#include "ols.hpp"
#include "wide_float.hpp"
#include "ols_precision.hpp"
#include "n1_checks.hpp"
#include "group_forward_certificate.hpp"
#include "exact_binary_products.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "hdfe/deterministic_parallel.hpp"
#include "hdfe/parallel_work_observer.hpp"
#include "hdfe/ieee_bits.hpp"

#ifdef HDFE_USE_OPENMP
#include <omp.h>
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440084436210485
#endif

namespace hdfe {
namespace detail {
namespace {

struct KahanSum {
    long double sum = 0.0L;
    long double c = 0.0L;
    void add(long double value) {
        const long double y = value - c;
        const long double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
};

inline int deterministic_ols_chunk_count(int n) {
    return deterministic_parallel_chunk_count(n);
}

inline int deterministic_ols_chunk_begin(int n, int chunk, int chunks) {
    return deterministic_parallel_chunk_begin(n, chunk, chunks);
}

inline int deterministic_ols_chunk_end(int n, int chunk, int chunks) {
    return deterministic_parallel_chunk_end(n, chunk, chunks);
}

class ObservedOlsRegion {
public:
    ObservedOlsRegion(ParallelWorkObserver* observer,
                      bool has_work,
                      int expected_team)
        : observer_(has_work ? observer : nullptr) {
        if (observer_ != nullptr) {
            observer_->begin_region(std::max(1, expected_team));
        }
    }

    ~ObservedOlsRegion() noexcept(false) {
        if (observer_ != nullptr) {
            observer_->end_region();
        }
    }

    void observe_work() const noexcept {
        if (observer_ != nullptr) {
            observer_->observe_work();
        }
    }

private:
    ParallelWorkObserver* observer_ = nullptr;
};

double weighted_sum_of_squares(const Eigen::VectorXd& values, const Eigen::VectorXd* weights) {
    const int n = static_cast<int>(values.size());
    if (n == 0) {
        return 0.0;
    }
    const double* v = values.data();
    KahanSum sum;
    if (!weights) {
        for (int i = 0; i < n; ++i) {
            const long double vi = static_cast<long double>(v[i]);
            sum.add(vi * vi);
        }
        return static_cast<double>(sum.sum);
    }
    const double* w = weights->data();
    for (int i = 0; i < n; ++i) {
        const long double vi = static_cast<long double>(v[i]);
        sum.add(vi * vi * static_cast<long double>(w[i]));
    }
    return static_cast<double>(sum.sum);
}

double weighted_sum_of_squares_with_n1(
    const Eigen::VectorXd& y,
    const Eigen::Ref<const Eigen::MatrixXd>& score_design,
    const Eigen::Ref<const Eigen::MatrixXd>* actual_design,
    const Eigen::VectorXd& coefficients,
    const Eigen::VectorXd& residuals,
    const Eigen::VectorXd* weights,
    N1NormalChunk& evidence) {
    const int n = static_cast<int>(residuals.size());
    const int p = static_cast<int>(score_design.cols());
    KahanSum sum;
    for (int i = 0; i < n; ++i) {
        const double u = residuals[i];
        const double w = weights ? (*weights)[i] : 1.0;
        const long double u_wide = static_cast<long double>(u);
        sum.add(u_wide * u_wide * static_cast<long double>(w));
        evidence.add(
            y[i], u, w, p,
            [&](int j) { return score_design(i, j); },
            [&](int j) {
                return actual_design ? (*actual_design)(i, j)
                                     : score_design(i, j);
            },
            [&](int j) { return coefficients[j]; });
    }
    return static_cast<double>(sum.sum);
}

double normal_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}



struct OlsInferenceRangeError:std::runtime_error {
    explicit OlsInferenceRangeError(const char* reason):std::runtime_error(
        std::string("OLS inference numerical range could not be preserved: ")+reason+
        "; no estimates returned") {}
};

bool inference_nonzero(double value) {
    return (ieee_bits_detail::bits(value)&0x7fffffffffffffffULL)!=0;
}

bool inference_nonzero(const OlsWide& value) {
    return inference_nonzero(value.hi) || inference_nonzero(value.lo);
}

bool inference_bad_product(double value) {
    return !ieee_finite(value) || std::abs(value)<std::numeric_limits<double>::min();
}

// Mark small totals for adaptive range inspection: subnormal summands can
// hide in a normal total. Crossing this margin alone never rejects a fit.
double inference_meat_floor(int rows) {
    return (std::numeric_limits<double>::min()*std::max(1,rows))/
        std::numeric_limits<double>::epsilon();
}

bool inference_score_term_risk(double weight,double residual,double x,bool squared,bool frequency) {
    if (!(weight>0) || !inference_nonzero(residual) || !inference_nonzero(x)) return false;
    int lower=0,upper=0;
    auto include=[&](double value,int copies) {
        const int stored=static_cast<int>((ieee_bits_detail::bits(value)>>52)&0x7ff);
        const int low=stored ? stored-1023 : -1074,high=stored ? stored-1023 : -1023;
        lower+=copies*std::min(0,low);upper+=copies*std::max(0,high+1);
    };
    include(weight,squared ? (frequency ? 1 : 2) : 1);
    include(residual,squared ? 2 : 1);include(x,squared ? 2 : 1);
    return lower<-1022 || upper>=1024;
}

bool inference_score_present(const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::VectorXd& residuals,const Eigen::VectorXd* weights,
    const Eigen::VectorXi* clusters,int column,bool inspect_range=false,bool frequency=false) {
    const int n=static_cast<int>(X.rows());
    if (!clusters) {
        for (int row=0;row<n;++row)
            if ((!weights || (*weights)[row]>0) && inference_nonzero(residuals[row]) &&
                (column<0 || inference_nonzero(X(row,column)))) {
                if (!inspect_range || inference_score_term_risk(weights ? (*weights)[row] : 1.0,
                    residuals[row],column<0 ? 1.0 : X(row,column),true,frequency)) return true;
            }
        return false;
    }
    // Only suspect zero columns reach this scan. One exact accumulator per
    // group preserves cancellation without allocating one per cluster.
    bool sorted=true;
    for (int row=1;row<n;++row) sorted=sorted && (*clusters)[row]>=(*clusters)[row-1];
    std::vector<int> order;
    if (!sorted) {
        order.resize(n);std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](int a,int b) {
            return (*clusters)[a]<(*clusters)[b] || ((*clusters)[a]==(*clusters)[b] && a<b);
        });
    }
    auto row_at=[&](int i) {return order.empty() ? i : order[i];};
    for (int begin=0;begin<n;) {
        const int id=(*clusters)[row_at(begin)];ExactBinaryProducts<> score;
        int end=begin;bool product_risk=false;
        do {
            const int row=row_at(end++);
            score.add_product(std::array<double,3>{weights ? (*weights)[row] : 1.0,
                residuals[row],column<0 ? 1.0 : X(row,column)});
            if (inspect_range) product_risk=product_risk || inference_score_term_risk(
                weights ? (*weights)[row] : 1.0,residuals[row],column<0 ? 1.0 : X(row,column),false,false);
        } while (end<n && (*clusters)[row_at(end)]==id);
        if (!score.valid()) return true;
        if (!score.exactly_zero()) {
            if (!inspect_range || product_risk) return true;
            const auto interval=score.interval();
            const long double lower=std::min(std::abs(interval.first),std::abs(interval.second));
            const long double upper=std::max(std::abs(interval.first),std::abs(interval.second));
            if (!(lower>=0x1p-511L) || !(upper<0x1p512L)) return true;
        }
        begin=end;
    }
    return false;
}

template<class Derived>
bool inference_meat_needs_wide(const Eigen::MatrixBase<Derived>& meat,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const Eigen::VectorXd& residuals,
    const Eigen::VectorXd* weights,const Eigen::VectorXi* clusters,bool scale_loss=false,bool frequency=false) {
    const double floor=inference_meat_floor(static_cast<int>(X.rows()));
    bool positive=false;
    for (int j=0;j<meat.rows();++j) {
        const double diagonal=meat(j,j);
        if (!ieee_finite(diagonal) || diagonal<0) return true;
        if (inference_nonzero(diagonal)) {
            positive=true;
            if (diagonal<floor && inference_score_present(X,residuals,weights,clusters,j,true,frequency)) return true;
        }
        else if (inference_score_present(X,residuals,weights,clusters,j)) return true;
    }
    // All-zero meat is safe only after each column's exact support/cancellation
    // check. A lost scalar on an irrelevant row must not revoke that zero.
    return scale_loss && positive;
}

void require_cluster_auxiliary_range(double u2,const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::VectorXd& residuals,const Eigen::VectorXd* weights,const Eigen::VectorXi& clusters) {
    if (!ieee_finite(u2) || u2<0 ||
        (inference_nonzero(u2) && u2<inference_meat_floor(static_cast<int>(X.rows())) &&
            inference_score_present(X,residuals,weights,&clusters,-1,true)) ||
        (!inference_nonzero(u2) && inference_score_present(X,residuals,weights,&clusters,-1)))
        throw OlsInferenceRangeError("cluster residual-total square lost range");
}

// Tiny/huge off-diagonal operands can lose range in B*M or (B*M)*B even
// when the resulting diagonal is finite. Only the small matrices are read.
template<class Bread,class Meat>
bool inference_sandwich_needs_wide(const Eigen::MatrixBase<Bread>& bread,
    const Eigen::MatrixBase<Meat>& meat) {
    int low_b=2048,high_b=-2048,low_m=2048,high_m=-2048;
    auto inspect=[](double value,int& low,int& high) {
        if (!ieee_finite(value)) return false;
        if (inference_nonzero(value)) {
            const int stored=static_cast<int>((ieee_bits_detail::bits(value)>>52)&0x7ff);
            low=std::min(low,stored ? stored-1023 : -1074);
            high=std::max(high,stored ? stored-1023 : -1023);
        }
        return true;
    };
    for (int j=0;j<bread.rows();++j) for (int k=0;k<bread.cols();++k)
        if (!inspect(bread(j,k),low_b,high_b) || !inspect(meat(j,k),low_m,high_m)) return true;
    if (high_m==-2048) return false;
    if (high_b==-2048) return true;
    int margin=3;for (int size=std::max(1,static_cast<int>(bread.rows()))-1;size;size>>=1) ++margin;
    return low_b+low_m<-1022 || 2*low_b+low_m<-1022 ||
        high_b+high_m+margin>=1024 || 2*high_b+high_m+2*margin>=1024;
}

OlsWide wide_inference_product(const OlsWide& a,const OlsWide& b,bool& loss) {
    const OlsWide value=a*b;
    loss=loss || !value.finite() ||
        (inference_nonzero(a) && inference_nonzero(b) && inference_bad_product(value.value()));
    return value;
}

OlsWide wide_inference_add(const OlsWide& a,const OlsWide& b,bool& loss) {
    const OlsWide value=a+b;
    if (!value.finite()) loss=true;
    else if (inference_bad_product(value.value()) && (inference_nonzero(a) || inference_nonzero(b))) {
        ExactBinaryProducts<> exact;
        for (double part:{a.hi,a.lo,b.hi,b.lo}) exact.add_product(std::array<double,1>{part});
        if (!exact.valid() || !exact.exactly_zero()) loss=true;
    }
    return value;
}

OlsWide checked_inference_product(const OlsWide& a,const OlsWide& b) {
    bool loss=false;const auto value=wide_inference_product(a,b,loss);
    if (loss) throw OlsInferenceRangeError("wide inference product underflow/overflow");
    return value;
}

OlsWide checked_inference_add(const OlsWide& a,const OlsWide& b) {
    bool loss=false;const auto value=wide_inference_add(a,b,loss);
    if (loss) throw OlsInferenceRangeError("wide inference sum lost a nonzero value");
    return value;
}

template<class Derived>
bool normal_equations_need_wide(const Eigen::MatrixBase<Derived>& gram) {
    const int p=static_cast<int>(gram.rows());Eigen::VectorXd scale(p);
    for(int j=0;j<p;++j) {
        if(!ieee_finite(gram(j,j)) || !(gram(j,j)>0)) return true;
        scale[j]=std::sqrt(gram(j,j));
    }
    if(p<2) return false;
    Eigen::MatrixXd normalized=gram;
    normalized.array().colwise()/=scale.array();
    normalized.array().rowwise()/=scale.transpose().array();
    if(!ieee_all_finite(normalized)) return true;
    Eigen::LDLT<Eigen::MatrixXd> factor(normalized);
    if(factor.info()!=Eigen::Success || !factor.isPositive()) return true;
    if(!ieee_all_finite(factor.vectorD()) ||
       (factor.vectorD().array()<=0.0).any()) return true;
    const double reciprocal=factor.rcond();
    // Solver selection only: no rank, estimator or convergence target changes.
    return !ieee_finite(reciprocal) || reciprocal<1e-8;
}

OlsResult run_ols_wide(const Eigen::VectorXd& y,const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::VectorXd* weights,const Eigen::VectorXi* cluster,
    const std::vector<Eigen::VectorXi>* multiway,StandardErrorType se_type,
    double tss,double within_tss,double n_effective,bool frequency,
    const Eigen::Ref<const Eigen::MatrixXd>* residual_design,ParallelWorkObserver* observer,
    ClusterDofMethod g_df=ClusterDofMethod::Min,bool g_adj=true,
    const Eigen::MatrixXd* score_low=nullptr,
    const Eigen::VectorXd* normalization_means=nullptr,
    Eigen::VectorXd* intercept_covariance=nullptr);

struct OlsOriginShift {
    int constant = -1;
    Eigen::VectorXd origins;
};

template<class Derived>
std::optional<OlsOriginShift> find_ols_origin_shift(
    const Eigen::MatrixBase<Derived>& gram,
    const Eigen::Ref<const Eigen::MatrixXd>& X) {
    const int n = static_cast<int>(X.rows()), p = static_cast<int>(X.cols());
    if (n == 0 || p < 2) return std::nullopt;
    int constant = -1;
    for (int j = 0; j < p; ++j) {
        if (X(0,j) == 1.0 && X(n-1,j) == 1.0 && (X.col(j).array() == 1.0).all()) {
            constant = j;
            break;
        }
    }
    if (constant < 0 || !(gram(constant,constant) > 0.0)) return std::nullopt;
    OlsOriginShift shift{constant, Eigen::VectorXd::Zero(p)};
    bool needed = false;
    for (int j = 0; j < p; ++j) {
        if (j == constant || !(gram(j,j) > 0.0)) continue;
        const double correlation = std::abs(gram(j,constant)) /
            std::sqrt(gram(j,j)) / std::sqrt(gram(constant,constant));
        const double origin = gram(j,constant) / gram(constant,constant);
        // Arithmetic dispatch only, using the existing IV Gram band.
        if (ieee_finite(correlation) && ieee_finite(origin) && origin != 0.0 &&
            1.0 - correlation * correlation <= 0x1p-13) {
            shift.origins[j] = origin;
            needed = true;
        }
    }
    return needed ? std::optional<OlsOriginShift>(std::move(shift)) : std::nullopt;
}

OlsResult run_ols_origin_shifted(
    const OlsOriginShift& shift,
    const Eigen::VectorXd& y, const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::VectorXd* weights, const Eigen::VectorXi* cluster,
    const std::vector<Eigen::VectorXi>* multiway, StandardErrorType se_type,
    double tss, double within_tss, double n_effective, bool frequency,
    const Eigen::Ref<const Eigen::MatrixXd>* residual_design,
    bool explicit_threads, ParallelWorkObserver* observer,
    ClusterDofMethod g_df = ClusterDofMethod::Min, bool g_adj = true) {
    Eigen::MatrixXd shifted = X, actual;
    if (residual_design) actual = *residual_design;
    auto subtract_origins = [&](Eigen::MatrixXd& matrix) {
        for (int j = 0; j < matrix.cols(); ++j) {
            if (shift.origins[j] == 0.0) continue;
            const OlsWide origin(shift.origins[j]);
            for (int i = 0; i < matrix.rows(); ++i) {
                const OlsWide value = OlsWide(matrix(i,j)) - origin;
                if (!value.finite() || value.lo != 0.0) return false;
                matrix(i,j) = value.hi;
            }
        }
        return true;
    };
    if (!subtract_origins(shifted) || (residual_design && !subtract_origins(actual))) {
        return run_ols_wide(y,X,weights,cluster,multiway,se_type,tss,within_tss,
                            n_effective,frequency,residual_design,observer,g_df,g_adj);
    }
    std::optional<Eigen::Ref<const Eigen::MatrixXd>> actual_ref;
    if (residual_design) actual_ref.emplace(actual);
    OlsResult result = multiway
        ? run_ols_multiway_with_score_low(y,shifted,weights,multiway,se_type,
            tss,within_tss,g_df,g_adj,n_effective,actual_ref ? &*actual_ref : nullptr,
            explicit_threads,observer,nullptr)
        : run_ols_with_score_low(y,shifted,weights,cluster,se_type,tss,within_tss,
            n_effective,frequency,actual_ref ? &*actual_ref : nullptr,
            explicit_threads,observer,nullptr);

    const int p = static_cast<int>(X.cols()), c = shift.constant;
    OlsWide intercept(result.coefficients[c]);
    for (int j = 0; j < p; ++j)
        intercept -= OlsWide(shift.origins[j]) * OlsWide(result.coefficients[j]);
    result.coefficients[c] = intercept.value();
    // A congruence in the small coefficient space, retaining cancellation.
    auto restore_covariance = [&](Eigen::MatrixXd& matrix) {
        std::vector<OlsWide> row(static_cast<std::size_t>(p)), column(static_cast<std::size_t>(p));
        for (int j = 0; j < p; ++j) {
            row[j] = OlsWide(matrix(c,j));
            column[j] = OlsWide(matrix(j,c));
            for (int k = 0; k < p; ++k) {
                row[j] -= OlsWide(shift.origins[k]) * OlsWide(matrix(k,j));
                column[j] -= OlsWide(matrix(j,k)) * OlsWide(shift.origins[k]);
            }
        }
        OlsWide diagonal = row[c];
        for (int j = 0; j < p; ++j)
            diagonal -= row[j] * OlsWide(shift.origins[j]);
        for (int j = 0; j < p; ++j) {
            matrix(c,j) = row[j].value();
            matrix(j,c) = column[j].value();
        }
        matrix(c,c) = diagonal.value();
    };
    restore_covariance(result.xtx_inv);
    restore_covariance(result.covariance);
    if (result.cluster_ux.size() == p) {
        for (int j = 0; j < p; ++j)
            result.cluster_ux[j] = (OlsWide(result.cluster_ux[j]) +
                OlsWide(shift.origins[j]) * OlsWide(result.cluster_u2)).value();
    }
    if (!ieee_all_finite(result.coefficients) || !ieee_all_finite(result.covariance) ||
        !ieee_all_finite(result.xtx_inv))
        throw OlsInferenceRangeError("OLS origin transformation exceeded its numerical range");
    result.std_errors = result.covariance.diagonal().array().cwiseMax(0.0).sqrt();
    constexpr double z = 1.959963984540054;
    for (int j = 0; j < p; ++j) {
        const double se = result.std_errors[j], b = result.coefficients[j];
        result.tvalues[j] = se > 0.0 ? b / se : 0.0;
        result.pvalues[j] = se > 0.0 ? 2 * (1 - normal_cdf(std::abs(result.tvalues[j]))) : 1.0;
        result.conf_int(j,0) = b - z * se;
        result.conf_int(j,1) = b + z * se;
    }
    return result;
}

template <int P>
OlsResult run_ols_fast_impl(const Eigen::VectorXd& y,
                            const Eigen::Ref<const Eigen::MatrixXd>& X,
                            const Eigen::VectorXd* weights,
                            StandardErrorType se_type,
                            double total_sum_squares,
                            double within_sum_squares,
                            double n_effective,
                            bool weights_are_frequencies,
                            ParallelWorkObserver* parallel_observer) {
    static_assert(P > 0, "P must be positive");
    if (se_type == StandardErrorType::Cluster) {
        throw std::runtime_error("run_ols_fast_impl does not support clustered inference");
    }
    const int n = static_cast<int>(y.size());
    if (X.rows() != n || X.cols() != P) {
        throw std::runtime_error("run_ols_fast_impl called with inconsistent dimensions");
    }

    const double* y_ptr = y.data();
    const double* w_ptr = weights ? weights->data() : nullptr;
    const Eigen::Index x_rows = X.rows();
    const double* x_ptrs[P];
    for (int j = 0; j < P; ++j) {
        x_ptrs[j] = X.data() + static_cast<Eigen::Index>(j) * x_rows;
    }

    int threads = 1;
#ifdef HDFE_USE_OPENMP
    threads = std::max(1, omp_get_max_threads());
#endif

    using MatPP = Eigen::Matrix<double, P, P>;
    using VecP = Eigen::Matrix<double, P, 1>;

    const int chunks = deterministic_ols_chunk_count(n);
    std::vector<MatPP> xtx_tls(static_cast<std::size_t>(chunks), MatPP::Zero());
    std::vector<VecP> xty_tls(static_cast<std::size_t>(chunks), VecP::Zero());

    {
        ObservedOlsRegion observed(
            parallel_observer, chunks > 0, threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int chunk = 0; chunk < chunks; ++chunk) {
            observed.observe_work();
            MatPP& xtx_local = xtx_tls[static_cast<std::size_t>(chunk)];
            VecP& xty_local = xty_tls[static_cast<std::size_t>(chunk)];
            const int begin = deterministic_ols_chunk_begin(n, chunk, chunks);
            const int end = deterministic_ols_chunk_end(n, chunk, chunks);
            for (int i = begin; i < end; ++i) {
                const double w = w_ptr ? w_ptr[i] : 1.0;
                const double yi = y_ptr[i];
                double xvals[P];
                for (int j = 0; j < P; ++j) {
                    xvals[j] = x_ptrs[j][i];
                }
                for (int j = 0; j < P; ++j) {
                    xty_local(j) += w * xvals[j] * yi;
                    for (int k = 0; k <= j; ++k) {
                        xtx_local(j, k) += w * xvals[j] * xvals[k];
                    }
                }
            }
        }
    }

    MatPP xtx = MatPP::Zero();
    VecP xty = VecP::Zero();
    for (int chunk = 0; chunk < chunks; ++chunk) {
        xtx.noalias() += xtx_tls[static_cast<std::size_t>(chunk)];
        xty.noalias() += xty_tls[static_cast<std::size_t>(chunk)];
    }
    for (int j = 0; j < P; ++j) {
        for (int k = j + 1; k < P; ++k) {
            xtx(j, k) = xtx(k, j);
        }
    }

    if (auto shift = find_ols_origin_shift(xtx, X))
        return run_ols_origin_shifted(*shift,y,X,weights,nullptr,nullptr,se_type,
            total_sum_squares,within_sum_squares,n_effective,weights_are_frequencies,
            nullptr,false,parallel_observer);
    if(normal_equations_need_wide(xtx))
        return run_ols_wide(y,X,weights,nullptr,nullptr,se_type,total_sum_squares,
            within_sum_squares,n_effective,weights_are_frequencies,nullptr,parallel_observer);
    Eigen::LDLT<MatPP> solver;
    solver.compute(xtx);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize X'X; matrix may be singular");
    }
    VecP beta = solver.solve(xty);
    MatPP xtx_inv = solver.solve(MatPP::Identity());

    Eigen::VectorXd residuals(n);
    double* resid_ptr = residuals.data();
    std::shared_ptr<N1NormalEvidence> n1_normal;
    if (n1_capture_active) {
        n1_normal = std::make_shared<N1NormalEvidence>(chunks, P);
    }

    std::vector<long double> rss_tls(static_cast<std::size_t>(chunks), 0.0L);
    std::atomic<bool> inference_scale_loss{false};
    std::vector<MatPP> meat_tls;
    if (se_type == StandardErrorType::Robust) {
        meat_tls.assign(static_cast<std::size_t>(chunks), MatPP::Zero());
    }

    {
        ObservedOlsRegion observed(
            parallel_observer, chunks > 0, threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int chunk = 0; chunk < chunks; ++chunk) {
            observed.observe_work();
            N1NormalChunk* n1_chunk =
                n1_normal ? &n1_normal->chunks[static_cast<std::size_t>(chunk)]
                          : nullptr;
            KahanSum rss_local;
            MatPP meat_local = MatPP::Zero();
            const int begin = deterministic_ols_chunk_begin(n, chunk, chunks);
            const int end = deterministic_ols_chunk_end(n, chunk, chunks);
            for (int i = begin; i < end; ++i) {
                double xvals[P];
                for (int j = 0; j < P; ++j) {
                    xvals[j] = x_ptrs[j][i];
                }
                double fitted = 0.0;
                for (int j = 0; j < P; ++j) {
                    fitted += xvals[j] * beta(j);
                }
                const double u = y_ptr[i] - fitted;
                resid_ptr[i] = u;

                const double w = w_ptr ? w_ptr[i] : 1.0;
                rss_local.add(static_cast<long double>(w) *
                              static_cast<long double>(u) * static_cast<long double>(u));
                if (n1_chunk) {
                    n1_chunk->add(
                        y_ptr[i], u, w, P,
                        [&](int j) { return xvals[j]; },
                        [&](int j) { return xvals[j]; },
                        [&](int j) { return beta(j); });
                }

                if (se_type == StandardErrorType::Robust) {
                    // fweight: replication semantics -> linear w in the robust meat;
                    // aweight/pweight: w^2 (matches Stata/areg/reghdfe).
                    const double w2 =
                        w_ptr ? (weights_are_frequencies ? w : (w * w)) : 1.0;
                    const double scale = w2 * u * u;
                    if (inference_nonzero(u) && w>0 && inference_bad_product(scale))
                        for (int j=0;j<P;++j) if (inference_nonzero(xvals[j])) {
                            inference_scale_loss.store(true,std::memory_order_relaxed);break;
                        }
                    for (int j = 0; j < P; ++j) {
                        for (int k = 0; k <= j; ++k) {
                            meat_local(j, k) += scale * xvals[j] * xvals[k];
                        }
                    }
                }
            }

            rss_tls[static_cast<std::size_t>(chunk)] = rss_local.sum;
            if (se_type == StandardErrorType::Robust) {
                meat_tls[static_cast<std::size_t>(chunk)] = std::move(meat_local);
            }
        }
    }

    long double rss = 0.0L;
    for (long double v : rss_tls) {
        rss += v;
    }
    const double rss_d = static_cast<double>(rss);

    MatPP meat = MatPP::Zero();
    if (se_type == StandardErrorType::Robust) {
        for (int chunk = 0; chunk < chunks; ++chunk) {
            meat.noalias() += meat_tls[static_cast<std::size_t>(chunk)];
        }
        for (int j = 0; j < P; ++j) {
            for (int k = j + 1; k < P; ++k) {
                meat(j, k) = meat(k, j);
            }
        }
    }

    if (se_type==StandardErrorType::Robust &&
        (inference_meat_needs_wide(meat,X,residuals,weights,nullptr,
             inference_scale_loss.load(std::memory_order_relaxed),weights_are_frequencies) ||
         inference_sandwich_needs_wide(xtx_inv,meat))) {
        n1_normal.reset();
        return run_ols_wide(y,X,weights,nullptr,nullptr,se_type,total_sum_squares,
            within_sum_squares,n_effective,weights_are_frequencies,nullptr,parallel_observer);
    }

    const double df_resid = std::max(0.0, n_effective - static_cast<double>(P));
    const double sigma2 = (df_resid > 0.0) ? rss_d / df_resid : 0.0;

    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(P, P);
    if (se_type == StandardErrorType::Homoskedastic) {
        covariance = sigma2 * xtx_inv;
    } else {
        covariance = xtx_inv * meat * xtx_inv;
    }

    Eigen::VectorXd std_errors = covariance.diagonal().array().cwiseMax(0.0).sqrt();
    Eigen::VectorXd tvalues = Eigen::VectorXd::Zero(P);
    Eigen::VectorXd pvalues = Eigen::VectorXd::Zero(P);
    Eigen::MatrixXd conf_int(P, 2);
    constexpr double kZ = 1.959963984540054;
    for (int j = 0; j < P; ++j) {
        if (std_errors(j) > 0) {
            tvalues(j) = beta(j) / std_errors(j);
            const double tail = 1.0 - normal_cdf(std::abs(tvalues(j)));
            pvalues(j) = 2.0 * tail;
        } else {
            tvalues(j) = 0.0;
            pvalues(j) = 1.0;
        }
        conf_int(j, 0) = beta(j) - kZ * std_errors(j);
        conf_int(j, 1) = beta(j) + kZ * std_errors(j);
    }

    OlsResult result;
    result.coefficients = Eigen::VectorXd(beta);
    result.std_errors = std::move(std_errors);
    result.tvalues = std::move(tvalues);
    result.pvalues = std::move(pvalues);
    result.conf_int = std::move(conf_int);
    result.residuals = std::move(residuals);
    result.covariance = std::move(covariance);
    result.xtx_inv = std::move(xtx_inv);
    result.df_resid = df_resid;
    result.rss = rss_d;
    result.tss = total_sum_squares;
    result.within_tss = within_sum_squares;
    const double missing_r2 = std::numeric_limits<double>::quiet_NaN();
    result.r2 = (total_sum_squares > 0.0)
                    ? 1.0 - rss_d / total_sum_squares
                    : missing_r2;
    result.r2_within = (within_sum_squares > 0.0)
                           ? 1.0 - rss_d / within_sum_squares
                           : missing_r2;
    result.sigma2 = sigma2;
    result.nobs = n;
    result.n1_normal = std::move(n1_normal);
    return result;
}

template <int P>
OlsResult run_ols_fast_from_xtx_impl(const Eigen::VectorXd& y,
                                     const Eigen::Ref<const Eigen::MatrixXd>& X,
                                     const Eigen::VectorXd* weights,
                                     StandardErrorType se_type,
                                     double total_sum_squares,
                                     double within_sum_squares,
                                     const Eigen::Matrix<double, P, P>& xtx,
                                     const Eigen::Matrix<double, P, 1>& xty,
                                     double n_effective,
                                     bool weights_are_frequencies,
                                     ParallelWorkObserver* parallel_observer) {
    static_assert(P > 0, "P must be positive");
    if (se_type == StandardErrorType::Cluster) {
        throw std::runtime_error("run_ols_fast_from_xtx_impl does not support clustered inference");
    }
    const int n = static_cast<int>(y.size());
    if (X.rows() != n || X.cols() != P) {
        throw std::runtime_error("run_ols_fast_from_xtx_impl called with inconsistent dimensions");
    }

    const double* y_ptr = y.data();
    const double* w_ptr = weights ? weights->data() : nullptr;
    const Eigen::Index x_rows = X.rows();
    const double* x_ptrs[P];
    for (int j = 0; j < P; ++j) {
        x_ptrs[j] = X.data() + static_cast<Eigen::Index>(j) * x_rows;
    }

    Eigen::LDLT<Eigen::Matrix<double, P, P>> solver;
    solver.compute(xtx);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize X'X; matrix may be singular");
    }
    Eigen::Matrix<double, P, 1> beta = solver.solve(xty);
    Eigen::Matrix<double, P, P> xtx_inv =
        solver.solve(Eigen::Matrix<double, P, P>::Identity());

    Eigen::VectorXd residuals(n);
    double* resid_ptr = residuals.data();

    int threads = 1;
#ifdef HDFE_USE_OPENMP
    threads = std::max(1, omp_get_max_threads());
#endif

    const int chunks = deterministic_ols_chunk_count(n);
    std::shared_ptr<N1NormalEvidence> n1_normal;
    if (n1_capture_active) {
        n1_normal = std::make_shared<N1NormalEvidence>(chunks, P);
    }
    std::vector<long double> rss_tls(static_cast<std::size_t>(chunks), 0.0L);
    std::atomic<bool> inference_scale_loss{false};
    std::vector<Eigen::Matrix<double, P, P>> meat_tls;
    if (se_type == StandardErrorType::Robust) {
        meat_tls.assign(static_cast<std::size_t>(chunks),
                        Eigen::Matrix<double, P, P>::Zero());
    }

    {
        ObservedOlsRegion observed(
            parallel_observer, chunks > 0, threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int chunk = 0; chunk < chunks; ++chunk) {
            observed.observe_work();
            N1NormalChunk* n1_chunk =
                n1_normal ? &n1_normal->chunks[static_cast<std::size_t>(chunk)]
                          : nullptr;
            KahanSum rss_local;
            Eigen::Matrix<double, P, P> meat_local =
                Eigen::Matrix<double, P, P>::Zero();
            const int begin = deterministic_ols_chunk_begin(n, chunk, chunks);
            const int end = deterministic_ols_chunk_end(n, chunk, chunks);
            for (int i = begin; i < end; ++i) {
                double xvals[P];
                for (int j = 0; j < P; ++j) {
                    xvals[j] = x_ptrs[j][i];
                }
                double fitted = 0.0;
                for (int j = 0; j < P; ++j) {
                    fitted += xvals[j] * beta(j);
                }
                const double u = y_ptr[i] - fitted;
                resid_ptr[i] = u;

                const double w = w_ptr ? w_ptr[i] : 1.0;
                rss_local.add(static_cast<long double>(w) *
                              static_cast<long double>(u) * static_cast<long double>(u));
                if (n1_chunk) {
                    n1_chunk->add(
                        y_ptr[i], u, w, P,
                        [&](int j) { return xvals[j]; },
                        [&](int j) { return xvals[j]; },
                        [&](int j) { return beta(j); });
                }

                if (se_type == StandardErrorType::Robust) {
                    // fweight: replication semantics -> linear w in the robust meat;
                    // aweight/pweight: w^2 (matches Stata/areg/reghdfe).
                    const double w2 =
                        w_ptr ? (weights_are_frequencies ? w : (w * w)) : 1.0;
                    const double scale = w2 * u * u;
                    if (inference_nonzero(u) && w>0 && inference_bad_product(scale))
                        for (int j=0;j<P;++j) if (inference_nonzero(xvals[j])) {
                            inference_scale_loss.store(true,std::memory_order_relaxed);break;
                        }
                    for (int j = 0; j < P; ++j) {
                        for (int k = 0; k <= j; ++k) {
                            meat_local(j, k) += scale * xvals[j] * xvals[k];
                        }
                    }
                }
            }

            rss_tls[static_cast<std::size_t>(chunk)] = rss_local.sum;
            if (se_type == StandardErrorType::Robust) {
                meat_tls[static_cast<std::size_t>(chunk)] = std::move(meat_local);
            }
        }
    }

    long double rss = 0.0L;
    for (long double v : rss_tls) {
        rss += v;
    }
    const double rss_d = static_cast<double>(rss);

    Eigen::Matrix<double, P, P> meat = Eigen::Matrix<double, P, P>::Zero();
    if (se_type == StandardErrorType::Robust) {
        for (int chunk = 0; chunk < chunks; ++chunk) {
            meat.noalias() += meat_tls[static_cast<std::size_t>(chunk)];
        }
        for (int j = 0; j < P; ++j) {
            for (int k = j + 1; k < P; ++k) {
                meat(j, k) = meat(k, j);
            }
        }
    }

    if (se_type==StandardErrorType::Robust &&
        (inference_meat_needs_wide(meat,X,residuals,weights,nullptr,
             inference_scale_loss.load(std::memory_order_relaxed),weights_are_frequencies) ||
         inference_sandwich_needs_wide(xtx_inv,meat))) {
        n1_normal.reset();
        return run_ols_wide(y,X,weights,nullptr,nullptr,se_type,total_sum_squares,
            within_sum_squares,n_effective,weights_are_frequencies,nullptr,parallel_observer);
    }

    const double df_resid = std::max(0.0, n_effective - static_cast<double>(P));
    const double sigma2 = (df_resid > 0.0) ? rss_d / df_resid : 0.0;

    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(P, P);
    if (se_type == StandardErrorType::Homoskedastic) {
        covariance = sigma2 * xtx_inv;
    } else {
        covariance = xtx_inv * meat * xtx_inv;
    }

    Eigen::VectorXd std_errors = covariance.diagonal().array().cwiseMax(0.0).sqrt();
    Eigen::VectorXd tvalues = Eigen::VectorXd::Zero(P);
    Eigen::VectorXd pvalues = Eigen::VectorXd::Zero(P);
    Eigen::MatrixXd conf_int(P, 2);
    constexpr double kZ = 1.959963984540054;
    for (int j = 0; j < P; ++j) {
        if (std_errors(j) > 0) {
            tvalues(j) = beta(j) / std_errors(j);
            const double tail = 1.0 - normal_cdf(std::abs(tvalues(j)));
            pvalues(j) = 2.0 * tail;
        } else {
            tvalues(j) = 0.0;
            pvalues(j) = 1.0;
        }
        conf_int(j, 0) = beta(j) - kZ * std_errors(j);
        conf_int(j, 1) = beta(j) + kZ * std_errors(j);
    }

    OlsResult result;
    result.coefficients = Eigen::VectorXd(beta);
    result.std_errors = std::move(std_errors);
    result.tvalues = std::move(tvalues);
    result.pvalues = std::move(pvalues);
    result.conf_int = std::move(conf_int);
    result.residuals = std::move(residuals);
    result.covariance = std::move(covariance);
    result.xtx_inv = std::move(xtx_inv);
    result.df_resid = df_resid;
    result.rss = rss_d;
    result.tss = total_sum_squares;
    result.within_tss = within_sum_squares;
    const double missing_r2 = std::numeric_limits<double>::quiet_NaN();
    result.r2 = (total_sum_squares > 0.0)
                    ? 1.0 - rss_d / total_sum_squares
                    : missing_r2;
    result.r2_within = (within_sum_squares > 0.0)
                           ? 1.0 - rss_d / within_sum_squares
                           : missing_r2;
    result.sigma2 = sigma2;
    result.nobs = n;
    result.n1_normal = std::move(n1_normal);
    return result;
}

Eigen::MatrixXd compute_covariance(const Eigen::MatrixXd& xtx_inv,
                                    const Eigen::Ref<const Eigen::MatrixXd>& WX,
                                    const Eigen::Ref<const Eigen::MatrixXd>& original_X,
                                    const Eigen::VectorXd& residuals,
                                    const Eigen::VectorXd* original_weights,
                                    const Eigen::VectorXd* sqrt_weights,
                                    const Eigen::VectorXi* clusters,
                                    StandardErrorType se_type,
                                    double sigma2,
                                    double df_resid,
                                    double n_effective,
                                    int* num_clusters_out,
                                    Eigen::VectorXd* cluster_ux_out,
                                    double* cluster_u2_out,
                                    bool weights_are_frequencies,
                                    bool num_threads_explicit,
                                    ParallelWorkObserver* parallel_observer) {
    const int p = static_cast<int>(WX.cols());
    const int n = static_cast<int>(WX.rows());
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p, p);
    if (num_clusters_out) {
        *num_clusters_out = 0;
    }
    if (cluster_ux_out) {
        cluster_ux_out->resize(0);
    }
    if (cluster_u2_out) {
        *cluster_u2_out = 0.0;
    }
    if (se_type == StandardErrorType::Homoskedastic) {
        cov = sigma2 * xtx_inv;
        return cov;
    }

    if (se_type == StandardErrorType::Robust) {
        Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(p, p);
        auto meat_lower = meat.selfadjointView<Eigen::Lower>();
        bool scale_loss=false;
        for (int i = 0; i < n; ++i) {
            // WX already carries sqrt(w), so WX_i WX_i' = w*xx'. For aweight/pweight
            // the robust meat needs w^2*xx' (scale by u*sqrt(w) -> u^2*w). For fweight
            // (replication) it needs w*xx' (scale by u -> u^2), matching Stata/areg/reghdfe.
            const double scaled_u = (sqrt_weights && !weights_are_frequencies)
                                        ? residuals(i) * (*sqrt_weights)(i)
                                        : residuals(i);
            const double w = scaled_u * scaled_u;
            if (inference_nonzero(residuals(i)) && (!original_weights || (*original_weights)[i]>0) &&
                inference_bad_product(w))
                for (int j=0;j<p;++j) if (inference_nonzero(original_X(i,j))) {scale_loss=true;break;}
            meat_lower.rankUpdate(WX.row(i).transpose(), w);
        }
        if (inference_meat_needs_wide(meat,original_X,residuals,original_weights,nullptr,scale_loss,weights_are_frequencies) ||
            inference_sandwich_needs_wide(xtx_inv,meat)) throw OlsInferenceRangeError("HC meat");
        cov = xtx_inv * meat_lower * xtx_inv;
        if (!ieee_all_finite(cov)) throw OlsInferenceRangeError("covariance scaling overflow");
        return cov;
    }

    if (!clusters) {
        throw std::runtime_error("Cluster-robust errors requested but no cluster vector was provided");
    }
    if (clusters->size() != n) {
        throw std::runtime_error("Cluster vector length must equal the number of observations");
    }
    int min_id = (*clusters)(0);
    int max_id = (*clusters)(0);
    int adjacent_matches = 0;
    bool nondecreasing = true;
    int prev_id = (*clusters)(0);
    for (int i = 1; i < n; ++i) {
        const int v = (*clusters)(i);
        min_id = std::min(min_id, v);
        max_id = std::max(max_id, v);
        if (v == prev_id) {
            ++adjacent_matches;
        }
        if (v < prev_id) {
            nondecreasing = false;
        }
        prev_id = v;
    }
    const long long range_ll =
        static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL;
    constexpr long long kDenseRangeCap = 50000000LL;
    const bool dense_ok =
        min_id >= 0 && range_ll > 0 && range_ll <= kDenseRangeCap &&
        range_ll <= static_cast<long long>(n) * 2LL;
    const int range = dense_ok ? static_cast<int>(range_ll) : 0;

    std::atomic<bool> inference_scale_loss{false};
    std::vector<double> aggregated;
    std::vector<double> aggregated_u;
    int next_cluster = 0;
    if (dense_ok) {
        const double run_frac =
            (n > 1) ? static_cast<double>(adjacent_matches) / static_cast<double>(n - 1) : 0.0;
        constexpr double kRunFastThreshold = 0.20;
        const bool use_run_fast = (run_frac >= kRunFastThreshold);
        if (nondecreasing && p <= 8) {
            std::vector<int> run_starts;
            std::vector<int> run_ends;
            run_starts.reserve(static_cast<std::size_t>(range));
            run_ends.reserve(static_cast<std::size_t>(range));
            int i = 0;
            while (i < n) {
                const int start = i;
                do {
                    ++i;
                } while (i < n && (*clusters)(i) == (*clusters)(start));
                run_starts.push_back(start);
                run_ends.push_back(i);
            }
            next_cluster = static_cast<int>(run_starts.size());
            std::vector<double> run_scores(
                static_cast<std::size_t>(next_cluster) * static_cast<std::size_t>(p));
            std::vector<double> run_u_sums(static_cast<std::size_t>(next_cluster));

            // Per-run scores are each accumulated serially by one thread in
            // observation order and combined serially in run order below, so
            // the result is bit-identical at any thread count. The historical
            // 8-thread cap remains an auto-only performance heuristic below
            // the large-n gate. An explicit request is honored up to the
            // OpenMP runtime capacity.
            int cluster_threads = 1;
#ifdef HDFE_USE_OPENMP
            cluster_threads =
                (num_threads_explicit || n >= 4194304)
                    ? std::max(1, omp_get_max_threads())
                    : std::min(std::max(1, omp_get_max_threads()), 8);
#endif
            {
                ObservedOlsRegion observed(
                    parallel_observer, next_cluster > 0, cluster_threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(cluster_threads)
#endif
                for (int r = 0; r < next_cluster; ++r) {
                    observed.observe_work();
                    const int start = run_starts[static_cast<std::size_t>(r)];
                    const int end = run_ends[static_cast<std::size_t>(r)];
                    double run_score[8] =
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    double run_u_sum = 0.0;
                    for (int obs = start; obs < end; ++obs) {
                        const double scaled_u =
                            sqrt_weights
                                ? residuals(obs) * (*sqrt_weights)(obs)
                                : residuals(obs);
                        if (sqrt_weights && inference_nonzero(residuals(obs)) &&
                            inference_nonzero((*sqrt_weights)(obs)) && inference_bad_product(scaled_u))
                            inference_scale_loss.store(true,std::memory_order_relaxed);
                        for (int j = 0; j < p; ++j) {
                            run_score[j] += WX(obs, j) * scaled_u;
                        }
                        double u_score = scaled_u;
                        if (sqrt_weights) {
                            u_score *= (*sqrt_weights)(obs);
                        }
                        run_u_sum += u_score;
                    }
                    double* score =
                        run_scores.data() +
                        static_cast<std::size_t>(r) *
                            static_cast<std::size_t>(p);
                    for (int j = 0; j < p; ++j) {
                        score[static_cast<std::size_t>(j)] = run_score[j];
                    }
                    run_u_sums[static_cast<std::size_t>(r)] = run_u_sum;
                }
            }

            Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(p, p);
            auto meat_lower = meat.selfadjointView<Eigen::Lower>();
            Eigen::VectorXd cluster_ux;
            if (cluster_ux_out) {
                cluster_ux = Eigen::VectorXd::Zero(p);
            }
            double cluster_u2 = 0.0;
            for (int r = 0; r < next_cluster; ++r) {
                const Eigen::Map<const Eigen::VectorXd> score(
                    run_scores.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(p),
                    p);
                meat_lower.rankUpdate(score, 1.0);
                const double ug = run_u_sums[static_cast<std::size_t>(r)];
                cluster_u2 += ug * ug;
                if (cluster_ux_out) {
                    cluster_ux.noalias() += ug * score;
                }
            }
            double scale = 1.0;
            const int G = next_cluster;
            if (num_clusters_out) {
                *num_clusters_out = G;
            }
            if (cluster_u2_out) require_cluster_auxiliary_range(cluster_u2,original_X,residuals,original_weights,*clusters);
            if (cluster_u2_out) {
                *cluster_u2_out = cluster_u2;
            }
            if (cluster_ux_out) {
                *cluster_ux_out = std::move(cluster_ux);
            }
            if (G > 1 && df_resid > 0.0) {
                scale = (static_cast<double>(G) / (G - 1.0)) *
                        ((n_effective - 1.0) / df_resid);
            }
            if (inference_meat_needs_wide(meat,original_X,residuals,original_weights,clusters,
                    inference_scale_loss.load(std::memory_order_relaxed)) ||
                inference_sandwich_needs_wide(xtx_inv,meat)) throw OlsInferenceRangeError("cluster meat");
            cov = xtx_inv * meat_lower * xtx_inv;
            cov *= scale;
            if (!ieee_all_finite(cov)) throw OlsInferenceRangeError("covariance scaling overflow");
            return cov;
        }
        aggregated.assign(static_cast<std::size_t>(range) * static_cast<std::size_t>(p), 0.0);
        aggregated_u.assign(static_cast<std::size_t>(range), 0.0);
        std::vector<uint8_t> seen(static_cast<std::size_t>(range), 0);
        if (use_run_fast) {
            int i = 0;
            if (p <= 8) {
                while (i < n) {
                    const int idx = (*clusters)(i) - min_id;
                    if (!seen[static_cast<std::size_t>(idx)]) {
                        seen[static_cast<std::size_t>(idx)] = 1;
                        ++next_cluster;
                    }

                    double run_score[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    double run_u = 0.0;
                    do {
                        const double scaled_u =
                            sqrt_weights ? residuals(i) * (*sqrt_weights)(i) : residuals(i);
                        if (sqrt_weights && inference_nonzero(residuals(i)) &&
                            inference_nonzero((*sqrt_weights)(i)) && inference_bad_product(scaled_u))
                            inference_scale_loss.store(true,std::memory_order_relaxed);
                        for (int j = 0; j < p; ++j) {
                            run_score[j] += WX(i, j) * scaled_u;
                        }
                        double u_score = scaled_u;
                        if (sqrt_weights) {
                            u_score *= (*sqrt_weights)(i);
                        }
                        run_u += u_score;
                        ++i;
                    } while (i < n && ((*clusters)(i) - min_id) == idx);

                    double* score = aggregated.data() +
                                    static_cast<std::size_t>(idx) * static_cast<std::size_t>(p);
                    for (int j = 0; j < p; ++j) {
                        score[static_cast<std::size_t>(j)] += run_score[j];
                    }
                    aggregated_u[static_cast<std::size_t>(idx)] += run_u;
                }
            } else {
                std::vector<double> run_score(static_cast<std::size_t>(p), 0.0);
                while (i < n) {
                    const int idx = (*clusters)(i) - min_id;
                    if (!seen[static_cast<std::size_t>(idx)]) {
                        seen[static_cast<std::size_t>(idx)] = 1;
                        ++next_cluster;
                    }

                    std::fill(run_score.begin(), run_score.end(), 0.0);
                    double run_u = 0.0;
                    do {
                        const double scaled_u =
                            sqrt_weights ? residuals(i) * (*sqrt_weights)(i) : residuals(i);
                        if (sqrt_weights && inference_nonzero(residuals(i)) &&
                            inference_nonzero((*sqrt_weights)(i)) && inference_bad_product(scaled_u))
                            inference_scale_loss.store(true,std::memory_order_relaxed);
                        for (int j = 0; j < p; ++j) {
                            run_score[static_cast<std::size_t>(j)] += WX(i, j) * scaled_u;
                        }
                        double u_score = scaled_u;
                        if (sqrt_weights) {
                            u_score *= (*sqrt_weights)(i);
                        }
                        run_u += u_score;
                        ++i;
                    } while (i < n && ((*clusters)(i) - min_id) == idx);

                    double* score = aggregated.data() +
                                    static_cast<std::size_t>(idx) * static_cast<std::size_t>(p);
                    for (int j = 0; j < p; ++j) {
                        score[static_cast<std::size_t>(j)] +=
                            run_score[static_cast<std::size_t>(j)];
                    }
                    aggregated_u[static_cast<std::size_t>(idx)] += run_u;
                }
            }
        } else {
            for (int i = 0; i < n; ++i) {
                const int idx = (*clusters)(i) - min_id;
                if (!seen[static_cast<std::size_t>(idx)]) {
                    seen[static_cast<std::size_t>(idx)] = 1;
                    ++next_cluster;
                }
                const double scaled_u =
                    sqrt_weights ? residuals(i) * (*sqrt_weights)(i) : residuals(i);
                if (sqrt_weights && inference_nonzero(residuals(i)) &&
                    inference_nonzero((*sqrt_weights)(i)) && inference_bad_product(scaled_u))
                    inference_scale_loss.store(true,std::memory_order_relaxed);
                double* score = aggregated.data() +
                                static_cast<std::size_t>(idx) * static_cast<std::size_t>(p);
                for (int j = 0; j < p; ++j) {
                    score[static_cast<std::size_t>(j)] += WX(i, j) * scaled_u;
                }
                double u_score = scaled_u;
                if (sqrt_weights) {
                    u_score *= (*sqrt_weights)(i);
                }
                aggregated_u[static_cast<std::size_t>(idx)] += u_score;
            }
        }
        Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(p, p);
        auto meat_lower = meat.selfadjointView<Eigen::Lower>();
        Eigen::VectorXd cluster_ux;
        if (cluster_ux_out) {
            cluster_ux = Eigen::VectorXd::Zero(p);
        }
        double cluster_u2 = 0.0;
        for (int g = 0; g < range; ++g) {
            if (!seen[static_cast<std::size_t>(g)]) {
                continue;
            }
            const Eigen::Map<const Eigen::VectorXd> score(
                aggregated.data() + static_cast<std::size_t>(g) * static_cast<std::size_t>(p), p);
            meat_lower.rankUpdate(score, 1.0);
            const double ug = aggregated_u[static_cast<std::size_t>(g)];
            cluster_u2 += ug * ug;
            if (cluster_ux_out) {
                cluster_ux.noalias() += ug * score;
            }
        }
        double scale = 1.0;
        const int G = next_cluster;
        if (num_clusters_out) {
            *num_clusters_out = G;
        }
        if (cluster_u2_out) require_cluster_auxiliary_range(cluster_u2,original_X,residuals,original_weights,*clusters);
        if (cluster_u2_out) {
            *cluster_u2_out = cluster_u2;
        }
        if (cluster_ux_out) {
            *cluster_ux_out = std::move(cluster_ux);
        }
        if (G > 1 && df_resid > 0.0) {
            scale = (static_cast<double>(G) / (G - 1.0)) *
                    ((n_effective - 1.0) / df_resid);
        }
        if (inference_meat_needs_wide(meat,original_X,residuals,original_weights,clusters,
                inference_scale_loss.load(std::memory_order_relaxed)) ||
            inference_sandwich_needs_wide(xtx_inv,meat)) throw OlsInferenceRangeError("cluster meat");
        cov = xtx_inv * meat_lower * xtx_inv;
        cov *= scale;
        if (!ieee_all_finite(cov)) throw OlsInferenceRangeError("covariance scaling overflow");
        return cov;
    }

    std::unordered_map<int, int> cluster_map;
    cluster_map.reserve(static_cast<std::size_t>(n));
    aggregated.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(p) /
                       static_cast<std::size_t>(8));
    aggregated_u.reserve(static_cast<std::size_t>(n) / static_cast<std::size_t>(8));
    for (int i = 0; i < n; ++i) {
        const int raw_id = (*clusters)(i);
        auto it = cluster_map.find(raw_id);
        int cluster_idx;
        if (it == cluster_map.end()) {
            cluster_idx = next_cluster;
            ++next_cluster;
            cluster_map.emplace(raw_id, cluster_idx);
            aggregated.resize(static_cast<std::size_t>(next_cluster) * static_cast<std::size_t>(p),
                              0.0);
            aggregated_u.resize(static_cast<std::size_t>(next_cluster), 0.0);
        } else {
            cluster_idx = it->second;
        }
        const double scaled_u = sqrt_weights ? residuals(i) * (*sqrt_weights)(i) : residuals(i);
        if (sqrt_weights && inference_nonzero(residuals(i)) &&
            inference_nonzero((*sqrt_weights)(i)) && inference_bad_product(scaled_u))
            inference_scale_loss.store(true,std::memory_order_relaxed);
        double* score = aggregated.data() +
                        static_cast<std::size_t>(cluster_idx) * static_cast<std::size_t>(p);
        for (int j = 0; j < p; ++j) {
            score[static_cast<std::size_t>(j)] += WX(i, j) * scaled_u;
        }
        double u_score = scaled_u;
        if (sqrt_weights) {
            u_score *= (*sqrt_weights)(i);
        }
        aggregated_u[static_cast<std::size_t>(cluster_idx)] += u_score;
    }
    Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(p, p);
    auto meat_lower = meat.selfadjointView<Eigen::Lower>();
    Eigen::VectorXd cluster_ux;
    if (cluster_ux_out) {
        cluster_ux = Eigen::VectorXd::Zero(p);
    }
    double cluster_u2 = 0.0;
    for (int g = 0; g < next_cluster; ++g) {
        const Eigen::Map<const Eigen::VectorXd> score(
            aggregated.data() + static_cast<std::size_t>(g) * static_cast<std::size_t>(p), p);
        meat_lower.rankUpdate(score, 1.0);
        const double ug = aggregated_u[static_cast<std::size_t>(g)];
        cluster_u2 += ug * ug;
        if (cluster_ux_out) {
            cluster_ux.noalias() += ug * score;
        }
    }
    double scale = 1.0;
    const int G = next_cluster;
    if (num_clusters_out) {
        *num_clusters_out = G;
    }
    if (cluster_u2_out) require_cluster_auxiliary_range(cluster_u2,original_X,residuals,original_weights,*clusters);
    if (cluster_u2_out) {
        *cluster_u2_out = cluster_u2;
    }
    if (cluster_ux_out) {
        *cluster_ux_out = std::move(cluster_ux);
    }
    if (G > 1 && df_resid > 0.0) {
        scale = (static_cast<double>(G) / (G - 1.0)) *
                ((n_effective - 1.0) / df_resid);
    }
    if (inference_meat_needs_wide(meat,original_X,residuals,original_weights,clusters,
            inference_scale_loss.load(std::memory_order_relaxed)) ||
        inference_sandwich_needs_wide(xtx_inv,meat)) throw OlsInferenceRangeError("cluster meat");
    cov = xtx_inv * meat_lower * xtx_inv;
    cov *= scale;
    if (!ieee_all_finite(cov)) throw OlsInferenceRangeError("covariance scaling overflow");
    return cov;
}

Eigen::MatrixXd compute_cluster_meat(const Eigen::MatrixXd& WX,
                                     const Eigen::Ref<const Eigen::MatrixXd>& original_X,
                                     const Eigen::VectorXd& residuals,
                                     const Eigen::VectorXd* original_weights,
                                     const Eigen::VectorXd* sqrt_weights,
                                     const Eigen::VectorXi& clusters,
                                     int* num_clusters_out) {
    const int p = static_cast<int>(WX.cols());
    const int n = static_cast<int>(WX.rows());
    if (clusters.size() != n) {
        throw std::runtime_error("Cluster vector length must equal the number of observations");
    }
    int min_id = clusters(0);
    int max_id = clusters(0);
    int adjacent_matches = 0;
    for (int i = 1; i < n; ++i) {
        const int v = clusters(i);
        min_id = std::min(min_id, v);
        max_id = std::max(max_id, v);
        if (v == clusters(i - 1)) {
            ++adjacent_matches;
        }
    }
    const long long range_ll =
        static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL;
    constexpr long long kDenseRangeCap = 50000000LL;
    const bool dense_ok =
        min_id >= 0 && range_ll > 0 && range_ll <= kDenseRangeCap &&
        range_ll <= static_cast<long long>(n) * 2LL;
    const int range = dense_ok ? static_cast<int>(range_ll) : 0;

    bool scale_loss=false;
    std::vector<double> aggregated;
    int next_cluster = 0;
    if (dense_ok) {
        const double run_frac =
            (n > 1) ? static_cast<double>(adjacent_matches) / static_cast<double>(n - 1) : 0.0;
        constexpr double kRunFastThreshold = 0.20;
        const bool use_run_fast = (run_frac >= kRunFastThreshold);
        aggregated.assign(static_cast<std::size_t>(range) * static_cast<std::size_t>(p), 0.0);
        std::vector<uint8_t> seen(static_cast<std::size_t>(range), 0);
        if (use_run_fast) {
            int i = 0;
            if (p <= 8) {
                while (i < n) {
                    const int idx = clusters(i) - min_id;
                    if (!seen[static_cast<std::size_t>(idx)]) {
                        seen[static_cast<std::size_t>(idx)] = 1;
                        ++next_cluster;
                    }

                    double run_score[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    do {
                        const double scaled_u =
                            sqrt_weights ? residuals(i) * (*sqrt_weights)(i) : residuals(i);
                        if (sqrt_weights && inference_nonzero(residuals(i)) &&
                            inference_nonzero((*sqrt_weights)(i)) && inference_bad_product(scaled_u)) scale_loss=true;
                        for (int j = 0; j < p; ++j) {
                            run_score[j] += WX(i, j) * scaled_u;
                        }
                        ++i;
                    } while (i < n && (clusters(i) - min_id) == idx);

                    double* score = aggregated.data() +
                                    static_cast<std::size_t>(idx) * static_cast<std::size_t>(p);
                    for (int j = 0; j < p; ++j) {
                        score[static_cast<std::size_t>(j)] += run_score[j];
                    }
                }
            } else {
                std::vector<double> run_score(static_cast<std::size_t>(p), 0.0);
                while (i < n) {
                    const int idx = clusters(i) - min_id;
                    if (!seen[static_cast<std::size_t>(idx)]) {
                        seen[static_cast<std::size_t>(idx)] = 1;
                        ++next_cluster;
                    }

                    std::fill(run_score.begin(), run_score.end(), 0.0);
                    do {
                        const double scaled_u =
                            sqrt_weights ? residuals(i) * (*sqrt_weights)(i) : residuals(i);
                        if (sqrt_weights && inference_nonzero(residuals(i)) &&
                            inference_nonzero((*sqrt_weights)(i)) && inference_bad_product(scaled_u)) scale_loss=true;
                        for (int j = 0; j < p; ++j) {
                            run_score[static_cast<std::size_t>(j)] += WX(i, j) * scaled_u;
                        }
                        ++i;
                    } while (i < n && (clusters(i) - min_id) == idx);

                    double* score = aggregated.data() +
                                    static_cast<std::size_t>(idx) * static_cast<std::size_t>(p);
                    for (int j = 0; j < p; ++j) {
                        score[static_cast<std::size_t>(j)] +=
                            run_score[static_cast<std::size_t>(j)];
                    }
                }
            }
        } else {
            for (int i = 0; i < n; ++i) {
                const int idx = clusters(i) - min_id;
                if (!seen[static_cast<std::size_t>(idx)]) {
                    seen[static_cast<std::size_t>(idx)] = 1;
                    ++next_cluster;
                }
                const double scaled_u =
                    sqrt_weights ? residuals(i) * (*sqrt_weights)(i) : residuals(i);
                if (sqrt_weights && inference_nonzero(residuals(i)) &&
                    inference_nonzero((*sqrt_weights)(i)) && inference_bad_product(scaled_u)) scale_loss=true;
                double* score = aggregated.data() +
                                static_cast<std::size_t>(idx) * static_cast<std::size_t>(p);
                for (int j = 0; j < p; ++j) {
                    score[static_cast<std::size_t>(j)] += WX(i, j) * scaled_u;
                }
            }
        }
        Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(p, p);
        auto meat_lower = meat.selfadjointView<Eigen::Lower>();
        for (int g = 0; g < range; ++g) {
            if (!seen[static_cast<std::size_t>(g)]) {
                continue;
            }
            const Eigen::Map<const Eigen::VectorXd> score(
                aggregated.data() + static_cast<std::size_t>(g) * static_cast<std::size_t>(p), p);
            meat_lower.rankUpdate(score, 1.0);
        }
        if (num_clusters_out) {
            *num_clusters_out = next_cluster;
        }
        if (inference_meat_needs_wide(meat,original_X,residuals,original_weights,&clusters,scale_loss))
            throw OlsInferenceRangeError("multiway component meat");
        return meat.selfadjointView<Eigen::Lower>();
    }

    std::unordered_map<int, int> cluster_map;
    cluster_map.reserve(static_cast<std::size_t>(n));
    aggregated.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(p) /
                       static_cast<std::size_t>(8));
    for (int i = 0; i < n; ++i) {
        const int raw_id = clusters(i);
        auto it = cluster_map.find(raw_id);
        int cluster_idx;
        if (it == cluster_map.end()) {
            cluster_idx = next_cluster;
            ++next_cluster;
            cluster_map.emplace(raw_id, cluster_idx);
            aggregated.resize(static_cast<std::size_t>(next_cluster) * static_cast<std::size_t>(p),
                              0.0);
        } else {
            cluster_idx = it->second;
        }
        const double scaled_u = sqrt_weights ? residuals(i) * (*sqrt_weights)(i) : residuals(i);
        if (sqrt_weights && inference_nonzero(residuals(i)) &&
            inference_nonzero((*sqrt_weights)(i)) && inference_bad_product(scaled_u)) scale_loss=true;
        double* score = aggregated.data() +
                        static_cast<std::size_t>(cluster_idx) * static_cast<std::size_t>(p);
        for (int j = 0; j < p; ++j) {
            score[static_cast<std::size_t>(j)] += WX(i, j) * scaled_u;
        }
    }
    Eigen::MatrixXd meat = Eigen::MatrixXd::Zero(p, p);
    auto meat_lower = meat.selfadjointView<Eigen::Lower>();
    for (int g = 0; g < next_cluster; ++g) {
        const Eigen::Map<const Eigen::VectorXd> score(
            aggregated.data() + static_cast<std::size_t>(g) * static_cast<std::size_t>(p), p);
        meat_lower.rankUpdate(score, 1.0);
    }
    if (num_clusters_out) {
        *num_clusters_out = next_cluster;
    }
    if (inference_meat_needs_wide(meat,original_X,residuals,original_weights,&clusters,scale_loss))
        throw OlsInferenceRangeError("multiway component meat");
    return meat.selfadjointView<Eigen::Lower>();
}

Eigen::VectorXi combine_clusters(const std::vector<Eigen::VectorXi>& clusters,
                                 const std::vector<int>& dims) {
    if (dims.empty()) {
        throw std::runtime_error("Cannot combine an empty set of cluster dimensions");
    }
    const int n = clusters[static_cast<std::size_t>(dims[0])].size();
    for (const int d : dims) {
        if (clusters[static_cast<std::size_t>(d)].size() != n) {
            throw std::runtime_error("All cluster vectors must have the same length");
        }
    }
    if (dims.size() == 1) {
        return clusters[static_cast<std::size_t>(dims[0])];
    }

    Eigen::VectorXi current = clusters[static_cast<std::size_t>(dims[0])];
    for (std::size_t pos = 1; pos < dims.size(); ++pos) {
        const Eigen::VectorXi& next = clusters[static_cast<std::size_t>(dims[pos])];
        std::unordered_map<std::uint64_t, int> map;
        map.reserve(static_cast<std::size_t>(n));
        Eigen::VectorXi combined(n);
        int next_id = 0;
        for (int i = 0; i < n; ++i) {
            const std::uint32_t a = static_cast<std::uint32_t>(current(i));
            const std::uint32_t b = static_cast<std::uint32_t>(next(i));
            const std::uint64_t key = (static_cast<std::uint64_t>(a) << 32) | b;
            auto it = map.find(key);
            if (it == map.end()) {
                map.emplace(key, next_id);
                combined(i) = next_id;
                ++next_id;
            } else {
                combined(i) = it->second;
            }
        }
        current = std::move(combined);
    }
    return current;
}

Eigen::MatrixXd compute_covariance_multiway(const Eigen::MatrixXd& xtx_inv,
                                            const Eigen::MatrixXd& WX,
                                            const Eigen::Ref<const Eigen::MatrixXd>& original_X,
                                            const Eigen::VectorXd& residuals,
                                            const Eigen::VectorXd* original_weights,
                                            const Eigen::VectorXd* sqrt_weights,
                                            const std::vector<Eigen::VectorXi>& clusters,
                                            double df_resid,
                                            double n_effective,
                                            int* num_clusters_out,
                                            ClusterDofMethod g_df,
                                            bool g_adj) {
    const int p = static_cast<int>(WX.cols());
    const int n = static_cast<int>(WX.rows());
    const int m = static_cast<int>(clusters.size());
    if (m <= 0) {
        throw std::runtime_error("At least one cluster dimension is required");
    }
    if (m > 20) {
        throw std::runtime_error("Multi-way clustering supports up to 20 cluster dimensions");
    }
    for (const auto& c : clusters) {
        if (c.size() != n) {
            throw std::runtime_error("Cluster vector length must equal the number of observations");
        }
    }

    int min_clusters = std::numeric_limits<int>::max();
    for (const auto& c : clusters) {
        std::unordered_map<int, int> uniq;
        uniq.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            uniq.emplace(c(i), 1);
        }
        min_clusters = std::min(min_clusters, static_cast<int>(uniq.size()));
    }
    if (num_clusters_out) {
        *num_clusters_out = (min_clusters == std::numeric_limits<int>::max()) ? 0 : min_clusters;
    }

    Eigen::MatrixXd meat_total = Eigen::MatrixXd::Zero(p, p);
    const std::uint64_t max_mask = (static_cast<std::uint64_t>(1) << m);
    std::vector<int> dims;
    dims.reserve(static_cast<std::size_t>(m));
    for (std::uint64_t mask = 1; mask < max_mask; ++mask) {
        dims.clear();
        for (int j = 0; j < m; ++j) {
            if (mask & (static_cast<std::uint64_t>(1) << j)) {
                dims.push_back(j);
            }
        }
        Eigen::VectorXi combined = combine_clusters(clusters, dims);
        int G = 0;
        Eigen::MatrixXd meat = compute_cluster_meat(WX,original_X,residuals,original_weights,sqrt_weights,combined,&G);
        double scale = 1.0;
        if (g_df == ClusterDofMethod::Conventional && g_adj && G > 1) {
            scale = static_cast<double>(G) / (G - 1.0);
        }
        const double sign = (dims.size() % 2 == 1) ? 1.0 : -1.0;
        meat_total.noalias() += sign * scale * meat;
    }

    double scale = 1.0;
    if (df_resid > 0.0) {
        scale *= (n_effective - 1.0) / df_resid;
    }
    if (g_df == ClusterDofMethod::Min && g_adj && min_clusters > 1) {
        scale *= static_cast<double>(min_clusters) / (min_clusters - 1.0);
    }

    if (inference_sandwich_needs_wide(xtx_inv,meat_total)) throw OlsInferenceRangeError("multiway sandwich products");
    Eigen::MatrixXd cov = xtx_inv * meat_total * xtx_inv;
    cov *= scale;
    if (!ieee_all_finite(cov)) throw OlsInferenceRangeError("multiway covariance scaling overflow");
    return cov;
}

OlsResult run_ols_wide(const Eigen::VectorXd& y,const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::VectorXd* weights,const Eigen::VectorXi* cluster,
    const std::vector<Eigen::VectorXi>* multiway,StandardErrorType se_type,
    double tss,double within_tss,double n_effective,bool frequency,
    const Eigen::Ref<const Eigen::MatrixXd>* residual_design,ParallelWorkObserver* observer,
    ClusterDofMethod g_df,bool g_adj,const Eigen::MatrixXd* score_low,
    const Eigen::VectorXd* normalization_means,
    Eigen::VectorXd* intercept_covariance) {
    using Number=OlsWide;
    using Matrix=std::vector<Number>;
    const int n=static_cast<int>(y.size()),p=static_cast<int>(X.cols());
    const std::size_t square=static_cast<std::size_t>(p)*p;
    auto fail=[](const char* why) {throw std::runtime_error(std::string("OLS extended precision: ")+why+"; no estimates returned");};
    if(!ieee_all_finite(y) || !ieee_all_finite(X)) fail("input is not finite");
    if(weights && (weights->size()!=n || !ieee_all_finite(*weights) || weights->minCoeff()<0)) fail("invalid weights");
    if(residual_design && !ieee_all_finite(*residual_design)) fail("actual-regressor design is not finite");
    if(score_low && (score_low->rows()!=n || score_low->cols()!=p || !ieee_all_finite(*score_low))) fail("invalid score low part");
    const bool capture_intercept = normalization_means && intercept_covariance;
    if(capture_intercept && (normalization_means->size()!=p || !ieee_all_finite(*normalization_means))) fail("invalid intercept normalization means");
    auto score_coordinate=[&](int row,int column) {
        return score_low ? Number::pair(X(row,column),(*score_low)(row,column)) : Number(X(row,column));
    };
    std::vector<double> scales(p),origins(p,0.0);
    std::vector<Number> shifts(p);
    int constant=-1;
    for(int j=0;j<p && constant<0;++j)
        if(X(0,j)!=0.0 && (X.col(j).array()==X(0,j)).all() &&
           (!score_low || (score_low->col(j).array()==(*score_low)(0,j)).all())) constant=j;
    if(constant>=0) for(int j=0;j<p;++j) if(j!=constant) {
        origins[j]=X(0,j);shifts[j]=Number(origins[j])/score_coordinate(0,constant);
    }
    auto coordinate=[&](int row,int column,bool actual) {
        const bool original=actual && residual_design;
        Number value=original ? Number((*residual_design)(row,column)) : score_coordinate(row,column);
        if(constant>=0 && column!=constant) {
            const Number c=original ? Number((*residual_design)(row,constant)) : score_coordinate(row,constant);
            const Number origin=score_coordinate(0,constant);
            if(c.hi==origin.hi && c.lo==origin.lo) value-=Number(origins[column]);
            else value-=shifts[column]*c;
        }
        return value;
    };
    Matrix inverse_scales(p);
    for(int j=0;j<p;++j) {
        scales[j]=0.0;double smallest=std::numeric_limits<double>::infinity();
        for(int row=0;row<n;++row) {
            const Number value=coordinate(row,j,false);const double magnitude=std::abs(value.value());
            if (!value.finite()) throw OlsInferenceRangeError("wide design centering overflow");
            if (inference_nonzero(value) && !inference_nonzero(magnitude))
                throw OlsInferenceRangeError("wide design normalization lost a nonzero coordinate");
            if (constant>=0 && j!=constant && !inference_nonzero(value) && X(row,j)!=origins[j])
                throw OlsInferenceRangeError("wide design centering lost a nonzero coordinate");
            scales[j]=std::max(scales[j],magnitude);
            if (inference_nonzero(magnitude)) smallest=std::min(smallest,magnitude);
        }
        if(!(scales[j]>0)) fail("selected design is rank deficient");
        inverse_scales[j]=Number(1)/Number(scales[j]);
        (void)checked_inference_product(Number(smallest),inverse_scales[j]);
    }
    const double y_origin=constant>=0 ? y[0] : 0.0;
    const Number response_shift=constant>=0 ? Number(y_origin)/score_coordinate(0,constant) : Number();
    auto response_coordinate=[&](int row,bool actual) {
        Number value(y[row]);
        if(actual && residual_design && constant>=0 &&
           ((*residual_design)(row,constant)!=X(0,constant) || (score_low && (*score_low)(0,constant)!=0.0)))
            value-=response_shift*Number((*residual_design)(row,constant));
        else value-=Number(y_origin);
        return value;
    };
    double y_scale=0,smallest_y=std::numeric_limits<double>::infinity();
    for(int row=0;row<n;++row) {
        const Number value=response_coordinate(row,false);const double magnitude=std::abs(value.value());
        if (!value.finite()) throw OlsInferenceRangeError("wide response centering overflow");
        if ((inference_nonzero(value) && !inference_nonzero(magnitude)) ||
            (constant>=0 && !inference_nonzero(value) && y[row]!=y_origin))
            throw OlsInferenceRangeError("wide response normalization lost a nonzero coordinate");
        y_scale=std::max(y_scale,magnitude);
        if (inference_nonzero(magnitude)) smallest_y=std::min(smallest_y,magnitude);
    }
    if(!(y_scale>0)) y_scale=1;
    const Number inverse_y=Number(1)/Number(y_scale);
    if (ieee_finite(smallest_y)) (void)checked_inference_product(Number(smallest_y),inverse_y);
    int threads=1;
#ifdef HDFE_USE_OPENMP
    threads=std::max(1,omp_get_max_threads());
#endif
    const int chunks=deterministic_ols_chunk_count(n);
    const std::size_t stride=square+p+(capture_intercept ? 1 : 0);
    Matrix partial(static_cast<std::size_t>(chunks)*stride);
    {
        ObservedOlsRegion observed(observer,chunks>0,threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for(int chunk=0;chunk<chunks;++chunk) {
            observed.observe_work();Matrix row(p);
            auto* values=partial.data()+static_cast<std::size_t>(chunk)*stride;
            for(int i=deterministic_ols_chunk_begin(n,chunk,chunks);i<deterministic_ols_chunk_end(n,chunk,chunks);++i) {
                const Number w(weights ? (*weights)[i] : 1),response=response_coordinate(i,false)*inverse_y;
                if(capture_intercept) values[square+p]+=w;
                for(int j=0;j<p;++j) row[j]=coordinate(i,j,false)*inverse_scales[j];
                for(int j=0;j<p;++j) {
                    values[square+j]+=w*row[j]*response;
                    for(int k=0;k<=j;++k) values[static_cast<std::size_t>(j)*p+k]+=w*row[j]*row[k];
                }
            }
        }
    }
    Matrix gram(square),rhs(p);
    Number intercept_weight_sum;
    for(int chunk=0;chunk<chunks;++chunk) {
        const auto* values=partial.data()+static_cast<std::size_t>(chunk)*stride;
        if(capture_intercept) intercept_weight_sum+=values[square+p];
        for(int j=0;j<p;++j) {
            rhs[j]+=values[square+j];
            for(int k=0;k<=j;++k) gram[static_cast<std::size_t>(j)*p+k]+=values[static_cast<std::size_t>(j)*p+k];
        }
    }
    Matrix lower(square),diagonal(p);
    for(int k=0;k<p;++k) {
        Number pivot=gram[static_cast<std::size_t>(k)*p+k];
        for(int j=0;j<k;++j) pivot-=lower[static_cast<std::size_t>(k)*p+j]*lower[static_cast<std::size_t>(k)*p+j]*diagonal[j];
        if(!pivot.finite() || !(pivot.value()>0)) fail("cannot establish positive numerical rank");
        diagonal[k]=pivot;lower[static_cast<std::size_t>(k)*p+k]=Number(1);
        for(int i=k+1;i<p;++i) {
            Number value=gram[static_cast<std::size_t>(i)*p+k];
            for(int j=0;j<k;++j) value-=lower[static_cast<std::size_t>(i)*p+j]*diagonal[j]*lower[static_cast<std::size_t>(k)*p+j];
            lower[static_cast<std::size_t>(i)*p+k]=value/pivot;
        }
    }
    auto solve=[&](Matrix value) {
        for(int i=0;i<p;++i) for(int j=0;j<i;++j) value[i]-=lower[static_cast<std::size_t>(i)*p+j]*value[j];
        for(int i=0;i<p;++i) value[i]=value[i]/diagonal[i];
        for(int i=p-1;i>=0;--i) for(int j=i+1;j<p;++j) value[i]-=lower[static_cast<std::size_t>(j)*p+i]*value[j];
        for(const auto& v:value) if(!v.finite()) fail("linear solve exceeded its numerical range");
        return value;
    };
    const Matrix beta=solve(rhs);
    Matrix bread(square);
    for(int j=0;j<p;++j) {
        Matrix unit(p);unit[j]=Number(1);const auto column=solve(std::move(unit));
        for(int i=0;i<p;++i) bread[static_cast<std::size_t>(i)*p+j]=column[i];
    }
    Matrix transform(constant>=0 ? square : 0);
    OlsResult result;result.coefficients.resize(p);result.residuals.resize(n);
    if(constant>=0) {
        for(int j=0;j<p;++j) {
            transform[static_cast<std::size_t>(j)*p+j]=inverse_scales[j];
            if(j!=constant) transform[static_cast<std::size_t>(constant)*p+j]=checked_inference_product(-shifts[j],inverse_scales[j]);
        }
        for(int i=0;i<p;++i) {
            Number value;
            for(int j=0;j<p;++j) value=checked_inference_add(value,checked_inference_product(
                checked_inference_product(transform[static_cast<std::size_t>(i)*p+j],beta[j]),Number(y_scale)));
            if(i==constant) value=checked_inference_add(value,response_shift);
            result.coefficients[i]=value.value();
        }
    } else for(int i=0;i<p;++i)
        result.coefficients[i]=checked_inference_product(checked_inference_product(beta[i],Number(y_scale)),inverse_scales[i]).value();
    Matrix residual(n),rss_chunks(chunks),meat(square);
    Matrix meat_chunks(se_type==StandardErrorType::Robust ? static_cast<std::size_t>(chunks)*square : 0);
    Matrix intercept_cross(capture_intercept ? p : 0);
    Number intercept_total;
    Matrix intercept_chunks(capture_intercept && se_type==StandardErrorType::Robust ?
        static_cast<std::size_t>(chunks)*(p+1) : 0);
    std::atomic<bool> wide_scale_loss{false};
    std::shared_ptr<N1NormalEvidence> n1_normal;
    if(n1_capture_active) n1_normal=std::make_shared<N1NormalEvidence>(chunks,p);
    {
        ObservedOlsRegion observed(observer,chunks>0,threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for(int chunk=0;chunk<chunks;++chunk) {
            observed.observe_work();Matrix row(p);
            N1NormalChunk* n1_chunk=n1_normal ? &n1_normal->chunks[static_cast<std::size_t>(chunk)] : nullptr;
            for(int i=deterministic_ols_chunk_begin(n,chunk,chunks);i<deterministic_ols_chunk_end(n,chunk,chunks);++i) {
                Number u=response_coordinate(i,true)*inverse_y;
                for(int j=0;j<p;++j) {
                    row[j]=coordinate(i,j,false)*inverse_scales[j];
                    const Number actual=residual_design ? coordinate(i,j,true)*inverse_scales[j] : row[j];
                    u-=actual*beta[j];
                }
                bool scale_loss=false;
                residual[i]=u;
                result.residuals[i]=wide_inference_product(u,Number(y_scale),scale_loss).value();
                const double w_value=weights ? (*weights)[i] : 1;
                const Number w(w_value);
                const Number weighted_u=wide_inference_product(w,u,scale_loss);
                const Number rss_term=wide_inference_product(weighted_u,u,scale_loss);
                rss_chunks[chunk]=wide_inference_add(rss_chunks[chunk],rss_term,scale_loss);
                if(n1_chunk) n1_chunk->add(y[i],result.residuals[i],w_value,p,
                    [&](int j) {return X(i,j);},
                    [&](int j) {return residual_design ? (*residual_design)(i,j) : X(i,j);},
                    [&](int j) {return result.coefficients[j];});
                if(se_type==StandardErrorType::Robust) {
                    bool has_score=false;
                    if (inference_nonzero(u) && (!weights || (*weights)[i]>0))
                        for (const auto& x:row) has_score=has_score || inference_nonzero(x);
                    Number multiplier;
                    if (has_score) {
                        const Number weight_factor=frequency ? w : wide_inference_product(w,w,scale_loss);
                        multiplier=wide_inference_product(wide_inference_product(weight_factor,u,scale_loss),u,scale_loss);
                    }
                    if(capture_intercept) {
                        Number mean_multiplier=multiplier;
                        if(!has_score && inference_nonzero(u) && (!weights || (*weights)[i]>0)) {
                            const Number factor=frequency ? w : wide_inference_product(w,w,scale_loss);
                            mean_multiplier=wide_inference_product(wide_inference_product(factor,u,scale_loss),u,scale_loss);
                        }
                        auto* auxiliary=intercept_chunks.data()+static_cast<std::size_t>(chunk)*(p+1);
                        auxiliary[p]+=mean_multiplier;
                        for(int j=0;j<p;++j) auxiliary[j]+=mean_multiplier*row[j];
                    }
                    auto* destination=meat_chunks.data()+static_cast<std::size_t>(chunk)*square;
                    for(int j=0;j<p;++j) {
                        // Only diagonal products need the extra range check;
                        // the p-by-p accumulation keeps its original formula.
                        (void)wide_inference_product(wide_inference_product(multiplier,row[j],scale_loss),row[j],scale_loss);
                        for(int k=0;k<=j;++k)
                            destination[static_cast<std::size_t>(j)*p+k]+=multiplier*row[j]*row[k];
                    }
                }
                if (scale_loss) wide_scale_loss.store(true,std::memory_order_relaxed);
            }
        }
    }
    Number rss;
    for(int chunk=0;chunk<chunks;++chunk) {
        rss+=rss_chunks[chunk];
        if(capture_intercept && se_type==StandardErrorType::Robust) {
            const auto* auxiliary=intercept_chunks.data()+static_cast<std::size_t>(chunk)*(p+1);
            intercept_total+=auxiliary[p];
            for(int j=0;j<p;++j) intercept_cross[j]+=auxiliary[j];
        }
        if(se_type==StandardErrorType::Robust)
            for(std::size_t j=0;j<square;++j) meat[j]+=meat_chunks[static_cast<std::size_t>(chunk)*square+j];
    }
    if (wide_scale_loss.load(std::memory_order_relaxed))
        throw OlsInferenceRangeError("wide residual/HC scale product lost range");
    result.rss=checked_inference_product(checked_inference_product(rss,Number(y_scale)),Number(y_scale)).value();
    if(!ieee_all_finite(result.residuals) || !ieee_finite(result.rss)) fail("residual output exceeded its numerical range");
    if (se_type==StandardErrorType::Robust) {
        for (int j=0;j<p;++j) {
            const auto& entry=meat[static_cast<std::size_t>(j)*p+j];
            const double diagonal=entry.value();
            if (!entry.finite() || diagonal<0 ||
                (inference_nonzero(entry) && diagonal<std::numeric_limits<double>::min()))
                throw OlsInferenceRangeError("wide HC meat outside resolved range");
            if (!inference_nonzero(entry)) for (int i=0;i<n;++i)
                if ((!weights || (*weights)[i]>0) && inference_nonzero(residual[i]) &&
                    inference_nonzero(coordinate(i,j,false)*inverse_scales[j]))
                    throw OlsInferenceRangeError("wide HC lost a positive meat diagonal");
        }
    }
    const double effective=n_effective>0 ? n_effective : n;
    const double df=std::max(0.0,effective-p);
    Matrix intercept_component_cross(capture_intercept ? p : 0);
    Number intercept_component_total;
    auto cluster_meat=[&](const Eigen::VectorXi& ids,bool auxiliary) {
        if(ids.size()!=n) fail("cluster labels do not align with observations");
        std::unordered_map<int,int> mapping;
        std::vector<Matrix> scores;Matrix totals;
        for(int row=0;row<n;++row) {
            auto found=mapping.find(ids[row]);
            if(found==mapping.end()) {
                const int id=static_cast<int>(scores.size());found=mapping.emplace(ids[row],id).first;
                scores.emplace_back(p);totals.emplace_back();
            }
            const int group=found->second;
            const Number wu=checked_inference_product(Number(weights ? (*weights)[row] : 1),residual[row]);
            totals[group]=checked_inference_add(totals[group],wu);
            for(int j=0;j<p;++j) scores[group][j]=checked_inference_add(scores[group][j],
                checked_inference_product(checked_inference_product(wu,coordinate(row,j,false)),inverse_scales[j]));
        }
        Matrix value(square),cross(p);Number total_u2;
        for(std::size_t g=0;g<scores.size();++g) {
            for(int j=0;j<p;++j) {
                if (inference_nonzero(scores[g][j]) &&
                    inference_bad_product((scores[g][j]*scores[g][j]).value()))
                    throw OlsInferenceRangeError("wide cluster score square lost range");
                for(int k=0;k<=j;++k) value[static_cast<std::size_t>(j)*p+k]+=scores[g][j]*scores[g][k];
                if(auxiliary || capture_intercept) cross[j]=checked_inference_add(cross[j],checked_inference_product(totals[g],scores[g][j]));
            }
            if(auxiliary || capture_intercept) total_u2=checked_inference_add(total_u2,checked_inference_product(totals[g],totals[g]));
        }
        for (int j=0;j<p;++j) {
            const auto& entry=value[static_cast<std::size_t>(j)*p+j];
            const double diagonal=entry.value();
            if (!entry.finite() || diagonal<0 ||
                (inference_nonzero(entry) && diagonal<std::numeric_limits<double>::min()))
                throw OlsInferenceRangeError("wide cluster component meat outside resolved range");
            if (!inference_nonzero(entry)) for (const auto& score:scores)
                if (inference_nonzero(score[j])) throw OlsInferenceRangeError("wide cluster lost a positive meat diagonal");
        }
        if(capture_intercept) {
            intercept_component_cross=cross;
            intercept_component_total=total_u2;
        }
        if(auxiliary) {
            result.cluster_ux.resize(p);
            for(int j=0;j<p;++j) result.cluster_ux[j]=checked_inference_product(checked_inference_product(
                checked_inference_add(checked_inference_product(cross[j],Number(scales[j])),
                    checked_inference_product(total_u2,Number(origins[j]))),Number(y_scale)),Number(y_scale)).value();
            result.cluster_u2=checked_inference_product(checked_inference_product(total_u2,Number(y_scale)),Number(y_scale)).value();
        }
        return std::make_pair(std::move(value),static_cast<int>(scores.size()));
    };
    Number adjustment(1);
    if(se_type==StandardErrorType::Cluster) {
        if(multiway) {
            const int dimensions=static_cast<int>(multiway->size());
            if(dimensions<1 || dimensions>20) fail("multiway clustering requires 1 to 20 dimensions");
            int minimum=std::numeric_limits<int>::max();
            for(std::uint64_t mask=1;mask<(1ULL<<dimensions);++mask) {
                std::vector<int> dims;
                for(int d=0;d<dimensions;++d) if(mask&(1ULL<<d)) dims.push_back(d);
                auto block=cluster_meat(combine_clusters(*multiway,dims),false);
                if(dims.size()==1) minimum=std::min(minimum,block.second);
                Number multiplier(dims.size()%2 ? 1 : -1);
                if(g_df==ClusterDofMethod::Conventional && g_adj && block.second>1)
                    multiplier=multiplier*Number(block.second)/Number(block.second-1);
                for(std::size_t j=0;j<square;++j)
                    meat[j]=checked_inference_add(meat[j],checked_inference_product(multiplier,block.first[j]));
                if(capture_intercept) {
                    intercept_total+=multiplier*intercept_component_total;
                    for(int j=0;j<p;++j) intercept_cross[j]+=multiplier*intercept_component_cross[j];
                }
            }
            result.num_clusters=minimum;
            if(df>0) adjustment=Number(effective-1)/Number(df);
            if(g_df==ClusterDofMethod::Min && g_adj && minimum>1)
                adjustment=adjustment*Number(minimum)/Number(minimum-1);
        } else {
            if(!cluster) fail("cluster labels were not supplied");
            auto block=cluster_meat(*cluster,true);meat=std::move(block.first);result.num_clusters=block.second;
            if(capture_intercept) {
                intercept_cross=intercept_component_cross;
                intercept_total=intercept_component_total;
            }
            if(block.second>1 && df>0)
                adjustment=Number(block.second)/Number(block.second-1)*Number(effective-1)/Number(df);
            result.cov_scale=adjustment.value();
        }
    }
    Matrix covariance(square);
    if(se_type==StandardErrorType::Homoskedastic) {
        const Number sigma=df>0 ? rss/Number(df) : Number();
        for(std::size_t j=0;j<square;++j) covariance[j]=checked_inference_product(sigma,bread[j]);
    } else {
        for(int i=0;i<p;++i) for(int j=i+1;j<p;++j) meat[static_cast<std::size_t>(i)*p+j]=meat[static_cast<std::size_t>(j)*p+i];
        Matrix intermediate(square);
        for(int i=0;i<p;++i) for(int j=0;j<p;++j) for(int k=0;k<p;++k)
            intermediate[static_cast<std::size_t>(i)*p+j]=checked_inference_add(intermediate[static_cast<std::size_t>(i)*p+j],
                checked_inference_product(bread[static_cast<std::size_t>(i)*p+k],meat[static_cast<std::size_t>(k)*p+j]));
        for(int i=0;i<p;++i) for(int j=0;j<p;++j) {
            for(int k=0;k<p;++k) covariance[static_cast<std::size_t>(i)*p+j]=checked_inference_add(
                covariance[static_cast<std::size_t>(i)*p+j],checked_inference_product(
                    intermediate[static_cast<std::size_t>(i)*p+k],bread[static_cast<std::size_t>(k)*p+j]));
            covariance[static_cast<std::size_t>(i)*p+j]=checked_inference_product(covariance[static_cast<std::size_t>(i)*p+j],adjustment);
        }
    }
    if(capture_intercept) {
        // Congruence of the augmented sandwich, before rounding bread/meat.
        // T maps normalized coefficients back to the original slope units.
        auto basis=[&](int row,int column) {
            return constant>=0 ? transform[static_cast<std::size_t>(row)*p+column] :
                (row==column ? inverse_scales[row] : Number());
        };
        Matrix means(p),cross(p),covariance_means(p);
        for(int j=0;j<p;++j) for(int i=0;i<p;++i)
            means[j]+=basis(i,j)*Number((*normalization_means)[i]);
        Number mean_variance;
        if(se_type==StandardErrorType::Homoskedastic) {
            mean_variance=df>0 ? (rss/Number(df))/intercept_weight_sum : Number();
        } else {
            mean_variance=(adjustment*intercept_total/intercept_weight_sum)/intercept_weight_sum;
            for(int i=0;i<p;++i) {
                for(int j=0;j<p;++j) cross[i]+=bread[static_cast<std::size_t>(i)*p+j]*intercept_cross[j];
                cross[i]=(adjustment*cross[i])/intercept_weight_sum;
            }
        }
        for(int i=0;i<p;++i) for(int j=0;j<p;++j)
            covariance_means[i]+=covariance[static_cast<std::size_t>(i)*p+j]*means[j];
        Number variance=mean_variance;
        for(int j=0;j<p;++j) variance+=means[j]*(covariance_means[j]-Number(2)*cross[j]);
        intercept_covariance->resize(p+1);
        for(int i=0;i<p;++i) {
            Number value;
            for(int j=0;j<p;++j) value+=basis(i,j)*(cross[j]-covariance_means[j]);
            (*intercept_covariance)[i]=checked_inference_product(
                checked_inference_product(value,Number(y_scale)),Number(y_scale)).value();
        }
        (*intercept_covariance)[p]=checked_inference_product(
            checked_inference_product(variance,Number(y_scale)),Number(y_scale)).value();
        if(!ieee_all_finite(*intercept_covariance)) fail("intercept inference exceeded its numerical range");
    }
    result.xtx_inv.resize(p,p);result.covariance.resize(p,p);
    if(constant>=0) {
        auto change_basis=[&](const Matrix& input,bool response_scale) {
            Matrix left(square),output(square);
            for(int i=0;i<p;++i) for(int j=0;j<p;++j) for(int k=0;k<p;++k) {
                const Number factor=response_scale ? checked_inference_product(transform[static_cast<std::size_t>(i)*p+k],Number(y_scale)) : transform[static_cast<std::size_t>(i)*p+k];
                left[static_cast<std::size_t>(i)*p+j]=checked_inference_add(left[static_cast<std::size_t>(i)*p+j],
                    checked_inference_product(factor,input[static_cast<std::size_t>(k)*p+j]));
            }
            for(int i=0;i<p;++i) for(int j=0;j<p;++j) for(int k=0;k<p;++k) {
                const Number factor=response_scale ? checked_inference_product(transform[static_cast<std::size_t>(j)*p+k],Number(y_scale)) : transform[static_cast<std::size_t>(j)*p+k];
                output[static_cast<std::size_t>(i)*p+j]=checked_inference_add(output[static_cast<std::size_t>(i)*p+j],
                    checked_inference_product(left[static_cast<std::size_t>(i)*p+k],factor));
            }
            return output;
        };
        const Matrix original_bread=change_basis(bread,false),original_covariance=change_basis(covariance,true);
        for(int i=0;i<p;++i) {
            for(int j=0;j<p;++j) {
                result.xtx_inv(i,j)=original_bread[static_cast<std::size_t>(i)*p+j].value();
                result.covariance(i,j)=original_covariance[static_cast<std::size_t>(i)*p+j].value();
            }
        }
    } else for(int i=0;i<p;++i) {
        for(int j=0;j<p;++j) {
            const auto at=static_cast<std::size_t>(i)*p+j;
            result.xtx_inv(i,j)=checked_inference_product(checked_inference_product(bread[at],inverse_scales[i]),inverse_scales[j]).value();
            result.covariance(i,j)=checked_inference_product(checked_inference_product(checked_inference_product(
                checked_inference_product(covariance[at],Number(y_scale)),inverse_scales[i]),Number(y_scale)),inverse_scales[j]).value();
        }
    }
    if(!ieee_all_finite(result.coefficients) || !ieee_all_finite(result.xtx_inv) || !ieee_all_finite(result.covariance) ||
       !ieee_all_finite(result.cluster_ux) || !ieee_finite(result.cluster_u2)) fail("inference output exceeded its numerical range");
    result.std_errors=result.covariance.diagonal().array().cwiseMax(0.0).sqrt();
    result.tvalues.resize(p);result.pvalues.resize(p);result.conf_int.resize(p,2);
    constexpr double z=1.959963984540054;
    for(int j=0;j<p;++j) {
        result.tvalues[j]=result.std_errors[j]>0 ? result.coefficients[j]/result.std_errors[j] : 0;
        result.pvalues[j]=result.std_errors[j]>0 ? 2*(1-normal_cdf(std::abs(result.tvalues[j]))) : 1;
        result.conf_int(j,0)=result.coefficients[j]-z*result.std_errors[j];
        result.conf_int(j,1)=result.coefficients[j]+z*result.std_errors[j];
    }
    result.df_resid=df;result.sigma2=df>0 ? result.rss/df : 0;result.nobs=n;
    result.tss=tss;result.within_tss=within_tss;
    result.r2=tss>0 ? 1-result.rss/tss : std::numeric_limits<double>::quiet_NaN();
    result.r2_within=within_tss>0 ? 1-result.rss/within_tss : std::numeric_limits<double>::quiet_NaN();
    result.n1_normal=std::move(n1_normal);
    return result;
}

}  // namespace

OlsResult precise_classical_reference(const Eigen::VectorXd& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const Eigen::VectorXd* weights,
    bool frequency,ParallelWorkObserver* observer) {
    const double effective=frequency && weights ? weights->sum() : y.size();
    return run_ols_wide(y,X,weights,nullptr,nullptr,StandardErrorType::Homoskedastic,
        0,0,effective,frequency,nullptr,observer);
}

bool exact_two_fe_span(const Eigen::Ref<const Eigen::VectorXd>& raw,
    const Eigen::VectorXi& first,const Eigen::VectorXi& second) {
    const int n=static_cast<int>(raw.size());
    std::unordered_map<int,std::size_t> a_ids,b_ids;
    std::vector<std::size_t> a(n),b(n);
    for(int row=0;row<n;++row) {
        a[row]=a_ids.emplace(first[row],a_ids.size()).first->second;
        b[row]=b_ids.emplace(second[row],b_ids.size()).first->second;
    }
    const std::size_t split=a_ids.size(),nodes=split+b_ids.size();
    std::vector<std::vector<int>> edges(nodes);
    for(int row=0;row<n;++row) {
        b[row]+=split;edges[a[row]].push_back(row);edges[b[row]].push_back(row);
    }
    std::vector<OlsWide> value(nodes);std::vector<unsigned char> seen(nodes,0);
    std::vector<std::size_t> queue;queue.reserve(nodes);
    for(std::size_t root=0;root<nodes;++root) if(!seen[root]) {
        queue.clear();queue.push_back(root);seen[root]=1;
        for(std::size_t next=0;next<queue.size();++next) {
            const auto node=queue[next];
            for(int row:edges[node]) {
                const auto other=a[row]==node ? b[row] : a[row];
                if(!seen[other]) {
                    value[other]=OlsWide(raw[row])-value[node];
                    if(!value[other].finite()) return false;
                    seen[other]=1;queue.push_back(other);
                }
            }
        }
    }
    // Exact binary verification makes this a proof, independent of the
    // precision of graph propagation, normalization or FE solver tolerances.
    for(int row=0;row<n;++row) {
        GroupExactDyadicSum residual;residual.add(raw[row]);
        residual.add(value[a[row]].hi,1,true);residual.add(value[a[row]].lo,1,true);
        residual.add(value[b[row]].hi,1,true);residual.add(value[b[row]].lo,1,true);
        if(!residual.exactly_zero()) return false;
    }
    return true;
}

bool exact_level_projection(const Eigen::VectorXd& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,const Eigen::VectorXi& fe,
    const Eigen::VectorXd* weights,const Eigen::VectorXd& within_y,
    const Eigen::Ref<const Eigen::MatrixXd>& within_X) {
    const int n=static_cast<int>(y.size()),rhs=static_cast<int>(X.cols())+1;
    std::unordered_map<int,std::vector<int>> groups;
    for(int row=0;row<n;++row) {
        if(weights) {
            const auto bits=ieee_bits_detail::bits((*weights)[row]);
            if((bits&0x7fffffffffffffffULL) && !(bits&0x7ff0000000000000ULL)) return false;
        }
        if(!weights || (*weights)[row]>0) groups[fe[row]].push_back(row);
    }
    auto input=[&](int row,int j) {return j ? X(row,j-1) : y[row];};
    auto residual=[&](int row,int j) {return j ? within_X(row,j-1) : within_y[row];};
    for(const auto& group:groups) {
        std::vector<OlsWide> alpha(rhs);
        std::vector<GroupExactDyadicSum> moments(rhs);
        for(int j=0;j<rhs;++j) alpha[j]=OlsWide(input(group.second[0],j))-OlsWide(residual(group.second[0],j));
        for(int row:group.second) for(int j=0;j<rhs;++j) {
            if(!alpha[j].finite()) return false;
            GroupExactDyadicSum affine;
            affine.add(input(row,j));affine.add(residual(row,j),1,true);
            affine.add(alpha[j].hi,1,true);affine.add(alpha[j].lo,1,true);
            if(!affine.exactly_zero()) return false;
            const double w=weights ? (*weights)[row] : 1.0,r=residual(row,j);
            const auto rbits=ieee_bits_detail::bits(r)&0x7fffffffffffffffULL;
            if(rbits && !(rbits&0x7ff0000000000000ULL)) return false;
            const double product=OlsWide::mul(w,r);
            const auto pbits=ieee_bits_detail::bits(product)&0x7fffffffffffffffULL;
            if(!ieee_finite(product) || (!pbits && w!=0 && rbits)) return false;
            // Keep the error-free product, including its low word, away from
            // subnormal flushing on release platforms. Otherwise decline proof.
            if(pbits && (pbits>>52)<123) return false;
            volatile double error=std::fma(w,r,-product);
            if(!ieee_finite(error)) return false;
            moments[j].add(product);moments[j].add(error);
        }
        for(auto& moment:moments) if(!moment.exactly_zero()) return false;
    }
    return !groups.empty();
}

std::vector<std::uint8_t> precise_ols_rank_mask(
    const Eigen::Ref<const Eigen::MatrixXd>& X,const std::vector<int>& columns,
    const Eigen::VectorXd* weights,bool center,double tolerance,
    const std::vector<int>& priority,int threads,ParallelWorkObserver* observer,
    Eigen::MatrixXd* directions,Eigen::MatrixXd* full_basis) {
    using Number=OlsWide;using Vector=std::vector<Number>;
    const int n=static_cast<int>(X.rows()),p=static_cast<int>(columns.size());
    std::vector<std::uint8_t> drop(p,0);
    if(directions) directions->resize(p,0);
    if(full_basis) full_basis->resize(p,0);
    if(!p) return drop;
    if(!ieee_all_finite(X) || (weights && !ieee_all_finite(*weights)))
        throw std::runtime_error("OLS rank check received non-finite values; no estimates returned");
    Vector origins(p),inverse_scale(p);
    for(int j=0;j<p;++j)
        origins[j]=Number(center ? X(0,columns[j]) : 0.0);
    const int chunks=deterministic_ols_chunk_count(n);
    std::vector<double> scale_chunks(static_cast<std::size_t>(chunks)*p,0.0);
    {
        ObservedOlsRegion observed(observer,chunks>0,threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for(int chunk=0;chunk<chunks;++chunk) {
            observed.observe_work();
            auto* scales=scale_chunks.data()+static_cast<std::size_t>(chunk)*p;
            const int begin=deterministic_ols_chunk_begin(n,chunk,chunks);
            const int end=deterministic_ols_chunk_end(n,chunk,chunks);
            for(int j=0;j<p;++j) {
                double scale=0;
                for(int i=begin;i<end;++i)
                    scale=std::max(scale,std::abs((Number(X(i,columns[j]))-origins[j]).value()));
                scales[j]=scale;
            }
        }
    }
    for(int j=0;j<p;++j) {
        double scale=0;
        for(int chunk=0;chunk<chunks;++chunk)
            scale=std::max(scale,scale_chunks[static_cast<std::size_t>(chunk)*p+j]);
        inverse_scale[j]=scale>0 ? Number(1)/Number(scale) : Number(1);
    }
    const std::size_t square=static_cast<std::size_t>(p)*p,stride=square+p+1;
    Vector partial(static_cast<std::size_t>(chunks)*stride);
    {
        ObservedOlsRegion observed(observer,chunks>0,threads);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for(int chunk=0;chunk<chunks;++chunk) {
            observed.observe_work();Vector row(p);
            auto* values=partial.data()+static_cast<std::size_t>(chunk)*stride;
            const int begin=deterministic_ols_chunk_begin(n,chunk,chunks);
            const int end=deterministic_ols_chunk_end(n,chunk,chunks);
            if(!weights) values[square+p]=Number(static_cast<double>(end-begin));
            for(int i=begin;i<end;++i) {
                const Number weight(weights ? (*weights)[i] : 1);
                if(weights) values[square+p]+=weight;
                for(int j=0;j<p;++j) row[j]=(Number(X(i,columns[j]))-origins[j])*inverse_scale[j];
                for(int j=0;j<p;++j) {
                    const Number weighted_row=weight*row[j];
                    if(center) values[square+j]+=weighted_row;
                    for(int k=0;k<=j;++k) values[static_cast<std::size_t>(j)*p+k]+=weighted_row*row[k];
                }
            }
        }
    }
    Vector sums(stride);
    for(int chunk=0;chunk<chunks;++chunk) for(std::size_t j=0;j<stride;++j)
        sums[j]+=partial[static_cast<std::size_t>(chunk)*stride+j];
    if(!sums[square+p].finite() || !(sums[square+p].value()>0))
        throw std::runtime_error("OLS rank check has invalid total weight; no estimates returned");
    for(int j=0;j<p;++j) for(int k=0;k<=j;++k) {
        const auto at=static_cast<std::size_t>(j)*p+k;
        if(center) sums[at]-=sums[square+j]*sums[square+k]/sums[square+p];
        sums[static_cast<std::size_t>(k)*p+j]=sums[at];
    }
    std::vector<int> order(p),kept;
    std::iota(order.begin(),order.end(),0);
    if(!priority.empty()) std::stable_sort(order.begin(),order.end(),[&](int a,int b) {
        const int pa=columns[a]<static_cast<int>(priority.size()) ? priority[columns[a]] : 0;
        const int pb=columns[b]<static_cast<int>(priority.size()) ? priority[columns[b]] : 0;
        return pa!=pb ? pa>pb : a<b;
    });
    Vector lower(square),diagonal(p);
    std::vector<Vector> combinations;
    std::vector<Eigen::VectorXd> contrasts,complete_basis;
    const Number limit=Number(tolerance)*Number(tolerance);
    for(int pos:order) {
        const Number base=sums[static_cast<std::size_t>(pos)*p+pos];
        if(!base.finite()) throw std::runtime_error("OLS rank calculation exceeded its numerical range; no estimates returned");
        if(!(base.value()>0)) {drop[pos]=1;continue;}
        Vector row(kept.size());Number residual=base;
        for(std::size_t j=0;j<kept.size();++j) {
            Number value=sums[static_cast<std::size_t>(pos)*p+kept[j]];
            for(std::size_t k=0;k<j;++k) value-=row[k]*diagonal[k]*lower[j*p+k];
            row[j]=value/diagonal[j];residual-=row[j]*row[j]*diagonal[j];
        }
        if(!residual.finite()) throw std::runtime_error("OLS rank precision is not attainable; no estimates returned");
        if((residual-limit*base).value()<=0) {drop[pos]=1;continue;}
        // Optional proof outputs; ordinary rank callers do not allocate them.
        if (directions || full_basis) {
            Vector combination(p);combination[pos]=Number(1);
            bool mixed=false;
            for (std::size_t k=0;k<row.size();++k) {
                mixed=mixed || row[k].value()!=0.0;
                for (int j=0;j<p;++j) combination[j]-=row[k]*combinations[k][j];
            }
            if (mixed || full_basis) {
                Eigen::VectorXd contrast(p);
                for (int j=0;j<p;++j) {
                    contrast[j]=(combination[j]*inverse_scale[j]).value();
                    if (!ieee_finite(contrast[j]))
                        throw std::runtime_error("OLS audit direction exceeded its numerical range");
                }
                if (directions && mixed) contrasts.push_back(contrast);
                if (full_basis) complete_basis.push_back(std::move(contrast));
            }
            combinations.push_back(std::move(combination));
        }
        const auto next=kept.size();diagonal[next]=residual;lower[next*p+next]=Number(1);
        for(std::size_t j=0;j<row.size();++j) lower[next*p+j]=row[j];
        kept.push_back(pos);
    }
    if (directions) {
        directions->resize(p,static_cast<int>(contrasts.size()));
        for (std::size_t j=0;j<contrasts.size();++j) directions->col(j)=contrasts[j];
    }
    if (full_basis) {
        full_basis->resize(p,static_cast<int>(complete_basis.size()));
        for (std::size_t j=0;j<complete_basis.size();++j) full_basis->col(j)=complete_basis[j];
    }
    return drop;
}

OlsResult run_ols_fast_from_xtx(const Eigen::VectorXd& y,
                                const Eigen::Ref<const Eigen::MatrixXd>& X,
                                const Eigen::VectorXd* weights,
                                StandardErrorType se_type,
                                double total_sum_squares,
                                double within_sum_squares,
                                const Eigen::MatrixXd& xtx,
                                const Eigen::VectorXd& xty,
                                double n_effective,
                                bool weights_are_frequencies,
                                bool num_threads_explicit,
                                ParallelWorkObserver* parallel_observer) {
    if (se_type == StandardErrorType::Cluster) {
        throw std::runtime_error("run_ols_fast_from_xtx does not support clustered inference");
    }
    const int n = static_cast<int>(y.size());
    if (X.rows() != n) {
        throw std::runtime_error("run_ols_fast_from_xtx called with inconsistent dimensions");
    }
    const int p = static_cast<int>(X.cols());
    if (p <= 0 || xtx.rows() != p || xtx.cols() != p || xty.size() != p) {
        throw std::runtime_error("run_ols_fast_from_xtx called with invalid dimensions");
    }
    const double n_eff = (n_effective > 0.0) ? n_effective : static_cast<double>(n);
    if (auto shift = find_ols_origin_shift(xtx, X))
        return run_ols_origin_shifted(*shift,y,X,weights,nullptr,nullptr,se_type,
            total_sum_squares,within_sum_squares,n_eff,weights_are_frequencies,
            nullptr,num_threads_explicit,parallel_observer);
    if(normal_equations_need_wide(xtx))
        return run_ols_wide(y,X,weights,nullptr,nullptr,se_type,total_sum_squares,
            within_sum_squares,n_eff,weights_are_frequencies,nullptr,parallel_observer);
    switch (p) {
        case 1:
            return run_ols_fast_from_xtx_impl<1>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies,
                                                 parallel_observer);
        case 2:
            return run_ols_fast_from_xtx_impl<2>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies,
                                                 parallel_observer);
        case 3:
            return run_ols_fast_from_xtx_impl<3>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies,
                                                 parallel_observer);
        case 4:
            return run_ols_fast_from_xtx_impl<4>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies,
                                                 parallel_observer);
        case 5:
            return run_ols_fast_from_xtx_impl<5>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies,
                                                 parallel_observer);
        case 6:
            return run_ols_fast_from_xtx_impl<6>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies,
                                                 parallel_observer);
        case 7:
            return run_ols_fast_from_xtx_impl<7>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies,
                                                 parallel_observer);
        case 8:
            return run_ols_fast_from_xtx_impl<8>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies,
                                                 parallel_observer);
        default:
            break;
    }
    return run_ols(y, X, weights, nullptr, se_type, total_sum_squares, within_sum_squares, n_eff,
                   weights_are_frequencies, nullptr, num_threads_explicit,
                   parallel_observer);
}

OlsResult run_ols(const Eigen::VectorXd& y,
                  const Eigen::Ref<const Eigen::MatrixXd>& X,
                  const Eigen::VectorXd* weights,
                  const Eigen::VectorXi* clusters,
                  StandardErrorType se_type,
                  double total_sum_squares,
                  double within_sum_squares,
                  double n_effective,
                  bool weights_are_frequencies,
                  const Eigen::MatrixXd* X_for_residuals,
                  bool num_threads_explicit,
                  ParallelWorkObserver* parallel_observer) {
    std::optional<Eigen::Ref<const Eigen::MatrixXd>> actual;
    if (X_for_residuals) actual.emplace(*X_for_residuals);
    return run_ols_with_score_low(y,X,weights,clusters,se_type,total_sum_squares,
        within_sum_squares,n_effective,weights_are_frequencies,actual ? &*actual : nullptr,
        num_threads_explicit,parallel_observer,nullptr);
}

OlsResult run_ols_with_score_low(const Eigen::VectorXd& y,
                  const Eigen::Ref<const Eigen::MatrixXd>& X,
                  const Eigen::VectorXd* weights,
                  const Eigen::VectorXi* clusters,
                  StandardErrorType se_type,
                  double total_sum_squares,
                  double within_sum_squares,
                  double n_effective,
                  bool weights_are_frequencies,
                  const Eigen::Ref<const Eigen::MatrixXd>* X_for_residuals,
                  bool num_threads_explicit,
                  ParallelWorkObserver* parallel_observer,
                           const Eigen::MatrixXd* score_low,
    const Eigen::VectorXd* normalization_means,
    Eigen::VectorXd* intercept_covariance) {
    if (intercept_covariance) intercept_covariance->resize(0);
    const int n = static_cast<int>(y.size());
    if (X.rows() != n) {
        throw std::runtime_error("OLS called with inconsistent dimensions");
    }
    const int p = static_cast<int>(X.cols());
    if (n == 0 || p == 0) {
        throw std::runtime_error("Need positive observations and regressors for estimation");
    }
    if (X_for_residuals &&
        (X_for_residuals->rows() != n || X_for_residuals->cols() != p)) {
        throw std::runtime_error("Residual design matrix must match X in run_ols");
    }
    const double n_eff = (n_effective > 0.0) ? n_effective : static_cast<double>(n);

    if (se_type != StandardErrorType::Cluster && clusters == nullptr &&
        X_for_residuals == nullptr) {
        switch (p) {
            case 1:
                return run_ols_fast_impl<1>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies,
                                            parallel_observer);
            case 2:
                return run_ols_fast_impl<2>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies,
                                            parallel_observer);
            case 3:
                return run_ols_fast_impl<3>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies,
                                            parallel_observer);
            case 4:
                return run_ols_fast_impl<4>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies,
                                            parallel_observer);
            case 5:
                return run_ols_fast_impl<5>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies,
                                            parallel_observer);
            case 6:
                return run_ols_fast_impl<6>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies,
                                            parallel_observer);
            case 7:
                return run_ols_fast_impl<7>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies,
                                            parallel_observer);
            case 8:
                return run_ols_fast_impl<8>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies,
                                            parallel_observer);
            default:
                break;
        }
    }

    // Large unweighted problems take a copy-free, thread-parallel path: WX/Wy
    // would be byte-for-byte copies of X/y that are only ever read, and the
    // Eigen GEMM/matvec passes below run single-threaded over multi-GB arrays.
    // The size threshold and the weighted serial path are auto-only
    // profitability choices. An explicit request selects the same
    // deterministic arithmetic graph whenever there are observations to
    // process; weighted score storage is still materialized for covariance.
    constexpr int kParallelOlsMinObs = 4194304;
    bool parallel_moments = false;
    int ols_threads = 1;
#ifdef HDFE_USE_OPENMP
    ols_threads = std::max(1, omp_get_max_threads());
    parallel_moments =
        num_threads_explicit ||
        ((weights == nullptr) && n >= kParallelOlsMinObs);
#endif

    Eigen::VectorXd Wy;
    Eigen::MatrixXd WX;
    Eigen::VectorXd sqrt_weights;
    Eigen::MatrixXd xtx;
    Eigen::VectorXd xty;
    if (!parallel_moments) {
        Wy = y;
        WX = X;
        if (weights) {
            if (weights->size() != n) {
                throw std::runtime_error("Weights must align with y in run_ols");
            }
            sqrt_weights = weights->array().sqrt();
            Wy.array() *= sqrt_weights.array();
            for (int j = 0; j < p; ++j) {
                WX.col(j).array() *= sqrt_weights.array();
            }
        }
        xtx = WX.transpose() * WX;
        xty = WX.transpose() * Wy;
    }
#ifdef HDFE_USE_OPENMP
    else {
        const double* weight_ptr = nullptr;
        if (weights) {
            if (weights->size() != n) {
                throw std::runtime_error(
                    "Weights must align with y in run_ols");
            }
            sqrt_weights = weights->array().sqrt();
            WX = X;
            for (int j = 0; j < p; ++j) {
                WX.col(j).array() *= sqrt_weights.array();
            }
            weight_ptr = weights->data();
        }
        // Fixed logical chunks make the arithmetic graph independent of the
        // number of OpenMP workers. Threads only execute chunks; the partials
        // are always combined in chunk-index order.
        const int chunks = deterministic_ols_chunk_count(n);
        std::vector<Eigen::MatrixXd> xtx_tls(static_cast<std::size_t>(chunks),
                                             Eigen::MatrixXd::Zero(p, p));
        std::vector<Eigen::VectorXd> xty_tls(static_cast<std::size_t>(chunks),
                                             Eigen::VectorXd::Zero(p));
        const double* y_ptr = y.data();
        const double* x_base = X.data();
        const Eigen::Index x_rows = X.rows();
        {
            ObservedOlsRegion observed(
                parallel_observer, chunks > 0, ols_threads);
#pragma omp parallel for schedule(static) num_threads(ols_threads)
            for (int chunk = 0; chunk < chunks; ++chunk) {
                observed.observe_work();
                Eigen::MatrixXd& xtx_local =
                    xtx_tls[static_cast<std::size_t>(chunk)];
                Eigen::VectorXd& xty_local =
                    xty_tls[static_cast<std::size_t>(chunk)];
                const int begin =
                    deterministic_ols_chunk_begin(n, chunk, chunks);
                const int end =
                    deterministic_ols_chunk_end(n, chunk, chunks);
                for (int i = begin; i < end; ++i) {
                    const double yi = y_ptr[i];
                    const double w = weight_ptr ? weight_ptr[i] : 1.0;
                    for (int j = 0; j < p; ++j) {
                        const double xij =
                            x_base[static_cast<Eigen::Index>(j) * x_rows + i];
                        xty_local(j) += w * xij * yi;
                        for (int k = 0; k <= j; ++k) {
                            xtx_local(j, k) +=
                                w * xij *
                                x_base[
                                    static_cast<Eigen::Index>(k) * x_rows + i];
                        }
                    }
                }
            }
        }
        xtx = Eigen::MatrixXd::Zero(p, p);
        xty = Eigen::VectorXd::Zero(p);
        for (int chunk = 0; chunk < chunks; ++chunk) {
            xtx.noalias() += xtx_tls[static_cast<std::size_t>(chunk)];
            xty.noalias() += xty_tls[static_cast<std::size_t>(chunk)];
        }
        for (int j = 0; j < p; ++j) {
            for (int k = j + 1; k < p; ++k) {
                xtx(j, k) = xtx(k, j);
            }
        }
    }
#endif

    if (!score_low && !normalization_means) {
        if (auto shift = find_ols_origin_shift(xtx, X))
            return run_ols_origin_shifted(*shift,y,X,weights,clusters,nullptr,se_type,
                total_sum_squares,within_sum_squares,n_eff,weights_are_frequencies,
                X_for_residuals,num_threads_explicit,parallel_observer);
    }
    if(normal_equations_need_wide(xtx))
        return run_ols_wide(y,X,weights,clusters,nullptr,se_type,total_sum_squares,
            within_sum_squares,n_eff,weights_are_frequencies,X_for_residuals,parallel_observer,ClusterDofMethod::Min,true,score_low,normalization_means,intercept_covariance);
    Eigen::LDLT<Eigen::MatrixXd> solver;
    solver.compute(xtx);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize X'X; matrix may be singular");
    }
    Eigen::VectorXd beta = solver.solve(xty);
    Eigen::MatrixXd xtx_inv = solver.solve(Eigen::MatrixXd::Identity(p, p));

    Eigen::VectorXd residuals(n);
    double rss = 0.0;
    std::shared_ptr<N1NormalEvidence> n1_normal;
    if (n1_capture_active) {
        const int evidence_chunks =
            parallel_moments ? deterministic_ols_chunk_count(n) : 1;
        n1_normal =
            std::make_shared<N1NormalEvidence>(evidence_chunks, p);
    }
    if (!parallel_moments) {
        Eigen::VectorXd fitted =
            X_for_residuals ? Eigen::VectorXd((*X_for_residuals) * beta)
                            : Eigen::VectorXd(X * beta);
        residuals = y - fitted;
        rss = n1_normal
                  ? weighted_sum_of_squares_with_n1(
                        y, X, X_for_residuals, beta, residuals, weights,
                        n1_normal->chunks[0])
                  : weighted_sum_of_squares(residuals, weights);
    }
#ifdef HDFE_USE_OPENMP
    else {
        // Each residual element is written by exactly one thread with the
        // same per-element column accumulation order; RSS chunks and their
        // combine order are fixed independently of the team size.
        const double* y_ptr = y.data();
        const double* x_base = X_for_residuals ? X_for_residuals->data() : X.data();
        const Eigen::Index x_rows = X_for_residuals ? X_for_residuals->outerStride() : X.outerStride();
        double* resid_ptr = residuals.data();
        const double* weight_ptr = weights ? weights->data() : nullptr;
        const int chunks = deterministic_ols_chunk_count(n);
        std::vector<long double> rss_tls(static_cast<std::size_t>(chunks), 0.0L);
        {
            ObservedOlsRegion observed(
                parallel_observer, chunks > 0, ols_threads);
#pragma omp parallel for schedule(static) num_threads(ols_threads)
            for (int chunk = 0; chunk < chunks; ++chunk) {
                observed.observe_work();
                N1NormalChunk* n1_chunk =
                    n1_normal
                        ? &n1_normal->chunks[static_cast<std::size_t>(chunk)]
                        : nullptr;
                KahanSum rss_local;
                const int begin =
                    deterministic_ols_chunk_begin(n, chunk, chunks);
                const int end =
                    deterministic_ols_chunk_end(n, chunk, chunks);
                for (int i = begin; i < end; ++i) {
                    double fitted_i = 0.0;
                    for (int j = 0; j < p; ++j) {
                        fitted_i +=
                            x_base[
                                static_cast<Eigen::Index>(j) * x_rows + i] *
                            beta(j);
                    }
                    const double u = y_ptr[i] - fitted_i;
                    resid_ptr[i] = u;
                    const long double w =
                        static_cast<long double>(
                            weight_ptr ? weight_ptr[i] : 1.0);
                    rss_local.add(
                        w * static_cast<long double>(u) *
                        static_cast<long double>(u));
                    if (n1_chunk) {
                        n1_chunk->add(
                            y_ptr[i], u,
                            weight_ptr ? weight_ptr[i] : 1.0, p,
                            [&](int j) {
                                return X(i, j);
                            },
                            [&](int j) {
                                return X_for_residuals
                                           ? (*X_for_residuals)(i, j)
                                           : X(i, j);
                            },
                            [&](int j) { return beta(j); });
                    }
                }
                rss_tls[static_cast<std::size_t>(chunk)] = rss_local.sum;
            }
        }
        long double rss_acc = 0.0L;
        for (long double v : rss_tls) {
            rss_acc += v;
        }
        rss = static_cast<double>(rss_acc);
    }
#endif
    const double df_resid = std::max(0.0, n_eff - static_cast<double>(p));
    const double sigma2 = (df_resid > 0.0) ? rss / df_resid
                                         : 0.0;

    const Eigen::VectorXd* sqrt_ptr = weights ? &sqrt_weights : nullptr;
    int num_clusters = 0;
    Eigen::VectorXd cluster_ux;
    double cluster_u2 = 0.0;
    // On the copy-free path WX was never materialized; the unweighted scores
    // read X directly (identical values to the historical WX copy).
    const Eigen::Ref<const Eigen::MatrixXd> WX_used =
        (parallel_moments && weights == nullptr)
            ? Eigen::Ref<const Eigen::MatrixXd>(X)
            : Eigen::Ref<const Eigen::MatrixXd>(WX);
    Eigen::MatrixXd covariance;
    try {
        covariance = compute_covariance(xtx_inv,WX_used,X,residuals,weights,sqrt_ptr,clusters,
            se_type,sigma2,df_resid,n_eff,&num_clusters,
            (se_type==StandardErrorType::Cluster) ? &cluster_ux : nullptr,
            (se_type==StandardErrorType::Cluster) ? &cluster_u2 : nullptr,
            weights_are_frequencies,num_threads_explicit,parallel_observer);
    } catch (const OlsInferenceRangeError&) {
        n1_normal.reset();
        return run_ols_wide(y,X,weights,clusters,nullptr,se_type,total_sum_squares,
            within_sum_squares,n_eff,weights_are_frequencies,X_for_residuals,parallel_observer,ClusterDofMethod::Min,true,score_low,normalization_means,intercept_covariance);
    }
    double cov_scale = 1.0;
    if (se_type == StandardErrorType::Cluster) {
        const int G = num_clusters;
        if (G > 1 && df_resid > 0.0) {
            cov_scale = (static_cast<double>(G) / (G - 1.0)) *
                        ((n_eff - 1.0) / df_resid);
        }
    }

    Eigen::VectorXd std_errors = covariance.diagonal().array().cwiseMax(0.0).sqrt();
    Eigen::VectorXd tvalues = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd pvalues = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd conf_int(p, 2);
    constexpr double kZ = 1.959963984540054;
    for (int j = 0; j < p; ++j) {
        if (std_errors(j) > 0) {
            tvalues(j) = beta(j) / std_errors(j);
            const double tail = 1.0 - normal_cdf(std::abs(tvalues(j)));
            pvalues(j) = 2.0 * tail;
        } else {
            tvalues(j) = 0.0;
            pvalues(j) = 1.0;
        }
        conf_int(j, 0) = beta(j) - kZ * std_errors(j);
        conf_int(j, 1) = beta(j) + kZ * std_errors(j);
    }

    OlsResult result;
    result.coefficients = std::move(beta);
    result.std_errors = std::move(std_errors);
    result.tvalues = std::move(tvalues);
    result.pvalues = std::move(pvalues);
    result.conf_int = std::move(conf_int);
    result.residuals = std::move(residuals);
    result.covariance = std::move(covariance);
    result.xtx_inv = std::move(xtx_inv);
    result.cluster_ux = std::move(cluster_ux);
    result.cluster_u2 = cluster_u2;
    result.cov_scale = cov_scale;
    result.df_resid = df_resid;
    result.rss = rss;
    result.tss = total_sum_squares;
    result.within_tss = within_sum_squares;
    const double missing_r2 = std::numeric_limits<double>::quiet_NaN();
    result.r2 = (total_sum_squares > 0.0)
                    ? 1.0 - rss / total_sum_squares
                    : missing_r2;
    result.r2_within = (within_sum_squares > 0.0)
                           ? 1.0 - rss / within_sum_squares
                           : missing_r2;
    result.sigma2 = sigma2;
    result.nobs = n;
    result.num_clusters = num_clusters;
    result.n1_normal = std::move(n1_normal);
    return result;
}

Eigen::MatrixXd multiway_cluster_sandwich(
    const Eigen::MatrixXd& bread,
    const Eigen::Ref<const Eigen::MatrixXd>& scores,
    const Eigen::VectorXd& residuals,
    const Eigen::VectorXd* weights,
    const std::vector<Eigen::VectorXi>& clusters,
    ClusterDofMethod g_df,
    bool g_adj) {
    const int n = static_cast<int>(scores.rows());
    const int p = static_cast<int>(scores.cols());
    if (n == 0 || p == 0 || residuals.size() != n) {
        throw std::runtime_error("Invalid inputs for multiway covariance rebuild");
    }
    if (bread.rows() != p || bread.cols() != p) {
        throw std::runtime_error("Bread dimension mismatch in multiway covariance rebuild");
    }
    Eigen::MatrixXd WS = scores;
    Eigen::VectorXd sqrt_weights;
    if (weights) {
        if (weights->size() != n) {
            throw std::runtime_error("Weights must align with scores in multiway covariance rebuild");
        }
        sqrt_weights = weights->array().sqrt();
        for (int j = 0; j < p; ++j) {
            WS.col(j).array() *= sqrt_weights.array();
        }
    }
    const Eigen::VectorXd* sqrt_ptr = weights ? &sqrt_weights : nullptr;
    int min_clusters = 0;
    // Conventional factors belong to each inclusion-exclusion component.
    // The caller maps only the remaining common scale from its slope block.
    return compute_covariance_multiway(bread,WS,scores,residuals,weights,sqrt_ptr,clusters,
                                       /*df_resid=*/0.0, static_cast<double>(n),
                                       &min_clusters, g_df,
                                       g_adj && g_df == ClusterDofMethod::Conventional);
}

OlsResult run_ols_multiway(const Eigen::VectorXd& y,
                           const Eigen::Ref<const Eigen::MatrixXd>& X,
                           const Eigen::VectorXd* weights,
                           const std::vector<Eigen::VectorXi>* clusters,
                           StandardErrorType se_type,
                           double total_sum_squares,
                           double within_sum_squares,
                           ClusterDofMethod g_df,
                           bool g_adj,
                           double n_effective,
                           const Eigen::MatrixXd* X_for_residuals,
                           bool num_threads_explicit,
                           ParallelWorkObserver* parallel_observer) {
    std::optional<Eigen::Ref<const Eigen::MatrixXd>> actual;
    if (X_for_residuals) actual.emplace(*X_for_residuals);
    return run_ols_multiway_with_score_low(y,X,weights,clusters,se_type,total_sum_squares,
        within_sum_squares,g_df,g_adj,n_effective,actual ? &*actual : nullptr,num_threads_explicit,
        parallel_observer,nullptr);
}

OlsResult run_ols_multiway_with_score_low(const Eigen::VectorXd& y,
                           const Eigen::Ref<const Eigen::MatrixXd>& X,
                           const Eigen::VectorXd* weights,
                           const std::vector<Eigen::VectorXi>* clusters,
                           StandardErrorType se_type,
                           double total_sum_squares,
                           double within_sum_squares,
                           ClusterDofMethod g_df,
                           bool g_adj,
                           double n_effective,
                           const Eigen::Ref<const Eigen::MatrixXd>* X_for_residuals,
                           bool num_threads_explicit,
                           ParallelWorkObserver* parallel_observer,
                           const Eigen::MatrixXd* score_low,
    const Eigen::VectorXd* normalization_means,
    Eigen::VectorXd* intercept_covariance) {
    if (intercept_covariance) intercept_covariance->resize(0);
    if (se_type != StandardErrorType::Cluster) {
        throw std::runtime_error("run_ols_multiway is only valid for clustered inference");
    }
    if (!clusters || clusters->empty()) {
        throw std::runtime_error(
            "Cluster-robust errors requested but no cluster variables were provided");
    }
    if (clusters->size() == 1) {
        return run_ols_with_score_low(y, X, weights, &(*clusters)[0], se_type, total_sum_squares, within_sum_squares,
                       n_effective, false, X_for_residuals,
                       num_threads_explicit, parallel_observer, score_low, normalization_means, intercept_covariance);
    }

    const int n = static_cast<int>(y.size());
    if (X.rows() != n) {
        throw std::runtime_error("OLS called with inconsistent dimensions");
    }
    const int p = static_cast<int>(X.cols());
    if (n == 0 || p == 0) {
        throw std::runtime_error("Need positive observations and regressors for estimation");
    }
    if (X_for_residuals &&
        (X_for_residuals->rows() != n || X_for_residuals->cols() != p)) {
        throw std::runtime_error("Residual design matrix must match X in run_ols_multiway");
    }
    const double n_eff = (n_effective > 0.0) ? n_effective : static_cast<double>(n);
    for (const auto& c : *clusters) {
        if (c.size() != n) {
            throw std::runtime_error("Cluster vector length must equal the number of observations");
        }
    }

    Eigen::VectorXd Wy = y;
    Eigen::MatrixXd WX = X;
    Eigen::VectorXd sqrt_weights;
    if (weights) {
        if (weights->size() != n) {
            throw std::runtime_error("Weights must align with y in run_ols_multiway");
        }
        sqrt_weights = weights->array().sqrt();
        Wy.array() *= sqrt_weights.array();
        for (int j = 0; j < p; ++j) {
            WX.col(j).array() *= sqrt_weights.array();
        }
    }

    bool parallel_moments = false;
    int ols_threads = 1;
#ifdef HDFE_USE_OPENMP
    ols_threads = std::max(1, omp_get_max_threads());
    parallel_moments = num_threads_explicit;
#endif

    Eigen::MatrixXd xtx;
    Eigen::VectorXd xty;
    if (!parallel_moments) {
        xtx = WX.transpose() * WX;
        xty = WX.transpose() * Wy;
    }
#ifdef HDFE_USE_OPENMP
    else {
        // Explicit requests use fixed logical chunks. The arithmetic graph is
        // therefore identical for explicit 1, 8, 48, or any other runtime-
        // permitted team size.
        const int chunks = deterministic_ols_chunk_count(n);
        std::vector<Eigen::MatrixXd> xtx_tls(
            static_cast<std::size_t>(chunks),
            Eigen::MatrixXd::Zero(p, p));
        std::vector<Eigen::VectorXd> xty_tls(
            static_cast<std::size_t>(chunks),
            Eigen::VectorXd::Zero(p));
        const double* y_ptr = y.data();
        const double* x_base = X.data();
        const double* weight_ptr = weights ? weights->data() : nullptr;
        const Eigen::Index x_rows = X.rows();
        {
            ObservedOlsRegion observed(
                parallel_observer, chunks > 0, ols_threads);
#pragma omp parallel for schedule(static) num_threads(ols_threads)
            for (int chunk = 0; chunk < chunks; ++chunk) {
                observed.observe_work();
                Eigen::MatrixXd& xtx_local =
                    xtx_tls[static_cast<std::size_t>(chunk)];
                Eigen::VectorXd& xty_local =
                    xty_tls[static_cast<std::size_t>(chunk)];
                const int begin =
                    deterministic_ols_chunk_begin(n, chunk, chunks);
                const int end =
                    deterministic_ols_chunk_end(n, chunk, chunks);
                for (int i = begin; i < end; ++i) {
                    const double yi = y_ptr[i];
                    const double w = weight_ptr ? weight_ptr[i] : 1.0;
                    for (int j = 0; j < p; ++j) {
                        const double xij =
                            x_base[
                                static_cast<Eigen::Index>(j) * x_rows + i];
                        xty_local(j) += w * xij * yi;
                        for (int k = 0; k <= j; ++k) {
                            xtx_local(j, k) +=
                                w * xij *
                                x_base[
                                    static_cast<Eigen::Index>(k) * x_rows + i];
                        }
                    }
                }
            }
        }
        xtx = Eigen::MatrixXd::Zero(p, p);
        xty = Eigen::VectorXd::Zero(p);
        for (int chunk = 0; chunk < chunks; ++chunk) {
            xtx.noalias() += xtx_tls[static_cast<std::size_t>(chunk)];
            xty.noalias() += xty_tls[static_cast<std::size_t>(chunk)];
        }
        for (int j = 0; j < p; ++j) {
            for (int k = j + 1; k < p; ++k) {
                xtx(j, k) = xtx(k, j);
            }
        }
    }
#endif

    if (!score_low && !normalization_means) {
        if (auto shift = find_ols_origin_shift(xtx, X))
            return run_ols_origin_shifted(*shift,y,X,weights,nullptr,clusters,se_type,
                total_sum_squares,within_sum_squares,n_eff,false,
                X_for_residuals,num_threads_explicit,parallel_observer,g_df,g_adj);
    }
    if(normal_equations_need_wide(xtx))
        return run_ols_wide(y,X,weights,nullptr,clusters,se_type,total_sum_squares,
            within_sum_squares,n_eff,false,X_for_residuals,parallel_observer,g_df,g_adj,score_low,normalization_means,intercept_covariance);
    Eigen::LDLT<Eigen::MatrixXd> solver;
    solver.compute(xtx);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize X'X; matrix may be singular");
    }
    Eigen::VectorXd beta = solver.solve(xty);
    Eigen::MatrixXd xtx_inv = solver.solve(Eigen::MatrixXd::Identity(p, p));

    Eigen::VectorXd residuals(n);
    double rss = 0.0;
    std::shared_ptr<N1NormalEvidence> n1_normal;
    if (n1_capture_active) {
        const int evidence_chunks =
            parallel_moments ? deterministic_ols_chunk_count(n) : 1;
        n1_normal =
            std::make_shared<N1NormalEvidence>(evidence_chunks, p);
    }
    if (!parallel_moments) {
        Eigen::VectorXd fitted =
            X_for_residuals
                ? Eigen::VectorXd((*X_for_residuals) * beta)
                : Eigen::VectorXd(X * beta);
        residuals = y - fitted;
        rss = n1_normal
                  ? weighted_sum_of_squares_with_n1(
                        y, X, X_for_residuals, beta, residuals, weights,
                        n1_normal->chunks[0])
                  : weighted_sum_of_squares(residuals, weights);
    }
#ifdef HDFE_USE_OPENMP
    else {
        const double* y_ptr = y.data();
        const double* x_base =
            X_for_residuals ? X_for_residuals->data() : X.data();
        const Eigen::Index x_rows =
            X_for_residuals ? X_for_residuals->outerStride() : X.outerStride();
        const double* weight_ptr = weights ? weights->data() : nullptr;
        double* residual_ptr = residuals.data();
        const int chunks = deterministic_ols_chunk_count(n);
        std::vector<long double> rss_tls(
            static_cast<std::size_t>(chunks), 0.0L);
        {
            ObservedOlsRegion observed(
                parallel_observer, chunks > 0, ols_threads);
#pragma omp parallel for schedule(static) num_threads(ols_threads)
            for (int chunk = 0; chunk < chunks; ++chunk) {
                observed.observe_work();
                N1NormalChunk* n1_chunk =
                    n1_normal
                        ? &n1_normal->chunks[static_cast<std::size_t>(chunk)]
                        : nullptr;
                KahanSum rss_local;
                const int begin =
                    deterministic_ols_chunk_begin(n, chunk, chunks);
                const int end =
                    deterministic_ols_chunk_end(n, chunk, chunks);
                for (int i = begin; i < end; ++i) {
                    double fitted_i = 0.0;
                    for (int j = 0; j < p; ++j) {
                        fitted_i +=
                            x_base[
                                static_cast<Eigen::Index>(j) * x_rows + i] *
                            beta(j);
                    }
                    const double u = y_ptr[i] - fitted_i;
                    residual_ptr[i] = u;
                    const long double w =
                        static_cast<long double>(
                            weight_ptr ? weight_ptr[i] : 1.0);
                    rss_local.add(
                        w * static_cast<long double>(u) *
                        static_cast<long double>(u));
                    if (n1_chunk) {
                        n1_chunk->add(
                            y_ptr[i], u,
                            weight_ptr ? weight_ptr[i] : 1.0, p,
                            [&](int j) {
                                return X(i, j);
                            },
                            [&](int j) {
                                return X_for_residuals
                                           ? (*X_for_residuals)(i, j)
                                           : X(i, j);
                            },
                            [&](int j) { return beta(j); });
                    }
                }
                rss_tls[static_cast<std::size_t>(chunk)] = rss_local.sum;
            }
        }
        long double rss_acc = 0.0L;
        for (long double value : rss_tls) {
            rss_acc += value;
        }
        rss = static_cast<double>(rss_acc);
    }
#endif

    const double df_resid = std::max(0.0, n_eff - static_cast<double>(p));
    const double sigma2 =
        (df_resid > 0.0) ? rss / df_resid : 0.0;

    const Eigen::VectorXd* sqrt_ptr = weights ? &sqrt_weights : nullptr;
    int min_clusters = 0;
    // Inclusion-exclusion covariance remains deliberately serial. Only the
    // deterministic bread/xty and residual/RSS loops above report parallel
    // work; this serial phase must not manufacture observer evidence.
    Eigen::MatrixXd covariance;
    try {
        covariance = compute_covariance_multiway(xtx_inv,WX,X,residuals,weights,sqrt_ptr,
            *clusters,df_resid,n_eff,&min_clusters,g_df,g_adj);
    } catch (const OlsInferenceRangeError&) {
        n1_normal.reset();
        return run_ols_wide(y,X,weights,nullptr,clusters,se_type,total_sum_squares,
            within_sum_squares,n_eff,false,X_for_residuals,parallel_observer,g_df,g_adj,score_low,normalization_means,intercept_covariance);
    }

    Eigen::VectorXd std_errors = covariance.diagonal().array().cwiseMax(0.0).sqrt();
    Eigen::VectorXd tvalues = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd pvalues = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd conf_int(p, 2);
    constexpr double kZ = 1.959963984540054;
    for (int j = 0; j < p; ++j) {
        if (std_errors(j) > 0) {
            tvalues(j) = beta(j) / std_errors(j);
            const double tail = 1.0 - normal_cdf(std::abs(tvalues(j)));
            pvalues(j) = 2.0 * tail;
        } else {
            tvalues(j) = 0.0;
            pvalues(j) = 1.0;
        }
        conf_int(j, 0) = beta(j) - kZ * std_errors(j);
        conf_int(j, 1) = beta(j) + kZ * std_errors(j);
    }

    OlsResult result;
    result.coefficients = std::move(beta);
    result.std_errors = std::move(std_errors);
    result.tvalues = std::move(tvalues);
    result.pvalues = std::move(pvalues);
    result.conf_int = std::move(conf_int);
    result.residuals = std::move(residuals);
    result.covariance = std::move(covariance);
    result.xtx_inv = std::move(xtx_inv);
    result.df_resid = df_resid;
    result.rss = rss;
    result.tss = total_sum_squares;
    result.within_tss = within_sum_squares;
    const double missing_r2 = std::numeric_limits<double>::quiet_NaN();
    result.r2 = (total_sum_squares > 0.0)
                    ? 1.0 - rss / total_sum_squares
                    : missing_r2;
    result.r2_within = (within_sum_squares > 0.0)
                           ? 1.0 - rss / within_sum_squares
                           : missing_r2;
    result.sigma2 = sigma2;
    result.nobs = n;
    result.num_clusters = min_clusters;
    result.n1_normal = std::move(n1_normal);
    return result;
}

}  // namespace detail
}  // namespace hdfe
