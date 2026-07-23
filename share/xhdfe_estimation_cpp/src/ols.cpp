#include "ols.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

double normal_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

template <int P>
OlsResult run_ols_fast_impl(const Eigen::VectorXd& y,
                            const Eigen::Ref<const Eigen::MatrixXd>& X,
                            const Eigen::VectorXd* weights,
                            StandardErrorType se_type,
                            double total_sum_squares,
                            double within_sum_squares,
                            double n_effective,
                            bool weights_are_frequencies) {
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

    std::vector<MatPP> xtx_tls(static_cast<std::size_t>(threads), MatPP::Zero());
    std::vector<VecP> xty_tls(static_cast<std::size_t>(threads), VecP::Zero());

#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
        int tid = 0;
#ifdef HDFE_USE_OPENMP
        tid = omp_get_thread_num();
#endif
        MatPP& xtx_local = xtx_tls[static_cast<std::size_t>(tid)];
        VecP& xty_local = xty_tls[static_cast<std::size_t>(tid)];

#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
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

    MatPP xtx = MatPP::Zero();
    VecP xty = VecP::Zero();
    for (int t = 0; t < threads; ++t) {
        xtx.noalias() += xtx_tls[static_cast<std::size_t>(t)];
        xty.noalias() += xty_tls[static_cast<std::size_t>(t)];
    }
    for (int j = 0; j < P; ++j) {
        for (int k = j + 1; k < P; ++k) {
            xtx(j, k) = xtx(k, j);
        }
    }

    Eigen::LDLT<MatPP> solver;
    solver.compute(xtx);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize X'X; matrix may be singular");
    }
    VecP beta = solver.solve(xty);
    MatPP xtx_inv = solver.solve(MatPP::Identity());

    Eigen::VectorXd residuals(n);
    double* resid_ptr = residuals.data();

    std::vector<long double> rss_tls(static_cast<std::size_t>(threads), 0.0L);
    std::vector<MatPP> meat_tls;
    if (se_type == StandardErrorType::Robust) {
        meat_tls.assign(static_cast<std::size_t>(threads), MatPP::Zero());
    }

#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
        int tid = 0;
#ifdef HDFE_USE_OPENMP
        tid = omp_get_thread_num();
#endif
        KahanSum rss_local;
        MatPP meat_local = MatPP::Zero();

#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
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

            if (se_type == StandardErrorType::Robust) {
                // fweight: replication semantics -> linear w in the robust meat;
                // aweight/pweight: w^2 (matches Stata/areg/reghdfe).
                const double w2 = w_ptr ? (weights_are_frequencies ? w : (w * w)) : 1.0;
                const double scale = w2 * u * u;
                for (int j = 0; j < P; ++j) {
                    for (int k = 0; k <= j; ++k) {
                        meat_local(j, k) += scale * xvals[j] * xvals[k];
                    }
                }
            }
        }

        rss_tls[static_cast<std::size_t>(tid)] = rss_local.sum;
        if (se_type == StandardErrorType::Robust) {
            meat_tls[static_cast<std::size_t>(tid)] = std::move(meat_local);
        }
    }

    long double rss = 0.0L;
    for (long double v : rss_tls) {
        rss += v;
    }
    const double rss_d = static_cast<double>(rss);

    MatPP meat = MatPP::Zero();
    if (se_type == StandardErrorType::Robust) {
        for (int t = 0; t < threads; ++t) {
            meat.noalias() += meat_tls[static_cast<std::size_t>(t)];
        }
        for (int j = 0; j < P; ++j) {
            for (int k = j + 1; k < P; ++k) {
                meat(j, k) = meat(k, j);
            }
        }
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
    result.r2 = (total_sum_squares > 0.0) ? 1.0 - rss_d / total_sum_squares : 1.0;
    result.r2_within = (within_sum_squares > 0.0) ? 1.0 - rss_d / within_sum_squares : 1.0;
    result.sigma2 = sigma2;
    result.nobs = n;
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
                                     bool weights_are_frequencies) {
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

    std::vector<long double> rss_tls(static_cast<std::size_t>(threads), 0.0L);
    std::vector<Eigen::Matrix<double, P, P>> meat_tls;
    if (se_type == StandardErrorType::Robust) {
        meat_tls.assign(static_cast<std::size_t>(threads),
                        Eigen::Matrix<double, P, P>::Zero());
    }

#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
        int tid = 0;
#ifdef HDFE_USE_OPENMP
        tid = omp_get_thread_num();
#endif
        KahanSum rss_local;
        Eigen::Matrix<double, P, P> meat_local = Eigen::Matrix<double, P, P>::Zero();

#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
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

            if (se_type == StandardErrorType::Robust) {
                // fweight: replication semantics -> linear w in the robust meat;
                // aweight/pweight: w^2 (matches Stata/areg/reghdfe).
                const double w2 = w_ptr ? (weights_are_frequencies ? w : (w * w)) : 1.0;
                const double scale = w2 * u * u;
                for (int j = 0; j < P; ++j) {
                    for (int k = 0; k <= j; ++k) {
                        meat_local(j, k) += scale * xvals[j] * xvals[k];
                    }
                }
            }
        }

        rss_tls[static_cast<std::size_t>(tid)] = rss_local.sum;
        if (se_type == StandardErrorType::Robust) {
            meat_tls[static_cast<std::size_t>(tid)] = std::move(meat_local);
        }
    }

    long double rss = 0.0L;
    for (long double v : rss_tls) {
        rss += v;
    }
    const double rss_d = static_cast<double>(rss);

    Eigen::Matrix<double, P, P> meat = Eigen::Matrix<double, P, P>::Zero();
    if (se_type == StandardErrorType::Robust) {
        for (int t = 0; t < threads; ++t) {
            meat.noalias() += meat_tls[static_cast<std::size_t>(t)];
        }
        for (int j = 0; j < P; ++j) {
            for (int k = j + 1; k < P; ++k) {
                meat(j, k) = meat(k, j);
            }
        }
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
    result.r2 = (total_sum_squares > 0.0) ? 1.0 - rss_d / total_sum_squares : 1.0;
    result.r2_within = (within_sum_squares > 0.0) ? 1.0 - rss_d / within_sum_squares : 1.0;
    result.sigma2 = sigma2;
    result.nobs = n;
    return result;
}

Eigen::MatrixXd compute_covariance(const Eigen::MatrixXd& xtx_inv,
                                    const Eigen::Ref<const Eigen::MatrixXd>& WX,
                                    const Eigen::VectorXd& residuals,
                                    const Eigen::VectorXd* sqrt_weights,
                                    const Eigen::VectorXi* clusters,
                                    StandardErrorType se_type,
                                    double sigma2,
                                    double df_resid,
                                    double n_effective,
                                    int* num_clusters_out,
                                    Eigen::VectorXd* cluster_ux_out,
                                    double* cluster_u2_out,
                                    bool weights_are_frequencies) {
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
        for (int i = 0; i < n; ++i) {
            // WX already carries sqrt(w), so WX_i WX_i' = w*xx'. For aweight/pweight
            // the robust meat needs w^2*xx' (scale by u*sqrt(w) -> u^2*w). For fweight
            // (replication) it needs w*xx' (scale by u -> u^2), matching Stata/areg/reghdfe.
            const double scaled_u = (sqrt_weights && !weights_are_frequencies)
                                        ? residuals(i) * (*sqrt_weights)(i)
                                        : residuals(i);
            const double w = scaled_u * scaled_u;
            meat_lower.rankUpdate(WX.row(i).transpose(), w);
        }
        cov = xtx_inv * meat_lower * xtx_inv;
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

#ifdef HDFE_USE_OPENMP
            // Per-run scores are each accumulated serially by one thread in
            // observation order and combined serially in run order below, so
            // the result is bit-identical at any thread count. The historical
            // 8-thread cap is kept below the large-n gate and lifted above it,
            // where the scan is memory-bandwidth bound.
            const int cluster_threads =
                (n >= 4194304) ? std::max(1, omp_get_max_threads())
                               : std::min(std::max(1, omp_get_max_threads()), 8);
#pragma omp parallel for schedule(static) num_threads(cluster_threads)
#endif
            for (int r = 0; r < next_cluster; ++r) {
                const int start = run_starts[static_cast<std::size_t>(r)];
                const int end = run_ends[static_cast<std::size_t>(r)];
                double run_score[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                double run_u_sum = 0.0;
                for (int obs = start; obs < end; ++obs) {
                    const double scaled_u =
                        sqrt_weights ? residuals(obs) * (*sqrt_weights)(obs) : residuals(obs);
                    for (int j = 0; j < p; ++j) {
                        run_score[j] += WX(obs, j) * scaled_u;
                    }
                    double u_score = scaled_u;
                    if (sqrt_weights) {
                        u_score *= (*sqrt_weights)(obs);
                    }
                    run_u_sum += u_score;
                }
                double* score = run_scores.data() +
                                static_cast<std::size_t>(r) * static_cast<std::size_t>(p);
                for (int j = 0; j < p; ++j) {
                    score[static_cast<std::size_t>(j)] = run_score[j];
                }
                run_u_sums[static_cast<std::size_t>(r)] = run_u_sum;
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
            cov = xtx_inv * meat_lower * xtx_inv;
            cov *= scale;
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
        cov = xtx_inv * meat_lower * xtx_inv;
        cov *= scale;
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
    cov = xtx_inv * meat_lower * xtx_inv;
    cov *= scale;
    return cov;
}

Eigen::MatrixXd compute_cluster_meat(const Eigen::MatrixXd& WX,
                                     const Eigen::VectorXd& residuals,
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
                                            const Eigen::VectorXd& residuals,
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
        Eigen::MatrixXd meat = compute_cluster_meat(WX, residuals, sqrt_weights, combined, &G);
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

    Eigen::MatrixXd cov = xtx_inv * meat_total * xtx_inv;
    cov *= scale;
    return cov;
}

}  // namespace

OlsResult run_ols_fast_from_xtx(const Eigen::VectorXd& y,
                                const Eigen::Ref<const Eigen::MatrixXd>& X,
                                const Eigen::VectorXd* weights,
                                StandardErrorType se_type,
                                double total_sum_squares,
                                double within_sum_squares,
                                const Eigen::MatrixXd& xtx,
                                const Eigen::VectorXd& xty,
                                double n_effective,
                                bool weights_are_frequencies) {
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
    switch (p) {
        case 1:
            return run_ols_fast_from_xtx_impl<1>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies);
        case 2:
            return run_ols_fast_from_xtx_impl<2>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies);
        case 3:
            return run_ols_fast_from_xtx_impl<3>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies);
        case 4:
            return run_ols_fast_from_xtx_impl<4>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies);
        case 5:
            return run_ols_fast_from_xtx_impl<5>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies);
        case 6:
            return run_ols_fast_from_xtx_impl<6>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies);
        case 7:
            return run_ols_fast_from_xtx_impl<7>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies);
        case 8:
            return run_ols_fast_from_xtx_impl<8>(y, X, weights, se_type, total_sum_squares,
                                                 within_sum_squares, xtx, xty, n_eff,
                                                 weights_are_frequencies);
        default:
            break;
    }
    return run_ols(y, X, weights, nullptr, se_type, total_sum_squares, within_sum_squares, n_eff,
                   weights_are_frequencies);
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
                  const Eigen::MatrixXd* X_for_residuals) {
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
                                            within_sum_squares, n_eff, weights_are_frequencies);
            case 2:
                return run_ols_fast_impl<2>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies);
            case 3:
                return run_ols_fast_impl<3>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies);
            case 4:
                return run_ols_fast_impl<4>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies);
            case 5:
                return run_ols_fast_impl<5>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies);
            case 6:
                return run_ols_fast_impl<6>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies);
            case 7:
                return run_ols_fast_impl<7>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies);
            case 8:
                return run_ols_fast_impl<8>(y, X, weights, se_type, total_sum_squares,
                                            within_sum_squares, n_eff, weights_are_frequencies);
            default:
                break;
        }
    }

    // Large unweighted problems take a copy-free, thread-parallel path: WX/Wy
    // would be byte-for-byte copies of X/y that are only ever read, and the
    // Eigen GEMM/matvec passes below run single-threaded over multi-GB
    // arrays. Below the gate (and for every weighted call) the historical
    // code runs verbatim, so results there are bit-identical by construction.
    constexpr int kParallelOlsMinObs = 4194304;
    bool parallel_unweighted = false;
    int ols_threads = 1;
#ifdef HDFE_USE_OPENMP
    ols_threads = std::max(1, omp_get_max_threads());
    parallel_unweighted = (weights == nullptr) && n >= kParallelOlsMinObs && ols_threads > 1;
#endif

    Eigen::VectorXd Wy;
    Eigen::MatrixXd WX;
    Eigen::VectorXd sqrt_weights;
    Eigen::MatrixXd xtx;
    Eigen::VectorXd xty;
    if (!parallel_unweighted) {
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
        // Deterministic TLS accumulation: static row ranges per thread,
        // partials combined in thread-id order.
        std::vector<Eigen::MatrixXd> xtx_tls(static_cast<std::size_t>(ols_threads),
                                             Eigen::MatrixXd::Zero(p, p));
        std::vector<Eigen::VectorXd> xty_tls(static_cast<std::size_t>(ols_threads),
                                             Eigen::VectorXd::Zero(p));
        const double* y_ptr = y.data();
        const double* x_base = X.data();
        const Eigen::Index x_rows = X.rows();
#pragma omp parallel num_threads(ols_threads)
        {
            const int tid = omp_get_thread_num();
            Eigen::MatrixXd& xtx_local = xtx_tls[static_cast<std::size_t>(tid)];
            Eigen::VectorXd& xty_local = xty_tls[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const double yi = y_ptr[i];
                for (int j = 0; j < p; ++j) {
                    const double xij = x_base[static_cast<Eigen::Index>(j) * x_rows + i];
                    xty_local(j) += xij * yi;
                    for (int k = 0; k <= j; ++k) {
                        xtx_local(j, k) +=
                            xij * x_base[static_cast<Eigen::Index>(k) * x_rows + i];
                    }
                }
            }
        }
        xtx = Eigen::MatrixXd::Zero(p, p);
        xty = Eigen::VectorXd::Zero(p);
        for (int t = 0; t < ols_threads; ++t) {
            xtx.noalias() += xtx_tls[static_cast<std::size_t>(t)];
            xty.noalias() += xty_tls[static_cast<std::size_t>(t)];
        }
        for (int j = 0; j < p; ++j) {
            for (int k = j + 1; k < p; ++k) {
                xtx(j, k) = xtx(k, j);
            }
        }
    }
#endif

    Eigen::LDLT<Eigen::MatrixXd> solver;
    solver.compute(xtx);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize X'X; matrix may be singular");
    }
    Eigen::VectorXd beta = solver.solve(xty);
    Eigen::MatrixXd xtx_inv = solver.solve(Eigen::MatrixXd::Identity(p, p));

    Eigen::VectorXd residuals(n);
    double rss = 0.0;
    if (!parallel_unweighted) {
        Eigen::VectorXd fitted =
            X_for_residuals ? Eigen::VectorXd((*X_for_residuals) * beta)
                            : Eigen::VectorXd(X * beta);
        residuals = y - fitted;
        rss = weighted_sum_of_squares(residuals, weights);
    }
#ifdef HDFE_USE_OPENMP
    else {
        // Each residual element is written by exactly one thread with the
        // same per-element column accumulation order as the Eigen matvec;
        // the RSS uses per-thread Kahan partials combined in thread-id order.
        const double* y_ptr = y.data();
        const double* x_base = X_for_residuals ? X_for_residuals->data() : X.data();
        const Eigen::Index x_rows = X_for_residuals ? X_for_residuals->rows() : X.rows();
        double* resid_ptr = residuals.data();
        std::vector<long double> rss_tls(static_cast<std::size_t>(ols_threads), 0.0L);
#pragma omp parallel num_threads(ols_threads)
        {
            const int tid = omp_get_thread_num();
            KahanSum rss_local;
#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                double fitted_i = 0.0;
                for (int j = 0; j < p; ++j) {
                    fitted_i += x_base[static_cast<Eigen::Index>(j) * x_rows + i] * beta(j);
                }
                const double u = y_ptr[i] - fitted_i;
                resid_ptr[i] = u;
                rss_local.add(static_cast<long double>(u) * static_cast<long double>(u));
            }
            rss_tls[static_cast<std::size_t>(tid)] = rss_local.sum;
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
        parallel_unweighted ? Eigen::Ref<const Eigen::MatrixXd>(X)
                            : Eigen::Ref<const Eigen::MatrixXd>(WX);
    Eigen::MatrixXd covariance = compute_covariance(xtx_inv, WX_used, residuals, sqrt_ptr, clusters,
                                                    se_type, sigma2, df_resid, n_eff, &num_clusters,
                                                    (se_type == StandardErrorType::Cluster) ? &cluster_ux : nullptr,
                                                    (se_type == StandardErrorType::Cluster) ? &cluster_u2 : nullptr,
                                                    weights_are_frequencies);
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
    result.r2 = (total_sum_squares > 0.0) ? 1.0 - rss / total_sum_squares : 1.0;
    result.r2_within = (within_sum_squares > 0.0) ? 1.0 - rss / within_sum_squares : 1.0;
    result.sigma2 = sigma2;
    result.nobs = n;
    result.num_clusters = num_clusters;
    return result;
}

Eigen::MatrixXd multiway_cluster_sandwich(
    const Eigen::MatrixXd& bread,
    const Eigen::Ref<const Eigen::MatrixXd>& scores,
    const Eigen::VectorXd& residuals,
    const Eigen::VectorXd* weights,
    const std::vector<Eigen::VectorXi>& clusters) {
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
    // df_resid=0 and g_adj=false: no global or per-combination df factors —
    // the caller maps the composite scaling from its scaled slope block.
    return compute_covariance_multiway(bread, WS, residuals, sqrt_ptr, clusters,
                                       /*df_resid=*/0.0, static_cast<double>(n),
                                       &min_clusters, ClusterDofMethod::Min,
                                       /*g_adj=*/false);
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
                           const Eigen::MatrixXd* X_for_residuals) {
    if (se_type != StandardErrorType::Cluster) {
        throw std::runtime_error("run_ols_multiway is only valid for clustered inference");
    }
    if (!clusters || clusters->empty()) {
        throw std::runtime_error(
            "Cluster-robust errors requested but no cluster variables were provided");
    }
    if (clusters->size() == 1) {
        return run_ols(y, X, weights, &(*clusters)[0], se_type, total_sum_squares, within_sum_squares,
                       n_effective, false, X_for_residuals);
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

    Eigen::MatrixXd xtx = WX.transpose() * WX;
    Eigen::VectorXd xty = WX.transpose() * Wy;
    Eigen::LDLT<Eigen::MatrixXd> solver;
    solver.compute(xtx);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to factorize X'X; matrix may be singular");
    }
    Eigen::VectorXd beta = solver.solve(xty);
    Eigen::MatrixXd xtx_inv = solver.solve(Eigen::MatrixXd::Identity(p, p));

    Eigen::VectorXd fitted = X_for_residuals ? Eigen::VectorXd((*X_for_residuals) * beta)
                                             : Eigen::VectorXd(X * beta);
    Eigen::VectorXd residuals = y - fitted;

    const double rss = weighted_sum_of_squares(residuals, weights);
    const double df_resid = std::max(0.0, n_eff - static_cast<double>(p));
    const double sigma2 =
        (df_resid > 0.0) ? rss / df_resid : 0.0;

    const Eigen::VectorXd* sqrt_ptr = weights ? &sqrt_weights : nullptr;
    int min_clusters = 0;
    Eigen::MatrixXd covariance = compute_covariance_multiway(
        xtx_inv, WX, residuals, sqrt_ptr, *clusters, df_resid, n_eff, &min_clusters, g_df, g_adj);

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
    result.r2 = (total_sum_squares > 0.0) ? 1.0 - rss / total_sum_squares : 1.0;
    result.r2_within = (within_sum_squares > 0.0) ? 1.0 - rss / within_sum_squares : 1.0;
    result.sigma2 = sigma2;
    result.nobs = n;
    result.num_clusters = min_clusters;
    return result;
}

}  // namespace detail
}  // namespace hdfe
