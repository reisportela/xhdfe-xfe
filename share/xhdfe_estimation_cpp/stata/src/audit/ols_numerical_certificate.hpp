#ifndef XHDFE_OLS_NUMERICAL_CERTIFICATE_HPP
#define XHDFE_OLS_NUMERICAL_CERTIFICATE_HPP

#include "../exact_binary_products.hpp"
#include "../ols_precision.hpp"
#include "ordinary_certification.hpp"
#include "hdfe/ieee_bits.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#ifdef HDFE_USE_OPENMP
#include <omp.h>
#endif

namespace hdfe { namespace detail {

struct OlsProofInterval {
    using Real=long double;
    Real lo=0,hi=0;
    static Real up(Real x) {return std::nextafter(x,std::numeric_limits<Real>::infinity());}
    static Real down(Real x) {return std::nextafter(x,-std::numeric_limits<Real>::infinity());}
    static OlsProofInterval point(Real x) {return {x,x};}
    OlsProofInterval operator-() const {return {-hi,-lo};}
    OlsProofInterval operator+(const OlsProofInterval& b) const {
        if (b.lo==0 && b.hi==0) return *this;
        if (lo==0 && hi==0) return b;
        volatile Real a=lo+b.lo,c=hi+b.hi;return {down(a),up(c)};
    }
    OlsProofInterval operator-(const OlsProofInterval& b) const {return *this+(-b);}
    OlsProofInterval operator*(const OlsProofInterval& b) const {
        if ((lo==0 && hi==0) || (b.lo==0 && b.hi==0)) return {};
        volatile Real a=lo*b.lo,c=lo*b.hi,d=hi*b.lo,e=hi*b.hi;
        return {down(std::min({Real(a),Real(c),Real(d),Real(e)})),up(std::max({Real(a),Real(c),Real(d),Real(e)}))};
    }
    Real absolute() const {return std::max(std::abs(lo),std::abs(hi));}
};

struct OlsNumericalCertificate {
    long double singular_lower=0;
    long double coefficient_error=std::numeric_limits<long double>::infinity();
    long double residual_norm=std::numeric_limits<long double>::infinity();
    long double normal_defect=std::numeric_limits<long double>::infinity();
    long double normalized_error=0;
    long double normalized_matrix_norm=0;
    OlsProofInterval residual_squared;
    std::vector<int> basis_exponents;
    std::vector<OlsProofInterval> bread;
};

inline OlsNumericalCertificate certify_ols_numerics(
    const Eigen::VectorXd& y,const Eigen::MatrixXd& X,const Eigen::VectorXd* weights,
    const Eigen::VectorXd& beta,const Eigen::MatrixXd& basis,int threads,
    ParallelWorkObserver* observer,bool include_bread=false) {
    if (!ordinary_audit_enabled()) return {};
    ++ordinary_audit_work_count;
    using Sum=ExactBinaryProducts<>;using I=OlsProofInterval;using Real=long double;
    using Matrix=Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic>;
    using Vector=Eigen::Matrix<Real,Eigen::Dynamic,1>;
    const int n=static_cast<int>(X.rows()),p=static_cast<int>(X.cols());
    if (!p || beta.size()!=p || basis.rows()!=p || basis.cols()!=p)
        throw std::runtime_error("Ordinary FE coefficient certificate received an incomplete basis; no estimates returned");
    const std::size_t square=static_cast<std::size_t>(p)*p;
    std::vector<std::pair<int,int>> pairs;
    for (int j=0;j<=p;++j) for (int k=j;k<=p;++k) pairs.emplace_back(j,k);
    const int team=std::max(1,threads);
    std::vector<std::vector<Sum>> partial(static_cast<std::size_t>(team),std::vector<Sum>(pairs.size()));
    if (observer) observer->begin_region(team);
#ifdef HDFE_USE_OPENMP
#pragma omp parallel num_threads(team)
#endif
    {
        int tid=0;
#ifdef HDFE_USE_OPENMP
        tid=omp_get_thread_num();
#endif
        bool observed=false;
#ifdef HDFE_USE_OPENMP
#pragma omp for schedule(static)
#endif
        for (int row=0;row<n;++row) {
            if (!observed && observer) observer->observe_work();observed=true;
            const double w=weights ? (*weights)[row] : 1.0;
            for (std::size_t m=0;m<pairs.size();++m) {
                const auto [j,k]=pairs[m];
                const double a=j==p ? y[row] : X(row,j),b=k==p ? y[row] : X(row,k);
                partial[tid][m].add_product(std::array<double,3>{w,a,b});
            }
        }
    }
    if (observer) observer->end_region();
    std::vector<Sum> gram(square),rhs(p);Sum yty;
    for (std::size_t m=0;m<pairs.size();++m) {
        Sum total;
        for (const auto& values:partial) total.merge(values[m]);
        if (!total.valid()) throw std::runtime_error("Ordinary FE exact OLS moments exceeded their integer range; no estimates returned");
        const auto [j,k]=pairs[m];
        if (j==p) yty=std::move(total);
        else if (k==p) rhs[j]=std::move(total);
        else {gram[j*p+k]=total;gram[k*p+j]=std::move(total);}
    }
    partial.clear();
    std::vector<Sum> defect=rhs;
    for (int j=0;j<p;++j) for (int k=0;k<p;++k) defect[j].add_scaled(gram[j*p+k],beta[k],true);
    Sum residual_squared=yty;
    for (int j=0;j<p;++j) {
        residual_squared.add_scaled(rhs[j],beta[j],true);
        residual_squared.add_scaled(defect[j],beta[j],true);
    }
    if (!residual_squared.valid() || residual_squared.magnitude().first)
        throw std::runtime_error("Ordinary FE exact OLS residual norm was not established; no estimates returned");
    OlsNumericalCertificate result;
    const auto rss_interval=residual_squared.interval();
    result.residual_squared={std::max(0.0L,rss_interval.first),rss_interval.second};
    result.residual_norm=residual_squared.exactly_zero() ? 0 : I::up(std::sqrt(residual_squared.upper_absolute()));
    result.normal_defect=0;
    for (const auto& value:defect) result.normal_defect=I::up(result.normal_defect+value.upper_absolute());

    // H=C' G C and C'g are exact. Binary diagonal scaling keeps H small
    // without squaring a floating-point condition number or trusting bread.
    std::vector<Sum> left(square),transformed(square),projected(p);
    for (int j=0;j<p;++j) for (int a=0;a<p;++a) if (basis(a,j)!=0) {
        projected[j].add_scaled(defect[a],basis(a,j));
        for (int k=0;k<p;++k) left[j*p+k].add_scaled(gram[a*p+k],basis(a,j));
    }
    for (int j=0;j<p;++j) for (int k=0;k<p;++k)
        for (int b=0;b<p;++b) if (basis(b,k)!=0) transformed[j*p+k].add_scaled(left[j*p+b],basis(b,k));
    std::vector<int> exponent(p);
    for (int j=0;j<p;++j) {
        const auto& diagonal=transformed[j*p+j];
        if (!diagonal.valid() || diagonal.exactly_zero() || diagonal.magnitude().first)
            throw std::runtime_error("Ordinary FE coefficient basis has no verified positive norm; no estimates returned");
        const int top=diagonal.top_exponent();
        exponent[j]=top>=0 ? top/2 : -((-top+1)/2);
    }
    Matrix H(p,p);Vector h(p);std::vector<I> enclosed(square),enclosed_h(p);
    Real eigen_lower=std::numeric_limits<Real>::infinity();
    for (int j=0;j<p;++j) {
        const auto value=projected[j].interval(-exponent[j]);enclosed_h[j]={value.first,value.second};
        h[j]=value.first/2+value.second/2;
        for (int k=0;k<p;++k) {
            const auto entry=transformed[j*p+k].interval(-exponent[j]-exponent[k]);
            enclosed[j*p+k]={entry.first,entry.second};H(j,k)=entry.first/2+entry.second/2;
        }
        Real radius=0;
        for (int k=0;k<p;++k) if (k!=j) radius=I::up(radius+enclosed[j*p+k].absolute());
        eigen_lower=std::min(eigen_lower,I::down(enclosed[j*p+j].lo-radius));
    }
    if (!(eigen_lower>0) || !ieee_finite(static_cast<double>(eigen_lower)))
        throw std::runtime_error("Ordinary FE coefficient basis could not be bounded away from singularity; no estimates returned");
    result.basis_exponents=exponent;
    for (int j=0;j<p;++j) {
        Real row_sum=0;
        for (int k=0;k<p;++k) row_sum=I::up(row_sum+enclosed[j*p+k].absolute());
        result.normalized_matrix_norm=std::max(result.normalized_matrix_norm,row_sum);
    }
    std::vector<I> scaled_basis(square);std::vector<Real> row_squared(p,0);Real basis_squared=0;
    for (int j=0;j<p;++j) for (int k=0;k<p;++k) {
        Sum entry;entry.add_product(std::array<double,1>{basis(j,k)});
        const auto bounds=entry.interval(-exponent[k]);scaled_basis[j*p+k]={bounds.first,bounds.second};
        const Real upper=scaled_basis[j*p+k].absolute();
        basis_squared=I::up(basis_squared+I::up(upper*upper));
        row_squared[j]=I::up(row_squared[j]+I::up(upper*upper));
    }
    const Real basis_norm=I::up(std::sqrt(basis_squared));
    result.singular_lower=I::down(std::sqrt(I::down(eigen_lower/basis_squared)));
    if (!(result.singular_lower>0))
        throw std::runtime_error("Ordinary FE singular-value bound exceeded its numerical range; no estimates returned");
    if (include_bread) {
        const Matrix inverse=H.ldlt().solve(Matrix::Identity(p,p));
        Real inverse_residual=0;
        for (int j=0;j<p;++j) for (int k=0;k<p;++k) {
            I defect=I::point(j==k ? 1 : 0);
            for (int a=0;a<p;++a) defect=defect-enclosed[j*p+a]*I::point(inverse(a,k));
            inverse_residual=I::up(inverse_residual+defect.absolute());
        }
        const Real inverse_error=I::up(inverse_residual/eigen_lower);
        result.bread.resize(square);
        for (int j=0;j<p;++j) for (int k=0;k<p;++k) {
            I value;
            for (int a=0;a<p;++a) for (int b=0;b<p;++b)
                value=value+scaled_basis[j*p+a]*I::point(inverse(a,b))*scaled_basis[k*p+b];
            const I rounding=I::point(I::up(std::sqrt(row_squared[j])))*
                I::point(I::up(std::sqrt(row_squared[k])))*I::point(inverse_error);
            result.bread[j*p+k]=value+I{-rounding.hi,rounding.hi};
        }
    }

    // Enclose the actual OLS coefficient error in the stabilised coordinates.
    // This preserves a tiny defect in a weak direction when large components
    // cancel, instead of replacing its action by ||g||/sigma_min^2.
    if (std::all_of(defect.begin(),defect.end(),[](const Sum& s){return s.exactly_zero();})) {
        result.coefficient_error=0;result.normal_defect=0;return result;
    }
    const Vector correction=H.ldlt().solve(h);
    Real residual_upper=0,coefficient_upper=0,normalized_upper=0;
    for (int j=0;j<p;++j) {
        I remainder=enclosed_h[j],delta;
        for (int k=0;k<p;++k) {
            remainder=remainder-enclosed[j*p+k]*I::point(correction[k]);
            delta=delta+scaled_basis[j*p+k]*I::point(correction[k]);
        }
        residual_upper=I::up(residual_upper+remainder.absolute());
        coefficient_upper=I::up(coefficient_upper+delta.absolute());
        normalized_upper=I::up(normalized_upper+std::abs(correction[j]));
    }
    const Real correction_error=I::up(residual_upper/eigen_lower);
    result.normalized_error=I::up(normalized_upper+correction_error);
    result.coefficient_error=I::up(coefficient_upper+I::up(basis_norm*correction_error));
    if (!ieee_finite(static_cast<double>(result.coefficient_error)))
        throw std::runtime_error("Ordinary FE OLS coefficient error could not be enclosed; no estimates returned");
    return result;
}

}}
#endif
