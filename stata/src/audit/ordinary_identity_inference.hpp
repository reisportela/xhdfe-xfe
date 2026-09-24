#ifndef XHDFE_ORDINARY_IDENTITY_INFERENCE_HPP
#define XHDFE_ORDINARY_IDENTITY_INFERENCE_HPP

#include "ordinary_final_inference.hpp"
#include "../ols_precision.hpp"

namespace hdfe { namespace detail {

// A bounded certificate for an unchanged RHS. Large models fail this size
// filter before any observation is read; exact moments are the acceptance test.
inline bool ordinary_small_identity_projection(
    const Eigen::Ref<const Eigen::VectorXd>& y,const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,const AbsorptionResult& result) {
    const auto n=y.size(),p=X.cols();
    if (n<=0 || p<=0 || p>32 || fes.empty() || result.fe_levels.size()!=fes.size() ||
        result.y_tilde.size()!=n || result.X_tilde.rows()!=n || result.X_tilde.cols()!=p) return false;
    std::uint64_t levels=0;
    for (int count:result.fe_levels) {
        if (count<=0 || count>64) return false;
        levels+=static_cast<std::uint64_t>(count);
        if (levels>64) return false;
    }
    if (static_cast<std::uint64_t>(n)>100000ULL/(levels+static_cast<std::uint64_t>(p))) return false;
    for (Eigen::Index row=0;row<n;++row) {
        if (ieee_bits_detail::bits(y[row])!=ieee_bits_detail::bits(result.y_tilde[row])) return false;
        for (Eigen::Index j=0;j<p;++j)
            if (ieee_bits_detail::bits(X(row,j))!=ieee_bits_detail::bits(result.X_tilde(row,j))) return false;
    }
    for (const auto& fe:fes)
        if (fe.size()!=n || !exact_level_projection(result.y_tilde,result.X_tilde,fe,
                nullptr,result.y_tilde,result.X_tilde)) return false;
    return true;
}

// Called only after the bounded identity proof. This measures stored residual
// formation exactly instead of multiplying a huge global signal by epsilon.
inline bool ordinary_small_residual_formation(
    const Eigen::VectorXd& y,const Eigen::MatrixXd& X,const OlsResult& fit,long double& bound) {
    using I=OlsProofInterval;using Sum=ExactBinaryProducts<>;
    if (fit.residuals.size()!=y.size() || fit.coefficients.size()!=X.cols()) return false;
    I squared;
    for (Eigen::Index row=0;row<y.size();++row) {
        Sum error;error.add_product(std::array<double,1>{y[row]});
        error.add_product(std::array<double,1>{fit.residuals[row]},true);
        for (Eigen::Index j=0;j<X.cols();++j)
            error.add_product(std::array<double,2>{X(row,j),fit.coefficients[j]},true);
        if (!error.valid()) return false;
        if (!error.exactly_zero()) {
            const I radius=I::point(error.upper_absolute());
            squared=squared+radius*radius;
        }
    }
    bound=ordinary_final_root(squared.hi);
    return bound>=0 && bound<=std::numeric_limits<long double>::max();
}

} }
#endif
