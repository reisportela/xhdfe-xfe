#ifndef XHDFE_ORDINARY_STRUCTURED_COEFFICIENT_HPP
#define XHDFE_ORDINARY_STRUCTURED_COEFFICIENT_HPP

#include "ols_numerical_certificate.hpp"

namespace hdfe { namespace detail {

struct OrdinaryStructuredCoefficientInput {
    bool ready=false;
    Eigen::VectorXd coefficients;
    OlsNumericalCertificate numerical;
    Eigen::MatrixXd auxiliary_homoskedastic_covariance;
    long double auxiliary_covariance_df=0;
    long double limit=0;
    long double covariance_limit=0;
    long double residual_roundoff_floor=0;
};

struct OrdinaryStructuredHomoskedasticCovarianceCheck {
    bool eligible=false;
    bool passed=false;
    bool exact_zero=false;
    bool numerical_perfect=false;
    long double eta_squared=std::numeric_limits<long double>::infinity();
    long double phi_squared=std::numeric_limits<long double>::infinity();
    long double delta=std::numeric_limits<long double>::infinity();
    OlsProofInterval candidate_rss;
    OlsProofInterval true_rss;
    long double rho_lower=0;
    long double rho_upper=std::numeric_limits<long double>::infinity();
    long double fe_relative_effect=std::numeric_limits<long double>::infinity();
    long double native_relative_effect=std::numeric_limits<long double>::infinity();
    long double relative_effect=std::numeric_limits<long double>::infinity();
};

struct OrdinaryStructuredHomoskedasticNativeCheck {
    bool eligible=false;
    bool passed=false;
    bool exact_zero=false;
    OlsProofInterval candidate_rss;
    long double relative_effect=std::numeric_limits<long double>::infinity();
};

// Let U=A+E_F+E_P and v=b+e_F+e_P, where A=P X, b=P y,
// P residualizes the FE space, and the F/P components are orthogonal.
// For exact OLS beta on U,v, R=v-U beta and delta=e-E beta:
// A'(b-A beta) = -A'delta_P - E_P'P R - E_F'delta_F.
// FE-space errors therefore enter the coefficient bound quadratically.
// This certificate says nothing about covariance; its existing checks remain
// mandatory. Numerical OLS error is included, not presumed negligible.
inline long double ordinary_structured_coefficient_bound(
    const OrdinaryStructuredCoefficientInput& input,
    const std::vector<long double>& fe,const std::vector<long double>& affine) {
    using I=OlsProofInterval;using Real=long double;
    const Real infinity=std::numeric_limits<Real>::infinity();
    const int p=static_cast<int>(input.coefficients.size());
    if (!input.ready || !p || fe.size()!=static_cast<std::size_t>(p+1) || affine.size()!=fe.size()) return infinity;
    const auto& numerical=input.numerical;
    if (!(numerical.singular_lower>0) || !(numerical.coefficient_error>=0) || !(numerical.residual_norm>=0)) return infinity;
    I f_squared,a_squared,delta_f=I::point(fe[0]),delta_a=I::point(affine[0]);
    for (int j=0;j<=p;++j) {
        if (!(fe[j]>=0) || !(affine[j]>=0) || fe[j]>std::numeric_limits<Real>::max() || affine[j]>std::numeric_limits<Real>::max()) return infinity;
        if (!j) continue;
        const I f=I::point(fe[j]),a=I::point(affine[j]);
        f_squared=f_squared+f*f;a_squared=a_squared+a*a;
        const I coefficient=I::point(std::abs(static_cast<Real>(input.coefficients[j-1])))+
            I::point(numerical.coefficient_error);
        delta_f=delta_f+f*coefficient;delta_a=delta_a+a*coefficient;
    }
    const Real eta_f=I::up(std::sqrt(f_squared.hi)),eta_a=I::up(std::sqrt(a_squared.hi));
    const I s=I::point(numerical.singular_lower);
    const Real remaining=(s*s-f_squared).lo;
    if (!(remaining>0)) return infinity;
    const Real lower=I::down(I::down(std::sqrt(remaining))-eta_a);
    if (!(lower>0)) return infinity;
    const I inverse{I::down(1/lower),I::up(1/lower)};
    const I error=I::point(numerical.coefficient_error)+delta_a*inverse+
        (I::point(eta_a)*I::point(numerical.residual_norm)+I::point(eta_f)*delta_f)*inverse*inverse;
    return error.hi>=0 ? error.hi : infinity;
}

// Enclose the exact candidate RSS/bread covariance and compare the auxiliary
// native OLS point result against it. This veto is also required when a later
// established projection fallback, rather than the structured FE bound,
// accepts the candidate.
inline OrdinaryStructuredHomoskedasticNativeCheck
ordinary_structured_homoskedastic_native_check(
    const OrdinaryStructuredCoefficientInput& input) {
    using I=OlsProofInterval;using Real=long double;
    OrdinaryStructuredHomoskedasticNativeCheck check;
    const Real maximum=std::numeric_limits<Real>::max();
    const int p=static_cast<int>(input.coefficients.size());
    const auto& numerical=input.numerical;
    auto finite_nonnegative=[&](Real value) {return value>=0 && value<=maximum;};
    if (!input.ready || !p || numerical.bread.size()!=static_cast<std::size_t>(p)*p ||
        input.auxiliary_homoskedastic_covariance.rows()!=p ||
        input.auxiliary_homoskedastic_covariance.cols()!=p ||
        !(input.auxiliary_covariance_df>0) || !(input.auxiliary_covariance_df<=maximum) ||
        !finite_nonnegative(input.covariance_limit) ||
        !finite_nonnegative(numerical.normalized_matrix_norm) ||
        !finite_nonnegative(numerical.normalized_error) ||
        !(numerical.residual_squared.lo>=0) ||
        !(numerical.residual_squared.hi>=numerical.residual_squared.lo) ||
        !(numerical.residual_squared.hi<=maximum)) return check;
    const I reduction=I::point(numerical.normalized_matrix_norm)*
        I::point(numerical.normalized_error)*I::point(numerical.normalized_error);
    const Real candidate_lo=std::max(0.0L,
        I::down(numerical.residual_squared.lo-reduction.hi));
    const Real candidate_hi=numerical.residual_squared.hi;
    if (!finite_nonnegative(candidate_lo) || !finite_nonnegative(candidate_hi) ||
        candidate_lo>candidate_hi) return check;
    check.candidate_rss={candidate_lo,candidate_hi};check.eligible=true;
    if (candidate_lo==0 && candidate_hi==0) {
        for (int j=0;j<p;++j) for (int k=0;k<p;++k)
            if (input.auxiliary_homoskedastic_covariance(j,k)!=0) return check;
        check.exact_zero=true;check.relative_effect=0;check.passed=true;return check;
    }
    if (!(candidate_lo>0)) return check;
    const I inverse_df{I::down(1/input.auxiliary_covariance_df),
                       I::up(1/input.auxiliary_covariance_df)};
    const I covariance_scale=check.candidate_rss*inverse_df;
    std::vector<I> covariance(static_cast<std::size_t>(p)*p);
    for (int j=0;j<p;++j) for (int k=0;k<p;++k)
        covariance[static_cast<std::size_t>(j)*p+k]=
            covariance_scale*numerical.bread[static_cast<std::size_t>(j)*p+k];
    Real native_effect=0;
    for (int j=0;j<p;++j) for (int k=0;k<p;++k) {
        const I& value=covariance[static_cast<std::size_t>(j)*p+k];
        const I& left=covariance[static_cast<std::size_t>(j)*p+j];
        const I& right=covariance[static_cast<std::size_t>(k)*p+k];
        if (!(left.lo>0) || !(right.lo>0) || !(value.lo<=value.hi)) return check;
        const Real left_root=I::down(std::sqrt(left.lo));
        const Real right_root=I::down(std::sqrt(right.lo));
        const Real scale=(I::point(left_root)*I::point(right_root)).lo;
        if (!(scale>0)) return check;
        const Real point=input.auxiliary_homoskedastic_covariance(j,k);
        if (!(point>=-std::numeric_limits<double>::max()) ||
            !(point<=std::numeric_limits<double>::max())) return check;
        const Real error=I::up(std::max(std::abs(point-value.lo),std::abs(point-value.hi)));
        native_effect=std::max(native_effect,I::up(error/scale));
    }
    if (!finite_nonnegative(native_effect)) return check;
    check.relative_effect=native_effect;
    check.passed=native_effect<=input.covariance_limit;
    return check;
}

// For U=A+E+F, with E=P U-A and F=(I-P)U, bound the change in
// RSS*(U'WU)^-1 caused by the certified FE and affine errors. This controls
// homoskedastic covariance only; robust and clustered meat require different
// information. An inconclusive interval deliberately returns passed=false so
// the established independent reference remains authoritative.
inline OrdinaryStructuredHomoskedasticCovarianceCheck
ordinary_structured_homoskedastic_covariance_check(
    const OrdinaryStructuredCoefficientInput& input,
    const std::vector<long double>& fe,const std::vector<long double>& affine,
    long double coefficient_bound) {
    using I=OlsProofInterval;using Real=long double;
    OrdinaryStructuredHomoskedasticCovarianceCheck check;
    const Real maximum=std::numeric_limits<Real>::max();
    const int p=static_cast<int>(input.coefficients.size());
    const auto& numerical=input.numerical;
    auto finite_nonnegative=[&](Real value) {
        return value>=0 && value<=maximum;
    };
    if (!input.ready || !p || fe.size()!=static_cast<std::size_t>(p+1) ||
        affine.size()!=fe.size() || numerical.bread.size()!=static_cast<std::size_t>(p)*p ||
        input.auxiliary_homoskedastic_covariance.rows()!=p ||
        input.auxiliary_homoskedastic_covariance.cols()!=p ||
        !finite_nonnegative(coefficient_bound) ||
        !(input.auxiliary_covariance_df>0) || !(input.auxiliary_covariance_df<=maximum) ||
        !finite_nonnegative(input.covariance_limit) ||
        !finite_nonnegative(numerical.normalized_matrix_norm) ||
        !finite_nonnegative(numerical.normalized_error) ||
        !(numerical.residual_squared.lo>=0) ||
        !(numerical.residual_squared.hi>=numerical.residual_squared.lo) ||
        !(numerical.residual_squared.hi<=maximum)) return check;

    const auto native=ordinary_structured_homoskedastic_native_check(input);
    if (!native.eligible) return check;
    std::vector<Real> e(p),f(p);
    for (int j=0;j<=p;++j) {
        if (!finite_nonnegative(fe[j]) || !finite_nonnegative(affine[j])) return check;
        if (j) {e[j-1]=affine[j];f[j-1]=fe[j];}
    }
    I eta_squared_interval,phi_squared_interval;
    for (int j=0;j<p;++j) for (int k=0;k<p;++k) {
        const auto& entry=numerical.bread[static_cast<std::size_t>(j)*p+k];
        if (!(entry.lo<=entry.hi)) return check;
        const Real bread=entry.absolute();
        if (!finite_nonnegative(bread)) return check;
        const I eta_product=I::point(bread)*I::point(e[j])*I::point(e[k]);
        const I phi_product=I::point(bread)*I::point(f[j])*I::point(f[k]);
        eta_squared_interval=eta_squared_interval+eta_product;
        phi_squared_interval=phi_squared_interval+phi_product;
    }
    const Real eta_squared=eta_squared_interval.hi;
    const Real phi_squared=phi_squared_interval.hi;
    if (!finite_nonnegative(eta_squared) || !finite_nonnegative(phi_squared)) return check;
    const Real eta=eta_squared==0 ? 0 : I::up(std::sqrt(eta_squared));
    const I delta_interval=I::point(2)*I::point(eta)+I::point(eta_squared)+
        I::point(phi_squared);
    const Real delta=delta_interval.hi;
    if (!finite_nonnegative(delta) || !(delta<1)) return check;
    check.eta_squared=eta_squared;check.phi_squared=phi_squared;check.delta=delta;

    const Real candidate_lo=native.candidate_rss.lo;
    const Real candidate_hi=native.candidate_rss.hi;
    check.candidate_rss=native.candidate_rss;

    I eps_e_interval=I::point(affine[0]),eps_f_interval=I::point(fe[0]);
    for (int j=0;j<p;++j) {
        const I beta=I::point(std::abs(static_cast<Real>(input.coefficients[j])))+
            I::point(coefficient_bound);
        eps_e_interval=eps_e_interval+I::point(e[j])*beta;
        eps_f_interval=eps_f_interval+I::point(f[j])*beta;
    }
    const Real eps_e=eps_e_interval.hi,eps_f=eps_f_interval.hi;
    if (!finite_nonnegative(eps_e) || !finite_nonnegative(eps_f)) return check;

    // The existing globally numerical-perfect policy uses an absolute
    // covariance allowance floor^2 * bread. A relative variance target has
    // no meaning at zero noise. Both candidate and true residual norms must
    // lie inside that same FP64 envelope, with the FE Gram still verified.
    const Real floor=input.residual_roundoff_floor;
    if (floor>0 && floor<=maximum) {
        const I floor_squared=I::point(floor)*I::point(floor);
        const Real candidate_root=candidate_hi==0 ? 0 : I::up(std::sqrt(candidate_hi));
        const I true_root=I::point(candidate_root)+I::point(eps_e);
        const Real true_upper=(true_root*true_root).hi;
        const I plus=I::point(1)+I::point(delta),minus=I::point(1)-I::point(delta);
        if (candidate_hi<=floor_squared.lo && true_upper<=floor_squared.lo &&
            minus.lo>0 && floor_squared.hi<=maximum) {
            const I inverse_df{I::down(1/input.auxiliary_covariance_df),
                               I::up(1/input.auxiliary_covariance_df)};
            const I rss_scale=I{0,true_upper}*inverse_df;
            const Real distortion=I::up(delta/minus.lo);
            bool passed=true;Real worst=0;
            for (int j=0;j<p && passed;++j) for (int k=0;k<p;++k) {
                const I& left=numerical.bread[static_cast<std::size_t>(j)*p+j];
                const I& right=numerical.bread[static_cast<std::size_t>(k)*p+k];
                if (!(left.lo>0) || !(right.lo>0)) {passed=false;break;}
                const I diagonal_lower=I::point(I::down(std::sqrt(left.lo)))*
                    I::point(I::down(std::sqrt(right.lo)));
                const I diagonal_upper=I::point(I::up(std::sqrt(left.hi)))*
                    I::point(I::up(std::sqrt(right.hi)));
                const Real padding=(I::point(distortion)*diagonal_upper).hi;
                const I true_bread=numerical.bread[static_cast<std::size_t>(j)*p+k]+
                    I{-padding,padding};
                const I covariance=rss_scale*true_bread;
                const Real point=input.auxiliary_homoskedastic_covariance(j,k);
                const Real error=I::up(std::max(std::abs(point-covariance.lo),
                                               std::abs(point-covariance.hi)));
                const I allowance=floor_squared*diagonal_lower*
                    I{I::down(1/plus.hi),I::up(1/plus.hi)};
                if (!(allowance.lo>0) || !finite_nonnegative(error) ||
                    error>allowance.lo) {passed=false;break;}
                worst=std::max(worst,I::up(error/allowance.lo));
            }
            if (passed) {
                check.eligible=true;check.passed=true;check.numerical_perfect=true;
                check.true_rss={0,true_upper};
                // This ratio is relative to the absolute rounding allowance,
                // not relative to a variance that can be mathematically zero.
                check.relative_effect=worst;
                return check;
            }
        }
    }

    check.eligible=true;
    if (candidate_lo==0 && candidate_hi==0) {
        if (eps_e!=0 || !native.exact_zero || !native.passed) return check;
        check.exact_zero=true;check.true_rss={0,0};check.rho_lower=1;check.rho_upper=1;
        check.fe_relative_effect=0;check.native_relative_effect=0;check.relative_effect=0;
        check.passed=true;return check;
    }
    if (!(candidate_lo>0)) return check;

    const Real eps_f_squared=(I::point(eps_f)*I::point(eps_f)).hi;
    const Real radicand=std::max(0.0L,I::down(candidate_lo-eps_f_squared));
    Real true_root_lower=I::down(std::sqrt(radicand));
    true_root_lower=std::max(0.0L,I::down(true_root_lower-eps_e));
    const Real true_lo=std::max(0.0L,
        (I::point(true_root_lower)*I::point(true_root_lower)).lo);
    const Real candidate_root_hi=I::up(std::sqrt(candidate_hi));
    const Real true_root_hi=I::up(candidate_root_hi+eps_e);
    const Real true_hi=(I::point(true_root_hi)*I::point(true_root_hi)).hi;
    if (!(true_lo>0) || !finite_nonnegative(true_hi) || true_lo>true_hi) return check;
    check.true_rss={true_lo,true_hi};

    const I one_plus_interval=I::point(1)+I::point(delta);
    const I one_minus_interval=I::point(1)-I::point(delta);
    const Real one_plus=one_plus_interval.hi,one_minus=one_minus_interval.lo;
    if (!(one_minus>0)) return check;
    Real rho_lower=I::down(true_lo/candidate_hi);
    rho_lower=I::down(rho_lower/one_plus);
    Real rho_upper=I::up(true_hi/candidate_lo);
    rho_upper=I::up(rho_upper/one_minus);
    if (!(rho_lower>0) || !(rho_upper>=rho_lower) || !(rho_upper<=maximum)) return check;
    check.rho_lower=rho_lower;check.rho_upper=rho_upper;

    Real lower_effect=I::up(I::up(1/rho_lower)-1);
    Real upper_effect=I::up(1-I::down(1/rho_upper));
    const Real fe_effect=std::max({0.0L,lower_effect,upper_effect});
    const Real native_true_effect=I::up(native.relative_effect/rho_lower);
    const Real total=I::up(fe_effect+native_true_effect);
    if (!finite_nonnegative(total)) return check;
    check.fe_relative_effect=fe_effect;check.native_relative_effect=native_true_effect;
    check.relative_effect=total;check.passed=total<=input.covariance_limit;
    return check;
}

}}
#endif
