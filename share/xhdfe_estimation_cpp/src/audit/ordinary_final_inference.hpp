#ifndef XHDFE_ORDINARY_FINAL_INFERENCE_HPP
#define XHDFE_ORDINARY_FINAL_INFERENCE_HPP

#include "hdfe/parallel_work_observer.hpp"
#include "ordinary_sandwich_profile.hpp"
#include "ordinary_structured_coefficient.hpp"

namespace hdfe { namespace detail {

struct OrdinaryFinalInferenceMeans {
    Eigen::VectorXd x;
    double y=0,denominator=0;
    std::vector<OlsProofInterval> enclosed_x;
    OlsProofInterval enclosed_y;
    bool ready=false;
};

struct OrdinaryFinalInferenceProof {
    bool ready=false,passed=false;
    const char* reason="unprepared";
    long double limit=0,coefficient_limit=0,coefficient_bound=0;
    long double residual_error_squared=0,residual_formation=0,contraction=0;
    std::vector<OlsProofInterval> covariance;
    std::vector<long double> fe_effect,allowance,score_error;
    std::vector<long double> noise_quantum,perfect_envelope;
    Eigen::VectorXd coefficients;
    int columns=0;
    bool reference_used=false;
};

struct OrdinaryFinalInferenceCandidate {
    OlsResult fit;
    OrdinaryFinalInferenceProof proof;
};

inline long double ordinary_final_root(long double x) {
    return x==0 ? 0 : OlsProofInterval::up(std::sqrt(x));
}

inline long double ordinary_final_root_lower(long double x) {
    return x>0 ? std::max(0.0L,OlsProofInterval::down(std::sqrt(x))) : 0;
}

inline long double ordinary_final_noise_quantum(long double rss_lower,long double bread_lower) {
    using I=OlsProofInterval;
    if (!(rss_lower>0 && bread_lower>0)) return 0;
    const long double product=(I::point(rss_lower)*I::point(bread_lower)).lo;
    return std::max(0.0L,(I::point(64*std::numeric_limits<double>::epsilon())*
        I::point(ordinary_final_root_lower(product))).lo);
}

// The point expression is the existing cached one-way intercept expression.
// The terminal guard checks the actual postprocessed matrix independently.
inline Eigen::MatrixXd ordinary_final_native_covariance(
    const OlsResult& fit,const OrdinaryFinalInferenceMeans& means) {
    const int p=static_cast<int>(fit.coefficients.size()),n=fit.nobs;
    if (p<=0 || n<=0 || means.x.size()!=p || fit.cluster_ux.size()!=p ||
        fit.covariance.rows()!=p || fit.covariance.cols()!=p || !(fit.cov_scale>0))
        throw std::runtime_error("Ordinary final inference: incomplete native one-way covariance; no estimates returned");
    Eigen::MatrixXd value(p+1,p+1);
    value.topLeftCorner(p,p)=fit.covariance;
    const double inv_n=1.0/static_cast<double>(n);
    const Eigen::MatrixXd unscaled=fit.covariance/fit.cov_scale;
    const Eigen::VectorXd B=fit.xtx_inv*means.x;
    const Eigen::VectorXd cov_xbar=unscaled*means.x;
    Eigen::VectorXd cross=inv_n*(fit.xtx_inv*fit.cluster_ux)-cov_xbar;
    const double base=inv_n*inv_n*fit.cluster_u2-
        2.0*inv_n*B.dot(fit.cluster_ux)+means.x.dot(cov_xbar);
    cross*=fit.cov_scale;
    long double intercept=base;intercept*=static_cast<long double>(fit.cov_scale);
    if (intercept<0) intercept=0; // Same existing native nonnegative convention.
    for (int j=0;j<p;++j) value(j,p)=value(p,j)=cross[j];
    value(p,p)=static_cast<double>(intercept);
    return value;
}

inline bool ordinary_final_covariance_readback(const OrdinaryFinalInferenceProof& proof,
    const Eigen::MatrixXd& native,double scale) {
    using I=OlsProofInterval;using Real=long double;
    const int q=proof.columns;
    if (!proof.ready || q<=0 || native.rows()!=q || native.cols()!=q ||
        !(scale>0) || !ieee_finite(scale) ||
        (!proof.noise_quantum.empty() && proof.noise_quantum.size()!=static_cast<std::size_t>(q)) ||
        (!proof.perfect_envelope.empty() && proof.perfect_envelope.size()!=static_cast<std::size_t>(q)*q)) return false;
    const I multiplier=I::point(scale);
    std::vector<Real> lower(q),raw_root(q);
    for (int j=0;j<q;++j) {
        const auto index=static_cast<std::size_t>(j)*q+j;
        const I raw=proof.covariance[index]-I::point(proof.fe_effect[index]);
        raw_root[j]=ordinary_final_root_lower(raw.lo);
        const I diagonal=raw*multiplier;
        lower[j]=std::max(0.0L,diagonal.lo);
    }
    // Two scalar multiplications bound the final SSC operations. This is
    // arithmetic allowance only; no FE error enters it.
    const Real gamma=OrdinarySandwichSum::gamma(2,std::numeric_limits<double>::epsilon());
    for (int j=0;j<q;++j) for (int k=0;k<q;++k) {
        const auto index=static_cast<std::size_t>(j)*q+k;
        const Real point=native(j,k);
        const I interval=proof.covariance[index]*multiplier;
        if (!ieee_finite(native(j,k)) || !(interval.lo<=interval.hi)) return false;
        const Real difference=std::max(std::abs(point-interval.lo),std::abs(point-interval.hi));
        const Real delta=point==interval.lo && point==interval.hi ? 0 : I::up(difference);
        const I error=I::point(proof.fe_effect[index])*multiplier+I::point(delta);
        I roundoff=I::point(proof.allowance[index]);
        // A zero lower bound means positive variance is unproved, not that
        // variance is zero. Certified-positive pairs retain their old target.
        if (!proof.noise_quantum.empty() && (raw_root[j]==0 || raw_root[k]==0)) {
            const I left=I::point(proof.noise_quantum[j]),right=I::point(proof.noise_quantum[k]);
            roundoff=roundoff+left*I::point(raw_root[k])+I::point(raw_root[j])*right+left*right;
        }
        if (!proof.perfect_envelope.empty()) roundoff=roundoff+I::point(proof.perfect_envelope[index]);
        const I arithmetic=roundoff*multiplier+
            I::point(gamma)*I::point(std::abs(point));
        const Real left=lower[j]==0 ? 0 : std::max(0.0L,I::down(std::sqrt(lower[j])));
        const Real right=lower[k]==0 ? 0 : std::max(0.0L,I::down(std::sqrt(lower[k])));
        const Real diagonal_scale=(I::point(left)*I::point(right)).lo;
        const I allowed=I::point(proof.limit)*I::point(std::max(0.0L,diagonal_scale))+arithmetic;
        if (!(error.hi>=0 && error.hi<=std::numeric_limits<Real>::max()) ||
            !(allowed.lo>=0 && allowed.hi<=std::numeric_limits<Real>::max()) ||
            !(interval.absolute()<=std::numeric_limits<Real>::max()) || error.hi>allowed.lo) return false;
    }
    return true;
}

inline OrdinaryFinalInferenceProof ordinary_final_inference_bound(
    const OlsResult& fit,const OlsNumericalCertificate& numerical,
    const OrdinarySandwichProfile& profile,const OrdinarySandwichFullMoments& moments,
    const OrdinaryFinalInferenceMeans& means,const std::vector<long double>& fe,
    const std::vector<long double>& affine,const HdfeOptions& options,long double coefficient_limit) {
    using I=OlsProofInterval;using Real=long double;
    OrdinaryFinalInferenceProof proof;
    const int p=static_cast<int>(fit.coefficients.size()),q=p+1;
    const Real maximum=std::numeric_limits<Real>::max();
    auto positive_finite=[&](Real x) {return x>=0 && x<=maximum;};
    if (!p || fit.nobs<=0 || !profile.eligible || !profile.fp64_floor_valid || !means.ready ||
        means.x.size()!=p || means.enclosed_x.size()!=static_cast<std::size_t>(p) ||
        profile.column_norm.size()!=static_cast<std::size_t>(p) ||
        profile.max_cell_influence.size()!=static_cast<std::size_t>(p) ||
        numerical.bread.size()!=static_cast<std::size_t>(p)*p ||
        fe.size()!=static_cast<std::size_t>(q) || affine.size()!=fe.size() ||
        moments.covariance.size()!=static_cast<std::size_t>(q)*q ||
        moments.allowance.size()!=moments.covariance.size()) {
        proof.reason="incomplete_proof";return proof;
    }
    proof.limit=std::max(options.tol,64*std::numeric_limits<double>::epsilon());
    proof.coefficient_limit=coefficient_limit;
    OrdinaryStructuredCoefficientInput coefficient;
    coefficient.ready=true;coefficient.coefficients=fit.coefficients;coefficient.numerical=numerical;
    proof.coefficient_bound=ordinary_structured_coefficient_bound(coefficient,fe,affine);
    if (!(proof.coefficient_bound<=proof.coefficient_limit)) {
        proof.reason="coefficient_bound";return proof;
    }
    I phi2,eta2,dF=I::point(fe[0]),dP=I::point(affine[0]);
    for (int j=0;j<q;++j) {
        if (!positive_finite(fe[j]) || !positive_finite(affine[j])) {
            proof.reason="projection_bound_range";return proof;
        }
        if (!j) continue;
        const I f=I::point(fe[j]),a=I::point(affine[j]),b=I::point(std::abs(Real(fit.coefficients[j-1])));
        phi2=phi2+f*f;eta2=eta2+a*a;dF=dF+f*b;dP=dP+a*b;
    }
    const Real phi=ordinary_final_root(phi2.hi),eta=ordinary_final_root(eta2.hi);
    const I su=I::point(numerical.singular_lower);
    const Real remaining=(su*su-phi2).lo;
    const Real s=remaining>0 ? I::down(I::down(std::sqrt(remaining))-eta) : -1;
    if (!(s>0)) {proof.reason="singular_bound";return proof;}
    const I inverse{I::down(1/s),I::up(1/s)};
    const Real kappa=ordinary_final_root((phi2+eta2).hi);
    const I nu=I::point(ordinary_final_root(numerical.normalized_matrix_norm))*
        I::point(numerical.normalized_error);
    const Real projection=std::min(((I::point(1)+I::point(kappa)*inverse)*nu).hi,
        (I::point(numerical.normal_defect)*inverse).hi);
    const I correction=I::point(projection)+(I::point(phi)*dF+
        I::point(eta)*I::point(numerical.residual_norm))*inverse;
    const I D2=dF*dF+dP*dP+correction*correction;
    proof.residual_error_squared=D2.hi;

    bool beta_zero=true;
    I prediction;
    for (int j=0;j<p;++j) {
        beta_zero=beta_zero && (ieee_bits_detail::bits(fit.coefficients[j])&0x7fffffffffffffffULL)==0;
        prediction=prediction+I::point(std::abs(Real(fit.coefficients[j])))*
            I::point(profile.column_norm[j]);
    }
    Real rho=0;
    if (moments.residual_formation_bound>=0 && moments.residual_formation_bound<=maximum)
        rho=moments.residual_formation_bound;
    else if (!((beta_zero && moments.response_matches) ||
          (numerical.residual_squared.hi==0 && moments.residual_zero))) {
        const Real u=std::numeric_limits<double>::epsilon()/2.0L;
        const Real gamma=OrdinarySandwichSum::gamma(static_cast<std::uint64_t>(p)+1,u);
        // At most p products, p additions and the final subtraction per row.
        // min(), rather than denorm_min(), also covers a flush-to-zero runtime.
        const I tiny=I::point(ordinary_final_root(static_cast<Real>(fit.nobs)))*
            I::point(2.0L*p+1)*I::point(std::numeric_limits<double>::min());
        const I numerator=I::point(u)*I::point(numerical.residual_norm)+
            I::point(1+u)*I::point(gamma)*prediction+tiny;
        rho=(numerator*I{I::down(1/(1-2*u)),I::up(1/(1-2*u))}).hi;
    }
    proof.residual_formation=rho;
    const Real dR=(I::point(ordinary_final_root(D2.hi))+I::point(rho)).hi;
    if (!positive_finite(dR)) {proof.reason="residual_bound_range";return proof;}
    const Real sigma=ordinary_final_root(profile.max_cell_residual);
    std::vector<Real> H(static_cast<std::size_t>(p)*p),scores(q),scales(p),rhs(p),errors(q);
    for (int j=0;j<q;++j)
        scores[j]=ordinary_final_root(std::max(0.0L,moments.covariance[static_cast<std::size_t>(j)*q+j].hi));
    for (int j=0;j<p;++j) {
        const Real diagonal=numerical.bread[static_cast<std::size_t>(j)*p+j].hi;
        if (!(diagonal>0 && diagonal<=maximum)) {proof.reason="bread_range";return proof;}
        scales[j]=std::ldexp(1.0L,std::ilogb(diagonal)/2);
        I x_error;
        for (int k=0;k<p;++k) {
            I h;
            for (int l=0;l<p;++l) {
                const I delta=I::point(profile.column_norm[k])*I::point(affine[l+1])+
                    I::point(affine[k+1])*I::point(profile.column_norm[l])+
                    I::point(fe[k+1])*I::point(fe[l+1])+I::point(affine[k+1])*I::point(affine[l+1]);
                h=h+delta*I::point(numerical.bread[static_cast<std::size_t>(l)*p+j].absolute());
            }
            H[static_cast<std::size_t>(j)*p+k]=h.hi;
            const I f=I::point(fe[k+1]),a=I::point(affine[k+1]);
            x_error=x_error+I::point(numerical.bread[static_cast<std::size_t>(k)*p+j].absolute())*
                I::point(ordinary_final_root((f*f+a*a).hi));
        }
        I value=I::point(ordinary_final_root(profile.max_cell_influence[j]))*I::point(dR)+
            (I::point(sigma)+I::point(dR))*x_error;
        for (int k=0;k<p;++k)
            value=value+I::point(H[static_cast<std::size_t>(j)*p+k])*I::point(scores[k]);
        rhs[j]=value.hi;
    }
    Real ell=0,maximum_rhs=0;
    for (int j=0;j<p;++j) {
        I row;
        for (int k=0;k<p;++k) row=row+I::point(H[static_cast<std::size_t>(j)*p+k])*
            I::point(scales[k])*I{I::down(1/scales[j]),I::up(1/scales[j])};
        ell=std::max(ell,row.hi);
        maximum_rhs=std::max(maximum_rhs,(I::point(rhs[j])*I{I::down(1/scales[j]),I::up(1/scales[j])}).hi);
    }
    proof.contraction=ell;
    if (!(ell>=0 && ell<1)) {proof.reason="score_contraction";return proof;}
    const Real denominator=I::down(1-ell);
    for (int j=0;j<p;++j)
        errors[j]=(I::point(scales[j])*I::point(maximum_rhs)*
            I{I::down(1/denominator),I::up(1/denominator)}).hi;
    I intercept=I::point(ordinary_final_root(static_cast<Real>(profile.max_cell_rows)))*
        I{I::down(1.0L/fit.nobs),I::up(1.0L/fit.nobs)}*I::point(dR);
    for (int j=0;j<p;++j) intercept=intercept+I::point(means.enclosed_x[j].absolute())*I::point(errors[j]);
    errors[p]=intercept.hi;
    proof.columns=q;proof.covariance=moments.covariance;proof.allowance=moments.allowance;
    proof.score_error=errors;proof.fe_effect.resize(static_cast<std::size_t>(q)*q);
    const Real residual_lower=std::max(0.0L,(I::point(ordinary_final_root_lower(
        numerical.residual_squared.lo))-I::point(dR)).lo);
    const Real rss_lower=std::max(0.0L,(I::point(residual_lower)*I::point(residual_lower)).lo);
    proof.noise_quantum.resize(q);
    for (int j=0;j<p;++j) {
        const I f=I::point(fe[j+1]),a=I::point(affine[j+1]);
        const Real norm_upper=(I::point(profile.column_norm[j])+
            I::point(ordinary_final_root((f*f+a*a).hi))).hi;
        const Real squared=(I::point(norm_upper)*I::point(norm_upper)).hi;
        const Real bread_lower=squared>0 ? std::max(0.0L,I::down(1/squared)) : 0;
        proof.noise_quantum[j]=ordinary_final_noise_quantum(rss_lower,bread_lower);
    }
    proof.noise_quantum[p]=ordinary_final_noise_quantum(rss_lower,I::down(1.0L/fit.nobs));
    for (int j=0;j<q;++j) for (int k=0;k<q;++k) {
        const I value=I::point(scores[j])*I::point(errors[k])+I::point(errors[j])*I::point(scores[k])+
            I::point(errors[j])*I::point(errors[k]);
        if (!positive_finite(value.hi)) {proof.reason="covariance_bound_range";return proof;}
        proof.fe_effect[static_cast<std::size_t>(j)*q+k]=value.hi;
    }
    proof.coefficients.resize(q);proof.coefficients.head(p)=fit.coefficients;
    proof.coefficients[p]=means.y-means.x.dot(fit.coefficients);
    I alpha=means.enclosed_y,alpha_error;
    for (int j=0;j<p;++j) {
        alpha=alpha-means.enclosed_x[j]*I::point(fit.coefficients[j]);
        alpha_error=alpha_error+I::point(means.enclosed_x[j].absolute())*I::point(proof.coefficient_bound);
    }
    const Real alpha_native=proof.coefficients[p];
    const Real alpha_round=I::up(std::max(std::abs(alpha_native-alpha.lo),std::abs(alpha_native-alpha.hi)));
    const I alpha_total=alpha_error+I::point(alpha_round);
    const Real alpha_scale=std::max(1.0L,I::down(std::abs(alpha_native)-alpha_total.hi));
    if (!(alpha_total.hi<=(I::point(proof.coefficient_limit)*I::point(alpha_scale)).lo)) {
        proof.reason="intercept_coefficient_bound";return proof;
    }
    proof.ready=true;
    proof.passed=ordinary_final_covariance_readback(proof,ordinary_final_native_covariance(fit,means),fit.cov_scale);
    proof.reason=proof.passed ? "ok" : "full_covariance_bound";
    return proof;
}

} }
#endif
