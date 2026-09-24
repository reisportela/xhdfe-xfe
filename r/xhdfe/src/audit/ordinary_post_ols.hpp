#pragma once

#include "ordinary_certification.hpp"
#include "ordinary_three_fe_support.hpp"
#include "ordinary_identity_inference.hpp"
#include <functional>
#include <memory>
#include <map>

namespace hdfe { namespace detail {

// Synchronous, fit-owned receipt. It is created only after the last N1 retry
// and rank selection. No lease survives a move, another fit, or a cache hit.
class OrdinaryPostOlsIdentity final {
    std::vector<std::function<bool()>> witnesses_;
public:
    OrdinaryPostOlsIdentity()=default;
    OrdinaryPostOlsIdentity(const OrdinaryPostOlsIdentity&)=delete;
    OrdinaryPostOlsIdentity& operator=(const OrdinaryPostOlsIdentity&)=delete;
    OrdinaryPostOlsIdentity(OrdinaryPostOlsIdentity&&)=delete;
    OrdinaryPostOlsIdentity& operator=(OrdinaryPostOlsIdentity&&)=delete;
    template<class Derived> void pin(const Eigen::MatrixBase<Derived>& expression) {
        const auto& value=expression.derived();
        using Scalar=typename Derived::Scalar;
        const auto* pointer=value.data();
        const auto rows=value.rows(),cols=value.cols(),inner=value.innerStride(),outer=value.outerStride();
        auto copy=std::make_shared<std::vector<unsigned char>>(static_cast<std::size_t>(value.size())*sizeof(Scalar));
        for(Eigen::Index j=0;j<cols;++j) for(Eigen::Index i=0;i<rows;++i)
            std::memcpy(copy->data()+(static_cast<std::size_t>(j)*rows+i)*sizeof(Scalar),
                pointer+(Derived::IsRowMajor ? i*outer+j*inner : j*outer+i*inner),sizeof(Scalar));
        witnesses_.emplace_back([&value,pointer,rows,cols,inner,outer,copy]() {
            if(value.data()!=pointer || value.rows()!=rows || value.cols()!=cols ||
                value.innerStride()!=inner || value.outerStride()!=outer) return false;
            for(Eigen::Index j=0;j<cols;++j) for(Eigen::Index i=0;i<rows;++i)
                if(std::memcmp(copy->data()+(static_cast<std::size_t>(j)*rows+i)*sizeof(Scalar),
                    pointer+(Derived::IsRowMajor ? i*outer+j*inner : j*outer+i*inner),sizeof(Scalar))) return false;
            return true;
        });
    }
    void pin(const std::vector<int>& value) {
        const auto copy=value;const auto* pointer=value.data();
        witnesses_.emplace_back([&value,pointer,copy]() {return value.data()==pointer && value==copy;});
    }
    template<class Producer> void pin_metadata(Producer producer) {
        const auto before=producer();
        witnesses_.emplace_back([producer,before]() {return producer()==before;});
    }
    bool matches() const {
        for(const auto& witness:witnesses_) if(!witness()) return false;
        return true;
    }
};

struct OrdinaryProjectionComponents {
    std::vector<long double> fe,affine;
    bool ready=false;
    const char* reason="projection_bound_unavailable";
};

inline OrdinaryProjectionComponents ordinary_post_projection_components(
    const Eigen::VectorXd& y,const Eigen::MatrixXd& X,
    const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights,
    const AbsorptionResult& result,const HdfeOptions& options) {
    OrdinaryProjectionComponents out;
    if(!ordinary_audit_enabled()) {out.reason="audit_disabled";return out;}
    using I=OlsProofInterval;using Real=long double;
    const int n=static_cast<int>(y.size()),q=static_cast<int>(X.cols())+1;
    out.fe.assign(q,0);out.affine.assign(q,0);
    if(fes.empty()) {out.ready=true;out.reason="no_projection";return out;}
    if(fes.size()==1) {
        if(exact_level_projection(y,X,fes[0],weights,result.y_tilde,result.X_tilde)) {
            out.ready=true;out.reason="exact_one_fe";return out;
        }
        std::map<int,std::vector<int>> groups;
        for(int row=0;row<n;++row) if(!weights || (*weights)[row]>0) groups[fes[0][row]].push_back(row);
        for(int rhs=0;rhs<q;++rhs) {
            I energy,affine;
            auto raw=[&](int i) {return rhs ? X(i,rhs-1) : y[i];};
            auto residual=[&](int i) {return rhs ? result.X_tilde(i,rhs-1) : result.y_tilde[i];};
            for(const auto& group:groups) {
                I mass,gradient;
                // Any witness is admissible; its full defect is enclosed.
                const Real witness=Real(raw(group.second.front()))-residual(group.second.front());
                for(int i:group.second) {
                    const I w=I::point(weights ? (*weights)[i] : 1);
                    mass=mass+w;gradient=gradient+w*I::point(residual(i));
                    const I defect=I::point(raw(i))-I::point(residual(i))-I::point(witness);
                    const I radius=I::point(defect.absolute());affine=affine+w*radius*radius;
                }
                if(!(mass.lo>0)) {out.reason="one_fe_weight_range";return out;}
                const I radius=I::point(gradient.absolute());
                energy=energy+radius*radius*I{I::down(1/mass.hi),I::up(1/mass.lo)};
            }
            out.fe[rhs]=ordinary_final_root(std::max(0.0L,energy.hi));
            out.affine[rhs]=ordinary_final_root(std::max(0.0L,affine.hi));
        }
    } else {
        // Collect all component bounds. This sentinel never becomes a solver
        // target or an acceptance limit; the later proof supplies its contract.
        const std::vector<Real> collect(q,std::numeric_limits<Real>::max());
        if(fes.size()==2) {
            const auto check=certify_ordinary_tree_space(y,X,fes,weights,result,collect);
            if(!check.eligible || check.fe_bound.size()!=static_cast<std::size_t>(q)) return out;
            out.fe=check.fe_bound;out.affine=check.affine_bound;
        } else if(fes.size()==3) {
            OrdinaryThreeFeSupportWorkspace workspace;
            const auto check=certify_ordinary_three_fe_support(y,X,fes,weights,result,collect,
                &workspace,options.num_threads,nullptr);
            if(!check.eligible || check.fe_bound.size()!=static_cast<std::size_t>(q)) return out;
            out.fe=check.fe_bound;out.affine=check.affine_bound;
        } else {out.reason="projection_component_scope";return out;}
    }
    for(int j=0;j<q;++j)
        if(!(out.fe[j]>=0 && out.fe[j]<=std::numeric_limits<Real>::max() &&
             out.affine[j]>=0 && out.affine[j]<=std::numeric_limits<Real>::max())) return out;
    out.ready=true;out.reason="projection_components";return out;
}

inline OrdinaryFinalInferenceMeans ordinary_post_means(const Eigen::VectorXd& y,
    const Eigen::MatrixXd& X,const Eigen::VectorXd* weights) {
    using I=OlsProofInterval;
    OrdinaryFinalInferenceMeans means;
    ExactBinaryProducts<> total;
    for(int i=0;i<y.size();++i) total.add_product(std::array<double,1>{weights ? (*weights)[i] : 1.0});
    const auto mass=total.interval();
    if(!total.valid() || !(mass.first>0)) return means;
    const I inverse{I::down(1/mass.second),I::up(1/mass.first)};
    means.x.resize(X.cols());means.enclosed_x.resize(X.cols());
    means.denominator=static_cast<double>((mass.first+mass.second)/2);
    for(int j=-1;j<X.cols();++j) {
        ExactBinaryProducts<> sum;
        for(int i=0;i<y.size();++i)
            sum.add_product(std::array<double,2>{weights ? (*weights)[i] : 1.0,j<0 ? y[i] : X(i,j)});
        if(!sum.valid()) return means;
        const auto interval=sum.interval();const I value=I{interval.first,interval.second}*inverse;
        const double point=static_cast<double>(value.lo/2+value.hi/2);
        if(!ieee_finite(point)) return means;
        if(j<0) {means.enclosed_y=value;means.y=point;}
        else {means.enclosed_x[j]=value;means.x[j]=point;}
    }
    means.ready=true;return means;
}

inline bool ordinary_post_coefficient_readback(const Eigen::VectorXd& native,
    const OlsResult& fit,const OrdinaryFinalInferenceMeans& means,bool recovered,
    long double bound,long double limit) {
    using I=OlsProofInterval;using Real=long double;
    const int p=static_cast<int>(fit.coefficients.size());
    if(native.size()!=p+(recovered ? 1 : 0) || !(bound>=0 && bound<=std::numeric_limits<Real>::max())) return false;
    for(int j=0;j<p;++j) {
        const Real error=I::up(std::abs(Real(native[j])-fit.coefficients[j])+bound);
        const Real scale=std::max(1.0L,I::down(std::abs(Real(fit.coefficients[j]))-bound));
        if(!(error<=limit*scale)) return false;
    }
    if(recovered) {
        if(!means.ready) return false;
        I alpha=means.enclosed_y,padding;
        for(int j=0;j<p;++j) {
            alpha=alpha-means.enclosed_x[j]*I::point(fit.coefficients[j]);
            padding=padding+I::point(means.enclosed_x[j].absolute())*I::point(bound);
        }
        alpha=alpha+I{-padding.hi,padding.hi};
        const Real error=I::up(std::max(std::abs(Real(native[p])-alpha.lo),std::abs(Real(native[p])-alpha.hi)));
        const Real scale=std::max(1.0L,std::min(std::abs(alpha.lo),std::abs(alpha.hi)));
        if(!(error<=limit*scale)) return false;
    }
    return true;
}

inline bool ordinary_post_sandwich_readback(const OrdinarySandwichProfile& profile,
    const Eigen::MatrixXd& covariance,double multiplier) {
    using I=OlsProofInterval;using Real=long double;
    const int p=static_cast<int>(profile.score_lower.size());
    if(!profile.eligible || !profile.fp64_floor_valid || covariance.rows()!=p || covariance.cols()!=p ||
        !(multiplier>0) || !ieee_finite(multiplier)) return false;
    for(int j=0;j<p;++j) {
        const Real low=profile.score_lower[j],high=profile.score_upper[j];
        const I interval=I{I::down(low*low),I::up(high*high)}*I::point(multiplier);
        const Real floor=profile.fp64_score_floor[j];
        const I allowance=(I::point(2)*I::point(high)*I::point(floor)+I::point(floor)*I::point(floor))*I::point(multiplier);
        const Real point=covariance(j,j);
        if(!ieee_finite(static_cast<double>(point)) || point<interval.lo-allowance.hi || point>interval.hi+allowance.hi) return false;
    }
    return true;
}

// A failed sufficient bound is inconclusive. A separated final point is a
// demonstrated error only when the complete independent enclosure is ready.
inline bool ordinary_post_full_separated(const OrdinaryFinalInferenceProof& proof,
    const Eigen::MatrixXd& native,double scale) {
    using I=OlsProofInterval;using Real=long double;
    const int p=proof.columns;
    if(!proof.ready || native.rows()!=p || native.cols()!=p || !(scale>0)) return false;
    for(int j=0;j<p;++j) for(int k=0;k<p;++k) {
        const auto at=static_cast<std::size_t>(j)*p+k;
        const I enclosed=(proof.covariance[at]+I{-proof.fe_effect[at],proof.fe_effect[at]})*I::point(scale);
        const Real left=std::max(0.0L,((proof.covariance[static_cast<std::size_t>(j)*p+j]+
            I::point(proof.fe_effect[static_cast<std::size_t>(j)*p+j]))*I::point(scale)).hi);
        const Real right=std::max(0.0L,((proof.covariance[static_cast<std::size_t>(k)*p+k]+
            I::point(proof.fe_effect[static_cast<std::size_t>(k)*p+k]))*I::point(scale)).hi);
        I arithmetic=I::point(proof.allowance[at])*I::point(scale);
        if(!proof.perfect_envelope.empty()) arithmetic=arithmetic+I::point(proof.perfect_envelope[at])*I::point(scale);
        if(!proof.noise_quantum.empty()) {
            const I a=I::point(proof.noise_quantum[j])*I::point(ordinary_final_root(scale));
            const I b=I::point(proof.noise_quantum[k])*I::point(ordinary_final_root(scale));
            arithmetic=arithmetic+a*I::point(ordinary_final_root(right))+b*I::point(ordinary_final_root(left))+a*b;
        }
        const Real allowed=(I::point(proof.limit)*I::point(ordinary_final_root(left))*I::point(ordinary_final_root(right))+arithmetic+
            I::point(OrdinarySandwichSum::gamma(2,std::numeric_limits<double>::epsilon()))*I::point(std::abs(Real(native(j,k))))).hi;
        if(!(enclosed.lo<=enclosed.hi && enclosed.absolute()<=std::numeric_limits<Real>::max() &&
             allowed>=0 && allowed<=std::numeric_limits<Real>::max())) continue;
        const Real point=native(j,k);
        if(point<I::down(enclosed.lo-allowed) || point>I::up(enclosed.hi+allowed)) return true;
    }
    return false;
}

inline bool ordinary_post_full_rss_readback(const OrdinaryFinalInferenceProof& proof,
    const OlsNumericalCertificate& numerical,double native,double scale) {
    using I=OlsProofInterval;
    if(!proof.ready || !(scale>0) || !ieee_finite(native)) return false;
    const I radius=I::point(ordinary_final_root(proof.residual_error_squared))+
        I::point(proof.residual_formation);
    const long double low=std::max(0.0L,(I::point(ordinary_final_root_lower(numerical.residual_squared.lo))-radius).lo);
    const long double high=(I::point(ordinary_final_root(numerical.residual_squared.hi))+radius).hi;
    const I interval=I{std::max(0.0L,(I::point(low)*I::point(low)).lo),
        (I::point(high)*I::point(high)).hi}*I::point(scale);
    const long double error=I::up(std::max(std::abs(static_cast<long double>(native)-interval.lo),
        std::abs(static_cast<long double>(native)-interval.hi)));
    const I allowed=I::point(1e-8L)*I::point(std::max(0.0L,interval.lo))+
        I::point(OrdinarySandwichSum::gamma(2,std::numeric_limits<double>::epsilon()))*I::point(std::abs(native));
    return error<=allowed.lo;
}

inline void run_ordinary_post_ols_audit(
    OrdinaryAuditReceipt& receipt,const Eigen::Ref<const Eigen::VectorXd>& raw_y,
    const Eigen::Ref<const Eigen::MatrixXd>& raw_design,
    const std::vector<Eigen::VectorXi>& fes,const Eigen::VectorXd* weights,
    const std::vector<Eigen::VectorXi>* clusters,const HdfeOptions& options,
    const AbsorptionResult& accepted,const Eigen::Ref<const Eigen::MatrixXd>& design,
    const std::vector<int>& kept,const OlsResult& fit,const HdfeResults& reported,
    bool recovered_intercept,bool explicit_intercept,double covariance_df,
    double robust_scale,double cluster_ratio,double reported_rss_scale,bool inference_unavailable,
    bool iv,bool slopes) {
    if(!ordinary_audit_enabled()) return;
    if(slopes) {receipt.unsupported("heterogeneous_slopes");return;}
    if(iv) {
        receipt.families[0]=run_ordinary_n2_audit(raw_y,raw_design,fes,weights,options,accepted,clusters,false);
        for(int j=1;j<4;++j) receipt.families[j]={OrdinaryAuditFinding::Unsupported,"iv_score_design"};
        return;
    }
    if(inference_unavailable) {
        receipt.families[0]=run_ordinary_n2_audit(raw_y,raw_design,fes,weights,options,accepted,clusters,false);
        for(int j=1;j<4;++j) receipt.families[j]={OrdinaryAuditFinding::Unsupported,"inference_unavailable"};
        return;
    }
    ++ordinary_audit_work_count;
    OrdinaryPostOlsIdentity identity;
    identity.pin(raw_y);identity.pin(raw_design);identity.pin(design);
    identity.pin(accepted.y_tilde);identity.pin(accepted.X_tilde);
    identity.pin(accepted.fe_levels);identity.pin(accepted.sweep_order_used);
    identity.pin(fit.coefficients);identity.pin(fit.xtx_inv);identity.pin(fit.residuals);
    identity.pin(fit.covariance);identity.pin(fit.cluster_ux);
    identity.pin(reported.coefficients);identity.pin(reported.covariance);identity.pin(reported.residuals);
    identity.pin(reported.sample_index);identity.pin(reported.omitted_reason);identity.pin(reported.fe_num_levels);
    identity.pin(kept);identity.pin(options.collinear_priority);
    for(const auto& fe:fes) identity.pin(fe);
    if(weights) identity.pin(*weights);
    if(clusters) for(const auto& cluster:*clusters) identity.pin(cluster);
    identity.pin_metadata([&]() {
        auto bits=[](double value) {return ieee_bits_detail::bits(value);};
        return std::vector<std::uint64_t>{
            bits(options.tol),bits(options.fe_tolerance),std::uint64_t(options.tolerance_mode),
            std::uint64_t(options.convergence_criterion),std::uint64_t(options.absorption_method),
            std::uint64_t(options.max_iter),std::uint64_t(options.weights_are_frequencies),
            std::uint64_t(options.fit_intercept),std::uint64_t(options.retain_fixed_effects),
            std::uint64_t(options.dof_method),std::uint64_t(options.dof_adjust_clusters),std::uint64_t(options.dof_adjust_continuous),
            std::uint64_t(options.stats_style),std::uint64_t(options.se_type),
            std::uint64_t(options.ssc_k_adj),std::uint64_t(options.ssc_k_fixef),std::uint64_t(options.ssc_k_exact),
            std::uint64_t(options.ssc_g_adj),std::uint64_t(options.ssc_g_df),bits(options.ssc_t_df),
            std::uint64_t(accepted.iterations),std::uint64_t(accepted.converged),std::uint64_t(accepted.precision_certified),
            std::uint64_t(accepted.gpu_used),std::uint64_t(accepted.gpu_status_code),
            std::uint64_t(accepted.gpu_attempted),std::uint64_t(accepted.gpu_absorption_converged),
            std::uint64_t(accepted.gpu_absorption_iterations),
            bits(accepted.abs_residual_rel),bits(accepted.abs_residual),
            bits(accepted.krylov_max_final_backward_error),bits(accepted.krylov_internal_tolerance),
            bits(fit.rss),bits(fit.df_resid),bits(fit.sigma2),bits(fit.cov_scale),bits(fit.cluster_u2),
            std::uint64_t(fit.nobs),std::uint64_t(fit.num_clusters),
            bits(reported.rss),bits(reported.sigma2),bits(reported.df_resid),bits(reported.df_resid_unadj),
            bits(reported.df_a),bits(reported.df_a_nested),bits(reported.nobs_effective),
            std::uint64_t(reported.nobs),std::uint64_t(reported.num_clusters),std::uint64_t(reported.num_iterations),
            bits(covariance_df),bits(robust_scale),bits(cluster_ratio),bits(reported_rss_scale),
            std::uint64_t(recovered_intercept),std::uint64_t(explicit_intercept),
            std::uint64_t(receipt.iterations),std::uint64_t(receipt.cache_hit),std::uint64_t(receipt.n1_refinements)};
    });
    receipt.families[0]=run_ordinary_n2_audit(raw_y,raw_design,fes,weights,options,accepted,clusters,true);
    try {
        const int p=static_cast<int>(design.cols());
        const int q=p+(recovered_intercept ? 1 : 0);
        if(p<=0 || p>64 || fit.coefficients.size()!=p ||
            p!=static_cast<int>(kept.size())+(explicit_intercept ? 1 : 0)) {
            for(int j=1;j<4;++j) receipt.families[j]={OrdinaryAuditFinding::Unsupported,"retained_design_scope"};
        } else {
            HdfeOptions proof_options=options;proof_options.parallel_observer=nullptr;
            // Frozen output contracts, never forwarded to a solve/refinement.
            proof_options.tol=1e-8;
            const long double coefficient_limit=options.tolerance_mode==ToleranceMode::XhdfeFast ?
                std::max(options.tol,64*std::numeric_limits<double>::epsilon()) : 1e-9L;
            Eigen::MatrixXd final_X=design,original_X(raw_y.size(),p);
            std::vector<int> reported_columns=kept;
            for(int j=0;j<static_cast<int>(kept.size());++j) original_X.col(j)=raw_design.col(kept[j]);
            if(explicit_intercept) {
                original_X.col(p-1)=raw_design.col(raw_design.cols()-1);
                reported_columns.push_back(static_cast<int>(reported.coefficients.size())-1);
            }
            if(recovered_intercept) reported_columns.push_back(static_cast<int>(reported.coefficients.size())-1);
            Eigen::VectorXd final_beta(q);Eigen::MatrixXd final_V(q,q);
            for(int j=0;j<q;++j) {
                final_beta[j]=reported.coefficients[reported_columns[j]];
                for(int k=0;k<q;++k) final_V(j,k)=reported.covariance(reported_columns[j],reported_columns[k]);
            }
            std::vector<int> all(p);std::iota(all.begin(),all.end(),0);
            Eigen::MatrixXd directions,basis;
            const auto rank=precise_ols_rank_mask(final_X,all,weights,false,
                64*std::numeric_limits<double>::epsilon(),{},std::max(1,options.num_threads),nullptr,&directions,&basis);
            if(std::any_of(rank.begin(),rank.end(),[](std::uint8_t value){return value!=0;}) || basis.cols()!=p)
                throw std::runtime_error("final_retained_basis_inconclusive");
            const Eigen::VectorXd final_y=accepted.y_tilde;
            const auto numerical=certify_ols_numerics(final_y,final_X,weights,fit.coefficients,basis,
                std::max(1,options.num_threads),nullptr,true);
            AbsorptionResult selected;selected.y_tilde=final_y;selected.X_tilde=final_X;selected.fe_levels=accepted.fe_levels;
            const Eigen::VectorXd original_y=raw_y;
            const auto components=ordinary_post_projection_components(original_y,original_X,fes,weights,selected,proof_options);
            const auto means=ordinary_post_means(original_y,original_X,weights);
            OrdinaryStructuredCoefficientInput coefficient;
            coefficient.ready=components.ready;coefficient.coefficients=fit.coefficients;coefficient.numerical=numerical;
            coefficient.covariance_limit=1e-8L;coefficient.limit=coefficient_limit;
            const long double coefficient_bound=ordinary_structured_coefficient_bound(coefficient,components.fe,components.affine);
            const bool coefficients_ok=ordinary_post_coefficient_readback(final_beta,fit,means,recovered_intercept,
                coefficient_bound,coefficient_limit);
            if(options.se_type==StandardErrorType::Homoskedastic) {
                coefficient.auxiliary_homoskedastic_covariance=final_V.topLeftCorner(p,p);
                coefficient.auxiliary_covariance_df=covariance_df;
                long double response=0,prediction=0;
                for(int i=0;i<final_y.size();++i) {
                    const long double root=std::sqrt(weights ? (*weights)[i] : 1.0L);
                    response=std::hypot(response,root*final_y[i]);
                    prediction=std::hypot(prediction,root*(static_cast<long double>(final_y[i])-fit.residuals[i]));
                }
                coefficient.residual_roundoff_floor=64*std::numeric_limits<double>::epsilon()*(response+prediction);
                const auto check=ordinary_structured_homoskedastic_covariance_check(coefficient,components.fe,components.affine,coefficient_bound);
                bool rss_ok=false;
                if(check.eligible && check.true_rss.lo>=0 && reported_rss_scale>0) {
                    using I=OlsProofInterval;
                    const I interval=check.true_rss*I::point(reported_rss_scale);
                    const long double error=std::max(std::abs(static_cast<long double>(reported.rss)-interval.lo),
                        std::abs(static_cast<long double>(reported.rss)-interval.hi));
                    const long double allowance=1e-8L*interval.lo+
                        (I::point(coefficient.residual_roundoff_floor)*I::point(coefficient.residual_roundoff_floor)*I::point(reported_rss_scale)).hi;
                    rss_ok=error<=allowance;
                }
                const bool passed=components.ready && coefficients_ok && check.passed && rss_ok;
                receipt.families[1]={passed ? OrdinaryAuditFinding::Passed : OrdinaryAuditFinding::BoundInconclusive,
                    passed ? "structured_homoskedastic_retained_covariance" :
                    !components.ready ? components.reason : !coefficients_ok ? "coefficient_readback_bound" :
                    !rss_ok ? "rss_readback_bound" : "structured_homoskedastic_bound"};
                receipt.families[2]={OrdinaryAuditFinding::Unsupported,"vce_homoskedastic"};
                receipt.families[3]={OrdinaryAuditFinding::Unsupported,"full_v_requires_oneway_cluster"};
            } else if(options.se_type==StandardErrorType::Cluster && (!clusters || clusters->size()!=1)) {
                receipt.families[1]={OrdinaryAuditFinding::Unsupported,"vce_not_homoskedastic"};
                receipt.families[2]=receipt.families[3]={OrdinaryAuditFinding::Unsupported,"multiway"};
            } else {
                receipt.families[1]={OrdinaryAuditFinding::Unsupported,"vce_not_homoskedastic"};
                const bool full_scope=!weights && options.se_type==StandardErrorType::Cluster && recovered_intercept && means.ready;
                OrdinarySandwichFullMoments moments;
                if(full_scope) {moments.mean_points=means.x;moments.means=means.enclosed_x;moments.response=&final_y;}
                const auto profile=ordinary_sandwich_profile(final_X,weights,clusters,proof_options,fit,numerical,
                    full_scope ? &moments : nullptr);
                const double scale=options.se_type==StandardErrorType::Cluster ? fit.cov_scale*cluster_ratio : fit.cov_scale*robust_scale;
                const bool moments_ok=ordinary_post_sandwich_readback(profile,final_V.topLeftCorner(p,p),scale);
                receipt.families[2]={moments_ok ? OrdinaryAuditFinding::Passed : OrdinaryAuditFinding::BoundInconclusive,
                    moments_ok ? "sandwich_candidate_moments" : profile.eligible ? "sandwich_native_moment_bound" : profile.reason};
                if(!full_scope) receipt.families[3]={OrdinaryAuditFinding::Unsupported,
                    weights ? "full_v_weighted" : !recovered_intercept ? "intercept_not_recovered" : "full_v_requires_oneway_cluster"};
                else if(!components.ready) receipt.families[3]={OrdinaryAuditFinding::BoundInconclusive,components.reason};
                else {
                    if(ordinary_small_identity_projection(original_y,original_X,fes,selected))
                        ordinary_small_residual_formation(final_y,final_X,fit,moments.residual_formation_bound);
                    const auto proof=ordinary_final_inference_bound(fit,numerical,profile,moments,means,
                        components.fe,components.affine,proof_options,coefficient_limit);
                    const bool rss_ok=ordinary_post_full_rss_readback(proof,numerical,reported.rss,reported_rss_scale);
                    const bool passed=coefficients_ok && rss_ok && proof.ready && ordinary_final_covariance_readback(proof,final_V,scale);
                    const bool separated=!passed && options.tolerance_mode==ToleranceMode::ReghdfeComparable &&
                        ordinary_post_full_separated(proof,final_V,scale);
                    receipt.families[3]={passed ? OrdinaryAuditFinding::Passed : separated ? OrdinaryAuditFinding::ErrorDemonstrated :
                        OrdinaryAuditFinding::BoundInconclusive,
                        passed ? "full_oneway_covariance_and_recovered_intercept" : separated ? "final_covariance_interval_separation" :
                        !coefficients_ok ? "coefficient_readback_bound" : !proof.ready ? proof.reason :
                        !rss_ok ? "rss_readback_bound" : "full_covariance_readback_bound"};
                }
            }
        }
    } catch(const std::exception& error) {
        for(int j=1;j<4;++j) if(receipt.families[j].reason=="not_requested")
            receipt.families[j]={OrdinaryAuditFinding::BoundInconclusive,error.what()};
    }
    receipt.identity_match=identity.matches();
    if(!receipt.identity_match)
        for(auto& family:receipt.families) family={OrdinaryAuditFinding::BoundInconclusive,"fit_identity_changed"};
}

} }
