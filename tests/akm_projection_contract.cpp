// Closed-form two-firm oracle. No dense inverse, fitted covariance, or
// production projection operator is used to obtain the expected interval.
#include "hdfe/akm_kss.hpp"
#include "hdfe/ieee_bits.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void close(double actual,double expected,const char* label) {
    if(!hdfe::detail::ieee_finite(actual) ||
       std::abs(actual-expected)>1e-9+1e-8*std::abs(expected))
        throw std::runtime_error(std::string("projection oracle mismatch: ")+label);
}
double geometric(double value) {
    double total=1.,power=1.;
    for(int k=1;k<5;++k){power*=value;total+=power;}
    return total;
}

void run(bool controls,double loading,bool jla,double scale,double shift) {
    constexpr int workers=64,n=4*workers;
    Eigen::VectorXi w(n),f(n);
    Eigen::VectorXd y(n),v(n),firm_pattern(n);
    Eigen::MatrixXd x(n,1);
    double vv=0.,fy=0.,vy=0.;
    for(int worker=0;worker<workers;++worker) {
        const double s=1.+worker%3;
        const double alpha=.3*std::sin(.37*(worker+1));
        for(int j=0;j<4;++j) {
            const int i=4*worker+j;
            w[i]=worker;f[i]=j/2;
            firm_pattern[i]=j<2 ? -.5:.5;
            v[i]=(j%2 ? -s:s);
            x(i,0)=loading*firm_pattern[i]+v[i];
            y[i]=alpha+.7*firm_pattern[i]+.35*x(i,0)+.1*std::sin(.73*(i+1));
            vv+=v[i]*v[i];fy+=firm_pattern[i]*y[i];vy+=v[i]*y[i];
        }
    }
    const double raw_difference=fy/workers;
    const double gamma=controls ? vy/vv:0.;
    const double difference=raw_difference-loading*gamma;
    const double lambda_f=1./workers;
    double variance=0.;
    for(int worker=0;worker<workers;++worker) {
        const double s=1.+worker%3,local_vv=4.*s*s,lambda_v=local_vv/vv;
        double local_fy=0.,local_vy=0.;
        for(int j=0;j<4;++j) {
            const int i=4*worker+j;
            local_fy+=firm_pattern[i]*y[i];local_vy+=v[i]*y[i];
        }
        const double f_score=geometric(lambda_f)*(local_fy-raw_difference)
            +std::pow(lambda_f,5)*local_fy;
        const double v_score=geometric(lambda_v)*(local_vy/local_vv-gamma)
            +std::pow(lambda_v,5)*local_vy/local_vv;
        const double score=-.5*f_score/workers+.5*loading*local_vv*v_score/vv;
        variance+=score*score;
    }
    const double multiplier=static_cast<double>(n)/(n-1.);
    const double centre=std::abs(difference)/2.;
    // Displayed dimension is two although the balanced design has rank one.
    // P(chi-square(2)<=q)=1-exp(-q/2), so the critical value is analytic.
    const double critical=-2.*std::log(.05),radius=std::sqrt(critical*variance);
    const double lower=multiplier*std::pow(std::max(0.,centre-radius),2);
    const double upper=multiplier*std::pow(centre+radius,2);
    const double se=2.*multiplier*centre*std::sqrt(variance);
    y=y.array()*scale+shift;
    hdfe::akm::AkmOptions options;
    options.num_threads=2;
    options.sigma_centering=hdfe::akm::SigmaCentering::PersonYear;
    options.leverage_method=jla ? hdfe::akm::LeverageMethod::Jla : hdfe::akm::LeverageMethod::Exact;
    options.jla_draws=200;
    const auto point=hdfe::akm::akm_kss_decompose(y,w,f,controls ? &x:nullptr,options);
    options.compute_se=true;
    const auto result=hdfe::akm::akm_kss_decompose(y,w,f,controls ? &x:nullptr,options);
    if(!point.converged || !result.converged || !result.projection_inference)
        throw std::runtime_error("projection fit failed");
    const auto& inference=*result.projection_inference;
    if(inference.threads_used!=2 || inference.parallel_workers_active!=2 ||
       inference.dimension!=2 || inference.rank_diagnostic!=1 ||
       inference.clusters!=workers || result.sample.n_obs!=n)
        throw std::runtime_error("projection metadata/real worker contract failed");
    close(result.kss.var_psi/(scale*scale),point.kss.var_psi/(scale*scale),"unchanged point");
    close(result.plugin.var_psi/(scale*scale),multiplier*centre*centre,"plug-in target");
    close(inference.se[0]/(scale*scale),se,"regular delta SE");
    close(inference.ci_lb[0]/(scale*scale),lower,"lower projection bound");
    close(inference.ci_ub[0]/(scale*scale),upper,"upper projection bound");
    close(inference.se[1]/(scale*scale),0.,"structural covariance SE");
    close(inference.ci_lb[1]/(scale*scale),0.,"structural covariance lower bound");
    close(inference.ci_ub[1]/(scale*scale),0.,"structural covariance upper bound");
    close(inference.critical_value,critical,"chi-square critical value");
}
}

int main() {
    try {
        int cases=0;
        for(int model=0;model<4;++model) for(bool jla:{false,true}) {
            const bool controls=model!=0;
            const double loading=model<2 ? 0. : (model==2 ? 5.:10.);
            for(int exponent:{-500,-200,0,200,500}) {
                run(controls,loading,jla,std::ldexp(1.,exponent),0.);++cases;
            }
            run(controls,loading,jla,1.,1e6);++cases;
        }
        Eigen::VectorXd y(4),fw=Eigen::VectorXd::Constant(4,2.);
        Eigen::VectorXi worker(4),firm(4);
        y<<1.,2.,3.,5.;worker<<0,0,1,1;firm<<0,1,0,1;
        hdfe::akm::AkmOptions unsupported;
        unsupported.sigma_centering=hdfe::akm::SigmaCentering::PersonYear;
        unsupported.eigen_diagnostics=true;
        bool rejected=false;
        try { hdfe::akm::akm_kss_decompose(y,worker,firm,nullptr,unsupported,nullptr,&fw); }
        catch(const std::runtime_error& error) { rejected=std::string(error.what()).find("fweights")!=std::string::npos; }
        if(!rejected) throw std::runtime_error("eigen-only fweight guard failed");
        std::cout << "AKM_PROJECTION_CLOSED_FORM: PASS " << cases << " cases, 1 expected refusal\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';return 1;
    }
}
