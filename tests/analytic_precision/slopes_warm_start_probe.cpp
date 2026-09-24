#ifndef XHDFE_SLOPE_PROBE_SOURCE
#define XHDFE_SLOPE_PROBE_SOURCE "fe_absorption.cpp"
#endif
#include XHDFE_SLOPE_PROBE_SOURCE

#include <cstring>
#include <iostream>
#include <string>

// Set to zero when compiling this probe against the unchanged C0 source.
#ifndef XHDFE_SLOPE_PROBE_HAS_SEED
#define XHDFE_SLOPE_PROBE_HAS_SEED 1
#endif

namespace {
using namespace hdfe;
using namespace hdfe::detail;

void require(bool value,const std::string& label) {
    if(!value) throw std::runtime_error(label);
}

template<class A,class B> bool same_values(const A& a,const B& b) {
    return a.rows()==b.rows() && a.cols()==b.cols() &&
        (!a.size() || !std::memcmp(a.data(),b.data(),static_cast<std::size_t>(a.size())*sizeof(double)));
}

void same_result(const AbsorptionResult& a,const AbsorptionResult& b,const std::string& label) {
    require(same_values(a.y_tilde,b.y_tilde) && same_values(a.X_tilde,b.X_tilde),label+" point");
    require(a.converged==b.converged && a.iterations==b.iterations &&
            a.precision_certified==b.precision_certified && a.fe_levels==b.fe_levels &&
            a.sweep_order_used==b.sweep_order_used && a.gpu_used==b.gpu_used &&
            a.gpu_attempted==b.gpu_attempted && a.gpu_status_code==b.gpu_status_code,
            label+" state");
    require(a.fe_group_ids==b.fe_group_ids && a.fe_means.size()==b.fe_means.size() &&
            a.fe_weight_sums.size()==b.fe_weight_sums.size(),label+" retained dimensions");
    for(std::size_t d=0;d<a.fe_means.size();++d)
        require(same_values(a.fe_means[d],b.fe_means[d]),label+" retained means");
    for(std::size_t d=0;d<a.fe_weight_sums.size();++d)
        require(same_values(a.fe_weight_sums[d],b.fe_weight_sums[d]),label+" retained weights");
}

HdfeOptions stage_options(ToleranceMode mode) {
    HdfeOptions options;
    options.tol=1e-8;options.max_iter=1;options.num_threads=2;options.num_threads_explicit=true;
    options.convergence_check_interval=1;
    options.tolerance_mode=mode;
    options.convergence_criterion=ConvergenceCriterion::Auto;
    options.from_auto=false;options.retain_fixed_effects=false;
    options.absorption_method=AbsorptionMethod::GaussSeidel;
    options.symmetric_sweep=false;
    return options;
}

struct Fixture {
    Eigen::VectorXd y;
    Eigen::MatrixXd X;
    std::vector<Eigen::VectorXi> fes;
    std::vector<HeterogeneousSlopeTerm> slopes;
    AbsorptionResult seed;
    explicit Fixture(int n):y(n),X(n,1),fes(2,Eigen::VectorXi(n)),slopes(1) {
        slopes[0].fe_index=1;slopes[0].include_intercept=true;slopes[0].values.resize(n);
        seed.y_tilde.resize(n);seed.X_tilde.resize(n,1);seed.converged=true;
        for(int i=0;i<n;++i) {
            fes[0][i]=(i%16)/4;fes[1][i]=i%3;
            slopes[0].values[i]=(i%5)-2.0;
            y[i]=((i%17)*(i%17)%23)/8.0+2.0*fes[1][i];
            X(i,0)=(i%7)/4.0+fes[1][i]/8.0;
            // Subtract only an intercept of the second absorbed term.
            const double component=fes[1][i]+1.0;
            seed.y_tilde[i]=y[i]-component;
            seed.X_tilde(i,0)=X(i,0)-0.5*component;
        }
    }
};

AbsorptionResult cold(const Fixture& f,const Eigen::VectorXd* weights,const HdfeOptions& options) {
    return absorb_fixed_effects_v6_mixed(f.y,f.X,f.fes,weights,options,
                                        AbsorptionMethod::GaussSeidel,f.slopes);
}

#if XHDFE_SLOPE_PROBE_HAS_SEED
AbsorptionResult seeded(const Fixture& f,const Eigen::VectorXd* weights,
                        const HdfeOptions& options,const AbsorptionResult& seed) {
    return absorb_fixed_effects_v6_mixed_impl(f.y,f.X,f.fes,weights,options,
                                             AbsorptionMethod::GaussSeidel,f.slopes,&seed);
}

void check_original_references() {
    Fixture f(16);
    f.fes.resize(1);f.fes[0].setZero();f.slopes[0].fe_index=0;
    f.y.setConstant(2.0);f.X.setConstant(4.0);
    f.seed.y_tilde.setZero();f.seed.X_tilde.setZero();
    for(int i=0;i<f.y.size();++i) f.slopes[0].values[i]=(i&1) ? 1.0 : -1.0;
    for(auto mode:{ToleranceMode::XhdfeFast,ToleranceMode::ReghdfeComparable}) {
        const auto options=stage_options(mode);
        const auto correct=seeded(f,nullptr,options,f.seed);
        const auto baseline=cold(f,nullptr,options);
        // The actual C0 kernel on the seed as input is the negative control:
        // replacing the original stopping references admits its first check.
        const auto changed_reference=absorb_fixed_effects_v6_mixed(
            f.seed.y_tilde,f.seed.X_tilde,f.fes,nullptr,options,
            AbsorptionMethod::GaussSeidel,f.slopes);
        require(!correct.converged && correct.iterations==1,"original first-check references");
        same_result(correct,baseline,"original reference control");
        require(changed_reference.converged,"reference control must distinguish the wrong initialization");
    }
}

void check_seed_point() {
    for(int n:{16,200000}) {
        const Fixture f(n);
        const auto options=stage_options(ToleranceMode::XhdfeFast);
        const auto warm=seeded(f,nullptr,options,f.seed);
        const auto baseline=cold(f,nullptr,options);
        const auto point_reference=absorb_fixed_effects_v6_mixed(
            f.seed.y_tilde,f.seed.X_tilde,f.fes,nullptr,options,
            AbsorptionMethod::GaussSeidel,f.slopes);
        require(!warm.converged && !point_reference.converged &&
                warm.iterations==1 && point_reference.iterations==1,
                "one-iteration point fixture must not include terminal polishing");
        require(same_values(warm.y_tilde,point_reference.y_tilde) &&
                same_values(warm.X_tilde,point_reference.X_tilde),"seeded CPU point");
        require(!same_values(warm.y_tilde,baseline.y_tilde) ||
                !same_values(warm.X_tilde,baseline.X_tilde),"point fixture must exercise the seed");
    }
}

void check_excluded_domains() {
    const Fixture f(16);
    for(auto mode:{ToleranceMode::XhdfeFast,ToleranceMode::ReghdfeComparable}) {
        for(int control=0;control<8;++control) {
            auto options=stage_options(mode);
            auto seed=f.seed;
            Eigen::VectorXd weights=Eigen::VectorXd::Ones(f.y.size());
            const Eigen::VectorXd* w=nullptr;
            if(control==0) w=&weights;
            if(control==1) {
                for(int i=0;i<weights.size();++i) weights[i]=(i&1) ? 1e-12 : 1.0;
                w=&weights;
            }
            if(control==2) options.retain_fixed_effects=true;
            if(control==3) seed.gpu_used=true;
            if(control==4) seed.gpu_attempted=true;
            if(control==5) seed.converged=false;
            if(control==6) seed.X_tilde.resize(f.y.size(),0);
            if(control==7) seed.y_tilde[0]=std::numeric_limits<double>::quiet_NaN();
            same_result(seeded(f,w,options,seed),cold(f,w,options),
                        "excluded domain "+std::to_string(control));
        }
    }
}
#endif

void print_cold_controls() {
    const Fixture f(16);
    for(auto mode:{ToleranceMode::XhdfeFast,ToleranceMode::ReghdfeComparable})
    for(bool retain:{false,true}) for(bool weighted:{false,true}) {
        auto options=stage_options(mode);options.retain_fixed_effects=retain;
        Eigen::VectorXd weights=Eigen::VectorXd::Ones(f.y.size());
        const auto value=cold(f,weighted ? &weights : nullptr,options);
        std::cout<<int(mode)<<' '<<retain<<' '<<weighted<<' '<<value.converged<<' '
                 <<value.iterations<<' '<<value.gpu_attempted;
        for(Eigen::Index i=0;i<value.y_tilde.size();++i)
            std::cout<<' '<<std::hex<<ieee_bits_detail::bits(value.y_tilde[i])<<std::dec;
        for(Eigen::Index j=0;j<value.X_tilde.cols();++j)
            for(Eigen::Index i=0;i<value.X_tilde.rows();++i)
                std::cout<<' '<<std::hex<<ieee_bits_detail::bits(value.X_tilde(i,j))<<std::dec;
        std::cout<<'\n';
    }
}
}

int main(int argc,char** argv) {
    try {
        require(hdfe::detail::resolve_gpu_backend()==hdfe::detail::GpuBackend::Cpu,
                "run this CPU probe with XHDFE_GPU_BACKEND=cpu");
        const std::string mode=argc>1 ? argv[1] : "--seed";
        if(mode=="--cold") {print_cold_controls();return 0;}
        require(mode=="--seed","expected --seed or --cold");
#if XHDFE_SLOPE_PROBE_HAS_SEED
        check_original_references();check_seed_point();check_excluded_domains();
        std::cout<<"CPU_SEED_REFERENCE_POINT_EXCLUSIONS_PASS\n";
        return 0;
#else
        throw std::runtime_error("C0 does not contain the private seed path");
#endif
    } catch(const std::exception& error) {
        std::cerr<<"PROBE_FAIL "<<error.what()<<'\n';return 1;
    }
}
