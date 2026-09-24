// Private eligibility tests; no estimator fits or runtime-performance claims.
#include "additive_cell_projection.hpp"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#if defined(__i386__) || defined(__x86_64__)
#include <fpu_control.h>
#endif

namespace acp=hdfe::detail::additive_cell_projection;
using hdfe::HdfeOptions;
using hdfe::detail::ParallelWorkObserver;

struct Fixture {
    Eigen::VectorXd y,w,expected_y;
    Eigen::MatrixXd X,expected_X;
    std::vector<int> left,right;
    int storage_left=0,storage_right=0,levels_left=0,levels_right=0;
    acp::FeView first() const {return {left.data(),storage_left,levels_left};}
    acp::FeView second() const {return {right.data(),storage_right,levels_right};}
};

Fixture make_fixture(int L=3,int R=5,int per_cell=64,double beta=.75,int exponent=20,
                     bool permute=false,int recode=0,bool omit_last=false) {
    const int n=(L*R-(omit_last ? 1 : 0))*per_cell;
    Fixture f;f.y.resize(n);f.w.resize(n);f.X.resize(n,1);
    f.expected_y.resize(n);f.expected_X.resize(n,1);f.left.resize(n);f.right.resize(n);
    f.levels_left=L;f.levels_right=R;
    f.storage_left=recode==2 ? 8 : L;f.storage_right=recode==2 ? 8 : R;
    auto label=[&](int value,int levels) {
        return recode==2 ? ((value+1)*8/levels-1) : (recode==1 ? levels-1-value : value);
    };
    int row=0;
    for(int a=0;a<L;++a) for(int b=0;b<R;++b) {
        if(omit_last && a==L-1 && b==R-1) continue;
        for(int k=0;k<per_cell;++k,++row) {
            const double A=2*(k%2)-1,B=2*((k/2)%2)-1;
            f.left[row]=label(a,L);f.right[row]=label(b,R);
            f.X(row,0)=A+a/4.0-b/2.0;
            f.y[row]=beta*f.X(row,0)+8*a-16*b+std::ldexp(B,-exponent);
            f.w[row]=std::ldexp(1.0,(a+2*b)%5-2);
            f.expected_X(row,0)=A;
            f.expected_y[row]=beta*A+std::ldexp(B,-exponent);
        }
    }
    if(permute) {
        Fixture previous=f;std::vector<int> order(n);std::iota(order.begin(),order.end(),0);
        std::mt19937 generator(20260912);std::shuffle(order.begin(),order.end(),generator);
        for(int i=0;i<n;++i) {
            const int j=order[i];f.y[i]=previous.y[j];f.w[i]=previous.w[j];
            f.X.row(i)=previous.X.row(j);f.expected_y[i]=previous.expected_y[j];
            f.expected_X.row(i)=previous.expected_X.row(j);
            f.left[i]=previous.left[j];f.right[i]=previous.right[j];
        }
    }
    return f;
}

template<class A,class B> bool same_eigen(const A& a,const B& b) {
    return a.rows()==b.rows() && a.cols()==b.cols() &&
        std::memcmp(a.data(),b.data(),static_cast<std::size_t>(a.size())*sizeof(typename A::Scalar))==0;
}

bool same_inputs(const Fixture& a,const Fixture& b) {
    return same_eigen(a.y,b.y) && same_eigen(a.X,b.X) && same_eigen(a.w,b.w) &&
        a.left==b.left && a.right==b.right;
}

bool same_plan(const acp::Plan& a,const acp::Plan& b) {
    return a.n==b.n && a.cols==b.cols && a.threads==b.threads && a.cells==b.cells &&
        a.storage==b.storage && a.levels==b.levels && a.cell==b.cell &&
        a.means==b.means && a.alphas==b.alphas && a.cell_mass==b.cell_mass;
}

struct PlanGuard {
    acp::Plan& plan;
    acp::Plan before;
    const void* cell;
    const void* means;
    const void* alphas;
    std::array<std::size_t,3> capacity;
    explicit PlanGuard(acp::Plan& value):plan(value),before(value),cell(value.cell.data()),
        means(value.means.data()),alphas(value.alphas.data()),
        capacity{{value.cell.capacity(),value.means.capacity(),value.alphas.capacity()}} {}
    bool unchanged() const {
        return same_plan(plan,before) && cell==plan.cell.data() && means==plan.means.data() &&
            alphas==plan.alphas.data() && capacity==std::array<std::size_t,3>{{
                plan.cell.capacity(),plan.means.capacity(),plan.alphas.capacity()}};
    }
};

acp::Plan sentinel_plan() {
    acp::Plan p;p.n=-77;p.cols=19;p.threads=23;p.cells=11;
    p.storage={{7,5}};p.levels={{2,3}};p.cell={1,7,19};
    p.means={-1.25,6.5};p.alphas={9.75,-8};p.cell_mass[3]=16.5;return p;
}

HdfeOptions options(int threads=1) {
    HdfeOptions o;o.tolerance_mode=hdfe::ToleranceMode::ReghdfeComparable;
    o.absorption_method=hdfe::AbsorptionMethod::Auto;o.tol=1e-8;o.max_iter=1000;
    o.num_threads=threads;o.num_threads_explicit=true;return o;
}

struct Record {std::string name;bool passed;int requested=0,team=0,workers=0;};
std::vector<Record> records;
void check(std::string name,bool passed,int requested=0,int team=0,int workers=0) {
    if(!passed) std::cerr<<"FAIL "<<name<<'\n';
    records.push_back({std::move(name),passed,requested,team,workers});
}

void positive(std::string name,const Fixture& f,HdfeOptions o) {
    const Fixture before=f;ParallelWorkObserver observer;o.parallel_observer=&observer;
    acp::Plan plan;bool prepared=false,applied=false,threw=false;
    try {prepared=acp::prepare(f.y,f.X,f.first(),f.second(),&f.w,o,plan);}
    catch(...) {threw=true;}
    check(name+"_prepare",prepared && !threw && plan.n==f.y.size() && plan.cols==f.X.cols());
    check(name+"_sum_workers",prepared && observer.max_team_size()==o.num_threads &&
        observer.active_workers()==o.num_threads,o.num_threads,observer.max_team_size(),observer.active_workers());
    observer.reset();hdfe::detail::AbsorptionResult result;
    try {applied=acp::try_cpu(f.y,f.X,f.first(),f.second(),&f.w,o,result);}
    catch(...) {threw=true;}
    check(name+"_projection",applied && !threw && same_eigen(result.y_tilde,f.expected_y) &&
        same_eigen(result.X_tilde,f.expected_X));
    check(name+"_application_workers",applied && observer.max_team_size()==o.num_threads &&
        observer.active_workers()==o.num_threads,o.num_threads,observer.max_team_size(),observer.active_workers());
    bool metadata=applied && result.fe_levels==std::vector<int>{f.levels_left,f.levels_right};
    if(o.retain_fixed_effects) {
        metadata=metadata && result.fe_group_ids.size()==2 && result.fe_alpha_y.size()==2;
        if(metadata) {
            metadata=result.fe_group_ids[0]==f.left && result.fe_group_ids[1]==f.right;
            for(int i=0;i<f.y.size() && metadata;++i) {
                const double a=result.fe_alpha_y[0][f.left[i]],b=result.fe_alpha_y[1][f.right[i]];
                metadata=hdfe::detail::weighted_two_by_two::add(a,b)==plan.means[plan.cell[i]];
            }
        }
    }
    check(name+"_metadata",metadata);
    check(name+"_inputs_unchanged",same_inputs(f,before));
}

void negative(std::string name,const Fixture& f,HdfeOptions o) {
    const Fixture before=f;acp::Plan output=sentinel_plan();PlanGuard guard(output);
    bool accepted=false,threw=false;
    try {accepted=acp::prepare(f.y,f.X,f.first(),f.second(),&f.w,o,output);}
    catch(...) {threw=true;}
    check(name+"_ineligible",!accepted && !threw);
    check(name+"_plan_unchanged",guard.unchanged());
    check(name+"_inputs_unchanged",same_inputs(f,before));
}

int main() {
    const int capacity=hdfe::detail::runtime_thread_capacity();
    check("native_significand_available",acp::native_significand_available());
    const std::array<std::array<int,2>,6> grids{{{{3,5}},{{5,3}},{{4,4}},{{2,7}},{{7,2}},{{8,8}}}};
    int case_number=0;
    for(const auto& grid:grids) for(double beta:{.75,-1.25,0.0}) for(int exponent:{12,20,30}) {
        const int id=case_number++;
        Fixture f=make_fixture(grid[0],grid[1],64,beta,exponent,id%2,id%3);
        HdfeOptions o=options(std::min(2,capacity));
        o.fit_intercept=(id%2)==0;o.retain_fixed_effects=(id%4)>=2;
        positive("analytic_"+std::to_string(id),f,o);
    }
    std::vector<int> skipped_threads;
    for(int threads:{1,2,8,16,24,48}) {
        if(threads>capacity) {skipped_threads.push_back(threads);continue;}
        auto o=options(threads);o.retain_fixed_effects=true;
        positive("thread_"+std::to_string(threads),make_fixture(3,5,64,-1.25,30,true,2),o);
    }
    const Fixture base=make_fixture();
    auto o=options(std::min(2,capacity));
    Fixture f=base;
    for(int i=0;i<f.y.size();++i) f.y[i]+=(f.left[i]*f.right[i])/8.0;
    negative("y_interaction",f,o);
    f=base;for(int i=0;i<f.y.size();++i) f.X(i,0)+=(f.left[i]*f.right[i])/4.0;
    negative("x_interaction",f,o);
    f=base;f.w[0]=std::nextafter(f.w[0],std::numeric_limits<double>::infinity());
    negative("within_cell_weight_nextafter",f,o);
    f=base;f.w[0]=0;negative("zero_weight",f,o);
    negative("missing_cartesian_cell",make_fixture(3,5,64,.75,20,false,0,true),o);
    f=make_fixture(3,5,3);f.X.setZero();
    for(int i=0;i<f.y.size();++i) f.y[i]=i%3==2 ? 1.0 : 0.0;
    negative("nonrepresentable_third_mean",f,o);
    f=base;f.y[0]=std::ldexp(1.0,-100);negative("sum_bit_budget",f,o);
    negative("nine_left_levels",make_fixture(9,3),o);
    negative("nine_right_levels",make_fixture(3,9),o);
    f=base;f.X=Eigen::MatrixXd::Ones(f.y.size(),33);negative("thirty_three_columns",f,o);
    negative("existing_two_by_two",make_fixture(2,2),o);
    auto changed=o;changed.tolerance_mode=hdfe::ToleranceMode::XhdfeFast;
    negative("fast_mode",base,changed);
    changed=o;changed.tolerance_mode=hdfe::ToleranceMode::StrictResidual;
    negative("strict_mode",base,changed);
    int method=0;
    for(auto value:{hdfe::AbsorptionMethod::GaussSeidel,hdfe::AbsorptionMethod::SymmetricGaussSeidel,
        hdfe::AbsorptionMethod::Jacobi,hdfe::AbsorptionMethod::Schwarz,
        hdfe::AbsorptionMethod::Lsmr,hdfe::AbsorptionMethod::Mlsmr}) {
        changed=o;changed.absorption_method=value;changed.from_auto=false;
        negative("explicit_method_"+std::to_string(method++),base,changed);
    }
    changed=o;changed.use_krylov=true;negative("explicit_krylov",base,changed);
    changed=o;changed.use_sparse_solver=true;negative("explicit_sparse",base,changed);

    changed=o;changed.absorption_method=hdfe::AbsorptionMethod::GaussSeidel;changed.from_auto=false;
    {
        acp::ScopedPublicAutoRequest automatic(true);
        positive("resolved_public_auto",base,changed);
        {
            acp::ScopedPublicAutoRequest explicit_request(false);
            negative("nested_explicit_request",base,changed);
        }
        check("public_auto_context_restored",acp::scope(changed,base.first(),base.second(),2,1,&base.w));
        auto fast=changed;fast.tolerance_mode=hdfe::ToleranceMode::XhdfeFast;
        negative("public_auto_does_not_enable_fast",base,fast);
    }
    check("public_auto_context_cleared",!acp::scope(changed,base.first(),base.second(),2,1,&base.w));

#if defined(__i386__) || defined(__x86_64__)
    {
        const Fixture before=base;acp::Plan output=sentinel_plan();PlanGuard guard(output);
        fpu_control_t saved,reduced,after;_FPU_GETCW(saved);
        reduced=static_cast<fpu_control_t>((saved&~_FPU_EXTENDED)|_FPU_DOUBLE);
        bool accepted=false,threw=false,probe=false;
        _FPU_SETCW(reduced);
        probe=acp::native_significand_available();
        try {accepted=acp::prepare(base.y,base.X,base.first(),base.second(),&base.w,o,output);}
        catch(...) {threw=true;}
        _FPU_GETCW(after);_FPU_SETCW(saved);
        fpu_control_t restored;_FPU_GETCW(restored);
        check("caller_reduced_precision_ineligible",!probe && !accepted && !threw);
        check("caller_controlword_untouched_and_restored",after==reduced && restored==saved);
        check("caller_reduced_plan_unchanged",guard.unchanged());
        check("caller_reduced_inputs_unchanged",same_inputs(base,before));
    }
#ifdef HDFE_USE_OPENMP
    if(capacity>=2) {
        const Fixture before=base;acp::Plan output=sentinel_plan();PlanGuard guard(output);
        std::thread::id target;fpu_control_t saved=0,reduced=0;
        bool worker_probe=true;const int previous_dynamic=omp_get_dynamic();omp_set_dynamic(0);
#pragma omp parallel num_threads(2)
        {
            if(omp_get_thread_num()==1) {
                target=std::this_thread::get_id();_FPU_GETCW(saved);
                reduced=static_cast<fpu_control_t>((saved&~_FPU_EXTENDED)|_FPU_DOUBLE);
                _FPU_SETCW(reduced);worker_probe=acp::native_significand_available();
            }
        }
        bool accepted=false,threw=false;
        try {accepted=acp::prepare(base.y,base.X,base.first(),base.second(),&base.w,options(2),output);}
        catch(...) {threw=true;}
        bool found=false,untouched=false,restored=false;
#pragma omp parallel num_threads(2)
        {
            if(std::this_thread::get_id()==target) {
                fpu_control_t current;_FPU_GETCW(current);untouched=current==reduced;
                _FPU_SETCW(saved);_FPU_GETCW(current);restored=current==saved;found=true;
            }
        }
        omp_set_dynamic(previous_dynamic);
        check("reused_worker_reduced_precision_ineligible",!worker_probe && !accepted && !threw);
        check("reused_worker_controlword_untouched_and_restored",found && untouched && restored);
        check("reused_worker_plan_unchanged",guard.unchanged());
        check("reused_worker_inputs_unchanged",same_inputs(base,before));
        check("caller_precision_after_worker",acp::native_significand_available());
    }
#endif
#endif
    const int failures=static_cast<int>(std::count_if(records.begin(),records.end(),[](const auto& r){return !r.passed;}));
    std::cout<<"{\"checks\":"<<records.size()<<",\"failures\":"<<failures<<",\"runtime_capacity\":"<<capacity;
    std::cout<<",\"skipped_thread_requests\":[";
    for(std::size_t i=0;i<skipped_threads.size();++i) {if(i) std::cout<<',';std::cout<<skipped_threads[i];}
    std::cout<<"],\"rows\":[";
    for(std::size_t i=0;i<records.size();++i) {
        if(i) std::cout<<',';const auto& row=records[i];
        std::cout<<"{\"name\":"<<std::quoted(row.name)<<",\"passed\":"<<(row.passed ? "true" : "false")
            <<",\"requested\":"<<row.requested<<",\"team\":"<<row.team<<",\"workers\":"<<row.workers<<'}';
    }
    std::cout<<"]}\n";return failures ? 1 : 0;
}
