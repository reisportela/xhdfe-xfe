#include "reduced_schur_pcg_cuda_controller.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace rp=hdfe::reduced_pcg;
struct CpuOperator {
    int calls=0,replacements=0;
    void prepare(const rp::internal::Geometry&,const rp::Config&,int,std::size_t){}
    bool replacement_on_device()const noexcept{return false;}
    void begin_group(int){}
    void pin_host(double*,std::size_t){}
    void unpin_host() noexcept{}
    template<int W> void apply(const rp::internal::Geometry& g,const rp::internal::Team& team,
        const double* in,double* out,double* T,rp::internal::Work& work){++calls;rp::internal::apply_operator<W>(g,team,in,out,T,work);}
    template<int W> void apply_replacement(const rp::internal::Geometry& g,const rp::internal::Team& team,
        const double* x,double* q,double* T,rp::internal::Work& work){++replacements;rp::internal::apply_operator<W>(g,team,x,q,T,work);}
};
// Package 2 (C1) and package 3-lite: what solve() asks of the operator. The real width
// before each group; ONE registration before the first product, released once after
// the last, on every exit (normal, decline, exception): of p and q (adjacent slices of
// the arena) when the replacement stays on the CPU, of x, r, p and q when the operator
// says it computes the replacement too (device_replacement); every device product
// inside it, the iterative one on p into q, the replacement on x into q, each
// replacement after an iterative product of the same iteration (p -> x -> p). The CPU
// operator computes.
struct RecordingOperator {
    std::vector<int> widths;
    std::vector<char> kinds;  // 'p' iterative, 'x' replacement
    int pins=0,unpins=0,applies=0,replacements=0,throw_at=-1,throw_at_replacement=-1;
    std::uintptr_t begin=0,end=0;
    std::size_t pinned_count=0;
    bool inside=true,pinned=false,order=true,operands=true,device_replacement=false;
    void prepare(const rp::internal::Geometry&,const rp::Config&,int,std::size_t){}
    bool replacement_on_device()const noexcept{return device_replacement;}
    void begin_group(int width){widths.push_back(width);}
    void pin_host(double* b,std::size_t count){
        ++pins;pinned=true;begin=reinterpret_cast<std::uintptr_t>(b);end=begin+count*sizeof(double);pinned_count=count;
    }
    void unpin_host() noexcept{++unpins;pinned=false;}
    bool in_range(const double* p,std::size_t bytes)const{
        const auto a=reinterpret_cast<std::uintptr_t>(p);return a>=begin&&a+bytes<=end;
    }
    template<int W> void apply(const rp::internal::Geometry& g,const rp::internal::Team& team,
        const double* in,double* out,double* T,rp::internal::Work& work){
        const std::size_t mw=static_cast<std::size_t>(g.m)*W,bytes=mw*sizeof(double);
        inside=inside&&in_range(in,bytes)&&in_range(out,bytes);
        operands=operands&&in==work.arena.data()+2U*mw&&out==work.arena.data()+3U*mw;
        order=order&&pinned&&!widths.empty();
        kinds.push_back('p');
        if(applies++==throw_at)throw std::runtime_error("injected operator failure (test)");
        rp::internal::apply_operator<W>(g,team,in,out,T,work);
    }
    template<int W> void apply_replacement(const rp::internal::Geometry& g,const rp::internal::Team& team,
        const double* x,double* q,double* T,rp::internal::Work& work){
        const std::size_t mw=static_cast<std::size_t>(g.m)*W,bytes=mw*sizeof(double);
        if(device_replacement)inside=inside&&in_range(x,bytes)&&in_range(q,bytes);
        operands=operands&&x==work.arena.data()&&q==work.arena.data()+3U*mw;
        order=order&&pinned&&!kinds.empty()&&kinds.back()=='p';
        kinds.push_back('x');
        if(replacements++==throw_at_replacement)throw std::runtime_error("injected replacement failure (test)");
        rp::internal::apply_operator<W>(g,team,x,q,T,work);
    }
};
int main(){
    int cases=0;
    for(int cols:{1,2,4,8,11})for(int variation=0;variation<6;++variation){
        const int n=1024,k=3;std::vector<std::vector<int>> codes(k,std::vector<int>(n));int levels[]={64,19,5};
        std::vector<std::vector<double>> y(cols,std::vector<double>(n)),a(cols,std::vector<double>(n)),b=a;
        for(int i=0;i<n;++i){codes[0][i]=i/16;codes[1][i]=(i*7+i/13)%19;codes[2][i]=(i+i/11)%5;
            for(int c=0;c<cols;++c)y[c][i]=cols>=4&&c==0?0.:cols>=4&&c==1?codes[0][i]:std::sin(.11*i+.3*c)+.5*codes[0][i]-.1*codes[1][i];}
        std::vector<const int*> ids(k);for(int j=0;j<k;++j)ids[j]=codes[j].data();
        std::vector<const double*> in(cols);std::vector<double*> oa(cols),ob(cols);for(int c=0;c<cols;++c){in[c]=y[c].data();oa[c]=a[c].data();ob[c]=b[c].data();}
        rp::Problem p;p.n=n;p.k=k;p.codes=ids.data();p.levels=levels;p.eliminated=0;p.cols=cols;p.columns=in.data();p.residuals=oa.data();
        rp::Config cfg;cfg.threads=4;cfg.blocks=16;cfg.chat_stop=1e-12;cfg.max_iter=400;cfg.force_merge=variation%2;
        if(variation==1)cfg.test_zero_curvature_at=1;
        if(variation==2){cfg.test_recurrence_chat_at=1;cfg.test_recurrence_chat_value=1e-20;}
        if(variation==3)cfg.max_iter=1;
        if(variation==4)cfg.force_decline_iterate=true;
        if(variation==5)cfg.chat_stop=0;
        auto old=rp::solve(p,cfg);p.residuals=ob.data();CpuOperator op;auto now=hdfe::reduced_cuda_controller::solve(p,cfg,op);
        if(old.solved!=now.solved||old.decline!=now.decline||old.iterations!=now.iterations||old.columns.size()!=now.columns.size())throw std::runtime_error("controller decision drift");
        for(std::size_t c=0;c<old.columns.size();++c){const auto& x=old.columns[c];const auto& z=now.columns[c];
            if(x.stop!=z.stop||x.iterations!=z.iterations||x.formed_checks!=z.formed_checks||x.first_tau_pass!=z.first_tau_pass)throw std::runtime_error("column stopping drift");}
        if(old.solved)for(int c=0;c<cols;++c)if(std::memcmp(a[c].data(),b[c].data(),n*sizeof(double)))throw std::runtime_error("CPU operator residual bytes drift");
        ++cases;
        if(old.solved&&op.calls>0&&op.replacements==0)throw std::runtime_error("no replacement reached the operator");
        // The same problem through the recording operator, with the replacement on the CPU
        // (registration of p and q) and on the device (of x, r, p and q): widths, one
        // registration, the products inside it and on their operands, p -> x -> p, its
        // release; and once more with a failure thrown at the second iterative product,
        // and at the first replacement (the registration is still released exactly once).
        for(int device=0;device<2;++device)for(int fail=0;fail<3;++fail){
            RecordingOperator rec;rec.device_replacement=device==1;
            rec.throw_at=fail==1?1:-1;rec.throw_at_replacement=fail==2?0:-1;
            std::vector<std::vector<double>> c2(cols,std::vector<double>(n));std::vector<double*> oc(cols);
            for(int c=0;c<cols;++c)oc[c]=c2[c].data();
            p.residuals=oc.data();
            bool thrown=false;rp::Report r;
            try{r=hdfe::reduced_cuda_controller::solve(p,cfg,rec);}catch(const std::runtime_error&){thrown=true;}
            const int groups=(cols+7)/8,per_group=(cols+groups-1)/groups;
            std::vector<int> want;for(int g=0;g<groups;++g)want.push_back(std::min(per_group,cols-g*per_group));
            // A decline or a failure stops the groups: a prefix of the widths, all of them when solved.
            const bool prefix=rec.widths.size()<=want.size()&&std::equal(rec.widths.begin(),rec.widths.end(),want.begin());
            if(!prefix||(!thrown&&r.solved&&rec.widths!=want))throw std::runtime_error("group widths");
            if(rec.pins!=1||rec.unpins!=1||rec.pinned)throw std::runtime_error("registration not exactly once, or not released");
            if(!rec.inside||!rec.order||!rec.operands)throw std::runtime_error("a product outside the registered range, on other operands, or out of order");
            const std::size_t lanes_used=static_cast<std::size_t>(rp::internal::lanes_for(per_group));
            const std::size_t m_reduced=static_cast<std::size_t>(levels[1]+levels[2]);
            if(rec.pinned_count!=(device?4U:2U)*m_reduced*lanes_used)throw std::runtime_error("registered range: p, q (2 slices) or x, r, p, q (4)");
            if(fail==1&&rec.applies>1&&!thrown)throw std::runtime_error("the injected failure did not leave solve()");
            if(fail==2&&rec.replacements>0&&!thrown)throw std::runtime_error("the failure at a replacement did not leave solve()");
            if(!fail&&r.solved){
                // The recording operator computes with the CPU operator: the same residual bytes.
                for(int c=0;c<cols;++c)if(std::memcmp(a[c].data(),c2[c].data(),n*sizeof(double)))throw std::runtime_error("recording operator residual bytes drift");
                if(rec.replacements==0)throw std::runtime_error("no replacement reached the recording operator");
            }
            ++cases;
        }
        p.residuals=oa.data();
    }
    std::cout<<"CUDA_CONTROLLER_CPU_PARITY_PASS "<<cases<<"\n";
}
