// Compiles byte-identical builder bodies extracted by the offline Python guard.
#include "ac_factor_fragment.hpp"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>

namespace {
using hdfe::detail::MlsmrSchurEdge;
using Factor=hdfe::detail::MlsmrApproxCholFactor;
template<class T>bool identical(const std::vector<T>& a,const std::vector<T>& b){
    return a.size()==b.size() && (a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
bool same(const Factor& a,const Factor& b){return a.n==b.n &&
    identical(a.vertices,b.vertices)&&identical(a.offsets,b.offsets)&&
    identical(a.neighbor_indices,b.neighbor_indices)&&identical(a.fractions,b.fractions)&&
    identical(a.inv_diagonal,b.inv_diagonal);}
template<class T>void dump(std::FILE* file,const std::vector<T>& a){
    const std::uint64_t length=a.size();
    if(std::fwrite(&length,sizeof(length),1,file)!=1 ||
       (!a.empty() && std::fwrite(a.data(),sizeof(T),a.size(),file)!=a.size()))
        throw std::runtime_error("factor archive write failed");
}
void dump(std::FILE* file,const Factor& f){
    if(std::fwrite(&f.n,sizeof(f.n),1,file)!=1)throw std::runtime_error("factor archive write failed");
    dump(file,f.vertices);dump(file,f.offsets);dump(file,f.neighbor_indices);
    dump(file,f.fractions);dump(file,f.inv_diagonal);
}
std::vector<MlsmrSchurEdge> graph(int n,int family,std::uint64_t seed){
    std::mt19937_64 rng(seed);std::vector<MlsmrSchurEdge> edges;
    auto add=[&](int a,int b,double weight=1.0){if(a!=b){if(a>b)std::swap(a,b);edges.push_back({a,b,weight});}};
    if(family==0)return edges;
    if(family==1||family==2||family==8){
        for(int v=1;v<n;++v){add(v-1,v);if(family==8){add(v-1,v,0.25);add(v-1,v,2.0e-14);}}
        if(family==2)add(0,n-1);
    }else if(family==3){for(int v=1;v<n;++v)add(0,v,std::ldexp(1.0,-v%4));}
    else if(family==4){for(int v=1;v<n;++v)if(v%(n/4+1))add(v-1,v);}
    else if(family==5){for(int v=0;v<n;++v)for(int k:{1,7,13})add(v,(v+k)%n);}
    if(family==6||family==8){for(int k=0;k<6*n;++k){
        const int a=rng()%n;const int b=rng()%n;const int exponent=rng()%4;
        add(a,b,std::ldexp(1.0,-exponent));
    }}
    if(family==7){for(int a=0;a<n;++a)for(int b=a+1;b<n;++b)
        if((rng()&3U)==0)add(a,b,std::ldexp(1.0,-static_cast<int>(rng()%4)));}
    return edges;
}
void finite(const Factor& f){
    for(double v:f.fractions)if(!hdfe::detail::ieee_finite(v))throw std::runtime_error("nonfinite fraction");
    for(double v:f.inv_diagonal)if(!hdfe::detail::ieee_finite(v))throw std::runtime_error("nonfinite inverse diagonal");
}
int run(int argc,char** argv){
    if(argc<2)throw std::runtime_error("archive path required");
    const bool negative=argc>2 && std::string(argv[2])=="negative";
    std::FILE* file=std::fopen(argv[1],"wbx");
    if(!file)throw std::runtime_error("archive exists or cannot be created");
    int cases=0;
    auto check_case=[&](int n,int family,std::uint64_t seed,int split){
        auto edges=graph(n,family,seed);
        auto reference=Factor::build_reference(n,edges,seed^0x93481aULL,split);
        auto arena=Factor::build_arena(n,edges,seed^0x93481aULL,split);
        if(!reference||!arena)throw std::runtime_error("factor missing");
        finite(*reference);finite(*arena);
        if(negative && !arena->fractions.empty()){
            arena->fractions[0]=std::nextafter(arena->fractions[0],std::numeric_limits<double>::infinity());
            if(same(*reference,*arena))throw std::runtime_error("negative control was accepted");
            std::puts("AC_FACTOR_NEGATIVE_CONTROL_REJECTED");return false;
        }
        if(!same(*reference,*arena))throw std::runtime_error("full factor differs from reference");
        if(n>=262144 && edges.size()>=(std::size_t{1}<<16)){
            auto dispatched=Factor::build(n,edges,seed^0x93481aULL,split);
            if(!dispatched||!same(*arena,*dispatched))throw std::runtime_error("dispatched factor differs");
        }
        dump(file,*arena);++cases;return true;
    };
    if(negative){
        const bool completed=check_case(64,1,11,1);std::fclose(file);return completed?2:1;
    }
    for(int family=0;family<9;++family)for(int seed=0;seed<16;++seed)for(int split:{1,2,4})
        check_case(32+32*(seed%8),family,static_cast<std::uint64_t>(family)*1000003U+seed,split);
    for(int family:{1,3,5,6,7,8})check_case(family==7?1024:4096,family,91827+family,2);
    for(int family:{1,3,5})check_case(262144,family,283746+family,2);
    if(std::fclose(file)!=0)throw std::runtime_error("factor archive close failed");
    std::printf("AC_FACTOR_MEMORY_CONTRACT_PASS cases=%d\n",cases);return 0;
}
}
int main(int argc,char** argv){
    try{return run(argc,argv);}
    catch(const std::exception& e){std::fprintf(stderr,"AC_FACTOR_MEMORY_CONTRACT_FAIL: %s\n",e.what());return 1;}
}
