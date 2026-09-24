#if defined(XHDFE_SCALE_IMPLEMENTATION)
#include "../../src/fe_recovery_scale.hpp"
extern "C" void new_scale(double* p,long long n,int e) {
    Eigen::Map<Eigen::VectorXd> v(p,n);
    hdfe::detail::scale_fe_recovery_vector(v,e);
}
#elif defined(XHDFE_SCALE_REFERENCE)
#include <cmath>
extern "C" void reference_scale(double* p,long long n,int e) {
    for(long long i=0;i<n;++i) p[i]=std::ldexp(p[i],e);
}
#else
#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cfenv>
#include <climits>
#ifdef __SSE__
#include <xmmintrin.h>
#endif
extern "C" void new_scale(double*,long long,int);
extern "C" void reference_scale(double*,long long,int);
static double value(std::uint64_t bits){double x;std::memcpy(&x,&bits,8);return x;}
static std::uint64_t bits(double x){std::uint64_t r;std::memcpy(&r,&x,8);return r;}
int main(){
 const int saved_round=std::fegetround();
#ifdef __SSE__
 const unsigned saved=_mm_getcsr();
 auto set_control=[](unsigned x){_mm_setcsr(x);};
 auto get_control=[](){return _mm_getcsr();};
 const std::vector<unsigned> control_flags{0u,1u<<15,1u<<6,(1u<<15)|(1u<<6)};
#else
 const unsigned saved=0;
 auto set_control=[](unsigned){};
 auto get_control=[](){return 0u;};
 const std::vector<unsigned> control_flags{0u};
#endif
 std::vector<double> normal{0.,-0.};
 const std::uint64_t fractions[]={0,1,UINT64_C(0x8000000000000),UINT64_C(0xfffffffffffff)};
 for(int sign=0;sign<2;++sign)for(int e=1;e<2047;++e)for(auto fraction:fractions)
  normal.push_back(value((std::uint64_t(sign)<<63)|(std::uint64_t(e)<<52)|fraction));
 std::vector<std::vector<double>> sets{normal,{0.,-0.,value(1),value(UINT64_C(0x8000000000000001)),1.,-1.},
  {1.,-1.,value(UINT64_C(0x7ff0000000000000)),value(UINT64_C(0xfff0000000000000)),value(UINT64_C(0x7ff8000000000042)),value(UINT64_C(0x7ff0000000000042))}};
 std::vector<int> exponents;for(int e=-1075;e<=1075;++e)exponents.push_back(e);exponents.push_back(INT_MIN);exponents.push_back(INT_MAX);
 std::uint64_t cases=0,values=0;
 for(int rounding:{FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO}){
  std::fesetround(rounding);
  const unsigned base=get_control() & ~((1u<<15)|(1u<<6));
  for(unsigned flags:control_flags){
   set_control(base|flags);
   for(const auto& input:sets)for(int exponent:exponents){
    auto b=input,c=input;new_scale(b.data(),b.size(),exponent);reference_scale(c.data(),c.size(),exponent);
    ++cases;values+=input.size();
    for(std::size_t i=0;i<b.size();++i)if(bits(b[i])!=bits(c[i])){
     std::cerr<<"MISMATCH rounding="<<rounding<<" flags="<<flags<<" exponent="<<exponent<<" index="<<i
      <<" input="<<std::hex<<bits(input[i])<<" new="<<bits(b[i])<<" ref="<<bits(c[i])<<std::dec<<"\n";
     set_control(saved);std::fesetround(saved_round);return 1;
    }
   }
  }
 }
 set_control(saved);std::fesetround(saved_round);
 std::cout<<"{\"status\":\"PASS\",\"cases\":"<<cases<<",\"values\":"<<values<<",\"rounding_modes\":4,\"ftz_daz_states\":"<<control_flags.size()<<"}\n";
}

#endif
