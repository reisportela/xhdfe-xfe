#include "ols_precision.hpp"
#include <cstring>
#include <iostream>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

int main() {
    using namespace hdfe::detail;
    Eigen::VectorXi first(4),second(4),constant(4);
    first<<0,0,1,1;second<<0,1,0,1;constant.setZero();
    Eigen::VectorXd additive(4),wrong(4),y(4),weights(4);
    additive<<0,2,3,5;wrong<<0,2,3,6;y<<1,-1,1,-1;weights.setOnes();
    Eigen::MatrixXd X=y;
    int checks=0;
#if defined(__SSE__)
    const unsigned csr=_mm_getcsr();
#endif
    for(bool flush:{false,true}) {
#if defined(__SSE__)
        _mm_setcsr(flush ? csr|0x8040 : csr&~0x8040U);
#endif
        if(!exact_two_fe_span(additive,first,second) ||
           exact_two_fe_span(wrong,first,second)) return 1;
        checks+=2;
        if(!exact_level_projection(y,X,constant,&weights,y,X) ||
           exact_level_projection(y,X,constant,&weights,2*y,X)) return 2;
        checks+=2;
        std::uint64_t bits=1;double tiny;std::memcpy(&tiny,&bits,8);
        wrong.setZero();std::memcpy(&wrong[3],&tiny,8);
        if(exact_two_fe_span(wrong,first,second)) return 3;
        weights.setOnes();std::memcpy(&weights[0],&tiny,8);
        if(exact_level_projection(y,X,constant,&weights,y,X)) return 4;
        weights.setOnes();checks+=2;
    }
#if defined(__SSE__)
    _mm_setcsr(csr);
#endif
    std::cout<<"OLS_EXACT_OMISSION_CERTIFICATE_PASS "<<checks<<'\n';
}
