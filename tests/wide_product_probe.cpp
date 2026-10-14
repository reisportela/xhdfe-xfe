#include <Eigen/Core>
#include "wide_float.hpp"
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iostream>
#if defined(__SSE__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

int main() {
    // Exercise subnormals under IEEE arithmetic, including when the test
    // executable was linked with fast-math's startup environment.
    if (std::fesetenv(FE_DFL_ENV)) return 2;
#if defined(__SSE__) || defined(_M_X64)
    // MinGW fesetenv does not clear fast-math's SSE flush modes.
    _mm_setcsr(_mm_getcsr() & ~0x8040U);
    if (_mm_getcsr() & 0x8040U) return 3;
#endif
    std::uint64_t ab, bb;
    while (std::cin >> std::hex >> ab >> bb) {
        double a, b;
        std::memcpy(&a,&ab,sizeof(a)); std::memcpy(&b,&bb,sizeof(b));
        volatile double rounded=a*b;
#if defined(__SIZEOF_INT128__)
        const double error=hdfe::detail::OlsWide::exact_product_error(a,b,rounded);
#else
        const double error=std::fma(a,b,-rounded);
#endif
        const double p=rounded;
        const auto wide=hdfe::detail::OlsWide(a)*hdfe::detail::OlsWide(b);
        std::uint64_t pb, eb, wh, wl;
        std::memcpy(&pb,&p,sizeof(p)); std::memcpy(&eb,&error,sizeof(error));
        std::memcpy(&wh,&wide.hi,sizeof(wh)); std::memcpy(&wl,&wide.lo,sizeof(wl));
        std::cout << std::hex << pb << ' ' << eb << ' ' << wh << ' ' << wl << '\n';
    }
    return std::cin.eof() ? 0 : 1;
}
