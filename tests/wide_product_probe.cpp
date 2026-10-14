#include <Eigen/Core>
#include "wide_float.hpp"
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iostream>

int main() {
    // Exercise subnormals under IEEE arithmetic, including when the test
    // executable was linked with fast-math's startup environment.
    if (std::fesetenv(FE_DFL_ENV)) return 2;
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
        std::uint64_t pb, eb;
        std::memcpy(&pb,&p,sizeof(p)); std::memcpy(&eb,&error,sizeof(error));
        std::cout << std::hex << pb << ' ' << eb << '\n';
    }
    return std::cin.eof() ? 0 : 1;
}
