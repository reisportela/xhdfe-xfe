// Two facts the exact kernels rely on, checked on the toolchain and target
// of the build:
//   1. std::fma() rounds once (a libm that rounds twice is not the
//      arithmetic the lockstep kernels were validated with);
//   2. hdfe::detail::target_mul_add() computes a * b + c the way a plain
//      loop compiled with the same flags computes it — one rounding where
//      the compiler contracts the pair into a fused multiply-add, two with
//      XHDFE_FP_CONTRACT_OFF and -ffp-contract=off — which makes the fused Gauss-Seidel step and the
//      certificate give the bits of the two-pass kernels.
// The inputs make the two roundings visible: (1 + 2^-27)(1 - 2^-27) = 1 - 2^-54
// is exact in a fused operation and rounds to 1 on its own.
#include "hdfe/deterministic_parallel.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// Probe the semantics inside the same exact FP scope as the product kernels.
XHDFE_EXACT_FP
__attribute__((noinline)) double exact_fma(double a, double b, double c) {
    XHDFE_EXACT_FP_BODY
    return std::fma(a, b, c);
}

// Exact kernels also contain ordinary product-plus-sum expressions. Their
// scope must keep the two roundings separate even when the target has FMA.
XHDFE_EXACT_FP
__attribute__((noinline)) double exact_plain(double a, double b, double c) {
    XHDFE_EXACT_FP_BODY
    const double product = a * b;
    return product + c;
}


// A loop the compiler sees as the two-pass kernels see theirs: no attribute,
// free to contract.
__attribute__((noinline)) void plain_mul_add(const double* a,
                                             const double* b,
                                             const double* c,
                                             double* out,
                                             int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = a[i] * b[i] + c[i];
    }
}

__attribute__((noinline)) void exact_mul_add(const double* a,
                                             const double* b,
                                             const double* c,
                                             double* out,
                                             int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = hdfe::detail::target_mul_add(a[i], b[i], c[i]);
    }
}

}  // namespace

int main() {
    const double eps27 = std::ldexp(1.0, -27);
    volatile double a_hard = 1.0 + eps27;
    volatile double b_hard = 1.0 - eps27;
    volatile double c_hard = -1.0;
    const double fused = exact_fma(a_hard, b_hard, c_hard);
    if (fused != -std::ldexp(1.0, -54)) {
        std::fprintf(stderr, "FAIL: std::fma rounds twice on this toolchain (%a)\n", fused);
        return 1;
    }
    const double separate = exact_plain(a_hard, b_hard, c_hard);
    if (separate != 0.0) {
        std::fprintf(stderr,
                     "FAIL: XHDFE_EXACT_FP_BODY contracted a*b+c (%a)\n",
                     separate);
        return 1;
    }

    constexpr int n = 64;
    std::vector<double> a(n), b(n), c(n), plain(n), exact(n);
    for (int i = 0; i < n; ++i) {
        // Full mantissas, with the hard pair in every eighth slot.
        a[i] = (i % 8 == 0) ? a_hard : 1.0 + std::ldexp(static_cast<double>(i * 7919 % 1013), -20) + std::ldexp(1.0, -50);
        b[i] = (i % 8 == 0) ? b_hard : 1.0 - std::ldexp(static_cast<double>(i * 104729 % 997), -21) - std::ldexp(3.0, -51);
        c[i] = (i % 8 == 0) ? c_hard : -0.75 + std::ldexp(static_cast<double>(i), -30);
    }
    plain_mul_add(a.data(), b.data(), c.data(), plain.data(), n);
    exact_mul_add(a.data(), b.data(), c.data(), exact.data(), n);
    int differ = 0;
    for (int i = 0; i < n; ++i) {
        differ += std::memcmp(&plain[i], &exact[i], sizeof(double)) != 0;
    }
    if (differ) {
        std::fprintf(stderr,
                     "FAIL: target_mul_add differs from a plain a*b+c loop in %d of %d "
                     "entries (__FP_FAST_FMA %s, contract-off policy %s)\n",
                     differ, n,
#if defined(__FP_FAST_FMA) || defined(FP_FAST_FMA)
                     "defined",
#else
                     "undefined",
#endif
#if defined(XHDFE_FP_CONTRACT_OFF)
                     "enabled"
#else
                     "disabled"
#endif
        );
        return 1;
    }
    return 0;
}
