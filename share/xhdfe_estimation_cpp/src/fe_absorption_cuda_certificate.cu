// WP3 — device-native certificate verifier (dedicated translation unit).
//
// This TU exists under precondition P1 of
// XHDFE_GPU_NATIVE_CERTIFICATE_MAX_PERFORMANCE_PLAN_CLAUDE_20260728.md: the
// verifier must not share compilation flags, kernels, scratch or reduction
// trees with the solver (§6.7), and must be compiled with strict FP semantics
// (§7.6): --fmad=false --prec-div=true --prec-sqrt=true --ftz=false on the
// device side, no fast-math anywhere, and -frecord-gcc-switches on the host
// side so the §15.3 flag gate can prove the flags from the object itself.
//
// Current contents (WP3 increment 1): the DECISION core of §6.5 — the
// quadratic-form comparison under directed rounding — plus its test entry
// points. The segmented reduction kernels for S(j,m,c)[g] and Σ|termo| (§6.6,
// §7.2) land in the next increment; nothing in the production path calls this
// TU yet, so it is inert until the shadow-mode wiring arrives.
//
// Decision contract (§6.5), implemented WITHOUT sqrt or division:
//   with A_sq = Op_sq(j,m), B_sq = Orig_sq(c), R = Res_sq(j,m,c):
//     T_lower = rd( rd(L*L) *rd A_sq_lower *rd B_sq_lower )
//     T_upper = ru( ru(L*L) *ru A_sq_upper *ru B_sq_upper )
//     PASS          <=> R_upper <= T_lower
//     FAIL          <=> R_lower >  T_upper
//     INDETERMINATE <=> otherwise
//     ERROR         <=> any input non-finite (never PASS)
//   Degeneracies: A_sq_lower == 0 or B_sq_lower == 0 => T_lower == 0 =>
//   PASS only if R_upper == 0 (fail-closed by construction).

#if defined(__FAST_MATH__) || defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "certificate verifier TU must not be compiled with fast-math (plan §7.6)"
#endif

#include <cuda_runtime.h>

#include <cstdint>

namespace hdfe_cert {

enum class CertVerdict : int {
    kPass = 0,
    kFail = 1,
    kIndeterminate = 2,
    kError = 3,
};

struct DecisionInput {
    double r_lower;      // lower bound on Res_sq(j,m,c)
    double r_upper;      // upper bound on Res_sq(j,m,c)
    double a_sq_lower;   // lower bound on Op_sq(j,m)
    double a_sq_upper;   // upper bound on Op_sq(j,m)
    double b_sq_lower;   // lower bound on Orig_sq(c)
    double b_sq_upper;   // upper bound on Orig_sq(c)
};

// __dadd_rd/__dmul_rd round toward -inf, __dadd_ru/__dmul_ru toward +inf,
// independently of the prevailing rounding mode and immune to contraction
// (this TU is compiled with --fmad=false; FMA may only appear via explicit
// intrinsics accounted in the bound, and the decision core uses none).
__device__ inline CertVerdict decide_one(const DecisionInput& in, double L) {
    const bool finite =
        isfinite(in.r_lower) && isfinite(in.r_upper) &&
        isfinite(in.a_sq_lower) && isfinite(in.a_sq_upper) &&
        isfinite(in.b_sq_lower) && isfinite(in.b_sq_upper) && isfinite(L) &&
        in.r_lower >= 0.0 && in.a_sq_lower >= 0.0 && in.b_sq_lower >= 0.0 &&
        in.r_upper >= in.r_lower && in.a_sq_upper >= in.a_sq_lower &&
        in.b_sq_upper >= in.b_sq_lower && L >= 0.0;
    if (!finite) {
        return CertVerdict::kError;
    }
    const double l_sq_lower = __dmul_rd(L, L);
    const double l_sq_upper = __dmul_ru(L, L);
    const double t_lower =
        __dmul_rd(__dmul_rd(l_sq_lower, in.a_sq_lower), in.b_sq_lower);
    const double t_upper =
        __dmul_ru(__dmul_ru(l_sq_upper, in.a_sq_upper), in.b_sq_upper);
    // t_upper can overflow to +inf; a +inf threshold upper bound is a valid
    // (conservative) bound, so it only widens INDETERMINATE, never mis-PASSes.
    if (in.r_upper <= t_lower) {
        return CertVerdict::kPass;
    }
    if (in.r_lower > t_upper) {
        return CertVerdict::kFail;
    }
    return CertVerdict::kIndeterminate;
}

__global__ void decide_kernel(const DecisionInput* inputs,
                              int count,
                              double L,
                              int* verdicts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        verdicts[idx] = static_cast<int>(decide_one(inputs[idx], L));
    }
}

}  // namespace hdfe_cert

// Test-only C entry: evaluate a batch of decision inputs on the DEVICE and
// return the verdicts. Used by the WP3 boundary harness (below / exactly at /
// above the threshold, §6.8) and by the oracle-containment tests; never called
// from the production fit path.
extern "C" int xhdfe_cert_decide_device(const hdfe_cert::DecisionInput* inputs,
                                        int count,
                                        double L,
                                        int* verdicts_out) {
    if (count <= 0) {
        return 1;
    }
    hdfe_cert::DecisionInput* d_in = nullptr;
    int* d_out = nullptr;
    if (cudaMalloc(&d_in, sizeof(hdfe_cert::DecisionInput) * count) !=
        cudaSuccess) {
        return 2;
    }
    if (cudaMalloc(&d_out, sizeof(int) * count) != cudaSuccess) {
        cudaFree(d_in);
        return 2;
    }
    int rc = 0;
    if (cudaMemcpy(d_in, inputs, sizeof(hdfe_cert::DecisionInput) * count,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        rc = 3;
    } else {
        const int block = 128;
        const int grid = (count + block - 1) / block;
        hdfe_cert::decide_kernel<<<grid, block>>>(d_in, count, L, d_out);
        if (cudaGetLastError() != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            rc = 4;
        } else if (cudaMemcpy(verdicts_out, d_out, sizeof(int) * count,
                              cudaMemcpyDeviceToHost) != cudaSuccess) {
            rc = 3;
        }
    }
    cudaFree(d_in);
    cudaFree(d_out);
    return rc;
}
