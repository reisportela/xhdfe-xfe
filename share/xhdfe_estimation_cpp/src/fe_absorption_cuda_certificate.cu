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

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fe_absorption_cuda.hpp"

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

// ---------------------------------------------------------------------------
// WP3 increment 2 — shadow-mode device certificate.
//
// Everything below computes, on the device and with this TU's own kernels and
// reduction trees (§6.7), the aggregate quantities the HOST certificate
// decides on —
//     Res_sq[c]  = Σ_dims Σ_g ( Σ_{i∈g} w_i · resid[i,c] )²
//     Op_sq_agg  = Σ_dims Σ_i w_i²
//     Orig_sq[c] = Σ_i original[i,c]²
// — together with the §6.6 error bracket needed by the §6.5 directed-rounding
// decision. Scope of this increment: the standard surface (no heterogeneous
// slopes; unit or strictly-positive analytic weights). Slopes/GI/savefe are
// WP4 kernels.
//
// Reduction shapes and their K (tree depth) used in γ_K = K·u/(1−K·u):
//   * column sum-squares (Orig/Op): ONE fixed block of 1024 threads,
//     grid-stride serial per thread then shared tree.
//     K = ceil(n/1024) + 10 (+1 for the squaring of each term).
//   * contiguous dims: one block of 256 per group (grid-stride over groups),
//     serial per-thread stride + shared tree; deterministic.
//     K1 = ceil(n_gmax/256) + 8 (+1 for the w·v product rounding).
//   * non-contiguous dims: per-element atomicAdd into S[g,c] / A[g,c].
//     Order-free bound (§6.6): K1 = n_gmax (+1 product rounding). The W=4096
//     tile tree of §7.2 tightens this in a later increment; with the corpus's
//     n_g and L there are ≥2 orders of slack even at K = n_g.
//   * group finalisation into per-(dim,c) scalars: atomicAdd per group.
//     K2 = G_total + dims (aggregation across dims included).
// Σ|termo| (A) is accumulated in the SAME tree as S (§6.6 requirement).
// ---------------------------------------------------------------------------

namespace {

constexpr double kU = 1.1102230246251565404e-16;   // 2^-53
constexpr double kEta = 4.9406564584124654e-324;   // 2^-1074 (denormal min)
constexpr int kColBlock = 1024;
constexpr int kSegBlock = 256;

__host__ __device__ inline double gamma_of(double k_depth) {
    // γ_K = K·u/(1−K·u), monotone; caller guarantees K·u < 1. On the device
    // this is written with directed intrinsics for two reasons: (1) the FMA
    // gate caught nvcc contracting the plain `1.0 - k*u` into fma even under
    // --fmad=false (audit A-02, second occurrence), and intrinsics are never
    // contracted; (2) rounding ku UP, the denominator DOWN and the quotient
    // UP deliberately OVER-estimates γ, which is conservative for every use:
    // it widens the error terms (upper side) and deflates the (1−γ) lower
    // brackets further.
#ifdef __CUDA_ARCH__
    const double ku = __dmul_ru(k_depth, kU);
    const double den = __dadd_rd(1.0, -ku);
    return __ddiv_ru(ku, den);
#else
    const double ku = k_depth * kU;
    return ku / (1.0 - ku);
#endif
}

// Fixed-shape sum of squares per column (c = 0 → y, c ≥ 1 → X col c-1),
// deterministic: single block, serial grid-stride per thread, shared tree.
__global__ void colwise_sumsq_kernel(const double* y,
                                     const double* x,
                                     int n,
                                     int cols,
                                     int ld,
                                     double* out /* cols+1 */) {
    __shared__ double sh[kColBlock];
    for (int c = 0; c <= cols; ++c) {
        const double* v = (c == 0) ? y : (x + static_cast<std::size_t>(c - 1) * ld);
        double acc = 0.0;
        for (int i = threadIdx.x; i < n; i += kColBlock) {
            const double t = v[i];
            // Explicit __dmul_rn/__dadd_rn: NVHPC 25.3 nvcc contracts a*b+c
            // into DFMA even under --fmad=false (audit A-02); intrinsics are
            // never contracted, so the §6.6 one-rounding-per-op model holds
            // in the object, not just on the command line.
            acc = __dadd_rn(acc, __dmul_rn(t, t));
        }
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (int s = kColBlock / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sh[threadIdx.x] = __dadd_rn(sh[threadIdx.x], sh[threadIdx.x + s]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            out[c] = sh[0];
        }
        __syncthreads();
    }
}

// Op_sq for one (dim, moment): Σ mult², mult = w (intercept/standard moment)
// or w·z (slope moment). Same fixed single-block tree as the column kernel.
__global__ void weights_sumsq_kernel(const double* w,
                                     const double* z,
                                     int n,
                                     double* out) {
    __shared__ double sh[kColBlock];
    double acc = 0.0;
    for (int i = threadIdx.x; i < n; i += kColBlock) {
        double t = (w != nullptr) ? w[i] : 1.0;
        if (z != nullptr) {
            t = __dmul_rn(t, z[i]);
        }
        acc = __dadd_rn(acc, __dmul_rn(t, t));
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kColBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh[threadIdx.x] = __dadd_rn(sh[threadIdx.x], sh[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *out = sh[0];
    }
}

// Contiguity probe: monotone non-decreasing gid over rows.
__global__ void contiguity_kernel(const int* gid, int n, int* flag_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < n && gid[i] < gid[i - 1]) {
        atomicExch(flag_out, 0);
    }
}

// group_ptr derivation for a contiguous dim: start[g] via binary search
// (deterministic, one thread per group), start[G] = n.
__global__ void group_ptr_kernel(const int* gid, int n, int groups, int* ptr) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g > groups) {
        return;
    }
    if (g == groups) {
        ptr[groups] = n;
        return;
    }
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (gid[mid] < g) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    ptr[g] = lo;
}

// Contiguous accumulation: one block per group (grid-stride), shared tree per
// column; S and A (Σ|termo|) written directly — no atomics, deterministic.
// mult(i, moment): moment 0 is the intercept/standard weight; slope moments
// multiply by z (two roundings before the add — the caller accounts 2u in K).
__device__ inline double cert_mult(const double* w,
                                   const double* z,
                                   int moment,
                                   bool slope_intercept,
                                   int i) {
    double m = (w != nullptr) ? w[i] : 1.0;
    if (z != nullptr && !(slope_intercept && moment == 0)) {
        m = __dmul_rn(m, z[i]);
    }
    return m;
}

__global__ void seg_accum_kernel(const double* resid_y,
                                 const double* resid_x,
                                 const double* w,
                                 const double* slope_z,
                                 bool slope_intercept,
                                 int moments,
                                 const int* ptr,
                                 int groups,
                                 int cols,
                                 int ld,
                                 double* S,   // groups x (moments x (cols+1))
                                 double* A,
                                 double* W,   // groups
                                 int* CNT) {  // groups
    __shared__ double sh_s[kSegBlock];
    __shared__ double sh_a[kSegBlock];
    const int rhsc = cols + 1;
    const int vpg = moments * rhsc;
    for (int g = blockIdx.x; g < groups; g += gridDim.x) {
        const int beg = ptr[g];
        const int end = ptr[g + 1];
        for (int vi = 0; vi < vpg; ++vi) {
            const int moment = vi / rhsc;
            const int c = vi % rhsc;
            const double* v =
                (c == 0) ? resid_y
                         : (resid_x + static_cast<std::size_t>(c - 1) * ld);
            double s = 0.0;
            double a = 0.0;
            for (int i = beg + threadIdx.x; i < end; i += kSegBlock) {
                const double wi =
                    cert_mult(w, slope_z, moment, slope_intercept, i);
                const double t = __dmul_rn(wi, v[i]);
                s = __dadd_rn(s, t);
                a = __dadd_rn(a, fabs(t));
            }
            sh_s[threadIdx.x] = s;
            sh_a[threadIdx.x] = a;
            __syncthreads();
            for (int st = kSegBlock / 2; st > 0; st >>= 1) {
                if (threadIdx.x < st) {
                    sh_s[threadIdx.x] =
                        __dadd_rn(sh_s[threadIdx.x], sh_s[threadIdx.x + st]);
                    sh_a[threadIdx.x] =
                        __dadd_rn(sh_a[threadIdx.x], sh_a[threadIdx.x + st]);
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                S[static_cast<std::size_t>(g) * vpg + vi] = sh_s[0];
                A[static_cast<std::size_t>(g) * vpg + vi] = sh_a[0];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            CNT[g] = end - beg;
        }
        double ws = 0.0;
        for (int i = beg + threadIdx.x; i < end; i += kSegBlock) {
            ws += (w != nullptr) ? w[i] : 1.0;
        }
        sh_s[threadIdx.x] = ws;
        __syncthreads();
        for (int st = kSegBlock / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) {
                sh_s[threadIdx.x] += sh_s[threadIdx.x + st];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            W[g] = sh_s[0];
        }
        __syncthreads();
    }
}

// Non-contiguous accumulation: per-element atomics (order-free §6.6 bound).
__global__ void atomic_accum_kernel(const double* resid_y,
                                    const double* resid_x,
                                    const double* w,
                                    const double* slope_z,
                                    bool slope_intercept,
                                    int moments,
                                    const int* gid,
                                    int n,
                                    int groups,
                                    int cols,
                                    int ld,
                                    double* S,
                                    double* A,
                                    double* W,
                                    int* CNT,
                                    int* bad_gid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int g = gid[i];
    if (g < 0 || g >= groups) {
        // Corrupt gid: record and bail — the finish path turns this into
        // ERROR (fail-closed). Writing through it would corrupt device
        // memory, which is exactly what the §6.7 cross-check exists to
        // detect (audit A-05).
        atomicExch(bad_gid, 1);
        return;
    }
    const int rhsc = cols + 1;
    const int vpg = moments * rhsc;
    for (int vi = 0; vi < vpg; ++vi) {
        const int moment = vi / rhsc;
        const int c = vi % rhsc;
        const double wi = cert_mult(w, slope_z, moment, slope_intercept, i);
        const double v =
            (c == 0) ? resid_y[i]
                     : resid_x[static_cast<std::size_t>(c - 1) * ld + i];
        const double t = __dmul_rn(wi, v);
        atomicAdd(&S[static_cast<std::size_t>(g) * vpg + vi], t);
        atomicAdd(&A[static_cast<std::size_t>(g) * vpg + vi], fabs(t));
    }
    atomicAdd(&W[g], (w != nullptr) ? w[i] : 1.0);
    atomicAdd(&CNT[g], 1);
}

// Per-(dim,c) scalar accumulators.
struct DimColAccum {
    double ss;  // Σ_g S²
    double sa;  // Σ_g |S|·A
    double aa;  // Σ_g A²
    double a1;  // Σ_g A
    double sn;  // Σ_g |S|·n_g
};

__device__ inline void atomic_max_double_nonneg(double* addr, double value) {
    // Non-negative doubles order like their bit patterns.
    unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long v = __double_as_longlong(value);
    unsigned long long old = *a;
    while (old < v) {
        const unsigned long long assumed = old;
        old = atomicCAS(a, assumed, v);
        if (old == assumed) {
            break;
        }
    }
}

__global__ void group_finalize_kernel(const double* S,
                                      const double* A,
                                      const double* W,
                                      const int* CNT,
                                      const double* solver_weight_sums,
                                      int groups,
                                      int cols,
                                      int moments,
                                      DimColAccum* acc /* cols+1; moments folded */,
                                      double* wsum_maxdiff,
                                      double* wsum_max,
                                      int* nmax,
                                      unsigned long long* empty) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    const int rhsc = cols + 1;
    const int vpg = moments * rhsc;
    const int cnt = CNT[g];
    if (cnt == 0) {
        atomicAdd(empty, 1ULL);
    }
    atomicMax(nmax, cnt);
    if (solver_weight_sums != nullptr) {
        atomic_max_double_nonneg(wsum_maxdiff,
                                 fabs(W[g] - solver_weight_sums[g]));
        atomic_max_double_nonneg(wsum_max, fabs(W[g]));
    }
    // The host contract folds moments into the per-column aggregates
    // (residual_norm_sq[c] and operator_norm_sq sum over dims AND moments),
    // so the accumulators do the same.
    for (int vi = 0; vi < vpg; ++vi) {
        const int c = vi % rhsc;
        const double sv = S[static_cast<std::size_t>(g) * vpg + vi];
        const double a = A[static_cast<std::size_t>(g) * vpg + vi];
        const double as = fabs(sv);
        atomicAdd(&acc[c].ss, sv * sv);
        atomicAdd(&acc[c].sa, as * a);
        atomicAdd(&acc[c].aa, a * a);
        atomicAdd(&acc[c].a1, a);
        atomicAdd(&acc[c].sn, as * static_cast<double>(cnt));
    }
}

// Final bracket + decision, one thread. All outward-rounded steps use the
// directed intrinsics; to-nearest values are diagnostics only.
struct ShadowDeviceOut {
    int verdict;
    int host_like_pass;
    double rel_max;
    double rel_max_perdim;
    double r_lower;
    double r_upper;
};

__global__ void bracket_decide_kernel(const DimColAccum* acc,  // dims x (cols+1)
                                      const double* k1_per_dim,  // host-computed depths
                                      const int* groups_per_dim,
                                      const int* moments_per_dim,
                                      const double* op_per_dim,  // dims x 2
                                      int dims,
                                      int cols,
                                      const double* orig_sq,   // cols+1
                                      double orig_k,
                                      double op_k,
                                      double n_rows,
                                      double L,
                                      int exact_zero_ok,
                                      ShadowDeviceOut* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int vpg = cols + 1;
    int worst = 0;  // PASS
    double worst_r_lo = 0.0, worst_r_up = 0.0;
    double rel_max = 0.0;
    double rel_max_perdim = 0.0;

    // Op_sq aggregate = Σ_dims Σ_moments op[d][m], each with the column-tree
    // bound, combined with directed rounding.
    const double g_op = gamma_of(op_k);
    // §6.6 (audit A-01): each accumulated fl(t·t) carries u·t² relative AND an
    // additive η floor — a purely relative bracket under-covers exactly in the
    // subnormal regime, where a false PASS becomes constructible. Widen both
    // Orig and Op brackets by n·η per accumulated term (outward-rounded).
    const double eta_n = __dmul_ru(kEta, n_rows);
    double op_agg = 0.0;    // to-nearest, for the host-semantics diagnostic
    double a_up = 0.0;
    double a_dn = 0.0;
    for (int d = 0; d < dims; ++d) {
        for (int m = 0; m < moments_per_dim[d]; ++m) {
            const double op = op_per_dim[d * 2 + m];
            op_agg += op;
            a_up = __dadd_ru(
                a_up,
                __dadd_ru(__dmul_ru(op, __dadd_ru(1.0, g_op)), eta_n));
            a_dn = __dadd_rd(
                a_dn,
                __dadd_rd(__dmul_rd(op, __dadd_rd(1.0, -g_op)), -eta_n));
        }
    }
    if (!(a_dn > 0.0)) {
        a_dn = 0.0;
    }
    const double g_orig = gamma_of(orig_k);

    long long total_groups = 0;
    for (int d = 0; d < dims; ++d) {
        total_groups += groups_per_dim[d];
    }
    const double g2 = gamma_of(static_cast<double>(total_groups) +
                               static_cast<double>(dims) + 16.0);

    for (int c = 0; c <= cols; ++c) {
        // Aggregate the per-dim scalars with outward rounding.
        double ss_up = 0.0, ss_dn = 0.0, err_up = 0.0;
        double a1_tot = 0.0;  // exact-zero channel detector
        double perdim_ss[1];  // silence unused warning pattern
        (void)perdim_ss;
        for (int d = 0; d < dims; ++d) {
            const DimColAccum a = acc[d * vpg + c];
            const double g1 = gamma_of(k1_per_dim[d]);
            // e_S(g) ≤ γ1·A_g + (n_g/2)η  ⇒
            // Σ_g (2|S|e + e²) ≤ 2γ1·SA + γ1²·AA + η·SN (+ η·A1 guard)
            double e = __dmul_ru(2.0, __dmul_ru(g1, a.sa));
            e = __dadd_ru(e, __dmul_ru(__dmul_ru(g1, g1), a.aa));
            e = __dadd_ru(e, __dmul_ru(kEta, a.sn));
            e = __dadd_ru(e, __dmul_ru(kEta, a.a1));
            err_up = __dadd_ru(err_up, e);
            ss_up = __dadd_ru(ss_up, a.ss);
            ss_dn = __dadd_rd(ss_dn, a.ss);
            a1_tot += a.a1;

            // Per-dimension diagnostic (P3): rel_d over the dim's own Op.
            double op_d = 0.0;
            for (int m = 0; m < moments_per_dim[d]; ++m) {
                op_d += op_per_dim[d * 2 + m];
            }
            const double denom_d = op_d * orig_sq[c];
            if (denom_d > 0.0) {
                const double rel_d = sqrt(a.ss / denom_d);
                if (rel_d > rel_max_perdim) {
                    rel_max_perdim = rel_d;
                }
            } else if (a.ss > 0.0) {
                rel_max_perdim = 1.0 / 0.0;
            }
        }
        // Reduction-of-squares rounding on the device's own SS accumulation.
        err_up = __dadd_ru(err_up, __dmul_ru(g2, ss_up));
        err_up = __dadd_ru(err_up,
                           __dmul_ru(kEta, static_cast<double>(total_groups)));
        // Exact-zero channel (re-audit R-01): sound ONLY when every group
        // multiplier is identically 1.0 — no weights and no slope dims
        // (exact_zero_ok, host-computed). Then t = fl(1·v) = v bit-exact, so
        // a1_tot == 0 means every accumulated v was ±0, every addition was
        // 0+0 — exact — and the true residual is exactly zero: no additive
        // error floor applies. With weights or slopes, t = fl(w·v) (or
        // fl(w·z·v)) can underflow to zero for nonzero w·v, so a1_tot == 0
        // does NOT imply a zero true residual and the §6.6 η floor must
        // stand (all-zeros inputs then yield INDET: fail-safe, accepted).
        // Do not "simplify" this gate to a moments count: a slope dim
        // without intercept has moments == 1 (its only moment IS the slope
        // moment, mult = w·z) and would re-open the underflow false PASS.
        if (exact_zero_ok && a1_tot == 0.0 && ss_up == 0.0) {
            err_up = 0.0;
        }
        const double r_up = __dadd_ru(ss_up, err_up);
        double r_dn = __dadd_rd(ss_dn, -err_up);
        if (!(r_dn > 0.0)) {
            r_dn = 0.0;
        }

        const double b = orig_sq[c];
        const double b_up =
            __dadd_ru(__dmul_ru(b, __dadd_ru(1.0, g_orig)), eta_n);
        double b_dn =
            __dadd_rd(__dmul_rd(b, __dadd_rd(1.0, -g_orig)), -eta_n);
        if (!(b_dn > 0.0)) {
            b_dn = 0.0;
        }

        hdfe_cert::DecisionInput in;
        in.r_lower = r_dn;
        in.r_upper = r_up;
        in.a_sq_lower = a_dn;
        in.a_sq_upper = a_up;
        in.b_sq_lower = b_dn;
        in.b_sq_upper = b_up;
        const int v = static_cast<int>(hdfe_cert::decide_one(in, L));
        // Worst-verdict ordering: ERROR > FAIL > INDET > PASS.
        const int rank[4] = {0, 2, 1, 3};
        if (rank[v] > rank[worst]) {
            worst = v;
            worst_r_lo = r_dn;
            worst_r_up = r_up;
        }

        // Host-semantics diagnostic (to-nearest, aggregate contract).
        const double scale = sqrt(op_agg) * sqrt(b);
        double rel;
        if (scale > 0.0) {
            rel = sqrt(ss_up) / scale;
        } else {
            rel = (ss_up > 0.0) ? 1.0 / 0.0 : 0.0;
        }
        if (rel > rel_max) {
            rel_max = rel;
        }
        if (c == 0 && worst_r_up == 0.0 && worst == 0) {
            worst_r_lo = r_dn;
            worst_r_up = r_up;
        }
    }
    out->verdict = worst;
    out->host_like_pass = (rel_max <= L) ? 1 : 0;
    out->rel_max = rel_max;
    out->rel_max_perdim = rel_max_perdim;
    out->r_lower = worst_r_lo;
    out->r_upper = worst_r_up;
}

}  // anonymous namespace

}  // namespace hdfe_cert

// Host-side shadow driver lives in hdfe::detail to match the header
// declarations; it uses the global hdfe_cert kernels.
namespace hdfe {
namespace detail {

struct ShadowDim {
    const int* d_gid = nullptr;
    const double* d_weight_sums = nullptr;
    const double* d_slope_z = nullptr;  // device slope values, slope dims only
    bool slope_intercept = false;
    int moments = 1;                    // 1 standard; 1-2 for slope dims
    int groups = 0;
    bool contiguous = false;
    int* d_group_ptr = nullptr;   // groups+1, contiguous dims only
    double* d_S = nullptr;
    double* d_A = nullptr;
    double* d_W = nullptr;
    int* d_CNT = nullptr;
};

struct CudaCertShadowHandle {
    int n = 0;
    int cols = 0;
    int ld = 0;
    const double* d_weights = nullptr;
    std::vector<ShadowDim> dims;
    double* d_orig_sq = nullptr;      // cols+1
    double* d_op = nullptr;           // dims x 2 (per moment; unused slot 0)
    int max_moments = 1;
    double* d_scalars = nullptr;      // dims*(cols+1) DimColAccum + tails
    hdfe_cert::DimColAccum* d_acc = nullptr;
    double* d_wsum_maxdiff = nullptr;
    double* d_wsum_max = nullptr;
    int* d_nmax = nullptr;            // per dim
    unsigned long long* d_empty = nullptr;
    hdfe_cert::ShadowDeviceOut* d_out = nullptr;
    double* d_k1 = nullptr;           // per dim depths
    int* d_groups = nullptr;          // per dim group counts
    int* d_moments = nullptr;         // per dim moment counts
    int* d_bad_gid = nullptr;         // A-05: corrupt-gid flag (fail-closed)
    double begin_ms = 0.0;
    bool begin_ok = false;
};

namespace {

inline bool sh_ok(cudaError_t e) { return e == cudaSuccess; }

template <typename T>
bool sh_alloc(T** p, std::size_t count) {
    return sh_ok(cudaMalloc(p, sizeof(T) * count)) &&
           sh_ok(cudaMemset(*p, 0, sizeof(T) * count));
}

double now_ms_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

}  // namespace

CudaCertShadowHandle* cuda_cert_shadow_begin(
    const double* d_y_original,
    const double* d_x_original,
    int n,
    int cols,
    int ld,
    const double* d_weights_or_null,
    const int* const* d_group_ids,
    const double* const* d_weight_sums,
    const int* num_groups,
    int num_dims,
    const double* const* d_slope_values,
    const unsigned char* slope_has_intercept) {
    if (n <= 0 || cols < 0 || num_dims <= 0) {
        return nullptr;
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto* h = new (std::nothrow) CudaCertShadowHandle();
    if (h == nullptr) {
        return nullptr;
    }
    h->n = n;
    h->cols = cols;
    h->ld = ld;
    h->d_weights = d_weights_or_null;
    h->dims.resize(static_cast<std::size_t>(num_dims));
    const int vpg = cols + 1;

    bool ok = sh_alloc(&h->d_orig_sq, static_cast<std::size_t>(vpg)) &&
              sh_alloc(&h->d_op, static_cast<std::size_t>(num_dims) * 2);
    // Solver-invariant quantities, captured from the freshly-uploaded
    // originals with this TU's own fixed-tree kernels (§6.7, §7.2). Op_sq is
    // per (dim, moment): standard dims use Σw²; slope dims add Σ(w·z)² for
    // the slope moment (and Σw² for the intercept moment when present).
    if (ok) {
        hdfe_cert::colwise_sumsq_kernel<<<1, hdfe_cert::kColBlock>>>(
            d_y_original, d_x_original, n, cols, ld, h->d_orig_sq);
        ok = sh_ok(cudaGetLastError());
    }
    for (int d = 0; ok && d < num_dims; ++d) {
        const double* z =
            (d_slope_values != nullptr) ? d_slope_values[d] : nullptr;
        const bool has_int =
            (slope_has_intercept != nullptr) && slope_has_intercept[d] != 0;
        const int moments = (z != nullptr) ? (has_int ? 2 : 1) : 1;
        ShadowDim& sd = h->dims[static_cast<std::size_t>(d)];
        sd.d_slope_z = z;
        sd.slope_intercept = (z != nullptr) && has_int;
        sd.moments = moments;
        if (moments > h->max_moments) {
            h->max_moments = moments;
        }
        for (int m = 0; ok && m < moments; ++m) {
            // moment 0 with an intercept (or a standard dim) uses mult = w;
            // the slope moment uses mult = w·z.
            const bool slope_moment = (z != nullptr) && !(sd.slope_intercept && m == 0);
            hdfe_cert::weights_sumsq_kernel<<<1, hdfe_cert::kColBlock>>>(
                d_weights_or_null, slope_moment ? z : nullptr, n,
                h->d_op + static_cast<std::size_t>(d) * 2 + m);
            ok = sh_ok(cudaGetLastError());
        }
    }
    int* d_flag = nullptr;
    if (ok) {
        ok = sh_ok(cudaMalloc(&d_flag, sizeof(int)));
    }
    for (int d = 0; ok && d < num_dims; ++d) {
        ShadowDim& sd = h->dims[static_cast<std::size_t>(d)];
        sd.d_gid = d_group_ids[d];
        sd.d_weight_sums = d_weight_sums[d];
        sd.groups = num_groups[d];
        if (sd.groups <= 0) {
            ok = false;
            break;
        }
        const int one = 1;
        ok = sh_ok(cudaMemcpy(d_flag, &one, sizeof(int),
                              cudaMemcpyHostToDevice));
        if (!ok) {
            break;
        }
        const int blocks = (n + 255) / 256;
        hdfe_cert::contiguity_kernel<<<blocks, 256>>>(sd.d_gid, n, d_flag);
        int flag = 0;
        ok = sh_ok(cudaGetLastError()) &&
             sh_ok(cudaMemcpy(&flag, d_flag, sizeof(int),
                              cudaMemcpyDeviceToHost));
        sd.contiguous = ok && flag == 1;
        if (ok && sd.contiguous) {
            ok = sh_ok(cudaMalloc(&sd.d_group_ptr,
                                  sizeof(int) * (static_cast<std::size_t>(
                                                     sd.groups) +
                                                 1)));
            if (ok) {
                const int gblocks = (sd.groups + 1 + 255) / 256;
                hdfe_cert::group_ptr_kernel<<<gblocks, 256>>>(
                    sd.d_gid, n, sd.groups, sd.d_group_ptr);
                ok = sh_ok(cudaGetLastError());
            }
        }
        if (ok) {
            const std::size_t cells = static_cast<std::size_t>(sd.groups) *
                                      static_cast<std::size_t>(sd.moments) *
                                      vpg;
            ok = sh_alloc(&sd.d_S, cells) && sh_alloc(&sd.d_A, cells) &&
                 sh_alloc(&sd.d_W, static_cast<std::size_t>(sd.groups)) &&
                 sh_alloc(&sd.d_CNT, static_cast<std::size_t>(sd.groups));
        }
    }
    if (d_flag != nullptr) {
        cudaFree(d_flag);
    }
    if (ok) {
        ok = sh_alloc(&h->d_acc,
                      static_cast<std::size_t>(num_dims) * vpg) &&
             sh_alloc(&h->d_wsum_maxdiff, 1) &&
             sh_alloc(&h->d_wsum_max, 1) &&
             sh_alloc(&h->d_nmax, static_cast<std::size_t>(num_dims)) &&
             sh_alloc(&h->d_empty, 1) &&
             sh_alloc(&h->d_out, 1) &&
             sh_alloc(&h->d_k1, static_cast<std::size_t>(num_dims)) &&
             sh_alloc(&h->d_groups, static_cast<std::size_t>(num_dims)) &&
             sh_alloc(&h->d_moments, static_cast<std::size_t>(num_dims)) &&
             sh_alloc(&h->d_bad_gid, 1);
    }
    if (ok) {
        ok = sh_ok(cudaDeviceSynchronize());
    }
    h->begin_ok = ok;
    h->begin_ms = now_ms_since(t0);
    if (!ok) {
        cuda_cert_shadow_abort(h);
        return nullptr;
    }
    return h;
}

bool cuda_cert_shadow_finish(CudaCertShadowHandle* h,
                             const double* d_y_residual,
                             const double* d_x_residual,
                             double certificate_limit,
                             CudaCertShadowResult* out) {
    if (out == nullptr) {
        return false;
    }
    *out = CudaCertShadowResult{};
    out->verdict = 3;  // ERROR until proven otherwise (fail-closed)
    if (h == nullptr || !h->begin_ok) {
        cuda_cert_shadow_abort(h);
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int n = h->n;
    const int cols = h->cols;
    const int vpg = cols + 1;
    const int dims = static_cast<int>(h->dims.size());
    bool ok = true;

    for (int d = 0; ok && d < dims; ++d) {
        ShadowDim& sd = h->dims[static_cast<std::size_t>(d)];
        if (sd.contiguous) {
            const int gblocks =
                sd.groups < 4096 ? sd.groups : 4096;
            hdfe_cert::seg_accum_kernel<<<gblocks, hdfe_cert::kSegBlock>>>(
                d_y_residual, d_x_residual, h->d_weights, sd.d_slope_z,
                sd.slope_intercept, sd.moments, sd.d_group_ptr,
                sd.groups, cols, h->ld, sd.d_S, sd.d_A, sd.d_W, sd.d_CNT);
        } else {
            const int blocks = (n + 255) / 256;
            hdfe_cert::atomic_accum_kernel<<<blocks, 256>>>(
                d_y_residual, d_x_residual, h->d_weights, sd.d_slope_z,
                sd.slope_intercept, sd.moments, sd.d_gid, n, sd.groups, cols,
                h->ld, sd.d_S, sd.d_A, sd.d_W, sd.d_CNT, h->d_bad_gid);
        }
        ok = sh_ok(cudaGetLastError());
        if (ok) {
            const int gblocks = (sd.groups + 255) / 256;
            hdfe_cert::group_finalize_kernel<<<gblocks, 256>>>(
                sd.d_S, sd.d_A, sd.d_W, sd.d_CNT, sd.d_weight_sums,
                sd.groups, cols, sd.moments,
                h->d_acc + static_cast<std::size_t>(d) * vpg,
                h->d_wsum_maxdiff, h->d_wsum_max, h->d_nmax + d, h->d_empty);
            ok = sh_ok(cudaGetLastError());
        }
    }

    // Depths for γ_K, from the observed n_gmax per dim (D2H of dims ints).
    std::vector<int> nmax(static_cast<std::size_t>(dims), 0);
    std::vector<int> groups(static_cast<std::size_t>(dims), 0);
    std::vector<int> moments(static_cast<std::size_t>(dims), 1);
    if (ok) {
        ok = sh_ok(cudaMemcpy(nmax.data(), h->d_nmax, sizeof(int) * dims,
                              cudaMemcpyDeviceToHost));
    }
    if (ok) {
        std::vector<double> k1(static_cast<std::size_t>(dims), 0.0);
        for (int d = 0; d < dims; ++d) {
            const ShadowDim& sd =
                h->dims[static_cast<std::size_t>(d)];
            groups[static_cast<std::size_t>(d)] = sd.groups;
            moments[static_cast<std::size_t>(d)] = sd.moments;
            const double depth =
                sd.contiguous
                    ? (static_cast<double>(
                           (nmax[static_cast<std::size_t>(d)] +
                            hdfe_cert::kSegBlock - 1) /
                           hdfe_cert::kSegBlock) +
                       8.0)
                    : static_cast<double>(nmax[static_cast<std::size_t>(d)]);
            // +1 for the w·v product rounding; slope moments carry one more
            // (mult = w·z is two roundings before the add, §6.6) — count 2
            // for the whole dim, conservative for its intercept moment too.
            k1[static_cast<std::size_t>(d)] =
                depth + ((sd.d_slope_z != nullptr) ? 2.0 : 1.0);
        }
        ok = sh_ok(cudaMemcpy(h->d_k1, k1.data(), sizeof(double) * dims,
                              cudaMemcpyHostToDevice)) &&
             sh_ok(cudaMemcpy(h->d_groups, groups.data(), sizeof(int) * dims,
                              cudaMemcpyHostToDevice)) &&
             sh_ok(cudaMemcpy(h->d_moments, moments.data(),
                              sizeof(int) * dims, cudaMemcpyHostToDevice));
    }
    if (ok) {
        // Column-tree depth for Orig/Op: ceil(n/1024)+10, +1 squaring; slope
        // moments carry the extra w·z rounding, +1 more (conservative for
        // every moment when any slope dim is present).
        const double col_k =
            static_cast<double>((n + hdfe_cert::kColBlock - 1) /
                                hdfe_cert::kColBlock) +
            11.0 + ((h->max_moments > 1) ? 1.0 : 0.0);
        // R-01 predicate: the exact-zero channel is sound only under
        // mult ≡ 1.0 — no weights AND no slope dim anywhere (a slope-only
        // dim has moments==1, so max_moments cannot substitute for this).
        int exact_zero_ok = (h->d_weights == nullptr) ? 1 : 0;
        for (int d = 0; d < dims && exact_zero_ok; ++d) {
            if (h->dims[static_cast<std::size_t>(d)].d_slope_z != nullptr) {
                exact_zero_ok = 0;
            }
        }
        hdfe_cert::bracket_decide_kernel<<<1, 1>>>(
            h->d_acc, h->d_k1, h->d_groups, h->d_moments, h->d_op, dims,
            cols, h->d_orig_sq, col_k, col_k, static_cast<double>(n),
            certificate_limit, exact_zero_ok, h->d_out);
        ok = sh_ok(cudaGetLastError());
    }
    hdfe_cert::ShadowDeviceOut dev{};
    unsigned long long empty = 0;
    double wdiff = 0.0;
    int bad_gid = 0;
    if (ok) {
        ok = sh_ok(cudaMemcpy(&dev, h->d_out, sizeof(dev),
                              cudaMemcpyDeviceToHost)) &&
             sh_ok(cudaMemcpy(&empty, h->d_empty, sizeof(empty),
                              cudaMemcpyDeviceToHost)) &&
             sh_ok(cudaMemcpy(&wdiff, h->d_wsum_maxdiff, sizeof(wdiff),
                              cudaMemcpyDeviceToHost)) &&
             sh_ok(cudaMemcpy(&bad_gid, h->d_bad_gid, sizeof(bad_gid),
                              cudaMemcpyDeviceToHost)) &&
             sh_ok(cudaDeviceSynchronize());
    }
    double wsum_max = 0.0;
    if (ok) {
        ok = sh_ok(cudaMemcpy(&wsum_max, h->d_wsum_max, sizeof(wsum_max),
                              cudaMemcpyDeviceToHost));
    }
    int nmax_all = 0;
    for (int d = 0; d < dims; ++d) {
        if (nmax[static_cast<std::size_t>(d)] > nmax_all) {
            nmax_all = nmax[static_cast<std::size_t>(d)];
        }
    }
    const double wsum_gamma =
        (2.0 * static_cast<double>(nmax_all) * 1.1102230246251565404e-16);
    // Conservative host-side inflation in lieu of directed rounding.
    const double wsum_tol =
        wsum_gamma * wsum_max * (1.0 + 1e-12) +
        static_cast<double>(n) * 4.9406564584124654e-324;
    if (ok && bad_gid != 0) {
        // A-05: at least one out-of-range gid — the accumulation is
        // incomplete by design. ERROR, never a verdict.
        dev.verdict = 3;
        dev.host_like_pass = -1;
    }
    if (ok && !(wdiff <= wsum_tol)) {
        // R-05: the weight-sums cross-check is the shadow's own integrity
        // probe (device group sums vs an independent device recount). If it
        // fails, the certificate inputs cannot be trusted, so no PASS/FAIL
        // verdict may stand — degrade to ERROR (fail-closed), same policy
        // as bad_gid. NaN wdiff also lands here (!(NaN <= tol)).
        dev.verdict = 3;
        dev.host_like_pass = -1;
    }
    if (ok) {
        out->verdict = dev.verdict;
        out->host_like_pass = dev.host_like_pass;
        out->rel_max = dev.rel_max;
        out->rel_max_perdim = dev.rel_max_perdim;
        out->r_lower = dev.r_lower;
        out->r_upper = dev.r_upper;
        out->wsum_maxdiff = wdiff;
        out->wsum_tol = wsum_tol;
        out->wsum_ok = (wdiff <= wsum_tol) ? 1 : 0;
        out->empty_groups = static_cast<long long>(empty);
        out->dims = dims;
        out->rhs = vpg;
    }
    out->shadow_begin_ms = h->begin_ms;
    out->shadow_finish_ms = now_ms_since(t0);
    cuda_cert_shadow_abort(h);
    return ok;
}

void cuda_cert_shadow_abort(CudaCertShadowHandle* h) {
    if (h == nullptr) {
        return;
    }
    for (auto& sd : h->dims) {
        if (sd.d_group_ptr) cudaFree(sd.d_group_ptr);
        if (sd.d_S) cudaFree(sd.d_S);
        if (sd.d_A) cudaFree(sd.d_A);
        if (sd.d_W) cudaFree(sd.d_W);
        if (sd.d_CNT) cudaFree(sd.d_CNT);
    }
    if (h->d_orig_sq) cudaFree(h->d_orig_sq);
    if (h->d_op) cudaFree(h->d_op);
    if (h->d_acc) cudaFree(h->d_acc);
    if (h->d_wsum_maxdiff) cudaFree(h->d_wsum_maxdiff);
    if (h->d_wsum_max) cudaFree(h->d_wsum_max);
    if (h->d_nmax) cudaFree(h->d_nmax);
    if (h->d_empty) cudaFree(h->d_empty);
    if (h->d_out) cudaFree(h->d_out);
    if (h->d_k1) cudaFree(h->d_k1);
    if (h->d_groups) cudaFree(h->d_groups);
    if (h->d_moments) cudaFree(h->d_moments);
    if (h->d_bad_gid) cudaFree(h->d_bad_gid);
    delete h;
}

}  // namespace detail
}  // namespace hdfe

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
