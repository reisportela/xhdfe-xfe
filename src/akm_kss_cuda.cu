// CUDA backend for the AKM/KSS two-way solver: Jacobi-PCG on the grounded
// worker-partialled firm Laplacian S = Df - B' Dw^{-1} B, matrix-free over
// the match structure. Opt-in from hdfe::akm (use_gpu); the CPU path is the
// reference. Reductions use CUB device sums (run-to-run deterministic on a
// given device).

#include "hdfe/akm_kss_cuda.hpp"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace hdfe {
namespace akm {

namespace {

constexpr int kBlock = 256;

inline bool cuda_ok(cudaError_t st) { return st == cudaSuccess; }

__global__ void k_iota_i64(std::int64_t* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (i < n) out[i] = static_cast<std::int64_t>(i);
}

__global__ void k_mult_b(int n_workers, const std::int64_t* w_ptr, const int* m_f,
                         const double* m_c, const double* x, const double* inv_dw,
                         double* t) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_workers) return;
    double acc = 0.0;
    for (std::int64_t a = w_ptr[w]; a < w_ptr[w + 1]; ++a) {
        acc += m_c[a] * x[m_f[a]];
    }
    t[w] = acc * inv_dw[w];
}

__global__ void k_mult_bt_finish(int n_firms, int ground, const std::int64_t* f_ptr,
                                 const std::int64_t* f_matches, const int* m_w,
                                 const double* m_c, const double* s,
                                 const double* Df, const double* p, double* y) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_firms) return;
    double acc = 0.0;
    for (std::int64_t a = f_ptr[f]; a < f_ptr[f + 1]; ++a) {
        const std::int64_t m = f_matches[a];
        acc += m_c[m] * s[m_w[m]];
    }
    y[f] = (f == ground) ? 0.0 : Df[f] * p[f] - acc;
}

__global__ void k_axpy(int n, double a, const double* x, double* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void k_xpay(int n, double a, const double* x, double* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + a * y[i];
}

__global__ void k_hadamard(int n, const double* a, const double* b, double* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void k_mul_pair(int n, const double* a, const double* b, double* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}


__global__ void k_mult_b_multi(int n_workers, int nb, const std::int64_t* w_ptr,
                               const int* m_f, const double* m_c, const double* x,
                               const double* inv_dw, const char* active, double* t,
                               int J, int N) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    if (w >= n_workers || l >= nb || !active[l]) return;
    const double* xl = x + static_cast<std::size_t>(l) * J;
    double acc = 0.0;
    for (std::int64_t a = w_ptr[w]; a < w_ptr[w + 1]; ++a) {
        acc += m_c[a] * xl[m_f[a]];
    }
    t[static_cast<std::size_t>(l) * N + w] = acc * inv_dw[w];
}

__global__ void k_mult_bt_finish_multi(int n_firms, int nb, int ground,
                                       const std::int64_t* f_ptr,
                                       const std::int64_t* f_matches, const int* m_w,
                                       const double* m_c, const double* s,
                                       const double* Df, const double* p, double* y,
                                       const char* active, int J, int N) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    if (f >= n_firms || l >= nb || !active[l]) return;
    const double* sl = s + static_cast<std::size_t>(l) * N;
    double acc = 0.0;
    for (std::int64_t a = f_ptr[f]; a < f_ptr[f + 1]; ++a) {
        const std::int64_t m = f_matches[a];
        acc += m_c[m] * sl[m_w[m]];
    }
    const std::size_t off = static_cast<std::size_t>(l) * J + f;
    y[off] = (f == ground) ? 0.0 : Df[f] * p[off] - acc;
}

// One PCG update, preserving the two legacy AXPY expressions element by
// element while avoiding a second scalar upload and a second full launch.
__global__ void k_cg_update_multi(int n, int nb, const double* alpha,
                                  const char* active, const double* p,
                                  const double* Ap, double* x, double* r) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    if (i >= n || l >= nb || !active[l]) return;
    const std::size_t off = static_cast<std::size_t>(l) * n + i;
    const double a = alpha[l];
    x[off] += a * p[off];
    const double neg_a = -a;
    r[off] += neg_a * Ap[off];
}

__global__ void k_xpay_multi(int n, int nb, const double* beta, const char* active,
                             const double* x, double* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    if (i >= n || l >= nb || !active[l]) return;
    const std::size_t off = static_cast<std::size_t>(l) * n + i;
    y[off] = x[off] + beta[l] * y[off];
}

__global__ void k_hadamard_multi(int n, int nb, const double* a, const char* active,
                                 const double* b, double* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    if (i >= n || l >= nb || !active[l]) return;
    const std::size_t off = static_cast<std::size_t>(l) * n + i;
    out[off] = a[i] * b[off];
}

__global__ void k_mul_pair_multi(int n, int nb, const double* a, const double* b,
                                 const char* active, double* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    if (i >= n || l >= nb) return;
    const std::size_t off = static_cast<std::size_t>(l) * n + i;
    // inactive lanes contribute zeros so the segmented sums stay defined
    out[off] = active[l] ? a[off] * b[off] : 0.0;
}

__global__ void k_seg_offsets(int nb, int n, int* offs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= nb) offs[i] = i * n;
}

}  // namespace

bool akm_cuda_stable_sort_u64(const std::uint64_t* keys,
                              std::int64_t* order_out,
                              std::size_t n) {
    if (n == 0) return true;
    if (keys == nullptr || order_out == nullptr ||
        n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        !akm_cuda_available()) {
        return false;
    }

    std::uint64_t* d_keys_in = nullptr;
    std::uint64_t* d_keys_out = nullptr;
    std::int64_t* d_order_in = nullptr;
    std::int64_t* d_order_out = nullptr;
    void* d_temp = nullptr;
    auto cleanup = [&]() {
        cudaFree(d_temp);
        cudaFree(d_order_out);
        cudaFree(d_order_in);
        cudaFree(d_keys_out);
        cudaFree(d_keys_in);
    };

    const std::size_t key_bytes = n * sizeof(std::uint64_t);
    const std::size_t order_bytes = n * sizeof(std::int64_t);
    bool ok = cuda_ok(cudaMalloc(&d_keys_in, key_bytes)) &&
              cuda_ok(cudaMalloc(&d_keys_out, key_bytes)) &&
              cuda_ok(cudaMalloc(&d_order_in, order_bytes)) &&
              cuda_ok(cudaMalloc(&d_order_out, order_bytes)) &&
              cuda_ok(cudaMemcpy(d_keys_in, keys, key_bytes,
                                 cudaMemcpyHostToDevice));
    if (!ok) {
        cleanup();
        return false;
    }

    const unsigned blocks = static_cast<unsigned>((n + kBlock - 1) / kBlock);
    k_iota_i64<<<blocks, kBlock>>>(d_order_in, n);
    if (!cuda_ok(cudaGetLastError())) {
        cleanup();
        return false;
    }

    std::size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                                    d_keys_in, d_keys_out,
                                    d_order_in, d_order_out,
                                    static_cast<int>(n));
    if (!cuda_ok(cudaGetLastError()) ||
        !cuda_ok(cudaMalloc(&d_temp, temp_bytes))) {
        cleanup();
        return false;
    }
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
                                    d_keys_in, d_keys_out,
                                    d_order_in, d_order_out,
                                    static_cast<int>(n));
    ok = cuda_ok(cudaGetLastError()) &&
         cuda_ok(cudaMemcpy(order_out, d_order_out, order_bytes,
                            cudaMemcpyDeviceToHost));
    cleanup();
    return ok;
}

struct AkmCudaContext {
    int N = 0;
    int J = 0;
    int ground = 0;
    std::size_t M = 0;
    // structure
    std::int64_t* d_w_ptr = nullptr;
    std::int64_t* d_f_ptr = nullptr;
    std::int64_t* d_f_matches = nullptr;
    int* d_m_w = nullptr;
    int* d_m_f = nullptr;
    double* d_m_c = nullptr;
    double* d_inv_dw = nullptr;
    double* d_Df = nullptr;
    double* d_inv_diag = nullptr;
    // workspace
    double* d_x = nullptr;
    double* d_r = nullptr;
    double* d_z = nullptr;
    double* d_p = nullptr;
    double* d_Ap = nullptr;
    double* d_tw = nullptr;
    double* d_pair = nullptr;
    double* d_dot = nullptr;
    void* d_cub_tmp = nullptr;
    std::size_t cub_tmp_bytes = 0;
    // multi-RHS workspace (allocated lazily on the first batched solve)
    int mr_lanes = 0;
    double* d_mx = nullptr;
    double* d_mr = nullptr;
    double* d_mz = nullptr;
    double* d_mp = nullptr;
    double* d_mAp = nullptr;
    double* d_mtw = nullptr;
    double* d_mpair = nullptr;
    double* d_mdots = nullptr;
    double* d_mscal = nullptr;
    char* d_mactive = nullptr;
    int* d_moffs = nullptr;
    void* d_mcub_tmp = nullptr;
    std::size_t mcub_tmp_bytes = 0;
    // pinned host staging (perf 09jul2026): page-locked buffers for the
    // rhs/z packs and the per-step scalar traffic of the multi-lane solver
    // (pageable transfers stage through the driver's internal bounce buffer
    // and pay extra latency on every CG step). All are optional — a null
    // pointer means pinned allocation failed and the pageable fallback is
    // used. Pinning changes transfer speed only, never values.
    double* h_pack = nullptr;   // J * mr_lanes (rhs upload / z download)
    double* h_dots = nullptr;   // 2 * mr_lanes (paired segmented-reduce results)
    double* h_scal = nullptr;   // mr_lanes (alpha/beta uploads)
    char* h_active = nullptr;   // mr_lanes (active-mask uploads)
    double* h_dot1 = nullptr;   // single-lane dot result (device_dot)

    ~AkmCudaContext() {
        cudaFree(d_w_ptr);
        cudaFree(d_f_ptr);
        cudaFree(d_f_matches);
        cudaFree(d_m_w);
        cudaFree(d_m_f);
        cudaFree(d_m_c);
        cudaFree(d_inv_dw);
        cudaFree(d_Df);
        cudaFree(d_inv_diag);
        cudaFree(d_x);
        cudaFree(d_r);
        cudaFree(d_z);
        cudaFree(d_p);
        cudaFree(d_Ap);
        cudaFree(d_tw);
        cudaFree(d_pair);
        cudaFree(d_dot);
        cudaFree(d_cub_tmp);
        cudaFree(d_mx);
        cudaFree(d_mr);
        cudaFree(d_mz);
        cudaFree(d_mp);
        cudaFree(d_mAp);
        cudaFree(d_mtw);
        cudaFree(d_mpair);
        cudaFree(d_mdots);
        cudaFree(d_mscal);
        cudaFree(d_mactive);
        cudaFree(d_moffs);
        cudaFree(d_mcub_tmp);
        cudaFreeHost(h_pack);
        cudaFreeHost(h_dots);
        cudaFreeHost(h_scal);
        cudaFreeHost(h_active);
        cudaFreeHost(h_dot1);
    }
};

bool akm_cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

namespace {

template <typename T>
bool upload(T** dst, const T* src, std::size_t count) {
    if (!cuda_ok(cudaMalloc(dst, sizeof(T) * count))) return false;
    return cuda_ok(cudaMemcpy(*dst, src, sizeof(T) * count, cudaMemcpyHostToDevice));
}

// deterministic device dot product via CUB
double device_dot(AkmCudaContext* c, const double* a, const double* b, int n) {
    const int grid = (n + kBlock - 1) / kBlock;
    k_mul_pair<<<grid, kBlock>>>(n, a, b, c->d_pair);
    std::size_t need = 0;
    cub::DeviceReduce::Sum(nullptr, need, c->d_pair, c->d_dot, n);
    if (need > c->cub_tmp_bytes) {
        cudaFree(c->d_cub_tmp);
        if (!cuda_ok(cudaMalloc(&c->d_cub_tmp, need))) return 0.0;
        c->cub_tmp_bytes = need;
    }
    cub::DeviceReduce::Sum(c->d_cub_tmp, c->cub_tmp_bytes, c->d_pair, c->d_dot, n);
    double out = 0.0;
    if (c->h_dot1 != nullptr) {
        cudaMemcpy(c->h_dot1, c->d_dot, sizeof(double), cudaMemcpyDeviceToHost);
        out = *c->h_dot1;
    } else {
        cudaMemcpy(&out, c->d_dot, sizeof(double), cudaMemcpyDeviceToHost);
    }
    return out;
}

// Ap = S p (grounded): tw = Dw^{-1} (B p); Ap = Df.p - B' tw; Ap[ground]=0
void apply_S(AkmCudaContext* c, const double* p, double* Ap) {
    const int gw = (c->N + kBlock - 1) / kBlock;
    const int gf = (c->J + kBlock - 1) / kBlock;
    k_mult_b<<<gw, kBlock>>>(c->N, c->d_w_ptr, c->d_m_f, c->d_m_c, p, c->d_inv_dw,
                             c->d_tw);
    k_mult_bt_finish<<<gf, kBlock>>>(c->J, c->ground, c->d_f_ptr, c->d_f_matches,
                                     c->d_m_w, c->d_m_c, c->d_tw, c->d_Df, p, Ap);
}

}  // namespace

AkmCudaContext* akm_cuda_create(int n_workers, int n_firms, const int* m_w,
                                const int* m_f, const double* m_c,
                                std::size_t n_matches, const std::int64_t* w_ptr,
                                const std::int64_t* f_ptr, const std::int64_t* f_matches,
                                const double* Dw, const double* Df,
                                const double* inv_diagS) {
    if (!akm_cuda_available()) return nullptr;
    AkmCudaContext* c = new (std::nothrow) AkmCudaContext();
    if (c == nullptr) return nullptr;
    c->N = n_workers;
    c->J = n_firms;
    c->ground = n_firms - 1;
    c->M = n_matches;
    // host inverse of Dw for the fused kernel
    double* inv_dw_host = new (std::nothrow) double[n_workers];
    if (inv_dw_host == nullptr) {
        delete c;
        return nullptr;
    }
    for (int w = 0; w < n_workers; ++w) inv_dw_host[w] = 1.0 / Dw[w];
    bool ok = upload(&c->d_w_ptr, w_ptr, static_cast<std::size_t>(n_workers) + 1) &&
              upload(&c->d_f_ptr, f_ptr, static_cast<std::size_t>(n_firms) + 1) &&
              upload(&c->d_f_matches, f_matches, n_matches) &&
              upload(&c->d_m_w, m_w, n_matches) &&
              upload(&c->d_m_f, m_f, n_matches) &&
              upload(&c->d_m_c, m_c, n_matches) &&
              upload(&c->d_inv_dw, inv_dw_host, static_cast<std::size_t>(n_workers)) &&
              upload(&c->d_Df, Df, static_cast<std::size_t>(n_firms)) &&
              upload(&c->d_inv_diag, inv_diagS, static_cast<std::size_t>(n_firms));
    delete[] inv_dw_host;
    const std::size_t J = static_cast<std::size_t>(n_firms);
    ok = ok && cuda_ok(cudaMalloc(&c->d_x, sizeof(double) * J)) &&
         cuda_ok(cudaMalloc(&c->d_r, sizeof(double) * J)) &&
         cuda_ok(cudaMalloc(&c->d_z, sizeof(double) * J)) &&
         cuda_ok(cudaMalloc(&c->d_p, sizeof(double) * J)) &&
         cuda_ok(cudaMalloc(&c->d_Ap, sizeof(double) * J)) &&
         cuda_ok(cudaMalloc(&c->d_tw, sizeof(double) * static_cast<std::size_t>(n_workers))) &&
         cuda_ok(cudaMalloc(&c->d_pair, sizeof(double) * J)) &&
         cuda_ok(cudaMalloc(&c->d_dot, sizeof(double)));
    if (!ok) {
        akm_cuda_destroy(c);
        return nullptr;
    }
    if (cudaMallocHost(&c->h_dot1, sizeof(double)) != cudaSuccess) {
        c->h_dot1 = nullptr;  // optional; pageable fallback
    }
    return c;
}

void akm_cuda_destroy(AkmCudaContext* ctx) { delete ctx; }
int akm_cuda_max_lanes() { return 32; }

namespace {

// Per-lane dots via one segmented reduce. The device-only primitive lets two
// independent reductions be queued before a single host synchronization.
bool seg_dots_device(AkmCudaContext* c, int nb, const double* a,
                     const double* b, double* out_device) {
    const int J = c->J;
    dim3 grid((J + kBlock - 1) / kBlock, nb);
    k_mul_pair_multi<<<grid, kBlock>>>(J, nb, a, b, c->d_mactive, c->d_mpair);
    std::size_t need = 0;
    cub::DeviceSegmentedReduce::Sum(nullptr, need, c->d_mpair, out_device, nb,
                                    c->d_moffs, c->d_moffs + 1);
    if (need > c->mcub_tmp_bytes) {
        cudaFree(c->d_mcub_tmp);
        if (!cuda_ok(cudaMalloc(&c->d_mcub_tmp, need))) return false;
        c->mcub_tmp_bytes = need;
    }
    cub::DeviceSegmentedReduce::Sum(c->d_mcub_tmp, c->mcub_tmp_bytes, c->d_mpair,
                                    out_device, nb, c->d_moffs, c->d_moffs + 1);
    return cuda_ok(cudaGetLastError());
}

bool seg_dots(AkmCudaContext* c, int nb, const double* a, const double* b,
              double* out_host) {
    if (!seg_dots_device(c, nb, a, b, c->d_mdots)) return false;
    double* dst = (c->h_dots != nullptr) ? c->h_dots : out_host;
    if (!cuda_ok(cudaMemcpy(dst, c->d_mdots, sizeof(double) * nb,
                            cudaMemcpyDeviceToHost)))
        return false;
    if (dst != out_host) std::memcpy(out_host, dst, sizeof(double) * nb);
    return true;
}

bool seg_dots_pair(AkmCudaContext* c, int nb,
                   const double* a0, const double* b0,
                   const double* a1, const double* b1,
                   double* out_host) {
    if (!seg_dots_device(c, nb, a0, b0, c->d_mdots)) return false;
    if (!seg_dots_device(c, nb, a1, b1, c->d_mdots + nb)) return false;
    double* dst = (c->h_dots != nullptr) ? c->h_dots : out_host;
    if (!cuda_ok(cudaMemcpy(dst, c->d_mdots, sizeof(double) * 2 * nb,
                            cudaMemcpyDeviceToHost)))
        return false;
    if (dst != out_host) std::memcpy(out_host, dst, sizeof(double) * 2 * nb);
    return true;
}

bool mr_ensure(AkmCudaContext* c, int nb) {
    if (c->mr_lanes >= nb) return true;
    const std::size_t J = static_cast<std::size_t>(c->J);
    const std::size_t N = static_cast<std::size_t>(c->N);
    const std::size_t L = static_cast<std::size_t>(nb);
    // free-and-null before reallocating: cudaMalloc leaves *ptr untouched on
    // failure, so a stale (already freed) pointer would otherwise survive a
    // partial OOM and be double-freed by the destructor.
    cudaFree(c->d_mx); c->d_mx = nullptr;
    cudaFree(c->d_mr); c->d_mr = nullptr;
    cudaFree(c->d_mz); c->d_mz = nullptr;
    cudaFree(c->d_mp); c->d_mp = nullptr;
    cudaFree(c->d_mAp); c->d_mAp = nullptr;
    cudaFree(c->d_mtw); c->d_mtw = nullptr;
    cudaFree(c->d_mpair); c->d_mpair = nullptr;
    cudaFree(c->d_mdots); c->d_mdots = nullptr;
    cudaFree(c->d_mscal); c->d_mscal = nullptr;
    cudaFree(c->d_mactive); c->d_mactive = nullptr;
    cudaFree(c->d_moffs); c->d_moffs = nullptr;
    cudaFreeHost(c->h_pack); c->h_pack = nullptr;
    cudaFreeHost(c->h_dots); c->h_dots = nullptr;
    cudaFreeHost(c->h_scal); c->h_scal = nullptr;
    cudaFreeHost(c->h_active); c->h_active = nullptr;
    bool ok = cuda_ok(cudaMalloc(&c->d_mx, sizeof(double) * J * L)) &&
              cuda_ok(cudaMalloc(&c->d_mr, sizeof(double) * J * L)) &&
              cuda_ok(cudaMalloc(&c->d_mz, sizeof(double) * J * L)) &&
              cuda_ok(cudaMalloc(&c->d_mp, sizeof(double) * J * L)) &&
              cuda_ok(cudaMalloc(&c->d_mAp, sizeof(double) * J * L)) &&
              cuda_ok(cudaMalloc(&c->d_mtw, sizeof(double) * N * L)) &&
              cuda_ok(cudaMalloc(&c->d_mpair, sizeof(double) * J * L)) &&
              cuda_ok(cudaMalloc(&c->d_mdots, sizeof(double) * 2 * L)) &&
              cuda_ok(cudaMalloc(&c->d_mscal, sizeof(double) * L)) &&
              cuda_ok(cudaMalloc(&c->d_mactive, L)) &&
              cuda_ok(cudaMalloc(&c->d_moffs, sizeof(int) * (L + 1)));
    if (!ok) {
        c->mr_lanes = 0;
        return false;
    }
    // pinned host staging is best-effort: on failure the pointers stay null
    // and the callers use pageable transfers (values are identical).
    if (cudaMallocHost(&c->h_pack, sizeof(double) * J * L) != cudaSuccess) {
        c->h_pack = nullptr;
    }
    if (cudaMallocHost(&c->h_dots, sizeof(double) * 2 * L) != cudaSuccess) {
        c->h_dots = nullptr;
    }
    if (cudaMallocHost(&c->h_scal, sizeof(double) * L) != cudaSuccess) {
        c->h_scal = nullptr;
    }
    if (cudaMallocHost(&c->h_active, L) != cudaSuccess) {
        c->h_active = nullptr;
    }
    c->mr_lanes = nb;
    return true;
}

}  // namespace

double* akm_cuda_pack_buffer(AkmCudaContext* c, int nb) {
    if (c == nullptr || nb < 1 || nb > akm_cuda_max_lanes()) return nullptr;
    if (!mr_ensure(c, nb)) return nullptr;
    return c->h_pack;  // may be null: pinned staging is best-effort
}

int akm_cuda_solve_S_multi(AkmCudaContext* c, const double* rhs_pack,
                           double* z_pack, int nb, double tol, int max_iter,
                           int* iters_out) {
    if (nb < 1 || nb > akm_cuda_max_lanes()) return -1;
    if (!mr_ensure(c, nb)) return -1;
    const int J = c->J;
    const std::size_t JL = static_cast<std::size_t>(J) * nb;
    dim3 gJ((J + kBlock - 1) / kBlock, nb);
    dim3 gN((c->N + kBlock - 1) / kBlock, nb);
    // pinned staging helpers for the per-step scalar traffic (identical
    // values; only the transfer path changes)
    const auto up_active = [c, nb](const char* v) {
        const char* src = v;
        if (c->h_active != nullptr) {
            std::memcpy(c->h_active, v, static_cast<std::size_t>(nb));
            src = c->h_active;
        }
        cudaMemcpy(c->d_mactive, src, static_cast<std::size_t>(nb),
                   cudaMemcpyHostToDevice);
    };
    const auto up_scal = [c, nb](const double* v) {
        const double* src = v;
        if (c->h_scal != nullptr) {
            std::memcpy(c->h_scal, v, sizeof(double) * nb);
            src = c->h_scal;
        }
        cudaMemcpy(c->d_mscal, src, sizeof(double) * nb, cudaMemcpyHostToDevice);
    };
    {
        const int go = (nb + 1 + kBlock - 1) / kBlock;
        k_seg_offsets<<<go, kBlock>>>(nb, J, c->d_moffs);
    }
    if (!cuda_ok(cudaMemcpy(c->d_mr, rhs_pack, sizeof(double) * JL,
                            cudaMemcpyHostToDevice)))
        return -1;
    // pin grounds; x = 0; all lanes active
    cudaMemset(c->d_mx, 0, sizeof(double) * JL);
    {
        std::vector<char> act(nb, 1);
        for (int l = 0; l < nb; ++l) {
            cudaMemset(c->d_mr + static_cast<std::size_t>(l) * J + c->ground, 0,
                       sizeof(double));
        }
        up_active(act.data());
    }
    std::vector<double> rhs_norm2(nb), rz(nb), tol2(nb), pAp(nb);
    std::vector<double> rr_rz(static_cast<std::size_t>(2 * nb));
    std::vector<double> alpha(nb), beta(nb);
    std::vector<char> active(nb, 1), ok_lane(nb, 0);
    std::vector<int> its(nb, 0);
    if (!seg_dots(c, nb, c->d_mr, c->d_mr, rhs_norm2.data())) return -1;
    for (int l = 0; l < nb; ++l) {
        if (rhs_norm2[l] == 0.0) {
            active[l] = 0;
            ok_lane[l] = 1;
        }
        tol2[l] = tol * tol * rhs_norm2[l];
    }
    up_active(active.data());
    k_hadamard_multi<<<gJ, kBlock>>>(J, nb, c->d_inv_diag, c->d_mactive, c->d_mr,
                                     c->d_mz);
    cudaMemcpy(c->d_mp, c->d_mz, sizeof(double) * JL, cudaMemcpyDeviceToDevice);
    if (!seg_dots(c, nb, c->d_mr, c->d_mz, rz.data())) return -1;

    auto any_active = [&]() {
        for (int l = 0; l < nb; ++l)
            if (active[l]) return true;
        return false;
    };
    while (any_active()) {
        // Ap = S p for active lanes
        k_mult_b_multi<<<gN, kBlock>>>(c->N, nb, c->d_w_ptr, c->d_m_f, c->d_m_c,
                                       c->d_mp, c->d_inv_dw, c->d_mactive, c->d_mtw,
                                       J, c->N);
        k_mult_bt_finish_multi<<<gJ, kBlock>>>(J, nb, c->ground, c->d_f_ptr,
                                               c->d_f_matches, c->d_m_w, c->d_m_c,
                                               c->d_mtw, c->d_Df, c->d_mp, c->d_mAp,
                                               c->d_mactive, J, c->N);
        if (!seg_dots(c, nb, c->d_mp, c->d_mAp, pAp.data())) return -1;
        bool active_changed = false;
        for (int l = 0; l < nb; ++l) {
            if (!active[l]) {
                alpha[l] = 0.0;
                continue;
            }
            if (!(pAp[l] > 0.0) || !(pAp[l] < 1e300)) {
                active[l] = 0;
                active_changed = true;
                alpha[l] = 0.0;
                continue;
            }
            alpha[l] = rz[l] / pAp[l];
        }
        // In the normal path the mask cannot change between consecutive
        // residual checks, so avoid an otherwise redundant H2D transfer.
        if (active_changed) up_active(active.data());
        up_scal(alpha.data());
        k_cg_update_multi<<<gJ, kBlock>>>(J, nb, c->d_mscal, c->d_mactive,
                                          c->d_mp, c->d_mAp, c->d_mx, c->d_mr);
        // Precondition first, then queue rr and r'z reductions back-to-back.
        // Both CUB reductions retain their legacy shapes/order, but a single
        // D2H transfer/synchronization returns both scalar vectors.
        k_hadamard_multi<<<gJ, kBlock>>>(J, nb, c->d_inv_diag, c->d_mactive,
                                         c->d_mr, c->d_mz);
        if (!seg_dots_pair(c, nb, c->d_mr, c->d_mr,
                           c->d_mr, c->d_mz, rr_rz.data()))
            return -1;
        for (int l = 0; l < nb; ++l) {
            if (!active[l]) continue;
            ++its[l];
            if (rr_rz[static_cast<std::size_t>(l)] <= tol2[l]) {
                ok_lane[l] = 1;
                active[l] = 0;
            } else if (its[l] >= max_iter) {
                active[l] = 0;
            }
        }
        up_active(active.data());
        if (!any_active()) break;
        for (int l = 0; l < nb; ++l) {
            const double rz_new = rr_rz[static_cast<std::size_t>(nb + l)];
            beta[l] = (active[l] && rz[l] != 0.0) ? rz_new / rz[l] : 0.0;
            if (active[l]) rz[l] = rz_new;
        }
        up_scal(beta.data());
        k_xpay_multi<<<gJ, kBlock>>>(J, nb, c->d_mscal, c->d_mactive, c->d_mz,
                                     c->d_mp);
    }
    if (!cuda_ok(cudaMemcpy(z_pack, c->d_mx, sizeof(double) * JL,
                            cudaMemcpyDeviceToHost)))
        return -1;
    for (int l = 0; l < nb; ++l) {
        z_pack[static_cast<std::size_t>(l) * J + c->ground] = 0.0;
        if (!ok_lane[l]) {
            // zero the failed lane's solution (caller marks failure)
            for (int j = 0; j < J; ++j)
                z_pack[static_cast<std::size_t>(l) * J + j] = 0.0;
            iters_out[l] = -1;
        } else {
            iters_out[l] = its[l];
        }
    }
    return 0;
}


int akm_cuda_solve_S(AkmCudaContext* c, const double* rhs, double* z, double tol,
                     int max_iter) {
    const int J = c->J;
    const int gf = (J + kBlock - 1) / kBlock;
    if (!cuda_ok(cudaMemcpy(c->d_r, rhs, sizeof(double) * J, cudaMemcpyHostToDevice)))
        return -1;
    // pin ground of r; x = 0
    cudaMemset(c->d_r + c->ground, 0, sizeof(double));
    cudaMemset(c->d_x, 0, sizeof(double) * J);
    const double rhs_norm2 = device_dot(c, c->d_r, c->d_r, J);
    if (rhs_norm2 == 0.0) {
        for (int f = 0; f < J; ++f) z[f] = 0.0;
        return 0;
    }
    const double tol2 = tol * tol * rhs_norm2;
    k_hadamard<<<gf, kBlock>>>(J, c->d_inv_diag, c->d_r, c->d_z);
    cudaMemcpy(c->d_p, c->d_z, sizeof(double) * J, cudaMemcpyDeviceToDevice);
    double rz = device_dot(c, c->d_r, c->d_z, J);
    int it = 0;
    bool converged = false;
    for (; it < max_iter; ++it) {
        apply_S(c, c->d_p, c->d_Ap);
        const double pAp = device_dot(c, c->d_p, c->d_Ap, J);
        if (!(pAp > 0.0) || !(pAp < 1e300)) break;
        const double a = rz / pAp;
        k_axpy<<<gf, kBlock>>>(J, a, c->d_p, c->d_x);
        k_axpy<<<gf, kBlock>>>(J, -a, c->d_Ap, c->d_r);
        const double rr = device_dot(c, c->d_r, c->d_r, J);
        if (rr <= tol2) {
            ++it;
            converged = true;
            break;
        }
        k_hadamard<<<gf, kBlock>>>(J, c->d_inv_diag, c->d_r, c->d_z);
        const double rz_new = device_dot(c, c->d_r, c->d_z, J);
        k_xpay<<<gf, kBlock>>>(J, rz_new / rz, c->d_z, c->d_p);
        rz = rz_new;
    }
    if (!converged) {
        const double rr = device_dot(c, c->d_r, c->d_r, J);
        if (rr > tol2) return -1;
    }
    if (!cuda_ok(cudaMemcpy(z, c->d_x, sizeof(double) * J, cudaMemcpyDeviceToHost)))
        return -1;
    z[c->ground] = 0.0;
    return it;
}

}  // namespace akm
}  // namespace hdfe
