// CUDA backend for the AKM/KSS two-way solver: Jacobi-PCG on the grounded
// worker-partialled firm Laplacian S = Df - B' Dw^{-1} B, matrix-free over
// the match structure. Opt-in from hdfe::akm (use_gpu); the CPU path is the
// reference. Reductions use CUB device sums (run-to-run deterministic on a
// given device).

#include "hdfe/akm_kss_cuda.hpp"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <new>

namespace hdfe {
namespace akm {

namespace {

constexpr int kBlock = 256;

inline bool cuda_ok(cudaError_t st) { return st == cudaSuccess; }

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

}  // namespace

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
    cudaMemcpy(&out, c->d_dot, sizeof(double), cudaMemcpyDeviceToHost);
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
    return c;
}

void akm_cuda_destroy(AkmCudaContext* ctx) { delete ctx; }

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
