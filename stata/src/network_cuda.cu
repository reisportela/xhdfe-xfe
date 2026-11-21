#include "network_device.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#if defined(__linux__)
#include <dlfcn.h>
#include <mutex>
#include <string>
#endif

namespace {
void check(cudaError_t s) {
    if (s != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(s));
}
void check(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("mixed network cuBLAS operation failed");
}
void check(cusparseStatus_t s) {
    if (s != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error("mixed network cuSPARSE operation failed");
}

// The cuBLAS and cuSPARSE functions of the network route, under the names the
// libraries export (the _v2 names behind the cublas_v2.h macros); the headers
// still give their types. On Linux they are resolved at the first call of
// solve_network_device, once per process, so the hybrid artefacts do not list
// the two libraries as needed and a CPU fit never loads them. Elsewhere they
// are the linked functions, as before.
struct MathLibraries {
    decltype(&::cublasCreate_v2) cublasCreate_v2 = nullptr;
    decltype(&::cublasDestroy_v2) cublasDestroy_v2 = nullptr;
    decltype(&::cublasSetStream_v2) cublasSetStream_v2 = nullptr;
    decltype(&::cublasSetMathMode) cublasSetMathMode = nullptr;
    decltype(&::cublasDgemm_v2) cublasDgemm_v2 = nullptr;
    decltype(&::cusparseCreate) cusparseCreate = nullptr;
    decltype(&::cusparseDestroy) cusparseDestroy = nullptr;
    decltype(&::cusparseSetStream) cusparseSetStream = nullptr;
    decltype(&::cusparseCreateCsr) cusparseCreateCsr = nullptr;
    decltype(&::cusparseCreateDnMat) cusparseCreateDnMat = nullptr;
    decltype(&::cusparseDestroySpMat) cusparseDestroySpMat = nullptr;
    decltype(&::cusparseDestroyDnMat) cusparseDestroyDnMat = nullptr;
    decltype(&::cusparseSpMM_bufferSize) cusparseSpMM_bufferSize = nullptr;
    decltype(&::cusparseSpMM) cusparseSpMM = nullptr;
};

#if defined(__linux__)
#define XHDFE_NETWORK_SONAME_(name, major) name ".so." #major
#define XHDFE_NETWORK_SONAME(name, major) XHDFE_NETWORK_SONAME_(name, major)
// The sonames of the major versions this file is compiled against, which are
// the ones the build linked before. The libraries are found the way the linked
// ones were: the caller's run path, LD_LIBRARY_PATH, the loader cache. They
// stay open for the life of the process (never dlclose), so the functions
// outlive every handle and stream made from them.
constexpr const char *kCublasLibrary = XHDFE_NETWORK_SONAME("libcublas", CUBLAS_VER_MAJOR);
constexpr const char *kCusparseLibrary = XHDFE_NETWORK_SONAME("libcusparse", CUSPARSE_VER_MAJOR);

void *open_library(const char *name, std::string &why) {
    dlerror();
    void *library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        const char *error = dlerror();
        why = error != nullptr ? error : "unknown dlopen error";
    }
    return library;
}

template <class F>
void take(void *library, const char *symbol, F &slot, std::string &missing) {
    dlerror();
    void *address = dlsym(library, symbol);
    if (address == nullptr) {
        missing += missing.empty() ? "" : ", ";
        missing += symbol;
        return;
    }
    slot = reinterpret_cast<F>(address);
}

// All fourteen functions at once; an empty string when every one was found.
std::string load_libraries(MathLibraries &api) {
    std::string why;
    void *blas = open_library(kCublasLibrary, why);
    if (blas == nullptr)
        return std::string("mixed network: cannot open ") + kCublasLibrary + ": " + why;
    void *sparse = open_library(kCusparseLibrary, why);
    if (sparse == nullptr) {
        // cuSPARSE needs nvJitLink of the same CUDA major, which a toolkit may
        // keep in another directory (the HPC SDK keeps it beside cudart, not
        // beside cuSPARSE): open it through the caller's search path, then
        // cuSPARSE once more. The first error is the one reported.
        std::string ignored;
        const std::string jit = "libnvJitLink.so." + std::to_string(CUDART_VERSION / 1000);
        if (open_library(jit.c_str(), ignored) != nullptr)
            sparse = open_library(kCusparseLibrary, ignored);
        if (sparse == nullptr)
            return std::string("mixed network: cannot open ") + kCusparseLibrary + ": " + why;
    }
    std::string missing_blas, missing_sparse;
    take(blas, "cublasCreate_v2", api.cublasCreate_v2, missing_blas);
    take(blas, "cublasDestroy_v2", api.cublasDestroy_v2, missing_blas);
    take(blas, "cublasSetStream_v2", api.cublasSetStream_v2, missing_blas);
    take(blas, "cublasSetMathMode", api.cublasSetMathMode, missing_blas);
    take(blas, "cublasDgemm_v2", api.cublasDgemm_v2, missing_blas);
    take(sparse, "cusparseCreate", api.cusparseCreate, missing_sparse);
    take(sparse, "cusparseDestroy", api.cusparseDestroy, missing_sparse);
    take(sparse, "cusparseSetStream", api.cusparseSetStream, missing_sparse);
    take(sparse, "cusparseCreateCsr", api.cusparseCreateCsr, missing_sparse);
    take(sparse, "cusparseCreateDnMat", api.cusparseCreateDnMat, missing_sparse);
    take(sparse, "cusparseDestroySpMat", api.cusparseDestroySpMat, missing_sparse);
    take(sparse, "cusparseDestroyDnMat", api.cusparseDestroyDnMat, missing_sparse);
    take(sparse, "cusparseSpMM_bufferSize", api.cusparseSpMM_bufferSize, missing_sparse);
    take(sparse, "cusparseSpMM", api.cusparseSpMM, missing_sparse);
    if (!missing_blas.empty())
        return std::string("mixed network: ") + kCublasLibrary + " lacks " + missing_blas;
    if (!missing_sparse.empty())
        return std::string("mixed network: ") + kCusparseLibrary + " lacks " + missing_sparse;
    return std::string();
}

// Resolved once per process under std::call_once; every later call, from any
// thread, reads the same table or throws the same error as the first, the way
// a failed cuBLAS call throws (check above).
const MathLibraries &math_libraries() {
    static std::once_flag once;
    static MathLibraries api;
    static std::string error;
    std::call_once(once, [] { error = load_libraries(api); });
    if (!error.empty())
        throw std::runtime_error(error);
    return api;
}
#else
const MathLibraries &math_libraries() {
    static const MathLibraries api{
        &::cublasCreate_v2,      &::cublasDestroy_v2,      &::cublasSetStream_v2,
        &::cublasSetMathMode,    &::cublasDgemm_v2,        &::cusparseCreate,
        &::cusparseDestroy,      &::cusparseSetStream,     &::cusparseCreateCsr,
        &::cusparseCreateDnMat,  &::cusparseDestroySpMat,  &::cusparseDestroyDnMat,
        &::cusparseSpMM_bufferSize, &::cusparseSpMM};
    return api;
}
#endif

template <class T> struct Buffer {
    T *p = nullptr;
    size_t n;
    explicit Buffer(size_t size, const T *host = nullptr) : n(size) {
        if (n) {
            check(cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T)));
            if (host) {
                auto rc = cudaMemcpy(p, host, n * sizeof(T), cudaMemcpyHostToDevice);
                if (rc != cudaSuccess) {
                    cudaFree(p);
                    p = nullptr;
                    check(rc);
                }
            }
        }
    }
    Buffer(const Buffer &) = delete;
    ~Buffer() {
        if (p)
            cudaFree(p);
    }
};
struct Handles {
    const MathLibraries &api;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    cusparseHandle_t sparse = nullptr;
    cusparseSpMatDescr_t h = nullptr;
    cusparseDnMatDescr_t v = nullptr, q = nullptr;
    explicit Handles(const MathLibraries &libraries) : api(libraries) {}
    ~Handles() {
        if (q)
            api.cusparseDestroyDnMat(q);
        if (v)
            api.cusparseDestroyDnMat(v);
        if (h)
            api.cusparseDestroySpMat(h);
        if (sparse)
            api.cusparseDestroy(sparse);
        if (blas)
            api.cublasDestroy_v2(blas);
        if (stream)
            cudaStreamDestroy(stream);
    }
};
constexpr int block_size = 256;
__global__ void local_solve(int firms, int k, const int *ix, const double *inv, const double *r,
                            double *z) {
    for (long long j = blockIdx.x * blockDim.x + threadIdx.x; j < (long long)firms * k;
         j += (long long)blockDim.x * gridDim.x) {
        const int f = j / k, c = j % k;
        const int *a = ix + 3LL * f;
        const double *m = inv + 9LL * f;
        double v[3];
        for (int t = 0; t < 3; ++t)
            v[t] = a[t] < 0 ? 0. : r[(long long)a[t] * k + c];
        for (int t = 0; t < 3; ++t)
            if (a[t] >= 0)
                z[(long long)a[t] * k + c] =
                    m[3 * t] * v[0] + m[3 * t + 1] * v[1] + m[3 * t + 2] * v[2];
    }
}

__device__ double block_sum(double x) {
    __shared__ double partial[8];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int s = 16; s; s >>= 1)
        x += __shfl_down_sync(0xffffffff, x, s);
    if (lane == 0)
        partial[warp] = x;
    __syncthreads();
    x = warp == 0 && lane < 8 ? partial[lane] : 0.;
    if (warp == 0)
        for (int s = 16; s; s >>= 1)
            x += __shfl_down_sync(0xffffffff, x, s);
    return x;
}
__global__ void dot_parts(const double *a, const double *b, int n, int k, double *p) {
    const int c = blockIdx.y;
    double sum = 0.;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        auto pos = (long long)i * k + c;
        sum += a[pos] * b[pos];
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0)
        p[(long long)c * gridDim.x + blockIdx.x] = sum;
}
__global__ void dot_finish(const double *p, int blocks, double *out) {
    int c = blockIdx.x;
    double sum = 0.;
    for (int i = threadIdx.x; i < blocks; i += blockDim.x)
        sum += p[c * blocks + i];
    sum = block_sum(sum);
    if (threadIdx.x == 0)
        out[c] = sum;
}
__global__ void ratio(const double *a, const double *b, int k, double *result, int *live) {
    int c = threadIdx.x;
    if (c < k) {
        double val = (a[c] > 0. && b[c] > 0.) ? a[c] / b[c] : 0.;
        if (!isfinite(val) || !(val > 0.))
            live[c] = 0;
        result[c] = live[c] ? val : 0.;
    }
}
__global__ void take_step(long long count, int k, const double *alpha, const double *p,
                          const double *q, double *x, double *r) {
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (long long)gridDim.x * blockDim.x) {
        double a = alpha[i % k];
        x[i] += a * p[i];
        r[i] -= a * q[i];
    }
}
__global__ void direction(long long count, int k, const int *live, const double *beta,
                          const double *z, double *p) {
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (long long)gridDim.x * blockDim.x) {
        int c = i % k;
        p[i] = live[c] ? z[i] + beta[c] * p[i] : 0.;
    }
}
} // namespace

int solve_network_device(const NetworkDeviceInput &d, void *ctx, NetworkProbe probe,
                         NetworkAccept accept) {
    const int n = d.n, k = d.rhs, t = d.periods;
    const size_t width = size_t(n) * k;
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total));
    const size_t required = width * sizeof(double) * 6 + size_t(d.nnz) * 12 +
                            size_t(n) * t * sizeof(double) + size_t(d.firms) * 84;
    if (required > free - free / 8)
        return 0;
    const MathLibraries &api = math_libraries();
    Handles e(api);
    check(cudaStreamCreateWithFlags(&e.stream, cudaStreamNonBlocking));
    check(api.cublasCreate_v2(&e.blas));
    check(api.cublasSetStream_v2(e.blas, e.stream));
    check(api.cublasSetMathMode(e.blas, CUBLAS_PEDANTIC_MATH));
    check(api.cusparseCreate(&e.sparse));
    check(api.cusparseSetStream(e.sparse, e.stream));
    Buffer<int> ptr(n + 1, d.ptr), col(d.nnz, d.col), ix(size_t(d.firms) * 3, d.index);
    Buffer<double> val(d.nnz, d.values), K(size_t(n) * t, d.border),
        inv(size_t(d.firms) * 9, d.inverse);
    Buffer<double> state(width), r(width, d.b), p(width), q(width), z(width), temp(size_t(t) * k);
    const int grid = std::max(1, std::min(2048, (n + block_size - 1) / block_size));
    Buffer<double> part(size_t(grid) * k), rho0(k), rho1(k), denom(k), step(k);
    Buffer<int> live(k);
    check(cudaMemsetAsync(state.p, 0, width * sizeof(double), e.stream));
    check(cudaMemsetAsync(live.p, 1, k * sizeof(int), e.stream));
    check(api.cusparseCreateCsr(&e.h, n, n, d.nnz, ptr.p, col.p, val.p, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    check(api.cusparseCreateDnMat(&e.v, n, k, k, p.p, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    check(api.cusparseCreateDnMat(&e.q, n, k, k, q.p, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    const double one = 1., zero = 0., neg = -1.;
    size_t bytes = 0;
    auto sparse_alg = CUSPARSE_SPMM_CSR_ALG2;
    check(api.cusparseSpMM_bufferSize(e.sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, &one, e.h, e.v, &one, e.q,
                                  CUDA_R_64F, sparse_alg, &bytes));
    Buffer<unsigned char> buffer(bytes);
    auto inner = [&](const double *a, const double *b, double *out) {
        dot_parts<<<dim3(grid, k), block_size, 0, e.stream>>>(a, b, n, k, part.p);
        dot_finish<<<k, block_size, 0, e.stream>>>(part.p, grid, out);
    };
    auto prec = [&]() {
        local_solve<<<grid, block_size, 0, e.stream>>>(d.firms, k, ix.p, inv.p, r.p, z.p);
    };
    prec();
    check(cudaMemcpyAsync(p.p, z.p, width * sizeof(double), cudaMemcpyDeviceToDevice, e.stream));
    double *rho = rho0.p;
    double *next = rho1.p;
    inner(r.p, z.p, rho);
    std::vector<double> host(width);
    std::vector<int> host_live(k);
    for (int it = 1; it <= d.budget; ++it) {
        check(api.cublasDgemm_v2(e.blas, CUBLAS_OP_N, CUBLAS_OP_T, k, t, n, &one, p.p, k, K.p, t, &zero,
                          temp.p, k));
        check(api.cublasDgemm_v2(e.blas, CUBLAS_OP_N, CUBLAS_OP_N, k, n, t, &neg, temp.p, k, K.p, t, &zero,
                          q.p, k));
        check(api.cusparseSpMM(e.sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_NON_TRANSPOSE, &one, e.h, e.v, &one, e.q, CUDA_R_64F,
                           sparse_alg, buffer.p));
        inner(p.p, q.p, denom.p);
        ratio<<<1, 64, 0, e.stream>>>(rho, denom.p, k, step.p, live.p);
        take_step<<<grid, block_size, 0, e.stream>>>(width, k, step.p, p.p, q.p, state.p, r.p);
        if (it % 16 == 0 || it == d.budget) {
            check(cudaMemcpyAsync(host.data(), r.p, width * sizeof(double), cudaMemcpyDeviceToHost,
                                  e.stream));
            check(cudaMemcpyAsync(host_live.data(), live.p, k * sizeof(int), cudaMemcpyDeviceToHost,
                                  e.stream));
            check(cudaStreamSynchronize(e.stream));
            const bool exhausted = std::none_of(host_live.begin(), host_live.end(),
                                                [](int lane) { return lane != 0; });
            if (exhausted || probe(ctx, it, host.data())) {
                check(cudaMemcpyAsync(host.data(), state.p, width * sizeof(double),
                                      cudaMemcpyDeviceToHost, e.stream));
                check(cudaStreamSynchronize(e.stream));
                if (accept(ctx, it, host.data()))
                    return it;
            }
            if (exhausted)
                return -it;
        }
        prec();
        inner(r.p, z.p, next);
        ratio<<<1, 64, 0, e.stream>>>(next, rho, k, step.p, live.p);
        direction<<<grid, block_size, 0, e.stream>>>(width, k, live.p, step.p, z.p, p.p);
        std::swap(rho, next);
        check(cudaPeekAtLastError());
    }
    return -d.budget;
}
