#include "fe_absorption_cuda.hpp"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace hdfe {
namespace detail {
namespace {

constexpr int kBlockSize = 256;

bool strict_residual_tolerance_mode(const HdfeOptions& options) {
    return options.tolerance_mode == ToleranceMode::StrictResidual;
}

double effective_absorption_tolerance(const HdfeOptions& options) {
    if (options.tol > 0.0 &&
        options.tolerance_mode == ToleranceMode::ReghdfeComparable) {
        return std::min(options.tol, 1.0e-9);
    }
    return options.tol;
}

double group_individual_absorption_tolerance(const HdfeOptions& options) {
    const double tol = effective_absorption_tolerance(options);
    if (tol > 0.0 &&
        options.tolerance_mode == ToleranceMode::ReghdfeComparable) {
        return std::min(tol, 1.0e-12);
    }
    return tol;
}

// Prototype knob (measurement phase): see fe_absorption.cpp counterpart.
bool honest_tol_trigger_enabled() {
    static const bool enabled = []() {
        const char* e = std::getenv("XHDFE_TOL_TRIGGER");
        return e != nullptr && (e[0] == 'c' || e[0] == 'C' || e[0] == '1');
    }();
    return enabled;
}

inline void cuda_check(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        const char* err = cudaGetErrorString(status);
        throw std::runtime_error(std::string(msg) + ": " + (err ? err : "unknown"));
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept { move_from(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            move_from(other);
        }
        return *this;
    }

    void allocate(std::size_t count) {
        if (count == 0) {
            reset();
            return;
        }
        if (ptr_ && count <= size_) {
            return;
        }
        reset();
        size_ = count;
        cuda_check(cudaMalloc(&ptr_, sizeof(T) * count), "cudaMalloc failed");
    }

    void reset() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return size_; }

private:
    void move_from(DeviceBuffer& other) {
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// Pinned host staging for large host<->device copies. Pageable cudaMemcpy of
// multi-GB buffers runs far below PCIe rate (the CUDA driver pipelines them
// through a small internal bounce buffer); a double-buffered pinned pipeline
// recovers most of the link bandwidth. Below kStagedCopyMinBytes the plain
// cudaMemcpy path is kept so small transfers are byte-for-byte unaffected.
// The staged copies move exactly the same bytes; results are bit-identical.
class PinnedStage {
public:
    ~PinnedStage() { reset(); }

    PinnedStage() = default;
    PinnedStage(const PinnedStage&) = delete;
    PinnedStage& operator=(const PinnedStage&) = delete;

    bool ensure(std::size_t chunk_bytes) {
        if (ptr_ && chunk_ >= chunk_bytes) {
            return true;
        }
        reset();
        void* p = nullptr;
        if (cudaMallocHost(&p, chunk_bytes * 2) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        ptr_ = static_cast<unsigned char*>(p);
        chunk_ = chunk_bytes;
        return true;
    }

    unsigned char* data(int idx) {
        return ptr_ + static_cast<std::size_t>(idx) * chunk_;
    }

    void reset() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
            chunk_ = 0;
        }
    }

private:
    unsigned char* ptr_ = nullptr;
    std::size_t chunk_ = 0;
};

constexpr std::size_t kStagedCopyMinBytes = 64ull << 20;
constexpr std::size_t kStagedChunkBytes = 32ull << 20;

void staged_memcpy_h2d(void* dst_dev,
                       const void* src_host,
                       std::size_t bytes,
                       PinnedStage& stage,
                       const char* msg) {
    if (bytes < kStagedCopyMinBytes || !stage.ensure(kStagedChunkBytes)) {
        cuda_check(cudaMemcpy(dst_dev, src_host, bytes, cudaMemcpyHostToDevice), msg);
        return;
    }
    cudaEvent_t ev[2];
    cuda_check(cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming),
               "staged h2d event create failed");
    if (cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming) != cudaSuccess) {
        (void)cudaEventDestroy(ev[0]);
        cuda_check(cudaMemcpy(dst_dev, src_host, bytes, cudaMemcpyHostToDevice), msg);
        return;
    }
    const unsigned char* src = static_cast<const unsigned char*>(src_host);
    unsigned char* dst = static_cast<unsigned char*>(dst_dev);
    std::size_t off = 0;
    int buf = 0;
    while (off < bytes) {
        const std::size_t len = std::min(kStagedChunkBytes, bytes - off);
        // Wait for this stage buffer's previous async upload to drain, then
        // refill it while the other buffer's upload is still in flight.
        cuda_check(cudaEventSynchronize(ev[buf]), "staged h2d event sync failed");
        std::memcpy(stage.data(buf), src + off, len);
        cuda_check(cudaMemcpyAsync(dst + off, stage.data(buf), len,
                                   cudaMemcpyHostToDevice),
                   msg);
        cuda_check(cudaEventRecord(ev[buf]), "staged h2d event record failed");
        off += len;
        buf ^= 1;
    }
    cuda_check(cudaEventSynchronize(ev[0]), "staged h2d final sync failed");
    cuda_check(cudaEventSynchronize(ev[1]), "staged h2d final sync failed");
    (void)cudaEventDestroy(ev[0]);
    (void)cudaEventDestroy(ev[1]);
}

void staged_memcpy_d2h(void* dst_host,
                       const void* src_dev,
                       std::size_t bytes,
                       PinnedStage& stage,
                       const char* msg) {
    if (bytes < kStagedCopyMinBytes || !stage.ensure(kStagedChunkBytes)) {
        cuda_check(cudaMemcpy(dst_host, src_dev, bytes, cudaMemcpyDeviceToHost), msg);
        return;
    }
    cudaEvent_t ev[2];
    cuda_check(cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming),
               "staged d2h event create failed");
    if (cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming) != cudaSuccess) {
        (void)cudaEventDestroy(ev[0]);
        cuda_check(cudaMemcpy(dst_host, src_dev, bytes, cudaMemcpyDeviceToHost), msg);
        return;
    }
    const unsigned char* src = static_cast<const unsigned char*>(src_dev);
    unsigned char* dst = static_cast<unsigned char*>(dst_host);
    std::size_t off = 0;
    int buf = 0;
    bool have_prev = false;
    int prev_buf = 0;
    std::size_t prev_off = 0;
    std::size_t prev_len = 0;
    while (off < bytes) {
        const std::size_t len = std::min(kStagedChunkBytes, bytes - off);
        cuda_check(cudaMemcpyAsync(stage.data(buf), src + off, len,
                                   cudaMemcpyDeviceToHost),
                   msg);
        cuda_check(cudaEventRecord(ev[buf]), "staged d2h event record failed");
        if (have_prev) {
            cuda_check(cudaEventSynchronize(ev[prev_buf]),
                       "staged d2h event sync failed");
            // Intentionally single-threaded: the destination arrays are
            // re-read by the whole post-absorption pipeline, and a parallel
            // first-touch here would fragment their transparent huge pages
            // (see PERF_PREP_APPLIED.md for the regression this caused).
            std::memcpy(dst + prev_off, stage.data(prev_buf), prev_len);
        }
        have_prev = true;
        prev_buf = buf;
        prev_off = off;
        prev_len = len;
        off += len;
        buf ^= 1;
    }
    cuda_check(cudaEventSynchronize(ev[prev_buf]), "staged d2h event sync failed");
    std::memcpy(dst + prev_off, stage.data(prev_buf), prev_len);
    (void)cudaEventDestroy(ev[0]);
    (void)cudaEventDestroy(ev[1]);
}

__global__ __launch_bounds__(256, 4) void accumulate_sums_atomic_kernel(const double* y,
                                              const double* X,
                                              int n,
                                              int cols,
                                              int ld,
                                              const int* gid,
                                              const double* weights,
                                              bool unit_weights,
                                              double* sum_y,
                                              double* sum_x,
                                              int groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int g = gid[i];
    const double w = unit_weights ? 1.0 : weights[i];
    atomicAdd(&sum_y[g], w * y[i]);
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        atomicAdd(&sum_x[static_cast<std::size_t>(j) * groups + g],
                  w * X[static_cast<std::size_t>(j) * ld + i]);
    }
}

__global__ __launch_bounds__(256, 4) void accumulate_slope_sums_atomic_kernel(
    const double* y,
    const double* X,
    int n,
    int cols,
    int ld,
    const int* gid,
    const double* z,
    const double* weights,
    bool unit_weights,
    double* sum_y,
    double* sum_zy,
    double* sum_x,
    double* sum_zx,
    int groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int g = gid[i];
    if (g < 0 || g >= groups) {
        return;
    }
    const double w = unit_weights ? 1.0 : weights[i];
    const double zi = z[i];
    const double yi = y[i];
    atomicAdd(&sum_y[g], w * yi);
    atomicAdd(&sum_zy[g], w * zi * yi);
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        const double xi = X[static_cast<std::size_t>(j) * ld + i];
        atomicAdd(&sum_x[static_cast<std::size_t>(j) * groups + g], w * xi);
        atomicAdd(&sum_zx[static_cast<std::size_t>(j) * groups + g], w * zi * xi);
    }
}

__global__ __launch_bounds__(256, 4) void accumulate_edge_sums_kernel(const double* y,
                                                                      const double* X,
                                                                      int n,
                                                                      int cols,
                                                                      int ld,
                                                                      int edge_stride,
                                                                      const int* edge_group,
                                                                      double* sum_y,
                                                                      double* sum_x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int e = edge_group[i];
    atomicAdd(&sum_y[e], y[i]);
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        atomicAdd(&sum_x[static_cast<std::size_t>(j) * edge_stride + e],
                  X[static_cast<std::size_t>(j) * ld + i]);
    }
}

__global__ __launch_bounds__(256, 4) void apply_edge_residuals_kernel(double* y,
                                                                      double* X,
                                                                      int n,
                                                                      int cols,
                                                                      int ld,
                                                                      int edge_stride,
                                                                      const int* edge_group,
                                                                      const int* edge_counts,
                                                                      const double* sum_y,
                                                                      const double* sum_x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int e = edge_group[i];
    const double inv = 1.0 / static_cast<double>(edge_counts[e]);
    y[i] -= sum_y[e] * inv;
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        X[static_cast<std::size_t>(j) * ld + i] -=
            sum_x[static_cast<std::size_t>(j) * edge_stride + e] * inv;
    }
}

// Accumulate group sums using warp-segmented reductions to cut down on atomics
// when observations are locally sorted by gid (common for panel data).
// Correctness does not rely on global sorting: repeated gid values that are not
// contiguous within a warp simply form multiple segments whose sums accumulate.
__global__ __launch_bounds__(256, 2) void accumulate_sums_segmented_kernel(const double* y,
                                                 const double* X,
                                                 int n,
                                                 int cols,
                                                 int ld,
                                                 const int* gid,
                                                 const double* weights,
                                                 bool unit_weights,
                                                 double* sum_y,
                                                 double* sum_x,
                                                 int groups) {
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
    static_assert(kBlockSize % kWarpSize == 0, "kBlockSize must be a multiple of warp size");

    using WarpReduce = cub::WarpReduce<double, kWarpSize>;
    __shared__ typename WarpReduce::TempStorage temp_storage[kWarpsPerBlock];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool in_range = (i < n);
    const int g = in_range ? gid[i] : -1;
    const double w = (!in_range) ? 0.0 : (unit_weights ? 1.0 : weights[i]);
    const double vy = w * (in_range ? y[i] : 0.0);

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    const int prev_g = __shfl_up_sync(0xFFFFFFFFu, g, 1);
    const int head = (lane == 0) || (g != prev_g);

    const double agg_y = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(vy, head);
    if (in_range && head) {
        atomicAdd(&sum_y[g], agg_y);
    }

#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        const double vx = w * (in_range ? X[static_cast<std::size_t>(j) * ld + i] : 0.0);
        const double agg_x = WarpReduce(temp_storage[warp_id]).HeadSegmentedSum(vx, head);
        if (in_range && head) {
            atomicAdd(&sum_x[static_cast<std::size_t>(j) * groups + g], agg_x);
        }
    }
}

// Privatized accumulation for low-cardinality FEs: each block holds a
// per-group shared-memory sum, with fast shared atomics, then flushes to
// global once per (block, group). Reduces global atomic contention by a
// factor of `blocks_n` compared with a plain global-atomic kernel.
// Required shared memory per block: groups * (1 + cols) * 8 bytes.
// Caller must ensure this fits in the dynamic shared-memory budget.
__global__ __launch_bounds__(256, 2) void accumulate_sums_privatized_kernel(
    const double* y,
    const double* X,
    int n,
    int cols,
    int ld,
    const int* gid,
    const double* weights,
    bool unit_weights,
    double* sum_y,
    double* sum_x,
    int groups) {
    extern __shared__ double shared_sums[];
    double* shared_y = shared_sums;
    double* shared_x = shared_sums + groups;
    const int total_x = groups * cols;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    for (int g = tid; g < groups; g += block_size) {
        shared_y[g] = 0.0;
    }
    for (int k = tid; k < total_x; k += block_size) {
        shared_x[k] = 0.0;
    }
    __syncthreads();

    const int i = blockIdx.x * block_size + tid;
    if (i < n) {
        const int g = gid[i];
        const double w = unit_weights ? 1.0 : weights[i];
        atomicAdd(&shared_y[g], w * y[i]);
#pragma unroll 4
        for (int j = 0; j < cols; ++j) {
            atomicAdd(&shared_x[static_cast<std::size_t>(j) * groups + g],
                      w * X[static_cast<std::size_t>(j) * ld + i]);
        }
    }
    __syncthreads();

    for (int g = tid; g < groups; g += block_size) {
        const double v = shared_y[g];
        if (v != 0.0) {
            atomicAdd(&sum_y[g], v);
        }
    }
    for (int k = tid; k < total_x; k += block_size) {
        const double v = shared_x[k];
        if (v != 0.0) {
            atomicAdd(&sum_x[k], v);
        }
    }
}

__global__ __launch_bounds__(256, 4) void compute_means_kernel(double* sum_y,
                                     double* sum_x,
                                     const double* weight_sums,
                                     int groups,
                                     int cols) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    const double denom = weight_sums[g];
    const double inv = denom > 0.0 ? 1.0 / denom : 0.0;
    sum_y[g] *= inv;
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        sum_x[static_cast<std::size_t>(j) * groups + g] *= inv;
    }
}

__global__ __launch_bounds__(256, 4) void compute_slope_coefficients_kernel(
    double* sum_y,
    double* sum_zy,
    double* sum_x,
    double* sum_zx,
    const double* weight_sums,
    const double* sum_z,
    const double* sum_zz,
    int groups,
    int cols,
    bool include_intercept) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    constexpr double kDetTol = 1e-12;
    const double sw = weight_sums[g];
    const double sz = sum_z[g];
    const double szz = sum_zz[g];
    const double sy = sum_y[g];
    const double szy = sum_zy[g];
    double alpha = 0.0;
    double gamma = 0.0;
    if (include_intercept) {
        if (sw > 0.0) {
            const double det = sw * szz - sz * sz;
            const double scale = fmax(1.0, sw * szz);
            if (det > kDetTol * scale) {
                alpha = (szz * sy - sz * szy) / det;
                gamma = (-sz * sy + sw * szy) / det;
            } else {
                alpha = sy / sw;
            }
        }
    } else {
        const double scale = fmax(1.0, szz);
        if (szz > kDetTol * scale) {
            gamma = szy / szz;
        }
    }
    sum_y[g] = alpha;
    sum_zy[g] = gamma;

#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j) * groups + g;
        const double sx = sum_x[idx];
        const double szx = sum_zx[idx];
        double ax = 0.0;
        double gx = 0.0;
        if (include_intercept) {
            if (sw > 0.0) {
                const double det = sw * szz - sz * sz;
                const double scale = fmax(1.0, sw * szz);
                if (det > kDetTol * scale) {
                    ax = (szz * sx - sz * szx) / det;
                    gx = (-sz * sx + sw * szx) / det;
                } else {
                    ax = sx / sw;
                }
            }
        } else {
            const double scale = fmax(1.0, szz);
            if (szz > kDetTol * scale) {
                gx = szx / szz;
            }
        }
        sum_x[idx] = ax;
        sum_zx[idx] = gx;
    }
}

__global__ __launch_bounds__(256, 4) void accumulate_alpha_kernel(const double* mean_y,
                                        const double* mean_x,
                                        double* alpha_y,
                                        double* alpha_x,
                                        int groups,
                                        int cols,
                                        double scale) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    if (alpha_y) {
        alpha_y[g] += scale * mean_y[g];
    }
    if (alpha_x) {
#pragma unroll 4
        for (int j = 0; j < cols; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j) * groups + g;
            alpha_x[idx] += scale * mean_x[idx];
        }
    }
}

__global__ __launch_bounds__(256, 4) void apply_means_kernel(double* y,
                                   double* X,
                                   int n,
                                   int cols,
                                   int ld,
                                   const int* gid,
                                   const double* mean_y,
                                   const double* mean_x,
                                   int groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int g = gid[i];
    y[i] -= mean_y[g];
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        X[static_cast<std::size_t>(j) * ld + i] -=
            mean_x[static_cast<std::size_t>(j) * groups + g];
    }
}

__global__ __launch_bounds__(256, 4) void apply_slope_kernel(double* y,
                                   double* X,
                                   int n,
                                   int cols,
                                   int ld,
                                   const int* gid,
                                   const double* z,
                                   const double* alpha_y,
                                   const double* gamma_y,
                                   const double* alpha_x,
                                   const double* gamma_x,
                                   int groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int g = gid[i];
    const double zi = z[i];
    y[i] -= alpha_y[g] + gamma_y[g] * zi;
#pragma unroll 4
    for (int j = 0; j < cols; ++j) {
        const std::size_t idx = static_cast<std::size_t>(j) * groups + g;
        X[static_cast<std::size_t>(j) * ld + i] -= alpha_x[idx] + gamma_x[idx] * zi;
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void apply_means_sumsq_kernel(double* y,
                                         double* X,
                                         int n,
                                         int cols,
                                         int ld,
                                         const int* gid,
                                         const double* mean_y,
                                         const double* mean_x,
                                         int groups,
                                         double* sumsq_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    if (i < n) {
        const int g = gid[i];
        double yi = y[i] - mean_y[g];
        y[i] = yi;
        local += yi * yi;
#pragma unroll 4
        for (int j = 0; j < cols; ++j) {
            double xi =
                X[static_cast<std::size_t>(j) * ld + i] -
                mean_x[static_cast<std::size_t>(j) * groups + g];
            X[static_cast<std::size_t>(j) * ld + i] = xi;
            local += xi * xi;
        }
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(sumsq_out, block_sum);
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void apply_slope_sumsq_kernel(double* y,
                                         double* X,
                                         int n,
                                         int cols,
                                         int ld,
                                         const int* gid,
                                         const double* z,
                                         const double* alpha_y,
                                         const double* gamma_y,
                                         const double* alpha_x,
                                         const double* gamma_x,
                                         int groups,
                                         double* sumsq_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    if (i < n) {
        const int g = gid[i];
        const double zi = z[i];
        double yi = y[i] - alpha_y[g] - gamma_y[g] * zi;
        y[i] = yi;
        local += yi * yi;
#pragma unroll 4
        for (int j = 0; j < cols; ++j) {
            const std::size_t idx = static_cast<std::size_t>(j) * groups + g;
            double xi =
                X[static_cast<std::size_t>(j) * ld + i] - alpha_x[idx] - gamma_x[idx] * zi;
            X[static_cast<std::size_t>(j) * ld + i] = xi;
            local += xi * xi;
        }
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(sumsq_out, block_sum);
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void sumsq_kernel(const double* y,
	                             const double* X,
	                             int n,
                             int cols,
                             int ld,
                             double* sumsq_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    if (i < n) {
        const double yi = y[i];
        local += yi * yi;
#pragma unroll 4
        for (int j = 0; j < cols; ++j) {
            const double xi = X[static_cast<std::size_t>(j) * ld + i];
            local += xi * xi;
        }
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(sumsq_out, block_sum);
    }
}

__device__ double reldif_device(double newer, double older) {
    return fabs(newer - older) / (fabs(older) + 1.0);
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void reldif_update_sums_kernel(
    const double* y_new,
    const double* y_old,
    const double* X_new,
    const double* X_old,
    const double* weights,
    bool unit_weights,
    int n,
    int cols,
    int ld,
    double* sums_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int component = blockIdx.y;
    const int width = cols + 1;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    double local = 0.0;
    for (int i = tid; i < n; i += stride) {
        const double w = unit_weights ? 1.0 : weights[i];
        if (component == width) {
            local += w;
        } else if (component == 0) {
            local += w * reldif_device(y_new[i], y_old[i]);
        } else {
            const int j = component - 1;
            const std::size_t idx = static_cast<std::size_t>(j) * ld + i;
            local += w * reldif_device(X_new[idx], X_old[idx]);
        }
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(&sums_out[component], block_sum);
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void irons_tuck_stats_kernel(const double* x,
                                        const double* gx,
                                        const double* ggx,
                                        int n,
                                        double* stats_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_vprod;
    __shared__ typename BlockReduce::TempStorage temp_ssq;
    __shared__ typename BlockReduce::TempStorage temp_d1sq;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    double local_vprod = 0.0;
    double local_ssq = 0.0;
    double local_d1sq = 0.0;
#pragma unroll 4
    for (int i = tid; i < n; i += stride) {
        const double gx_val = gx[i];
        const double delta_gx = ggx[i] - gx_val;
        const double delta2 = delta_gx - gx_val + x[i];
        local_vprod += delta_gx * delta2;
        local_ssq += delta2 * delta2;
        local_d1sq += delta_gx * delta_gx;
    }
    const double block_vprod = BlockReduce(temp_vprod).Sum(local_vprod);
    const double block_ssq = BlockReduce(temp_ssq).Sum(local_ssq);
    const double block_d1sq = BlockReduce(temp_d1sq).Sum(local_d1sq);
    if (threadIdx.x == 0) {
        atomicAdd(&stats_out[0], block_vprod);
        atomicAdd(&stats_out[1], block_ssq);
        atomicAdd(&stats_out[2], block_d1sq);
    }
}

__global__ __launch_bounds__(256, 4) void irons_tuck_update_kernel(double* x,
                                         const double* gx,
                                         const double* ggx,
                                         int n,
                                         double coef) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
#pragma unroll 4
    for (int i = tid; i < n; i += stride) {
        const double delta_gx = ggx[i] - gx[i];
        x[i] = ggx[i] - coef * delta_gx;
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void component_sumsq_kernel(const double* y,
                                                                   const double* X,
                                                                   int n,
                                                                   int cols,
                                                                   int ld,
                                                                   double* sums_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int component = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    double local = 0.0;
    for (int i = tid; i < n; i += stride) {
        double v = 0.0;
        if (component == 0) {
            v = y[i];
        } else {
            const int j = component - 1;
            if (j < cols) {
                v = X[static_cast<std::size_t>(j) * ld + i];
            }
        }
        local += v * v;
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(&sums_out[component], block_sum);
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void component_dot_kernel(const double* y_a,
                                                                const double* y_b,
                                                                const double* X_a,
                                                                const double* X_b,
                                                                int n,
                                                                int cols,
                                                                int ld,
                                                                double* sums_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;

    const int component = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    double local = 0.0;
    for (int i = tid; i < n; i += stride) {
        double a = 0.0;
        double b = 0.0;
        if (component == 0) {
            a = y_a[i];
            b = y_b[i];
        } else {
            const int j = component - 1;
            if (j < cols) {
                const std::size_t idx = static_cast<std::size_t>(j) * ld + i;
                a = X_a[idx];
                b = X_b[idx];
            }
        }
        local += a * b;
    }
    const double block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(&sums_out[component], block_sum);
    }
}

__global__ __launch_bounds__(256, 4) void projection_residual_kernel(double* proj_y,
                                                                     const double* src_y,
                                                                     double* proj_x,
                                                                     const double* src_x,
                                                                     std::size_t total,
                                                                     int n,
                                                                     int ld) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t idx = tid; idx < total; idx += stride) {
        if (idx < static_cast<std::size_t>(n)) {
            proj_y[idx] = src_y[idx] - proj_y[idx];
        } else {
            const std::size_t x_idx = idx - static_cast<std::size_t>(n);
            proj_x[x_idx] = src_x[x_idx] - proj_x[x_idx];
        }
    }
}

__global__ __launch_bounds__(256, 4) void scaled_subtract_columns_kernel(double* target_y,
                                                                        double* target_x,
                                                                        const double* direction_y,
                                                                        const double* direction_x,
                                                                        const double* alpha,
                                                                        std::size_t total,
                                                                        int n,
                                                                        int ld) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t idx = tid; idx < total; idx += stride) {
        if (idx < static_cast<std::size_t>(n)) {
            target_y[idx] -= alpha[0] * direction_y[idx];
        } else {
            const std::size_t x_idx = idx - static_cast<std::size_t>(n);
            const int j = static_cast<int>(x_idx / static_cast<std::size_t>(ld));
            target_x[x_idx] -= alpha[j + 1] * direction_x[x_idx];
        }
    }
}

__global__ __launch_bounds__(256, 4) void cg_direction_update_kernel(double* u_y,
                                                                     double* u_x,
                                                                     const double* r_y,
                                                                     const double* r_x,
                                                                     const double* beta,
                                                                     std::size_t total,
                                                                     int n,
                                                                     int ld) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t idx = tid; idx < total; idx += stride) {
        if (idx < static_cast<std::size_t>(n)) {
            u_y[idx] = r_y[idx] + beta[0] * u_y[idx];
        } else {
            const std::size_t x_idx = idx - static_cast<std::size_t>(n);
            const int j = static_cast<int>(x_idx / static_cast<std::size_t>(ld));
            u_x[x_idx] = r_x[x_idx] + beta[j + 1] * u_x[x_idx];
        }
    }
}

// Reduce max(|values[i]|) over [0, n). IEEE-754 non-negative doubles preserve
// their ordering when reinterpreted as unsigned 64-bit, so atomicMax on the
// bit representation is correct for |x|.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void max_abs_kernel(const double* values,
                                                            int n,
                                                            double* max_out) {
    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    double local_max = 0.0;
    for (int i = tid; i < n; i += stride) {
        const double v = fabs(values[i]);
        if (v > local_max) {
            local_max = v;
        }
    }
    const double block_max = BlockReduce(temp).Reduce(local_max, cub::Max());
    if (threadIdx.x == 0 && block_max > 0.0) {
        unsigned long long* addr_ull =
            reinterpret_cast<unsigned long long*>(max_out);
        unsigned long long new_ull = __double_as_longlong(block_max);
        atomicMax(addr_ull, new_ull);
    }
}

__global__ void jacobi_update_kernel(double* y,
                                     double* X,
                                     int n,
                                     int cols,
                                     int ld,
                                     const int* const* gid_ptrs,
                                     const double* const* mean_y_ptrs,
                                     const double* const* mean_x_ptrs,
                                     const int* groups,
                                     int dims,
                                     double relaxation) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    double delta_y = 0.0;
    for (int dim = 0; dim < dims; ++dim) {
        const int g = gid_ptrs[dim][i];
        delta_y += mean_y_ptrs[dim][g];
        for (int j = 0; j < cols; ++j) {
            X[static_cast<std::size_t>(j) * ld + i] -=
                relaxation * mean_x_ptrs[dim][static_cast<std::size_t>(j) * groups[dim] + g];
        }
    }
    y[i] -= relaxation * delta_y;
}

__global__ void gi_denom_kernel(const int* group_ptr,
                                const int* group_individual,
                                const double* group_scale,
                                const double* weights,
                                bool unit_weights,
                                int groups,
                                double* denom) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    const double w = unit_weights ? 1.0 : weights[g];
    const double scale = group_scale[g];
    const double contrib = w * scale * scale;
    const int begin = group_ptr[g];
    const int end = group_ptr[g + 1];
    for (int pos = begin; pos < end; ++pos) {
        const int i = group_individual[pos];
        atomicAdd(&denom[i], contrib);
    }
}

__global__ void gi_denom_by_individual_kernel(const int* individual_ptr,
                                              const int* individual_group,
                                              const double* group_scale,
                                              const double* weights,
                                              bool unit_weights,
                                              int groups,
                                              int individuals,
                                              double* denom) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= individuals) {
        return;
    }
    const int begin = individual_ptr[i];
    const int end = individual_ptr[i + 1];
    double acc = 0.0;
    for (int pos = begin; pos < end; ++pos) {
        const int g = individual_group[pos];
        if (g < 0 || g >= groups) {
            continue;
        }
        const double scale = group_scale[g];
        const double w = unit_weights ? 1.0 : weights[g];
        acc += w * scale * scale;
    }
    denom[i] = acc > 0.0 ? acc : 1.0;
}

// Gauss-Seidel sweep over one color class of individuals. Individuals within
// a color share no observation row, so the reads (numer) and the atomicAdd
// updates touch disjoint rows and the launch is race-free and deterministic.
// Colors are processed sequentially by the host loop, which makes the overall
// sweep a valid Gauss-Seidel pass (same fixed point as the CPU sweep). A
// single launch over ALL individuals (the previous scheme) is NOT convergent:
// two individuals sharing a row read the same residual and double-subtract,
// which oscillates instead of contracting.
__global__ void gi_color_sweep_kernel(double* y,
                                      double* X,
                                      int groups,
                                      int cols,
                                      int ld,
                                      const int* individual_ptr,
                                      const int* individual_group,
                                      const double* group_scale,
                                      const double* weights,
                                      bool unit_weights,
                                      const double* denom,
                                      const int* color_individuals,
                                      int color_begin,
                                      int color_count,
                                      double relaxation) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= color_count) {
        return;
    }
    const int i = color_individuals[color_begin + t];
    const int begin = individual_ptr[i];
    const int end = individual_ptr[i + 1];
    const double inv_denom = 1.0 / (denom[i] > 0.0 ? denom[i] : 1.0);

    double numer_y = 0.0;
    if (cols > 0 && cols <= 8) {
        double numer_x[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int pos = begin; pos < end; ++pos) {
            const int g = individual_group[pos];
            if (g < 0 || g >= groups) {
                continue;
            }
            const double scale = group_scale[g];
            const double w = unit_weights ? 1.0 : weights[g];
            const double weight_scale = w * scale;
            numer_y += weight_scale * y[g];
            for (int j = 0; j < cols; ++j) {
                numer_x[j] += weight_scale * X[static_cast<std::size_t>(j) * ld + g];
            }
        }
        const double alpha_y = numer_y * inv_denom;
        double alpha_x[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int j = 0; j < cols; ++j) {
            alpha_x[j] = numer_x[j] * inv_denom;
        }
        for (int pos = begin; pos < end; ++pos) {
            const int g = individual_group[pos];
            if (g < 0 || g >= groups) {
                continue;
            }
            const double scale = group_scale[g];
            atomicAdd(&y[g], -relaxation * scale * alpha_y);
            for (int j = 0; j < cols; ++j) {
                atomicAdd(&X[static_cast<std::size_t>(j) * ld + g],
                          -relaxation * scale * alpha_x[j]);
            }
        }
        return;
    }

    // Generic fallback for cols > 8: loop over columns to avoid large per-thread buffers.
    for (int pos = begin; pos < end; ++pos) {
        const int g = individual_group[pos];
        if (g < 0 || g >= groups) {
            continue;
        }
        const double scale = group_scale[g];
        const double w = unit_weights ? 1.0 : weights[g];
        const double weight_scale = w * scale;
        numer_y += weight_scale * y[g];
    }
    const double alpha_y = numer_y * inv_denom;
    for (int pos = begin; pos < end; ++pos) {
        const int g = individual_group[pos];
        if (g < 0 || g >= groups) {
            continue;
        }
        const double scale = group_scale[g];
        atomicAdd(&y[g], -relaxation * scale * alpha_y);
    }
    for (int j = 0; j < cols; ++j) {
        double numer = 0.0;
        for (int pos = begin; pos < end; ++pos) {
            const int g = individual_group[pos];
            if (g < 0 || g >= groups) {
                continue;
            }
            const double scale = group_scale[g];
            const double w = unit_weights ? 1.0 : weights[g];
            const double weight_scale = w * scale;
            numer += weight_scale * X[static_cast<std::size_t>(j) * ld + g];
        }
        const double alpha = numer * inv_denom;
        for (int pos = begin; pos < end; ++pos) {
            const int g = individual_group[pos];
            if (g < 0 || g >= groups) {
                continue;
            }
            const double scale = group_scale[g];
            atomicAdd(&X[static_cast<std::size_t>(j) * ld + g],
                      -relaxation * scale * alpha);
        }
    }
}

__global__ void gi_accumulate_numer_kernel(const double* y,
                                          const double* X,
                                          int groups,
                                          int cols,
                                          int ld,
                                          const int* group_ptr,
                                          const int* group_individual,
                                          const double* group_scale,
                                          const double* weights,
                                          bool unit_weights,
                                          int individuals,
                                          double* numer_y,
                                          double* numer_x) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    const double w = unit_weights ? 1.0 : weights[g];
    const double scale = group_scale[g];
    const double coeff = w * scale;
    const double y_contrib = coeff * y[g];
    const int begin = group_ptr[g];
    const int end = group_ptr[g + 1];

    if (cols > 0 && cols <= 8) {
        double x_contrib[8];
        for (int j = 0; j < cols; ++j) {
            x_contrib[j] = coeff * X[static_cast<std::size_t>(j) * ld + g];
        }
        for (int pos = begin; pos < end; ++pos) {
            const int i = group_individual[pos];
            if (i < 0 || i >= individuals) {
                continue;
            }
            atomicAdd(&numer_y[i], y_contrib);
            for (int j = 0; j < cols; ++j) {
                atomicAdd(&numer_x[static_cast<std::size_t>(j) * individuals + i], x_contrib[j]);
            }
        }
        return;
    }

    for (int pos = begin; pos < end; ++pos) {
        const int i = group_individual[pos];
        if (i < 0 || i >= individuals) {
            continue;
        }
        atomicAdd(&numer_y[i], y_contrib);
        for (int j = 0; j < cols; ++j) {
            const double x_contrib = coeff * X[static_cast<std::size_t>(j) * ld + g];
            atomicAdd(&numer_x[static_cast<std::size_t>(j) * individuals + i], x_contrib);
        }
    }
}

__global__ void gi_compute_alpha_kernel(const double* numer_y,
                                       const double* numer_x,
                                       const double* denom,
                                       int individuals,
                                       int cols,
                                       double* alpha_y,
                                       double* alpha_x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= individuals) {
        return;
    }
    const double d = denom[i] > 0.0 ? denom[i] : 1.0;
    const double inv = 1.0 / d;
    alpha_y[i] = numer_y[i] * inv;
    for (int j = 0; j < cols; ++j) {
        alpha_x[static_cast<std::size_t>(j) * individuals + i] =
            numer_x[static_cast<std::size_t>(j) * individuals + i] * inv;
    }
}

__global__ void gi_apply_update_kernel(double* y,
                                      double* X,
                                      int groups,
                                      int cols,
                                      int ld,
                                      const int* group_ptr,
                                      const int* group_individual,
                                      const double* group_scale,
                                      const double* alpha_y,
                                      const double* alpha_x,
                                      int individuals,
                                      double relaxation) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) {
        return;
    }
    const double scale = group_scale[g];
    const int begin = group_ptr[g];
    const int end = group_ptr[g + 1];

    double sum_y = 0.0;
    if (cols > 0) {
        if (cols <= 8) {
            double sum_x[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int pos = begin; pos < end; ++pos) {
                const int i = group_individual[pos];
                if (i < 0 || i >= individuals) {
                    continue;
                }
                sum_y += alpha_y[i];
                for (int j = 0; j < cols; ++j) {
                    sum_x[j] += alpha_x[static_cast<std::size_t>(j) * individuals + i];
                }
            }
            y[g] -= relaxation * scale * sum_y;
            for (int j = 0; j < cols; ++j) {
                X[static_cast<std::size_t>(j) * ld + g] -= relaxation * scale * sum_x[j];
            }
            return;
        }
        // Generic fallback: scan the small group neighborhood once per column.
        for (int pos = begin; pos < end; ++pos) {
            const int i = group_individual[pos];
            if (i < 0 || i >= individuals) {
                continue;
            }
            sum_y += alpha_y[i];
        }
        y[g] -= relaxation * scale * sum_y;
        for (int j = 0; j < cols; ++j) {
            double sum_x = 0.0;
            for (int pos = begin; pos < end; ++pos) {
                const int i = group_individual[pos];
                if (i < 0 || i >= individuals) {
                    continue;
                }
                sum_x += alpha_x[static_cast<std::size_t>(j) * individuals + i];
            }
            X[static_cast<std::size_t>(j) * ld + g] -= relaxation * scale * sum_x;
        }
        return;
    }

    for (int pos = begin; pos < end; ++pos) {
        const int i = group_individual[pos];
        if (i < 0 || i >= individuals) {
            continue;
        }
        sum_y += alpha_y[i];
    }
    y[g] -= relaxation * scale * sum_y;
}

struct CudaFeDevice {
    int num_groups = 0;
    int num_levels_present = 0;
    bool is_slope = false;
    bool slope_has_intercept = false;
    bool use_segmented_sums = false;
    // Privatized accumulation: each CUDA block keeps per-group partial sums
    // in shared memory, then flushes to global once per (block,group).
    // Drastically reduces atomic contention when num_groups is small compared
    // to n (e.g., year, a handful of industries).
    bool use_privatized_sums = false;
    std::size_t privatized_shmem_bytes = 0;
    DeviceBuffer<int> gid;
    DeviceBuffer<double> weight_sums;
    DeviceBuffer<double> slope_z;
    DeviceBuffer<double> slope_sum_z;
    DeviceBuffer<double> slope_sum_zz;
    DeviceBuffer<double> sum_y;
    DeviceBuffer<double> sum_zy;
    DeviceBuffer<double> sum_x;
    DeviceBuffer<double> sum_zx;
    DeviceBuffer<double> alpha_y;
    DeviceBuffer<double> alpha_x;
    DeviceBuffer<double> alpha_y_gx;
    DeviceBuffer<double> alpha_x_gx;
    DeviceBuffer<double> alpha_y_ggx;
    DeviceBuffer<double> alpha_x_ggx;
    DeviceBuffer<double> alpha_y_acc_a;
    DeviceBuffer<double> alpha_x_acc_a;
    DeviceBuffer<double> alpha_y_acc_b;
    DeviceBuffer<double> alpha_x_acc_b;
    DeviceBuffer<double> alpha_y_acc_c;
    DeviceBuffer<double> alpha_x_acc_c;
};

struct CudaWorkspace {
    std::vector<CudaFeDevice> fe_dev;
    DeviceBuffer<double> d_y;
    DeviceBuffer<double> d_x;
	DeviceBuffer<double> d_weights;
	DeviceBuffer<double> d_sumsq;
    DeviceBuffer<double> d_y_prev_update;
    DeviceBuffer<double> d_x_prev_update;
    DeviceBuffer<double> d_update_sums;
    DeviceBuffer<double> d_cg_alpha;
    DeviceBuffer<double> d_cg_beta;
	DeviceBuffer<double> d_y_gx;
    DeviceBuffer<double> d_y_ggx;
    DeviceBuffer<double> d_x_gx;
    DeviceBuffer<double> d_x_ggx;
    DeviceBuffer<double> d_stats;
    DeviceBuffer<double> d_y_acc_a;
    DeviceBuffer<double> d_y_acc_b;
    DeviceBuffer<double> d_y_acc_c;
    DeviceBuffer<double> d_x_acc_a;
    DeviceBuffer<double> d_x_acc_b;
    DeviceBuffer<double> d_x_acc_c;

    // group()/individual() CUDA buffers.
    DeviceBuffer<int> gi_group_ptr;
    DeviceBuffer<int> gi_group_individual;
    DeviceBuffer<int> gi_individual_ptr;
    DeviceBuffer<int> gi_individual_group;
    DeviceBuffer<int> gi_color_individuals;
    DeviceBuffer<double> gi_group_scale;
    DeviceBuffer<double> gi_denom;
    DeviceBuffer<double> gi_numer_y;
    DeviceBuffer<double> gi_numer_x;
    DeviceBuffer<double> gi_alpha_y;
    DeviceBuffer<double> gi_alpha_x;

    // Reusable pinned staging for large host<->device copies.
    PinnedStage h_stage;
};

struct AlphaStatePtrs {
    std::vector<double*> y;
    std::vector<double*> x;
};

static thread_local CudaWorkspace cuda_workspace;

bool needs_reghdfe_update_check(const HdfeOptions& options) {
    return options.convergence_criterion == ConvergenceCriterion::Reghdfe ||
           options.convergence_criterion == ConvergenceCriterion::Both;
}

bool convergence_reached(const HdfeOptions& options,
                         double norm_error,
                         double update_error,
                         double tol) {
    const bool norm_ok = norm_error < tol;
    const bool reghdfe_ok = update_error <= tol;
    switch (options.convergence_criterion) {
        case ConvergenceCriterion::Auto:
        case ConvergenceCriterion::NormChange:
            return norm_ok;
        case ConvergenceCriterion::Reghdfe:
            return reghdfe_ok;
        case ConvergenceCriterion::Both:
            return norm_ok && reghdfe_ok;
    }
    return norm_ok;
}

double auto_reghdfe_cg_dominant_share_threshold_cuda() {
    static const double threshold = []() {
        const char* e = std::getenv("XHDFE_AUTO_CG_DOM_SHARE");
        if (e != nullptr) {
            const double v = std::atof(e);
            return v > 0.0 ? v : 0.0;
        }
        return 0.37;
    }();
    return threshold;
}

// ---- GPU-CG adaptive gate ----------------------------------------------------------------
// Route ill-conditioned (high projected MAP-iters) comparable jobs to the EXISTING block-PCG
// even when the dominant-FE-share gate (should_use_cuda_auto_default_reghdfe_cg) declines.
// On such graphs (e.g. the difficult worker+firm+year DGP) the Irons-Tuck MAP comparable path
// converges to a point ~2e-7 off reghdfe and needs ~530 sweeps, whereas the block-PCG matches
// reghdfe to ~1e-9 in fewer iterations. This mirrors the CPU Schwarz auto-gate: it reuses the
// SAME XHDFE_SCHWARZ_NMIN / XHDFE_SCHWARZ_MSWITCH knobs and the SAME plain-GS difficulty probe
// so CPU and GPU switch on exactly the same datasets. It is comparable-only and from_auto-only
// BY DESIGN: fast mode keeps the already-optimal IT-MAP (where unpreconditioned CG would need
// MORE iterations, i.e. a timing regression), and explicit absorption-method choices are
// honored. Disable independently with XHDFE_GPU_CG_AUTO=0.
// OPT-IN, default OFF. The GPU-CG substitution is a net win only in a narrow regime: ~1M-scale
// pathological 3-FE graphs, where the block-PCG converges in fewer iterations than IT-MAP AND
// closes the ~2e-7 comparable-mode coefficient gap (IT-MAP comparable stalls there; the CPU
// comparable path reaches ~1e-13 via Schwarz). It REGRESSES at larger scale -- the native 10M
// difficult graph needs ~1989 unpreconditioned-CG iters vs IT-MAP's ~530, i.e. ~4x slower -- and
// the host-side difficulty probe adds O(n) cost even to in-band datasets that end up declining.
// Per the project's strict non-regression mandate it is therefore disabled by default and must
// be requested explicitly with XHDFE_GPU_CG_AUTO=1. When enabled it is still bounded by an
// upper-n guard (XHDFE_GPU_CG_NMAX) so opted-in users never hit the large-scale regression.
bool gpu_cg_auto_enabled() {
    const char* e = std::getenv("XHDFE_GPU_CG_AUTO");
    return e != nullptr && e[0] != '0';
}
int64_t gpu_cg_nmax() {
    static const int64_t v = []() {
        const char* e = std::getenv("XHDFE_GPU_CG_NMAX");
        return e ? std::max<int64_t>(1, std::atoll(e)) : static_cast<int64_t>(2000000);
    }();
    return v;
}
int64_t schwarz_auto_nmin_cuda() {
    static const int64_t v = []() {
        const char* e = std::getenv("XHDFE_SCHWARZ_NMIN");
        return e ? std::max<int64_t>(1, std::atoll(e)) : static_cast<int64_t>(750000);
    }();
    return v;
}
double schwarz_auto_mswitch_cuda() {
    static const double v = []() {
        const char* e = std::getenv("XHDFE_SCHWARZ_MSWITCH");
        return e ? std::atof(e) : 50000.0;
    }();
    return v;
}
bool gpu_cg_diag_enabled() { return std::getenv("XHDFE_GPU_CG_DIAG") != nullptr; }
// Bound the (serial, host-side) difficulty probe to a stride-sampled subset so it never adds
// meaningful wall-clock on very large datasets that the gate ends up declining (e.g. a 46M-row
// real panel: the full probe costs ~5.5s, the 2M-row sample ~0.25s). n at or below the cap runs
// the full probe unchanged. The switch/no-switch margin is large (~50x: easy panels project a
// few thousand iters, pathological graphs hundreds of thousands), so a uniform edge sub-sample
// does not flip the decision; re-validated on ready (->MAP) and difficult 1M/10M (->CG).
int64_t gpu_cg_probe_cap() {
    static const int64_t v = []() {
        const char* e = std::getenv("XHDFE_GPU_CG_PROBE_CAP");
        return e ? std::max<int64_t>(1, std::atoll(e)) : static_cast<int64_t>(2000000);
    }();
    return v;
}

// Plain-Gauss-Seidel MAP-difficulty probe on a COPY of host y. Identical in form to the CPU
// schwarz_probe_projected_iters so the projected-iters scale and the mswitch calibration carry
// over unchanged. Returns +inf when not contracting. Does NOT mutate the caller's y.
double schwarz_probe_projected_iters_cuda(const std::vector<const int*>& g,
                                          const std::vector<int>& ng,
                                          const double* y, int n, double tol) {
    const int K0 = 5, W = 4, NS = K0 + W;
    const int D = static_cast<int>(g.size());
    std::vector<std::vector<double>> cnt(D), sum(D);
    for (int d = 0; d < D; ++d) {
        cnt[d].assign(ng[d], 0.0);
        sum[d].assign(ng[d], 0.0);
        for (int i = 0; i < n; ++i) cnt[d][g[d][i]] += 1.0;
    }
    std::vector<double> yp(y, y + n);
    auto norm2 = [&]() {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += yp[i] * yp[i];
        return std::sqrt(s);
    };
    double prevn = std::max(norm2(), 1e-300);
    double logprod = 0.0;
    int nr = 0;
    for (int it = 0; it < NS; ++it) {
        for (int d = 0; d < D; ++d) {
            auto& s = sum[d];
            std::fill(s.begin(), s.end(), 0.0);
            const int* gd = g[d];
            for (int i = 0; i < n; ++i) s[gd[i]] += yp[i];
            for (size_t j = 0; j < s.size(); ++j) s[j] = cnt[d][j] > 0.0 ? s[j] / cnt[d][j] : 0.0;
            for (int i = 0; i < n; ++i) yp[i] -= s[gd[i]];
        }
        double nn = norm2();
        if (it >= K0 && prevn > 1e-300) {
            double r = nn / prevn;
            if (r > 0.0 && r < 1.0) { logprod += std::log(r); nr++; }
        }
        prevn = nn;
    }
    if (nr == 0) return 1e18;
    const double rho_bar = std::exp(logprod / nr);
    if (!(rho_bar > 0.0 && rho_bar < 1.0)) return 1e18;
    const double proj = std::log(tol) / std::log(rho_bar);
    return proj > 0.0 ? proj : 1e18;
}

bool should_use_cuda_adaptive_cg(const std::vector<GpuFeInput>& fe_inputs,
                                 int n,
                                 const Eigen::VectorXd* weights,
                                 const HdfeOptions& options,
                                 bool store_alphas,
                                 const Eigen::Ref<const Eigen::VectorXd>& y) {
    if (!gpu_cg_auto_enabled()) {
        return false;  // opt-in only (default OFF) -- see gpu_cg_auto_enabled() rationale
    }
    // Comparable + auto-default only (mirror should_use_cuda_auto_default_reghdfe_cg), and only
    // when the absorption method was left on Auto (from_auto) so explicit choices are honored.
    if (options.convergence_criterion != ConvergenceCriterion::Auto ||
        options.tolerance_mode != ToleranceMode::ReghdfeComparable ||
        !options.from_auto) {
        return false;
    }
    if (store_alphas || options.retain_fixed_effects || weights != nullptr || n <= 0) {
        return false;
    }
    const std::size_t dims = fe_inputs.size();
    if (dims < 2 || dims > 3) {
        return false;
    }
    if (static_cast<int64_t>(n) < schwarz_auto_nmin_cuda()) {
        return false;
    }
    if (static_cast<int64_t>(n) > gpu_cg_nmax()) {
        return false;  // above the validated-win regime; IT-MAP avoids the CG-at-scale regression
    }
    std::vector<const int*> g;
    std::vector<int> ng;
    g.reserve(dims);
    ng.reserve(dims);
    for (const auto& fi : fe_inputs) {
        if (!fi.group_ids || fi.num_groups <= 0) {
            return false;
        }
        g.push_back(fi.group_ids);
        ng.push_back(fi.num_groups);
    }
    const double eff_tol = effective_absorption_tolerance(options);
    // Stride-sample the probe input when n exceeds the cap so the serial host probe stays cheap.
    const int64_t cap = gpu_cg_probe_cap();
    int m = n;
    std::vector<const int*> gp = g;
    const double* yptr = y.data();
    std::vector<int> gs_storage;
    std::vector<const int*> gs_ptrs;
    std::vector<double> ys;
    if (cap > 0 && static_cast<int64_t>(n) > cap) {
        const int stride = static_cast<int>(static_cast<int64_t>(n) / cap);
        m = (n + stride - 1) / stride;
        ys.resize(static_cast<size_t>(m));
        gs_storage.resize(static_cast<size_t>(dims) * static_cast<size_t>(m));
        for (size_t d = 0; d < dims; ++d) {
            int* dst = gs_storage.data() + d * static_cast<size_t>(m);
            const int* src = g[d];
            int k = 0;
            for (int i = 0; i < n && k < m; i += stride) dst[k++] = src[i];
        }
        int k = 0;
        for (int i = 0; i < n && k < m; i += stride) ys[k++] = yptr[i];
        gs_ptrs.resize(dims);
        for (size_t d = 0; d < dims; ++d) gs_ptrs[d] = gs_storage.data() + d * static_cast<size_t>(m);
        gp = gs_ptrs;
        yptr = ys.data();
    }
    const double proj = schwarz_probe_projected_iters_cuda(gp, ng, yptr, m, eff_tol);
    const bool go = proj > schwarz_auto_mswitch_cuda();
    if (gpu_cg_diag_enabled()) {
        std::fprintf(stderr,
                     "[gpu-cg-gate] n=%d probe_rows=%d ndim=%zu proj_iters=%.0f mswitch=%.0f nmin=%lld -> %s\n",
                     n, m, dims, proj, schwarz_auto_mswitch_cuda(),
                     static_cast<long long>(schwarz_auto_nmin_cuda()), go ? "CG" : "MAP");
    }
    return go;
}

bool should_use_cuda_auto_default_reghdfe_cg(const std::vector<GpuFeInput>& fe_inputs,
                                             int n,
                                             const Eigen::VectorXd* weights,
                                             const HdfeOptions& options,
                                             bool store_alphas) {
    if (options.convergence_criterion != ConvergenceCriterion::Auto ||
        options.tolerance_mode != ToleranceMode::ReghdfeComparable) {
        return false;
    }
    if (store_alphas || options.retain_fixed_effects || weights != nullptr ||
        fe_inputs.size() < 3 || n <= 0) {
        return false;
    }

    const double threshold = auto_reghdfe_cg_dominant_share_threshold_cuda();
    if (!(threshold > 0.0 && threshold <= 1.0)) {
        return false;
    }

    double max_share = 0.0;
    const double denom = static_cast<double>(n);
    for (const auto& fi : fe_inputs) {
        if (!fi.weight_sums || fi.num_groups <= 0) {
            return false;
        }
        double max_weight = 0.0;
        for (int g = 0; g < fi.num_groups; ++g) {
            max_weight = std::max(max_weight, fi.weight_sums[g]);
        }
        max_share = std::max(max_share, max_weight / denom);
    }
    return max_share >= threshold;
}

bool should_use_cuda_reghdfe_cg(const std::vector<GpuFeInput>& fe_inputs,
                                int n,
                                const Eigen::VectorXd* weights,
                                const HdfeOptions& options,
                                bool store_alphas) {
    const bool explicit_reghdfe =
        options.convergence_criterion != ConvergenceCriterion::Auto &&
        options.convergence_criterion != ConvergenceCriterion::NormChange;
    const bool auto_default =
        should_use_cuda_auto_default_reghdfe_cg(fe_inputs, n, weights, options,
                                                store_alphas);
    if (!explicit_reghdfe && !auto_default) {
        return false;
    }
    const std::size_t dims = fe_inputs.size();
    if (store_alphas || options.retain_fixed_effects || dims < 2) {
        return false;
    }
    const char* disabled = std::getenv("XHDFE_CUDA_REGHDFE_CG");
    if (disabled != nullptr && disabled[0] == '0') {
        return false;
    }
    return weights == nullptr;
}

struct ExactEdgeUnionFind {
    std::vector<int> parent;
    std::vector<unsigned char> rank;

    explicit ExactEdgeUnionFind(int n)
        : parent(static_cast<std::size_t>(n)), rank(static_cast<std::size_t>(n), 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[static_cast<std::size_t>(x)] != x) {
            parent[static_cast<std::size_t>(x)] =
                parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
            x = parent[static_cast<std::size_t>(x)];
        }
        return x;
    }

    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) {
            return;
        }
        const std::size_t ia = static_cast<std::size_t>(ra);
        const std::size_t ib = static_cast<std::size_t>(rb);
        if (rank[ia] < rank[ib]) {
            parent[ia] = rb;
        } else if (rank[ia] > rank[ib]) {
            parent[ib] = ra;
        } else {
            parent[ib] = ra;
            ++rank[ia];
        }
    }
};

struct TreeDuplicateEdgeGroups {
    std::vector<int> obs_edge;
    std::vector<int> edge_counts;
    int total_levels = 0;
    int components = 0;
};

bool build_tree_duplicate_edge_groups(const std::vector<GpuFeInput>& fe_inputs,
                                      int n,
                                      TreeDuplicateEdgeGroups& out) {
    out = TreeDuplicateEdgeGroups{};
    if (fe_inputs.size() != 2 || n <= 0 || n > 20000) {
        return false;
    }
    const int g0 = fe_inputs[0].num_groups;
    const int g1 = fe_inputs[1].num_groups;
    if (g0 <= 0 || g1 <= 0 || fe_inputs[0].group_ids == nullptr ||
        fe_inputs[1].group_ids == nullptr) {
        return false;
    }
    const int total_levels = g0 + g1;
    if (total_levels < n - 2 || total_levels > n + 2) {
        return false;
    }

    ExactEdgeUnionFind uf(total_levels);
    std::unordered_map<std::uint64_t, int> edge_to_group;
    edge_to_group.reserve(static_cast<std::size_t>(n));
    out.obs_edge.resize(static_cast<std::size_t>(n));
    out.edge_counts.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        const int a = fe_inputs[0].group_ids[i];
        const int b = fe_inputs[1].group_ids[i];
        if (a < 0 || a >= g0 || b < 0 || b >= g1) {
            out = TreeDuplicateEdgeGroups{};
            return false;
        }
        const std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
            static_cast<std::uint32_t>(b);
        auto it = edge_to_group.find(key);
        if (it == edge_to_group.end()) {
            const int e = static_cast<int>(out.edge_counts.size());
            edge_to_group.emplace(key, e);
            out.obs_edge[static_cast<std::size_t>(i)] = e;
            out.edge_counts.push_back(1);
            uf.unite(a, g0 + b);
        } else {
            out.obs_edge[static_cast<std::size_t>(i)] = it->second;
            ++out.edge_counts[static_cast<std::size_t>(it->second)];
        }
    }

    std::unordered_map<int, int> roots;
    roots.reserve(static_cast<std::size_t>(total_levels));
    for (int node = 0; node < total_levels; ++node) {
        roots.emplace(uf.find(node), 1);
    }
    out.components = static_cast<int>(roots.size());
    out.total_levels = total_levels;

    const int unique_edges = static_cast<int>(out.edge_counts.size());
    const int duplicate_rows = n - unique_edges;
    if (duplicate_rows <= 0) {
        out = TreeDuplicateEdgeGroups{};
        return false;
    }
    if (unique_edges != total_levels - out.components) {
        out = TreeDuplicateEdgeGroups{};
        return false;
    }
    return true;
}

bool absorb_tree_duplicate_edges_cuda(const Eigen::Ref<const Eigen::VectorXd>& y,
                                      const Eigen::Ref<const Eigen::MatrixXd>& X,
                                      const TreeDuplicateEdgeGroups& edge_groups,
                                      const std::vector<GpuFeInput>& fe_inputs,
                                      AbsorptionResult& result) {
    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    const int ld = static_cast<int>(X.rows());
    const int unique_edges = static_cast<int>(edge_groups.edge_counts.size());
    if (n <= 0 || unique_edges <= 0 || static_cast<int>(edge_groups.obs_edge.size()) != n ||
        ld != n) {
        result.converged = false;
        return false;
    }

    try {
        cudaGetLastError();
        DeviceBuffer<double> d_y;
        DeviceBuffer<double> d_x;
        DeviceBuffer<int> d_edge_group;
        DeviceBuffer<int> d_edge_counts;
        DeviceBuffer<double> d_sum_y;
        DeviceBuffer<double> d_sum_x;

        d_y.allocate(static_cast<std::size_t>(n));
        cuda_check(cudaMemcpy(d_y.data(), y.data(), sizeof(double) * n, cudaMemcpyHostToDevice),
                   "cudaMemcpy exact tree y failed");
        if (cols > 0) {
            d_x.allocate(static_cast<std::size_t>(ld) * cols);
            cuda_check(cudaMemcpy(d_x.data(), X.data(), sizeof(double) * ld * cols,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy exact tree X failed");
        }

        d_edge_group.allocate(static_cast<std::size_t>(n));
        cuda_check(cudaMemcpy(d_edge_group.data(), edge_groups.obs_edge.data(),
                              sizeof(int) * n, cudaMemcpyHostToDevice),
                   "cudaMemcpy exact tree edge ids failed");
        d_edge_counts.allocate(static_cast<std::size_t>(unique_edges));
        cuda_check(cudaMemcpy(d_edge_counts.data(), edge_groups.edge_counts.data(),
                              sizeof(int) * unique_edges, cudaMemcpyHostToDevice),
                   "cudaMemcpy exact tree edge counts failed");

        d_sum_y.allocate(static_cast<std::size_t>(unique_edges));
        cuda_check(cudaMemset(d_sum_y.data(), 0, sizeof(double) * unique_edges),
                   "cudaMemset exact tree sum_y failed");
        if (cols > 0) {
            d_sum_x.allocate(static_cast<std::size_t>(unique_edges) * cols);
            cuda_check(cudaMemset(d_sum_x.data(), 0,
                                  sizeof(double) * unique_edges * cols),
                       "cudaMemset exact tree sum_x failed");
        }

        const int blocks = (n + kBlockSize - 1) / kBlockSize;
        accumulate_edge_sums_kernel<<<blocks, kBlockSize>>>(
            d_y.data(), cols > 0 ? d_x.data() : nullptr, n, cols, ld, unique_edges,
            d_edge_group.data(), d_sum_y.data(), cols > 0 ? d_sum_x.data() : nullptr);
        cuda_check(cudaGetLastError(), "exact tree edge-sum kernel launch failed");

        apply_edge_residuals_kernel<<<blocks, kBlockSize>>>(
            d_y.data(), cols > 0 ? d_x.data() : nullptr, n, cols, ld, unique_edges,
            d_edge_group.data(), d_edge_counts.data(), d_sum_y.data(),
            cols > 0 ? d_sum_x.data() : nullptr);
        cuda_check(cudaGetLastError(), "exact tree residual kernel launch failed");

        result.y_tilde.resize(n);
        cuda_check(cudaMemcpy(result.y_tilde.data(), d_y.data(), sizeof(double) * n,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy exact tree y_tilde failed");
        if (cols > 0) {
            result.X_tilde.resize(ld, cols);
            cuda_check(cudaMemcpy(result.X_tilde.data(), d_x.data(),
                                  sizeof(double) * ld * cols, cudaMemcpyDeviceToHost),
                       "cudaMemcpy exact tree X_tilde failed");
        } else {
            result.X_tilde.resize(ld, 0);
        }

        result.fe_levels.clear();
        result.fe_levels.reserve(fe_inputs.size());
        for (const auto& fe : fe_inputs) {
            result.fe_levels.push_back(fe.num_levels_present);
        }
        result.sweep_order_used = {0, 1};
        result.iterations = 1;
        result.converged = true;
        return true;
    } catch (const std::exception& e) {
        if (std::getenv("XHDFE_DEBUG_GPU") != nullptr) {
            std::fprintf(stderr, "[gpu] exact tree absorption failed: %s\n", e.what());
        }
        cudaGetLastError();
        result.converged = false;
        result.iterations = 0;
        return false;
    } catch (...) {
        if (std::getenv("XHDFE_DEBUG_GPU") != nullptr) {
            std::fprintf(stderr, "[gpu] exact tree absorption failed: unknown\n");
        }
        cudaGetLastError();
        result.converged = false;
        result.iterations = 0;
        return false;
    }
}

struct FeProfileStats {
    int num_groups = 0;
    int num_levels_present = 0;
    double mean = 0.0;
    double stddev = 0.0;
    double cv = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    double singleton_ratio = 0.0;
};

FeProfileStats profile_fe(const GpuFeInput& fe, bool unit_weights) {
    FeProfileStats stats;
    const int groups = fe.num_groups;
    stats.num_groups = groups;
    stats.num_levels_present = fe.num_levels_present;
    if (groups <= 0 || fe.weight_sums == nullptr) {
        return stats;
    }
    double sum = 0.0;
    double sumsq = 0.0;
    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();
    int singletons = 0;
    constexpr double kSingletonTol = 1e-12;
    for (int g = 0; g < groups; ++g) {
        const double v = fe.weight_sums[g];
        sum += v;
        sumsq += v * v;
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        if (v <= 1.0 + kSingletonTol) {
            ++singletons;
        }
    }
    stats.mean = sum / static_cast<double>(groups);
    const double var =
        std::max(0.0, sumsq / static_cast<double>(groups) - stats.mean * stats.mean);
    stats.stddev = std::sqrt(var);
    stats.cv = stats.mean > 0.0 ? stats.stddev / stats.mean : 0.0;
    stats.min_val = std::isfinite(min_val) ? min_val : 0.0;
    stats.max_val = std::isfinite(max_val) ? max_val : 0.0;
    stats.singleton_ratio = unit_weights
                                ? static_cast<double>(singletons) / static_cast<double>(groups)
                                : 0.0;
    return stats;
}

enum class FeDifficulty { Easy, Medium, Hard, Extreme };

FeDifficulty classify_problem(const std::vector<FeProfileStats>& profiles) {
    const int dims = static_cast<int>(profiles.size());
    double max_singleton = 0.0;
    double max_cv = 0.0;
    for (const auto& p : profiles) {
        max_singleton = std::max(max_singleton, p.singleton_ratio);
        max_cv = std::max(max_cv, p.cv);
    }
    if (dims >= 4 && (max_singleton >= 0.30 || max_cv >= 2.0)) {
        return FeDifficulty::Extreme;
    }
    if (dims >= 4 || max_singleton >= 0.20 || max_cv >= 1.5) {
        return FeDifficulty::Hard;
    }
    if (dims >= 3 || max_singleton >= 0.10 || max_cv >= 1.0) {
        return FeDifficulty::Medium;
    }
    return FeDifficulty::Easy;
}

std::vector<std::size_t> select_two_largest_dims(const std::vector<FeProfileStats>& profiles) {
    if (profiles.size() < 2) {
        return {};
    }
    int best = -1;
    int second = -1;
    int best_groups = -1;
    int second_groups = -1;
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        const int groups = profiles[i].num_groups;
        if (groups > best_groups) {
            second = best;
            second_groups = best_groups;
            best = static_cast<int>(i);
            best_groups = groups;
        } else if (groups > second_groups) {
            second = static_cast<int>(i);
            second_groups = groups;
        }
    }
    if (best < 0 || second < 0) {
        return {};
    }
    std::vector<std::size_t> dims;
    dims.reserve(2);
    dims.push_back(static_cast<std::size_t>(best));
    dims.push_back(static_cast<std::size_t>(second));
    return dims;
}

std::vector<std::size_t> filter_order(const std::vector<std::size_t>& order,
                                      const std::vector<std::size_t>& dims) {
    if (order.empty() || dims.empty()) {
        return {};
    }
    std::vector<uint8_t> wanted(order.size(), 0);
    for (const std::size_t dim : dims) {
        if (dim < wanted.size()) {
            wanted[dim] = 1;
        }
    }
    std::vector<std::size_t> filtered;
    filtered.reserve(dims.size());
    for (const std::size_t dim : order) {
        if (dim < wanted.size() && wanted[dim]) {
            filtered.push_back(dim);
        }
    }
    return filtered;
}

}  // namespace

bool cuda_backend_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (std::getenv("XHDFE_CUDA_AVAIL_DIAG") != nullptr) {
        std::fprintf(stderr, "[xhdfe-cuda] cudaGetDeviceCount status=%d (%s) count=%d\n",
                     static_cast<int>(status), cudaGetErrorString(status), count);
    }
    if (status != cudaSuccess) {
        return false;
    }
    return count > 0;
}

bool absorb_fixed_effects_cuda(const Eigen::Ref<const Eigen::VectorXd>& y,
                               const Eigen::Ref<const Eigen::MatrixXd>& X,
                               const std::vector<GpuFeInput>& fe_inputs,
                               const Eigen::VectorXd* weights,
                               const std::vector<std::size_t>& sweep_order,
                               const HdfeOptions& options,
                               AbsorptionMethod method,
                               AbsorptionResult& result) {
    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    const int ld = static_cast<int>(X.rows());
    if (ld != n) {
        return false;
    }
    if (fe_inputs.empty()) {
        result.y_tilde = y;
        result.X_tilde = X;
        result.iterations = 0;
        result.converged = true;
        return true;
    }

    bool any_slope = false;
    for (const auto& fe : fe_inputs) {
        any_slope = any_slope || fe.is_slope;
        if (fe.is_slope &&
            (!fe.slope_values || !fe.slope_sum_z || !fe.slope_sum_zz)) {
            return false;
        }
    }
    if (any_slope && method == AbsorptionMethod::Jacobi) {
        return false;
    }

    // Default ON (10jun2026): restores the exact 1-iteration solve for
    // saturated tree/path graphs (zigzag class) that the feature-branch
    // kernel restructure had regressed to tens of thousands of accelerated
    // iterations with degraded SEs. Shape-gated by
    // build_tree_duplicate_edge_groups; disable with XHDFE_CUDA_EXACT_TREE=0.
    const char* exact_tree_env = std::getenv("XHDFE_CUDA_EXACT_TREE");
    const bool use_exact_tree_shortcut =
        exact_tree_env == nullptr || exact_tree_env[0] != '0';
    if (use_exact_tree_shortcut && !any_slope && weights == nullptr &&
        !options.retain_fixed_effects) {
        TreeDuplicateEdgeGroups edge_groups;
        if (build_tree_duplicate_edge_groups(fe_inputs, n, edge_groups)) {
            return absorb_tree_duplicate_edges_cuda(y, X, edge_groups, fe_inputs, result);
        }
    }

    const bool store_alphas =
        !any_slope &&
        options.retain_fixed_effects &&
        options.fe_recovery_method == FeRecoveryMethod::Hybrid &&
        cols <= options.savefe_fastpath_max_cols;
    // Mirror the CPU acceleration gate: large n always, plus ill-conditioned
    // sparse multi-FE graphs detected via low mean group occupancy
    // (observations per FE level). Tree-like citation / board-membership /
    // matched-pair panels (e.g. patents, directors) otherwise run unaccelerated
    // and need thousands of Gauss-Seidel sweeps; the Irons-Tuck path converges
    // in tens. Threshold tunable via XHDFE_ACCEL_OCC (benchmarking only).
    double min_fe_occupancy_gpu = -1.0;
    for (const auto& fi : fe_inputs) {
        const int levels = fi.num_levels_present > 0 ? fi.num_levels_present : fi.num_groups;
        if (levels > 0) {
            const double occ = static_cast<double>(n) / static_cast<double>(levels);
            if (min_fe_occupancy_gpu < 0.0 || occ < min_fe_occupancy_gpu) {
                min_fe_occupancy_gpu = occ;
            }
        }
    }
    static const double accel_occ_threshold_gpu = []() {
        const char* e = std::getenv("XHDFE_ACCEL_OCC");
        return (e != nullptr) ? std::atof(e) : 12.0;  // keep in sync with kAccelOccupancyThreshold (CPU)
    }();
    const bool ill_conditioned_graph_gpu =
        fe_inputs.size() >= 2 && min_fe_occupancy_gpu >= 0.0 &&
        min_fe_occupancy_gpu <= accel_occ_threshold_gpu;
    bool use_reghdfe_cg =
        should_use_cuda_reghdfe_cg(fe_inputs, n, weights, options, store_alphas);
    if (!use_reghdfe_cg) {
        // GPU-CG adaptive gate: hand ill-conditioned high-iteration comparable jobs to the
        // existing block-PCG. On these graphs the IT-MAP comparable path stalls ~2e-7 off
        // reghdfe in ~530 sweeps; the PCG matches reghdfe to ~1e-9 in fewer iterations. Only
        // runs the probe when the dominant-share gate already declined. Fast mode untouched.
        use_reghdfe_cg =
            should_use_cuda_adaptive_cg(fe_inputs, n, weights, options, store_alphas, y);
    }
    const bool use_accel =
        use_reghdfe_cg ||
        (fe_inputs.size() >= 2 && (n >= 200000 || ill_conditioned_graph_gpu));
    // Match the CPU work-bound guarantee: for occupancy-triggered acceleration
    // (small, possibly pathological graphs) cap accelerated iterations so the
    // accelerated path never performs more sweeps than the plain solver would.
    const bool accel_from_occupancy_gpu = use_accel && (n < 200000);
    const int accel_iter_budget =
        accel_from_occupancy_gpu ? std::max(1, (options.max_iter * 2) / 5) : options.max_iter;
    constexpr int kGrandAccelInterval = 4;
    const bool accel_store_alphas = store_alphas && use_accel;
    const bool accel_grand_alphas = accel_store_alphas && (kGrandAccelInterval > 0);
    const double fe_tol =
        options.fe_tolerance > 0.0 ? options.fe_tolerance : options.tol;
    double convergence_tol = effective_absorption_tolerance(options);
    // See fe_absorption.cpp: reghdfe-comparable mode switches the accelerated
    // loop to the norm-of-change criterion at the NOMINAL tolerance.
    const bool honest_mode =
        honest_tol_trigger_enabled() ||
        options.tolerance_mode == ToleranceMode::ReghdfeComparable;
    const double honest_tol = options.tol;
    // Heterogeneous-slope designs: the CPU mixed absorber resolves the Auto
    // criterion under reghdfe-comparable to the Reghdfe mean-reldif update
    // check (resolved_mixed_convergence_criterion in fe_absorption.cpp). The
    // honest norm-of-change exit is a different, stricter stopping metric and
    // made GPU comparable run ~2x the CPU iterations on slope designs. Mirror
    // the CPU criterion here so both backends stop on the same validated
    // rule. Kill switch: XHDFE_GPU_MIXED_REGHDFE_CRIT=0.
    static const bool mixed_crit_enabled = []() {
        const char* e = std::getenv("XHDFE_GPU_MIXED_REGHDFE_CRIT");
        return !(e != nullptr && e[0] == '0');
    }();
    const bool mixed_update_criterion =
        mixed_crit_enabled && any_slope &&
        options.tolerance_mode == ToleranceMode::ReghdfeComparable &&
        options.convergence_criterion == ConvergenceCriterion::Auto &&
        !honest_tol_trigger_enabled();
    // IT->CG divergence hand-off (mirror of fe_absorption.cpp). When the
    // Irons-Tuck safeguard detects divergence (norm rises past cg_bail_factor x
    // its running minimum) or the accelerator grinds past cg_iter_cap honest
    // iterations on an ill-conditioned graph (github-class), hand the current
    // post-sweep iterate to the stable block-CG instead of plain Gauss-Seidel.
    // No-op on well-conditioned data: the iterate norm is monotone and every
    // supported dataset converges well under the cap, so the 16 others stay
    // bit-identical. Tunable via XHDFE_PACKED_CG_FALLBACK / XHDFE_CG_BAIL_FACTOR
    // / XHDFE_CG_ITER_CAP (kept in sync with the CPU constants).
    static const bool cuda_cg_fallback_enabled = []() {
        const char* e = std::getenv("XHDFE_PACKED_CG_FALLBACK");
        return !(e != nullptr && e[0] == '0');
    }();
    static const double cg_bail_factor = []() {
        const char* e = std::getenv("XHDFE_CG_BAIL_FACTOR");
        const double v = (e != nullptr) ? std::atof(e) : 1.5;
        return v > 1.0 ? v : 1.5;
    }();
    static const int cg_iter_cap = []() {
        const char* e = std::getenv("XHDFE_CG_ITER_CAP");
        if (e != nullptr) {
            const int v = std::atoi(e);
            return v > 0 ? v : 0;
        }
        return 1000;
    }();
    // The block-CG handles only the no-alpha, unit-weight, multi-FE case (same
    // preconditions as should_use_cuda_reghdfe_cg minus the criterion gate).
    const bool cuda_cg_fallback_ok =
        cuda_cg_fallback_enabled && honest_mode && !store_alphas &&
        !options.retain_fixed_effects && fe_inputs.size() >= 2 && weights == nullptr;
    bool accel_diverged = false;
    if (store_alphas && fe_tol > 0.0 && convergence_tol > 0.0) {
        constexpr double kFeTolScale = 1e-8;
        convergence_tol = std::min(convergence_tol, fe_tol * kFeTolScale);
    }

    try {
        // Consume any stale, sticky CUDA error left by a prior failed or
        // non-converged call in this process. CUDA runtime errors are sticky and
        // are otherwise surfaced by the next CUDA API call (here the first
        // cudaMalloc/cudaMemcpy), which would spuriously fail this absorption and
        // (under gpubackend(cuda)) be reported as rc=498 even though the GPU is
        // healthy. This was the cause of the transient "CUDA absorption failed"
        // results observed when many estimations run back-to-back in one session.
        cudaGetLastError();
        const bool unit_weights = (weights == nullptr);
        CudaWorkspace& workspace = cuda_workspace;

        DeviceBuffer<double>& d_y = workspace.d_y;
        d_y.allocate(static_cast<std::size_t>(n));
        staged_memcpy_h2d(d_y.data(), y.data(), sizeof(double) * n, workspace.h_stage,
                          "cudaMemcpy y failed");

        const std::size_t x_size = static_cast<std::size_t>(ld) * cols;
        DeviceBuffer<double>& d_x = workspace.d_x;
        if (cols > 0) {
            d_x.allocate(x_size);
            staged_memcpy_h2d(d_x.data(), X.data(), sizeof(double) * ld * cols,
                              workspace.h_stage, "cudaMemcpy X failed");
        }

        DeviceBuffer<double>& d_weights = workspace.d_weights;
        if (!unit_weights) {
            d_weights.allocate(static_cast<std::size_t>(n));
            staged_memcpy_h2d(d_weights.data(), weights->data(), sizeof(double) * n,
                              workspace.h_stage, "cudaMemcpy weights failed");
        }

        std::vector<CudaFeDevice>& fe_dev = workspace.fe_dev;
        // Keep the persistent workspace sized to EXACTLY the current number of
        // fixed effects. Previously this only grew (`<`), so after a model with
        // more FEs, a model with fewer FEs left stale extra CudaFeDevice entries
        // behind. Several solve loops iterate over fe_dev.size() (the polish
        // sweeps, the grand-acceleration alpha handling, the final demean), so
        // those stale dimensions were demeaned with a buffer (gid/sum) sized for
        // the *previous* dataset's larger n -> an out-of-bounds GPU read (the
        // rc=498 "illegal memory access" cascade) and corrupted residuals.
        // Truncating to fe_inputs.size() makes every dimension loop correct.
        if (fe_dev.size() != fe_inputs.size()) {
            fe_dev.resize(fe_inputs.size());
        }
        const std::size_t active_fes = fe_inputs.size();
        for (std::size_t idx = 0; idx < fe_inputs.size(); ++idx) {
            const auto& fe = fe_inputs[idx];
            CudaFeDevice& dev = fe_dev[idx];
            dev.num_groups = fe.num_groups;
            dev.num_levels_present = fe.num_levels_present;
            dev.is_slope = fe.is_slope;
            dev.slope_has_intercept = fe.slope_has_intercept;
            // Dispatch strategy for the accumulation kernel:
            //   1) Privatized (shared-memory partial sums) if num_groups *
            //      (1 + cols) fits in ~40 KB shared memory per block. Best for
            //      low-cardinality FEs (year, industry, etc.) where global
            //      atomic contention otherwise dominates.
            //   2) Warp-segmented otherwise: never slower than a plain atomic
            //      kernel and collapses same-group threads into one atomic.
            // The atomic-only branch is no longer selected directly; segmented
            // subsumes it on modern hardware (H100+).
            constexpr std::size_t kPrivatizedShmemBudget = 40 * 1024;  // bytes
            const std::size_t shmem_needed =
                static_cast<std::size_t>(fe.num_groups) *
                (1 + static_cast<std::size_t>(cols)) * sizeof(double);
            dev.use_privatized_sums =
                (fe.num_groups > 0 && shmem_needed <= kPrivatizedShmemBudget);
            dev.privatized_shmem_bytes = dev.use_privatized_sums ? shmem_needed : 0;
            dev.use_segmented_sums = !dev.use_privatized_sums;
            dev.gid.allocate(static_cast<std::size_t>(n));
            staged_memcpy_h2d(dev.gid.data(), fe.group_ids, sizeof(int) * n,
                              workspace.h_stage, "cudaMemcpy group ids failed");
            dev.weight_sums.allocate(static_cast<std::size_t>(fe.num_groups));
            staged_memcpy_h2d(dev.weight_sums.data(), fe.weight_sums,
                              sizeof(double) * fe.num_groups, workspace.h_stage,
                              "cudaMemcpy weight sums failed");
            if (dev.is_slope) {
                dev.slope_z.allocate(static_cast<std::size_t>(n));
                staged_memcpy_h2d(dev.slope_z.data(), fe.slope_values, sizeof(double) * n,
                                  workspace.h_stage, "cudaMemcpy slope values failed");
                dev.slope_sum_z.allocate(static_cast<std::size_t>(fe.num_groups));
                dev.slope_sum_zz.allocate(static_cast<std::size_t>(fe.num_groups));
                staged_memcpy_h2d(dev.slope_sum_z.data(), fe.slope_sum_z,
                                  sizeof(double) * fe.num_groups, workspace.h_stage,
                                  "cudaMemcpy slope sum_z failed");
                staged_memcpy_h2d(dev.slope_sum_zz.data(), fe.slope_sum_zz,
                                  sizeof(double) * fe.num_groups, workspace.h_stage,
                                  "cudaMemcpy slope sum_zz failed");
                dev.sum_zy.allocate(static_cast<std::size_t>(fe.num_groups));
                if (cols > 0) {
                    dev.sum_zx.allocate(static_cast<std::size_t>(fe.num_groups) * cols);
                } else {
                    dev.sum_zx.reset();
                }
            } else {
                dev.slope_z.reset();
                dev.slope_sum_z.reset();
                dev.slope_sum_zz.reset();
                dev.sum_zy.reset();
                dev.sum_zx.reset();
            }
            dev.sum_y.allocate(static_cast<std::size_t>(fe.num_groups));
            if (cols > 0) {
                dev.sum_x.allocate(static_cast<std::size_t>(fe.num_groups) * cols);
            }
            if (store_alphas) {
                dev.alpha_y.allocate(static_cast<std::size_t>(fe.num_groups));
                cuda_check(cudaMemset(dev.alpha_y.data(), 0,
                                      sizeof(double) * fe.num_groups),
                           "cudaMemset alpha_y failed");
                if (cols > 0) {
                    dev.alpha_x.allocate(static_cast<std::size_t>(fe.num_groups) * cols);
                    cuda_check(cudaMemset(dev.alpha_x.data(), 0,
                                          sizeof(double) * fe.num_groups * cols),
                               "cudaMemset alpha_x failed");
                } else {
                    dev.alpha_x.reset();
                }
                if (accel_store_alphas) {
                    dev.alpha_y_gx.allocate(static_cast<std::size_t>(fe.num_groups));
                    cuda_check(cudaMemset(dev.alpha_y_gx.data(), 0,
                                          sizeof(double) * fe.num_groups),
                               "cudaMemset alpha_y_gx failed");
                    dev.alpha_y_ggx.allocate(static_cast<std::size_t>(fe.num_groups));
                    cuda_check(cudaMemset(dev.alpha_y_ggx.data(), 0,
                                          sizeof(double) * fe.num_groups),
                               "cudaMemset alpha_y_ggx failed");
                    if (cols > 0) {
                        dev.alpha_x_gx.allocate(static_cast<std::size_t>(fe.num_groups) * cols);
                        cuda_check(cudaMemset(dev.alpha_x_gx.data(), 0,
                                              sizeof(double) * fe.num_groups * cols),
                                   "cudaMemset alpha_x_gx failed");
                        dev.alpha_x_ggx.allocate(static_cast<std::size_t>(fe.num_groups) * cols);
                        cuda_check(cudaMemset(dev.alpha_x_ggx.data(), 0,
                                              sizeof(double) * fe.num_groups * cols),
                                   "cudaMemset alpha_x_ggx failed");
                    } else {
                        dev.alpha_x_gx.reset();
                        dev.alpha_x_ggx.reset();
                    }
                    if (accel_grand_alphas) {
                        dev.alpha_y_acc_a.allocate(static_cast<std::size_t>(fe.num_groups));
                        dev.alpha_y_acc_b.allocate(static_cast<std::size_t>(fe.num_groups));
                        dev.alpha_y_acc_c.allocate(static_cast<std::size_t>(fe.num_groups));
                        cuda_check(cudaMemset(dev.alpha_y_acc_a.data(), 0,
                                              sizeof(double) * fe.num_groups),
                                   "cudaMemset alpha_y_acc_a failed");
                        cuda_check(cudaMemset(dev.alpha_y_acc_b.data(), 0,
                                              sizeof(double) * fe.num_groups),
                                   "cudaMemset alpha_y_acc_b failed");
                        cuda_check(cudaMemset(dev.alpha_y_acc_c.data(), 0,
                                              sizeof(double) * fe.num_groups),
                                   "cudaMemset alpha_y_acc_c failed");
                        if (cols > 0) {
                            dev.alpha_x_acc_a.allocate(
                                static_cast<std::size_t>(fe.num_groups) * cols);
                            dev.alpha_x_acc_b.allocate(
                                static_cast<std::size_t>(fe.num_groups) * cols);
                            dev.alpha_x_acc_c.allocate(
                                static_cast<std::size_t>(fe.num_groups) * cols);
                            cuda_check(cudaMemset(dev.alpha_x_acc_a.data(), 0,
                                                  sizeof(double) * fe.num_groups * cols),
                                       "cudaMemset alpha_x_acc_a failed");
                            cuda_check(cudaMemset(dev.alpha_x_acc_b.data(), 0,
                                                  sizeof(double) * fe.num_groups * cols),
                                       "cudaMemset alpha_x_acc_b failed");
                            cuda_check(cudaMemset(dev.alpha_x_acc_c.data(), 0,
                                                  sizeof(double) * fe.num_groups * cols),
                                       "cudaMemset alpha_x_acc_c failed");
                        } else {
                            dev.alpha_x_acc_a.reset();
                            dev.alpha_x_acc_b.reset();
                            dev.alpha_x_acc_c.reset();
                        }
                    } else {
                        dev.alpha_y_acc_a.reset();
                        dev.alpha_y_acc_b.reset();
                        dev.alpha_y_acc_c.reset();
                        dev.alpha_x_acc_a.reset();
                        dev.alpha_x_acc_b.reset();
                        dev.alpha_x_acc_c.reset();
                    }
                } else {
                    dev.alpha_y_gx.reset();
                    dev.alpha_x_gx.reset();
                    dev.alpha_y_ggx.reset();
                    dev.alpha_x_ggx.reset();
                    dev.alpha_y_acc_a.reset();
                    dev.alpha_y_acc_b.reset();
                    dev.alpha_y_acc_c.reset();
                    dev.alpha_x_acc_a.reset();
                    dev.alpha_x_acc_b.reset();
                    dev.alpha_x_acc_c.reset();
                }
            } else {
                dev.alpha_y.reset();
                dev.alpha_x.reset();
                dev.alpha_y_gx.reset();
                dev.alpha_x_gx.reset();
                dev.alpha_y_ggx.reset();
                dev.alpha_x_ggx.reset();
                dev.alpha_y_acc_a.reset();
                dev.alpha_y_acc_b.reset();
                dev.alpha_y_acc_c.reset();
                dev.alpha_x_acc_a.reset();
                dev.alpha_x_acc_b.reset();
                dev.alpha_x_acc_c.reset();
            }
        }

        result.fe_levels.clear();
        result.fe_levels.reserve(fe_inputs.size());
        for (std::size_t dim = 0; dim < fe_inputs.size(); ++dim) {
            result.fe_levels.push_back(fe_dev[dim].num_levels_present);
        }

        AlphaStatePtrs alpha_state_ptrs;
        AlphaStatePtrs alpha_gx_ptrs;
        AlphaStatePtrs alpha_ggx_ptrs;
        AlphaStatePtrs alpha_acc_a_ptrs;
        AlphaStatePtrs alpha_acc_b_ptrs;
        AlphaStatePtrs alpha_acc_c_ptrs;
        if (store_alphas) {
            const std::size_t dims = fe_inputs.size();
            alpha_state_ptrs.y.assign(dims, nullptr);
            alpha_state_ptrs.x.assign(dims, nullptr);
            for (std::size_t dim = 0; dim < dims; ++dim) {
                alpha_state_ptrs.y[dim] = fe_dev[dim].alpha_y.data();
                alpha_state_ptrs.x[dim] = cols > 0 ? fe_dev[dim].alpha_x.data() : nullptr;
            }
            if (accel_store_alphas) {
                alpha_gx_ptrs.y.assign(dims, nullptr);
                alpha_gx_ptrs.x.assign(dims, nullptr);
                alpha_ggx_ptrs.y.assign(dims, nullptr);
                alpha_ggx_ptrs.x.assign(dims, nullptr);
                for (std::size_t dim = 0; dim < dims; ++dim) {
                    alpha_gx_ptrs.y[dim] = fe_dev[dim].alpha_y_gx.data();
                    alpha_gx_ptrs.x[dim] = cols > 0 ? fe_dev[dim].alpha_x_gx.data() : nullptr;
                    alpha_ggx_ptrs.y[dim] = fe_dev[dim].alpha_y_ggx.data();
                    alpha_ggx_ptrs.x[dim] = cols > 0 ? fe_dev[dim].alpha_x_ggx.data() : nullptr;
                }
                if (accel_grand_alphas) {
                    alpha_acc_a_ptrs.y.assign(dims, nullptr);
                    alpha_acc_a_ptrs.x.assign(dims, nullptr);
                    alpha_acc_b_ptrs.y.assign(dims, nullptr);
                    alpha_acc_b_ptrs.x.assign(dims, nullptr);
                    alpha_acc_c_ptrs.y.assign(dims, nullptr);
                    alpha_acc_c_ptrs.x.assign(dims, nullptr);
                    for (std::size_t dim = 0; dim < dims; ++dim) {
                        alpha_acc_a_ptrs.y[dim] = fe_dev[dim].alpha_y_acc_a.data();
                        alpha_acc_a_ptrs.x[dim] = cols > 0 ? fe_dev[dim].alpha_x_acc_a.data()
                                                           : nullptr;
                        alpha_acc_b_ptrs.y[dim] = fe_dev[dim].alpha_y_acc_b.data();
                        alpha_acc_b_ptrs.x[dim] = cols > 0 ? fe_dev[dim].alpha_x_acc_b.data()
                                                           : nullptr;
                        alpha_acc_c_ptrs.y[dim] = fe_dev[dim].alpha_y_acc_c.data();
                        alpha_acc_c_ptrs.x[dim] = cols > 0 ? fe_dev[dim].alpha_x_acc_c.data()
                                                           : nullptr;
                    }
                }
            }
        }

	        DeviceBuffer<double>& d_sumsq = workspace.d_sumsq;
	        d_sumsq.allocate(1);
	        const bool use_update_error =
	            needs_reghdfe_update_check(options) || mixed_update_criterion;
	        DeviceBuffer<double>& d_y_prev_update = workspace.d_y_prev_update;
	        DeviceBuffer<double>& d_x_prev_update = workspace.d_x_prev_update;
	        DeviceBuffer<double>& d_update_sums = workspace.d_update_sums;
	        if (use_update_error) {
	            d_y_prev_update.allocate(static_cast<std::size_t>(n));
	            if (cols > 0) {
	                d_x_prev_update.allocate(x_size);
	            }
	            d_update_sums.allocate(static_cast<std::size_t>(cols + 2));
	        } else if (cuda_cg_fallback_ok) {
	            // The CG divergence fallback uses d_update_sums (size cg_width =
	            // cols+1) for its per-column reductions; allocate it even when the
	            // reghdfe update check is not otherwise active.
	            d_update_sums.allocate(static_cast<std::size_t>(cols + 2));
	        }

	        auto compute_sumsq = [&](const double* y_ptr, const double* x_ptr, double& out) {
            cuda_check(cudaMemset(d_sumsq.data(), 0, sizeof(double)), "cudaMemset sumsq failed");
            const int blocks = (n + kBlockSize - 1) / kBlockSize;
            sumsq_kernel<kBlockSize><<<blocks, kBlockSize>>>(
                y_ptr, x_ptr, n, cols, ld, d_sumsq.data());
            cuda_check(cudaGetLastError(), "sumsq kernel launch failed");
            cuda_check(cudaMemcpy(&out, d_sumsq.data(), sizeof(double), cudaMemcpyDeviceToHost),
	                       "cudaMemcpy sumsq failed");
	        };

	        auto copy_update_snapshot = [&](const double* y_ptr, const double* x_ptr) {
	            if (!use_update_error) {
	                return;
	            }
	            cuda_check(cudaMemcpy(d_y_prev_update.data(), y_ptr, sizeof(double) * n,
	                                  cudaMemcpyDeviceToDevice),
	                       "cudaMemcpy y convergence snapshot failed");
	            if (cols > 0) {
	                cuda_check(cudaMemcpy(d_x_prev_update.data(), x_ptr, sizeof(double) * x_size,
	                                      cudaMemcpyDeviceToDevice),
	                           "cudaMemcpy X convergence snapshot failed");
	            }
	        };

	        auto compute_update_error = [&](const double* y_new,
	                                        const double* x_new,
	                                        double& out) {
	            out = std::numeric_limits<double>::max();
	            if (!use_update_error) {
	                return;
	            }
	            const int width = cols + 1;
	            cuda_check(cudaMemset(d_update_sums.data(), 0,
	                                  sizeof(double) * static_cast<std::size_t>(width + 1)),
	                       "cudaMemset update sums failed");
	            const int blocks = (n + kBlockSize - 1) / kBlockSize;
	            const dim3 grid(blocks, width + 1);
	            reldif_update_sums_kernel<kBlockSize><<<grid, kBlockSize>>>(
	                y_new,
	                d_y_prev_update.data(),
	                cols > 0 ? x_new : nullptr,
	                cols > 0 ? d_x_prev_update.data() : nullptr,
	                unit_weights ? nullptr : d_weights.data(),
	                unit_weights,
	                n,
	                cols,
	                ld,
	                d_update_sums.data());
	            cuda_check(cudaGetLastError(), "reldif update kernel launch failed");
	            std::vector<double> host(static_cast<std::size_t>(width + 1), 0.0);
	            cuda_check(cudaMemcpy(host.data(), d_update_sums.data(),
	                                  sizeof(double) * static_cast<std::size_t>(width + 1),
	                                  cudaMemcpyDeviceToHost),
	                       "cudaMemcpy update sums failed");
	            const double denom = host[static_cast<std::size_t>(width)];
	            if (!(denom > 0.0)) {
	                throw std::runtime_error("Weights must sum to a positive value");
	            }
	            double max_mean = 0.0;
	            for (int j = 0; j < width; ++j) {
	                max_mean = std::max(max_mean, host[static_cast<std::size_t>(j)] / denom);
	            }
	            out = max_mean;
	        };

	        auto accumulate_alpha = [&](CudaFeDevice& fe,
                                    double scale,
                                    double* alpha_y_ptr,
                                    double* alpha_x_ptr) {
            if (!alpha_y_ptr && !alpha_x_ptr) {
                return;
            }
            const int blocks_g = (fe.num_groups + kBlockSize - 1) / kBlockSize;
            accumulate_alpha_kernel<<<blocks_g, kBlockSize>>>(
                fe.sum_y.data(),
                fe.sum_x.data(),
                alpha_y_ptr,
                alpha_x_ptr,
                fe.num_groups,
                cols,
                scale);
            cuda_check(cudaGetLastError(), "accumulate alpha kernel launch failed");
        };

        auto compute_means_only = [&](CudaFeDevice& fe,
                                      double* y_ptr,
                                      double* x_ptr,
                                      double alpha_scale,
                                      double* alpha_y_ptr,
                                      double* alpha_x_ptr) {
            const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
            const int blocks_g = (fe.num_groups + kBlockSize - 1) / kBlockSize;
            cuda_check(cudaMemset(fe.sum_y.data(), 0, sizeof(double) * fe.num_groups),
                       "cudaMemset sum_y failed");
            if (cols > 0) {
                cuda_check(cudaMemset(fe.sum_x.data(), 0,
                                      sizeof(double) * fe.num_groups * cols),
                           "cudaMemset sum_x failed");
            }

            if (fe.is_slope) {
                cuda_check(cudaMemset(fe.sum_zy.data(), 0, sizeof(double) * fe.num_groups),
                           "cudaMemset sum_zy failed");
                if (cols > 0) {
                    cuda_check(cudaMemset(fe.sum_zx.data(), 0,
                                          sizeof(double) * fe.num_groups * cols),
                               "cudaMemset sum_zx failed");
                }
                accumulate_slope_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    fe.slope_z.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_zy.data(),
                    fe.sum_x.data(),
                    fe.sum_zx.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_slope_sums kernel launch failed");

                compute_slope_coefficients_kernel<<<blocks_g, kBlockSize>>>(
                    fe.sum_y.data(),
                    fe.sum_zy.data(),
                    fe.sum_x.data(),
                    fe.sum_zx.data(),
                    fe.weight_sums.data(),
                    fe.slope_sum_z.data(),
                    fe.slope_sum_zz.data(),
                    fe.num_groups,
                    cols,
                    fe.slope_has_intercept);
                cuda_check(cudaGetLastError(),
                           "compute_slope_coefficients kernel launch failed");
                return;
            }

            if (fe.use_privatized_sums) {
                accumulate_sums_privatized_kernel<<<blocks_n, kBlockSize,
                                                     fe.privatized_shmem_bytes>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_sums (privatized) kernel launch failed");
            } else if (fe.use_segmented_sums) {
                accumulate_sums_segmented_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_sums (segmented) kernel launch failed");
            } else {
                accumulate_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(), "accumulate_sums (atomic) kernel launch failed");
            }

            compute_means_kernel<<<blocks_g, kBlockSize>>>(
                fe.sum_y.data(),
                fe.sum_x.data(),
                fe.weight_sums.data(),
                fe.num_groups,
                cols);
            cuda_check(cudaGetLastError(), "compute_means kernel launch failed");
            accumulate_alpha(fe, alpha_scale, alpha_y_ptr, alpha_x_ptr);
        };

        auto run_demean = [&](CudaFeDevice& fe,
                              double* y_ptr,
                              double* x_ptr,
                              bool compute_check,
                              double& sumsq_out,
                              double alpha_scale,
                              double* alpha_y_ptr,
                              double* alpha_x_ptr) {
            const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
            const int blocks_g = (fe.num_groups + kBlockSize - 1) / kBlockSize;
            cuda_check(cudaMemset(fe.sum_y.data(), 0, sizeof(double) * fe.num_groups),
                       "cudaMemset sum_y failed");
            if (cols > 0) {
                cuda_check(cudaMemset(fe.sum_x.data(), 0,
                                      sizeof(double) * fe.num_groups * cols),
                           "cudaMemset sum_x failed");
            }

            if (fe.is_slope) {
                cuda_check(cudaMemset(fe.sum_zy.data(), 0, sizeof(double) * fe.num_groups),
                           "cudaMemset sum_zy failed");
                if (cols > 0) {
                    cuda_check(cudaMemset(fe.sum_zx.data(), 0,
                                          sizeof(double) * fe.num_groups * cols),
                               "cudaMemset sum_zx failed");
                }
                accumulate_slope_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    fe.slope_z.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_zy.data(),
                    fe.sum_x.data(),
                    fe.sum_zx.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_slope_sums kernel launch failed");

                compute_slope_coefficients_kernel<<<blocks_g, kBlockSize>>>(
                    fe.sum_y.data(),
                    fe.sum_zy.data(),
                    fe.sum_x.data(),
                    fe.sum_zx.data(),
                    fe.weight_sums.data(),
                    fe.slope_sum_z.data(),
                    fe.slope_sum_zz.data(),
                    fe.num_groups,
                    cols,
                    fe.slope_has_intercept);
                cuda_check(cudaGetLastError(),
                           "compute_slope_coefficients kernel launch failed");

                if (compute_check) {
                    cuda_check(cudaMemset(d_sumsq.data(), 0, sizeof(double)),
                               "cudaMemset sumsq failed");
                    apply_slope_sumsq_kernel<kBlockSize><<<blocks_n, kBlockSize>>>(
                        y_ptr,
                        x_ptr,
                        n,
                        cols,
                        ld,
                        fe.gid.data(),
                        fe.slope_z.data(),
                        fe.sum_y.data(),
                        fe.sum_zy.data(),
                        fe.sum_x.data(),
                        fe.sum_zx.data(),
                        fe.num_groups,
                        d_sumsq.data());
                    cuda_check(cudaGetLastError(),
                               "apply_slope_sumsq kernel launch failed");
                    cuda_check(cudaMemcpy(&sumsq_out, d_sumsq.data(), sizeof(double),
                                          cudaMemcpyDeviceToHost),
                               "cudaMemcpy sumsq failed");
                } else {
                    apply_slope_kernel<<<blocks_n, kBlockSize>>>(
                        y_ptr,
                        x_ptr,
                        n,
                        cols,
                        ld,
                        fe.gid.data(),
                        fe.slope_z.data(),
                        fe.sum_y.data(),
                        fe.sum_zy.data(),
                        fe.sum_x.data(),
                        fe.sum_zx.data(),
                        fe.num_groups);
                    cuda_check(cudaGetLastError(), "apply_slope kernel launch failed");
                }
                return;
            }

            if (fe.use_privatized_sums) {
                accumulate_sums_privatized_kernel<<<blocks_n, kBlockSize,
                                                     fe.privatized_shmem_bytes>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_sums (privatized) kernel launch failed");
            } else if (fe.use_segmented_sums) {
                accumulate_sums_segmented_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_sums (segmented) kernel launch failed");
            } else {
                accumulate_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(), "accumulate_sums (atomic) kernel launch failed");
            }

            compute_means_kernel<<<blocks_g, kBlockSize>>>(
                fe.sum_y.data(),
                fe.sum_x.data(),
                fe.weight_sums.data(),
                fe.num_groups,
                cols);
            cuda_check(cudaGetLastError(), "compute_means kernel launch failed");
            accumulate_alpha(fe, alpha_scale, alpha_y_ptr, alpha_x_ptr);

            if (compute_check) {
                cuda_check(cudaMemset(d_sumsq.data(), 0, sizeof(double)),
                           "cudaMemset sumsq failed");
                apply_means_sumsq_kernel<kBlockSize><<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups,
                    d_sumsq.data());
                cuda_check(cudaGetLastError(), "apply_means_sumsq kernel launch failed");
                cuda_check(cudaMemcpy(&sumsq_out, d_sumsq.data(), sizeof(double),
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy sumsq failed");
            } else {
                apply_means_kernel<<<blocks_n, kBlockSize>>>(
                    y_ptr,
                    x_ptr,
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(), "apply_means kernel launch failed");
            }
        };

        const int dims = static_cast<int>(active_fes);
        if (method == AbsorptionMethod::Jacobi) {
            std::vector<const int*> gid_ptrs(static_cast<std::size_t>(dims));
            std::vector<const double*> mean_y_ptrs(static_cast<std::size_t>(dims));
            std::vector<const double*> mean_x_ptrs(static_cast<std::size_t>(dims));
            std::vector<int> groups(static_cast<std::size_t>(dims));
            for (int d = 0; d < dims; ++d) {
                gid_ptrs[static_cast<std::size_t>(d)] = fe_dev[d].gid.data();
                mean_y_ptrs[static_cast<std::size_t>(d)] = fe_dev[d].sum_y.data();
                mean_x_ptrs[static_cast<std::size_t>(d)] =
                    cols > 0 ? fe_dev[d].sum_x.data() : nullptr;
                groups[static_cast<std::size_t>(d)] = fe_dev[d].num_groups;
            }

            DeviceBuffer<const int*> d_gid_ptrs(static_cast<std::size_t>(dims));
            DeviceBuffer<const double*> d_mean_y_ptrs(static_cast<std::size_t>(dims));
            DeviceBuffer<const double*> d_mean_x_ptrs(static_cast<std::size_t>(dims));
            DeviceBuffer<int> d_groups(static_cast<std::size_t>(dims));

            cuda_check(cudaMemcpy(d_gid_ptrs.data(), gid_ptrs.data(),
                                  sizeof(const int*) * dims, cudaMemcpyHostToDevice),
                       "cudaMemcpy gid ptrs failed");
            cuda_check(cudaMemcpy(d_mean_y_ptrs.data(), mean_y_ptrs.data(),
                                  sizeof(const double*) * dims, cudaMemcpyHostToDevice),
                       "cudaMemcpy mean y ptrs failed");
            if (cols > 0) {
                cuda_check(cudaMemcpy(d_mean_x_ptrs.data(), mean_x_ptrs.data(),
                                      sizeof(const double*) * dims, cudaMemcpyHostToDevice),
                           "cudaMemcpy mean x ptrs failed");
            }
            cuda_check(cudaMemcpy(d_groups.data(), groups.data(), sizeof(int) * dims,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy groups failed");

            double relaxation = options.jacobi_relaxation;
            if (relaxation <= 0.0) {
                relaxation = 2.0 / (static_cast<double>(dims) + 1.0);
            }
            relaxation = std::min(relaxation, 1.0);

	            double prev_norm = 0.0;
	            compute_sumsq(d_y.data(), d_x.data(), prev_norm);
	            prev_norm = std::sqrt(prev_norm);
	            copy_update_snapshot(d_y.data(), d_x.data());
	            const int check_interval = std::max(1, options.convergence_check_interval);
            int last_check_iter = -1;
            bool converged = false;

            for (int iter = 0; iter < options.max_iter; ++iter) {
                for (int d = 0; d < dims; ++d) {
                    double* alpha_y_ptr = nullptr;
                    double* alpha_x_ptr = nullptr;
                    if (store_alphas) {
                        alpha_y_ptr = alpha_state_ptrs.y[static_cast<std::size_t>(d)];
                        alpha_x_ptr = alpha_state_ptrs.x[static_cast<std::size_t>(d)];
                    }
                    compute_means_only(fe_dev[d], d_y.data(), d_x.data(), relaxation,
                                       alpha_y_ptr, alpha_x_ptr);
                }

                const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
                jacobi_update_kernel<<<blocks_n, kBlockSize>>>(
                    d_y.data(),
                    d_x.data(),
                    n,
                    cols,
                    ld,
                    d_gid_ptrs.data(),
                    d_mean_y_ptrs.data(),
                    cols > 0 ? d_mean_x_ptrs.data() : nullptr,
                    d_groups.data(),
                    dims,
                    relaxation);
                cuda_check(cudaGetLastError(), "jacobi update kernel launch failed");

                const bool do_check =
                    (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
                if (do_check) {
                    double curr_norm = 0.0;
                    compute_sumsq(d_y.data(), d_x.data(), curr_norm);
                    curr_norm = std::sqrt(curr_norm);
                    const double denom = std::max(1.0, prev_norm);
                    const double rel_change = std::abs(curr_norm - prev_norm) / denom;
	                    const int step = iter - last_check_iter;
	                    last_check_iter = iter;
	                    prev_norm = curr_norm;
	                    double update_error = std::numeric_limits<double>::max();
	                    compute_update_error(d_y.data(), d_x.data(), update_error);
	                    copy_update_snapshot(d_y.data(), d_x.data());
	                    const double norm_error = rel_change / static_cast<double>(step);
	                    if (step > 0 &&
	                        convergence_reached(options, norm_error, update_error, convergence_tol)) {
	                        result.iterations = iter + 1;
                        converged = true;
                        break;
                    }
                }
            }

            result.converged = converged;
            if (!converged) {
                result.iterations = options.max_iter;
            } else if (result.iterations == 0) {
                result.iterations = 1;
            }
        } else {
            bool use_symmetric = options.symmetric_sweep && sweep_order.size() > 1;
            if (use_reghdfe_cg) {
                use_symmetric = true;
            }

            std::vector<FeProfileStats> fe_profiles;
            fe_profiles.reserve(fe_inputs.size());
            for (const auto& fe : fe_inputs) {
                fe_profiles.push_back(profile_fe(fe, unit_weights));
            }

            const FeDifficulty difficulty = classify_problem(fe_profiles);
            bool enable_two_stage = false;
            std::vector<std::size_t> two_fe_order;
            int iter_warmup = 0;
            int iter_two_fe = 0;
            if (fe_profiles.size() >= 3 && n >= 200000) {
                int first = -1;
                int second = -1;
                int third = -1;
                for (const auto& p : fe_profiles) {
                    const int groups = p.num_groups;
                    if (groups > first) {
                        third = second;
                        second = first;
                        first = groups;
                    } else if (groups > second) {
                        third = second;
                        second = groups;
                    } else if (groups > third) {
                        third = groups;
                    }
                }
                if (first > 0 && third > 0) {
                    constexpr double kTwoStageThirdRatioMax = 0.05;
                    if (static_cast<double>(third) <=
                        kTwoStageThirdRatioMax * static_cast<double>(first)) {
                        enable_two_stage = true;
                        if (difficulty == FeDifficulty::Extreme) {
                            iter_warmup = 5;
                            iter_two_fe = 5;
                        } else if (difficulty >= FeDifficulty::Hard) {
                            iter_warmup = 8;
                            iter_two_fe = 8;
                        } else {
                            iter_warmup = 5;
                            iter_two_fe = 8;
                        }
                    }
                }
            }
            if (enable_two_stage) {
                const std::vector<std::size_t> two_dims = select_two_largest_dims(fe_profiles);
                two_fe_order = filter_order(sweep_order, two_dims);
                if (two_fe_order.size() != 2) {
                    enable_two_stage = false;
                    iter_warmup = 0;
                    iter_two_fe = 0;
                }
            }

            bool converged = false;

            auto compute_norm = [&]() {
                double norm = 0.0;
                compute_sumsq(d_y.data(), d_x.data(), norm);
                return std::sqrt(norm);
            };

            auto run_sweep = [&](double* y_ptr,
                                 double* x_ptr,
                                 const std::vector<std::size_t>& order,
                                 bool compute_check,
                                 double& sumsq_out,
                                 const AlphaStatePtrs* alpha_ptrs) {
                if (order.empty()) {
                    if (compute_check) {
                        sumsq_out = 0.0;
                    }
                    return;
                }
                double sumsq = 0.0;
                const bool do_symmetric = use_symmetric && order.size() > 1;
                for (std::size_t pos = 0; pos < order.size(); ++pos) {
                    const std::size_t dim = order[pos];
                    const bool last_call =
                        compute_check && !do_symmetric && (pos + 1 == order.size());
                    double* alpha_y_ptr = nullptr;
                    double* alpha_x_ptr = nullptr;
                    if (alpha_ptrs) {
                        alpha_y_ptr = alpha_ptrs->y[dim];
                        alpha_x_ptr = alpha_ptrs->x[dim];
                    }
                    run_demean(fe_dev[dim], y_ptr, x_ptr, last_call, sumsq, 1.0,
                               alpha_y_ptr, alpha_x_ptr);
                }
                if (do_symmetric) {
                    for (std::size_t idx = order.size(); idx-- > 0;) {
                        const std::size_t dim = order[idx];
                        const bool last_call = compute_check && (idx == 0);
                        double* alpha_y_ptr = nullptr;
                        double* alpha_x_ptr = nullptr;
                        if (alpha_ptrs) {
                            alpha_y_ptr = alpha_ptrs->y[dim];
                            alpha_x_ptr = alpha_ptrs->x[dim];
                        }
                        run_demean(fe_dev[dim], y_ptr, x_ptr, last_call, sumsq, 1.0,
                                   alpha_y_ptr, alpha_x_ptr);
                    }
                }
                if (compute_check) {
                    sumsq_out = sumsq;
                }
            };

            const int check_interval = std::max(1, options.convergence_check_interval);

            if (use_accel) {
                DeviceBuffer<double>& d_y_gx = workspace.d_y_gx;
                DeviceBuffer<double>& d_y_ggx = workspace.d_y_ggx;
                DeviceBuffer<double>& d_x_gx = workspace.d_x_gx;
                DeviceBuffer<double>& d_x_ggx = workspace.d_x_ggx;
                d_y_gx.allocate(static_cast<std::size_t>(n));
                d_y_ggx.allocate(static_cast<std::size_t>(n));
                if (cols > 0) {
                    d_x_gx.allocate(x_size);
                    d_x_ggx.allocate(x_size);
                }

                DeviceBuffer<double>& d_stats = workspace.d_stats;
                d_stats.allocate(3);
                DeviceBuffer<double>& d_cg_alpha = workspace.d_cg_alpha;
                DeviceBuffer<double>& d_cg_beta = workspace.d_cg_beta;
                const int cg_width = cols + 1;
                if (use_reghdfe_cg || cuda_cg_fallback_ok) {
                    // Allocate CG scalar buffers when CG can run either as the
                    // explicit accelerator or as the IT divergence fallback.
                    d_cg_alpha.allocate(static_cast<std::size_t>(cg_width));
                    d_cg_beta.allocate(static_cast<std::size_t>(cg_width));
                }
                const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
                const int blocks_x = (static_cast<int>(x_size) + kBlockSize - 1) / kBlockSize;
                const std::size_t total_values = static_cast<std::size_t>(n) + x_size;
                const int blocks_values =
                    static_cast<int>((total_values + kBlockSize - 1) / kBlockSize);
                const int iter_grand_acc = kGrandAccelInterval;

                DeviceBuffer<double>& d_y_acc_a = workspace.d_y_acc_a;
                DeviceBuffer<double>& d_y_acc_b = workspace.d_y_acc_b;
                DeviceBuffer<double>& d_y_acc_c = workspace.d_y_acc_c;
                DeviceBuffer<double>& d_x_acc_a = workspace.d_x_acc_a;
                DeviceBuffer<double>& d_x_acc_b = workspace.d_x_acc_b;
                DeviceBuffer<double>& d_x_acc_c = workspace.d_x_acc_c;
                if (iter_grand_acc > 0) {
                    d_y_acc_a.allocate(static_cast<std::size_t>(n));
                    d_y_acc_b.allocate(static_cast<std::size_t>(n));
                    d_y_acc_c.allocate(static_cast<std::size_t>(n));
                    if (cols > 0) {
                        d_x_acc_a.allocate(x_size);
                        d_x_acc_b.allocate(x_size);
                        d_x_acc_c.allocate(x_size);
                    }
                }

                auto copy_alpha_state = [&](const AlphaStatePtrs& dst,
                                            const AlphaStatePtrs& src) {
                    for (std::size_t dim = 0; dim < active_fes; ++dim) {
                        if (!dst.y[dim] || !src.y[dim]) {
                            continue;
                        }
                        const int groups = fe_dev[dim].num_groups;
                        cuda_check(cudaMemcpy(dst.y[dim], src.y[dim],
                                              sizeof(double) * groups,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy alpha_y state failed");
                        if (cols > 0 && dst.x[dim] && src.x[dim]) {
                            const std::size_t count =
                                static_cast<std::size_t>(groups) * static_cast<std::size_t>(cols);
                            cuda_check(cudaMemcpy(dst.x[dim], src.x[dim],
                                                  sizeof(double) * count,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy alpha_x state failed");
                        }
                    }
                };

                auto lincomb_alpha_state = [&](const AlphaStatePtrs& out,
                                               const AlphaStatePtrs& a,
                                               const AlphaStatePtrs& b,
                                               double coef) {
                    for (std::size_t dim = 0; dim < active_fes; ++dim) {
                        if (!out.y[dim] || !a.y[dim] || !b.y[dim]) {
                            continue;
                        }
                        const int groups = fe_dev[dim].num_groups;
                        const int blocks_g = (groups + kBlockSize - 1) / kBlockSize;
                        irons_tuck_update_kernel<<<blocks_g, kBlockSize>>>(
                            out.y[dim], a.y[dim], b.y[dim], groups, coef);
                        cuda_check(cudaGetLastError(),
                                   "alpha lincomb kernel launch failed");
                        if (cols > 0 && out.x[dim] && a.x[dim] && b.x[dim]) {
                            const std::size_t count =
                                static_cast<std::size_t>(groups) * static_cast<std::size_t>(cols);
                            const int blocks_x =
                                (static_cast<int>(count) + kBlockSize - 1) / kBlockSize;
                            irons_tuck_update_kernel<<<blocks_x, kBlockSize>>>(
                                out.x[dim], a.x[dim], b.x[dim],
                                static_cast<int>(count), coef);
                            cuda_check(cudaGetLastError(),
                                       "alpha lincomb X kernel launch failed");
                        }
                    }
                };

                auto run_accel_phase = [&](const std::vector<std::size_t>& order,
                                           int max_iter,
                                           int& iter_used) -> bool {
                    iter_used = 0;
                    if (max_iter <= 0 || order.empty()) {
                        return false;
	                    }
	                    double prev_norm = compute_norm();
	                    copy_update_snapshot(d_y.data(), d_x.data());
	                    int last_check_iter = -1;
	                    int grand_acc = 0;
                    // Adaptive-restart safeguard against Irons-Tuck divergence
                    // on ill-conditioned FE graphs (mirrors the CPU packed/SoA
                    // accel loops in fe_absorption.cpp). No-op on
                    // well-conditioned data (monotone iterate norm).
                    constexpr double kAccelDivergeFactor = 4.0;
                    double best_resid = prev_norm;
                    bool accel_suspended = false;
                    int accel_restart_count = 0;
                    double max_ratio_seen = 1.0;

                    cuda_check(cudaMemcpy(d_y_gx.data(), d_y.data(), sizeof(double) * n,
                                          cudaMemcpyDeviceToDevice),
                               "cudaMemcpy y_gx failed");
                    if (cols > 0) {
                        cuda_check(cudaMemcpy(d_x_gx.data(), d_x.data(),
                                              sizeof(double) * x_size,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy x_gx failed");
                    }
                    if (store_alphas) {
                        copy_alpha_state(alpha_gx_ptrs, alpha_state_ptrs);
                    }
                    double ignored = 0.0;
                    run_sweep(d_y_gx.data(), d_x_gx.data(), order, false, ignored,
                              store_alphas ? &alpha_gx_ptrs : nullptr);

                    for (int iter = 0; iter < max_iter; ++iter) {
                        const bool do_check = (check_interval == 1 || iter < check_interval ||
                                               iter % check_interval == 0);
                        cuda_check(cudaMemcpy(d_y_ggx.data(), d_y_gx.data(), sizeof(double) * n,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy y_ggx failed");
                        if (cols > 0) {
                            cuda_check(cudaMemcpy(d_x_ggx.data(), d_x_gx.data(),
                                                  sizeof(double) * x_size,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy x_ggx failed");
                        }
                        if (store_alphas) {
                            copy_alpha_state(alpha_ggx_ptrs, alpha_gx_ptrs);
                        }
                        run_sweep(d_y_ggx.data(), d_x_ggx.data(), order, false, ignored,
                                  store_alphas ? &alpha_ggx_ptrs : nullptr);

                        cuda_check(cudaMemset(d_stats.data(), 0, sizeof(double) * 3),
                                   "cudaMemset stats failed");
                        irons_tuck_stats_kernel<kBlockSize><<<blocks_n, kBlockSize>>>(
                            d_y.data(), d_y_gx.data(), d_y_ggx.data(), n, d_stats.data());
                        cuda_check(cudaGetLastError(), "irons tuck stats kernel launch failed");
                        if (cols > 0) {
                            irons_tuck_stats_kernel<kBlockSize><<<blocks_x, kBlockSize>>>(
                                d_x.data(), d_x_gx.data(), d_x_ggx.data(),
                                static_cast<int>(x_size), d_stats.data());
                            cuda_check(cudaGetLastError(),
                                       "irons tuck stats X kernel launch failed");
                        }

                        double stats_host[3] = {0.0, 0.0, 0.0};
                        cuda_check(cudaMemcpy(stats_host, d_stats.data(), sizeof(double) * 3,
                                              cudaMemcpyDeviceToHost),
                                   "cudaMemcpy stats failed");
                        const double vprod = stats_host[0];
                        const double ssq = stats_host[1];
                        if (honest_mode && !mixed_update_criterion &&
                            std::sqrt(stats_host[2]) <=
                                honest_tol * std::max(1.0, prev_norm)) {
                            // One full sweep moved the iterate by less than
                            // tol: return ggx, the closest available iterate.
                            cuda_check(cudaMemcpy(d_y.data(), d_y_ggx.data(),
                                                  sizeof(double) * n,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy y honest-exit failed");
                            if (cols > 0) {
                                cuda_check(cudaMemcpy(d_x.data(), d_x_ggx.data(),
                                                      sizeof(double) * x_size,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy x honest-exit failed");
                            }
                            if (store_alphas) {
                                copy_alpha_state(alpha_state_ptrs, alpha_ggx_ptrs);
                            }
                            iter_used = iter + 1;
                            if (std::getenv("XHDFE_ACCEL_DEBUG") != nullptr) {
                                std::fprintf(stderr,
                                             "[ACCEL cuda CONVERGED iter_used=%d restarts=%d]\n",
                                             iter_used, accel_restart_count);
                            }
                            return true;
                        }
                        if (ssq == 0.0) {
                            cuda_check(cudaMemcpy(d_y.data(), d_y_gx.data(), sizeof(double) * n,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy y fallback failed");
                            if (cols > 0) {
                                cuda_check(cudaMemcpy(d_x.data(), d_x_gx.data(),
                                                      sizeof(double) * x_size,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy x fallback failed");
                            }
                            if (store_alphas) {
                                copy_alpha_state(alpha_state_ptrs, alpha_gx_ptrs);
                            }
                            iter_used = iter + 1;
                            return true;
                        }

                        double coef = vprod / ssq;
                        if (accel_suspended) {
                            // Diverging: take the plain (firmly non-expansive)
                            // sweep step instead of the IT extrapolation.
                            coef = 0.0;
                        }
                        irons_tuck_update_kernel<<<blocks_n, kBlockSize>>>(
                            d_y.data(), d_y_gx.data(), d_y_ggx.data(), n, coef);
                        cuda_check(cudaGetLastError(), "irons tuck update kernel launch failed");
                        if (cols > 0) {
                            irons_tuck_update_kernel<<<blocks_x, kBlockSize>>>(
                                d_x.data(), d_x_gx.data(), d_x_ggx.data(),
                                static_cast<int>(x_size), coef);
                            cuda_check(cudaGetLastError(),
                                       "irons tuck update X kernel launch failed");
                        }
                        if (store_alphas) {
                            lincomb_alpha_state(alpha_state_ptrs, alpha_gx_ptrs, alpha_ggx_ptrs,
                                                coef);
                        }

                        cuda_check(cudaMemcpy(d_y_gx.data(), d_y.data(), sizeof(double) * n,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy y_gx update failed");
                        if (cols > 0) {
                            cuda_check(cudaMemcpy(d_x_gx.data(), d_x.data(),
                                                  sizeof(double) * x_size,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy x_gx update failed");
                        }
                        if (store_alphas) {
                            copy_alpha_state(alpha_gx_ptrs, alpha_state_ptrs);
                        }

                        double sumsq = 0.0;
                        run_sweep(d_y_gx.data(), d_x_gx.data(), order, do_check, sumsq,
                                  store_alphas ? &alpha_gx_ptrs : nullptr);
                        if (do_check) {
                            const double curr_norm = std::sqrt(sumsq);
                            const double denom = std::max(1.0, prev_norm);
                            const double rel_change = std::abs(curr_norm - prev_norm) / denom;
                            const int step = iter - last_check_iter;
                            last_check_iter = iter;
                            prev_norm = curr_norm;
                            if (honest_mode) {
                                const double ratio =
                                    curr_norm / std::max(best_resid, 1e-300);
                                if (ratio > max_ratio_seen) {
                                    max_ratio_seen = ratio;
                                }
                                if (curr_norm <= best_resid) {
                                    best_resid = curr_norm;
                                    accel_suspended = false;
                                } else {
                                    if (cuda_cg_fallback_ok &&
                                        curr_norm > cg_bail_factor * best_resid) {
                                        // Diverging: hand the post-sweep iterate
                                        // (d_*_gx) to the stable block-CG.
                                        cuda_check(cudaMemcpy(d_y.data(), d_y_gx.data(),
                                                              sizeof(double) * n,
                                                              cudaMemcpyDeviceToDevice),
                                                   "cudaMemcpy y cg-bail failed");
                                        if (cols > 0) {
                                            cuda_check(cudaMemcpy(d_x.data(), d_x_gx.data(),
                                                                  sizeof(double) * x_size,
                                                                  cudaMemcpyDeviceToDevice),
                                                       "cudaMemcpy x cg-bail failed");
                                        }
                                        accel_diverged = true;
                                        iter_used = iter + 1;
                                        return false;
                                    }
                                    if (curr_norm > kAccelDivergeFactor * best_resid) {
                                        if (!accel_suspended) {
                                            ++accel_restart_count;
                                            grand_acc = 0;
                                        }
                                        accel_suspended = true;
                                    }
                                }
                            }
                            if (mixed_update_criterion && step > 0) {
                                // Mirror of the CPU mixed absorber comparable
                                // exit: weighted mean reldif of the post-sweep
                                // iterate vs the previous check <= nominal tol.
                                double update_error =
                                    std::numeric_limits<double>::max();
                                compute_update_error(d_y_gx.data(), d_x_gx.data(),
                                                     update_error);
                                copy_update_snapshot(d_y_gx.data(), d_x_gx.data());
                                if (update_error <= options.tol) {
                                    cuda_check(cudaMemcpy(d_y.data(), d_y_gx.data(),
                                                          sizeof(double) * n,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy y converged failed");
                                    if (cols > 0) {
                                        cuda_check(cudaMemcpy(d_x.data(), d_x_gx.data(),
                                                              sizeof(double) * x_size,
                                                              cudaMemcpyDeviceToDevice),
                                                   "cudaMemcpy x converged failed");
                                    }
                                    if (store_alphas) {
                                        copy_alpha_state(alpha_state_ptrs,
                                                         alpha_gx_ptrs);
                                    }
                                    iter_used = iter + 1;
                                    return true;
                                }
                            }
                            if (!honest_mode && step > 0 &&
                                (rel_change / static_cast<double>(step)) < convergence_tol) {
                                cuda_check(cudaMemcpy(d_y.data(), d_y_gx.data(),
                                                      sizeof(double) * n,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy y converged failed");
                                if (cols > 0) {
                                    cuda_check(cudaMemcpy(d_x.data(), d_x_gx.data(),
                                                          sizeof(double) * x_size,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy x converged failed");
                                }
                                if (store_alphas) {
                                    copy_alpha_state(alpha_state_ptrs, alpha_gx_ptrs);
                                }
                                iter_used = iter + 1;
                                return true;
                            }
                        }

                        if (cuda_cg_fallback_ok && cg_iter_cap > 0 &&
                            !accel_diverged && (iter + 1) >= cg_iter_cap) {
                            // Still grinding far past where any well-conditioned
                            // graph converges: hand the post-sweep iterate to CG.
                            cuda_check(cudaMemcpy(d_y.data(), d_y_gx.data(),
                                                  sizeof(double) * n,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy y cg-cap failed");
                            if (cols > 0) {
                                cuda_check(cudaMemcpy(d_x.data(), d_x_gx.data(),
                                                      sizeof(double) * x_size,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy x cg-cap failed");
                            }
                            accel_diverged = true;
                            iter_used = iter + 1;
                            return false;
                        }

                        if (!accel_suspended && iter_grand_acc > 0 &&
                            ((iter + 1) % iter_grand_acc == 0)) {
                            ++grand_acc;
                            if (grand_acc == 1) {
                                cuda_check(cudaMemcpy(d_y_acc_a.data(), d_y_gx.data(),
                                                      sizeof(double) * n,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy acc a failed");
                                if (cols > 0) {
                                    cuda_check(cudaMemcpy(d_x_acc_a.data(), d_x_gx.data(),
                                                          sizeof(double) * x_size,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy acc a X failed");
                                }
                                if (store_alphas && accel_grand_alphas) {
                                    copy_alpha_state(alpha_acc_a_ptrs, alpha_gx_ptrs);
                                }
                            } else if (grand_acc == 2) {
                                cuda_check(cudaMemcpy(d_y_acc_b.data(), d_y_gx.data(),
                                                      sizeof(double) * n,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy acc b failed");
                                if (cols > 0) {
                                    cuda_check(cudaMemcpy(d_x_acc_b.data(), d_x_gx.data(),
                                                          sizeof(double) * x_size,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy acc b X failed");
                                }
                                if (store_alphas && accel_grand_alphas) {
                                    copy_alpha_state(alpha_acc_b_ptrs, alpha_gx_ptrs);
                                }
                            } else {
                                cuda_check(cudaMemcpy(d_y_acc_c.data(), d_y_gx.data(),
                                                      sizeof(double) * n,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy acc c failed");
                                if (cols > 0) {
                                    cuda_check(cudaMemcpy(d_x_acc_c.data(), d_x_gx.data(),
                                                          sizeof(double) * x_size,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy acc c X failed");
                                }
                                if (store_alphas && accel_grand_alphas) {
                                    copy_alpha_state(alpha_acc_c_ptrs, alpha_gx_ptrs);
                                }

                                cuda_check(cudaMemset(d_stats.data(), 0, sizeof(double) * 3),
                                           "cudaMemset stats failed");
                                irons_tuck_stats_kernel<kBlockSize><<<blocks_n, kBlockSize>>>(
                                    d_y_acc_a.data(), d_y_acc_b.data(), d_y_acc_c.data(),
                                    n, d_stats.data());
                                cuda_check(cudaGetLastError(),
                                           "grand irons tuck stats kernel launch failed");
                                if (cols > 0) {
                                    irons_tuck_stats_kernel<kBlockSize><<<blocks_x, kBlockSize>>>(
                                        d_x_acc_a.data(), d_x_acc_b.data(), d_x_acc_c.data(),
                                        static_cast<int>(x_size), d_stats.data());
                                    cuda_check(cudaGetLastError(),
                                               "grand irons tuck stats X kernel launch failed");
                                }

                                double acc_stats[2] = {0.0, 0.0};
                                cuda_check(cudaMemcpy(acc_stats, d_stats.data(),
                                                      sizeof(double) * 2,
                                                      cudaMemcpyDeviceToHost),
                                           "cudaMemcpy acc stats failed");
                                if (acc_stats[1] != 0.0) {
                                    const double acc_coef = acc_stats[0] / acc_stats[1];
                                    irons_tuck_update_kernel<<<blocks_n, kBlockSize>>>(
                                        d_y_acc_a.data(), d_y_acc_b.data(), d_y_acc_c.data(),
                                        n, acc_coef);
                                    cuda_check(cudaGetLastError(),
                                               "grand irons tuck update kernel launch failed");
                                    if (cols > 0) {
                                        irons_tuck_update_kernel<<<blocks_x, kBlockSize>>>(
                                            d_x_acc_a.data(), d_x_acc_b.data(), d_x_acc_c.data(),
                                            static_cast<int>(x_size), acc_coef);
                                        cuda_check(cudaGetLastError(),
                                                   "grand irons tuck update X kernel launch failed");
                                    }
                                    if (store_alphas && accel_grand_alphas) {
                                        lincomb_alpha_state(alpha_acc_a_ptrs, alpha_acc_b_ptrs,
                                                            alpha_acc_c_ptrs, acc_coef);
                                    }
                                }

                                cuda_check(cudaMemcpy(d_y.data(), d_y_acc_a.data(),
                                                      sizeof(double) * n,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy acc result failed");
                                if (cols > 0) {
                                    cuda_check(cudaMemcpy(d_x.data(), d_x_acc_a.data(),
                                                          sizeof(double) * x_size,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy acc result X failed");
                                }
                                if (store_alphas && accel_grand_alphas) {
                                    copy_alpha_state(alpha_state_ptrs, alpha_acc_a_ptrs);
                                }
                                cuda_check(cudaMemcpy(d_y_gx.data(), d_y.data(),
                                                      sizeof(double) * n,
                                                      cudaMemcpyDeviceToDevice),
                                           "cudaMemcpy acc y_gx failed");
                                if (cols > 0) {
                                    cuda_check(cudaMemcpy(d_x_gx.data(), d_x.data(),
                                                          sizeof(double) * x_size,
                                                          cudaMemcpyDeviceToDevice),
                                               "cudaMemcpy acc x_gx failed");
                                }
                                if (store_alphas) {
                                    copy_alpha_state(alpha_gx_ptrs, alpha_state_ptrs);
                                }
                                run_sweep(d_y_gx.data(), d_x_gx.data(), order, false, ignored,
                                          store_alphas ? &alpha_gx_ptrs : nullptr);
                                grand_acc = 0;
                            }
                        }
                    }

                    cuda_check(cudaMemcpy(d_y.data(), d_y_gx.data(), sizeof(double) * n,
                                          cudaMemcpyDeviceToDevice),
                               "cudaMemcpy y fallback final failed");
                    if (cols > 0) {
                        cuda_check(cudaMemcpy(d_x.data(), d_x_gx.data(),
                                              sizeof(double) * x_size,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy x fallback final failed");
                    }
                    if (store_alphas) {
                        copy_alpha_state(alpha_state_ptrs, alpha_gx_ptrs);
                    }
                    iter_used = max_iter;
                    return false;
                };

                // Stable block-CG over the symmetric (I - G) demean operator,
                // reused both as the explicit reghdfe CG accelerator (gate below)
                // and as the IT divergence fallback. CG re-derives its residual
                // from d_y/d_x, so it is safe to run on any handed-off iterate.
                auto run_cuda_cg = [&](const std::vector<std::size_t>& cg_order) {
                    // CG needs the symmetric (forward+backward) sweep so that
                    // (I - G) is self-adjoint. The gate path already forces this
                    // (use_reghdfe_cg => use_symmetric=true at setup); for the
                    // divergence fallback run_accel_phase has finished, so setting
                    // it here only affects the CG sweeps, matching the gate.
                    use_symmetric = true;
                    auto safe_ratio = [](double num, double den) {
                        constexpr double eps = std::numeric_limits<double>::epsilon();
                        return std::abs(den) <= eps ? 0.0 : num / den;
                    };
                    auto compute_component_sumsq = [&](const double* y_ptr,
                                                        const double* x_ptr,
                                                        std::vector<double>& out) {
                        out.assign(static_cast<std::size_t>(cg_width), 0.0);
                        cuda_check(cudaMemset(d_update_sums.data(), 0,
                                              sizeof(double) *
                                                  static_cast<std::size_t>(cg_width)),
                                   "cudaMemset CG sumsq failed");
                        const dim3 grid(blocks_n, cg_width);
                        component_sumsq_kernel<kBlockSize><<<grid, kBlockSize>>>(
                            y_ptr, x_ptr, n, cols, ld, d_update_sums.data());
                        cuda_check(cudaGetLastError(), "CG sumsq kernel launch failed");
                        cuda_check(cudaMemcpy(out.data(), d_update_sums.data(),
                                              sizeof(double) *
                                                  static_cast<std::size_t>(cg_width),
                                              cudaMemcpyDeviceToHost),
                                   "cudaMemcpy CG sumsq failed");
                    };
                    auto compute_component_dot = [&](const double* y_a,
                                                     const double* y_b,
                                                     const double* x_a,
                                                     const double* x_b,
                                                     std::vector<double>& out) {
                        out.assign(static_cast<std::size_t>(cg_width), 0.0);
                        cuda_check(cudaMemset(d_update_sums.data(), 0,
                                              sizeof(double) *
                                                  static_cast<std::size_t>(cg_width)),
                                   "cudaMemset CG dot failed");
                        const dim3 grid(blocks_n, cg_width);
                        component_dot_kernel<kBlockSize><<<grid, kBlockSize>>>(
                            y_a, y_b, x_a, x_b, n, cols, ld, d_update_sums.data());
                        cuda_check(cudaGetLastError(), "CG dot kernel launch failed");
                        cuda_check(cudaMemcpy(out.data(), d_update_sums.data(),
                                              sizeof(double) *
                                                  static_cast<std::size_t>(cg_width),
                                              cudaMemcpyDeviceToHost),
                                   "cudaMemcpy CG dot failed");
                    };
                    auto projection_sweep = [&](const double* src_y,
                                                const double* src_x,
                                                double* proj_y,
                                                double* proj_x) {
                        cuda_check(cudaMemcpy(proj_y, src_y, sizeof(double) * n,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy CG projection y failed");
                        if (cols > 0) {
                            cuda_check(cudaMemcpy(proj_x, src_x, sizeof(double) * x_size,
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy CG projection X failed");
                        }
                        double ignored = 0.0;
                        run_sweep(proj_y, proj_x, cg_order, false, ignored, nullptr);
                        projection_residual_kernel<<<blocks_values, kBlockSize>>>(
                            proj_y, src_y, proj_x, src_x, total_values, n, ld);
                        cuda_check(cudaGetLastError(), "CG projection residual kernel failed");
                    };
                    auto subtract_columns = [&](double* target_y,
                                                double* target_x,
                                                const double* direction_y,
                                                const double* direction_x,
                                                const std::vector<double>& alpha) {
                        cuda_check(cudaMemcpy(d_cg_alpha.data(), alpha.data(),
                                              sizeof(double) *
                                                  static_cast<std::size_t>(cg_width),
                                              cudaMemcpyHostToDevice),
                                   "cudaMemcpy CG alpha failed");
                        scaled_subtract_columns_kernel<<<blocks_values, kBlockSize>>>(
                            target_y, target_x, direction_y, direction_x, d_cg_alpha.data(),
                            total_values, n, ld);
                        cuda_check(cudaGetLastError(), "CG scaled subtract kernel failed");
                    };
                    auto update_direction = [&](double* u_y,
                                                double* u_x,
                                                const double* r_y,
                                                const double* r_x,
                                                const std::vector<double>& beta) {
                        cuda_check(cudaMemcpy(d_cg_beta.data(), beta.data(),
                                              sizeof(double) *
                                                  static_cast<std::size_t>(cg_width),
                                              cudaMemcpyHostToDevice),
                                   "cudaMemcpy CG beta failed");
                        cg_direction_update_kernel<<<blocks_values, kBlockSize>>>(
                            u_y, u_x, r_y, r_x, d_cg_beta.data(), total_values, n, ld);
                        cuda_check(cudaGetLastError(), "CG direction update kernel failed");
                    };

                    std::vector<double> improvement;
                    std::vector<double> ssr;
                    std::vector<double> ssr_old;
                    std::vector<double> denom;
                    std::vector<double> alpha(static_cast<std::size_t>(cg_width), 0.0);
                    std::vector<double> beta(static_cast<std::size_t>(cg_width), 0.0);
                    std::vector<double> recent(static_cast<std::size_t>(cg_width), 0.0);

                    compute_component_sumsq(d_y.data(), d_x.data(), improvement);
                    projection_sweep(d_y.data(), d_x.data(), d_y_gx.data(), d_x_gx.data());
                    compute_component_sumsq(d_y_gx.data(), d_x_gx.data(), ssr);
                    cuda_check(cudaMemcpy(d_y_ggx.data(), d_y_gx.data(), sizeof(double) * n,
                                          cudaMemcpyDeviceToDevice),
                               "cudaMemcpy CG direction y failed");
                    if (cols > 0) {
                        cuda_check(cudaMemcpy(d_x_ggx.data(), d_x_gx.data(),
                                              sizeof(double) * x_size,
                                              cudaMemcpyDeviceToDevice),
                                   "cudaMemcpy CG direction X failed");
                    }

                    int cg_iters = 0;
                    for (int iter = 1; iter <= options.max_iter; ++iter) {
                        projection_sweep(d_y_ggx.data(), d_x_ggx.data(),
                                         d_y_acc_a.data(), d_x_acc_a.data());
                        compute_component_dot(d_y_ggx.data(), d_y_acc_a.data(),
                                              d_x_ggx.data(), d_x_acc_a.data(), denom);
                        for (int k = 0; k < cg_width; ++k) {
                            alpha[static_cast<std::size_t>(k)] =
                                safe_ratio(ssr[static_cast<std::size_t>(k)],
                                           denom[static_cast<std::size_t>(k)]);
                            recent[static_cast<std::size_t>(k)] =
                                alpha[static_cast<std::size_t>(k)] *
                                ssr[static_cast<std::size_t>(k)];
                            improvement[static_cast<std::size_t>(k)] -=
                                recent[static_cast<std::size_t>(k)];
                        }

                        subtract_columns(d_y.data(), d_x.data(),
                                         d_y_ggx.data(), d_x_ggx.data(), alpha);
                        subtract_columns(d_y_gx.data(), d_x_gx.data(),
                                         d_y_acc_a.data(), d_x_acc_a.data(), alpha);

                        ssr_old = ssr;
                        compute_component_sumsq(d_y_gx.data(), d_x_gx.data(), ssr);
                        for (int k = 0; k < cg_width; ++k) {
                            beta[static_cast<std::size_t>(k)] =
                                safe_ratio(ssr[static_cast<std::size_t>(k)],
                                           ssr_old[static_cast<std::size_t>(k)]);
                        }
                        update_direction(d_y_ggx.data(), d_x_ggx.data(),
                                         d_y_gx.data(), d_x_gx.data(), beta);

                        double update_error = 0.0;
                        for (int k = 0; k < cg_width; ++k) {
                            constexpr double eps_floor = 1e-15;
                            const double num =
                                std::max(0.0, recent[static_cast<std::size_t>(k)]);
                            const double den = std::max(
                                std::abs(improvement[static_cast<std::size_t>(k)]),
                                std::numeric_limits<double>::epsilon());
                            const double err = std::sqrt(num < eps_floor ? 0.0 : num / den);
                            update_error = std::max(update_error, err);
                        }
                        cg_iters = iter;
                        if (update_error <= options.tol) {
                            converged = true;
                            break;
                        }
                    }
                    result.iterations = cg_iters > 0 ? cg_iters : options.max_iter;
                    if (std::getenv("XHDFE_ACCEL_DEBUG") != nullptr) {
                        std::fprintf(stderr, "[ACCEL cuda CG iters=%d converged=%d]\n",
                                     result.iterations, converged ? 1 : 0);
                    }
                };

                const bool cg_fallback_possible =
                    cuda_cg_fallback_ok && !sweep_order.empty();
                if (use_reghdfe_cg && !sweep_order.empty()) {
                    run_cuda_cg(sweep_order);
                } else if (!enable_two_stage) {
                    int phase_iters = 0;
                    converged = run_accel_phase(sweep_order, accel_iter_budget, phase_iters);
                    if (!converged && accel_diverged && cg_fallback_possible) {
                        run_cuda_cg(sweep_order);
                    } else if (converged) {
                        result.iterations = phase_iters;
                    }
                } else {
                    int total_iters = 0;
                    int remaining = options.max_iter;
                    if (iter_warmup > 0 && remaining > 0) {
                        int phase_iters = 0;
                        converged = run_accel_phase(sweep_order,
                                                    std::min(iter_warmup, remaining),
                                                    phase_iters);
                        total_iters += phase_iters;
                        remaining -= phase_iters;
                    }
                    if (!converged && !accel_diverged && !two_fe_order.empty() &&
                        iter_two_fe > 0 && remaining > 0) {
                        int phase_iters = 0;
                        run_accel_phase(two_fe_order,
                                        std::min(iter_two_fe, remaining),
                                        phase_iters);
                        total_iters += phase_iters;
                        remaining -= phase_iters;
                    }
                    if (!converged && !accel_diverged && remaining > 0) {
                        int phase_iters = 0;
                        converged = run_accel_phase(sweep_order, remaining, phase_iters);
                        total_iters += phase_iters;
                        remaining -= phase_iters;
                    }
                    if (!converged && accel_diverged && cg_fallback_possible) {
                        run_cuda_cg(sweep_order);
                    } else if (converged) {
                        result.iterations = total_iters;
                    }
                }
            } else {
                auto run_simple_phase = [&](const std::vector<std::size_t>& order,
                                            int max_iter,
                                            int& iter_used) -> bool {
                    iter_used = 0;
                    if (max_iter <= 0 || order.empty()) {
                        return false;
	                    }
	                    double prev_norm = compute_norm();
	                    copy_update_snapshot(d_y.data(), d_x.data());
	                    int last_check_iter = -1;
	                    for (int iter = 0; iter < max_iter; ++iter) {
                        const bool do_check = (check_interval == 1 || iter < check_interval ||
                                               iter % check_interval == 0);
                        double sumsq = 0.0;
                        run_sweep(d_y.data(), d_x.data(), order, do_check, sumsq,
                                  store_alphas ? &alpha_state_ptrs : nullptr);
                        if (do_check) {
                            const double curr_norm = std::sqrt(sumsq);
                            const double denom = std::max(1.0, prev_norm);
                            const double rel_change = std::abs(curr_norm - prev_norm) / denom;
	                            const int step = iter - last_check_iter;
	                            last_check_iter = iter;
	                            prev_norm = curr_norm;
	                            double update_error = std::numeric_limits<double>::max();
	                            compute_update_error(d_y.data(), d_x.data(), update_error);
	                            copy_update_snapshot(d_y.data(), d_x.data());
	                            const double norm_error = rel_change / static_cast<double>(step);
	                            if (step > 0 &&
	                                convergence_reached(options, norm_error, update_error,
	                                                    convergence_tol)) {
	                                iter_used = iter + 1;
                                return true;
                            }
                        }
                    }
                    iter_used = max_iter;
                    return false;
                };

                if (!enable_two_stage) {
                    int phase_iters = 0;
                    converged = run_simple_phase(sweep_order, options.max_iter, phase_iters);
                    if (converged) {
                        result.iterations = phase_iters;
                    }
                } else {
                    int total_iters = 0;
                    int remaining = options.max_iter;
                    if (iter_warmup > 0 && remaining > 0) {
                        int phase_iters = 0;
                        converged = run_simple_phase(sweep_order,
                                                     std::min(iter_warmup, remaining),
                                                     phase_iters);
                        total_iters += phase_iters;
                        remaining -= phase_iters;
                    }
                    if (!converged && !two_fe_order.empty() && iter_two_fe > 0 && remaining > 0) {
                        int phase_iters = 0;
                        run_simple_phase(two_fe_order,
                                         std::min(iter_two_fe, remaining),
                                         phase_iters);
                        total_iters += phase_iters;
                        remaining -= phase_iters;
                    }
                    if (!converged && remaining > 0) {
                        int phase_iters = 0;
                        converged = run_simple_phase(sweep_order, remaining, phase_iters);
                        total_iters += phase_iters;
                        remaining -= phase_iters;
                    }
                    if (converged) {
                        result.iterations = total_iters;
                    }
                }
            }

            result.converged = converged;
            if (!converged) {
                result.iterations = options.max_iter;
            } else if (result.iterations == 0) {
                result.iterations = 1;
            }
        }

        if (result.converged && active_fes > 0) {
            constexpr int kPolishSweeps = 6;
            const bool strict_tolerance = strict_residual_tolerance_mode(options);
            const int max_polish_sweeps =
                strict_tolerance ? std::max(0, options.max_iter - result.iterations)
                                 : kPolishSweeps;
            const double polish_tol = strict_tolerance ? options.tol : 0.0;
            const bool polish_symmetric =
                options.symmetric_sweep && active_fes > 1;
            double final_max = strict_tolerance ? std::numeric_limits<double>::infinity() : 0.0;
            int polish_done = 0;
            for (int polish = 0; polish < max_polish_sweeps; ++polish) {
                if (strict_tolerance) {
                    cuda_check(cudaMemset(d_sumsq.data(), 0, sizeof(double)),
                               "cudaMemset polish max_abs failed");
                }
                for (std::size_t dim = 0; dim < active_fes; ++dim) {
                    double ignored = 0.0;
                    double* alpha_y_ptr = nullptr;
                    double* alpha_x_ptr = nullptr;
                    if (store_alphas) {
                        alpha_y_ptr = alpha_state_ptrs.y[dim];
                        alpha_x_ptr = alpha_state_ptrs.x[dim];
                    }
                    run_demean(fe_dev[dim], d_y.data(), d_x.data(), false, ignored, 1.0,
                               alpha_y_ptr, alpha_x_ptr);
                    if (strict_tolerance) {
                        const int blocks_g =
                            (fe_dev[dim].num_groups + kBlockSize - 1) / kBlockSize;
                        max_abs_kernel<kBlockSize><<<blocks_g, kBlockSize>>>(
                            fe_dev[dim].sum_y.data(), fe_dev[dim].num_groups,
                            d_sumsq.data());
                        cuda_check(cudaGetLastError(),
                                   "polish max_abs kernel launch failed");
                    }
                }
                if (polish_symmetric) {
                    for (std::size_t idx = active_fes; idx-- > 0;) {
                        double ignored = 0.0;
                        double* alpha_y_ptr = nullptr;
                        double* alpha_x_ptr = nullptr;
                        if (store_alphas) {
                            alpha_y_ptr = alpha_state_ptrs.y[idx];
                            alpha_x_ptr = alpha_state_ptrs.x[idx];
                        }
                        run_demean(fe_dev[idx], d_y.data(), d_x.data(), false, ignored, 1.0,
                                   alpha_y_ptr, alpha_x_ptr);
                        if (strict_tolerance) {
                            const int blocks_g =
                                (fe_dev[idx].num_groups + kBlockSize - 1) / kBlockSize;
                            max_abs_kernel<kBlockSize><<<blocks_g, kBlockSize>>>(
                                fe_dev[idx].sum_y.data(), fe_dev[idx].num_groups,
                                d_sumsq.data());
                            cuda_check(cudaGetLastError(),
                                       "polish max_abs kernel launch failed");
                        }
                    }
                }
                ++polish_done;
                if (strict_tolerance) {
                    cuda_check(cudaMemcpy(&final_max, d_sumsq.data(), sizeof(double),
                                          cudaMemcpyDeviceToHost),
                               "cudaMemcpy polish max_abs failed");
                    if (final_max <= polish_tol) {
                        break;
                    }
                }
            }
            if (strict_tolerance) {
                result.iterations += polish_done;
                if (final_max > polish_tol) {
                    result.converged = false;
                }
            }
        }

        result.y_tilde.resize(n);
        staged_memcpy_d2h(result.y_tilde.data(), d_y.data(), sizeof(double) * n,
                          workspace.h_stage, "cudaMemcpy y_tilde failed");
        if (cols > 0) {
            result.X_tilde.resize(ld, cols);
            staged_memcpy_d2h(result.X_tilde.data(), d_x.data(),
                              sizeof(double) * ld * cols, workspace.h_stage,
                              "cudaMemcpy X_tilde failed");
        } else {
            result.X_tilde.resize(ld, 0);
        }

        if (options.retain_fixed_effects) {
            result.fe_group_ids.resize(fe_inputs.size());
            result.fe_weight_sums.resize(fe_inputs.size());
            if (store_alphas) {
                result.fe_alpha_y.resize(fe_inputs.size());
                result.fe_alpha_X.resize(fe_inputs.size());
                for (std::size_t dim = 0; dim < fe_inputs.size(); ++dim) {
                    const int groups = fe_inputs[dim].num_groups;
                    result.fe_group_ids[dim].assign(fe_inputs[dim].group_ids,
                                                    fe_inputs[dim].group_ids + n);
                    result.fe_weight_sums[dim].resize(groups);
                    std::memcpy(result.fe_weight_sums[dim].data(), fe_inputs[dim].weight_sums,
                                sizeof(double) * groups);
                    result.fe_alpha_y[dim].resize(groups);
                    cuda_check(cudaMemcpy(result.fe_alpha_y[dim].data(),
                                          fe_dev[dim].alpha_y.data(),
                                          sizeof(double) * groups, cudaMemcpyDeviceToHost),
                               "cudaMemcpy fe alpha_y failed");
                    result.fe_alpha_X[dim].resize(groups, cols);
                    if (cols > 0) {
                        cuda_check(cudaMemcpy(result.fe_alpha_X[dim].data(),
                                              fe_dev[dim].alpha_x.data(),
                                              sizeof(double) * groups * cols,
                                              cudaMemcpyDeviceToHost),
                                   "cudaMemcpy fe alpha_X failed");
                    }
                }
                result.fe_means.clear();
            } else {
                result.fe_means.resize(fe_inputs.size());
                for (std::size_t dim = 0; dim < fe_inputs.size(); ++dim) {
                    const int groups = fe_inputs[dim].num_groups;
                    result.fe_group_ids[dim].assign(fe_inputs[dim].group_ids,
                                                    fe_inputs[dim].group_ids + n);
                    result.fe_weight_sums[dim].resize(groups);
                    std::memcpy(result.fe_weight_sums[dim].data(), fe_inputs[dim].weight_sums,
                                sizeof(double) * groups);
                    result.fe_means[dim].resize(groups);
                    cuda_check(cudaMemcpy(result.fe_means[dim].data(), fe_dev[dim].sum_y.data(),
                                          sizeof(double) * groups, cudaMemcpyDeviceToHost),
                               "cudaMemcpy fe means failed");
                }
            }
        }

        result.sweep_order_used.clear();
        if (method == AbsorptionMethod::Jacobi || sweep_order.empty()) {
            result.sweep_order_used.reserve(fe_inputs.size());
            for (std::size_t dim = 0; dim < fe_inputs.size(); ++dim) {
                result.sweep_order_used.push_back(static_cast<int>(dim));
            }
        } else {
            result.sweep_order_used.reserve(sweep_order.size());
            for (const std::size_t dim : sweep_order) {
                result.sweep_order_used.push_back(static_cast<int>(dim));
            }
        }

        return true;
    } catch (const std::exception& e) {
        if (std::getenv("XHDFE_DEBUG_GPU") != nullptr) {
            std::fprintf(stderr, "[gpu] absorb_fixed_effects_cuda failed: %s\n", e.what());
        }
        cudaGetLastError();  // clear the error so the next call starts clean
        return false;
    } catch (...) {
        if (std::getenv("XHDFE_DEBUG_GPU") != nullptr) {
            std::fprintf(stderr, "[gpu] absorb_fixed_effects_cuda failed: unknown\n");
        }
        cudaGetLastError();
        return false;
    }
}

bool absorb_fixed_effects_group_individual_cuda(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<GpuFeInput>& standard_fe_inputs,
    const GroupIndividualStructure& gi,
    const Eigen::VectorXd* weights,
    const std::vector<std::size_t>& sweep_order,
    const HdfeOptions& options,
    AbsorptionMethod method,
    AbsorptionResult& result) {
    const int n = static_cast<int>(y.size());
    const int cols = static_cast<int>(X.cols());
    const int ld = static_cast<int>(X.rows());
    if (n <= 0 || ld != n) {
        return false;
    }
    if (gi.num_groups != n || gi.num_individuals <= 0) {
        return false;
    }
    if (static_cast<int>(gi.group_ptr.size()) != n + 1) {
        return false;
    }
    if (static_cast<int>(gi.group_scale.size()) != n) {
        return false;
    }
    if (gi.group_ptr.empty()) {
        return false;
    }
    if (static_cast<std::size_t>(gi.group_ptr.back()) != gi.group_individual.size()) {
        return false;
    }
    if (method == AbsorptionMethod::Jacobi) {
        return false;
    }

    std::vector<std::size_t> order = sweep_order;
    const std::size_t k = standard_fe_inputs.size();
    auto default_order = [&]() {
        order.clear();
        order.reserve(k);
        for (std::size_t d = 0; d < k; ++d) {
            order.push_back(d);
        }
    };
    if (k > 0) {
        if (order.empty()) {
            default_order();
        } else {
            std::vector<uint8_t> seen(k, 0);
            bool ok = true;
            for (const std::size_t d : order) {
                if (d >= k || seen[d]) {
                    ok = false;
                    break;
                }
                seen[d] = 1;
            }
            if (!ok || order.size() != k) {
                default_order();
            }
        }
    } else {
        order.clear();
    }

    double relaxation = options.jacobi_relaxation;
    if (!(relaxation > 0.0 && relaxation <= 1.0)) {
        relaxation = 1.0;
    }
    relaxation = std::min(relaxation, 1.0);

    try {
        const bool unit_weights = (weights == nullptr);
        const bool use_symmetric =
            (method == AbsorptionMethod::SymmetricGaussSeidel) && ((order.size() + 1) > 1);

        CudaWorkspace& workspace = cuda_workspace;

        DeviceBuffer<double>& d_y = workspace.d_y;
        d_y.allocate(static_cast<std::size_t>(n));
        cuda_check(cudaMemcpy(d_y.data(), y.data(), sizeof(double) * n, cudaMemcpyHostToDevice),
                   "cudaMemcpy y failed");

        DeviceBuffer<double>& d_x = workspace.d_x;
        if (cols > 0) {
            d_x.allocate(static_cast<std::size_t>(ld) * cols);
            cuda_check(cudaMemcpy(d_x.data(), X.data(), sizeof(double) * ld * cols,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy X failed");
        } else {
            d_x.reset();
        }

        DeviceBuffer<double>& d_weights = workspace.d_weights;
        if (!unit_weights) {
            d_weights.allocate(static_cast<std::size_t>(n));
            cuda_check(cudaMemcpy(d_weights.data(), weights->data(), sizeof(double) * n,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy weights failed");
        } else {
            d_weights.reset();
        }

        std::vector<CudaFeDevice>& fe_dev = workspace.fe_dev;
        if (fe_dev.size() != k) {
            fe_dev.resize(k);
        }
        for (std::size_t idx = 0; idx < k; ++idx) {
            const auto& fe = standard_fe_inputs[idx];
            CudaFeDevice& dev = fe_dev[idx];
            dev.num_groups = fe.num_groups;
            dev.num_levels_present = fe.num_levels_present;
            // Same privatized-vs-segmented dispatch as the main path.
            constexpr std::size_t kPrivatizedShmemBudget = 40 * 1024;
            const std::size_t shmem_needed =
                static_cast<std::size_t>(fe.num_groups) *
                (1 + static_cast<std::size_t>(cols)) * sizeof(double);
            dev.use_privatized_sums =
                (fe.num_groups > 0 && shmem_needed <= kPrivatizedShmemBudget);
            dev.privatized_shmem_bytes = dev.use_privatized_sums ? shmem_needed : 0;
            dev.use_segmented_sums = !dev.use_privatized_sums;
            dev.gid.allocate(static_cast<std::size_t>(n));
            staged_memcpy_h2d(dev.gid.data(), fe.group_ids, sizeof(int) * n,
                              workspace.h_stage, "cudaMemcpy group ids failed");
            dev.weight_sums.allocate(static_cast<std::size_t>(fe.num_groups));
            staged_memcpy_h2d(dev.weight_sums.data(), fe.weight_sums,
                              sizeof(double) * fe.num_groups, workspace.h_stage,
                              "cudaMemcpy weight sums failed");
            dev.sum_y.allocate(static_cast<std::size_t>(fe.num_groups));
            if (cols > 0) {
                dev.sum_x.allocate(static_cast<std::size_t>(fe.num_groups) * cols);
            } else {
                dev.sum_x.reset();
            }
        }

        // Copy group/individual structure (individual -> groups adjacency + group scale).
        DeviceBuffer<double>& d_group_scale = workspace.gi_group_scale;
        d_group_scale.allocate(static_cast<std::size_t>(n));
        cuda_check(cudaMemcpy(d_group_scale.data(), gi.group_scale.data(),
                              sizeof(double) * static_cast<std::size_t>(n),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy gi.group_scale failed");

        const int individuals = gi.num_individuals;
        if (static_cast<int>(gi.individual_ptr.size()) != individuals + 1) {
            return false;
        }
        if (gi.individual_ptr.empty()) {
            return false;
        }
        if (static_cast<std::size_t>(gi.individual_ptr.back()) != gi.individual_group.size()) {
            return false;
        }

        DeviceBuffer<int>& d_ind_ptr = workspace.gi_individual_ptr;
        d_ind_ptr.allocate(static_cast<std::size_t>(individuals) + 1);
        cuda_check(cudaMemcpy(d_ind_ptr.data(), gi.individual_ptr.data(),
                              sizeof(int) * (static_cast<std::size_t>(individuals) + 1),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy gi.individual_ptr failed");

        const std::size_t edges = gi.individual_group.size();
        DeviceBuffer<int>& d_ind_group = workspace.gi_individual_group;
        d_ind_group.allocate(edges);
        cuda_check(cudaMemcpy(d_ind_group.data(), gi.individual_group.data(),
                              sizeof(int) * edges, cudaMemcpyHostToDevice),
                   "cudaMemcpy gi.individual_group failed");

        // Greedy-color the individuals so individuals sharing an observation
        // row land in different colors: each color is then one race-free
        // Gauss-Seidel kernel launch (see gi_color_sweep_kernel). The fixed
        // visit order makes the coloring (and the sweep) deterministic.
        std::vector<int> color(static_cast<std::size_t>(individuals), 0);
        int num_colors = 1;
        {
            std::vector<int> used;  // used[c] == i marks color c taken by a neighbor of i
            for (int i = 0; i < individuals; ++i) {
                const int beg = gi.individual_ptr[static_cast<std::size_t>(i)];
                const int end = gi.individual_ptr[static_cast<std::size_t>(i) + 1];
                for (int pos = beg; pos < end; ++pos) {
                    const int g = gi.individual_group[static_cast<std::size_t>(pos)];
                    if (g < 0 || g >= n) {
                        continue;
                    }
                    const int gbeg = gi.group_ptr[static_cast<std::size_t>(g)];
                    const int gend = gi.group_ptr[static_cast<std::size_t>(g) + 1];
                    for (int pos2 = gbeg; pos2 < gend; ++pos2) {
                        const int j = gi.group_individual[static_cast<std::size_t>(pos2)];
                        if (j < 0 || j >= i) {
                            continue;  // only already-colored neighbors matter
                        }
                        const int cj = color[static_cast<std::size_t>(j)];
                        if (cj >= static_cast<int>(used.size())) {
                            used.resize(static_cast<std::size_t>(cj) + 1, -1);
                        }
                        used[static_cast<std::size_t>(cj)] = i;
                    }
                }
                int c = 0;
                while (c < static_cast<int>(used.size()) &&
                       used[static_cast<std::size_t>(c)] == i) {
                    ++c;
                }
                color[static_cast<std::size_t>(i)] = c;
                num_colors = std::max(num_colors, c + 1);
            }
        }
        std::vector<int> color_offsets(static_cast<std::size_t>(num_colors) + 1, 0);
        for (int i = 0; i < individuals; ++i) {
            ++color_offsets[static_cast<std::size_t>(color[static_cast<std::size_t>(i)]) + 1];
        }
        for (int c = 0; c < num_colors; ++c) {
            color_offsets[static_cast<std::size_t>(c) + 1] +=
                color_offsets[static_cast<std::size_t>(c)];
        }
        std::vector<int> individuals_by_color(static_cast<std::size_t>(individuals), 0);
        {
            std::vector<int> cursor(color_offsets.begin(), color_offsets.end() - 1);
            for (int i = 0; i < individuals; ++i) {
                const int c = color[static_cast<std::size_t>(i)];
                individuals_by_color[static_cast<std::size_t>(
                    cursor[static_cast<std::size_t>(c)]++)] = i;
            }
        }
        DeviceBuffer<int>& d_color_individuals = workspace.gi_color_individuals;
        d_color_individuals.allocate(static_cast<std::size_t>(individuals));
        cuda_check(cudaMemcpy(d_color_individuals.data(), individuals_by_color.data(),
                              sizeof(int) * static_cast<std::size_t>(individuals),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy gi color order failed");

        DeviceBuffer<double>& d_denom = workspace.gi_denom;
        d_denom.allocate(static_cast<std::size_t>(individuals));
        const int blocks_i = (individuals + kBlockSize - 1) / kBlockSize;
        gi_denom_by_individual_kernel<<<blocks_i, kBlockSize>>>(
            d_ind_ptr.data(),
            d_ind_group.data(),
            d_group_scale.data(),
            unit_weights ? nullptr : d_weights.data(),
            unit_weights,
            n,
            individuals,
            d_denom.data());
        cuda_check(cudaGetLastError(), "gi denom (by individual) kernel launch failed");

        // Keep the unused buffers around for workspace reuse, but avoid forcing allocations here.
        workspace.gi_numer_y.reset();
        workspace.gi_numer_x.reset();
        workspace.gi_alpha_y.reset();
        workspace.gi_alpha_x.reset();

        DeviceBuffer<double>& d_sumsq = workspace.d_sumsq;
        d_sumsq.allocate(1);

        auto compute_norm = [&](double& out_norm) {
            cuda_check(cudaMemset(d_sumsq.data(), 0, sizeof(double)), "cudaMemset sumsq failed");
            const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
            sumsq_kernel<kBlockSize><<<blocks_n, kBlockSize>>>(
                d_y.data(), d_x.data(), n, cols, ld, d_sumsq.data());
            cuda_check(cudaGetLastError(), "sumsq kernel launch failed");
            double sumsq = 0.0;
            cuda_check(cudaMemcpy(&sumsq, d_sumsq.data(), sizeof(double), cudaMemcpyDeviceToHost),
                       "cudaMemcpy sumsq failed");
            out_norm = std::sqrt(sumsq);
        };

        auto run_standard_demean = [&](CudaFeDevice& fe) {
            const int blocks_n = (n + kBlockSize - 1) / kBlockSize;
            const int blocks_h = (fe.num_groups + kBlockSize - 1) / kBlockSize;
            cuda_check(cudaMemset(fe.sum_y.data(), 0, sizeof(double) * fe.num_groups),
                       "cudaMemset sum_y failed");
            if (cols > 0) {
                cuda_check(cudaMemset(fe.sum_x.data(), 0,
                                      sizeof(double) * fe.num_groups * cols),
                           "cudaMemset sum_x failed");
            }
            if (fe.use_privatized_sums) {
                accumulate_sums_privatized_kernel<<<blocks_n, kBlockSize,
                                                     fe.privatized_shmem_bytes>>>(
                    d_y.data(),
                    d_x.data(),
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_sums (privatized) kernel launch failed");
            } else if (fe.use_segmented_sums) {
                accumulate_sums_segmented_kernel<<<blocks_n, kBlockSize>>>(
                    d_y.data(),
                    d_x.data(),
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(),
                           "accumulate_sums (segmented) kernel launch failed");
            } else {
                accumulate_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                    d_y.data(),
                    d_x.data(),
                    n,
                    cols,
                    ld,
                    fe.gid.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    fe.sum_y.data(),
                    fe.sum_x.data(),
                    fe.num_groups);
                cuda_check(cudaGetLastError(), "accumulate_sums (atomic) kernel launch failed");
            }
            compute_means_kernel<<<blocks_h, kBlockSize>>>(
                fe.sum_y.data(),
                fe.sum_x.data(),
                fe.weight_sums.data(),
                fe.num_groups,
                cols);
            cuda_check(cudaGetLastError(), "compute_means kernel launch failed");
            apply_means_kernel<<<blocks_n, kBlockSize>>>(
                d_y.data(),
                d_x.data(),
                n,
                cols,
                ld,
                fe.gid.data(),
                fe.sum_y.data(),
                fe.sum_x.data(),
                fe.num_groups);
            cuda_check(cudaGetLastError(), "apply_means kernel launch failed");
        };

        auto run_individual_sweep = [&]() {
            for (int c = 0; c < num_colors; ++c) {
                const int color_begin = color_offsets[static_cast<std::size_t>(c)];
                const int color_count =
                    color_offsets[static_cast<std::size_t>(c) + 1] - color_begin;
                if (color_count <= 0) {
                    continue;
                }
                const int blocks_c = (color_count + kBlockSize - 1) / kBlockSize;
                gi_color_sweep_kernel<<<blocks_c, kBlockSize>>>(
                    d_y.data(),
                    d_x.data(),
                    n,
                    cols,
                    ld,
                    d_ind_ptr.data(),
                    d_ind_group.data(),
                    d_group_scale.data(),
                    unit_weights ? nullptr : d_weights.data(),
                    unit_weights,
                    d_denom.data(),
                    d_color_individuals.data(),
                    color_begin,
                    color_count,
                    relaxation);
                cuda_check(cudaGetLastError(), "gi color sweep kernel launch failed");
            }
        };

        double prev_norm = 0.0;
        compute_norm(prev_norm);
        const int check_interval = std::max(1, options.convergence_check_interval);
        int last_check_iter = -1;
        bool converged = false;

        for (int iter = 0; iter < options.max_iter; ++iter) {
            for (const std::size_t dim : order) {
                run_standard_demean(fe_dev[dim]);
            }
            run_individual_sweep();

            if (use_symmetric) {
                run_individual_sweep();
                for (std::size_t idx = order.size(); idx-- > 0;) {
                    run_standard_demean(fe_dev[order[idx]]);
                }
            }

            const bool do_check =
                (check_interval == 1 || iter < check_interval || iter % check_interval == 0);
            if (do_check) {
                double curr_norm = 0.0;
                compute_norm(curr_norm);
                const double denom = std::max(1.0, prev_norm);
                const double rel_change = std::abs(curr_norm - prev_norm) / denom;
                const int step = iter - last_check_iter;
                last_check_iter = iter;
                prev_norm = curr_norm;
                if (step > 0 &&
                    (rel_change / static_cast<double>(step)) <
                        group_individual_absorption_tolerance(options)) {
                    result.iterations = iter + 1;
                    converged = true;
                    break;
                }
            }
        }

        result.converged = converged;
        if (!converged) {
            result.iterations = options.max_iter;
        } else if (result.iterations == 0) {
            result.iterations = 1;
        }

        result.y_tilde.resize(n);
        cuda_check(cudaMemcpy(result.y_tilde.data(), d_y.data(), sizeof(double) * n,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy y_tilde failed");
        result.X_tilde.resize(ld, cols);
        if (cols > 0) {
            cuda_check(cudaMemcpy(result.X_tilde.data(), d_x.data(),
                                  sizeof(double) * ld * cols, cudaMemcpyDeviceToHost),
                       "cudaMemcpy X_tilde failed");
        }

        result.sweep_order_used.clear();
        result.sweep_order_used.reserve(order.size());
        for (const std::size_t dim : order) {
            result.sweep_order_used.push_back(static_cast<int>(dim));
        }
        return true;
    } catch (...) {
        return false;
    }
}

// GPU port of fe_recovery_max_delta (see fe_absorption.cpp:5996). For each FE
// dimension in sweep_order: (a) accumulate weighted group sums of `residual`,
// (b) convert to group means, (c) track the maximum |mean_g| across dims and
// groups, (d) subtract the means from `residual` in-place (Gauss-Seidel).
// Returns the final max |mean| across all dims via out_max_delta. On any CUDA
// error returns false; caller falls back to CPU implementation.
bool fe_recovery_max_delta_cuda(const Eigen::Ref<const Eigen::VectorXd>& residual,
                                const std::vector<GpuFeInput>& fe_inputs,
                                const Eigen::VectorXd* weights,
                                const std::vector<std::size_t>& sweep_order,
                                double& out_max_delta) {
    try {
        const int n = static_cast<int>(residual.size());
        const std::size_t dims = fe_inputs.size();
        if (n == 0 || dims == 0) {
            out_max_delta = 0.0;
            return true;
        }
        if (weights && weights->size() != n) {
            return false;
        }

        std::vector<std::size_t> order = sweep_order;
        if (order.empty()) {
            order.resize(dims);
            std::iota(order.begin(), order.end(), 0);
        }
        for (const std::size_t d : order) {
            if (d >= dims) {
                return false;
            }
        }

        const bool unit_weights = (weights == nullptr);
        const int blocks_n = (n + kBlockSize - 1) / kBlockSize;

        DeviceBuffer<double> d_y;
        DeviceBuffer<double> d_w;
        d_y.allocate(static_cast<std::size_t>(n));
        cuda_check(cudaMemcpy(d_y.data(), residual.data(), sizeof(double) * n,
                              cudaMemcpyHostToDevice),
                   "fe_recovery_max_delta: H2D residual failed");
        if (!unit_weights) {
            d_w.allocate(static_cast<std::size_t>(n));
            cuda_check(cudaMemcpy(d_w.data(), weights->data(), sizeof(double) * n,
                                  cudaMemcpyHostToDevice),
                       "fe_recovery_max_delta: H2D weights failed");
        }

        std::vector<DeviceBuffer<int>> d_gids(dims);
        std::vector<DeviceBuffer<double>> d_weight_sums(dims);
        std::vector<DeviceBuffer<double>> d_sum_y(dims);
        for (std::size_t d = 0; d < dims; ++d) {
            const int groups = fe_inputs[d].num_groups;
            if (groups <= 0 || fe_inputs[d].group_ids == nullptr ||
                fe_inputs[d].weight_sums == nullptr) {
                return false;
            }
            d_gids[d].allocate(static_cast<std::size_t>(n));
            cuda_check(cudaMemcpy(d_gids[d].data(), fe_inputs[d].group_ids,
                                  sizeof(int) * n, cudaMemcpyHostToDevice),
                       "fe_recovery_max_delta: H2D group_ids failed");
            d_weight_sums[d].allocate(static_cast<std::size_t>(groups));
            cuda_check(cudaMemcpy(d_weight_sums[d].data(), fe_inputs[d].weight_sums,
                                  sizeof(double) * groups, cudaMemcpyHostToDevice),
                       "fe_recovery_max_delta: H2D weight_sums failed");
            d_sum_y[d].allocate(static_cast<std::size_t>(groups));
        }

        DeviceBuffer<double> d_max_abs;
        d_max_abs.allocate(1);
        cuda_check(cudaMemset(d_max_abs.data(), 0, sizeof(double)),
                   "fe_recovery_max_delta: memset max_abs failed");

        for (const std::size_t d : order) {
            const int groups = fe_inputs[d].num_groups;
            const int blocks_g = (groups + kBlockSize - 1) / kBlockSize;

            cuda_check(cudaMemset(d_sum_y[d].data(), 0, sizeof(double) * groups),
                       "fe_recovery_max_delta: memset sum_y failed");

            // Accumulate weighted group sums of y. cols=0 → X/sum_x branch is
            // skipped inside the kernel, so nullptrs are safe.
            accumulate_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                d_y.data(), /*X=*/nullptr, n, /*cols=*/0, /*ld=*/n,
                d_gids[d].data(),
                unit_weights ? nullptr : d_w.data(), unit_weights,
                d_sum_y[d].data(), /*sum_x=*/nullptr, groups);
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta: accumulate kernel launch");

            // Convert sums to means (divide by weight_sums; zero-safe inside).
            compute_means_kernel<<<blocks_g, kBlockSize>>>(
                d_sum_y[d].data(), /*sum_x=*/nullptr, d_weight_sums[d].data(),
                groups, /*cols=*/0);
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta: compute_means kernel launch");

            // Reduce max|mean_g| into d_max_abs (monotone in the |.|).
            max_abs_kernel<kBlockSize><<<blocks_g, kBlockSize>>>(
                d_sum_y[d].data(), groups, d_max_abs.data());
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta: max_abs kernel launch");

            // Subtract means from residual in-place (Gauss-Seidel).
            apply_means_kernel<<<blocks_n, kBlockSize>>>(
                d_y.data(), /*X=*/nullptr, n, /*cols=*/0, /*ld=*/n,
                d_gids[d].data(), d_sum_y[d].data(),
                /*mean_x=*/nullptr, groups);
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta: apply_means kernel launch");
        }

        double host_max = 0.0;
        cuda_check(cudaMemcpy(&host_max, d_max_abs.data(), sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "fe_recovery_max_delta: D2H max_abs failed");
        out_max_delta = host_max;
        return true;
    } catch (...) {
        return false;
    }
}

bool fe_recovery_max_delta_cuda_cached(const Eigen::Ref<const Eigen::VectorXd>& residual,
                                       const std::vector<GpuFeInput>& fe_inputs,
                                       const Eigen::VectorXd* weights,
                                       const std::vector<std::size_t>& sweep_order,
                                       double& out_max_delta) {
    try {
        const int n = static_cast<int>(residual.size());
        const std::size_t dims = fe_inputs.size();
        if (n == 0 || dims == 0) {
            out_max_delta = 0.0;
            return true;
        }
        if (weights && weights->size() != n) {
            return false;
        }

        std::vector<std::size_t> order = sweep_order;
        if (order.empty()) {
            order.resize(dims);
            std::iota(order.begin(), order.end(), 0);
        }
        for (const std::size_t d : order) {
            if (d >= dims) {
                return false;
            }
        }

        CudaWorkspace& workspace = cuda_workspace;
        if (workspace.fe_dev.size() != dims) {
            return false;
        }
        for (std::size_t d = 0; d < dims; ++d) {
            const int groups = fe_inputs[d].num_groups;
            if (groups <= 0 || fe_inputs[d].weight_sums == nullptr) {
                return false;
            }
            const CudaFeDevice& dev = workspace.fe_dev[d];
            if (dev.num_groups != groups ||
                dev.num_levels_present != fe_inputs[d].num_levels_present ||
                dev.gid.size() < static_cast<std::size_t>(n) ||
                dev.weight_sums.size() < static_cast<std::size_t>(groups) ||
                dev.sum_y.size() < static_cast<std::size_t>(groups)) {
                return false;
            }
        }

        const bool unit_weights = (weights == nullptr);
        const int blocks_n = (n + kBlockSize - 1) / kBlockSize;

        DeviceBuffer<double>& d_y = workspace.d_y;
        d_y.allocate(static_cast<std::size_t>(n));
        cuda_check(cudaMemcpy(d_y.data(), residual.data(), sizeof(double) * n,
                              cudaMemcpyHostToDevice),
                   "fe_recovery_max_delta_cached: H2D residual failed");

        DeviceBuffer<double>& d_w = workspace.d_weights;
        if (!unit_weights) {
            d_w.allocate(static_cast<std::size_t>(n));
            cuda_check(cudaMemcpy(d_w.data(), weights->data(), sizeof(double) * n,
                                  cudaMemcpyHostToDevice),
                       "fe_recovery_max_delta_cached: H2D weights failed");
        }

        DeviceBuffer<double>& d_max_abs = workspace.d_sumsq;
        d_max_abs.allocate(1);
        cuda_check(cudaMemset(d_max_abs.data(), 0, sizeof(double)),
                   "fe_recovery_max_delta_cached: memset max_abs failed");

        for (const std::size_t d : order) {
            CudaFeDevice& dev = workspace.fe_dev[d];
            const int groups = fe_inputs[d].num_groups;
            const int blocks_g = (groups + kBlockSize - 1) / kBlockSize;

            cuda_check(cudaMemset(dev.sum_y.data(), 0, sizeof(double) * groups),
                       "fe_recovery_max_delta_cached: memset sum_y failed");

            accumulate_sums_atomic_kernel<<<blocks_n, kBlockSize>>>(
                d_y.data(), /*X=*/nullptr, n, /*cols=*/0, /*ld=*/n,
                dev.gid.data(),
                unit_weights ? nullptr : d_w.data(), unit_weights,
                dev.sum_y.data(), /*sum_x=*/nullptr, groups);
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta_cached: accumulate kernel launch");

            compute_means_kernel<<<blocks_g, kBlockSize>>>(
                dev.sum_y.data(), /*sum_x=*/nullptr, dev.weight_sums.data(),
                groups, /*cols=*/0);
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta_cached: compute_means kernel launch");

            max_abs_kernel<kBlockSize><<<blocks_g, kBlockSize>>>(
                dev.sum_y.data(), groups, d_max_abs.data());
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta_cached: max_abs kernel launch");

            apply_means_kernel<<<blocks_n, kBlockSize>>>(
                d_y.data(), /*X=*/nullptr, n, /*cols=*/0, /*ld=*/n,
                dev.gid.data(), dev.sum_y.data(),
                /*mean_x=*/nullptr, groups);
            cuda_check(cudaGetLastError(),
                       "fe_recovery_max_delta_cached: apply_means kernel launch");
        }

        double host_max = 0.0;
        cuda_check(cudaMemcpy(&host_max, d_max_abs.data(), sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "fe_recovery_max_delta_cached: D2H max_abs failed");
        out_max_delta = host_max;
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace detail
}  // namespace hdfe
