// The CUDA operator of the reduced tier (package 2, level A, 17 November 2026).
// The organisation, the classes and the admission are described in
// reduced_schur_pcg_cuda.hpp; the design and its bench validation in
// Verifications/reduced_cuda_perf_20261116/p2/ (DESENHO_P2.md, BENCH_LOG.md).
#include "reduced_schur_pcg_cuda.hpp"

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace hdfe::detail::reduced_cuda {
namespace {

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// CUDA errors. An allocation that ran out is `on_memory` (Memory for device
// memory, Pinned for page-locked host memory); no device or driver is
// Unavailable; anything else is Fatal: an invalid argument, a duplicate
// registration or an earlier asynchronous error is not scarcity, whatever the
// call. The error is reported by the exception and then cleared from the
// runtime's last error, so that it cannot reappear at a later, unrelated check
// (a sticky error cannot be cleared and stays).
void check(cudaError_t status, const char* what, ErrorKind on_memory = ErrorKind::Fatal) {
    if (status == cudaSuccess) return;
    ErrorKind kind = ErrorKind::Fatal;
    if (status == cudaErrorMemoryAllocation && on_memory != ErrorKind::Fatal) {
        kind = on_memory;
    } else if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
        kind = ErrorKind::Unavailable;
    }
    (void)cudaGetLastError();
    throw Error(kind, std::string("CUDA reduced operator: ") + what + ": " + cudaGetErrorString(status));
}

#if defined(XHDFE_REDUCED_CUDA_TEST_HOOKS)
std::atomic<std::size_t> g_live_bytes{0};  // tests: device bytes of every live operator
#endif

// What the operator holds on the device. Every allocation is checked against the
// admitted peak before it is made.
struct Ledger {
    std::size_t live = 0, peak = 0, limit = 0;
};

template <class T>
struct DeviceArray {
    T* ptr = nullptr;
    std::size_t count = 0;
    Ledger* ledger = nullptr;
    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    ~DeviceArray() { release(); }
    std::size_t bytes() const noexcept { return count * sizeof(T); }
    void allocate(Ledger& l, std::size_t n, const char* what, bool inject = false) {
        release();
        const std::size_t want = checked_mul(n, sizeof(T));
        if (checked_add(l.live, want) > l.limit)
            throw Error(ErrorKind::Fatal, std::string("CUDA reduced operator: allocation beyond its admission: ") + what);
        if (inject) check(cudaErrorMemoryAllocation, what, ErrorKind::Memory);  // test injection
        if (want) check(cudaMalloc(reinterpret_cast<void**>(&ptr), want), what, ErrorKind::Memory);
        count = n;
        ledger = &l;
        l.live += want;
        l.peak = std::max(l.peak, l.live);
#if defined(XHDFE_REDUCED_CUDA_TEST_HOOKS)
        g_live_bytes += want;
#endif
    }
    void release() noexcept {
        if (ptr) (void)cudaFree(ptr);
        if (ledger) ledger->live -= bytes();
#if defined(XHDFE_REDUCED_CUDA_TEST_HOOKS)
        if (ledger) g_live_bytes -= bytes();
#endif
        ptr = nullptr;
        count = 0;
        ledger = nullptr;
    }
};

struct Stream {
    cudaStream_t value = nullptr;
    ~Stream() { if (value) (void)cudaStreamDestroy(value); }
};
struct Event {
    cudaEvent_t value = nullptr;
    ~Event() { if (value) (void)cudaEventDestroy(value); }
};
// Declared after the buffers it protects: on every exit the stream is drained
// before they are freed.
struct DrainOnExit {
    cudaStream_t stream;
    ~DrainOnExit() { if (stream) (void)cudaStreamSynchronize(stream); }
};

struct PinnedChunk {
    void* ptr = nullptr;
    PinnedChunk() = default;
    PinnedChunk(const PinnedChunk&) = delete;
    PinnedChunk& operator=(const PinnedChunk&) = delete;
    ~PinnedChunk() { if (ptr) (void)cudaFreeHost(ptr); }
    void allocate(std::size_t bytes, bool inject) {
        if (inject) check(cudaErrorMemoryAllocation, "page-locked upload chunk (test injection)", ErrorKind::Pinned);
        check(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), "page-locked upload chunk", ErrorKind::Pinned);
    }
};

using RunFn = std::function<void(int, const std::function<void(int)>&)>;

// Uploads through two page-locked chunks: the host copy into one chunk (on the
// caller's team, in a fixed number of byte ranges) overlaps the DMA from the
// other; a chunk is refilled only after its DMA is done, and the stream is
// drained before the chunks are freed.
struct Staging {
    cudaStream_t stream;
    const RunFn& run;
    PinnedChunk buf[2];
    Event done[2];
    std::size_t chunk = 0;
    int next = 0;
    bool used[2] = {false, false};
    Staging(cudaStream_t s, std::size_t chunk_bytes, const RunFn& r, bool inject) : stream(s), run(r) {
        chunk = std::max<std::size_t>(sizeof(double), chunk_bytes / sizeof(double) * sizeof(double));
        for (int b = 0; b < 2; ++b) {
            buf[b].allocate(chunk, inject && b == 1);
            check(cudaEventCreateWithFlags(&done[b].value, cudaEventDisableTiming), "event creation", ErrorKind::Memory);
        }
    }
    Staging(const Staging&) = delete;
    Staging& operator=(const Staging&) = delete;
    ~Staging() { (void)cudaStreamSynchronize(stream); }
    void put(void* dst, const void* src, std::size_t bytes) {
        const char* from = static_cast<const char*>(src);
        char* to = static_cast<char*>(dst);
        for (std::size_t off = 0; off < bytes; off += chunk) {
            const std::size_t len = std::min(chunk, bytes - off);
            const int b = next;
            next ^= 1;
            if (used[b]) check(cudaEventSynchronize(done[b].value), "staged upload");
            char* stage = static_cast<char*>(buf[b].ptr);
            const int parts = staging_parts(len);
            if (run && len >= (std::size_t{1} << 20)) {
                run(parts, [&](int part) {
                    const std::size_t a0 = len * static_cast<std::size_t>(part) / parts;
                    const std::size_t a1 = len * static_cast<std::size_t>(part + 1) / parts;
                    std::memcpy(stage + a0, from + off + a0, a1 - a0);
                });
            } else {
                std::memcpy(stage, from + off, len);
            }
            check(cudaMemcpyAsync(to + off, stage, len, cudaMemcpyHostToDevice, stream), "staged upload");
            check(cudaEventRecord(done[b].value, stream), "staged upload");
            used[b] = true;
        }
    }
};

unsigned blocks_for(long long n, int per_block) {
    const long long b = (n + per_block - 1) / per_block;
    if (b > static_cast<long long>(INT_MAX)) throw Error(ErrorKind::Memory, "CUDA reduced operator: grid too large");
    return static_cast<unsigned>(std::max(1LL, b));
}

// ---------------------------------------------------------------------------
// Kernels. Thread organisation: org_for(width) (groups of width threads inside a
// warp; ws, the device row stride, is the width).

struct Dims {
    int nb;
    int off[kMaxDims];
    const int* cc[kMaxDims];
};

struct K1Params {
    const double* in; double* T; double* PA;
    const int* op_seg;  // the geometry's order (K1L)
    const int* k1_seg;  // K1's order: the same segments grouped by length (k1_segment_order)
    const int* cell_ptr; const double* seg_inv; const int* mult;
    Dims d;
    int npriv; int priv_j[kMaxDims]; int priv_base[kMaxDims]; int S;
    int L_op; int short_max;
    int ws, gpw, G;
};

// u at cell c: the operand rows of every remaining effect, added in effect order from 0.0.
__device__ __forceinline__ double gather_u(const K1Params& p, int c, int lane) {
    double u = 0.0;
#pragma unroll
    for (int j = 0; j < kMaxDims; ++j)
        if (j < p.d.nb) u += __ldg(p.in + static_cast<std::size_t>(p.d.off[j] + __ldg(p.d.cc[j] + c)) * p.ws + lane);
    return u;
}

// K1: segments of at most short_max cells, one group per segment, taken in K1's
// order (k1_seg: grouped by length). A segment of at most CACHE cells keeps its u in
// registers; the arithmetic and its order are those of the two walks (mean, then
// deviations). The final combination: one owner per output (strided), groups added
// in group order, after a barrier that every thread reaches.
// Four blocks of 256 threads per multiprocessor fit in registers (at most 64 each).
template <int CACHE>
__global__ void __launch_bounds__(kThreads, 4) k1_segments(K1Params p) {
    extern __shared__ double acc[];  // G x S x ws, private to each group
    int g, lane;
    bool in_group;
    place_of(static_cast<int>(threadIdx.x), p.ws, p.gpw, g, lane, in_group);
    const std::size_t acc_count = static_cast<std::size_t>(p.G) * p.S * p.ws;
    for (std::size_t e = threadIdx.x; e < acc_count; e += blockDim.x) acc[e] = 0.0;  // groups without work too
    __syncthreads();
    if (in_group) {
#pragma unroll 1  // one segment at a time (the register cache is per segment; 64 registers, no spill)
        for (int i = 0; i < kSegPerGroup; ++i) {
            const long long s = k1_segment(blockIdx.x, g, i, p.G);
            if (s >= p.L_op) break;  // validated before k1_seg[s] is read
            const int l = p.k1_seg[s];
            const int c0 = p.cell_ptr[l], c1 = p.cell_ptr[l + 1];
            const int n = c1 - c0;
            if (n > p.short_max) continue;  // K1L's (the same test there)
            double* mine = acc + static_cast<std::size_t>(g) * p.S * p.ws;
            if (CACHE > 0 && n <= CACHE) {
                double uc[CACHE > 0 ? CACHE : 1];
                double t = 0.0;
#pragma unroll
                for (int k = 0; k < CACHE; ++k) {
                    if (k < n) {
                        const int c = c0 + k;
                        const double u = gather_u(p, c, lane);
                        uc[k] = u;
                        const double mc = p.mult ? static_cast<double>(__ldg(p.mult + c)) : 1.0;
                        t += mc * u;
                    }
                }
                const double mean = t * p.seg_inv[l];
                p.T[static_cast<std::size_t>(l) * p.ws + lane] = mean;
                if (p.S > 0) {
#pragma unroll
                    for (int k = 0; k < CACHE; ++k) {
                        if (k < n) {
                            const int c = c0 + k;
                            const double mc = p.mult ? static_cast<double>(__ldg(p.mult + c)) : 1.0;
                            const double dev = mc * (uc[k] - mean);
                            for (int q = 0; q < p.npriv; ++q)
                                mine[static_cast<std::size_t>(p.priv_base[q] + __ldg(p.d.cc[p.priv_j[q]] + c)) * p.ws + lane] += dev;
                        }
                    }
                }
                continue;
            }
            double t = 0.0;
            for (int c = c0; c < c1; ++c) {
                const double u = gather_u(p, c, lane);
                const double mc = p.mult ? static_cast<double>(p.mult[c]) : 1.0;
                t += mc * u;
            }
            const double mean = t * p.seg_inv[l];
            p.T[static_cast<std::size_t>(l) * p.ws + lane] = mean;
            if (p.S > 0) {
                for (int c = c0; c < c1; ++c) {
                    const double u = gather_u(p, c, lane);
                    const double mc = p.mult ? static_cast<double>(p.mult[c]) : 1.0;
                    const double dev = mc * (u - mean);
                    for (int q = 0; q < p.npriv; ++q)
                        mine[static_cast<std::size_t>(p.priv_base[q] + p.d.cc[p.priv_j[q]][c]) * p.ws + lane] += dev;
                }
            }
        }
    }
    __syncthreads();  // every thread, with or without a segment
    for (int e = threadIdx.x; e < p.S * p.ws; e += blockDim.x) {
        const int lam = e / p.ws, ln = e - lam * p.ws;
        double sum = 0.0;
        for (int gg = 0; gg < p.G; ++gg) sum += acc[(static_cast<std::size_t>(gg) * p.S + lam) * p.ws + ln];
        p.PA[(static_cast<std::size_t>(blockIdx.x) * p.S + lam) * p.ws + ln] = sum;
    }
}

// K1L: one block per long segment. Items of `item` cells, each summed in cell order
// by one group; the item sums combined by an in-place pairwise tree on item index
// (at stride k, item i += item i + k for i a multiple of 2k) in the global partials
// PL, whose slots for this segment start at long_base; then the deviations of the
// private effects into the shared accumulators, combined as in K1 into the slot
// pa_first + block of PA.
struct K1LParams {
    K1Params k;
    const int* long_pos; const long long* long_base; double* PL; int item; int pa_first;
};

__global__ void k1_long(K1LParams q) {
    const K1Params& p = q.k;
    extern __shared__ double acc[];
    __shared__ double mean_s[8];
    int g, lane;
    bool in_group;
    place_of(static_cast<int>(threadIdx.x), p.ws, p.gpw, g, lane, in_group);
    const std::size_t acc_count = static_cast<std::size_t>(p.G) * p.S * p.ws;
    for (std::size_t e = threadIdx.x; e < acc_count; e += blockDim.x) acc[e] = 0.0;
    const int s = q.long_pos[blockIdx.x];
    const int l = p.op_seg[s];
    const int c0 = p.cell_ptr[l], c1 = p.cell_ptr[l + 1];
    const int items = (c1 - c0 + q.item - 1) / q.item;
    double* PL = q.PL + static_cast<std::size_t>(q.long_base[blockIdx.x]) * p.ws;
    if (in_group) {
        for (int it = g; it < items; it += p.G) {
            const int a0 = c0 + it * q.item, a1 = min(c1, a0 + q.item);
            double t = 0.0;
            for (int c = a0; c < a1; ++c) {
                double u = 0.0;
                for (int j = 0; j < p.d.nb; ++j)
                    u += p.in[static_cast<std::size_t>(p.d.off[j] + p.d.cc[j][c]) * p.ws + lane];
                const double mc = p.mult ? static_cast<double>(p.mult[c]) : 1.0;
                t += mc * u;
            }
            PL[static_cast<std::size_t>(it) * p.ws + lane] = t;
        }
    }
    __syncthreads();
    for (int stride = 1; stride < items; stride <<= 1) {
        const int pairs = (items + 2 * stride - 1) / (2 * stride);
        for (int e = threadIdx.x; e < pairs * p.ws; e += blockDim.x) {
            const int pr = e / p.ws, ln = e - pr * p.ws;
            const int i = pr * 2 * stride;
            if (i + stride < items) PL[static_cast<std::size_t>(i) * p.ws + ln] += PL[static_cast<std::size_t>(i + stride) * p.ws + ln];
        }
        __syncthreads();
    }
    if (static_cast<int>(threadIdx.x) < p.ws) {
        const double mean = PL[threadIdx.x] * p.seg_inv[l];
        mean_s[threadIdx.x] = mean;
        p.T[static_cast<std::size_t>(l) * p.ws + threadIdx.x] = mean;
    }
    __syncthreads();
    if (in_group && p.S > 0) {
        const double mean = mean_s[lane];
        double* mine = acc + static_cast<std::size_t>(g) * p.S * p.ws;
        for (int it = g; it < items; it += p.G) {
            const int a0 = c0 + it * q.item, a1 = min(c1, a0 + q.item);
            for (int c = a0; c < a1; ++c) {
                double u = 0.0;
                for (int j = 0; j < p.d.nb; ++j)
                    u += p.in[static_cast<std::size_t>(p.d.off[j] + p.d.cc[j][c]) * p.ws + lane];
                const double mc = p.mult ? static_cast<double>(p.mult[c]) : 1.0;
                const double dev = mc * (u - mean);
                for (int k = 0; k < p.npriv; ++k)
                    mine[static_cast<std::size_t>(p.priv_base[k] + p.d.cc[p.priv_j[k]][c]) * p.ws + lane] += dev;
            }
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < p.S * p.ws; e += blockDim.x) {
        const int lam = e / p.ws, ln = e - lam * p.ws;
        double sum = 0.0;
        for (int gg = 0; gg < p.G; ++gg) sum += acc[(static_cast<std::size_t>(gg) * p.S + lam) * p.ws + ln];
        p.PA[(static_cast<std::size_t>(q.pa_first + blockIdx.x) * p.S + lam) * p.ws + ln] = sum;
    }
}

// K2: the levels of one transposed or built effect j; one group per unit, units in
// launch order (decreasing length); a unit writes its level's output row or its
// logical PB slot.
struct K2Params {
    const double* in; const double* T; double* out; double* PB;
    const int* u_level; const int* u_begin; const int* u_end; const int* u_slot; const int* order; long long units;
    const int* seg; const int* other[kMaxDims]; const int* mult;
    int j; Dims d;
    int ws, gpw, G;
};

__global__ void k2_levels(K2Params p) {
    int g, lane;
    bool in_group;
    place_of(static_cast<int>(threadIdx.x), p.ws, p.gpw, g, lane, in_group);
    const long long ui = static_cast<long long>(blockIdx.x) * p.G + g;
    if (!in_group || ui >= p.units) return;  // no barrier in this kernel
    const int u = p.order[ui];
    const int lev = p.u_level[u], slot = p.u_slot[u];
    double* dst = slot < 0 ? p.out + static_cast<std::size_t>(p.d.off[p.j] + lev) * p.ws
                           : p.PB + static_cast<std::size_t>(slot) * p.ws;
    const double own = p.in[static_cast<std::size_t>(p.d.off[p.j] + lev) * p.ws + lane];
    double acc = 0.0;
    const int a1 = p.u_end[u];
#pragma unroll 4
    for (int a = p.u_begin[u]; a < a1; ++a) {
        const int l = p.seg[a];
        double v = 0.0;
        for (int jj = 0; jj < p.d.nb; ++jj)
            v += jj == p.j ? own : p.in[static_cast<std::size_t>(p.d.off[jj] + p.other[jj][a]) * p.ws + lane];
        const double mc = p.mult ? static_cast<double>(p.mult[a]) : 1.0;
        acc += mc * (v - p.T[static_cast<std::size_t>(l) * p.ws + lane]);
    }
    dst[lane] = acc;
}

// K3: one block per output; thread t adds partials t, t + 256, ... in order, then a
// halving tree over t.
__device__ double block_tree(double v, double* buf) {
    buf[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    const double r = buf[0];
    __syncthreads();
    return r;
}

struct K3PParams {
    const double* PA; double* out; long long slots; int S; int ws;
    const int* lam_out;  // S: output row (offset of the effect + level)
};

__global__ void k3_private(K3PParams p) {
    __shared__ double buf[kThreads];
    const int lam = blockIdx.x / p.ws, ln = blockIdx.x - lam * p.ws;
    double v = 0.0;
    for (long long s = threadIdx.x; s < p.slots; s += blockDim.x) v += p.PA[(static_cast<std::size_t>(s) * p.S + lam) * p.ws + ln];
    const double r = block_tree(v, buf);
    if (threadIdx.x == 0) p.out[static_cast<std::size_t>(p.lam_out[lam]) * p.ws + ln] = r;
}

struct K3LParams {
    const double* PB; double* out; const int* lv_out; const int* lv_base; const int* lv_items; int ws;
};

__global__ void k3_long(K3LParams p) {
    __shared__ double buf[kThreads];
    const int lv = blockIdx.x;
    const int base = p.lv_base[lv], items = p.lv_items[lv];
    double* dst = p.out + static_cast<std::size_t>(p.lv_out[lv]) * p.ws;
    for (int ln = 0; ln < p.ws; ++ln) {
        double v = 0.0;
        for (int i = threadIdx.x; i < items; i += blockDim.x) v += p.PB[static_cast<std::size_t>(base + i) * p.ws + ln];
        const double r = block_tree(v, buf);
        if (threadIdx.x == 0) dst[ln] = r;
    }
}

// ---------------------------------------------------------------------------
// B: the level-ordered arrays of an effect neither private nor transposed.
__global__ void b_seg_counts(const int* op_seg, const int* cell_ptr, int L_op, int* cnt) {
    const long long s = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= L_op) return;
    const int l = op_seg[s];
    cnt[s] = cell_ptr[l + 1] - cell_ptr[l];
}

// The stable sequence of operator cells: position q -> (cell, segment id).
__global__ void b_enumerate(const int* op_seg, const int* cell_ptr, const int* qbase, int L_op, int* qcell, int* qseg) {
    const long long s = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= L_op) return;
    const int l = op_seg[s];
    const int c0 = cell_ptr[l], c1 = cell_ptr[l + 1];
    int q = qbase[s];
    for (int c = c0; c < c1; ++c, ++q) {
        qcell[q] = c;
        qseg[q] = l;
    }
}

// Counts of each level inside a tile of `tile` positions (counts do not depend on order).
__global__ void b_tile_hist(const int* code, const int* qcell, long long C_op, int tile, int levels, int* H) {
    extern __shared__ int h[];
    for (int i = threadIdx.x; i < levels; i += blockDim.x) h[i] = 0;
    __syncthreads();
    const long long q0 = static_cast<long long>(blockIdx.x) * tile;
    const long long q1 = min(C_op, q0 + tile);
    for (long long q = q0 + threadIdx.x; q < q1; q += blockDim.x) atomicAdd(&h[code[qcell[q]]], 1);
    __syncthreads();
    for (int i = threadIdx.x; i < levels; i += blockDim.x) H[static_cast<std::size_t>(blockIdx.x) * levels + i] = h[i];
}

__global__ void b_level_totals(const int* H, int tiles, int levels, int* totals) {
    const int lev = blockIdx.x * blockDim.x + threadIdx.x;
    if (lev >= levels) return;
    int sum = 0;
    for (int t = 0; t < tiles; ++t) sum += H[static_cast<std::size_t>(t) * levels + lev];
    totals[lev] = sum;
}

__global__ void b_set_end(int* ptr, int levels, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) ptr[levels] = value;
}

// The cursor of (tile, level): the level's start, then the tiles before it, in order.
__global__ void b_level_offsets(int* H, const int* ptr, int tiles, int levels) {
    const int lev = blockIdx.x * blockDim.x + threadIdx.x;
    if (lev >= levels) return;
    int run = ptr[lev];
    for (int t = 0; t < tiles; ++t) {
        int& slot = H[static_cast<std::size_t>(t) * levels + lev];
        const int c = slot;
        slot = run;
        run += c;
    }
}

// Stable scatter: one warp per tile, positions in order, 32 at a time; the rank of a
// position among the equal codes of its 32 is the number of lower lanes with that
// code (__match_any_sync); the cursors advance by the count after every lane has read
// them. No atomic decides a position.
struct BScatter {
    const int* code; const int* qcell; const int* qseg; long long C_op; int tile; int levels; int j;
    const int* H; Dims d; const int* mult;
    int* seg; int* other[kMaxDims]; int* omult; int* cell;
};

__global__ void b_scatter(BScatter p) {
    extern __shared__ int cursor_all[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long long tile = static_cast<long long>(blockIdx.x) * (blockDim.x >> 5) + warp;
    const long long tiles = (p.C_op + p.tile - 1) / p.tile;
    int* cursor = cursor_all + static_cast<std::size_t>(warp) * p.levels;
    if (tile >= tiles) return;  // whole warp, no block barrier below
    for (int i = lane; i < p.levels; i += 32) cursor[i] = p.H[static_cast<std::size_t>(tile) * p.levels + i];
    __syncwarp();
    const long long q0 = tile * p.tile, q1 = min(p.C_op, q0 + p.tile);
    for (long long qb = q0; qb < q1; qb += 32) {
        const long long q = qb + lane;
        const bool valid = q < q1;
        const unsigned active = __ballot_sync(0xffffffffu, valid);
        int pos = 0, c = 0, code = 0;
        unsigned peers = 0;
        if (valid) {
            c = p.qcell[q];
            code = p.code[c];
            peers = __match_any_sync(active, code);
            const unsigned lower = peers & ((1u << lane) - 1u);
            pos = cursor[code] + __popc(lower);
        }
        __syncwarp();
        if (valid && lane == 31 - __clz(peers)) cursor[code] += __popc(peers);
        __syncwarp();
        if (valid) {
            p.seg[pos] = p.qseg[q];
            for (int jj = 0; jj < p.d.nb; ++jj)
                if (jj != p.j) p.other[jj][pos] = p.d.cc[jj][c];
            if (p.omult) p.omult[pos] = p.mult[c];
#if defined(XHDFE_REDUCED_CUDA_TEST_HOOKS)
            if (p.cell) p.cell[pos] = c;
#endif
        }
    }
}

// The radix alternative: keys = codes in operator-cell order, values = positions.
__global__ void b_keys(const int* code, const int* qcell, long long C_op, int* keys, int* vals) {
    const long long q = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= C_op) return;
    keys[q] = code[qcell[q]];
    vals[q] = static_cast<int>(q);
}
__global__ void b_ptr_from_sorted(const int* keys, long long n, int levels, int* ptr) {
    const int lev = blockIdx.x * blockDim.x + threadIdx.x;
    if (lev > levels) return;
    long long lo = 0, hi = n;  // the first position with key >= lev
    while (lo < hi) {
        const long long mid = (lo + hi) / 2;
        if (keys[mid] < lev) lo = mid + 1; else hi = mid;
    }
    ptr[lev] = static_cast<int>(lo);
}
struct BGather {
    const int* vals; const int* qcell; const int* qseg; long long n; int j; Dims d; const int* mult;
    int* seg; int* other[kMaxDims]; int* omult; int* cell;
};
__global__ void b_gather(BGather p) {
    const long long a = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (a >= p.n) return;
    const int q = p.vals[a];
    const int c = p.qcell[q];
    p.seg[a] = p.qseg[q];
    for (int jj = 0; jj < p.d.nb; ++jj)
        if (jj != p.j) p.other[jj][a] = p.d.cc[jj][c];
    if (p.omult) p.omult[a] = p.mult[c];
#if defined(XHDFE_REDUCED_CUDA_TEST_HOOKS)
    if (p.cell) p.cell[a] = c;
#endif
}

// The caller's rows of stride W arrive whole; the device packs the live lanes into
// rows of ws and unpacks the result, writing +0.0 into the padding lanes.
__global__ void k_pack(const double* inW, double* in, long long m, int W, int ws) {
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= m * ws) return;
    const long long r = e / ws;
    in[e] = inW[r * W + (e - r * ws)];
}
__global__ void k_unpack(const double* out, double* outW, long long m, int W, int ws) {
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= m * W) return;
    const long long r = e / W;
    const int lane = static_cast<int>(e - r * W);
    outW[e] = lane < ws ? out[r * ws + lane] : 0.0;
}

int radix_bits(int levels) {
    int bits = 1;
    while (bits < 31 && (1LL << bits) < levels) ++bits;
    return bits;
}

}  // namespace

// ---------------------------------------------------------------------------
struct Operator::State {
    Ledger ledger;
    Stream stream;
    Event ev[9];
    GeometryView g;
    Options o;
    Plan plan;
    CubSizes cub;
    Admission adm;
    Statistics st;
    Org org;
    int width = 0;
    int smem_optin = 0;
    std::size_t k1_static = 0, k1l_static = 0;
    // resident
    DeviceArray<int> cell_ptr, op_seg, k1_seg, mult;
    DeviceArray<double> seg_inv;
    DeviceArray<int> cc[kMaxDims];
    struct Level {
        DeviceArray<int> ptr, seg, mult, cell;
        DeviceArray<int> other[kMaxDims];
        DeviceArray<int> u_level, u_begin, u_end, u_slot, order;
        long long units = 0;
        bool has_mult = false;
    } lv[kMaxDims];
    DeviceArray<int> lam_out, long_pos;
    DeviceArray<long long> long_base;
    DeviceArray<int> lvl_out, lvl_base, lvl_items;
    int long_levels_total = 0;
    // per product
    DeviceArray<double> in, out, inW, outW, T, PA, PL, PB;
    // the caller's registered range
    double* reg_begin = nullptr;
    std::size_t reg_count = 0;

    ~State() {
        if (stream.value) (void)cudaStreamSynchronize(stream.value);  // nothing in flight when memory is freed
        if (reg_begin) (void)cudaHostUnregister(reg_begin);
    }
    void setup(const GeometryView& view, const Options& options, Admission* record);
    void build_b(int j, unsigned char* scratch, bool inject, std::vector<int>& host_ptr);
    void run_kernels();
    void record_event(int k) {
        if (o.profile) check(cudaEventRecord(ev[k].value, stream.value), "event");
    }
};

void Operator::State::setup(const GeometryView& view, const Options& options, Admission* record) {
    const auto t_all = Clock::now();
    g = view;
    o = options;
    const RunFn run = std::move(o.run);  // used only in this function
    o.run = nullptr;
    st.timed = o.profile;
#if defined(XHDFE_REDUCED_CUDA_TEST_HOOKS)
    const bool cells = o.test_source_cells;
#else
    if (o.test_source_cells) throw Error(ErrorKind::Fatal, "CUDA reduced operator: the source-cell test hook is not compiled in");
    const bool cells = false;
#endif
    const auto lap = [&](double& field, Clock::time_point& from) {
        const auto now = Clock::now();
        field += std::chrono::duration<double>(now - from).count();
        from = now;
    };
    auto from = Clock::now();

    // The device first: the classes depend on its shared memory.
    int devices = 0;
    check(cudaGetDeviceCount(&devices), "device count", ErrorKind::Memory);
    if (!devices) throw Error(ErrorKind::Unavailable, "CUDA reduced operator: no device");
    check(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking), "stream creation", ErrorKind::Memory);
    int device = 0;
    check(cudaGetDevice(&device), "device", ErrorKind::Memory);
    check(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device), "shared memory", ErrorKind::Memory);
    cudaFuncAttributes fa{};
    check(cudaFuncGetAttributes(&fa, k1_segments<kK1Cache>), "kernel attributes");
    k1_static = fa.sharedSizeBytes;
    check(cudaFuncGetAttributes(&fa, k1_long), "kernel attributes");
    k1l_static = fa.sharedSizeBytes;
    if (o.profile)
        for (Event& e : ev) check(cudaEventCreate(&e.value), "event creation", ErrorKind::Memory);
    const std::size_t smem_static = std::max(k1_static, k1l_static);
    const std::size_t smem_limit = static_cast<std::size_t>(smem_optin) > smem_static ? static_cast<std::size_t>(smem_optin) - smem_static : 0;
    lap(st.device_seconds, from);

    plan = plan_operator(g, o, smem_limit);
    lap(st.plan_seconds, from);

    // CUB's own temporary sizes for these shapes, before the admission (C3).
    for (int j = 0; j < g.nb; ++j) {
        if (plan.cls[j] != 'B') continue;
        // The same iterator types as the calls in build_b.
        check(cub::DeviceScan::ExclusiveSum(nullptr, cub.scan_segments, static_cast<int*>(nullptr), static_cast<int*>(nullptr),
                                            g.L_op, stream.value), "scan size");
        if (plan.radix[j]) {
            cub::DoubleBuffer<int> keys(nullptr, nullptr), vals(nullptr, nullptr);
            check(cub::DeviceRadixSort::SortPairs(nullptr, cub.radix[j], keys, vals, static_cast<int>(g.C_op), 0,
                                                  radix_bits(g.dims[j].levels), stream.value), "radix sort size");
        } else {
            check(cub::DeviceScan::ExclusiveSum(nullptr, cub.scan_levels[j], static_cast<int*>(nullptr), static_cast<int*>(nullptr),
                                                g.dims[j].levels, stream.value), "scan size");
        }
    }
    adm = admit(g, o, plan, cub, cells);
    adm.budget = o.budget;
    std::size_t free_b = 0, total_b = 0;
    check(cudaMemGetInfo(&free_b, &total_b), "available memory", ErrorKind::Memory);
    adm.free_at_entry = free_b;
    if (record) *record = adm;
    if (o.host_budget && (o.host_reserved > o.host_budget || adm.host_setup > o.host_budget - o.host_reserved))
        throw Error(ErrorKind::Memory, "CUDA reduced operator: host budget");
    if (o.budget && adm.device_peak > o.budget) throw Error(ErrorKind::Memory, "CUDA reduced operator: device budget");
    if (adm.device_peak > free_b) throw Error(ErrorKind::Memory, "CUDA reduced operator: free device memory");
    adm.admitted = true;
    if (record) *record = adm;
    ledger.limit = adm.device_peak;
    lap(st.device_seconds, from);

    // Page-locked upload chunks (refused for lack of resources: Pinned).
    Staging staging(stream.value, o.chunk_bytes, run, o.test_failure == TestFailure::Staging);
    lap(st.staging_seconds, from);

    // Resident uploads, then the per-product buffers while they are in flight.
    const std::size_t W = static_cast<std::size_t>(o.lanes), m = static_cast<std::size_t>(g.m);
    const std::size_t L = static_cast<std::size_t>(g.L), C = static_cast<std::size_t>(g.C);
    const auto upload = [&](auto& dst, const auto* src, std::size_t n, const char* what) {
        dst.allocate(ledger, n, what);
        if (n) staging.put(dst.ptr, src, n * sizeof(*src));
        st.uploaded_bytes = checked_add(st.uploaded_bytes, n * sizeof(*src));
    };
    upload(cell_ptr, g.cell_ptr, L + 1U, "cell_ptr");
    upload(op_seg, g.op_seg, static_cast<std::size_t>(g.L_op), "op_seg");
    std::size_t host_order = 0;  // the real host peak of K1's order (the order and its counts)
    {
        // K1's order on the fit's team; freed once staged (put has read it when it returns).
        std::vector<int> order(static_cast<std::size_t>(g.L_op));
        k1_segment_order(g, o.short_seg, order.data(), [&](int parts, const std::function<void(int)>& body) {
            if (run) run(parts, body);
            else for (int part = 0; part < parts; ++part) body(part);
        });
        upload(k1_seg, order.data(), order.size(), "k1_segment_order");
        host_order = checked_add(order.capacity() * sizeof(int), k1_order_count_bytes(o.short_seg, g.L_op));
    }
    upload(seg_inv, g.seg_inv, L, "seg_inv");
    for (int j = 0; j < g.nb; ++j) upload(cc[j], g.cc[j], C, "codes");
    if (g.mult) upload(mult, g.mult, C, "mult");
    for (int j = 0; j < g.nb; ++j) {
        if (plan.cls[j] != 'T') continue;
        const DimView& d = g.dims[j];
        const std::size_t E = static_cast<std::size_t>(plan.entries[j]);
        upload(lv[j].ptr, d.tr_ptr, static_cast<std::size_t>(d.levels) + 1U, "level_ptr");
        upload(lv[j].seg, d.tr_seg, E, "level_seg");
        for (int jj = 0; jj < g.nb; ++jj)
            if (jj != j) upload(lv[j].other[jj], d.tr_other[jj], E, "level_others");
        lv[j].has_mult = d.has_mult;  // has_mult == false: multiplicity one, .mult never read
        if (d.has_mult) upload(lv[j].mult, d.tr_mult, E, "level_mult");
    }
    std::size_t host_early = 0;  // the real host peak of the lists below (capacities)
    {
        std::vector<int> lam(static_cast<std::size_t>(plan.S));
        std::size_t at = 0;
        for (int q = 0; q < plan.npriv; ++q)
            for (int lev = 0; lev < g.dims[plan.priv_j[q]].levels; ++lev) lam[at++] = g.dims[plan.priv_j[q]].offset + lev;
        std::vector<int> pos(static_cast<std::size_t>(plan.n_long));
        std::vector<long long> base(static_cast<std::size_t>(plan.n_long));
        long long items = 0;
        at = 0;
        for (int s = 0; s < g.L_op; ++s) {
            const int l = g.op_seg[s];
            const int n = g.cell_ptr[l + 1] - g.cell_ptr[l];
            if (n > o.short_seg) {
                pos[at] = s;
                base[at] = items;
                items += (n + o.item - 1) / o.item;
                ++at;
            }
        }
        if (at != static_cast<std::size_t>(plan.n_long) || items != plan.pl_items)
            throw Error(ErrorKind::Fatal, "CUDA reduced operator: long segments");
        upload(lam_out, lam.data(), lam.size(), "private_outputs");
        upload(long_pos, pos.data(), pos.size(), "long_segments");
        upload(long_base, base.data(), base.size(), "long_segments");
        host_early = checked_add(checked_add(lam.capacity() * sizeof(int), pos.capacity() * sizeof(int)),
                                 base.capacity() * sizeof(long long));
    }
    lap(st.upload_seconds, from);
    in.allocate(ledger, m * W, "in", o.test_failure == TestFailure::Allocation);
    out.allocate(ledger, m * W, "out");
    if (W > 1) {
        inW.allocate(ledger, m * W, "registered operand rows");
        outW.allocate(ledger, m * W, "registered result rows");
    }
    T.allocate(ledger, L * W, "T");
    PA.allocate(ledger, checked_mul(static_cast<std::size_t>(plan.pa_slots), checked_mul(static_cast<std::size_t>(plan.S), W)), "PA");
    PL.allocate(ledger, checked_mul(static_cast<std::size_t>(plan.pl_items), W), "PL");
    lap(st.alloc_seconds, from);
    check(cudaStreamSynchronize(stream.value), "uploads");
    lap(st.upload_seconds, from);

    // B effects: built on the device in one scratch allocation (freed, with nothing
    // in flight, before the units exist); B's ptr comes back for the units.
    std::vector<int> host_ptr[kMaxDims];
    std::size_t host_ptrs = 0;
    {
        DeviceArray<unsigned char> scratch;
        if (adm.device[Admission::Scratch] > 0) scratch.allocate(ledger, adm.device[Admission::Scratch], "b_build");
        const DrainOnExit drain{stream.value};  // destroyed before the scratch
        bool first = true;
        for (int j = 0; j < g.nb; ++j) {
            if (plan.cls[j] != 'B') continue;
            build_b(j, scratch.ptr, first && o.test_failure == TestFailure::Build, host_ptr[j]);
            first = false;
            host_ptrs = checked_add(host_ptrs, host_ptr[j].capacity() * sizeof(int));
        }
        if (o.profile) {
            std::size_t free_now = 0, total_now = 0;
            check(cudaMemGetInfo(&free_now, &total_now), "available memory", ErrorKind::Memory);
            adm.free_at_peak = free_now;
        }
    }
    lap(st.build_seconds, from);

    // Units of every T or B effect, the long-level lists, PB.
    std::size_t host_units = 0;
    {
        UnitCount counts[kMaxDims];
        long long long_levels = 0, slots_total = 0;
        for (int j = 0; j < g.nb; ++j) {
            if (plan.cls[j] == 'P') continue;
            const DimView& d = g.dims[j];
            counts[j] = plan.cls[j] == 'T' ? plan.units[j] : count_units(host_ptr[j].data(), d.levels, o.short_level, o.item);
            if (counts[j].units > plan.units[j].units || counts[j].slots > plan.units[j].slots ||
                counts[j].long_levels > plan.units[j].long_levels)
                throw Error(ErrorKind::Fatal, "CUDA reduced operator: units beyond their admission");
            long_levels += counts[j].long_levels;
            slots_total += counts[j].slots;
        }
        LongLevels longs;
        longs.size_for(static_cast<std::size_t>(long_levels));
        long long pb_next = 0;
        for (int j = 0; j < g.nb; ++j) {
            if (plan.cls[j] == 'P') continue;
            const DimView& d = g.dims[j];
            const int* ptr = plan.cls[j] == 'T' ? d.tr_ptr : host_ptr[j].data();
            UnitLists units;
            build_units(ptr, d.levels, d.offset, o.short_level, o.item, pb_next, counts[j], units, longs);
            host_units = std::max(host_units, checked_add(units.capacity_bytes(), sort_count_bytes(o.short_level, o.item)));
            Level& lj = lv[j];
            upload(lj.u_level, units.level.data(), units.level.size(), "units");
            upload(lj.u_begin, units.begin.data(), units.begin.size(), "units");
            upload(lj.u_end, units.end.data(), units.end.size(), "units");
            upload(lj.u_slot, units.slot.data(), units.slot.size(), "units");
            upload(lj.order, units.order.data(), units.order.size(), "units");
            lj.units = counts[j].units;
            pb_next += counts[j].slots;
            st.units += counts[j].units;
            st.long_levels += counts[j].long_levels;
        }
        if (pb_next != slots_total || longs.next != longs.out.size())
            throw Error(ErrorKind::Fatal, "CUDA reduced operator: unit lists");
        PB.allocate(ledger, checked_mul(static_cast<std::size_t>(pb_next), W), "PB");
        upload(lvl_out, longs.out.data(), longs.out.size(), "long_levels");
        upload(lvl_base, longs.base.data(), longs.base.size(), "long_levels");
        upload(lvl_items, longs.items.data(), longs.items.size(), "long_levels");
        long_levels_total = static_cast<int>(longs.out.size());
        st.pb_slots = pb_next;
        check(cudaStreamSynchronize(stream.value), "units");
        host_units = checked_add(host_units, longs.capacity_bytes());
    }
    lap(st.units_seconds, from);

    // K1's dynamic shared memory above 48 KiB needs the explicit opt-in. The same
    // value (the device's limit) every time, so that concurrent fits agree.
    if (plan.k1_smem > (std::size_t{48} << 10)) {
        check(cudaFuncSetAttribute(k1_segments<kK1Cache>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(static_cast<std::size_t>(smem_optin) - k1_static)), "shared memory opt-in");
        check(cudaFuncSetAttribute(k1_long, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(static_cast<std::size_t>(smem_optin) - k1l_static)), "shared memory opt-in");
    }

    // The host peak, from real capacities: the chunks, then the largest of K1's order
    // (with its counts), the early lists and (B's ptrs + one effect's unit lists + the
    // long-level lists).
    adm.host_setup_real = checked_add(adm.page_locked_setup, std::max({host_order, host_early, checked_add(host_ptrs, host_units)}));
    if (adm.host_setup_real > adm.host_setup)
        throw Error(ErrorKind::Fatal, "CUDA reduced operator: host bytes beyond their admission");
    adm.allocated_peak = ledger.peak;
    adm.allocated_iteration = ledger.live;
    if (o.profile) {
        std::size_t free_now = 0, total_now = 0;
        check(cudaMemGetInfo(&free_now, &total_now), "available memory", ErrorKind::Memory);
        adm.free_after_setup = free_now;
        if (adm.free_at_peak == 0) adm.free_at_peak = free_now;  // no B build: the iteration is the peak
    }
    if (record) *record = adm;

    for (int j = 0; j < g.nb; ++j) {
        st.classes[j] = plan.cls[j];
        st.b_radix += plan.cls[j] == 'B' && plan.radix[j];
    }
    st.private_levels = plan.S;
    st.long_segments = plan.n_long;
    st.k1_blocks = plan.k1_blocks_max;
    st.pa_slots = plan.pa_slots;
    st.pl_items = plan.pl_items;
    st.host_layout_bytes = adm.host_setup_real;
    st.device_payload_bytes = ledger.live;
    st.setup_seconds = seconds(t_all);
    // The staging chunks are freed here, after the stream is drained (~Staging).
}

void Operator::State::build_b(int j, unsigned char* scratch, bool inject, std::vector<int>& host_ptr) {
    const ScratchLayout sl = scratch_layout(g, plan, j, cub);
    int* cnt = reinterpret_cast<int*>(scratch + sl.cnt);
    int* qbase = reinterpret_cast<int*>(scratch + sl.qbase);
    int* qcell = reinterpret_cast<int*>(scratch + sl.qcell);
    int* qseg = reinterpret_cast<int*>(scratch + sl.qseg);
    void* scan_temp = scratch + sl.scan;
    const cudaStream_t s = stream.value;
    const int mj = g.dims[j].levels;
    const long long Cop = g.C_op;
    Level& lj = lv[j];
    if (g.L_op > 0) {
        b_seg_counts<<<blocks_for(g.L_op, kThreads), kThreads, 0, s>>>(op_seg.ptr, cell_ptr.ptr, g.L_op, cnt);
        check(cudaGetLastError(), "b_seg_counts");
        std::size_t bytes = cub.scan_segments;
        check(cub::DeviceScan::ExclusiveSum(scan_temp, bytes, cnt, qbase, g.L_op, s), "scan");
        b_enumerate<<<blocks_for(g.L_op, kThreads), kThreads, 0, s>>>(op_seg.ptr, cell_ptr.ptr, qbase, g.L_op, qcell, qseg);
        check(cudaGetLastError(), "b_enumerate");
    }
    // The effect's arrays (the Build injection: after the first build kernels are queued).
    lj.ptr.allocate(ledger, static_cast<std::size_t>(mj) + 1U, "level_ptr", inject);
    lj.seg.allocate(ledger, static_cast<std::size_t>(Cop), "level_seg");
    for (int jj = 0; jj < g.nb; ++jj)
        if (jj != j) lj.other[jj].allocate(ledger, static_cast<std::size_t>(Cop), "level_others");
    lj.has_mult = g.mult != nullptr;
    if (lj.has_mult) lj.mult.allocate(ledger, static_cast<std::size_t>(Cop), "level_mult");
    int* cell_out = nullptr;
#if defined(XHDFE_REDUCED_CUDA_TEST_HOOKS)
    if (o.test_source_cells) {
        lj.cell.allocate(ledger, static_cast<std::size_t>(Cop), "level_cell (test hook)");
        cell_out = lj.cell.ptr;
    }
#endif
    Dims dd{};
    dd.nb = g.nb;
    for (int jj = 0; jj < g.nb; ++jj) {
        dd.off[jj] = g.dims[jj].offset;
        dd.cc[jj] = cc[jj].ptr;
    }
    if (!plan.radix[j]) {
        int* H = reinterpret_cast<int*>(scratch + sl.hist);
        int* totals = reinterpret_cast<int*>(scratch + sl.totals);
        const long long tiles = sl.tiles;
        if (tiles > 0) {  // C_op = 0: no tile, every level empty
            b_tile_hist<<<static_cast<unsigned>(tiles), kThreads, static_cast<std::size_t>(mj) * sizeof(int), s>>>(
                cc[j].ptr, qcell, Cop, kTile, mj, H);
            check(cudaGetLastError(), "b_tile_hist");
        }
        b_level_totals<<<blocks_for(mj, kThreads), kThreads, 0, s>>>(H, static_cast<int>(tiles), mj, totals);
        check(cudaGetLastError(), "b_level_totals");
        std::size_t bytes = cub.scan_levels[j];
        check(cub::DeviceScan::ExclusiveSum(scan_temp, bytes, totals, lj.ptr.ptr, mj, s), "scan");
        b_set_end<<<1, 1, 0, s>>>(lj.ptr.ptr, mj, static_cast<int>(Cop));
        check(cudaGetLastError(), "b_set_end");
        b_level_offsets<<<blocks_for(mj, kThreads), kThreads, 0, s>>>(H, lj.ptr.ptr, static_cast<int>(tiles), mj);
        check(cudaGetLastError(), "b_level_offsets");
        BScatter sc{};
        sc.code = cc[j].ptr; sc.qcell = qcell; sc.qseg = qseg; sc.C_op = Cop; sc.tile = kTile; sc.levels = mj; sc.j = j;
        sc.H = H; sc.d = dd; sc.mult = mult.ptr; sc.seg = lj.seg.ptr;
        for (int jj = 0; jj < g.nb; ++jj) sc.other[jj] = lj.other[jj].ptr;
        sc.omult = lj.has_mult ? lj.mult.ptr : nullptr;
        sc.cell = cell_out;
        constexpr int warps = 4;
        const std::size_t smem = static_cast<std::size_t>(warps) * static_cast<std::size_t>(mj) * sizeof(int);
        if (smem > (std::size_t{48} << 10)) throw Error(ErrorKind::Fatal, "CUDA reduced operator: scatter cursors");
        b_scatter<<<blocks_for(tiles, warps), 32 * warps, smem, s>>>(sc);
        check(cudaGetLastError(), "b_scatter");
    } else {
        // The DoubleBuffer overload (O(P) temporary) over the buffers counted in the admission.
        int* k0 = reinterpret_cast<int*>(scratch + sl.keys);
        int* v0 = reinterpret_cast<int*>(scratch + sl.vals);
        cub::DoubleBuffer<int> keys(k0, k0 + Cop), vals(v0, v0 + Cop);
        if (Cop > 0) {
            b_keys<<<blocks_for(Cop, kThreads), kThreads, 0, s>>>(cc[j].ptr, qcell, Cop, k0, v0);
            check(cudaGetLastError(), "b_keys");
        }
        std::size_t bytes = cub.radix[j];
        check(cub::DeviceRadixSort::SortPairs(scratch + sl.radix, bytes, keys, vals, static_cast<int>(Cop), 0, radix_bits(mj), s),
              "radix sort");
        b_ptr_from_sorted<<<blocks_for(static_cast<long long>(mj) + 1, kThreads), kThreads, 0, s>>>(keys.Current(), Cop, mj, lj.ptr.ptr);
        check(cudaGetLastError(), "b_ptr");
        BGather bg{};
        bg.vals = vals.Current(); bg.qcell = qcell; bg.qseg = qseg; bg.n = Cop; bg.j = j; bg.d = dd; bg.mult = mult.ptr;
        bg.seg = lj.seg.ptr;
        for (int jj = 0; jj < g.nb; ++jj) bg.other[jj] = lj.other[jj].ptr;
        bg.omult = lj.has_mult ? lj.mult.ptr : nullptr;
        bg.cell = cell_out;
        b_gather<<<blocks_for(Cop, kThreads), kThreads, 0, s>>>(bg);
        check(cudaGetLastError(), "b_gather");
    }
    // B's ptr to the host, for the units (its download and synchronisation are part of the set-up).
    host_ptr.assign(static_cast<std::size_t>(mj) + 1U, 0);
    check(cudaMemcpyAsync(host_ptr.data(), lj.ptr.ptr, host_ptr.size() * sizeof(int), cudaMemcpyDeviceToHost, s), "ptr download");
    check(cudaStreamSynchronize(s), "level build");
}

void Operator::State::run_kernels() {
    const cudaStream_t s = stream.value;
    Dims d{};
    d.nb = g.nb;
    for (int j = 0; j < g.nb; ++j) {
        d.off[j] = g.dims[j].offset;
        d.cc[j] = cc[j].ptr;
    }
    K1Params k{};
    k.in = in.ptr; k.T = T.ptr; k.PA = PA.ptr; k.op_seg = op_seg.ptr; k.k1_seg = k1_seg.ptr; k.cell_ptr = cell_ptr.ptr;
    k.seg_inv = seg_inv.ptr;
    k.mult = g.mult ? mult.ptr : nullptr; k.d = d; k.npriv = plan.npriv; k.S = plan.S;
    for (int q = 0; q < plan.npriv; ++q) { k.priv_j[q] = plan.priv_j[q]; k.priv_base[q] = plan.priv_base[q]; }
    k.L_op = g.L_op; k.short_max = o.short_seg;
    k.ws = org.ws; k.gpw = org.gpw; k.G = org.groups;
    const long long blocks = k1_blocks(g.L_op, width);
    const std::size_t smem = k1_smem_bytes(plan.S, width);
    if (blocks > 0) {
        k1_segments<kK1Cache><<<static_cast<unsigned>(blocks), kThreads, smem, s>>>(k);
        check(cudaGetLastError(), "k1");
    }
    record_event(3);
    if (plan.n_long > 0) {
        K1LParams q{};
        q.k = k; q.long_pos = long_pos.ptr; q.long_base = long_base.ptr; q.PL = PL.ptr; q.item = o.item;
        q.pa_first = static_cast<int>(blocks);
        k1_long<<<static_cast<unsigned>(plan.n_long), kThreads, smem, s>>>(q);
        check(cudaGetLastError(), "k1l");
    }
    record_event(4);
    for (int j = 0; j < g.nb; ++j) {
        if (plan.cls[j] == 'P') continue;
        Level& lj = lv[j];
        if (lj.units == 0) continue;
        K2Params p{};
        p.in = in.ptr; p.T = T.ptr; p.out = out.ptr; p.PB = PB.ptr;
        p.u_level = lj.u_level.ptr; p.u_begin = lj.u_begin.ptr; p.u_end = lj.u_end.ptr; p.u_slot = lj.u_slot.ptr;
        p.order = lj.order.ptr; p.units = lj.units;
        p.seg = lj.seg.ptr;
        for (int jj = 0; jj < g.nb; ++jj) p.other[jj] = jj == j ? nullptr : lj.other[jj].ptr;
        p.mult = lj.has_mult ? lj.mult.ptr : nullptr;
        p.j = j; p.d = d; p.ws = org.ws; p.gpw = org.gpw; p.G = org.groups;
        k2_levels<<<blocks_for(lj.units, org.groups), kThreads, 0, s>>>(p);
        check(cudaGetLastError(), "k2");
    }
    record_event(5);
    if (plan.S > 0) {
        K3PParams p{};
        p.PA = PA.ptr; p.out = out.ptr; p.slots = blocks + plan.n_long; p.S = plan.S; p.ws = org.ws; p.lam_out = lam_out.ptr;
        k3_private<<<static_cast<unsigned>(plan.S * org.ws), kThreads, 0, s>>>(p);
        check(cudaGetLastError(), "k3 private");
    }
    if (long_levels_total > 0) {
        K3LParams p{};
        p.PB = PB.ptr; p.out = out.ptr; p.lv_out = lvl_out.ptr; p.lv_base = lvl_base.ptr; p.lv_items = lvl_items.ptr; p.ws = org.ws;
        k3_long<<<static_cast<unsigned>(long_levels_total), kThreads, 0, s>>>(p);
        check(cudaGetLastError(), "k3 long");
    }
    record_event(6);
}

Operator::Operator(const GeometryView& geometry, const Options& options, Admission* record) : state_(std::make_unique<State>()) {
    state_->setup(geometry, options, record);
}
Operator::~Operator() = default;
const Statistics& Operator::statistics() const noexcept { return state_->st; }
const Admission& Operator::admission() const noexcept { return state_->adm; }
char Operator::effect_class(int j) const noexcept { return j >= 0 && j < kMaxDims ? state_->plan.cls[j] : 0; }
std::size_t Operator::host_state_bytes() noexcept { return sizeof(Operator) + sizeof(State); }
std::size_t Operator::live_device_bytes() noexcept {
#if defined(XHDFE_REDUCED_CUDA_TEST_HOOKS)
    return g_live_bytes.load();
#else
    return 0;
#endif
}

void Operator::begin_group(int width) {
    State& s = *state_;
    if (width < 1 || width > s.o.lanes) throw Error(ErrorKind::Fatal, "CUDA reduced operator: group width out of range");
    s.width = width;
    s.org = org_for(width);
}

void Operator::register_host(double* begin, std::size_t count) {
    State& s = *state_;
    if (s.reg_begin != nullptr) throw Error(ErrorKind::Fatal, "CUDA reduced operator: the rows are already registered");
    if (begin == nullptr || count == 0) throw Error(ErrorKind::Fatal, "CUDA reduced operator: an empty registration");
    const std::size_t bytes = checked_mul(count, sizeof(double));
    const auto t0 = Clock::now();
    if (s.o.test_failure == TestFailure::Register)
        check(cudaErrorMemoryAllocation, "registration of the operand and result rows (test injection)", ErrorKind::Pinned);
    check(cudaHostRegister(begin, bytes, cudaHostRegisterDefault), "registration of the operand and result rows", ErrorKind::Pinned);
    s.reg_begin = begin;
    s.reg_count = count;
    if (s.o.profile) s.st.register_seconds += seconds(t0);
}

void Operator::unregister_host() noexcept {
    State& s = *state_;
    if (s.reg_begin == nullptr) return;
    const auto t0 = Clock::now();
    if (s.stream.value) (void)cudaStreamSynchronize(s.stream.value);  // nothing in flight
    (void)cudaHostUnregister(s.reg_begin);
    s.reg_begin = nullptr;
    s.reg_count = 0;
    if (s.o.profile) s.st.unregister_seconds += seconds(t0);
}

void Operator::apply(const double* input, double* output) {
    State& s = *state_;
    if (s.width < 1) throw Error(ErrorKind::Fatal, "CUDA reduced operator: no group width");
    const int W = s.o.lanes, ws = s.width;
    const long long m = s.g.m;
    const std::size_t count = static_cast<std::size_t>(m) * static_cast<std::size_t>(W);
    const std::size_t bytes = count * sizeof(double);
    // Only the registered rows: a pageable transfer would be a silent slower path.
    const auto inside = [&](const double* p) {
        const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p), b = reinterpret_cast<std::uintptr_t>(s.reg_begin);
        return s.reg_begin != nullptr && a >= b && (a - b) % sizeof(double) == 0 &&
               (a - b) / sizeof(double) <= s.reg_count && s.reg_count - (a - b) / sizeof(double) >= count;
    };
    if (!inside(input) || !inside(output))
        throw Error(ErrorKind::Fatal, "CUDA reduced operator: operand or result outside the registered rows");
    const auto t0 = s.o.profile ? Clock::now() : Clock::time_point{};
    const cudaStream_t st = s.stream.value;
    s.record_event(0);
    if (ws == W) {
        check(cudaMemcpyAsync(s.in.ptr, input, bytes, cudaMemcpyHostToDevice, st), "operand upload");
        s.record_event(1);
    } else {
        check(cudaMemcpyAsync(s.inW.ptr, input, bytes, cudaMemcpyHostToDevice, st), "operand upload");
        s.record_event(1);
        k_pack<<<blocks_for(m * ws, kThreads), kThreads, 0, st>>>(s.inW.ptr, s.in.ptr, m, W, ws);
        check(cudaGetLastError(), "pack");
    }
    s.record_event(2);
    s.run_kernels();
    if (ws == W) {
        s.record_event(7);
        check(cudaMemcpyAsync(output, s.out.ptr, bytes, cudaMemcpyDeviceToHost, st), "result download");
    } else {
        k_unpack<<<blocks_for(m * W, kThreads), kThreads, 0, st>>>(s.out.ptr, s.outW.ptr, m, W, ws);
        check(cudaGetLastError(), "unpack");
        s.record_event(7);
        check(cudaMemcpyAsync(output, s.outW.ptr, bytes, cudaMemcpyDeviceToHost, st), "result download");
    }
    s.record_event(8);
    check(cudaStreamSynchronize(st), "operator completion");
    if (s.o.profile) {
        float ms[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < 8; ++k) check(cudaEventElapsedTime(&ms[k], s.ev[k].value, s.ev[k + 1].value), "timing");
        Statistics& t = s.st;
        t.h2d_dma_ms += ms[0]; t.pack_ms += ms[1]; t.k1_ms += ms[2]; t.k1l_ms += ms[3]; t.k2_ms += ms[4]; t.k3_ms += ms[5];
        t.unpack_ms += ms[6]; t.d2h_dma_ms += ms[7];
        t.h2d_ms += ms[0] + ms[1];
        t.kernel_ms += ms[2] + ms[3] + ms[4] + ms[5];
        t.d2h_ms += ms[6] + ms[7];
        t.apply_seconds += seconds(t0);
    }
    ++s.st.calls;
    s.st.h2d_bytes += bytes;
    s.st.d2h_bytes += bytes;
}

Operator::LevelOrderCopy Operator::download_level_order(int j) const {
    State& s = *state_;
    LevelOrderCopy c;
    if (j < 0 || j >= s.g.nb || s.plan.cls[j] != 'B') return c;
    check(cudaStreamSynchronize(s.stream.value), "download");
    auto get = [&](const DeviceArray<int>& a, std::vector<int>& v) {
        v.resize(a.count);
        if (a.count) check(cudaMemcpy(v.data(), a.ptr, a.count * sizeof(int), cudaMemcpyDeviceToHost), "download");
    };
    const State::Level& lj = s.lv[j];
    get(lj.ptr, c.ptr);
    get(lj.seg, c.seg);
    get(lj.cell, c.cell);
    if (lj.has_mult) get(lj.mult, c.mult);
    c.other.resize(static_cast<std::size_t>(s.g.nb));
    for (int jj = 0; jj < s.g.nb; ++jj)
        if (jj != j) get(lj.other[jj], c.other[static_cast<std::size_t>(jj)]);
    return c;
}

}  // namespace hdfe::detail::reduced_cuda
