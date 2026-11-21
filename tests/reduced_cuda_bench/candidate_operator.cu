// Candidate CUDA reduced operator, bench only (package 2, stage 2b, 17 November 2026).
// See candidate_operator.hpp for the organisation and DESENHO_P2.md (Amendment 1, B1-B10).
#include "candidate_operator.hpp"

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace xhdfe_bench::cand {
namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

constexpr int kThreads = 256;  // threads per block of every kernel but the scatter
constexpr int kMaxDims = 4;

// The CUDA errors keep the mapping of the tier's operator (Memory only for an
// allocation that ran out; no device or driver: Unavailable; anything else
// Fatal). Page-locking refused for lack of resources is Pinned (B7); an invalid
// argument, a duplicate registration or an earlier asynchronous error is not.
void check(cudaError_t status, const char* what, ErrorKind on_memory = ErrorKind::Fatal) {
    if (status == cudaSuccess) return;
    ErrorKind kind = ErrorKind::Fatal;
    if (status == cudaErrorMemoryAllocation && on_memory != ErrorKind::Fatal) {
        kind = on_memory;
        (void)cudaGetLastError();  // clears only this allocation failure (sticky errors stay)
    } else if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
        kind = ErrorKind::Unavailable;
    }
    throw Error(kind, std::string("candidate operator: ") + what + ": " + cudaGetErrorString(status));
}

template <class T>
struct DeviceArray {
    T* ptr = nullptr;
    std::size_t count = 0;
    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    ~DeviceArray() { release(); }
    void allocate(std::size_t n, const char* what) {
        release();
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw Error(ErrorKind::Memory, "byte count overflow");
        if (n) check(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)), what, ErrorKind::Memory);
        count = n;
    }
    void release() {
        if (ptr) (void)cudaFree(ptr);
        ptr = nullptr;
        count = 0;
    }
    std::size_t bytes() const { return count * sizeof(T); }
};

struct PinnedArray {
    double* ptr = nullptr;
    std::size_t count = 0;
    ~PinnedArray() { release(); }
    void allocate(std::size_t n) {
        release();
        if (n) check(cudaHostAlloc(reinterpret_cast<void**>(&ptr), n * sizeof(double), cudaHostAllocDefault),
                     "page-locked staging", ErrorKind::Pinned);
        count = n;
    }
    void release() {
        if (ptr) (void)cudaFreeHost(ptr);
        ptr = nullptr;
        count = 0;
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

int next_pow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

// ---------------------------------------------------------------------------
// Thread organisation: groups of gsz threads inside a warp (never across one),
// groups_per_warp of them; lane < width computes.
struct Org {
    int ws = 8;           // row stride on the device (doubles)
    int gsz = 8;          // threads per group
    int gpw = 4;          // groups per warp
    int groups = 32;      // groups per block of kThreads
    int width = 8;        // live lanes of the current group
};

__device__ inline void place(int gsz, int gpw, int& g, int& lane, bool& in_group) {
    const int warp = threadIdx.x >> 5, lw = threadIdx.x & 31;
    const int giw = lw / gsz;
    in_group = giw < gpw;
    g = warp * gpw + giw;
    lane = lw - giw * gsz;
}

struct Dims {
    int nb;
    int off[kMaxDims];
    const int* cc[kMaxDims];
};

// K1: segments of at most short_max cells; one group per segment.
struct K1Params {
    const double* in; double* T; double* PA;
    const int* op_seg; const int* cell_ptr; const double* seg_inv; const int* mult;
    const int* k1_seg;  // K1's operator segments: op_seg, or op_seg grouped by length (seg_order 1); K1L reads op_seg
    Dims d;
    int npriv; int priv_j[kMaxDims]; int priv_base[kMaxDims]; int S;
    int L_op; int short_max; int seg_per_group;
    int ws, gsz, gpw, G, width;
};

// u at cell c: the operand rows of every remaining effect, added in effect order from 0.0.
__device__ __forceinline__ double gather_u(const K1Params& p, int c, int lane) {
    double u = 0.0;
#pragma unroll
    for (int j = 0; j < kMaxDims; ++j)
        if (j < p.d.nb) u += __ldg(p.in + static_cast<std::size_t>(p.d.off[j] + __ldg(p.d.cc[j] + c)) * p.ws + lane);
    return u;
}

// CACHE > 0: a segment of at most CACHE cells keeps its u in registers (fully
// unrolled, so the loads of all its cells can be in flight together); the
// arithmetic and its order are those of the two walks.
template <int CACHE>
__global__ void k1_segments(K1Params p) {
    extern __shared__ double acc[];  // G x S x ws, private to each group
    int g, lane;
    bool in_group;
    place(p.gsz, p.gpw, g, lane, in_group);
    const std::size_t acc_count = static_cast<std::size_t>(p.G) * p.S * p.ws;
    for (std::size_t e = threadIdx.x; e < acc_count; e += blockDim.x) acc[e] = 0.0;  // groups without work included
    __syncthreads();
    if (in_group && lane < p.width) {
        const long long base = static_cast<long long>(blockIdx.x) * p.G * p.seg_per_group + g;
        for (int i = 0; i < p.seg_per_group; ++i) {
            const long long s = base + static_cast<long long>(p.G) * i;
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
    for (int e = threadIdx.x; e < p.S * p.width; e += blockDim.x) {  // strided, one owner per output
        const int lam = e / p.width, ln = e - lam * p.width;
        double sum = 0.0;
        for (int gg = 0; gg < p.G; ++gg) sum += acc[(static_cast<std::size_t>(gg) * p.S + lam) * p.ws + ln];
        p.PA[(static_cast<std::size_t>(blockIdx.x) * p.S + lam) * p.ws + ln] = sum;
    }
}

// The K1 instantiation of a register cache (0, 8, 16, 24 or 32 cells; values between round down).
// The cache changes where u is kept, not which values are added or in what order.
using K1Fn = void (*)(K1Params);
K1Fn k1_kernel(int cache) {
    return cache >= 32 ? k1_segments<32> : cache >= 24 ? k1_segments<24> : cache >= 16 ? k1_segments<16>
         : cache >= 8 ? k1_segments<8> : k1_segments<0>;
}

// K1L: one block per long segment. Items of `item` cells, each summed in cell
// order by one group; the item sums combined by an in-place pairwise tree on
// item index (at stride k, item i += item i + k for i a multiple of 2k) in the
// global partials PL; then the deviations of the private effects.
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
    place(p.gsz, p.gpw, g, lane, in_group);
    const bool live = in_group && lane < p.width;
    const std::size_t acc_count = static_cast<std::size_t>(p.G) * p.S * p.ws;
    for (std::size_t e = threadIdx.x; e < acc_count; e += blockDim.x) acc[e] = 0.0;
    const int s = q.long_pos[blockIdx.x];
    const int l = p.op_seg[s];
    const int c0 = p.cell_ptr[l], c1 = p.cell_ptr[l + 1];
    const int items = (c1 - c0 + q.item - 1) / q.item;
    double* PL = q.PL + static_cast<std::size_t>(q.long_base[blockIdx.x]) * p.ws;
    if (live) {
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
        for (int e = threadIdx.x; e < pairs * p.width; e += blockDim.x) {
            const int pr = e / p.width, ln = e - pr * p.width;
            const int i = pr * 2 * stride;
            if (i + stride < items) PL[static_cast<std::size_t>(i) * p.ws + ln] += PL[static_cast<std::size_t>(i + stride) * p.ws + ln];
        }
        __syncthreads();
    }
    if (threadIdx.x < p.width) {
        const double mean = PL[threadIdx.x] * p.seg_inv[l];
        mean_s[threadIdx.x] = mean;
        p.T[static_cast<std::size_t>(l) * p.ws + threadIdx.x] = mean;
    }
    __syncthreads();
    if (live && p.S > 0) {
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
    for (int e = threadIdx.x; e < p.S * p.width; e += blockDim.x) {
        const int lam = e / p.width, ln = e - lam * p.width;
        double sum = 0.0;
        for (int gg = 0; gg < p.G; ++gg) sum += acc[(static_cast<std::size_t>(gg) * p.S + lam) * p.ws + ln];
        p.PA[(static_cast<std::size_t>(q.pa_first + blockIdx.x) * p.S + lam) * p.ws + ln] = sum;
    }
}

// K2: the levels of one transposed or built effect j; one group per unit.
struct K2Params {
    const double* in; const double* T; double* out; double* PB;
    const int* u_level; const int* u_begin; const int* u_end; const int* u_slot; const int* order; long long units;
    const int* seg; const int* other[kMaxDims]; const int* mult;
    int j; Dims d;
    int ws, gsz, gpw, G, width;
};

__global__ void k2_levels(K2Params p) {
    int g, lane;
    bool in_group;
    place(p.gsz, p.gpw, g, lane, in_group);
    const long long ui = static_cast<long long>(blockIdx.x) * p.G + g;
    if (!in_group || ui >= p.units || lane >= p.ws) return;  // no barrier in this kernel
    const int u = p.order ? p.order[ui] : static_cast<int>(ui);
    const int lev = p.u_level[u], slot = p.u_slot[u];
    double* dst = slot < 0 ? p.out + static_cast<std::size_t>(p.d.off[p.j] + lev) * p.ws
                           : p.PB + static_cast<std::size_t>(slot) * p.ws;
    if (lane >= p.width) {  // padding lane of the padded form
        dst[lane] = 0.0;
        return;
    }
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

// K3: one block per output; thread t adds partials t, t + 256, ... in order,
// then a halving tree over t.
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
    const double* PA; double* out; long long slots; int S; int ws; int width;
    const int* lam_out;  // S: output level index (offset of the effect + level)
};

__global__ void k3_private(K3PParams p) {
    __shared__ double buf[kThreads];
    const int lam = blockIdx.x / p.ws, ln = blockIdx.x - lam * p.ws;
    double* dst = p.out + static_cast<std::size_t>(p.lam_out[lam]) * p.ws + ln;
    if (ln >= p.width) {
        if (threadIdx.x == 0) *dst = 0.0;
        return;
    }
    double v = 0.0;
    for (long long s = threadIdx.x; s < p.slots; s += blockDim.x) v += p.PA[(static_cast<std::size_t>(s) * p.S + lam) * p.ws + ln];
    const double r = block_tree(v, buf);
    if (threadIdx.x == 0) *dst = r;
}

struct K3LParams {
    const double* PB; double* out; const int* lv_out; const int* lv_base; const int* lv_items; int ws; int width;
};

__global__ void k3_long(K3LParams p) {
    __shared__ double buf[kThreads];
    const int lv = blockIdx.x;
    const int base = p.lv_base[lv], items = p.lv_items[lv];
    double* dst = p.out + static_cast<std::size_t>(p.lv_out[lv]) * p.ws;
    for (int ln = 0; ln < p.ws; ++ln) {
        if (ln >= p.width) {
            if (threadIdx.x == 0) dst[ln] = 0.0;
            continue;
        }
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

// Stable scatter: one warp per tile, positions in order, 32 at a time; the rank
// of a position among the equal codes of its 32 is the number of lower lanes
// with that code (__match_any_sync), the cursors advance by the count after
// every lane has read them. No atomic decides a position.
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
            if (p.cell) p.cell[pos] = c;
        }
    }
}

// Radix alternative: keys = codes in operator-cell order, values = positions.
__global__ void b_keys(const int* code, const int* qcell, long long C_op, int* keys, int* vals) {
    const long long q = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= C_op) return;
    keys[q] = code[qcell[q]];
    vals[q] = static_cast<int>(q);
}
__global__ void b_ptr_from_sorted(const int* keys, long long n, int levels, int* ptr) {
    const int lev = blockIdx.x * blockDim.x + threadIdx.x;
    if (lev > levels) return;
    long long lo = 0, hi = n;  // first position with key >= lev
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
    if (p.cell) p.cell[a] = c;
}

// Registered transfers with compacted device rows: the caller's rows of stride W go over the link whole;
// the device packs the live lanes and unpacks the result, writing +0.0 into the padding lanes.
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

unsigned blocks_for(long long n, int threads) {
    const long long b = (n + threads - 1) / threads;
    if (b > static_cast<long long>(INT_MAX)) throw Error(ErrorKind::Memory, "grid too large");
    return static_cast<unsigned>(std::max(1LL, b));
}

template <class T>
void upload(DeviceArray<T>& dst, const T* src, std::size_t n, cudaStream_t stream, const char* what, std::size_t& bytes) {
    dst.allocate(n, what);
    if (n) check(cudaMemcpyAsync(dst.ptr, src, n * sizeof(T), cudaMemcpyHostToDevice, stream), what);
    bytes += n * sizeof(T);
}

// Uploads through two page-locked chunks: the host copy of one chunk (16 threads)
// overlaps the DMA of the other; a chunk is reused only after its DMA is done.
struct StagedUploader {
    double* buf[2] = {nullptr, nullptr};
    cudaEvent_t done[2] = {nullptr, nullptr};
    std::size_t chunk = 0;
    int next = 0;
    bool used[2] = {false, false};
    void put(void* dst, const void* src, std::size_t bytes, cudaStream_t stream) {
        const char* s8 = static_cast<const char*>(src);
        char* d8 = static_cast<char*>(dst);
        for (std::size_t off = 0; off < bytes; off += chunk) {
            const std::size_t len = std::min(chunk, bytes - off);
            const int b = next;
            next ^= 1;
            if (used[b]) check(cudaEventSynchronize(done[b]), "staged upload");
            char* stage = reinterpret_cast<char*>(buf[b]);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(16)
#endif
            for (int part = 0; part < 16; ++part) {
                const std::size_t a0 = len * static_cast<std::size_t>(part) / 16, a1 = len * static_cast<std::size_t>(part + 1) / 16;
                std::memcpy(stage + a0, s8 + off + a0, a1 - a0);
            }
            check(cudaMemcpyAsync(d8 + off, stage, len, cudaMemcpyHostToDevice, stream), "staged upload");
            check(cudaEventRecord(done[b], stream), "staged upload");
            used[b] = true;
        }
    }
};

template <class T>
void upload_staged(StagedUploader* su, DeviceArray<T>& dst, const T* src, std::size_t n, cudaStream_t stream, const char* what, std::size_t& bytes) {
    if (su == nullptr) return upload(dst, src, n, stream, what, bytes);
    dst.allocate(n, what);
    if (n) su->put(dst.ptr, src, n * sizeof(T), stream);
    bytes += n * sizeof(T);
}

}  // namespace

// ---------------------------------------------------------------------------
struct Operator::State {
    GeometryView g;
    Options o;
    Stats st;
    Admission adm;
    Org org;
    int width = 0;
    int occupancy_width = 0;
    Stream stream;
    Event ev[8];
    // resident
    DeviceArray<int> cell_ptr, op_seg, mult;
    DeviceArray<int> k1_seg;  // seg_order 1: op_seg grouped by length, for K1 only
    DeviceArray<double> seg_inv;
    DeviceArray<int> cc[kMaxDims];
    char cls[kMaxDims] = {0, 0, 0, 0};
    struct Level {
        DeviceArray<int> ptr, seg, mult, cell;
        DeviceArray<int> other[kMaxDims];
        DeviceArray<int> u_level, u_begin, u_end, u_slot, order;
        long long units = 0;
        std::vector<int> host_ptr;
        bool has_mult = false;
        const int* seg_p = nullptr; const int* other_p[kMaxDims] = {nullptr, nullptr, nullptr, nullptr}; const int* mult_p = nullptr;
        long long pb_first = 0, pb_slots = 0;
    } lv[kMaxDims];
    int npriv = 0, priv_j[kMaxDims] = {0, 0, 0, 0}, priv_base[kMaxDims] = {0, 0, 0, 0}, S = 0;
    DeviceArray<int> lam_out;
    std::vector<int> long_pos_h;
    std::vector<long long> long_base_h;
    DeviceArray<int> long_pos;
    DeviceArray<long long> long_base;
    long long pl_items = 0;
    DeviceArray<int> lvl_out, lvl_base, lvl_items;   // long levels of every T/B effect, for K3
    int long_levels_total = 0;
    long long k1_blocks = 0;
    // per product
    DeviceArray<double> in, out, T, PA, PB, PL;
    DeviceArray<double> inW, outW;  // registered transfers with compacted device rows
    PinnedArray pin_in, pin_out, chunk[2];
    Event chunk_ev[2];
    const double* reg_in = nullptr;
    double* reg_out = nullptr;

    void setup(const GeometryView& view, const Options& opt);
    void configure(int w);
    void run_kernels();
};

namespace {
std::size_t add(std::vector<std::pair<std::string, std::size_t>>& list, const std::string& name, std::size_t bytes) {
    list.emplace_back(name, bytes);
    return bytes;
}
std::size_t total(const std::vector<std::pair<std::string, std::size_t>>& list) {
    std::size_t s = 0;
    for (const auto& e : list) {
        if (e.second > std::numeric_limits<std::size_t>::max() - s) throw Error(ErrorKind::Memory, "byte count overflow");
        s += e.second;
    }
    return s;
}
}  // namespace

std::string Admission::text() const {
    std::ostringstream out;
    auto list = [&](const char* title, const std::vector<std::pair<std::string, std::size_t>>& v) {
        out << title << ':';
        for (const auto& e : v) out << ' ' << e.first << '=' << e.second;
        out << ';';
    };
    list("device_resident", device_resident);
    list("device_setup_scratch", device_setup_scratch);
    list("device_product", device_product);
    list("host", host);
    out << " device_setup_peak=" << device_setup_peak << " device_iteration_peak=" << device_iteration_peak
        << " device_peak=" << device_peak << " host_peak=" << host_peak << " free_at_entry=" << free_at_entry;
    return out.str();
}

void Operator::State::configure(int w) {
    if (w < 1 || w > o.lanes) throw Error(ErrorKind::Fatal, "candidate operator: width out of range");
    width = w;
    switch (o.width_mode) {
        case WidthMode::Padded: org.ws = o.lanes; org.gsz = o.lanes; break;
        case WidthMode::Sub8: org.ws = w; org.gsz = next_pow2(w); break;
        case WidthMode::InWarp: org.ws = w; org.gsz = w; break;
    }
    org.gpw = 32 / org.gsz;
    org.groups = (kThreads / 32) * org.gpw;
    org.width = w;
}

void Operator::State::setup(const GeometryView& view, const Options& opt) {
    const auto t_all = Clock::now();
    g = view;
    o = opt;
    if (g.nb < 1 || g.nb > kMaxDims) throw Error(ErrorKind::Fatal, "candidate operator: 1 to 4 remaining effects");
    if (o.lanes != 1 && o.lanes != 2 && o.lanes != 4 && o.lanes != 8) throw Error(ErrorKind::Fatal, "candidate operator: lanes 1, 2, 4 or 8");
    if (g.C_op > INT_MAX || g.C > INT_MAX) throw Error(ErrorKind::Memory, "candidate operator: cells beyond int");
    configure(o.lanes);

    // Classes: private while the cumulative levels stay <= private_max (smallest effects first), else
    // transposed when the CPU made a transpose, else built.
    std::vector<int> by_size(g.nb);
    std::iota(by_size.begin(), by_size.end(), 0);
    std::stable_sort(by_size.begin(), by_size.end(), [&](int a, int b) { return g.dims[a].levels < g.dims[b].levels; });
    int cumulative = 0;
    for (int j : by_size) {
        if (cumulative + g.dims[j].levels <= o.private_max) {
            cls[j] = 'P';
            cumulative += g.dims[j].levels;
        }
    }
    for (int j = 0; j < g.nb; ++j)
        if (!cls[j]) cls[j] = g.dims[j].transposed ? 'T' : 'B';
    for (int j = 0; j < g.nb; ++j)
        if (cls[j] == 'P') { priv_j[npriv] = j; priv_base[npriv] = S; S += g.dims[j].levels; ++npriv; }
    st.private_levels = S;
    st.classes.assign(cls, cls + g.nb);

    // Long segments (exclusive with K1 by the same test) and their items.
    long long items_long = 0;
    for (int s = 0; s < g.L_op; ++s) {
        const int l = g.op_seg[s];
        const int n = g.cell_ptr[l + 1] - g.cell_ptr[l];
        if (n > o.short_seg) {
            long_pos_h.push_back(s);
            long_base_h.push_back(items_long);
            items_long += (n + o.item - 1) / o.item;
        }
    }
    pl_items = items_long;
    st.long_segments = static_cast<int>(long_pos_h.size());

    // Admission (B1): every device buffer by phase, before the first allocation.
    const auto t_adm = Clock::now();
    const std::size_t W = static_cast<std::size_t>(o.lanes), m = static_cast<std::size_t>(g.m), L = static_cast<std::size_t>(g.L);
    const std::size_t Cop = static_cast<std::size_t>(g.C_op), C = static_cast<std::size_t>(g.C);
    // The K1 grid depends on the groups per block; size P_A for the largest grid over every width the operator may see.
    long long pa_blocks = 0;
    for (int w = 1; w <= o.lanes; ++w) {
        Org t = org;
        configure(w);
        const long long blocks = (static_cast<long long>(g.L_op) + static_cast<long long>(org.groups) * o.seg_per_group - 1) /
                                 (static_cast<long long>(org.groups) * o.seg_per_group);
        pa_blocks = std::max(pa_blocks, blocks);
        org = t;
    }
    configure(o.lanes);
    st.k1_blocks = pa_blocks;
    st.pa_slots = pa_blocks + static_cast<long long>(long_pos_h.size());
    auto& R = adm.device_resident;
    auto& X = adm.device_setup_scratch;
    auto& P = adm.device_product;
    add(R, "cell_ptr", (L + 1) * sizeof(int));
    add(R, "op_seg", static_cast<std::size_t>(g.L_op) * sizeof(int));
    if (o.seg_order == 1) add(R, "k1_segment_order", static_cast<std::size_t>(g.L_op) * sizeof(int));
    add(R, "seg_inv", L * sizeof(double));
    add(R, "codes", static_cast<std::size_t>(g.nb) * C * sizeof(int));
    if (g.mult) add(R, "mult", C * sizeof(int));
    add(R, "private_outputs", static_cast<std::size_t>(S) * sizeof(int));
    add(R, "long_segment_lists", long_pos_h.size() * (sizeof(int) + sizeof(long long)));
    std::size_t scratch_b = 0;
    long long total_long_levels = 0;
    for (int j = 0; j < g.nb; ++j) {
        if (cls[j] == 'P') continue;
        const std::size_t mj = static_cast<std::size_t>(g.dims[j].levels);
        const bool hm = g.mult != nullptr;
        add(R, "level_ptr_" + std::to_string(j), (mj + 1) * sizeof(int));
        add(R, "level_seg_others_" + std::to_string(j), static_cast<std::size_t>(g.nb) * Cop * sizeof(int));
        if (hm) add(R, "level_mult_" + std::to_string(j), Cop * sizeof(int));
        // Units and partial slots (B1): exact from ptr for T effects. For B effects ptr exists only after
        // the build, so the admission takes the bound units <= m_j + C_op/item (each level one unit plus
        // ceil(n/item) - 1 <= n/item more for a long one) and slots <= C_op/item + C_op/(short_level + 1)
        // (each long level has more than short_level entries); the allocation after the build is exact.
        long long units = 0, slots = 0, longs = 0;
        if (cls[j] == 'T') {
            for (int lev = 0; lev < g.dims[j].levels; ++lev) {
                const long long n = static_cast<long long>(g.dims[j].tr_ptr[lev + 1]) - g.dims[j].tr_ptr[lev];
                if (n > o.short_level) { const long long k = (n + o.item - 1) / o.item; units += k; slots += k; ++longs; }
                else units += 1;
            }
        } else {
            units = static_cast<long long>(mj) + static_cast<long long>(Cop) / o.item;
            longs = static_cast<long long>(Cop) / (o.short_level + 1);
            slots = static_cast<long long>(Cop) / o.item + longs;
            add(R, "level_cell_" + std::to_string(j), Cop * sizeof(int));  // source cells, kept for the identity test
            const std::size_t tiles = (Cop + static_cast<std::size_t>(o.tile) - 1) / static_cast<std::size_t>(o.tile);
            std::size_t sb = 2 * static_cast<std::size_t>(g.L_op) * sizeof(int) + 2 * Cop * sizeof(int);  // counts, bases, cell, segment
            const bool radix = o.b_method == BMethod::Radix ||
                               (o.b_method == BMethod::Auto && mj * sizeof(int) > 12 * 1024);
            sb += radix ? 4 * Cop * sizeof(int) + (std::size_t{64} << 20)            // keys, values (double buffers), CUB temp bound
                        : tiles * mj * sizeof(int) + mj * sizeof(int);              // tile histograms (then cursors), totals
            scratch_b = std::max(scratch_b, sb);
            add(adm.host, "level_ptr_host_" + std::to_string(j), (mj + 1) * sizeof(int));
        }
        add(R, "units_" + std::to_string(j), static_cast<std::size_t>(units) * 5 * sizeof(int));
        add(adm.host, "unit_lists_" + std::to_string(j), static_cast<std::size_t>(units) * 5 * sizeof(int));
        add(P, "PB_" + std::to_string(j), static_cast<std::size_t>(slots) * W * sizeof(double));
        total_long_levels += longs;
    }
    add(R, "long_level_lists", static_cast<std::size_t>(total_long_levels) * 3 * sizeof(int));
    if (scratch_b) add(X, "b_build", scratch_b);
    add(X, "cub_scan_temp", std::size_t{16} << 20);
    add(P, "in", m * W * sizeof(double));
    add(P, "out", m * W * sizeof(double));
    add(P, "T", L * W * sizeof(double));
    add(P, "PA", static_cast<std::size_t>(st.pa_slots) * static_cast<std::size_t>(std::max(1, S)) * W * sizeof(double));
    add(P, "PL", static_cast<std::size_t>(pl_items) * W * sizeof(double));
    if (o.transfer == Transfer::Registered && o.width_mode != WidthMode::Padded)
        add(P, "registered_stride_pair", 2 * m * W * sizeof(double));
    const std::size_t resident = total(R), scratch = total(X), product = total(P);
    adm.device_setup_peak = resident + scratch + product;  // the product buffers are allocated before the B build
    adm.device_iteration_peak = resident + product;
    adm.device_peak = std::max(adm.device_setup_peak, adm.device_iteration_peak);
    // Host: the page-locked pair lives from the set-up to the release, with Work (B1).
    if (o.transfer == Transfer::Pinned) add(adm.host, "pinned_pair", 2 * m * W * sizeof(double));
    if (o.transfer == Transfer::PinnedChunks) add(adm.host, "pinned_chunks", 2 * o.chunk_bytes);
    if (o.staged_uploads) add(adm.host, "upload_chunks (set-up only)", 2 * o.chunk_bytes);
    add(adm.host, "long_lists_host", long_pos_h.capacity() * sizeof(int) + long_base_h.capacity() * sizeof(long long) +
                                     static_cast<std::size_t>(total_long_levels) * 3 * sizeof(int) + static_cast<std::size_t>(S) * sizeof(int));
    adm.host_peak = total(adm.host);
    int devices = 0;
    check(cudaGetDeviceCount(&devices), "device count", ErrorKind::Unavailable);
    if (!devices) throw Error(ErrorKind::Unavailable, "candidate operator: no device");
    check(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking), "stream");
    std::size_t free_b = 0, total_b = 0;
    check(cudaMemGetInfo(&free_b, &total_b), "free memory");
    adm.free_at_entry = free_b;
    const std::size_t limit = o.budget ? std::min(o.budget, free_b) : free_b;
    st.admission_s = since(t_adm);
    if (adm.device_peak > limit) throw Error(ErrorKind::Memory, "candidate operator: admission");
    for (Event& e : ev) check(cudaEventCreate(&e.value), "event");

    // Resident uploads: pageable (as today's operator) or through two page-locked chunks.
    auto t_up = Clock::now();
    std::size_t up = 0;
    PinnedArray up_chunk[2];
    Event up_done[2];
    StagedUploader stager;
    StagedUploader* su = nullptr;
    if (o.staged_uploads) {
        const std::size_t n = std::max<std::size_t>(1, o.chunk_bytes / sizeof(double));
        for (int b2 = 0; b2 < 2; ++b2) {
            up_chunk[b2].allocate(n);
            check(cudaEventCreateWithFlags(&up_done[b2].value, cudaEventDisableTiming), "event");
            stager.buf[b2] = up_chunk[b2].ptr;
            stager.done[b2] = up_done[b2].value;
        }
        stager.chunk = n * sizeof(double);
        su = &stager;
    }
    upload_staged(su, cell_ptr, g.cell_ptr, L + 1, stream.value, "cell_ptr", up);
    upload_staged(su, op_seg, g.op_seg, static_cast<std::size_t>(g.L_op), stream.value, "op_seg", up);
    if (o.seg_order == 1) {
        // K1's order: the operator segments by decreasing cells, stable (a counting sort over the lengths up to
        // short_seg + 1: K1 skips the longer ones, which K1L takes in the geometry's order).
        const auto t_ord = Clock::now();
        const int top = o.short_seg + 1;
        std::vector<long long> start(static_cast<std::size_t>(top) + 2, 0);
        auto key = [&](int s) {
            const int l = g.op_seg[s];
            const int n = g.cell_ptr[l + 1] - g.cell_ptr[l];
            return top - std::min(n, top);  // 0 for the longest
        };
        for (int s = 0; s < g.L_op; ++s) ++start[static_cast<std::size_t>(key(s)) + 1];
        for (int k = 0; k <= top; ++k) start[static_cast<std::size_t>(k) + 1] += start[static_cast<std::size_t>(k)];
        std::vector<int> order(static_cast<std::size_t>(g.L_op));
        for (int s = 0; s < g.L_op; ++s) order[static_cast<std::size_t>(start[static_cast<std::size_t>(key(s))]++)] = g.op_seg[s];
        upload_staged(su, k1_seg, order.data(), order.size(), stream.value, "k1_segment_order", up);
        check(cudaStreamSynchronize(stream.value), "k1 order");  // the host vector is freed below
        st.seg_order_s = since(t_ord);
    }
    upload_staged(su, seg_inv, g.seg_inv, L, stream.value, "seg_inv", up);
    for (int j = 0; j < g.nb; ++j) upload_staged(su, cc[j], g.cc[j], C, stream.value, "codes", up);
    if (g.mult) upload_staged(su, mult, g.mult, C, stream.value, "mult", up);
    for (int j = 0; j < g.nb; ++j) {
        if (cls[j] != 'T') continue;
        const auto& dv = g.dims[j];
        Level& L2 = lv[j];
        upload_staged(su, L2.ptr, dv.tr_ptr, static_cast<std::size_t>(dv.levels) + 1, stream.value, "tr_ptr", up);
        const std::size_t E = static_cast<std::size_t>(dv.tr_ptr[dv.levels]);
        upload_staged(su, L2.seg, dv.tr_seg, E, stream.value, "tr_seg", up);
        for (int jj = 0; jj < g.nb; ++jj)
            if (jj != j) upload_staged(su, L2.other[jj], dv.tr_other[jj], E, stream.value, "tr_other", up);
        L2.has_mult = dv.has_mult;  // has_mult == false: multiplicity one, .mult not read
        if (dv.has_mult) upload_staged(su, L2.mult, dv.tr_mult, E, stream.value, "tr_mult", up);
        L2.host_ptr.assign(dv.tr_ptr, dv.tr_ptr + dv.levels + 1);
    }
    check(cudaStreamSynchronize(stream.value), "uploads");  // nothing in flight when the chunks are freed
    st.upload_s = since(t_up);
    st.uploaded_bytes = up;
    auto t_alloc = Clock::now();
    in.allocate(m * W, "in");
    out.allocate(m * W, "out");
    T.allocate(L * W, "T");
    PA.allocate(static_cast<std::size_t>(st.pa_slots) * static_cast<std::size_t>(std::max(1, S)) * W, "PA");
    PL.allocate(static_cast<std::size_t>(pl_items) * W, "PL");
    if (o.transfer == Transfer::Registered && o.width_mode != WidthMode::Padded) {
        inW.allocate(m * W, "registered operand rows");
        outW.allocate(m * W, "registered result rows");
    }
    st.alloc_s = since(t_alloc);

    // B effects: build on the device, then download ptr for the units (B5).
    const auto t_b = Clock::now();
    for (int j = 0; j < g.nb; ++j) {
        if (cls[j] != 'B') continue;
        Level& L2 = lv[j];
        const int mj = g.dims[j].levels;
        Event e[7];
        for (Event& x : e) check(cudaEventCreate(&x.value), "event");
        check(cudaEventRecord(e[0].value, stream.value), "event");
        DeviceArray<int> cnt, qbase, qcell, qseg;
        cnt.allocate(static_cast<std::size_t>(g.L_op), "b_cnt");
        qbase.allocate(static_cast<std::size_t>(g.L_op), "b_qbase");
        qcell.allocate(Cop, "b_qcell");
        qseg.allocate(Cop, "b_qseg");
        b_seg_counts<<<blocks_for(g.L_op, kThreads), kThreads, 0, stream.value>>>(op_seg.ptr, cell_ptr.ptr, g.L_op, cnt.ptr);
        check(cudaGetLastError(), "b_seg_counts");
        std::size_t temp_bytes = 0;
        check(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, cnt.ptr, qbase.ptr, g.L_op, stream.value), "scan size");
        DeviceArray<unsigned char> temp;
        temp.allocate(std::max<std::size_t>(1, temp_bytes), "scan temp");
        check(cub::DeviceScan::ExclusiveSum(temp.ptr, temp_bytes, cnt.ptr, qbase.ptr, g.L_op, stream.value), "scan");
        b_enumerate<<<blocks_for(g.L_op, kThreads), kThreads, 0, stream.value>>>(op_seg.ptr, cell_ptr.ptr, qbase.ptr, g.L_op, qcell.ptr, qseg.ptr);
        check(cudaGetLastError(), "b_enumerate");
        check(cudaEventRecord(e[1].value, stream.value), "event");
        L2.ptr.allocate(static_cast<std::size_t>(mj) + 1, "b_ptr");
        L2.seg.allocate(Cop, "b_seg");
        for (int jj = 0; jj < g.nb; ++jj)
            if (jj != j) L2.other[jj].allocate(Cop, "b_other");
        L2.has_mult = g.mult != nullptr;
        if (L2.has_mult) L2.mult.allocate(Cop, "b_mult");
        L2.cell.allocate(Cop, "b_cell");
        BScatter sc{};
        Dims dd{};
        dd.nb = g.nb;
        for (int jj = 0; jj < g.nb; ++jj) { dd.off[jj] = g.dims[jj].offset; dd.cc[jj] = cc[jj].ptr; }
        const bool radix = o.b_method == BMethod::Radix ||
                           (o.b_method == BMethod::Auto && static_cast<std::size_t>(mj) * sizeof(int) > 12 * 1024);
        if (!radix) {
            const long long tiles = (static_cast<long long>(Cop) + o.tile - 1) / o.tile;
            DeviceArray<int> H, totals;
            H.allocate(static_cast<std::size_t>(tiles) * static_cast<std::size_t>(mj), "b_hist");
            totals.allocate(static_cast<std::size_t>(mj), "b_totals");
            if (tiles > 0) {  // C_op = 0: no tile, every level empty
                b_tile_hist<<<static_cast<unsigned>(tiles), kThreads, static_cast<std::size_t>(mj) * sizeof(int), stream.value>>>(
                    cc[j].ptr, qcell.ptr, static_cast<long long>(Cop), o.tile, mj, H.ptr);
                check(cudaGetLastError(), "b_tile_hist");
            }
            check(cudaEventRecord(e[2].value, stream.value), "event");
            b_level_totals<<<blocks_for(mj, kThreads), kThreads, 0, stream.value>>>(H.ptr, static_cast<int>(tiles), mj, totals.ptr);
            check(cudaGetLastError(), "b_level_totals");
            std::size_t t2 = 0;
            check(cub::DeviceScan::ExclusiveSum(nullptr, t2, totals.ptr, L2.ptr.ptr, mj, stream.value), "scan size");
            DeviceArray<unsigned char> temp2;
            temp2.allocate(std::max<std::size_t>(1, t2), "scan temp");
            check(cub::DeviceScan::ExclusiveSum(temp2.ptr, t2, totals.ptr, L2.ptr.ptr, mj, stream.value), "scan");
            const int cop_int = static_cast<int>(Cop);
            check(cudaMemcpyAsync(L2.ptr.ptr + mj, &cop_int, sizeof(int), cudaMemcpyHostToDevice, stream.value), "ptr end");
            check(cudaStreamSynchronize(stream.value), "ptr end");
            check(cudaEventRecord(e[3].value, stream.value), "event");
            b_level_offsets<<<blocks_for(mj, kThreads), kThreads, 0, stream.value>>>(H.ptr, L2.ptr.ptr, static_cast<int>(tiles), mj);
            check(cudaGetLastError(), "b_level_offsets");
            check(cudaEventRecord(e[4].value, stream.value), "event");
            sc.code = cc[j].ptr; sc.qcell = qcell.ptr; sc.qseg = qseg.ptr; sc.C_op = static_cast<long long>(Cop);
            sc.tile = o.tile; sc.levels = mj; sc.j = j; sc.H = H.ptr; sc.d = dd; sc.mult = mult.ptr;
            sc.seg = L2.seg.ptr;
            for (int jj = 0; jj < g.nb; ++jj) sc.other[jj] = L2.other[jj].ptr;
            sc.omult = L2.has_mult ? L2.mult.ptr : nullptr;
            sc.cell = L2.cell.ptr;
            const int warps = 4;
            const std::size_t smem = static_cast<std::size_t>(warps) * static_cast<std::size_t>(mj) * sizeof(int);
            if (smem > 48 * 1024)
                check(cudaFuncSetAttribute(b_scatter, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)), "scatter smem");
            b_scatter<<<blocks_for(tiles, warps), 32 * warps, smem, stream.value>>>(sc);
            check(cudaGetLastError(), "b_scatter");
            check(cudaEventRecord(e[5].value, stream.value), "event");
        } else {
            DeviceArray<int> kin, kout, vin, vout;
            kin.allocate(Cop, "radix keys"); kout.allocate(Cop, "radix keys");
            vin.allocate(Cop, "radix values"); vout.allocate(Cop, "radix values");
            b_keys<<<blocks_for(static_cast<long long>(Cop), kThreads), kThreads, 0, stream.value>>>(cc[j].ptr, qcell.ptr, static_cast<long long>(Cop), kin.ptr, vin.ptr);
            check(cudaGetLastError(), "b_keys");
            check(cudaEventRecord(e[2].value, stream.value), "event");
            int bits = 1;
            while (bits < 31 && (1LL << bits) < mj) ++bits;
            std::size_t t3 = 0;
            check(cub::DeviceRadixSort::SortPairs(nullptr, t3, kin.ptr, kout.ptr, vin.ptr, vout.ptr, static_cast<int>(Cop), 0, bits, stream.value), "radix size");
            DeviceArray<unsigned char> temp3;
            temp3.allocate(std::max<std::size_t>(1, t3), "radix temp");
            check(cub::DeviceRadixSort::SortPairs(temp3.ptr, t3, kin.ptr, kout.ptr, vin.ptr, vout.ptr, static_cast<int>(Cop), 0, bits, stream.value), "radix");
            check(cudaEventRecord(e[3].value, stream.value), "event");
            b_ptr_from_sorted<<<blocks_for(mj + 1, kThreads), kThreads, 0, stream.value>>>(kout.ptr, static_cast<long long>(Cop), mj, L2.ptr.ptr);
            check(cudaGetLastError(), "b_ptr");
            check(cudaEventRecord(e[4].value, stream.value), "event");
            BGather bg{};
            bg.vals = vout.ptr; bg.qcell = qcell.ptr; bg.qseg = qseg.ptr; bg.n = static_cast<long long>(Cop); bg.j = j; bg.d = dd; bg.mult = mult.ptr;
            bg.seg = L2.seg.ptr;
            for (int jj = 0; jj < g.nb; ++jj) bg.other[jj] = L2.other[jj].ptr;
            bg.omult = L2.has_mult ? L2.mult.ptr : nullptr;
            bg.cell = L2.cell.ptr;
            b_gather<<<blocks_for(static_cast<long long>(Cop), kThreads), kThreads, 0, stream.value>>>(bg);
            check(cudaGetLastError(), "b_gather");
            check(cudaEventRecord(e[5].value, stream.value), "event");
        }
        L2.host_ptr.resize(static_cast<std::size_t>(mj) + 1);
        check(cudaMemcpyAsync(L2.host_ptr.data(), L2.ptr.ptr, (static_cast<std::size_t>(mj) + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream.value), "ptr download");
        check(cudaEventRecord(e[6].value, stream.value), "event");
        check(cudaStreamSynchronize(stream.value), "b build");
        for (int k = 0; k < 6; ++k) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, e[k].value, e[k + 1].value) == cudaSuccess) st.b_steps_ms[k] += ms;
            else (void)cudaGetLastError();
        }
    }
    st.b_build_s = since(t_b);

    // Units of every T/B effect: a level of <= short_level entries is one unit; a longer one has
    // ceil(n/item) units, each with its logical (level, item) slot. Launch order by decreasing length.
    const auto t_units = Clock::now();
    std::vector<int> lvl_out_h, lvl_base_h, lvl_items_h;
    long long pb_next = 0;
    for (int j = 0; j < g.nb; ++j) {
        if (cls[j] == 'P') continue;
        Level& L2 = lv[j];
        const int mj = g.dims[j].levels;
        std::vector<int> ul, ub, ue, us;
        ul.reserve(static_cast<std::size_t>(mj));
        long long slots = 0;
        for (int lev = 0; lev < mj; ++lev) {
            const int a0 = L2.host_ptr[static_cast<std::size_t>(lev)], a1 = L2.host_ptr[static_cast<std::size_t>(lev) + 1];
            const long long n = static_cast<long long>(a1) - a0;
            if (n <= o.short_level) {
                ul.push_back(lev); ub.push_back(a0); ue.push_back(a1); us.push_back(-1);
            } else {
                const long long k = (n + o.item - 1) / o.item;
                lvl_out_h.push_back(g.dims[j].offset + lev);
                lvl_base_h.push_back(static_cast<int>(pb_next + slots));
                lvl_items_h.push_back(static_cast<int>(k));
                for (long long it = 0; it < k; ++it) {
                    const long long b0 = a0 + it * o.item;
                    ul.push_back(lev); ub.push_back(static_cast<int>(b0));
                    ue.push_back(static_cast<int>(std::min<long long>(a1, b0 + o.item)));
                    us.push_back(static_cast<int>(pb_next + slots + it));
                }
                slots += k;
                ++st.long_levels[j];
            }
        }
        L2.units = static_cast<long long>(ul.size());
        L2.pb_first = pb_next;
        L2.pb_slots = slots;
        pb_next += slots;
        st.units[j] = L2.units;
        st.pb_slots[j] = slots;
        std::vector<int> order;
        if (o.sort_units) {  // a stable counting sort by decreasing length (lengths <= max(short_level, item))
            const int maxlen = std::max(o.short_level, o.item);
            std::vector<long long> start(static_cast<std::size_t>(maxlen) + 2, 0);
            for (std::size_t u = 0; u < ul.size(); ++u) ++start[static_cast<std::size_t>(maxlen - (ue[u] - ub[u])) + 1];
            for (int k = 0; k <= maxlen; ++k) start[static_cast<std::size_t>(k) + 1] += start[static_cast<std::size_t>(k)];
            order.resize(ul.size());
            for (std::size_t u = 0; u < ul.size(); ++u)
                order[static_cast<std::size_t>(start[static_cast<std::size_t>(maxlen - (ue[u] - ub[u]))]++)] = static_cast<int>(u);
        }
        std::size_t dummy = 0;
        upload(L2.u_level, ul.data(), ul.size(), stream.value, "units", dummy);
        upload(L2.u_begin, ub.data(), ub.size(), stream.value, "units", dummy);
        upload(L2.u_end, ue.data(), ue.size(), stream.value, "units", dummy);
        upload(L2.u_slot, us.data(), us.size(), stream.value, "units", dummy);
        if (o.sort_units) upload(L2.order, order.data(), order.size(), stream.value, "units", dummy);
        L2.seg_p = L2.seg.ptr;
        for (int jj = 0; jj < g.nb; ++jj) L2.other_p[jj] = jj == j ? nullptr : L2.other[jj].ptr;
        L2.mult_p = L2.has_mult ? L2.mult.ptr : nullptr;
        check(cudaStreamSynchronize(stream.value), "units");
    }
    long_levels_total = static_cast<int>(lvl_out_h.size());
    PB.allocate(static_cast<std::size_t>(pb_next) * W, "PB");
    std::size_t dummy = 0;
    upload(lvl_out, lvl_out_h.data(), lvl_out_h.size(), stream.value, "long levels", dummy);
    upload(lvl_base, lvl_base_h.data(), lvl_base_h.size(), stream.value, "long levels", dummy);
    upload(lvl_items, lvl_items_h.data(), lvl_items_h.size(), stream.value, "long levels", dummy);
    std::vector<int> lam_out_h;
    for (int q = 0; q < npriv; ++q)
        for (int lev = 0; lev < g.dims[priv_j[q]].levels; ++lev) lam_out_h.push_back(g.dims[priv_j[q]].offset + lev);
    upload(lam_out, lam_out_h.data(), lam_out_h.size(), stream.value, "private outputs", dummy);
    upload(long_pos, long_pos_h.data(), long_pos_h.size(), stream.value, "long segments", dummy);
    upload(long_base, long_base_h.data(), long_base_h.size(), stream.value, "long segments", dummy);
    check(cudaStreamSynchronize(stream.value), "lists");
    st.units_s = since(t_units);

    // Page-locked staging (B7: refusal for lack of resources is Pinned).
    const auto t_pin = Clock::now();
    if (o.transfer == Transfer::Pinned) {
        pin_in.allocate(m * W);
        pin_out.allocate(m * W);
    } else if (o.transfer == Transfer::PinnedChunks) {
        const std::size_t n = std::max<std::size_t>(1, o.chunk_bytes / sizeof(double));
        chunk[0].allocate(n);
        chunk[1].allocate(n);
        for (Event& x : chunk_ev) check(cudaEventCreateWithFlags(&x.value, cudaEventDisableTiming), "event");
    }
    st.pinned_s = since(t_pin);

    // Compiled kernels (B8.3).
    cudaFuncAttributes a{};
    check(cudaFuncGetAttributes(&a, k1_segments<16>), "attributes");
    st.k1_regs = a.numRegs;
    st.k1_local = static_cast<int>(a.localSizeBytes);
    check(cudaFuncGetAttributes(&a, k2_levels), "attributes");
    st.k2_regs = a.numRegs;
    st.k2_local = static_cast<int>(a.localSizeBytes);
    const std::size_t smem = static_cast<std::size_t>(org.groups) * static_cast<std::size_t>(S) * static_cast<std::size_t>(org.ws) * sizeof(double);
    auto k1 = k1_kernel(o.k1_cache);
    check(cudaFuncGetAttributes(&a, k1), "attributes");
    st.k1_regs = a.numRegs;
    st.k1_local = static_cast<int>(a.localSizeBytes);
    if (smem > 48 * 1024)
        check(cudaFuncSetAttribute(k1, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)), "k1 smem");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&st.k1_blocks_per_sm, k1, kThreads, smem), "occupancy");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&st.k2_blocks_per_sm, k2_levels, kThreads, 0), "occupancy");
    st.k1_smem = smem;
    st.setup_total_s = since(t_all);
}

void Operator::State::run_kernels() {
    const int w = width;
    Dims d{};
    d.nb = g.nb;
    for (int j = 0; j < g.nb; ++j) { d.off[j] = g.dims[j].offset; d.cc[j] = cc[j].ptr; }
    K1Params k{};
    k.in = in.ptr; k.T = T.ptr; k.PA = PA.ptr; k.op_seg = op_seg.ptr; k.cell_ptr = cell_ptr.ptr; k.seg_inv = seg_inv.ptr;
    k.k1_seg = o.seg_order == 1 ? k1_seg.ptr : op_seg.ptr;
    k.mult = g.mult ? mult.ptr : nullptr; k.d = d; k.npriv = npriv; k.S = S;
    for (int q = 0; q < npriv; ++q) { k.priv_j[q] = priv_j[q]; k.priv_base[q] = priv_base[q]; }
    k.L_op = g.L_op; k.short_max = o.short_seg; k.seg_per_group = o.seg_per_group;
    k.ws = org.ws; k.gsz = org.gsz; k.gpw = org.gpw; k.G = org.groups; k.width = w;
    const long long blocks = (static_cast<long long>(g.L_op) + static_cast<long long>(org.groups) * o.seg_per_group - 1) /
                             (static_cast<long long>(org.groups) * o.seg_per_group);
    const std::size_t smem = static_cast<std::size_t>(org.groups) * static_cast<std::size_t>(S) * static_cast<std::size_t>(org.ws) * sizeof(double);
    auto k1 = k1_kernel(o.k1_cache);
    if (smem > 48 * 1024) {
        check(cudaFuncSetAttribute(k1, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)), "k1 smem");
        check(cudaFuncSetAttribute(k1_long, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)), "k1l smem");
    }
    if (st.k1_smem != smem || occupancy_width != w) {  // the configuration of this width, reported once per change
        st.k1_smem = smem;
        occupancy_width = w;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&st.k1_blocks_per_sm, k1, kThreads, smem), "occupancy");
    }
    if (blocks > 0) {
        k1<<<static_cast<unsigned>(blocks), kThreads, smem, stream.value>>>(k);
        check(cudaGetLastError(), "k1");
    }
    if (o.profile) check(cudaEventRecord(ev[2].value, stream.value), "event");
    if (!long_pos_h.empty()) {
        K1LParams q{};
        q.k = k; q.long_pos = long_pos.ptr; q.long_base = long_base.ptr; q.PL = PL.ptr; q.item = o.item;
        q.pa_first = static_cast<int>(blocks);
        k1_long<<<static_cast<unsigned>(long_pos_h.size()), kThreads, smem, stream.value>>>(q);
        check(cudaGetLastError(), "k1l");
    }
    if (o.profile) check(cudaEventRecord(ev[3].value, stream.value), "event");
    for (int j = 0; j < g.nb; ++j) {
        if (cls[j] == 'P') continue;
        Level& L2 = lv[j];
        K2Params p{};
        p.in = in.ptr; p.T = T.ptr; p.out = out.ptr; p.PB = PB.ptr;
        p.u_level = L2.u_level.ptr; p.u_begin = L2.u_begin.ptr; p.u_end = L2.u_end.ptr; p.u_slot = L2.u_slot.ptr;
        p.order = o.sort_units ? L2.order.ptr : nullptr; p.units = L2.units;
        p.seg = L2.seg_p; for (int jj = 0; jj < g.nb; ++jj) p.other[jj] = L2.other_p[jj]; p.mult = L2.mult_p;
        p.j = j; p.d = d; p.ws = org.ws; p.gsz = org.gsz; p.gpw = org.gpw; p.G = org.groups; p.width = w;
        if (L2.units > 0) {
            k2_levels<<<blocks_for(L2.units, org.groups), kThreads, 0, stream.value>>>(p);
            check(cudaGetLastError(), "k2");
        }
    }
    if (o.profile) check(cudaEventRecord(ev[4].value, stream.value), "event");
    if (S > 0) {
        K3PParams p{};
        p.PA = PA.ptr; p.out = out.ptr; p.slots = static_cast<long long>(blocks) + static_cast<long long>(long_pos_h.size());
        p.S = S; p.ws = org.ws; p.width = w; p.lam_out = lam_out.ptr;
        k3_private<<<static_cast<unsigned>(S * org.ws), kThreads, 0, stream.value>>>(p);
        check(cudaGetLastError(), "k3 private");
    }
    if (long_levels_total > 0) {
        K3LParams p{};
        p.PB = PB.ptr; p.out = out.ptr; p.lv_out = lvl_out.ptr; p.lv_base = lvl_base.ptr; p.lv_items = lvl_items.ptr;
        p.ws = org.ws; p.width = w;
        k3_long<<<static_cast<unsigned>(long_levels_total), kThreads, 0, stream.value>>>(p);
        check(cudaGetLastError(), "k3 long");
    }
    if (o.profile) check(cudaEventRecord(ev[5].value, stream.value), "event");
}

Operator::Operator(const GeometryView& g, const Options& o) : s_(std::make_unique<State>()) { s_->setup(g, o); }
Operator::~Operator() {
    if (s_ && s_->stream.value) (void)cudaStreamSynchronize(s_->stream.value);  // nothing in flight when memory is freed
    if (s_) unregister();
}

void Operator::begin_group(int width) { s_->configure(width); }

void Operator::register_host(const double* in, double* out) {
    State& s = *s_;
    if (s.o.transfer != Transfer::Registered) throw Error(ErrorKind::Fatal, "candidate operator: not the registered transfer");
    unregister();
    const std::size_t bytes = static_cast<std::size_t>(s.g.m) * s.o.lanes * sizeof(double);
    check(cudaHostRegister(const_cast<double*>(in), bytes, cudaHostRegisterDefault), "register operand", ErrorKind::Pinned);
    s.reg_in = in;
    const cudaError_t st = cudaHostRegister(out, bytes, cudaHostRegisterDefault);
    if (st != cudaSuccess) {
        (void)cudaHostUnregister(const_cast<double*>(in));
        s.reg_in = nullptr;
        check(st, "register result", ErrorKind::Pinned);
    }
    s.reg_out = out;
}

void Operator::unregister() {
    State& s = *s_;
    if (s.stream.value) (void)cudaStreamSynchronize(s.stream.value);
    if (s.reg_in) (void)cudaHostUnregister(const_cast<double*>(s.reg_in));
    if (s.reg_out) (void)cudaHostUnregister(s.reg_out);
    s.reg_in = nullptr;
    s.reg_out = nullptr;
}

namespace {
// Pack rows [r0, r1) of the caller's stride-W vector into a stride-ws buffer (lanes < ws).
void pack_rows(const double* src, double* dst, int W, int ws, long long r0, long long r1) {
    if (ws == W) {
        std::memcpy(dst, src + static_cast<std::size_t>(r0) * W, static_cast<std::size_t>(r1 - r0) * W * sizeof(double));
        return;
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (r1 - r0 > 65536)
#endif
    for (long long r = r0; r < r1; ++r)
        std::memcpy(dst + static_cast<std::size_t>(r - r0) * ws, src + static_cast<std::size_t>(r) * W, static_cast<std::size_t>(ws) * sizeof(double));
}
// Unpack: lanes < ws from the buffer, lanes [ws, W) written +0.0.
void unpack_rows(const double* src, double* dst, int W, int ws, long long r0, long long r1) {
    if (ws == W) {
        std::memcpy(dst + static_cast<std::size_t>(r0) * W, src, static_cast<std::size_t>(r1 - r0) * W * sizeof(double));
        return;
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (r1 - r0 > 65536)
#endif
    for (long long r = r0; r < r1; ++r) {
        double* row = dst + static_cast<std::size_t>(r) * W;
        std::memcpy(row, src + static_cast<std::size_t>(r - r0) * ws, static_cast<std::size_t>(ws) * sizeof(double));
        for (int w = ws; w < W; ++w) row[w] = 0.0;
    }
}
void parallel_copy(double* dst, const double* src, std::size_t n) {
#if defined(_OPENMP)
    const int parts = 16;
#pragma omp parallel for schedule(static) num_threads(16)
    for (int p = 0; p < parts; ++p) {
        const std::size_t a = n * p / parts, b = n * (p + 1) / parts;
        std::memcpy(dst + a, src + a, (b - a) * sizeof(double));
    }
#else
    std::memcpy(dst, src, n * sizeof(double));
#endif
}
}  // namespace

void Operator::apply(const double* in_host, double* out_host) {
    State& s = *s_;
    const auto t0 = Clock::now();
    const int W = s.o.lanes, ws = s.org.ws;
    const long long m = s.g.m;
    const std::size_t dev_bytes = static_cast<std::size_t>(m) * ws * sizeof(double);
    cudaStream_t st = s.stream.value;
    double pack_ms = 0.0, unpack_ms = 0.0;
    if (s.o.profile) check(cudaEventRecord(s.ev[0].value, st), "event");
    switch (s.o.transfer) {
        case Transfer::Pageable:
        case Transfer::Registered: {
            if (s.o.transfer == Transfer::Registered && (s.reg_in != in_host || s.reg_out != out_host))
                throw Error(ErrorKind::Fatal, "candidate operator: unregistered vectors");
            if (ws == W) {
                check(cudaMemcpyAsync(s.in.ptr, in_host, dev_bytes, cudaMemcpyHostToDevice, st), "operand upload");
            } else {
                if (s.o.transfer != Transfer::Registered) throw Error(ErrorKind::Fatal, "candidate operator: pageable transfers need the padded form");
                const std::size_t full = static_cast<std::size_t>(m) * W * sizeof(double);
                check(cudaMemcpyAsync(s.inW.ptr, in_host, full, cudaMemcpyHostToDevice, st), "operand upload");
                k_pack<<<blocks_for(m * ws, kThreads), kThreads, 0, st>>>(s.inW.ptr, s.in.ptr, m, W, ws);
                check(cudaGetLastError(), "pack");
            }
            break;
        }
        case Transfer::Pinned: {
            const auto tp = Clock::now();
            if (ws == W) parallel_copy(s.pin_in.ptr, in_host, static_cast<std::size_t>(m) * W);
            else pack_rows(in_host, s.pin_in.ptr, W, ws, 0, m);
            pack_ms = 1e3 * since(tp);
            check(cudaMemcpyAsync(s.in.ptr, s.pin_in.ptr, dev_bytes, cudaMemcpyHostToDevice, st), "operand upload");
            break;
        }
        case Transfer::PinnedChunks: {
            const long long rows = std::max<long long>(1, static_cast<long long>(s.chunk[0].count) / ws);
            int k = 0;
            for (long long r0 = 0; r0 < m; r0 += rows, ++k) {
                const long long r1 = std::min(m, r0 + rows);
                const int b = k & 1;
                if (k >= 2) check(cudaEventSynchronize(s.chunk_ev[b].value), "chunk reuse");  // its last DMA done
                const auto tp = Clock::now();
                pack_rows(in_host, s.chunk[b].ptr, W, ws, r0, r1);
                pack_ms += 1e3 * since(tp);
                check(cudaMemcpyAsync(s.in.ptr + static_cast<std::size_t>(r0) * ws, s.chunk[b].ptr,
                                      static_cast<std::size_t>(r1 - r0) * ws * sizeof(double), cudaMemcpyHostToDevice, st), "chunk upload");
                check(cudaEventRecord(s.chunk_ev[b].value, st), "event");
            }
            break;
        }
    }
    if (s.o.profile) check(cudaEventRecord(s.ev[1].value, st), "event");
    s.run_kernels();
    switch (s.o.transfer) {
        case Transfer::Pageable:
        case Transfer::Registered:
            if (ws == W) {
                check(cudaMemcpyAsync(out_host, s.out.ptr, dev_bytes, cudaMemcpyDeviceToHost, st), "result download");
            } else {
                k_unpack<<<blocks_for(m * W, kThreads), kThreads, 0, st>>>(s.out.ptr, s.outW.ptr, m, W, ws);
                check(cudaGetLastError(), "unpack");
                check(cudaMemcpyAsync(out_host, s.outW.ptr, static_cast<std::size_t>(m) * W * sizeof(double), cudaMemcpyDeviceToHost, st), "result download");
            }
            if (s.o.profile) check(cudaEventRecord(s.ev[6].value, st), "event");
            check(cudaStreamSynchronize(st), "operator completion");
            break;
        case Transfer::Pinned: {
            check(cudaMemcpyAsync(s.pin_out.ptr, s.out.ptr, dev_bytes, cudaMemcpyDeviceToHost, st), "result download");
            if (s.o.profile) check(cudaEventRecord(s.ev[6].value, st), "event");
            check(cudaStreamSynchronize(st), "operator completion");
            const auto tu = Clock::now();
            if (ws == W) parallel_copy(out_host, s.pin_out.ptr, static_cast<std::size_t>(m) * W);
            else unpack_rows(s.pin_out.ptr, out_host, W, ws, 0, m);
            unpack_ms = 1e3 * since(tu);
            break;
        }
        case Transfer::PinnedChunks: {
            const long long rows = std::max<long long>(1, static_cast<long long>(s.chunk[0].count) / ws);
            const long long chunks = (m + rows - 1) / rows;
            auto issue = [&](long long k) {
                const long long r0 = k * rows, r1 = std::min(m, r0 + rows);
                check(cudaMemcpyAsync(s.chunk[k & 1].ptr, s.out.ptr + static_cast<std::size_t>(r0) * ws,
                                      static_cast<std::size_t>(r1 - r0) * ws * sizeof(double), cudaMemcpyDeviceToHost, st), "chunk download");
                check(cudaEventRecord(s.chunk_ev[k & 1].value, st), "event");
            };
            for (long long k = 0; k < std::min<long long>(2, chunks); ++k) issue(k);
            for (long long k = 0; k < chunks; ++k) {
                check(cudaEventSynchronize(s.chunk_ev[k & 1].value), "chunk download");
                const long long r0 = k * rows, r1 = std::min(m, r0 + rows);
                const auto tu = Clock::now();
                unpack_rows(s.chunk[k & 1].ptr, out_host, W, ws, r0, r1);
                unpack_ms += 1e3 * since(tu);
                if (k + 2 < chunks) issue(k + 2);
            }
            if (s.o.profile) check(cudaEventRecord(s.ev[6].value, st), "event");
            check(cudaStreamSynchronize(st), "operator completion");
            break;
        }
    }
    if (s.o.profile) {
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, s.ev[0].value, s.ev[1].value), "timing"); s.st.h2d_ms += ms;
        check(cudaEventElapsedTime(&ms, s.ev[1].value, s.ev[2].value), "timing"); s.st.k1_ms += ms;
        check(cudaEventElapsedTime(&ms, s.ev[2].value, s.ev[3].value), "timing"); s.st.k1l_ms += ms;
        check(cudaEventElapsedTime(&ms, s.ev[3].value, s.ev[4].value), "timing"); s.st.k2_ms[0] += ms;
        check(cudaEventElapsedTime(&ms, s.ev[4].value, s.ev[5].value), "timing"); s.st.k3_ms += ms;
        check(cudaEventElapsedTime(&ms, s.ev[5].value, s.ev[6].value), "timing"); s.st.d2h_ms += ms;
    }
    s.st.pack_ms += pack_ms;
    s.st.unpack_ms += unpack_ms;
    ++s.st.calls;
    s.st.apply_ms += 1e3 * since(t0);
}

const Stats& Operator::stats() const noexcept { return s_->st; }
const Admission& Operator::admission() const noexcept { return s_->adm; }
char Operator::dim_class(int j) const { return s_->cls[j]; }

Operator::LevelOrderCopy Operator::download_level_order(int j) const {
    State& s = *s_;
    LevelOrderCopy c;
    if (s.cls[j] != 'B') return c;
    auto& L2 = s.lv[j];
    auto get = [&](const DeviceArray<int>& a, std::vector<int>& v) {
        v.resize(a.count);
        if (a.count) check(cudaMemcpy(v.data(), a.ptr, a.count * sizeof(int), cudaMemcpyDeviceToHost), "download");
    };
    get(L2.ptr, c.ptr);
    get(L2.seg, c.seg);
    get(L2.cell, c.cell);
    if (L2.has_mult) get(L2.mult, c.mult);
    c.other.resize(static_cast<std::size_t>(s.g.nb));
    for (int jj = 0; jj < s.g.nb; ++jj)
        if (jj != j) get(L2.other[jj], c.other[static_cast<std::size_t>(jj)]);
    return c;
}

// ---------------------------------------------------------------------------
TransferResult measure_transfers(int m, int lanes, int width, int reps, std::size_t chunk_bytes) {
    TransferResult r;
    const std::size_t n_full = static_cast<std::size_t>(m) * lanes, n_pack = static_cast<std::size_t>(m) * width;
    std::vector<double> host_in(n_full), host_out(n_full);
    for (std::size_t i = 0; i < n_full; ++i) host_in[i] = (i % lanes) < static_cast<std::size_t>(width) ? 1.0 + 1e-9 * static_cast<double>(i % 977) : 0.0;
    Stream stream;
    check(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking), "stream");
    DeviceArray<double> dev;
    dev.allocate(n_full, "device vector");
    Event e0, e1;
    check(cudaEventCreate(&e0.value), "event");
    check(cudaEventCreate(&e1.value), "event");
    auto timed = [&](auto&& body) {  // device time of an async body on the stream (ms)
        check(cudaEventRecord(e0.value, stream.value), "event");
        body();
        check(cudaEventRecord(e1.value, stream.value), "event");
        check(cudaEventSynchronize(e1.value), "event");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, e0.value, e1.value), "timing");
        return static_cast<double>(ms);
    };
    auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
    std::vector<double> a, b;
    for (int i = 0; i < reps; ++i) {
        a.push_back(timed([&] { check(cudaMemcpyAsync(dev.ptr, host_in.data(), n_full * 8, cudaMemcpyHostToDevice, stream.value), "h2d"); }));
        b.push_back(timed([&] { check(cudaMemcpyAsync(host_out.data(), dev.ptr, n_full * 8, cudaMemcpyDeviceToHost, stream.value), "d2h"); }));
    }
    r.pageable_h2d_ms = median(a);
    r.pageable_d2h_ms = median(b);
    auto t = Clock::now();
    PinnedArray pin_in, pin_out;
    pin_in.allocate(n_full);
    pin_out.allocate(n_full);
    r.pinned_alloc_s = since(t);
    a.clear(); b.clear();
    for (int i = 0; i < reps; ++i) {
        a.push_back(timed([&] { check(cudaMemcpyAsync(dev.ptr, pin_in.ptr, n_pack * 8, cudaMemcpyHostToDevice, stream.value), "h2d"); }));
        b.push_back(timed([&] { check(cudaMemcpyAsync(pin_out.ptr, dev.ptr, n_pack * 8, cudaMemcpyDeviceToHost, stream.value), "d2h"); }));
    }
    r.pinned_h2d_ms = median(a);
    r.pinned_d2h_ms = median(b);
    a.clear(); b.clear();
    for (int i = 0; i < reps; ++i) {
        a.push_back(timed([&] { check(cudaMemcpyAsync(dev.ptr, pin_in.ptr, n_full * 8, cudaMemcpyHostToDevice, stream.value), "h2d"); }));
        b.push_back(timed([&] { check(cudaMemcpyAsync(pin_out.ptr, dev.ptr, n_full * 8, cudaMemcpyDeviceToHost, stream.value), "d2h"); }));
    }
    r.pinned_full_h2d_ms = median(a);
    r.pinned_full_d2h_ms = median(b);
    std::vector<double> cyc, pk, upk;
    for (int i = 0; i < reps; ++i) {  // pack, DMA, DMA back, unpack: wall time
        const auto c0 = Clock::now();
        if (width == lanes) parallel_copy(pin_in.ptr, host_in.data(), n_full); else pack_rows(host_in.data(), pin_in.ptr, lanes, width, 0, m);
        const double p_ms = 1e3 * since(c0);
        check(cudaMemcpyAsync(dev.ptr, pin_in.ptr, n_pack * 8, cudaMemcpyHostToDevice, stream.value), "h2d");
        check(cudaMemcpyAsync(pin_out.ptr, dev.ptr, n_pack * 8, cudaMemcpyDeviceToHost, stream.value), "d2h");
        check(cudaStreamSynchronize(stream.value), "sync");
        const auto u0 = Clock::now();
        if (width == lanes) parallel_copy(host_out.data(), pin_out.ptr, n_full); else unpack_rows(pin_out.ptr, host_out.data(), lanes, width, 0, m);
        upk.push_back(1e3 * since(u0));
        pk.push_back(p_ms);
        cyc.push_back(1e3 * since(c0));
    }
    r.cycle_ms = median(cyc);
    r.cycle_pack_ms = median(pk);
    r.cycle_unpack_ms = median(upk);
    t = Clock::now();
    pin_in.release();
    pin_out.release();
    r.pinned_free_s = since(t);
    // Double-buffered chunks.
    PinnedArray ch[2];
    const std::size_t cn = std::max<std::size_t>(1, chunk_bytes / sizeof(double));
    ch[0].allocate(cn);
    ch[1].allocate(cn);
    Event ce[2];
    for (Event& x : ce) check(cudaEventCreateWithFlags(&x.value, cudaEventDisableTiming), "event");
    const long long rows = std::max<long long>(1, static_cast<long long>(cn) / width);
    const long long chunks = (m + rows - 1) / rows;
    cyc.clear();
    for (int i = 0; i < reps; ++i) {
        const auto c0 = Clock::now();
        for (long long k = 0; k < chunks; ++k) {
            const long long r0 = k * rows, r1 = std::min<long long>(m, r0 + rows);
            if (k >= 2) check(cudaEventSynchronize(ce[k & 1].value), "reuse");
            pack_rows(host_in.data(), ch[k & 1].ptr, lanes, width, r0, r1);
            check(cudaMemcpyAsync(dev.ptr + static_cast<std::size_t>(r0) * width, ch[k & 1].ptr, static_cast<std::size_t>(r1 - r0) * width * 8,
                                  cudaMemcpyHostToDevice, stream.value), "h2d");
            check(cudaEventRecord(ce[k & 1].value, stream.value), "event");
        }
        auto issue = [&](long long k) {
            const long long r0 = k * rows, r1 = std::min<long long>(m, r0 + rows);
            check(cudaMemcpyAsync(ch[k & 1].ptr, dev.ptr + static_cast<std::size_t>(r0) * width, static_cast<std::size_t>(r1 - r0) * width * 8,
                                  cudaMemcpyDeviceToHost, stream.value), "d2h");
            check(cudaEventRecord(ce[k & 1].value, stream.value), "event");
        };
        for (long long k = 0; k < std::min<long long>(2, chunks); ++k) issue(k);
        for (long long k = 0; k < chunks; ++k) {
            check(cudaEventSynchronize(ce[k & 1].value), "d2h");
            const long long r0 = k * rows, r1 = std::min<long long>(m, r0 + rows);
            unpack_rows(ch[k & 1].ptr, host_out.data(), lanes, width, r0, r1);
            if (k + 2 < chunks) issue(k + 2);
        }
        check(cudaStreamSynchronize(stream.value), "sync");
        cyc.push_back(1e3 * since(c0));
    }
    r.chunked_cycle_ms = median(cyc);
    // Registration of the caller's vectors (padded rows only: the host keeps stride W).
    t = Clock::now();
    check(cudaHostRegister(host_in.data(), n_full * 8, cudaHostRegisterDefault), "register", ErrorKind::Pinned);
    check(cudaHostRegister(host_out.data(), n_full * 8, cudaHostRegisterDefault), "register", ErrorKind::Pinned);
    r.register_s = since(t);
    a.clear(); b.clear(); cyc.clear();
    for (int i = 0; i < reps; ++i) {
        a.push_back(timed([&] { check(cudaMemcpyAsync(dev.ptr, host_in.data(), n_full * 8, cudaMemcpyHostToDevice, stream.value), "h2d"); }));
        b.push_back(timed([&] { check(cudaMemcpyAsync(host_out.data(), dev.ptr, n_full * 8, cudaMemcpyDeviceToHost, stream.value), "d2h"); }));
        const auto c0 = Clock::now();
        check(cudaMemcpyAsync(dev.ptr, host_in.data(), n_full * 8, cudaMemcpyHostToDevice, stream.value), "h2d");
        check(cudaMemcpyAsync(host_out.data(), dev.ptr, n_full * 8, cudaMemcpyDeviceToHost, stream.value), "d2h");
        check(cudaStreamSynchronize(stream.value), "sync");
        cyc.push_back(1e3 * since(c0));
    }
    r.registered_h2d_ms = median(a);
    r.registered_d2h_ms = median(b);
    r.registered_cycle_ms = median(cyc);
    // Registered rows of stride lanes, live lanes only on the link (2D copies), the padding of the result
    // written +0.0 on the host: the whole path of registration with packing.
    if (width < lanes) {
        a.clear(); b.clear(); cyc.clear();
        std::vector<double> z;
        for (int i = 0; i < reps; ++i) {
            a.push_back(timed([&] { check(cudaMemcpy2DAsync(dev.ptr, static_cast<std::size_t>(width) * 8, host_in.data(), static_cast<std::size_t>(lanes) * 8,
                                                            static_cast<std::size_t>(width) * 8, static_cast<std::size_t>(m), cudaMemcpyHostToDevice, stream.value), "h2d 2d"); }));
            b.push_back(timed([&] { check(cudaMemcpy2DAsync(host_out.data(), static_cast<std::size_t>(lanes) * 8, dev.ptr, static_cast<std::size_t>(width) * 8,
                                                            static_cast<std::size_t>(width) * 8, static_cast<std::size_t>(m), cudaMemcpyDeviceToHost, stream.value), "d2h 2d"); }));
            const auto c0 = Clock::now();
            check(cudaMemcpy2DAsync(dev.ptr, static_cast<std::size_t>(width) * 8, host_in.data(), static_cast<std::size_t>(lanes) * 8,
                                    static_cast<std::size_t>(width) * 8, static_cast<std::size_t>(m), cudaMemcpyHostToDevice, stream.value), "h2d 2d");
            check(cudaMemcpy2DAsync(host_out.data(), static_cast<std::size_t>(lanes) * 8, dev.ptr, static_cast<std::size_t>(width) * 8,
                                    static_cast<std::size_t>(width) * 8, static_cast<std::size_t>(m), cudaMemcpyDeviceToHost, stream.value), "d2h 2d");
            check(cudaStreamSynchronize(stream.value), "sync");
            const auto z0 = Clock::now();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(16)
#endif
            for (long long rr = 0; rr < m; ++rr)
                for (int w = width; w < lanes; ++w) host_out[static_cast<std::size_t>(rr) * lanes + w] = 0.0;
            z.push_back(1e3 * since(z0));
            cyc.push_back(1e3 * since(c0));
        }
        r.registered2d_h2d_ms = median(a);
        r.registered2d_d2h_ms = median(b);
        r.registered2d_cycle_ms = median(cyc);
        r.registered2d_zero_ms = median(z);
    }
    t = Clock::now();
    (void)cudaHostUnregister(host_in.data());
    (void)cudaHostUnregister(host_out.data());
    r.unregister_s = since(t);
    return r;
}

}  // namespace xhdfe_bench::cand
