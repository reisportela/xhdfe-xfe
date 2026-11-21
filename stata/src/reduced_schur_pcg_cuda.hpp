#ifndef HDFE_REDUCED_SCHUR_PCG_CUDA_HPP
#define HDFE_REDUCED_SCHUR_PCG_CUDA_HPP
// The CUDA operator of the reduced tier (package 2, level A, 17 November 2026):
// out = S in for the implicit Schur complement of the CPU geometry, one group of
// columns at a time, at the group's real width. The kernels and the device set-up
// are in reduced_schur_pcg_cuda.cu; the host-side plan below (classes, long
// segments, units, admission) is plain C++, so that it can be tested without a
// device (tests/reduced_cuda_memory_index_contract.cpp).
//
// The product. Lanes never mix; no atomic operation takes part in a value; every
// output has one writer; the padding lanes of the caller's rows come back +0.0.
//   K1   segments of at most short_seg cells: one group of `width` threads per
//        segment (groups never straddle a warp), the mean summed in cell order,
//        the private effects' deviations into per-group shared accumulators,
//        combined in group order per block;
//   K1L  longer segments: one block each, items of `item` cells summed in cell
//        order, a pairwise tree on item index in global partials (PL), then the
//        private deviations as in K1;
//   K2   the levels of every transposed (T) or device-built (B) effect: one group
//        per unit (a level of at most short_level entries, or one item of a
//        longer level), entries in level order;
//   K3   fixed trees: the private effects over the block partials (PA), the long
//        levels over their item partials (PB).
// Classes of the remaining effects: P (private accumulators) for the smallest
// effects while their cumulative levels stay within private_max and their
// accumulators fit the device's shared memory; otherwise T when the CPU geometry
// made a transpose (its arrays are uploaded as they are), else B (the same arrays
// built on the device by a stable counting sort over tiles, or by a stable radix
// sort when the level count exceeds kTilesMaxLevels).
//
// Transfers. The caller's operand and result rows (stride `lanes`) are page-locked
// by ONE registration of their covering range for the whole solve
// (register_host / unregister_host); whole rows go over the link, and the device
// packs the live lanes into rows of `width` and unpacks the result, writing +0.0
// into the padding lanes.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hdfe/ieee_bits.hpp"

#if defined(__CUDACC__)
#define HDFE_REDUCED_CUDA_HD __host__ __device__
#else
#define HDFE_REDUCED_CUDA_HD
#endif

namespace hdfe::detail::reduced_cuda {

// Memory: an allocation or the admission refused; Unavailable: no device or
// driver; Fatal: anything else (it propagates without estimates); Pinned:
// page-locking of host memory refused for lack of resources (a decline with the
// reason `pinned`). Pinned is the last value so that the others keep theirs.
enum class ErrorKind { Memory, Unavailable, Fatal, Pinned };

class Error : public std::runtime_error {
public:
    Error(ErrorKind kind, const std::string& message) : std::runtime_error(message), kind_(kind) {}
    ErrorKind kind() const noexcept { return kind_; }
private:
    ErrorKind kind_;
};

constexpr int kThreads = 256;          // threads per block of every kernel but the scatter
constexpr int kMaxDims = 4;            // remaining effects (the tier takes 2 to 5 effects)
constexpr int kSegPerGroup = 16;       // K1: segments per group per block
constexpr int kK1Cache = 16;           // K1: a segment of at most this many cells keeps u in registers
constexpr int kTile = 4096;            // B: positions per tile of the counting sort
constexpr int kTilesMaxLevels = 3072;  // B: the counting sort's cursors (4 warps x levels x 4 bytes) stay within 48 KiB
constexpr std::size_t kAlign = 256;    // sub-buffers of the build scratch

// A remaining effect as the CPU geometry holds it.
struct DimView {
    int levels = 0;
    int offset = 0;                    // in the reduced vector
    bool transposed = false;           // build_geometry made tr[j]
    const int* tr_ptr = nullptr;       // levels + 1
    const int* tr_seg = nullptr;       // entry -> ORIGINAL segment id (indexes T)
    const int* tr_other[kMaxDims] = {nullptr, nullptr, nullptr, nullptr};  // [jj], jj != j
    const int* tr_mult = nullptr;      // read only when has_mult
    bool has_mult = false;
};

// The CPU geometry as the operator reads it (borrowed pointers; nothing copied).
struct GeometryView {
    int nb = 0, m = 0, L = 0, L_op = 0;
    long long C = 0, C_op = 0;
    const int* cell_ptr = nullptr;     // L + 1
    const int* op_seg = nullptr;       // L_op
    const double* seg_inv = nullptr;   // L: 1 / rows
    const int* cc[kMaxDims] = {nullptr, nullptr, nullptr, nullptr};  // segment-order codes (C each)
    const int* mult = nullptr;         // merged geometry only (C)
    DimView dims[kMaxDims];
};

enum class BMethod { Auto, Tiles, Radix };

// Test injections (XHDFE_TEST_REDUCED_PCG_CUDA_FAIL): a failure at one point,
// produced through the same error mapping as the real one.
//   Allocation  the first per-product buffer, after the resident uploads were queued (Memory);
//   Build       a level array of the first B effect, after its first build kernels (Memory);
//   Staging     the second page-locked upload chunk (Pinned);
//   Register    the registration of the caller's rows (Pinned).
enum class TestFailure { None, Allocation, Build, Staging, Register };

struct Options {
    int lanes = 8;                     // W, the stride of the caller's rows: 1, 2, 4 or 8
    int private_max = 48;              // cumulative levels of the private effects
    int short_seg = 256;               // K1 takes segments of at most this many cells
    int short_level = 256;             // a level of at most this many entries is one K2 unit
    int item = 256;                    // items of long segments and of long levels
    BMethod b_method = BMethod::Auto;
    std::size_t budget = 0;            // device admission budget in bytes; 0: the free memory only
    std::size_t host_budget = 0;       // host bytes the set-up may hold (0: unchecked)
    std::size_t host_reserved = 0;     // host bytes already held when the set-up runs
    std::size_t chunk_bytes = std::size_t{8} << 20;  // each of the two page-locked upload chunks
    // The slices of m x lanes doubles of the caller's workspace that the caller
    // registers for the iteration (admission only: locked, not allocated): 2 for
    // p and q (package 2), 4 for x, r, p and q (package 3-lite's replacement product).
    int registered_slices = 2;
    bool profile = false;              // events per kernel, clocks of the set-up (the trace)
    // The host team for the copies into the upload chunks; empty: one thread.
    // Used only inside the constructor.
    std::function<void(int, const std::function<void(int)>&)> run;
    TestFailure test_failure = TestFailure::None;
    // The source cell of every entry of a B effect, for the identity test; honoured
    // only in builds with XHDFE_REDUCED_CUDA_TEST_HOOKS (never in a library or plugin).
    bool test_source_cells = false;
};

// Every buffer of the operator, by phase, counted before the first allocation.
// Device phases, in the order of the set-up:
//   resident  the geometry's arrays, K1's segment order, the transposes' arrays,
//             the private outputs, the long-segment lists;
//   product   in, out, the registered stride pair (W > 1), T, PA, PL;
//   build     the level arrays of every B effect;
//   scratch   the B build (freed before the units exist);
//   units     the K2 units, the long-level lists and PB (exact for T; for B the
//             bounds of the plan, the allocation after the build is exact).
// device_setup_peak = resident + product + build + scratch;
// device_iteration_peak = resident + product + build + units.
// Host (the set-up only; nothing stays after it): the two upload chunks
// (page-locked), K1's segment order with its counts, the long-segment lists, the
// downloaded ptr of every B effect, the unit lists of one effect at a time with
// their sort counts, the long-level lists and the private outputs. Page-locked
// during the iteration: the caller's registered rows (Options::registered_slices;
// inside its workspace: no new bytes).
struct Admission {
    enum Phase : unsigned char { Resident, Product, Build, Scratch, Units, Host };
    struct Entry {
        const char* name = "";
        int effect = -1;
        Phase phase = Resident;
        std::size_t bytes = 0;
    };
    static constexpr int kMaxEntries = 64;
    Entry entries[kMaxEntries];
    int count = 0;
    std::size_t device[5] = {0, 0, 0, 0, 0};   // by Phase (Resident .. Units)
    std::size_t device_setup_peak = 0, device_iteration_peak = 0, device_peak = 0;
    std::size_t host_setup = 0;                // admitted (an upper bound of the real peak)
    std::size_t host_setup_real = 0;           // the real peak, from vector capacities, after the set-up
    std::size_t page_locked_setup = 0, page_locked_iteration = 0;
    std::size_t budget = 0, free_at_entry = 0;
    std::size_t cub_scan_bytes = 0, cub_radix_bytes = 0;
    // What the operator asked of the device (cudaMalloc), and, in trace mode only,
    // the free memory at the set-up's peak and after it.
    std::size_t allocated_peak = 0, allocated_iteration = 0;
    std::size_t free_at_peak = 0, free_after_setup = 0;
    bool admitted = false;
};

inline std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw Error(ErrorKind::Memory, "CUDA reduced operator: byte count overflow");
    return a + b;
}
inline std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw Error(ErrorKind::Memory, "CUDA reduced operator: byte count overflow");
    return a * b;
}
inline std::size_t align_up(std::size_t bytes) {
    return checked_add(bytes, kAlign - 1) / kAlign * kAlign;
}

// The copy ranges of one upload chunk on the host team: 64 KiB each, fixed by the
// data whatever the team (128 for a full chunk of 8 MiB).
inline int staging_parts(std::size_t len) {
    return static_cast<int>(std::max<std::size_t>(1, len >> 16));
}

// K1's segment order (the K1 iteration of 17 November): the operator segments grouped
// by length, a stable sort by decreasing cells, lengths above short_seg counting as
// one (K1 leaves those to K1L). The groups of a warp advance together, so a warp
// takes as long as its longest segment; with equal lengths side by side the groups
// of a warp carry the same work. A fixed permutation of the geometry: the bits stay a
// function of the geometry alone (not those of the natural order: the private sums
// of a block gather other segments). Built over at most kOrderParts contiguous parts
// of the operator segments (fixed, whatever the team): counts per part, prefixes in
// (key, part) order, a stable scatter per part, the result of a serial stable sort.
// `run(parts, body)` runs body(part) once for every part, in any order or at once.
constexpr int kOrderParts = 64;
inline int k1_order_key(const GeometryView& g, int s, int top) {
    const int l = g.op_seg[s];
    const int n = g.cell_ptr[l + 1] - g.cell_ptr[l];
    return top - std::min(n, top);  // 0 for the longest
}
inline int k1_order_parts(int L_op) { return std::max(1, std::min(kOrderParts, L_op)); }
// The host bytes of the order's counts (the order itself: L_op ints).
inline std::size_t k1_order_count_bytes(int short_seg, int L_op) {
    return checked_mul(checked_mul(static_cast<std::size_t>(k1_order_parts(L_op)), static_cast<std::size_t>(short_seg) + 2U),
                       sizeof(long long));
}
template <class Run>
inline void k1_segment_order(const GeometryView& g, int short_seg, int* out, Run&& run) {
    const int top = short_seg + 1, keys = top + 1;
    const int P = k1_order_parts(g.L_op);
    const long long L_op = g.L_op;
    std::vector<long long> at(static_cast<std::size_t>(P) * static_cast<std::size_t>(keys), 0);
    run(P, [&](int p) {
        long long* c = at.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(keys);
        const int s0 = static_cast<int>(L_op * p / P), s1 = static_cast<int>(L_op * (p + 1) / P);
        for (int s = s0; s < s1; ++s) ++c[k1_order_key(g, s, top)];
    });
    long long next = 0;
    for (int k = 0; k < keys; ++k)
        for (int p = 0; p < P; ++p) {
            long long& c = at[static_cast<std::size_t>(p) * static_cast<std::size_t>(keys) + static_cast<std::size_t>(k)];
            const long long n = c;
            c = next;
            next += n;
        }
    run(P, [&](int p) {
        long long* c = at.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(keys);
        const int s0 = static_cast<int>(L_op * p / P), s1 = static_cast<int>(L_op * (p + 1) / P);
        for (int s = s0; s < s1; ++s) out[c[k1_order_key(g, s, top)]++] = g.op_seg[s];
    });
}

inline void add_entry(Admission& a, const char* name, int effect, Admission::Phase phase, std::size_t bytes) {
    if (a.count >= Admission::kMaxEntries) throw Error(ErrorKind::Fatal, "CUDA reduced operator: admission entries");
    a.entries[a.count++] = Admission::Entry{name, effect, phase, bytes};
    if (phase == Admission::Host) a.host_setup = checked_add(a.host_setup, bytes);
    else a.device[phase] = checked_add(a.device[phase], bytes);
}

// Thread organisation at a real width: groups of `width` threads inside a warp
// (never across one), floor(32 / width) of them per warp; ws, the device row
// stride, is the width.
struct Org {
    int ws = 1, gsz = 1, gpw = 32, groups = 256;
};
HDFE_REDUCED_CUDA_HD inline Org org_for(int width) {
    Org o;
    o.ws = width;
    o.gsz = width;
    o.gpw = 32 / width;
    o.groups = (kThreads / 32) * o.gpw;
    return o;
}
// Thread t of a block: its group, its lane in the group, and whether it has one.
HDFE_REDUCED_CUDA_HD inline void place_of(int t, int gsz, int gpw, int& group, int& lane, bool& in_group) {
    const int warp = t >> 5, lw = t & 31;
    const int giw = lw / gsz;
    in_group = giw < gpw;
    group = warp * gpw + giw;
    lane = lw - giw * gsz;
}
// K1: the i-th segment of group g of block b (64-bit; validated against L_op before use).
HDFE_REDUCED_CUDA_HD inline long long k1_segment(long long block, int group, int i, int groups) {
    return block * groups * kSegPerGroup + group + static_cast<long long>(groups) * i;
}
inline long long k1_blocks(long long L_op, int width) {
    const long long per_block = static_cast<long long>(org_for(width).groups) * kSegPerGroup;
    return (L_op + per_block - 1) / per_block;
}
// K1's shared accumulators at one width: groups x S x width doubles.
inline std::size_t k1_smem_bytes(int S, int width) {
    const Org o = org_for(width);
    return checked_mul(checked_mul(static_cast<std::size_t>(o.groups), static_cast<std::size_t>(S)),
                       checked_mul(static_cast<std::size_t>(o.ws), sizeof(double)));
}
inline std::size_t k1_smem_max(int S, int lanes) {
    std::size_t most = 0;
    for (int w = 1; w <= lanes; ++w) most = std::max(most, k1_smem_bytes(S, w));
    return most;
}

// Units of one level-ordered effect: a level of at most short_level entries is one
// unit (slot -1: K2 writes the output row); a longer one has ceil(n / item) units,
// each with its logical slot base + item index in PB (K3 sums them). Counted from
// ptr with every count checked; ptr must be non-decreasing from 0.
struct UnitCount {
    long long units = 0, slots = 0, long_levels = 0;
};
inline UnitCount count_units(const int* ptr, int levels, int short_level, int item) {
    if (ptr == nullptr || levels < 0 || ptr[0] != 0 || short_level < 1 || item < 1)
        throw Error(ErrorKind::Fatal, "CUDA reduced operator: invalid level pointers");
    UnitCount c;
    for (int lev = 0; lev < levels; ++lev) {
        const long long n = static_cast<long long>(ptr[lev + 1]) - ptr[lev];
        if (n < 0) throw Error(ErrorKind::Fatal, "CUDA reduced operator: invalid level pointers");
        if (n > short_level) {
            const long long k = (n + item - 1) / item;
            c.units += k;
            c.slots += k;
            ++c.long_levels;
        } else {
            ++c.units;
        }
    }
    if (c.units > std::numeric_limits<int>::max() || c.slots > std::numeric_limits<int>::max())
        throw Error(ErrorKind::Memory, "CUDA reduced operator: unit count");
    return c;
}
// The bounds of the same counts before ptr exists (a B effect): every level one unit
// plus ceil(n / item) - 1 <= n / item more for a long one; a long level has more than
// short_level entries.
inline UnitCount bound_units(long long levels, long long C_op, int short_level, int item) {
    UnitCount c;
    c.long_levels = std::min(levels, C_op / (static_cast<long long>(short_level) + 1));
    c.units = levels + C_op / item;
    c.slots = C_op / item + c.long_levels + 1;
    return c;
}

// The unit lists of one effect (level, begin, end, slot) and the launch order by
// decreasing length (a stable counting sort; slots stay logical); the long levels
// go into the lists of every effect for K3 (output index, first slot, items).
struct UnitLists {
    std::vector<int> level, begin, end, slot, order;
    std::size_t capacity_bytes() const {
        std::size_t b = 0;
        for (const auto* v : {&level, &begin, &end, &slot, &order})
            b = checked_add(b, checked_mul(v->capacity(), sizeof(int)));
        return b;
    }
};
struct LongLevels {
    std::vector<int> out, base, items;
    std::size_t next = 0;
    void size_for(std::size_t count) {
        out.assign(count, 0);
        base.assign(count, 0);
        items.assign(count, 0);
        next = 0;
    }
    std::size_t capacity_bytes() const {
        return checked_mul(checked_add(checked_add(out.capacity(), base.capacity()), items.capacity()), sizeof(int));
    }
};
inline std::size_t sort_count_bytes(int short_level, int item) {
    return checked_mul(static_cast<std::size_t>(std::max(short_level, item)) + 2U, sizeof(long long));
}
inline void build_units(const int* ptr, int levels, int offset, int short_level, int item, long long slot_base,
                        const UnitCount& count, UnitLists& out, LongLevels& longs) {
    const std::size_t U = static_cast<std::size_t>(count.units);
    if (longs.next + static_cast<std::size_t>(count.long_levels) > longs.out.size())
        throw Error(ErrorKind::Fatal, "CUDA reduced operator: long-level lists");
    out.level.assign(U, 0);
    out.begin.assign(U, 0);
    out.end.assign(U, 0);
    out.slot.assign(U, 0);
    out.order.assign(U, 0);
    std::size_t u = 0;
    long long slots = 0, ll = 0;
    for (int lev = 0; lev < levels; ++lev) {
        const int a0 = ptr[lev], a1 = ptr[lev + 1];
        const long long n = static_cast<long long>(a1) - a0;
        if (n <= short_level) {
            if (u >= U) throw Error(ErrorKind::Fatal, "CUDA reduced operator: unit count mismatch");
            out.level[u] = lev; out.begin[u] = a0; out.end[u] = a1; out.slot[u] = -1; ++u;
            continue;
        }
        const long long k = (n + item - 1) / item;
        if (u + static_cast<std::size_t>(k) > U || ll >= count.long_levels)
            throw Error(ErrorKind::Fatal, "CUDA reduced operator: unit count mismatch");
        longs.out[longs.next] = offset + lev;
        longs.base[longs.next] = static_cast<int>(slot_base + slots);
        longs.items[longs.next] = static_cast<int>(k);
        ++longs.next;
        ++ll;
        for (long long it = 0; it < k; ++it) {
            const long long b0 = a0 + it * item;
            out.level[u] = lev;
            out.begin[u] = static_cast<int>(b0);
            out.end[u] = static_cast<int>(std::min<long long>(a1, b0 + item));
            out.slot[u] = static_cast<int>(slot_base + slots + it);
            ++u;
        }
        slots += k;
    }
    if (u != U || ll != count.long_levels || slots != count.slots)
        throw Error(ErrorKind::Fatal, "CUDA reduced operator: unit count mismatch");
    // Launch order: decreasing length (lengths <= max(short_level, item)), stable.
    const int maxlen = std::max(short_level, item);
    std::vector<long long> start(static_cast<std::size_t>(maxlen) + 2U, 0);
    for (std::size_t i = 0; i < U; ++i) ++start[static_cast<std::size_t>(maxlen - (out.end[i] - out.begin[i])) + 1U];
    for (int k = 0; k <= maxlen; ++k) start[static_cast<std::size_t>(k) + 1U] += start[static_cast<std::size_t>(k)];
    for (std::size_t i = 0; i < U; ++i)
        out.order[static_cast<std::size_t>(start[static_cast<std::size_t>(maxlen - (out.end[i] - out.begin[i]))]++)] = static_cast<int>(i);
}

// The plan: classes, long segments, exact units of the T effects. One pass over
// the operator segments validates op_seg, cell_ptr and seg_inv on the way.
struct Plan {
    char cls[kMaxDims] = {0, 0, 0, 0};
    int npriv = 0, priv_j[kMaxDims] = {0, 0, 0, 0}, priv_base[kMaxDims] = {0, 0, 0, 0}, S = 0;
    long long n_long = 0, pl_items = 0;          // segments of more than short_seg cells, and their items
    long long k1_blocks_max = 0, pa_slots = 0;   // the most K1 blocks over every width; + n_long
    UnitCount units[kMaxDims];                   // exact for T, bounds for B
    long long entries[kMaxDims] = {0, 0, 0, 0};  // level-ordered entries of each T or B effect
    bool radix[kMaxDims] = {false, false, false, false};
    std::size_t k1_smem = 0;                     // over every width up to lanes
};

inline Plan plan_operator(const GeometryView& g, const Options& o, std::size_t smem_limit) {
    const auto bad = [] { throw Error(ErrorKind::Fatal, "CUDA reduced operator: invalid geometry view"); };
    if (g.nb < 1 || g.nb > kMaxDims || g.m < 0 || g.L < 0 || g.L_op < 0 || g.L_op > g.L || g.C < 0 ||
        g.C_op < 0 || g.C_op > g.C || g.cell_ptr == nullptr || (g.L_op > 0 && (g.op_seg == nullptr || g.seg_inv == nullptr)))
        bad();
    if (o.lanes != 1 && o.lanes != 2 && o.lanes != 4 && o.lanes != 8) bad();
    if (o.short_seg < 1 || o.short_level < 1 || o.item < 1 || o.private_max < 0) bad();
    if (g.C > std::numeric_limits<int>::max() || g.C_op > std::numeric_limits<int>::max())
        throw Error(ErrorKind::Memory, "CUDA reduced operator: cells beyond int");
    long long m = 0;
    for (int j = 0; j < g.nb; ++j) {
        const DimView& d = g.dims[j];
        // An array with no element may arrive as a null pointer (no cell: no code to read).
        if (d.levels < 1 || d.offset != m || (g.C > 0 && g.cc[j] == nullptr)) bad();
        m += d.levels;
    }
    if (m != g.m || g.cell_ptr[0] != 0 || g.cell_ptr[g.L] != g.C) bad();
    Plan p;
    // Private: the smallest effects first while the cumulative levels stay within
    // private_max and K1's accumulators fit the device's shared memory.
    int by_size[kMaxDims] = {0, 1, 2, 3};
    std::stable_sort(by_size, by_size + g.nb, [&](int a, int b) { return g.dims[a].levels < g.dims[b].levels; });
    int cumulative = 0;
    for (int q = 0; q < g.nb; ++q) {
        const int j = by_size[q];
        const int next = cumulative + g.dims[j].levels;
        if (next > o.private_max || k1_smem_max(next, o.lanes) > smem_limit) break;
        p.cls[j] = 'P';
        cumulative = next;
    }
    for (int j = 0; j < g.nb; ++j) {
        if (p.cls[j] == 'P') {
            p.priv_j[p.npriv] = j;
            p.priv_base[p.npriv] = p.S;
            p.S += g.dims[j].levels;
            ++p.npriv;
        } else {
            p.cls[j] = g.dims[j].transposed ? 'T' : 'B';
        }
    }
    p.k1_smem = k1_smem_max(p.S, o.lanes);
    // Operator segments: validation, and the long ones (K1L) with their items.
    long long cells = 0;
    for (int s = 0; s < g.L_op; ++s) {
        const int l = g.op_seg[s];
        if (l < 0 || l >= g.L) bad();
        const long long c0 = g.cell_ptr[l], c1 = g.cell_ptr[l + 1];
        if (c0 < 0 || c1 < c0 || c1 > g.C || !hdfe::detail::ieee_finite(g.seg_inv[l]) || !(g.seg_inv[l] > 0.0))
            bad();
        cells += c1 - c0;
        if (c1 - c0 > o.short_seg) {
            ++p.n_long;
            p.pl_items += (c1 - c0 + o.item - 1) / o.item;
        }
    }
    if (cells != g.C_op) bad();
    for (int w = 1; w <= o.lanes; ++w) p.k1_blocks_max = std::max(p.k1_blocks_max, k1_blocks(g.L_op, w));
    if (p.k1_blocks_max > std::numeric_limits<int>::max() || p.n_long > std::numeric_limits<int>::max() ||
        p.pl_items > std::numeric_limits<int>::max())
        throw Error(ErrorKind::Memory, "CUDA reduced operator: grid size");
    p.pa_slots = p.k1_blocks_max + p.n_long;
    for (int j = 0; j < g.nb; ++j) {
        const DimView& d = g.dims[j];
        if (p.cls[j] == 'T') {
            if (d.tr_ptr == nullptr) bad();
            p.units[j] = count_units(d.tr_ptr, d.levels, o.short_level, o.item);
            p.entries[j] = d.tr_ptr[d.levels];
            if (p.entries[j] != g.C_op) bad();
            if (p.entries[j] > 0) {  // entries to read: their arrays must exist
                if (d.tr_seg == nullptr || (d.has_mult && d.tr_mult == nullptr)) bad();
                for (int jj = 0; jj < g.nb; ++jj)
                    if (jj != j && d.tr_other[jj] == nullptr) bad();
            }
        } else if (p.cls[j] == 'B') {
            p.units[j] = bound_units(d.levels, g.C_op, o.short_level, o.item);
            p.entries[j] = g.C_op;
            p.radix[j] = o.b_method == BMethod::Radix || (o.b_method == BMethod::Auto && d.levels > kTilesMaxLevels);
        }
    }
    return p;
}

// CUB's own temporary sizes, as it reports them for these shapes (queried by the
// operator before the admission): the scans over L_op and over each B effect's
// levels, and the radix sort of each B effect that takes it.
struct CubSizes {
    std::size_t scan_segments = 0;
    std::size_t scan_levels[kMaxDims] = {0, 0, 0, 0};
    std::size_t radix[kMaxDims] = {0, 0, 0, 0};
};

// The build scratch of one B effect, in the order it is carved (each sub-buffer
// aligned to kAlign): counts and bases over the operator segments, the source cell
// and segment of every operator cell, the scan temporary, then either the tile
// histograms (later the cursors) and the level totals, or the radix sort's keys and
// values in double buffers and its temporary.
struct ScratchLayout {
    std::size_t cnt = 0, qbase = 0, qcell = 0, qseg = 0, scan = 0, hist = 0, totals = 0, keys = 0, vals = 0, radix = 0;
    std::size_t bytes = 0;
    long long tiles = 0;
};
inline ScratchLayout scratch_layout(const GeometryView& g, const Plan& p, int j, const CubSizes& cub) {
    ScratchLayout s;
    const std::size_t Lop = static_cast<std::size_t>(g.L_op), Cop = static_cast<std::size_t>(g.C_op);
    const std::size_t mj = static_cast<std::size_t>(g.dims[j].levels);
    std::size_t at = 0;
    const auto carve = [&](std::size_t bytes) {
        const std::size_t here = at;
        at = checked_add(at, align_up(bytes));
        return here;
    };
    s.cnt = carve(checked_mul(Lop, sizeof(int)));
    s.qbase = carve(checked_mul(Lop, sizeof(int)));
    s.qcell = carve(checked_mul(Cop, sizeof(int)));
    s.qseg = carve(checked_mul(Cop, sizeof(int)));
    s.scan = carve(std::max(cub.scan_segments, cub.scan_levels[j]));
    if (p.radix[j]) {
        s.keys = carve(checked_mul(checked_mul(Cop, 2U), sizeof(int)));
        s.vals = carve(checked_mul(checked_mul(Cop, 2U), sizeof(int)));
        s.radix = carve(cub.radix[j]);
    } else {
        s.tiles = (g.C_op + kTile - 1) / kTile;
        s.hist = carve(checked_mul(checked_mul(static_cast<std::size_t>(s.tiles), mj), sizeof(int)));
        s.totals = carve(checked_mul(mj, sizeof(int)));
    }
    s.bytes = at;
    return s;
}

// The admission, before the first allocation (the sizes the set-up allocates, in
// its order). Device peaks per phase as in Admission; the free memory and the
// budget are checked by the operator.
inline Admission admit(const GeometryView& g, const Options& o, const Plan& p, const CubSizes& cub, bool test_cells) {
    Admission a;
    using A = Admission;
    const std::size_t W = static_cast<std::size_t>(o.lanes), m = static_cast<std::size_t>(g.m);
    const std::size_t L = static_cast<std::size_t>(g.L), Lop = static_cast<std::size_t>(g.L_op);
    const std::size_t C = static_cast<std::size_t>(g.C), Cop = static_cast<std::size_t>(g.C_op);
    const std::size_t I = sizeof(int), D = sizeof(double);
    add_entry(a, "cell_ptr", -1, A::Resident, checked_mul(L + 1U, I));
    add_entry(a, "op_seg", -1, A::Resident, checked_mul(Lop, I));
    add_entry(a, "k1_segment_order", -1, A::Resident, checked_mul(Lop, I));
    add_entry(a, "seg_inv", -1, A::Resident, checked_mul(L, D));
    for (int j = 0; j < g.nb; ++j) add_entry(a, "codes", j, A::Resident, checked_mul(C, I));
    if (g.mult != nullptr) add_entry(a, "mult", -1, A::Resident, checked_mul(C, I));
    for (int j = 0; j < g.nb; ++j) {
        if (p.cls[j] != 'T') continue;
        const std::size_t E = static_cast<std::size_t>(p.entries[j]);
        add_entry(a, "level_ptr", j, A::Resident, checked_mul(static_cast<std::size_t>(g.dims[j].levels) + 1U, I));
        add_entry(a, "level_seg_others", j, A::Resident, checked_mul(checked_mul(E, static_cast<std::size_t>(g.nb)), I));
        if (g.dims[j].has_mult) add_entry(a, "level_mult", j, A::Resident, checked_mul(E, I));
    }
    add_entry(a, "private_outputs", -1, A::Resident, checked_mul(static_cast<std::size_t>(p.S), I));
    add_entry(a, "long_segments", -1, A::Resident, checked_mul(static_cast<std::size_t>(p.n_long), I + sizeof(long long)));
    const std::size_t rows = checked_mul(checked_mul(m, W), D);
    add_entry(a, "in", -1, A::Product, rows);
    add_entry(a, "out", -1, A::Product, rows);
    if (W > 1) add_entry(a, "registered_stride_pair", -1, A::Product, checked_mul(rows, 2U));
    add_entry(a, "T", -1, A::Product, checked_mul(checked_mul(L, W), D));
    add_entry(a, "PA", -1, A::Product,
              checked_mul(checked_mul(static_cast<std::size_t>(p.pa_slots), static_cast<std::size_t>(p.S)), checked_mul(W, D)));
    add_entry(a, "PL", -1, A::Product, checked_mul(static_cast<std::size_t>(p.pl_items), checked_mul(W, D)));
    std::size_t scratch = 0;
    std::size_t host_ptrs = 0, host_units = 0, long_levels = 0;
    for (int j = 0; j < g.nb; ++j) {
        if (p.cls[j] == 'P') continue;
        const std::size_t U = static_cast<std::size_t>(p.units[j].units);
        if (p.cls[j] == 'B') {
            add_entry(a, "level_ptr", j, A::Build, checked_mul(static_cast<std::size_t>(g.dims[j].levels) + 1U, I));
            add_entry(a, "level_seg_others", j, A::Build, checked_mul(checked_mul(Cop, static_cast<std::size_t>(g.nb)), I));
            if (g.mult != nullptr) add_entry(a, "level_mult", j, A::Build, checked_mul(Cop, I));
            if (test_cells) add_entry(a, "level_cell_test_hook", j, A::Build, checked_mul(Cop, I));
            scratch = std::max(scratch, scratch_layout(g, p, j, cub).bytes);
            host_ptrs = checked_add(host_ptrs, checked_mul(static_cast<std::size_t>(g.dims[j].levels) + 1U, I));
        }
        add_entry(a, "units", j, A::Units, checked_mul(checked_mul(U, 5U), I));
        add_entry(a, "PB", j, A::Units, checked_mul(static_cast<std::size_t>(p.units[j].slots), checked_mul(W, D)));
        long_levels = checked_add(long_levels, static_cast<std::size_t>(p.units[j].long_levels));
        // Host lists of one effect at a time (8 int vectors at most, see UnitLists) and the sort counts.
        host_units = std::max(host_units, checked_add(checked_mul(checked_mul(U, 5U), I), sort_count_bytes(o.short_level, o.item)));
    }
    add_entry(a, "long_levels", -1, A::Units, checked_mul(checked_mul(long_levels, 3U), I));
    if (scratch > 0) add_entry(a, "b_build", -1, A::Scratch, scratch);
    // Host, the set-up only (an upper bound: these never all live at once).
    const std::size_t chunks = checked_mul(o.chunk_bytes, 2U);
    add_entry(a, "upload_chunks_page_locked", -1, A::Host, chunks);
    add_entry(a, "k1_segment_order_host", -1, A::Host, checked_add(checked_mul(Lop, I), k1_order_count_bytes(o.short_seg, g.L_op)));
    add_entry(a, "long_segments_host", -1, A::Host, checked_mul(static_cast<std::size_t>(p.n_long), I + sizeof(long long)));
    add_entry(a, "b_level_ptr_host", -1, A::Host, host_ptrs);
    add_entry(a, "unit_lists_host", -1, A::Host, host_units);
    add_entry(a, "long_levels_host", -1, A::Host, checked_mul(checked_mul(long_levels, 3U), I));
    add_entry(a, "private_outputs_host", -1, A::Host, checked_mul(static_cast<std::size_t>(p.S), I));
    a.page_locked_setup = chunks;
    // The caller's rows (p and q, or x, r, p and q): registered, not allocated.
    a.page_locked_iteration = checked_mul(rows, static_cast<std::size_t>(std::max(0, o.registered_slices)));
    const std::size_t base = checked_add(checked_add(a.device[A::Resident], a.device[A::Product]), a.device[A::Build]);
    a.device_setup_peak = checked_add(base, a.device[A::Scratch]);
    a.device_iteration_peak = checked_add(base, a.device[A::Units]);
    a.device_peak = std::max(a.device_setup_peak, a.device_iteration_peak);
    a.cub_scan_bytes = cub.scan_segments;
    for (int j = 0; j < g.nb; ++j) {
        a.cub_scan_bytes = std::max(a.cub_scan_bytes, cub.scan_levels[j]);
        a.cub_radix_bytes = std::max(a.cub_radix_bytes, cub.radix[j]);
    }
    return a;
}

struct Statistics {
    // The operator line (event=operator).
    std::size_t host_layout_bytes = 0;     // the set-up's real host peak
    std::size_t device_payload_bytes = 0;  // device bytes held during the iteration
    std::size_t h2d_bytes = 0, d2h_bytes = 0;
    int calls = 0;
    bool timed = false;
    double setup_seconds = 0.0, apply_seconds = 0.0;
    double h2d_ms = 0.0, kernel_ms = 0.0, d2h_ms = 0.0;  // h2d: DMA + pack; d2h: unpack + DMA
    // The attribution line (trace only): the set-up's phases, and per product.
    double plan_seconds = 0.0, device_seconds = 0.0, staging_seconds = 0.0, upload_seconds = 0.0,
           alloc_seconds = 0.0, build_seconds = 0.0, units_seconds = 0.0, register_seconds = 0.0,
           unregister_seconds = 0.0;
    double h2d_dma_ms = 0.0, pack_ms = 0.0, k1_ms = 0.0, k1l_ms = 0.0, k2_ms = 0.0, k3_ms = 0.0,
           unpack_ms = 0.0, d2h_dma_ms = 0.0;
    // Structure (always).
    char classes[kMaxDims + 1] = {0, 0, 0, 0, 0};
    int private_levels = 0, b_radix = 0;
    long long long_segments = 0, long_levels = 0, units = 0, pb_slots = 0, k1_blocks = 0, pa_slots = 0, pl_items = 0;
    std::size_t uploaded_bytes = 0;
};

class Operator {
public:
    // Plans, admits and sets up; `record` (optional) receives the admission as soon
    // as it is computed, also when the set-up then fails.
    Operator(const GeometryView& geometry, const Options& options, Admission* record = nullptr);
    ~Operator();
    Operator(const Operator&) = delete;
    Operator& operator=(const Operator&) = delete;
    // The real width of the next group (1..lanes), from solve(), before EACH group.
    void begin_group(int width);
    // ONE registration (page-locking) of the caller's covering range of `count`
    // doubles, kept until unregister_host(); apply() reads and writes only inside it.
    void register_host(double* begin, std::size_t count);
    // Waits for the stream (nothing in flight), then releases the registration.
    void unregister_host() noexcept;
    // in, out: m rows of `lanes` doubles inside the registered range; the lanes of out
    // from the width on are written +0.0.
    void apply(const double* in, double* out);
    const Statistics& statistics() const noexcept;
    const Admission& admission() const noexcept;
    char effect_class(int j) const noexcept;
    static std::size_t host_state_bytes() noexcept;
    // Tests: the device-built level-ordered arrays of a B effect (cell: test-hook
    // builds with Options::test_source_cells only, else empty).
    struct LevelOrderCopy {
        std::vector<int> ptr, seg, mult, cell;
        std::vector<std::vector<int>> other;
    };
    LevelOrderCopy download_level_order(int j) const;
    // Tests: the device bytes held by every operator alive in this process
    // (test-hook builds; 0 otherwise).
    static std::size_t live_device_bytes() noexcept;
private:
    struct State;
    std::unique_ptr<State> state_;
};

bool enabled() noexcept;

} // namespace hdfe::detail::reduced_cuda
#endif
