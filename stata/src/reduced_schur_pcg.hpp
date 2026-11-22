#ifndef HDFE_REDUCED_SCHUR_PCG_HPP
#define HDFE_REDUCED_SCHUR_PCG_HPP

// R1 (on by default since the 2.32 candidate, XHDFE_REDUCED_PCG=0 turns it off;
// design and audit in Verifications/perf_reduced_system_20261113/r1/).
//
// Exact elimination of one fixed effect (e) and Jacobi-preconditioned
// conjugate gradients on the Schur complement of the remaining effects:
//     S x = g,  S = D2'(I - P1)D2,  g = D2'(I - P1)v,
// where P1 replaces each value by the mean of its level of e. The columns of
// a group are independent recurrences that share each traversal of the
// geometry. The observations are read only to form g (with the norms the
// stopping rule needs) and, at the end, to form the residuals
// v - D1 a1 - D2 x with a1 = mean over each level of e of (v - D2 x).
//
// Self-contained on purpose (plain pointers and sizes, the standard library
// and OpenMP), unit weights only. Every floating-point reduction runs over a
// partition fixed by the data and by Config::blocks, never by the team, and
// adds its partials in partition order: the bits do not depend on the number
// of threads. Non-finite values are detected on the IEEE bits, because the
// core is compiled with -ffast-math.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

// Buffer's 64-byte blocks (17nov2026): the aligned operator new of C++17 is not
// available to the macOS Intel slice at its deployment target (10.12), so Apple
// targets take them from posix_memalign and give them back with std::free. A test
// build can force that branch elsewhere with XHDFE_REDUCED_PCG_POSIX_MEMALIGN.
#if defined(__APPLE__) || defined(XHDFE_REDUCED_PCG_POSIX_MEMALIGN)
#define RPCG_BUFFER_POSIX_MEMALIGN 1
#include <cstdlib>
#endif

// Arithmetic as written inside the row-level kernels, whatever the flags of
// the build (the core uses -ffast-math): (v - hi) - (small + s) must not be
// reassociated into v - (hi + small + s), which would bring back eps |mean|
// per row on a column dominated by a constant. Same attributes as
// XHDFE_EXACT_FP (include/hdfe/deterministic_parallel.hpp), restated so that
// this header stays self-contained.
#if defined(__clang__)
#define RPCG_EXACT_FP
#define RPCG_EXACT_FP_BODY _Pragma("float_control(precise, on)") _Pragma("clang fp contract(off)")
#elif defined(__GNUC__)
#define RPCG_EXACT_FP                                                      \
    __attribute__((optimize("fp-contract=off", "no-associative-math",      \
                            "no-reciprocal-math", "signed-zeros",          \
                            "no-unsafe-math-optimizations")))
#define RPCG_EXACT_FP_BODY
#else
#define RPCG_EXACT_FP
#define RPCG_EXACT_FP_BODY
#endif

// The kernel of the operator's segment pass for large effects
// (internal::hinted_segment_part) needs the vector types of GCC, and is offered
// only where its sums are known to be the loop's. With Clang they are not: the
// loop's sums over four remaining effects come out reassociated under the
// build's fast-math flags and the kernel's do not (Clang 20, 21nov2026), so
// Clang builds keep the loop for every shape.
#if defined(__GNUC__) && !defined(__clang__)
#define RPCG_HINTED_SEGMENT_KERNEL 1
#else
#define RPCG_HINTED_SEGMENT_KERNEL 0
#endif

namespace hdfe {
namespace reduced_pcg {

// Why a column stopped.
enum class Stop : int {
    None = 0,
    Zero,   // a zero column: no iteration
    Span,   // in the span of the eliminated effect: no iteration
    Test2,  // tau <= atol, confirmed on the replaced residual
    Test1,  // compatible system, confirmed on the replaced residual
    Chat,   // chat <= Config::chat_stop, confirmed on the replaced residual (D10)
};

// Why the whole attempt was declined (control returns to the caller's chain).
enum class Decline : int {
    None = 0,
    Input,
    Memory,
    Stall,
    Ceiling,
    Projected,
    NonFinite,
    Breakdown,
    Forced,
};

inline const char* stop_name(Stop stop) noexcept {
    switch (stop) {
        case Stop::Zero: return "zero";
        case Stop::Span: return "span";
        case Stop::Test2: return "test2";
        case Stop::Test1: return "test1";
        case Stop::Chat: return "chat";
        case Stop::None:
        default: return "none";
    }
}

inline const char* decline_name(Decline decline) noexcept {
    switch (decline) {
        case Decline::Input: return "input";
        case Decline::Memory: return "memory";
        case Decline::Stall: return "stall";
        case Decline::Ceiling: return "ceiling";
        case Decline::Projected: return "projected";
        case Decline::NonFinite: return "nonfinite";
        case Decline::Breakdown: return "breakdown";
        case Decline::Forced: return "forced";
        case Decline::None:
        default: return "none";
    }
}

// Parallel-work observer, as three callbacks.
struct Hooks {
    void* context = nullptr;
    void (*begin)(void*, int) = nullptr;
    void (*observe)(void*) = nullptr;
    void (*end)(void*) = nullptr;
};

struct Config {
    int threads = 1;
    int blocks = 48;                 // logical partition, independent of `threads`
    double atol = 1.0e-11;           // Krylov floor of the mode
    double btol = 1.0e-11;
    double trigger = 1.0e-11;        // acceptance trigger: reported as chat, a stop condition only with `companion`
    bool companion = false;          // also require chat <= trigger before a column stops (study switch)
    // D10 (pre-registration Verifications/perf_stata_path_20261114/d10/, reading B):
    // when positive, a column stops when its chat, confirmed on the replaced
    // residual, is at most chat_stop, and tau <= atol is no longer an exit on any
    // path; Zero, Span and test 1 stay; the stall and projection declines read
    // chat / chat_stop. 0 keeps the rule before D10, bit for bit.
    double chat_stop = 0.0;
    int max_iter = 1500;
    int stall_window = 50;
    int refresh_period = 50;
    double floor_factor = 64.0;
    double merge_threshold = 0.8;
    int merge_sample_stride = 64;
    bool dominant_origin = true;     // certificate scale rule of reghdfe-comparable
    // Segment pass of the operator (apply_operator, phase A). Where an effect's
    // rows of the reduced vector do not fit in cache (internal::kPrefetchMinBytes)
    // and no cell is repeated, the pass goes through a kernel that requests those
    // rows this many cells ahead (software prefetch) and adds the rows as
    // vectors (internal::hinted_segment_part; GCC builds only). 0 keeps the loop
    // of 2.34.1 for every shape. The kernel forms the same sums in the same
    // order, so the bits do not depend on this field: checked by the unit
    // test (21nov2026, Verifications/fereg_liu_study_20261121/).
    int prefetch_cells = 48;
    // Admission: an attempt whose memory_bound() exceeds this is declined
    // (Decline::Memory) before it allocates; the adapter sets admission_budget().
    std::size_t memory_budget = std::numeric_limits<std::size_t>::max();
    int force_effect_class = -1;     // tests: -1 automatic, 0 all private, 1 all transposed
    int force_merge = -1;            // tests: -1 automatic, 0 never, 1 always
    bool force_decline_geometry = false;
    bool force_decline_iterate = false;
    int test_zero_curvature_at = -1; // tests: read the curvature p'Sp as 0 at this iteration
    int test_recurrence_chat_at = -1;          // tests: at this iteration the recurrence's chat reads as
    double test_recurrence_chat_value = 0.0;   // this value (a candidate, or none, before the replacement)
    Hooks hooks;
    // Tests: called for every active column after every iteration.
    void* iteration_context = nullptr;
    void (*on_iteration)(void*, int column, int iteration, double tau, double chat) = nullptr;
};

struct Problem {
    int n = 0;
    int k = 0;                               // fixed effects, 2..5
    const int* const* codes = nullptr;       // k arrays of n codes in [0, levels[b])
    const int* levels = nullptr;             // k
    int eliminated = 0;
    int cols = 0;
    const double* const* columns = nullptr;  // cols arrays of n
    double* const* residuals = nullptr;      // cols arrays of n, written when solved
};

struct ColumnReport {
    Stop stop = Stop::None;
    int iterations = 0;
    int first_tau_pass = -1;   // first iteration with tau <= atol (-1: never)
    double tau = 0.0;
    double chat = 0.0;         // predicted certificate at the stop
    double s_rel = 0.0;        // current residual norm over the column norm
    int formed_checks = 0;     // passes over the column to measure the formed residual
};

struct Report {
    bool solved = false;
    Decline decline = Decline::None;
    int decline_column = -1;
    int iterations = 0;
    std::vector<ColumnReport> columns;
    int segments = 0;
    int operator_segments = 0;
    long long cells = 0;
    bool sorted = false;
    bool merged = false;
    int large_effects = 0;
    int groups = 0;
    int lanes = 0;
    std::size_t bytes = 0;
    double geometry_seconds = 0.0;
    double rhs_seconds = 0.0;
    double iterate_seconds = 0.0;
    double recover_seconds = 0.0;
    double residual_seconds = 0.0;
    long long hinted_applies = 0;  // applies of the operator that went through internal::hinted_segment_part
};

// Admission budget of one attempt (2.32): 16 bytes per row for every numeric
// column and every fixed-effect identifier of the fit, and at least 1 GiB.
// What an attempt allocates is dominated by per-row index structures (codes,
// cells, permutation, scratch) that scale with the identifiers rather than the
// columns: a budget on the columns alone (16 n cols) kept narrow fits on large
// data out of the tier (one regressor on ready: bound 1.86 GB, budget 1.48 GB).
inline std::size_t admission_budget(int n, int cols, int fixed_effects) noexcept {
    const std::size_t rows = static_cast<std::size_t>(std::max(0, n));
    const std::size_t per_row = 16U * (static_cast<std::size_t>(std::max(0, cols)) +
                                       static_cast<std::size_t>(std::max(0, fixed_effects)));
    if (per_row != 0 && rows > std::numeric_limits<std::size_t>::max() / per_row)
        return std::numeric_limits<std::size_t>::max();
    return std::max(std::size_t{1} << 30, rows * per_row);
}

namespace internal {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int kVectorUnit = 4096;
constexpr int kStats = 5;  // sum v^2, sum (v - mean_l)^2, sum v, sum (v - v0), sum (v - v0)^2
// hinted_segment_part takes the operator's segment pass when an effect's rows of
// the reduced vector take at least this many bytes (levels * lanes * 8), about
// where they stop fitting in a core's cache. A rule of the data and the columns
// only, and a cautious one. On synthetic mobility panels of 3 M rows (21nov2026,
// 1 and 16 threads) the kernel saved 11 to 32 % of an iteration at 8 lanes at
// every size from 0.1 to 9 MB; at 4 lanes it was within 2 % of the loop up to
// 0.5 MB and saved 14 to 26 % from 1.5 MB at 16 threads. Shapes below the
// threshold keep the loop of 2.34.1 until they are measured on real panels.
constexpr std::size_t kPrefetchMinBytes = std::size_t{2} << 20;
constexpr int kMaxRemaining = 4;  // the tier takes at most 5 effects: 4 remain
constexpr bool kHintedSegmentKernel = RPCG_HINTED_SEGMENT_KERNEL != 0;

inline bool finite(double value) noexcept {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

inline long long part_begin(long long n, int b, int parts) noexcept {
    const long long base = n / parts;
    const long long extra = n % parts;
    return static_cast<long long>(b) * base + std::min<long long>(b, extra);
}

inline double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// 64-byte aligned array; allocation failure throws std::bad_alloc.
template <class T>
class Buffer {
public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    ~Buffer() { release(); }

    void allocate(std::size_t count) {
        release();
        if (count == 0) count = 1;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
#if defined(RPCG_BUFFER_POSIX_MEMALIGN)
        void* block = nullptr;
        if (posix_memalign(&block, 64, count * sizeof(T)) != 0 || block == nullptr) throw std::bad_alloc();
        data_ = static_cast<T*>(block);
#else
        data_ = static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t(64)));
#endif
        size_ = count;
    }
    void zero() noexcept {
        if (data_ != nullptr) std::memset(static_cast<void*>(data_), 0, size_ * sizeof(T));
    }
    void release() noexcept {
        if (data_ != nullptr) {
#if defined(RPCG_BUFFER_POSIX_MEMALIGN)
            std::free(data_);
#else
            ::operator delete(data_, std::align_val_t(64));
#endif
            data_ = nullptr;
            size_ = 0;
        }
    }
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

// The team the caller resolved. Work units are fixed by the data; the team
// takes them in contiguous chunks (at most `chunk_target` per region, a number
// fixed by the configuration, never by the team), so a region with many small
// units does not pay one dispatch per unit. Every unit is still computed on
// its own and combined by its caller in a fixed order: the chunking changes
// who computes what, never the bits. Each participating worker reports once
// per region to the observer hooks.
class Team {
public:
    Team(int threads, const Hooks& hooks, int chunk_target = 384)
        : threads_(std::max(1, threads)), hooks_(hooks), chunk_target_(std::max(1, chunk_target)) {}
    int threads() const noexcept { return threads_; }

    template <class Body>
    void run(int units, Body&& body) const {
        if (units <= 0) return;
        const int chunks = std::min(units, chunk_target_);
        if (hooks_.begin != nullptr) hooks_.begin(hooks_.context, threads_);
        const int team = threads_;
#if defined(_OPENMP)
#pragma omp parallel num_threads(team) if (team > 1 && chunks > 1)
#endif
        {
            bool observed = false;
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 1)
#endif
            for (int chunk = 0; chunk < chunks; ++chunk) {
                if (!observed) {
                    if (hooks_.observe != nullptr) hooks_.observe(hooks_.context);
                    observed = true;
                }
                const int u1 = static_cast<int>(part_begin(units, chunk + 1, chunks));
                for (int unit = static_cast<int>(part_begin(units, chunk, chunks)); unit < u1; ++unit) body(unit);
            }
        }
        if (hooks_.end != nullptr) hooks_.end(hooks_.context);
    }

private:
    int threads_;
    Hooks hooks_;
    int chunk_target_;
};

inline int thread_slot() noexcept {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// Contiguous ranges of `count` items with about equal weight; `prefix` has
// count + 1 non-decreasing entries. Fixed by the data and `parts` only.
inline std::vector<int> balanced_bounds(const int* prefix, int count, int parts) {
    parts = std::max(1, parts);
    std::vector<int> bounds(static_cast<std::size_t>(parts) + 1U, count);
    bounds[0] = 0;
    const long long first = count > 0 ? prefix[0] : 0;
    const long long total = count > 0 ? static_cast<long long>(prefix[count]) - first : 0;
    for (int b = 1; b < parts; ++b) {
        const long long target = first + part_begin(total, b, parts);
        const int* it = std::lower_bound(prefix, prefix + count + 1, target,
                                         [](int value, long long t) { return value < t; });
        const int at = std::min(count, static_cast<int>(it - prefix));
        bounds[static_cast<std::size_t>(b)] = std::max(bounds[static_cast<std::size_t>(b) - 1U], at);
    }
    return bounds;
}

struct Transpose {
    Buffer<int> ptr;                 // levels + 1
    Buffer<int> seg;                 // entry -> level of e of its cell
    std::vector<Buffer<int>> other;  // entry -> code of every other remaining effect
    Buffer<int> mult;                // merged geometry only
    Buffer<int> item_level;
    Buffer<int> item_begin;
    Buffer<int> item_end;
    Buffer<int> item_ptr;            // levels + 1: the items of each level
    std::vector<int> item_blocks;    // contiguous blocks of items with about equal entries
    int items = 0;
    bool has_mult = false;
};

struct Geometry {
    int n = 0, k = 0, e = 0, nb = 0, L = 0, m = 0;
    std::vector<int> remaining;      // effect of each remaining position j
    std::vector<int> offset;         // nb + 1 offsets in the reduced vector
    bool sorted = false;
    bool merged = false;
    bool copied = false;
    Buffer<int> perm;                // position in segment order -> row (unsorted only)
    Buffer<int> row_ptr;             // L + 1, positions in segment order
    Buffer<int> cell_ptr_own;        // L + 1 (merged only)
    const int* cell_ptr = nullptr;   // row_ptr or cell_ptr_own
    std::vector<Buffer<int>> cell_code_own;
    std::vector<const int*> cc;      // nb: code of remaining effect j at cell c
    Buffer<int> mult_own;
    const int* mult = nullptr;       // null: multiplicity one everywhere
    long long C = 0;
    long long C_op = 0;
    Buffer<int> op_seg;              // segments with more than one distinct cell
    int L_op = 0;
    std::vector<int> op_bounds;      // blocks over operator segments, by cells
    std::vector<int> all_bounds;     // blocks over all segments, by cells
    std::vector<int> row_bounds;     // blocks over all segments, by rows
    std::vector<char> large;         // per remaining position
    std::vector<int> small_list;     // positions with private accumulators
    std::vector<int> small_offset;   // their offsets in a private block
    std::vector<int> small_target;   // private index -> reduced index
    int small_levels = 0;
    std::vector<Transpose> tr;       // per remaining position (large ones filled)
    Buffer<double> seg_inv;          // L: 1/n_l (0 for empty levels)
    Buffer<double> diag;             // m: S_jj
    Buffer<double> minv;             // m: 1/S_jj, 0 on null levels
};

inline int rows_of(const Geometry& g, int l) noexcept {
    return g.row_ptr[static_cast<std::size_t>(l) + 1U] - g.row_ptr[static_cast<std::size_t>(l)];
}

inline int row_at(const Geometry& g, int position) noexcept {
    return g.sorted ? position : g.perm[static_cast<std::size_t>(position)];
}

inline bool same_tuple(const Problem& p, const Geometry& g, int a, int b) noexcept {
    for (int j = 0; j < g.nb; ++j) {
        const int* codes = p.codes[g.remaining[static_cast<std::size_t>(j)]];
        if (codes[a] != codes[b]) return false;
    }
    return true;
}

// Sorts the rows of segment l by their tuple of remaining codes and returns
// the number of distinct tuples; `scratch` holds at least n_l ints.
inline int sort_segment_tuples(const Problem& p, const Geometry& g, int l, int* scratch) {
    const int r0 = g.row_ptr[static_cast<std::size_t>(l)];
    const int r1 = g.row_ptr[static_cast<std::size_t>(l) + 1U];
    for (int r = r0; r < r1; ++r) scratch[r - r0] = row_at(g, r);
    std::sort(scratch, scratch + (r1 - r0), [&](int a, int b) {
        for (int j = 0; j < g.nb; ++j) {
            const int* codes = p.codes[g.remaining[static_cast<std::size_t>(j)]];
            if (codes[a] != codes[b]) return codes[a] < codes[b];
        }
        return a < b;
    });
    int distinct = r1 > r0 ? 1 : 0;
    for (int i = 1; i < r1 - r0; ++i)
        if (!same_tuple(p, g, scratch[i - 1], scratch[i])) ++distinct;
    return distinct;
}

inline int lanes_for(int cols) {
    int w = 1;
    while (w < cols && w < 8) w *= 2;
    return w;
}

// Effect class: private accumulators when their reduction (2 P m_j W 8 bytes,
// W = 8 so the class does not depend on the columns) costs less than a
// transposed stream (4 nb C bytes).
inline bool is_large(long double mj, long double nb, long double C, int blocks, int forced) {
    if (forced == 0) return false;
    if (forced == 1) return true;
    return 2.0L * blocks * mj * 64.0L > 4.0L * nb * C;
}

// Upper bound of what the attempt allocates, from shapes only (C = n).
inline std::size_t memory_bound(const Problem& p, const Geometry& g, const Config& cfg,
                                bool merge, int lanes) {
    const long double n = p.n, L = g.L, m = g.m, nb = g.nb;
    const int B = std::max(1, cfg.blocks);
    const long double C = n, mu = merge ? 1.0L : 0.0L, W = lanes;
    long double bytes = 0.0L;
    if (!g.sorted) bytes += 8.0L * n;                                  // permutation, radix buffer
    if (!g.sorted || merge) bytes += 4.0L * (nb + mu) * C + 4.0L * (L + 1.0L);
    bytes += 4.0L * mu * n;  // segment-sort scratch: the merge sample, then the merge passes (never both)
    bytes += 4.0L * (L + 1.0L) + 8.0L * L + 4.0L * L + 4.0L * L;       // pointers, 1/n_l, lists
    long double small_levels = 0.0L;
    for (int j = 0; j < g.nb; ++j) {
        const long double mj = p.levels[g.remaining[static_cast<std::size_t>(j)]];
        if (is_large(mj, nb, C, B, cfg.force_effect_class)) {
            bytes += 4.0L * (mj + 1.0L) + 4.0L * (nb + mu) * C + 4.0L * B * mj +
                     16.0L * (mj + C / 1024.0L + 1.0L) + 8.0L * W * (mj + C / 1024.0L + 1.0L);
        } else {
            small_levels += mj;
        }
    }
    bytes += 16.0L * m;                                                // diagonal, its inverse
    bytes += 8.0L * W * (5.0L * m + 3.0L * L);                         // g, x, r, p, q, T, vbar, corr
    bytes += 8.0L * W * B * small_levels + 9.0L * B * small_levels;    // private blocks
    if (bytes > static_cast<long double>(std::numeric_limits<std::size_t>::max() / 2))
        return std::numeric_limits<std::size_t>::max();
    return static_cast<std::size_t>(bytes);
}

// Stable LSD radix sort (8-bit digits) of the rows by their level of e, over
// fixed blocks of positions: inside each level the rows keep the input order.
inline void radix_by_eliminated(const Problem& p, Geometry& g, const Team& team, int blocks) {
    const int n = p.n;
    const int* key = p.codes[g.e];
    int bits = 1;
    while (bits < 31 && (1LL << bits) < static_cast<long long>(g.L)) ++bits;
    const int passes = (bits + 7) / 8;
    Buffer<int> spare;
    g.perm.allocate(static_cast<std::size_t>(n));
    spare.allocate(static_cast<std::size_t>(n));
    const int parts = std::max(1, std::min(blocks, n));
    std::vector<int> hist(static_cast<std::size_t>(parts) * 256U);
    int* src = g.perm.data();
    int* dst = spare.data();
    for (int pass = 0; pass < passes; ++pass) {
        const int shift = 8 * pass;
        const bool identity = pass == 0;
        std::fill(hist.begin(), hist.end(), 0);
        team.run(parts, [&](int b) {
            int* h = hist.data() + static_cast<std::size_t>(b) * 256U;
            const int i0 = static_cast<int>(part_begin(n, b, parts));
            const int i1 = static_cast<int>(part_begin(n, b + 1, parts));
            for (int i = i0; i < i1; ++i) ++h[(key[identity ? i : src[i]] >> shift) & 255];
        });
        int position = 0;
        for (int digit = 0; digit < 256; ++digit) {
            for (int b = 0; b < parts; ++b) {
                int& slot = hist[static_cast<std::size_t>(b) * 256U + static_cast<std::size_t>(digit)];
                const int count = slot;
                slot = position;
                position += count;
            }
        }
        team.run(parts, [&](int b) {
            int* cursor = hist.data() + static_cast<std::size_t>(b) * 256U;
            const int i0 = static_cast<int>(part_begin(n, b, parts));
            const int i1 = static_cast<int>(part_begin(n, b + 1, parts));
            for (int i = i0; i < i1; ++i) {
                const int row = identity ? i : src[i];
                dst[cursor[(key[row] >> shift) & 255]++] = row;
            }
        });
        std::swap(src, dst);
    }
    if (src != g.perm.data())
        std::memcpy(g.perm.data(), src, sizeof(int) * static_cast<std::size_t>(n));
}

// Level pointers of e from keys that never decrease along the positions.
template <class KeyAt>
inline void fill_level_pointers(int n, int L, KeyAt key_at, int* ptr, const Team& team, int blocks) {
    const int first = key_at(0);
    const int last = key_at(n - 1);
    for (int level = 0; level <= first; ++level) ptr[level] = 0;
    for (int level = last + 1; level <= L; ++level) ptr[level] = n;
    const int parts = std::max(1, std::min(blocks, n));
    team.run(parts, [&](int b) {
        const int i0 = std::max(1, static_cast<int>(part_begin(n, b, parts)));
        const int i1 = static_cast<int>(part_begin(n, b + 1, parts));
        for (int i = i0; i < i1; ++i) {
            const int before = key_at(i - 1);
            const int now = key_at(i);
            for (int level = before + 1; level <= now; ++level) ptr[level] = i;
        }
    });
}

inline Decline build_geometry(const Problem& p, const Config& cfg, const Team& team, int lanes,
                              Geometry& g, std::size_t* bytes_out) {
    const int n = p.n;
    const int B = std::max(1, cfg.blocks);
    g.n = n;
    g.k = p.k;
    g.e = p.eliminated;
    g.nb = p.k - 1;
    g.L = p.levels[g.e];
    // Point 19 (17nov2026): the reduced vector is indexed by int throughout the
    // tier, so the levels of the effects that are not eliminated must sum to at
    // most INT_MAX. The sum is formed in 64 bits before g.offset and g.m (and the
    // admission below) read it; one that does not fit declines the attempt
    // (Decline::Input) before anything is allocated, and the caller runs its
    // chain. Every sum that fits is formed exactly as before.
    long long reduced_levels = 0;
    for (int b = 0; b < p.k; ++b)
        if (b != g.e) reduced_levels += p.levels[b];
    if (reduced_levels > static_cast<long long>(std::numeric_limits<int>::max())) return Decline::Input;
    g.offset.assign(1, 0);
    for (int b = 0; b < p.k; ++b) {
        if (b == g.e) continue;
        g.remaining.push_back(b);
        g.offset.push_back(g.offset.back() + p.levels[b]);
    }
    g.m = g.offset.back();
    const int* key = p.codes[g.e];

    {
        const int parts = std::max(1, std::min(B, n));
        std::vector<char> ok(static_cast<std::size_t>(parts), 1);
        team.run(parts, [&](int b) {
            const int i0 = std::max(1, static_cast<int>(part_begin(n, b, parts)));
            const int i1 = static_cast<int>(part_begin(n, b + 1, parts));
            char good = 1;
            for (int i = i0; i < i1; ++i) good = static_cast<char>(good & (key[i - 1] <= key[i]));
            ok[static_cast<std::size_t>(b)] = good;
        });
        g.sorted = std::all_of(ok.begin(), ok.end(), [](char v) { return v != 0; });
    }

    // Admission from shapes, with the merge assumed possible.
    *bytes_out = memory_bound(p, g, cfg, cfg.force_merge != 0, lanes);
    if (*bytes_out > cfg.memory_budget) return Decline::Memory;

    if (!g.sorted) radix_by_eliminated(p, g, team, B);
    g.row_ptr.allocate(static_cast<std::size_t>(g.L) + 1U);
    if (g.sorted) {
        fill_level_pointers(n, g.L, [&](int i) { return key[i]; }, g.row_ptr.data(), team, B);
    } else {
        const int* perm = g.perm.data();
        fill_level_pointers(n, g.L, [&](int i) { return key[perm[i]]; }, g.row_ptr.data(), team, B);
    }
    int max_rows = 1;
    for (int l = 0; l < g.L; ++l) max_rows = std::max(max_rows, rows_of(g, l));
    g.row_bounds = balanced_bounds(g.row_ptr.data(), g.L, B);
    const int row_parts = static_cast<int>(g.row_bounds.size()) - 1;

    // Merge when a deterministic sample of segments (one in `stride`)
    // projects at most merge_threshold * N cells.
    bool merge = cfg.force_merge == 1;
    if (cfg.force_merge == -1) {
        std::vector<int> scratch(static_cast<std::size_t>(max_rows));
        long long sampled_rows = 0, sampled_cells = 0;
        const int stride = std::max(1, cfg.merge_sample_stride);
        for (int l = 0; l < g.L; l += stride) {
            if (rows_of(g, l) == 0) continue;
            sampled_rows += rows_of(g, l);
            sampled_cells += sort_segment_tuples(p, g, l, scratch.data());
        }
        merge = sampled_rows > 0 && static_cast<long double>(sampled_cells) <=
                                        static_cast<long double>(cfg.merge_threshold) * sampled_rows;
    }
    g.merged = merge;
    g.copied = merge || !g.sorted;

    // Cells, and the number of distinct cells of each segment.
    g.cc.assign(static_cast<std::size_t>(g.nb), nullptr);
    std::vector<int> distinct(static_cast<std::size_t>(g.L), 0);
    if (merge) {
        // Scratch of the segment sorts: one slice per block of segments, as
        // long as the block's largest segment, so at most n ints in all and the
        // same for every team and schedule; allocated before the parallel
        // regions and kept for both passes. Scratch only: the sorts, and so
        // every bit downstream, do not depend on where it lives.
        std::vector<std::size_t> slice(static_cast<std::size_t>(row_parts) + 1U, 0);
        for (int b = 0; b < row_parts; ++b) {
            int widest = 0;
            for (int l = g.row_bounds[static_cast<std::size_t>(b)];
                 l < g.row_bounds[static_cast<std::size_t>(b) + 1U]; ++l)
                widest = std::max(widest, rows_of(g, l));
            slice[static_cast<std::size_t>(b) + 1U] = slice[static_cast<std::size_t>(b)] + static_cast<std::size_t>(widest);
        }
        Buffer<int> scratch;
        scratch.allocate(slice[static_cast<std::size_t>(row_parts)]);
        team.run(row_parts, [&](int b) {
            int* s = scratch.data() + slice[static_cast<std::size_t>(b)];
            for (int l = g.row_bounds[static_cast<std::size_t>(b)];
                 l < g.row_bounds[static_cast<std::size_t>(b) + 1U]; ++l)
                distinct[static_cast<std::size_t>(l)] = sort_segment_tuples(p, g, l, s);
        });
        g.cell_ptr_own.allocate(static_cast<std::size_t>(g.L) + 1U);
        g.cell_ptr_own[0] = 0;
        for (int l = 0; l < g.L; ++l)
            g.cell_ptr_own[static_cast<std::size_t>(l) + 1U] =
                g.cell_ptr_own[static_cast<std::size_t>(l)] + distinct[static_cast<std::size_t>(l)];
        g.C = g.cell_ptr_own[static_cast<std::size_t>(g.L)];
        g.cell_code_own.resize(static_cast<std::size_t>(g.nb));
        for (int j = 0; j < g.nb; ++j)
            g.cell_code_own[static_cast<std::size_t>(j)].allocate(static_cast<std::size_t>(g.C));
        g.mult_own.allocate(static_cast<std::size_t>(g.C));
        team.run(row_parts, [&](int b) {
            int* s = scratch.data() + slice[static_cast<std::size_t>(b)];
            for (int l = g.row_bounds[static_cast<std::size_t>(b)];
                 l < g.row_bounds[static_cast<std::size_t>(b) + 1U]; ++l) {
                const int rows = rows_of(g, l);
                if (rows == 0) continue;
                sort_segment_tuples(p, g, l, s);
                int c = g.cell_ptr_own[static_cast<std::size_t>(l)];
                int count = 1;
                for (int i = 1; i <= rows; ++i) {
                    if (i < rows && same_tuple(p, g, s[i - 1], s[i])) {
                        ++count;
                        continue;
                    }
                    for (int j = 0; j < g.nb; ++j)
                        g.cell_code_own[static_cast<std::size_t>(j)][static_cast<std::size_t>(c)] =
                            p.codes[g.remaining[static_cast<std::size_t>(j)]][s[i - 1]];
                    g.mult_own[static_cast<std::size_t>(c)] = count;
                    ++c;
                    count = 1;
                }
            }
        });
        g.cell_ptr = g.cell_ptr_own.data();
        g.mult = g.mult_own.data();
        for (int j = 0; j < g.nb; ++j)
            g.cc[static_cast<std::size_t>(j)] = g.cell_code_own[static_cast<std::size_t>(j)].data();
    } else {
        g.cell_ptr = g.row_ptr.data();
        g.C = n;
        if (g.copied) {
            g.cell_code_own.resize(static_cast<std::size_t>(g.nb));
            for (int j = 0; j < g.nb; ++j)
                g.cell_code_own[static_cast<std::size_t>(j)].allocate(static_cast<std::size_t>(n));
            const int parts = std::max(1, std::min(B, n));
            team.run(parts, [&](int b) {
                const int i0 = static_cast<int>(part_begin(n, b, parts));
                const int i1 = static_cast<int>(part_begin(n, b + 1, parts));
                for (int j = 0; j < g.nb; ++j) {
                    const int* codes = p.codes[g.remaining[static_cast<std::size_t>(j)]];
                    int* out = g.cell_code_own[static_cast<std::size_t>(j)].data();
                    for (int i = i0; i < i1; ++i) out[i] = codes[g.perm[static_cast<std::size_t>(i)]];
                }
            });
            for (int j = 0; j < g.nb; ++j)
                g.cc[static_cast<std::size_t>(j)] = g.cell_code_own[static_cast<std::size_t>(j)].data();
        } else {
            for (int j = 0; j < g.nb; ++j)
                g.cc[static_cast<std::size_t>(j)] = p.codes[g.remaining[static_cast<std::size_t>(j)]];
        }
        // A segment whose cells all carry one tuple contributes nothing.
        team.run(row_parts, [&](int b) {
            for (int l = g.row_bounds[static_cast<std::size_t>(b)];
                 l < g.row_bounds[static_cast<std::size_t>(b) + 1U]; ++l) {
                const int c0 = g.cell_ptr[l], c1 = g.cell_ptr[l + 1];
                int many = 0;
                for (int c = c0 + 1; c < c1 && many == 0; ++c)
                    for (int j = 0; j < g.nb && many == 0; ++j)
                        many = g.cc[static_cast<std::size_t>(j)][c] != g.cc[static_cast<std::size_t>(j)][c0];
                distinct[static_cast<std::size_t>(l)] = c1 > c0 ? 1 + many : 0;
            }
        });
    }

    // Operator segments and the partitions over them.
    g.L_op = 0;
    for (int l = 0; l < g.L; ++l) g.L_op += distinct[static_cast<std::size_t>(l)] >= 2 ? 1 : 0;
    g.op_seg.allocate(static_cast<std::size_t>(g.L_op));
    {
        int s = 0;
        for (int l = 0; l < g.L; ++l)
            if (distinct[static_cast<std::size_t>(l)] >= 2) g.op_seg[static_cast<std::size_t>(s++)] = l;
        std::vector<int> prefix(static_cast<std::size_t>(g.L_op) + 1U, 0);
        for (int t = 0; t < g.L_op; ++t) {
            const int l = g.op_seg[static_cast<std::size_t>(t)];
            prefix[static_cast<std::size_t>(t) + 1U] =
                prefix[static_cast<std::size_t>(t)] + (g.cell_ptr[l + 1] - g.cell_ptr[l]);
        }
        g.C_op = prefix[static_cast<std::size_t>(g.L_op)];
        g.op_bounds = balanced_bounds(prefix.data(), g.L_op, B);
        g.all_bounds = balanced_bounds(g.cell_ptr, g.L, B);
    }
    const int op_parts = static_cast<int>(g.op_bounds.size()) - 1;

    g.seg_inv.allocate(static_cast<std::size_t>(g.L));
    for (int l = 0; l < g.L; ++l)
        g.seg_inv[static_cast<std::size_t>(l)] = rows_of(g, l) > 0 ? 1.0 / static_cast<double>(rows_of(g, l)) : 0.0;

    g.large.assign(static_cast<std::size_t>(g.nb), 0);
    g.tr.resize(static_cast<std::size_t>(g.nb));
    for (int j = 0; j < g.nb; ++j) {
        const int mj = p.levels[g.remaining[static_cast<std::size_t>(j)]];
        const bool large = is_large(mj, g.nb, static_cast<long double>(g.C), B, cfg.force_effect_class);
        g.large[static_cast<std::size_t>(j)] = large ? 1 : 0;
        if (!large) {
            g.small_list.push_back(j);
            g.small_offset.push_back(g.small_levels);
            for (int lev = 0; lev < mj; ++lev) g.small_target.push_back(g.offset[static_cast<std::size_t>(j)] + lev);
            g.small_levels += mj;
        }
    }

    g.diag.allocate(static_cast<std::size_t>(g.m));
    g.diag.zero();
    std::vector<char> partial(static_cast<std::size_t>(std::max(1, g.m)), 0);

    // Transposes of the large effects, their work items and their diagonal.
    const int item_size = static_cast<int>(std::max<long long>(1024, g.C_op / (4LL * B) + 1));
    for (int j = 0; j < g.nb; ++j) {
        if (!g.large[static_cast<std::size_t>(j)]) continue;
        Transpose& t = g.tr[static_cast<std::size_t>(j)];
        const int mj = p.levels[g.remaining[static_cast<std::size_t>(j)]];
        const int* code = g.cc[static_cast<std::size_t>(j)];
        std::vector<int> cursor(static_cast<std::size_t>(op_parts) * static_cast<std::size_t>(mj), 0);
        team.run(op_parts, [&](int b) {
            int* cnt = cursor.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(mj);
            for (int s = g.op_bounds[static_cast<std::size_t>(b)]; s < g.op_bounds[static_cast<std::size_t>(b) + 1U]; ++s) {
                const int l = g.op_seg[static_cast<std::size_t>(s)];
                for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) ++cnt[code[c]];
            }
        });
        t.ptr.allocate(static_cast<std::size_t>(mj) + 1U);
        t.ptr[0] = 0;
        for (int lev = 0; lev < mj; ++lev) {
            int total = 0;
            for (int b = 0; b < op_parts; ++b)
                total += cursor[static_cast<std::size_t>(b) * static_cast<std::size_t>(mj) + static_cast<std::size_t>(lev)];
            t.ptr[static_cast<std::size_t>(lev) + 1U] = t.ptr[static_cast<std::size_t>(lev)] + total;
        }
        const int level_units = (mj + kVectorUnit - 1) / kVectorUnit;
        team.run(level_units, [&](int unit) {
            const int lv1 = std::min(mj, (unit + 1) * kVectorUnit);
            for (int lev = unit * kVectorUnit; lev < lv1; ++lev) {
                int position = t.ptr[static_cast<std::size_t>(lev)];
                for (int b = 0; b < op_parts; ++b) {
                    int& slot = cursor[static_cast<std::size_t>(b) * static_cast<std::size_t>(mj) + static_cast<std::size_t>(lev)];
                    const int count = slot;
                    slot = position;
                    position += count;
                }
            }
        });
        const std::size_t entries = static_cast<std::size_t>(t.ptr[static_cast<std::size_t>(mj)]);
        t.seg.allocate(entries);
        t.other.resize(static_cast<std::size_t>(g.nb));
        for (int jj = 0; jj < g.nb; ++jj)
            if (jj != j) t.other[static_cast<std::size_t>(jj)].allocate(entries);
        t.has_mult = g.mult != nullptr;
        if (t.has_mult) t.mult.allocate(entries);
        team.run(op_parts, [&](int b) {
            int* cur = cursor.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(mj);
            for (int s = g.op_bounds[static_cast<std::size_t>(b)]; s < g.op_bounds[static_cast<std::size_t>(b) + 1U]; ++s) {
                const int l = g.op_seg[static_cast<std::size_t>(s)];
                for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) {
                    const std::size_t pos = static_cast<std::size_t>(cur[code[c]]++);
                    t.seg[pos] = l;
                    for (int jj = 0; jj < g.nb; ++jj)
                        if (jj != j) t.other[static_cast<std::size_t>(jj)][pos] = g.cc[static_cast<std::size_t>(jj)][c];
                    if (t.has_mult) t.mult[pos] = g.mult[c];
                }
            }
        });
        std::vector<int> il, ib, ie;
        t.item_ptr.allocate(static_cast<std::size_t>(mj) + 1U);
        for (int lev = 0; lev < mj; ++lev) {
            t.item_ptr[static_cast<std::size_t>(lev)] = static_cast<int>(il.size());
            const int e1 = t.ptr[static_cast<std::size_t>(lev) + 1U];
            for (int a = t.ptr[static_cast<std::size_t>(lev)]; a < e1; a += item_size) {
                il.push_back(lev);
                ib.push_back(a);
                ie.push_back(std::min(e1, a + item_size));
            }
        }
        t.item_ptr[static_cast<std::size_t>(mj)] = static_cast<int>(il.size());
        t.items = static_cast<int>(il.size());
        t.item_level.allocate(il.size());
        t.item_begin.allocate(il.size());
        t.item_end.allocate(il.size());
        std::copy(il.begin(), il.end(), t.item_level.data());
        std::copy(ib.begin(), ib.end(), t.item_begin.data());
        std::copy(ie.begin(), ie.end(), t.item_end.data());
        // Blocks of items balanced by entries, the units of phase B: a level
        // list is one item (or a few), so one unit per item would cost one
        // dispatch per level of the effect.
        {
            std::vector<int> prefix(il.size() + 1U, 0);
            for (std::size_t it = 0; it < il.size(); ++it) prefix[it + 1U] = prefix[it] + (ie[it] - ib[it]);
            t.item_blocks = balanced_bounds(prefix.data(), t.items, 8 * B);
        }
        // S_jj = sum_l n_lj (n_l - n_lj) / n_l: n_lj aggregates the run of
        // entries of segment l inside the list of level j before squaring.
        const int off = g.offset[static_cast<std::size_t>(j)];
        team.run(level_units, [&](int unit) {
            const int lv1 = std::min(mj, (unit + 1) * kVectorUnit);
            for (int lev = unit * kVectorUnit; lev < lv1; ++lev) {
                double sum = 0.0;
                char nonzero = 0;
                int a = t.ptr[static_cast<std::size_t>(lev)];
                const int e1 = t.ptr[static_cast<std::size_t>(lev) + 1U];
                while (a < e1) {
                    const int l = t.seg[static_cast<std::size_t>(a)];
                    long long nlj = 0;
                    while (a < e1 && t.seg[static_cast<std::size_t>(a)] == l) {
                        nlj += t.has_mult ? t.mult[static_cast<std::size_t>(a)] : 1;
                        ++a;
                    }
                    const long long nl = rows_of(g, l);
                    if (nlj < nl) {
                        nonzero = 1;
                        sum += static_cast<double>(nlj) * static_cast<double>(nl - nlj) / static_cast<double>(nl);
                    }
                }
                g.diag[static_cast<std::size_t>(off + lev)] = sum;
                partial[static_cast<std::size_t>(off + lev)] = nonzero;
            }
        });
    }

    // Diagonal of the small effects, per block of operator segments, levels
    // aggregated inside each segment before squaring; blocks added in order.
    if (g.small_levels > 0) {
        const std::size_t width = static_cast<std::size_t>(g.small_levels);
        std::vector<double> block_sum(static_cast<std::size_t>(op_parts) * width, 0.0);
        std::vector<char> block_flag(static_cast<std::size_t>(op_parts) * width, 0);
        std::vector<long long> counter(static_cast<std::size_t>(op_parts) * width, 0);
        team.run(op_parts, [&](int b) {
            double* sum = block_sum.data() + static_cast<std::size_t>(b) * width;
            char* flag = block_flag.data() + static_cast<std::size_t>(b) * width;
            long long* cnt = counter.data() + static_cast<std::size_t>(b) * width;
            for (int s = g.op_bounds[static_cast<std::size_t>(b)]; s < g.op_bounds[static_cast<std::size_t>(b) + 1U]; ++s) {
                const int l = g.op_seg[static_cast<std::size_t>(s)];
                const long long nl = rows_of(g, l);
                for (std::size_t q = 0; q < g.small_list.size(); ++q) {
                    const int j = g.small_list[q];
                    const int base = g.small_offset[q];
                    for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c)
                        cnt[base + g.cc[static_cast<std::size_t>(j)][c]] += g.mult != nullptr ? g.mult[c] : 1;
                    for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) {
                        const int idx = base + g.cc[static_cast<std::size_t>(j)][c];
                        const long long nlj = cnt[idx];
                        if (nlj == 0) continue;  // already taken from this segment
                        if (nlj < nl) {
                            flag[idx] = 1;
                            sum[idx] += static_cast<double>(nlj) * static_cast<double>(nl - nlj) / static_cast<double>(nl);
                        }
                        cnt[idx] = 0;
                    }
                }
            }
        });
        for (int idx = 0; idx < g.small_levels; ++idx) {
            double sum = 0.0;
            char nonzero = 0;
            for (int b = 0; b < op_parts; ++b) {
                sum += block_sum[static_cast<std::size_t>(b) * width + static_cast<std::size_t>(idx)];
                nonzero = static_cast<char>(nonzero | block_flag[static_cast<std::size_t>(b) * width + static_cast<std::size_t>(idx)]);
            }
            const int target = g.small_target[static_cast<std::size_t>(idx)];
            g.diag[static_cast<std::size_t>(target)] = sum;
            partial[static_cast<std::size_t>(target)] = nonzero;
        }
    }
    // Exact integer test: S_jj = 0 iff every segment touching j lies in j.
    g.minv.allocate(static_cast<std::size_t>(g.m));
    for (int i = 0; i < g.m; ++i)
        g.minv[static_cast<std::size_t>(i)] =
            partial[static_cast<std::size_t>(i)] != 0 ? 1.0 / g.diag[static_cast<std::size_t>(i)] : 0.0;
    return Decline::None;
}

// ---------------------------------------------------------------------------
// Solver for one group of W lanes.

struct Work {
    Buffer<double> g;        // m * W
    Buffer<double> arena;    // W * (4m + L): x, r, p, q, T; private blocks of the RHS pass
    Buffer<double> vbar;     // L * W: level means of v, rounded (hi)
    Buffer<double> corr;     // L * W: mean of (v - hi) per level, then that minus the mean of D2 x
    Buffer<double> priv;     // op_parts * small_levels * W
    std::vector<Buffer<double>> partials;  // per large effect: items * W
    Buffer<double> units;    // vector units * 8 * W
    Buffer<double> stats;    // row parts * kStats * W
    int prefetch_cells = 0;  // Config::prefetch_cells of the solve (0: the loop of 2.34.1)
    long long hinted_applies = 0;  // applies whose segment pass went through hinted_segment_part
};

template <int W>
inline void gather_cell(const Geometry& g, const double* in, int c, double* u) {
    for (int w = 0; w < W; ++w) u[w] = 0.0;
    for (int j = 0; j < g.nb; ++j) {
        const double* src = in + static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)] +
                                                          g.cc[static_cast<std::size_t>(j)][c]) * W;
        for (int w = 0; w < W; ++w) u[w] += src[w];
    }
}

#if RPCG_HINTED_SEGMENT_KERNEL
// A row of W lanes as one value: a vector type of GCC, unaligned and allowed
// to alias double.
template <int W>
struct LaneRow;
template <>
struct LaneRow<1> {
    typedef double type __attribute__((may_alias));
};
template <>
struct LaneRow<2> {
    typedef double type __attribute__((vector_size(16), aligned(8), may_alias));
};
template <>
struct LaneRow<4> {
    typedef double type __attribute__((vector_size(32), aligned(8), may_alias));
};
template <>
struct LaneRow<8> {
    typedef double type __attribute__((vector_size(64), aligned(8), may_alias));
};

// One part of the operator's segment pass for the shapes of
// Config::prefetch_cells: multiplicity one everywhere and at least one effect
// whose rows of `in` do not fit in cache. The sums of the loop in
// apply_operator as GCC compiles it, lane by lane and in the same order, here
// kept as written: the same bits (tests/reduced_schur_pcg_unit.cpp). Each row is
// one vector addition (left to the vectoriser, the second gather of that loop
// came out as W scalar additions per row); the rows of the effects in `hinted`
// are requested `ahead` cells before they are gathered; and the cell loop
// reads its pointers from this frame. Cells are visited in increasing order,
// so the hint runs across segments and parts; a segment uses it when all its
// cells have a cell `ahead` positions on, which leaves out only the last cells
// of the codes.
template <int W>
RPCG_EXACT_FP inline void hinted_segment_part(const Geometry& g, const double* in, double* T, double* mine, int b,
                                              const int* hinted, int hints, int ahead) {
    RPCG_EXACT_FP_BODY
    typedef typename LaneRow<W>::type Row;
    const int nb = g.nb;
    const double* row[kMaxRemaining];
    const int* code[kMaxRemaining];
    for (int j = 0; j < nb; ++j) {
        row[j] = in + static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)]) * W;
        code[j] = g.cc[static_cast<std::size_t>(j)];
    }
    const double* ahead_row[kMaxRemaining];
    const int* ahead_code[kMaxRemaining];
    for (int q = 0; q < hints; ++q) {
        ahead_row[q] = row[hinted[q]];
        ahead_code[q] = code[hinted[q]] + ahead;
    }
    const int smalls = mine != nullptr ? static_cast<int>(g.small_list.size()) : 0;
    double* small_row[kMaxRemaining];
    const int* small_code[kMaxRemaining];
    for (int q = 0; q < smalls; ++q) {
        small_row[q] = mine + static_cast<std::size_t>(g.small_offset[static_cast<std::size_t>(q)]) * W;
        small_code[q] = code[g.small_list[static_cast<std::size_t>(q)]];
    }
    const long long ahead_end = g.C - ahead;
    for (int s = g.op_bounds[static_cast<std::size_t>(b)]; s < g.op_bounds[static_cast<std::size_t>(b) + 1U]; ++s) {
        const int l = g.op_seg[static_cast<std::size_t>(s)];
        const int c0 = g.cell_ptr[l], c1 = g.cell_ptr[l + 1];
        const bool look = c1 <= ahead_end;
        Row t = {};
        for (int c = c0; c < c1; ++c) {
            if (look)
                for (int q = 0; q < hints; ++q)
                    __builtin_prefetch(ahead_row[q] + static_cast<std::size_t>(ahead_code[q][c]) * W, 0, 3);
            Row u = {};
            for (int j = 0; j < nb; ++j)
                u += *reinterpret_cast<const Row*>(row[j] + static_cast<std::size_t>(code[j][c]) * W);
            t += u;
        }
        t *= g.seg_inv[static_cast<std::size_t>(l)];
        std::memcpy(T + static_cast<std::size_t>(l) * W, &t, sizeof(Row));
        if (smalls == 0) continue;
        for (int c = c0; c < c1; ++c) {
            Row u = {};
            for (int j = 0; j < nb; ++j)
                u += *reinterpret_cast<const Row*>(row[j] + static_cast<std::size_t>(code[j][c]) * W);
            const Row d = u - t;
            for (int q = 0; q < smalls; ++q)
                *reinterpret_cast<Row*>(small_row[q] + static_cast<std::size_t>(small_code[q][c]) * W) += d;
        }
    }
}
#endif

// out = S in. Phase A by segment (means of D2 in, private blocks of the small
// effects); phase B by level of each large effect, with an owner per item.
template <int W>
inline void apply_operator(const Geometry& g, const Team& team, const double* in, double* out,
                           double* T, Work& work) {
    const int op_parts = static_cast<int>(g.op_bounds.size()) - 1;
    const std::size_t small_width = static_cast<std::size_t>(g.small_levels) * W;
    double* priv = work.priv.data();
#if RPCG_HINTED_SEGMENT_KERNEL
    // The shapes of Config::prefetch_cells go through hinted_segment_part.
    const int ahead = work.prefetch_cells;
    int hinted[kMaxRemaining];
    int hints = 0;
    if (ahead > 0 && ahead < g.C && g.mult == nullptr && g.nb <= kMaxRemaining)
        for (int j = 0; j < g.nb; ++j) {
            const std::size_t levels = static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j) + 1U] -
                                                                g.offset[static_cast<std::size_t>(j)]);
            if (levels * W * sizeof(double) >= kPrefetchMinBytes) hinted[hints++] = j;
        }
    if (hints > 0) {
        ++work.hinted_applies;
        team.run(op_parts, [&](int b) {
            double* mine = small_width > 0 ? priv + static_cast<std::size_t>(b) * small_width : nullptr;
            if (mine != nullptr) std::fill(mine, mine + small_width, 0.0);
            hinted_segment_part<W>(g, in, T, mine, b, hinted, hints, ahead);
        });
    } else
#endif
    team.run(op_parts, [&](int b) {
        double* mine = small_width > 0 ? priv + static_cast<std::size_t>(b) * small_width : nullptr;
        if (mine != nullptr) std::fill(mine, mine + small_width, 0.0);
        for (int s = g.op_bounds[static_cast<std::size_t>(b)]; s < g.op_bounds[static_cast<std::size_t>(b) + 1U]; ++s) {
            const int l = g.op_seg[static_cast<std::size_t>(s)];
            const int c0 = g.cell_ptr[l], c1 = g.cell_ptr[l + 1];
            double t[W];
            for (int w = 0; w < W; ++w) t[w] = 0.0;
            for (int c = c0; c < c1; ++c) {
                double u[W];
                gather_cell<W>(g, in, c, u);
                const double mc = g.mult != nullptr ? static_cast<double>(g.mult[c]) : 1.0;
                for (int w = 0; w < W; ++w) t[w] += mc * u[w];
            }
            const double inv = g.seg_inv[static_cast<std::size_t>(l)];
            double* Tl = T + static_cast<std::size_t>(l) * W;
            for (int w = 0; w < W; ++w) {
                t[w] *= inv;
                Tl[w] = t[w];
            }
            if (mine == nullptr) continue;
            for (int c = c0; c < c1; ++c) {
                double u[W];
                gather_cell<W>(g, in, c, u);
                const double mc = g.mult != nullptr ? static_cast<double>(g.mult[c]) : 1.0;
                for (std::size_t q = 0; q < g.small_list.size(); ++q) {
                    const int j = g.small_list[q];
                    double* dst = mine + static_cast<std::size_t>(g.small_offset[q] + g.cc[static_cast<std::size_t>(j)][c]) * W;
                    for (int w = 0; w < W; ++w) dst[w] += mc * (u[w] - t[w]);
                }
            }
        }
    });
    if (small_width > 0) {
        const int units = (g.small_levels + kVectorUnit - 1) / kVectorUnit;
        team.run(units, [&](int unit) {
            const int i1 = std::min(g.small_levels, (unit + 1) * kVectorUnit);
            for (int idx = unit * kVectorUnit; idx < i1; ++idx) {
                double acc[W];
                for (int w = 0; w < W; ++w) acc[w] = 0.0;
                for (int b = 0; b < op_parts; ++b) {
                    const double* src = priv + static_cast<std::size_t>(b) * small_width + static_cast<std::size_t>(idx) * W;
                    for (int w = 0; w < W; ++w) acc[w] += src[w];
                }
                double* dst = out + static_cast<std::size_t>(g.small_target[static_cast<std::size_t>(idx)]) * W;
                for (int w = 0; w < W; ++w) dst[w] = acc[w];
            }
        });
    }
    for (int j = 0; j < g.nb; ++j) {
        if (!g.large[static_cast<std::size_t>(j)]) continue;
        const Transpose& t = g.tr[static_cast<std::size_t>(j)];
        double* part = work.partials[static_cast<std::size_t>(j)].data();
        const int off = g.offset[static_cast<std::size_t>(j)];
        const int item_blocks = static_cast<int>(t.item_blocks.size()) - 1;
        team.run(item_blocks, [&](int blk) {
            for (int it = t.item_blocks[static_cast<std::size_t>(blk)];
                 it < t.item_blocks[static_cast<std::size_t>(blk) + 1U]; ++it) {
                const int lev = t.item_level[static_cast<std::size_t>(it)];
                const int e0 = t.item_begin[static_cast<std::size_t>(it)];
                const int e1 = t.item_end[static_cast<std::size_t>(it)];
                double acc[W];
                for (int w = 0; w < W; ++w) acc[w] = 0.0;
                for (int a = e0; a < e1; ++a) {
#if defined(__GNUC__)
                    if (a + 16 < e1)
                        __builtin_prefetch(T + static_cast<std::size_t>(t.seg[static_cast<std::size_t>(a) + 16U]) * W, 0, 0);
#endif
                    double u[W];
                    for (int w = 0; w < W; ++w) u[w] = 0.0;
                    for (int jj = 0; jj < g.nb; ++jj) {
                        const int code = jj == j ? lev : t.other[static_cast<std::size_t>(jj)][static_cast<std::size_t>(a)];
                        const double* src = in + static_cast<std::size_t>(g.offset[static_cast<std::size_t>(jj)] + code) * W;
                        for (int w = 0; w < W; ++w) u[w] += src[w];
                    }
                    const double* Tl = T + static_cast<std::size_t>(t.seg[static_cast<std::size_t>(a)]) * W;
                    const double mc = t.has_mult ? static_cast<double>(t.mult[static_cast<std::size_t>(a)]) : 1.0;
                    for (int w = 0; w < W; ++w) acc[w] += mc * (u[w] - Tl[w]);
                }
                double* dst = part + static_cast<std::size_t>(it) * W;
                for (int w = 0; w < W; ++w) dst[w] = acc[w];
            }
        });
        const int mj = g.offset[static_cast<std::size_t>(j) + 1U] - off;
        const int units = (mj + kVectorUnit - 1) / kVectorUnit;
        team.run(units, [&](int unit) {
            const int lv1 = std::min(mj, (unit + 1) * kVectorUnit);
            for (int lev = unit * kVectorUnit; lev < lv1; ++lev) {
                double acc[W];
                for (int w = 0; w < W; ++w) acc[w] = 0.0;
                for (int it = t.item_ptr[static_cast<std::size_t>(lev)]; it < t.item_ptr[static_cast<std::size_t>(lev) + 1U]; ++it) {
                    const double* src = part + static_cast<std::size_t>(it) * W;
                    for (int w = 0; w < W; ++w) acc[w] += src[w];
                }
                double* dst = out + static_cast<std::size_t>(off + lev) * W;
                for (int w = 0; w < W; ++w) dst[w] = acc[w];
            }
        });
    }
}

// Per-lane sums over the reduced vector, in fixed units added in order. The
// body updates level i and adds its contributions to acc[s * W + w].
template <int W, int S, class Body>
inline void lane_sums(int m, const Team& team, Work& work, double* total, Body&& body) {
    const int units = std::max(1, (m + kVectorUnit - 1) / kVectorUnit);
    double* buffer = work.units.data();
    team.run(units, [&](int unit) {
        double acc[S * W];
        for (int s = 0; s < S * W; ++s) acc[s] = 0.0;
        const int i1 = std::min(m, (unit + 1) * kVectorUnit);
        for (int i = unit * kVectorUnit; i < i1; ++i) body(i, acc);
        std::memcpy(buffer + static_cast<std::size_t>(unit) * S * W, acc, sizeof(acc));
    });
    for (int s = 0; s < S * W; ++s) total[s] = 0.0;
    for (int unit = 0; unit < units; ++unit)
        for (int s = 0; s < S * W; ++s) total[s] += buffer[static_cast<std::size_t>(unit) * S * W + static_cast<std::size_t>(s)];
}

// One segment of the RHS pass: the level mean as hi + delta (hi the rounded
// mean, delta the mean of v - hi), the deviations (v - hi) - delta scattered
// into the private copy `acc` of g, and the per-column sums in `st`.
template <int W>
RPCG_EXACT_FP inline void rhs_segment(const Problem& p, const Geometry& g, int l, const double* const* col,
                                      int width, const double* origin, double* mean, double* delta,
                                      double* acc, double* st) {
    RPCG_EXACT_FP_BODY
    const int r0 = g.row_ptr[static_cast<std::size_t>(l)];
    const int r1 = g.row_ptr[static_cast<std::size_t>(l) + 1U];
    for (int w = 0; w < W; ++w) {
        mean[w] = 0.0;
        delta[w] = 0.0;
    }
    if (r0 == r1) return;
    const int* perm = g.sorted ? nullptr : g.perm.data();
    for (int rr = r0; rr < r1; ++rr) {
        const int row = perm != nullptr ? perm[rr] : rr;
        for (int w = 0; w < width; ++w) mean[w] += col[w][row];
    }
    const double inv = g.seg_inv[static_cast<std::size_t>(l)];
    for (int w = 0; w < width; ++w) mean[w] *= inv;
    for (int rr = r0; rr < r1; ++rr) {
        const int row = perm != nullptr ? perm[rr] : rr;
        for (int w = 0; w < width; ++w) delta[w] += col[w][row] - mean[w];
    }
    for (int w = 0; w < width; ++w) delta[w] *= inv;
    for (int rr = r0; rr < r1; ++rr) {
        const int row = perm != nullptr ? perm[rr] : rr;
        double d[W];
        for (int w = 0; w < width; ++w) {
            const double v = col[w][row];
            d[w] = (v - mean[w]) - delta[w];
            const double shifted = v - origin[w];
            st[0 * W + w] += v * v;
            st[1 * W + w] += d[w] * d[w];
            st[2 * W + w] += v;
            st[3 * W + w] += shifted;
            st[4 * W + w] += shifted * shifted;
        }
        for (int j = 0; j < g.nb; ++j) {
            const int code = p.codes[g.remaining[static_cast<std::size_t>(j)]][row];
            double* dst = acc + static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)] + code) * W;
            for (int w = 0; w < width; ++w) dst[w] += d[w];
        }
    }
}

// Rows [i0, i1) of v - D1 a1 - D2 x as (v - hi) - (small + D2 x), small =
// delta - mean(D2 x) of the row's level of e. Writes `res` (when given) or
// adds the squares of the masked lanes to `acc`.
template <int W>
RPCG_EXACT_FP inline void residual_rows(const Problem& p, const Geometry& g, int i0, int i1, const double* x,
                                        const double* hi_table, const double* small_table,
                                        const double* const* col, double* const* res, int width,
                                        const bool* mask, double* acc) {
    RPCG_EXACT_FP_BODY
    const int* key = p.codes[g.e];
    for (int i = i0; i < i1; ++i) {
        double s[W];
        for (int w = 0; w < W; ++w) s[w] = 0.0;
        for (int j = 0; j < g.nb; ++j) {
            const double* src = x + static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)] +
                                                             p.codes[g.remaining[static_cast<std::size_t>(j)]][i]) * W;
            for (int w = 0; w < W; ++w) s[w] += src[w];
        }
        const double* hi = hi_table + static_cast<std::size_t>(key[i]) * W;
        const double* small = small_table + static_cast<std::size_t>(key[i]) * W;
        if (res != nullptr) {
            for (int w = 0; w < width; ++w) res[w][i] = (col[w][i] - hi[w]) - (small[w] + s[w]);
        } else {
            for (int w = 0; w < width; ++w) {
                if (!mask[w]) continue;
                const double d = (col[w][i] - hi[w]) - (small[w] + s[w]);
                acc[w] += d * d;
            }
        }
    }
}

struct Lane {
    bool active = false;   // still iterating
    Stop stop = Stop::None;
    int column = -1;
    int iterations = 0;
    int first_tau_pass = -1;
    int formed_checks = 0;
    double v2 = 0.0, vnorm = 0.0, w2 = 0.0, wnorm = 0.0, denom = 0.0;
    double rho = 0.0, alpha_prev = 0.0, beta = 0.0, a2 = 0.0, s2 = 0.0;
    bool formed = false;   // the energy identity is at its rounding floor: s is measured on the formed residual
    double s_formed = 0.0;
    double tau = kInf, chat = kInf;
    double best = kInf, best_prev = kInf;
    bool have_prev = false;
};

// ||v - D1 a1(x) - D2 x||^2 per masked lane, formed from the data: the small
// part of a1(x), delta - mean(D2 x), into T for every segment (T is free
// between operator applications), then one pass over the rows of the masked
// columns, in fixed blocks, as (v - hi) - (that + D2 x).
template <int W>
inline void formed_residual_sq(const Problem& p, const Geometry& g, const Team& team, Work& work,
                               const double* x, const double* vbar, const double* corr, double* T,
                               const double* const* col, const bool* mask, double* out) {
    const int all_parts = static_cast<int>(g.all_bounds.size()) - 1;
    team.run(all_parts, [&](int b) {
        for (int l = g.all_bounds[static_cast<std::size_t>(b)]; l < g.all_bounds[static_cast<std::size_t>(b) + 1U]; ++l) {
            if (rows_of(g, l) == 0) continue;
            double t[W];
            for (int w = 0; w < W; ++w) t[w] = 0.0;
            for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) {
                double u[W];
                gather_cell<W>(g, x, c, u);
                const double mc = g.mult != nullptr ? static_cast<double>(g.mult[c]) : 1.0;
                for (int w = 0; w < W; ++w) t[w] += mc * u[w];
            }
            const double inv = g.seg_inv[static_cast<std::size_t>(l)];
            const double* delta = corr + static_cast<std::size_t>(l) * W;
            double* small = T + static_cast<std::size_t>(l) * W;
            for (int w = 0; w < W; ++w) small[w] = delta[w] - t[w] * inv;
        }
    });
    const int n = p.n;
    const int parts = std::max(1, std::min(static_cast<int>(g.row_bounds.size()) - 1, n));
    double* partial = work.stats.data();
    team.run(parts, [&](int b) {
        double acc[W];
        for (int w = 0; w < W; ++w) acc[w] = 0.0;
        const int i0 = static_cast<int>(part_begin(n, b, parts));
        const int i1 = static_cast<int>(part_begin(n, b + 1, parts));
        residual_rows<W>(p, g, i0, i1, x, vbar, T, col, nullptr, W, mask, acc);
        std::memcpy(partial + static_cast<std::size_t>(b) * W, acc, sizeof(acc));
    });
    for (int w = 0; w < W; ++w) out[w] = 0.0;
    for (int b = 0; b < parts; ++b)
        for (int w = 0; w < W; ++w) out[w] += partial[static_cast<std::size_t>(b) * W + static_cast<std::size_t>(w)];
}

template <int W>
inline Decline run_group(const Problem& p, const Config& cfg, const Geometry& g, const Team& team,
                         Work& work, int col0, int width, Report& rep) {
    const int m = g.m, L = g.L, n = p.n;
    double* gv = work.g.data();
    double* x = work.arena.data();
    double* r = x + static_cast<std::size_t>(m) * W;
    double* pv = r + static_cast<std::size_t>(m) * W;
    double* q = pv + static_cast<std::size_t>(m) * W;
    double* T = q + static_cast<std::size_t>(m) * W;
    double* vbar = work.vbar.data();
    double* corr = work.corr.data();
    const double* minv = g.minv.data();
    const double* diag = g.diag.data();
    const double* col[W];
    double* res[W];
    double origin[W];
    for (int w = 0; w < W; ++w) {
        col[w] = w < width ? p.columns[col0 + w] : nullptr;
        res[w] = w < width ? p.residuals[col0 + w] : nullptr;
        origin[w] = w < width ? col[w][0] : 0.0;
    }

    // RHS pass, by segment: the level means, the deviations scattered into
    // private copies of g (in the arena, before x, r, p, q and T use it), and
    // the per-column sums of the stopping rule and of the certificate scale.
    auto t0 = std::chrono::steady_clock::now();
    const int rhs_parts = std::max(1, std::min(static_cast<int>(g.row_bounds.size()) - 1,
                                               std::max(1, (4 * m + L) / std::max(1, m))));
    const std::vector<int> rhs_bounds = balanced_bounds(g.row_ptr.data(), L, rhs_parts);
    const int rparts = static_cast<int>(rhs_bounds.size()) - 1;
    const std::size_t mw = static_cast<std::size_t>(m) * W;
    double* stats = work.stats.data();
    team.run(rparts, [&](int b) {
        double* acc = x + static_cast<std::size_t>(b) * mw;
        std::fill(acc, acc + mw, 0.0);
        double st[kStats * W];
        for (int s = 0; s < kStats * W; ++s) st[s] = 0.0;
        // The level mean as hi + delta, so the deviations sum to zero at the
        // precision of the deviations, not of the mean (rhs_segment).
        for (int l = rhs_bounds[static_cast<std::size_t>(b)]; l < rhs_bounds[static_cast<std::size_t>(b) + 1U]; ++l)
            rhs_segment<W>(p, g, l, col, width, origin, vbar + static_cast<std::size_t>(l) * W,
                           corr + static_cast<std::size_t>(l) * W, acc, st);
        std::memcpy(stats + static_cast<std::size_t>(b) * kStats * W, st, sizeof(st));
    });
    {
        const int units = std::max(1, (m + kVectorUnit - 1) / kVectorUnit);
        team.run(units, [&](int unit) {
            const std::size_t i0 = static_cast<std::size_t>(unit) * kVectorUnit * W;
            const std::size_t i1 = std::min(mw, static_cast<std::size_t>(unit + 1) * kVectorUnit * W);
            for (std::size_t i = i0; i < i1; ++i) {
                double sum = 0.0;
                for (int b = 0; b < rparts; ++b) sum += x[static_cast<std::size_t>(b) * mw + i];
                gv[i] = sum;
            }
        });
    }
    Lane lane[W];
    int k = 0;
    bool iterating = false;
    std::chrono::steady_clock::time_point t_iterate{};
    // The reports of this group's columns, on every exit: a declined attempt
    // still shows how far each column got (iterations, tau, chat).
    auto report_lanes = [&]() {
        for (int w = 0; w < width; ++w) {
            const Lane& ln = lane[w];
            ColumnReport& out = rep.columns[static_cast<std::size_t>(col0 + w)];
            out.stop = ln.active ? Stop::None : ln.stop;
            out.iterations = ln.active ? k : ln.iterations;
            out.first_tau_pass = ln.first_tau_pass;
            out.formed_checks = ln.formed_checks;
            const bool trivial = ln.stop == Stop::Zero || ln.stop == Stop::Span;
            out.tau = trivial ? 0.0 : ln.tau;
            out.chat = trivial ? 0.0 : ln.chat;
            const double s = trivial ? ln.wnorm : (ln.formed ? ln.s_formed : std::sqrt(std::max(0.0, ln.s2)));
            out.s_rel = ln.vnorm > 0.0 ? s / ln.vnorm : 0.0;
            rep.iterations = std::max(rep.iterations, out.iterations);
        }
    };
    auto finish = [&](Decline decline) {
        if (iterating) rep.iterate_seconds += seconds_since(t_iterate);
        report_lanes();
        return decline;
    };
    {
        long double tot[kStats * W];
        for (int s = 0; s < kStats * W; ++s) tot[s] = 0.0L;
        for (int b = 0; b < rparts; ++b)
            for (int s = 0; s < kStats * W; ++s)
                tot[s] += stats[static_cast<std::size_t>(b) * kStats * W + static_cast<std::size_t>(s)];
        // Denominator of the certificate (certify_absorption_result,
        // fe_absorption.cpp:24812-24858): sqrt(K n) times the column norm, the
        // centred one in the dominant-origin case of reghdfe-comparable.
        const long double op_norm = std::sqrt(static_cast<long double>(p.k) * static_cast<long double>(n));
        for (int w = 0; w < width; ++w) {
            Lane& ln = lane[w];
            ln.column = col0 + w;
            const long double raw = tot[0 * W + w];
            const long double sum = tot[2 * W + w];
            long double norm_sq = raw;
            if (cfg.dominant_origin && n > 0 && sum * (sum / n) > (1.0L - 1.0L / 1024.0L) * raw) {
                const long double s1 = tot[3 * W + w];
                const long double centered = tot[4 * W + w] - s1 * (s1 / n);
                if (centered > 0.0L && centered < raw / 1024.0L) norm_sq = centered;
            }
            ln.v2 = static_cast<double>(raw);
            ln.vnorm = std::sqrt(ln.v2);
            ln.w2 = static_cast<double>(tot[1 * W + w]);
            ln.wnorm = std::sqrt(ln.w2);
            ln.denom = static_cast<double>(op_norm * std::sqrt(norm_sq));
            // A non-finite input or sum is never read as a zero or a span.
            if (!finite(ln.v2) || !finite(ln.w2) || !finite(static_cast<double>(sum)) ||
                !finite(static_cast<double>(tot[4 * W + w]))) {
                rep.decline_column = ln.column;
                return finish(Decline::NonFinite);
            }
            if (!(raw > 0.0L)) {
                ln.stop = Stop::Zero;
            } else if (ln.wnorm <= cfg.btol * ln.vnorm) {
                ln.stop = Stop::Span;
            } else {
                ln.active = true;
            }
        }
    }
    rep.rhs_seconds += seconds_since(t0);

    // Lockstep PCG from x = 0: r = g, p = M+ r; the lanes never mix.
    t_iterate = std::chrono::steady_clock::now();
    iterating = true;
    bool live[W];
    for (int w = 0; w < W; ++w) live[w] = lane[w].active;
    std::fill(x, x + mw, 0.0);
    std::fill(q, q + mw, 0.0);
    double tot[8 * W];
    lane_sums<W, 2>(m, team, work, tot, [&](int i, double* acc) {
        const std::size_t base = static_cast<std::size_t>(i) * W;
        for (int w = 0; w < W; ++w) {
            const double rv = live[w] ? gv[base + w] : 0.0;
            r[base + w] = rv;
            pv[base + w] = minv[i] * rv;
            acc[0 * W + w] += minv[i] * rv * rv;
            acc[1 * W + w] += rv * rv;
        }
    });
    for (int w = 0; w < width; ++w) {
        Lane& ln = lane[w];
        if (!ln.active) continue;
        ln.rho = tot[0 * W + w];
        ln.s2 = ln.w2;
        if (!finite(ln.rho) || !finite(tot[1 * W + w])) {
            rep.decline_column = ln.column;
            return finish(Decline::NonFinite);
        }
        ln.chat = std::sqrt(tot[1 * W + w]) / ln.denom;
        if (!(ln.rho > 0.0)) {
            // tau = 0: no preconditioned residual left (whatever remains lies
            // on null levels, which no x can change). The certificate judges.
            ln.tau = 0.0;
            ln.first_tau_pass = 0;
            if (cfg.chat_stop > 0.0) {
                // D10: r = g is formed exactly here, so chat needs no other
                // confirmation; above the target the column stays active and
                // the first step declines (Breakdown: nothing to reduce).
                if (ln.chat <= cfg.chat_stop) {
                    ln.stop = Stop::Chat;
                    ln.active = false;
                }
            } else {
                ln.stop = Stop::Test2;
                ln.active = false;
            }
        }
    }

    // Stop measures of one lane. The compatible-system exit is LSQR's test1
    // in the coordinates of the preconditioned reduced operator: PCG with
    // M = diag(S) is CG on B^'B^, B^ = (I - P1) D2 M^(-1/2), in y = M^(1/2) x;
    // for min ||w - B^ y|| the test reads ||w - B^ y|| <= btol ||w|| +
    // atol ||B^|| ||y||, with ||B^|| estimated by A^ and ||y|| = sqrt(x'Mx)
    // (MLSMR applies the same rule to its own operator, fe_absorption.cpp:
    // 22319-22324). ||w|| <= ||v||, so it is never looser than the original norm.
    const int max_iter = std::max(1, cfg.max_iter);
    auto evaluate = [&](Lane& ln, double rho, double rr, double xmx) {
        const double s = ln.formed ? ln.s_formed : (ln.s2 > 0.0 ? std::sqrt(ln.s2) : 0.0);
        const double a = std::sqrt(std::max(0.0, ln.a2));
        if (!(rho > 0.0)) ln.tau = 0.0;
        else ln.tau = (s > 0.0 && a > 0.0) ? std::sqrt(rho) / (a * s) : kInf;
        ln.chat = std::sqrt(std::max(0.0, rr)) / ln.denom;
        const bool test2 = ln.tau <= cfg.atol;
        const bool test1 = (ln.formed || ln.s2 > 0.0) &&
                           s <= cfg.btol * ln.wnorm + cfg.atol * a * std::sqrt(std::max(0.0, xmx));
        if (test2 && ln.first_tau_pass < 0) ln.first_tau_pass = k;
        const bool companion_ok = !cfg.companion || ln.chat <= cfg.trigger;
        if (cfg.chat_stop > 0.0) {
            // D10: the target replaces test 2; test 1 stays.
            const bool test_chat = ln.chat <= cfg.chat_stop;
            ln.stop = companion_ok ? (test_chat ? Stop::Chat : (test1 ? Stop::Test1 : Stop::None)) : Stop::None;
        } else {
            ln.stop = companion_ok ? (test2 ? Stop::Test2 : (test1 ? Stop::Test1 : Stop::None)) : Stop::None;
        }
        return ln.stop != Stop::None;
    };
    auto any_active = [&]() {
        for (int w = 0; w < W; ++w)
            if (lane[w].active) return true;
        return false;
    };
    while (any_active()) {
        if (k >= max_iter) {
            for (int w = 0; w < W; ++w)
                if (lane[w].active) {
                    rep.decline_column = lane[w].column;
                    break;
                }
            return finish(Decline::Ceiling);
        }
        apply_operator<W>(g, team, pv, q, T, work);
        lane_sums<W, 1>(m, team, work, tot, [&](int i, double* acc) {
            const std::size_t base = static_cast<std::size_t>(i) * W;
            for (int w = 0; w < W; ++w) acc[w] += pv[base + w] * q[base + w];
        });
        double alpha[W];
        double rho_old[W];
        for (int w = 0; w < W; ++w) {
            alpha[w] = 0.0;
            rho_old[w] = lane[w].rho;
            Lane& ln = lane[w];
            if (!ln.active) continue;
            // Curvature is checked before the division.
            const double sigma = cfg.test_zero_curvature_at == k + 1 ? 0.0 : tot[w];
            if (!finite(sigma)) {
                rep.decline_column = ln.column;
                return finish(Decline::NonFinite);
            }
            if (!(sigma > 0.0)) {
                rep.decline_column = ln.column;
                return finish(Decline::Breakdown);
            }
            alpha[w] = ln.rho / sigma;
        }
        lane_sums<W, 3>(m, team, work, tot, [&](int i, double* acc) {
            const std::size_t base = static_cast<std::size_t>(i) * W;
            for (int w = 0; w < W; ++w) {
                const double xv = x[base + w] + alpha[w] * pv[base + w];
                const double rv = r[base + w] - alpha[w] * q[base + w];
                x[base + w] = xv;
                r[base + w] = rv;
                acc[0 * W + w] += minv[i] * rv * rv;
                acc[1 * W + w] += rv * rv;
                acc[2 * W + w] += diag[i] * xv * xv;
            }
        });
        ++k;
        bool candidate[W];
        bool enter_formed[W];
        bool any_candidate = false, any_enter = false;
        for (int w = 0; w < W; ++w) {
            candidate[w] = false;
            enter_formed[w] = false;
            Lane& ln = lane[w];
            if (!ln.active) continue;
            const double rho_new = tot[0 * W + w];
            // Trace of the CG Lanczos matrix T_k: 1/alpha_i + beta_(i-1)/alpha_(i-1).
            ln.a2 += 1.0 / alpha[w] + (k > 1 ? ln.beta / ln.alpha_prev : 0.0);
            if (!ln.formed) ln.s2 -= alpha[w] * rho_old[w];  // energy identity
            if (!finite(rho_new) || !finite(ln.s2) || !finite(ln.a2) || !finite(alpha[w])) {
                rep.decline_column = ln.column;
                return finish(Decline::NonFinite);
            }
            ln.beta = rho_new / rho_old[w];
            ln.alpha_prev = alpha[w];
            ln.rho = rho_new;
            if (!ln.formed && !(ln.s2 > cfg.floor_factor * kEps * ln.w2)) {
                enter_formed[w] = true;
                any_enter = true;
            }
            double rr = tot[1 * W + w];
            if (cfg.test_recurrence_chat_at == k) {
                const double forced = cfg.test_recurrence_chat_value * ln.denom;
                rr = forced * forced;
            }
            candidate[w] = evaluate(ln, rho_new, rr, tot[2 * W + w]);
            any_candidate = any_candidate || candidate[w];
        }
        const bool periodic = k % std::max(1, cfg.refresh_period) == 0;
        const bool refresh = any_candidate || periodic || any_enter;
        if (refresh) {
            // Residual replacement: r := g - S x. The search direction is kept
            // (no restart), so the recurrence and the Lanczos trace continue.
            apply_operator<W>(g, team, x, q, T, work);
            bool act[W];
            for (int w = 0; w < W; ++w) act[w] = lane[w].active;
            double fresh[5 * W];
            lane_sums<W, 5>(m, team, work, fresh, [&](int i, double* acc) {
                const std::size_t base = static_cast<std::size_t>(i) * W;
                for (int w = 0; w < W; ++w) {
                    if (!act[w]) continue;
                    const double rv = gv[base + w] - q[base + w];
                    r[base + w] = rv;
                    const double xv = x[base + w];
                    acc[0 * W + w] += minv[i] * rv * rv;
                    acc[1 * W + w] += rv * rv;
                    acc[2 * W + w] += diag[i] * xv * xv;
                    acc[3 * W + w] += xv * gv[base + w];
                    acc[4 * W + w] += xv * rv;
                }
            });
            bool formed_mask[W];
            bool any_formed = false;
            for (int w = 0; w < W; ++w) {
                formed_mask[w] = false;
                Lane& ln = lane[w];
                if (!ln.active) continue;
                const double rho_new = fresh[0 * W + w];
                if (!finite(rho_new)) {
                    rep.decline_column = ln.column;
                    return finish(Decline::NonFinite);
                }
                ln.beta = rho_new / rho_old[w];
                ln.rho = rho_new;
                if (!ln.formed) {
                    ln.s2 = ln.w2 - (fresh[3 * W + w] + fresh[4 * W + w]);
                    if (!(ln.s2 > cfg.floor_factor * kEps * ln.w2)) enter_formed[w] = true;
                }
                // Near the span of the effects both energy formulas cancel:
                // the exit is decided on the formed residual of the lane.
                if (ln.formed || enter_formed[w]) {
                    formed_mask[w] = true;
                    any_formed = true;
                }
            }
            if (any_formed) {
                double sq[W];
                formed_residual_sq<W>(p, g, team, work, x, vbar, corr, T, col, formed_mask, sq);
                for (int w = 0; w < W; ++w) {
                    if (!formed_mask[w]) continue;
                    Lane& ln = lane[w];
                    if (!finite(sq[w])) {
                        rep.decline_column = ln.column;
                        return finish(Decline::NonFinite);
                    }
                    if (!ln.formed && !(cfg.chat_stop > 0.0)) {
                        // A new measure of tau: the progress window starts
                        // again (with D10 the window reads chat, whose scale
                        // does not change here).
                        ln.best = kInf;
                        ln.have_prev = false;
                    }
                    ln.formed = true;
                    ln.s_formed = std::sqrt(std::max(0.0, sq[w]));
                    ++ln.formed_checks;
                }
            }
            for (int w = 0; w < W; ++w) {
                Lane& ln = lane[w];
                if (!ln.active) continue;
                candidate[w] = evaluate(ln, ln.rho, fresh[1 * W + w], fresh[2 * W + w]);
            }
        }
        for (int w = 0; w < W; ++w) {
            Lane& ln = lane[w];
            if (!ln.active) continue;
            if (cfg.on_iteration != nullptr)
                cfg.on_iteration(cfg.iteration_context, ln.column, k, ln.tau, ln.chat);
            if (candidate[w] && refresh) {
                ln.active = false;
                ln.iterations = k;
                continue;
            }
            ln.stop = Stop::None;
            // Progress: distance to the stop (1 = passes); stall and projection
            // are read at the periodic replacements only.
            const double measure = cfg.chat_stop > 0.0 ? ln.chat / cfg.chat_stop
                                   : cfg.companion       ? std::max(ln.tau / cfg.atol, ln.chat / cfg.trigger)
                                                         : ln.tau / cfg.atol;
            if (measure < ln.best) ln.best = measure;
            const int window = std::max(1, cfg.stall_window);
            if (periodic && k % window == 0) {
                if (ln.have_prev) {
                    if (!(ln.best <= 0.5 * ln.best_prev)) {
                        rep.decline_column = ln.column;
                        return finish(Decline::Stall);
                    }
                    const double rate = std::pow(ln.best / ln.best_prev, 1.0 / window);
                    if (ln.best > 1.0 && rate > 0.0 && rate < 1.0) {
                        const double projected = k + std::log(1.0 / ln.best) / std::log(rate);
                        if (projected > max_iter) {
                            rep.decline_column = ln.column;
                            return finish(Decline::Projected);
                        }
                    }
                }
                ln.best_prev = ln.best;
                ln.have_prev = true;
            }
        }
        bool still[W];
        for (int w = 0; w < W; ++w) still[w] = lane[w].active;
        lane_sums<W, 1>(m, team, work, tot, [&](int i, double* acc) {
            static_cast<void>(acc);
            const std::size_t base = static_cast<std::size_t>(i) * W;
            for (int w = 0; w < W; ++w)
                pv[base + w] = still[w] ? minv[i] * r[base + w] + lane[w].beta * pv[base + w] : 0.0;
        });
        if (cfg.force_decline_iterate) return finish(Decline::Forced);
    }
    rep.iterate_seconds += seconds_since(t_iterate);
    iterating = false;

    // a1 = mean over each level of e of (v - D2 x) = hi + (delta - mean(D2 x)),
    // on every segment; hi stays in vbar, the small part replaces delta.
    t0 = std::chrono::steady_clock::now();
    const int all_parts = static_cast<int>(g.all_bounds.size()) - 1;
    team.run(all_parts, [&](int b) {
        for (int l = g.all_bounds[static_cast<std::size_t>(b)]; l < g.all_bounds[static_cast<std::size_t>(b) + 1U]; ++l) {
            if (rows_of(g, l) == 0) continue;
            double t[W];
            for (int w = 0; w < W; ++w) t[w] = 0.0;
            for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) {
                double u[W];
                gather_cell<W>(g, x, c, u);
                const double mc = g.mult != nullptr ? static_cast<double>(g.mult[c]) : 1.0;
                for (int w = 0; w < W; ++w) t[w] += mc * u[w];
            }
            const double inv = g.seg_inv[static_cast<std::size_t>(l)];
            double* small = corr + static_cast<std::size_t>(l) * W;
            for (int w = 0; w < W; ++w) small[w] -= t[w] * inv;
        }
    });
    rep.recover_seconds += seconds_since(t0);

    // Residuals in row order: (v - hi) - ((delta - mean(D2 x)) + D2 x).
    t0 = std::chrono::steady_clock::now();
    const int parts = std::max(1, std::min(static_cast<int>(g.row_bounds.size()) - 1, n));
    team.run(parts, [&](int b) {
        const int i0 = static_cast<int>(part_begin(n, b, parts));
        const int i1 = static_cast<int>(part_begin(n, b + 1, parts));
        residual_rows<W>(p, g, i0, i1, x, vbar, corr, col, res, width, nullptr, nullptr);
    });
    rep.residual_seconds += seconds_since(t0);

    report_lanes();
    return Decline::None;
}

}  // namespace internal

// Builds the geometry (tests): S_jj for every remaining level, in the order
// of the reduced vector, and the null levels as zeros of the inverse.
inline std::vector<double> schur_diagonal(const Problem& p, const Config& cfg) {
    internal::Team team(cfg.threads, cfg.hooks, 8 * std::max(1, cfg.blocks));
    internal::Geometry g;
    std::size_t bytes = 0;
    std::vector<double> out;
    if (internal::build_geometry(p, cfg, team, 1, g, &bytes) != Decline::None) return out;
    out.resize(static_cast<std::size_t>(g.m));
    for (int i = 0; i < g.m; ++i)
        out[static_cast<std::size_t>(i)] = g.minv[static_cast<std::size_t>(i)] > 0.0 ? g.diag[static_cast<std::size_t>(i)] : 0.0;
    return out;
}

inline Report solve(const Problem& p, const Config& cfg) {
    Report rep;
    rep.columns.assign(static_cast<std::size_t>(std::max(0, p.cols)), ColumnReport{});
    bool valid = p.n > 0 && p.k >= 2 && p.k <= 5 && p.cols > 0 && p.codes != nullptr &&
                 p.levels != nullptr && p.columns != nullptr && p.residuals != nullptr &&
                 p.eliminated >= 0 && p.eliminated < p.k;
    for (int b = 0; valid && b < p.k; ++b) valid = p.levels[b] >= 1 && p.codes[b] != nullptr;
    if (!valid) {
        rep.decline = Decline::Input;
        return rep;
    }
    const internal::Team team(cfg.threads, cfg.hooks, 8 * std::max(1, cfg.blocks));
    const int groups = (p.cols + 7) / 8;
    const int per_group = (p.cols + groups - 1) / groups;
    const int lanes = internal::lanes_for(per_group);
    rep.groups = groups;
    rep.lanes = lanes;
    try {
        auto t0 = std::chrono::steady_clock::now();
        internal::Geometry g;
        const Decline geometry = internal::build_geometry(p, cfg, team, lanes, g, &rep.bytes);
        rep.geometry_seconds = internal::seconds_since(t0);
        rep.sorted = g.sorted;
        rep.merged = g.merged;
        rep.segments = g.L;
        rep.operator_segments = g.L_op;
        rep.cells = g.C;
        for (int j = 0; j < g.nb && geometry == Decline::None; ++j) rep.large_effects += g.large[static_cast<std::size_t>(j)];
        if (geometry != Decline::None) {
            rep.decline = geometry;
            return rep;
        }
        if (cfg.force_decline_geometry) {
            rep.decline = Decline::Forced;
            return rep;
        }
        internal::Work work;
        work.prefetch_cells = std::max(0, cfg.prefetch_cells);
        const std::size_t W = static_cast<std::size_t>(lanes);
        const std::size_t m = static_cast<std::size_t>(g.m), L = static_cast<std::size_t>(g.L);
        const int op_parts = static_cast<int>(g.op_bounds.size()) - 1;
        work.g.allocate(m * W);
        work.arena.allocate(W * (4U * m + L));
        work.vbar.allocate(L * W);
        work.corr.allocate(L * W);
        work.priv.allocate(static_cast<std::size_t>(op_parts) * static_cast<std::size_t>(g.small_levels) * W);
        work.partials.resize(static_cast<std::size_t>(g.nb));
        for (int j = 0; j < g.nb; ++j)
            if (g.large[static_cast<std::size_t>(j)])
                work.partials[static_cast<std::size_t>(j)].allocate(static_cast<std::size_t>(g.tr[static_cast<std::size_t>(j)].items) * W);
        work.units.allocate((m / internal::kVectorUnit + 1U) * 8U * W);
        work.stats.allocate((g.row_bounds.size() + 1U) * internal::kStats * W);
        for (int group = 0; group < groups; ++group) {
            const int col0 = group * per_group;
            const int width = std::min(per_group, p.cols - col0);
            Decline d = Decline::None;
            switch (lanes) {
                case 1: d = internal::run_group<1>(p, cfg, g, team, work, col0, width, rep); break;
                case 2: d = internal::run_group<2>(p, cfg, g, team, work, col0, width, rep); break;
                case 4: d = internal::run_group<4>(p, cfg, g, team, work, col0, width, rep); break;
                default: d = internal::run_group<8>(p, cfg, g, team, work, col0, width, rep); break;
            }
            rep.hinted_applies = work.hinted_applies;
            if (d != Decline::None) {
                rep.decline = d;
                return rep;
            }
        }
        rep.solved = true;
    } catch (const std::bad_alloc&) {
        rep.solved = false;
        rep.decline = Decline::Memory;
    }
    return rep;
}

}  // namespace reduced_pcg
}  // namespace hdfe

#endif  // HDFE_REDUCED_SCHUR_PCG_HPP
