#ifndef XHDFE_TESTS_REDUCED_CUDA_BENCH_CANDIDATE_OPERATOR_HPP
#define XHDFE_TESTS_REDUCED_CUDA_BENCH_CANDIDATE_OPERATOR_HPP
// Candidate CUDA reduced operator (package 2, stage 2b, 17 November 2026), in the
// bench only: the design note DESENHO_P2.md as amended (Amendment 1, B1-B10).
// The caller hands it a plain view of the CPU geometry (no CUDA or OpenMP types
// here); it uploads what the geometry holds, builds on the device the
// level-ordered arrays of the effects that are neither private nor transposed,
// and computes out = S in with:
//   K1   segments of at most `short_seg` cells: one group of threads per segment,
//        the mean sequential in cell order, the private effects' deviations into
//        per-group shared accumulators, combined in group order per block;
//   K1L  longer segments: one block each, items of `item` cells summed in order,
//        an in-place pairwise tree on item index in global partials;
//   K2   the levels of every transposed or built effect, one group per unit
//        (a level of at most `short_level` entries, or `item` entries of a longer one);
//   K3   fixed trees: private effects over the block partials, long levels over items.
// Lanes never mix; no atomic operation takes part in a value; every output is
// written by one thread; the padding lanes of `out` are written +0.0.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace xhdfe_bench::cand {

enum class ErrorKind { Memory, Unavailable, Pinned, Fatal };

class Error : public std::runtime_error {
public:
    Error(ErrorKind kind, const std::string& message) : std::runtime_error(message), kind_(kind) {}
    ErrorKind kind() const noexcept { return kind_; }
private:
    ErrorKind kind_;
};

// A remaining effect as the CPU geometry holds it.
struct DimView {
    int levels = 0;
    int offset = 0;
    bool transposed = false;           // build_geometry made tr[j]
    const int* tr_ptr = nullptr;       // levels + 1
    const int* tr_seg = nullptr;       // entry -> ORIGINAL segment id (indexes T)
    const int* tr_other[4] = {nullptr, nullptr, nullptr, nullptr};  // [jj], jj != j
    const int* tr_mult = nullptr;      // only when has_mult
    bool has_mult = false;
};

struct GeometryView {
    int nb = 0, m = 0, L = 0, L_op = 0;
    long long C = 0, C_op = 0;
    const int* cell_ptr = nullptr;     // L + 1
    const int* op_seg = nullptr;       // L_op
    const double* seg_inv = nullptr;   // L: 1 / rows
    const int* cc[4] = {nullptr, nullptr, nullptr, nullptr};  // segment-order codes of each remaining effect
    const int* mult = nullptr;         // merged geometry only
    DimView dims[4];
};

enum class WidthMode { Padded, Sub8, InWarp };   // device lanes: W | real width in subgroups of 2^k threads | width threads, floor(32/width) groups per warp
enum class Transfer { Pageable, Pinned, PinnedChunks, Registered };
enum class BMethod { Auto, Tiles, Radix };

struct Options {
    int lanes = 8;                       // W: the stride of the caller's vectors
    WidthMode width_mode = WidthMode::Padded;
    Transfer transfer = Transfer::Pinned;
    BMethod b_method = BMethod::Auto;
    int private_max = 48;                // cumulative levels of the private effects
    int short_seg = 256;                 // K1 handles segments of at most this many cells
    int short_level = 256;               // a level of at most this many entries is one K2 unit
    int item = 256;                      // items of long segments and long levels
    int seg_per_group = 16;              // K1: segments per group per block
    int k1_cache = 16;                   // K1: segments of at most this many cells (0, 8, 16, 24 or 32) keep u in registers
    // K1's segment order (K1 iteration, 17 November): 0 the geometry's; 1 grouped by length, a stable sort by
    // decreasing cells (a fixed permutation of the geometry: K1L, the B build, slots and owners keep theirs).
    int seg_order = 0;
    int tile = 4096;                     // B: tiles of the stable counting sort
    bool sort_units = true;              // launch K2 units by decreasing length (slots stay logical)
    std::size_t chunk_bytes = std::size_t{8} << 20;  // PinnedChunks, and the staged uploads
    bool staged_uploads = true;          // resident uploads through two page-locked chunks
    bool profile = true;                 // per-kernel events
    std::size_t budget = 0;              // admission budget (bytes); 0: no budget, free memory only
};

// Every device and host buffer of the operator, by phase (B1, B7).
struct Admission {
    std::vector<std::pair<std::string, std::size_t>> device_resident, device_setup_scratch, device_product, host;
    std::size_t device_setup_peak = 0, device_iteration_peak = 0, device_peak = 0, host_peak = 0;
    std::size_t free_at_entry = 0;
    std::string text() const;
};

struct Stats {
    int calls = 0;
    // set-up (s)
    double admission_s = 0, alloc_s = 0, upload_s = 0, units_s = 0, b_build_s = 0, b_ptr_download_s = 0, pinned_s = 0, setup_total_s = 0;
    double seg_order_s = 0;                      // building and uploading K1's segment order (seg_order 1)
    double b_steps_ms[6] = {0, 0, 0, 0, 0, 0};   // enumerate, histogram, totals+scan, offsets, scatter, ptr download (or radix)
    double teardown_s = 0;
    // per product (accumulated ms)
    double pack_ms = 0, h2d_ms = 0, k1_ms = 0, k1l_ms = 0, k2_ms[4] = {0, 0, 0, 0}, k3_ms = 0, d2h_ms = 0, unpack_ms = 0, apply_ms = 0;
    std::size_t uploaded_bytes = 0;
    int private_levels = 0, long_segments = 0, long_levels[4] = {0, 0, 0, 0};
    long long units[4] = {0, 0, 0, 0}, pb_slots[4] = {0, 0, 0, 0}, k1_blocks = 0, pa_slots = 0;
    std::string classes;                 // per remaining effect: P, T or B
    // compiled K1 / K2 (B8.3)
    int k1_regs = 0, k1_local = 0, k1_blocks_per_sm = 0, k2_regs = 0, k2_local = 0, k2_blocks_per_sm = 0;
    std::size_t k1_smem = 0;
};

class Operator {
public:
    Operator(const GeometryView& g, const Options& o);
    ~Operator();
    Operator(const Operator&) = delete;
    Operator& operator=(const Operator&) = delete;
    // The width of the next group (from solve(), before each group): 1..lanes.
    void begin_group(int width);
    // in, out: m x lanes, stride lanes; out's lanes >= width are written +0.0.
    void apply(const double* in, double* out);
    // Registered transfers only: page-lock the caller's vectors (released by the destructor or unregister()).
    void register_host(const double* in, double* out);
    void unregister();
    const Stats& stats() const noexcept;
    const Admission& admission() const noexcept;
    // Tests: the device-built level-ordered arrays of remaining effect j (B dimensions only).
    struct LevelOrderCopy { std::vector<int> ptr, seg, mult, cell; std::vector<std::vector<int>> other; };
    LevelOrderCopy download_level_order(int j) const;
    char dim_class(int j) const;
private:
    struct State;
    std::unique_ptr<State> s_;
};

// B8.1: page-locked transfers and the whole pack/DMA/unpack cycle at one size.
struct TransferResult {
    double pinned_alloc_s = 0, pinned_free_s = 0, register_s = 0, unregister_s = 0;
    double pageable_h2d_ms = 0, pageable_d2h_ms = 0, pinned_h2d_ms = 0, pinned_d2h_ms = 0;  // pinned: packed rows (width)
    double pinned_full_h2d_ms = 0, pinned_full_d2h_ms = 0;                                 // pinned: padded rows (lanes)
    double cycle_ms = 0, cycle_pack_ms = 0, cycle_unpack_ms = 0, chunked_cycle_ms = 0, registered_cycle_ms = 0;
    double registered_h2d_ms = 0, registered_d2h_ms = 0;
    double registered2d_h2d_ms = 0, registered2d_d2h_ms = 0, registered2d_cycle_ms = 0, registered2d_zero_ms = 0;
};
TransferResult measure_transfers(int m, int lanes, int width, int reps, std::size_t chunk_bytes);

}  // namespace xhdfe_bench::cand
#endif
