// Standalone bench of the reduced tier's iterative operator S p (package 2,
// stages 2a-2b and the integration, 17 November 2026). Synthetic geometries only
// (synthetic_problem.hpp): nothing is read from disk. Arms on the same Geometry, built
// by the CPU tier's own build_geometry:
//   cpu       the CPU apply_operator<W> alone, on the team (16 threads by default);
//   cpuw      the same at W = the real width (the live columns), for reference;
//   gpu       the operator of the tier BEFORE package 2, a frozen copy kept for the bench
//             only (frozen_20261115/; make_layout + Operator, as the tier built it);
//   cand      the stage-2b candidate (candidate_operator.cu), its variants by option, as
//             validated in the bench (frozen with it);
//   prod      the INTEGRATED operator (src/reduced_schur_pcg_cuda.cu, compiled here with
//             XHDFE_REDUCED_CUDA_TEST_HOOKS for the source-cell check), as the tier builds
//             it: the view of the geometry, one registration of adjacent p and q;
//   xfer      page-locked transfers and the pack / DMA / unpack cycle at this size;
//   levelref  the host reference of the level-ordered arrays of every remaining
//             effect (team-parallel stable counting sort, checked against a serial
//             one and, where build_geometry made one, against its transpose).
// Every device product is compared with the CPU one per lane against a bound from
// the operation counts (check_lanes); two calls must agree bit for bit; cand and prod
// also run A -> B -> A (widths) with sentinel-filled outputs, and their device-built
// level-ordered arrays are compared with the host reference; prod is compared bit for
// bit with cand when both run at the same width (cand in the winner's configuration).
// Prints one line of aggregates per run, starting REDUCED_CUDA_BENCH.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "candidate_operator.hpp"
#include "frozen_20261115/frozen_layout.hpp"
#include "frozen_20261115/frozen_operator.hpp"
#include "hdfe/deterministic_parallel.hpp"
#include "level_order_reference.hpp"
#include "reduced_schur_pcg.hpp"
#include "reduced_schur_pcg_cuda.hpp"
#include "reduced_schur_pcg_cuda_layout.hpp"
#include "reduced_schur_pcg_cuda_memory.hpp"
#include "synthetic_problem.hpp"

namespace rp = hdfe::reduced_pcg;
namespace rpi = hdfe::reduced_pcg::internal;
namespace gpu = hdfe::detail::reduced_cuda;      // the integrated operator (prod)
namespace frozen = xhdfe_bench::frozen;           // the operator before package 2 (gpu)
namespace cand = xhdfe_bench::cand;
using Clock = std::chrono::steady_clock;

namespace {

double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

struct Stat {
    double median = 0.0, min = 0.0;
    int count = 0;
};

Stat stat_of(std::vector<double> v) {
    Stat s;
    s.count = static_cast<int>(v.size());
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    s.median = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    s.min = v.front();
    return s;
}

struct Options {
    xhdfe_bench::Shape shape;
    std::string label;
    int lanes = 8;
    int width = -1;           // live lanes (columns); -1: all lanes
    int threads = 16;
    int blocks = 0;           // 0: deterministic_parallel_chunk_count(n), as the tier
    int products = 20;
    int warmup = 3;
    std::string arms = "cpu,cpuw,gpu,levelref";
    int merge = -1;           // Config::force_merge
    int effect_class = -1;    // Config::force_effect_class
    std::size_t budget = 0;   // 0: admission_budget(n, width, k), as the tier
    int levelref_parts = 64;  // logical blocks of the host reference (a constant, not the team)
    std::uint64_t operand_seed = 7;
    cand::Options cand;
    int aba_width = -1;       // B of the A -> B -> A check; -1: max(1, width - 2)
    int xfer_reps = 20;
    bool registered_adjacent = false;  // registered transfers: p and q adjacent slices of one 64-byte (not page) aligned buffer
    gpu::Options prod;        // the integrated operator's structural options (defaults: the tier's)
    bool prod_cells = false;  // prod: build the source cells of B effects and check them (memory and time not as in a fit)
};

bool has_arm(const Options& o, const std::string& arm) {
    std::stringstream ss(o.arms);
    std::string item;
    while (std::getline(ss, item, ','))
        if (item == arm) return true;
    return false;
}

std::vector<int> int_list(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stoi(item));
    return out;
}

// The three plan shapes (n, levels with the eliminated effect first, live
// columns) of the traced fits of 16-17 November, and small adversarial ones.
void preset(const std::string& name, Options& o) {
    auto& s = o.shape;
    s.name = name;
    if (name == "ready") {
        // Calibrated by one number, `tail` (skew 1.2), so that the level items of the large effect
        // match the stage-0 trace (b_items 672 801 / 875 590 / 1 338 459) with few empty levels.
        s.rows = 46156187; s.levels = {5074679, 610089, 26}; o.width = 5; s.skew = 1.2; s.tail = 0.5;
    } else if (name == "firstreg") {
        s.rows = 55947171; s.levels = {5948793, 36, 799265}; o.width = 7; s.skew = 1.2; s.tail = 0.5;
    } else if (name == "simulated") {
        s.rows = 173163263; s.levels = {20000000, 1000000, 500, 15}; o.width = 5; s.tail = 0.0;
    } else if (name == "mixed-uniform26") {
        // K1 iteration (17 November): segments uniform on 2..26 cells (mean 14), ready's other effects.
        s.rows = 46156187; s.levels = {3296870, 610089, 26}; o.width = 5; s.skew = 1.2; s.tail = 0.5;
        s.length_law = xhdfe_bench::Shape::LengthLaw::Uniform; s.len_lo = 2; s.len_hi = 26;
    } else if (name == "mixed-tail27") {
        // A short-heavy mixture with a tail to 27 cells, mean 9.10: ready's segments and effects.
        s.rows = 46156187; s.levels = {5074679, 610089, 26}; o.width = 5; s.skew = 1.2; s.tail = 0.5;
        s.length_law = xhdfe_bench::Shape::LengthLaw::Mixture; s.len_lo = 2; s.len_mid = 12; s.len_hi = 27; s.len_p = 0.8388;
    } else if (name == "mixed-tail36") {
        // The same with a tail to 36 cells, mean 9.40: firstreg's segments and effects.
        s.rows = 55947171; s.levels = {5948793, 36, 799265}; o.width = 7; s.skew = 1.2; s.tail = 0.5;
        s.length_law = xhdfe_bench::Shape::LengthLaw::Mixture; s.len_lo = 2; s.len_mid = 12; s.len_hi = 36; s.len_p = 0.8626;
    } else if (name == "small") {
        s.rows = 400000; s.levels = {45000, 6000, 20}; o.width = 5;
    } else if (name.rfind("adv-", 0) == 0) {
        s.rows = 300000; s.levels = {33000, 4000, 300, 12}; o.width = 5;
        if (name == "adv-unsorted") s.sorted = false;
        else if (name == "adv-merged") { s.dup = 0.3; o.merge = 1; }
        else if (name == "adv-single-cell") s.single_cell_segment = true;
        else if (name == "adv-left-out") s.left_out_segment = true;
        else if (name == "adv-null-levels") { s.unused_levels = 7; s.stayer_level = true; }
        else if (name == "adv-disconnected") s.disconnected = true;
        else if (name == "adv-long-level") s.long_level_share = 0.3;
        else if (name == "adv-long-segment") s.long_segment_rows = 5000;
        else if (name == "adv-class0") o.effect_class = 0;
        else if (name == "adv-class1") o.effect_class = 1;
        else throw std::invalid_argument("unknown adversarial shape " + name);
    } else if (name != "custom") {
        throw std::invalid_argument("unknown shape " + name);
    }
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i + 1 < argc; i += 2)
        if (std::string(argv[i]) == "--shape") preset(argv[i + 1], o);
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        const std::string v = argv[i + 1];
        auto& s = o.shape;
        auto& c = o.cand;
        if (key == "--shape") continue;
        else if (key == "--label") o.label = v;
        else if (key == "--rows") s.rows = std::stoll(v);
        else if (key == "--levels") s.levels = int_list(v);
        else if (key == "--skew") s.skew = std::stod(v);
        else if (key == "--tail") s.tail = std::stod(v);
        else if (key == "--move") s.move = std::stod(v);
        else if (key == "--year-like-max") s.year_like_max = std::stoi(v);
        else if (key == "--spread") s.spread = std::stoi(v);
        else if (key == "--sorted") s.sorted = std::stoi(v) != 0;
        else if (key == "--dup") s.dup = std::stod(v);
        else if (key == "--long-segment") s.long_segment_rows = std::stoll(v);
        else if (key == "--single-cell") s.single_cell_segment = std::stoi(v) != 0;
        else if (key == "--left-out") s.left_out_segment = std::stoi(v) != 0;
        else if (key == "--stayer") s.stayer_level = std::stoi(v) != 0;
        else if (key == "--unused-levels") s.unused_levels = std::stoi(v);
        else if (key == "--disconnected") s.disconnected = std::stoi(v) != 0;
        else if (key == "--long-level-share") s.long_level_share = std::stod(v);
        else if (key == "--lengths") {  // spread | uniform:LO:HI | mix:P:LO:MID:HI
            std::vector<std::string> f;
            std::stringstream ss(v);
            for (std::string item; std::getline(ss, item, ':');) f.push_back(item);
            if (f.size() == 1 && f[0] == "spread") s.length_law = xhdfe_bench::Shape::LengthLaw::MeanSpread;
            else if (f.size() == 3 && f[0] == "uniform") {
                s.length_law = xhdfe_bench::Shape::LengthLaw::Uniform; s.len_lo = std::stoi(f[1]); s.len_hi = std::stoi(f[2]);
            } else if (f.size() == 5 && f[0] == "mix") {
                s.length_law = xhdfe_bench::Shape::LengthLaw::Mixture; s.len_p = std::stod(f[1]);
                s.len_lo = std::stoi(f[2]); s.len_mid = std::stoi(f[3]); s.len_hi = std::stoi(f[4]);
            } else throw std::invalid_argument("--lengths spread|uniform:LO:HI|mix:P:LO:MID:HI");
        }
        else if (key == "--seed") s.seed = std::stoull(v);
        else if (key == "--lanes") o.lanes = std::stoi(v);
        else if (key == "--width") o.width = std::stoi(v);
        else if (key == "--threads") o.threads = std::stoi(v);
        else if (key == "--blocks") o.blocks = std::stoi(v);
        else if (key == "--products") o.products = std::stoi(v);
        else if (key == "--warmup") o.warmup = std::stoi(v);
        else if (key == "--arms") o.arms = v;
        else if (key == "--merge") o.merge = v == "auto" ? -1 : v == "on" ? 1 : v == "off" ? 0 : throw std::invalid_argument("--merge auto|on|off");
        else if (key == "--effect-class") o.effect_class = v == "auto" ? -1 : std::stoi(v);
        else if (key == "--budget") o.budget = static_cast<std::size_t>(std::stoull(v));
        else if (key == "--levelref-parts") o.levelref_parts = std::stoi(v);
        else if (key == "--operand-seed") o.operand_seed = std::stoull(v);
        else if (key == "--aba-width") o.aba_width = std::stoi(v);
        else if (key == "--xfer-reps") o.xfer_reps = std::stoi(v);
        else if (key == "--registered-adjacent") o.registered_adjacent = std::stoi(v) != 0;
        else if (key == "--cand-width-mode")
            c.width_mode = v == "padded" ? cand::WidthMode::Padded : v == "sub8" ? cand::WidthMode::Sub8
                         : v == "inwarp" ? cand::WidthMode::InWarp : throw std::invalid_argument("--cand-width-mode padded|sub8|inwarp");
        else if (key == "--cand-transfer")
            c.transfer = v == "pageable" ? cand::Transfer::Pageable : v == "pinned" ? cand::Transfer::Pinned
                       : v == "chunks" ? cand::Transfer::PinnedChunks : v == "registered" ? cand::Transfer::Registered
                       : throw std::invalid_argument("--cand-transfer pageable|pinned|chunks|registered");
        else if (key == "--cand-b")
            c.b_method = v == "auto" ? cand::BMethod::Auto : v == "tiles" ? cand::BMethod::Tiles : v == "radix" ? cand::BMethod::Radix
                       : throw std::invalid_argument("--cand-b auto|tiles|radix");
        else if (key == "--cand-private-max") c.private_max = std::stoi(v);
        else if (key == "--cand-short-seg") c.short_seg = std::stoi(v);
        else if (key == "--cand-short-level") c.short_level = std::stoi(v);
        else if (key == "--cand-item") c.item = std::stoi(v);
        else if (key == "--cand-seg-per-group") c.seg_per_group = std::stoi(v);
        else if (key == "--cand-k1-cache") c.k1_cache = std::stoi(v);
        else if (key == "--cand-staged-uploads") c.staged_uploads = std::stoi(v) != 0;
        else if (key == "--cand-tile") c.tile = std::stoi(v);
        else if (key == "--cand-sort-units") c.sort_units = std::stoi(v) != 0;
        else if (key == "--cand-chunk-mb") c.chunk_bytes = static_cast<std::size_t>(std::stod(v) * 1048576.0);
        else if (key == "--cand-seg-order")
            c.seg_order = v == "natural" ? 0 : v == "length" ? 1 : throw std::invalid_argument("--cand-seg-order natural|length");
        else if (key == "--prod-b")
            o.prod.b_method = v == "auto" ? gpu::BMethod::Auto : v == "tiles" ? gpu::BMethod::Tiles : v == "radix" ? gpu::BMethod::Radix
                            : throw std::invalid_argument("--prod-b auto|tiles|radix");
        else if (key == "--prod-private-max") o.prod.private_max = std::stoi(v);
        else if (key == "--prod-items") { o.prod.short_seg = o.prod.short_level = o.prod.item = std::stoi(v); }
        else if (key == "--prod-cells") o.prod_cells = std::stoi(v) != 0;
        else throw std::invalid_argument("unknown option " + key);
    }
    if (o.lanes < 1 || o.lanes > 8) throw std::invalid_argument("--lanes must be 1..8");
    if (o.width < 0 || o.width > o.lanes) o.width = o.lanes;
    if (o.width < 1) throw std::invalid_argument("--width must be 1..lanes");
    if (o.products < 1 || o.warmup < 2) throw std::invalid_argument("--products >= 1 and --warmup >= 2 (two calls are compared)");
    if (o.shape.rows <= 0 || o.shape.levels.size() < 2) throw std::invalid_argument("a shape: --shape NAME or --rows and --levels");
    if (o.aba_width < 0) o.aba_width = std::max(1, o.width - 2);
    o.cand.lanes = o.lanes;
    o.prod.lanes = o.lanes;
    return o;
}

// ---------------------------------------------------------------------------
// CPU arm: apply_operator<W> alone, with the workspace the controller gives it.

template <int W>
Stat cpu_products(const rpi::Geometry& g, const rpi::Team& team, const double* in, double* out, double* second,
                  int warmup, int products, bool& bit_equal) {
    rpi::Work work;
    const int op_parts = static_cast<int>(g.op_bounds.size()) - 1;
    work.priv.allocate(static_cast<std::size_t>(op_parts) * static_cast<std::size_t>(g.small_levels) * W);
    work.partials.resize(static_cast<std::size_t>(g.nb));
    for (int j = 0; j < g.nb; ++j)
        if (g.large[static_cast<std::size_t>(j)])
            work.partials[static_cast<std::size_t>(j)].allocate(static_cast<std::size_t>(g.tr[static_cast<std::size_t>(j)].items) * W);
    rpi::Buffer<double> T;
    T.allocate(static_cast<std::size_t>(g.L) * W);
    std::vector<double> ms;
    for (int it = 0; it < warmup + products; ++it) {
        double* dst = it == 1 ? second : out;
        const auto t0 = Clock::now();
        rpi::apply_operator<W>(g, team, in, dst, T.data(), work);
        const double s = since(t0);
        if (it >= warmup) ms.push_back(1e3 * s);
    }
    bit_equal = std::memcmp(out, second, sizeof(double) * static_cast<std::size_t>(g.m) * W) == 0;
    return stat_of(ms);
}

Stat cpu_dispatch(int W, const rpi::Geometry& g, const rpi::Team& team, const double* in, double* out, double* second,
                  int warmup, int products, bool& bit_equal) {
    switch (W) {
        case 1: return cpu_products<1>(g, team, in, out, second, warmup, products, bit_equal);
        case 2: return cpu_products<2>(g, team, in, out, second, warmup, products, bit_equal);
        case 3: return cpu_products<3>(g, team, in, out, second, warmup, products, bit_equal);
        case 4: return cpu_products<4>(g, team, in, out, second, warmup, products, bit_equal);
        case 5: return cpu_products<5>(g, team, in, out, second, warmup, products, bit_equal);
        case 6: return cpu_products<6>(g, team, in, out, second, warmup, products, bit_equal);
        case 7: return cpu_products<7>(g, team, in, out, second, warmup, products, bit_equal);
        default: return cpu_products<8>(g, team, in, out, second, warmup, products, bit_equal);
    }
}

// ---------------------------------------------------------------------------
// GPU arm: the operator before package 2 (the frozen copy), built as the tier built it.

struct GpuResult {
    std::string error;
    double layout_s = 0.0, setup_s = 0.0, layout_free_s = 0.0, release_s = 0.0;
    frozen::LayoutSplit layout_split;
    frozen::SetupSplit setup_split;
    std::size_t payload = 0, uploaded = 0, host_layout_bytes = 0, host_layout_peak = 0, host_geometry_bytes = 0;
    long long a_items = 0, b_items = 0;
    Stat kernel, h2d, d2h, apply, seg_items, seg_finish, lvl_items, lvl_finish;
    bool bit_equal = false;
};

GpuResult gpu_products(const rpi::Geometry& g, int W, std::size_t budget, const double* in, double* out,
                       int warmup, int products) {
    GpuResult r;
    try {
        const std::size_t state_bytes = gpu::host_sum({frozen::Operator::host_state_bytes()});
        r.host_geometry_bytes = gpu::geometry_resident_bytes(g);
        const std::size_t reserved = gpu::host_sum({r.host_geometry_bytes, state_bytes, sizeof(frozen::Layout)});
        auto t0 = Clock::now();
        frozen::Layout layout = frozen::make_layout(g, W, budget, reserved, &r.layout_split);
        r.layout_s = since(t0);
        r.host_layout_bytes = layout.host_bytes();
        r.host_layout_peak = gpu::host_sum({layout.host_bytes(), static_cast<std::size_t>(g.m) * sizeof(int), sizeof(layout)});
        r.payload = layout.device_bytes();
        r.a_items = static_cast<long long>(layout.a_item_begin.size());
        r.b_items = static_cast<long long>(layout.b_item_begin.size());
        const std::size_t device_only = sizeof(double) * static_cast<std::size_t>(W) *
            (2U * static_cast<std::size_t>(g.m) + static_cast<std::size_t>(layout.segments) +
             static_cast<std::size_t>(r.a_items) + static_cast<std::size_t>(r.b_items));
        r.uploaded = r.payload - device_only;
        t0 = Clock::now();
        auto op = std::make_unique<frozen::Operator>(layout, budget, true, &r.setup_split);
        r.setup_s = since(t0);
        t0 = Clock::now();
        layout = frozen::Layout{};
        r.layout_free_s = since(t0);
        const std::size_t count = static_cast<std::size_t>(g.m) * W;
        std::vector<double> second(count), scratch(count);
        std::vector<double> kernel, h2d, d2h, apply, si, sf, li, lf;
        for (int it = 0; it < warmup + products; ++it) {
            frozen::KernelSplit split;
            const frozen::Statistics before = op->statistics();
            double* dst = it == 0 ? out : it == 1 ? second.data() : scratch.data();
            const auto t = Clock::now();
            op->apply(in, dst, &split);
            const double wall = since(t);
            const frozen::Statistics after = op->statistics();
            if (it < warmup) continue;
            kernel.push_back(after.kernel_ms - before.kernel_ms);
            h2d.push_back(after.h2d_ms - before.h2d_ms);
            d2h.push_back(after.d2h_ms - before.d2h_ms);
            apply.push_back(1e3 * wall);
            si.push_back(split.segment_items_ms);
            sf.push_back(split.segment_finish_ms);
            li.push_back(split.level_items_ms);
            lf.push_back(split.level_finish_ms);
        }
        r.bit_equal = std::memcmp(out, second.data(), sizeof(double) * count) == 0;
        r.kernel = stat_of(kernel); r.h2d = stat_of(h2d); r.d2h = stat_of(d2h); r.apply = stat_of(apply);
        r.seg_items = stat_of(si); r.seg_finish = stat_of(sf); r.lvl_items = stat_of(li); r.lvl_finish = stat_of(lf);
        t0 = Clock::now();
        op.reset();
        r.release_s = since(t0);
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    return r;
}

// ---------------------------------------------------------------------------
// Entries per output level (operator cells) and the longest segment.
struct LevelCounts {
    std::vector<long long> entries;  // m
    int cmax = 0;
    long long seg_over16 = 0, cells_over16 = 0;  // operator segments of more than 16 cells (K1's register cache), and their cells
};

LevelCounts level_counts(const rpi::Geometry& g) {
    LevelCounts r;
    r.entries.assign(static_cast<std::size_t>(g.m), 0);
    for (int s = 0; s < g.L_op; ++s) {
        const int l = g.op_seg[static_cast<std::size_t>(s)];
        const int c0 = g.cell_ptr[l], c1 = g.cell_ptr[l + 1];
        r.cmax = std::max(r.cmax, c1 - c0);
        if (c1 - c0 > 16) { ++r.seg_over16; r.cells_over16 += c1 - c0; }
        for (int c = c0; c < c1; ++c)
            for (int j = 0; j < g.nb; ++j)
                ++r.entries[static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)] + g.cc[static_cast<std::size_t>(j)][c])];
    }
    return r;
}

// Device against CPU, per lane. Both compute out_i = sum over the entries a of
// level i of m_a (u_a - mean_s(a)), u_a = sum over the remaining effects of the
// operand, mean_s = (sum over the cells of s of m_c u_c) / rows(s). For any
// order of these sums (the CPU's, possibly reassociated by -ffast-math, or the
// device's trees) the computed value differs from the exact one by at most
// gamma_K * B_i (Higham's gamma_K = K u / (1 - K u), u = 2^-53) with
//   B_i = sum_a m_a (U_a + M_s(a)),  U_a = sum |operand terms|,  M_s = (sum_c m_c U_c) / rows(s),
//   K   = 2 E_i + Cmax + nb + parts + 8,
// E_i the entries of level i, Cmax the most cells of a segment, parts the
// CPU's private partitions (its small effects add one partial per partition).
// So |device - CPU| <= 2 gamma_K B_i; the ratio to that bound must be <= 1. The
// bound comes from the operation counts only, never from the observed error.
// Reported too: max |device - CPU| / B_i (the difference against the size of the
// terms summed) and the largest absolute limit 2 gamma_K B_i.

struct LaneCheck {
    double max_abs = 0.0, max_ratio = 0.0, max_rel = 0.0, max_limit = 0.0;
    long long zero_bound_mismatch = 0, nonfinite = 0;
};

bool zero_bits(double v) {
    std::uint64_t b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return b == 0;
}

std::vector<LaneCheck> check_lanes(const rpi::Geometry& g, const LevelCounts& lc, const double* in, const double* a, int a_stride,
                                   const double* b, int b_stride, int W_in, int width, int parts) {
    const int m = g.m, nb = g.nb;
    std::vector<LaneCheck> res(static_cast<std::size_t>(width));
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int w = 0; w < width; ++w) {
        std::vector<long double> bound(static_cast<std::size_t>(m), 0.0L);
        auto U = [&](int c) {
            long double u = 0.0L;
            for (int j = 0; j < nb; ++j)
                u += std::fabs(static_cast<long double>(
                    in[static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)] + g.cc[static_cast<std::size_t>(j)][c]) * W_in + w]));
            return u;
        };
        for (int s = 0; s < g.L_op; ++s) {
            const int l = g.op_seg[static_cast<std::size_t>(s)];
            const int c0 = g.cell_ptr[l], c1 = g.cell_ptr[l + 1];
            long double A = 0.0L;
            for (int c = c0; c < c1; ++c) A += (g.mult ? g.mult[c] : 1) * U(c);
            const long double M = A * static_cast<long double>(g.seg_inv[static_cast<std::size_t>(l)]);
            for (int c = c0; c < c1; ++c) {
                const long double t = (g.mult ? g.mult[c] : 1) * (U(c) + M);
                for (int j = 0; j < nb; ++j)
                    bound[static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)] + g.cc[static_cast<std::size_t>(j)][c])] += t;
            }
        }
        LaneCheck& r = res[static_cast<std::size_t>(w)];
        const long double unit = std::ldexp(1.0L, -53);
        for (int i = 0; i < m; ++i) {
            const double x = a[static_cast<std::size_t>(i) * a_stride + w], y = b[static_cast<std::size_t>(i) * b_stride + w];
            if (!rpi::finite(x) || !rpi::finite(y)) { ++r.nonfinite; continue; }
            const double diff = std::fabs(x - y);
            r.max_abs = std::max(r.max_abs, diff);
            const long double K = 2.0L * lc.entries[static_cast<std::size_t>(i)] + lc.cmax + nb + parts + 8;
            const long double limit = 2.0L * (K * unit / (1.0L - K * unit)) * bound[static_cast<std::size_t>(i)];
            r.max_limit = std::max(r.max_limit, static_cast<double>(limit));
            if (limit == 0.0L) {
                if (diff != 0.0) ++r.zero_bound_mismatch;
                continue;
            }
            r.max_ratio = std::max(r.max_ratio, static_cast<double>(diff / limit));
            r.max_rel = std::max(r.max_rel, static_cast<double>(diff / bound[static_cast<std::size_t>(i)]));
        }
    }
    return res;
}

bool padding_zero(const double* v, int m, int W, int width) {
    for (int i = 0; i < m; ++i)
        for (int w = width; w < W; ++w)
            if (!zero_bits(v[static_cast<std::size_t>(i) * W + w])) return false;
    return true;
}

std::string load_average() {
    std::ifstream f("/proc/loadavg");
    std::string a, b, c;
    f >> a >> b >> c;
    return a + "/" + b + "/" + c;
}

template <class T>
std::string join(const std::vector<T>& v, const char* fmt) {
    std::string out;
    char buf[64];
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::snprintf(buf, sizeof(buf), fmt, v[i]);
        out += (i ? "," : "");
        out += buf;
    }
    return out;
}

void emit_check(std::ostringstream& line, const std::string& prefix, const std::vector<LaneCheck>& lanes, bool padding_ok) {
    std::vector<double> diff, ratio, rel, limit;
    long long mismatch = 0, nonfinite = 0;
    double worst = 0.0;
    for (const auto& c : lanes) {
        diff.push_back(c.max_abs);
        ratio.push_back(c.max_ratio);
        rel.push_back(c.max_rel);
        limit.push_back(c.max_limit);
        mismatch += c.zero_bound_mismatch;
        nonfinite += c.nonfinite;
        worst = std::max(worst, c.max_ratio);
    }
    line << ' ' << prefix << "diff_max=" << join(diff, "%.3e") << ' ' << prefix << "ratio_max=" << join(ratio, "%.3e")
         << ' ' << prefix << "rel_max=" << join(rel, "%.3e") << ' ' << prefix << "limit_max=" << join(limit, "%.3e")
         << ' ' << prefix << "zero_bound_mismatch=" << mismatch << ' ' << prefix << "nonfinite=" << nonfinite
         << ' ' << prefix << "padding_zero=" << padding_ok << ' ' << prefix << "bound_check="
         << ((worst <= 1.0 && mismatch == 0 && nonfinite == 0 && padding_ok) ? "pass" : "FAIL");
}

cand::GeometryView cand_view(const rpi::Geometry& g) {
    cand::GeometryView v;
    v.nb = g.nb; v.m = g.m; v.L = g.L; v.L_op = g.L_op; v.C = g.C; v.C_op = g.C_op;
    v.cell_ptr = g.cell_ptr; v.op_seg = g.op_seg.data(); v.seg_inv = g.seg_inv.data(); v.mult = g.mult;
    for (int j = 0; j < g.nb; ++j) {
        v.cc[j] = g.cc[static_cast<std::size_t>(j)];
        auto& d = v.dims[j];
        d.levels = g.offset[static_cast<std::size_t>(j) + 1U] - g.offset[static_cast<std::size_t>(j)];
        d.offset = g.offset[static_cast<std::size_t>(j)];
        d.transposed = g.large[static_cast<std::size_t>(j)] != 0;
        if (d.transposed) {
            const auto& t = g.tr[static_cast<std::size_t>(j)];
            d.tr_ptr = t.ptr.data();
            d.tr_seg = t.seg.data();
            for (int jj = 0; jj < g.nb; ++jj) d.tr_other[jj] = jj == j ? nullptr : t.other[static_cast<std::size_t>(jj)].data();
            d.has_mult = t.has_mult;
            d.tr_mult = t.has_mult ? t.mult.data() : nullptr;
        }
    }
    return v;
}

const char* mode_name(cand::WidthMode m) { return m == cand::WidthMode::Padded ? "padded" : m == cand::WidthMode::Sub8 ? "sub8" : "inwarp"; }
const char* transfer_name(cand::Transfer t) {
    return t == cand::Transfer::Pageable ? "pageable" : t == cand::Transfer::Pinned ? "pinned" : t == cand::Transfer::PinnedChunks ? "chunks" : "registered";
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        o = parse(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "reduced_cuda_bench: %s\n", e.what());
        return 2;
    }
    std::ostringstream line;
    line.setf(std::ios::fixed);
    line.precision(3);
    const auto& s = o.shape;
    line << "REDUCED_CUDA_BENCH shape=" << s.name;
    if (!o.label.empty()) line << " label=" << o.label;
    line << " rows=" << s.rows << " levels=" << join(s.levels, "%d") << " skew=" << s.skew << " tail=" << s.tail << " move=" << s.move
         << " sorted=" << (s.sorted ? 1 : 0) << " dup=" << s.dup << " long_segment=" << s.long_segment_rows
         << " single_cell=" << s.single_cell_segment << " left_out=" << s.left_out_segment << " stayer=" << s.stayer_level
         << " unused_levels=" << s.unused_levels << " disconnected=" << s.disconnected << " long_level_share=" << s.long_level_share
         << " lengths=" << (s.length_law == xhdfe_bench::Shape::LengthLaw::MeanSpread ? std::string("spread")
                            : s.length_law == xhdfe_bench::Shape::LengthLaw::Uniform
                                ? "uniform:" + std::to_string(s.len_lo) + ":" + std::to_string(s.len_hi)
                                : "mix:" + std::to_string(s.len_p) + ":" + std::to_string(s.len_lo) + ":" + std::to_string(s.len_mid) +
                                      ":" + std::to_string(s.len_hi))
         << " seed=" << s.seed << " lanes=" << o.lanes << " width=" << o.width << " threads=" << o.threads
         << " merge=" << o.merge << " effect_class=" << o.effect_class << " load=" << load_average();
    try {
        auto t0 = Clock::now();
        const xhdfe_bench::Codes codes = xhdfe_bench::generate(s);
        const double generate_s = since(t0);
        const int k = static_cast<int>(codes.codes.size());
        const int n = codes.n;
        std::vector<const int*> code_ptrs(static_cast<std::size_t>(k));
        for (int b = 0; b < k; ++b) code_ptrs[static_cast<std::size_t>(b)] = codes.codes[static_cast<std::size_t>(b)].data();
        rp::Problem p;
        p.n = n;
        p.k = k;
        p.codes = code_ptrs.data();
        p.levels = codes.levels.data();
        p.eliminated = 0;
        p.cols = o.width;
        rp::Config cfg;
        cfg.threads = o.threads;
        cfg.blocks = o.blocks > 0 ? o.blocks : std::max(1, hdfe::detail::deterministic_parallel_chunk_count(n));
        cfg.memory_budget = o.budget ? o.budget : rp::admission_budget(n, o.width, k);
        cfg.force_merge = o.merge;
        cfg.force_effect_class = o.effect_class;
        const rpi::Team team(cfg.threads, cfg.hooks, 8 * std::max(1, cfg.blocks));
        rpi::Geometry g;
        std::size_t cpu_bytes = 0;
        t0 = Clock::now();
        const rp::Decline declined = rpi::build_geometry(p, cfg, team, o.lanes, g, &cpu_bytes);
        const double geometry_s = since(t0);
        line << " blocks=" << cfg.blocks << " budget=" << cfg.memory_budget << " generate_s=" << generate_s
             << " geometry_s=" << geometry_s << " cpu_bound_bytes=" << cpu_bytes;
        if (declined != rp::Decline::None) {
            line << " geometry=declined_" << rp::decline_name(declined);
            std::printf("%s\n", line.str().c_str());
            return 3;
        }
        std::string classes;
        for (int j = 0; j < g.nb; ++j) classes += g.large[static_cast<std::size_t>(j)] ? 'T' : 'S';
        const int op_parts = static_cast<int>(g.op_bounds.size()) - 1;
        const LevelCounts lc = level_counts(g);
        std::vector<long long> empty, emax, minv0;
        for (int j = 0; j < g.nb; ++j) {
            long long e = 0, mx = 0, z = 0;
            for (int i = g.offset[static_cast<std::size_t>(j)]; i < g.offset[static_cast<std::size_t>(j) + 1U]; ++i) {
                e += lc.entries[static_cast<std::size_t>(i)] == 0;
                mx = std::max(mx, lc.entries[static_cast<std::size_t>(i)]);
                z += g.minv[static_cast<std::size_t>(i)] == 0.0;
            }
            empty.push_back(e);
            emax.push_back(mx);
            minv0.push_back(z);
        }
        line << " classes=" << classes << " sorted_geometry=" << g.sorted << " merged=" << g.merged << " C=" << g.C
             << " C_op=" << g.C_op << " L=" << g.L << " L_op=" << g.L_op << " m=" << g.m << " op_parts=" << op_parts
             << " cmax=" << lc.cmax << " seg_over16=" << lc.seg_over16 << " cells_over16=" << lc.cells_over16
             << " empty_levels=" << join(empty, "%lld") << " max_level_entries=" << join(emax, "%lld")
             << " minv_zero=" << join(minv0, "%lld") << " host_geometry_bytes=" << gpu::geometry_resident_bytes(g);

        // The operand: the live lanes uniform in [-1, 1), the padding lanes +0.0.
        const int W = o.lanes, width = o.width;
        const std::size_t count = static_cast<std::size_t>(g.m) * W;
        std::vector<double> in(count, 0.0);
        xhdfe_bench::SplitMix64 rng(o.operand_seed);
        for (int i = 0; i < g.m; ++i)
            for (int w = 0; w < width; ++w) in[static_cast<std::size_t>(i) * W + w] = 2.0 * rng.uniform() - 1.0;

        std::vector<double> cpu_out(count, 0.0), cpu_second(count, 0.0);
        bool cpu_ran = false;
        if (has_arm(o, "cpu")) {
            bool equal = false;
            const Stat c = cpu_dispatch(W, g, team, in.data(), cpu_out.data(), cpu_second.data(), o.warmup, o.products, equal);
            cpu_ran = true;
            line << " cpu_med_ms=" << c.median << " cpu_min_ms=" << c.min << " cpu_n=" << c.count << " cpu_bit_equal=" << equal;
        }
        if (has_arm(o, "cpuw") && width != W) {
            const std::size_t cw = static_cast<std::size_t>(g.m) * width;
            std::vector<double> in_w(cw), out_w(cw), second_w(cw);
            for (int i = 0; i < g.m; ++i)
                for (int w = 0; w < width; ++w) in_w[static_cast<std::size_t>(i) * width + w] = in[static_cast<std::size_t>(i) * W + w];
            bool equal = false;
            const Stat c = cpu_dispatch(width, g, team, in_w.data(), out_w.data(), second_w.data(), o.warmup, o.products, equal);
            line << " cpuw_med_ms=" << c.median << " cpuw_min_ms=" << c.min << " cpuw_bit_equal=" << equal;
        }
        if (has_arm(o, "xfer")) {
            const cand::TransferResult x = cand::measure_transfers(g.m, W, width, o.xfer_reps, o.cand.chunk_bytes);
            line.precision(4);
            line << " xfer_bytes_padded=" << count * 8 << " xfer_bytes_packed=" << static_cast<std::size_t>(g.m) * width * 8
                 << " xfer_pinned_alloc_s=" << x.pinned_alloc_s << " xfer_pinned_free_s=" << x.pinned_free_s
                 << " xfer_register_s=" << x.register_s << " xfer_unregister_s=" << x.unregister_s;
            line.precision(3);
            line << " xfer_pageable_h2d_ms=" << x.pageable_h2d_ms << " xfer_pageable_d2h_ms=" << x.pageable_d2h_ms
                 << " xfer_pinned_full_h2d_ms=" << x.pinned_full_h2d_ms << " xfer_pinned_full_d2h_ms=" << x.pinned_full_d2h_ms
                 << " xfer_pinned_packed_h2d_ms=" << x.pinned_h2d_ms << " xfer_pinned_packed_d2h_ms=" << x.pinned_d2h_ms
                 << " xfer_cycle_ms=" << x.cycle_ms << " xfer_cycle_pack_ms=" << x.cycle_pack_ms << " xfer_cycle_unpack_ms=" << x.cycle_unpack_ms
                 << " xfer_chunked_cycle_ms=" << x.chunked_cycle_ms << " xfer_registered_h2d_ms=" << x.registered_h2d_ms
                 << " xfer_registered_d2h_ms=" << x.registered_d2h_ms << " xfer_registered_cycle_ms=" << x.registered_cycle_ms
                 << " xfer_registered2d_h2d_ms=" << x.registered2d_h2d_ms << " xfer_registered2d_d2h_ms=" << x.registered2d_d2h_ms
                 << " xfer_registered2d_cycle_ms=" << x.registered2d_cycle_ms << " xfer_registered2d_zero_ms=" << x.registered2d_zero_ms;
        }
        if (has_arm(o, "gpu")) {
            if (W != 1 && W != 2 && W != 4 && W != 8) {
                line << " gpu=unsupported_lanes";
            } else {
                std::vector<double> gpu_out(count, 0.0);
                const GpuResult r = gpu_products(g, W, cfg.memory_budget, in.data(), gpu_out.data(), o.warmup, o.products);
                if (!r.error.empty()) {
                    std::string e = r.error;
                    std::replace(e.begin(), e.end(), ' ', '_');
                    line << " gpu_error=" << e;
                } else {
                    line.precision(4);
                    line << " layout_s=" << r.layout_s << " layout_zero_s=" << r.layout_split.zero_seconds
                         << " layout_copy_s=" << r.layout_split.copy_seconds << " layout_sort_s=" << r.layout_split.sort_seconds
                         << " layout_items_s=" << r.layout_split.items_seconds << " setup_s=" << r.setup_s
                         << " validate_s=" << r.setup_split.validate_seconds << " device_init_s=" << r.setup_split.device_seconds
                         << " device_alloc_s=" << r.setup_split.alloc_seconds << " upload_copy_s=" << r.setup_split.copy_seconds
                         << " upload_sync_s=" << r.setup_split.sync_seconds << " layout_free_s=" << r.layout_free_s
                         << " release_s=" << r.release_s;
                    line.precision(3);
                    line << " payload_bytes=" << r.payload << " uploaded_bytes=" << r.uploaded
                         << " host_layout_bytes=" << r.host_layout_bytes << " host_layout_peak=" << r.host_layout_peak
                         << " a_items=" << r.a_items << " b_items=" << r.b_items
                         << " gpu_kernel_med_ms=" << r.kernel.median << " gpu_kernel_min_ms=" << r.kernel.min
                         << " segment_items_med_ms=" << r.seg_items.median << " segment_items_min_ms=" << r.seg_items.min
                         << " segment_finish_med_ms=" << r.seg_finish.median << " segment_finish_min_ms=" << r.seg_finish.min
                         << " level_items_med_ms=" << r.lvl_items.median << " level_items_min_ms=" << r.lvl_items.min
                         << " level_finish_med_ms=" << r.lvl_finish.median << " level_finish_min_ms=" << r.lvl_finish.min
                         << " h2d_med_ms=" << r.h2d.median << " d2h_med_ms=" << r.d2h.median
                         << " apply_med_ms=" << r.apply.median << " apply_min_ms=" << r.apply.min << " gpu_n=" << r.kernel.count
                         << " gpu_bit_equal=" << r.bit_equal;
                    if (cpu_ran)
                        emit_check(line, "", check_lanes(g, lc, in.data(), cpu_out.data(), W, gpu_out.data(), W, W, width, op_parts),
                                   padding_zero(cpu_out.data(), g.m, W, width) && padding_zero(gpu_out.data(), g.m, W, width));
                }
            }
        }
        // The candidate's first product, kept for the bit-for-bit comparison with prod
        // when the candidate runs prod's structure. The register cache does not enter the
        // condition: a cached segment and a two-walk one add the same values in the same
        // order. Nor does the segment order, which the line records (cand_seg_order):
        // prod takes K1's segments grouped by length since the K1 iteration (17 November),
        // so a candidate in the length order must equal prod everywhere, and one in the
        // natural order only where the length order is the identity (every segment of
        // one length); elsewhere the order moves segments between blocks and groups and
        // the private sums change order.
        std::vector<double> cand_out;
        const auto& oc = o.cand;
        const bool cand_like_prod =
            oc.width_mode == cand::WidthMode::InWarp && oc.transfer == cand::Transfer::Registered &&
            oc.seg_per_group == gpu::kSegPerGroup && oc.tile == gpu::kTile && oc.sort_units && oc.private_max == o.prod.private_max &&
            oc.short_seg == o.prod.short_seg && oc.short_level == o.prod.short_level && oc.item == o.prod.item &&
            static_cast<int>(oc.b_method) == static_cast<int>(o.prod.b_method);
        if (has_arm(o, "cand")) {
            const auto& c = o.cand;
            line << " cand_width_mode=" << mode_name(c.width_mode) << " cand_transfer=" << transfer_name(c.transfer)
                 << " cand_short_seg=" << c.short_seg << " cand_short_level=" << c.short_level << " cand_item=" << c.item
                 << " cand_seg_per_group=" << c.seg_per_group << " cand_tile=" << c.tile << " cand_sort_units=" << c.sort_units
                 << " cand_k1_cache=" << c.k1_cache << " cand_staged_uploads=" << c.staged_uploads
                 << " cand_seg_order=" << (c.seg_order == 1 ? "length" : "natural");
            try {
                cand::Options co = c;
                co.budget = cfg.memory_budget;
                auto op = std::make_unique<cand::Operator>(cand_view(g), co);
                const cand::Stats& st0 = op->stats();
                line.precision(4);
                line << " cand_classes=" << st0.classes << " cand_setup_s=" << st0.setup_total_s << " cand_upload_s=" << st0.upload_s
                     << " cand_uploaded_bytes=" << st0.uploaded_bytes << " cand_alloc_s=" << st0.alloc_s << " cand_b_build_s=" << st0.b_build_s
                     << " cand_b_steps_ms=" << join(std::vector<double>(st0.b_steps_ms, st0.b_steps_ms + 6), "%.2f")
                     << " cand_units_s=" << st0.units_s << " cand_pinned_s=" << st0.pinned_s << " cand_seg_order_s=" << st0.seg_order_s;
                line.precision(3);
                line << " cand_units=" << join(std::vector<long long>(st0.units, st0.units + g.nb), "%lld")
                     << " cand_pb_slots=" << join(std::vector<long long>(st0.pb_slots, st0.pb_slots + g.nb), "%lld")
                     << " cand_long_levels=" << join(std::vector<int>(st0.long_levels, st0.long_levels + g.nb), "%d")
                     << " cand_long_segments=" << st0.long_segments << " cand_private_levels=" << st0.private_levels
                     << " cand_k1_blocks=" << st0.k1_blocks << " cand_pa_slots=" << st0.pa_slots
                     << " cand_k1_regs=" << st0.k1_regs << " cand_k1_local=" << st0.k1_local << " cand_k1_smem=" << st0.k1_smem
                     << " cand_k1_blocks_per_sm=" << st0.k1_blocks_per_sm << " cand_k2_regs=" << st0.k2_regs
                     << " cand_k2_local=" << st0.k2_local << " cand_k2_blocks_per_sm=" << st0.k2_blocks_per_sm;
                std::string adm = op->admission().text();
                std::replace(adm.begin(), adm.end(), ' ', '|');
                line << " cand_admission=" << adm;
                std::vector<double> out1(count), out2(count), scratch(count);
                // As in the controller, p and q may be neighbouring slices of one arena, 64-byte aligned
                // but not page aligned: then they share a page.
                std::vector<double> arena;
                const double* in_p = in.data();
                double* scratch_p = scratch.data();
                if (o.registered_adjacent) {
                    arena.assign(2 * count + 16, 0.0);
                    std::size_t offset = 0;
                    while ((reinterpret_cast<std::uintptr_t>(arena.data() + offset) % 4096) == 0 ||
                           (reinterpret_cast<std::uintptr_t>(arena.data() + offset) % 64) != 0) ++offset;
                    std::copy(in.begin(), in.end(), arena.begin() + static_cast<std::ptrdiff_t>(offset));
                    in_p = arena.data() + offset;
                    scratch_p = arena.data() + offset + count;
                    line << " cand_registered_adjacent=1 cand_p_page_offset=" << (reinterpret_cast<std::uintptr_t>(in_p) % 4096)
                         << " cand_q_page_offset=" << (reinterpret_cast<std::uintptr_t>(scratch_p) % 4096);
                }
                if (c.transfer == cand::Transfer::Registered) op->register_host(in_p, scratch_p);
                std::vector<double> apply, h2d, k1, k1l, k2, k3, d2h, pack, unpack;
                op->begin_group(width);
                for (int it = 0; it < o.warmup + o.products; ++it) {
                    double* dst = scratch_p;
                    const cand::Stats before = op->stats();
                    op->apply(in_p, dst);
                    const cand::Stats& after = op->stats();
                    if (it == 0) std::copy(scratch_p, scratch_p + count, out1.begin());
                    if (it == 1) std::copy(scratch_p, scratch_p + count, out2.begin());
                    if (it < o.warmup) continue;
                    apply.push_back(after.apply_ms - before.apply_ms);
                    h2d.push_back(after.h2d_ms - before.h2d_ms);
                    k1.push_back(after.k1_ms - before.k1_ms);
                    k1l.push_back(after.k1l_ms - before.k1l_ms);
                    k2.push_back(after.k2_ms[0] - before.k2_ms[0]);
                    k3.push_back(after.k3_ms - before.k3_ms);
                    d2h.push_back(after.d2h_ms - before.d2h_ms);
                    pack.push_back(after.pack_ms - before.pack_ms);
                    unpack.push_back(after.unpack_ms - before.unpack_ms);
                }
                std::vector<double> kern;
                for (std::size_t i = 0; i < k1.size(); ++i) kern.push_back(k1[i] + k1l[i] + k2[i] + k3[i]);
                line << " cand_run_k1_smem=" << op->stats().k1_smem << " cand_run_k1_blocks_per_sm=" << op->stats().k1_blocks_per_sm;
                const Stat sa = stat_of(apply), sk = stat_of(kern);
                line << " cand_kernel_med_ms=" << sk.median << " cand_kernel_min_ms=" << sk.min
                     << " cand_k1_med_ms=" << stat_of(k1).median << " cand_k1l_med_ms=" << stat_of(k1l).median
                     << " cand_k2_med_ms=" << stat_of(k2).median << " cand_k2_min_ms=" << stat_of(k2).min
                     << " cand_k3_med_ms=" << stat_of(k3).median << " cand_k1_min_ms=" << stat_of(k1).min
                     << " cand_h2d_med_ms=" << stat_of(h2d).median << " cand_d2h_med_ms=" << stat_of(d2h).median
                     << " cand_pack_med_ms=" << stat_of(pack).median << " cand_unpack_med_ms=" << stat_of(unpack).median
                     << " cand_apply_med_ms=" << sa.median << " cand_apply_min_ms=" << sa.min
                     << " cand_bit_equal=" << (std::memcmp(out1.data(), out2.data(), count * sizeof(double)) == 0);
                if (cand_like_prod) cand_out = out1;
                if (cpu_ran)
                    emit_check(line, "cand_", check_lanes(g, lc, in.data(), cpu_out.data(), W, out1.data(), W, W, width, op_parts),
                               padding_zero(out1.data(), g.m, W, width));
                // A -> B -> A: widths width, aba_width, width; outputs pre-filled with a sentinel. Registered
                // transfers register each call's pair (and release it) around the call.
                if (c.transfer == cand::Transfer::Registered) op->unregister();
                {
                    const int wb = std::min(o.aba_width, W);
                    std::uint64_t sentinel_bits = 0x7ff8deadbeef0001ULL;
                    double sentinel;
                    std::memcpy(&sentinel, &sentinel_bits, sizeof(sentinel));
                    std::vector<double> a1(count, sentinel), bb(count, sentinel), a2(count, sentinel);
                    std::vector<double> in_b(in);
                    for (int i = 0; i < g.m; ++i)
                        for (int w = wb; w < W; ++w) in_b[static_cast<std::size_t>(i) * W + w] = 0.0;  // B's padding is zero in the controller
                    const bool reg = c.transfer == cand::Transfer::Registered;
                    auto call = [&](int w, const std::vector<double>& x, std::vector<double>& y) {
                        if (reg) op->register_host(x.data(), y.data());
                        op->begin_group(w);
                        op->apply(x.data(), y.data());
                        if (reg) op->unregister();
                    };
                    call(width, in, a1);
                    call(wb, in_b, bb);
                    call(width, in, a2);
                    auto sentinel_left = [&](const std::vector<double>& v) {
                        for (double x : v) {
                            std::uint64_t bits = 0;
                            std::memcpy(&bits, &x, sizeof(bits));
                            if (bits == sentinel_bits) return true;
                        }
                        return false;
                    };
                    const bool aba_same = std::memcmp(a1.data(), a2.data(), count * sizeof(double)) == 0 &&
                                          std::memcmp(a1.data(), out1.data(), count * sizeof(double)) == 0;
                    line << " cand_aba_width=" << wb << " cand_aba_bit_equal=" << aba_same
                         << " cand_aba_sentinel_left=" << (sentinel_left(a1) || sentinel_left(bb) || sentinel_left(a2));
                    if (cpu_ran)
                        emit_check(line, "cand_aba_b_", check_lanes(g, lc, in.data(), cpu_out.data(), W, bb.data(), W, W, wb, op_parts),
                                   padding_zero(bb.data(), g.m, W, wb));
                }
                // Device-built level-ordered arrays against the host reference (B5), every array.
                std::string bcheck;
                for (int j = 0; j < g.nb; ++j) {
                    if (op->dim_class(j) != 'B') continue;
                    const auto dev = op->download_level_order(j);
                    const xhdfe_bench::LevelOrder ref = xhdfe_bench::build_level_order(
                        g, j, o.levelref_parts, [&](int units, auto&& body) { team.run(units, body); });
                    bool same = dev.ptr == ref.ptr && xhdfe_bench::same_ints(dev.seg.data(), dev.seg.size(), ref.seg.get(), ref.seg.size) &&
                                xhdfe_bench::same_ints(dev.cell.data(), dev.cell.size(), ref.cell.get(), ref.cell.size) &&
                                xhdfe_bench::same_ints(dev.mult.data(), dev.mult.size(), ref.mult.get(), ref.mult.size);
                    for (int jj = 0; jj < g.nb; ++jj)
                        if (jj != j)
                            same = same && xhdfe_bench::same_ints(dev.other[static_cast<std::size_t>(jj)].data(), dev.other[static_cast<std::size_t>(jj)].size(),
                                                                  ref.other[static_cast<std::size_t>(jj)].get(), ref.other[static_cast<std::size_t>(jj)].size);
                    bcheck += (bcheck.empty() ? "" : ",") + std::to_string(j) + (same ? ":identical" : ":DIFFERENT");
                }
                if (!bcheck.empty()) line << " cand_b_identity=" << bcheck;
                const auto t_rel = Clock::now();
                op.reset();
                line << " cand_teardown_s=" << since(t_rel);
            } catch (const std::exception& e) {
                std::string msg = e.what();
                std::replace(msg.begin(), msg.end(), ' ', '_');
                line << " cand_error=" << msg;
            }
        }
        if (has_arm(o, "prod")) {
            const gpu::Options& po = o.prod;
            line << " prod_private_max=" << po.private_max << " prod_short_seg=" << po.short_seg << " prod_short_level=" << po.short_level
                 << " prod_item=" << po.item
                 << " prod_b=" << (po.b_method == gpu::BMethod::Auto ? "auto" : po.b_method == gpu::BMethod::Tiles ? "tiles" : "radix")
                 << " prod_cells=" << o.prod_cells;
            try {
                gpu::Options popt = po;
                popt.budget = cfg.memory_budget;
                popt.profile = true;
                popt.test_source_cells = o.prod_cells;
                popt.run = [&team](int units, const std::function<void(int)>& body) { team.run(units, body); };
                const auto t_setup = Clock::now();
                auto op = std::make_unique<gpu::Operator>(gpu::view_of(g), popt);
                const double setup_wall = since(t_setup);
                const gpu::Statistics& st0 = op->statistics();
                const gpu::Admission& adm = op->admission();
                line.precision(4);
                line << " prod_classes=" << st0.classes << " prod_setup_s=" << setup_wall << " prod_setup_device_s=" << st0.device_seconds
                     << " prod_setup_plan_s=" << st0.plan_seconds << " prod_setup_staging_s=" << st0.staging_seconds
                     << " prod_setup_upload_s=" << st0.upload_seconds << " prod_setup_alloc_s=" << st0.alloc_seconds
                     << " prod_setup_build_s=" << st0.build_seconds << " prod_setup_units_s=" << st0.units_seconds;
                line.precision(3);
                line << " prod_uploaded_bytes=" << st0.uploaded_bytes << " prod_private_levels=" << st0.private_levels
                     << " prod_long_segments=" << st0.long_segments << " prod_long_levels=" << st0.long_levels << " prod_units=" << st0.units
                     << " prod_pb_slots=" << st0.pb_slots << " prod_b_radix=" << st0.b_radix
                     << " prod_device_setup_peak=" << adm.device_setup_peak << " prod_device_iteration_peak=" << adm.device_iteration_peak
                     << " prod_allocated_peak=" << adm.allocated_peak << " prod_allocated_iteration=" << adm.allocated_iteration
                     << " prod_device_free_entry=" << adm.free_at_entry << " prod_device_free_peak=" << adm.free_at_peak
                     << " prod_device_free_after=" << adm.free_after_setup << " prod_cub_scan=" << adm.cub_scan_bytes
                     << " prod_cub_radix=" << adm.cub_radix_bytes << " prod_host_setup=" << adm.host_setup
                     << " prod_host_setup_real=" << adm.host_setup_real;
                // p and q as solve() holds them: adjacent slices of one 64-byte (not page) aligned arena.
                rpi::Buffer<double> arena;
                arena.allocate(2 * count);
                double* p = arena.data();
                double* q = p + count;
                std::copy(in.begin(), in.end(), p);
                std::fill(q, q + count, 0.0);
                const auto t_reg = Clock::now();
                op->register_host(p, 2 * count);
                const double reg_s = since(t_reg);
                op->begin_group(width);
                std::vector<double> out1(count), out2(count);
                std::vector<double> apply, h2d, k1, k1l, k2, k3, d2h, pack, unpack, wall;
                for (int it = 0; it < o.warmup + o.products; ++it) {
                    const gpu::Statistics before = op->statistics();
                    const auto t = Clock::now();
                    op->apply(p, q);
                    const double w_ms = 1e3 * since(t);
                    const gpu::Statistics& after = op->statistics();
                    if (it == 0) std::copy(q, q + count, out1.begin());
                    if (it == 1) std::copy(q, q + count, out2.begin());
                    if (it < o.warmup) continue;
                    wall.push_back(w_ms);
                    h2d.push_back(after.h2d_dma_ms - before.h2d_dma_ms);
                    pack.push_back(after.pack_ms - before.pack_ms);
                    k1.push_back(after.k1_ms - before.k1_ms);
                    k1l.push_back(after.k1l_ms - before.k1l_ms);
                    k2.push_back(after.k2_ms - before.k2_ms);
                    k3.push_back(after.k3_ms - before.k3_ms);
                    unpack.push_back(after.unpack_ms - before.unpack_ms);
                    d2h.push_back(after.d2h_dma_ms - before.d2h_dma_ms);
                }
                std::vector<double> kern;
                for (std::size_t i = 0; i < k1.size(); ++i) kern.push_back(k1[i] + k1l[i] + k2[i] + k3[i]);
                const Stat sw = stat_of(wall), sk = stat_of(kern);
                line << " prod_register_s=" << reg_s << " prod_kernel_med_ms=" << sk.median << " prod_kernel_min_ms=" << sk.min
                     << " prod_k1_med_ms=" << stat_of(k1).median << " prod_k1l_med_ms=" << stat_of(k1l).median
                     << " prod_k2_med_ms=" << stat_of(k2).median << " prod_k3_med_ms=" << stat_of(k3).median
                     << " prod_h2d_med_ms=" << stat_of(h2d).median << " prod_pack_med_ms=" << stat_of(pack).median
                     << " prod_unpack_med_ms=" << stat_of(unpack).median << " prod_d2h_med_ms=" << stat_of(d2h).median
                     << " prod_apply_med_ms=" << sw.median << " prod_apply_min_ms=" << sw.min << " prod_n=" << sk.count
                     << " prod_bit_equal=" << (std::memcmp(out1.data(), out2.data(), count * sizeof(double)) == 0);
                if (!cand_out.empty())
                    line << " prod_vs_cand_bit_equal=" << (std::memcmp(out1.data(), cand_out.data(), count * sizeof(double)) == 0);
                if (cpu_ran)
                    emit_check(line, "prod_", check_lanes(g, lc, in.data(), cpu_out.data(), W, out1.data(), W, W, width, op_parts),
                               padding_zero(out1.data(), g.m, W, width));
                // A -> B -> A: widths width, aba_width, width; q pre-filled with a sentinel each time.
                {
                    const int wb = std::min(o.aba_width, W);
                    std::uint64_t sentinel_bits = 0x7ff8deadbeef0001ULL;
                    double sentinel;
                    std::memcpy(&sentinel, &sentinel_bits, sizeof(sentinel));
                    std::vector<double> in_b(in);
                    for (int i = 0; i < g.m; ++i)
                        for (int w = wb; w < W; ++w) in_b[static_cast<std::size_t>(i) * W + w] = 0.0;  // the controller zeroes them
                    auto call = [&](int w, const std::vector<double>& x) {
                        std::copy(x.begin(), x.end(), p);
                        std::fill(q, q + count, sentinel);
                        op->begin_group(w);
                        op->apply(p, q);
                        return std::vector<double>(q, q + count);
                    };
                    const std::vector<double> a1 = call(width, in), bb = call(wb, in_b), a2 = call(width, in);
                    auto sentinel_left = [&](const std::vector<double>& v) {
                        for (double x : v) {
                            std::uint64_t bits = 0;
                            std::memcpy(&bits, &x, sizeof(bits));
                            if (bits == sentinel_bits) return true;
                        }
                        return false;
                    };
                    const bool aba_same = std::memcmp(a1.data(), a2.data(), count * sizeof(double)) == 0 &&
                                          std::memcmp(a1.data(), out1.data(), count * sizeof(double)) == 0;
                    line << " prod_aba_width=" << wb << " prod_aba_bit_equal=" << aba_same
                         << " prod_aba_sentinel_left=" << (sentinel_left(a1) || sentinel_left(bb) || sentinel_left(a2));
                    if (cpu_ran)
                        emit_check(line, "prod_aba_b_", check_lanes(g, lc, in.data(), cpu_out.data(), W, bb.data(), W, W, wb, op_parts),
                                   padding_zero(bb.data(), g.m, W, wb));
                }
                const auto t_unreg = Clock::now();
                op->unregister_host();
                line << " prod_unregister_s=" << since(t_unreg);
                // Device-built level-ordered arrays against the host reference (the source cells with --prod-cells 1).
                std::string bcheck;
                for (int j = 0; j < g.nb; ++j) {
                    if (op->effect_class(j) != 'B') continue;
                    const auto dev = op->download_level_order(j);
                    const xhdfe_bench::LevelOrder ref = xhdfe_bench::build_level_order(
                        g, j, o.levelref_parts, [&](int units, auto&& body) { team.run(units, body); });
                    bool same = dev.ptr == ref.ptr && xhdfe_bench::same_ints(dev.seg.data(), dev.seg.size(), ref.seg.get(), ref.seg.size) &&
                                xhdfe_bench::same_ints(dev.mult.data(), dev.mult.size(), ref.mult.get(), ref.mult.size) &&
                                (!o.prod_cells || xhdfe_bench::same_ints(dev.cell.data(), dev.cell.size(), ref.cell.get(), ref.cell.size));
                    for (int jj = 0; jj < g.nb; ++jj)
                        if (jj != j)
                            same = same && xhdfe_bench::same_ints(dev.other[static_cast<std::size_t>(jj)].data(), dev.other[static_cast<std::size_t>(jj)].size(),
                                                                  ref.other[static_cast<std::size_t>(jj)].get(), ref.other[static_cast<std::size_t>(jj)].size);
                    bcheck += (bcheck.empty() ? "" : ",") + std::to_string(j) + (same ? ":identical" : ":DIFFERENT");
                }
                if (!bcheck.empty()) line << " prod_b_identity=" << bcheck;
                const auto t_rel = Clock::now();
                op.reset();
                line << " prod_teardown_s=" << since(t_rel);
            } catch (const std::exception& e) {
                std::string msg = e.what();
                std::replace(msg.begin(), msg.end(), ' ', '_');
                line << " prod_error=" << msg;
            }
        }
        if (has_arm(o, "levelref")) {
            std::vector<std::string> parts;
            bool all = true;
            for (int j = 0; j < g.nb; ++j) {
                t0 = Clock::now();
                const xhdfe_bench::LevelOrder par = xhdfe_bench::build_level_order(
                    g, j, o.levelref_parts, [&](int units, auto&& body) { team.run(units, body); });
                const double par_s = since(t0);
                t0 = Clock::now();
                const xhdfe_bench::LevelOrder ser = xhdfe_bench::build_level_order_serial(g, j);
                const double ser_s = since(t0);
                const bool same = xhdfe_bench::same_level_order(par, ser);
                const int vs_tr = g.large[static_cast<std::size_t>(j)] ? (xhdfe_bench::same_as_transpose(par, g, j) ? 1 : 0) : -1;
                all = all && same && vs_tr != 0;
                char buf[160];
                std::snprintf(buf, sizeof(buf), "%d:%.4f/%.4f/%s/%s", j, par_s, ser_s, same ? "serial_ok" : "SERIAL_DIFF",
                              vs_tr < 0 ? "no_transpose" : vs_tr ? "transpose_ok" : "TRANSPOSE_DIFF");
                parts.push_back(buf);
            }
            std::string joined;
            for (std::size_t i = 0; i < parts.size(); ++i) joined += (i ? "," : "") + parts[i];
            line << " levelref_parts=" << o.levelref_parts << " levelref=" << joined << " levelref_check=" << (all ? "pass" : "FAIL");
        }
    } catch (const std::exception& e) {
        std::string msg = e.what();
        std::replace(msg.begin(), msg.end(), ' ', '_');
        line << " error=" << msg;
        std::printf("%s\n", line.str().c_str());
        return 1;
    }
    std::printf("%s\n", line.str().c_str());
    return 0;
}
