// The CUDA operator of the reduced tier against the CPU operator (package 2, Amendment 2,
// C6; 17 November 2026). Built from src/reduced_schur_pcg_cuda.cu with
// XHDFE_REDUCED_CUDA_TEST_HOOKS (the source cells of the device-built arrays and the
// count of live device bytes exist only in such builds), with the library's CUDA flags.
// Synthetic geometries only. Checks, each printed:
//  - every live lane of every product within the operation-count bound of the CPU
//    product (Higham's gamma_K with K = 2 E_i + Cmax + nb + parts + 8; the bound comes
//    from the operation counts, never from the error seen), the padding lanes +0.0, two
//    calls bit for bit, at the real widths 8, 7, 6, 5, 3 (8 lanes), 4, 3 (4), 2, 1 (2), 1;
//  - A -> B -> A (width w, a narrower last group, w) with the result rows pre-filled
//    with sentinels: A's results identical, nothing left unwritten, B's padding +0.0;
//  - the device-built level-ordered arrays (B effects, tiles and radix) identical to the
//    host reference on ptr, seg, every other code, mult and the source cell;
//  - C_op = 0 (no operator segment, every level empty) gives +0.0 everywhere;
//  - K1's segment order (17 November): the same bits with the host parts of the set-up
//    run serially, in reverse order or on a team;
//  - package 3-lite: p -> x -> p through one registration of x, r, p and q;
//  - failures injected in the middle of the set-up (an allocation, the build, the
//    page-locked chunks) and at the registration: the kind (Memory or Pinned), nothing
//    left allocated or registered, no error left behind, and the process then computes
//    the same bits again; a one-byte budget, illegal widths, a product outside the
//    registered rows and a range already registered elsewhere are refused as they must be.
#include "reduced_schur_pcg.hpp"
#include "reduced_schur_pcg_cuda.hpp"
#include "reduced_schur_pcg_cuda_layout.hpp"
#include "level_order_reference.hpp"
#include "synthetic_problem.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rp = hdfe::reduced_pcg;
namespace rpi = hdfe::reduced_pcg::internal;
namespace gpu = hdfe::detail::reduced_cuda;

namespace {

int failures = 0, checks = 0;
void expect(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << '\n';
    }
}

bool zero_bits(double v) {
    std::uint64_t b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return b == 0;
}
const std::uint64_t kSentinelBits = 0x7ff8deadbeef0001ULL;
double sentinel() {
    double s;
    std::memcpy(&s, &kSentinelBits, sizeof(s));
    return s;
}
bool is_sentinel(double v) {
    std::uint64_t b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return b == kSentinelBits;
}

struct Problem {
    std::string name;
    std::vector<std::vector<int>> codes;
    std::vector<int> levels;
    int n = 0;
    int merge = -1, effect_class = -1;
};

Problem from_shape(const xhdfe_bench::Shape& s, int merge = -1, int effect_class = -1) {
    Problem p;
    const xhdfe_bench::Codes c = xhdfe_bench::generate(s);
    p.name = s.name;
    p.codes = c.codes;
    p.levels = c.levels;
    p.n = c.n;
    p.merge = merge;
    p.effect_class = effect_class;
    return p;
}

// The small shapes of the former smoke test (k = 2..5, a segment of 32 768 rows,
// forced merging and effect classes).
Problem old_shape(int shape) {
    Problem p;
    const int n = shape == 1 ? 65536 : 4096, k = shape == 5 ? 4 : shape == 6 ? 5 : shape == 2 ? 3 : 2;
    p.name = "old" + std::to_string(shape);
    p.n = n;
    p.codes.assign(static_cast<std::size_t>(k), std::vector<int>(static_cast<std::size_t>(n)));
    p.levels = k >= 3 ? std::vector<int>{256, 31, 7, 3, 2} : std::vector<int>{256, 100};
    p.levels.resize(static_cast<std::size_t>(k));
    for (int i = 0; i < n; ++i) {
        p.codes[0][static_cast<std::size_t>(i)] = shape == 1 ? (i < 32768 ? 0 : 1 + i % 255) : i / 16;
        p.codes[1][static_cast<std::size_t>(i)] = shape == 3 ? (i / 16) % 4 : shape == 4 ? i % 4 : i % 97;
        if (k >= 3)
            for (int j = 1; j < k; ++j) p.codes[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = (i * (j + 5) + i / 11) % p.levels[static_cast<std::size_t>(j)];
    }
    p.merge = shape == 1 ? 0 : 1;
    p.effect_class = shape % 2;
    return p;
}

struct Built {
    rpi::Geometry g;
    std::unique_ptr<rpi::Team> team;
    int op_parts = 0;
};

void build(const Problem& pr, int W, Built& b) {
    std::vector<const int*> ids(pr.codes.size());
    for (std::size_t j = 0; j < pr.codes.size(); ++j) ids[j] = pr.codes[j].data();
    rp::Problem p;
    p.n = pr.n;
    p.k = static_cast<int>(pr.codes.size());
    p.eliminated = 0;
    p.codes = ids.data();
    p.levels = pr.levels.data();
    p.cols = W;
    rp::Config cfg;
    cfg.threads = 8;
    cfg.blocks = 16;
    cfg.force_merge = pr.merge;
    cfg.force_effect_class = pr.effect_class;
    cfg.memory_budget = std::size_t{2} << 30;
    b.team = std::make_unique<rpi::Team>(cfg.threads, rp::Hooks{}, 128);
    std::size_t bytes = 0;
    if (rpi::build_geometry(p, cfg, *b.team, W, b.g, &bytes) != rp::Decline::None) throw std::runtime_error(pr.name + ": geometry declined");
    b.op_parts = static_cast<int>(b.g.op_bounds.size()) - 1;
}

template <int W>
void cpu_product(const Built& b, const double* in, double* out) {
    const rpi::Geometry& g = b.g;
    rpi::Work work;
    work.priv.allocate(static_cast<std::size_t>(b.op_parts) * static_cast<std::size_t>(g.small_levels) * W);
    work.partials.resize(static_cast<std::size_t>(g.nb));
    for (int j = 0; j < g.nb; ++j)
        if (g.large[static_cast<std::size_t>(j)]) work.partials[static_cast<std::size_t>(j)].allocate(static_cast<std::size_t>(g.tr[static_cast<std::size_t>(j)].items) * W);
    rpi::Buffer<double> T;
    T.allocate(static_cast<std::size_t>(g.L) * W);
    rpi::apply_operator<W>(g, *b.team, in, out, T.data(), work);
}
void cpu_dispatch(int W, const Built& b, const double* in, double* out) {
    switch (W) {
        case 1: return cpu_product<1>(b, in, out);
        case 2: return cpu_product<2>(b, in, out);
        case 4: return cpu_product<4>(b, in, out);
        default: return cpu_product<8>(b, in, out);
    }
}

// |device - CPU| <= 2 gamma_K B_i per live lane (see the header comment); returns the
// largest ratio to that limit, or a value > 1 on any failure.
double bound_ratio(const rpi::Geometry& g, int parts, const double* in, const double* a, const double* b, int W, int width) {
    const int m = g.m, nb = g.nb;
    std::vector<long long> entries(static_cast<std::size_t>(m), 0);
    int cmax = 0;
    for (int s = 0; s < g.L_op; ++s) {
        const int l = g.op_seg[static_cast<std::size_t>(s)];
        cmax = std::max(cmax, g.cell_ptr[l + 1] - g.cell_ptr[l]);
        for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c)
            for (int j = 0; j < nb; ++j) ++entries[static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)] + g.cc[static_cast<std::size_t>(j)][c])];
    }
    double worst = 0.0;
    const long double unit = std::ldexp(1.0L, -53);
    for (int w = 0; w < width; ++w) {
        std::vector<long double> bound(static_cast<std::size_t>(m), 0.0L);
        auto U = [&](int c) {
            long double u = 0.0L;
            for (int j = 0; j < nb; ++j)
                u += std::fabs(static_cast<long double>(in[static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)] + g.cc[static_cast<std::size_t>(j)][c]) * W + w]));
            return u;
        };
        for (int s = 0; s < g.L_op; ++s) {
            const int l = g.op_seg[static_cast<std::size_t>(s)];
            long double A = 0.0L;
            for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) A += (g.mult ? g.mult[c] : 1) * U(c);
            const long double M = A * static_cast<long double>(g.seg_inv[static_cast<std::size_t>(l)]);
            for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) {
                const long double t = (g.mult ? g.mult[c] : 1) * (U(c) + M);
                for (int j = 0; j < nb; ++j) bound[static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)] + g.cc[static_cast<std::size_t>(j)][c])] += t;
            }
        }
        for (int i = 0; i < m; ++i) {
            const double x = a[static_cast<std::size_t>(i) * W + w], y = b[static_cast<std::size_t>(i) * W + w];
            if (!rpi::finite(x) || !rpi::finite(y)) return 2.0;
            const double diff = std::fabs(x - y);
            const long double K = 2.0L * entries[static_cast<std::size_t>(i)] + cmax + nb + parts + 8;
            const long double limit = 2.0L * (K * unit / (1.0L - K * unit)) * bound[static_cast<std::size_t>(i)];
            if (limit == 0.0L) {
                if (diff != 0.0) return 2.0;
                continue;
            }
            worst = std::max(worst, static_cast<double>(diff / limit));
        }
    }
    return worst;
}

bool padding_zero(const double* v, int m, int W, int width) {
    for (int i = 0; i < m; ++i)
        for (int w = width; w < W; ++w)
            if (!zero_bits(v[static_cast<std::size_t>(i) * W + w])) return false;
    return true;
}

// The caller's rows as the controller holds them: p and q adjacent in one buffer, one
// registration of the covering range.
struct Rows {
    std::vector<double> buf;
    double* p = nullptr;
    double* q = nullptr;
    std::size_t count = 0;
    Rows(int m, int W) : buf(2U * static_cast<std::size_t>(m) * W, 0.0), count(static_cast<std::size_t>(m) * W) {
        p = buf.data();
        q = buf.data() + count;
    }
};

gpu::Options options(int W) {
    gpu::Options o;
    o.lanes = W;
    o.test_source_cells = true;
    return o;
}

std::vector<double> operand(int m, int W, int width, std::uint64_t seed) {
    std::vector<double> x(static_cast<std::size_t>(m) * W, 0.0);
    xhdfe_bench::SplitMix64 rng(seed);
    for (int i = 0; i < m; ++i)
        for (int w = 0; w < width; ++w) x[static_cast<std::size_t>(i) * W + w] = 2.0 * rng.uniform() - 1.0;
    return x;
}

// One device product of `x` at `width` into a sentinel-filled q; returns q.
std::vector<double> device_product(gpu::Operator& op, Rows& rows, const std::vector<double>& x, int width) {
    std::copy(x.begin(), x.end(), rows.p);
    std::fill(rows.q, rows.q + rows.count, sentinel());
    op.begin_group(width);
    op.apply(rows.p, rows.q);
    return std::vector<double>(rows.q, rows.q + rows.count);
}

std::string classes_of(const gpu::Operator& op, int nb) {
    std::string c;
    for (int j = 0; j < nb; ++j) c += op.effect_class(j);
    return c;
}

// Products at every real width of W, A -> B -> A, and the B arrays; returns the
// first product at full width (for the injections' reference).
std::vector<double> run_case(const Problem& pr, int W, const std::vector<int>& widths, gpu::Options o, const char* tag) {
    Built b;
    build(pr, W, b);
    const rpi::Geometry& g = b.g;
    o.lanes = W;
    gpu::Operator op(gpu::view_of(g), o);
    Rows rows(g.m, W);
    op.register_host(rows.p, 2U * rows.count);
    std::vector<double> first;
    std::ostringstream line;
    line << "OPERATOR_SMOKE " << pr.name << ' ' << tag << " W=" << W << " classes=" << classes_of(op, g.nb) << " C=" << g.C
         << " C_op=" << g.C_op << " L_op=" << g.L_op << " m=" << g.m << " merged=" << g.merged;
    double worst = 0.0;
    for (int width : widths) {
        const std::vector<double> x = operand(g.m, W, width, 1000003ULL * static_cast<std::uint64_t>(width) + 17);
        std::vector<double> ref(x.size(), 0.0);
        cpu_dispatch(W, b, x.data(), ref.data());
        const std::vector<double> q1 = device_product(op, rows, x, width);
        const std::vector<double> q2 = device_product(op, rows, x, width);
        const std::string at = pr.name + " " + tag + " W=" + std::to_string(W) + " width=" + std::to_string(width);
        expect(std::memcmp(q1.data(), q2.data(), q1.size() * sizeof(double)) == 0, at + ": two calls bit for bit");
        expect(padding_zero(q1.data(), g.m, W, width), at + ": padding lanes +0.0");
        const double ratio = bound_ratio(g, b.op_parts, x.data(), q1.data(), ref.data(), W, width);
        expect(ratio <= 1.0, at + ": within the operation-count bound (ratio " + std::to_string(ratio) + ")");
        worst = std::max(worst, ratio);
        if (first.empty() && width == W) first = q1;
        // A -> B -> A, B a narrower group (the last group of a fit), outputs pre-filled.
        if (width >= 2) {
            const int wb = width >= 3 ? width - 2 : 1;
            std::vector<double> xb = x;
            for (int i = 0; i < g.m; ++i)
                for (int w = wb; w < W; ++w) xb[static_cast<std::size_t>(i) * W + w] = 0.0;  // the controller zeroes them
            const std::vector<double> a1 = device_product(op, rows, x, width);
            const std::vector<double> bb = device_product(op, rows, xb, wb);
            const std::vector<double> a2 = device_product(op, rows, x, width);
            bool left = false;
            for (const auto* v : {&a1, &bb, &a2})
                for (double d : *v) left = left || is_sentinel(d);
            expect(!left, at + ": A -> B -> A, every entry written");
            expect(std::memcmp(a1.data(), q1.data(), q1.size() * sizeof(double)) == 0 &&
                       std::memcmp(a2.data(), q1.data(), q1.size() * sizeof(double)) == 0,
                   at + ": A -> B -> A, A identical");
            expect(padding_zero(bb.data(), g.m, W, wb), at + ": A -> B -> A, B's padding +0.0");
            const double rb = bound_ratio(g, b.op_parts, x.data(), bb.data(), ref.data(), W, wb);
            expect(rb <= 1.0, at + ": A -> B -> A, B within the bound");
        }
    }
    op.unregister_host();
    // The device-built arrays against the host reference, every array and the source cells.
    std::string bcheck;
    for (int j = 0; j < g.nb; ++j) {
        if (op.effect_class(j) != 'B') continue;
        const auto dev = op.download_level_order(j);
        const xhdfe_bench::LevelOrder ref = xhdfe_bench::build_level_order(g, j, 64, [&](int units, auto&& body) { b.team->run(units, body); });
        bool same = dev.ptr == ref.ptr && xhdfe_bench::same_ints(dev.seg.data(), dev.seg.size(), ref.seg.get(), ref.seg.size) &&
                    xhdfe_bench::same_ints(dev.cell.data(), dev.cell.size(), ref.cell.get(), ref.cell.size) &&
                    xhdfe_bench::same_ints(dev.mult.data(), dev.mult.size(), ref.mult.get(), ref.mult.size);
        for (int jj = 0; jj < g.nb; ++jj)
            if (jj != j)
                same = same && xhdfe_bench::same_ints(dev.other[static_cast<std::size_t>(jj)].data(), dev.other[static_cast<std::size_t>(jj)].size(),
                                                      ref.other[static_cast<std::size_t>(jj)].get(), ref.other[static_cast<std::size_t>(jj)].size);
        expect(same, pr.name + " " + tag + ": B arrays of effect " + std::to_string(j) + " identical to the host reference");
        bcheck += (bcheck.empty() ? "" : ",") + std::to_string(j) + (same ? ":identical" : ":DIFFERENT");
    }
    // What the device was asked for equals what was admitted: the set-up's peak exactly, the
    // iteration's at most (B's units are bounds before the build), exactly without a B effect.
    const gpu::Admission& adm = op.admission();
    bool built = false;
    for (int j = 0; j < g.nb; ++j) built = built || op.effect_class(j) == 'B';
    expect(adm.admitted && adm.allocated_peak == std::max(adm.device_setup_peak, adm.allocated_iteration) &&
               adm.allocated_iteration <= adm.device_iteration_peak && (built || adm.allocated_iteration == adm.device_iteration_peak) &&
               adm.allocated_iteration == op.statistics().device_payload_bytes && adm.host_setup_real <= adm.host_setup,
           pr.name + " " + tag + ": allocations equal to the admission");
    const gpu::Statistics& st = op.statistics();
    line << " admitted_peak=" << adm.device_peak << " allocated_peak=" << adm.allocated_peak << " b_radix=" << st.b_radix << " long_segments=" << st.long_segments << " long_levels=" << st.long_levels
         << " units=" << st.units << " calls=" << st.calls << " worst_ratio=" << worst << " b_identity=" << (bcheck.empty() ? "none" : bcheck);
    std::cout << line.str() << '\n';
    return first;
}

template <class F>
gpu::ErrorKind kind_of(F&& f, bool& thrown) {
    thrown = false;
    try {
        f();
    } catch (const gpu::Error& e) {
        thrown = true;
        return e.kind();
    }
    return gpu::ErrorKind::Fatal;
}

bool clean_device() {
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

}  // namespace

int main() {
    try {
        // 1. The former smoke shapes at every lane count and real width.
        const std::vector<std::vector<int>> widths = {{}, {1}, {2, 1}, {}, {4, 3}, {}, {}, {}, {8, 7, 6, 5, 3}};  // by W
        for (int shape = 0; shape < 7; ++shape)
            for (int W : {1, 2, 4, 8}) run_case(old_shape(shape), W, widths[static_cast<std::size_t>(W)], options(W), "auto");

        // 2. The adversarial geometries (300 000 rows; levels 33 000, 4 000, 300, 12) at
        // width 5 of 8 lanes, with A -> B -> A to 3; adv-class0 builds its 4 000-level
        // effect by the radix sort.
        for (const char* name : {"adv-unsorted", "adv-merged", "adv-single-cell", "adv-left-out", "adv-null-levels",
                                 "adv-disconnected", "adv-long-level", "adv-long-segment", "adv-class0", "adv-class1"}) {
            xhdfe_bench::Shape s;
            s.name = name;
            s.rows = 300000;
            s.levels = {33000, 4000, 300, 12};
            int merge = -1, effect_class = -1;
            const std::string n = name;
            if (n == "adv-unsorted") s.sorted = false;
            else if (n == "adv-merged") { s.dup = 0.3; merge = 1; }
            else if (n == "adv-single-cell") s.single_cell_segment = true;
            else if (n == "adv-left-out") s.left_out_segment = true;
            else if (n == "adv-null-levels") { s.unused_levels = 7; s.stayer_level = true; }
            else if (n == "adv-disconnected") s.disconnected = true;
            else if (n == "adv-long-level") s.long_level_share = 0.3;
            else if (n == "adv-long-segment") s.long_segment_rows = 5000;
            else if (n == "adv-class0") effect_class = 0;
            else effect_class = 1;
            run_case(from_shape(s, merge, effect_class), 8, {5}, options(8), "auto");
        }
        // 16/17 (K1's register cache): segments of 13 to 19 cells; 255/256/257 (K1 and K1L):
        // segments of 254 to 258 cells (unmerged, so cells are rows).
        {
            xhdfe_bench::Shape s;
            s.name = "b16";
            s.rows = 320000;
            s.levels = {20000, 3000, 200, 10};
            s.spread = 3;
            run_case(from_shape(s, 0, -1), 8, {7, 5}, options(8), "auto");
            s.name = "b256";
            s.rows = 307200;
            s.levels = {1200, 3000, 100, 10};
            s.spread = 2;
            run_case(from_shape(s, 0, -1), 8, {8, 5}, options(8), "auto");
        }
        // K1's segment order (17 November) and the upload copies are functions of the
        // geometry alone: the same bits with the host parts run serially, in reverse order
        // or on a team of eight threads (lengths 13 to 19 and a private effect: the order
        // is not the identity and it moves the private sums).
        {
            xhdfe_bench::Shape s;
            s.name = "b16-team";
            s.rows = 320000;
            s.levels = {20000, 3000, 200, 10};
            s.spread = 3;
            const Problem pr = from_shape(s, 0, -1);  // alive while the geometry borrows its codes
            Built b;
            build(pr, 8, b);
            const std::vector<double> x = operand(b.g.m, 8, 5, 4242);
            std::vector<std::vector<double>> q;
            for (int how = 0; how < 3; ++how) {
                gpu::Options o = options(8);
                if (how == 1)
                    o.run = [](int parts, const std::function<void(int)>& body) {
                        for (int part = parts - 1; part >= 0; --part) body(part);
                    };
                if (how == 2) o.run = [&b](int parts, const std::function<void(int)>& body) { b.team->run(parts, body); };
                gpu::Operator op(gpu::view_of(b.g), o);
                Rows rows(b.g.m, 8);
                op.register_host(rows.p, 2U * rows.count);
                q.push_back(device_product(op, rows, x, 5));
                op.unregister_host();
            }
            expect(std::memcmp(q[0].data(), q[1].data(), q[0].size() * sizeof(double)) == 0 &&
                       std::memcmp(q[0].data(), q[2].data(), q[0].size() * sizeof(double)) == 0,
                   "K1's order: the same bits with the host parts serial, reversed or on a team");
        }
        // Package 3-lite: the replacement product on x and the iterative one on p through ONE
        // registration of x, r, p and q (four adjacent slices, as run_group's arena), alternating
        // p -> x -> p with distinct operands and sentinel-filled results: p's two results bit for
        // bit, x's equal to the product of the same values placed in p's slot, each within the
        // bound of the CPU product of its operand, the padding +0.0, r and the operands untouched,
        // four slices in the admission.
        {
            xhdfe_bench::Shape s;
            s.name = "alternation";
            s.rows = 300000;
            s.levels = {33000, 4000, 300, 12};
            const Problem pr = from_shape(s, -1, -1);
            Built b;
            build(pr, 8, b);
            const int W = 8, width = 5, m = b.g.m;
            const std::size_t mw = static_cast<std::size_t>(m) * W;
            gpu::Options o = options(W);
            o.registered_slices = 4;
            gpu::Operator op(gpu::view_of(b.g), o);
            std::vector<double> arena(4 * mw, 0.0);
            double* x = arena.data();
            double* r = x + mw;
            double* p = r + mw;
            double* q = p + mw;
            const std::vector<double> xs = operand(m, W, width, 77), ps = operand(m, W, width, 99);
            std::copy(xs.begin(), xs.end(), x);
            std::copy(ps.begin(), ps.end(), p);
            std::fill(r, r + mw, sentinel());
            op.register_host(arena.data(), 4 * mw);
            op.begin_group(width);
            const auto product = [&](const double* in) {
                std::fill(q, q + mw, sentinel());
                op.apply(in, q);
                return std::vector<double>(q, q + mw);
            };
            const std::vector<double> q1 = product(p), q2 = product(x), q3 = product(p);
            const auto same = [](const std::vector<double>& a, const std::vector<double>& c) {
                return std::memcmp(a.data(), c.data(), a.size() * sizeof(double)) == 0;
            };
            expect(same(q1, q3), "p -> x -> p: p's two results bit for bit");
            bool r_kept = true;
            for (std::size_t i = 0; i < mw; ++i) r_kept = r_kept && is_sentinel(r[i]);
            expect(r_kept, "p -> x -> p: r never written");
            expect(std::equal(xs.begin(), xs.end(), x) && std::equal(ps.begin(), ps.end(), p), "p -> x -> p: operands unchanged");
            std::copy(xs.begin(), xs.end(), p);
            expect(same(product(p), q2), "x's result = the same values' result from p's slot");
            std::vector<double> rx(mw, 0.0), rp(mw, 0.0);
            cpu_dispatch(W, b, xs.data(), rx.data());
            cpu_dispatch(W, b, ps.data(), rp.data());
            expect(bound_ratio(b.g, b.op_parts, xs.data(), q2.data(), rx.data(), W, width) <= 1.0, "x's result within the bound");
            expect(bound_ratio(b.g, b.op_parts, ps.data(), q1.data(), rp.data(), W, width) <= 1.0, "p's result within the bound");
            expect(padding_zero(q1.data(), m, W, width) && padding_zero(q2.data(), m, W, width), "p -> x -> p: padding +0.0");
            expect(op.admission().page_locked_iteration == 4 * mw * sizeof(double), "admission: four registered slices");
            op.unregister_host();
        }
        // The radix path forced on a small merged shape, and the tiles path with no
        // private effect (every remaining effect built), items of 2 (K1L, K2 items, K3).
        {
            xhdfe_bench::Shape s;
            s.name = "radix-forced";
            s.rows = 120000;
            s.levels = {9000, 2500, 40};
            s.dup = 0.2;
            gpu::Options o = options(8);
            o.b_method = gpu::BMethod::Radix;
            o.private_max = 0;
            run_case(from_shape(s, 1, 0), 8, {7, 5}, o, "radix_all_built");
            gpu::Options t = options(4);
            t.b_method = gpu::BMethod::Tiles;
            t.private_max = 0;
            t.short_seg = 2;
            t.short_level = 2;
            t.item = 2;
            run_case(from_shape(s, 1, 0), 4, {4, 3}, t, "tiles_items_of_2");
            gpu::Options pv = options(8);  // K1L with the private accumulators
            pv.short_seg = 2;
            pv.short_level = 2;
            pv.item = 2;
            run_case(from_shape(s, 1, 1), 8, {8, 6}, pv, "private_items_of_2");
        }

        // 3. C_op = 0: two single-cell segments, every level empty, every effect built.
        {
            const int cell_ptr[] = {0, 1, 2}, c0[] = {0, 1}, c1[] = {0, 2};
            const double inv[] = {1.0, 1.0};
            gpu::GeometryView v;
            v.nb = 2; v.m = 5; v.L = 2; v.L_op = 0; v.C = 2; v.C_op = 0;
            int none = 0;
            v.cell_ptr = cell_ptr; v.op_seg = &none; v.seg_inv = inv; v.cc[0] = c0; v.cc[1] = c1;
            v.dims[0].levels = 2; v.dims[0].offset = 0; v.dims[1].levels = 3; v.dims[1].offset = 2;
            for (int pm : {48, 0}) {
                gpu::Options o = options(4);
                o.private_max = pm;
                gpu::Operator op(v, o);
                Rows rows(5, 4);
                op.register_host(rows.p, 2U * rows.count);
                const std::vector<double> q = device_product(op, rows, operand(5, 4, 3, 5), 3);
                bool zeros = true;
                for (double d : q) zeros = zeros && zero_bits(d);
                expect(zeros, "C_op = 0 (" + classes_of(op, 2) + "): every output +0.0");
                op.unregister_host();
            }
            // No segment and no cell at all: the arrays without elements are null pointers (the
            // rational oracle's zero_operator fixtures); private, built and transposed effects.
            gpu::GeometryView e;
            const int none_ptr[] = {0}, tr0[] = {0, 0, 0, 0}, tr1[] = {0, 0, 0};
            e.nb = 2; e.m = 5; e.L = 0; e.L_op = 0; e.C = 0; e.C_op = 0; e.cell_ptr = none_ptr;
            e.dims[0].levels = 3; e.dims[0].offset = 0; e.dims[1].levels = 2; e.dims[1].offset = 3;
            for (int variant = 0; variant < 3; ++variant) {
                gpu::GeometryView v2 = e;
                gpu::Options o = options(2);
                if (variant >= 1) o.private_max = 0;
                if (variant == 2) {
                    v2.dims[0].transposed = true; v2.dims[0].tr_ptr = tr0;
                    v2.dims[1].transposed = true; v2.dims[1].tr_ptr = tr1;
                }
                gpu::Operator op(v2, o);
                Rows rows(5, 2);
                op.register_host(rows.p, 2U * rows.count);
                const std::vector<double> q = device_product(op, rows, operand(5, 2, 2, 7), 2);
                bool zeros = true;
                for (double d : q) zeros = zeros && zero_bits(d);
                expect(zeros, "no segment, no cell (" + classes_of(op, 2) + "): every output +0.0");
                op.unregister_host();
            }
            std::cout << "OPERATOR_SMOKE c_op_zero done\n";
        }

        // 4. Failures in the middle of the set-up and at the registration, then the same
        // process again. No transpose: the 4 000-level effect built by the radix sort,
        // the 300-level one by the tiles, the 12-level one private.
        {
            xhdfe_bench::Shape s;
            s.name = "inject";
            s.rows = 300000;
            s.levels = {33000, 4000, 300, 12};
            const Problem pr = from_shape(s, -1, 0);
            Built b;
            build(pr, 8, b);
            const auto view = gpu::view_of(b.g);
            const std::vector<double> x = operand(b.g.m, 8, 5, 99);
            std::vector<double> reference;
            {
                gpu::Operator op(view, options(8));
                Rows rows(b.g.m, 8);
                op.register_host(rows.p, 2U * rows.count);
                reference = device_product(op, rows, x, 5);
                op.unregister_host();
            }
            const std::size_t live0 = gpu::Operator::live_device_bytes();
            expect(live0 == 0, "no device bytes held after an operator is destroyed");
            const struct { gpu::TestFailure f; gpu::ErrorKind kind; const char* name; } points[] = {
                {gpu::TestFailure::Allocation, gpu::ErrorKind::Memory, "allocation"},
                {gpu::TestFailure::Build, gpu::ErrorKind::Memory, "build"},
                {gpu::TestFailure::Staging, gpu::ErrorKind::Pinned, "staging"}};
            for (const auto& pt : points) {
                gpu::Options o = options(8);
                o.test_failure = pt.f;
                gpu::Admission record;
                bool thrown = false;
                const gpu::ErrorKind k = kind_of([&] { gpu::Operator op(view, o, &record); }, thrown);
                expect(thrown && k == pt.kind, std::string("injected ") + pt.name + ": the expected kind");
                expect(gpu::Operator::live_device_bytes() == live0, std::string("injected ") + pt.name + ": nothing left allocated");
                expect(clean_device(), std::string("injected ") + pt.name + ": no CUDA error left, nothing in flight");
                expect(record.admitted, std::string("injected ") + pt.name + ": admitted before it failed");
            }
            {
                gpu::Options o = options(8);
                o.test_failure = gpu::TestFailure::Register;
                gpu::Operator op(view, o);
                Rows rows(b.g.m, 8);
                bool thrown = false;
                const gpu::ErrorKind k = kind_of([&] { op.register_host(rows.p, 2U * rows.count); }, thrown);
                expect(thrown && k == gpu::ErrorKind::Pinned, "injected register: Pinned");
                // Nothing registered: the range can be registered (and released) at once.
                const cudaError_t r = cudaHostRegister(rows.p, 2U * rows.count * sizeof(double), cudaHostRegisterDefault);
                expect(r == cudaSuccess, "injected register: the range is not left registered");
                if (r == cudaSuccess) (void)cudaHostUnregister(rows.p);
                bool refused = false;
                op.begin_group(5);
                const gpu::ErrorKind u = kind_of([&] { op.apply(rows.p, rows.q); }, refused);
                expect(refused && u == gpu::ErrorKind::Fatal, "a product outside the registered rows: Fatal");
                expect(clean_device(), "injected register: no CUDA error left");
            }
            {
                // A range already registered elsewhere is not scarcity: Fatal (B7), and the
                // other registration is untouched.
                gpu::Operator op(view, options(8));
                Rows rows(b.g.m, 8);
                expect(cudaHostRegister(rows.p, 2U * rows.count * sizeof(double), cudaHostRegisterDefault) == cudaSuccess,
                       "the test's own registration");
                bool thrown = false;
                const gpu::ErrorKind k = kind_of([&] { op.register_host(rows.p, 2U * rows.count); }, thrown);
                expect(thrown && k == gpu::ErrorKind::Fatal, "a duplicate registration: Fatal");
                expect(cudaHostUnregister(rows.p) == cudaSuccess, "the test's own registration released");
                (void)cudaGetLastError();
                bool w0 = false, w9 = false;
                const gpu::ErrorKind k0 = kind_of([&] { op.begin_group(0); }, w0);
                const gpu::ErrorKind k9 = kind_of([&] { op.begin_group(9); }, w9);
                expect(w0 && k0 == gpu::ErrorKind::Fatal && w9 && k9 == gpu::ErrorKind::Fatal, "illegal widths 0 and lanes + 1: Fatal");
            }
            {
                gpu::Options o = options(8);
                o.budget = 1;
                bool thrown = false;
                const gpu::ErrorKind k = kind_of([&] { gpu::Operator op(view, o); }, thrown);
                expect(thrown && k == gpu::ErrorKind::Memory, "a one-byte budget: Memory, before any allocation");
                expect(gpu::Operator::live_device_bytes() == live0, "a one-byte budget: nothing allocated");
            }
            {
                gpu::Operator op(view, options(8));
                Rows rows(b.g.m, 8);
                op.register_host(rows.p, 2U * rows.count);
                const std::vector<double> again = device_product(op, rows, x, 5);
                op.unregister_host();
                expect(std::memcmp(again.data(), reference.data(), again.size() * sizeof(double)) == 0,
                       "after every injected failure, the same process computes the same bits");
            }
            expect(gpu::Operator::live_device_bytes() == live0, "every operator released");
            std::cout << "OPERATOR_SMOKE injections done\n";
        }
    } catch (const std::exception& e) {
        std::cout << "FAIL exception: " << e.what() << '\n';
        return 1;
    }
    if (failures) {
        std::cout << "CUDA_OPERATOR_SMOKE_FAIL " << failures << " of " << checks << '\n';
        return 1;
    }
    std::cout << "CUDA_OPERATOR_SMOKE_PASS " << checks << '\n';
    return 0;
}
