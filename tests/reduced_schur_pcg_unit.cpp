// Unit tests of the R1 kernel (src/reduced_schur_pcg.hpp) on synthetic data,
// against a dense least-squares oracle in long double. One compiler process,
// the project's flags, from the worktree root (one command line):
//   nice -n 10 c++ -std=gnu++17 -O3 -DNDEBUG -march=native -mtune=native
//     -ffast-math -fno-finite-math-only -funroll-loops -ftree-vectorize
//     -fomit-frame-pointer -fno-signed-zeros -fno-trapping-math -fassociative-math
//     -freciprocal-math -fopenmp -I src -isystem r/xhdfe/src/eigen
//     tests/reduced_schur_pcg_unit.cpp -o build_r1_unit/reduced_schur_pcg_unit
// Run: OMP_WAIT_POLICY=passive build_r1_unit/reduced_schur_pcg_unit

#include "reduced_schur_pcg.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------- allocation meter

// Every operator new of this process goes through the replacements below; each
// block carries a header with its size, so the live bytes and their peak are
// known at any time. The skewed-shape case compares the peak of one solve with
// the bound the kernel reported before allocating (Report::bytes). Buffer
// allocates with align_val_t(64), so the aligned forms are replaced too. The
// meter changes where blocks live, never what is computed.
namespace meter {
std::atomic<long long> live{0};
std::atomic<long long> peak{0};

inline void* allocate(std::size_t size, std::size_t align) {
    const std::size_t header = std::max<std::size_t>(64, align);
    const std::size_t total = (size + 2 * header - 1) / header * header;
    void* base = std::aligned_alloc(header, total);
    if (base == nullptr) throw std::bad_alloc();
    char* block = static_cast<char*>(base) + header;
    std::memcpy(block - 8, &header, sizeof(header));
    std::memcpy(block - 16, &size, sizeof(size));
    const long long now = live.fetch_add(static_cast<long long>(size)) + static_cast<long long>(size);
    long long seen = peak.load();
    while (now > seen && !peak.compare_exchange_weak(seen, now)) {
    }
    return block;
}

inline void* allocate_nothrow(std::size_t size, std::size_t align) noexcept {
    try {
        return allocate(size, align);
    } catch (...) {
        return nullptr;
    }
}

inline void release(void* ptr) noexcept {
    if (ptr == nullptr) return;
    char* block = static_cast<char*>(ptr);
    std::size_t header = 0, size = 0;
    std::memcpy(&header, block - 8, sizeof(header));
    std::memcpy(&size, block - 16, sizeof(size));
    live.fetch_sub(static_cast<long long>(size));
    std::free(block - header);
}
}  // namespace meter

void* operator new(std::size_t size) { return meter::allocate(size, alignof(std::max_align_t)); }
void* operator new[](std::size_t size) { return meter::allocate(size, alignof(std::max_align_t)); }
void* operator new(std::size_t size, std::align_val_t a) { return meter::allocate(size, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t size, std::align_val_t a) { return meter::allocate(size, static_cast<std::size_t>(a)); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    return meter::allocate_nothrow(size, alignof(std::max_align_t));
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return meter::allocate_nothrow(size, alignof(std::max_align_t));
}
void* operator new(std::size_t size, std::align_val_t a, const std::nothrow_t&) noexcept {
    return meter::allocate_nothrow(size, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t size, std::align_val_t a, const std::nothrow_t&) noexcept {
    return meter::allocate_nothrow(size, static_cast<std::size_t>(a));
}
void operator delete(void* p) noexcept { meter::release(p); }
void operator delete[](void* p) noexcept { meter::release(p); }
void operator delete(void* p, std::size_t) noexcept { meter::release(p); }
void operator delete[](void* p, std::size_t) noexcept { meter::release(p); }
void operator delete(void* p, std::align_val_t) noexcept { meter::release(p); }
void operator delete[](void* p, std::align_val_t) noexcept { meter::release(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { meter::release(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { meter::release(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { meter::release(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { meter::release(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { meter::release(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { meter::release(p); }

namespace rp = hdfe::reduced_pcg;
using MatL = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;
using VecL = Eigen::Matrix<long double, Eigen::Dynamic, 1>;

static int g_checks = 0;
static int g_failures = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        ++g_checks;                                                   \
        if (!(cond)) {                                                \
            ++g_failures;                                             \
            std::printf("FAIL line %d: ", __LINE__);                  \
            std::printf(__VA_ARGS__);                                 \
            std::printf("\n");                                        \
        }                                                             \
    } while (0)

static const int kThreads[] = {1, 2, 8, 16, 24, 48};

struct Design {
    std::string name;
    int n = 0;
    std::vector<std::vector<int>> codes;  // per effect
    std::vector<int> levels;
    int eliminated = 0;
    std::vector<std::vector<double>> cols;
};

// ---------------------------------------------------------------- designs

// Rows grouped by `units` levels of the eliminated effect (2..max_rows rows
// each), remaining codes drawn uniformly; columns: y = effects + noise,
// x1 ~ N(0,1).
static Design random_panel(unsigned seed, int units, int max_rows, const std::vector<int>& remaining,
                           int extra_x = 1) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> rows_dist(2, max_rows);
    std::normal_distribution<double> normal(0.0, 1.0);
    Design d;
    d.name = "random";
    const int k = 1 + static_cast<int>(remaining.size());
    d.codes.assign(static_cast<std::size_t>(k), {});
    d.levels.assign(static_cast<std::size_t>(k), 0);
    d.eliminated = 0;
    d.levels[0] = units;
    for (std::size_t j = 0; j < remaining.size(); ++j) d.levels[j + 1] = remaining[j];
    for (int u = 0; u < units; ++u) {
        const int rows = rows_dist(rng);
        for (int r = 0; r < rows; ++r) {
            d.codes[0].push_back(u);
            for (std::size_t j = 0; j < remaining.size(); ++j)
                d.codes[j + 1].push_back(std::uniform_int_distribution<int>(0, remaining[j] - 1)(rng));
        }
    }
    d.n = static_cast<int>(d.codes[0].size());
    std::vector<std::vector<double>> effect(static_cast<std::size_t>(k));
    for (int b = 0; b < k; ++b) {
        effect[static_cast<std::size_t>(b)].resize(static_cast<std::size_t>(d.levels[static_cast<std::size_t>(b)]));
        for (double& e : effect[static_cast<std::size_t>(b)]) e = normal(rng);
    }
    std::vector<double> y(static_cast<std::size_t>(d.n));
    for (int i = 0; i < d.n; ++i) {
        double v = 0.3 * normal(rng);
        for (int b = 0; b < k; ++b) v += effect[static_cast<std::size_t>(b)][static_cast<std::size_t>(d.codes[static_cast<std::size_t>(b)][static_cast<std::size_t>(i)])];
        y[static_cast<std::size_t>(i)] = v;
    }
    d.cols.push_back(y);
    for (int x = 0; x < extra_x; ++x) {
        std::vector<double> col(static_cast<std::size_t>(d.n));
        for (double& v : col) v = normal(rng) + 0.5;
        d.cols.push_back(col);
    }
    return d;
}

// Worker-firm-year panel: each worker has a home firm and moves to a random
// one in a share `mobility` of its rows; consecutive years from a random start.
static Design mobility_panel(unsigned seed, int workers, int firms, int years, double mobility) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> rows_dist(2, 8);
    std::uniform_int_distribution<int> firm_dist(0, firms - 1);
    std::uniform_int_distribution<int> start_dist(0, years - 8);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    Design d;
    d.name = "mobility";
    d.codes.assign(3, {});
    d.levels = {workers, firms, years};
    std::vector<double> wfe(static_cast<std::size_t>(workers)), ffe(static_cast<std::size_t>(firms)),
        yfe(static_cast<std::size_t>(years));
    for (double& v : wfe) v = normal(rng);
    for (double& v : ffe) v = normal(rng);
    for (double& v : yfe) v = 0.1 * normal(rng);
    std::vector<double> y, x1, x2;
    for (int u = 0; u < workers; ++u) {
        const int rows = rows_dist(rng), home = firm_dist(rng), start = start_dist(rng);
        for (int r = 0; r < rows; ++r) {
            const int firm = unit(rng) < mobility ? firm_dist(rng) : home;
            const int year = start + r;
            d.codes[0].push_back(u);
            d.codes[1].push_back(firm);
            d.codes[2].push_back(year);
            const double a = normal(rng);
            x1.push_back(a);
            x2.push_back(1.0e3 + normal(rng));
            y.push_back(wfe[static_cast<std::size_t>(u)] + ffe[static_cast<std::size_t>(firm)] +
                        yfe[static_cast<std::size_t>(year)] + 0.5 * a + 0.3 * normal(rng));
        }
    }
    d.n = static_cast<int>(y.size());
    d.cols = {y, x1, x2};
    return d;
}

static void shuffle_rows(Design& d, unsigned seed) {
    std::vector<int> order(static_cast<std::size_t>(d.n));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    for (auto& codes : d.codes) {
        std::vector<int> copy = codes;
        for (int i = 0; i < d.n; ++i) codes[static_cast<std::size_t>(i)] = copy[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
    }
    for (auto& col : d.cols) {
        std::vector<double> copy = col;
        for (int i = 0; i < d.n; ++i) col[static_cast<std::size_t>(i)] = copy[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
    }
}

// A column of the span of every effect: D a for random a.
static std::vector<double> span_column(const Design& d, unsigned seed, double scale = 1.0, double offset = 0.0) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> col(static_cast<std::size_t>(d.n), offset);
    for (std::size_t b = 0; b < d.codes.size(); ++b) {
        std::vector<double> a(static_cast<std::size_t>(d.levels[b]));
        for (double& v : a) v = scale * normal(rng);
        for (int i = 0; i < d.n; ++i) col[static_cast<std::size_t>(i)] += a[static_cast<std::size_t>(d.codes[b][static_cast<std::size_t>(i)])];
    }
    return col;
}

// ---------------------------------------------------------------- oracle

struct Oracle {
    MatL D;
    Eigen::CompleteOrthogonalDecomposition<MatL> cod;
    explicit Oracle(const Design& d) {
        int total = 0;
        for (int lv : d.levels) total += lv;
        D = MatL::Zero(d.n, total);
        int off = 0;
        for (std::size_t b = 0; b < d.codes.size(); ++b) {
            for (int i = 0; i < d.n; ++i) D(i, off + d.codes[b][static_cast<std::size_t>(i)]) = 1.0L;
            off += d.levels[b];
        }
        cod.setThreshold(1.0e-12L);
        cod.compute(D);
    }
    std::vector<double> residual(const std::vector<double>& v) const {
        VecL b(static_cast<Eigen::Index>(v.size()));
        for (std::size_t i = 0; i < v.size(); ++i) b(static_cast<Eigen::Index>(i)) = v[i];
        const VecL a = cod.solve(b);
        const VecL r = b - D * a;
        std::vector<double> out(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(r(static_cast<Eigen::Index>(i)));
        return out;
    }
};

// abs_residual_rel as certify_absorption_result defines it for unit weights:
// ||D' r|| / (sqrt(K n) ||v||), with the dominant-origin scale.
static double certificate(const Design& d, const std::vector<double>& v, const std::vector<double>& res) {
    long double num = 0.0L;
    for (std::size_t b = 0; b < d.codes.size(); ++b) {
        std::vector<long double> sums(static_cast<std::size_t>(d.levels[b]), 0.0L);
        for (int i = 0; i < d.n; ++i) sums[static_cast<std::size_t>(d.codes[b][static_cast<std::size_t>(i)])] += res[static_cast<std::size_t>(i)];
        for (long double s : sums) num += s * s;
    }
    long double raw = 0.0L, sum = 0.0L;
    for (double x : v) {
        raw += static_cast<long double>(x) * x;
        sum += x;
    }
    long double norm_sq = raw;
    const long double n = d.n;
    if (sum * (sum / n) > (1.0L - 1.0L / 1024.0L) * raw) {
        const long double origin = v[0];
        long double shifted = 0.0L;
        for (double x : v) shifted += static_cast<long double>(x) - origin;
        const long double mean = shifted / n;
        long double centered = 0.0L;
        for (double x : v) {
            const long double delta = static_cast<long double>(x) - origin - mean;
            centered += delta * delta;
        }
        if (centered > 0.0L && centered < raw / 1024.0L) norm_sq = centered;
    }
    const long double scale = std::sqrt(static_cast<long double>(d.codes.size()) * n) * std::sqrt(norm_sq);
    if (scale <= 0.0L) return num > 0.0L ? 1.0e308 : 0.0;
    return static_cast<double>(std::sqrt(num) / scale);
}

static double norm2(const std::vector<double>& v) {
    long double s = 0.0L;
    for (double x : v) s += static_cast<long double>(x) * x;
    return static_cast<double>(std::sqrt(s));
}

// ---------------------------------------------------------------- kernel runs

struct Run {
    rp::Report rep;
    std::vector<std::vector<double>> res;
};

static rp::Config base_config(int threads) {
    rp::Config c;
    c.threads = threads;
    c.blocks = 48;
    c.atol = 1.0e-11;
    c.btol = 1.0e-11;
    c.trigger = 1.0e-11;
    c.max_iter = 1500;
    return c;
}

// FNV-1a over every run's residual bits and per-column report (not the
// timings): equal digests mean equal results between two builds.
static std::uint64_t g_digest = UINT64_C(1469598103934665603);
static void digest_bytes(const void* data, std::size_t size) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        g_digest ^= p[i];
        g_digest *= UINT64_C(1099511628211);
    }
}

static Run run_kernel(const Design& d, const rp::Config& cfg);

static void digest_run(const Run& run) {
    const int solved = run.rep.solved ? 1 : 0, decline = static_cast<int>(run.rep.decline);
    digest_bytes(&solved, sizeof(solved));
    digest_bytes(&decline, sizeof(decline));
    digest_bytes(&run.rep.iterations, sizeof(run.rep.iterations));
    for (const auto& c : run.rep.columns) {
        const int stop = static_cast<int>(c.stop);
        digest_bytes(&stop, sizeof(stop));
        digest_bytes(&c.iterations, sizeof(c.iterations));
        digest_bytes(&c.tau, sizeof(c.tau));
        digest_bytes(&c.chat, sizeof(c.chat));
    }
    if (run.rep.solved)
        for (const auto& r : run.res) digest_bytes(r.data(), r.size() * sizeof(double));
}

static Run run_kernel(const Design& d, const rp::Config& cfg) {
    Run run;
    const int k = static_cast<int>(d.codes.size());
    std::vector<const int*> codes(static_cast<std::size_t>(k));
    for (int b = 0; b < k; ++b) codes[static_cast<std::size_t>(b)] = d.codes[static_cast<std::size_t>(b)].data();
    run.res.assign(d.cols.size(), std::vector<double>(static_cast<std::size_t>(d.n), 0.0));
    std::vector<const double*> cols(d.cols.size());
    std::vector<double*> res(d.cols.size());
    for (std::size_t c = 0; c < d.cols.size(); ++c) {
        cols[c] = d.cols[c].data();
        res[c] = run.res[c].data();
    }
    rp::Problem p;
    p.n = d.n;
    p.k = k;
    p.codes = codes.data();
    p.levels = d.levels.data();
    p.eliminated = d.eliminated;
    p.cols = static_cast<int>(d.cols.size());
    p.columns = cols.data();
    p.residuals = res.data();
    run.rep = rp::solve(p, cfg);
    digest_run(run);
    return run;
}

static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// The same report and residual bits at every team size.
static void check_threads(const Design& d, rp::Config cfg, const char* label) {
    cfg.threads = 1;
    const Run base = run_kernel(d, cfg);
    for (int t : kThreads) {
        if (t == 1) continue;
        cfg.threads = t;
        const Run other = run_kernel(d, cfg);
        bool same = other.rep.solved == base.rep.solved && other.rep.decline == base.rep.decline &&
                    other.rep.iterations == base.rep.iterations;
        for (std::size_t c = 0; same && c < base.rep.columns.size(); ++c) {
            const auto& a = base.rep.columns[c];
            const auto& b = other.rep.columns[c];
            same = a.stop == b.stop && a.iterations == b.iterations && a.first_tau_pass == b.first_tau_pass &&
                   same_bits(a.tau, b.tau) && same_bits(a.chat, b.chat) && a.formed_checks == b.formed_checks;
        }
        if (base.rep.solved)
            for (std::size_t c = 0; same && c < base.res.size(); ++c)
                same = std::memcmp(base.res[c].data(), other.res[c].data(), sizeof(double) * base.res[c].size()) == 0;
        CHECK(same, "%s: threads=%d differ from threads=1", label, t);
    }
}

// Solved, residuals at the oracle's within tol * ||v||, certificate measure
// at most 1e-11 for every column.
static void check_oracle(const Design& d, const Run& run, const Oracle& oracle, const char* label,
                         double tol = 1.0e-9) {
    CHECK(run.rep.solved, "%s: declined (%s)", label, rp::decline_name(run.rep.decline));
    if (!run.rep.solved) return;
    for (std::size_t c = 0; c < d.cols.size(); ++c) {
        const std::vector<double> expected = oracle.residual(d.cols[c]);
        std::vector<double> diff(expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) diff[i] = run.res[c][i] - expected[i];
        const double vnorm = norm2(d.cols[c]);
        const double err = norm2(diff);
        CHECK(err <= tol * vnorm || err == 0.0, "%s col %zu: ||res - oracle|| = %.3e > %.1e * ||v|| = %.3e",
              label, c, err, tol, tol * vnorm);
        const double cert = certificate(d, d.cols[c], run.res[c]);
        CHECK(cert <= 1.0e-11, "%s col %zu: certificate measure %.3e > 1e-11 (stop %s, iterations %d)", label, c,
              cert, rp::stop_name(run.rep.columns[c].stop), run.rep.columns[c].iterations);
    }
}

// ---------------------------------------------------------------- dense references

// B = (I - P1) D2, dense, in the order of the reduced vector.
static MatL dense_reduced(const Design& d) {
    int m = 0;
    std::vector<int> offset;
    for (std::size_t b = 0; b < d.codes.size(); ++b) {
        if (static_cast<int>(b) == d.eliminated) continue;
        offset.push_back(m);
        m += d.levels[b];
    }
    MatL D2 = MatL::Zero(d.n, m);
    int j = 0;
    for (std::size_t b = 0; b < d.codes.size(); ++b) {
        if (static_cast<int>(b) == d.eliminated) continue;
        for (int i = 0; i < d.n; ++i) D2(i, offset[static_cast<std::size_t>(j)] + d.codes[b][static_cast<std::size_t>(i)]) = 1.0L;
        ++j;
    }
    const int L = d.levels[static_cast<std::size_t>(d.eliminated)];
    MatL means = MatL::Zero(L, m);
    std::vector<int> count(static_cast<std::size_t>(L), 0);
    const auto& e = d.codes[static_cast<std::size_t>(d.eliminated)];
    for (int i = 0; i < d.n; ++i) {
        means.row(e[static_cast<std::size_t>(i)]) += D2.row(i);
        ++count[static_cast<std::size_t>(e[static_cast<std::size_t>(i)])];
    }
    for (int l = 0; l < L; ++l)
        if (count[static_cast<std::size_t>(l)] > 0) means.row(l) /= static_cast<long double>(count[static_cast<std::size_t>(l)]);
    MatL B = D2;
    for (int i = 0; i < d.n; ++i) B.row(i) -= means.row(e[static_cast<std::size_t>(i)]);
    return B;
}

static void check_diagonal(const Design& d, const char* label) {
    rp::Problem p;
    std::vector<const int*> codes;
    for (const auto& c : d.codes) codes.push_back(c.data());
    p.n = d.n;
    p.k = static_cast<int>(d.codes.size());
    p.codes = codes.data();
    p.levels = d.levels.data();
    p.eliminated = d.eliminated;
    const std::vector<double> diag = rp::schur_diagonal(p, base_config(4));
    const MatL B = dense_reduced(d);
    const VecL dense = (B.transpose() * B).diagonal();
    CHECK(static_cast<Eigen::Index>(diag.size()) == dense.size(), "%s: diagonal size", label);
    if (static_cast<Eigen::Index>(diag.size()) != dense.size()) return;
    int nulls = 0;
    for (Eigen::Index i = 0; i < dense.size(); ++i) {
        const long double expected = dense(i) < 1.0e-12L ? 0.0L : dense(i);
        nulls += expected == 0.0L ? 1 : 0;
        const long double err = std::fabs(static_cast<long double>(diag[static_cast<std::size_t>(i)]) - expected);
        CHECK(err <= 1.0e-12L * std::max<long double>(1.0L, expected), "%s: S_jj[%ld] = %.17g, dense %.17Lg", label,
              static_cast<long>(i), diag[static_cast<std::size_t>(i)], expected);
    }
    std::printf("  %s: %ld levels, %d null\n", label, static_cast<long>(dense.size()), nulls);
}

// tau_k of the kernel against LSQR's test2 on the Jacobi-scaled reduced
// problem min ||w - B^ y||, computed densely by Golub-Kahan with full
// reorthogonalization: ||B^' s_k|| / (||B_k||_F ||s_k||).
struct TauRecorder {
    std::vector<double> tau;
    static void record(void* self, int column, int iteration, double tau, double) {
        auto* me = static_cast<TauRecorder*>(self);
        if (column != 0) return;
        if (static_cast<int>(me->tau.size()) < iteration) me->tau.resize(static_cast<std::size_t>(iteration), -1.0);
        me->tau[static_cast<std::size_t>(iteration) - 1U] = tau;
    }
};

static void check_tau_against_lsqr(const Design& d, const char* label) {
    MatL B = dense_reduced(d);
    const VecL diag = (B.transpose() * B).diagonal();
    for (Eigen::Index j = 0; j < B.cols(); ++j)
        B.col(j) *= diag(j) > 1.0e-12L ? 1.0L / std::sqrt(diag(j)) : 0.0L;
    const auto& e = d.codes[static_cast<std::size_t>(d.eliminated)];
    const int L = d.levels[static_cast<std::size_t>(d.eliminated)];
    VecL w(d.n);
    {
        std::vector<long double> mean(static_cast<std::size_t>(L), 0.0L);
        std::vector<int> count(static_cast<std::size_t>(L), 0);
        for (int i = 0; i < d.n; ++i) {
            mean[static_cast<std::size_t>(e[static_cast<std::size_t>(i)])] += d.cols[0][static_cast<std::size_t>(i)];
            ++count[static_cast<std::size_t>(e[static_cast<std::size_t>(i)])];
        }
        for (int i = 0; i < d.n; ++i)
            w(i) = d.cols[0][static_cast<std::size_t>(i)] - mean[static_cast<std::size_t>(e[static_cast<std::size_t>(i)])] /
                                                             count[static_cast<std::size_t>(e[static_cast<std::size_t>(i)])];
    }
    const int kmax = 6;
    std::vector<VecL> U, V;
    std::vector<long double> alpha, beta;
    const long double beta1 = w.norm();
    U.push_back(w / beta1);
    VecL v = B.transpose() * U[0];
    alpha.push_back(v.norm());
    V.push_back(v / alpha[0]);
    std::vector<double> test2;
    for (int i = 0; i < kmax; ++i) {
        VecL u = B * V[static_cast<std::size_t>(i)] - alpha[static_cast<std::size_t>(i)] * U[static_cast<std::size_t>(i)];
        for (const VecL& q : U) u -= q.dot(u) * q;
        beta.push_back(u.norm());
        U.push_back(u / beta.back());
        VecL vv = B.transpose() * U.back() - beta.back() * V[static_cast<std::size_t>(i)];
        for (const VecL& q : V) vv -= q.dot(vv) * q;
        alpha.push_back(vv.norm());
        V.push_back(vv / alpha.back());
        const int kk = i + 1;
        MatL Bk = MatL::Zero(kk + 1, kk);
        long double frob = 0.0L;
        for (int c = 0; c < kk; ++c) {
            Bk(c, c) = alpha[static_cast<std::size_t>(c)];
            Bk(c + 1, c) = beta[static_cast<std::size_t>(c)];
            frob += alpha[static_cast<std::size_t>(c)] * alpha[static_cast<std::size_t>(c)] +
                    beta[static_cast<std::size_t>(c)] * beta[static_cast<std::size_t>(c)];
        }
        VecL rhs = VecL::Zero(kk + 1);
        rhs(0) = beta1;
        const VecL z = Bk.colPivHouseholderQr().solve(rhs);
        VecL y = VecL::Zero(B.cols());
        for (int c = 0; c < kk; ++c) y += z(c) * V[static_cast<std::size_t>(c)];
        const VecL s = w - B * y;
        test2.push_back(static_cast<double>((B.transpose() * s).norm() / (std::sqrt(frob) * s.norm())));
    }
    TauRecorder recorder;
    rp::Config cfg = base_config(1);
    cfg.iteration_context = &recorder;
    cfg.on_iteration = &TauRecorder::record;
    Design single = d;
    single.cols.resize(1);
    const Run run = run_kernel(single, cfg);
    CHECK(run.rep.solved, "%s: tau run declined", label);
    for (int i = 0; i < kmax && i < static_cast<int>(recorder.tau.size()); ++i) {
        const double ratio = recorder.tau[static_cast<std::size_t>(i)] / test2[static_cast<std::size_t>(i)];
        CHECK(std::fabs(ratio - 1.0) <= 1.0e-6, "%s: k=%d tau=%.12e test2=%.12e", label, i + 1,
              recorder.tau[static_cast<std::size_t>(i)], test2[static_cast<std::size_t>(i)]);
    }
    std::printf("  %s: tau_1..%d match LSQR test2 (first %.3e, last %.3e)\n", label,
                std::min(kmax, static_cast<int>(recorder.tau.size())), test2.front(), test2.back());
}

// ---------------------------------------------------------------- cases

static void standard_case(const Design& d, const char* label, rp::Config cfg = base_config(1)) {
    const Oracle oracle(d);
    const Run run = run_kernel(d, cfg);
    check_oracle(d, run, oracle, label);
    check_threads(d, cfg, label);
    std::printf("  %s: n=%d k=%zu solved=%d iterations=%d sorted=%d merged=%d large=%d cells=%lld\n", label, d.n,
                d.codes.size(), run.rep.solved ? 1 : 0, run.rep.iterations, run.rep.sorted ? 1 : 0,
                run.rep.merged ? 1 : 0, run.rep.large_effects, run.rep.cells);
}

// The skewed shape of review round 14, at 1M rows instead of 10M: two effects
// of two levels, rows sorted by the eliminated one, whose first level holds all
// rows but two; one regressor. The merge sample accepts the merge, so both
// merge passes run and one block sorts a segment of n - 2 rows.
static Design skewed_pair(unsigned seed, int n) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Design d;
    d.name = "skewed";
    d.n = n;
    d.codes.assign(2, std::vector<int>(static_cast<std::size_t>(n), 0));
    d.levels = {2, 2};
    d.eliminated = 0;
    const double a[2] = {0.3, -0.7}, b[2] = {1.1, 0.4};
    std::vector<double> y(static_cast<std::size_t>(n)), x(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const std::size_t r = static_cast<std::size_t>(i);
        d.codes[0][r] = i >= n - 2 ? 1 : 0;
        d.codes[1][r] = i >= n - 2 ? i - (n - 2) : (i % 3 == 0 ? 1 : 0);
        x[r] = normal(rng);
        y[r] = 0.5 * x[r] + a[d.codes[0][r]] + b[d.codes[1][r]] + 0.3 * normal(rng);
    }
    d.cols = {y, x};
    return d;
}

// One solve with the inputs and outputs prepared beforehand, so that the peak
// counts only what the attempt itself allocates (the bound's scope).
struct Metered {
    rp::Report rep;
    long long peak = 0;
};

static Metered metered_solve(const Design& d, const rp::Config& cfg) {
    const int k = static_cast<int>(d.codes.size());
    std::vector<const int*> codes(static_cast<std::size_t>(k));
    for (int b = 0; b < k; ++b) codes[static_cast<std::size_t>(b)] = d.codes[static_cast<std::size_t>(b)].data();
    std::vector<std::vector<double>> out(d.cols.size(), std::vector<double>(static_cast<std::size_t>(d.n), 0.0));
    std::vector<const double*> cols(d.cols.size());
    std::vector<double*> res(d.cols.size());
    for (std::size_t c = 0; c < d.cols.size(); ++c) {
        cols[c] = d.cols[c].data();
        res[c] = out[c].data();
    }
    rp::Problem p;
    p.n = d.n;
    p.k = k;
    p.codes = codes.data();
    p.levels = d.levels.data();
    p.eliminated = d.eliminated;
    p.cols = static_cast<int>(d.cols.size());
    p.columns = cols.data();
    p.residuals = res.data();
    Metered m;
    const long long before = meter::live.load();
    meter::peak.store(before);
    m.rep = rp::solve(p, cfg);
    m.peak = meter::peak.load() - before;
    return m;
}

int main() {
    std::printf("R1 kernel unit tests\n");

    // 2 to 5 effects, sorted rows, automatic classes.
    for (int k = 2; k <= 5; ++k) {
        std::vector<int> rem;
        const int sizes[] = {25, 9, 6, 4};
        for (int j = 0; j < k - 1; ++j) rem.push_back(sizes[j]);
        Design d = random_panel(100 + static_cast<unsigned>(k), 110, 6, rem, 2);
        const std::string label = "random K=" + std::to_string(k);
        standard_case(d, label.c_str());
        rp::Config forced = base_config(1);
        forced.force_effect_class = 1;
        standard_case(d, (label + " transposed").c_str(), forced);
        forced.force_effect_class = 0;
        standard_case(d, (label + " private").c_str(), forced);
    }

    // Mixed classes (some effects private, some transposed), the production
    // mix of ready and firstreg: two blocks move the small effects private.
    for (int k = 3; k <= 4; ++k) {
        std::vector<int> rem;
        const int sizes[] = {25, 9, 6};
        for (int j = 0; j < k - 1; ++j) rem.push_back(sizes[j]);
        Design d = random_panel(200 + static_cast<unsigned>(k), 110, 6, rem, 2);
        rp::Config cfg = base_config(1);
        cfg.blocks = 2;
        const std::string label = "mixed classes K=" + std::to_string(k);
        standard_case(d, label.c_str(), cfg);
        const Run run = run_kernel(d, cfg);
        CHECK(run.rep.large_effects > 0 && run.rep.large_effects < k - 1, "%s: %d large of %d", label.c_str(),
              run.rep.large_effects, k - 1);
    }

    // Eleven columns: two lane groups.
    {
        Design d = random_panel(300, 120, 6, {18, 6}, 10);
        standard_case(d, "eleven columns");
        const Run run = run_kernel(d, base_config(1));
        CHECK(run.rep.groups == 2, "eleven columns: %d groups", run.rep.groups);
    }

    // Rows not sorted by the eliminated effect, with and without merging.
    {
        Design d = random_panel(7, 120, 7, {20, 7}, 2);
        shuffle_rows(d, 99);
        rp::Config cfg = base_config(1);
        cfg.force_merge = 0;
        standard_case(d, "unsorted unmerged", cfg);
        cfg.force_merge = 1;
        standard_case(d, "unsorted merged", cfg);
        cfg.force_effect_class = 1;
        standard_case(d, "unsorted merged transposed", cfg);
    }

    // Repeated cells (several rows per worker-firm-year) and one-cell segments.
    {
        Design d = random_panel(11, 90, 5, {12, 4}, 1);
        const int n0 = d.n;
        for (int i = 0; i < n0; i += 3) {  // duplicate every third row
            for (auto& c : d.codes) c.push_back(c[static_cast<std::size_t>(i)]);
            for (auto& col : d.cols) col.push_back(col[static_cast<std::size_t>(i)] + 0.25);
        }
        // worker levels whose rows all share one tuple
        const int L = d.levels[0];
        for (int r = 0; r < 3; ++r) {
            d.codes[0].push_back(L);
            d.codes[1].push_back(5);
            d.codes[2].push_back(2);
            d.cols[0].push_back(1.0 + r);
            d.cols[1].push_back(-0.5 * r);
        }
        d.levels[0] = L + 1;
        d.n = static_cast<int>(d.codes[0].size());
        // keep rows sorted by the eliminated effect
        std::vector<int> order(static_cast<std::size_t>(d.n));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return d.codes[0][static_cast<std::size_t>(a)] < d.codes[0][static_cast<std::size_t>(b)]; });
        Design s = d;
        for (std::size_t b = 0; b < d.codes.size(); ++b)
            for (int i = 0; i < d.n; ++i) s.codes[b][static_cast<std::size_t>(i)] = d.codes[b][static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
        for (std::size_t c = 0; c < d.cols.size(); ++c)
            for (int i = 0; i < d.n; ++i) s.cols[c][static_cast<std::size_t>(i)] = d.cols[c][static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
        rp::Config cfg = base_config(1);
        cfg.force_merge = 0;
        standard_case(s, "repeated cells unmerged", cfg);
        cfg.force_merge = 1;
        standard_case(s, "repeated cells merged", cfg);
        check_diagonal(s, "diagonal with shared levels");
    }

    // Disconnected components, nested effects, multiway dependencies, null levels.
    {
        Design a = random_panel(21, 60, 6, {10, 5}, 1);
        Design b = random_panel(22, 60, 6, {10, 5}, 1);
        Design d = a;
        for (std::size_t e = 0; e < d.codes.size(); ++e)
            for (int i = 0; i < b.n; ++i)
                d.codes[e].push_back(b.codes[e][static_cast<std::size_t>(i)] + a.levels[e]);
        for (std::size_t c = 0; c < d.cols.size(); ++c) d.cols[c].insert(d.cols[c].end(), b.cols[c].begin(), b.cols[c].end());
        for (std::size_t e = 0; e < d.levels.size(); ++e) d.levels[e] = a.levels[e] + b.levels[e];
        d.n = a.n + b.n;
        standard_case(d, "two components");
    }
    {
        // firm nested in region: region(f) = f % 4
        Design d = random_panel(31, 120, 6, {24, 6}, 1);
        for (int i = 0; i < d.n; ++i) d.codes[2][static_cast<std::size_t>(i)] = d.codes[1][static_cast<std::size_t>(i)] % 4;
        d.levels[2] = 4;
        standard_case(d, "nested firm in region");
        // a duplicated effect: null space beyond one constant per component
        Design dup = random_panel(32, 120, 6, {18, 6}, 1);
        dup.codes.push_back(dup.codes[1]);
        dup.levels.push_back(dup.levels[1]);
        standard_case(dup, "duplicated effect");
        check_diagonal(dup, "diagonal duplicated effect");
    }
    {
        // Zero-diagonal coordinates: codes with gaps and firms of stayers.
        Design d = random_panel(41, 100, 6, {15, 5}, 2);
        for (int i = 0; i < d.n; ++i) d.codes[1][static_cast<std::size_t>(i)] *= 2;  // odd firm codes empty
        d.levels[1] = 30;
        // workers 0..9 only ever in firm 29 (odd, otherwise empty): a level of stayers
        for (int i = 0; i < d.n; ++i)
            if (d.codes[0][static_cast<std::size_t>(i)] < 10) d.codes[1][static_cast<std::size_t>(i)] = 29;
        standard_case(d, "null levels (gaps and stayers)");
        check_diagonal(d, "diagonal null levels");
    }

    // Special columns: zero, constant within each eliminated level, in the
    // span of every effect, a dominant constant (C1), and the same design
    // scaled by 2^500 and 2^-500.
    {
        Design d = random_panel(51, 120, 6, {16, 6}, 0);
        std::mt19937_64 rng(5);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::vector<double> zero(static_cast<std::size_t>(d.n), 0.0);
        std::vector<double> per_level(static_cast<std::size_t>(d.levels[0]));
        for (double& v : per_level) v = normal(rng);
        std::vector<double> in_e(static_cast<std::size_t>(d.n)), dominant(static_cast<std::size_t>(d.n));
        for (int i = 0; i < d.n; ++i) {
            in_e[static_cast<std::size_t>(i)] = per_level[static_cast<std::size_t>(d.codes[0][static_cast<std::size_t>(i)])];
            dominant[static_cast<std::size_t>(i)] = 1.0e6 + normal(rng);
        }
        d.cols.push_back(zero);
        d.cols.push_back(in_e);
        d.cols.push_back(span_column(d, 6));
        d.cols.push_back(dominant);
        const Oracle oracle(d);
        const Run run = run_kernel(d, base_config(1));
        check_oracle(d, run, oracle, "special columns");
        check_threads(d, base_config(1), "special columns");
        if (run.rep.solved) {
            CHECK(run.rep.columns[1].stop == rp::Stop::Zero && run.rep.columns[1].iterations == 0, "zero column: %s",
                  rp::stop_name(run.rep.columns[1].stop));
            CHECK(run.rep.columns[2].stop == rp::Stop::Span && run.rep.columns[2].iterations == 0,
                  "column constant within e: %s", rp::stop_name(run.rep.columns[2].stop));
            const auto& sp = run.rep.columns[3];
            std::printf("  span column: stop=%s iterations=%d formed_checks=%d s_rel=%.3e\n", rp::stop_name(sp.stop),
                        sp.iterations, sp.formed_checks, sp.s_rel);
            // C1: chat at the stop against the certificate measure of the formed residual.
            const double cert = certificate(d, d.cols[4], run.res[4]);
            const double chat = run.rep.columns[4].chat;
            CHECK(cert > 0.0 && chat / cert > 0.5 && chat / cert < 2.0,
                  "dominant constant: chat=%.3e certificate=%.3e (C1 denominator)", chat, cert);
            std::printf("  dominant constant: chat=%.3e certificate=%.3e\n", chat, cert);
            // The formed residual of that column keeps no eps |mean| per row:
            // its level sums over the eliminated effect are at the rounding of
            // the deviations (1e6 times smaller than with a reassociated formula).
            std::vector<long double> sums(static_cast<std::size_t>(d.levels[0]), 0.0L);
            for (int i = 0; i < d.n; ++i)
                sums[static_cast<std::size_t>(d.codes[0][static_cast<std::size_t>(i)])] += run.res[4][static_cast<std::size_t>(i)];
            long double eliminated_sq = 0.0L;
            for (long double s : sums) eliminated_sq += s * s;
            CHECK(std::sqrt(eliminated_sq) < 1.0e-12L, "dominant constant: ||D1' res|| = %.3Le",
                  std::sqrt(eliminated_sq));
        }
        // Scaling by a power of two is exact as long as no square overflows or
        // underflows: same iterations and stops, residual bits scaled.
        Design unit = d;
        unit.cols.pop_back();  // without the dominant constant
        const Run reference = run_kernel(unit, base_config(1));
        for (const int exponent : {300, -300}) {
            const double scale = std::ldexp(1.0, exponent);
            Design s = unit;
            for (auto& col : s.cols)
                for (double& v : col) v *= scale;
            const Run scaled = run_kernel(s, base_config(1));
            bool same = scaled.rep.solved == reference.rep.solved && scaled.rep.iterations == reference.rep.iterations;
            for (std::size_t c = 0; same && c < reference.rep.columns.size(); ++c)
                same = scaled.rep.columns[c].stop == reference.rep.columns[c].stop &&
                       scaled.rep.columns[c].iterations == reference.rep.columns[c].iterations &&
                       same_bits(scaled.rep.columns[c].tau, reference.rep.columns[c].tau);
            for (std::size_t c = 0; same && c < reference.res.size(); ++c)
                for (int i = 0; same && i < d.n; ++i)
                    same = same_bits(scaled.res[c][static_cast<std::size_t>(i)],
                                     reference.res[c][static_cast<std::size_t>(i)] * scale);
            CHECK(same, "scale 2^%d: iterations, stops, tau or residual bits differ", exponent);
        }
        // 1e6 * 2^500 overflows when squared: a clean decline, never a result.
        Design overflow = d;
        for (double& v : overflow.cols.back()) v *= std::ldexp(1.0, 500);
        const Run over = run_kernel(overflow, base_config(1));
        CHECK(!over.rep.solved && over.rep.decline == rp::Decline::NonFinite, "overflowing squares: %s",
              rp::decline_name(over.rep.decline));
    }

    // rho == 0 with a non-zero residual: every worker sits in one cell, so
    // g = 0 while v keeps a within-cell residual.
    {
        Design d;
        d.codes.assign(3, {});
        std::mt19937_64 rng(61);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::vector<double> y;
        for (int u = 0; u < 80; ++u) {
            const int firm = u % 7, year = u % 3;
            for (int r = 0; r < 3; ++r) {
                d.codes[0].push_back(u);
                d.codes[1].push_back(firm);
                d.codes[2].push_back(year);
                y.push_back(normal(rng));
            }
        }
        d.levels = {80, 7, 3};
        d.n = static_cast<int>(y.size());
        d.cols.push_back(y);
        const Oracle oracle(d);
        const Run run = run_kernel(d, base_config(1));
        check_oracle(d, run, oracle, "rho == 0");
        CHECK(run.rep.solved && run.rep.columns[0].stop == rp::Stop::Test2 && run.rep.columns[0].iterations == 0 &&
                  run.rep.columns[0].tau == 0.0,
              "rho == 0: stop %s after %d iterations", rp::stop_name(run.rep.columns[0].stop),
              run.rep.columns[0].iterations);
        check_threads(d, base_config(1), "rho == 0");
    }

    // A right-hand side made slightly inconsistent by rounding: a column of
    // the span with large offsets. It must be solved to the oracle or declined.
    {
        Design d = random_panel(71, 120, 6, {16, 6}, 0);
        d.cols.push_back(span_column(d, 72, 1.0e8, 3.0e9));
        const Oracle oracle(d);
        const Run run = run_kernel(d, base_config(1));
        if (run.rep.solved) {
            const std::vector<double> expected = oracle.residual(d.cols[1]);
            std::vector<double> diff(expected.size());
            for (std::size_t i = 0; i < expected.size(); ++i) diff[i] = run.res[1][i] - expected[i];
            CHECK(norm2(diff) <= 1.0e-9 * norm2(d.cols[1]), "inconsistent RHS: residual %.3e", norm2(diff));
        }
        std::printf("  inconsistent RHS: solved=%d decline=%s stop=%s formed_checks=%d\n", run.rep.solved ? 1 : 0,
                    rp::decline_name(run.rep.decline), rp::stop_name(run.rep.columns[1].stop),
                    run.rep.columns[1].formed_checks);
        check_threads(d, base_config(1), "inconsistent RHS");
    }

    // The diagonal and the stopping measure.
    {
        Design d = random_panel(81, 70, 6, {14, 5}, 0);
        check_diagonal(d, "diagonal random");
        check_tau_against_lsqr(d, "tau vs LSQR");
        Design two = random_panel(82, 70, 6, {12}, 0);
        check_tau_against_lsqr(two, "tau vs LSQR (K=2)");
    }

    // Declines.
    {
        Design d = random_panel(91, 120, 6, {20, 6}, 1);
        rp::Config cfg = base_config(1);
        cfg.max_iter = 2;
        Run run = run_kernel(d, cfg);
        CHECK(!run.rep.solved && run.rep.decline == rp::Decline::Ceiling, "ceiling: %s", rp::decline_name(run.rep.decline));
        // A declined attempt still reports how far each column got.
        CHECK(run.rep.iterations == 2 && run.rep.decline_column >= 0 && run.rep.columns[0].iterations == 2 &&
                  run.rep.columns[0].tau > 0.0 && run.rep.columns[0].tau < 1.0e300 && run.rep.iterate_seconds > 0.0,
              "ceiling report: iterations=%d column=%d col0 iterations=%d tau=%.3e", run.rep.iterations,
              run.rep.decline_column, run.rep.columns[0].iterations, run.rep.columns[0].tau);
        cfg = base_config(1);
        cfg.memory_budget = 1;
        run = run_kernel(d, cfg);
        CHECK(!run.rep.solved && run.rep.decline == rp::Decline::Memory, "memory: %s", rp::decline_name(run.rep.decline));
        cfg = base_config(1);
        cfg.force_decline_geometry = true;
        run = run_kernel(d, cfg);
        CHECK(!run.rep.solved && run.rep.decline == rp::Decline::Forced, "forced geometry: %s", rp::decline_name(run.rep.decline));
        cfg = base_config(1);
        cfg.force_decline_iterate = true;
        run = run_kernel(d, cfg);
        CHECK(!run.rep.solved && run.rep.decline == rp::Decline::Forced, "forced iterate: %s", rp::decline_name(run.rep.decline));
        cfg = base_config(1);
        cfg.test_zero_curvature_at = 2;
        run = run_kernel(d, cfg);
        CHECK(!run.rep.solved && run.rep.decline == rp::Decline::Breakdown, "zero curvature: %s", rp::decline_name(run.rep.decline));
        Design nan = d;
        nan.cols[1][17] = std::numeric_limits<double>::quiet_NaN();
        run = run_kernel(nan, base_config(1));
        CHECK(!run.rep.solved && run.rep.decline == rp::Decline::NonFinite, "NaN input: %s", rp::decline_name(run.rep.decline));
        // A long chain of workers and firms with an unreachable floor: the
        // stall or the projection must end the attempt before the ceiling.
        Design chain;
        chain.codes.assign(2, {});
        std::vector<double> y;
        for (int u = 0; u < 400; ++u) {
            chain.codes[0].push_back(u);
            chain.codes[1].push_back(u / 2);
            y.push_back(std::sin(0.37 * u));
            chain.codes[0].push_back(u);
            chain.codes[1].push_back(std::min(199, u / 2 + 1));
            y.push_back(std::cos(0.11 * u));
        }
        chain.levels = {400, 200};
        chain.n = static_cast<int>(y.size());
        chain.cols.push_back(y);
        cfg = base_config(1);
        cfg.atol = 1.0e-40;
        cfg.btol = 1.0e-40;
        run = run_kernel(chain, cfg);
        CHECK(!run.rep.solved && (run.rep.decline == rp::Decline::Stall || run.rep.decline == rp::Decline::Projected),
              "unreachable floor: %s after %d", rp::decline_name(run.rep.decline), run.rep.iterations);
        CHECK(run.rep.iterations > 0 && run.rep.columns[0].iterations == run.rep.iterations,
              "unreachable floor report: iterations=%d", run.rep.iterations);
        std::printf("  unreachable floor: decline=%s after %d iterations\n", rp::decline_name(run.rep.decline),
                    run.rep.iterations);
        check_threads(chain, cfg, "unreachable floor");
        // The same chain at the real floor: solved to the oracle.
        standard_case(chain, "chain");
    }

    // A mid-size panel (about 2M rows) where the class rule itself mixes the
    // effects (firm transposed, year private). No dense oracle at this size:
    // the certificate measure of every column, the agreement with a solve with
    // every effect transposed, and the bits at 1, 8 and 16 threads.
    {
        const Design d = mobility_panel(400, 400000, 20000, 20, 0.15);
        rp::Config cfg = base_config(1);
        const Run run = run_kernel(d, cfg);
        CHECK(run.rep.solved, "mid-size: declined (%s)", rp::decline_name(run.rep.decline));
        CHECK(run.rep.large_effects == 1, "mid-size: %d large effects, expected the firm only", run.rep.large_effects);
        rp::Config transposed = cfg;
        transposed.force_effect_class = 1;
        const Run other = run_kernel(d, transposed);
        for (std::size_t c = 0; run.rep.solved && c < d.cols.size(); ++c) {
            const double cert = certificate(d, d.cols[c], run.res[c]);
            CHECK(cert <= 1.0e-11, "mid-size col %zu: certificate measure %.3e", c, cert);
            std::vector<double> diff(static_cast<std::size_t>(d.n));
            for (int i = 0; i < d.n; ++i) diff[static_cast<std::size_t>(i)] = run.res[c][static_cast<std::size_t>(i)] - other.res[c][static_cast<std::size_t>(i)];
            CHECK(norm2(diff) <= 1.0e-8 * norm2(d.cols[c]), "mid-size col %zu: mixed vs transposed %.3e", c, norm2(diff));
        }
        std::printf("  mid-size threads=1: %.2f ms per iteration\n",
                    1.0e3 * run.rep.iterate_seconds / std::max(1, run.rep.iterations));
        for (const int t : {8, 16}) {
            cfg.threads = t;
            const Run threaded = run_kernel(d, cfg);
            bool same = threaded.rep.iterations == run.rep.iterations;
            for (std::size_t c = 0; same && c < run.res.size(); ++c)
                same = std::memcmp(run.res[c].data(), threaded.res[c].data(), sizeof(double) * run.res[c].size()) == 0;
            CHECK(same, "mid-size: threads=%d differ from threads=1", t);
            std::printf("  mid-size threads=%d: %.2f ms per iteration\n", t,
                        1.0e3 * threaded.rep.iterate_seconds / std::max(1, threaded.rep.iterations));
        }
        std::printf("  mid-size: n=%d solved=%d iterations=%d large=%d cells=%lld geometry=%.2fs iterate=%.2fs\n", d.n,
                    run.rep.solved ? 1 : 0, run.rep.iterations, run.rep.large_effects, run.rep.cells,
                    run.rep.geometry_seconds, run.rep.iterate_seconds);
    }

    // The kernel of the operator's segment pass for large effects
    // (Config::prefetch_cells, internal::hinted_segment_part, 21nov2026) gives
    // the bits of the loop it replaces. A mobility panel whose firm rows exceed
    // internal::kPrefetchMinBytes at every lane width, solved with the kernel
    // off and on: at 4 lanes at three distances and at 1 and 16 threads; at 1,
    // 2 and 8 lanes at the default distance (the 8-lane group carries a zero
    // column and a column in the span of the eliminated effect); three random
    // panels with one, three and four remaining effects. Report::hinted_applies
    // shows that the kernel ran (where the build has it:
    // internal::kHintedSegmentKernel). Then the two shapes that keep the loop
    // whatever the field says: every other row three
    // times over with the cells merged (multiplicities), and the mid-size panel
    // (rows that fit in cache). Kept out of the digest, so that the digest
    // still compares with the builds before the kernel.
    {
        const std::uint64_t digest_before = g_digest;
        const auto same_run = [](const Run& off, const Run& on) {
            bool same = on.rep.solved == off.rep.solved && on.rep.decline == off.rep.decline &&
                        on.rep.iterations == off.rep.iterations;
            for (std::size_t c = 0; same && c < off.rep.columns.size(); ++c) {
                const auto& a = off.rep.columns[c];
                const auto& b = on.rep.columns[c];
                same = a.stop == b.stop && a.iterations == b.iterations && same_bits(a.tau, b.tau) && same_bits(a.chat, b.chat);
            }
            for (std::size_t c = 0; same && c < off.res.size(); ++c)
                same = std::memcmp(off.res[c].data(), on.res[c].data(), sizeof(double) * off.res[c].size()) == 0;
            return same;
        };
        const Design big = mobility_panel(401, 900000, 270000, 20, 0.15);
        for (const int lanes : {4, 1, 2, 8}) {
            Design d = big;
            if (lanes == 1) d.cols.resize(1);
            if (lanes == 2) d.cols.resize(2);
            if (lanes == 8) {
                d.cols.push_back(std::vector<double>(static_cast<std::size_t>(d.n), 0.0));  // zero column
                std::vector<double> span(static_cast<std::size_t>(d.n));                    // a worker effect only
                for (int i = 0; i < d.n; ++i) span[static_cast<std::size_t>(i)] = 0.001 * (d.codes[0][static_cast<std::size_t>(i)] % 977);
                d.cols.push_back(span);
                std::vector<double> mix(static_cast<std::size_t>(d.n));
                for (int i = 0; i < d.n; ++i)
                    mix[static_cast<std::size_t>(i)] = d.cols[1][static_cast<std::size_t>(i)] * d.cols[2][static_cast<std::size_t>(i)] * 1.0e-3;
                d.cols.push_back(mix);
            }
            const std::size_t bytes = static_cast<std::size_t>(d.levels[1]) * static_cast<std::size_t>(lanes) * sizeof(double);
            CHECK(bytes >= rp::internal::kPrefetchMinBytes, "kernel lanes=%d: %zu bytes below the threshold", lanes, bytes);
            for (const int t : {1, 16}) {
                if (lanes != 4 && t == 1) continue;
                rp::Config cfg = base_config(t);
                cfg.prefetch_cells = 0;
                const Run off = run_kernel(d, cfg);
                CHECK(off.rep.solved && off.rep.lanes == lanes && !off.rep.merged && off.rep.hinted_applies == 0,
                      "kernel lanes=%d: solved=%d lanes=%d merged=%d hinted_applies=%lld", lanes, off.rep.solved ? 1 : 0, off.rep.lanes,
                      off.rep.merged ? 1 : 0, off.rep.hinted_applies);
                for (const int ahead : {48, 1, 4096}) {
                    if (lanes != 4 && ahead != 48) continue;
                    cfg.prefetch_cells = ahead;
                    const Run on = run_kernel(d, cfg);
                    CHECK(on.rep.iterations > 0 && (rp::internal::kHintedSegmentKernel ? on.rep.hinted_applies >= on.rep.iterations
                                                                                        : on.rep.hinted_applies == 0),
                          "kernel lanes=%d threads=%d distance %d: %lld applies through the kernel, %d iterations", lanes, t, ahead,
                          on.rep.hinted_applies, on.rep.iterations);
                    CHECK(same_run(off, on), "kernel lanes=%d threads=%d: distance %d differs from the loop", lanes, t, ahead);
                }
                std::printf("  kernel lanes=%d threads=%d: n=%d cols=%zu iterations=%d\n", lanes, t, d.n, d.cols.size(), off.rep.iterations);
            }
        }
        for (const std::vector<int>& remaining :
             {std::vector<int>{90000}, std::vector<int>{90000, 400, 30}, std::vector<int>{90000, 400, 30, 6}}) {
            const Design d = random_panel(31 + static_cast<unsigned>(remaining.size()), 300000, 8, remaining, 2);
            for (const int t : {1, 16}) {
                rp::Config cfg = base_config(t);
                cfg.prefetch_cells = 0;
                const Run off = run_kernel(d, cfg);
                cfg.prefetch_cells = 48;
                const Run on = run_kernel(d, cfg);
                CHECK(off.rep.solved && off.rep.lanes == 4 && off.rep.hinted_applies == 0 && on.rep.iterations > 0 &&
                          (rp::internal::kHintedSegmentKernel ? on.rep.hinted_applies >= on.rep.iterations : on.rep.hinted_applies == 0),
                      "kernel %zu remaining effects threads=%d: solved=%d lanes=%d applies off=%lld on=%lld iterations=%d", remaining.size(), t,
                      off.rep.solved ? 1 : 0, off.rep.lanes, off.rep.hinted_applies, on.rep.hinted_applies, on.rep.iterations);
                CHECK(same_run(off, on), "kernel %zu remaining effects threads=%d: differs from the loop", remaining.size(), t);
                std::printf("  kernel %zu remaining effects threads=%d: n=%d iterations=%d large=%d\n", remaining.size(), t, d.n,
                            off.rep.iterations, off.rep.large_effects);
            }
        }
        {
            const Design base = mobility_panel(402, 250000, 80000, 20, 0.15);
            Design d = base;
            for (auto& c : d.codes) c.clear();
            for (auto& col : d.cols) col.clear();
            for (int i = 0; i < base.n; ++i)
                for (int r = 0; r < (i % 2 == 0 ? 3 : 1); ++r) {  // rows stay sorted by worker
                    for (std::size_t b = 0; b < base.codes.size(); ++b) d.codes[b].push_back(base.codes[b][static_cast<std::size_t>(i)]);
                    for (std::size_t c = 0; c < base.cols.size(); ++c) d.cols[c].push_back(base.cols[c][static_cast<std::size_t>(i)] + 0.25 * r);
                }
            d.n = static_cast<int>(d.codes[0].size());
            const Design small = mobility_panel(401, 400000, 20000, 20, 0.15);
            for (const Design* shape : std::initializer_list<const Design*>{&d, &small}) {
                const bool repeated = shape == &d;
                for (const int t : {1, 16}) {
                    rp::Config cfg = base_config(t);
                    if (repeated) cfg.force_merge = 1;
                    cfg.prefetch_cells = 0;
                    const Run off = run_kernel(*shape, cfg);
                    CHECK(off.rep.solved && off.rep.merged == repeated, "kernel %s: solved=%d merged=%d", repeated ? "repeated cells" : "small rows",
                          off.rep.solved ? 1 : 0, off.rep.merged ? 1 : 0);
                    cfg.prefetch_cells = 48;
                    const Run on = run_kernel(*shape, cfg);
                    CHECK(on.rep.hinted_applies == 0, "kernel %s threads=%d: the kernel ran on a shape that keeps the loop",
                          repeated ? "repeated cells" : "small rows", t);
                    CHECK(same_run(off, on), "kernel %s threads=%d: the field changed a shape that keeps the loop",
                          repeated ? "repeated cells" : "small rows", t);
                    std::printf("  kernel not used (%s) threads=%d: n=%d iterations=%d merged=%d\n", repeated ? "repeated cells" : "small rows", t,
                                shape->n, off.rep.iterations, off.rep.merged ? 1 : 0);
                }
            }
        }
        g_digest = digest_before;
    }

    // The merge scratch at 48 threads on the skewed shape (2.32, review round
    // 14): the results against the oracle and equal at every team size, and the
    // bound the kernel reports before allocating at least the peak it then
    // allocates, for the geometry alone and for the whole attempt. The previous
    // scratch (one segment of the largest size per thread) asked for
    // 48 * (n - 2) ints here. Kept out of the digest, so that the digest still
    // compares with the R1 build's.
    {
        const std::uint64_t digest_before = g_digest;
        const int n = 1000000;
        const Design d = skewed_pair(1414, n);
        standard_case(d, "skewed merge");
        rp::Config cfg = base_config(48);
        cfg.force_decline_geometry = true;
        const Metered geometry = metered_solve(d, cfg);
        CHECK(geometry.rep.merged, "skewed merge: the merge did not run");
        CHECK(geometry.rep.decline == rp::Decline::Forced, "skewed geometry: %s", rp::decline_name(geometry.rep.decline));
        CHECK(geometry.peak <= static_cast<long long>(geometry.rep.bytes),
              "skewed geometry: peak %lld bytes above the bound %zu", geometry.peak, geometry.rep.bytes);
        cfg.force_decline_geometry = false;
        const Metered whole = metered_solve(d, cfg);
        CHECK(whole.rep.solved, "skewed attempt: declined (%s)", rp::decline_name(whole.rep.decline));
        CHECK(whole.peak <= static_cast<long long>(whole.rep.bytes),
              "skewed attempt: peak %lld bytes above the bound %zu", whole.peak, whole.rep.bytes);
        std::printf("  skewed merge: n=%d threads=48 bound=%zu peak geometry=%lld peak attempt=%lld "
                    "previous scratch=%lld\n", n, whole.rep.bytes, geometry.peak, whole.peak,
                    48LL * static_cast<long long>(n - 2) * static_cast<long long>(sizeof(int)));
        // The same comparison on the mid-size mobility panel below (many
        // segments, so the O(L) terms of the bound are exercised too).
        const Design mid = mobility_panel(400, 400000, 20000, 20, 0.15);
        rp::Config mid_cfg = base_config(48);
        mid_cfg.force_decline_geometry = true;
        const Metered mid_geometry = metered_solve(mid, mid_cfg);
        mid_cfg.force_decline_geometry = false;
        const Metered mid_whole = metered_solve(mid, mid_cfg);
        CHECK(mid_geometry.peak <= static_cast<long long>(mid_geometry.rep.bytes),
              "mid-size geometry: peak %lld bytes above the bound %zu", mid_geometry.peak, mid_geometry.rep.bytes);
        CHECK(mid_whole.rep.solved && mid_whole.peak <= static_cast<long long>(mid_whole.rep.bytes),
              "mid-size attempt: peak %lld bytes above the bound %zu", mid_whole.peak, mid_whole.rep.bytes);
        std::printf("  mid-size metered: n=%d threads=48 merged=%d segments=%d bound=%zu peak geometry=%lld "
                    "peak attempt=%lld\n", mid.n, mid_whole.rep.merged ? 1 : 0, mid_whole.rep.segments,
                    mid_whole.rep.bytes, mid_geometry.peak, mid_whole.peak);
        g_digest = digest_before;
    }

    // Admission budget (2.32, review of 14 Nov 2026): max(1 GiB, 16 n (cols +
    // k)). The three narrow fits the auditor traced (event=plan bound) are
    // admitted, where the columns-only budget declined them; a design whose
    // bound still exceeds the budget is declined before it allocates.
    {
        struct Traced {
            const char* name;
            int n, cols, k;
            std::size_t bound;
        };
        const Traced traced[] = {
            {"ready, one regressor", 46156187, 2, 3, std::size_t{1856895080}},
            {"simulated_panel, one regressor", 173163263, 2, 4, std::size_t{8003956303}},
            {"simulated_panel, two regressors", 173163263, 3, 4, std::size_t{9063098715}},
        };
        for (const Traced& t : traced) {
            const std::size_t budget = rp::admission_budget(t.n, t.cols, t.k);
            const std::size_t columns_only =
                std::max<std::size_t>(std::size_t{1} << 30, 16U * static_cast<std::size_t>(t.n) * t.cols);
            CHECK(budget >= t.bound, "%s: budget %zu below the bound %zu", t.name, budget, t.bound);
            CHECK(columns_only < t.bound, "%s: the columns-only budget %zu admitted it", t.name, columns_only);
            std::printf("  admission %s: bound=%zu budget=%zu (columns only %zu)\n", t.name, t.bound, budget,
                        columns_only);
        }
        CHECK(rp::admission_budget(1000, 1, 2) == (std::size_t{1} << 30), "admission floor");
        CHECK(rp::admission_budget(100000000, 3, 2) == std::size_t{16} * 100000000U * 5U, "admission formula");

        // Declined: 600 000 rows, two effects of 300 000 levels, 1024 blocks.
        // The transposed effect's private partial sums (4 B m_j bytes) put the
        // bound above 1 GiB, the budget's floor; the kernel declines at
        // admission, before allocating them.
        const int rows = 600000, levels = 300000;
        Design wide;
        wide.name = "admission";
        wide.n = rows;
        wide.codes.assign(2, std::vector<int>(static_cast<std::size_t>(rows), 0));
        wide.levels = {levels, levels};
        wide.eliminated = 0;
        for (int i = 0; i < rows; ++i) {
            wide.codes[0][static_cast<std::size_t>(i)] = i / 2;
            wide.codes[1][static_cast<std::size_t>(i)] = (i / 2 + i % 2) % levels;
        }
        wide.cols = {span_column(wide, 77)};
        rp::Config cfg = base_config(1);
        cfg.blocks = 1024;
        cfg.memory_budget = rp::admission_budget(wide.n, static_cast<int>(wide.cols.size()),
                                                 static_cast<int>(wide.codes.size()));
        const Metered declined = metered_solve(wide, cfg);
        CHECK(!declined.rep.solved && declined.rep.decline == rp::Decline::Memory &&
                  declined.rep.bytes > cfg.memory_budget,
              "admission: %s with bound %zu and budget %zu", rp::decline_name(declined.rep.decline),
              declined.rep.bytes, cfg.memory_budget);
        std::printf("  admission declined: n=%d bound=%zu budget=%zu decline=%s\n", wide.n, declined.rep.bytes,
                    cfg.memory_budget, rp::decline_name(declined.rep.decline));
    }

    std::printf("digest %016llx\n", static_cast<unsigned long long>(g_digest));
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
