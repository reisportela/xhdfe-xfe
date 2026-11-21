// Point 19 (17 November 2026): the reduced tier sums the levels of the effects it does not
// eliminate in 64 bits before anything reads the sum (build_geometry in
// src/reduced_schur_pcg.hpp), and declines with Decline::Input when the sum does not fit
// the tier's int index. Header-level: no device, no data, no library. Built with
// -fsanitize=signed-integer-overflow -fno-sanitize-recover=all (in trap mode, which needs
// no sanitizer runtime), so that a sum formed in int stops the test. Every caller of
// build_geometry is exercised: the CPU tier's solve(), the CUDA controller's solve() (with a
// CPU operator that must not be prepared) and schur_diagonal().
//
//   reduced_pcg_level_sum_contract [all|overflow|exact|ordinary]
//
//   overflow  problems of six rows whose remaining effects declare 2^31 levels or more (two
//             to four remaining effects): every caller declines with Decline::Input, reports
//             no admission bytes, never prepares the operator, and nothing larger than 1 MiB
//             is allocated during the call;
//   exact     the same shapes declaring exactly INT_MAX remaining levels: the guard lets them
//             through, the admission (budget 64 MiB) declines them with Decline::Memory and
//             nonzero bytes, nothing larger than 1 MiB allocated;
//   ordinary  ordinary shapes (two to five effects, rows sorted or not, 1 to 11 columns)
//             solve; the CPU tier and the CUDA controller agree bit for bit; every residual
//             has a mean within each level of each effect below 1e-8 of the column's largest
//             value; one line ORDINARY_DIGEST <hash> per shape, the bits of the outcome and of
//             the residuals, for comparison with a build of the header before the guard.
// Prints LEVEL_SUM_CONTRACT_PASS <checks> at the end; any failure exits nonzero.
#include "reduced_schur_pcg_cuda_controller.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace rp = hdfe::reduced_pcg;

namespace {

// Largest single allocation while tracking is on.
bool tracking = false;
std::size_t largest = 0;
void note(std::size_t bytes) noexcept {
    if (tracking && bytes > largest) largest = bytes;
}
void* plain(std::size_t bytes) {
    note(bytes);
    if (void* p = std::malloc(bytes != 0 ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void* aligned(std::size_t bytes, std::size_t alignment) {
    note(bytes);
#if defined(_WIN32)
    if (void* p = _aligned_malloc(bytes != 0 ? bytes : 1, alignment)) return p;
    throw std::bad_alloc();
#else
    void* p = nullptr;
    if (posix_memalign(&p, std::max(alignment, sizeof(void*)), bytes != 0 ? bytes : 1) != 0) throw std::bad_alloc();
    return p;
#endif
}
void release_aligned(void* p) noexcept {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

}  // namespace

void* operator new(std::size_t bytes) { return plain(bytes); }
void* operator new[](std::size_t bytes) { return plain(bytes); }
void* operator new(std::size_t bytes, std::align_val_t a) { return aligned(bytes, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t bytes, std::align_val_t a) { return aligned(bytes, static_cast<std::size_t>(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { release_aligned(p); }
void operator delete[](void* p, std::align_val_t) noexcept { release_aligned(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { release_aligned(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { release_aligned(p); }

namespace {

int checks = 0;
void expect(bool ok, const std::string& what) {
    ++checks;
    if (!ok) throw std::runtime_error(what);
}

// A CPU operator for the CUDA controller; prepare() records that the set-up was reached.
struct CpuOperator {
    int prepared = 0;
    void prepare(const rp::internal::Geometry&, const rp::Config&, int, std::size_t) { ++prepared; }
    bool replacement_on_device() const noexcept { return false; }
    void begin_group(int) {}
    void pin_host(double*, std::size_t) {}
    void unpin_host() noexcept {}
    template <int W>
    void apply(const rp::internal::Geometry& g, const rp::internal::Team& team, const double* in, double* out,
               double* T, rp::internal::Work& work) {
        rp::internal::apply_operator<W>(g, team, in, out, T, work);
    }
    template <int W>
    void apply_replacement(const rp::internal::Geometry& g, const rp::internal::Team& team, const double* x,
                           double* q, double* T, rp::internal::Work& work) {
        rp::internal::apply_operator<W>(g, team, x, q, T, work);
    }
};

struct Shape {
    std::vector<int> levels;   // declared levels, eliminated effect first
    int n = 6;
};

// A problem of `n` rows whose effect b has codes in [0, levels[b]); the eliminated effect is 0.
struct Built {
    std::vector<std::vector<int>> codes;
    std::vector<const int*> code_ptrs;
    std::vector<int> levels;
    std::vector<std::vector<double>> columns, residuals;
    std::vector<const double*> in;
    std::vector<double*> out;
    rp::Problem problem;
};

Built build(const Shape& s, int cols) {
    Built b;
    const int k = static_cast<int>(s.levels.size());
    b.levels = s.levels;
    b.codes.assign(static_cast<std::size_t>(k), std::vector<int>(static_cast<std::size_t>(s.n)));
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < s.n; ++i) b.codes[j][i] = (i * (j + 1)) % std::min(s.levels[j], s.n);
    for (int j = 0; j < k; ++j) b.code_ptrs.push_back(b.codes[j].data());
    b.columns.assign(static_cast<std::size_t>(cols), std::vector<double>(static_cast<std::size_t>(s.n)));
    b.residuals = b.columns;
    for (int c = 0; c < cols; ++c)
        for (int i = 0; i < s.n; ++i) b.columns[c][i] = std::sin(0.7 * i + c) + 0.1 * c;
    for (int c = 0; c < cols; ++c) {
        b.in.push_back(b.columns[c].data());
        b.out.push_back(b.residuals[c].data());
    }
    b.problem.n = s.n;
    b.problem.k = k;
    b.problem.codes = b.code_ptrs.data();
    b.problem.levels = b.levels.data();
    b.problem.eliminated = 0;
    b.problem.cols = cols;
    b.problem.columns = b.in.data();
    b.problem.residuals = b.out.data();
    return b;
}

rp::Config config(std::size_t budget) {
    rp::Config cfg;
    cfg.threads = 4;
    cfg.blocks = 16;
    cfg.chat_stop = 1e-12;
    cfg.memory_budget = budget;
    return cfg;
}

// Every caller of build_geometry on a shape declaring huge levels: the expected decline,
// the admission bytes reported (zero or not), the operator never prepared, nothing large.
void declines(const Shape& s, rp::Decline want, bool bytes_expected, std::size_t budget, const char* label) {
    Built b = build(s, 2);
    const rp::Config cfg = config(budget);
    largest = 0;
    tracking = true;
    const rp::Report cpu = rp::solve(b.problem, cfg);
    CpuOperator op;
    const rp::Report cuda = hdfe::reduced_cuda_controller::solve(b.problem, cfg, op);
    const std::vector<double> diagonal = rp::schur_diagonal(b.problem, cfg);
    tracking = false;
    const std::string what = std::string(label) + ": ";
    expect(!cpu.solved && cpu.decline == want, what + "CPU solve() declined with " + rp::decline_name(cpu.decline));
    expect(!cuda.solved && cuda.decline == want, what + "controller declined with " + rp::decline_name(cuda.decline));
    expect(op.prepared == 0, what + "the controller prepared its operator");
    expect(diagonal.empty(), what + "schur_diagonal() built a geometry");
    expect((cpu.bytes > 0) == bytes_expected && (cuda.bytes > 0) == bytes_expected,
           what + "admission bytes " + std::to_string(cpu.bytes));
    expect(largest <= (std::size_t{1} << 20), what + "largest allocation " + std::to_string(largest));
    std::printf("CASE %s decline=%s bytes=%zu largest_allocation=%zu\n", label, rp::decline_name(cpu.decline),
                cpu.bytes, largest);
}

void overflow_cases() {
    const int big = 1 << 30;
    // Remaining sums 2^31, 2^31 and 2^31 + 2^30 + ... with two to four remaining effects;
    // the last one passes 2^32, where an int sum would wrap back to a small value.
    declines({{big + 7, big, big}}, rp::Decline::Input, false, SIZE_MAX, "overflow_k3_2^31");
    declines({{big + 7, big / 2, big / 2, big / 2, big / 2}}, rp::Decline::Input, false, SIZE_MAX,
             "overflow_k5_2^31");
    declines({{INT_MAX, INT_MAX, INT_MAX, INT_MAX, 3}}, rp::Decline::Input, false, SIZE_MAX,
             "overflow_k5_3x2^31");
    declines({{big + 7, big, big}}, rp::Decline::Input, false, 64U << 20, "overflow_k3_small_budget");
}

void exact_cases() {
    const int big = 1 << 30;
    // INT_MAX exactly: not caught by the guard, declined by the admission.
    declines({{big + 7, big, big - 1}}, rp::Decline::Memory, true, 64U << 20, "exact_k3_INT_MAX");
    declines({{big + 7, big / 2, big / 2, big / 2, big / 2 - 1}}, rp::Decline::Memory, true, 64U << 20,
             "exact_k5_INT_MAX");
}

std::uint64_t fnv(const void* data, std::size_t bytes, std::uint64_t h) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

void ordinary_cases() {
    struct Ordinary {
        int n, cols;
        std::vector<int> levels;   // eliminated (the largest) first
        bool shuffled;
    };
    const std::vector<Ordinary> shapes = {
        {3000, 1, {300, 40}, false},       {3000, 3, {300, 40}, true},
        {4096, 8, {256, 31, 7}, false},    {4096, 11, {256, 31, 7}, true},
        {6000, 2, {600, 45, 12, 5}, true}, {6000, 5, {600, 45, 12, 5, 3}, false},
    };
    int index = 0;
    for (const Ordinary& o : shapes) {
        const int n = o.n, k = static_cast<int>(o.levels.size());
        std::vector<std::vector<int>> codes(static_cast<std::size_t>(k), std::vector<int>(static_cast<std::size_t>(n)));
        for (int i = 0; i < n; ++i) {
            const int r = o.shuffled ? static_cast<int>((static_cast<long long>(i) * 7919) % n) : i;
            codes[0][i] = r * o.levels[0] / n;  // every level present
            for (int j = 1; j < k; ++j) codes[j][i] = (r * (2 * j + 3) + r / (j + 11)) % o.levels[j];
        }
        std::vector<const int*> ids;
        for (int j = 0; j < k; ++j) ids.push_back(codes[j].data());
        std::vector<std::vector<double>> y(static_cast<std::size_t>(o.cols), std::vector<double>(static_cast<std::size_t>(n)));
        for (int c = 0; c < o.cols; ++c)
            for (int i = 0; i < n; ++i)
                y[c][i] = std::sin(0.013 * i + 0.4 * c) + 0.5 * codes[0][i] / o.levels[0] - 0.2 * codes[k - 1][i] +
                          0.05 * std::cos(0.37 * i * (c + 1));
        std::vector<std::vector<double>> a(y), b(y);
        std::vector<const double*> in;
        std::vector<double*> oa, ob;
        for (int c = 0; c < o.cols; ++c) {
            in.push_back(y[c].data());
            oa.push_back(a[c].data());
            ob.push_back(b[c].data());
        }
        rp::Problem p;
        p.n = n;
        p.k = k;
        p.codes = ids.data();
        p.levels = o.levels.data();
        p.eliminated = 0;
        p.cols = o.cols;
        p.columns = in.data();
        p.residuals = oa.data();
        const rp::Config cfg = config(SIZE_MAX);
        const rp::Report cpu = rp::solve(p, cfg);
        p.residuals = ob.data();
        CpuOperator op;
        const rp::Report cuda = hdfe::reduced_cuda_controller::solve(p, cfg, op);
        const std::string what = "ordinary shape " + std::to_string(index) + ": ";
        expect(cpu.solved, what + "not solved (" + rp::decline_name(cpu.decline) + ")");
        expect(cuda.solved && cuda.iterations == cpu.iterations, what + "controller differs");
        for (int c = 0; c < o.cols; ++c)
            expect(std::memcmp(a[c].data(), b[c].data(), sizeof(double) * static_cast<std::size_t>(n)) == 0,
                   what + "controller residual bits differ");
        for (int c = 0; c < o.cols; ++c) {
            double scale = 0.0;
            for (int i = 0; i < n; ++i) scale = std::max(scale, std::abs(y[c][i]));
            for (int j = 0; j < k; ++j) {
                std::vector<double> sum(static_cast<std::size_t>(o.levels[j]), 0.0);
                std::vector<int> count(static_cast<std::size_t>(o.levels[j]), 0);
                for (int i = 0; i < n; ++i) {
                    sum[codes[j][i]] += a[c][i];
                    ++count[codes[j][i]];
                }
                double worst = 0.0;
                for (int l = 0; l < o.levels[j]; ++l)
                    if (count[l] > 0) worst = std::max(worst, std::abs(sum[l] / count[l]));
                expect(worst <= 1e-8 * scale, what + "a level mean of the residual is " + std::to_string(worst));
            }
        }
        std::uint64_t h = 1469598103934665603ULL;
        const int outcome[] = {cpu.solved ? 1 : 0, static_cast<int>(cpu.decline), cpu.iterations};
        h = fnv(outcome, sizeof(outcome), h);
        for (const rp::ColumnReport& col : cpu.columns) {
            const int stop[] = {static_cast<int>(col.stop), col.iterations, col.first_tau_pass, col.formed_checks};
            h = fnv(stop, sizeof(stop), h);
            h = fnv(&col.tau, sizeof(col.tau), h);
            h = fnv(&col.chat, sizeof(col.chat), h);
        }
        for (int c = 0; c < o.cols; ++c) h = fnv(a[c].data(), sizeof(double) * static_cast<std::size_t>(n), h);
        std::printf("ORDINARY_DIGEST shape=%d k=%d cols=%d iterations=%d hash=%016llx\n", index, k, o.cols,
                    cpu.iterations, static_cast<unsigned long long>(h));
        ++index;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string which = argc > 1 ? argv[1] : "all";
    try {
        if (which == "all" || which == "overflow") overflow_cases();
        if (which == "all" || which == "exact") exact_cases();
        if (which == "all" || which == "ordinary") ordinary_cases();
    } catch (const std::exception& e) {
        std::printf("LEVEL_SUM_CONTRACT_FAIL %s\n", e.what());
        return 1;
    }
    std::printf("LEVEL_SUM_CONTRACT_PASS %d checks (%s)\n", checks, which.c_str());
    return 0;
}
