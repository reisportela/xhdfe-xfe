#ifdef HDFE_USE_CUDA
// Experimental CUDA operator tier, OFF unless the fit requests it
// (HdfeOptions::reduced_pcg_cuda), else unless XHDFE_REDUCED_PCG_CUDA=1.
// The existing effective reduced-tier switch is also required by the caller.
// CPU recurrence/replacement/certification stay in their existing arithmetic.

#include "reduced_schur_pcg_entry.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "hdfe/deterministic_parallel.hpp"
#include "hdfe/ieee_bits.hpp"
#include "hdfe/parallel_work_observer.hpp"
#include "reduced_schur_pcg.hpp"
#include "sparse_rank.hpp"


#include "reduced_schur_pcg_cuda_layout.hpp"
#include "reduced_schur_pcg_cuda_memory.hpp"
#include "reduced_schur_pcg_cuda_controller.hpp"
namespace hdfe { namespace detail { namespace {
namespace rp=hdfe::reduced_pcg;
bool companion_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("XHDFE_REDUCED_PCG_COMPANION");
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }();
    return enabled;
}

// XHDFE_REDUCED_PCG_CHAT_STOP: "0" keeps the tier's stopping rule before D10
// (pre-registration Verifications/perf_stata_path_20261114/d10/); anything
// else, or nothing, applies D10 wherever its scope holds
// (reduced_pcg_chat_stop_applies). Read once per process; the stop, the
// declines and the cache tag all follow this one reading.
bool chat_stop_switch() {
    static const bool on = [] {
        const char* e = std::getenv("XHDFE_REDUCED_PCG_CHAT_STOP");
        return !(e != nullptr && e[0] == '0' && e[1] == '\0');
    }();
    return on;
}

// XHDFE_TEST_REDUCED_PCG_DECLINE: decline at a fixed stage (tests only);
// bad_alloc throws std::bad_alloc during the adapter's setup, which only the
// last-resort exit below can turn into a decline.
int forced_decline_stage() {
    static const int stage = [] {
        const char* e = std::getenv("XHDFE_TEST_REDUCED_PCG_DECLINE");
        if (e == nullptr) return 0;
        const std::string value(e);
        if (value == "geometry") return 1;
        if (value == "iterate") return 2;
        if (value == "residuals") return 3;
        if (value == "certificate") return 4;
        if (value == "bad_alloc") return 5;
        return 0;
    }();
    return stage;
}

// The adapter's last exit, reached when the setup or the reporting of a
// decline throws: the attempt is declined and the outcome says so without
// allocating (a decline already recorded keeps its reason and iterations);
// the final trace line is written only if that still works.
std::optional<AbsorptionResult> decline_last_resort(ReducedPcgOutcome* outcome, const char* reason,
                                                    int iterations) noexcept {
    if (outcome != nullptr && outcome->status != 2) {
        outcome->status = 2;
        outcome->reason = reason;
        outcome->iterations = iterations;
    }
    try {
        const char* recorded = outcome != nullptr ? outcome->reason : reason;
        const int spent = outcome != nullptr ? outcome->iterations : iterations;
        reduced_pcg_final_line(std::string("status=declined reason=") + recorded +
                               " column=-1 iterations=" + std::to_string(spent) + " rel=-1");
    } catch (...) {
    }
    return std::nullopt;
}

// The two tolerance formulas of src/fe_absorption.cpp, restated for this
// unit (keep them in step with the originals):
// effective_absorption_tolerance (fe_absorption.cpp:435-441) and
// krylov_parity_tolerance (fe_absorption.cpp:443-455) as MLSMR sees it,
// i.e. with absorption_method = Mlsmr.
double effective_tolerance(const HdfeOptions& options) {
    if (options.tol > 0.0 && options.tolerance_mode == ToleranceMode::ReghdfeComparable)
        return std::min(options.tol, 1.0e-9);
    return options.tol;
}

double mlsmr_parity_tolerance(const HdfeOptions& options) {
    const double tol = effective_tolerance(options);
    const bool parity_mode = options.tolerance_mode == ToleranceMode::ReghdfeComparable ||
                             options.tolerance_mode == ToleranceMode::StrictResidual;
    if (tol > 0.0 && parity_mode && options.ordinary_krylov_parity_floor) return std::min(tol, 1.0e-11);
    return tol;
}

void observer_begin(void* context, int team) {
    static_cast<ParallelWorkObserver*>(context)->begin_region(team);
}
void observer_observe(void* context) {
    static_cast<ParallelWorkObserver*>(context)->observe_work();
}
void observer_end(void* context) {
    static_cast<ParallelWorkObserver*>(context)->end_region();
}

struct CompactCodes {
    std::vector<int> codes;
    int levels = 0;
    int min_id = 0;  // of the raw ids, for the certificate (certify_tier_absorption)
    int max_id = 0;
};

// The sorted branch of compact_codes(): a sort of (id, row) pairs, codes in
// increasing order of the ids; returns the number of distinct ids.
int sorted_compact_codes(const int* src, int n, int* codes) {
    std::vector<std::pair<int, int>> keyed(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) keyed[static_cast<std::size_t>(i)] = {src[i], i};
    std::sort(keyed.begin(), keyed.end());
    int next = -1;
    int previous = 0;
    for (std::size_t i = 0; i < keyed.size(); ++i) {
        if (next < 0 || keyed[i].first != previous) {
            ++next;
            previous = keyed[i].first;
        }
        codes[static_cast<std::size_t>(keyed[i].second)] = next;
    }
    return next + 1;
}

// Codes 0..G-1 in increasing order of the raw ids, so G is the number of
// distinct ids (num_levels_present of the absorbers' indexers): a dense remap
// over [min, max] when that range is at most 2n; otherwise the same ranks from
// the bitmap index of src/sparse_rank.hpp on the team when it fits its budget
// (phase 1 of the Stata-path plan), and a sort of (id, row) pairs when it does
// not or XHDFE_SPARSE_RANK=0. Deterministic, and independent of the team.
CompactCodes compact_codes(const Eigen::VectorXi& raw, const rp::internal::Team& team, int blocks) {
    CompactCodes out;
    const int n = static_cast<int>(raw.size());
    out.codes.resize(static_cast<std::size_t>(n));
    if (n == 0) return out;
    const int* src = raw.data();
    const int parts = std::max(1, std::min(blocks, n));
    std::vector<int> low(static_cast<std::size_t>(parts)), high(static_cast<std::size_t>(parts));
    team.run(parts, [&](int b) {
        const int i0 = static_cast<int>(rp::internal::part_begin(n, b, parts));
        const int i1 = static_cast<int>(rp::internal::part_begin(n, b + 1, parts));
        int lo = src[i0], hi = src[i0];
        for (int i = i0 + 1; i < i1; ++i) {
            lo = std::min(lo, src[i]);
            hi = std::max(hi, src[i]);
        }
        low[static_cast<std::size_t>(b)] = lo;
        high[static_cast<std::size_t>(b)] = hi;
    });
    const int min_id = *std::min_element(low.begin(), low.end());
    const int max_id = *std::max_element(high.begin(), high.end());
    out.min_id = min_id;
    out.max_id = max_id;
    const long long range = static_cast<long long>(max_id) - static_cast<long long>(min_id) + 1LL;
    if (range > 0 && range <= 2LL * n) {
        const std::size_t size = static_cast<std::size_t>(range);
        std::unique_ptr<std::atomic<unsigned char>[]> seen(new std::atomic<unsigned char>[size]);
        for (std::size_t i = 0; i < size; ++i) seen[i].store(0, std::memory_order_relaxed);
        team.run(parts, [&](int b) {
            const int i0 = static_cast<int>(rp::internal::part_begin(n, b, parts));
            const int i1 = static_cast<int>(rp::internal::part_begin(n, b + 1, parts));
            for (int i = i0; i < i1; ++i) {
                std::atomic<unsigned char>& flag = seen[static_cast<std::size_t>(src[i] - min_id)];
                if (flag.load(std::memory_order_relaxed) == 0) flag.store(1, std::memory_order_relaxed);
            }
        });
        std::vector<int> remap(size, -1);
        int next = 0;
        for (std::size_t g = 0; g < size; ++g)
            if (seen[g].load(std::memory_order_relaxed) != 0) remap[g] = next++;
        team.run(parts, [&](int b) {
            const int i0 = static_cast<int>(rp::internal::part_begin(n, b, parts));
            const int i1 = static_cast<int>(rp::internal::part_begin(n, b + 1, parts));
            for (int i = i0; i < i1; ++i)
                out.codes[static_cast<std::size_t>(i)] = remap[static_cast<std::size_t>(src[i] - min_id)];
        });
        out.levels = next;
        return out;
    }
    const auto sparse_t0 = std::chrono::steady_clock::now();
    if (sparse_rank::enabled()) {
        const auto run = [&team](int units, auto&& body) { team.run(units, body); };
        sparse_rank::Index index;
        if (sparse_rank::build_index(src, n, min_id, max_id, run, index)) {
            sparse_rank::map_ranks(src, n, index, out.codes.data(), run);
            out.levels = index.levels;
            index.clear();
            if (sparse_rank::verify_enabled()) {
                std::vector<int> ref(static_cast<std::size_t>(n));
                const int ref_levels = sorted_compact_codes(src, n, ref.data());
                sparse_rank::verify_check(ref_levels == out.levels, "tier_codes", "levels");
                sparse_rank::verify_check(ref == out.codes, "tier_codes", "codes");
                sparse_rank::verify_ok("tier_codes");
            }
            sparse_rank::profile_line("tier_codes", "bitmap", "index", n, min_id, max_id,
                                      out.levels, team.threads(), sparse_t0);
            return out;
        }
    }
    out.levels = sorted_compact_codes(src, n, out.codes.data());
    sparse_rank::profile_line("tier_codes", "serial", sparse_rank::serial_reason(), n, min_id,
                              max_id, out.levels, 1, sparse_t0);
    return out;
}

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// XHDFE_SHARED_FE_CODING: "0" makes the attempt's certificate number the
// identifiers again, as before phase 2 of the Stata-path plan (design note
// Verifications/perf_stata_path_20261114/fase2/DESENHO_FASE2.md, section 6);
// anything else, or nothing, hands it the attempt's codes. Read once per process.
bool shared_fe_coding_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("XHDFE_SHARED_FE_CODING");
        return !(e != nullptr && e[0] == '0' && e[1] == '\0');
    }();
    return on;
}


// Trace-only record of the attribution line (event=attribution): consecutive
// stamps on one clock, so the named spans tile the attempt; the operator's own
// splits (set-up phases, registration, kernels) are in its Statistics. Never read
// without trace.
// The products of the attempt by kind, for the trace line event=replacement (package
// 3-lite): the iterative products S p and the replacements S x, on the device or on
// the CPU, counted and timed (host wall; the device's transfers and kernels of the
// replacements from the operator's events).
struct ProductCounts {
    long long iterative_device = 0, iterative_zero = 0, replacement_device = 0, replacement_cpu = 0;
    double iterative_seconds = 0.0, replacement_device_seconds = 0.0, replacement_cpu_seconds = 0.0;
    double replacement_h2d_ms = 0.0, replacement_kernel_ms = 0.0, replacement_d2h_ms = 0.0;
};
struct ReplacementCompare;

struct TierClocks {
    using Clock = std::chrono::steady_clock;
    Clock::time_point codes_from{}, codes_to{}, solve_from{}, solve_to{}, released{},
        certificate_from{}, certificate_to{};
    Clock::time_point view_from{}, view_to{}, operator_from{}, operator_to{};
    hdfe::reduced_cuda_controller::SolveStamps solve;
    // Package 3-lite: the products by kind, the doubles registered for the iteration,
    // and the qualification mode's state (set only when that mode is on).
    ProductCounts products;
    std::size_t registered_doubles = 0;
    ReplacementCompare* qualification = nullptr;
};

// XHDFE_TEST_REDUCED_PCG_CUDA_FAIL (tests only, read at every attempt): a failure
// injected at one point of the operator's set-up or registration (allocation,
// build, staging, register; see reduced_cuda::TestFailure).
reduced_cuda::TestFailure injected_failure() {
    const char* e = std::getenv("XHDFE_TEST_REDUCED_PCG_CUDA_FAIL");
    if (e == nullptr) return reduced_cuda::TestFailure::None;
    const std::string value(e);
    if (value == "allocation") return reduced_cuda::TestFailure::Allocation;
    if (value == "build") return reduced_cuda::TestFailure::Build;
    if (value == "staging") return reduced_cuda::TestFailure::Staging;
    if (value == "register") return reduced_cuda::TestFailure::Register;
    return reduced_cuda::TestFailure::None;
}

const char* decline_reason(reduced_cuda::ErrorKind kind) {
    return kind == reduced_cuda::ErrorKind::Memory ? "memory"
         : kind == reduced_cuda::ErrorKind::Pinned ? "pinned" : "gpu_unavailable";
}

// XHDFE_REDUCED_PCG_CUDA_REPLACEMENT (package 3-lite; on by default since 17
// November 2026, the owner's decision): the residual replacement S x is computed
// with the device operator, only where D10 is in force for the attempt
// (config.chat_stop > 0); elsewhere, or with the variable exactly "0", the
// replacement stays the CPU product. Read once per process.
bool replacement_switch() {
    static const bool on = [] {
        const char* e = std::getenv("XHDFE_REDUCED_PCG_CUDA_REPLACEMENT");
        return !(e != nullptr && e[0] == '0' && e[1] == '\0');
    }();
    return on;
}

// XHDFE_TEST_REDUCED_PCG_CUDA_REPLACEMENT_COMPARE (qualification only, read at
// every attempt; never in the default path): exactly "1" computes both products at
// every replacement and prints one comparison line per replacement and lane;
// "strict" does the same with a zero bound (any difference is a divergence: the
// test of the refusal replacement_divergence).
int replacement_compare_requested() {
    const char* e = std::getenv("XHDFE_TEST_REDUCED_PCG_CUDA_REPLACEMENT_COMPARE");
    if (e == nullptr) return 0;
    if (e[0] == '1' && e[1] == '\0') return 1;
    return std::string(e) == "strict" ? 2 : 0;
}

// XHDFE_TEST_REDUCED_PCG_CUDA_FAIL=replacement (tests only, read at every attempt):
// a fatal error at the first replacement whose product the device computes, once per
// process, so that the same process can then show that it fits again.
bool injected_replacement_failure() {
    const char* e = std::getenv("XHDFE_TEST_REDUCED_PCG_CUDA_FAIL");
    return e != nullptr && std::string(e) == "replacement";
}
bool replacement_fault_now() {
    static std::atomic<bool> fired{false};
    return !fired.exchange(true);
}

// XHDFE_TEST_REDUCED_PCG_CUDA_RECURRENCE_CHAT=k:value (tests only, read at every
// attempt): Config's existing test hook, the recurrence's chat read as `value` at
// iteration k (a candidate the replacement then confirms or rejects).
void recurrence_chat_hook(rp::Config& config) {
    const char* e = std::getenv("XHDFE_TEST_REDUCED_PCG_CUDA_RECURRENCE_CHAT");
    if (e == nullptr) return;
    const std::string s(e);
    const std::size_t colon = s.find(':');
    if (colon == std::string::npos) return;
    try {
        const int at = std::stoi(s.substr(0, colon));
        const double value = std::stod(s.substr(colon + 1));
        config.test_recurrence_chat_at = at;
        config.test_recurrence_chat_value = value;
    } catch (const std::exception&) {
    }
}

// The qualification mode of package 3-lite. At every replacement both products S x
// are computed on the same x (the device's is the one used when 3-lite is in force,
// the CPU's otherwise) and compared lane by lane. run_group's lane state is not
// visible from the adapter, so a shadow follows it from what the adapter sees, with
// run_group's expressions and sum order: the RHS sums (work.stats, read at the
// group's first product, before a formed residual reuses them) give each lane's
// denominator, w2 and wnorm; the iterative products, x, r and p give rho, alpha,
// beta, the Lanczos trace a2, s2 and the formed state; a formed residual of its own
// (the same function, into T, free between products, with partials of its own) gives
// s_formed. This unit is built with -ffast-math, so the same expressions may round
// differently here and in run_group (a few units in the last place): the controller's
// tau and chat, received through Config::on_iteration, check the shadow at every
// iteration, and each line carries its verdict (Pending, below). The bound of
// |S x (device) - S x (CPU)| per row is
// the one of the operator smoke test and oracle (2 gamma_K B_i, K = 2 E_i + Cmax + nb +
// parts + 8); the bounds of chat and s2 add the rounding of the residual's formation
// and of its sums (RELATORIO_P3.md gives the derivation).
struct ShadowLane {
    bool active = false, formed = false;
    double rho = 0.0, rho_old = 0.0, alpha = 0.0, alpha_prev = 0.0, beta = 0.0, a2 = 0.0, s2 = 0.0;
    double w2 = 0.0, wnorm = 0.0, denom = 0.0, s_formed = 0.0;
    bool expect = false;           // an expectation of (tau, chat) for the iteration just done
    double tau = 0.0, chat = 0.0;
};

struct ReplacementCompare {
    using Clock = std::chrono::steady_clock;
    const rp::Problem* problem = nullptr;
    rp::Config cfg;
    int lanes = 0, width = 0, group = -1, k = 0;
    bool initialized = false, pending_recurrence = false;
    ShadowLane lane[8];
    rp::internal::Work own;                 // lane sums' units and the formed residual's partials
    std::vector<double> other;              // the product not used (m x lanes)
    std::vector<long long> entries;         // E_i of the bound (per level), with cmax and parts
    int cmax = 0, parts = 0;
    std::vector<std::vector<long double>> magnitude;  // B_i per lane
    struct Hook { int k = -1; double tau = 0.0, chat = 0.0; };
    std::vector<Hook> hook;                 // the controller's last (tau, chat) per column
    // A comparison line waiting for the controller's tau and chat of its iteration. Layer 1 (both products, the
    // difference and its bound, chat and s2 per product, the Chat confirmation and the s2 floor crossing) is
    // verified when the used product's chat equals the controller's bit for bit. Layer 2 (the formed state, test 1,
    // the stop) also needs tau: exact when tau is equal bit for bit; "margin" when tau is within 1e-14 relative and
    // every layer-2 margin is at least 1e-12 relative (a rounding difference of the shadow cannot move the
    // decision); else unverified.
    struct Pending {
        int lane = 0, column = 0, k = 0;
        std::string fields;
        double test1_margin = 0.0, s2_margin = 0.0;
    };
    std::vector<Pending> pending;
    // A difference above its bound on some row (round 28, adjudication 4: with the comparison running, a divergence
    // above a valid bound declines the attempt with its own reason, replacement_divergence, even if the certificate
    // would pass; a bound too wide to conclude is not a divergence).
    bool divergent = false, strict = false;
    long long lines = 0, differ = 0, within = 0, checked = 0, chat_mismatch = 0, tau_mismatch = 0;
    long long chat_verified = 0, layer2_exact = 0, layer2_margin = 0, layer2_unverified = 0;
    double tau_rel_max = 0.0, chat_rel_max = 0.0;
    double cpu_seconds = 0.0, device_seconds = 0.0, shadow_seconds = 0.0;

    void prepare(const rp::Config& c, int W) {
        cfg = c;
        lanes = W;
        hook.assign(static_cast<std::size_t>(std::max(0, problem != nullptr ? problem->cols : 0)), Hook{});
    }
    double* other_rows() { return other.data(); }
    void on_hook(int column, int iteration, double tau, double chat) {
        if (column < 0 || column >= static_cast<int>(hook.size())) return;
        hook[static_cast<std::size_t>(column)] = Hook{iteration, tau, chat};
    }
    static void hook_entry(void* context, int column, int iteration, double tau, double chat) {
        static_cast<ReplacementCompare*>(context)->on_hook(column, iteration, tau, chat);
    }
    int col0() const {
        const int cols = problem->cols, groups = (cols + 7) / 8, per_group = (cols + groups - 1) / groups;
        return group * per_group;
    }
    static bool same(double a, double b) {
        std::uint64_t x = 0, y = 0;
        std::memcpy(&x, &a, sizeof(x));
        std::memcpy(&y, &b, sizeof(y));
        return x == y;
    }
    // The expectations of the iteration just done against the controller's hook values;
    // the pending lines are printed with the verdict.
    static double rel(double a, double b) {
        if (same(a, b)) return 0.0;
        const double scale = std::max(std::fabs(a), std::fabs(b));
        return scale > 0.0 && std::isfinite(scale) ? std::fabs(a - b) / scale : 1.0;
    }
    void flush() {
        for (int w = 0; w < width; ++w) {
            ShadowLane& ln = lane[w];
            if (!ln.expect) continue;
            ln.expect = false;
            const Hook& h = hook[static_cast<std::size_t>(col0() + w)];
            ++checked;
            const double rc = h.k == k ? rel(h.chat, ln.chat) : 1.0, rt = h.k == k ? rel(h.tau, ln.tau) : 1.0;
            chat_mismatch += rc > 0.0 ? 1 : 0;
            tau_mismatch += rt > 0.0 ? 1 : 0;
            chat_rel_max = std::max(chat_rel_max, rc);
            tau_rel_max = std::max(tau_rel_max, rt);
            if ((rc > 0.0 || rt > 0.0) && chat_mismatch + tau_mismatch <= 40) {  // the first ones, for the report
                std::ostringstream o;
                o.precision(17);
                o << "xhdfe reduced_cuda event=shadow_mismatch group=" << group << " col=" << col0() + w << " k=" << k
                  << " hook_k=" << h.k << " hook_tau=" << h.tau << " shadow_tau=" << ln.tau << " hook_chat=" << h.chat
                  << " shadow_chat=" << ln.chat << " formed=" << (ln.formed ? 1 : 0);
                std::cerr << o.str() << '\n';
            }
        }
        for (const Pending& line : pending) {
            const Hook& h = hook[static_cast<std::size_t>(line.column)];
            const ShadowLane& ln = lane[line.lane];
            const bool chat_ok = h.k == line.k && same(h.chat, ln.chat);
            const double rt = h.k == line.k ? rel(h.tau, ln.tau) : 1.0;
            const char* layer2 = !chat_ok ? "unverified"
                               : rt == 0.0 ? "exact"
                               : rt <= 1e-14 && std::fabs(line.test1_margin) >= 1e-12 && std::fabs(line.s2_margin) >= 1e-12 ? "margin"
                                                                                                                       : "unverified";
            chat_verified += chat_ok ? 1 : 0;
            layer2_exact += std::strcmp(layer2, "exact") == 0 ? 1 : 0;
            layer2_margin += std::strcmp(layer2, "margin") == 0 ? 1 : 0;
            layer2_unverified += std::strcmp(layer2, "unverified") == 0 ? 1 : 0;
            std::ostringstream o;
            o.precision(3);
            o << " tau_rel_diff=" << rt << " chat_verified=" << (chat_ok ? 1 : 0) << " layer2=" << layer2;
            std::cerr << line.fields << o.str() << '\n';
        }
        pending.clear();
    }
    void begin_group(int w) {
        flush();
        ++group;
        width = w;
        k = 0;
        initialized = false;
        pending_recurrence = false;
        for (ShadowLane& ln : lane) ln = ShadowLane{};
    }

    // run_group's evaluate on a copy of the lane: tau, chat and the stop it would set.
    rp::Stop evaluate(const ShadowLane& ln, double rho, double rr, double xmx, double& tau, double& chat) const {
        const double s = ln.formed ? ln.s_formed : (ln.s2 > 0.0 ? std::sqrt(ln.s2) : 0.0);
        const double a = std::sqrt(std::max(0.0, ln.a2));
        if (!(rho > 0.0)) tau = 0.0;
        else tau = (s > 0.0 && a > 0.0) ? std::sqrt(rho) / (a * s) : rp::internal::kInf;
        chat = std::sqrt(std::max(0.0, rr)) / ln.denom;
        const bool test2 = tau <= cfg.atol;
        const bool test1 = (ln.formed || ln.s2 > 0.0) &&
                           s <= cfg.btol * ln.wnorm + cfg.atol * a * std::sqrt(std::max(0.0, xmx));
        const bool companion_ok = !cfg.companion || chat <= cfg.trigger;
        if (cfg.chat_stop > 0.0) {
            const bool test_chat = chat <= cfg.chat_stop;
            return companion_ok ? (test_chat ? rp::Stop::Chat : (test1 ? rp::Stop::Test1 : rp::Stop::None)) : rp::Stop::None;
        }
        return companion_ok ? (test2 ? rp::Stop::Test2 : (test1 ? rp::Stop::Test1 : rp::Stop::None)) : rp::Stop::None;
    }

    // The group's first product: the lanes' constants from the RHS sums, as run_group forms them.
    template <int W>
    void initialise(const rp::internal::Geometry& g, const rp::internal::Team& team, rp::internal::Work& work) {
        const int m = g.m, L = g.L, n = problem->n;
        if (own.units.size() == 0) {
            own.units.allocate((static_cast<std::size_t>(m) / rp::internal::kVectorUnit + 1U) * 8U * static_cast<std::size_t>(lanes));
            own.stats.allocate((g.row_bounds.size() + 1U) * rp::internal::kStats * static_cast<std::size_t>(lanes));
            other.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(lanes), 0.0);
        }
        const int rhs_parts = std::max(1, std::min(static_cast<int>(g.row_bounds.size()) - 1,
                                                   std::max(1, (4 * m + L) / std::max(1, m))));
        const std::vector<int> rhs_bounds = rp::internal::balanced_bounds(g.row_ptr.data(), L, rhs_parts);
        const int rparts = static_cast<int>(rhs_bounds.size()) - 1;
        const double* stats = work.stats.data();
        constexpr int kS = rp::internal::kStats;
        long double tot[kS * W];
        for (int s = 0; s < kS * W; ++s) tot[s] = 0.0L;
        for (int b = 0; b < rparts; ++b)
            for (int s = 0; s < kS * W; ++s) tot[s] += stats[static_cast<std::size_t>(b) * kS * W + static_cast<std::size_t>(s)];
        const long double op_norm = std::sqrt(static_cast<long double>(problem->k) * static_cast<long double>(n));
        for (int w = 0; w < width; ++w) {
            ShadowLane& ln = lane[w];
            const long double raw = tot[0 * W + w];
            const long double sum = tot[2 * W + w];
            long double norm_sq = raw;
            if (cfg.dominant_origin && n > 0 && sum * (sum / n) > (1.0L - 1.0L / 1024.0L) * raw) {
                const long double s1 = tot[3 * W + w];
                const long double centered = tot[4 * W + w] - s1 * (s1 / n);
                if (centered > 0.0L && centered < raw / 1024.0L) norm_sq = centered;
            }
            ln.w2 = static_cast<double>(tot[1 * W + w]);
            ln.wnorm = std::sqrt(ln.w2);
            ln.denom = static_cast<double>(op_norm * std::sqrt(norm_sq));
            ln.s2 = ln.w2;
        }
        // rho = sum of minv r^2 over the live lanes, as run_group's first lane sums form it.
        const double* r = work.arena.data() + static_cast<std::size_t>(m) * W;
        const double* minv = g.minv.data();
        double rho[W];
        rp::internal::lane_sums<W, 1>(m, team, own, rho, [&](int i, double* acc) {
            const std::size_t base = static_cast<std::size_t>(i) * W;
            for (int w = 0; w < W; ++w) {
                const double rv = r[base + w];
                acc[w] += minv[i] * rv * rv;
            }
        });
        for (int w = 0; w < width; ++w) lane[w].rho = rho[w];
        initialized = true;
    }

    // The recurrence's update of the iteration just done (run_group's lane loop after
    // ++k), from x and r as they stand; the stop it would set is the expectation unless
    // a replacement follows. Returns the candidate and formed-entry flags per lane.
    template <int W>
    void recurrence(const rp::internal::Geometry& g, const rp::internal::Team& team, rp::internal::Work& work,
                    bool* candidate, bool* enter) {
        const int m = g.m;
        const double* x = work.arena.data();
        const double* r = x + static_cast<std::size_t>(m) * W;
        const double* minv = g.minv.data();
        const double* diag = g.diag.data();
        double tot[3 * W];
        rp::internal::lane_sums<W, 3>(m, team, own, tot, [&](int i, double* acc) {
            const std::size_t base = static_cast<std::size_t>(i) * W;
            for (int w = 0; w < W; ++w) {
                const double xv = x[base + w];
                const double rv = r[base + w];
                acc[0 * W + w] += minv[i] * rv * rv;
                acc[1 * W + w] += rv * rv;
                acc[2 * W + w] += diag[i] * xv * xv;
            }
        });
        for (int w = 0; w < W; ++w) {
            candidate[w] = false;
            enter[w] = false;
            if (w >= width) continue;
            ShadowLane& ln = lane[w];
            if (!ln.active) continue;
            const double rho_new = tot[0 * W + w];
            ln.a2 += 1.0 / ln.alpha + (k > 1 ? ln.beta / ln.alpha_prev : 0.0);
            if (!ln.formed) ln.s2 -= ln.alpha * ln.rho_old;
            ln.beta = rho_new / ln.rho_old;
            ln.alpha_prev = ln.alpha;
            ln.rho = rho_new;
            if (!ln.formed && !(ln.s2 > cfg.floor_factor * rp::internal::kEps * ln.w2)) enter[w] = true;
            double rr = tot[1 * W + w];
            if (cfg.test_recurrence_chat_at == k) {
                const double forced = cfg.test_recurrence_chat_value * ln.denom;
                rr = forced * forced;
            }
            candidate[w] = evaluate(ln, rho_new, rr, tot[2 * W + w], ln.tau, ln.chat) != rp::Stop::None;
            ln.expect = true;
        }
    }

    // Before the iterative product of a new iteration: the previous iteration's
    // recurrence when no replacement followed it, the check of its expectations, and
    // the lanes active in this one (a stopped lane's direction is +-0 in every row).
    template <int W>
    void before_iterative(const rp::internal::Geometry& g, const rp::internal::Team& team, const double* pv,
                          rp::internal::Work& work) {
        const auto t0 = Clock::now();
        if (!initialized) {
            initialise<W>(g, team, work);
        } else if (pending_recurrence) {
            bool candidate[W], enter[W];
            recurrence<W>(g, team, work, candidate, enter);
            pending_recurrence = false;
        }
        flush();
        const std::size_t m = static_cast<std::size_t>(g.m);
        for (int w = 0; w < width; ++w) {
            bool any = false;
            for (std::size_t i = 0; i < m && !any; ++i) any = pv[i * W + static_cast<std::size_t>(w)] != 0.0;
            lane[w].active = any;
            lane[w].rho_old = lane[w].rho;
        }
        shadow_seconds += std::chrono::duration<double>(Clock::now() - t0).count();
    }
    // After it: sigma = p' S p in run_group's order, alpha, and the iteration count.
    template <int W>
    void after_iterative(const rp::internal::Geometry& g, const rp::internal::Team& team, const double* pv, const double* q) {
        const auto t0 = Clock::now();
        double sigma[W];
        rp::internal::lane_sums<W, 1>(g.m, team, own, sigma, [&](int i, double* acc) {
            const std::size_t base = static_cast<std::size_t>(i) * W;
            for (int w = 0; w < W; ++w) acc[w] += pv[base + w] * q[base + w];
        });
        for (int w = 0; w < width; ++w)
            if (lane[w].active) lane[w].alpha = lane[w].rho / (cfg.test_zero_curvature_at == k + 1 ? 0.0 : sigma[w]);
        ++k;
        pending_recurrence = true;
        shadow_seconds += std::chrono::duration<double>(Clock::now() - t0).count();
    }

    // At a replacement: the recurrence of this iteration, then both products' fresh sums,
    // decisions and bounds; the shadow follows the used product.
    template <int W>
    void compare(const rp::internal::Geometry& g, const rp::internal::Team& team, rp::internal::Work& work,
                 const double* q_cpu, const double* q_dev, bool device_used, double* T);
};

const char* stop_label(rp::Stop s) {
    return s == rp::Stop::Chat ? "chat" : s == rp::Stop::Test1 ? "test1" : s == rp::Stop::Test2 ? "test2" : "none";
}

template <int W>
void ReplacementCompare::compare(const rp::internal::Geometry& g, const rp::internal::Team& team, rp::internal::Work& work,
                                 const double* q_cpu, const double* q_dev, bool device_used, double* T) {
    const auto t0 = Clock::now();
    const int m = g.m, nb = g.nb;
    const double* x = work.arena.data();
    const double* gv = work.g.data();
    const double* minv = g.minv.data();
    const double* diag = g.diag.data();
    // The recurrence of this iteration (r is still the recurrence's here).
    bool candidate[W], enter_rec[W];
    recurrence<W>(g, team, work, candidate, enter_rec);
    pending_recurrence = false;
    const bool periodic = k % std::max(1, cfg.refresh_period) == 0;
    bool act[W];
    for (int w = 0; w < W; ++w) act[w] = w < width && lane[w].active;
    // Both products' fresh sums, as run_group forms them (r is not written here).
    const auto fresh_of = [&](const double* qx, double* f) {
        rp::internal::lane_sums<W, 5>(m, team, own, f, [&](int i, double* acc) {
            const std::size_t base = static_cast<std::size_t>(i) * W;
            for (int w = 0; w < W; ++w) {
                if (!act[w]) continue;
                const double rv = gv[base + w] - qx[base + w];
                const double xv = x[base + w];
                acc[0 * W + w] += minv[i] * rv * rv;
                acc[1 * W + w] += rv * rv;
                acc[2 * W + w] += diag[i] * xv * xv;
                acc[3 * W + w] += xv * gv[base + w];
                acc[4 * W + w] += xv * rv;
            }
        });
    };
    double fc[5 * W], fd[5 * W];
    fresh_of(q_cpu, fc);
    fresh_of(q_dev, fd);
    // run_group's fresh block on a copy of the lane, per product; the formed residual
    // (product-independent: it reads x and the data) wherever either product needs it.
    ShadowLane tc[W], td[W];
    bool enter_c[W], enter_d[W], mask[W];
    bool any_mask = false;
    const auto fresh_update = [&](int w, const double* f, ShadowLane& t, bool& enter) {
        t = lane[w];
        t.beta = f[0 * W + w] / lane[w].rho_old;
        t.rho = f[0 * W + w];
        enter = enter_rec[w];
        if (!t.formed) {
            t.s2 = t.w2 - (f[3 * W + w] + f[4 * W + w]);
            if (!(t.s2 > cfg.floor_factor * rp::internal::kEps * t.w2)) enter = true;
        }
    };
    for (int w = 0; w < W; ++w) {
        mask[w] = false;
        enter_c[w] = enter_d[w] = false;
        if (!act[w]) continue;
        fresh_update(w, fc, tc[w], enter_c[w]);
        fresh_update(w, fd, td[w], enter_d[w]);
        mask[w] = lane[w].formed || enter_c[w] || enter_d[w];
        any_mask = any_mask || mask[w];
    }
    double sq[W];
    for (int w = 0; w < W; ++w) sq[w] = 0.0;
    if (any_mask) {
        const double* col[W];
        for (int w = 0; w < W; ++w) col[w] = w < width ? problem->columns[col0() + w] : nullptr;
        rp::internal::formed_residual_sq<W>(*problem, g, team, own, x, work.vbar.data(), work.corr.data(), T, col, mask, sq);
    }
    rp::Stop sc[W], sd[W];
    double tau_c[W], tau_d[W], chat_c[W], chat_d[W];
    for (int w = 0; w < W; ++w) {
        if (!act[w]) continue;
        if (lane[w].formed || enter_c[w]) {
            tc[w].formed = true;
            tc[w].s_formed = std::sqrt(std::max(0.0, sq[w]));
        }
        if (lane[w].formed || enter_d[w]) {
            td[w].formed = true;
            td[w].s_formed = std::sqrt(std::max(0.0, sq[w]));
        }
        sc[w] = evaluate(tc[w], tc[w].rho, fc[1 * W + w], fc[2 * W + w], tau_c[w], chat_c[w]);
        sd[w] = evaluate(td[w], td[w].rho, fd[1 * W + w], fd[2 * W + w], tau_d[w], chat_d[w]);
    }
    // The bound of |S x (device) - S x (CPU)| per row: 2 gamma_K B_i, B_i the magnitudes
    // of the terms of row i (tests/reduced_cuda_operator_smoke.cpp, the oracle's bound).
    if (entries.empty()) {
        entries.assign(static_cast<std::size_t>(m), 0);
        cmax = 0;
        parts = static_cast<int>(g.op_bounds.size()) - 1;
        for (int s = 0; s < g.L_op; ++s) {
            const int l = g.op_seg[static_cast<std::size_t>(s)];
            cmax = std::max(cmax, g.cell_ptr[l + 1] - g.cell_ptr[l]);
            for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c)
                for (int j = 0; j < nb; ++j)
                    ++entries[static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)] + g.cc[static_cast<std::size_t>(j)][c])];
        }
        magnitude.assign(static_cast<std::size_t>(lanes), std::vector<long double>(static_cast<std::size_t>(m), 0.0L));
    }
    team.run(width, [&](int w) {
        if (!act[w]) return;
        std::vector<long double>& B = magnitude[static_cast<std::size_t>(w)];
        std::fill(B.begin(), B.end(), 0.0L);
        const auto U = [&](int c) {
            long double u = 0.0L;
            for (int j = 0; j < nb; ++j)
                u += std::fabs(static_cast<long double>(
                    x[static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)] + g.cc[static_cast<std::size_t>(j)][c]) * W + static_cast<std::size_t>(w)]));
            return u;
        };
        for (int s = 0; s < g.L_op; ++s) {
            const int l = g.op_seg[static_cast<std::size_t>(s)];
            long double A = 0.0L;
            for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) A += (g.mult != nullptr ? g.mult[c] : 1) * U(c);
            const long double M = A * static_cast<long double>(g.seg_inv[static_cast<std::size_t>(l)]);
            for (int c = g.cell_ptr[l]; c < g.cell_ptr[l + 1]; ++c) {
                const long double t = (g.mult != nullptr ? g.mult[c] : 1) * (U(c) + M);
                for (int j = 0; j < nb; ++j)
                    B[static_cast<std::size_t>(g.offset[static_cast<std::size_t>(j)] + g.cc[static_cast<std::size_t>(j)][c])] += t;
            }
        }
    });
    const long double u = std::ldexp(1.0L, -53);
    const auto gamma = [&](long double n) { return n * u / (1.0L - n * u); };
    const long double g_sum = gamma(static_cast<long double>(m) + 2.0L);
    for (int w = 0; w < width; ++w) {
        if (!act[w]) continue;
        const std::vector<long double>& B = magnitude[static_cast<std::size_t>(w)];
        long double max_diff = 0.0L, ratio = 0.0L, limit_sq = 0.0L, x_limit = 0.0L, x_res_c = 0.0L, x_res_d = 0.0L;
        bool over = false;
        for (int i = 0; i < m; ++i) {
            const std::size_t at = static_cast<std::size_t>(i) * W + static_cast<std::size_t>(w);
            const long double e = std::fabs(static_cast<long double>(q_dev[at]) - static_cast<long double>(q_cpu[at]));
            const long double K = 2.0L * entries[static_cast<std::size_t>(i)] + cmax + nb + parts + 8;
            const long double limit = 2.0L * gamma(K) * B[static_cast<std::size_t>(i)];
            max_diff = std::max(max_diff, e);
            if (limit > 0.0L) ratio = std::max(ratio, e / limit);
            else if (e > 0.0L) over = true;
            const long double ax = std::fabs(static_cast<long double>(x[at]));
            limit_sq += limit * limit;
            x_limit += ax * limit;
            x_res_c += ax * std::fabs(static_cast<long double>(gv[at]) - q_cpu[at]);
            x_res_d += ax * std::fabs(static_cast<long double>(gv[at]) - q_dev[at]);
        }
        if (over || ratio > 1.0L || (strict && max_diff > 0.0L)) divergent = true;
        const ShadowLane& ln = lane[w];
        // chat = sqrt(sum (g - q)^2) / denom: |chat_dev - chat_cpu| <= ||e||_2 / denom plus
        // the rounding of each (residual, square, sum: gamma_(m+2); sqrt and division: 2u).
        const long double chat_bound = (std::sqrt(limit_sq) / ln.denom +
                                        (g_sum / 2.0L + 2.0L * u) * (static_cast<long double>(chat_c[w]) + chat_d[w])) * (1.0L + 8.0L * u);
        // s2 = w2 - (x'g + x'(g - q)): |s2_dev - s2_cpu| <= sum |x| |e| plus the rounding of
        // both x'(g - q) (gamma_(m+2) of their magnitudes) and of the two final operations.
        const long double s2c = ln.w2 - (fc[3 * W + w] + fc[4 * W + w]);
        const long double s2d = ln.w2 - (fd[3 * W + w] + fd[4 * W + w]);
        const long double s2_bound =
            (x_limit + g_sum * (x_res_c + x_res_d) +
             u * (std::fabs(static_cast<long double>(fc[3 * W + w]) + fc[4 * W + w]) + std::fabs(static_cast<long double>(fd[3 * W + w]) + fd[4 * W + w])) +
             u * (std::fabs(s2c) + std::fabs(s2d))) * (1.0L + 8.0L * u);
        const double floor = cfg.floor_factor * rp::internal::kEps * ln.w2;
        const double chat_used = device_used ? chat_d[w] : chat_c[w];
        const bool formed_c = tc[w].formed, formed_d = td[w].formed;
        const bool differs = sc[w] != sd[w] || formed_c != formed_d;
        const bool near_chat = cfg.chat_stop > 0.0 && std::fabs(static_cast<long double>(chat_used) - cfg.chat_stop) <= chat_bound;
        const bool near_floor = !ln.formed && std::fabs((device_used ? s2d : s2c) - static_cast<long double>(floor)) <= s2_bound;
        // Layer-2 margins of the used product, relative: test 1 (s against its right-hand side) and the s2 floor
        // (the distance to it over w2; 1 when the lane was formed before, as s2 is not read then).
        const ShadowLane& lu = device_used ? td[w] : tc[w];
        const double* fu = device_used ? fd : fc;
        const double s_used = lu.formed ? lu.s_formed : (lu.s2 > 0.0 ? std::sqrt(lu.s2) : 0.0);
        const double rhs1 = cfg.btol * lu.wnorm + cfg.atol * std::sqrt(std::max(0.0, lu.a2)) * std::sqrt(std::max(0.0, fu[2 * W + w]));
        const double test1_margin = rhs1 > 0.0 ? (rhs1 - s_used) / rhs1 : 1.0;
        const double s2_margin = ln.formed ? 1.0 : ln.w2 > 0.0 ? static_cast<double>(((device_used ? s2d : s2c) - floor) / ln.w2) : 1.0;
        ++lines;
        differ += differs ? 1 : 0;
        within += (near_chat || near_floor) ? 1 : 0;
        std::string why;
        const auto add = [&](const char* s) { why += (why.empty() ? "" : "+") + std::string(s); };
        if (candidate[w]) add("candidate");
        if (enter_rec[w]) add("formed_entry");
        if (periodic) add("periodic");
        if (why.empty()) add("other_lane");
        std::ostringstream o;
        o.precision(17);
        o << "xhdfe reduced_cuda event=replacement_compare group=" << group << " col=" << col0() + w << " lane=" << w
          << " k=" << k << " why=" << why << " used=" << (device_used ? "device" : "cpu")
          << " max_abs_diff=" << static_cast<double>(max_diff) << " bound_ratio=" << (over ? -1.0 : static_cast<double>(ratio))
          << " chat_cpu=" << chat_c[w] << " chat_dev=" << chat_d[w]
          << " chat_diff=" << std::fabs(chat_d[w] - chat_c[w]) << " chat_bound=" << static_cast<double>(chat_bound)
          << " chat_stop=" << cfg.chat_stop << " formed_before=" << (ln.formed ? 1 : 0)
          << " s2_cpu=" << static_cast<double>(s2c) << " s2_dev=" << static_cast<double>(s2d)
          << " s2_diff=" << static_cast<double>(std::fabs(s2d - s2c)) << " s2_bound=" << static_cast<double>(s2_bound)
          << " s2_floor=" << floor << " stop_cpu=" << stop_label(sc[w]) << " stop_dev=" << stop_label(sd[w])
          << " formed_cpu=" << (formed_c ? 1 : 0) << " formed_dev=" << (formed_d ? 1 : 0)
          << " decision_differs=" << (differs ? 1 : 0) << " threshold_within_bound=" << ((near_chat || near_floor) ? 1 : 0);
        o.precision(3);
        o << " chat_margin=" << (cfg.chat_stop > 0.0 ? (chat_used - cfg.chat_stop) / cfg.chat_stop : 0.0)
          << " test1_margin=" << test1_margin << " s2_margin=" << s2_margin;
        pending.push_back(Pending{w, col0() + w, k, o.str(), test1_margin, s2_margin});
    }
    // The shadow follows the used product; its (tau, chat) are the iteration's expectation.
    for (int w = 0; w < W; ++w) {
        if (!act[w]) continue;
        lane[w] = device_used ? td[w] : tc[w];
        lane[w].tau = device_used ? tau_d[w] : tau_c[w];
        lane[w].chat = device_used ? chat_d[w] : chat_c[w];
        lane[w].expect = true;
    }
    shadow_seconds += std::chrono::duration<double>(Clock::now() - t0).count();
}

struct GpuIterativeOperator {
    using Clock = std::chrono::steady_clock;
    std::unique_ptr<reduced_cuda::Operator> op;
    reduced_cuda::Statistics saved;
    reduced_cuda::Admission admission;  // as computed, also when the set-up declined
    std::size_t adapter_bytes=0,host_peak_bound=0,host_budget=0,device_budget=0;
    std::size_t host_geometry_bytes=0,host_layout_peak=0,host_cpu_peak=0;
    TierClocks* clocks=nullptr;  // profiling (the trace) when set
    reduced_cuda::TestFailure test_failure=reduced_cuda::TestFailure::None;
    // Package 3-lite: the replacement on the device (in force: the switch and D10, set
    // in prepare); the qualification mode (compare: set by the attempt, which then
    // connects clocks and its ReplacementCompare); a fatal error injected at a
    // replacement (tests). Three flags in the padding after test_failure: the adapter's
    // size, which the host admission counts, is package 2's. The counts, the timers and
    // the qualification's state live in the attempt's TierClocks.
    bool replacement_in_force=false,compare=false,fail_at_replacement=false;
    static double since_seconds(Clock::time_point t0) {return std::chrono::duration<double>(Clock::now()-t0).count();}
    void prepare(const rp::internal::Geometry& g,const rp::Config& cfg,int lanes,std::size_t cpu_bytes) {
        host_budget=cfg.memory_budget;device_budget=cfg.memory_budget;
        const auto state_bytes=reduced_cuda::host_sum({sizeof(*this),reduced_cuda::Operator::host_state_bytes()});
        host_cpu_peak=reduced_cuda::host_sum({cpu_bytes,adapter_bytes,state_bytes});
        if(host_cpu_peak>host_budget)
            throw reduced_cuda::Error(reduced_cuda::ErrorKind::Memory,"CUDA reduced tier: host reserved budget");
        if(g.C_op==0)return;
        host_geometry_bytes=reduced_cuda::geometry_resident_bytes(g);
        if(clocks)clocks->view_from=TierClocks::Clock::now();
        const reduced_cuda::GeometryView view=reduced_cuda::view_of(g);  // borrows the geometry; nothing copied
        if(clocks)clocks->view_to=TierClocks::Clock::now();
        reduced_cuda::Options o;
        o.lanes=lanes;o.budget=device_budget;o.profile=clocks!=nullptr;o.test_failure=test_failure;
        // 3-lite only under D10 (config.chat_stop > 0); with it, or with the qualification
        // mode, the one registration covers x, r, p and q (four slices instead of two).
        replacement_in_force=replacement_switch()&&cfg.chat_stop>0.0;
        o.registered_slices=(replacement_in_force||compare)?4:2;
        if(compare)clocks->qualification->prepare(cfg,lanes);
        // The set-up's host structures are admitted with the geometry resident.
        o.host_budget=host_budget;
        o.host_reserved=reduced_cuda::host_sum({host_geometry_bytes,adapter_bytes,state_bytes});
        // The copies into the upload chunks run on the fit's team (its observer counts them).
        const rp::internal::Team team(cfg.threads,cfg.hooks,8*std::max(1,cfg.blocks));
        o.run=[&team](int parts,const std::function<void(int)>& body){team.run(parts,body);};
        if(clocks)clocks->operator_from=TierClocks::Clock::now();
        op=std::make_unique<reduced_cuda::Operator>(view,o,&admission);
        if(clocks)clocks->operator_to=TierClocks::Clock::now();
        host_layout_peak=admission.host_setup_real;
        host_peak_bound=reduced_cuda::host_phase_peak(cpu_bytes,host_geometry_bytes,host_layout_peak,adapter_bytes,state_bytes);
        if(host_peak_bound>host_budget)
            throw reduced_cuda::Error(reduced_cuda::ErrorKind::Memory,"CUDA reduced tier: host set-up budget");
    }
    // solve(): the real width before each group, and ONE registration of p and q (of x,
    // r, p and q when the device computes replacement products too).
    bool replacement_on_device()const noexcept{return op&&(replacement_in_force||compare);}
    void begin_group(int width){
        if(op)op->begin_group(width);
        if(compare&&op)clocks->qualification->begin_group(width);
    }
    void pin_host(double* begin,std::size_t count){
        if(!op)return;
        op->register_host(begin,count);
        if(clocks)clocks->registered_doubles=count;
    }
    void unpin_host() noexcept{if(op)op->unregister_host();}
    template<int W> void apply(const rp::internal::Geometry& g,const rp::internal::Team& team,
        const double* in,double* out,double*,rp::internal::Work& work) {
        if(op){
            if(compare)clocks->qualification->before_iterative<W>(g,team,in,work);
            const auto t0=clocks?Clock::now():Clock::time_point{};
            op->apply(in,out);
            if(clocks){clocks->products.iterative_seconds+=since_seconds(t0);++clocks->products.iterative_device;}
            if(compare)clocks->qualification->after_iterative<W>(g,team,in,out);
        }else{
            std::fill(out,out+static_cast<std::size_t>(g.m)*W,0.0);
            if(clocks)++clocks->products.iterative_zero;
        }
    }
    // run_group's residual replacement q = S x: the CPU product of package 2, or (3-lite
    // in force) the device's; the qualification mode computes both on the same x.
    template<int W> void apply_replacement(const rp::internal::Geometry& g,const rp::internal::Team& team,
        const double* x,double* q,double* T,rp::internal::Work& work) {
        const bool device=op&&replacement_in_force;
        if((device||(compare&&op))&&fail_at_replacement&&replacement_fault_now())
            throw reduced_cuda::Error(reduced_cuda::ErrorKind::Fatal,"CUDA reduced tier: injected fatal error at a replacement (test only)");
        if(compare&&op){
            ReplacementCompare& c=*clocks->qualification;
            double* kept=c.other_rows();
            if(device){
                device_replacement(x,q);
                const auto t0=Clock::now();
                rp::internal::apply_operator<W>(g,team,x,kept,T,work);  // T, priv, partials: free between products
                c.cpu_seconds+=since_seconds(t0);
                c.compare<W>(g,team,work,kept,q,true,T);
            }else{
                // The device's product first into q, kept; then the CPU's, the one used, into q
                // (apply_operator writes every row: q holds exactly package 2's product).
                const auto t0=Clock::now();
                op->apply(x,q);
                c.device_seconds+=since_seconds(t0);
                std::copy(q,q+static_cast<std::size_t>(g.m)*W,kept);
                cpu_replacement<W>(g,team,x,q,T,work);
                c.compare<W>(g,team,work,q,kept,false,T);
            }
            return;
        }
        if(device)device_replacement(x,q);
        else cpu_replacement<W>(g,team,x,q,T,work);
    }
    void device_replacement(const double* x,double* q){
        const reduced_cuda::Statistics before=clocks?op->statistics():reduced_cuda::Statistics{};
        const auto t0=clocks?Clock::now():Clock::time_point{};
        op->apply(x,q);
        if(clocks){
            const reduced_cuda::Statistics& after=op->statistics();
            ProductCounts& c=clocks->products;
            c.replacement_device_seconds+=since_seconds(t0);
            c.replacement_h2d_ms+=after.h2d_ms-before.h2d_ms;
            c.replacement_kernel_ms+=after.kernel_ms-before.kernel_ms;
            c.replacement_d2h_ms+=after.d2h_ms-before.d2h_ms;
            ++c.replacement_device;
        }
    }
    template<int W> void cpu_replacement(const rp::internal::Geometry& g,const rp::internal::Team& team,
        const double* x,double* q,double* T,rp::internal::Work& work){
        const auto t0=clocks?Clock::now():Clock::time_point{};
        rp::internal::apply_operator<W>(g,team,x,q,T,work);
        if(clocks){clocks->products.replacement_cpu_seconds+=since_seconds(t0);++clocks->products.replacement_cpu;}
    }
    void release(){
        if(compare&&clocks&&clocks->qualification)clocks->qualification->flush();
        if(op){saved=op->statistics();op.reset();}
    }
    const reduced_cuda::Statistics& statistics()const noexcept{return op?op->statistics():saved;}
};

std::optional<AbsorptionResult> attempt_reduced_pcg_cuda(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const HdfeOptions& options,
    double accept_trigger,
    ReducedPcgOutcome* outcome,
    int& declined_iterations) {
    const auto start = std::chrono::steady_clock::now();
    const int trace = reduced_pcg_trace_level();
    const int n = static_cast<int>(y.size());
    const int p = static_cast<int>(X.cols());
    const int cols = 1 + p;
    const int k = static_cast<int>(fes.size());
    const int stage = forced_decline_stage();

    // The attempt reports to a private observer, published on acceptance only.
    ParallelWorkObserver observer;
    if (stage == 5) throw std::bad_alloc();
    rp::Hooks hooks;
    hooks.context = &observer;
    hooks.begin = observer_begin;
    hooks.observe = observer_observe;
    hooks.end = observer_end;
    const int threads = std::max(1, options.num_threads);
    const int blocks = std::max(1, deterministic_parallel_chunk_count(std::max(1, n)));
    const rp::internal::Team team(threads, hooks, 8 * blocks);

    std::ostringstream lines;
    lines.precision(10);
    rp::Report report;
    TierClocks tier_clocks;  // trace only (and the qualification mode of package 3-lite)
    const auto stamp = [&](TierClocks::Clock::time_point& at) {
        if (trace > 0) at = TierClocks::Clock::now();
    };
    // Package 3-lite's qualification mode (tests only): its state outlives the operator.
    const int compare_mode = replacement_compare_requested();
    const bool compare = compare_mode > 0;
    std::unique_ptr<ReplacementCompare> qualification;
    if (compare) {
        qualification = std::make_unique<ReplacementCompare>();
        qualification->strict = compare_mode == 2;
        tier_clocks.qualification = qualification.get();
    }
    GpuIterativeOperator gpu_operator;
    gpu_operator.clocks = trace > 0 || compare ? &tier_clocks : nullptr;
    gpu_operator.test_failure = injected_failure();
    gpu_operator.compare = compare;
    gpu_operator.fail_at_replacement = injected_replacement_failure();
    gpu_operator.adapter_bytes = static_cast<std::size_t>(n) * (4U * k + 8U * cols) +
        static_cast<std::size_t>(k) * (sizeof(CompactCodes) + sizeof(int) + sizeof(void*)) +
        static_cast<std::size_t>(cols) * 2U * sizeof(void*);
    std::size_t budget = 0;  // the admission budget, for the plan line
    // D10's target for this attempt (0: the rule before D10), one decade below
    // the trigger, when the switch is on and the trigger is the threshold in force.
    const double chat_stop = reduced_pcg_chat_stop_applies(options, accept_trigger) ? 0.1 * accept_trigger : 0.0;
    double compact_seconds = 0.0, certificate_seconds = 0.0;
    int eliminated = -1;
    std::vector<int> levels(static_cast<std::size_t>(k), 0);
    auto phases = [&]() {
        if (trace <= 0) return;
        const std::pair<const char*, double> phase[] = {
            {"codes", compact_seconds},           {"geometry", report.geometry_seconds},
            {"rhs", report.rhs_seconds},          {"iterate", report.iterate_seconds},
            {"recover", report.recover_seconds},  {"residuals", report.residual_seconds},
            {"certificate", certificate_seconds}};
        for (const auto& entry : phase)
            lines << "xhdfe reduced_pcg event=phase name=" << entry.first << " seconds=" << entry.second << '\n';
    };
    auto plan = [&]() {
        if (trace <= 0) return;
        lines << "xhdfe reduced_pcg event=plan n=" << n << " nfe=" << k << " cols=" << cols
              << " eliminated=" << eliminated << " levels=";
        for (int b = 0; b < k; ++b) lines << (b ? "," : "") << levels[static_cast<std::size_t>(b)];
        lines << " cells=" << report.cells << " segments=" << report.segments
              << " operator_segments=" << report.operator_segments << " sorted=" << (report.sorted ? 1 : 0)
              << " merged=" << (report.merged ? 1 : 0) << " large=" << report.large_effects
              << " groups=" << report.groups << " lanes=" << report.lanes << " bytes=" << report.bytes
              << " budget=" << budget
              << " team=" << threads << " blocks=" << blocks << " companion=" << (companion_enabled() ? 1 : 0);
        if (chat_stop > 0.0) lines << " chat_stop=" << chat_stop;
        lines << '\n';
    };
    auto column_lines = [&](const std::vector<double>* rel) {
        if (trace <= 0) return;
        for (int c = 0; c < static_cast<int>(report.columns.size()); ++c) {
            const rp::ColumnReport& col = report.columns[static_cast<std::size_t>(c)];
            lines << "xhdfe reduced_pcg event=column col=" << c << " iterations=" << col.iterations
                  << " first_tau_pass=" << col.first_tau_pass << " reason=" << rp::stop_name(col.stop)
                  << " tau=" << col.tau << " chat=" << col.chat << " s_rel=" << col.s_rel
                  << " formed_checks=" << col.formed_checks;
            if (rel != nullptr) lines << " rel=" << (*rel)[static_cast<std::size_t>(c)];
            lines << '\n';
        }
    };
    auto finish = [&](const std::string& fields) {
        if (trace > 0) {
            std::cerr << lines.str();
            lines.str("");
        }
        if (trace > 0) {
            const auto& s = gpu_operator.statistics();
            std::cerr << "xhdfe reduced_pcg_cuda event=operator calls=" << s.calls
                << " host_peak_bound=" << gpu_operator.host_peak_bound
                << " host_cpu_peak=" << gpu_operator.host_cpu_peak
                << " host_geometry_bytes=" << gpu_operator.host_geometry_bytes
                << " host_layout_peak=" << gpu_operator.host_layout_peak
                << " host_budget=" << gpu_operator.host_budget
                << " device_payload=" << s.device_payload_bytes
                << " device_budget=" << gpu_operator.device_budget
                << " h2d_bytes=" << s.h2d_bytes << " d2h_bytes=" << s.d2h_bytes
                << " timed=" << s.timed << " setup_seconds=" << s.setup_seconds
                << " apply_seconds=" << s.apply_seconds << " h2d_ms=" << s.h2d_ms
                << " kernel_ms=" << s.kernel_ms << " d2h_ms=" << s.d2h_ms << '\n';
            // The admission (trace only; its prefix is outside "xhdfe reduced_pcg", whose
            // lines the contracts compare): every device buffer counted before the first
            // allocation, by phase, next to what the operator then asked of the device
            // and the device's free memory at entry, at the set-up's peak and after it.
            const reduced_cuda::Admission& ad = gpu_operator.admission;
            using A = reduced_cuda::Admission;
            const auto signed_difference = [](std::size_t a, std::size_t b) {
                return a >= b ? static_cast<long long>(a - b) : -static_cast<long long>(b - a);
            };
            std::ostringstream m;
            m << "xhdfe reduced_cuda event=admission admitted=" << (ad.admitted ? 1 : 0) << " classes=" << s.classes
              << " private_levels=" << s.private_levels << " long_segments=" << s.long_segments
              << " long_levels=" << s.long_levels << " units=" << s.units << " pb_slots=" << s.pb_slots
              << " k1_blocks=" << s.k1_blocks << " pa_slots=" << s.pa_slots << " pl_items=" << s.pl_items
              << " b_radix=" << s.b_radix << " uploaded_bytes=" << s.uploaded_bytes
              << " device_resident_bytes=" << ad.device[A::Resident] << " device_product_bytes=" << ad.device[A::Product]
              << " device_build_bytes=" << ad.device[A::Build] << " device_scratch_bytes=" << ad.device[A::Scratch]
              << " device_units_bytes=" << ad.device[A::Units] << " device_setup_peak_bytes=" << ad.device_setup_peak
              << " device_iteration_peak_bytes=" << ad.device_iteration_peak << " device_budget_bytes=" << ad.budget
              << " cub_scan_bytes=" << ad.cub_scan_bytes << " cub_radix_bytes=" << ad.cub_radix_bytes
              << " device_allocated_peak_bytes=" << ad.allocated_peak
              << " device_allocated_iteration_bytes=" << ad.allocated_iteration
              << " device_free_entry_bytes=" << ad.free_at_entry << " device_free_peak_bytes=" << ad.free_at_peak
              << " device_free_after_setup_bytes=" << ad.free_after_setup;
            if (ad.admitted && ad.free_after_setup > 0)
                m << " device_taken_peak_bytes=" << signed_difference(ad.free_at_entry, ad.free_at_peak)
                  << " device_taken_iteration_bytes=" << signed_difference(ad.free_at_entry, ad.free_after_setup);
            m << " host_setup_bytes=" << ad.host_setup << " host_setup_real_bytes=" << ad.host_setup_real
              << " page_locked_setup_bytes=" << ad.page_locked_setup
              << " page_locked_iteration_bytes=" << ad.page_locked_iteration << " buffers=";
            for (int e = 0; e < ad.count; ++e) {
                const A::Entry& x = ad.entries[e];
                static const char* const phase[] = {"resident", "product", "build", "scratch", "units", "host"};
                m << (e ? "," : "") << phase[x.phase] << ':' << x.name;
                if (x.effect >= 0) m << '[' << x.effect << ']';
                m << ':' << x.bytes;
            }
            std::cerr << m.str() << '\n';
        }
        if (trace > 0) {
            // Attribution (trace only): consecutive spans that tile the attempt
            // up to this line; a span with a stamp never taken prints 0 and
            // untiled_seconds shows what no span covers. Its prefix is outside
            // "xhdfe reduced_pcg", whose lines the contracts compare; every timing
            // key ends in _seconds or _ms.
            using Clock = TierClocks::Clock;
            const Clock::time_point at = Clock::now();
            const auto span = [](Clock::time_point a, Clock::time_point b) {
                return a == Clock::time_point{} || b == Clock::time_point{}
                           ? 0.0 : std::chrono::duration<double>(b - a).count();
            };
            const TierClocks& c = tier_clocks;
            const auto& sv = c.solve;
            Clock::time_point last = start;  // the last stamp taken, in the order of the attempt
            for (const Clock::time_point t : {c.codes_from, c.codes_to, c.solve_from, sv.prepare_from,
                                              c.view_from, c.view_to, c.operator_from, c.operator_to,
                                              sv.prepare_to, sv.work_allocated,
                                              sv.teardown_from, c.solve_to, c.released,
                                              c.certificate_from, c.certificate_to})
                if (t != Clock::time_point{}) last = t;
            const double total = std::chrono::duration<double>(at - start).count();
            const double phased = compact_seconds + report.geometry_seconds + report.rhs_seconds +
                                  report.iterate_seconds + report.recover_seconds + report.residual_seconds +
                                  certificate_seconds;
            const std::pair<const char*, double> tiles[] = {
                {"prologue", span(start, c.codes_from)},
                {"codes_span", span(c.codes_from, c.codes_to)},
                {"pre_solve", span(c.codes_to, c.solve_from)},
                {"geometry_span", span(c.solve_from, sv.prepare_from)},
                {"prepare", span(sv.prepare_from, sv.prepare_to)},
                {"work_alloc", span(sv.prepare_to, sv.work_allocated)},
                {"groups_span", span(sv.work_allocated, sv.teardown_from)},
                {"teardown", span(sv.teardown_from, c.solve_to)},
                {"release", span(c.solve_to, c.released)},
                {"pre_certificate", span(c.released, c.certificate_from)},
                {"certificate_span", span(c.certificate_from, c.certificate_to)},
                {"epilogue", span(last, at)}};
            double tiled = 0.0;
            std::ostringstream a;
            a.setf(std::ios::fixed);
            a.precision(6);
            a << "xhdfe reduced_cuda event=attribution total_seconds=" << total
              << " phased_seconds=" << phased << " not_phased_seconds=" << total - phased;
            for (const auto& tile : tiles) {
                a << ' ' << tile.first << "_seconds=" << tile.second;
                tiled += tile.second;
            }
            a << " untiled_seconds=" << total - tiled;
            // Inside prepare: the view of the geometry, the operator's set-up and its
            // phases (setup_seconds of the operator line); then the registration of p
            // and q (inside work_alloc) and its release (inside teardown); per product,
            // the transfers, the pack and unpack and the four kernels (CUDA events).
            const auto& os = gpu_operator.statistics();
            a << " view_seconds=" << span(c.view_from, c.view_to)
              << " operator_seconds=" << span(c.operator_from, c.operator_to)
              << " setup_device_seconds=" << os.device_seconds << " setup_plan_seconds=" << os.plan_seconds
              << " setup_staging_seconds=" << os.staging_seconds << " setup_upload_seconds=" << os.upload_seconds
              << " setup_alloc_seconds=" << os.alloc_seconds << " setup_build_seconds=" << os.build_seconds
              << " setup_units_seconds=" << os.units_seconds << " register_seconds=" << os.register_seconds
              << " unregister_seconds=" << os.unregister_seconds;
            a.precision(3);
            a << " h2d_dma_ms=" << os.h2d_dma_ms << " pack_ms=" << os.pack_ms << " k1_ms=" << os.k1_ms
              << " k1l_ms=" << os.k1l_ms << " k2_ms=" << os.k2_ms << " k3_ms=" << os.k3_ms
              << " unpack_ms=" << os.unpack_ms << " d2h_dma_ms=" << os.d2h_dma_ms << '\n';
            std::cerr << a.str();
            // Package 3-lite (trace only; a line of its own, so that every line above is
            // package 2's): the products by kind, so that the CPU replacements are measured
            // with the switch off. The operator line's calls and transfer bytes count every
            // device product, iterative and replacement alike.
            const ProductCounts& pc = c.products;
            std::ostringstream rl;
            rl.setf(std::ios::fixed);
            rl.precision(6);
            rl << "xhdfe reduced_cuda event=replacement in_force=" << (gpu_operator.replacement_in_force ? 1 : 0)
               << " switch=" << (replacement_switch() ? 1 : 0) << " d10=" << (chat_stop > 0.0 ? 1 : 0)
               << " compare=" << (compare ? 1 : 0) << " registered_doubles=" << c.registered_doubles
               << " iterative_device=" << pc.iterative_device << " iterative_zero=" << pc.iterative_zero
               << " iterative_seconds=" << pc.iterative_seconds
               << " replacement_device=" << pc.replacement_device << " replacement_device_seconds=" << pc.replacement_device_seconds
               << " replacement_cpu=" << pc.replacement_cpu << " replacement_cpu_seconds=" << pc.replacement_cpu_seconds;
            rl.precision(3);
            rl << " replacement_h2d_ms=" << pc.replacement_h2d_ms << " replacement_kernel_ms=" << pc.replacement_kernel_ms
               << " replacement_d2h_ms=" << pc.replacement_d2h_ms << '\n';
            std::cerr << rl.str();
        }
        if (compare && tier_clocks.qualification != nullptr) {
            // The qualification mode's summary (with or without the trace).
            const ReplacementCompare& q = *tier_clocks.qualification;
            std::ostringstream ql;
            ql.setf(std::ios::fixed);
            ql.precision(6);
            ql << "xhdfe reduced_cuda event=replacement_compare_summary lines=" << q.lines
               << " divergent=" << (q.divergent ? 1 : 0) << " decision_differs=" << q.differ
               << " threshold_within_bound=" << q.within << " chat_verified=" << q.chat_verified
               << " layer2_exact=" << q.layer2_exact << " layer2_margin=" << q.layer2_margin
               << " layer2_unverified=" << q.layer2_unverified << " shadow_checks=" << q.checked
               << " chat_mismatches=" << q.chat_mismatch << " tau_mismatches=" << q.tau_mismatch;
            ql.setf(std::ios::scientific, std::ios::floatfield);
            ql.precision(2);
            ql << " chat_rel_max=" << q.chat_rel_max << " tau_rel_max=" << q.tau_rel_max;
            ql.setf(std::ios::fixed, std::ios::floatfield);
            ql.precision(6);
            ql
               << " compare_cpu_seconds=" << q.cpu_seconds << " compare_device_seconds=" << q.device_seconds
               << " shadow_seconds=" << q.shadow_seconds << '\n';
            std::cerr << ql.str();
        }
        reduced_pcg_final_line(fields + " backend=cuda seconds=" + std::to_string(since(start)));
    };
    // The outcome is recorded first, without allocating; the trace after it.
    auto decline = [&](const char* reason, double rel) -> std::optional<AbsorptionResult> {
        declined_iterations = report.iterations;
        if (outcome != nullptr) {
            outcome->status = 2;
            outcome->reason = reason;
            outcome->iterations = report.iterations;
        }
        plan();
        phases();
        if (report.decline != rp::Decline::None || report.solved) column_lines(nullptr);
        std::ostringstream fields;
        fields.precision(10);
        fields << "status=declined reason=" << reason << " column=" << report.decline_column
               << " iterations=" << report.iterations << " rel=" << rel;
        finish(fields.str());
        return std::nullopt;
    };

    try {
        // Safe test injection: never launches a faulting device kernel.
        const char* fatal_test=std::getenv("XHDFE_TEST_REDUCED_PCG_CUDA_FATAL");
        static thread_local std::uint64_t fatal_attempts=0;
        if(fatal_test && fatal_test[0] && fatal_test[1]=='\0') {
            if(fatal_attempts<std::numeric_limits<std::uint64_t>::max())++fatal_attempts;
            if(fatal_test[0]=='1' || (fatal_test[0]=='2' && fatal_attempts==2))
                throw reduced_cuda::Error(reduced_cuda::ErrorKind::Fatal,"CUDA reduced tier: injected fatal error (test only)");
        }
        stamp(tier_clocks.codes_from);
        auto t0 = std::chrono::steady_clock::now();
        std::vector<CompactCodes> compact(static_cast<std::size_t>(k));
        std::vector<const int*> code_ptrs(static_cast<std::size_t>(k));
        for (int b = 0; b < k; ++b) {
            compact[static_cast<std::size_t>(b)] = compact_codes(fes[static_cast<std::size_t>(b)], team, blocks);
            levels[static_cast<std::size_t>(b)] = compact[static_cast<std::size_t>(b)].levels;
            code_ptrs[static_cast<std::size_t>(b)] = compact[static_cast<std::size_t>(b)].codes.data();
        }
        compact_seconds = since(t0);
        stamp(tier_clocks.codes_to);
        eliminated = 0;
        for (int b = 1; b < k; ++b)
            if (levels[static_cast<std::size_t>(b)] > levels[static_cast<std::size_t>(eliminated)]) eliminated = b;

        AbsorptionResult result;
        result.y_tilde.resize(n);
        result.X_tilde.resize(n, p);
        std::vector<const double*> columns(static_cast<std::size_t>(cols));
        std::vector<double*> residuals(static_cast<std::size_t>(cols));
        columns[0] = y.data();
        residuals[0] = result.y_tilde.data();
        for (int j = 0; j < p; ++j) {
            columns[static_cast<std::size_t>(j) + 1U] = X.col(j).data();
            residuals[static_cast<std::size_t>(j) + 1U] = result.X_tilde.col(j).data();
        }

        rp::Problem problem;
        problem.n = n;
        problem.k = k;
        problem.codes = code_ptrs.data();
        problem.levels = levels.data();
        problem.eliminated = eliminated;
        problem.cols = cols;
        problem.columns = columns.data();
        problem.residuals = residuals.data();

        HdfeOptions mlsmr_view = options;
        mlsmr_view.absorption_method = AbsorptionMethod::Mlsmr;
        rp::Config config;
        config.threads = threads;
        config.blocks = blocks;
        config.atol = mlsmr_parity_tolerance(mlsmr_view);
        config.btol = config.atol;
        config.trigger = accept_trigger;
        config.companion = companion_enabled();
        config.chat_stop = chat_stop;
        config.max_iter = std::max(1, std::min(1500, options.max_iter));
        // The columns and the identifiers of the fit (rp::admission_budget).
        config.memory_budget = rp::admission_budget(n, cols, k);
        budget = config.memory_budget;
        config.hooks = hooks;
        config.force_decline_geometry = stage == 1;
        config.force_decline_iterate = stage == 2;
        recurrence_chat_hook(config);
        if (qualification) {
            // The qualification mode reads the controller's tau and chat at every iteration.
            qualification->problem = &problem;
            config.iteration_context = qualification.get();
            config.on_iteration = &ReplacementCompare::hook_entry;
        }

        stamp(tier_clocks.solve_from);
        report = hdfe::reduced_cuda_controller::solve(problem, config, gpu_operator,
                                                      trace > 0 ? &tier_clocks.solve : nullptr);
        stamp(tier_clocks.solve_to);
        gpu_operator.release();
        stamp(tier_clocks.released);
        if (!report.solved) return decline(rp::decline_name(report.decline), -1.0);
        if (gpu_operator.statistics().calls == 0) return decline("no_gpu_work", -1.0);
        if (qualification && qualification->divergent) return decline("replacement_divergence", -1.0);
        if (stage == 3) return decline("forced", -1.0);

        result.iterations = report.iterations;
        result.converged = true;
        result.fe_levels = levels;  // distinct ids per effect, in the input order (feeds the DoF)
        HdfeOptions certificate_options = options;
        certificate_options.parallel_observer = &observer;
        stamp(tier_clocks.certificate_from);
        t0 = std::chrono::steady_clock::now();
        if (shared_fe_coding_enabled()) {
            // The geometry and the solver's workspace are gone: the certificate may
            // take the codes over (code_ptrs is not read again). Whatever it takes
            // dies with its indexers, inside this try.
            std::vector<TierFeCodes> tier_codes(static_cast<std::size_t>(k));
            for (int b = 0; b < k; ++b) {
                CompactCodes& c = compact[static_cast<std::size_t>(b)];
                tier_codes[static_cast<std::size_t>(b)] = TierFeCodes{&c.codes, c.levels, c.min_id, c.max_id};
            }
            certify_tier_absorption(y, X, fes, certificate_options, result, tier_codes);
        } else {
            certify_absorption_result(y, X, fes, nullptr, certificate_options, {}, result);
        }
        certificate_seconds = since(t0);
        stamp(tier_clocks.certificate_to);
        const double rel = result.abs_residual_rel;
        if (stage == 4) return decline("forced", rel);
        const bool certified = result.converged && result.precision_certified &&
                               hdfe::detail::ieee_finite(rel);
        if (!certified) return decline("certificate", rel);
        if (!(rel <= accept_trigger)) return decline("trigger", rel);

        // The attempt's team as it stands at the decision, before the
        // trace-only work below; published last, to the fit through
        // reduced_pcg_fold_observation.
        const int decided_team = observer.max_team_size();
        const int decided_workers = observer.active_workers();
        std::vector<double> per_column;
        if (trace >= 2) {
            // The unchanged certificate on each column alone, after the decision.
            per_column.resize(static_cast<std::size_t>(cols));
            const Eigen::MatrixXd empty(n, 0);
            for (int c = 0; c < cols; ++c) {
                AbsorptionResult single;
                single.y_tilde = c == 0 ? result.y_tilde : Eigen::VectorXd(result.X_tilde.col(c - 1));
                single.X_tilde = Eigen::MatrixXd(n, 0);
                single.converged = true;
                const Eigen::VectorXd original = c == 0 ? Eigen::VectorXd(y) : Eigen::VectorXd(X.col(c - 1));
                certify_absorption_result(original, empty, fes, nullptr, certificate_options, {}, single);
                per_column[static_cast<std::size_t>(c)] = single.abs_residual_rel;
            }
        }
        plan();
        phases();
        column_lines(trace >= 2 ? &per_column : nullptr);
        std::ostringstream fields;
        fields.precision(10);
        fields << "status=accepted reason=none iterations=" << report.iterations << " rel=" << rel;
        finish(fields.str());
        // Nothing below can throw: the publication and the accepted outcome
        // come last, so that a failure above is a decline with nothing published.
        reduced_pcg_record_cuda_observation(decided_team, decided_workers);
        result.gpu_used = true;
        result.gpu_attempted = true;
        result.gpu_status_code = 1;
        result.gpu_absorption_converged = true;
        result.gpu_absorption_iterations = report.iterations;
        if (outcome != nullptr) {
            outcome->status = 1;
            outcome->reason = "none";
            outcome->iterations = report.iterations;
        }
        return result;
    } catch (const reduced_cuda::Error& error) {
        gpu_operator.release();
        if (error.kind() == reduced_cuda::ErrorKind::Fatal) throw;
        return decline(decline_reason(error.kind()), -1.0);
    } catch (const std::bad_alloc&) {
        return decline("memory", -1.0);
    } catch (const std::exception&) {
        return decline("exception", -1.0);
    }
}


} // namespace
// The process switch: XHDFE_REDUCED_PCG_CUDA exactly 1 (on) or exactly 0 (off),
// otherwise the built-in value kReducedPcgCudaBuiltIn (package D). Read once.
bool reduced_pcg_cuda_enabled() noexcept {
    static const bool on=[] {
        const char* e=std::getenv("XHDFE_REDUCED_PCG_CUDA");
        if(e&&e[0]=='1'&&e[1]=='\0')return true;
        if(e&&e[0]=='0'&&e[1]=='\0')return false;
        return kReducedPcgCudaBuiltIn;
    }();
    return on;
}
std::optional<AbsorptionResult> absorb_reduced_pcg_cuda(
    const Eigen::Ref<const Eigen::VectorXd>& y,const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,const HdfeOptions& options,double trigger,ReducedPcgOutcome* outcome) {
    int declined=0;
    try{return attempt_reduced_pcg_cuda(y,X,fes,options,trigger,outcome,declined);}
    catch(const reduced_cuda::Error& error){
        if(error.kind()==reduced_cuda::ErrorKind::Fatal)throw;
        return decline_last_resort(outcome,decline_reason(error.kind()),declined);
    }catch(const std::bad_alloc&){return decline_last_resort(outcome,"memory",declined);}
    catch(const std::exception&){return decline_last_resort(outcome,"exception",declined);}
}
} // namespace detail
} // namespace hdfe
#endif
