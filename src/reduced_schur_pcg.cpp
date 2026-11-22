// The reduced-system tier (on by default; XHDFE_REDUCED_PCG=0 turns it off):
// the adapter between the regressor and the self-contained kernel of
// reduced_schur_pcg.hpp. A translation unit of its own, so that the code
// generated for fe_absorption.cpp does not change with the tier.

#include "reduced_schur_pcg_entry.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
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

namespace hdfe {
namespace detail {
namespace {

namespace rp = hdfe::reduced_pcg;

thread_local int fit_depth = 0;
thread_local bool final_printed = false;
thread_local int accepted_team = 0;
thread_local int accepted_workers = 0;
// Point 13 (ReducedPcgFinalKeep): touched only while the trace is on.
thread_local bool final_keeping = false;    // the fit's final lines are kept, not printed
thread_local bool final_kept = false;       // the first of them is in final_kept_fields
thread_local bool final_deferred = false;   // it waits for the end of the outermost fit
thread_local const char* final_fallback = nullptr;
thread_local std::string final_kept_fields;

void emit_final(const std::string& fields) {
    std::ostringstream line;
    line << "xhdfe reduced_pcg event=final " << fields << '\n';
    std::cerr << line.str();
}

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

// XHDFE_REDUCED_PCG_PREFETCH: how many cells ahead the segment pass of the
// tier's operator requests the rows of the effects that do not fit in cache
// (rp::Config::prefetch_cells). "0" keeps the loop of 2.34.1 for every
// shape; an integer from 1 to 4096 replaces the default distance; anything
// else, or nothing, keeps the default. The bits of the fit do not depend on
// it. Read once per process.
int prefetch_cells_switch(int fallback) {
    static const int value = [] {
        const char* e = std::getenv("XHDFE_REDUCED_PCG_PREFETCH");
        if (e == nullptr || e[0] == '\0') return -1;
        char* end = nullptr;
        const long parsed = std::strtol(e, &end, 10);
        if (end == e || *end != '\0' || parsed < 0 || parsed > 4096) return -1;
        return static_cast<int>(parsed);
    }();
    return value >= 0 ? value : fallback;
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

}  // namespace

bool reduced_pcg_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("XHDFE_REDUCED_PCG");
        return !(e != nullptr && e[0] == '0' && e[1] == '\0');
    }();
    return enabled;
}

int reduced_pcg_trace_level() {
    static const int level = [] {
        const char* e = std::getenv("XHDFE_REDUCED_PCG_TRACE");
        // An empty value is level 0; e[1] is read only after e[0] proved non-empty.
        if (e == nullptr || e[0] == '\0' || e[1] != '\0') return 0;
        return e[0] == '1' ? 1 : (e[0] == '2' ? 2 : 0);
    }();
    return level;
}

bool reduced_pcg_effective(const HdfeOptions& options) {
    switch (options.reduced_pcg) {
        case ReducedPcgRequest::On:
            return true;
        case ReducedPcgRequest::Off:
            return false;
        case ReducedPcgRequest::Default:
        default:
            return reduced_pcg_enabled();
    }
}

bool reduced_pcg_chat_stop_applies(const HdfeOptions& options, double accept_trigger) {
    if (!chat_stop_switch()) return false;
    // The certificate's limit (certify_absorption_result, fe_absorption.cpp:
    // 24997-24999): D10 applies only when the trigger is the threshold in force.
    const double certificate_limit = std::max(8.0 * std::max(0.0, effective_tolerance(options)),
                                              64.0 * std::numeric_limits<double>::epsilon());
    return accept_trigger > 0.0 && certificate_limit >= accept_trigger;
}

ReducedPcgFitScope::ReducedPcgFitScope() : ReducedPcgFitScope(reduced_pcg_enabled()) {}

ReducedPcgFitScope::ReducedPcgFitScope(bool enabled) {
    if (!enabled || reduced_pcg_trace_level() <= 0) return;
    counted_ = true;
    if (fit_depth++ == 0) final_printed = false;
}

ReducedPcgFitScope::~ReducedPcgFitScope() {
    if (!counted_) return;
    if (--fit_depth == 0 && final_deferred) {
        // The line ReducedPcgFinalKeep::publish deferred: the fit's own when
        // it ended with estimates, the line of the code before otherwise.
        final_deferred = false;
        try {
            if (completed_ || final_fallback == nullptr) {
                emit_final(final_kept_fields);
            } else {
                emit_final(final_fallback);
            }
        } catch (...) {
        }
        final_fallback = nullptr;
        final_kept_fields.clear();
        return;
    }
    if (fit_depth == 0 && !final_printed) {
        final_printed = true;
        try {
            emit_final("status=ineligible reason=not_reached");
        } catch (...) {
        }
    }
}

void reduced_pcg_final_line(const std::string& fields) {
    // The caller holds the fit's effective switch on (see the header).
    if (reduced_pcg_trace_level() <= 0) return;
    if (final_keeping) {
        if (!final_kept) {
            final_kept_fields = fields;
            final_kept = true;
        }
        return;
    }
    if (fit_depth > 0 && final_printed) return;
    final_printed = true;
    emit_final(fields);
}

ReducedPcgFinalKeep::ReducedPcgFinalKeep(bool armed) {
    if (!armed || reduced_pcg_trace_level() <= 0 || final_keeping) return;
    if (fit_depth > 0 && final_printed) return;
    active_ = true;
    final_keeping = true;
    final_kept = false;
}

ReducedPcgFinalKeep::~ReducedPcgFinalKeep() {
    if (!active_) return;
    final_keeping = false;
    final_kept = false;
}

bool ReducedPcgFinalKeep::publish(const char* fallback) noexcept {
    if (!active_) return false;
    active_ = false;
    final_keeping = false;
    if (!final_kept) return false;
    final_kept = false;
    if (fit_depth <= 0) {
        // No public fit to wait for: printed now.
        try {
            emit_final(final_kept_fields);
        } catch (...) {
        }
        final_kept_fields.clear();
        return true;
    }
    final_printed = true;
    final_deferred = true;
    final_fallback = fallback;
    return true;
}

void reduced_pcg_forget_observation() {
    accepted_team = 0;
    accepted_workers = 0;
}

void reduced_pcg_fold_observation(int& team, int& workers) {
    team = std::max(team, accepted_team);
    workers = std::max(workers, accepted_workers);
    accepted_team = 0;
    accepted_workers = 0;
}

#ifdef HDFE_USE_CUDA
void reduced_pcg_record_cuda_observation(int team, int workers) noexcept {
    accepted_team = std::max(accepted_team, team);
    accepted_workers = std::max(accepted_workers, workers);
}
#endif

namespace {

// The attempt itself. Whatever it throws, the setup included, reaches the
// handlers of absorb_reduced_pcg below. Invariant: a value is returned if and
// only if the outcome says accepted (status 1).
std::optional<AbsorptionResult> attempt_reduced_pcg(
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
        reduced_pcg_final_line(fields + " seconds=" + std::to_string(since(start)));
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
        auto t0 = std::chrono::steady_clock::now();
        std::vector<CompactCodes> compact(static_cast<std::size_t>(k));
        std::vector<const int*> code_ptrs(static_cast<std::size_t>(k));
        for (int b = 0; b < k; ++b) {
            compact[static_cast<std::size_t>(b)] = compact_codes(fes[static_cast<std::size_t>(b)], team, blocks);
            levels[static_cast<std::size_t>(b)] = compact[static_cast<std::size_t>(b)].levels;
            code_ptrs[static_cast<std::size_t>(b)] = compact[static_cast<std::size_t>(b)].codes.data();
        }
        compact_seconds = since(t0);
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
        config.prefetch_cells = prefetch_cells_switch(config.prefetch_cells);
        config.max_iter = std::max(1, std::min(1500, options.max_iter));
        // The columns and the identifiers of the fit (rp::admission_budget).
        config.memory_budget = rp::admission_budget(n, cols, k);
        budget = config.memory_budget;
        config.hooks = hooks;
        config.force_decline_geometry = stage == 1;
        config.force_decline_iterate = stage == 2;

        report = rp::solve(problem, config);
        if (!report.solved) return decline(rp::decline_name(report.decline), -1.0);
        if (stage == 3) return decline("forced", -1.0);

        result.iterations = report.iterations;
        result.converged = true;
        result.fe_levels = levels;  // distinct ids per effect, in the input order (feeds the DoF)
        HdfeOptions certificate_options = options;
        certificate_options.parallel_observer = &observer;
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
        accepted_team = std::max(accepted_team, decided_team);
        accepted_workers = std::max(accepted_workers, decided_workers);
        if (outcome != nullptr) {
            outcome->status = 1;
            outcome->reason = "none";
            outcome->iterations = report.iterations;
        }
        return result;
    } catch (const std::bad_alloc&) {
        return decline("memory", -1.0);
    } catch (const std::exception&) {
        return decline("exception", -1.0);
    }
}

}  // namespace

#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
std::optional<AbsorptionResult> absorb_reduced_pcg(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const HdfeOptions& options,
    double accept_trigger,
    ReducedPcgOutcome* outcome) {
    // Iterations of the attempt as its last decline recorded them.
    int declined_iterations = 0;
    try {
        return attempt_reduced_pcg(y, X, fes, options, accept_trigger, outcome, declined_iterations);
    } catch (const std::bad_alloc&) {
        return decline_last_resort(outcome, "memory", declined_iterations);
    } catch (const std::exception&) {
        return decline_last_resort(outcome, "exception", declined_iterations);
    }
}

}  // namespace detail
}  // namespace hdfe

#include "mixed_network_impl.hpp"
