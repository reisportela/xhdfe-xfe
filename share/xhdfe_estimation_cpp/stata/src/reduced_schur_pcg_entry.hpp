#ifndef HDFE_REDUCED_SCHUR_PCG_ENTRY_HPP
#define HDFE_REDUCED_SCHUR_PCG_ENTRY_HPP

// The reduced-system tier (on by default; XHDFE_REDUCED_PCG=0 turns it off).
// Included only by src/reduced_schur_pcg.cpp, which implements it, and by
// src/hdfe_regressor_v11.cpp, which calls it once per ordinary absorption.

#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "fe_absorption.hpp"
#include "hdfe/hdfe_regressor.hpp"

namespace hdfe {
namespace detail {

// The process switch: the tier is on unless XHDFE_REDUCED_PCG is exactly 0
// (read once per process).
bool reduced_pcg_enabled();

// The effective switch of one fit: on when the fit requests it
// (HdfeOptions::reduced_pcg == On), off when it requests Off, otherwise the
// process switch. Every decision the regressor takes for a particular fit
// (entry, cache tag, observation fold, diagnostics) reads this, never the
// process switch alone.
bool reduced_pcg_effective(const HdfeOptions& options);

// D10 (pre-registration Verifications/perf_stata_path_20261114/d10/): true when
// the process switch XHDFE_REDUCED_PCG_CHAT_STOP (read once; "0" turns it off)
// is on and the attempt's acceptance threshold in force is the trigger, that is
// when the certificate's limit for these options is not below accept_trigger.
// Then the attempt stops its columns on the confirmed chat at 0.1 accept_trigger
// instead of tau, and the fit's absorption-cache key carries D10's tag; otherwise
// both are as before D10, bit for bit.
bool reduced_pcg_chat_stop_applies(const HdfeOptions& options, double accept_trigger);

// What one attempt did, for the fit's read-only diagnostics: status 1 when
// it published the fit (then, and only then, the adapter returns a value), 2
// when it declined; the reason ("none" when accepted), a static string so that
// recording it never allocates; the iterations the attempt spent.
struct ReducedPcgOutcome {
    int status = 0;
    const char* reason = "";
    int iterations = 0;
};

// XHDFE_REDUCED_PCG_TRACE (stderr): 1 prints plan, phase, column and final
// lines at constant cost; 2 adds the verified rel of each column. Read once,
// and only while the tier is on.
int reduced_pcg_trace_level();

// While the tier and its trace are on, every public fit prints exactly one
// "xhdfe reduced_pcg event=final" line: the tier's own, or
// "status=ineligible reason=not_reached" when the fit never reached it.
// Nested fits (zero weights, group()) share the line of the outer fit. The
// regressor passes the fit's effective switch; the default constructor uses
// the process switch. complete() marks a fit that ends with estimates; it only
// matters for a line ReducedPcgFinalKeep::publish deferred to the end of the fit.
class ReducedPcgFitScope {
public:
    ReducedPcgFitScope();
    explicit ReducedPcgFitScope(bool enabled);
    ~ReducedPcgFitScope();
    ReducedPcgFitScope(const ReducedPcgFitScope&) = delete;
    ReducedPcgFitScope& operator=(const ReducedPcgFitScope&) = delete;
    void complete() noexcept { completed_ = true; }

private:
    bool counted_ = false;
    bool completed_ = false;
};

// The final line of the current public fit (fields after "event=final"),
// unless that fit printed one already. Called only for a fit whose effective
// switch is on (by the tier, or by the regressor under that switch).
void reduced_pcg_final_line(const std::string& fields);

// Point 13 (17nov2026): correction E1 recomputes the projection of a fit with
// saved effects (hdfe_regressor_v11.cpp, publish_savefe_projection). While an
// armed keep is alive, the first final line of the fit (the projection's
// attempt) is kept instead of printed, and the destructor drops it. publish()
// makes it the fit's final line, printed when the outermost fit ends: the kept
// line if that fit ended with estimates (ReducedPcgFitScope::complete), else
// `fallback`, the line printed before the correction. A keep is armed only
// with the trace on and the fit's final line still to print; it allocates
// nothing otherwise, and nothing in it is read by any computation.
class ReducedPcgFinalKeep {
public:
    explicit ReducedPcgFinalKeep(bool armed);
    ~ReducedPcgFinalKeep();
    ReducedPcgFinalKeep(const ReducedPcgFinalKeep&) = delete;
    ReducedPcgFinalKeep& operator=(const ReducedPcgFinalKeep&) = delete;
    // True when a line was kept and is now the fit's final line; false when
    // nothing was kept (the caller then prints its own line).
    bool publish(const char* fallback) noexcept;

private:
    bool active_ = false;
};

// The attempt reports to a private observer, so that a declined one leaves
// the fit's diagnostics untouched. An accepted attempt's team and active
// workers are folded into threads_used and the active-worker count when the
// fit ends its parallel observation; the start of an observation forgets them.
void reduced_pcg_forget_observation();
void reduced_pcg_fold_observation(int& team, int& workers);
#ifdef HDFE_USE_CUDA
// Private observation publication for the opt-in CUDA tier; no CPU solver change.
void reduced_pcg_record_cuda_observation(int team, int workers) noexcept;
bool reduced_pcg_cuda_enabled() noexcept;
// The CUDA variant's request of one fit: on when the fit requests it
// (HdfeOptions::reduced_pcg_cuda == On), off when it requests Off, otherwise
// the process switch above (XHDFE_REDUCED_PCG_CUDA exactly 1, read once per
// process). Only the regressor's entry to the tier reads it, for a
// reghdfe-comparable fit that requests CUDA; the tier's effective switch
// (reduced_pcg_effective) must be on as well.
inline bool reduced_pcg_cuda_effective(const HdfeOptions& options) noexcept {
    switch (options.reduced_pcg_cuda) {
        case ReducedPcgRequest::On:
            return true;
        case ReducedPcgRequest::Off:
            return false;
        case ReducedPcgRequest::Default:
        default:
            return reduced_pcg_cuda_enabled();
    }
}
std::optional<AbsorptionResult> absorb_reduced_pcg_cuda(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const HdfeOptions& options, double accept_trigger, ReducedPcgOutcome* outcome);
#endif

// Exact elimination of the effect with the most levels and Jacobi PCG on the
// Schur complement of the others, certified by certify_absorption_result.
// Returns the result when it is certified and its verified abs_residual_rel
// is at most accept_trigger; otherwise nothing, and the caller runs its own
// chain from the original data. Unit weights, no slopes, 2 to 5 effects.
// The outcome, when given, receives the status, reason and iterations.
std::optional<AbsorptionResult> absorb_reduced_pcg(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const HdfeOptions& options,
    double accept_trigger,
    ReducedPcgOutcome* outcome = nullptr);

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_REDUCED_SCHUR_PCG_ENTRY_HPP
