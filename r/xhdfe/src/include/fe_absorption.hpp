#ifndef HDFE_FE_ABSORPTION_HPP
#define HDFE_FE_ABSORPTION_HPP

#include <Eigen/Dense>

#include <type_traits>
#include <memory>
#include <vector>

#include "hdfe/hdfe_regressor.hpp"

namespace hdfe {
namespace detail {

struct N1FeEvidence;

struct CudaForwardProbeSummary {
    int status = 0;  // 0 not run, 1 ok, 2 fit error, 3 CUDA error, 4 ineligible
    int continuation_sweeps = 0;
    double coefficient_drift_abs = 0.0;
    double rss_drift_rel = 0.0;
    double standard_error_drift_abs = 0.0;
    double covariance_drift_abs = 0.0;
    double contraction = 0.0;
    double coefficient_tail_abs = 0.0;
    double condition_proxy = 0.0;
};
static_assert(std::is_trivially_copyable<CudaForwardProbeSummary>::value,
              "CUDA forward-probe summary must remain POD-like");

struct GroupIndividualStructure {
    int num_groups = 0;
    int num_individuals = 0;
    std::vector<int> group_ptr;
    std::vector<int> group_individual;
    std::vector<int> individual_ptr;
    std::vector<int> individual_group;
    std::vector<double> group_scale;
};

// Codes of AbsorptionResult::cuda_tail. With three ordinary fixed effects in
// reghdfe-comparable mode, the CUDA path completes the projection after its
// sweeps (the "tail"): on the device with LSMR, or, when a short device probe
// predicts a long device solve and the CPU's own route selector would use
// MLSMR on this fit, on the host with MLSMR. kCudaTailNone when no completion
// ran (CPU fits, other routes, cache hits). The cold restart after a broken
// CG stage (F2-B) is probed the same way: kCudaTailHostMlsmr when the host
// MLSMR solved it from the original data; a restart kept on the device
// reports kCudaTailNone, as in 2.30.1.
constexpr int kCudaTailNone = 0;
constexpr int kCudaTailDeviceLsmr = 1;
constexpr int kCudaTailHostMlsmr = 2;

inline const char* cuda_tail_name(int code) noexcept {
    return code == kCudaTailDeviceLsmr ? "device-lsmr"
         : code == kCudaTailHostMlsmr  ? "host-mlsmr"
                                       : "none";
}

// Package C (18nov2026): the GPU status code of a weighted fit with two or more
// absorbed effect dimensions, in reghdfe-comparable or strict-residual mode,
// whose effective backend was CUDA and which the regressor ran as the CPU call
// on an explicit per-fit request. Name
// weighted_cpu_route. Only the code says so: the other GPU fields are the CPU
// call's. 6 is taken (Gelbach reads it as not_applicable).
constexpr int kGpuStatusWeightedCpuRoute = 7;

// Phase 5a: false under XHDFE_SLOPE_TEAM_INDEXERS=0 (read once per process in
// fe_absorption.cpp), which keeps the serial identifier builds of the route with
// heterogeneous slopes; otherwise they run on the team, bit for bit the same.
bool slope_team_indexers_enabled();

struct AbsorptionResult {
    std::shared_ptr<N1FeEvidence> n1_fe;
    Eigen::VectorXd y_tilde;
    Eigen::MatrixXd X_tilde;
    std::vector<int> fe_levels;
    std::vector<std::vector<int>> fe_group_ids;
    std::vector<Eigen::VectorXd> fe_means;
    std::vector<Eigen::VectorXd> fe_weight_sums;
    std::vector<Eigen::VectorXd> fe_alpha_y;
    std::vector<Eigen::MatrixXd> fe_alpha_X;
    std::vector<Eigen::VectorXd> fe_slope_alpha_y;
    std::vector<Eigen::MatrixXd> fe_slope_alpha_X;
    std::vector<int> sweep_order_used;
    int iterations = 0;
    bool converged = true;
    // A failed CG recurrence can lose components outside the FE span.
    // Its output must never seed a precision repair.
    bool cg_restart_required = false;
    double abs_residual = 0.0;      // verified ||D' W v_tilde||_2, max over RHS columns
    double abs_residual_rel = 0.0;  // max ||D' W v_tilde||_2 / (||D' W||_F ||v||_2)
    bool slope_block_checked = false;  // false in fast mode; block metrics then not applicable
    double slope_block_residual_rel = 0.0;  // max scale-invariant per-block/RHS certificate
    double slope_block_frobenius_rel = 0.0;  // max ||m||/(sqrt(sum d)||u_tilde||_W)
    double slope_block_rms_rel = 0.0;  // max RMS diagonal-scaled group moment
    double slope_block_max_rel = 0.0;  // max worst-group diagonal-scaled moment
    double slope_block_skipped_max_rel = 0.0;  // max skipped rank-deficient-group cosine
    int slope_certificate_worst_fe = -1;
    int slope_certificate_worst_moment = -1;  // 0 intercept/plain, 1 slope
    int slope_accuracy_retry_stages = 0;
    int slope_accuracy_retry_iterations = 0;
    double slope_internal_tolerance = 0.0;
    double krylov_internal_tolerance = 0.0;
    double krylov_max_final_backward_error = 0.0;
    double krylov_max_condition = 0.0;
    double krylov_max_condition_times_backward_error = 0.0;
    bool auto_routing_retry_policy_enabled = false;
    bool auto_routing_retry_eligible = false;
    bool auto_routing_retry_fired = false;
    int auto_routing_retry_status = 0;
    int auto_routing_retry_primary_method = -1;
    int auto_routing_retry_primary_iterations = 0;
    double auto_routing_retry_primary_abs_residual_rel = 0.0;
    int auto_routing_retry_iterations = 0;
    double auto_routing_retry_abs_residual_rel = 0.0;
    double auto_routing_retry_elapsed_seconds = 0.0;
    // Early routing (options.early_routing): 0 none, 1 a sweep or a gated
    // conjugate-gradient stage left at a checkpoint (predictive rule), 2 the
    // hand-off to a conjugate-gradient stage was not made (structural rule).
    // A left result is unconverged and is discarded: the regressor runs the
    // retry's MLSMR instead.
    int early_routing_left = 0;
    double early_routing_prediction = 0.0;  // kappa*q*tol at the leave
    double early_routing_kappa = 0.0;       // certificate / stop quantity
    double early_routing_rate = 0.0;        // q, decay of the stop quantity per iteration
    bool precision_certified = true;
    bool schwarz_used = false;  // true when the Schwarz/approx-Cholesky PCG path ran (forced or auto-gated)
    bool mlsmr_used = false;    // true when the MLSMR absorber ran via the auto-gate promotion
    bool gpu_used = false;
    int gpu_status_code = 0;  // 0 none, 1 used, 2 unavailable, 3 not converged, 4 failed, 5 CPU cache/profile
    bool gpu_attempted = false;
    bool gpu_absorption_converged = false;
    int gpu_absorption_iterations = 0;
    CudaForwardProbeSummary cuda_forward_probe;
    int cuda_accuracy_retry_trigger = 0;  // 0 none, 1 residual, 2 forward
    int cuda_accuracy_retry_stage = 0;    // 0 none, 1-3 warm, 4 cold
    int cuda_accuracy_retry_iterations = 0;
    int cuda_tail = kCudaTailNone;  // how the three-FE Comparable CUDA completion ran
    // F2-B ledger of a CUDA CG stage entered from the IT sweeps: the sweeps
    // before the hand-off that `iterations` does not count (2.30.1
    // accounting), why the sweeps handed off (1 divergence, 2 iteration cap),
    // and the wall seconds of the sweeps and of the CG stage (trace only; no
    // decision reads them). The cold restart that follows a broken CG stage
    // charges these sweeps to the fit's budget and to `iterations`.
    int cuda_cg_uncounted_it_iterations = 0;
    int cuda_it_handoff_reason = 0;
    double cuda_it_seconds = 0.0;
    double cuda_cg_seconds = 0.0;
    // P3-fix: cold restarts of a broken xhdfe-fast CUDA CG stage (0 none) and
    // whether the device LSMR completed the fit after them (fe_absorption_cuda.cu,
    // restart_fast_cg_after_breakdown). Diagnostics and the refusal message only.
    int cuda_fast_cg_restarts = 0;
    bool cuda_fast_lsmr_restart = false;
    // P3-fix (C): set by the bounded xhdfe-fast CUDA repair (hdfe_regressor_v11.cpp)
    // when its result converged and passed the public certificate but missed the
    // repair acceptance target, which it holds (0.5*tol); 0 otherwise. The fit is
    // refused as before; only the message changes.
    double fast_repair_target = 0.0;
    // P3-fix (3'), strict scope: a stage with the strict provenance restarted its
    // broken CG stage with the device LSMR and the strict check, or met a second
    // breakdown after the sequence had used that restart (fe_absorption_cuda.cu,
    // restart_strict_after_breakdown). The dispatch carries it into a "not
    // converged" result (fe_absorption.cpp); the regressor hands it to the fit's
    // one strict repair as HdfeOptions::strict_lsmr_consumed.
    bool cuda_strict_lsmr_restart = false;
    // Point 14 (17nov2026): set only by absorb_fixed_effects_krylov, on the
    // transforms its preconditioned conjugate gradients produced. The regressor
    // reports KrylovPcg when this result is published under a forced Krylov
    // route; no computation reads it. Last member, so that no other offset moves.
    bool krylov_pcg_used = false;
    // Package B, stage 2a' (18nov2026), appended so that no other offset moves: the
    // raw sums of squares of the original y and of every original column (instruments
    // included) as the last certificate of this result formed them, before any
    // normalisation; and the request, set by the regressor only for the
    // recertification after the strict continuation, that this certificate take them
    // over instead of forming them again (the dominant-origin normalisation of the
    // comparable mode still runs on them). The certificate clears the request.
    std::vector<long double> raw_original_norm_sq;
    bool reuse_raw_original_norm_sq = false;
    // The CPU LSMR/MLSMR Strict projection needs the absolute check on readback too.
    AbsorptionMethod strict_cpu_projection_method = AbsorptionMethod::Auto;
    long double strict_cpu_max_abs_mean = -1.0L;
    bool weighted_gs_moments_met = true;
    bool cuda_projection_moments_met = true;
    bool cuda_projection_repaired = false;
};

struct HeterogeneousSlopeTerm {
    int fe_index = -1;
    Eigen::VectorXd values;
    bool include_intercept = false;
};

struct FeRecoveryResult {
    std::vector<Eigen::VectorXd> contributions;
    int iterations = 0;
    double max_delta = 0.0;
    bool converged = true;
};

// The FE-collinearity constant: a regressor whose absorbed sum of squares is
// at most this fraction of its raw sum of squares (of its centred one when the
// model has a constant) is collinear with the absorbed effects. The regressor
// omits such a column (hdfe_regressor_v11.cpp), and the heterogeneous-slope
// block certificate judges it in its input scale (certify_absorption_result,
// HdfeOptions::fe_collinear_omission_cols).
constexpr double kFeCollinearTol = 1e-9;

// Compute an explicit normal-equation residual for the returned within
// transform. The diagnostic is independent of the solver's internal stopping
// proxy and uses fixed logical chunks, so its value is invariant to the OpenMP
// team size within one artefact/backend.
void certify_absorption_result(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    const std::vector<HeterogeneousSlopeTerm>& slopes,
    AbsorptionResult& result,
    const GroupIndividualStructure* group_individual = nullptr,
    bool refresh_cuda_projection = false);

// One weighted projection on the span of forward GS increments, within the
// remaining iteration budget. Called after the per-column moment check fails.
bool repair_weighted_gauss_seidel(
    const std::vector<Eigen::VectorXi>& fes,
    const Eigen::VectorXd& weights,
    const HdfeOptions& options,
    AbsorptionResult& result);

// Complete an inaccurate ordinary CUDA projection on the device, in the
// remaining iteration budget. The independent check sees the original data.
bool repair_cuda_projection(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    AbsorptionResult& result);

// The codes the reduced-system tier computed for one effect (phase 2 of the
// Stata-path plan, design note Verifications/perf_stata_path_20261114/fase2/
// DESENHO_FASE2.md, section 6): the ascending compact ranks 0..levels-1 of the
// effect's identifiers, whose minimum and maximum are min_id and max_id.
struct TierFeCodes {
    std::vector<int>* codes = nullptr;
    int levels = 0;
    int min_id = 0;
    int max_id = 0;
};

// certify_absorption_result(y, X, fes, nullptr, options, {}, result) for the
// tier's attempt, with its codes (one entry per effect of fes, in order). For
// every effect whose indexer would number the identifiers by the same ranks
// (sparse_rank::indexer_numbers_by_rank()) the certificate takes the codes vector
// over instead of numbering the identifiers again; the other effects are numbered
// as before and their vectors are left alone. The result is the same.
void certify_tier_absorption(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const HdfeOptions& options,
    AbsorptionResult& result,
    std::vector<TierFeCodes>& tier_codes);

// The Krylov parity floor of the fit's mode as MLSMR applies it: krylov_parity_tolerance
// (fe_absorption.cpp) with absorption_method = Mlsmr, the atol the reduced tier also uses
// (reduced_schur_pcg.cpp, mlsmr_parity_tolerance). It follows the public tolerance:
// min(effective tolerance, 1e-11) in reghdfe-comparable and strict-residual with the parity
// floor on (1e-11 at the default 1e-8), the effective tolerance otherwise. Used by correction
// E1 (hdfe_regressor_v11.cpp, publish_savefe_projection); not a new tolerance.
double mode_parity_floor(const HdfeOptions& options);

// Package B (strict-residual on CUDA, 17nov2026). The largest absolute group
// mean of `values` over the absorbed effects `fes`: a fresh pass,
// long-double sums per group of each effect in row order, the effects in parallel,
// each serial (deterministic). Stage 2a' (18nov2026): the groups are indexed by the
// codes themselves when they are non-negative and dense (largest code below 5e7
// and at most 2n, build_indexer's dense range), by build_indexer otherwise
// (sparse or negative identifiers); the sums and their order are the same.
// `finite` is false when a value or a group mean is not finite; `maximum` is then
// +infinity.
struct GroupMeanReading {
    long double maximum = 0.0L;
    bool finite = true;
    std::vector<long double> per_effect;
};
GroupMeanReading max_abs_group_mean(const Eigen::Ref<const Eigen::VectorXd>& values,
                                    const std::vector<Eigen::VectorXi>& fes,
                                    int threads,
                                    const Eigen::VectorXd* weights = nullptr);

// Package B: the continuation of a strict-residual fit on CUDA whose delivered
// residual failed D(e) <= tol. Gauss-Seidel sweeps of y_tilde and of every
// column of X_tilde (kept regressors, omitted ones, instruments), in `order`
// (the order of the effects the fit used; when empty, the order the GPU
// dispatch of absorb_fixed_effects_v6 computes for the original data: by
// profile with three or more effects, select_sweep_order on y_original with
// two). After each effect, the group means of the combination
// y_tilde - X_tilde * combination removed by it are read on the device (the
// in-sweep maximum, a trigger only); when a sweep's in-sweep maximum is at or
// below tol, a fresh device pass (the residual formed by row, one column
// reduced per effect) confirms, and the continuation stops iff the fresh
// maximum is at or below tol. It never refuses by that rule: at the last sweep
// max_sweeps allows the state is returned as it is. Confirmations are not
// sweeps. Returns false when the device did not serve the call (the caller
// refuses); y_tilde and X_tilde are then unspecified.
struct StrictContinuationReport {
    int sweeps = 0;
    int confirmations = 0;
    bool confirmed = false;         // stopped by a fresh maximum at or below tol
    double first_effect_in_sweep = -1.0;  // in-sweep maximum of order[0], first sweep
    double last_in_sweep = -1.0;
    double last_fresh = -1.0;
    std::vector<int> order;         // the order the sweeps used
    bool order_inherited = false;   // true: the fit's sweep_order_used
    double seconds_indexers = 0.0;
    double seconds_upload = 0.0;
    double seconds_sweeps = 0.0;
    double seconds_confirmations = 0.0;
    double seconds_download = 0.0;
    // Stage 2a'' (input, set by the regressor): 2 prints one trace line per fresh reading
    // and, for the trace alone, also takes one at power-of-two sweeps (no decision reads
    // it); 0 and 1 leave the work of stage 2a' unchanged.
    int trace_level = 0;
    // Stage 2c (b/RESPOSTA_2B_B_ADENDA.md): the continuation has no frozen exit. A window
    // of three equal power-of-two readings is no evidence of a stalled state there: on the
    // reviewer's deterministic counterexample at the continuation's entry (16 pairs, 32
    // rows, two effects A_j = (u_j, v_j) and B_j = (v_j, u_{j+1}), no regressors, e(u_j) =
    // -e(v_j) = 2a min(j, 16 - j), a = 43 * 2^-32, order A, B) the readings after sweeps 1, 2
    // and 4 are bit for bit a > 1e-8 and the fifth is 127a/128 < 1e-8, so such a rule
    // refuses at the fourth sweep a fit the trigger accepts at the sixth.
    const char* exit = "budget";  // confirmed or budget (output)
};
bool continue_strict_residual(Eigen::VectorXd& y_tilde,
                              Eigen::MatrixXd& X_tilde,
                              const Eigen::Ref<const Eigen::VectorXd>& y_original,
                              const std::vector<Eigen::VectorXi>& fes,
                              const std::vector<int>& order_used,
                              const Eigen::VectorXd& combination,
                              double tol,
                              int max_sweeps,
                              int threads,
                              StrictContinuationReport& report);

// Package B, stage 2a' (18nov2026; b/RESPOSTA_2A_B.md): the polish of a strict-residual
// fit in scope at a sweep exit. The regressor sets HdfeOptions::strict_sweep_exit on the
// primary absorption only. The CUDA sweep route (absorb_fixed_effects_cuda_impl), when its
// main solve ended in Gauss-Seidel sweeps (no LSMR completion, no tail hand-off, no CG
// stage) and after the six fixed sweeps of the comparable polish, reads it once: a
// provisional b0 from the Gram of [X~ | y~] formed on the device and solved on the host
// (strict_sweep_exit_solve; no Gram without regressors, b0 empty), then D0, the largest
// absolute group mean of y~ - X~ b0 measured fresh on the device; above `tol` it sweeps
// every column in the route's order, each sweep counted against max_iter, until a fresh
// confirmation is at or below `tol` (the in-sweep maximum of the combination is only the
// trigger), b0 fixed. It never marks non-convergence and never refuses: at the last sweep
// the budget allows the state goes on as it is. Skipped, the fit unchanged, with
// instruments, above kStrictSweepExitMaxColumns regressors, a non-finite Gram, or a rank
// that is deficient or doubtful (strict_sweep_exit_solve). The fields after `instruments`
// are the step's report (trace only).
constexpr int kStrictSweepExitMaxColumns = 32;
// Stage 2b, step 0 (18nov2026; b/RESPOSTA_2A3_B.md): in the direct form (no symmetric
// sweep) a fresh reading is also taken when the count of additional sweeps is a power of
// two (at most one fresh reading per sweep), and the step hands the state over to the
// host, nothing confirmed, when the readings taken at the last three power-of-two counts
// are bit for bit the same (`frozen`; trigger readings in between do not enter that
// test). XHDFE_STRICT_FROZEN_STOP: `0` (stage 2a' exactly) or anything else, unset
// included (frozen_stop 1, the default; stage 2c removed the value `all`, which now means
// the default). The symmetric form is stage 2a' under every value. `frozen_stop` and
// `trace_level` are inputs set by the regressor (XHDFE_STRICT_ACCEPT_TRACE=2 prints every
// fresh reading).
struct StrictSweepExitRequest {
    double tol = 0.0;           // the public tolerance (the primary's may be tighter)
    double rank_tol = 0.0;      // the regressor's rank tolerance, max(1e-6, 10 tol)
    bool instruments = false;   // the fit has instruments: the step is skipped
    int frozen_stop = 1;        // 0 (stage 2a'), 1 step (the default); the step sets the
                                // effective value (0 on the symmetric form)
    int trace_level = 0;        // 2: one trace line per fresh reading
    bool consumed = false;      // the step was reached (once per fit)
    const char* exit = "none";  // d0, confirmed, frozen, budget, device; none when not
                                // reached or skipped
    int readings = 0;           // fresh readings after D0
    double best = -1.0;         // the smallest fresh reading (D0 included)
    bool formed = false;        // b0 formed (or empty without regressors) and D0 measured
    const char* skipped = "";   // why b0 was not formed: instruments, columns, cg-exit,
                                // non-finite, fe-collinear, rank, device
    int columns = 0;
    double lambda_min = -1.0;   // smallest eigenvalue of the unit-diagonal Gram
    double seconds_raw = 0.0;   // raw sums of squares of the original columns (device)
    double seconds_gram = 0.0;
    double seconds_solve = 0.0;
    double seconds_total = 0.0;
    double d0 = -1.0;
    int sweeps = 0;             // additional sweeps (counted in iterations)
    int confirmations = 0;
    bool confirmed = false;     // D0 or a fresh confirmation at or below tol
    double last_in_sweep = -1.0;
    double last_fresh = -1.0;
};

// Package B, stage 2a': b0 from the device Gram of [X~ | y~] (`upper`: its upper triangle,
// row major, cols + 1 columns, y~ last). Skipped (false, `reason` set) when an entry is not
// finite; when a column's absorbed sum of squares is at most 1e3 * kFeCollinearTol times its
// raw one (`raw_ss`, the original column; the regressor omits a column at 1e0 times); or
// when the smallest eigenvalue of the unit-diagonal Gram is below (1e3 * rank_tol)^2, the
// regressor dropping a column whose residual norm after the others is at most rank_tol times
// its norm. Otherwise b0 solves the scaled system by LDLT.
bool strict_sweep_exit_solve(const std::vector<double>& upper,
                             int cols,
                             const std::vector<double>& raw_ss,
                             double rank_tol,
                             std::vector<double>& b0,
                             double& lambda_min,
                             const char*& reason);

// Authoritative convergence gate for group()/individual() absorption.
// The norm-change proxy is only a candidate trigger: this routine certifies
// the explicit backward error (and the strict maximum-mean condition when
// requested), then makes converged and precision_certified agree.
// Forward-error target accepted by the group/individual Krylov solver: the
// running LSMR condition estimate times the dual residual test must not
// exceed it before a dual stop is accepted. Derived from the public
// tolerance of the mode (100 x the mode's dual tolerance, floored at 1e-8)
// unless options.group_forward_tolerance carries an explicit value.
double group_forward_tolerance(const HdfeOptions& options);

// Weighted relative residual sum of squares of the ones vector projected on
// the span of the absorbed effects (ordinary FEs and the group/individual
// incidence). Decides whether a design spans the constant when its structure
// alone cannot (sum aggregation, no ordinary FE, uneven team sizes). One
// right-hand side solved by the exact direct route when the design is
// eligible, otherwise by the LSMR of the backend that did the absorption
// (gpu_cuda), outside the certificate, refinement and retry machinery: a
// consistent system has no dual residual to certify, and the decision only
// needs the residual norm.
double group_constant_projection_residual_rel(
    const std::vector<Eigen::VectorXi>& standard_fes,
    const GroupIndividualStructure& gi,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    bool gpu_cuda);

bool certify_group_individual_candidate(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& standard_fes,
    const GroupIndividualStructure& group_individual,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    AbsorptionResult& result);

// The sweep absorber. It reads y and X where the caller keeps them (no
// temporary copy at the call) and works on its own copy (result.y_tilde,
// result.X_tilde); the inputs must not overlap the result and must not change
// during the call.
AbsorptionResult absorb_fixed_effects(const Eigen::Ref<const Eigen::VectorXd>& y,
                                      const Eigen::Ref<const Eigen::MatrixXd>& X,
                                      const std::vector<Eigen::VectorXi>& fes,
                                      const Eigen::VectorXd* weights,
                                      const HdfeOptions& options);

AbsorptionResult absorb_fixed_effects_v6(const Eigen::Ref<const Eigen::VectorXd>& y,
                                         const Eigen::Ref<const Eigen::MatrixXd>& X,
                                         const std::vector<Eigen::VectorXi>& fes,
                                         const Eigen::VectorXd* weights,
                                         const HdfeOptions& options,
                                         AbsorptionMethod method,
                                         const std::vector<HeterogeneousSlopeTerm>& slopes = {});

AbsorptionResult absorb_fixed_effects_group_individual(const Eigen::VectorXd& y,
                                                       const Eigen::MatrixXd& X,
                                                       const std::vector<Eigen::VectorXi>& standard_fes,
                                                       const GroupIndividualStructure& gi,
                                                       const Eigen::VectorXd* weights,
                                                       const HdfeOptions& options,
                                                       AbsorptionMethod method);

AbsorptionResult absorb_fixed_effects_krylov(const Eigen::VectorXd& y,
                                             const Eigen::MatrixXd& X,
                                             const std::vector<Eigen::VectorXi>& fes,
                                             const Eigen::VectorXd* weights,
                                             const HdfeOptions& options);

// Matrix-free LSMR / modified-LSMR (MLSMR) additive-Schwarz absorber.
// Standalone, CPU-only; reached by explicit absorptionmethod(lsmr|mlsmr),
// explicit auto-mlsmr, or the default auto selector on eligible standard-FE
// designs. Fails closed for savefe/retain-FE paths.
AbsorptionResult absorb_fixed_effects_mlsmr(const Eigen::VectorXd& y,
                                            const Eigen::MatrixXd& X,
                                            const std::vector<Eigen::VectorXi>& fes,
                                            const Eigen::VectorXd* weights,
                                            const HdfeOptions& options);

FeRecoveryResult recover_fixed_effects(const Eigen::VectorXd& partial,
                                       const std::vector<Eigen::VectorXi>& fes,
                                       const Eigen::VectorXd* weights,
                                       const HdfeOptions& options);

FeRecoveryResult recover_fixed_effects(const Eigen::VectorXd& partial,
                                       const std::vector<Eigen::VectorXi>& fes,
                                       const Eigen::VectorXd* weights,
                                       const HdfeOptions& options,
                                       double solver_fe_tolerance);

FeRecoveryResult recover_fixed_effects_group_ids(const Eigen::VectorXd& partial,
                                                 const std::vector<std::vector<int>>& fe_group_ids,
                                                 const std::vector<int>& fe_levels,
                                                 const Eigen::VectorXd* weights,
                                                 const HdfeOptions& options,
                                                 const std::vector<Eigen::VectorXd>* weight_sums_override = nullptr);

FeRecoveryResult recover_fixed_effects_group_ids(const Eigen::VectorXd& partial,
                                                 const std::vector<std::vector<int>>& fe_group_ids,
                                                 const std::vector<int>& fe_levels,
                                                 const Eigen::VectorXd* weights,
                                                 const HdfeOptions& options,
                                                 const std::vector<Eigen::VectorXd>* weight_sums_override,
                                                 double solver_fe_tolerance);

// Explicit coordinates for a correction in an already-normalized problem.
// Existing overloads retain automatic normalization and their ABI.
FeRecoveryResult recover_fixed_effects_group_ids(const Eigen::VectorXd& partial,
                                                 const std::vector<std::vector<int>>& fe_group_ids,
                                                 const std::vector<int>& fe_levels,
                                                 const Eigen::VectorXd* weights,
                                                 const HdfeOptions& options,
                                                 const std::vector<Eigen::VectorXd>* weight_sums_override,
                                                 double solver_fe_tolerance,
                                                 int coordinate_exponent);


double fe_recovery_max_delta(Eigen::VectorXd& residual,
                             const std::vector<std::vector<int>>& fe_group_ids,
                             const std::vector<int>& fe_levels,
                             const Eigen::VectorXd* weights,
                             const HdfeOptions& options,
                             const std::vector<Eigen::VectorXd>* weight_sums_override = nullptr);

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_FE_ABSORPTION_HPP
