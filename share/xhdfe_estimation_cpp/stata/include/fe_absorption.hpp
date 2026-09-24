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
    double abs_residual = 0.0;      // verified ||D' W v_tilde||_2, max over RHS columns
    double abs_residual_rel = 0.0;  // max ||D' W v_tilde||_2 / (||D' W||_F ||v||_2)
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
    const GroupIndividualStructure* group_individual = nullptr);

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

AbsorptionResult absorb_fixed_effects(const Eigen::VectorXd& y,
                                      const Eigen::MatrixXd& X,
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
