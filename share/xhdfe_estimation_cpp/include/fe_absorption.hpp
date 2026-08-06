#ifndef HDFE_FE_ABSORPTION_HPP
#define HDFE_FE_ABSORPTION_HPP

#include <Eigen/Dense>

#include <vector>

#include "hdfe/hdfe_regressor.hpp"

namespace hdfe {
namespace detail {

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
    bool precision_certified = true;
    bool schwarz_used = false;  // true when the Schwarz/approx-Cholesky PCG path ran (forced or auto-gated)
    bool mlsmr_used = false;    // true when the MLSMR absorber ran via the auto-gate promotion
    bool gpu_used = false;
    int gpu_status_code = 0;  // 0 none, 1 used, 2 unavailable, 3 not converged, 4 failed, 5 CPU cache/profile
    bool gpu_attempted = false;
    bool gpu_absorption_converged = false;
    int gpu_absorption_iterations = 0;
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

FeRecoveryResult recover_fixed_effects_group_ids(const Eigen::VectorXd& partial,
                                                 const std::vector<std::vector<int>>& fe_group_ids,
                                                 const std::vector<int>& fe_levels,
                                                 const Eigen::VectorXd* weights,
                                                 const HdfeOptions& options,
                                                 const std::vector<Eigen::VectorXd>* weight_sums_override = nullptr);

double fe_recovery_max_delta(Eigen::VectorXd& residual,
                             const std::vector<std::vector<int>>& fe_group_ids,
                             const std::vector<int>& fe_levels,
                             const Eigen::VectorXd* weights,
                             const HdfeOptions& options,
                             const std::vector<Eigen::VectorXd>* weight_sums_override = nullptr);

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_FE_ABSORPTION_HPP
