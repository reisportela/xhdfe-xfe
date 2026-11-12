#ifndef HDFE_AKM_PROJECTION_HPP
#define HDFE_AKM_PROJECTION_HPP

#include <Eigen/Dense>
#include <vector>

namespace hdfe { namespace akm {

// Full-model, worker-cluster inference for the optional invariant point.
// Confidence intervals project one joint 95% contrast region. The delta
// standard errors apply only in the regular, first-order regime.
struct AkmProjectionResult {
    Eigen::VectorXd contrasts;
    Eigen::MatrixXd covariance;
    double se[2] = {0., 0.};
    double ci_lb[2] = {0., 0.};
    double ci_ub[2] = {0., 0.};
    double linearisation_index[2] = {0., 0.};
    double correction_over_se[2] = {0., 0.};
    int clusters = 0;
    int dimension = 0;
    int rank_diagnostic = 0;
    int max_cluster_rows = 0;
    int control_rank = 0;
    int polynomial_degree = 4;
    int threads_used = 1;
    int parallel_workers_active = 1;
    bool control_bread_reused = false;
    double critical_value = 0.;
    double projection_factor = 0.;
    double max_cluster_leverage_trace = 0.;
    double max_row_leverage = 0.;
    double covariance_condition = 0.;
    double max_trust_region_gap = 0.;
};

namespace projection {

// Internal interface. rows indexes the original y/X, while adjusted and the
// dense worker/firm ids refer to the retained sample. An empty control_bread
// requests a design-only fallback (e.g. zero residual variance in FWL).
AkmProjectionResult worker_projection_inference(
    const Eigen::VectorXd& y, const std::vector<int>& rows,
    const Eigen::VectorXd& adjusted, const Eigen::MatrixXd* controls,
    const std::vector<int>& control_columns, const Eigen::MatrixXd& control_bread,
    const std::vector<int>& worker, const std::vector<int>& firm,
    const Eigen::VectorXd& alpha, const Eigen::VectorXd& psi, int team);

} // namespace projection
}} // namespace hdfe::akm
#endif
