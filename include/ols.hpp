#ifndef HDFE_OLS_HPP
#define HDFE_OLS_HPP

#include <Eigen/Dense>

#include <vector>

#include "hdfe/hdfe_regressor.hpp"

namespace hdfe {
namespace detail {

struct OlsResult {
    Eigen::VectorXd coefficients;
    Eigen::VectorXd stderr;
    Eigen::VectorXd tvalues;
    Eigen::VectorXd pvalues;
    Eigen::MatrixXd conf_int;
    Eigen::VectorXd residuals;
    Eigen::MatrixXd covariance;
    Eigen::MatrixXd xtx_inv;      // (X'X)^{-1} from the core solve (for post-processing).
    Eigen::VectorXd cluster_ux;   // Sum_g (sum_i w_i u_i) * (sum_i w_i x_i u_i) (cluster only).
    double cluster_u2 = 0.0;      // Sum_g (sum_i w_i u_i)^2 (cluster only).
    double cov_scale = 1.0;       // Finite-sample scale applied to `covariance`.
    double df_resid = 0.0;
    double rss = 0.0;
    double tss = 0.0;
    double within_tss = 0.0;
    double r2 = 0.0;
    double r2_within = 0.0;
    double sigma2 = 0.0;
    int nobs = 0;
    int num_clusters = 0;
};

// X_for_residuals: when set (2SLS), the coefficients are solved on X (the
// instrumented second-stage design) but residuals/rss/sigma2/R2 and the robust
// or clustered meat use u = y - X_for_residuals * beta (the actual regressors),
// as required for valid 2SLS inference.
OlsResult run_ols(const Eigen::VectorXd& y,
                  const Eigen::Ref<const Eigen::MatrixXd>& X,
                  const Eigen::VectorXd* weights,
                  const Eigen::VectorXi* clusters,
                  StandardErrorType se_type,
                  double total_sum_squares,
                  double within_sum_squares,
                  double n_effective = -1.0,
                  bool weights_are_frequencies = false,
                  const Eigen::MatrixXd* X_for_residuals = nullptr);

OlsResult run_ols_fast_from_xtx(const Eigen::VectorXd& y,
                                const Eigen::Ref<const Eigen::MatrixXd>& X,
                                const Eigen::VectorXd* weights,
                                StandardErrorType se_type,
                                double total_sum_squares,
                                double within_sum_squares,
                                const Eigen::MatrixXd& xtx,
                                const Eigen::VectorXd& xty,
                                double n_effective = -1.0,
                                bool weights_are_frequencies = false);

// Raw multiway-cluster sandwich bread * [inclusion-exclusion meat] * bread
// with NO df scaling of any kind (the caller maps its composite small-sample
// scale from an already-scaled slope block). `scores` is the unweighted score
// design (reghdfe restores the original means and appends the constant
// column); weights fold in as sqrt(w) on both scores and residuals.
Eigen::MatrixXd multiway_cluster_sandwich(
    const Eigen::MatrixXd& bread,
    const Eigen::Ref<const Eigen::MatrixXd>& scores,
    const Eigen::VectorXd& residuals,
    const Eigen::VectorXd* weights,
    const std::vector<Eigen::VectorXi>& clusters);

OlsResult run_ols_multiway(const Eigen::VectorXd& y,
                           const Eigen::Ref<const Eigen::MatrixXd>& X,
                           const Eigen::VectorXd* weights,
                           const std::vector<Eigen::VectorXi>* clusters,
                           StandardErrorType se_type,
                           double total_sum_squares,
                           double within_sum_squares,
                           ClusterDofMethod g_df = ClusterDofMethod::Conventional,
                           bool g_adj = true,
                           double n_effective = -1.0,
                           const Eigen::MatrixXd* X_for_residuals = nullptr);

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_OLS_HPP
