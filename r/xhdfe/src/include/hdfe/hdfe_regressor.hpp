#ifndef HDFE_REGRESSOR_HPP
#define HDFE_REGRESSOR_HPP

#include <Eigen/Dense>

#include <vector>

#ifdef _WIN32
#ifdef stderr
#undef stderr
#endif
#endif

namespace hdfe {

/** \brief Supported covariance estimators for coefficient inference. */
enum class StandardErrorType { Homoskedastic, Robust, Cluster };

/** \brief Algorithmic strategy used for the fixed-effect absorber. */
enum class AbsorptionMethod { Auto, GaussSeidel, SymmetricGaussSeidel, Jacobi, Schwarz, Lsmr, Mlsmr, AutoMlsmr };

/** \brief Numerical stopping rule used by the alternating-projection absorber. */
enum class ConvergenceCriterion { Auto, NormChange, Reghdfe, Both };

/** \brief Algorithmic strategy used to recover fixed effects when saving them. */
enum class FeRecoveryMethod { Map, Hybrid };

/** \brief Degree-of-freedom adjustment strategy for absorbed fixed effects (reghdfe-style). */
enum class DofAdjustmentMethod { All, None, FirstPair, Pairwise };

/** \brief Fixed-effect DoF counting strategy for SSC (fixest-style). */
enum class FixefDofMethod { Full, None, Nonnested };

/** \brief Multiway cluster DoF strategy for SSC (fixest-style). */
enum class ClusterDofMethod { Min, Conventional };

/** \brief Statistics style for small-sample corrections and fit statistics. */
enum class StatsStyle { Reghdfe, Legacy };

/** \brief Absorber convergence mode. */
enum class ToleranceMode { XhdfeFast, ReghdfeComparable, StrictResidual };

/**
 * \brief Configuration parameters controlling numerical tolerances and options for HDFE estimation.
 */
struct HdfeOptions {
    StandardErrorType se_type = StandardErrorType::Robust; //!< Covariance estimator to use after estimation.
    double level = 95.0;                                   //!< Confidence level (percent) used for confidence intervals.
    bool fit_intercept = true;                             //!< Whether to append an intercept column to X automatically.
    double tol = 1e-8;                                     //!< Convergence tolerance for the fixed-effect absorber.
    ToleranceMode tolerance_mode = ToleranceMode::ReghdfeComparable; //!< Absorber stopping semantics.
                                                           //!< Default (since 2.7.0): reghdfe-comparable stops when one
                                                           //!< full sweep moves the working data by less than tol in
                                                           //!< relative norm (reghdfe's meaning of tolerance).
                                                           //!< XhdfeFast restores the pre-2.7.0 fast trigger;
                                                           //!< StrictResidual is the audited certificate mode.
    double fe_tolerance = 1e-6;                            //!< Convergence tolerance for fixed-effect recovery (savefes).
    FeRecoveryMethod fe_recovery_method = FeRecoveryMethod::Hybrid; //!< Fixed-effect recovery strategy.
    int max_iter = 100000;                                 //!< Maximum alternating-projection iterations.
    int convergence_check_interval = 1;                    //!< Check convergence every k absorption iterations (1 = every iteration).
    ConvergenceCriterion convergence_criterion = ConvergenceCriterion::Auto; //!< Heterogeneous-slope absorber stopping rule.
                                                           //!< Auto (default) follows tolerance_mode: reghdfe-comparable maps
                                                           //!< to the reghdfe-style update criterion, xhdfe-fast to normchange.
                                                           //!< Explicit values override; standard (slope-free) absorption is
                                                           //!< governed by tolerance_mode alone.
    int num_threads = 0;                                   //!< Threads for absorption (0 = library default / auto).
    bool drop_singletons = true;                           //!< When true, iteratively drop singleton observations before estimation (reghdfe default).
    bool weights_are_frequencies = false;                  //!< Interpret weights as frequency weights for N/df bookkeeping.
    bool retain_fixed_effects = false;                     //!< When true, recover and cache observation-level fixed effects.
    bool refine_stored_residuals = false;                  //!< When true, recompute stored residuals via a final absorption of y-Xb.
    int savefe_fastpath_max_cols = 8;                      //!< Max X columns to enable savefe alpha accumulation during absorption.
    bool symmetric_sweep = false;                          //!< When true, apply a forward and backward sweep per iteration for faster convergence.
    AbsorptionMethod absorption_method = AbsorptionMethod::Auto; //!< Absorption strategy selector (auto picks based on FE count and threads).
    bool from_auto = false;                                //!< Internal: set by the regressor when absorption_method was left Auto, so the adaptive Schwarz gate can fire after Auto is resolved to a concrete MAP method. Not user-facing.
    double jacobi_relaxation = 0.0;                         //!< Optional relaxation factor for Jacobi (<=0 uses default 2/(J+1)).
    bool use_krylov = false;                               //!< When true, use Krylov-PCG partialling-out (v7+ only).
    double krylov_lambda = 1e-6;                           //!< Ridge regularization for Krylov-PCG (v7+ only).
    int krylov_probe_iters = 0;                            //!< Internal: when >0, the Jacobi-PCG absorber caps the leading (y) solve at this many iterations and returns converged=false if it does not converge, so the adaptive Schwarz gate can fall back to the approx-Cholesky path on graphs where matrix-free PCG stalls. 0 = solve fully (no probe). Not user-facing.
    bool use_sparse_solver = false;                        //!< When true, attempt sparse direct/PCG absorption before MAP.
    double sparse_threshold = 0.0;                         //!< Max total FE levels to enable sparse absorption (0 = heuristic).
    DofAdjustmentMethod dof_method = DofAdjustmentMethod::All; //!< Degrees-of-freedom adjustment method (reghdfe default is all).
    bool dof_adjust_clusters = true;                       //!< Apply cluster nesting DoF adjustments when clustering on absorbed variables.
    bool dof_adjust_continuous = true;                     //!< Placeholder for reghdfe continuous-interaction DoF checks (no-op in this implementation).
    bool save_groupvar = false;                            //!< When true, compute and store the first mobility group variable (reghdfe groupvar()).
    std::vector<int> sweep_order_override;                 //!< Optional sweep order override (0-based FE indices); empty = auto.
    bool ssc_k_adj = true;                                 //!< Apply (N-1)/(N-K) small-sample adjustment (fixest K.adj).
    FixefDofMethod ssc_k_fixef = FixefDofMethod::Full;      //!< How to count fixed effects in K (fixest K.fixef).
    bool ssc_k_exact = true;                               //!< Use exact FE rank when counting K (fixest K.exact).
    bool ssc_g_adj = true;                                 //!< Apply G/(G-1) cluster adjustment (fixest G.adj).
    ClusterDofMethod ssc_g_df = ClusterDofMethod::Min;      //!< Multiway cluster DoF (fixest G.df).
    double ssc_t_df = -1.0;                                //!< Override t df (fixest t.df); <=0 disables.
    StatsStyle stats_style = StatsStyle::Reghdfe;          //!< Fit-statistics style (reghdfe default).
    std::vector<int> collinear_priority;                   //!< Column priority when dropping collinear regressors (higher = keep).
};

/**
 * \brief Dense summary of estimation results and auxiliary diagnostics.
 */
struct HdfeResults {
    Eigen::VectorXd coefficients;  //!< Estimated regression coefficients in the order passed to X (including intercept when active).
    Eigen::VectorXd stderr;        //!< Standard errors corresponding to each coefficient.
    Eigen::VectorXd tvalues;       //!< t-statistics computed from coefficients and the chosen covariance estimator.
    Eigen::VectorXd pvalues;       //!< Two-sided p-values based on the normal approximation.
    Eigen::MatrixXd conf_int;      //!< Column-wise confidence intervals (p x 2) for the requested confidence level (default 95%).
    Eigen::MatrixXd covariance;    //!< Variance-covariance matrix corresponding to the stored coefficients.
    Eigen::VectorXd residuals;     //!< Regression residuals after absorbing fixed effects.
    std::vector<int> omitted_reason; //!< Omitted-regressor reason codes (0=kept, 1=collinear with FEs, 2=other collinearity).
    int nobs = 0;                  //!< Number of observations used in estimation.
    int nobs_full = 0;             //!< Number of observations prior to singleton dropping (if enabled).
    int num_singletons = 0;        //!< Number of singleton observations dropped prior to estimation.
    double nobs_effective = 0.0;   //!< Effective N used for df/statistics (row count unless frequency weights are active).
    double nobs_full_effective = 0.0; //!< Effective full N prior to singleton dropping.
    double num_singletons_effective = 0.0; //!< Effective singleton count (nobs_full_effective - nobs_effective).
    Eigen::VectorXi sample_index;  //!< 0-based indices of kept observations into the input arrays (length nobs).
    double df_resid = 0.0;         //!< Residual degrees of freedom.
    double df_resid_unadj = 0.0;   //!< Unclustered residual degrees of freedom (before cluster df adjustment).
    double df_m = 0.0;             //!< Model degrees of freedom excluding the intercept.
    double df_a = 0.0;             //!< Degrees of freedom absorbed by fixed effects (reghdfe-style).
    double df_a_levels = 0.0;      //!< Raw FE level count (sum of levels across dimensions).
    double df_a_exact = 0.0;       //!< Exact FE DoF after redundancy (sum of non-redundant levels).
    double df_a_nested = 0.0;      //!< FE DoF nested within clusters (used in SSC adjustments).
    double r2 = 0.0;               //!< Overall (between) R-squared.
    double r2_within = 0.0;        //!< Within R-squared measured on transformed data.
    double sigma2 = 0.0;           //!< Estimated variance of residuals under the selected covariance model.
    double rss = 0.0;              //!< Residual sum of squares.
    double tss = 0.0;              //!< Total sum of squares.
    double tss_within = 0.0;       //!< Within total sum of squares.
    std::vector<int> fe_num_levels; //!< Number of levels in each fixed-effect dimension.
    std::vector<int> fe_base_levels; //!< Distinct FE levels present in the estimation sample per dimension (before any heterogeneous-slope DoF expansion).
    std::vector<int> fe_redundant;  //!< Redundant FE levels (M) per dimension (reghdfe-style).
    std::vector<int> fe_num_coefs;  //!< Non-redundant FE levels (K-M) per dimension.
    std::vector<int> fe_inexact;    //!< Flags for inexact redundancy counts per FE (1=inexact).
    std::vector<int> fe_nested;     //!< Flags for FEs nested within clusters (1=nested).
    int num_iterations = 0;        //!< Number of absorption iterations performed.
    Eigen::VectorXi groupvar;      //!< First mobility group (connected component) identifier, when requested via groupvar().
    std::vector<Eigen::VectorXd> fe_effects; //!< Observation-level contribution of each FE dimension (one vector per FE).
    std::vector<Eigen::VectorXd> fe_save_effects; //!< Variables to write for savefe/savefes (includes heterogeneous slope coefficients).
    int fe_recovery_iterations = 0; //!< Number of sweeps used to recover fixed effects when retained.
    double fe_recovery_max_delta = 0.0; //!< Maximum change observed in the final FE recovery sweep.
    bool fe_recovery_converged = true; //!< Whether the FE recovery loop converged within the iteration cap.
    bool converged = true;         //!< Flag indicating whether the absorber reached tolerance.
    int num_clusters = 0;          //!< Minimum number of clusters across dimensions (if clustered).
    std::vector<int> cluster_counts; //!< Number of clusters for each dimension (if clustered).
    std::vector<int> cluster_combo_counts; //!< Cluster counts for each multiway combination (optional).
    double cluster_scale = 1.0;    //!< Final scale applied to the cluster covariance matrix.
    bool vcv_psd_fixed = false;    //!< Multiway-cluster VCV needed the Cameron-Gelbach-Miller PSD adjustment.

    //! Saturated / perfect-fit design: the absorbed FEs plus the regressors span
    //! the data, so the within residuals are ~0 (within R-sq = 1). The point
    //! estimates are exact, but the error variance — and hence the standard
    //! errors — is not identified. True only with estimated regressors
    //! (df_m > 0) and a residual sum of squares negligible relative to the
    //! within total. Purely derived from already-computed quantities, so it adds
    //! no work to estimation. Lets a consumer distinguish a legitimate zero SE
    //! (saturated, e.g. synthetic-zigzag) from a zero/garbage SE produced by a
    //! failed or unstable solve (where rss is not ~0).
    bool is_saturated() const {
        return df_m > 0.0 && tss_within > 0.0 && rss <= 1.0e-9 * tss_within;
    }
};

}  // namespace hdfe

#endif  // HDFE_REGRESSOR_HPP
