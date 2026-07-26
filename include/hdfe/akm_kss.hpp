#ifndef HDFE_AKM_KSS_HPP
#define HDFE_AKM_KSS_HPP

#include <Eigen/Dense>

#include <cstdint>
#include <string>
#include <vector>

namespace hdfe {
namespace akm {

// Leave-out unit for the KSS correction. Match (worker-firm pair) is the
// canonical default (Kline-Saggio-Soelvsten 2020 / Saggio's LeaveOutTwoWay):
// the data are collapsed to match means, the two-way system is solved as
// WLS with weight = match length (sqrt-weight FGLS transform, which equals
// person-year OLS), and the machinery runs leave-one-row-out on the
// collapsed rows. Observation leaves a single person-year row out.
enum class LeaveOutLevel { Match, Observation };

// How the row leverages P_ii and the quadratic-form weights B_ii are
// computed. Exact solves one linear system per unique row of the design;
// Jla uses the KSS Johnson-Lindenstrauss approximation with Rademacher
// draws and deterministic seeding. Auto mirrors LeaveOutTwoWay: exact when
// the input has <= exact_max_rows person-year rows, JLA otherwise.
enum class LeverageMethod { Auto, Exact, Jla };

struct AkmOptions {
    LeaveOutLevel leave_out_level = LeaveOutLevel::Match;
    LeverageMethod leverage_method = LeverageMethod::Auto;
    bool prune = true;                   //!< Compute the leave-out connected set. Set false only when the input is already a leave-out sample (e.g. cleaned by an oracle) to compare on identical rows.
    int jla_draws = 200;                 //!< Rademacher simulations for the JLA path (LeaveOutTwoWay default).
    std::uint64_t seed = 20260705;       //!< Seed for deterministic JLA streams; thread/backend reductions may differ at the last-ulp level.
    int exact_max_rows = 10000;          //!< Auto rule: exact leverages when the input has <= this many rows (LeaveOutTwoWay rule).
    int direct_max_firms = 50000;        //!< Direct sparse Cholesky of the firm Laplacian when #firms <= this, else matrix-free PCG.
    long long direct_max_nnz = 40000000; //!< Skip the direct path when the projected firm-Laplacian triplet count exceeds this.
    double cg_tol = 1e-10;               //!< Relative residual tolerance for the PCG solver.
    int cg_max_iter = 0;                 //!< 0 = auto (scales with #firms).
    int num_threads = 0;                 //!< 0 = library default (OpenMP max threads).
    double fwl_tol = 1e-10;              //!< Absorber tolerance for the covariate (FWL) step.
    int fwl_max_iter = 100000;           //!< Absorber iteration cap for the covariate step.
    bool use_gpu = false;                //!< Route the PCG solves (JLA draws, SE simulations, lincom, point estimate) to CUDA when available. Opt-in; CPU is the reference (GPU differs at fp-reassociation level, deterministic per device).
    bool compute_se = false;             //!< Standard errors for the three variance components (KSS high-rank case, leave_out_COMPLETE machinery).
    int se_nsim = 1000;                  //!< Simulations for the quadratic part of the SE (oracle NSIM; seeded here, unlike the oracle).
    int se_sigma_grid = 1000;            //!< KGrid for the binned sigma-tilde fit (llr_fit subsample mode 4).
    bool se_sigma_lowess = false;        //!< Use the llr_fit mode-0 lowess surface fit of sigma_i on (Pii, Bii) instead of the binned mode-4 fit. Faithful port of the MATLAB 'lowess' surface (k-NN tricube local linear, normalized predictors, span = NT^(-1/3), NaN -> sigma_i). O(n^2) k-NN: intended for small/medium samples; the binned mode stays the validated default.
    bool eigen_diagnostics = false;      //!< With compute_se: top-eigenvalue diagnostics and the Andrews-Mikusheva q=1 confidence interval (leave_out_COMPLETE eigen_diagno path).
    int eig_trace_nsim = 100;            //!< Hutchinson draws for tr(Atilde^2) (oracle default).
    double ci_level_unused = 0.0;        //!< Reserved (the oracle tabulation is for its fixed level).
    int verbose = 0;                     //!< 0 = silent (default). 1 = phase announcements plus throttled intra-phase progress with elapsed time and an ETA on the long loops (JLA leverage draws, SE simulations). Output only — never changes any numeric result.
    void (*progress)(const char* line, void* user) = nullptr;  //!< Optional sink for the verbose lines (one line per call, no trailing newline). When null, lines go to stderr. Called only from the calling thread, never from inside a parallel region — a Stata plugin can route it to SF_display.
    void* progress_user = nullptr;       //!< Opaque pointer handed back to `progress`.
};

// Options for the standalone leave-out connected-set utility.  The graph
// semantics are identical to the sample-building phase of akm_kss_decompose;
// these switches control execution and progress reporting only.
struct LeaveOutSetOptions {
    int num_threads = 0;                 //!< 0 = preserve the caller/library default.
    bool use_gpu = false;                //!< Use CUDA for stable match-key sorting when profitable/available; graph pruning remains CPU.
    int verbose = 0;                     //!< 0 = silent; 1 = phase progress. Output only.
    void (*progress)(const char* line, void* user) = nullptr;
    void* progress_user = nullptr;
};

// Leave-out connected-set result. keep refers to the ORIGINAL input rows.
struct LeaveOutSetResult {
    std::vector<std::uint8_t> keep;  //!< 1 = row belongs to the leave-out connected sample.
    long long n_obs_input = 0;
    long long n_obs_connected = 0;   //!< Rows in the largest connected set (before leave-out pruning).
    long long n_obs = 0;             //!< Rows in the leave-out sample.
    int n_workers = 0;
    int n_firms = 0;
    long long n_matches = 0;
    int n_movers = 0;                //!< Workers with >= 2 distinct firms in the leave-out sample.
    int n_stayers = 0;               //!< Workers with a single firm in the leave-out sample.
    int prune_iterations = 0;
    int threads_used = 1;          //!< OpenMP budget used by sample construction.
    bool gpu_used = false;         //!< CUDA stable sort actually executed.
    int gpu_status_code = 0;       //!< 0 not requested, 1 used, 2 unavailable, 4 failed, 6 below profitability gate.
};

// One variance-decomposition column: the three canonical AKM components.
struct AkmComponents {
    double var_alpha = 0.0;
    double var_psi = 0.0;
    double cov_alpha_psi = 0.0;
};

struct AkmKssResult {
    LeaveOutSetResult sample;

    // Observation-level effects on the kept original rows (input order).
    // psi is centered to a zero person-year mean over the estimation sample
    // (alpha absorbs the shift); the variance components are invariant to
    // this normalization.
    Eigen::VectorXd alpha;
    Eigen::VectorXd psi;
    Eigen::VectorXd beta;            //!< Control coefficients (empty when no controls).

    // Variance decomposition, person-year weighted with 1/(n_py - 1)
    // (LeaveOutTwoWay convention): plug-in (biased baseline), AGSU
    // homoskedastic correction (pytwoway 'ho'), KSS heteroskedastic
    // leave-out correction.
    AkmComponents plugin;
    AkmComponents agsu;
    AkmComponents kss;
    double var_y = 0.0;              //!< Person-year variance of y on the leave-out sample (before residualizing controls).
    double sigma2_ho = 0.0;          //!< Homoskedastic sigma^2 = RSS_py / (n_py - (N + J - 1)) used by AGSU.

    // Working-row-level arrays (matches when collapsed, else observations):
    // statistical leverages, KSS sigma_i, and the row keys as original ids.
    Eigen::VectorXd pii;
    Eigen::VectorXd sigma_i;
    Eigen::VectorXi row_worker;      //!< Original worker id per working row.
    Eigen::VectorXi row_firm;        //!< Original firm id per working row.
    Eigen::VectorXd row_weight;      //!< Person-year weight per working row (match length; 1 at observation level).

    // KSS lincom (Proposition 1 / Remark 9): projection of the firm effects
    // on [1, Z] at the person-year level, with leave-out (KSS) and naive
    // White standard errors. Filled when Z is supplied to akm_kss_decompose;
    // coefficients exclude the constant (LeaveOutTwoWay convention).
    Eigen::VectorXd lincom_coef;
    Eigen::VectorXd lincom_se_kss;
    Eigen::VectorXd lincom_se_white;
    Eigen::VectorXd lincom_t;

    // Component standard errors (KSS high-rank/normal case; the
    // leave_out_COMPLETE estimator: V = (4*sum W_i^2 sigma~_i^2 - Var_sim)
    // / n_py^2, computed on the person-year block-leave-out representation).
    // theta_c_* are the leave_out_COMPLETE-convention point estimates the
    // SEs are centered on (they differ from the kss components by O(1/n):
    // uncentered y, 1/n normalization, oracle stayer conventions).
    // se_var_alpha / theta_c_var_alpha are NaN at match level: the canonical
    // leave_out_COMPLETE oracle reports only the firm and covariance
    // components there.  Use observation level for var(alpha) inference.
    double se_var_psi = 0.0;
    double se_cov_alpha_psi = 0.0;
    double se_var_alpha = 0.0;
    double theta_c_var_psi = 0.0;
    double theta_c_cov_alpha_psi = 0.0;
    double theta_c_var_alpha = 0.0;

    // Weak-identification diagnostics and Andrews-Mikusheva q=1 confidence
    // intervals (eigen_diagnostics = true; leave_out_COMPLETE conventions).
    // Arrays are indexed fe(0), cov(1), pe(2); pe entries are NaN at match
    // level under the leave_out_COMPLETE oracle rule.
    double eig_lambda1[3] = {0, 0, 0};        //!< Top eigenvalue of Atilde (unnormalized).
    double eig_share1[3] = {0, 0, 0};         //!< lambda_1^2 / tr(Atilde^2).
    double eig_share2[3] = {0, 0, 0};
    double eig_share3[3] = {0, 0, 0};
    double lindeberg_max_x1bar_sq[3] = {0, 0, 0};
    double gamma_sq[3] = {0, 0, 0};
    double f_stat[3] = {0, 0, 0};
    double theta_1[3] = {0, 0, 0};            //!< Curvature-adjusted point estimate.
    double ci_lb[3] = {0, 0, 0};              //!< AM q=1 confidence bound (lower).
    double ci_ub[3] = {0, 0, 0};
    double curvature[3] = {0, 0, 0};
    double b_1[3] = {0, 0, 0};                //!< b_1 = sum(x1bar .* y).
    double cov_r1_11[3] = {0, 0, 0};          //!< Sigma_1 entries (COV_R1).
    double cov_r1_12[3] = {0, 0, 0};
    double cov_r1_22[3] = {0, 0, 0};

    // Diagnostics.
    double max_pii = 0.0;            //!< Max leverage over mover rows (stayer match rows have Pii = 1 by construction).
    double mean_pii = 0.0;           //!< Mean leverage over mover rows.
    long long n_rows = 0;            //!< Working rows (matches when collapsed).
    bool leverages_exact = true;     //!< Exact vs JLA path actually used.
    bool gpu_used = false;           //!< CUDA solver actually used for the PCG solves.
    bool solver_direct = true;       //!< Direct Cholesky vs PCG actually used.
    int fwl_threads_used = 0;        //!< Effective absorber threads for controls (0 when no controls).
    int threads_used = 1;            //!< Effective OpenMP team for the two-way KSS solver.
    int jla_draws_used = 0;
    std::uint64_t seed_used = 0;
    long long solver_iterations = 0; //!< Total PCG iterations across all solves (0 when fully direct).
    bool converged = true;
    std::string notes;
};

// Largest leave-out connected set, matching LeaveOutTwoWay: largest
// connected set, then iteratively remove workers that are articulation
// points of the mover-firm bipartite graph and retake the largest connected
// component, and finally drop workers observed only once. The same sample
// serves both leave-out levels (the level changes the collapse and sigma_i,
// not the pruning).
LeaveOutSetResult leave_out_connected_set(const Eigen::VectorXi& worker_ids,
                                           const Eigen::VectorXi& firm_ids,
                                           const Eigen::VectorXd* fweights = nullptr,
                                           const LeaveOutSetOptions& options = LeaveOutSetOptions{});

// Full AKM + leave-out variance decomposition on the leave-out connected set.
// X may be nullptr (no controls); controls are partialled out at the
// person-year level using the existing xhdfe absorber (FWL) before the
// two-way machinery runs, as in LeaveOutTwoWay.
// Z (optional, original rows): covariates for the KSS lincom projection of
// the firm effects (a constant is added internally).
AkmKssResult akm_kss_decompose(const Eigen::VectorXd& y,
                               const Eigen::VectorXi& worker_ids,
                               const Eigen::VectorXi& firm_ids,
                               const Eigen::MatrixXd* X,
                               const AkmOptions& options,
                               const Eigen::MatrixXd* Z = nullptr,
                               const Eigen::VectorXd* fweights = nullptr);
// fweights: optional per-row positive-integer frequency weights (row i stands
// for fweights[i] identical person-year observations). Supported at match
// level for the point decomposition (plug-in/AGSU/KSS, exact and JLA
// leverages, controls); equals the row-expanded run (JLA streams included,
// which are keyed per match and per person-year count). Not yet available
// with leave_out_level=obs, compute_se/eigen_diagnostics, or Z (lincom):
// expand the data for those (identical results by construction).

}  // namespace akm

namespace gelbach {

// Gelbach (2016) conditional decomposition, HDFE-aware (M9B). One compiled
// implementation behind the Stata, Python and R front-ends; inference
// reproduces Gelbach's b1x2 exactly (homoskedastic, robust, cluster, with
// the gamma0/cov0 options). Absorbed FE blocks always receive the gamma0
// (aux-regression-only) variance treatment.
enum class GelbachVce { Unadjusted, Robust, Cluster };

struct GelbachOptions {
    GelbachVce vce = GelbachVce::Unadjusted;
    bool gamma0 = false;
    bool cov0 = false;
    bool use_gpu = false;                 //!< Request CUDA for the full-model absorption phase; CPU remains the default/reference.
    bool capture_sample_provenance = false; //!< Compute a stable retained-sample identifier. Opt-in so the default path pays no O(n) provenance cost.
    bool return_sample_index = false;      //!< Return zero-based retained input-row positions. Implies sample provenance.
    // Zero-based X1 columns that the caller explicitly permits the full
    // model to absorb. Every declared column must be classified by the HDFE
    // fit as collinear with the absorbed FEs (omitted_reason == 1); all other
    // X1/X2 columns remain subject to the standard fail-hard rank guard.
    std::vector<int> absorbed_x1;
    // Optional zero-based pair of added FE dimensions for the retained-sample
    // mobility diagnostic. Indices exclude the common-FE prefix used by the
    // extended overload. Empty preserves the historical first-two-added
    // default. This is diagnostic only with common FEs or 3+ added FEs; it is
    // not a global rank certificate.
    std::vector<int> connectivity_fe_pair;
    // Fail closed unless the per-FE-dimension X1-row split is certified.
    // At present that certificate exists only with no common FEs and exactly
    // two added FE dimensions forming one retained-sample mobility component.
    bool require_connected_fe_split = false;
    double tol = 1e-8;
    int num_threads = 0;
    int verbose = 0;                     //!< 0 = silent; 1 = phase progress. Output only.
    void (*progress)(const char* line, void* user) = nullptr;
    void* progress_user = nullptr;
};

struct GelbachResult {
    Eigen::VectorXd b_base;   //!< Base-specification coefficients on X1.
    Eigen::VectorXd b_full;   //!< Full coefficients on X1; declared absorbed targets are constrained to zero.
    Eigen::VectorXi x1_absorbed; //!< 1 where b_full is imposed zero because X1 is absorbed; 0 where estimated.
    Eigen::VectorXd x1_fe_collinear_ratio; //!< Per-X1 ||M_D x||^2 / ||x||^2 from the full-fit classifier.
    Eigen::VectorXi x1_near_collinear_mask; //!< 1 where the ratio is in the documented warning band above the omission boundary.
    Eigen::MatrixXd gamma;        //!< Full-model X2 coefficients, padded by block: rows=max block width, cols=observed blocks.
    Eigen::VectorXd beta2;        //!< Full-model X2 coefficients in original column order.
    Eigen::MatrixXd beta2_cov;    //!< Requested-VCE covariance of beta2 in original column order.
    Eigen::MatrixXd auxiliary_loadings; //!< True auxiliary loadings Gamma: rows=[x1..., _cons], cols=X2 in original order.
    Eigen::VectorXd auxiliary_loading_ss_ratio; //!< Per observed block, weighted fitted/raw sum-of-squares ratio for Gamma.
    Eigen::VectorXi auxiliary_loading_rank; //!< Numerical rank of each observed block's Gamma matrix.
    Eigen::VectorXd auxiliary_loading_condition_number; //!< Nonzero-singular-value condition number of each block's Gamma.
    Eigen::MatrixXd auxiliary_loading_max_abs_z; //!< Rows=[x1..., _cons], cols=observed blocks; largest marginal |z| for each Gamma row.
    Eigen::MatrixXd auxiliary_loading_pvalue; //!< Rows=[x1..., _cons], cols=observed blocks; Bonferroni-adjusted rowwise Gamma=0 p-value.
    Eigen::MatrixXi auxiliary_loading_test_evaluated; //!< 1 when the rowwise Gamma test was needed and evaluated; 0 otherwise.
    Eigen::VectorXd beta2_wald_stat; //!< Requested-VCE joint Wald statistic for beta2_g=0.
    Eigen::VectorXi beta2_wald_df; //!< Numerical covariance rank used by each beta2 Wald test.
    Eigen::VectorXd beta2_wald_pvalue; //!< Chi-square reference p-value for each beta2 Wald test.
    Eigen::MatrixXd contribution_gradient_norm; //!< Rows=[x1..., _cons], cols=observed blocks; norm of the product gradient.
    Eigen::MatrixXi regular_inference_valid; //!< Rows=[x1..., _cons], cols=observed blocks; 1 only when beta2_g!=0 or the corresponding Gamma row!=0 is statistically established.
    std::vector<std::string> regular_inference_status; //!< Column-major status for each observed contribution: regular_beta_nonzero, regular_loading_nonzero, nonregular_not_ruled_out, not_certified, or not_applicable_common_fe_intercept.
    bool regular_inference_all_valid = true; //!< True iff every observed-X2 contribution passes the conservative regularity gate.
    double regularity_test_alpha = 0.05; //!< Family-wise threshold used by the diagnostic gate.
    Eigen::MatrixXd delta;    //!< (p+1) x G contributions over [x1..., _cons]; group order = x2 groups then FE dims.
    Eigen::MatrixXd cov;      //!< (G*(p+1)) x (G*(p+1)) covariance of vec(delta).
    Eigen::MatrixXd base_cov; //!< Requested-VCE covariance of [b_base, base intercept].
    Eigen::MatrixXd cov_delta_bbase; //!< Cov(vec(delta), [b_base, base intercept]).
    Eigen::MatrixXd cov_total_bbase; //!< Cov(sum_g delta_g, [b_base, base intercept]).
    Eigen::VectorXd total;    //!< Summed contribution (= b_base - b_full over [x1, _cons]).
    Eigen::MatrixXd total_cov;
    int n_common_fes = 0;      //!< FE dimensions absorbed in both base and full specifications.
    bool common_fes_applied = false; //!< True when the conditional common-FE estimand is active.
    bool intercept_inference_available = true; //!< False with common FEs because the intercept allocation is normalization-dependent.
    std::string intercept_status = "estimated_no_common_fes"; //!< estimated_no_common_fes or not_certified_common_fes.
    double identity_gap = 0.0;
    long long n_obs_input = 0;
    long long n_obs = 0;       //!< Retained row count (historical public field).
    long long n_obs_effective = 0; //!< Retained rows normally; sum of retained fweights under frequency weighting.
    long long n_singletons_dropped = 0;
    Eigen::VectorXi sample_index; //!< Optional zero-based retained positions into the caller's input arrays; empty unless requested.
    std::string sample_hash; //!< Optional non-cryptographic identifier of n_obs_input plus sample_index.
    std::string sample_hash_algorithm; //!< fnv1a64-le-v1 when provenance was requested; empty otherwise.
    std::string sample_index_scope; //!< input_rows_zero_based when provenance was requested; empty otherwise.
    int n_mobility_components = 0; //!< Exact selected-pair added-FE components in the retained sample; 0 unless at least two added FEs.
    long long largest_mobility_component_n_obs = 0; //!< Physical rows in the row-largest selected-pair component.
    double largest_mobility_component_share = 0.0; //!< Retained-row share of the row-largest selected-pair component.
    double largest_mobility_component_weight_share = 0.0; //!< Retained-weight share of the weight-largest selected-pair component.
    bool fe_split_identified = false; //!< Whether per-FE-dimension X1-row contributions are certified normalization-invariant.
    std::string fe_split_status = "not_applicable"; //!< not_applicable, single_fe_dimension, identified_two_way, normalization_dependent, not_certified_multiway, or not_certified_with_common_fes.
    int connectivity_fe_index1 = -1; //!< First zero-based added-FE index used by the pair diagnostic; -1 when not applicable.
    int connectivity_fe_index2 = -1; //!< Second zero-based added-FE index used by the pair diagnostic; -1 when not applicable.
    bool connectivity_pair_explicit = false; //!< True when the caller selected the diagnostic pair.
    std::string connectivity_pair_status = "not_applicable"; //!< not_applicable, connected, or disconnected.
    std::string connected_mode = "diagnose"; //!< diagnose (default) or require.
    std::string mobility_component_scope = "not_applicable"; //!< first_two_fe_dimensions/selected_fe_pair without common FEs; first_two_added_fe_dimensions/selected_added_fe_pair with common FEs.
    double df_full = 0.0;
    double df_base = 0.0;
    int n_clusters = 0;          //!< Independent retained-sample clusters for cluster VCE; 0 otherwise.
    double fe_collinear_ss_ratio_tol = 1e-9; //!< Backend FE-absorption classification boundary: ||M_D x||^2 / ||x||^2.
    double near_fe_collinear_ss_ratio_warn_upper = 1e-4; //!< Inclusive upper edge of the diagnostic-only warning band.
    int few_cluster_warning_threshold = 30; //!< Warn when one-way cluster count is below this value.
    bool absorbed_target_inference_valid = true; //!< True unless absorbed-target inference is requested away from an absorbing FE cluster.
    int absorbing_fe_index = -1; //!< Zero-based added-FE dimension matching the cluster and absorbing every target; -1 if unavailable/not applicable.
    bool converged = true;
    int threads_used = 1;
    bool gpu_requested = false;
    bool gpu_used = false;
    int gpu_status_code = 0;
    std::string gpu_backend = "cpu";
    std::string gpu_status = "not_requested";
    bool gpu_attempted = false;
    bool gpu_absorption_converged = false;
    int gpu_absorption_iterations = 0;
    std::string notes;
};

// X2 holds the observed covariate groups side by side; x2_group_sizes gives
// the column count of each observed group (in order). In the extended
// overload, the first n_common_fes entries of fes are absorbed in both base
// and full and are not reported as contributions; the remaining entries are
// added only to the full model and form decomposable blocks after the observed
// groups. With common FEs, slope rows receive the full point/inference
// contract while the intercept row is deliberately not certified because its
// allocation depends on FE normalization. cluster may be null unless
// vce == Cluster.
// weights (optional): Stata-style aweights (freq_weights = false; normalized
// to sum to N internally) or fweights (freq_weights = true), matching b1x2's
// weighted estimators exactly.
GelbachResult decompose(const Eigen::VectorXd& y,
                        const Eigen::MatrixXd& X1,
                        const Eigen::MatrixXd& X2,
                        const std::vector<int>& x2_group_sizes,
                        const std::vector<Eigen::VectorXi>& fes,
                        int n_common_fes,
                        const Eigen::VectorXi* cluster,
                        const GelbachOptions& options,
                        const Eigen::VectorXd* weights = nullptr,
                        bool freq_weights = false);

// Backward-compatible lane: every FE is added only to the full model.
GelbachResult decompose(const Eigen::VectorXd& y,
                        const Eigen::MatrixXd& X1,
                        const Eigen::MatrixXd& X2,
                        const std::vector<int>& x2_group_sizes,
                        const std::vector<Eigen::VectorXi>& fes,
                        const Eigen::VectorXi* cluster,
                        const GelbachOptions& options,
                        const Eigen::VectorXd* weights = nullptr,
                        bool freq_weights = false);

}  // namespace gelbach
}  // namespace hdfe

#endif  // HDFE_AKM_KSS_HPP
