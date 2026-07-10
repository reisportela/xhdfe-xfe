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
    // se_var_alpha / theta_c_var_alpha are NaN at match level when stayers
    // are present (not identified; oracle rule).
    double se_var_psi = 0.0;
    double se_cov_alpha_psi = 0.0;
    double se_var_alpha = 0.0;
    double theta_c_var_psi = 0.0;
    double theta_c_cov_alpha_psi = 0.0;
    double theta_c_var_alpha = 0.0;

    // Weak-identification diagnostics and Andrews-Mikusheva q=1 confidence
    // intervals (eigen_diagnostics = true; leave_out_COMPLETE conventions).
    // Arrays are indexed fe(0), cov(1), pe(2); pe entries are NaN under the
    // match-level-with-stayers rule.
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
                                          const Eigen::VectorXd* fweights = nullptr);

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
    int num_threads = 0;
};

struct GelbachResult {
    Eigen::VectorXd b_base;   //!< Base-specification coefficients on X1.
    Eigen::VectorXd b_full;   //!< Full-specification coefficients on X1.
    Eigen::MatrixXd delta;    //!< (p+1) x G contributions over [x1..., _cons]; group order = x2 groups then FE dims.
    Eigen::MatrixXd cov;      //!< (G*(p+1)) x (G*(p+1)) covariance of vec(delta).
    Eigen::VectorXd total;    //!< Summed contribution (= b_base - b_full over [x1, _cons]).
    Eigen::MatrixXd total_cov;
    double identity_gap = 0.0;
    long long n_obs = 0;
    double df_full = 0.0;
    bool converged = true;
    std::string notes;
};

// X2 holds the observed covariate groups side by side; x2_group_sizes gives
// the column count of each observed group (in order). Each entry of fes is
// one absorbed FE dimension forming its own decomposable block, ordered
// after the observed groups. cluster may be null unless vce == Cluster.
// weights (optional): Stata-style aweights (freq_weights = false; normalized
// to sum to N internally) or fweights (freq_weights = true), matching b1x2's
// weighted estimators exactly.
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
