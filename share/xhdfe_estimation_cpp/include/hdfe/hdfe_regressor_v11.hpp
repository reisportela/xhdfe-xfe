#ifndef HDFE_REGRESSOR_V11_HPP
#define HDFE_REGRESSOR_V11_HPP

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

#include "fe_absorption.hpp"
#include "hdfe/hdfe_regressor.hpp"
#include "ols.hpp"

namespace hdfe {
namespace v11 {

enum class GroupAggregation { Mean, Sum };

struct ThreadingOptions {
    int default_threads = 0;       // 0 = hardware concurrency
    int max_threads = 0;           // 0 = no cap
    int min_parallel_rows = 20000;
    int target_rows_per_thread = 500000;
    bool symmetric_sweep = false;
};

struct GroupIndividualFeOptions {
    double tol_main = 1e-9;     // tolm in extractfes
    double tol_start = 1e-3;    // toli in extractfes
    double tol_final = 1e-9;    // tolf in extractfes
    int max_iter_main = 100000; // maxiter1 in extractfes
    int max_iter_solver = 1000; // maxiter2 in extractfes
    int verbose = 0;            // 0 = silent
    int accel = 2;              // 0 = none, 1 = linear, 2 = geometric
    int start_accel = 5;        // staccel
    int every_accel = 5;        // evaccel
    double factor = 1.0;        // tighten solver tol if mse < factor * tol
    double a1p1 = 0.75;         // linear acceleration parameter
    double a2p1 = 1e-8;         // geometric acceleration safeguard
    int a2p2 = 5;               // geometric acceleration exponent
};

struct GroupIndividualFeEstimates {
    Eigen::VectorXi individual_ids;
    Eigen::VectorXd individual_effects;
    std::vector<Eigen::VectorXi> fe_level_ids;
    std::vector<Eigen::VectorXd> fe_level_effects;
    int iterations = 0;
    bool converged = true;
    double mse = 0.0;
    int threads_requested = 0;
    int threads_effective = 1;
    int threads_used = 1;
    int parallel_workers_active = 1;
    int thread_capacity = 1;
    bool openmp_enabled = false;
    int thread_limit_code = 0;
    std::string thread_limit_reason = "none";
};

/**
 * \brief Exact connected-component summary for the first pair of FE dimensions.
 *
 * These are the mobility groups used by the Component-style FE normalization
 * and by reghdfe-compatible groupvar()/first-pair DoF accounting. Observation
 * counts always refer to physical rows. The weight share uses the supplied
 * positive weights and is therefore scale invariant.
 */
struct FeComponentStats {
    int num_components = 0;
    long long largest_component_n_obs = 0;
    double largest_component_obs_share = 0.0;
    double largest_component_weight_share = 0.0;
};

FeComponentStats compute_first_pair_component_stats(
    const Eigen::VectorXi& first_fe,
    const Eigen::VectorXi& second_fe,
    const Eigen::VectorXd* weights = nullptr);

class HdfeRegressorV11 {
public:
    explicit HdfeRegressorV11(HdfeOptions options = HdfeOptions{},
                              ThreadingOptions threading = ThreadingOptions{});
    HdfeRegressorV11(const HdfeRegressorV11& other);
    HdfeRegressorV11& operator=(const HdfeRegressorV11& other);
    HdfeRegressorV11(HdfeRegressorV11&&) noexcept = default;
    HdfeRegressorV11& operator=(HdfeRegressorV11&&) noexcept = default;

    void fit(const Eigen::Ref<const Eigen::VectorXd>& y,
             const Eigen::Ref<const Eigen::MatrixXd>& X,
             const std::vector<Eigen::VectorXi>& fes = {},
             const Eigen::VectorXd* weights = nullptr,
             const std::vector<Eigen::VectorXi>* clusters = nullptr,
             const Eigen::MatrixXd* instruments = nullptr,
             const std::vector<int>& endogenous_idx = {},
             const std::vector<detail::HeterogeneousSlopeTerm>* slopes = nullptr);

    // Compute within-transformed y and X after absorbing fixed effects, without running the
    // post-absorption regression step. This is the building block for hdfe-style partialling-out.
    detail::AbsorptionResult partial_out(
        const Eigen::Ref<const Eigen::VectorXd>& y,
        const Eigen::Ref<const Eigen::MatrixXd>& X,
        const std::vector<Eigen::VectorXi>& fes = {},
        const Eigen::VectorXd* weights = nullptr,
        const std::vector<Eigen::VectorXi>* clusters = nullptr,
        const std::vector<detail::HeterogeneousSlopeTerm>* slopes = nullptr);

    void fit_grouped(const Eigen::Ref<const Eigen::VectorXd>& y,
                     const Eigen::Ref<const Eigen::MatrixXd>& X,
                     const std::vector<Eigen::VectorXi>& fes,
                     const Eigen::VectorXi& group_ids,
                     const Eigen::VectorXi* individual_ids = nullptr,
                     GroupAggregation aggregation = GroupAggregation::Mean,
                     const Eigen::VectorXd* weights = nullptr,
                     const std::vector<Eigen::VectorXi>* clusters = nullptr);

    GroupIndividualFeEstimates extract_group_individual_fes(
        const Eigen::Ref<const Eigen::VectorXd>& y,
        const Eigen::Ref<const Eigen::MatrixXd>& X,
        const std::vector<Eigen::VectorXi>& fes,
        const Eigen::VectorXi& group_ids,
        const Eigen::VectorXi& individual_ids,
        GroupAggregation aggregation = GroupAggregation::Mean,
        const Eigen::VectorXd* weights = nullptr,
        const GroupIndividualFeOptions& options = GroupIndividualFeOptions{}) const;

    // Bindings whose weight semantics are selected per call must set this
    // explicitly before fit()/fit_grouped(). The Python binding resets it on
    // every fit call, so a previous fweight fit cannot leak into a later call.
    void set_weights_are_frequencies(bool value) noexcept {
        options_.weights_are_frequencies = value;
    }

    const HdfeResults& results() const noexcept { return results_; }
    int threads_used() const noexcept { return threads_used_; }
    int threads_requested() const noexcept { return threads_requested_; }
    int threads_effective() const noexcept { return threads_effective_; }
    int parallel_workers_active() const noexcept { return parallel_workers_active_; }
    int thread_capacity() const noexcept { return thread_capacity_; }
    bool openmp_enabled() const noexcept { return openmp_enabled_; }
    int thread_limit_code() const noexcept { return thread_limit_code_; }
    const std::string& thread_limit_reason() const noexcept {
        return thread_limit_reason_;
    }
    AbsorptionMethod absorption_method_used() const noexcept { return method_used_; }
    bool gpu_used() const noexcept { return gpu_used_; }
    int gpu_status_code() const noexcept { return gpu_status_code_; }
    bool gpu_attempted() const noexcept { return gpu_attempted_; }
    bool gpu_absorption_converged() const noexcept { return gpu_absorption_converged_; }
    int gpu_absorption_iterations() const noexcept { return gpu_absorption_iterations_; }
    const FeComponentStats& first_pair_component_stats() const noexcept {
        return first_pair_component_stats_;
    }

private:
    struct ThreadResolution {
        int requested = 0;
        int effective = 1;
        int capacity = 1;
        bool openmp_enabled = false;
        int limit_code = 0;
        std::string limit_reason = "none";
    };

    ThreadResolution resolve_threads(int n_rows, int num_fes) const;
    void begin_parallel_observation(const ThreadResolution& resolution);
    void end_parallel_observation();
    AbsorptionMethod select_method(std::size_t num_fes) const;
    void apply_common_postprocessing(const Eigen::Ref<const Eigen::VectorXd>& y,
                                     const Eigen::Ref<const Eigen::MatrixXd>& X,
                                     const Eigen::VectorXd* weights,
                                     const std::vector<int>& fe_levels,
                                     const detail::OlsResult& ols_result);

    HdfeOptions options_;
    ThreadingOptions threading_;
    HdfeResults results_;
    int threads_used_ = 1;
    int threads_requested_ = 0;
    int threads_effective_ = 1;
    int parallel_workers_active_ = 1;
    int thread_capacity_ = 1;
    bool openmp_enabled_ = false;
    int thread_limit_code_ = 0;
    std::string thread_limit_reason_ = "none";
    std::shared_ptr<detail::ParallelWorkObserver> parallel_observer_;
    AbsorptionMethod method_used_ = AbsorptionMethod::GaussSeidel;
    bool gpu_used_ = false;
    int gpu_status_code_ = 0;
    bool gpu_attempted_ = false;
    bool gpu_absorption_converged_ = false;
    int gpu_absorption_iterations_ = 0;
    FeComponentStats first_pair_component_stats_;
};

}  // namespace v11
}  // namespace hdfe

#endif  // HDFE_REGRESSOR_V11_HPP
