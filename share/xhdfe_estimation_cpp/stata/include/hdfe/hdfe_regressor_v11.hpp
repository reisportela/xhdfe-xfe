#ifndef HDFE_REGRESSOR_V11_HPP
#define HDFE_REGRESSOR_V11_HPP

#include <Eigen/Dense>

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
};

class HdfeRegressorV11 {
public:
    explicit HdfeRegressorV11(HdfeOptions options = HdfeOptions{},
                              ThreadingOptions threading = ThreadingOptions{});

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

    const HdfeResults& results() const noexcept { return results_; }
    int threads_used() const noexcept { return threads_used_; }
    AbsorptionMethod absorption_method_used() const noexcept { return method_used_; }
    bool gpu_used() const noexcept { return gpu_used_; }
    int gpu_status_code() const noexcept { return gpu_status_code_; }
    bool gpu_attempted() const noexcept { return gpu_attempted_; }
    bool gpu_absorption_converged() const noexcept { return gpu_absorption_converged_; }
    int gpu_absorption_iterations() const noexcept { return gpu_absorption_iterations_; }

private:
    int resolve_threads(int n_rows, int num_fes) const;
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
    AbsorptionMethod method_used_ = AbsorptionMethod::GaussSeidel;
    bool gpu_used_ = false;
    int gpu_status_code_ = 0;
    bool gpu_attempted_ = false;
    bool gpu_absorption_converged_ = false;
    int gpu_absorption_iterations_ = 0;
};

}  // namespace v11
}  // namespace hdfe

#endif  // HDFE_REGRESSOR_V11_HPP
