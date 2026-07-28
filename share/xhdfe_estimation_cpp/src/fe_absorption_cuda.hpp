#ifndef HDFE_FE_ABSORPTION_CUDA_HPP
#define HDFE_FE_ABSORPTION_CUDA_HPP

#include <cstddef>
#include <vector>

#include "fe_absorption.hpp"

namespace hdfe {
namespace detail {

struct GpuFeInput {
    const int* group_ids = nullptr;
    int num_groups = 0;
    int num_levels_present = 0;
    const double* weight_sums = nullptr;
    bool is_slope = false;
    bool slope_has_intercept = false;
    const double* slope_values = nullptr;
    const double* slope_sum_z = nullptr;
    const double* slope_sum_zz = nullptr;
};

struct GpuFeCertificateView {
    const int* group_ids = nullptr;
    std::size_t group_ids_size = 0;
    const int* group_ptr = nullptr;
    std::size_t group_ptr_size = 0;
    int num_groups = 0;
    bool groups_contiguous = false;
};

// CUDA-only post-solve certificate.  Its implementation deliberately lives
// in a separate translation unit so CUDA certificate optimizations cannot
// perturb compiler code generation for the CPU absorption solvers.
#ifdef HDFE_USE_CUDA
void certify_cuda_absorption_result(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<Eigen::VectorXi>& fes,
    const Eigen::VectorXd* weights,
    const HdfeOptions& options,
    const std::vector<HeterogeneousSlopeTerm>& slopes,
    const std::vector<GpuFeCertificateView>& fe_views,
    AbsorptionResult& result);
#endif

enum class GpuBackend { Cpu, Cuda, Metal };

// Task-local backend selection used by composite estimators that need to
// route one nested fit without mutating the process environment.  The
// override is confined to the calling host thread; nested scopes restore the
// previous value.  OpenMP worker regions do not select the backend.
class ScopedGpuBackendOverride {
public:
    explicit ScopedGpuBackendOverride(GpuBackend backend) noexcept;
    ~ScopedGpuBackendOverride() noexcept;

    ScopedGpuBackendOverride(const ScopedGpuBackendOverride&) = delete;
    ScopedGpuBackendOverride& operator=(const ScopedGpuBackendOverride&) = delete;

private:
    bool previous_active_ = false;
    GpuBackend previous_backend_ = GpuBackend::Cpu;
};

bool has_thread_gpu_backend_override() noexcept;
GpuBackend thread_gpu_backend_override() noexcept;

#ifdef HDFE_USE_CUDA
bool cuda_backend_available();

bool absorb_fixed_effects_cuda(const Eigen::Ref<const Eigen::VectorXd>& y,
                               const Eigen::Ref<const Eigen::MatrixXd>& X,
                               const std::vector<GpuFeInput>& fe_inputs,
                               const Eigen::VectorXd* weights,
                               const std::vector<std::size_t>& sweep_order,
                               const HdfeOptions& options,
                               AbsorptionMethod method,
                               AbsorptionResult& result);

bool absorb_fixed_effects_group_individual_cuda(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const std::vector<GpuFeInput>& standard_fe_inputs,
    const GroupIndividualStructure& gi,
    const Eigen::VectorXd* weights,
    const std::vector<std::size_t>& sweep_order,
    const HdfeOptions& options,
    AbsorptionMethod method,
    AbsorptionResult& result);

// Compute max over all FE dimensions of max_g |weighted_mean_g(residual)|,
// applying the Gauss-Seidel sweep (y -= mean_per_group) across dims in order.
// Used by the savefe path to verify the recovered residual is orthogonal to
// each FE basis. Returns true on success and writes to out_max_delta; false
// on CUDA failure (caller must fall back to CPU).
bool fe_recovery_max_delta_cuda(const Eigen::Ref<const Eigen::VectorXd>& residual,
                                const std::vector<GpuFeInput>& fe_inputs,
                                const Eigen::VectorXd* weights,
                                const std::vector<std::size_t>& sweep_order,
                                double& out_max_delta);

bool fe_recovery_max_delta_cuda_cached(const Eigen::Ref<const Eigen::VectorXd>& residual,
                                       const std::vector<GpuFeInput>& fe_inputs,
                                       const Eigen::VectorXd* weights,
                                       const std::vector<std::size_t>& sweep_order,
                                       double& out_max_delta);
#else
inline bool cuda_backend_available() { return false; }

inline bool absorb_fixed_effects_cuda(const Eigen::Ref<const Eigen::VectorXd>&,
                                      const Eigen::Ref<const Eigen::MatrixXd>&,
                                      const std::vector<GpuFeInput>&,
                                      const Eigen::VectorXd*,
                                      const std::vector<std::size_t>&,
                                      const HdfeOptions&,
                                      AbsorptionMethod,
                                      AbsorptionResult&) {
    return false;
}

inline bool absorb_fixed_effects_group_individual_cuda(
    const Eigen::Ref<const Eigen::VectorXd>&,
    const Eigen::Ref<const Eigen::MatrixXd>&,
    const std::vector<GpuFeInput>&,
    const GroupIndividualStructure&,
    const Eigen::VectorXd*,
    const std::vector<std::size_t>&,
    const HdfeOptions&,
    AbsorptionMethod,
    AbsorptionResult&) {
    return false;
}

inline bool fe_recovery_max_delta_cuda(const Eigen::Ref<const Eigen::VectorXd>&,
                                       const std::vector<GpuFeInput>&,
                                       const Eigen::VectorXd*,
                                       const std::vector<std::size_t>&,
                                       double&) {
    return false;
}

inline bool fe_recovery_max_delta_cuda_cached(const Eigen::Ref<const Eigen::VectorXd>&,
                                              const std::vector<GpuFeInput>&,
                                              const Eigen::VectorXd*,
                                              const std::vector<std::size_t>&,
                                              double&) {
    return false;
}
#endif

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_FE_ABSORPTION_CUDA_HPP
