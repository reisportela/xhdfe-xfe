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

// ---------------------------------------------------------------------------
// WP3 shadow-mode device certificate (plan §10-WP3). Computed entirely on the
// device by the dedicated verifier TU (fe_absorption_cuda_certificate.cu),
// with its own kernels, scratch and reduction trees (§6.7). In shadow mode
// the HOST certificate remains authoritative; the device result is logged and
// compared, never used for any public field. Activated per-fit by the caller
// only when XHDFE_GPU_CERT_SHADOW=1; entirely inert otherwise.
// ---------------------------------------------------------------------------
struct CudaCertShadowResult {
    int verdict = 3;              // 0 PASS, 1 FAIL, 2 INDETERMINATE, 3 ERROR
    int host_like_pass = -1;      // to-nearest rel_max <= L (host semantics)
    double rel_max = 0.0;         // to-nearest aggregate rel, max over columns
    double rel_max_perdim = 0.0;  // per-dimension max (§6.4 / P3 diagnostic)
    double r_lower = 0.0;         // Res_sq bracket of the deciding column
    double r_upper = 0.0;
    double wsum_maxdiff = 0.0;    // §6.7 gid/weight_sums cross-check
    long long empty_groups = 0;   // counted, never silently omitted (§6.5)
    double shadow_begin_ms = 0.0;
    double shadow_finish_ms = 0.0;
    int dims = 0;
    int rhs = 0;
};

struct CudaCertShadowHandle;  // opaque; owned by the verifier TU

CudaCertShadowHandle* cuda_cert_shadow_begin(
    const double* d_y_original,
    const double* d_x_original,
    int n,
    int cols,
    int ld,
    const double* d_weights_or_null,
    const int* const* d_group_ids,      // per-dim device pointers
    const double* const* d_weight_sums, // per-dim device pointers
    const int* num_groups,              // per-dim host values
    int num_dims);

// Runs the accumulation over the post-solve residuals still resident on the
// device, brackets Res_sq per §6.6, decides per §6.5, copies ONLY scalars
// back, frees all device scratch and destroys the handle. Returns false (with
// out->verdict=ERROR) on any CUDA failure — fail-closed.
bool cuda_cert_shadow_finish(CudaCertShadowHandle* handle,
                             const double* d_y_residual,
                             const double* d_x_residual,
                             double certificate_limit,
                             CudaCertShadowResult* out);

// Exception-path cleanup: frees device scratch and the handle, computes nothing.
void cuda_cert_shadow_abort(CudaCertShadowHandle* handle);
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
