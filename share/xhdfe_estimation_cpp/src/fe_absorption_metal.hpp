#ifndef HDFE_FE_ABSORPTION_METAL_HPP
#define HDFE_FE_ABSORPTION_METAL_HPP

#include "fe_absorption_cuda.hpp"

namespace hdfe {
namespace detail {

#ifdef HDFE_USE_METAL
bool metal_backend_available();

bool absorb_fixed_effects_metal(const Eigen::Ref<const Eigen::VectorXd>& y,
                                const Eigen::Ref<const Eigen::MatrixXd>& X,
                                const std::vector<GpuFeInput>& fe_inputs,
                                const Eigen::VectorXd* weights,
                                const std::vector<std::size_t>& sweep_order,
                                const HdfeOptions& options,
                                AbsorptionMethod method,
                                AbsorptionResult& result);
#else
inline bool metal_backend_available() { return false; }

inline bool absorb_fixed_effects_metal(const Eigen::Ref<const Eigen::VectorXd>&,
                                       const Eigen::Ref<const Eigen::MatrixXd>&,
                                       const std::vector<GpuFeInput>&,
                                       const Eigen::VectorXd*,
                                       const std::vector<std::size_t>&,
                                       const HdfeOptions&,
                                       AbsorptionMethod,
                                       AbsorptionResult&) {
    return false;
}
#endif

}  // namespace detail
}  // namespace hdfe

#endif  // HDFE_FE_ABSORPTION_METAL_HPP
