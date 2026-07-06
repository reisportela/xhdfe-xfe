#ifndef HDFE_AKM_KSS_CUDA_HPP
#define HDFE_AKM_KSS_CUDA_HPP

#include <cstddef>
#include <cstdint>

namespace hdfe {
namespace akm {

// Opaque CUDA solver context for the grounded worker-partialled firm
// Laplacian S = Df - B' Dw^{-1} B used by the AKM/KSS machinery. The
// context holds the device-resident match structure and PCG workspace; one
// context serves all solves of a decomposition call. Reductions use CUB
// device sums (run-to-run deterministic on a given device), so GPU results
// are reproducible per device but differ from the CPU path at the usual
// floating-point reassociation level.
struct AkmCudaContext;

bool akm_cuda_available();

// Builds the context (uploads the match CSR/CSC structure). Returns nullptr
// when CUDA is unavailable or allocation fails (caller falls back to CPU).
AkmCudaContext* akm_cuda_create(int n_workers, int n_firms,
                                const int* m_w, const int* m_f,
                                const double* m_c, std::size_t n_matches,
                                const std::int64_t* w_ptr,
                                const std::int64_t* f_ptr,
                                const std::int64_t* f_matches,
                                const double* Dw, const double* Df,
                                const double* inv_diagS);

void akm_cuda_destroy(AkmCudaContext* ctx);

// Jacobi-PCG solve of the grounded system S z = rhs (length J vectors with
// the ground entry pinned to zero). Returns the iteration count, or -1 when
// the relative-residual tolerance was not reached within max_iter.
int akm_cuda_solve_S(AkmCudaContext* ctx, const double* rhs, double* z,
                     double tol, int max_iter);

}  // namespace akm
}  // namespace hdfe

#endif  // HDFE_AKM_KSS_CUDA_HPP
