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

// Batched independent Jacobi-PCG: nb grounded systems S z_l = rhs_l solved
// simultaneously (one kernel launch per step serves every lane; convergence
// is tracked per lane and retired lanes are frozen on device). rhs/z are
// host row-major J x nb... actually lane-major: rhs_pack/z_pack hold nb
// contiguous length-J columns (lane l at offset l*J). iters_out[l] receives
// the per-lane iteration count, or -1 when that lane failed to converge.
// Returns 0 on success (individual lane failures are reported per lane),
// -1 on a CUDA error. Lanes beyond the context's allocated width fall back
// to sequential akm_cuda_solve_S calls by the caller.
int akm_cuda_solve_S_multi(AkmCudaContext* ctx, const double* rhs_pack,
                           double* z_pack, int nb, double tol, int max_iter,
                           int* iters_out);

// Maximum lane width supported by akm_cuda_solve_S_multi (workspace is
// allocated lazily up to this bound).
int akm_cuda_max_lanes();

// Page-locked host staging buffer of nb * J doubles owned by the context
// (sized on demand alongside the multi-RHS workspace). Packing the rhs
// columns directly into it makes the akm_cuda_solve_S_multi host<->device
// pack transfers pinned (DMA) instead of pageable. The same buffer may be
// passed as both rhs_pack and z_pack: the solve reads the rhs fully before
// writing z. Returns nullptr when pinned allocation is unavailable —
// callers then use their own (pageable) buffers. The buffer stays valid
// until the next akm_cuda_pack_buffer/akm_cuda_solve_S_multi call with a
// larger nb, or context destruction.
double* akm_cuda_pack_buffer(AkmCudaContext* ctx, int nb);

}  // namespace akm
}  // namespace hdfe

#endif  // HDFE_AKM_KSS_CUDA_HPP
