// Schwarz / approx-Cholesky preconditioned PCG within-demean (gated accelerator).
//
// Standalone, Eigen-free core: demeans Y and the k columns of X (column-major, n*k) on the
// orthogonal complement of the span of the FE dummies, by solving the normal equations
// D'D u = D'v with a block preconditioner — approx-Cholesky on the bipartite Laplacian of the
// two highest-cardinality FE dimensions (sign-flip A_if = S L_if S), diagonal 1/deg on the
// remaining dimensions; references pinned for full rank. Block multi-RHS (all columns at
// once) with a race-free CSR gather matvec. Multigraph or collapsed approx-Cholesky chosen by
// the fill heuristic (collapse when the smaller bipartite side is large, e.g. AKM-scale).
//
// This is the validated prototype (_akm_opt/schwarz_proto/schwarz7.cpp + schwarz8.cpp collapse)
// promoted to a production-shaped function. It is NOT wired into absorb_fixed_effects yet.
//
// Returns true on success (Y and X overwritten with the within residuals; iters_out set to the
// max PCG iteration count over columns). Returns false (Y/X untouched) if unsupported:
//   - ndim < 2, n <= 0, k+1 > 16 columns, or the requested column count exceeds capacity.
// Caller keeps MAP as the universal fallback for any false return.

#ifndef XHDFE_SCHWARZ_DEMEAN_HPP
#define XHDFE_SCHWARZ_DEMEAN_HPP

#include <cstdint>
#include <vector>

namespace xhdfe {

// Y: length n (overwritten). X: column-major n*k (overwritten). fe[d]: length-n 0-based level
// codes for dimension d. nlev[d]: number of levels in dimension d. threads<=0 -> use default.
// collapse: -1 = auto (heuristic), 0 = force multigraph, 1 = force collapsed approx-Chol.
bool schwarz_demean_raw(double* Y, double* X, int64_t n, int k,
                        const std::vector<const int32_t*>& fe,
                        const std::vector<int>& nlev,
                        int threads, double tol, int max_iter,
                        int* iters_out, int collapse = -1);

}  // namespace xhdfe

#endif  // XHDFE_SCHWARZ_DEMEAN_HPP
