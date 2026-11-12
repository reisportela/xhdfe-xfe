# Invariant AKM components and worker projection inference

Package documentation version: 2.31.1.20261112.

`sigma_centering="person_year"` (Stata: `sigmacentering(personyear)`) makes
the corrected point components invariant to a common change in the outcome's
level. The mean is computed on the retained person-year sample after FWL.
The default `leaveouttwoway` convention remains available for reproducibility.

With `compute_se=True` (Stata: `se` or `ci`), the invariant option also returns
regular-case delta standard errors and simultaneous 95% projection intervals
for firm variance and worker–firm covariance. The calculation includes the
uncertainty from estimating the controls. Requesting inference changes
neither the point estimates nor their leave-out unit.

```python
r = akm_kss(y, worker, firm, X=controls,
            sigma_centering="person_year", compute_se=True)
ci = r["projection_inference"]
```

```stata
xhdfeakm y, worker(worker) firm(firm) controls(x i.year) ///
    sigmacentering(personyear) se
```

```r
r <- xhdfe_akm_kss(y, worker, firm, X = controls,
                  sigma_centering = "person_year", compute_se = TRUE)
r$projection_inference
```

## Interpreting the results

- **Point components:** the existing person-year-centred correction, including
  its finite-sample bias. Invariance does not imply exact unbiasedness under
  arbitrary heteroskedasticity. The point is identical with and without `se`.
- **Delta SEs:** conservative first-order measures of uncertainty in the
  regular regime described below. They are conditional on the JLA seed and
  omit the sampling error of the bias correction. At a weak or zero gradient
  they need not describe the point estimator's sampling distribution.
- **Projection intervals:** images of one joint 95% confidence region for
  linear contrasts. They cover the structural firm variance and covariance
  simultaneously, asymptotically under the conditions below, including at
  weak signal. They can be wider than point ± 1.96 SE and need not contain the
  corrected point. For example, a corrected variance can be negative while
  the interval lies in the nonnegative parameter space. The point is never
  clipped and the interval is never stretched to include it.

Inference clusters by **worker**, allowing dependence across that worker's
jobs. This is distinct from the point correction's `leave_out_level`, which
retains the user's observation or match choice. The inference calculation
runs on CPU; a requested CUDA backend still applies to the existing point
and leverage solves, with their usual GPU-use diagnostics.

The implementation supports 2–128 firms and unit observation weights.
Worker-variance SEs/intervals, frequency-weight inference, sigma-lowess and
lincom are unavailable in this mode. Worker-variance point components remain
available. The legacy inference options retain their existing numerical
semantics, including their outcome-level dependence. Legacy inference on a
FWL-adjusted outcome does not propagate the uncertainty of estimating the
controls; it should not be interpreted as the full-model inference provided
by this option.

## Target and full-model operators

On the final sample, write

\[
Y=X\beta+e,\qquad X=[D\ F\ C],\qquad P=X(X'X)^-X',\quad M=I-P.
\]

`D` contains worker indicators, `F` the identified firm indicators, and `C`
the controls retained by the existing FWL omission rules. An omitted control
has its existing zero-coefficient convention. The new inference does not
choose a different sample, control specification, FE normalisation or solver
tolerance. Controls may vary within a match.

Let `p_j=n_j/n` be firm shares. Choose a deterministic orthonormal basis `U`
orthogonal to `sqrt(p)`. The implementation uses a Householder construction.
The first block of contrasts is

\[
\kappa_\psi=U'\operatorname{diag}(\sqrt p)\psi.
\]

Let `a_j` be the mean worker effect among the observations at firm `j`. The
second block is `kappa_a=U'diag(sqrt(p))a`. Put
`kappa=(kappa_psi',kappa_a')'`, of displayed dimension `d=2(J-1)`. Then

\[
\theta_\psi=\frac n{n-1}\kappa_\psi'\kappa_\psi,
\qquad
\theta_{a\psi}=\frac n{n-1}\kappa_a'\kappa_\psi.
\]

Thus both parameters are quadratic forms `kappa'A kappa` in a fixed-dimensional
linear target. Write its OLS estimate as `kappa_hat=K Y`. The matrix `K`
comes from the **full design**, including control estimation. In particular,
centering an adjusted outcome and reusing a two-FE inference formula is
insufficient.

The implementation reuses the unscaled homoskedastic FWL bread when its
scale and diagonal elements are normal positive floating-point numbers.
Covariance and residual variance carry the same degrees-of-freedom factor,
which cancels in their quotient. Zero/subnormal cases use an explicit
residualised-control Gram calculation. The small firm Schur block completes
the joint within-worker bread. Independent tests compare this construction
with a dense SVD of the complete design.

The response is internally normalised by an exact power of two before QR
and restored to the original units afterwards. This prevents squared column
norms from losing small-scale uncertainty. The optional point path also
rescales extreme normal-equation right-hand sides by a power of two, retaining
the same relative solver tolerance. Ordinary-scale point arithmetic and the
legacy route are retained.

## Completed residuals

For worker `g`, let `P_g=P_gg`, `s_g` be its vector of ones and `m_-g` the
mean of the **original** outcome outside that worker. Fix degree four:

\[
H_g=I+P_g+P_g^2+P_g^3+P_g^4,\quad C_g=P_g^5,
\qquad
r_g=H_g(MY)_g+C_g(Y_g-m_{-g}s_g).
\]

This is a design-only linear map `r=L Y`. Its diagonal block is exactly

\[
L_{gg}=H_g(I-P_g)+C_g=I,
\]

and `L 1=0`. It remains defined for singular stayer or local-control blocks;
no leave-worker-out inverse is required. Computation starts at
`v_g=Y_g-m_-g s_g` and applies `v_g=(MY)_g+P_g v_g` five times. Sparse worker
supports and local projector products avoid constructing the full `n×n`
projector.

The polynomial degree is fixed, not selected from the outcome or exposed as
a tuning parameter. It limits noise amplification while retaining independent
prediction error in directions that a leave-out fit cannot identify.

Define the nonnegative covariance estimate

\[
\widehat V_\kappa=\sum_g(K_g r_g)(K_g r_g)'.
\]

If cluster errors are mean-zero and uncorrelated across clusters, with
`Omega_g=Var(e_g)`, its expectation satisfies the exact identity

\[
E[\widehat V_\kappa]-V_\kappa
=\sum_g K_g\left[b_gb_g'+\sum_{h\ne g}L_{gh}\Omega_hL_{gh}'\right]K_g'
\succeq0,\qquad b=LX\beta.
\]

This expectation inequality alone is not a confidence-coverage theorem.

## Coverage theorem

Consider deterministic triangular arrays of designs. On the identified
contrast range, assume:

1. Independent mean-zero worker error vectors, uniformly bounded cluster
   sizes and fourth moments, and `c I <= Omega_g <= C I` for fixed positive
   constants `c,C`.
2. Fixed target rank `r`, positive definite target covariance `V_kappa`, and
   diffuse influence:
   `delta_n=max_g ||V_kappa^(-1/2) K_g||op -> 0`.

These conditions permit arbitrary control dimension subject to identification
and the stated influence condition. A small control-count/sample-size ratio
alone does not establish them. A firm identified by only a handful of workers
can violate the influence condition even in a large sample. There is no
requirement that the quadratic target have a nonzero gradient.

The following argument establishes asymptotic conservativeness of the
implemented construction. Let `A_g=V_kappa^(-1/2)K_g` and
`Z=sum_g A_g e_g`. Then `Var(Z)=I`, and
`sum_g ||A_g||F² <= r/c`. Lyapunov follows from the fourth-moment bound and
`delta_n -> 0`, so `Z` tends to a standard normal vector.

The norm of `L` is uniformly bounded: each local polynomial has norm at most
five and each `C_g` has norm at most one; the leave-other-mean part has bounded
Frobenius norm when cluster sizes are bounded. In the zero-mean case the
normalised covariance estimate `S0` concentrates around its expectation.
For a unit direction `u`, put `a_g=A_g'u` and
`W=blockdiag(a_g a_g')`. Its quadratic form has matrix `L' W L`, with squared
Frobenius norm bounded by a constant times `delta_n²`. Independence and bounded
fourth moments bound its variance by the same order. Polarisation in fixed
dimension gives operator-norm concentration, and the expectation identity
implies `lambda_min(S0) >= 1-o_p(1)`.

No bounded-mean condition is needed for coverage. To see this, write the
normalised cluster-score matrices as `D+Xi`, where column `g` of `D` is
`A_g b_g` and column `g` of `Xi` is `A_g(Le)_g`. If `Pi` projects onto the row
space of `D`, its rank is at most `r` and

\[
(D+\Xi)(D+\Xi)'\succeq\Xi(I-\Pi)\Xi'
=S_0-\Xi\Pi\Xi'.
\]

The expected trace of the subtracted term is bounded by a constant times
`r² delta_n²`, uniformly over the signal. Hence the minimum eigenvalue of
the normalised covariance estimate is at least `1-o_p(1)`. Combining this
with the normal limit gives asymptotic coverage of at least `1-alpha` for
the joint coefficient ellipsoid. Bounded means additionally bound interval
width, but are not necessary for this coverage argument.

Structural redundancies are allowed. The proof applies on the deterministic
identified range of `K`. Calibration with `chi-square(d,1-alpha)` for the
displayed dimension `d >= r` enlarges the region. The implementation keeps
all `d=2(J-1)` coordinates and obtains a covariance root from a QR factor of
the cluster scores; it does not truncate the region using a numerical rank
estimate or add a ridge. Rank estimates are diagnostics and are used to
detect an unavailable covariance on identified target directions.

The region is represented as

\[
\mathcal C=\{\widehat\kappa+Rz:\|z\|^2\le
\chi^2_{d,.95}\},\qquad RR'=\widehat V_\kappa.
\]

For either component, the reported bounds are the minimum and maximum of
`kappa'A kappa` over this same region. Consequently both projected intervals
cover their structural parameters simultaneously whenever the true contrast
vector lies in the region. The optimisation uses an eigendecomposition and a
scalar secular equation, including interior and hard cases. Dual bounds and
rounding allowances give outward endpoints for the computed region;
primal/dual gaps are reported. This is not interval arithmetic for the entire
fit: errors in forming the covariance root and quadratic forms are assessed
separately by the independent numerical tests.
The chi-square constants are fixed 95% values, independently checked against
80-digit regularised-gamma inversion.

## When the delta SE describes the corrected point

Write the corrected point as
`T=kappa_hat'A kappa_hat-Rhat`, with the actual implemented correction `Rhat`.
Let `w=V_kappa^(1/2) A kappa` and
`B=V_kappa^(1/2) A V_kappa^(1/2)`. In addition to the coverage conditions,
require

\[
\|B\|_{op}/\|w\|\to0,\qquad \widehat R/\|w\|\to_p0.
\]

Then `T-theta=2w'Z+Z'BZ-Rhat` is first-order normal, and the reported
`se_delta=2 sqrt(kappa_hat'A Vhat_kappa A kappa_hat)` is asymptotically
conservative. These are conditions on the gradient and the correction,
not simply on whether the parameter is zero. A zero covariance can be regular;
a zero firm variance has zero gradient.

The realised correction/SE ratio and linearisation index are diagnostics,
not proofs of these asymptotic conditions. The index compares a bound on the
quadratic remainder with the smallest gradient standard deviation inside the
joint region. Infinity means a zero gradient is not excluded. The primary
projection interval is always used; no preliminary test switches intervals.
Changing the JLA seed can change the corrected point, while this inference
uses no JLA approximation or simulation. It is conditional on the seed when
interpreted as a standard error for that point.

## Diagnostics and reproducibility

Python and R return `projection_inference`; Stata returns
`r(inference_method)`, `r(inference_cluster)`, `r(inference_backend)` and
`r(se_method)`. Shared quantities include the displayed contrast dimension,
numerical rank diagnostic, number/maximum size of worker clusters, retained
control rank, polynomial degree, chi-square critical value, projection/Wald
radius factor, observed inference workers, covariance condition and solver gap.

`max_cluster_leverage_trace` (Stata `r(proj_max_trace)`) is the largest trace
of a worker's design-whitened influence Gram. It upper-bounds the corresponding
largest eigenvalue and is a concentration diagnostic. It does not certify a
population covariance lower bound or an asymptotic sequence. Numerical rank,
non-finite arithmetic or unavailable covariance failures are explicit errors.
Covariance rank is checked on singular values of the design-whitened QR root,
so a low-noise direction is not discarded by squaring its condition number.
The reported covariance condition is the squared singular-value ratio.

Stata uses `r(proj_dimension)`, `r(proj_rank)`, `r(proj_clusters)`,
`r(proj_max_cluster)`, `r(proj_controls)`, `r(proj_degree)`,
`r(proj_critical)`, `r(proj_factor)`, `r(proj_max_leverage)`,
`r(proj_cov_condition)`, `r(proj_solver_gap)`, `r(proj_bread_reused)`,
`r(proj_threads_used)` and `r(proj_workers_active)`. Component diagnostics are
`r(proj_nonlinear_psi/cov)`, `r(proj_correction_psi/cov)` and
`r(proj_zero_gradient_psi/cov)`. An infinite index is missing in the Stata
scalar, with its zero-gradient flag set to one.

Independent tests in `tests/analytic_precision/akm_projection_contract.py`
compare the compiled package with a dense full-design oracle, including
controls, omitted/duplicate columns, within-worker dependence, shifts,
reference changes, exact/JLA point routes and GPU point solves. The separate
Monte Carlo evidence reports all cells, interval widths and limitations.
An observed coverage rate or a passing implementation test is not a
finite-sample guarantee. Independence, exogeneity and moment bounds remain
substantive modelling assumptions.

The default procedure follows
[Kline, Saggio and Sølvsten (2020)](https://eml.berkeley.edu/~pkline/papers/KSS2020.pdf).
Recent work on
[cluster-robust inference for quadratic forms](https://arxiv.org/html/2602.13537v2)
studies related leave-out variance problems; the completed-residual projection
argument above is the derivation for this implementation. The numerical
optimisation is a quadratic trust-region problem; see
[Adachi et al. (2017)](https://ora.ox.ac.uk/objects/uuid%3A2920b3ed-d418-4aa2-bcf4-9e091027af8a).
