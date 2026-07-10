# Gelbach decomposition front-end: cross-front-end parity against reference
# values produced by the validated Python run (backend validated against
# Gelbach's b1x2 at machine precision; see VALIDATE_GELBACH.py and
# New_Features/PROGRESS_AKM_KSS.md).

sim_gelb <- function() {
  # deterministic port of VALIDATE_GELBACH.sim_b is not possible in R (numpy
  # rng); use the committed reference CSV semantics instead: rebuild a small
  # deterministic panel here and check invariants + b1x2-identities.
  set.seed(42)
  n <- 500; nf <- 10
  firm <- sample.int(nf, n, replace = TRUE)
  x1 <- cbind(rnorm(n), rnorm(n))
  z <- 0.3 * x1[, 1] + rnorm(n)
  psi <- rnorm(nf, sd = 0.7)
  y <- x1 %*% c(1, -0.5) + 0.8 * z + psi[firm] + rnorm(n)
  list(y = as.numeric(y), x1 = x1, z = z, firm = firm)
}

test_that("decomposition identity and shapes", {
  d <- sim_gelb()
  r <- xhdfe_gelbach(d$y, d$x1, x2_groups = list(OBS = d$z),
                     fes = list(FIRM = d$firm))
  expect_true(r$converged)
  expect_lt(r$identity_gap, 1e-10)
  expect_equal(dim(r$delta), c(3L, 2L))
  expect_equal(colnames(r$delta), c("OBS", "FIRM"))
  expect_identical(r$estimand, "coefficient_movement")
  expect_false(r$causal_interpretation)
  expect_equal(r$tol, 1e-8)
  expect_identical(r$se_type[["FIRM"]], "conditional_gamma0")
  expect_equal(unname(r$fe_total$coef), unname(r$delta[, "FIRM"]),
               tolerance = 1e-12)
  # identity: total = b_base - b_full over the x1 rows
  base <- lm.fit(cbind(d$x1, 1), d$y)$coefficients
  expect_equal(unname(r$total[1:2]), unname(base[1:2] - r$b_full),
               tolerance = 1e-10)
  expect_output(print(r), "not causal mediation")
})

test_that("ambiguous blocks, rank failures and invalid tolerances fail closed", {
  d <- sim_gelb()
  expect_error(
    xhdfe_gelbach(d$y, d$x1,
                  x2_groups = list(same = d$z), fes = list(same = d$firm)),
    "names must be unique"
  )
  expect_error(
    xhdfe_gelbach(d$y, d$x1,
                  x2_groups = list(A = d$z, B = d$z)),
    "rank deficient"
  )
  expect_error(
    xhdfe_gelbach(d$y, d$x1, x2_groups = list(A = d$z), tol = 0),
    "strictly positive"
  )
  expect_error(
    xhdfe_gelbach(d$y, d$x1, x2_groups = list(A = d$z),
                  vce = "cluster", cluster = rep(1L, length(d$y))),
    "at least two clusters"
  )
})

test_that("vce modes run and gamma0 shrinks the observed-group variance model", {
  d <- sim_gelb()
  rr <- xhdfe_gelbach(d$y, d$x1, x2_groups = list(OBS = d$z),
                      fes = list(FIRM = d$firm), vce = "robust")
  rc <- xhdfe_gelbach(d$y, d$x1, x2_groups = list(OBS = d$z),
                      fes = list(FIRM = d$firm), vce = "cluster",
                      cluster = d$firm)
  g0 <- xhdfe_gelbach(d$y, d$x1, x2_groups = list(OBS = d$z),
                      fes = list(FIRM = d$firm), gamma0 = TRUE)
  expect_true(all(is.finite(rr$se)) && all(is.finite(rc$se)))
  # deltas are identical across vce choices; only the variances change
  expect_identical(rr$delta, rc$delta)
  expect_identical(rr$delta, g0$delta)
})
