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

cluster_base_cov <- function(y, x1, cluster) {
  X <- cbind(x1, 1)
  P <- solve(crossprod(X))
  b <- drop(P %*% crossprod(X, y))
  score <- X * drop(y - X %*% b)
  sums <- rowsum(score, cluster, reorder = FALSE)
  n <- nrow(X); k <- ncol(X); g <- nrow(sums)
  ((n - 1) / (n - k)) * (g / (g - 1)) * P %*% crossprod(sums) %*% P
}

unadjusted_base_cov <- function(y, x1) {
  X <- cbind(x1, 1)
  P <- solve(crossprod(X))
  b <- drop(P %*% crossprod(X, y))
  u <- drop(y - X %*% b)
  drop(crossprod(u)) / (nrow(X) - ncol(X)) * P
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
  expect_equal(r$df_base, nrow(d$x1) - ncol(d$x1) - 1, tolerance = 0)
  expect_equal(r$n_clusters, 0L, tolerance = 0)
  expect_equal(dim(r$gamma), c(1L, 1L))
  expect_identical(colnames(r$gamma), "OBS")
  expect_equal(unname(r$gamma[1, 1]),
               unname(lm.fit(cbind(d$x1, d$z,
                                   model.matrix(~ factor(d$firm))),
                             d$y)$
                      coefficients[3]),
               tolerance = 1e-10)
  expect_equal(unname(r$base_cov),
               unname(unadjusted_base_cov(d$y, d$x1)),
               tolerance = 1e-12)
  expect_equal(dim(r$cov_delta_bbase), c(6L, 3L))
  expect_equal(dim(r$cov_total_bbase), c(3L, 3L))
  expect_identical(names(r$x1_near_collinear_mask), r$x1_names)
  expect_identical(names(r$x1_fe_collinear_ratio), r$x1_names)
  expect_equal(r$near_fe_collinear_ss_ratio_warn_upper, 1e-4,
               tolerance = 0)
  expect_equal(r$few_cluster_warning_threshold, 30L, tolerance = 0)
  expect_gte(r$threads_used, 1L)
  expect_false(r$gpu_requested)
  expect_false(r$gpu_used)
  expect_identical(r$gpu_backend, "cpu")
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
  expect_error(
    xhdfe_gelbach(d$y, matrix(numeric(0), nrow = length(d$y), ncol = 0),
                  x2_groups = list(A = d$z)),
    "at least one focal column"
  )
  expect_error(
    xhdfe_gelbach(d$y, d$x1, x2_groups = list(A = d$z),
                  vce = "cluster", cluster = rep(1:4, length.out = 20)),
    "same length as y"
  )
  expect_error(
    xhdfe_gelbach(d$y, d$x1, x2_groups = list(A = d$z),
                  weights = rep("bad", length(d$y))),
    "numeric vector"
  )
  expect_error(
    xhdfe_gelbach(d$y, d$x1, x2_groups = list(A = factor(d$firm))),
    "generate full-rank indicators"
  )
  expect_error(
    xhdfe_gelbach(d$y, d$x1, x2_groups = list(A = d$z), gpu = NA),
    "non-missing logical"
  )

  grid <- seq(-1, 1, length.out = 5)
  expect_error(
    xhdfe_gelbach(grid, cbind(grid),
                  x2_groups = list(saturated = cbind(grid^2, grid^3, grid^4))),
    "residual degrees of freedom"
  )
})

test_that("vce modes run and gamma0 shrinks the observed-group variance model", {
  d <- sim_gelb()
  rr <- xhdfe_gelbach(d$y, d$x1, x2_groups = list(OBS = d$z),
                      fes = list(FIRM = d$firm), vce = "robust")
  expect_warning(
    rc <- xhdfe_gelbach(d$y, d$x1, x2_groups = list(OBS = d$z),
                        fes = list(FIRM = d$firm), vce = "cluster",
                        cluster = d$firm),
    "few clusters"
  )
  g0 <- xhdfe_gelbach(d$y, d$x1, x2_groups = list(OBS = d$z),
                      fes = list(FIRM = d$firm), gamma0 = TRUE)
  expect_true(all(is.finite(rr$se)) && all(is.finite(rc$se)))
  expect_equal(rc$n_clusters, length(unique(d$firm)), tolerance = 0)
  expect_match(rc$notes, "few clusters", ignore.case = TRUE)
  # deltas are identical across vce choices; only the variances change
  expect_identical(rr$delta, rc$delta)
  expect_identical(rr$delta, g0$delta)

  many_cluster <- expect_silent(
    xhdfe_gelbach(d$y, d$x1, x2_groups = list(OBS = d$z),
                  vce = "cluster",
                  cluster = rep(seq_len(50L), length.out = length(d$y)))
  )
  expect_equal(many_cluster$n_clusters, 50L, tolerance = 0)

  gpu_no_fe <- xhdfe_gelbach(
    d$y, d$x1, x2_groups = list(OBS = d$z), gpu = TRUE
  )
  expect_true(gpu_no_fe$gpu_requested)
  expect_false(gpu_no_fe$gpu_used)
  expect_identical(gpu_no_fe$gpu_backend, "cpu")
  expect_identical(gpu_no_fe$gpu_status, "not_applicable")
})

test_that("absorbed targets are opt-in constraints and standard rank guards remain", {
  set.seed(20260719)
  nw <- 50L
  tt <- 4L
  worker <- rep(seq_len(nw), each = tt)
  group <- rep(sample(0:1, nw, replace = TRUE), each = tt)
  within <- rep(seq_len(tt), nw) + rnorm(nw * tt, sd = 0.1)
  z <- 0.3 * group + 0.2 * within + rnorm(nw * tt)
  y <- (0.4 * group + 0.1 * within + 0.7 * z +
        rep(rnorm(nw), each = tt) + rnorm(nw * tt, sd = 0.4))
  x1 <- cbind(group = group, within = within)

  expect_error(
    xhdfe_gelbach(y, x1, x2_groups = list(OBS = z),
                  fes = list(WORKER = worker)),
    "rank deficient"
  )
  r <- xhdfe_gelbach(
    y, x1, x2_groups = list(OBS = z), fes = list(WORKER = worker),
    vce = "cluster", cluster = worker, absorbed_targets = "group")
  expect_true(r$converged)
  expect_lt(r$identity_gap, 1e-10)
  expect_identical(r$estimand, "absorbed_target_allocation")
  expect_identical(r$identity_status, "exact_ols_constrained")
  expect_identical(unname(r$b_full_status), c("imposed_zero", "estimated"))
  expect_identical(unname(r$focal_status), c("absorbed", "identified"))
  expect_identical(r$absorbed_mask, c(TRUE, FALSE))
  expect_identical(r$absorbed_targets, 0L)
  expect_identical(r$absorbed_target_names, "group")
  expect_equal(r$b_full[1], 0, tolerance = 0)
  expect_equal(r$total[1], r$b_base[1], tolerance = 1e-10)
  expect_identical(r$total_se_type,
                   "target_exact_base_vce_mixed_components")
  expect_identical(r$inference_status, "clustered_at_absorbing_fe")
  expect_true(r$absorbed_target_inference_valid)
  expect_identical(r$absorbing_fe_index, 0L)
  expect_equal(r$fe_collinear_ss_ratio_tol, 1e-9, tolerance = 0)
  base_cov <- cluster_base_cov(y, x1, worker)
  expect_equal(r$total_cov[1, 1], base_cov[1, 1], tolerance = 1e-12)
  expect_equal(r$base_cov[1, 1], base_cov[1, 1], tolerance = 1e-12)
  expect_equal(r$cov_total_bbase[1, 1], r$base_cov[1, 1],
               tolerance = 0)
  absorbed_share <- xhdfe_gelbach_tidy(
    r, share = "base", include_total = TRUE, include_full = FALSE
  )
  absorbed_total <- (
    absorbed_share$component == "total_movement" &
      absorbed_share$coefficient %in% r$absorbed_target_names
  )
  expect_identical(
    unname(absorbed_share$share_std_error[absorbed_total]),
    0
  )
  expect_equal(r$n_obs_input, length(y))
  expect_equal(r$n_singletons_dropped, 0)
  expect_true(all(is.finite(r$cov)))
  expect_output(print(r), "0 \\(imposed\\)")

  g0 <- xhdfe_gelbach(
    y, x1, x2_groups = list(OBS = z), fes = list(WORKER = worker),
    vce = "cluster", cluster = worker, gamma0 = TRUE,
    absorbed_targets = "group")
  c0 <- xhdfe_gelbach(
    y, x1, x2_groups = list(OBS = z), fes = list(WORKER = worker),
    vce = "cluster", cluster = worker, cov0 = TRUE,
    absorbed_targets = "group")
  expect_equal(g0$total_cov[1, 1], r$total_cov[1, 1], tolerance = 0)
  expect_equal(c0$total_cov[1, 1], r$total_cov[1, 1], tolerance = 0)
  expect_output(print(g0), "gamma0")

  expect_warning(
    bad_vce <- xhdfe_gelbach(
      y, x1, x2_groups = list(OBS = z), fes = list(WORKER = worker),
      vce = "robust", absorbed_targets = "group"),
    "inferential diagnostic"
  )
  expect_false(bad_vce$absorbed_target_inference_valid)
  expect_identical(bad_vce$inference_status,
                   "warning_unsupported_vce_or_cluster")

  fw <- rep(1:3, length.out = length(y))
  rf <- xhdfe_gelbach(
    y, x1, x2_groups = list(OBS = z), fes = list(WORKER = worker),
    vce = "cluster", cluster = worker, weights = fw, fweights = TRUE,
    absorbed_targets = "group")
  expect_equal(rf$n_obs, length(y), tolerance = 0)
  expect_equal(rf$n_obs_effective, sum(fw), tolerance = 0)

  expect_error(
    xhdfe_gelbach(y, x1, x2_groups = list(OBS = z),
                  fes = list(WORKER = worker),
                  absorbed_targets = c("group", "within")),
    "must be omitted specifically"
  )
})

test_that("focal reporting, signed shares and contrasts preserve the estimand", {
  d <- sim_gelb()
  colnames(d$x1) <- c("target", "common_control")
  base <- xhdfe_gelbach(
    d$y, d$x1, x2_groups = list(OBS = d$z),
    fes = list(FIRM = d$firm)
  )
  focal <- xhdfe_gelbach(
    d$y, d$x1, x2_groups = list(OBS = d$z),
    fes = list(FIRM = d$firm), focal = "target"
  )
  positional_legacy <- do.call(
    xhdfe_gelbach,
    list(d$y, d$x1, list(OBS = d$z), list(FIRM = d$firm),
         "unadjusted", NULL, FALSE, FALSE, 1e-8, 0L,
         NULL, FALSE, NULL, NULL)
  )

  expect_identical(focal$delta, base$delta)
  expect_identical(positional_legacy$delta, base$delta)
  expect_false(positional_legacy$gpu_requested)
  expect_identical(focal$cov, base$cov)
  expect_identical(focal$total, base$total)
  expect_true(focal$focal_selection_explicit)
  expect_identical(focal$focal_indices, 0L)
  expect_identical(focal$focal_names, "target")

  movement <- xhdfe_gelbach_tidy(
    focal, share = "movement", include_total = FALSE,
    include_full = FALSE
  )
  expect_equal(sum(movement$share), 1, tolerance = 2e-14)
  expect_true(all(is.finite(movement$share_std_error)))
  expect_identical(unique(movement$share_se_type),
                   "joint_covariance_delta_method")

  base_share <- xhdfe_gelbach_tidy(
    focal, share = "base", include_total = FALSE,
    include_full = FALSE
  )
  expect_true(all(is.finite(base_share$share)))
  expect_true(all(is.finite(base_share$share_std_error)))
  expect_identical(unique(base_share$share_se_type),
                   "joint_base_covariance_delta_method")
  k1 <- nrow(focal$delta)
  for (g in seq_along(focal$names)) {
    d_g <- focal$delta[1, g]
    b <- focal$b_base[1]
    v_g <- focal$cov[(g - 1L) * k1 + 1L,
                     (g - 1L) * k1 + 1L]
    c_gb <- focal$cov_delta_bbase[(g - 1L) * k1 + 1L, 1L]
    oracle <- sqrt(max(0, v_g / b^2 +
      d_g^2 * focal$base_cov[1, 1] / b^4 -
      2 * d_g * c_gb / b^3))
    expect_equal(base_share$share_std_error[g], oracle, tolerance = 1e-14)
  }

  base_share_total <- xhdfe_gelbach_tidy(
    focal, share = "base", include_total = TRUE,
    include_full = FALSE
  )
  total_row <- base_share_total$component == "total_movement"
  total <- focal$total[1]
  b <- focal$b_base[1]
  total_oracle <- sqrt(max(0, focal$total_cov[1, 1] / b^2 +
    total^2 * focal$base_cov[1, 1] / b^4 -
    2 * total * focal$cov_total_bbase[1, 1] / b^3))
  expect_equal(base_share_total$share_std_error[total_row], total_oracle,
               tolerance = 1e-14)

  fixed_share <- xhdfe_gelbach_tidy(
    focal, share = "base_fixed", include_total = FALSE,
    include_full = FALSE
  )
  expect_true(all(is.finite(fixed_share$share_std_error)))
  expect_identical(unique(fixed_share$share_se_type),
                   "fixed_base_denominator_scaling")

  undefined <- focal
  undefined$b_base[1] <- 0
  warning_count <- 0L
  undefined_rows <- withCallingHandlers(
    xhdfe_gelbach_tidy(
      undefined, share = "base", include_total = FALSE,
      include_full = FALSE
    ),
    warning = function(w) {
      warning_count <<- warning_count + 1L
      expect_match(conditionMessage(w), "denominator is undefined")
      invokeRestart("muffleWarning")
    }
  )
  expect_equal(warning_count, 1L)
  expect_true(all(is.na(undefined_rows$share)))

  total_contrast <- xhdfe_gelbach_contrast(
    focal, "target", c("OBS", "FIRM")
  )
  expect_equal(total_contrast$estimate, focal$total[1], tolerance = 2e-14)
  expect_equal(total_contrast$std_error, focal$total_se[1], tolerance = 2e-14)
  expect_identical(total_contrast$se_type,
                   "joint_covariance_including_conditional_fe")

  expect_error(
    xhdfe_gelbach(d$y, d$x1, x2_groups = list(OBS = d$z),
                  focal = "not_a_column"),
    "unknown x1 column"
  )
  expect_error(
    xhdfe_gelbach_contrast(focal, "target", "not_a_group"),
    "unknown group"
  )
  duplicated_names <- d$x1
  colnames(duplicated_names) <- c("same", "same")
  expect_error(
    xhdfe_gelbach(d$y, duplicated_names, x2_groups = list(OBS = d$z)),
    "must be unique"
  )
})
