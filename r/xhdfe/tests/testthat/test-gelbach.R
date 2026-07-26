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
  expect_equal(r$n_mobility_components, 0L, tolerance = 0)
  expect_false(r$fe_split_identified)
  expect_identical(r$fe_split_status, "single_fe_dimension")
  expect_identical(r$connectivity_fe_indices, integer(0))
  expect_identical(r$connectivity_fe_names, character(0))
  expect_false(r$connectivity_pair_explicit)
  expect_identical(r$connectivity_pair_status, "not_applicable")
  expect_identical(r$connected_mode, "diagnose")
  expect_identical(r$mobility_component_scope, "not_applicable")
  expect_identical(r$se_type[["FIRM"]], "conditional_gamma0")
  expect_equal(unname(r$fe_total$coef), unname(r$delta[, "FIRM"]),
               tolerance = 1e-12)
  # identity: total = b_base - b_full over the x1 rows
  base <- lm.fit(cbind(d$x1, 1), d$y)$coefficients
  expect_equal(unname(r$total[1:2]), unname(base[1:2] - r$b_full),
               tolerance = 1e-10)
  expect_output(print(r), "not causal mediation")
})

test_that("FE mobility diagnostics use the retained sample and fail loud", {
  set.seed(20260724)

  worker <- rep(seq_len(16L), each = 4L)
  period <- rep(0:3, times = 16L)
  firm <- ifelse(period < 2L, (worker - 1L) %% 8L + 1L,
                 worker %% 8L + 1L)
  n <- length(worker)
  x <- rnorm(n)
  z <- 0.25 * x + rnorm(n)
  y <- (0.7 * x + 0.4 * z + rnorm(16L)[worker] +
        rnorm(8L)[firm] + rnorm(n, sd = 0.3))
  connected <- xhdfe_gelbach(
    y, x, x2_groups = list(observed = z),
    fes = list(worker = worker, firm = firm)
  )
  expect_true(connected$converged)
  expect_equal(connected$n_mobility_components, 1L, tolerance = 0)
  expect_equal(connected$largest_mobility_component_n_obs, n, tolerance = 0)
  expect_equal(connected$largest_mobility_component_share, 1, tolerance = 0)
  expect_equal(connected$largest_mobility_component_weight_share, 1,
               tolerance = 0)
  expect_true(connected$fe_split_identified)
  expect_identical(connected$fe_split_status, "identified_two_way")
  expect_identical(connected$connectivity_fe_indices, c(0L, 1L))
  expect_identical(connected$connectivity_fe_names, c("worker", "firm"))
  expect_false(connected$connectivity_pair_explicit)
  expect_identical(connected$connectivity_pair_status, "connected")
  expect_identical(connected$connected_mode, "diagnose")
  expect_identical(connected$mobility_component_scope,
                   "first_two_fe_dimensions")
  required <- xhdfe_gelbach(
    y, x, x2_groups = list(observed = z),
    fes = list(worker = worker, firm = firm),
    connected = "require", connectivity_fes = c("firm", "worker")
  )
  expect_true(required$fe_split_identified)
  expect_identical(required$connected_mode, "require")
  expect_identical(required$connectivity_fe_indices, c(1L, 0L))
  expect_identical(required$connectivity_fe_names, c("firm", "worker"))
  expect_true(required$connectivity_pair_explicit)
  expect_identical(required$connectivity_pair_status, "connected")
  expect_identical(required$mobility_component_scope, "selected_fe_pair")
  expect_equal(required$cov, connected$cov, tolerance = 0)

  local_worker <- rep(seq_len(8L), each = 4L)
  local_period <- rep(0:3, times = 8L)
  local_firm <- ifelse(local_period < 2L,
                       (local_worker - 1L) %% 4L + 1L,
                       local_worker %% 4L + 1L)
  worker2 <- c(local_worker, 8L + local_worker, 17L)
  firm2 <- c(local_firm, 4L + local_firm, 9L)
  component <- c(rep(0L, length(local_worker)),
                 rep(1L, length(local_worker)), 2L)
  n_input <- length(worker2)
  x2 <- component + 0.2 * rnorm(n_input)
  z2 <- 0.3 * x2 + rnorm(n_input)
  y2 <- (0.8 * x2 + 0.5 * z2 + rnorm(17L)[worker2] +
         rnorm(9L)[firm2] + rnorm(n_input, sd = 0.3))
  weights <- ifelse(component == 0L, 1, 3)
  weights[n_input] <- 100
  expect_warning(
    disconnected <- xhdfe_gelbach(
      y2, x2, x2_groups = list(observed = z2),
      fes = list(worker = worker2, firm = firm2), weights = weights
    ),
    "normalization-dependent"
  )
  expect_true(disconnected$converged)
  expect_equal(disconnected$n_obs_input, n_input, tolerance = 0)
  expect_equal(disconnected$n_singletons_dropped, 1, tolerance = 0)
  expect_equal(disconnected$n_obs, n_input - 1, tolerance = 0)
  expect_equal(disconnected$n_mobility_components, 2L, tolerance = 0)
  expect_equal(disconnected$largest_mobility_component_n_obs, 32,
               tolerance = 0)
  expect_equal(disconnected$largest_mobility_component_share, 0.5,
               tolerance = 0)
  expect_equal(disconnected$largest_mobility_component_weight_share, 0.75,
               tolerance = 0)
  expect_false(disconnected$fe_split_identified)
  expect_identical(disconnected$fe_split_status,
                   "normalization_dependent")
  expect_identical(disconnected$connectivity_pair_status, "disconnected")
  expect_equal(
    unname(disconnected$fe_total$coef),
    unname(disconnected$delta[, "worker"] +
             disconnected$delta[, "firm"]),
    tolerance = 0
  )
  expect_error(
    xhdfe_gelbach(
      y2, x2, x2_groups = list(observed = z2),
      fes = list(worker = worker2, firm = firm2), weights = weights,
      connected = "require"
    ),
    "connected\\(require\\) failed"
  )

  occupation <- (worker + period) %% 5L + 1L
  expect_warning(
    multiway <- xhdfe_gelbach(
      y, x, x2_groups = list(observed = z),
      fes = list(worker = worker, firm = firm, occupation = occupation)
    ),
    "not connectivity-certified"
  )
  expect_equal(multiway$n_mobility_components, 1L, tolerance = 0)
  expect_false(multiway$fe_split_identified)
  expect_identical(multiway$fe_split_status, "not_certified_multiway")
  expect_identical(multiway$connectivity_pair_status, "connected")
  expect_output(print(multiway), "FE X1-row split: not_certified_multiway")

  bridge <- c(local_period %% 2L, local_period %% 2L, 0L)
  three_fes <- list(worker = worker2, firm = firm2, bridge = bridge)
  first_pair <- suppressWarnings(xhdfe_gelbach(
    y2, x2, x2_groups = list(observed = z2),
    fes = three_fes, weights = weights
  ))
  selected_pair <- suppressWarnings(xhdfe_gelbach(
    y2, x2, x2_groups = list(observed = z2),
    fes = three_fes, weights = weights,
    connectivity_fes = c("worker", "bridge")
  ))
  expect_equal(first_pair$n_mobility_components, 2L, tolerance = 0)
  expect_identical(first_pair$connectivity_pair_status, "disconnected")
  expect_identical(first_pair$fe_split_status, "not_certified_multiway")
  expect_equal(selected_pair$n_mobility_components, 1L, tolerance = 0)
  expect_identical(selected_pair$connectivity_pair_status, "connected")
  expect_identical(selected_pair$connectivity_fe_indices, c(0L, 2L))
  expect_identical(selected_pair$connectivity_fe_names,
                   c("worker", "bridge"))
  expect_true(selected_pair$connectivity_pair_explicit)
  expect_identical(selected_pair$mobility_component_scope,
                   "selected_fe_pair")
  expect_false(selected_pair$fe_split_identified)
  expect_identical(selected_pair$fe_split_status,
                   "not_certified_multiway")
  expect_equal(selected_pair$cov, first_pair$cov, tolerance = 0)
  expect_error(
    xhdfe_gelbach(
      y2, x2, x2_groups = list(observed = z2), fes = three_fes,
      connected = "require",
      connectivity_fes = c("worker", "bridge")
    ),
    "requires exactly two FE dimensions"
  )
  expect_error(
    xhdfe_gelbach(
      y2, x2, x2_groups = list(observed = z2), fes = three_fes,
      connectivity_fes = c("worker", "unknown")
    ),
    "unknown FE name"
  )
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
    xhdfe_gelbach(
      d$y, d$x1, x2_groups = list(A = d$z), num_threads = -1L
    ),
    "nonnegative integer"
  )
  expect_error(
    xhdfe_gelbach(
      d$y, d$x1, x2_groups = list(A = d$z), num_threads = 1.5
    ),
    "nonnegative integer"
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
  r <- suppressWarnings(xhdfe_gelbach(
    y, x1, x2_groups = list(OBS = z), fes = list(WORKER = worker),
    vce = "cluster", cluster = worker, absorbed_targets = "group"))
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
                   paste0("target_exact_base_vce_mixed_components",
                          "_conditional_only_diagnostic"))
  expect_identical(
    unname(r$fe_variance_status),
    c("conditional_only_between_fe_dominant",
      "conditional_only_between_fe_dominant")
  )
  expect_identical(r$inference_status, "clustered_at_absorbing_fe")
  expect_true(r$absorbed_target_inference_valid)
  expect_identical(r$absorbing_fe_index, 0L)
  expect_equal(r$fe_collinear_ss_ratio_tol, 1e-9, tolerance = 0)
  base_cov <- cluster_base_cov(y, x1, worker)
  expect_equal(r$total_cov[1, 1], base_cov[1, 1], tolerance = 1e-12)
  expect_equal(r$base_cov[1, 1], base_cov[1, 1], tolerance = 1e-12)
  expect_equal(r$cov_total_bbase[1, 1], r$base_cov[1, 1],
               tolerance = 0)
  absorbed_share <- suppressWarnings(xhdfe_gelbach_tidy(
    r, share = "base", include_total = TRUE, include_full = FALSE
  ))
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

  g0 <- suppressWarnings(xhdfe_gelbach(
    y, x1, x2_groups = list(OBS = z), fes = list(WORKER = worker),
    vce = "cluster", cluster = worker, gamma0 = TRUE,
    absorbed_targets = "group"))
  c0 <- suppressWarnings(xhdfe_gelbach(
    y, x1, x2_groups = list(OBS = z), fes = list(WORKER = worker),
    vce = "cluster", cluster = worker, cov0 = TRUE,
    absorbed_targets = "group"))
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
  rf <- suppressWarnings(xhdfe_gelbach(
    y, x1, x2_groups = list(OBS = z), fes = list(WORKER = worker),
    vce = "cluster", cluster = worker, weights = fw, fweights = TRUE,
    absorbed_targets = "group"))
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

test_that("nonregular product inference is diagnosed contribution by contribution", {
  n <- 256L
  focal <- rep(c(1, -1), length.out = n)
  orthogonal <- rep(c(1, 1, -1, -1), length.out = n)
  residual <- rep(c(rep(1, 4), rep(-1, 4)), length.out = n)
  y <- 1.2 * focal + 0.7 * residual

  fit <- NULL
  expect_warning(
    fit <- xhdfe_gelbach(
      y, cbind(focal = focal),
      x2_groups = list(orthogonal = orthogonal)
    ),
    "regular first-order delta-method inference is not established"
  )
  diag <- fit$regularity$orthogonal
  expect_equal(unname(fit$beta2), 0, tolerance = 2e-14)
  expect_equal(unname(fit$auxiliary_loadings),
               matrix(0, 2, 1), tolerance = 2e-14)
  expect_equal(unname(diag$contribution_gradient_norm),
               c(0, 0), tolerance = 2e-14)
  expect_false(fit$regular_inference_all_valid)
  expect_identical(
    unname(diag$regular_inference_valid), c(FALSE, FALSE)
  )
  expect_identical(
    unname(diag$regular_inference_status),
    c("nonregular_not_ruled_out", "nonregular_not_ruled_out")
  )
  expect_gt(diag$beta2_wald_pvalue, fit$regularity_test_alpha)
  expect_true(all(
    diag$auxiliary_loading_pvalue > fit$regularity_test_alpha
  ))
  expect_equal(dim(fit$beta2_cov), c(1L, 1L))
  expect_equal(dim(fit$regular_inference_valid), c(2L, 1L))

  tab <- xhdfe_gelbach_tidy(
    fit, focal = "focal", include_total = FALSE,
    include_full = FALSE
  )
  expect_false(tab$regular_inference_valid)
  expect_identical(
    tab$regular_inference_status, "nonregular_not_ruled_out"
  )
  expect_identical(
    tab$confidence_interval_status,
    "diagnostic_only_nonregular_not_ruled_out"
  )
  expect_match(tab$se_type, "nonregular_diagnostic_only$")
  expect_warning(
    xhdfe_gelbach_contrast(fit, "focal", "orthogonal"),
    "normal-theory interval is diagnostic only"
  )

  loaded <- NULL
  expect_warning(
    loaded <- xhdfe_gelbach(
      y, cbind(focal = focal),
      x2_groups = list(loaded = 0.8 * focal + orthogonal)
    ),
    "regular first-order delta-method inference is not established"
  )
  expect_identical(
    unname(loaded$regularity$loaded$regular_inference_valid),
    c(TRUE, FALSE)
  )
  expect_identical(
    unname(loaded$regularity$loaded$regular_inference_status),
    c("regular_loading_nonzero", "nonregular_not_ruled_out")
  )
})

test_that("common FEs condition base and full while added FEs decompose", {
  set.seed(20260726)
  n <- 720L
  common <- rep(seq_len(24L), each = n / 24L)
  added <- sample.int(18L, n, replace = TRUE)
  cluster <- rep(seq_len(60L), each = n / 60L)
  x1 <- cbind(x = rnorm(n), w = rnorm(n))
  x2 <- cbind(
    z1 = 0.35 * x1[, 1] + rnorm(n),
    z2 = -0.20 * x1[, 2] + rnorm(n)
  )
  y <- drop(
    x1 %*% c(1.1, -0.6) + x2 %*% c(0.7, -0.25) +
      rnorm(24L)[common] + rnorm(18L)[added] + rnorm(n, sd = 0.25)
  )

  dc <- model.matrix(~ factor(common))
  da <- model.matrix(~ factor(added))[, -1, drop = FALSE]
  base_oracle <- lm.fit(cbind(x1, dc), y)$coefficients[1:2]
  full_oracle <- lm.fit(cbind(x1, x2, dc, da), y)$coefficients[1:2]

  fit <- xhdfe_gelbach(
    y, x1, x2_groups = list(observed = x2),
    fes = list(added = added), common_fes = list(cohort = common),
    vce = "cluster", cluster = cluster, tol = 1e-10
  )
  expect_true(fit$converged)
  expect_equal(unname(fit$b_base), unname(base_oracle), tolerance = 2e-10)
  expect_equal(unname(fit$b_full), unname(full_oracle), tolerance = 2e-10)
  expect_equal(
    unname(fit$total[1:2]),
    unname(fit$b_base - fit$b_full),
    tolerance = 2e-10
  )
  expect_lt(fit$identity_gap, 2e-10)
  expect_identical(fit$common_fe_names, "cohort")
  expect_equal(fit$n_common_fes, 1L, tolerance = 0)
  expect_true(fit$common_fes_applied)
  expect_false(fit$intercept_inference_available)
  expect_identical(fit$intercept_status, "not_certified_common_fes")
  expect_identical(
    fit$identity_status, "exact_ols_conditional_common_fes"
  )
  expect_true(is.na(fit$total[3]))
  expect_true(is.na(fit$total_se[3]))
  expect_true(is.na(fit$base_cov[3, 3]))
  expect_identical(
    unname(fit$regular_inference_status[3, "observed"]),
    "not_applicable_common_fe_intercept"
  )
  expect_match(fit$notes, "common FEs were conditioned out")

  expect_error(
    xhdfe_gelbach(y, x1, common_fes = list(cohort = common)),
    "added fixed-effect dimension"
  )
  expect_error(
    xhdfe_gelbach(
      y, x1, x2_groups = list(same = x2),
      fes = list(added = added), common_fes = list(same = common)
    ),
    "names must be unique"
  )
  expect_error(
    xhdfe_gelbach(
      y, x1, x2_groups = list(observed = x2),
      fes = list(added = added, added2 = (added + common) %% 19L),
      common_fes = list(cohort = common), connected = "require"
    ),
    "no common FEs"
  )
  expect_warning(
    diagnosed <- xhdfe_gelbach(
      y, x1, x2_groups = list(observed = x2),
      fes = list(added = added, added2 = (added + common) %% 19L),
      common_fes = list(cohort = common)
    ),
    "not connectivity-certified"
  )
  expect_false(diagnosed$fe_split_identified)
  expect_identical(
    diagnosed$fe_split_status, "not_certified_with_common_fes"
  )
  expect_identical(
    diagnosed$mobility_component_scope,
    "first_two_added_fe_dimensions"
  )
  expect_identical(
    diagnosed$connectivity_fe_names, c("added", "added2")
  )
})

test_that("retained-sample provenance is opt-in, exact, and non-invasive", {
  i <- seq_len(25L)
  group <- c(rep(seq_len(6L), each = 4L), 7L)
  x <- (i - mean(i)) / sd(i)
  z <- sin(1.3 * i) + 0.2 * cos(0.7 * i)
  y <- 0.8 * x + 0.4 * z + (group %% 3L) / 5 + cos(0.43 * i)

  plain <- suppressWarnings(xhdfe_gelbach(
    y, cbind(target = x), x2_groups = list(observed = z),
    fes = list(group = group)
  ))
  audited <- suppressWarnings(xhdfe_gelbach(
    y, cbind(target = x), x2_groups = list(observed = z),
    fes = list(group = group), sample_info = TRUE
  ))

  expect_false(plain$sample_info_requested)
  expect_null(plain$sample_index)
  expect_null(plain$sample_mask)
  expect_null(plain$sample_hash)
  expect_true(audited$sample_info_requested)
  expect_identical(audited$sample_index, 0:23)
  expect_identical(which(audited$sample_mask) - 1L, 0:23)
  expect_false(audited$sample_mask[25L])
  expect_equal(sum(audited$sample_mask), audited$n_obs, tolerance = 0)
  expect_equal(audited$n_singletons_dropped, 1, tolerance = 0)
  expect_identical(audited$sample_hash, "2d4dcd55f696e111")
  expect_identical(audited$sample_hash_algorithm, "fnv1a64-le-v1")
  expect_identical(audited$sample_index_scope, "input_rows_zero_based")
  expect_identical(plain$b_base, audited$b_base)
  expect_identical(plain$b_full, audited$b_full)
  expect_identical(plain$delta, audited$delta)
  expect_identical(plain$cov, audited$cov)
  expect_error(
    xhdfe_gelbach(
      y, cbind(target = x), x2_groups = list(observed = z),
      sample_info = NA
    ),
    "sample_info must be one non-missing logical"
  )
})

test_that("weak share denominators are retained but diagnostically gated", {
  set.seed(20260725)
  n <- 800L
  x <- rnorm(n)
  z <- rnorm(n)
  design <- cbind(x, 1)
  y0 <- 0.7 * z + rnorm(n)
  residual <- drop(y0 - design %*% lm.fit(design, y0)$coefficients)
  fit <- xhdfe_gelbach(
    residual + 1e-11 * x, cbind(x = x),
    x2_groups = list(A = z)
  )
  expect_warning(
    tab <- xhdfe_gelbach_tidy(
      fit, focal = "x", share = "base",
      include_total = FALSE, include_full = FALSE
    ),
    "share_t_min"
  )
  expect_true(all(tab$share_defined))
  expect_true(all(tab$share_denominator_t < 3))
  expect_true(all(
    tab$share_interval_status ==
      "weak_denominator_delta_method_unreliable"
  ))
  expect_true(all(grepl(
    "_weak_denominator_diagnostic_only$",
    tab$share_se_type
  )))

  strong <- xhdfe_gelbach(
    0.6 * x + 0.8 * z + rnorm(n), cbind(x = x),
    x2_groups = list(A = z)
  )
  strong_tab <- xhdfe_gelbach_tidy(
    strong, focal = "x", share = "base",
    include_total = FALSE, include_full = FALSE
  )
  expect_true(all(strong_tab$share_denominator_t >= 3))
  expect_true(all(strong_tab$share_interval_status == "valid_first_order"))
})

test_that("between-FE-dominant focal variance gates conditional FE intervals", {
  set.seed(20260726)
  n <- 6000L
  firm <- sample.int(80L, n, replace = TRUE)
  alpha_values <- rnorm(80L)
  alpha <- alpha_values[firm]
  z <- rnorm(n)

  x_between <- sqrt(3) * alpha + rnorm(n)
  expect_warning(
    gated <- xhdfe_gelbach(
      x_between + alpha + rnorm(n), cbind(x = x_between),
      x2_groups = list(A = z), fes = list(firm = firm)
    ),
    "conditional FE-block"
  )
  expect_lte(unname(gated$x1_fe_collinear_ratio["x"]), 0.35)
  expect_identical(
    unname(gated$fe_variance_status["x"]),
    "conditional_only_between_fe_dominant"
  )
  expect_match(
    gated$se_type[["firm"]], "_conditional_only_diagnostic$"
  )
  expect_match(gated$total_se_type, "_conditional_only_diagnostic$")

  x_within <- 0.5 * alpha + rnorm(n)
  valid <- suppressWarnings(xhdfe_gelbach(
    x_within + alpha + rnorm(n), cbind(x = x_within),
    x2_groups = list(A = z), fes = list(firm = firm)
  ))
  expect_gt(unname(valid$x1_fe_collinear_ratio["x"]), 0.35)
  expect_identical(
    unname(valid$fe_variance_status["x"]), "valid_first_order"
  )
  expect_false(grepl(
    "_conditional_only_diagnostic$", valid$se_type[["firm"]]
  ))
})

test_that("filtered etables preserve every panel with a covariance-aware Other", {
  set.seed(20260727)
  n <- 1200L
  x <- rnorm(n)
  blocks <- list(
    a = 0.2 * x + rnorm(n),
    b = 0.3 * x + rnorm(n),
    c = 0.4 * x + rnorm(n),
    d = 0.5 * x + rnorm(n)
  )
  y <- x + 0.15 * blocks$a + 0.30 * blocks$b +
    0.45 * blocks$c + 0.60 * blocks$d + rnorm(n)
  fit <- xhdfe_gelbach(y, cbind(x = x), x2_groups = blocks, vce = "robust")

  for (keep in list(NULL, "a", c("a", "b"))) {
    tab <- xhdfe_gelbach_etable(
      fit, panels = "all", keep = keep, exact_match = TRUE
    )
    for (panel in c("levels", "share_base", "share_movement")) {
      shown <- tab[
        tab$coefficient == "x" & tab$panel == panel &
          tab$component_kind %in% c("x2", "fe", "filtered_aggregate"),
        , drop = FALSE
      ]
      total <- tab$estimate[
        tab$coefficient == "x" & tab$panel == panel &
          tab$component_kind == "total"
      ]
      expect_lte(abs(sum(shown$estimate) - total), 1e-12)
    }
  }

  one <- xhdfe_gelbach_etable(
    fit, panels = "levels", keep = "a", exact_match = TRUE
  )
  other <- one[one$component_name == "other_filtered", , drop = FALSE]
  omitted <- c(3L, 5L, 7L)
  expected_se <- sqrt(sum(fit$cov[omitted, omitted]))
  expect_equal(other$std_error, expected_se, tolerance = 1e-14)

  expect_warning(
    legacy <- xhdfe_gelbach_etable(
      fit, keep = "a", exact_match = TRUE, include_other = FALSE
    ),
    "do not preserve the Gelbach accounting identity"
  )
  expect_false(any(legacy$component_name == "other_filtered"))
})
