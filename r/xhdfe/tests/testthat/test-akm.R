# AKM + leave-out (KSS) front-end tests.
#
# The felsdvsimul reference values are the exact-path results validated at
# machine precision against Saggio's LeaveOutTwoWay (MATLAB, the canonical
# KSS oracle) and an independent dense NumPy oracle on 2026-07-05 (see
# New_Features/PROGRESS_AKM_KSS.md at the repository root).

akm_sim_panel <- function(n_w = 120L, n_f = 15L, reps = 4L, seed = 20260706L) {
  # dense mover panel: every worker changes firm every period
  set.seed(seed)
  worker <- rep(seq_len(n_w), each = reps)
  firm <- integer(n_w * reps)
  pos <- 1L
  for (w in seq_len(n_w)) {
    f <- sample.int(n_f, 1L)
    for (t in seq_len(reps)) {
      firm[pos] <- f
      pos <- pos + 1L
      nxt <- sample.int(n_f - 1L, 1L)
      f <- if (nxt < f) nxt else nxt + 1L
    }
  }
  a <- rnorm(n_w, 0, 0.6)
  p <- rnorm(n_f, 0, 0.4)
  y <- a[worker] + p[firm] + rnorm(n_w * reps) * 0.3
  list(y = y, worker = worker, firm = firm)
}

felsdv <- function() {
  path <- testthat::test_path("..", "..", "..", "..", "data", "felsdvsimul.dta")
  skip_if_not(file.exists(path), "felsdvsimul.dta not available")
  skip_if_not_installed("foreign")
  foreign::read.dta(path)
}

test_that("leave-out connected set matches the canonical sample", {
  d <- felsdv()
  s <- xhdfe_akm_leave_out_set(d$i, d$j)
  expect_equal(s$n_obs, 15)
  expect_equal(s$n_workers, 2L)
  expect_equal(s$n_firms, 4L)
  expect_equal(s$n_matches, 8)
  expect_equal(sum(s$keep), 15L)
})

test_that("exact decomposition reproduces the LeaveOutTwoWay-validated values", {
  d <- felsdv()
  fit <- xhdfe_akm_kss(d$y, d$i, d$j, leverages = "exact")
  expect_true(fit$converged)
  expect_true(fit$leverages_exact)
  expect_equal(fit$plugin$var_psi, 34.398203, tolerance = 1e-6)
  expect_equal(fit$kss$var_psi, 6.2254112695, tolerance = 1e-6)
  expect_equal(fit$kss$var_alpha, -6.2429443959, tolerance = 1e-6)
  expect_equal(fit$kss$cov_alpha_psi, 2.6372007926, tolerance = 1e-6)
  # felsdvsimul's leave-out sample is all-movers: every leverage is < 1
  expect_lt(max(fit$pii), 1)
  expect_equal(fit$max_pii, max(fit$pii), tolerance = 1e-12)
  expect_output(print(fit), "leave-out")
})

test_that("JLA path is deterministic under a fixed seed", {
  d <- felsdv()
  a <- xhdfe_akm_kss(d$y, d$i, d$j, leverages = "jla", jla_draws = 60L, seed = 42)
  b <- xhdfe_akm_kss(d$y, d$i, d$j, leverages = "jla", jla_draws = 60L, seed = 42)
  expect_identical(a$kss, b$kss)
  expect_identical(a$pii, b$pii)
  cc <- xhdfe_akm_kss(d$y, d$i, d$j, leverages = "jla", jla_draws = 60L, seed = 43)
  expect_false(identical(a$kss, cc$kss))
})

test_that("verbose reports progress without changing AKM/KSS results", {
  d <- felsdv()
  quiet <- xhdfe_akm_kss(d$y, d$i, d$j, leverages = "jla",
                         jla_draws = 8L, seed = 42L)
  progress <- capture.output(
    loud <- xhdfe_akm_kss(d$y, d$i, d$j, leverages = "jla",
                          jla_draws = 8L, seed = 42L, verbose = TRUE),
    type = "message"
  )
  expect_match(progress[[1]], "building the leave-out connected set")
  expect_true(any(grepl("JLA draws 8/8", progress, fixed = TRUE)))
  expect_identical(loud, quiet)
})

test_that("controls are partialled out and factor ids are accepted", {
  d <- felsdv()
  fit <- xhdfe_akm_kss(d$y, factor(paste0("w", d$i)), factor(paste0("f", d$j)),
                       X = cbind(d$x1, d$x2), leverages = "exact")
  expect_true(fit$converged)
  expect_length(fit$beta, 2L)
  expect_gte(fit$fwl_threads_used, 1L)
  expect_gte(fit$threads_used, 1L)
  expect_true(all(is.finite(unlist(fit$kss))))
  ref <- xhdfe_akm_kss(d$y, d$i, d$j, X = cbind(d$x1, d$x2), leverages = "exact")
  expect_equal(fit$kss, ref$kss, tolerance = 1e-12)
})

test_that("eigen diagnostics (AM weak-id CIs) are returned and coherent", {
  d <- akm_sim_panel()
  fit <- xhdfe_akm_kss(d$y, d$worker, d$firm, leverages = "exact",
                       eigen_diagnostics = TRUE)
  expect_true("weak_id" %in% names(fit))
  wk <- fit$weak_id$var_psi
  expect_true(is.finite(wk$lambda1))
  expect_true(is.finite(wk$ci_lb) && is.finite(wk$ci_ub))
  expect_lt(wk$ci_lb, wk$ci_ub)
  th <- fit$component_se$theta_var_psi
  expect_lt(wk$ci_lb, th)
  expect_gt(wk$ci_ub, th)
})

test_that("fweights equal the row-expanded run", {
  d <- akm_sim_panel()
  set.seed(7)
  w <- sample(1:3, length(d$y), replace = TRUE)
  idx <- rep(seq_along(d$y), w)
  fw_fit <- xhdfe_akm_kss(d$y, d$worker, d$firm, leverages = "exact",
                          fweights = w)
  ex_fit <- xhdfe_akm_kss(d$y[idx], d$worker[idx], d$firm[idx],
                          leverages = "exact")
  for (tab in c("plugin", "agsu", "kss")) {
    for (k in names(fw_fit[[tab]])) {
      expect_equal(fw_fit[[tab]][[k]], ex_fit[[tab]][[k]], tolerance = 1e-12)
    }
  }
  expect_error(xhdfe_akm_kss(d$y, d$worker, d$firm, fweights = w,
                             compute_se = TRUE),
               "fweights")
})

test_that("lowess sigma-tilde option runs and preserves thetas", {
  d <- akm_sim_panel()
  f0 <- xhdfe_akm_kss(d$y, d$worker, d$firm, leverages = "exact",
                      compute_se = TRUE)
  f1 <- xhdfe_akm_kss(d$y, d$worker, d$firm, leverages = "exact",
                      compute_se = TRUE, se_sigma_lowess = TRUE)
  expect_identical(f0$component_se$theta_var_psi,
                   f1$component_se$theta_var_psi)
  expect_true(f1$component_se$se_var_psi > 0)
})
