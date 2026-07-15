# R <-> Python parity: the R binding must reproduce the reference Python
# module (same C++ core, same compiler flags) on an identical dataset.
# Tolerances sit above the documented run-to-run OpenMP floor (~1e-13) and
# far below any econometrically meaningful difference.

skip_if_not_installed("jsonlite")
fixture_path <- test_path("fixtures", "parity_reference.json")
skip_if_not(file.exists(fixture_path), "parity fixture not generated")

ref_all <- jsonlite_fromJSON_lite(fixture_path)
dat <- make_parity_data()
d <- dat$main
g <- dat$grouped

TOL_COEF <- 1e-9
TOL_STAT <- 1e-7

expect_digest <- function(x, dig, tol = 1e-7, label = "digest") {
  if (is.null(dig) || dig$n == 0) {
    expect_true(length(x) == 0 || is.null(x), label = label)
    return(invisible())
  }
  expect_equal(length(x), dig$n, label = paste(label, "n"))
  expect_lt(abs(sum(x) - dig$sum), tol * max(1, abs(dig$sum)))
  expect_lt(abs(sum(x^2) - dig$sumsq), tol * max(1, abs(dig$sumsq)))
}

run_spec <- function(name, ...) {
  ref <- ref_all$specs[[name]]
  expect_false(is.null(ref), label = paste("fixture entry", name))
  m <- xhdfe_fit(...)
  expect_equal(unname(m$coefficients), ref$coef, tolerance = TOL_COEF,
               label = paste(name, "coef"))
  expect_equal(unname(m$se), ref$se, tolerance = TOL_STAT,
               label = paste(name, "se"))
  expect_equal(unname(m$pvalues), ref$pvalues, tolerance = 1e-6,
               label = paste(name, "pvalues"))
  expect_equal(unname(m$conf_int), ref$conf_int,
               tolerance = TOL_STAT, label = paste(name, "conf_int"),
               ignore_attr = TRUE)
  expect_identical(m$nobs, ref$nobs)
  expect_identical(m$nobs_full, ref$nobs_full)
  expect_identical(m$num_singletons, ref$num_singletons)
  expect_equal(m$df_r, ref$df_resid)
  expect_equal(m$df_r_unadj, ref$df_resid_unadj)
  expect_equal(m$df_m, ref$df_m)
  expect_equal(m$df_a, ref$df_a)
  expect_equal(m$df_a_levels, ref$df_a_levels)
  expect_equal(m$df_a_exact, ref$df_a_exact)
  expect_equal(m$df_a_nested, ref$df_a_nested)
  expect_equal(m$r2, ref$r2, tolerance = TOL_STAT)
  expect_equal(m$r2_within, ref$r2_within, tolerance = TOL_STAT)
  expect_equal(m$rss, ref$rss, tolerance = TOL_STAT)
  expect_equal(m$tss, ref$tss, tolerance = TOL_STAT)
  expect_equal(m$tss_within, ref$tss_within, tolerance = TOL_STAT)
  expect_identical(isTRUE(m$saturated), isTRUE(ref$saturated))
  expect_identical(isTRUE(m$converged), isTRUE(ref$converged))
  expect_identical(m$iterations, ref$num_iterations)
  expect_identical(m$fe_num_levels, as.integer(ref$fe_num_levels))
  expect_identical(m$num_clusters, ref$num_clusters)
  expect_equal(m$cluster_scale, ref$cluster_scale, tolerance = TOL_STAT)
  # Full-length residuals: compare on the estimation sample only.
  expect_digest(m$residuals[!is.na(m$residuals)], ref$residual_digest,
                tol = 1e-7, label = paste(name, "residuals"))
  invisible(m)
}

X2 <- as.matrix(d[, c("x1", "x2")])
XIV <- as.matrix(d[, c("x1", "x2", "xend")])
Z <- as.matrix(d[, c("ze1", "ze2")])

test_that("parity: plain OLS and 2-3 way FEs", {
  run_spec("nofe_unadj", d$y, X2)
  run_spec("fe2_unadj", d$y, X2, fes = list(d$id1, d$id2))
  run_spec("fe2_robust", d$y, X2, fes = list(d$id1, d$id2), vcov = "robust")
  run_spec("fe2_cluster1", d$y, X2, fes = list(d$id1, d$id2),
           cluster = list(d$id1))
  # The CGM PSD warning is expected here (mirrors reghdfe).
  suppressWarnings(
    run_spec("fe3_cluster2", d$y, X2, fes = list(d$id1, d$id2, d$id3),
             cluster = list(d$id1, d$id2))
  )
})

test_that("parity: weights", {
  run_spec("fe2_weights_robust", d$y, X2, fes = list(d$id1, d$id2),
           weights = d$w, vcov = "robust")
})

test_that("frequency weights match exact replicated-rows equivalence", {
  # fweight = k is definitionally identical to repeating the row k times;
  # coefficients, SEs, dof and fit stats must all agree.
  idx <- rep(seq_len(nrow(d)), times = d$wf)
  m_fw <- xhdfe_fit(d$y, X2, fes = list(d$id1, d$id2),
                    weights = d$wf, weights_type = "frequency",
                    vcov = "robust")
  m_rep <- xhdfe_fit(d$y[idx], X2[idx, ], fes = list(d$id1[idx], d$id2[idx]),
                     vcov = "robust")
  expect_equal(m_fw$coefficients, m_rep$coefficients, tolerance = 1e-9)
  expect_equal(m_fw$se, m_rep$se, tolerance = 1e-7)
  expect_equal(m_fw$df_r, m_rep$df_r)
  expect_equal(m_fw$rss, m_rep$rss, tolerance = 1e-7)
  expect_equal(m_fw$tss, m_rep$tss, tolerance = 1e-7)
  expect_equal(m_fw$nobs_effective, m_rep$nobs, tolerance = 1e-9)
})

test_that("parity: IV/2SLS", {
  run_spec("iv_fe2_cluster", d$y, XIV, fes = list(d$id1, d$id2),
           cluster = list(d$id1), instruments = Z, endogenous = 3)
})

test_that("parity: heterogeneous slopes", {
  run_spec("slopes_intercept", d$y, X2, fes = list(d$id1, d$id2),
           cluster = list(d$id1),
           slopes = list(list(fe = 2, values = d$z, include_intercept = TRUE)))
  run_spec("slopes_only", d$y, X2, fes = list(d$id1, d$id2), vcov = "robust",
           slopes = list(list(fe = 2, values = d$z, include_intercept = FALSE)))
})

test_that("parity: singleton handling", {
  run_spec("keepsing", d$y, X2, fes = list(d$id_hi, d$id2), vcov = "robust",
           keep_singletons = TRUE)
  run_spec("dropsing", d$y, X2, fes = list(d$id_hi, d$id2), vcov = "robust")
})

test_that("parity: DoF and SSC controls", {
  run_spec("dof_none", d$y, X2, fes = list(d$id1, d$id2),
           cluster = list(d$id1), dof = "none")
  run_spec("ssc_variants", d$y, X2, fes = list(d$id1, d$id2),
           cluster = list(d$id1),
           ssc = list(ssc_k_fixef = "nonnested", ssc_g_df = "conventional",
                      ssc_k_adj = FALSE))
})

test_that("parity: tolerance modes and reporting", {
  run_spec("tolmode_fast", d$y, X2, fes = list(d$id1, d$id2),
           tolerance_mode = "xhdfe-fast")
  run_spec("tolmode_strict", d$y, X2, fes = list(d$id1, d$id2),
           tolerance_mode = "strict-residual")
  run_spec("noconstant", d$y, X2, fit_intercept = FALSE)
  run_spec("level90", d$y, X2, fes = list(d$id1, d$id2), vcov = "robust",
           level = 90)
})

test_that("parity: absorption method overrides", {
  run_spec("method_mlsmr", d$y, X2, fes = list(d$id1, d$id2),
           absorption_method = "mlsmr")
  run_spec("method_symgs", d$y, X2, fes = list(d$id1, d$id2),
           absorption_method = "symmetric-gauss-seidel",
           symmetric_sweep = TRUE)
})

test_that("parity: FE recovery and mobility groups", {
  ref <- ref_all$specs[["savefe"]]
  m <- run_spec("savefe", d$y, X2, fes = list(d$id1, d$id2), save_fe = TRUE)
  expect_length(m$fe_effects, length(ref$fe_effect_digests))
  for (i in seq_along(m$fe_effects)) {
    expect_digest(m$fe_effects[[i]], ref$fe_effect_digests[[i]],
                  label = sprintf("fe_effects[%d]", i))
  }
  expect_identical(isTRUE(m$fe_recovery_converged),
                   isTRUE(ref$fe_recovery_converged))

  ref2 <- ref_all$specs[["groupvar"]]
  m2 <- run_spec("groupvar", d$y, X2, fes = list(d$id1, d$id2),
                 groupvar = TRUE)
  expect_digest(m2$groupvar, ref2$groupvar_digest, label = "groupvar")
})

test_that("parity: group-level outcomes", {
  ref <- ref_all$specs[["group_only"]]
  m <- xhdfe_fit(g$yg, as.matrix(g[, "xg", drop = FALSE]),
                 fes = list(g$year), group = g$grp)
  expect_equal(unname(m$coefficients), ref$coef, tolerance = TOL_COEF)
  expect_equal(unname(m$se), ref$se, tolerance = TOL_STAT)
  expect_identical(m$nobs, ref$nobs)
  expect_equal(m$r2, ref$r2, tolerance = TOL_STAT)
  expect_equal(m$rss, ref$rss, tolerance = TOL_STAT)
  expect_digest(m$residuals[!is.na(m$residuals)], ref$residual_digest,
                label = "group residuals")

  ref2 <- ref_all$specs[["group_individual"]]
  m2 <- xhdfe_fit(g$yg, as.matrix(g[, "xg", drop = FALSE]),
                  fes = list(g$ind), group = g$grp, individual = g$ind)
  expect_equal(unname(m2$coefficients), ref2$coef, tolerance = TOL_COEF)
  expect_equal(unname(m2$se), ref2$se, tolerance = TOL_STAT)
  expect_identical(m2$nobs, ref2$nobs)
  expect_equal(m2$rss, ref2$rss, tolerance = TOL_STAT)

  ref3 <- ref_all$specs[["group_fes_decomposition"]]
  dec <- xhdfe_group_fes(g$yg, as.matrix(g[, "xg", drop = FALSE]),
                         fes = list(g$ind), group = g$grp,
                         individual = g$ind)
  expect_identical(isTRUE(dec$converged), isTRUE(ref3$converged))
  expect_identical(dec$iterations, ref3$iterations)
  expect_true(all(is.finite(dec$individual_effects)))
  expect_setequal(dec$individual_ids, unique(g$ind))

  # Raw individual effects are not identified in this fixture: the dense
  # design has four null directions, so native and portable builds can choose
  # different representatives while producing the same group projection.
  # Validate that identified projection against an independent dense QR
  # oracle instead of pinning a build-dependent normalization.
  group_ids <- unique(g$grp)
  incidence <- matrix(0, nrow = length(group_ids),
                      ncol = length(dec$individual_ids))
  incidence[cbind(match(g$grp, group_ids),
                  match(g$ind, dec$individual_ids))] <- 1
  incidence <- incidence / rowSums(incidence)
  group_rows <- !duplicated(g$grp)
  design <- cbind(1, g$xg[group_rows], incidence)
  oracle <- lm.fit(design, g$yg[group_rows])
  expect_identical(ncol(design) - oracle$rank, 4L)

  projected <- unname(m2$coefficients["(Intercept)"]) +
    unname(m2$coefficients["xg"]) * g$xg[group_rows] +
    drop(incidence %*% dec$individual_effects)
  identified_mse <- mean((projected - oracle$fitted.values)^2)
  expect_lte(dec$mse, 1e-9)
  expect_lte(identified_mse, 1e-9)
  expect_lt(abs(identified_mse - dec$mse), 1e-12)
  expect_lt(abs(dec$mse - ref3$mse), 1e-12)
})
