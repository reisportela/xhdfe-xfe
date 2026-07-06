#!/usr/bin/env Rscript
## AKM + leave-out (KSS) worked example for xhdfe (R front-end).
##
## Two-way worker-firm estimation, leave-out variance decomposition, component
## standard errors and Andrews-Mikusheva confidence intervals.
##
## Numerical semantics follow Kline, Saggio & Solvsten (2020) as implemented in
## Saggio's LeaveOutTwoWay (the canonical KSS reference); the leave-out set and
## components are validated against it and against pytwoway.
##
## Run with:  Rscript examples/akm_kss_example.R
## Requires the installed xhdfe R package.

library(xhdfe)
set.seed(20260706)

## 1. A reproducible mover panel: every worker changes firm each period.
simulate_akm_panel <- function(n_workers = 400L, n_firms = 40L, reps = 5L) {
  worker <- rep(seq_len(n_workers), each = reps)
  firm <- integer(n_workers * reps)
  pos <- 1L
  for (w in seq_len(n_workers)) {
    f <- sample.int(n_firms, 1L)
    for (k in seq_len(reps)) {
      firm[pos] <- f
      pos <- pos + 1L
      nxt <- sample.int(n_firms - 1L, 1L)
      f <- if (nxt < f) nxt else nxt + 1L
    }
  }
  alpha <- rnorm(n_workers, 0, 0.6)
  psi <- rnorm(n_firms, 0, 0.4)
  x1 <- rnorm(n_workers * reps)
  y <- alpha[worker] + psi[firm] + 0.3 * x1 + rnorm(n_workers * reps) * 0.5
  list(y = y, worker = worker, firm = firm, x1 = x1)
}

d <- simulate_akm_panel()

## 2. Sample preparation: the leave-one-out connected set (KSS-identified).
s <- xhdfe_akm_leave_out_set(d$worker, d$firm)
cat(sprintf("leave-out sample: %d obs, %d workers (%d movers), %d firms\n",
            s$n_obs, s$n_workers, s$n_movers, s$n_firms))

## 3. AKM two-way estimation + variance decomposition (plug-in / AGSU / KSS).
fit <- xhdfe_akm_kss(d$y, d$worker, d$firm, leverages = "exact")
kss <- fit$kss
cat("\nKSS-corrected decomposition:\n")
cat(sprintf("  var(psi)        = %.5f\n", kss$var_psi))
cat(sprintf("  var(alpha)      = %.5f\n", kss$var_alpha))
cat(sprintf("  cov(alpha,psi)  = %.5f\n", kss$cov_alpha_psi))
cat(sprintf("  corr(alpha,psi) = %.5f\n", kss$corr_alpha_psi))
cat(sprintf("  shares of var(y): alpha %.3f, psi %.3f, 2cov %.3f\n",
            kss$share_var_alpha, kss$share_var_psi, kss$share_2cov))

## 4. Component standard errors + Andrews-Mikusheva confidence intervals.
fit <- xhdfe_akm_kss(d$y, d$worker, d$firm, leverages = "exact",
                     compute_se = TRUE, eigen_diagnostics = TRUE)
se <- fit$component_se
wk <- fit$weak_id$var_psi
cat("\nInference on var(psi):\n")
cat(sprintf("  KSS point est.  = %.5f\n", se$theta_var_psi))
cat(sprintf("  standard error  = %.5f\n", se$se_var_psi))
cat(sprintf("  AM 95%% CI       = [%.5f, %.5f]\n", wk$ci_lb, wk$ci_ub))
cat(sprintf("  F statistic     = %.3f (curvature %.3f)\n", wk$f_stat, wk$curvature))

## 5. Controls partialled out (FWL); pass gpu = TRUE for the CUDA backend.
fit <- xhdfe_akm_kss(d$y, d$worker, d$firm, X = matrix(d$x1, ncol = 1),
                     leverages = "jla", jla_draws = 200L)
cat(sprintf("\ncontrol coefficient on x1: %.5f\n", fit$beta[1]))

cat("\ndone.\n")
