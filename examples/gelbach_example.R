#!/usr/bin/env Rscript
## Gelbach (2016) conditional decomposition worked example for xhdfe (R).
##
## Decomposes the movement of a base coefficient (the education coefficient)
## between a short and a long regression into additive, order-invariant
## contributions from declared covariate and fixed-effect blocks. This is
## specification accounting, not causal mediation. Semantics match b1x2.
##
## Run with:  Rscript examples/gelbach_example.R
## Requires the installed xhdfe R package.

library(xhdfe)
set.seed(20260706)

n <- 5000L
n_firms <- 60L

## Education, job covariates and firm assignment share latent determinants.
## The example does not impose or validate a causal ordering among them.
ability <- rnorm(n)
educ <- 0.5 * ability + rnorm(n)
latent <- 0.4 * ability + rnorm(n)
firm_id <- floor((n_firms - 1) / (1 + exp(-latent)))
tenure <- rnorm(n) + 0.2 * ability
exper <- rnorm(n)
firmpay <- rnorm(n_firms, 0, 0.7)[firm_id + 1L]
y <- 0.5 * educ + 0.8 * ability + 0.1 * tenure + firmpay + rnorm(n)

## Gelbach decomposition into ability, job-covariate and firm-FE blocks.
res <- xhdfe_gelbach(
  y, x1 = educ,
  x2_groups = list(ability = ability,
                   job_covariates = cbind(tenure, exper)),
  fes = list(firm_fe = firm_id)
)

## res$delta / res$se are matrices: rows = c("x1_1", ..., "_cons"), cols = groups.
## For the single base regressor educ, use row "x1_1".
cat(sprintf("base (short) educ coefficient: %.5f\n", res$b_base[1]))
cat(sprintf("full (long)  educ coefficient: %.5f\n", res$b_full[1]))
cat(sprintf("total movement:                %.5f (se %.5f)\n",
            res$total[1], res$total_se[1]))
cat("\ncontribution of each declared block to the movement:\n")
for (name in colnames(res$delta)) {
  cat(sprintf("  %-6s: %+.5f  (se %.5f)\n", name,
              res$delta["x1_1", name], res$se["x1_1", name]))
}
cat(sprintf("\nsummation identity residual = %.2e\n", res$identity_gap))
cat(sprintf("absorbed-FE aggregate (conditional/gamma0 SE): %+.5f (se %.5f)\n",
            res$fe_total$coef[1], res$fe_total$se[1]))
cat("interpretation: coefficient-movement accounting; not causal mediation\n")

## Cluster-robust inference by firm.
res_cl <- xhdfe_gelbach(
  y, x1 = educ,
  x2_groups = list(ability = ability,
                   job_covariates = cbind(tenure, exper)),
  fes = list(firm_fe = firm_id),
  vce = "cluster", cluster = firm_id
)
cat("\ncluster-robust standard errors:\n")
for (name in colnames(res_cl$delta)) {
  cat(sprintf("  %-6s: se %.5f\n", name, res_cl$se["x1_1", name]))
}

cat("\ndone.\n")
