#!/usr/bin/env Rscript
## Gelbach (2016) conditional decomposition worked example for xhdfe (R).
##
## Decomposes the movement of a base coefficient (the return to education)
## between a short regression (y on educ) and a long regression (y on educ plus
## mediating channels and a firm fixed effect) into an additive,
## order-invariant contribution per channel. Semantics match Gelbach's b1x2.
##
## Run with:  Rscript examples/gelbach_example.R
## Requires the installed xhdfe R package.

library(xhdfe)
set.seed(20260706)

n <- 5000L
n_firms <- 60L

## educ is correlated with ability and with firm sorting, so the short-regression
## return to education is inflated by both channels.
ability <- rnorm(n)
educ <- 0.5 * ability + rnorm(n)
latent <- 0.4 * educ + rnorm(n)
firm_id <- floor((n_firms - 1) / (1 + exp(-latent)))
tenure <- rnorm(n) + 0.2 * educ
exper <- rnorm(n)
firmpay <- rnorm(n_firms, 0, 0.7)[firm_id + 1L]
y <- 0.5 * educ + 0.8 * ability + 0.1 * tenure + firmpay + rnorm(n)

## Gelbach decomposition of the education coefficient into a skill channel
## (ability), a job channel (tenure, exper) and a firm fixed-effect channel.
res <- xhdfe_gelbach(
  y, x1 = educ,
  x2_groups = list(skill = ability, job = cbind(tenure, exper)),
  fes = list(firm = firm_id)
)

## res$delta / res$se are matrices: rows = c("x1_1", ..., "_cons"), cols = groups.
## For the single base regressor educ, use row "x1_1".
cat(sprintf("base (short) educ coefficient: %.5f\n", res$b_base[1]))
cat(sprintf("full (long)  educ coefficient: %.5f\n", res$b_full[1]))
cat(sprintf("total movement:                %.5f (se %.5f)\n",
            res$total[1], res$total_se[1]))
cat("\ncontribution of each channel to the movement:\n")
for (name in colnames(res$delta)) {
  cat(sprintf("  %-6s: %+.5f  (se %.5f)\n", name,
              res$delta["x1_1", name], res$se["x1_1", name]))
}
cat(sprintf("\nsummation identity residual = %.2e\n", res$identity_gap))

## Cluster-robust inference by firm.
res_cl <- xhdfe_gelbach(
  y, x1 = educ,
  x2_groups = list(skill = ability, job = cbind(tenure, exper)),
  fes = list(firm = firm_id),
  vce = "cluster", cluster = firm_id
)
cat("\ncluster-robust standard errors:\n")
for (name in colnames(res_cl$delta)) {
  cat(sprintf("  %-6s: se %.5f\n", name, res_cl$se["x1_1", name]))
}

cat("\ndone.\n")
