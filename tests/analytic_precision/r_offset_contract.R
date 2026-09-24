args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
root <- normalizePath(args[1], mustWork = TRUE)
.libPaths(c(normalizePath(args[3], mustWork = TRUE), .libPaths()))
library(Rcpp)
api <- new.env(parent = globalenv())
dll <- dyn.load(normalizePath(args[2], mustWork = TRUE))
for (symbol in getDLLRegisteredRoutines(dll)[[".Call"]]) {
  assign(symbol$name, symbol, envir = api)
}
for (name in c("RcppExports.R", "utils.R", "estimate.R", "xhdfe.R",
               "predict.R", "methods.R")) {
  sys.source(file.path(root, "r/xhdfe/R", name), envir = api)
}
set.seed(71)
n <- 1500L
d <- data.frame(x = rnorm(n), o = rnorm(n), q = rnorm(n),
                w = runif(n, .5, 2), f = rep(1:20, length.out = n))
d$o <- d$o + 3*d$x
d$y <- .7*d$x + d$o + d$q + cos(d$f) + rnorm(n)
d$o[17] <- NA_real_
d$adjusted <- d$y - (d$o + d$q)
for (fe in c(FALSE, TRUE)) for (weighted in c(FALSE, TRUE)) {
  fml <- if (fe) y ~ x + offset(o) + offset(q) | f else y ~ x + offset(o) + offset(q)
  adjusted_fml <- if (fe) adjusted ~ x | f else adjusted ~ x
  reference_fml <- if (fe) y ~ x + offset(o) + offset(q) + factor(f) else y ~ x + offset(o) + offset(q)
  w <- if (weighted) d$w else NULL
  fit <- api$xhdfe(fml, d, weights = w, threads = 1)
  adjusted <- api$xhdfe(adjusted_fml, d, weights = w, threads = 1)
  oracle_weights <- w
  reference <- lm(reference_fml, d, weights = oracle_weights, na.action = na.exclude)
  stopifnot(abs(fit$coefficients["x"] - coef(reference)["x"]) < 1e-9,
            abs(fit$vcov["x", "x"] - vcov(reference)["x", "x"]) <=
              1e-8*vcov(reference)["x", "x"],
            max(abs(fit$residuals - residuals(reference)), na.rm = TRUE) < 1e-8,
            identical(fit$sample, adjusted$sample), !(17 %in% fit$sample),
            identical(fit$coefficients, adjusted$coefficients),
            identical(fit$vcov, adjusted$vcov), identical(fit$tss, adjusted$tss),
            max(abs(api$predict.xhdfe(fit, type = "xbd") - fitted(reference)), na.rm = TRUE) < 1e-8)
  expected <- api$predict.xhdfe(adjusted, d) + d$o + d$q
  stopifnot(isTRUE(all.equal(api$predict.xhdfe(fit, d), expected, tolerance = 1e-12)),
            is.na(api$predict.xhdfe(fit)[17]))
}
cat("PASS: R offsets, weighted/unweighted, FE, missing sample, predictions and lm oracle\n")
