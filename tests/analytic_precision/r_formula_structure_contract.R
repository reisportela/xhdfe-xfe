# R formula parsing and retained bootstrap sample against explicit designs.
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
               "predict.R", "methods.R", "akm.R", "gelbach.R", "gelbach_features.R")) {
  sys.source(file.path(root, "r/xhdfe/R", name), envir = api)
}
set.seed(91)
n <- 1800L
d <- data.frame(x = rnorm(n), z = rnorm(n), s = rnorm(n),
                a = rep(1:6, length.out = n), b = sample(1:5, n, TRUE),
                c = sample(1:4, n, TRUE))
d$y <- .7 * d$x + .3 * d$z + sin(d$a * d$b) * d$s + cos(d$a) + rnorm(n)
oracle <- function(design) {
  fit <- lm.fit(design, d$y)
  q <- fit$qr
  bread <- chol2inv(qr.R(q)[seq_len(q$rank), seq_len(q$rank), drop = FALSE])
  position <- match(1L, q$pivot[seq_len(q$rank)])
  c(b = unname(fit$coefficients[1]), v = sum(fit$residuals^2) / fit$df.residual * bread[position, position])
}
ab <- interaction(d$a, d$b, drop = TRUE)
cases <- list(
  list(y ~ x | a ^ b, model.matrix(~ x + ab, d)),
  list(y ~ x | (a ^ b), model.matrix(~ x + ab, d)),
  list(y ~ x | (a + b), model.matrix(~ x + factor(a) + factor(b), d)),
  list(y ~ x | a + (b + c), model.matrix(~ x + factor(a) + factor(b) + factor(c), d)),
  list(y ~ x | a ^ b[s], model.matrix(~ x + ab + ab:s, d)),
  list(y ~ x | (a ^ b)[s], model.matrix(~ x + ab + ab:s, d)),
  list(y ~ x | a ^ b[[s]], cbind(x = d$x, 1, model.matrix(~ 0 + ab) * d$s)),
  list(y ~ x | (a ^ b)[[s]], cbind(x = d$x, 1, model.matrix(~ 0 + ab) * d$s))
)
for (case in cases) {
  design <- case[[2]]
  ix <- match("x", colnames(design))
  design <- design[, c(ix, setdiff(seq_len(ncol(design)), ix)), drop = FALSE]
  ref <- oracle(design)
  model <- suppressWarnings(api$xhdfe(case[[1]], d, threads = 2, backend = "cpu"))
  stopifnot(isTRUE(model$converged), abs(model$coefficients[1] - ref["b"]) <= 1e-9,
            abs(model$vcov[1, 1] - ref["v"]) <= 1e-8 * ref["v"])
  cat("PASS", deparse(case[[1]]), "\n")
}
dd <- d[, c("y", "x", "z", "a")]
dot <- api$xhdfe(y ~ . | a, dd, threads = 2)
explicit <- api$xhdfe(y ~ x + z | a, dd, threads = 2)
stopifnot(identical(names(dot$coefficients), names(explicit$coefficients)),
          max(abs(dot$coefficients - explicit$coefficients)) <= 1e-12,
          max(abs(api$predict.xhdfe(dot, dd) - api$predict.xhdfe(explicit, dd))) <= 1e-12)
cat("PASS dot expansion excludes response and fixed effects; prediction parity\n")

nkeep <- 400L
f <- c(rep(1:20, each = 20), 21:100)
x <- rnorm(length(f)); z <- rnorm(length(f)); y <- .7 * x + .3 * z + rnorm(100)[f] + rnorm(length(f))
y[-seq_len(nkeep)] <- y[-seq_len(nkeep)] + 20 * x[-seq_len(nkeep)] + 100
for (method in c("pairs", "cluster_pairs")) {
  boot <- function(index) suppressWarnings(api$xhdfe_gelbach_bootstrap(
    y[index], x[index], list(z = z[index]), list(fe = f[index]),
    method = method, bootstrap_cluster = if (method == "cluster_pairs") f[index] else NULL,
    reps = 12, min_valid_reps = 10, seed = 381, num_threads = 2, sample_info = TRUE))
  full <- boot(seq_along(y)); trimmed <- boot(seq_len(nkeep))
  stopifnot(identical(full$sample_index, 0L:(nkeep - 1L)),
            full$bootstrap$n_rows_population == nkeep)
  for (key in names(full$bootstrap$draws)) {
    stopifnot(isTRUE(all.equal(full$bootstrap$draws[[key]],
                               trimmed$bootstrap$draws[[key]], tolerance = 1e-12)))
  }
  cat("PASS retained-sample bootstrap", method, "\n")
}
cat("R_FORMULA_STRUCTURE_CONTRACT PASS\n")
