# Run against explicit source, native-module and dependency-library paths.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("usage: Rscript --vanilla r_prediction_metadata_contract.R SOURCE_ROOT NATIVE_SO R_LIBRARY")
}
source_root <- normalizePath(args[1], mustWork = TRUE)
native_path <- normalizePath(args[2], mustWork = TRUE)
dependency_library <- normalizePath(args[3], mustWork = TRUE)
.libPaths(c(dependency_library, .libPaths()))
library(Rcpp)

source_files <- file.path(source_root, "r/xhdfe/R",
                          c("RcppExports.R", "utils.R", "estimate.R",
                            "xhdfe.R", "predict.R", "methods.R"))
inputs <- c(native_path, source_files)
hashes_before <- tools::md5sum(inputs)
stopifnot(!anyNA(hashes_before))
api <- new.env(parent = globalenv())
dll <- dyn.load(native_path)
for (symbol in getDLLRegisteredRoutines(dll)[[".Call"]]) {
  assign(symbol$name, symbol, envir = api)
}
for (path in source_files) sys.source(path, envir = api)
cat("SOURCE_ROOT", source_root, "\nNATIVE_MODULE", dll[["path"]],
    "\nNATIVE_MD5", unname(hashes_before[1]), "\n")

assert_close <- function(actual, expected, label, tolerance = 1e-10) {
  stopifnot(length(actual) == length(expected),
            identical(as.vector(is.na(actual)), as.vector(is.na(expected))))
  keep <- !is.na(expected)
  stopifnot(all(is.finite(actual[keep])), all(is.finite(expected[keep])))
  error <- if (any(keep)) max(abs(actual[keep] - expected[keep])) else 0
  scale <- max(1, abs(expected[keep]))
  if (error > tolerance * scale) {
    stop(label, ": max absolute error ", format(error, digits = 17))
  }
  invisible(error)
}

fit <- function(formula, data) {
  model <- api$xhdfe(formula, data, threads = 1, backend = "cpu")
  stopifnot(isTRUE(model$converged), isTRUE(model$precision_certified))
  model
}

# Independent explicit 2SLS oracle, with an intercept in the final column.
iv_oracle <- function(y, X, instruments) {
  projected <- qr.fitted(qr(instruments), X)
  beta <- qr.coef(qr(projected), y)
  residual <- y - drop(X %*% beta)
  variance <- sum(residual^2) / (length(y) - ncol(X))
  list(coefficients = beta,
       vcov = variance * chol2inv(chol(crossprod(projected))))
}

set.seed(32)
data <- data.frame(x = seq(-2, 2, length.out = 80), z = rnorm(80),
                   g = factor(rep(letters[1:4], 20)))
data$y <- 2 + 1.25 * data$x - .7 * data$x^2 + .3 * data$z +
  c(0, .2, -.3, .1)[as.integer(data$g)] + rnorm(80, sd = .2)
rows <- c(1, 4, 8, 12, 20, 39, 63, 80)
cases <- list(linear = y ~ x + z, factor = y ~ x + g,
              poly = y ~ poly(x, 2), bs = y ~ splines::bs(x, df = 4),
              ns = y ~ splines::ns(x, df = 4))

for (name in names(cases)) {
  formula <- cases[[name]]
  model <- fit(formula, data)
  oracle <- lm(formula, data)
  coefficient_names <- names(model$coefficients)
  beta_error <- assert_close(unname(model$coefficients),
                             unname(coef(oracle)[coefficient_names]),
                             paste(name, "beta"))
  covariance_error <- assert_close(unname(model$vcov),
                                   unname(vcov(oracle)[coefficient_names,
                                                       coefficient_names]),
                                   paste(name, "covariance"))
  all_error <- assert_close(api$predict.xhdfe(model, newdata = data),
                            model$xb_cache, paste(name, "all rows"))
  subset <- data[rows, , drop = FALSE]
  subset_error <- assert_close(api$predict.xhdfe(model, newdata = subset),
                               model$xb_cache[rows], paste(name, "subset"))
  assert_close(unname(api$predict.xhdfe(model, newdata = subset)),
               unname(predict(oracle, subset)), paste(name, "predict.lm"))
  subset$x[3] <- NA_real_
  assert_close(unname(api$predict.xhdfe(model, newdata = subset)),
               unname(predict(oracle, subset)), paste(name, "missing predictor"))
  if (name == "factor") {
    subset <- droplevels(data[which(data$g %in% c("a", "b"))[1:12], ])
    subset$g[2] <- NA
    previous_contrasts <- getOption("contrasts")
    options(contrasts = c("contr.helmert", "contr.poly"))
    assert_close(unname(api$predict.xhdfe(model, newdata = subset)),
                 unname(predict(oracle, subset)), "factor levels and contrasts")
    options(contrasts = previous_contrasts)
  }
  cat(sprintf("PASS %s beta=%.3g covariance=%.3g all=%.3g subset=%.3g NA=PASS\n",
              name, beta_error, covariance_error, all_error, subset_error))
}

options(contrasts = c("contr.sum", "contr.poly"))
data2 <- data.frame(x = rnorm(300), g = factor(rep(letters[1:3], 100)))
data2$z1 <- as.numeric(data2$g == "a") + rnorm(300, sd = .02)
data2$z2 <- as.numeric(data2$g == "b") + rnorm(300, sd = .02)
data2$y <- 2 + .2 * data2$x + c(1.25, -.7, -.55)[as.integer(data2$g)] +
  rnorm(300, sd = .02)
model <- fit(y ~ x | g ~ z1 + z2, data2)
X <- cbind(x = data2$x, model.matrix(~ g, data2)[, -1, drop = FALSE],
           "(Intercept)" = 1)
Z <- cbind(data2$x, data2$z1, data2$z2, 1)
oracle <- iv_oracle(data2$y, X, Z)
beta_error <- assert_close(unname(model$coefficients),
                           unname(oracle$coefficients), "IV factor beta")
covariance_error <- assert_close(unname(model$vcov), unname(oracle$vcov),
                                 "IV factor covariance")
options(contrasts = c("contr.helmert", "contr.poly"))
# The existing combined xlevels emits warnings for variables absent on one side.
predict_iv <- function(model, newdata) {
  suppressWarnings(api$predict.xhdfe(model, newdata = newdata))
}
all_error <- assert_close(predict_iv(model, data2), model$xb_cache,
                          "IV changed contrasts all rows")
subset_error <- assert_close(predict_iv(model, data2[rows, , drop = FALSE]),
                             model$xb_cache[rows], "IV changed contrasts subset")
subset <- data2[rows, , drop = FALSE]
subset$g[2] <- NA
expected <- model$xb_cache[rows]
expected[2] <- NA_real_
assert_close(predict_iv(model, subset), expected, "IV factor missing predictor")
cat(sprintf("PASS endo_contrasts beta=%.3g covariance=%.3g all=%.3g subset=%.3g NA=PASS\n",
            beta_error, covariance_error, all_error, subset_error))

data3 <- data.frame(x = rnorm(300), q = seq(-2, 2, length.out = 300))
data3$z1 <- data3$q + rnorm(300, sd = .03)
data3$z2 <- data3$q^2 + rnorm(300, sd = .03)
data3$y <- 2 + .2 * data3$x + 1.25 * data3$q - .7 * data3$q^2 +
  rnorm(300, sd = .02)
model <- fit(y ~ x | poly(q, 2) ~ z1 + z2, data3)
X <- cbind(x = data3$x, model.matrix(~ poly(q, 2), data3)[, -1, drop = FALSE],
           "(Intercept)" = 1)
Z <- cbind(data3$x, data3$z1, data3$z2, 1)
oracle <- iv_oracle(data3$y, X, Z)
beta_error <- assert_close(unname(model$coefficients),
                           unname(oracle$coefficients), "IV polynomial beta")
covariance_error <- assert_close(unname(model$vcov), unname(oracle$vcov),
                                 "IV polynomial covariance")
all_error <- assert_close(predict_iv(model, data3), model$xb_cache,
                          "IV polynomial all rows")
subset_error <- assert_close(predict_iv(model, data3[rows, , drop = FALSE]),
                             model$xb_cache[rows], "IV polynomial subset")
subset <- data3[rows, , drop = FALSE]
subset$q[3] <- NA_real_
expected <- model$xb_cache[rows]
expected[3] <- NA_real_
assert_close(predict_iv(model, subset), expected, "IV polynomial missing predictor")
cat(sprintf("PASS endo_poly beta=%.3g covariance=%.3g all=%.3g subset=%.3g NA=PASS\n",
            beta_error, covariance_error, all_error, subset_error))

stopifnot(identical(hashes_before, tools::md5sum(inputs)))
cat("R_PREDICTION_METADATA_CONTRACT PASS 7/7; source/native hashes unchanged\n")
