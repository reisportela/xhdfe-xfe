# Explicit source/native inputs; stdout report, no installation or file writes.
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(3L, 4L) ||
    (length(args) == 4L && args[4] != "control")) {
  stop("usage: Rscript --vanilla iv_conditioning_r.R SOURCE_ROOT NATIVE_SO R_LIBRARY [control]")
}
source_root <- normalizePath(args[1], mustWork = TRUE)
native_path <- normalizePath(args[2], mustWork = TRUE)
dependency_library <- normalizePath(args[3], mustWork = TRUE)
control <- length(args) == 4L
.libPaths(c(dependency_library, .libPaths()))
library(Rcpp)
source_files <- file.path(source_root, "r/xhdfe/R",
                          c("RcppExports.R", "utils.R", "estimate.R", "xhdfe.R"))
inputs <- c(native_path, source_files)
before <- tools::md5sum(inputs)
stopifnot(!anyNA(before))
api <- new.env(parent = globalenv())
dll <- dyn.load(native_path)
for (symbol in getDLLRegisteredRoutines(dll)[[".Call"]]) {
  assign(symbol$name, symbol, envir = api)
}
for (path in source_files) sys.source(path, envir = api)
cat("SOURCE_ROOT", source_root, "\nNATIVE_MODULE", dll[["path"]],
    "\nNATIVE_MD5", unname(before[1]), "\n")

row <- 0:127
data <- data.frame(w = 2 * (row %% 2) - 1,
                   t = 2 * ((row %/% 2) %% 2) - 1,
                   v = 2 * ((row %/% 4) %% 2) - 1,
                   e = 2 * ((row %/% 8) %% 2) - 1,
                   aw = 1 + row %/% 16, cl = row %% 7)
data$x <- data$t + data$v
data$y <- 2 * data$w + 3 * data$x + data$e

# The oracle uses the exact score span [w,t], never the ill-conditioned z.
oracle <- function(weighted, vce) {
  n <- nrow(data)
  weight <- if (weighted) data$aw * (n / sum(data$aw)) else rep(1, n)
  score <- cbind(data$w, data$t)
  actual <- cbind(data$w, data$x)
  bread <- solve(crossprod(score, weight * actual))
  df <- n - 2
  rss <- sum(weight * data$e^2)
  if (vce == "iid") {
    covariance <- rss / df * bread
  } else {
    scores <- score * (weight * data$e)
    if (vce == "robust") {
      meat <- crossprod(scores)
      correction <- n / df
    } else {
      totals <- rowsum(scores, data$cl, reorder = FALSE)
      meat <- crossprod(totals)
      correction <- 7 / 6 * (n - 1) / df
      df <- 6
    }
    covariance <- correction * bread %*% meat %*% t(bread)
  }
  list(beta = c(2, 3), vcov = covariance, rss = rss, df = df)
}

exponents <- if (control) 0 else c(0, 10, 20, 25, 27)
passed <- failed <- 0L
for (exponent in exponents) {
  data$z <- data$w + 2^(-exponent) * data$t
  for (weighted in c(FALSE, TRUE)) {
    for (mode in c("xhdfe-fast", "reghdfe-comparable")) {
      for (vce in c("iid", "robust", "cluster")) {
        label <- paste0("delta=2^-", exponent, " weighted=", weighted,
                        " mode=", mode, " vce=", vce)
        result <- tryCatch({
          reference <- oracle(weighted, vce)
          model <- api$xhdfe(y ~ 0 + w | x ~ z, data, vcov = vce,
                             cluster = if (vce == "cluster") "cl" else NULL,
                             weights = if (weighted) "aw" else NULL,
                             weights_type = "analytic", threads = 2,
                             drop_singletons = FALSE, backend = "cpu",
                             tolerance_mode = mode)
          stopifnot(identical(names(model$coefficients), c("w", "x")),
                    identical(dim(model$vcov), c(2L, 2L)),
                    all(is.finite(c(model$coefficients, model$vcov, model$se,
                                    model$residuals, model$rss, model$df_r))),
                    model$nobs == 128, model$df_a == 0,
                    identical(as.integer(model$sample), 1:128),
                    isTRUE(model$converged), isTRUE(model$precision_certified),
                    !isTRUE(model$gpu_used), model$df_r == reference$df)
          scale <- sqrt(outer(diag(reference$vcov), diag(reference$vcov)))
          beta_error <- max(abs(model$coefficients - reference$beta) /
                              pmax(1, abs(reference$beta)))
          covariance_error <- max(abs(model$vcov - reference$vcov) / scale)
          se_error <- max(abs(model$se - sqrt(diag(reference$vcov))) /
                            sqrt(diag(reference$vcov)))
          residual_error <- max(abs(model$residuals - data$e))
          rss_error <- abs(model$rss - reference$rss)
          stopifnot(beta_error <= 1e-9, covariance_error <= 1e-8,
                    se_error <= 1e-8, residual_error <= 1e-7, rss_error <= 1e-8)
          sprintf("PASS %s b=%.9g V=%.9g se=%.9g residual=%.9g RSS=%.9g",
                  label, beta_error, covariance_error, se_error,
                  residual_error, rss_error)
        }, error = function(error) {
          paste("FAIL", label, gsub("[\r\n]", " ", conditionMessage(error)))
        })
        cat(result, "\n")
        if (startsWith(result, "PASS ")) passed <- passed + 1L else failed <- failed + 1L
      }
    }
  }
}
unchanged <- identical(before, tools::md5sum(inputs))
status <- if (failed == 0L && unchanged) "PASS" else "FAIL"
cat("IV_CONDITIONING_R", status, "cases=", passed + failed,
    "passed=", passed, "failed=", failed, "control=", control,
    "inputs_unchanged=", unchanged, "\n")
quit(status = as.integer(status != "PASS"))
