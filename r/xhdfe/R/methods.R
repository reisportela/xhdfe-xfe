# S3 methods for xhdfe objects.

#' Methods for xhdfe models
#'
#' Standard S3 accessors for \code{\link{xhdfe}} fits. \code{print} shows a
#' reghdfe-style header (observations, absorbed dimensions with level
#' counts, standard-error type, convergence, absorber method, GPU use when
#' applicable) followed by the coefficient table and fit statistics.
#' \code{summary} additionally reports sums of squares, the full
#' degrees-of-freedom accounting, cluster diagnostics and backend details.
#' \code{coeftable} returns the numeric coefficient matrix (estimate,
#' standard error, t value, p value). \code{confint} returns the confidence
#' intervals computed by the backend at the estimation \code{level}, or
#' recomputes them from the t distribution with \code{df_r} degrees of
#' freedom for a different \code{level}. \code{logLik}, \code{AIC},
#' \code{BIC} and \code{deviance} use the Gaussian log-likelihood and RSS
#' (the Stata \code{e(ll)} / \code{estat ic} values).
#'
#' @param x,object an \code{xhdfe} model.
#' @param digits significant digits for printing.
#' @param parm,level as in \code{\link[stats]{confint}}; \code{level} may be
#'   a fraction (0.90) or percent (90).
#' @param ... passed on or ignored.
#' @return \code{coef}: named coefficient vector (the intercept, when
#'   reported, is last, as in Stata). \code{vcov}: the variance-covariance
#'   matrix. \code{coeftable}: a k x 4 matrix. \code{confint}: a k x 2
#'   matrix. \code{nobs}, \code{df.residual}, \code{deviance}: scalars.
#'   \code{residuals}: full-length vector (\code{NA} off-sample).
#'   \code{formula}: the model formula.
#' @seealso \code{\link{xhdfe}}, \code{\link{predict.xhdfe}},
#'   \code{\link{fixef.xhdfe}}
#' @name xhdfe-methods
NULL

#' @rdname xhdfe-methods
print.xhdfe <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("xhdfe: linear regression with multiple fixed effects (C++ backend)\n")
  if (!is.null(x$fml)) cat("Formula: ", deparse1(x$fml), "\n", sep = "")
  se_label <- switch(x$se_type,
                     unadjusted = "unadjusted",
                     robust = "robust (HC1)",
                     cluster = paste0("clustered (",
                                      paste(x$cluster_names, collapse = ", "), ")"))
  cat(sprintf("Observations: %s (full: %s, singletons dropped: %s)\n",
              format(x$nobs, big.mark = ","),
              format(x$nobs_full, big.mark = ","),
              format(x$num_singletons, big.mark = ",")))
  if (length(x$fe_num_levels)) {
    cat("Absorbed FEs: ",
        paste(sprintf("%s (%s)", x$fe_labels,
                      trimws(format(x$fe_num_levels, big.mark = ","))),
              collapse = ", "), "\n", sep = "")
  }
  cat(sprintf("Std. errors: %s | Converged: %s in %d iterations (%s)\n",
              se_label, ifelse(x$converged, "yes", "NO"), x$iterations,
              x$absorption_method_used))
  if (isTRUE(x$gpu_used)) {
    cat("GPU: used (", x$gpu_status, ", ", x$gpu_absorption_iterations,
        " absorber iterations)\n", sep = "")
  }
  if (isTRUE(x$saturated)) {
    cat("NOTE: saturated (perfect-fit) design; standard errors are not identified\n")
  }
  cat("\n")
  stats::printCoefmat(coeftable(x), digits = digits, P.values = TRUE,
                      has.Pvalue = TRUE, ...)
  cat(sprintf("\nR-squared: %.5f  Within R-sq.: %.5f  Adj. R-sq.: %.5f\n",
              x$r2, x$r2_within, x$r2_a))
  if (is.finite(x$F_stat)) {
    cat(sprintf("F(%d, %s) = %.2f  Prob > F = %.4g  RMSE: %.5g\n",
                as.integer(x$df_m), format(x$df_r), x$F_stat, x$F_p, x$rmse))
  } else {
    cat(sprintf("RMSE: %.5g\n", x$rmse))
  }
  invisible(x)
}

#' @rdname xhdfe-methods
coeftable <- function(object, ...) UseMethod("coeftable")

coeftable.xhdfe <- function(object, ...) {
  cbind(Estimate = object$coefficients,
        `Std. Error` = object$se,
        `t value` = object$tvalues,
        `Pr(>|t|)` = object$pvalues)
}

#' @rdname xhdfe-methods
summary.xhdfe <- function(object, ...) {
  object$coeftable <- coeftable(object)
  class(object) <- c("summary.xhdfe", "xhdfe")
  object
}

print.summary.xhdfe <- function(x, digits = max(3L, getOption("digits") - 3L),
                                ...) {
  print.xhdfe(x, digits = digits, ...)
  cat(sprintf("RSS: %.6g  TSS: %.6g  Within TSS: %.6g  sigma2: %.6g\n",
              x$rss, x$tss, x$tss_within, x$sigma2))
  cat(sprintf("df: model %d, residual %s (unadj. %s), absorbed %s (exact %s, nested %s)\n",
              as.integer(x$df_m), format(x$df_r), format(x$df_r_unadj),
              format(x$df_a), format(x$df_a_exact), format(x$df_a_nested)))
  if (x$se_type == "cluster") {
    cat(sprintf("Clusters: %s (per dimension: %s; scale %.6g)\n",
                format(x$num_clusters, big.mark = ","),
                paste(format(x$cluster_counts, big.mark = ","), collapse = ", "),
                x$cluster_scale))
  }
  cat(sprintf("Backend: %s | threads: %d | tolerance mode: %s\n",
              ifelse(isTRUE(x$gpu_used), "GPU (CUDA)", "CPU"),
              x$threads_used, x$tolerance_mode))
  invisible(x)
}

#' @rdname xhdfe-methods
coef.xhdfe <- function(object, ...) object$coefficients

#' @rdname xhdfe-methods
vcov.xhdfe <- function(object, ...) object$vcov

#' @rdname xhdfe-methods
confint.xhdfe <- function(object, parm, level, ...) {
  ci <- object$conf_int
  if (!missing(level) && res_level(level) != object$level) {
    alpha <- 1 - res_level(level) / 100
    crit <- stats::qt(1 - alpha / 2, df = object$df_r)
    ci <- cbind(object$coefficients - crit * object$se,
                object$coefficients + crit * object$se)
    dimnames(ci) <- list(names(object$coefficients),
                         sprintf("%g %%", c(100 * alpha / 2,
                                            100 * (1 - alpha / 2))))
  }
  if (!missing(parm)) ci <- ci[parm, , drop = FALSE]
  ci
}

#' @rdname xhdfe-methods
nobs.xhdfe <- function(object, ...) object$nobs

#' @rdname xhdfe-methods
df.residual.xhdfe <- function(object, ...) object$df_r

#' @rdname xhdfe-methods
residuals.xhdfe <- function(object, ...) object$residuals

#' @rdname xhdfe-methods
logLik.xhdfe <- function(object, ...) {
  structure(object$ll, df = object$df_m + 1, nobs = object$nobs,
            class = "logLik")
}

#' @rdname xhdfe-methods
deviance.xhdfe <- function(object, ...) object$rss

#' @rdname xhdfe-methods
formula.xhdfe <- function(x, ...) x$fml
