# Fixed-effect recovery accessors (savefe parity).

#' Extract recovered fixed-effect contributions
#'
#' Returns the per-observation contribution of each absorbed dimension,
#' recovered by the C++ backend when the model was estimated with
#' \code{save_fe = TRUE}. This is the R analogue of Stata's
#' \code{absorb(..., savefe)} / \code{savefes(prefix)} options: each element
#' is a full-length numeric vector (aligned with the input data, \code{NA}
#' outside the estimation sample) holding that dimension's additive
#' contribution to the fitted value.
#'
#' For a plain absorbed dimension the contribution is constant within each
#' level (the level's fixed effect). For a heterogeneous-slope dimension
#' (\code{fe[x]} / \code{fe[[x]]} terms) the contribution is
#' \eqn{slope_g \cdot x_{it}}, i.e. it varies within the level with the slope
#' variable, exactly like the \code{Slope1} variables saved by the Stata
#' command. As in Stata, saved fixed effects are contributions, not
#' identified level ids, and may not be separately identified in all designs.
#'
#' @param object an \code{xhdfe} model estimated with \code{save_fe = TRUE}.
#' @param ... unused.
#' @return An object of class \code{xhdfe_fixef}: a named list with one
#'   full-length numeric vector per absorbed dimension, with attributes
#'   \code{iterations}, \code{max_delta} and \code{converged} describing the
#'   FE recovery pass (the Stata \code{e(fe_recovery_*)} scalars).
#' @seealso \code{\link{xhdfe}} (arguments \code{save_fe},
#'   \code{fe_tolerance}, \code{fe_recovery_method});
#'   \code{\link{predict.xhdfe}} with \code{type = "d"} for the summed
#'   absorbed component.
#' @examples
#' d <- data.frame(f1 = rep(1:20, 25), f2 = rep(1:25, each = 20))
#' d$x <- rnorm(500); d$y <- d$x + d$f1 / 10 + d$f2 / 5 + rnorm(500)
#' m <- xhdfe(y ~ x | f1 + f2, d, save_fe = TRUE)
#' fe <- fixef(m)
#' str(fe)
#' # contributions sum to the absorbed component:
#' all.equal(fe[[1]] + fe[[2]], predict(m, type = "d"))
#' @name fixef.xhdfe
fixef <- function(object, ...) UseMethod("fixef")

#' @rdname fixef.xhdfe
fixef.xhdfe <- function(object, ...) {
  if (!length(object$fe_effects)) {
    stop("no recovered fixed effects on this object; re-estimate with ",
         "save_fe = TRUE", call. = FALSE)
  }
  out <- lapply(object$fe_effects, function(v) {
    full <- rep(NA_real_, object$n_input)
    full[object$sample] <- v
    full
  })
  names(out) <- names(object$fe_effects)
  attr(out, "iterations") <- object$fe_recovery_iterations
  attr(out, "max_delta") <- object$fe_recovery_max_delta
  attr(out, "converged") <- object$fe_recovery_converged
  class(out) <- "xhdfe_fixef"
  out
}

print.xhdfe_fixef <- function(x, ...) {
  cat("xhdfe recovered fixed-effect contributions (one vector per absorbed ",
      "dimension)\n", sep = "")
  cat("Dimensions:", paste(names(x), collapse = ", "), "\n")
  cat(sprintf("Recovery: %s in %d iterations (max delta %.3g)\n",
              ifelse(attr(x, "converged"), "converged", "NOT converged"),
              attr(x, "iterations"), attr(x, "max_delta")))
  invisible(x)
}
