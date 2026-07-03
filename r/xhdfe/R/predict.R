# predict()/fitted() for xhdfe objects, mirroring Stata's predict statistics.

#' Predictions from an xhdfe model
#'
#' Mirrors the statistics of the Stata command's \code{predict}
#' (\code{xhdfe_p}): the estimated equation is
#' \eqn{y = x\beta + d_{absorbed} + e}, and each \code{type} returns one of
#' its components. All in-sample statistics return full-length vectors
#' aligned with the input data (\code{NA} outside the estimation sample).
#'
#' @param object a fitted \code{xhdfe} model.
#' @param newdata optional data frame for out-of-sample prediction. Only
#'   \code{type = "xb"} is available with \code{newdata}: the absorbed fixed
#'   effects of unseen observations are unknown, exactly as in Stata.
#'   Requires a model estimated through the formula interface.
#' @param type the statistic to compute:
#'   \describe{
#'     \item{\code{"xb"}}{linear prediction from the reported coefficients
#'       (default; Stata \code{predict, xb}).}
#'     \item{\code{"xbd"}}{\code{xb} plus the absorbed fixed effects
#'       (Stata \code{xbd}).}
#'     \item{\code{"d"}}{the absorbed-effect component
#'       \eqn{d = y - xb - e} (Stata \code{d}).}
#'     \item{\code{"residuals"}}{the residual \eqn{e} (Stata
#'       \code{residuals}).}
#'     \item{\code{"dresiduals"}}{\eqn{y - xb}, i.e. absorbed component plus
#'       residual (Stata \code{dresiduals}).}
#'     \item{\code{"score"}}{alias for \code{"residuals"} (Stata maps
#'       \code{score} to the residuals for \code{margins} compatibility).}
#'     \item{\code{"stdp"}}{standard error of the linear prediction
#'       \code{xb} (Stata \code{stdp}).}
#'   }
#' @param ... unused.
#' @return A numeric vector. With \code{newdata}, of length
#'   \code{nrow(newdata)}; otherwise of length \code{object$n_input} with
#'   \code{NA} outside the estimation sample. In group()/individual() fits
#'   the in-sample statistics are at the collapsed group level.
#' @seealso \code{\link{xhdfe}}, \code{\link{fixef.xhdfe}}
#' @examples
#' d <- data.frame(f1 = rep(1:25, 20), f2 = rep(1:20, each = 25))
#' d$x <- rnorm(500); d$y <- d$x + d$f1 / 10 + d$f2 / 5 + rnorm(500)
#' m <- xhdfe(y ~ x | f1 + f2, d)
#' # y decomposes exactly into xb + d + e:
#' all.equal(predict(m) + predict(m, type = "d") + residuals(m), d$y)
#' predict(m, newdata = head(d))
#' @name predict.xhdfe
predict.xhdfe <- function(object, newdata = NULL,
                          type = c("xb", "xbd", "d", "residuals",
                                   "dresiduals", "score", "stdp"),
                          ...) {
  type <- match.arg(type)

  if (!is.null(newdata)) {
    if (type != "xb") {
      stop("with `newdata` only type = \"xb\" is available (the absorbed ",
           "effects of new observations are unknown)", call. = FALSE)
    }
    if (is.null(object$terms)) {
      stop("this object was fitted through the matrix interface; predict on ",
           "new data requires the formula interface", call. = FALSE)
    }
    tf <- stats::delete.response(object$terms)
    mf <- stats::model.frame(tf, data = newdata, na.action = stats::na.pass,
                             xlev = object$xlevels)
    X <- stats::model.matrix(tf, mf, contrasts.arg = object$contrasts)
    int_col <- match("(Intercept)", colnames(X))
    if (!is.na(int_col)) X <- X[, -int_col, drop = FALSE]
    if (!is.null(object$terms_endo)) {
      tfe <- stats::delete.response(object$terms_endo)
      mfe <- stats::model.frame(tfe, data = newdata,
                                na.action = stats::na.pass,
                                xlev = object$xlevels)
      Xe <- stats::model.matrix(tfe, mfe)
      ic <- match("(Intercept)", colnames(Xe))
      if (!is.na(ic)) Xe <- Xe[, -ic, drop = FALSE]
      X <- cbind(X, Xe)
    }
    b <- object$coefficients
    slope_names <- setdiff(names(b), "(Intercept)")
    missing_cols <- setdiff(slope_names, colnames(X))
    if (length(missing_cols)) {
      stop("newdata model matrix is missing columns for coefficients: ",
           paste(missing_cols, collapse = ", "), call. = FALSE)
    }
    bs <- b[slope_names]
    bs[!is.finite(bs)] <- 0
    xb <- drop(X[, slope_names, drop = FALSE] %*% bs)
    if (object$has_intercept) xb <- xb + b[["(Intercept)"]]
    return(xb)
  }

  if (is.null(object$xb_cache)) {
    stop("prediction caches are unavailable for this fit", call. = FALSE)
  }
  xb <- object$xb_cache
  u <- object$residuals
  y <- object$y_cache

  switch(type,
         xb = xb,
         residuals = u,
         score = u,
         xbd = y - u,
         d = y - u - xb,
         dresiduals = y - xb,
         stdp = object$stdp_cache)
}

#' @rdname predict.xhdfe
#' @usage \method{fitted}{xhdfe}(object, type = c("xb", "xbd"), ...)
fitted.xhdfe <- function(object, type = c("xb", "xbd"), ...) {
  type <- match.arg(type)
  predict.xhdfe(object, type = type)
}
