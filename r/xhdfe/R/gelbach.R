# Gelbach (2016) conditional decomposition, HDFE-aware (M9B front-end).

#' Gelbach coefficient-movement decomposition (HDFE-aware)
#'
#' Decomposes the movement of the \code{x1} coefficients between one base and
#' one full linear specification into simultaneous, order-invariant observed
#' covariate and absorbed fixed-effect contributions, following Gelbach
#' (2016). This is specification accounting, not causal mediation; causal
#' interpretation requires a separately justified research design.
#'
#' @param y Finite numeric outcome of length n.
#' @param x1 Finite base-covariate matrix or vector with n rows. Do not append a
#'   constant: the same intercept is implicit in both specifications.
#' @param x2_groups Named list of finite observed-covariate matrices or vectors
#'   with n rows. Each entry is one simultaneous added block.
#' @param fes Named list of length-n fixed-effect identifier vectors. Integer,
#'   factor and character identifiers are accepted.
#' @param vce \code{"unadjusted"} (default), \code{"robust"} or
#'   \code{"cluster"}.
#' @param cluster Length-n cluster identifiers, required only when
#'   \code{vce = "cluster"}. Clustering is one-way and needs at least two
#'   clusters.
#' @param gamma0 Retain the auxiliary-regression component of the variance,
#'   matching Gelbach's \code{b1x2} \code{gamma0} option.
#' @param cov0 Omit robust stacked-system cross terms, matching \code{b1x2}; it
#'   is a no-op for unadjusted inference.
#' @param tol Positive fixed-effect absorption tolerance. The default
#'   \code{1e-8} preserves the historical effective tolerance. This does not
#'   control FE-collinearity classification, whose separate squared-norm rule
#'   is \code{||M_D x||^2 / ||x||^2 <= 1e-9} and is returned as metadata.
#' @param num_threads OpenMP thread request/cap for supported CPU phases
#'   (0 = library default). Individual phases may choose fewer threads;
#'   \code{$threads_used} reports the maximum effective absorber/recovery
#'   team, not a promise that every auxiliary or covariance phase used that
#'   count. GPU kernels do not use this OpenMP setting.
#' @param weights Optional finite, strictly positive Stata-style weights:
#'   analytic by default, frequency when \code{fweights = TRUE}.
#' @param fweights Treat \code{weights} as positive integer frequency weights.
#' @param absorbed_targets Optional X1 column names, one-based indices, or a
#'   length-p logical mask. Activates the distinct absorbed-target allocation
#'   estimand. Every declared target must be fully absorbed by \code{fes}; its
#'   full-model coefficient is imposed at zero, not estimated.
#' @param focal Optional X1 column names, one-based indices, or a length-p
#'   logical mask used only to select focal rows for reporting. Every X1
#'   column remains in both specifications, so common controls can be included
#'   in \code{x1} without crowding the empirical table.
#' @param gpu Logical scalar. With \code{TRUE}, request CUDA for the
#'   full-model FE-absorption phase. The decomposition and covariance
#'   calculations remain on the CPU. A CPU-only build or unavailable device
#'   falls back truthfully; inspect the \code{$gpu_*} diagnostics. This
#'   argument is last to preserve the positional contract of earlier releases.
#'
#' @section Specification and sample contract:
#' All X1 columns remain in both models. Every \code{x2_groups} and \code{fes}
#' entry is added simultaneously to the full model; list order does not define
#' a sequential allocation. Block names must be unique and nonempty across
#' both lists. The wrapper requires finite numeric inputs and does not silently
#' remove missing rows, so construct one common complete-case sample before
#' calling it. Recursive FE singleton removal can still occur in the backend.
#' Inspect \code{$n_obs_input}, \code{$n_obs},
#' \code{$n_obs_effective}, and \code{$n_singletons_dropped}. Under frequency
#' weights, effective N is the sum of retained weights; \code{$n_obs} remains
#' the retained row count. Rank failures and undeclared omissions fail closed.
#'
#' @section Inference and covariance:
#' Observed-block inference reproduces \code{b1x2}'s random-design
#' stacked-moment variance for unadjusted, robust and one-way clustered VCEs.
#' It includes uncertainty in the auxiliary projections and is not the smaller
#' variance conditional on the realised design. Absorbed-FE blocks receive
#' conditional/\code{gamma0} variance, so totals combining observed and FE
#' blocks are labelled as mixed. For an absorbed target,
#' \code{total_j = b_base_j - 0} is the base-coefficient estimator; valid
#' inference requires clustering at an FE dimension that absorbs that target.
#' Other VCE choices are retained for descriptive accounting but warn and set
#' \code{$absorbed_target_inference_valid = FALSE}.
#'
#' The joint \code{$cov} matrix is group-major. With
#' \code{k1 = ncol(x1) + 1}, its ordering is
#' \code{[group1:x1, group1:_cons, group2:x1, group2:_cons, ...]}.
#' Use this object, or \code{xhdfe_gelbach_contrast()}, for sums and contrasts;
#' component SEs must not be added as if the blocks were independent.
#' \code{$base_cov}, \code{$cov_delta_bbase}, and
#' \code{$cov_total_bbase} complete the covariance contract needed for
#' base-coefficient-denominator shares. \code{xhdfe_gelbach_tidy(...,
#' share = "base")} uses all denominator and cross-covariance terms; it is
#' distinct from the retained descriptive \code{share = "base_fixed"}.
#'
#' @section Result schema:
#' Coefficients and contributions are in \code{$b_base}, \code{$b_full},
#' \code{$b_full_status}, \code{$gamma}, \code{$delta}, \code{$se},
#' \code{$cov}, \code{$base_cov}, \code{$cov_delta_bbase},
#' \code{$cov_total_bbase}, \code{$total}, \code{$total_se},
#' \code{$total_cov}, and optional \code{$fe_total}. Names and selectors are
#' in \code{$names},
#' \code{$group_kinds}, \code{$x1_names}, \code{$focal_indices}, and
#' \code{$focal_names}. Estimand and identification metadata are in
#' \code{$estimand}, \code{$identity_status}, \code{$absorbed_mask}, the legacy
#' backend alias \code{$x1_absorbed},
#' \code{$absorbed_targets}, \code{$absorbed_target_names}, and
#' \code{$focal_status}. Inference metadata are in \code{$vce},
#' \code{$gamma0}, \code{$cov0}, \code{$tol}, \code{$se_type},
#' \code{$total_se_type}, \code{$inference_status},
#' \code{$absorbed_target_inference_valid}, and
#' \code{$absorbing_fe_index}. Diagnostics include \code{$identity_gap},
#' \code{$converged}, \code{$notes}, \code{$df_base}, \code{$df_full},
#' \code{$n_clusters}, the sample counts, per-X1
#' \code{$x1_fe_collinear_ratio}/\code{$x1_near_collinear_mask}, both
#' FE-collinearity thresholds, \code{$few_cluster_warning_threshold},
#' \code{$threads_used}, and the truthful \code{$gpu_*} fields.
#' \code{$gamma} has one column per observed block and is padded with
#' non-finite entries to the widest block. \code{$causal_interpretation} is
#' always \code{FALSE}. R accepts one-based selector indices but
#' returns \code{$focal_indices}, \code{$absorbed_targets}, and
#' \code{$absorbing_fe_index} as zero-based cross-frontend metadata.
#'
#' @section Printing and reporting:
#' \code{print()} displays base and full coefficients, total movement and its
#' SE, component contributions and SEs, and the normalization-safe FE subtotal.
#' It respects the stored \code{focal} selector and accepts \code{digits}; an
#' absorbed target's full coefficient is printed as \code{0 (imposed)}.
#' \code{xhdfe_gelbach_tidy()} produces rectangular rows, confidence intervals,
#' and signed shares. \code{xhdfe_gelbach_contrast()} computes linear
#' combinations from the stored joint covariance.
#'
#' @section Numerical diagnostics:
#' A bounded-cost normalized-Gram check records a warning in \code{$notes}
#' when an observed x2 block is severely near-collinear; values are not
#' altered, but that block's split SE can be rounding/tolerance sensitive.
#' A separate per-X1 diagnostic warns, without changing any number, when
#' \code{||M_D x1_j||^2 / ||x1_j||^2} lies above the backend omission
#' boundary and at or below the documented warning-band upper limit. Such a
#' focal is nearly FE-collinear; inspect the mask and consider
#' \code{absorbed_targets} only when it is conceptually FE-invariant.
#' Cluster-meat FMA may differ from the former materialized path by one
#' last-place unit in well-conditioned cells, with coefficients/deltas
#' bit-identical. Inferential notes raise \code{warning()} even when the
#' decomposition identity converges. Always require \code{$converged} and
#' inspect \code{$notes}; a small \code{$identity_gap} checks the summation
#' identity only.
#'
#' @section Deliberate limits:
#' The optional \code{gpu = TRUE} request applies only to full-model
#' FE absorption; auxiliary regressions, covariance construction, and
#' reporting remain CPU work. It does not implement IV/LATE allocation,
#' multiway clustering,
#' wild-cluster bootstrap, nonconditional recovered-FE covariance, a separate
#' HDFE set common to base and full, dynamic-panel corrections, nonlinear or
#' distributional decompositions, Oaxaca wrappers, or causal mediation.
#' A binary outcome is estimated as a linear probability model; a
#' decomposition on a logit or other nonlinear scale is a separate estimator.
#' Explicit numeric columns can represent polynomials, bins, splines,
#' interactions, and low-dimensional effects common to both specifications.
#' Factor objects are deliberately refused: use \code{model.matrix()} (or
#' explicit generated indicators), drop its intercept and one reference
#' category, and pass the resulting full-rank numeric columns. Because an
#' intercept is implicit, categorical indicators common to both models must
#' not include all category dummies.
#'
#' @section Offline validation bundles:
#' The source package declares Rcpp as an \code{Imports}/\code{LinkingTo}
#' dependency. In an offline repository validation bundle that includes
#' \code{r/Rlib/Rcpp}, set \code{R_PROFILE_USER=/dev/null} and
#' \code{R_LIBS_USER} to the bundle's \code{r/Rlib} before installation, and
#' verify the dependency with \code{find.package("Rcpp")}. \code{r/Rlib} is a
#' local development/validation library, not part of the CRAN-style source
#' package or guaranteed to be present in every clone; otherwise supply Rcpp
#' from offline media first.
#'
#' @return An object of class \code{xhdfe_gelbach}. Contributions and their
#'   SEs have rows \code{[x1..., _cons]} and columns in observed-group then FE
#'   order. Use \code{xhdfe_gelbach_tidy()} for publication-ready rows and
#'   signed shares, and \code{xhdfe_gelbach_contrast()} for covariance-aware
#'   linear combinations.
#'
#' @examples
#' \dontrun{
#' # target and controls common to base and full; report only target
#' fit <- xhdfe_gelbach(
#'   y = dat$lwage,
#'   x1 = cbind(target = dat$female, age = dat$age),
#'   x2_groups = list(
#'     human_capital = cbind(dat$educ, dat$educ_sq),
#'     job = cbind(dat$tenure, dat$experience)
#'   ),
#'   fes = list(firm = dat$firm, occupation = dat$occupation),
#'   vce = "cluster", cluster = dat$worker,
#'   focal = "target"
#' )
#' rows <- xhdfe_gelbach_tidy(fit, share = "movement")
#' observed <- xhdfe_gelbach_contrast(
#'   fit, focal = "target", groups = c("human_capital", "job")
#' )
#'
#' # distinct constrained estimand: target is invariant within worker
#' absorbed <- xhdfe_gelbach(
#'   y = dat$lwage,
#'   x1 = cbind(target = dat$female, age = dat$age),
#'   x2_groups = list(job = cbind(dat$tenure, dat$experience)),
#'   fes = list(worker = dat$worker, firm = dat$firm),
#'   vce = "cluster", cluster = dat$worker,
#'   absorbed_targets = "target", focal = "target"
#' )
#' stopifnot(absorbed$b_full_status[["target"]] == "imposed_zero")
#' }
#' @export
xhdfe_gelbach <- function(y, x1, x2_groups = NULL, fes = NULL,
                          vce = "unadjusted", cluster = NULL,
                          gamma0 = FALSE, cov0 = FALSE, tol = 1e-8,
                          num_threads = 0L,
                          weights = NULL, fweights = FALSE,
                          absorbed_targets = NULL, focal = NULL,
                          gpu = FALSE) {
  y <- as.numeric(y)
  x1_is_numeric <- if (is.data.frame(x1)) {
    all(vapply(x1, function(z) is.numeric(z) && !is.factor(z), logical(1)))
  } else {
    is.numeric(x1) && !is.factor(x1)
  }
  if (!x1_is_numeric) {
    stop("x1 must contain explicit numeric columns; generate a full-rank ",
         "indicator matrix first (for example with model.matrix(), dropping ",
         "its intercept and one reference level)", call. = FALSE)
  }
  x1 <- as.matrix(x1)
  storage.mode(x1) <- "double"
  if (ncol(x1) == 0L) {
    stop("x1 must contain at least one focal column", call. = FALSE)
  }
  if (nrow(x1) != length(y)) {
    stop("y and x1 must have the same number of rows", call. = FALSE)
  }
  if (any(!is.finite(y)) || any(!is.finite(x1))) {
    stop("y and x1 must contain only finite values", call. = FALSE)
  }
  x1_names <- colnames(x1) %||% paste0("x1_", seq_len(ncol(x1)))
  if (anyNA(x1_names) || any(!nzchar(trimws(x1_names)))) {
    stop("x1 column names must be non-missing, non-empty strings",
         call. = FALSE)
  }
  x1_names <- trimws(x1_names)
  if (anyDuplicated(x1_names)) {
    stop("x1 column names must be unique", call. = FALSE)
  }
  if ("_cons" %in% x1_names) {
    stop("x1 column names may not use the reserved name '_cons'",
         call. = FALSE)
  }
  colnames(x1) <- x1_names
  focal_idx <- .gelbach_focal_indices(focal, x1_names, ncol(x1))
  if (length(tol) != 1L || !is.finite(tol) || tol <= 0) {
    stop("tol must be finite and strictly positive", call. = FALSE)
  }
  if (length(gpu) != 1L || is.na(gpu) || !is.logical(gpu)) {
    stop("gpu must be one non-missing logical value", call. = FALSE)
  }
  x2_groups <- as.list(x2_groups %||% list())
  fes <- as.list(fes %||% list())
  if (!length(x2_groups) && !length(fes)) {
    stop("provide at least one x2 group or fixed-effect dimension", call. = FALSE)
  }
  nm <- c(names(x2_groups), names(fes))
  if (length(nm) != length(x2_groups) + length(fes) ||
      anyNA(nm) || any(!nzchar(trimws(nm)))) {
    stop("every x2/FE block must have a non-empty name", call. = FALSE)
  }
  if (anyDuplicated(nm)) {
    stop("x2 and FE block names must be unique", call. = FALSE)
  }
  absorbed_idx <- integer(0)
  if (!is.null(absorbed_targets)) {
    if (is.character(absorbed_targets)) {
      if (is.null(colnames(x1))) {
        stop("character absorbed_targets requires x1 column names", call. = FALSE)
      }
      absorbed_idx <- match(absorbed_targets, colnames(x1))
      if (anyNA(absorbed_idx)) {
        stop("absorbed_targets contains an unknown x1 column name", call. = FALSE)
      }
    } else if (is.logical(absorbed_targets)) {
      if (length(absorbed_targets) != ncol(x1) || anyNA(absorbed_targets)) {
        stop("an absorbed_targets logical mask must be non-missing and have length p",
             call. = FALSE)
      }
      absorbed_idx <- which(absorbed_targets)
    } else {
      if (!is.numeric(absorbed_targets)) {
        stop("absorbed_targets must contain x1 names, indices, or a logical mask",
             call. = FALSE)
      }
      if (anyNA(absorbed_targets) ||
          any(!is.finite(absorbed_targets)) ||
          any(absorbed_targets != round(absorbed_targets))) {
        stop("absorbed_targets indices must be finite integers", call. = FALSE)
      }
      absorbed_idx <- as.integer(absorbed_targets)
      if (any(absorbed_idx < 1L | absorbed_idx > ncol(x1))) {
        stop("absorbed_targets index is outside the x1 column range", call. = FALSE)
      }
    }
    if (anyDuplicated(absorbed_idx)) {
      stop("absorbed_targets must be unique", call. = FALSE)
    }
    if (length(absorbed_idx) && !length(fes)) {
      stop("absorbed_targets requires at least one absorbed FE dimension",
           call. = FALSE)
    }
  }
  if (!vce %in% c("unadjusted", "robust", "cluster")) {
    stop("vce must be \"unadjusted\", \"robust\" or \"cluster\"", call. = FALSE)
  }
  sizes <- integer(0)
  X2 <- NULL
  for (g in x2_groups) {
    group_is_numeric <- if (is.data.frame(g)) {
      all(vapply(g, function(z) is.numeric(z) && !is.factor(z), logical(1)))
    } else {
      is.numeric(g) && !is.factor(g)
    }
    if (!group_is_numeric) {
      stop("x2 groups must contain explicit numeric columns; generate ",
           "full-rank indicators first (for example with model.matrix(), ",
           "dropping its intercept and one reference level)", call. = FALSE)
    }
    g <- as.matrix(g)
    storage.mode(g) <- "double"
    if (nrow(g) != length(y)) stop("x2 group has wrong length", call. = FALSE)
    if (ncol(g) == 0L) stop("x2 groups must contain at least one column", call. = FALSE)
    if (any(!is.finite(g))) stop("x2 groups must contain only finite values", call. = FALSE)
    sizes <- c(sizes, ncol(g))
    X2 <- if (is.null(X2)) g else cbind(X2, g)
  }
  fe_list <- lapply(fes, function(ids) .akm_id_codes(ids, "fe"))
  if (!is.null(cluster) && !identical(vce, "cluster")) {
    stop("cluster ids supplied but vce != \"cluster\"", call. = FALSE)
  }
  if (identical(vce, "cluster") && is.null(cluster)) {
    stop("vce = \"cluster\" requires cluster ids", call. = FALSE)
  }
  if (!is.null(cluster) && length(cluster) != length(y)) {
    stop("cluster must have the same length as y", call. = FALSE)
  }
  cl <- if (is.null(cluster)) NULL else .akm_id_codes(cluster, "cluster")
  if (!is.null(cl) && length(unique(cl)) < 2L) {
    stop("vce = \"cluster\" requires at least two clusters", call. = FALSE)
  }
  if (!is.null(weights) && !is.numeric(weights)) {
    stop("weights must be a numeric vector", call. = FALSE)
  }
  w <- if (is.null(weights)) NULL else as.numeric(weights)
  if (!is.null(w)) {
    if (length(w) != length(y) || any(!is.finite(w)) || any(w <= 0)) {
      stop("weights must be finite, strictly positive and have length n", call. = FALSE)
    }
    if (isTRUE(fweights) && any(w != round(w))) {
      stop("frequency weights must be integers", call. = FALSE)
    }
  } else if (isTRUE(fweights)) {
    stop("fweights = TRUE requires weights", call. = FALSE)
  }
  out <- .xhdfe_cpp_gelbach(y, x1, X2, as.integer(sizes), fe_list, cl,
                            as.character(vce), isTRUE(gamma0), isTRUE(cov0),
                            as.numeric(tol), as.integer(num_threads),
                            isTRUE(gpu), w,
                            isTRUE(fweights), as.integer(absorbed_idx - 1L))
  k1 <- ncol(x1) + 1L
  coefficient_names <- c(x1_names, "_cons")
  delta <- out$delta
  colnames(delta) <- nm
  rownames(delta) <- coefficient_names
  se <- vapply(seq_along(nm), function(g) {
    idx <- ((g - 1L) * k1 + 1L):(g * k1)
    sqrt(diag(out$cov[idx, idx, drop = FALSE]))
  }, numeric(k1))
  dimnames(se) <- dimnames(delta)
  out$delta <- delta
  out$se <- se
  out$total_se <- sqrt(diag(out$total_cov))
  dimnames(out$base_cov) <- list(coefficient_names, coefficient_names)
  dimnames(out$cov_total_bbase) <- list(coefficient_names,
                                        coefficient_names)
  if (length(nm)) {
    cross_rows <- unlist(lapply(nm, function(group) {
      paste(group, coefficient_names, sep = ":")
    }), use.names = FALSE)
    dimnames(out$cov_delta_bbase) <- list(cross_rows, coefficient_names)
  }
  if (length(x2_groups)) {
    colnames(out$gamma) <- names(x2_groups)
    rownames(out$gamma) <- paste0("within_block_", seq_len(nrow(out$gamma)))
  }
  out$names <- nm
  out$x1_names <- unname(x1_names)
  out$focal_indices <- unname(as.integer(focal_idx - 1L))
  out$focal_names <- unname(x1_names[focal_idx])
  out$focal_selection_explicit <- !is.null(focal)
  out$group_kinds <- setNames(c(rep("x2", length(x2_groups)),
                                rep("fe", length(fes))), nm)
  observed_se_type <- if (isTRUE(gamma0)) {
    "gamma0"
  } else if (isTRUE(cov0) && !identical(vce, "unadjusted")) {
    "cov0"
  } else {
    "full"
  }
  out$se_type <- setNames(ifelse(out$group_kinds == "fe",
                                 "conditional_gamma0", observed_se_type), nm)
  if (length(fes)) {
    fe_cols <- (length(x2_groups) + 1L):length(nm)
    fe_cov <- matrix(0, k1, k1)
    for (g in fe_cols) for (h in fe_cols) {
      gi <- ((g - 1L) * k1 + 1L):(g * k1)
      hi <- ((h - 1L) * k1 + 1L):(h * k1)
      fe_cov <- fe_cov + out$cov[gi, hi, drop = FALSE]
    }
    dimnames(fe_cov) <- list(rownames(delta), rownames(delta))
    out$fe_total <- list(
      members = nm[fe_cols],
      coef = rowSums(delta[, fe_cols, drop = FALSE]),
      cov = fe_cov,
      se = sqrt(diag(fe_cov)),
      se_type = "conditional_gamma0"
    )
  } else {
    out$fe_total <- NULL
  }
  out$vce <- vce
  out$tol <- as.numeric(tol)
  out$gpu <- isTRUE(gpu)
  names(out$x1_fe_collinear_ratio) <- x1_names
  names(out$x1_near_collinear_mask) <- x1_names
  absorbed_mask <- as.logical(out$x1_absorbed)
  out$absorbed_mask <- unname(absorbed_mask)
  out$absorbed_targets <- unname(which(absorbed_mask) - 1L)
  out$absorbed_target_names <-
    (colnames(x1) %||% paste0("x1_", seq_len(ncol(x1))))[absorbed_mask]
  out$b_full_status <- ifelse(absorbed_mask, "imposed_zero", "estimated")
  out$focal_status <- ifelse(absorbed_mask, "absorbed", "identified")
  names(out$b_full_status) <- names(out$focal_status) <-
    colnames(x1) %||% paste0("x1_", seq_len(ncol(x1)))
  absorbed_mode <- any(absorbed_mask)
  out$estimand <- if (absorbed_mode) "absorbed_target_allocation" else
    "coefficient_movement"
  out$identity_status <- if (absorbed_mode) "exact_ols_constrained" else
    "exact_ols"
  out$total_se_type <- if (length(fes)) {
    if (isTRUE(gamma0) || !length(x2_groups)) {
      "conditional_gamma0"
    } else if (isTRUE(cov0) && !identical(vce, "unadjusted")) {
      "mixed_cov0_observed_conditional_fe"
    } else {
      "mixed_full_observed_conditional_fe"
    }
  } else {
    observed_se_type
  }
  if (absorbed_mode) {
    out$total_se_type <- "target_exact_base_vce_mixed_components"
  }
  out$fe_collinear_relative_norm_tol <-
    sqrt(out$fe_collinear_ss_ratio_tol)
  out$inference_status <- if (!absorbed_mode) {
    "not_applicable"
  } else if (isTRUE(out$absorbed_target_inference_valid)) {
    "clustered_at_absorbing_fe"
  } else {
    "warning_unsupported_vce_or_cluster"
  }
  out$gamma0 <- isTRUE(gamma0)
  out$cov0 <- isTRUE(cov0)
  out$causal_interpretation <- FALSE
  class(out) <- "xhdfe_gelbach"
  if (isFALSE(out$converged)) {
    warning("xhdfe_gelbach: the decomposition did not converge or failed a ",
            "convergence cross-check - results are unreliable (see $notes).",
            call. = FALSE)
  } else if (nzchar(out$notes) && grepl("warning:", tolower(out$notes),
                                        fixed = TRUE)) {
    warning("xhdfe_gelbach: inferential diagnostic - ", out$notes,
            call. = FALSE)
  }
  out
}

`%||%` <- function(a, b) if (is.null(a)) b else a

.gelbach_focal_indices <- function(focal, labels, p) {
  if (is.null(focal)) return(seq_len(p))
  if (is.character(focal)) {
    idx <- match(focal, labels)
    if (anyNA(idx)) {
      stop("focal contains an unknown x1 column name", call. = FALSE)
    }
  } else if (is.logical(focal)) {
    if (length(focal) != p || anyNA(focal)) {
      stop("a focal logical mask must be non-missing and have length p",
           call. = FALSE)
    }
    idx <- which(focal)
  } else {
    if (!is.numeric(focal) || anyNA(focal) || any(!is.finite(focal)) ||
        any(focal != round(focal))) {
      stop("focal must contain x1 names, one-based indices, or a logical mask",
           call. = FALSE)
    }
    idx <- as.integer(focal)
    if (any(idx < 1L | idx > p)) {
      stop("focal index is outside the x1 column range", call. = FALSE)
    }
  }
  if (!length(idx)) stop("focal must select at least one x1 column", call. = FALSE)
  if (anyDuplicated(idx)) stop("focal entries must be unique", call. = FALSE)
  idx
}

#' @export
print.xhdfe_gelbach <- function(x, digits = 6, ...) {
  cat("Gelbach coefficient-movement decomposition (xhdfe backend)\n")
  cat("Specification accounting; not causal mediation.\n")
  if (identical(x$estimand, "absorbed_target_allocation")) {
    cat("Estimand: absorbed-target allocation. Full-model target zeros are imposed, not estimated.\n")
  }
  cat(sprintf("n = %s, vce = %s%s\n",
              format(x$n_obs_effective, big.mark = ","),
              x$vce, if (isTRUE(x$gamma0)) " (gamma0)" else ""))
  if (identical(x$vce, "cluster")) {
    cat(sprintf("Clusters: %d%s\n", x$n_clusters,
                if (x$n_clusters < x$few_cluster_warning_threshold) {
                  sprintf(" (fewer than %d; use cluster-robust inference with caution)",
                          x$few_cluster_warning_threshold)
                } else ""))
  }
  if (isTRUE(x$gpu_requested)) {
    cat(sprintf("GPU absorption: %s (backend=%s, used=%s, iterations=%d)\n",
                x$gpu_status, x$gpu_backend,
                if (isTRUE(x$gpu_used)) "yes" else "no",
                x$gpu_absorption_iterations))
  }
  p <- length(x$b_base)
  selected <- if (isTRUE(x$focal_selection_explicit)) {
    x$focal_indices + 1L
  } else {
    seq_len(p)
  }
  coef_names <- names(x$b_full_status) %||% rownames(x$delta)[seq_len(p)]
  full_text <- formatC(x$b_full, digits = digits, format = "g")
  full_text[x$b_full_status == "imposed_zero"] <- "0 (imposed)"
  movement <- cbind(
    Base = formatC(x$b_base, digits = digits, format = "g"),
    Full = full_text,
    Movement = formatC(x$total[seq_len(p)], digits = digits, format = "g"),
    `Std. err.` = formatC(x$total_se[seq_len(p)], digits = digits, format = "g")
  )
  rownames(movement) <- coef_names
  cat("\nCoefficient movement:\n")
  print(noquote(movement[selected, , drop = FALSE]), quote = FALSE, right = TRUE)
  cat("\nContributions (delta):\n")
  shown_rows <- if (isTRUE(x$focal_selection_explicit)) selected else seq_len(nrow(x$delta))
  print(round(x$delta[shown_rows, , drop = FALSE], digits))
  cat("\nStandard errors:\n")
  print(round(x$se[shown_rows, , drop = FALSE], digits))
  if (!is.null(x$fe_total)) {
    cat("Absorbed-FE aggregate (conditional/gamma0 SE):\n")
    print(round(cbind(coef = x$fe_total$coef, se = x$fe_total$se), digits))
  }
  cat(sprintf("\nTotal (= b_base - b_full): %s\n",
              paste(round(x$total, digits), collapse = " ")))
  cat("Total SE type:", x$total_se_type, "\n")
  if (!isTRUE(x$converged)) {
    warning("xhdfe_gelbach: the decomposition did not converge or failed a ",
            "convergence cross-check - results are unreliable (see notes).",
            call. = FALSE)
  }
  if (nzchar(x$notes)) cat("notes:", x$notes, "\n")
  invisible(x)
}

.gelbach_result_rows <- function(x, focal = NULL, include_intercept = FALSE) {
  p <- length(x$b_base)
  idx <- if (is.null(focal)) {
    x$focal_indices + 1L
  } else {
    .gelbach_focal_indices(focal, x$x1_names, p)
  }
  if (isTRUE(include_intercept)) idx <- c(idx, p + 1L)
  idx
}

.gelbach_row_cov <- function(x, row) {
  k1 <- length(x$total)
  pos <- (seq_along(x$names) - 1L) * k1 + row
  x$cov[pos, pos, drop = FALSE]
}

.gelbach_share_rows <- function(x, denominator, share_tol) {
  if (length(denominator) != 1L ||
      !denominator %in% c("base", "base_fixed", "movement")) {
    stop("share must be NULL, \"base\", \"base_fixed\" or \"movement\"",
         call. = FALSE)
  }
  if (length(share_tol) != 1L || !is.finite(share_tol) || share_tol < 0) {
    stop("share_tol must be finite and nonnegative", call. = FALSE)
  }
  k1 <- nrow(x$delta); p <- length(x$b_base); G <- ncol(x$delta)
  ans <- matrix(NA_real_, k1, G, dimnames = dimnames(x$delta))
  se <- matrix(NA_real_, k1, G, dimnames = dimnames(x$delta))
  defined <- rep(FALSE, k1)
  for (row in seq_len(k1)) {
    denom <- if (denominator %in% c("base", "base_fixed")) {
      if (row > p) next else x$b_base[row]
    } else {
      x$total[row]
    }
    if (!is.finite(denom) || abs(denom) <= share_tol) next
    defined[row] <- TRUE
    d <- x$delta[row, ]
    ans[row, ] <- d / denom
    V <- .gelbach_row_cov(x, row)
    if (identical(denominator, "base_fixed")) {
      se[row, ] <- sqrt(pmax(0, diag(V))) / abs(denom)
    } else if (identical(denominator, "base")) {
      base_var <- x$base_cov[row, row]
      for (g in seq_len(G)) {
        pos <- (g - 1L) * k1 + row
        cross <- x$cov_delta_bbase[pos, row]
        ratio_var <- (V[g, g] / denom^2 +
                      d[g]^2 * base_var / denom^4 -
                      2 * d[g] * cross / denom^3)
        se[row, g] <- sqrt(max(0, ratio_var))
      }
    } else if (identical(denominator, "movement")) {
      for (g in seq_len(G)) {
        grad <- rep(-d[g] / denom^2, G)
        grad[g] <- grad[g] + 1 / denom
        se[row, g] <- sqrt(max(0, drop(t(grad) %*% V %*% grad)))
      }
    }
  }
  list(coef = ans, se = se, defined = defined, denominator = denominator,
       se_type = if (identical(denominator, "base")) {
         "joint_base_covariance_delta_method"
       } else if (identical(denominator, "base_fixed")) {
         "fixed_base_denominator_scaling"
       } else {
         "joint_covariance_delta_method"
       }, units = "fraction", tol = share_tol)
}

#' Tidy Gelbach contributions, confidence intervals and signed shares
#'
#' This is reporting-only post-processing of an \code{xhdfe_gelbach} result.
#' No model is re-estimated. With \code{share = "movement"}, ratio uncertainty
#' uses the full joint covariance and the delta method. With
#' \code{share = "base"}, the delta method also uses denominator uncertainty
#' and \code{Cov(delta, b_base)}. The explicit
#' \code{share = "base_fixed"} reproduces fixed-denominator scaling and labels
#' that convention in the output.
#'
#' @param x An \code{xhdfe_gelbach} result.
#' @param focal Optional X1 names, one-based indices or logical mask. The
#'   default uses the reporting selector stored in \code{x}.
#' @param include_intercept Include the implicit intercept row.
#' @param include_total Include a total-movement row.
#' @param include_full Include the full-model residual coefficient row.
#' @param conf_level Normal-approximation confidence level.
#' @param share Optional \code{"base"}, \code{"base_fixed"}, or
#'   \code{"movement"} denominator.
#' @param share_tol Absolute threshold below which a share is undefined.
#' @return A data frame with one row per selected coefficient and component.
#'   Core columns are \code{coefficient}, \code{component},
#'   \code{component_kind}, \code{estimate}, \code{std_error},
#'   \code{conf_low}, \code{conf_high}, \code{conf_level}, and
#'   \code{se_type}. When \code{share} is requested, the result also contains
#'   \code{share}, \code{share_std_error}, \code{share_conf_low},
#'   \code{share_conf_high}, \code{share_defined},
#'   \code{share_denominator}, \code{share_se_type}, \code{share_units}, and
#'   \code{share_tol}.
#' @section Share contract:
#' \code{share = "movement"} divides by total coefficient movement and uses
#' the full joint component covariance with the delta method; the total share
#' is one with SE zero. For \code{share = "base"}, the component formula is
#' \code{Var(delta_g / b) = Var(delta_g) / b^2 +
#' delta_g^2 Var(b) / b^4 - 2 delta_g Cov(delta_g, b) / b^3}; the total row
#' uses the analogous \code{Cov(total, b_base)} expression.
#' \code{share = "base_fixed"} is an explicit descriptive convention that
#' scales component SEs while holding the reported base coefficient fixed.
#' Shares are fractions, remain signed, and are never truncated or
#' renormalized. Denominators with absolute value at or below
#' \code{share_tol} are marked undefined and returned as missing. Base-based
#' shares are also undefined for the implicit intercept because \code{b_base}
#' contains X1 coefficients only.
#' @section Added rows:
#' \code{include_total} appends \code{total_movement}. \code{include_full}
#' appends \code{full_model_residual}; its SE and interval are missing because
#' that covariance is not exposed by the public contract. These rows are
#' reporting identities and do not re-estimate either model.
#' @examples
#' \dontrun{
#' rows <- xhdfe_gelbach_tidy(
#'   fit, focal = "target", include_total = TRUE,
#'   include_full = TRUE, conf_level = 0.95, share = "movement"
#' )
#' subset(rows, coefficient == "target")
#' }
#' @export
xhdfe_gelbach_tidy <- function(x, focal = NULL, include_intercept = FALSE,
                               include_total = TRUE, include_full = TRUE,
                               conf_level = 0.95, share = NULL,
                               share_tol = 1e-12) {
  if (!inherits(x, "xhdfe_gelbach")) {
    stop("x must be an xhdfe_gelbach result", call. = FALSE)
  }
  if (length(conf_level) != 1L || !is.finite(conf_level) ||
      conf_level <= 0 || conf_level >= 1) {
    stop("conf_level must lie strictly between zero and one", call. = FALSE)
  }
  selected <- .gelbach_result_rows(x, focal, include_intercept)
  zcrit <- stats::qnorm(0.5 + conf_level / 2)
  sh <- if (is.null(share)) NULL else .gelbach_share_rows(x, share, share_tol)
  if (!is.null(sh) && any(!sh$defined[selected])) {
    undefined <- ifelse(selected <= length(x$b_base),
                        x$x1_names[pmin(selected, length(x$b_base))],
                        "_cons")
    undefined <- unique(undefined[!sh$defined[selected]])
    warning("xhdfe_gelbach_tidy: requested share denominator is undefined ",
            "(non-finite, an unavailable base intercept, or absolute value ",
            "<= share_tol) for: ", paste(undefined, collapse = ", "),
            ". Shares and intervals are returned as NA.",
            call. = FALSE)
  }
  records <- list(); cursor <- 0L; p <- length(x$b_base)
  add <- function(record) {
    cursor <<- cursor + 1L
    records[[cursor]] <<- as.data.frame(record, stringsAsFactors = FALSE)
  }
  for (row in selected) {
    label <- if (row <= p) x$x1_names[row] else "_cons"
    for (g in seq_along(x$names)) {
      est <- x$delta[row, g]; serr <- x$se[row, g]
      rec <- list(coefficient = label, component = x$names[g],
                  component_kind = unname(x$group_kinds[g]), estimate = est,
                  std_error = serr, conf_low = est - zcrit * serr,
                  conf_high = est + zcrit * serr, conf_level = conf_level,
                  se_type = unname(x$se_type[g]))
      if (!is.null(sh)) {
        s <- sh$coef[row, g]; ss <- sh$se[row, g]
        rec <- c(rec, list(share = s, share_std_error = ss,
                          share_conf_low = s - zcrit * ss,
                          share_conf_high = s + zcrit * ss,
                          share_defined = sh$defined[row],
                          share_denominator = sh$denominator,
                          share_se_type = sh$se_type,
                          share_units = sh$units, share_tol = sh$tol))
      }
      add(rec)
    }
    if (isTRUE(include_total)) {
      est <- x$total[row]; serr <- x$total_se[row]
      rec <- list(coefficient = label, component = "total_movement",
                  component_kind = "total", estimate = est,
                  std_error = serr, conf_low = est - zcrit * serr,
                  conf_high = est + zcrit * serr, conf_level = conf_level,
                  se_type = x$total_se_type)
      if (!is.null(sh)) {
        if (identical(share, "movement") && sh$defined[row]) {
          s <- 1; ss <- 0
        } else if (share %in% c("base", "base_fixed") &&
                   row <= p && sh$defined[row]) {
          s <- est / x$b_base[row]
          if (identical(share, "base_fixed")) {
            ss <- serr / abs(x$b_base[row])
          } else if (isTRUE(x$absorbed_mask[row])) {
            # For a declared absorbed target, total and b_base are the same
            # estimator. Preserve the point value while imposing the exact
            # zero variance of their ratio.
            ss <- 0
          } else {
            denom <- x$b_base[row]
            ratio_var <- (x$total_cov[row, row] / denom^2 +
                          est^2 * x$base_cov[row, row] / denom^4 -
                          2 * est * x$cov_total_bbase[row, row] / denom^3)
            ss <- sqrt(max(0, ratio_var))
          }
        } else {
          s <- ss <- NA_real_
        }
        rec <- c(rec, list(share = s, share_std_error = ss,
                          share_conf_low = s - zcrit * ss,
                          share_conf_high = s + zcrit * ss,
                          share_defined = sh$defined[row],
                          share_denominator = sh$denominator,
                          share_se_type = sh$se_type,
                          share_units = sh$units, share_tol = sh$tol))
      }
      add(rec)
    }
    if (isTRUE(include_full) && row <= p) {
      rec <- list(coefficient = label, component = "full_model_residual",
                  component_kind = "full_model", estimate = x$b_full[row],
                  std_error = NA_real_, conf_low = NA_real_, conf_high = NA_real_,
                  conf_level = conf_level,
                  se_type = "not_available_in_public_contract")
      if (!is.null(sh)) {
        s <- if (share %in% c("base", "base_fixed") && sh$defined[row]) {
          x$b_full[row] / x$b_base[row]
        } else NA_real_
        rec <- c(rec, list(share = s, share_std_error = NA_real_,
                          share_conf_low = NA_real_, share_conf_high = NA_real_,
                          share_defined = sh$defined[row],
                          share_denominator = sh$denominator,
                          share_se_type = "not_available_for_full_model_residual",
                          share_units = sh$units, share_tol = sh$tol))
      }
      add(rec)
    }
  }
  do.call(rbind, records)
}

#' Linear contrast of Gelbach components
#'
#' Computes a covariance-aware linear combination of named contributions for
#' exactly one X1 coefficient. This is post-processing of the stored joint
#' covariance; no model is re-estimated.
#'
#' @param x An \code{xhdfe_gelbach} result.
#' @param focal One X1 name, one-based index, or logical mask selecting exactly
#'   one X1 coefficient.
#' @param groups A character vector of group names (unit weights) or a named
#'   numeric vector of arbitrary finite weights. Names must match
#'   \code{x$names}; duplicate and all-zero specifications are rejected.
#' @param conf_level Normal-approximation confidence level.
#' @return A one-row data frame containing \code{coefficient},
#'   \code{estimate}, \code{std_error}, \code{conf_low}, \code{conf_high},
#'   \code{conf_level}, and \code{se_type}. A contrast containing any FE block
#'   is labelled \code{joint_covariance_including_conditional_fe}; this label
#'   preserves the conditional-FE inference qualification.
#' @examples
#' \dontrun{
#' # Sum two components with unit weights
#' xhdfe_gelbach_contrast(
#'   fit, focal = "target", groups = c("human_capital", "job")
#' )
#'
#' # Difference between two components
#' xhdfe_gelbach_contrast(
#'   fit, focal = "target", groups = c(human_capital = 1, job = -1)
#' )
#' }
#' @export
xhdfe_gelbach_contrast <- function(x, focal, groups, conf_level = 0.95) {
  if (!inherits(x, "xhdfe_gelbach")) {
    stop("x must be an xhdfe_gelbach result", call. = FALSE)
  }
  row <- .gelbach_result_rows(x, focal, FALSE)
  if (length(row) != 1L) stop("contrast requires exactly one focal coefficient", call. = FALSE)
  if (is.character(groups)) {
    if (any(!groups %in% x$names)) stop("contrast contains an unknown group", call. = FALSE)
    if (anyDuplicated(groups)) stop("contrast group names must be unique", call. = FALSE)
    w <- as.numeric(x$names %in% groups)
  } else {
    if (!is.numeric(groups) || is.null(names(groups)) ||
        any(!names(groups) %in% x$names)) {
      stop("numeric groups must be a named weight vector", call. = FALSE)
    }
    if (anyDuplicated(names(groups))) {
      stop("numeric contrast group names must be unique", call. = FALSE)
    }
    w <- setNames(rep(0, length(x$names)), x$names)
    w[names(groups)] <- groups
    w <- unname(w)
  }
  if (any(!is.finite(w)) || !any(w != 0)) {
    stop("contrast weights must be finite and not all zero", call. = FALSE)
  }
  if (length(conf_level) != 1L || !is.finite(conf_level) ||
      conf_level <= 0 || conf_level >= 1) {
    stop("conf_level must lie strictly between zero and one", call. = FALSE)
  }
  V <- .gelbach_row_cov(x, row)
  d <- x$delta[row, ]
  est <- sum(w * d); serr <- sqrt(max(0, drop(t(w) %*% V %*% w)))
  zcrit <- stats::qnorm(0.5 + conf_level / 2)
  included <- x$names[w != 0]
  data.frame(coefficient = x$x1_names[row], estimate = est,
             std_error = serr, conf_low = est - zcrit * serr,
             conf_high = est + zcrit * serr, conf_level = conf_level,
             se_type = if (any(x$group_kinds[included] == "fe")) {
               "joint_covariance_including_conditional_fe"
             } else "joint_covariance",
             stringsAsFactors = FALSE)
}
