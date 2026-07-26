# Bootstrap and reporting extensions for the Gelbach companion.
#
# These functions intentionally orchestrate the existing shared estimator.
# Every valid bootstrap replication is a complete xhdfe_gelbach() refit.

.gelbach_boot_positive_int <- function(value, name, minimum = 1L) {
  if (length(value) != 1L || is.na(value) || !is.numeric(value) ||
      !is.finite(value) || value != round(value) || value < minimum) {
    stop(name, " must be an integer >= ", minimum, call. = FALSE)
  }
  as.integer(value)
}

.gelbach_boot_subset <- function(values, index) {
  if (is.data.frame(values)) {
    return(values[index, , drop = FALSE])
  }
  if (is.matrix(values)) {
    return(values[index, , drop = FALSE])
  }
  values[index]
}

.gelbach_boot_subset_list <- function(values, index) {
  lapply(as.list(values %||% list()), .gelbach_boot_subset, index = index)
}

.gelbach_boot_point_arrays <- function(x) {
  list(
    delta = unname(x$delta),
    total = unname(x$total),
    b_base = unname(x$b_base),
    b_full = unname(x$b_full)
  )
}

.gelbach_boot_interval <- function(draws, point, conf_level, method) {
  nrep <- dim(draws)[1L]
  matrix_draws <- matrix(draws, nrow = nrep)
  point_values <- as.numeric(point)
  low <- high <- std_error <- rep(NA_real_, length(point_values))
  valid_counts <- integer(length(point_values))
  alpha <- 1 - conf_level
  for (column in seq_along(point_values)) {
    values <- matrix_draws[, column]
    values <- values[is.finite(values)]
    valid_counts[column] <- length(values)
    if (!length(values) || !is.finite(point_values[column])) next
    quantiles <- stats::quantile(
      values, probs = c(alpha / 2, 1 - alpha / 2),
      names = FALSE, type = 7
    )
    if (identical(method, "percentile")) {
      low[column] <- quantiles[1L]
      high[column] <- quantiles[2L]
    } else {
      low[column] <- 2 * point_values[column] - quantiles[2L]
      high[column] <- 2 * point_values[column] - quantiles[1L]
    }
    if (length(values) > 1L) std_error[column] <- stats::sd(values)
  }
  restore <- function(value) {
    if (is.null(dim(point))) {
      names(value) <- names(point)
      value
    } else {
      array(value, dim = dim(point), dimnames = dimnames(point))
    }
  }
  list(
    low = restore(low),
    high = restore(high),
    std_error = restore(std_error),
    valid_counts = restore(valid_counts)
  )
}

.gelbach_boot_streams <- function(reps, seed) {
  old_kind <- RNGkind()
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (had_seed) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  } else NULL
  restore <- function() {
    do.call(RNGkind, as.list(old_kind))
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }
  RNGkind("L'Ecuyer-CMRG")
  set.seed(seed)
  streams <- vector("list", reps)
  streams[[1L]] <- get(".Random.seed", envir = .GlobalEnv)
  if (reps > 1L) {
    for (replication in 2:reps) {
      streams[[replication]] <- parallel::nextRNGStream(
        streams[[replication - 1L]]
      )
    }
  }
  list(streams = streams, restore = restore)
}

#' Gelbach pairs bootstrap
#'
#' Fits the ordinary \code{xhdfe_gelbach()} result and completely re-estimates
#' it under observation-pairs or explicitly declared cluster-pairs
#' resampling. Replications use independent L'Ecuyer-CMRG streams, record
#' failures in a ledger and fail closed below \code{min_valid_reps}.
#' The same integer seed is reproducible within R, but does not request the
#' same resamples as Python's PCG64 streams or Stata's native sequential RNG.
#' Frequency-weight pairs bootstrap is refused because resampling compressed
#' rows differs from resampling their expanded copies. The
#' \code{$bootstrap} object records draws (when requested), interval matrices,
#' the complete ledger and failure counts. Percentile/basic intervals do not
#' by themselves cure product nonregularity, a weak denominator, few-cluster
#' inference or conditional recovered-FE uncertainty. This remains
#' specification accounting, not causal mediation.
#'
#' @param y,x1,x2_groups,fes Inputs passed to \code{xhdfe_gelbach()}.
#' @param method \code{"pairs"} or \code{"cluster_pairs"}.
#' @param bootstrap_cluster Required resampling unit for cluster pairs. It is
#'   never inferred from the VCE cluster.
#' @param reps Number of requested replications.
#' @param seed Nonnegative integer master seed.
#' @param conf_level Bootstrap confidence level.
#' @param ci_method \code{"percentile"} or \code{"basic"}.
#' @param min_valid_reps Minimum valid replications; defaults to 90 percent.
#' @param store_draws Store complete valid-replication draws.
#' @param require_gpu_used Fail a requested-GPU replication on fallback.
#' @param share_tol Absolute denominator threshold for bootstrap shares.
#' @param ... Further arguments passed to \code{xhdfe_gelbach()}, including
#'   \code{common_fes}, VCE, analytic weights and absorbed targets.
#' @return An \code{xhdfe_gelbach} object with a \code{bootstrap} member.
#' @examples
#' \dontrun{
#' boot <- xhdfe_gelbach_bootstrap(
#'   y, x1, x2_groups = list(job = job_controls),
#'   common_fes = list(year = year), method = "cluster_pairs",
#'   bootstrap_cluster = worker, reps = 499, seed = 42
#' )
#' boot$bootstrap$ledger
#' }
#' @export
xhdfe_gelbach_bootstrap <- function(
    y, x1, x2_groups = NULL, fes = NULL,
    method = c("pairs", "cluster_pairs"), bootstrap_cluster = NULL,
    reps = 999L, seed = 0L, conf_level = 0.95,
    ci_method = c("percentile", "basic"), min_valid_reps = NULL,
    store_draws = TRUE, require_gpu_used = FALSE, share_tol = 1e-12, ...) {
  method <- match.arg(method)
  ci_method <- match.arg(ci_method)
  reps <- .gelbach_boot_positive_int(reps, "reps", 2L)
  seed <- .gelbach_boot_positive_int(seed, "seed", 0L)
  if (length(conf_level) != 1L || !is.finite(conf_level) ||
      conf_level <= 0 || conf_level >= 1) {
    stop("conf_level must lie strictly between zero and one", call. = FALSE)
  }
  if (is.null(min_valid_reps)) {
    min_valid_reps <- ceiling(0.9 * reps)
  } else {
    min_valid_reps <- .gelbach_boot_positive_int(
      min_valid_reps, "min_valid_reps", 2L
    )
  }
  if (min_valid_reps > reps) {
    stop("min_valid_reps may not exceed reps", call. = FALSE)
  }
  if (length(store_draws) != 1L || is.na(store_draws) ||
      !is.logical(store_draws)) {
    stop("store_draws must be one non-missing logical value", call. = FALSE)
  }
  if (length(require_gpu_used) != 1L || is.na(require_gpu_used) ||
      !is.logical(require_gpu_used)) {
    stop("require_gpu_used must be one non-missing logical value",
         call. = FALSE)
  }
  if (length(share_tol) != 1L || !is.finite(share_tol) || share_tol < 0) {
    stop("share_tol must be finite and nonnegative", call. = FALSE)
  }

  y_vector <- as.numeric(y)
  n <- length(y_vector)
  x1_matrix <- as.matrix(x1)
  if (nrow(x1_matrix) != n) {
    stop("x1 must have the same number of rows as y", call. = FALSE)
  }
  dots <- list(...)
  if (isTRUE(require_gpu_used) && !isTRUE(dots$gpu)) {
    stop("require_gpu_used = TRUE requires gpu = TRUE", call. = FALSE)
  }
  common_fes <- as.list(dots$common_fes %||% list())
  weights <- dots$weights
  if (isTRUE(dots$fweights)) {
    stop(
      "frequency-weight pairs bootstrap is not exposed because record-level ",
      "and expanded-sample resampling are different estimands; expand the ",
      "data explicitly before bootstrapping",
      call. = FALSE
    )
  }
  if (!is.null(weights) && length(weights) != n) {
    stop("weights must have length n", call. = FALSE)
  }

  if (identical(method, "pairs")) {
    if (!is.null(bootstrap_cluster)) {
      stop("bootstrap_cluster is only valid for cluster_pairs",
           call. = FALSE)
    }
    cluster_groups <- NULL
    bootstrap_cluster_name <- NULL
  } else {
    bootstrap_cluster_name <- deparse(substitute(bootstrap_cluster))
    if (is.null(bootstrap_cluster) || length(bootstrap_cluster) != n ||
        anyNA(bootstrap_cluster)) {
      stop("cluster_pairs requires a non-missing bootstrap_cluster of length n",
           call. = FALSE)
    }
    cluster_code <- match(bootstrap_cluster, unique(bootstrap_cluster))
    cluster_groups <- split(seq_len(n), cluster_code)
    if (length(cluster_groups) < 2L) {
      stop("cluster-pairs bootstrap requires at least two clusters",
           call. = FALSE)
    }
  }

  point_args <- c(
    list(y = y, x1 = x1, x2_groups = x2_groups, fes = fes),
    dots
  )
  point <- do.call(xhdfe_gelbach, point_args)
  if (!isTRUE(point$converged)) {
    stop("the point decomposition did not converge; bootstrap aborted",
         call. = FALSE)
  }
  if (isTRUE(require_gpu_used) && !isTRUE(point$gpu_used)) {
    stop("GPU use was required but the point estimate used a fallback",
         call. = FALSE)
  }

  streams_state <- .gelbach_boot_streams(reps, seed)
  on.exit(streams_state$restore(), add = TRUE)
  valid <- list()
  ledger <- vector("list", reps)
  failures <- character(0)
  for (replication in seq_len(reps)) {
    assign(".Random.seed", streams_state$streams[[replication]],
           envir = .GlobalEnv)
    if (identical(method, "pairs")) {
      index <- sample.int(n, n, replace = TRUE)
    } else {
      selected <- sample.int(
        length(cluster_groups), length(cluster_groups), replace = TRUE
      )
      index <- unlist(cluster_groups[selected], use.names = FALSE)
    }
    rep_dots <- dots
    rep_dots$vce <- "unadjusted"
    rep_dots$cluster <- NULL
    rep_dots$common_fes <- .gelbach_boot_subset_list(common_fes, index)
    if (!is.null(weights)) rep_dots$weights <- weights[index]
    rep_args <- c(
      list(
        y = y_vector[index],
        x1 = x1_matrix[index, , drop = FALSE],
        x2_groups = .gelbach_boot_subset_list(x2_groups, index),
        fes = .gelbach_boot_subset_list(fes, index)
      ),
      rep_dots
    )
    fit <- tryCatch(
      suppressWarnings(do.call(xhdfe_gelbach, rep_args)),
      error = function(error) error
    )
    reason <- ""
    if (inherits(fit, "error")) {
      reason <- paste0(class(fit)[1L], ": ", conditionMessage(fit))
    } else if (!isTRUE(fit$converged)) {
      reason <- "decomposition_not_converged"
    } else if (isTRUE(require_gpu_used) && !isTRUE(fit$gpu_used)) {
      reason <- "gpu_fallback"
    } else {
      arrays <- .gelbach_boot_point_arrays(fit)
      p <- length(point$b_base)
      finite_slopes <- (
        all(is.finite(arrays$delta[seq_len(p), , drop = FALSE])) &&
        all(is.finite(arrays$total[seq_len(p)])) &&
        all(is.finite(arrays$b_base)) &&
        all(is.finite(arrays$b_full))
      )
      if (!finite_slopes) {
        reason <- "nonfinite_slope_result"
      } else if (!identical(dim(arrays$delta), dim(point$delta))) {
        reason <- "replicate_schema_changed"
      }
    }
    if (nzchar(reason)) {
      failures <- c(failures, reason)
      ledger[[replication]] <- data.frame(
        replication = replication, status = "failed", reason = reason,
        n_rows_drawn = length(index), n_obs_retained = NA_integer_,
        n_singletons_dropped = NA_integer_, identity_gap = NA_real_,
        gpu_used = NA, stringsAsFactors = FALSE
      )
    } else {
      valid[[length(valid) + 1L]] <- arrays
      ledger[[replication]] <- data.frame(
        replication = replication, status = "valid", reason = "",
        n_rows_drawn = length(index), n_obs_retained = fit$n_obs,
        n_singletons_dropped = fit$n_singletons_dropped,
        identity_gap = fit$identity_gap, gpu_used = fit$gpu_used,
        stringsAsFactors = FALSE
      )
    }
  }
  valid_count <- length(valid)
  failure_counts <- if (length(failures)) {
    sort(table(failures), decreasing = TRUE)
  } else integer(0)
  if (valid_count < min_valid_reps) {
    summary <- if (length(failure_counts)) {
      paste(
        paste0(as.integer(utils::head(failure_counts, 3L)), " x ",
               names(utils::head(failure_counts, 3L))),
        collapse = "; "
      )
    } else "no individual exception recorded"
    stop(
      "Gelbach bootstrap failed closed: ", valid_count, "/", reps,
      " valid replications, fewer than min_valid_reps=", min_valid_reps,
      ". Failures: ", summary, call. = FALSE
    )
  }
  if (valid_count < reps) {
    warning(
      "xhdfe_gelbach_bootstrap retained ", valid_count, "/", reps,
      " valid replications; inspect $bootstrap$ledger and $failure_counts.",
      call. = FALSE
    )
  }

  stack_matrix <- function(name) {
    aperm(simplify2array(lapply(valid, `[[`, name)), c(3L, 1L, 2L))
  }
  draws <- list(
    delta = stack_matrix("delta"),
    total = do.call(rbind, lapply(valid, `[[`, "total")),
    b_base = do.call(rbind, lapply(valid, `[[`, "b_base")),
    b_full = do.call(rbind, lapply(valid, `[[`, "b_full"))
  )
  k1 <- nrow(point$delta); groups <- ncol(point$delta)
  p <- length(point$b_base)
  draws$share_base <- array(NA_real_, c(valid_count, k1, groups))
  draws$share_movement <- array(NA_real_, c(valid_count, k1, groups))
  draws$full_share_base <- matrix(NA_real_, valid_count, p)
  draws$total_share_base <- matrix(NA_real_, valid_count, p)
  for (replication in seq_len(valid_count)) {
    for (row in seq_len(p)) {
      denominator <- draws$b_base[replication, row]
      if (is.finite(denominator) && abs(denominator) > share_tol) {
        draws$share_base[replication, row, ] <-
          draws$delta[replication, row, ] / denominator
        draws$full_share_base[replication, row] <-
          draws$b_full[replication, row] / denominator
        draws$total_share_base[replication, row] <-
          draws$total[replication, row] / denominator
      }
    }
    for (row in seq_len(k1)) {
      denominator <- draws$total[replication, row]
      if (is.finite(denominator) && abs(denominator) > share_tol) {
        draws$share_movement[replication, row, ] <-
          draws$delta[replication, row, ] / denominator
      }
    }
  }
  dimnames(draws$delta) <- list(
    NULL, rownames(point$delta), colnames(point$delta)
  )
  dimnames(draws$share_base) <- dimnames(draws$delta)
  dimnames(draws$share_movement) <- dimnames(draws$delta)
  colnames(draws$total) <- rownames(point$delta)
  colnames(draws$b_base) <- colnames(draws$b_full) <- point$x1_names
  colnames(draws$full_share_base) <- point$x1_names
  colnames(draws$total_share_base) <- point$x1_names

  point_arrays <- .gelbach_boot_point_arrays(point)
  point_share_base <- point_share_movement <-
    matrix(NA_real_, k1, groups, dimnames = dimnames(point$delta))
  for (row in seq_len(p)) {
    denominator <- point$b_base[row]
    if (is.finite(denominator) && abs(denominator) > share_tol) {
      point_share_base[row, ] <- point$delta[row, ] / denominator
    }
  }
  for (row in seq_len(k1)) {
    denominator <- point$total[row]
    if (is.finite(denominator) && abs(denominator) > share_tol) {
      point_share_movement[row, ] <- point$delta[row, ] / denominator
    }
  }
  point_full_share_base <- rep(NA_real_, p)
  point_total_share_base <- rep(NA_real_, p)
  valid_base <- is.finite(point$b_base) & abs(point$b_base) > share_tol
  point_full_share_base[valid_base] <-
    point$b_full[valid_base] / point$b_base[valid_base]
  point_total_share_base[valid_base] <-
    point$total[seq_len(p)][valid_base] / point$b_base[valid_base]
  names(point_full_share_base) <- point$x1_names
  names(point_total_share_base) <- point$x1_names
  interval_points <- c(
    point_arrays,
    list(
      share_base = point_share_base,
      share_movement = point_share_movement,
      full_share_base = point_full_share_base,
      total_share_base = point_total_share_base
    )
  )
  intervals <- lapply(names(interval_points), function(name) {
    .gelbach_boot_interval(
      draws[[name]], interval_points[[name]], conf_level, ci_method
    )
  })
  names(intervals) <- names(interval_points)

  point$bootstrap <- list(
    method = method,
    resampling_unit = if (identical(method, "pairs")) {
      "observation"
    } else "declared_cluster",
    bootstrap_cluster_name = bootstrap_cluster_name,
    seed = seed,
    rng = "LEcuyer-CMRG_independent_stream_per_replication",
    reps_requested = reps,
    reps_valid = valid_count,
    reps_failed = reps - valid_count,
    min_valid_reps = min_valid_reps,
    conf_level = conf_level,
    ci_method = ci_method,
    interval_status = "resampling_based_not_a_nonregularity_cure",
    share_tol = share_tol,
    component_names = point$names,
    coefficient_names = rownames(point$delta),
    ledger = do.call(rbind, ledger),
    failure_counts = failure_counts,
    intervals = intervals,
    draws = if (isTRUE(store_draws)) draws else NULL,
    draws_stored = isTRUE(store_draws),
    point_vce = point$vce,
    replicate_vce = "unadjusted_point_functional_only",
    gpu_required = isTRUE(require_gpu_used),
    gpu_used_all_valid = if (isTRUE(dots$gpu)) {
      all(vapply(ledger, function(row) {
        !identical(row$status, "valid") || isTRUE(row$gpu_used)
      }, logical(1)))
    } else FALSE
  )
  point
}

.gelbach_reporting_patterns <- function(value, name) {
  if (is.null(value)) return(character(0))
  value <- as.character(value)
  if (anyNA(value) || any(!nzchar(value))) {
    stop(name, " patterns must be non-missing, non-empty strings",
         call. = FALSE)
  }
  value
}

.gelbach_reporting_select <- function(names, keep = NULL, drop = NULL,
                                       exact_match = FALSE) {
  keep <- .gelbach_reporting_patterns(keep, "keep")
  drop <- .gelbach_reporting_patterns(drop, "drop")
  matches <- function(name, patterns) {
    if (!length(patterns)) return(FALSE)
    if (isTRUE(exact_match)) return(name %in% patterns)
    any(vapply(patterns, function(pattern) {
      tryCatch(
        suppressWarnings(grepl(pattern, name, perl = TRUE)),
        error = function(error) {
          stop(
            "invalid keep/drop regular expression: ",
            conditionMessage(error), call. = FALSE
          )
        }
      )
    }, logical(1)))
  }
  names[vapply(names, function(name) {
    (!length(keep) || matches(name, keep)) &&
      !(length(drop) && matches(name, drop))
  }, logical(1))]
}

.gelbach_reporting_panels <- function(panels) {
  aliases <- c(
    levels = "levels",
    share_base = "share_base",
    share_full = "share_base",
    share_movement = "share_movement",
    share_explained = "share_movement"
  )
  if (length(panels) == 1L && identical(panels, "all")) {
    return(c("levels", "share_base", "share_movement"))
  }
  if (anyNA(panels) || any(!panels %in% names(aliases))) {
    stop(
      "panels must be all, levels, share_base/share_full, or ",
      "share_movement/share_explained", call. = FALSE
    )
  }
  normalized <- unname(aliases[panels])
  if (anyDuplicated(normalized)) stop("panels may not contain duplicates",
                                      call. = FALSE)
  normalized
}

.gelbach_reporting_boot <- function(x, key, index) {
  if (is.null(x$bootstrap)) return(NULL)
  interval <- x$bootstrap$intervals[[key]]
  extract <- function(value) {
    if (length(index) == 1L) return(value[index])
    do.call(`[`, c(list(value), as.list(index), list(drop = TRUE)))
  }
  list(
    low = extract(interval$low),
    high = extract(interval$high),
    std_error = extract(interval$std_error),
    valid_count = extract(interval$valid_counts),
    method = x$bootstrap$ci_method
  )
}

.gelbach_reporting_filtered_aggregate <- function(
    x, row, members, panel, share_tol) {
  k1 <- nrow(x$delta)
  groups <- seq_along(x$names)
  member_index <- match(members, x$names)
  weights <- numeric(length(groups)); weights[member_index] <- 1
  positions <- (groups - 1L) * k1 + row
  covariance <- x$cov[positions, positions, drop = FALSE]
  contributions <- x$delta[row, ]
  numerator <- sum(contributions[member_index])
  numerator_variance <- drop(t(weights) %*% covariance %*% weights)
  if (identical(panel, "levels")) {
    return(c(
      estimate = numerator,
      std_error = sqrt(max(0, numerator_variance))
    ))
  }
  if (identical(panel, "share_base")) {
    denominator <- x$b_base[row]
    if (!is.finite(denominator) || abs(denominator) <= share_tol) {
      return(c(estimate = NA_real_, std_error = NA_real_))
    }
    covariance_num_den <- sum(
      x$cov_delta_bbase[positions[member_index], row]
    )
    denominator_variance <- x$base_cov[row, row]
    variance <- (
      numerator_variance / denominator^2 +
        numerator^2 * denominator_variance / denominator^4 -
        2 * numerator * covariance_num_den / denominator^3
    )
    return(c(
      estimate = numerator / denominator,
      std_error = if (is.finite(variance)) sqrt(max(0, variance)) else NA_real_
    ))
  }
  denominator <- x$total[row]
  if (!is.finite(denominator) || abs(denominator) <= share_tol) {
    return(c(estimate = NA_real_, std_error = NA_real_))
  }
  gradient <- weights / denominator -
    numerator * rep(1, length(groups)) / denominator^2
  variance <- drop(t(gradient) %*% covariance %*% gradient)
  c(
    estimate = numerator / denominator,
    std_error = sqrt(max(0, variance))
  )
}

#' Tabulate a Gelbach decomposition
#'
#' @param x An \code{xhdfe_gelbach} result.
#' @param panels Panels to include.
#' @param format \code{"data.frame"} (alias \code{"df"}),
#'   \code{"markdown"} (alias \code{"md"}), \code{"latex"} (alias
#'   \code{"tex"}), \code{"html"}, or optional \code{"gt"}. Unlike the
#'   Python frontend, R has no separate \code{"records"} format because the
#'   native dependency-free result is already a data frame.
#' @param focal,keep,drop,exact_match,labels Reporting selectors.
#' @param include_other Aggregate filtered components as
#'   \code{Other (filtered)} so each panel preserves the accounting identity.
#'   The aggregate SE uses the summed sub-block of the joint covariance,
#'   including cross-component terms. Setting this to \code{FALSE} restores
#'   the former filtered shape and warns that displayed rows do not close.
#' @param digits Number of printed decimals.
#' @param caption,notes Optional table text.
#' @param conf_level Analytic confidence level.
#' @param interval \code{"auto"}, \code{"normal"}, or \code{"bootstrap"}.
#' @param share_tol Share denominator threshold.
#' @param share_t_min Minimum absolute denominator t-statistic for valid
#'   first-order delta-method share inference.
#' @return A data frame, string, or optional \code{gt} object. Filtered
#'   data-frame output labels the aggregate \code{"Other (filtered)"} with
#'   \code{component_name = "other_filtered"}.
#' @details
#' \code{panels = "all"} emits \code{levels}, \code{share_base} (alias
#' \code{share_full}) and \code{share_movement} (alias
#' \code{share_explained}). Selection by \code{keep}/\code{drop} uses regular
#' expressions unless \code{exact_match = TRUE}; named \code{labels} are
#' applied afterwards. When a bootstrap result is supplied,
#' \code{interval = "auto"} consumes its bootstrap component, endpoint and
#' share intervals. The output formats are data.frame/df, markdown/md,
#' latex/tex, html and optional gt. This is post-processing and never refits
#' the model. Default notes retain the point estimator's diagnostic
#' \code{x$notes} rather than silently dropping warnings.
#' @examples
#' \dontrun{
#' tab <- xhdfe_gelbach_etable(
#'   fit, panels = "all", keep = "human", exact_match = TRUE,
#'   labels = c(human = "Human capital"), include_other = TRUE,
#'   share_t_min = 3
#' )
#' }
#' @export
xhdfe_gelbach_etable <- function(
    x, panels = "all", format = "data.frame", focal = NULL,
    keep = NULL, drop = NULL, exact_match = FALSE, labels = NULL,
    include_other = TRUE,
    digits = 3L, caption = NULL, notes = NULL, conf_level = 0.95,
    interval = c("auto", "normal", "bootstrap"), share_tol = 1e-12,
    share_t_min = 3.0) {
  if (!inherits(x, "xhdfe_gelbach")) {
    stop("x must be an xhdfe_gelbach result", call. = FALSE)
  }
  interval <- match.arg(interval)
  use_bootstrap <- identical(interval, "bootstrap") ||
    (identical(interval, "auto") && !is.null(x$bootstrap))
  if (use_bootstrap && is.null(x$bootstrap)) {
    stop("bootstrap intervals requested but x has no bootstrap", call. = FALSE)
  }
  digits <- .gelbach_boot_positive_int(digits, "digits", 0L)
  formats <- c("data.frame", "df", "markdown", "md", "latex", "tex",
               "html", "gt")
  if (length(format) != 1L || !format %in% formats) {
    stop("format must be data.frame/df, markdown/md, latex/tex, html or gt",
         call. = FALSE)
  }
  if (is.null(labels)) labels <- character(0)
  labels <- unlist(labels)
  if (length(labels) && (is.null(names(labels)) ||
      any(!names(labels) %in% x$names))) {
    stop("labels must be named by existing component names", call. = FALSE)
  }
  selected <- .gelbach_reporting_select(
    x$names, keep, drop, exact_match
  )
  omitted <- setdiff(x$names, selected)
  if (length(omitted) && !isTRUE(include_other)) {
    warning(
      "xhdfe_gelbach_etable: keep/drop filtered components without an ",
      "Other (filtered) aggregate; displayed components do not preserve ",
      "the Gelbach accounting identity.",
      call. = FALSE
    )
  }
  panel_names <- .gelbach_reporting_panels(panels)
  records <- list(); cursor <- 0L
  for (panel in panel_names) {
    share <- if (identical(panel, "levels")) NULL else if (
      identical(panel, "share_base")
    ) "base" else "movement"
    rows <- suppressWarnings(xhdfe_gelbach_tidy(
      x, focal = focal, include_intercept = FALSE,
      include_total = TRUE, include_full = TRUE,
      conf_level = conf_level, share = share, share_tol = share_tol,
      share_t_min = share_t_min
    ))
    coefficient_names <- unique(rows$coefficient)
    if (panel %in% c("levels", "share_base")) {
      for (coefficient in coefficient_names) {
        coefficient_index <- match(coefficient, rownames(x$delta))
        boot <- NULL
        if (identical(panel, "levels")) {
          estimate <- x$b_base[coefficient_index]
          std_error <- sqrt(max(
            0, x$base_cov[coefficient_index, coefficient_index]
          ))
          zcrit <- stats::qnorm(0.5 + conf_level / 2)
          low <- estimate - zcrit * std_error
          high <- estimate + zcrit * std_error
          if (use_bootstrap) {
            boot <- .gelbach_reporting_boot(
              x, "b_base", coefficient_index
            )
          }
        } else {
          estimate <- 1; std_error <- 0; low <- high <- 1
        }
        if (!is.null(boot)) {
          low <- boot$low; high <- boot$high
          std_error <- boot$std_error
          ci_method <- paste0("bootstrap_", boot$method)
          valid_count <- boot$valid_count
        } else {
          ci_method <- if (identical(panel, "share_base")) {
            "identity"
          } else if (use_bootstrap) "not_available" else "normal_delta"
          valid_count <- NA_integer_
        }
        cursor <- cursor + 1L
        records[[cursor]] <- data.frame(
          panel = panel, coefficient = coefficient,
          component = "base_model", component_name = "base_model",
          component_kind = "base_model", estimate = estimate,
          std_error = std_error, conf_low = low, conf_high = high,
          conf_level = if (!is.null(boot)) {
            x$bootstrap$conf_level
          } else conf_level,
          confidence_method = ci_method,
          bootstrap_valid_count = valid_count,
          stringsAsFactors = FALSE
        )
      }
    }
    for (record in seq_len(nrow(rows))) {
      row <- rows[record, , drop = FALSE]
      component <- row$component
      if (component %in% x$names && !component %in% selected) next
      coefficient_index <- match(row$coefficient, rownames(x$delta))
      if (identical(component, "total_movement") &&
          length(omitted) && isTRUE(include_other)) {
        aggregate <- .gelbach_reporting_filtered_aggregate(
          x, coefficient_index, omitted, panel, share_tol
        )
        zcrit <- stats::qnorm(0.5 + conf_level / 2)
        aggregate_method <- "normal_delta"
        if (!identical(panel, "levels") &&
            identical(
              row$share_interval_status,
              "weak_denominator_delta_method_unreliable"
            )) {
          aggregate_method <-
            "normal_delta_diagnostic_only_weak_denominator"
        }
        cursor <- cursor + 1L
        records[[cursor]] <- data.frame(
          panel = panel,
          coefficient = row$coefficient,
          component = "Other (filtered)",
          component_name = "other_filtered",
          component_kind = "filtered_aggregate",
          estimate = unname(aggregate["estimate"]),
          std_error = unname(aggregate["std_error"]),
          conf_low = unname(aggregate["estimate"] -
                            zcrit * aggregate["std_error"]),
          conf_high = unname(aggregate["estimate"] +
                             zcrit * aggregate["std_error"]),
          conf_level = conf_level,
          confidence_method = aggregate_method,
          bootstrap_valid_count = NA_integer_,
          stringsAsFactors = FALSE
        )
      }
      boot <- NULL
      if (identical(panel, "levels")) {
        estimate <- row$estimate; std_error <- row$std_error
        low <- row$conf_low; high <- row$conf_high
        if (use_bootstrap) {
          if (component %in% x$names) {
            boot <- .gelbach_reporting_boot(
              x, "delta", c(coefficient_index, match(component, x$names))
            )
          } else if (identical(component, "total_movement")) {
            boot <- .gelbach_reporting_boot(x, "total", coefficient_index)
          } else {
            boot <- .gelbach_reporting_boot(x, "b_full", coefficient_index)
          }
        }
      } else {
        estimate <- row$share; std_error <- row$share_std_error
        low <- row$share_conf_low; high <- row$share_conf_high
        if (use_bootstrap && component %in% x$names) {
          key <- if (identical(panel, "share_base")) {
            "share_base"
          } else "share_movement"
          boot <- .gelbach_reporting_boot(
            x, key, c(coefficient_index, match(component, x$names))
          )
        } else if (use_bootstrap &&
                   identical(component, "full_model_residual") &&
                   identical(panel, "share_base")) {
          boot <- .gelbach_reporting_boot(
            x, "full_share_base", coefficient_index
          )
        } else if (use_bootstrap &&
                   identical(component, "total_movement") &&
                   identical(panel, "share_base")) {
          boot <- .gelbach_reporting_boot(
            x, "total_share_base", coefficient_index
          )
        } else if (identical(component, "total_movement") &&
                   identical(panel, "share_movement") &&
                   is.finite(estimate)) {
          std_error <- 0; low <- high <- estimate
        }
      }
      if (!is.null(boot)) {
        low <- boot$low; high <- boot$high
        std_error <- boot$std_error
        ci_method <- paste0("bootstrap_", boot$method)
        valid_count <- boot$valid_count
      } else {
        if (use_bootstrap ||
            any(!is.finite(c(std_error, low, high)))) {
          ci_method <- "not_available"
        } else if (!identical(panel, "levels") &&
                   identical(
                     row$share_interval_status,
                     "weak_denominator_delta_method_unreliable"
                   )) {
          ci_method <-
            "normal_delta_diagnostic_only_weak_denominator"
        } else if (startsWith(
          as.character(row$confidence_interval_status),
          "diagnostic_only"
        )) {
          ci_method <- "normal_delta_diagnostic_only"
        } else {
          ci_method <- "normal_delta"
        }
        valid_count <- NA_integer_
      }
      cursor <- cursor + 1L
      records[[cursor]] <- data.frame(
        panel = panel,
        coefficient = row$coefficient,
        component = if (component %in% names(labels)) {
          unname(labels[component])
        } else component,
        component_name = component,
        component_kind = row$component_kind,
        estimate = estimate,
        std_error = std_error,
        conf_low = low,
        conf_high = high,
        conf_level = if (!is.null(boot)) {
          x$bootstrap$conf_level
        } else conf_level,
        confidence_method = ci_method,
        bootstrap_valid_count = valid_count,
        stringsAsFactors = FALSE
      )
    }
  }
  table <- do.call(rbind, records)
  default_note <- paste0(
    "Specification accounting, not causal mediation. VCE=", x$vce,
    "; identity=", x$identity_status, "."
  )
  if (!is.null(x$notes) && nzchar(trimws(x$notes))) {
    default_note <- paste(default_note, trimws(x$notes))
  }
  if (!is.null(x$bootstrap)) {
    default_note <- paste0(
      default_note, " Bootstrap=", x$bootstrap$method,
      ", valid reps=", x$bootstrap$reps_valid, "/",
      x$bootstrap$reps_requested,
      "; resampling intervals do not by themselves cure nonregularity."
    )
  }
  notes <- if (is.null(notes)) default_note else paste(default_note, notes)
  if (format %in% c("data.frame", "df")) return(table)
  format_number <- function(value) {
    ifelse(is.finite(value), formatC(value, digits = digits, format = "f"), "")
  }
  printable <- data.frame(
    Panel = table$panel,
    Coefficient = table$coefficient,
    Component = table$component,
    Estimate = format_number(table$estimate),
    `Std. error` = format_number(table$std_error),
    `CI low` = format_number(table$conf_low),
    `CI high` = format_number(table$conf_high),
    `CI method` = table$confidence_method,
    check.names = FALSE, stringsAsFactors = FALSE
  )
  if (format %in% c("markdown", "md")) {
    escape <- function(value) gsub("|", "\\\\|", value, fixed = TRUE)
    lines <- c(
      if (!is.null(caption)) c(paste0("**", escape(caption), "**"), ""),
      paste0("| ", paste(names(printable), collapse = " | "), " |"),
      paste0("|", paste(rep("---", ncol(printable)), collapse = "|"), "|")
    )
    for (row in seq_len(nrow(printable))) {
      lines <- c(lines, paste0(
        "| ", paste(escape(as.character(printable[row, ])),
                    collapse = " | "), " |"
      ))
    }
    return(paste(c(lines, "", paste0("Notes: ", notes)), collapse = "\n"))
  }
  if (format %in% c("latex", "tex")) {
    escape <- function(value) {
      value <- gsub("\\\\", "\\\\textbackslash{}", value)
      gsub("([&%$#_{}])", "\\\\\\1", value, perl = TRUE)
    }
    lines <- c(
      "\\begin{table}[htbp]", "\\centering",
      if (!is.null(caption)) paste0("\\caption{", escape(caption), "}"),
      "\\begin{tabular}{lllrrrrl}", "\\hline",
      paste0(paste(escape(names(printable)), collapse = " & "), " \\\\"),
      "\\hline"
    )
    for (row in seq_len(nrow(printable))) {
      lines <- c(lines, paste0(
        paste(escape(as.character(printable[row, ])), collapse = " & "),
        " \\\\"
      ))
    }
    lines <- c(
      lines, "\\hline", "\\end{tabular}",
      paste0("\\footnotesize ", escape(notes)), "\\end{table}"
    )
    return(paste(lines, collapse = "\n"))
  }
  if (identical(format, "html")) {
    escape <- function(value) {
      value <- gsub("&", "&amp;", value, fixed = TRUE)
      value <- gsub("<", "&lt;", value, fixed = TRUE)
      gsub(">", "&gt;", value, fixed = TRUE)
    }
    body <- vapply(seq_len(nrow(printable)), function(row) {
      paste0("<tr>", paste0(
        "<td>", escape(as.character(printable[row, ])), "</td>",
        collapse = ""
      ), "</tr>")
    }, character(1))
    return(paste0(
      "<table class=\"xhdfe-gelbach\">",
      if (!is.null(caption)) paste0("<caption>", escape(caption), "</caption>"),
      "<thead><tr>", paste0("<th>", escape(names(printable)), "</th>",
                            collapse = ""), "</tr></thead><tbody>",
      paste(body, collapse = ""), "</tbody><tfoot><tr><td colspan=\"8\">",
      escape(notes), "</td></tr></tfoot></table>"
    ))
  }
  if (!requireNamespace("gt", quietly = TRUE)) {
    stop("format = \"gt\" requires the optional gt package", call. = FALSE)
  }
  gt_table <- gt::gt(table)
  if (!is.null(caption)) gt_table <- gt::tab_header(gt_table, title = caption)
  gt_table
}

#' Gelbach waterfall data
#'
#' @param x An \code{xhdfe_gelbach} result.
#' @param focal Exactly one X1 coefficient.
#' @param keep,drop Regular expressions, or exact names when
#'   \code{exact_match = TRUE}.
#' @param exact_match Use exact component matching.
#' @param labels Named display labels.
#' @param include_other Aggregate filtered components as
#'   \code{Other (filtered)} to preserve the identity. Setting it to
#'   \code{FALSE} deliberately leaves a filtered residual.
#' @param share_tol Movement denominator threshold.
#' @return A list containing the waterfall rows and identity metadata.
#'   The final row contains \code{waterfall_residual}; it is approximately
#'   zero when \code{include_other = TRUE}.
#' @details Selection happens on original component names before labels.
#'   The residual aggregate makes the visual path an accounting identity; it
#'   does not add causal or path-dependent interpretation.
#' @examples
#' \dontrun{
#' specification <- xhdfe_gelbach_waterfall_data(
#'   fit, focal = "target", keep = "human",
#'   labels = c(human = "Human capital"), include_other = TRUE
#' )
#' }
#' @export
xhdfe_gelbach_waterfall_data <- function(
    x, focal = NULL, keep = NULL, drop = NULL, exact_match = FALSE,
    labels = NULL, include_other = TRUE, share_tol = 1e-12) {
  if (!inherits(x, "xhdfe_gelbach")) {
    stop("x must be an xhdfe_gelbach result", call. = FALSE)
  }
  selected_row <- .gelbach_result_rows(x, focal, FALSE)
  if (length(selected_row) != 1L) {
    stop("waterfall data requires exactly one focal coefficient",
         call. = FALSE)
  }
  row <- selected_row
  selected <- .gelbach_reporting_select(
    x$names, keep, drop, exact_match
  )
  omitted <- setdiff(x$names, selected)
  if (is.null(labels)) labels <- character(0)
  labels <- unlist(labels)
  if (length(labels) && (is.null(names(labels)) ||
      any(!names(labels) %in% x$names))) {
    stop("labels must be named by existing component names", call. = FALSE)
  }
  base <- x$b_base[row]; full <- x$b_full[row]; total <- x$total[row]
  rows <- list(list(
    position = 0L, name = "base_model", label = "Base model",
    kind = "endpoint", members = character(0), contribution = NA_real_,
    change = NA_real_, start = 0, end = base, value = base,
    share_of_movement = NA_real_
  ))
  plot_groups <- lapply(selected, function(name) list(name, name))
  if (isTRUE(include_other) && length(omitted)) {
    plot_groups[[length(plot_groups) + 1L]] <-
      list("other_filtered", omitted)
  }
  current <- base
  for (group in plot_groups) {
    name <- group[[1L]]; members <- group[[2L]]
    contribution <- sum(x$delta[row, members, drop = TRUE])
    ending <- current - contribution
    share <- if (is.finite(total) && abs(total) > share_tol) {
      contribution / total
    } else NA_real_
    rows[[length(rows) + 1L]] <- list(
      position = length(rows), name = name,
      label = if (identical(name, "other_filtered")) {
        "Other (filtered)"
      } else if (name %in% names(labels)) unname(labels[name]) else name,
      kind = if (identical(name, "other_filtered")) {
        "filtered_aggregate"
      } else unname(x$group_kinds[name]),
      members = members, contribution = contribution,
      change = -contribution, start = current, end = ending,
      value = ending, share_of_movement = share
    )
    current <- ending
  }
  rows[[length(rows) + 1L]] <- list(
    position = length(rows), name = "full_model", label = "Full model",
    kind = "endpoint", members = character(0), contribution = NA_real_,
    change = NA_real_, start = 0, end = full, value = full,
    share_of_movement = NA_real_, waterfall_residual = current - full
  )
  list(
    coefficient = rownames(x$delta)[row], base = base, full = full,
    movement = total, identity_gap = x$identity_gap,
    selected_components = selected, omitted_components = omitted,
    include_other = isTRUE(include_other), rows = rows
  )
}

#' Plot a Gelbach waterfall
#'
#' Uses base R graphics; filtered components are aggregated as
#' \code{Other (filtered)} by default so the visual remains an identity.
#'
#' @inheritParams xhdfe_gelbach_waterfall_data
#' @param annotate_shares Annotate shares of total movement.
#' @param main Plot title (the Python and Stata frontends call this
#'   \code{title}).
#' @param notes Optional footer.
#' @param col Endpoint, positive, negative and filtered colors.
#' @details This base-R graphics helper has no built-in file argument. Use an
#'   explicit graphics device such as \code{png()} or \code{pdf()}, call the
#'   function, and close the device with \code{dev.off()}.
#' @return Invisibly, the waterfall specification.
#' @examples
#' \dontrun{
#' xhdfe_gelbach_coefplot(
#'   fit, focal = "target", keep = "human", annotate_shares = TRUE,
#'   include_other = TRUE
#' )
#' }
#' @export
xhdfe_gelbach_coefplot <- function(
    x, focal = NULL, annotate_shares = TRUE, main = NULL,
    keep = NULL, drop = NULL, exact_match = FALSE, labels = NULL,
    notes = NULL, include_other = TRUE, share_tol = 1e-12,
    col = c("#374151", "#2563eb", "#dc2626", "#6b7280")) {
  specification <- xhdfe_gelbach_waterfall_data(
    x, focal = focal, keep = keep, drop = drop,
    exact_match = exact_match, labels = labels,
    include_other = include_other, share_tol = share_tol
  )
  rows <- specification$rows
  values <- unlist(lapply(rows, function(row) c(row$start, row$end)))
  limits <- range(c(0, values), finite = TRUE)
  padding <- max(diff(limits) * 0.12, 1e-8)
  graphics::plot(
    NA, xlim = c(0.5, length(rows) + 0.5),
    ylim = limits + c(-padding, padding),
    xaxt = "n", xlab = "", ylab = paste0(
      "Coefficient: ", specification$coefficient
    ),
    main = main %||% paste0(
      "Gelbach decomposition of ", specification$coefficient
    )
  )
  for (index in seq_along(rows)) {
    row <- rows[[index]]
    if (identical(row$kind, "endpoint")) {
      color <- col[1L]
      lower <- min(0, row$value); upper <- max(0, row$value)
    } else {
      color <- if (identical(row$kind, "filtered_aggregate")) {
        col[4L]
      } else if (row$contribution >= 0) col[2L] else col[3L]
      lower <- min(row$start, row$end); upper <- max(row$start, row$end)
    }
    graphics::rect(index - 0.36, lower, index + 0.36, upper,
                   col = color, border = NA)
    if (isTRUE(annotate_shares) && !identical(row$kind, "endpoint") &&
        is.finite(row$share_of_movement)) {
      graphics::text(
        index, (row$start + row$end) / 2,
        labels = sprintf("%.1f%%", 100 * row$share_of_movement),
        col = "white", cex = 0.8, font = 2
      )
    }
  }
  if (length(rows) > 1L) {
    for (index in seq_len(length(rows) - 1L)) {
      left <- rows[[index]]
      right <- rows[[index + 1L]]
      left_level <- if (identical(left$kind, "endpoint")) {
        left$value
      } else left$end
      right_level <- if (identical(right$kind, "endpoint")) {
        right$value
      } else right$start
      graphics::segments(
        index + 0.36, left_level, index + 0.64, right_level,
        col = "#9ca3af", lty = 2
      )
    }
  }
  graphics::abline(h = 0, col = "#111827", lwd = 0.8)
  graphics::axis(
    1, at = seq_along(rows),
    labels = vapply(rows, `[[`, character(1), "label"),
    las = 2, cex.axis = 0.8
  )
  footer <- notes %||% paste(
    "Movement = base - full. Filtered components are aggregated as",
    "'Other' so the waterfall remains an accounting identity."
  )
  graphics::mtext(footer, side = 1, line = 6, adj = 0, cex = 0.75)
  invisible(specification)
}
