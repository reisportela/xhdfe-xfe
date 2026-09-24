# Internal helpers: formula dissection, id encoding, option assembly.

# Split `y ~ x1 + x2 | fe1 + fe2 | endo ~ inst` into its pieces.
# `~` is left-associative, so R parses the two-tilde form as
# `(y ~ x | fe | endo) ~ inst`: the top-level LHS is itself a formula and the
# top-level RHS holds the instruments.
split_xhdfe_formula <- function(fml) {
  if (!inherits(fml, "formula") && !(is.call(fml) && identical(fml[[1L]], as.name("~")))) {
    stop("`fml` must be a two-sided formula, e.g. y ~ x1 + x2 | fe1 + fe2",
         call. = FALSE)
  }
  if (length(fml) != 3L) {
    stop("`fml` must be a two-sided formula, e.g. y ~ x1 + x2 | fe1 + fe2",
         call. = FALSE)
  }
  lhs <- fml[[2L]]
  rhs <- fml[[3L]]

  instruments <- NULL
  if (is.call(lhs) && identical(lhs[[1L]], as.name("~"))) {
    # IV form: (y ~ exo | fe | endo) ~ inst
    if (length(lhs) != 3L) {
      stop("Malformed IV part; use `y ~ exo | fe | endo ~ instruments`",
           call. = FALSE)
    }
    instruments <- rhs
    rhs <- lhs[[3L]]
    lhs <- lhs[[2L]]
  }

  parts <- list()
  while (is.call(rhs) && identical(rhs[[1L]], as.name("|"))) {
    parts <- c(list(rhs[[3L]]), parts)
    rhs <- rhs[[2L]]
  }
  parts <- c(list(rhs), parts)

  if (is.null(instruments)) {
    if (length(parts) > 2L) {
      stop("Too many `|` parts in the formula (without IV, at most `y ~ x | fe`)",
           call. = FALSE)
    }
    list(lhs = lhs,
         regressors = parts[[1L]],
         fe = if (length(parts) >= 2L) parts[[2L]] else NULL,
         endogenous = NULL,
         instruments = NULL)
  } else {
    if (length(parts) > 3L) {
      stop("Too many `|` parts in the formula (with IV, at most `y ~ exo | fe | endo ~ inst`)",
           call. = FALSE)
    }
    list(lhs = lhs,
         regressors = parts[[1L]],
         fe = if (length(parts) == 3L) parts[[2L]] else NULL,
         endogenous = parts[[length(parts)]],
         instruments = instruments)
  }
}

# Parentheses group the FE grammar; they do not request arithmetic evaluation.
unwrap_fe_parentheses <- function(expr) {
  while (is.call(expr) && identical(expr[[1L]], as.name("("))) expr <- expr[[2L]]
  expr
}

# Decompose the fixed-effects part into additive terms.
split_plus_terms <- function(expr) {
  expr <- unwrap_fe_parentheses(expr)
  if (is.call(expr) && identical(expr[[1L]], as.name("+"))) {
    if (length(expr) == 2L) return(split_plus_terms(expr[[2L]]))
    return(c(split_plus_terms(expr[[2L]]), split_plus_terms(expr[[3L]])))
  }
  list(expr)
}

# Parse one FE term. Grammar (fixest-compatible):
#   fe          plain absorbed intercepts
#   fe[x1, x2]  absorbed intercepts + group-specific slopes (Stata fe##c.x)
#   fe[[x1]]    group-specific slopes only, no intercepts    (Stata fe#c.x)
#   f1^f2       combined fixed effect (interaction of two or more ids)
# Returns list(fe_expr, slope_exprs (list), include_intercept, label).
parse_fe_term <- function(term) {
  term <- unwrap_fe_parentheses(term)
  if (is.call(term) && identical(term[[1L]], as.name("^"))) {
    left <- parse_fe_term(term[[2L]])
    right <- parse_fe_term(term[[3L]])
    if (length(left$slopes)) {
      stop("place slopes after the complete FE interaction, e.g. (a^b)[s]", call. = FALSE)
    }
    carrier <- call("^", left$fe_expr, right$fe_expr)
    if (length(right$slopes)) {
      right$fe_expr <- carrier
      right$label <- paste0("(", deparse1(carrier), ")",
                            if (right$include_intercept) "[" else "[[",
                            paste(vapply(right$slopes, deparse1, ""), collapse = ","),
                            if (right$include_intercept) "]" else "]]")
      return(right)
    }
    return(list(fe_expr = carrier, slopes = list(), include_intercept = TRUE,
                label = deparse1(term)))
  }
  if (is.call(term) && identical(term[[1L]], as.name("[["))) {
    fe_expr <- term[[2L]]
    slopes <- as.list(term)[-(1:2)]
    list(fe_expr = fe_expr, slopes = slopes, include_intercept = FALSE,
         label = paste0(deparse1(fe_expr), "[[",
                        paste(vapply(slopes, deparse1, ""), collapse = ","), "]]"))
  } else if (is.call(term) && identical(term[[1L]], as.name("["))) {
    fe_expr <- term[[2L]]
    slopes <- as.list(term)[-(1:2)]
    list(fe_expr = fe_expr, slopes = slopes, include_intercept = TRUE,
         label = paste0(deparse1(fe_expr), "[",
                        paste(vapply(slopes, deparse1, ""), collapse = ","), "]"))
  } else {
    list(fe_expr = term, slopes = list(), include_intercept = TRUE,
         label = deparse1(term))
  }
}

# Evaluate an FE expression in `data`; `a^b` combines ids.
eval_fe_expr <- function(expr, data, env) {
  expr <- unwrap_fe_parentheses(expr)
  if (is.call(expr) && identical(expr[[1L]], as.name("^"))) {
    left <- eval_fe_expr(expr[[2L]], data, env)
    right <- eval_fe_expr(expr[[3L]], data, env)
    return(interaction(left, right, drop = TRUE, lex.order = TRUE))
  }
  eval(expr, data, env)
}

regressor_terms <- function(spec, data, env) {
  fml <- stats::as.formula(call("~", spec$regressors), env = env)
  if (!"." %in% all.names(spec$regressors)) return(stats::terms(fml, data = data))
  excluded <- unique(c(all.vars(spec$lhs), all.vars(spec$fe),
                       all.vars(spec$endogenous), all.vars(spec$instruments)))
  dot_data <- if (is.null(names(data))) data else data[setdiff(names(data), excluded)]
  empty_dot <- !is.null(data) && !length(names(dot_data))
  if (empty_dot) {
    marker <- ".__xhdfe_empty_dot__"
    while (marker %in% c(names(data), all.names(fml))) marker <- paste0(marker, "_")
    dot_data <- setNames(list(0), marker)
  }
  fml <- stats::as.formula(call("~", spec$lhs, spec$regressors), env = env)
  terms <- stats::delete.response(stats::terms(fml, data = dot_data))
  if (empty_dot) {
    factors <- attr(terms, "factors")
    if (marker %in% rownames(factors)) {
      remove <- which(factors[marker, ] != 0)
      if (length(remove)) terms <- stats::drop.terms(terms, remove, keep.response = FALSE)
    }
  }
  terms
}

# Convert an arbitrary id vector to int32 codes for the C++ core.
# Integers pass through verbatim (level handling is the core's job);
# everything else is encoded to codes by first appearance. NAs are an error
# at this level: the formula interface filters them beforehand, and the
# matrix interface must not silently turn an NA into a valid level (which
# match()/unique() would otherwise do for character/logical input).
to_ids <- function(x, label = "id") {
  if (anyNA(x)) {
    stop(sprintf("`%s` contains missing values; the matrix interface does ",
                 label),
         "not drop NAs -- use xhdfe() or filter beforehand", call. = FALSE)
  }
  if (is.factor(x)) {
    return(as.integer(x))
  }
  if (is.integer(x)) {
    return(x)
  }
  if (is.numeric(x)) {
    r <- range(x, na.rm = TRUE)
    if (all(x == trunc(x), na.rm = TRUE) &&
        r[1L] >= -.Machine$integer.max && r[2L] <= .Machine$integer.max) {
      return(as.integer(x))
    }
    return(match(x, unique(x)))
  }
  if (is.character(x) || is.logical(x)) {
    return(match(x, unique(x)))
  }
  stop(sprintf("cannot encode `%s` as fixed-effect/cluster ids", label),
       call. = FALSE)
}

# Normalize a cluster specification (formula, character names, vector,
# matrix, data.frame or list of vectors) into a named list of vectors.
normalize_cluster_spec <- function(cluster, data, env) {
  if (is.null(cluster)) return(list())
  if (inherits(cluster, "formula")) {
    cenv <- environment(cluster)
    if (!is.null(cenv)) env <- cenv
    expr <- cluster[[length(cluster)]]
    terms <- split_plus_terms(expr)
    out <- lapply(terms, function(tm) eval_fe_expr(tm, data, env))
    names(out) <- vapply(terms, deparse1, "")
    return(out)
  }
  if (is.character(cluster) && !is.null(data) &&
      all(cluster %in% names(data))) {
    out <- lapply(cluster, function(nm) data[[nm]])
    names(out) <- cluster
    return(out)
  }
  if (is.data.frame(cluster)) {
    return(as.list(cluster))
  }
  if (is.matrix(cluster)) {
    out <- lapply(seq_len(ncol(cluster)), function(j) cluster[, j])
    names(out) <- colnames(cluster)
    return(out)
  }
  if (is.list(cluster)) {
    return(cluster)
  }
  # bare vector
  out <- list(cluster)
  names(out) <- deparse1(substitute(cluster))
  out
}

# Map user-facing vcov strings to the core se_type plus cluster requirement.
resolve_vcov <- function(vcov, has_cluster) {
  if (is.null(vcov)) {
    return(if (has_cluster) "cluster" else "unadjusted")
  }
  lower <- tolower(vcov)
  if (lower %in% c("iid", "unadjusted", "unadj", "ols", "homoskedastic",
                   "classical")) {
    if (has_cluster) {
      stop("cluster ids supplied together with vcov = \"", vcov, "\"",
           call. = FALSE)
    }
    return("unadjusted")
  }
  if (lower %in% c("robust", "hc1", "heteroskedastic")) {
    if (has_cluster) {
      stop("cluster ids supplied together with vcov = \"", vcov, "\"; use vcov = \"cluster\"",
           call. = FALSE)
    }
    return("robust")
  }
  if (lower %in% c("cluster", "clustered")) {
    if (!has_cluster) {
      stop("vcov = \"cluster\" requires `cluster` (or a `~ id` vcov formula)",
           call. = FALSE)
    }
    return("cluster")
  }
  stop("Unknown vcov type: ", vcov, call. = FALSE)
}

gpu_status_label <- function(code) {
  switch(as.character(code),
         "0" = "not_requested",
         "1" = "used",
         "2" = "backend_unavailable",
         "3" = "gpu_absorption_not_converged",
         "4" = "gpu_backend_failed",
         "5" = "cpu_cache_or_profile_result",
         "unknown")
}

# Generalized symmetric inverse used for the Wald model F test (mirrors
# Stata's syminv + diag0cnt accounting in xhdfe.ado).
sym_ginv <- function(V) {
  ev <- eigen(V, symmetric = TRUE)
  tol <- max(abs(ev$values)) * .Machine$double.eps * nrow(V)
  keep <- ev$values > tol
  inv <- ev$vectors[, keep, drop = FALSE] %*%
    (t(ev$vectors[, keep, drop = FALSE]) / ev$values[keep])
  list(inv = inv, rank = sum(keep))
}
