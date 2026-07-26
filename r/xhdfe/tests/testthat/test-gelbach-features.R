gelbach_feature_fixture <- function(seed = 20260725L) {
  set.seed(seed)
  n_clusters <- 36L
  cluster_size <- 8L
  n <- n_clusters * cluster_size
  cluster <- rep(seq_len(n_clusters), each = cluster_size)
  common <- rep(seq_len(cluster_size), n_clusters)
  added <- rep(seq_len(18L), length.out = n)
  x1 <- cbind(target = rnorm(n), baseline_control = rnorm(n))
  human <- cbind(
    education = 0.35 * x1[, 1L] + rnorm(n),
    experience = -0.20 * x1[, 2L] + rnorm(n)
  )
  job <- 0.25 * x1[, 1L] - 0.15 * x1[, 2L] + rnorm(n)
  y <- drop(
    x1 %*% c(0.9, -0.45) +
      human %*% c(0.6, -0.25) + 0.4 * job +
      rnorm(n_clusters, sd = 0.6)[cluster] +
      rnorm(cluster_size, sd = 0.4)[common] +
      rnorm(n, sd = 0.35)
  )
  list(
    y = y, x1 = x1,
    groups = list(human = human, job = job),
    common = list(year = common), added = list(occupation = added),
    cluster = cluster,
    weights = runif(n, 0.4, 2.2)
  )
}

.gelbach_feature_subset <- function(values, index) {
  if (is.matrix(values) || is.data.frame(values)) {
    return(values[index, , drop = FALSE])
  }
  values[index]
}


test_that("Gelbach iid bootstrap is reproducible and matches a full-refit oracle", {
  data <- gelbach_feature_fixture()
  set.seed(314159)
  rng_before <- .Random.seed
  kind_before <- RNGkind()

  fit <- xhdfe_gelbach_bootstrap(
    data$y, data$x1,
    x2_groups = data$groups,
    common_fes = data$common,
    vce = "cluster", cluster = data$cluster,
    method = "pairs", reps = 11L, seed = 12345L,
    min_valid_reps = 10L
  )

  expect_identical(.Random.seed, rng_before)
  expect_identical(RNGkind(), kind_before)
  expect_equal(fit$bootstrap$reps_valid, 11L, tolerance = 0)
  expect_equal(nrow(fit$bootstrap$ledger), 11L, tolerance = 0)
  expect_identical(fit$bootstrap$point_vce, "cluster")
  expect_identical(
    fit$bootstrap$replicate_vce,
    "unadjusted_point_functional_only"
  )
  expect_identical(
    fit$bootstrap$interval_status,
    "resampling_based_not_a_nonregularity_cure"
  )
  expect_setequal(
    names(fit$bootstrap$intervals),
    c(
      "delta", "total", "b_base", "b_full", "share_base",
      "share_movement", "full_share_base", "total_share_base"
    )
  )

  repeated <- xhdfe_gelbach_bootstrap(
    data$y, data$x1,
    x2_groups = data$groups,
    common_fes = data$common,
    vce = "cluster", cluster = data$cluster,
    method = "pairs", reps = 11L, seed = 12345L,
    min_valid_reps = 10L
  )
  expect_identical(
    fit$bootstrap$draws$delta,
    repeated$bootstrap$draws$delta
  )

  old_kind <- RNGkind()
  old_seed <- .Random.seed
  RNGkind("L'Ecuyer-CMRG")
  set.seed(12345L)
  index <- sample.int(length(data$y), length(data$y), replace = TRUE)
  do.call(RNGkind, as.list(old_kind))
  assign(".Random.seed", old_seed, envir = .GlobalEnv)
  oracle <- xhdfe_gelbach(
    data$y[index], data$x1[index, , drop = FALSE],
    x2_groups = lapply(data$groups, .gelbach_feature_subset, index = index),
    common_fes = lapply(data$common, `[`, index),
    vce = "unadjusted"
  )
  expect_equal(
    unname(fit$bootstrap$draws$delta[1L, , ]),
    unname(oracle$delta),
    tolerance = 0
  )
  expect_equal(
    unname(fit$bootstrap$draws$b_base[1L, ]),
    unname(oracle$b_base),
    tolerance = 0
  )
  expect_equal(
    unname(fit$bootstrap$draws$b_full[1L, ]),
    unname(oracle$b_full),
    tolerance = 0
  )

  alpha <- 1 - fit$bootstrap$conf_level
  expected <- unname(stats::quantile(
    fit$bootstrap$draws$delta[, 1L, 1L],
    c(alpha / 2, 1 - alpha / 2), type = 7
  ))
  expect_equal(
    c(
      fit$bootstrap$intervals$delta$low[1L, 1L],
      fit$bootstrap$intervals$delta$high[1L, 1L]
    ),
    expected,
    tolerance = 0
  )
})


test_that("declared-cluster bootstrap and basic intervals match their oracle", {
  data <- gelbach_feature_fixture()
  fit <- xhdfe_gelbach_bootstrap(
    data$y, data$x1,
    x2_groups = data$groups,
    common_fes = data$common,
    method = "cluster_pairs",
    bootstrap_cluster = data$cluster,
    reps = 9L, seed = 54321L, min_valid_reps = 8L,
    ci_method = "basic"
  )
  expect_identical(fit$bootstrap$resampling_unit, "declared_cluster")
  expect_equal(fit$bootstrap$reps_valid, 9L, tolerance = 0)

  groups <- split(seq_along(data$cluster), data$cluster)
  old_kind <- RNGkind()
  old_seed <- .Random.seed
  RNGkind("L'Ecuyer-CMRG")
  set.seed(54321L)
  selected <- sample.int(length(groups), length(groups), replace = TRUE)
  index <- unlist(groups[selected], use.names = FALSE)
  do.call(RNGkind, as.list(old_kind))
  assign(".Random.seed", old_seed, envir = .GlobalEnv)
  oracle <- xhdfe_gelbach(
    data$y[index], data$x1[index, , drop = FALSE],
    x2_groups = lapply(data$groups, .gelbach_feature_subset, index = index),
    common_fes = lapply(data$common, `[`, index),
    vce = "unadjusted"
  )
  expect_equal(
    unname(fit$bootstrap$draws$delta[1L, , ]),
    unname(oracle$delta),
    tolerance = 0
  )

  alpha <- 1 - fit$bootstrap$conf_level
  quantiles <- unname(stats::quantile(
    fit$bootstrap$draws$total[, 1L],
    c(alpha / 2, 1 - alpha / 2), type = 7
  ))
  expect_equal(
    c(
      fit$bootstrap$intervals$total$low[1L],
      fit$bootstrap$intervals$total$high[1L]
    ),
    c(
      2 * fit$total[1L] - quantiles[2L],
      2 * fit$total[1L] - quantiles[1L]
    ),
    tolerance = 0
  )
})


test_that("Gelbach bootstrap can require real CUDA in every valid refit", {
  skip_if(Sys.getenv("XHDFE_TEST_CUDA") != "1")
  data <- gelbach_feature_fixture()
  for (method in c("pairs", "cluster_pairs")) {
    fit <- xhdfe_gelbach_bootstrap(
      data$y, data$x1,
      x2_groups = data$groups,
      common_fes = data$common,
      fes = data$added,
      method = method,
      bootstrap_cluster = if (method == "cluster_pairs") {
        data$cluster
      } else NULL,
      reps = 3L, seed = 919L, min_valid_reps = 3L,
      gpu = TRUE, require_gpu_used = TRUE
    )
    expect_true(fit$gpu_used)
    expect_identical(fit$gpu_backend, "cuda")
    expect_identical(fit$gpu_status, "used")
    expect_true(fit$bootstrap$gpu_required)
    expect_true(fit$bootstrap$gpu_used_all_valid)
    expect_true(all(fit$bootstrap$ledger$status == "valid"))
    expect_true(all(fit$bootstrap$ledger$gpu_used))
  }
})


test_that("bootstrap validation is explicit and failure accounting is audible", {
  data <- gelbach_feature_fixture()
  expect_error(
    xhdfe_gelbach_bootstrap(
      data$y, data$x1, x2_groups = data$groups,
      method = "cluster_pairs", reps = 2L
    ),
    "requires a non-missing bootstrap_cluster"
  )
  expect_error(
    xhdfe_gelbach_bootstrap(
      data$y, data$x1, x2_groups = data$groups,
      method = "pairs", bootstrap_cluster = data$cluster, reps = 2L
    ),
    "only valid for cluster_pairs"
  )
  expect_error(
    xhdfe_gelbach_bootstrap(
      data$y, data$x1, x2_groups = data$groups,
      weights = rep(1, length(data$y)), fweights = TRUE, reps = 2L
    ),
    "expanded-sample"
  )
  expect_error(
    xhdfe_gelbach_bootstrap(
      data$y, data$x1, x2_groups = data$groups,
      require_gpu_used = TRUE, reps = 2L
    ),
    "requires gpu = TRUE"
  )

  n <- 80L
  rare <- as.numeric(seq_len(n) == 1L)
  z <- seq_len(n) / n + rnorm(n, sd = 0.2)
  y <- 0.8 * rare + 0.4 * z + rnorm(n)
  unstable <- suppressWarnings(xhdfe_gelbach_bootstrap(
    y, cbind(rare = rare), x2_groups = list(z = z),
    reps = 19L, seed = 101L, min_valid_reps = 2L
  ))
  expect_gt(unstable$bootstrap$reps_failed, 0L)
  expect_gt(unstable$bootstrap$reps_valid, 1L)
  expect_equal(
    nrow(unstable$bootstrap$ledger),
    unstable$bootstrap$reps_requested,
    tolerance = 0
  )
  expect_equal(
    sum(unstable$bootstrap$failure_counts),
    unstable$bootstrap$reps_failed,
    tolerance = 0
  )
  expect_error(
    suppressWarnings(xhdfe_gelbach_bootstrap(
      y, cbind(rare = rare), x2_groups = list(z = z),
      reps = 19L, seed = 101L, min_valid_reps = 19L
    )),
    "failed closed"
  )

  weighted <- xhdfe_gelbach_bootstrap(
    data$y, data$x1,
    x2_groups = data$groups,
    common_fes = data$common,
    weights = data$weights,
    reps = 3L, seed = 818L, min_valid_reps = 3L,
    store_draws = FALSE
  )
  point <- xhdfe_gelbach(
    data$y, data$x1,
    x2_groups = data$groups,
    common_fes = data$common,
    weights = data$weights
  )
  expect_equal(weighted$b_full, point$b_full, tolerance = 0)
  expect_null(weighted$bootstrap$draws)
  expect_false(weighted$bootstrap$draws_stored)
})


test_that("tables and waterfall reporting preserve the accounting identity", {
  data <- gelbach_feature_fixture()
  fit <- xhdfe_gelbach_bootstrap(
    data$y, data$x1,
    x2_groups = data$groups,
    common_fes = data$common,
    reps = 7L, seed = 787L, min_valid_reps = 7L
  )
  snapshot <- list(
    b_base = fit$b_base,
    b_full = fit$b_full,
    delta = fit$delta
  )
  table <- xhdfe_gelbach_etable(
    fit,
    panels = c("levels", "share_full", "share_explained"),
    format = "data.frame",
    keep = "human",
    labels = c(human = "Human capital")
  )
  expect_setequal(
    unique(table$panel),
    c("levels", "share_base", "share_movement")
  )
  expect_true(all(c("base_model", "total", "full_model") %in%
    table$component_kind))
  expect_true("Human capital" %in% table$component)
  expect_false("job" %in% table$component_name)
  expect_true("bootstrap_percentile" %in% table$confidence_method)

  markdown <- xhdfe_gelbach_etable(
    fit, format = "markdown", caption = "A table"
  )
  latex <- xhdfe_gelbach_etable(
    fit, format = "latex", caption = "A_table"
  )
  html <- xhdfe_gelbach_etable(
    fit, format = "html", caption = "<A>"
  )
  expect_match(markdown, "| Panel |", fixed = TRUE)
  expect_match(latex, "\\begin{table}", fixed = TRUE)
  expect_match(latex, "A\\_table", fixed = TRUE)
  expect_match(latex, "\n\\footnotesize ", fixed = TRUE)
  expect_false(grepl("\n\\\\footnotesize ", latex, fixed = TRUE))
  expect_match(html, "&lt;A&gt;", fixed = TRUE)
  full_rows <- table$component_name == "full_model_residual"
  full_missing <- full_rows & is.na(table$std_error)
  full_available <- full_rows & is.finite(table$std_error)
  expect_true(all(
    table$confidence_method[full_missing] == "not_available"
  ))
  expect_true(all(
    table$confidence_method[full_available] ==
      "bootstrap_percentile"
  ))

  waterfall <- xhdfe_gelbach_waterfall_data(
    fit, focal = "target", keep = "human",
    labels = c(human = "Human capital")
  )
  expect_true(any(vapply(
    waterfall$rows,
    function(row) identical(row$kind, "filtered_aggregate") &&
      identical(row$members, "job"),
    logical(1)
  )))
  expect_lt(
    abs(waterfall$rows[[length(waterfall$rows)]]$waterfall_residual),
    2e-12
  )
  expect_error(
    xhdfe_gelbach_waterfall_data(fit, focal = "target", keep = "["),
    "invalid regular expression"
  )

  target <- tempfile(fileext = ".png")
  grDevices::png(target)
  plotted <- xhdfe_gelbach_coefplot(
    fit, focal = "target", keep = "human",
    labels = c(human = "Human capital")
  )
  grDevices::dev.off()
  expect_true(file.exists(target))
  expect_lt(
    abs(plotted$rows[[length(plotted$rows)]]$waterfall_residual),
    2e-12
  )

  expect_identical(fit$b_base, snapshot$b_base)
  expect_identical(fit$b_full, snapshot$b_full)
  expect_identical(fit$delta, snapshot$delta)
})
