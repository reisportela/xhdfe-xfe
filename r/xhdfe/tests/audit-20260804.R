lib <- Sys.getenv("XHDFE_AUDIT_R_LIB", "/tmp/xhdfe-r-audit-lib")
.libPaths(c(lib, .libPaths()))
library(xhdfe)

set.seed(20260804)
n <- 240L
row <- 0:(n - 1L)
fe1 <- row %% 17L
fe2 <- row %% 13L
X <- cbind(x1 = rnorm(n), x2 = rnorm(n))
y <- drop(X %*% c(.4, -.7)) + .1 * fe1 - .05 * fe2 + rnorm(n)

y_bad <- y
y_bad[7L] <- Inf
stopifnot(inherits(try(xhdfe_fit(y_bad, X, list(fe1, fe2)), silent = TRUE),
                   "try-error"))
X_bad <- X
X_bad[9L, 2L] <- NaN
stopifnot(inherits(try(xhdfe_fit(y, X_bad, list(fe1, fe2)), silent = TRUE),
                   "try-error"))

extreme_scale <- seq(1, 2, length.out = 20L)
gram_error <- try(
  xhdfe_fit(extreme_scale, matrix(extreme_scale * 1e154, ncol = 1L),
            fes = list(), fit_intercept = FALSE,
            drop_singletons = FALSE, threads = 1L),
  silent = TRUE
)
stopifnot(inherits(gram_error, "try-error"),
          grepl("cross-product", gram_error, fixed = TRUE),
          grepl("rescale", gram_error, fixed = TRUE))

g1 <- xhdfe_fit(y, X, list(fe1, fe2), cluster = rep.int(0L, n),
                drop_singletons = FALSE, threads = 1L)
stopifnot(g1$num_clusters == 1L, all(is.na(g1$se)),
          all(is.na(g1$tvalues)), all(is.na(g1$pvalues)))

constant <- xhdfe_fit(rep.int(1, n), X, list(fe1, fe2),
                      drop_singletons = FALSE, threads = 1L)
stopifnot(constant$tss == 0, constant$tss_within == 0,
          is.na(constant$r2), is.na(constant$r2_within))

disconnected_fe1 <- c(0L, 0L, 1L, 1L, 2L, 2L, 3L, 3L)
disconnected_fe2 <- c(0L, 1L, 0L, 1L, 2L, 3L, 2L, 3L)
disconnected_X <- matrix(
  0:7 + c(0, .3, -.2, .4, .1, -.1, .2, -.3), ncol = 1L
)
disconnected_y <- 1.5 * disconnected_X[, 1L] +
  c(.1, -.1, .2, -.2, .3, -.3, .4, -.4)
default_dof <- xhdfe_fit(
  disconnected_y, disconnected_X, list(disconnected_fe1, disconnected_fe2),
  drop_singletons = FALSE, dof = NULL, threads = 1L
)
for (token_free in list(character(0), "", "   ", " , ")) {
  explicit_empty <- xhdfe_fit(
    disconnected_y, disconnected_X, list(disconnected_fe1, disconnected_fe2),
    drop_singletons = FALSE, dof = token_free, threads = 1L
  )
  stopifnot(explicit_empty$df_a == default_dof$df_a,
            identical(explicit_empty$fe_base_redundant,
                      default_dof$fe_base_redundant))
}

fw <- as.double(1L + row %% 3L)
expanded <- rep(seq_len(n), fw)
weighted <- xhdfe_fit(y, X, list(fe1, fe2), weights = fw,
                      weights_type = "frequency", drop_singletons = FALSE,
                      threads = 1L, tol = 1e-10)
literal <- xhdfe_fit(y[expanded], X[expanded, , drop = FALSE],
                     list(fe1[expanded], fe2[expanded]),
                     drop_singletons = FALSE, threads = 1L, tol = 1e-10)
stopifnot(max(abs(coef(weighted) - coef(literal))) <= 1e-12,
          max(abs(weighted$se - literal$se)) <= 1e-11,
          weighted$df_r == literal$df_r)

groups <- 1000L
chain1 <- rep.int(0:(groups - 1L), rep.int(3L, groups))
chain2 <- integer(3L * groups)
for (i in 0:(groups - 1L)) {
  pos <- 3L * i + 1:3
  chain2[pos] <- c(i, i, i + 1L)
}
chain1 <- c(chain1, groups - 1L)
chain2 <- c(chain2, groups)
chain_X <- cbind(rnorm(length(chain1)), rnorm(length(chain1)))
chain_y <- drop(chain_X %*% c(.7, -.2)) + rnorm(length(chain1))
messages <- character()
chain <- withCallingHandlers(
  xhdfe_fit(chain_y, chain_X, list(chain1, chain2),
            drop_singletons = FALSE, threads = 1L, tol = 1e-8,
            tolerance_mode = "reghdfe-comparable"),
  message = function(condition) {
    messages <<- c(messages, conditionMessage(condition))
    invokeRestart("muffleMessage")
  }
)
stopifnot(chain$converged, chain$precision_certified,
          !any(grepl("independent precision certificate", messages,
                     fixed = TRUE)))

forced_messages <- character()
forced_chain <- withCallingHandlers(
  xhdfe_fit(chain_y, chain_X, list(chain1, chain2),
            drop_singletons = FALSE, threads = 1L, tol = 1e-8,
            tolerance_mode = "reghdfe-comparable",
            absorption_method = "gauss-seidel"),
  message = function(condition) {
    forced_messages <<- c(forced_messages, conditionMessage(condition))
    invokeRestart("muffleMessage")
  }
)
stopifnot(forced_chain$converged, !forced_chain$precision_certified,
          any(grepl("independent precision certificate", forced_messages,
                    fixed = TRUE)))

if (identical(Sys.getenv("XHDFE_AUDIT_REQUIRE_CUDA"), "1")) {
  gpu_n <- 200000L
  gpu_row <- 0:(gpu_n - 1L)
  gpu_fes <- list(gpu_row %% 20000L, (gpu_row * 37L) %% 17003L)
  gpu_X <- cbind(sin(gpu_row * .001), cos(gpu_row * .0013))
  gpu_y <- drop(gpu_X %*% c(.6, -.25)) +
    .001 * gpu_fes[[1L]] - .0005 * gpu_fes[[2L]] +
    sin(gpu_row * .0021)
  cpu <- xhdfe_fit(gpu_y, gpu_X, gpu_fes, backend = "cpu", threads = 1L,
                   drop_singletons = FALSE)
  gpu <- xhdfe_fit(gpu_y, gpu_X, gpu_fes, backend = "cuda", threads = 1L,
                   drop_singletons = FALSE)
  stopifnot(cpu$converged, cpu$precision_certified,
            gpu$converged, gpu$precision_certified,
            isTRUE(gpu$gpu_used), identical(gpu$gpu_status, "used"),
            max(abs(coef(gpu) - coef(cpu))) <= 1e-10,
            max(abs(gpu$se - cpu$se)) <= 1e-8)
}

cat("PASS: R audit 20260804 remediation contracts\n")
