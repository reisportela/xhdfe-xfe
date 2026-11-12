# Exercise the new R argument and compiled bridge without installing a package.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
root <- normalizePath(args[1], mustWork = TRUE)
.libPaths(c(normalizePath(args[3], mustWork = TRUE), .libPaths()))
library(Rcpp)
api <- new.env(parent = globalenv())
dll <- dyn.load(normalizePath(args[2], mustWork = TRUE))
for (symbol in getDLLRegisteredRoutines(dll)[[".Call"]]) {
  assign(symbol$name, symbol, envir = api)
}
for (file in c("RcppExports.R", "akm.R")) {
  sys.source(file.path(root, "r/xhdfe/R", file), envir = api)
}

worker <- rep(c(0L, 0L, 1L, 1L), c(1L, 4L, 9L, 16L))
firm <- rep(c(0L, 1L, 0L, 1L), c(1L, 4L, 9L, 16L))
y <- rep(c(.2, 1.3, -.5, .7), c(1L, 4L, 9L, 16L))
components <- function(r) unlist(r$kss[c("var_alpha", "var_psi", "cov_alpha_psi")])
fit <- function(y, ...) suppressWarnings(api$xhdfe_akm_kss(
  y, worker, firm, num_threads = 1L, seed = 18112026, ...))
for (level in c("match", "obs")) {
  for (method in c("exact", "jla")) {
    old <- fit(y, leave_out_level = level, leverages = method)
    explicit <- fit(y, leave_out_level = level, leverages = method,
                    sigma_centering = "leaveouttwoway")
    stopifnot(identical(components(old), components(explicit)))
    inv <- fit(y, leave_out_level = level, leverages = method,
               sigma_centering = "person_year")
    shifted <- fit(y + 10, leave_out_level = level, leverages = method,
                   sigma_centering = "person_year")
    stopifnot(inv$converged, shifted$converged,
              identical(inv$sigma_centering, "person_year"),
              max(abs(components(inv) - components(shifted))) < 1e-10)
    if (level == "obs") stopifnot(max(abs(components(inv) - components(old))) < 1e-10)
  }
}
for (option in c("se_sigma_lowess")) {
  call <- list(y = y, worker = worker, firm = firm,
               sigma_centering = "person_year", num_threads = 1L)
  call[[option]] <- TRUE
  error <- tryCatch({ do.call(api$xhdfe_akm_kss, call); NULL }, error = identity)
  stopifnot(inherits(error, "error"), grepl("does not support", conditionMessage(error)))
}
error <- tryCatch({ fit(y, sigma_centering = "invalid"); NULL }, error = identity)
stopifnot(inherits(error, "error"))
tools::checkRd(file.path(root, "r/xhdfe/man/xhdfe_akm_kss.Rd"))
cat("AKM_CENTERING_R_CONTRACT_PASS\n")
