# Build/version information.

#' xhdfe build and backend information
#'
#' Reports the package version and how the compiled backend was built:
#' whether OpenMP threading is enabled, whether the optional CUDA absorber
#' was compiled in (install with \code{XHDFE_ENABLE_CUDA=ON}), the
#' compile-time default backend, and the current \code{XHDFE_GPU_BACKEND}
#' environment override if set. Use it to verify a GPU-capable installation
#' before requesting \code{backend = "cuda"}.
#'
#' @return An object of class \code{xhdfe_info}: a list with
#'   \code{version}, \code{cuda_enabled}, \code{openmp_enabled},
#'   \code{default_backend}, \code{compiler}, \code{march} (the
#'   \code{-march} target the binary was compiled for; \code{"portable"} means
#'   no package-level \code{-march} override and \code{"native"} means
#'   machine-specific), \code{cuda_arch} (e.g. \code{"sm_90"}, empty on CPU
#'   builds), \code{optimized}, \code{fast_math}, and \code{gpu_backend_env}.
#' @examples
#' xhdfe_info()
#' @seealso \code{\link{xhdfe}} (section \emph{GPU backends})
#' @name xhdfe_info
xhdfe_info <- function() {
  info <- .xhdfe_cpp_build_info()
  out <- list(
    version = as.character(utils::packageVersion("xhdfe")),
    cuda_enabled = info$cuda_enabled,
    openmp_enabled = info$openmp_enabled,
    default_backend = info$default_backend,
    compiler = info$compiler,
    march = info$march,
    cuda_arch = info$cuda_arch,
    optimized = info$optimized,
    fast_math = info$fast_math,
    gpu_backend_env = Sys.getenv("XHDFE_GPU_BACKEND", unset = "")
  )
  class(out) <- "xhdfe_info"
  out
}

print.xhdfe_info <- function(x, ...) {
  cat("xhdfe", x$version, "\n")
  cat("  OpenMP:", ifelse(x$openmp_enabled, "enabled", "DISABLED"), "\n")
  cat("  CUDA build:", ifelse(x$cuda_enabled,
                              paste0("enabled (", x$cuda_arch, ")"),
                              "not compiled"), "\n")
  cat("  Default backend:", x$default_backend, "\n")
  cat("  Compiler:", x$compiler, "| -march:", x$march, "\n")
  cat("  Reference flags:",
      ifelse(isTRUE(x$optimized) && isTRUE(x$fast_math),
             "yes (-O3, fast-math)",
             "NO -- not a reference-grade build"), "\n")
  if (nzchar(x$gpu_backend_env)) {
    cat("  XHDFE_GPU_BACKEND:", x$gpu_backend_env, "\n")
  }
  invisible(x)
}
