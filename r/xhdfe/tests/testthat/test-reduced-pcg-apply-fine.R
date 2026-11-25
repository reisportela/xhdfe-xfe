# reduced_pcg_apply_fine, the finer blocks of the reduced tier's segment pass as a request of the fit (package D,
# round 11 of 24 November 2026): the value is converted first and the integer that will be passed is validated,
# so that an object whose as.integer() method returns another value is judged by that value; and the C++ entry
# refuses a present value outside -1 (automatic) and 1 to 8 instead of passing it on to the core as "no request".

set.seed(20261124)
fine_n <- 6000
fine_dd <- data.frame(
  f1 = sample.int(300, fine_n, replace = TRUE),
  f2 = sample.int(40, fine_n, replace = TRUE),
  x1 = rnorm(fine_n)
)
fine_dd$y <- 0.5 * fine_dd$x1 + rnorm(300)[fine_dd$f1] + rnorm(40)[fine_dd$f2] + rnorm(fine_n)

test_that("reduced_pcg_apply_fine accepts NULL, \"auto\" and the factors 1 to 8", {
  for (good in list(NULL, "auto", 1, 8L, 4)) {
    m <- xhdfe(y ~ x1 | f1 + f2, fine_dd, reduced_pcg_apply_fine = good)
    expect_true(is.finite(coef(m)[["x1"]]))
  }
  m1 <- xhdfe(y ~ x1 | f1 + f2, fine_dd, reduced_pcg_apply_fine = 1)
  expect_equal(m1$reduced_pcg_fine_request, "1")
  ma <- xhdfe(y ~ x1 | f1 + f2, fine_dd, reduced_pcg_apply_fine = "auto")
  expect_equal(ma$reduced_pcg_fine_request, "auto")
})

test_that("reduced_pcg_apply_fine refuses what is not NULL, \"auto\" or an integer from 1 to 8", {
  for (bad in list("8", "Auto", 0, 9, -1, 2.5, NA_real_, c(1, 2), TRUE)) {
    expect_error(xhdfe(y ~ x1 | f1 + f2, fine_dd, reduced_pcg_apply_fine = bad),
                 "reduced_pcg_apply_fine")
  }
})

test_that("an object whose as.integer() returns another value is judged by that value", {
  # A number 4 whose as.integer() method returns 0: before round 11 the number was validated and the
  # conversion (0, "no request") passed on; now the converted value is validated, and refused.
  assign("as.integer.xhdfe_test_fine_zero", function(x, ...) 0L, envir = globalenv())
  on.exit(rm("as.integer.xhdfe_test_fine_zero", envir = globalenv()), add = TRUE)
  zero <- structure(4, class = "xhdfe_test_fine_zero")
  expect_identical(as.integer(zero), 0L)
  expect_error(xhdfe(y ~ x1 | f1 + f2, fine_dd, reduced_pcg_apply_fine = zero), "reduced_pcg_apply_fine")
  # and one whose as.integer() returns 3 for a number 3 asks for 3
  assign("as.integer.xhdfe_test_fine_three", function(x, ...) 3L, envir = globalenv())
  on.exit(rm("as.integer.xhdfe_test_fine_three", envir = globalenv()), add = TRUE)
  m <- xhdfe(y ~ x1 | f1 + f2, fine_dd, reduced_pcg_apply_fine = structure(3, class = "xhdfe_test_fine_three"))
  expect_equal(m$reduced_pcg_fine_request, "3")
})

test_that("the C++ entry validates the scalar request before converting to integer", {
  y <- fine_dd$y
  X <- matrix(fine_dd$x1, ncol = 1)
  fes <- list(as.integer(fine_dd$f1), as.integer(fine_dd$f2))
  base_opts <- list(tol = 1e-8, max_iter = 10000L, tolerance_mode = "reghdfe-comparable", num_threads = 1L,
                    se_type = "unadjusted")
  for (bad in list(8.5, -1.5, 1.5, 0L, 9L, -2L, 2^31, NA_integer_, NA_real_,
                   NaN, Inf, -Inf, TRUE, "8", factor("8"), c(1, 8), numeric())) {
    expect_error(xhdfe:::.xhdfe_cpp_partial_out(y, X, fes, NULL, list(),
                                                c(base_opts, list(reduced_pcg_apply_fine = bad))),
                 "reduced_pcg_apply_fine")
  }
  for (good in list(-1L, -1.0, 1L, 8L, 8.0, NULL)) {
    res <- xhdfe:::.xhdfe_cpp_partial_out(y, X, fes, NULL, list(), c(base_opts, list(reduced_pcg_apply_fine = good)))
    expect_true(isTRUE(res$converged))
  }
})
