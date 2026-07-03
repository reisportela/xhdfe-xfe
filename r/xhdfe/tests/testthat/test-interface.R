# Formula interface, S3 methods, and cross-package validation.

set.seed(42)
n <- 8000
dd <- data.frame(
  f1 = sample.int(150, n, replace = TRUE),
  f2 = sample.int(30, n, replace = TRUE),
  x1 = rnorm(n),
  x2 = rnorm(n),
  z = rnorm(n),
  w = runif(n, 0.4, 2.5)
)
dd$y <- 1 + 0.5 * dd$x1 - 0.3 * dd$x2 + dd$f1 / 75 + dd$f2 / 15 + rnorm(n)

test_that("xhdfe matches lm exactly on dummy-expanded designs", {
  m <- xhdfe(y ~ x1 + x2 | f1 + f2, dd)
  l <- lm(y ~ x1 + x2 + factor(f1) + factor(f2), dd)
  expect_equal(unname(coef(m)[1:2]), unname(coef(l)[2:3]), tolerance = 1e-10)
  expect_equal(unname(m$se[1:2]),
               unname(sqrt(diag(vcov(l)))[2:3]), tolerance = 1e-10)
  expect_equal(m$r2, summary(l)$r.squared, tolerance = 1e-10)
  expect_equal(m$df_r, l$df.residual)
  expect_equal(unname(m$rmse), summary(l)$sigma, tolerance = 1e-10)
  # residuals align observation by observation
  expect_equal(unname(residuals(m)), unname(residuals(l)), tolerance = 1e-8)
})

test_that("no-FE fit matches lm including intercept, F and loglik", {
  m <- xhdfe(y ~ x1 + x2, dd)
  l <- lm(y ~ x1 + x2, dd)
  expect_equal(unname(coef(m)), unname(coef(l)[c(2, 3, 1)]), tolerance = 1e-12)
  expect_equal(unname(m$se), unname(sqrt(diag(vcov(l)))[c(2, 3, 1)]),
               tolerance = 1e-12)
  expect_equal(m$F_stat, summary(l)$fstatistic[["value"]], tolerance = 1e-8)
  expect_equal(as.numeric(logLik(m)), as.numeric(logLik(l)), tolerance = 1e-8)
})

test_that("weights match lm weighted fits", {
  m <- xhdfe(y ~ x1 + x2 | f1, dd, weights = ~w)
  l <- lm(y ~ x1 + x2 + factor(f1), dd, weights = w)
  expect_equal(unname(coef(m)[1:2]), unname(coef(l)[2:3]), tolerance = 1e-10)
  expect_equal(unname(m$se[1:2]), unname(sqrt(diag(vcov(l)))[2:3]),
               tolerance = 1e-9)
})

test_that("formula sugar: factors, interactions, transformations", {
  dd2 <- dd
  dd2$cat <- factor(sample(letters[1:3], n, replace = TRUE))
  m <- xhdfe(y ~ x1 * cat + log(abs(x2) + 1) | f1, dd2)
  l <- lm(y ~ x1 * cat + log(abs(x2) + 1) + factor(f1), dd2)
  common <- setdiff(names(coef(m)), "(Intercept)")
  expect_equal(unname(coef(m)[common]), unname(coef(l)[common]),
               tolerance = 1e-9)
})

test_that("combined fixed effects f1^f2 equal explicit interaction ids", {
  m1 <- xhdfe(y ~ x1 + x2 | f1^f2, dd)
  ids <- as.integer(interaction(dd$f1, dd$f2, drop = TRUE))
  m2 <- xhdfe_fit(dd$y, as.matrix(dd[, c("x1", "x2")]), fes = list(ids))
  expect_equal(unname(coef(m1)), unname(coef(m2)), tolerance = 1e-12)
  expect_equal(m1$num_singletons, m2$num_singletons)
})

test_that("heterogeneous slope formula syntax maps to explicit slope terms", {
  m1 <- xhdfe(y ~ x1 + x2 | f1 + f2[z], dd)
  m2 <- xhdfe_fit(dd$y, as.matrix(dd[, c("x1", "x2")]),
                  fes = list(dd$f1, dd$f2),
                  slopes = list(list(fe = 2, values = dd$z,
                                     include_intercept = TRUE)))
  expect_equal(unname(coef(m1)), unname(coef(m2)), tolerance = 1e-12)
  m3 <- xhdfe(y ~ x1 + x2 | f1 + f2[[z]], dd)
  m4 <- xhdfe_fit(dd$y, as.matrix(dd[, c("x1", "x2")]),
                  fes = list(dd$f1, dd$f2),
                  slopes = list(list(fe = 2, values = dd$z,
                                     include_intercept = FALSE)))
  expect_equal(unname(coef(m3)), unname(coef(m4)), tolerance = 1e-12)
})

test_that("cluster specifications are equivalent across syntaxes", {
  m1 <- xhdfe(y ~ x1 + x2 | f1 + f2, dd, cluster = ~f1)
  m2 <- xhdfe(y ~ x1 + x2 | f1 + f2, dd, cluster = "f1")
  m3 <- xhdfe(y ~ x1 + x2 | f1 + f2, dd, vcov = ~f1)
  m4 <- xhdfe(y ~ x1 + x2 | f1 + f2, dd, cluster = dd$f1)
  expect_equal(m1$se, m2$se)
  expect_equal(m1$se, m3$se)
  expect_equal(m1$se, m4$se)
})

test_that("fixest agreement on coefficients and cluster SEs", {
  skip_if_not_installed("fixest")
  m <- xhdfe(y ~ x1 + x2 | f1 + f2, dd, cluster = ~f1, tol = 1e-10)
  f <- fixest::feols(y ~ x1 + x2 | f1 + f2, dd, cluster = ~f1,
                     fixef.tol = 1e-10)
  expect_equal(unname(coef(m)[1:2]), unname(coef(f)), tolerance = 1e-7)
  # reghdfe-style SSC corresponds to fixest's ssc(fixef.K = "nested",
  # cluster.df = "min", t.df = "min") (see the fixest ssc documentation)
  fs <- fixest::se(f, ssc = fixest::ssc(fixef.K = "nested",
                                        cluster.df = "min", t.df = "min"))
  expect_equal(unname(m$se[1:2]), as.numeric(fs), tolerance = 1e-6)
})

test_that("S3 surface behaves", {
  m <- xhdfe(y ~ x1 + x2 | f1 + f2, dd, cluster = ~f1, save_fe = TRUE)
  expect_s3_class(m, "xhdfe")
  expect_named(coef(m), c("x1", "x2", "(Intercept)"))
  expect_equal(dim(vcov(m)), c(3L, 3L))
  expect_equal(nobs(m), m$nobs)
  expect_equal(nrow(confint(m)), 3L)
  expect_output(print(m), "xhdfe")
  expect_output(print(summary(m)), "tolerance mode")
  expect_s3_class(fixef(m), "xhdfe_fixef")
  # y = xb + d + e reconstruction
  recon <- predict(m) + predict(m, type = "d") + residuals(m)
  expect_equal(unname(recon), dd$y, tolerance = 1e-10)
  fe <- fixef(m)
  expect_equal(unname(fe[[1]] + fe[[2]]),
               unname(predict(m, type = "d")), tolerance = 1e-8)
  # newdata prediction equals in-sample xb
  expect_equal(unname(predict(m, newdata = dd[1:7, ])),
               unname(predict(m)[1:7]), tolerance = 1e-10)
})

test_that("backend argument fails closed when CUDA is unavailable", {
  info <- xhdfe_info()
  if (!info$cuda_enabled) {
    # Either the core itself fails closed ("Requested GPU backend was
    # unavailable") or the R-level guard does; both are acceptable.
    expect_error(xhdfe(y ~ x1 | f1, dd, backend = "cuda"),
                 "GPU backend|CPU-fallback")
  } else {
    succeed()
  }
})

test_that("CUDA backend reports real GPU use and matches CPU", {
  skip_if_not(isTRUE(xhdfe_info()$cuda_enabled), "CPU-only build")
  m_cpu <- xhdfe(y ~ x1 + x2 | f1 + f2, dd, backend = "cpu")
  m_gpu <- xhdfe(y ~ x1 + x2 | f1 + f2, dd, backend = "cuda")
  expect_true(m_gpu$gpu_used)
  expect_identical(m_gpu$gpu_status, "used")
  expect_false(isTRUE(m_cpu$gpu_used))
  expect_equal(coef(m_gpu), coef(m_cpu), tolerance = 1e-10)
  expect_equal(m_gpu$se, m_cpu$se, tolerance = 1e-8)
})

test_that("demean matches manual within-transform for one FE", {
  dm <- xhdfe_demean(y ~ x1 | f1, dd)
  manual <- dd$y - ave(dd$y, dd$f1)
  expect_equal(unname(dm$demeaned[, "y"]), manual, tolerance = 1e-9)
})

test_that("errors are informative", {
  expect_error(xhdfe(y ~ x1 | f1, dd, vcov = "banana"), "Unknown vcov")
  expect_error(xhdfe(y ~ x1 | f1, dd, vcov = "cluster"), "requires")
  expect_error(xhdfe(y ~ x1 | f1, dd, tolerance_mode = "nope"),
               "Unknown tolerance mode")
  expect_error(xhdfe(y ~ x1 | f1, dd, absorption_method = "nope"),
               "Unknown absorption method")
  expect_error(xhdfe(y ~ x1 | f1, dd, level = 400), "level must be")
  expect_error(xhdfe(y ~ x1 | f1, dd, individual = "f2"), "requires")
  expect_error(xhdfe_fit(c(dd$y, NA), cbind(c(dd$x1, 1))), "missing values")
})

test_that("na.action and subset behave like the sample marker", {
  dd2 <- dd
  dd2$x1[1:50] <- NA
  m <- xhdfe(y ~ x1 + x2 | f1, dd2, subset = f2 > 3)
  expect_true(all(is.na(residuals(m)[1:50])))
  expect_true(all(is.na(residuals(m)[dd2$f2 <= 3])))
  expect_error(xhdfe(y ~ x1 + x2 | f1, dd2, na.action = "fail"),
               "missing values")
})
