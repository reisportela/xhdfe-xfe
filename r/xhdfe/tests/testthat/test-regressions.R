# Regression tests for defects found in the adversarial code review
# (2026-07-03). Each test pins the fixed behavior.

set.seed(99)
n <- 3000
rd <- data.frame(
  f1 = sample.int(80, n, replace = TRUE),
  f2 = sample.int(25, n, replace = TRUE),
  x1 = rnorm(n),
  z = rnorm(n),
  w = runif(n, 0.5, 2)
)
rd$endo <- 0.5 * rd$x1 + 0.6 * rd$z + rnorm(n) * 0.5
rd$y <- 1 + 0.4 * rd$x1 + 2 * rd$endo + rd$f1 / 40 + rd$f2 / 10 + rnorm(n)

test_that("predict(newdata) includes endogenous contributions for IV fits", {
  m <- xhdfe(y ~ x1 | f1 | endo ~ z, rd)
  in_sample <- predict(m)
  out_sample <- predict(m, newdata = rd)
  expect_equal(unname(out_sample), unname(in_sample), tolerance = 1e-10)
})

test_that("predict(newdata) fails closed on missing model columns", {
  m <- xhdfe(y ~ x1 | f1 | endo ~ z, rd)
  bad <- rd[, setdiff(names(rd), "endo")]
  expect_error(predict(m, newdata = bad))
})

test_that("predict(newdata) honours training factor levels", {
  rd2 <- rd
  rd2$cat <- factor(sample(letters[1:4], n, replace = TRUE))
  m <- xhdfe(y ~ x1 + cat | f1, rd2)
  sub <- rd2[rd2$cat %in% c("a", "b"), ][1:20, ]
  sub$cat <- droplevels(sub$cat)
  expect_equal(unname(predict(m, newdata = sub)),
               unname(predict(m)[as.integer(rownames(sub))]),
               tolerance = 1e-10)
})

test_that("xhdfe_demean realigns rows when singletons are dropped", {
  ds <- data.frame(f1 = c(rep(1:50, each = 4), 9001:9010))
  ds$x <- rnorm(nrow(ds))
  ds$y <- ds$x + ds$f1 / 25 + rnorm(nrow(ds))
  dm <- xhdfe_demean(y ~ x | f1, ds, drop_singletons = TRUE)
  expect_identical(length(dm$rows), nrow(dm$demeaned))
  expect_identical(dm$num_singletons, 10L)
  manual <- ds$y[dm$rows] - ave(ds$y[dm$rows], ds$f1[dm$rows])
  expect_equal(unname(dm$demeaned[, "y"]), manual, tolerance = 1e-9)
})

test_that("all-singleton designs raise an R error instead of crashing", {
  ds <- data.frame(f1 = 1:50, x = rnorm(50), y = rnorm(50))
  expect_error(xhdfe(y ~ x | f1, ds), "no observations remain")
})

test_that("na.action = 'fail' ignores NAs outside the subset", {
  rd2 <- rd
  rd2$x1[rd2$f2 == 1] <- NA
  m <- xhdfe(y ~ x1 | f1, rd2, subset = f2 > 1, na.action = "fail")
  expect_s3_class(m, "xhdfe")
  expect_error(xhdfe(y ~ x1 | f1, rd2, na.action = "fail"), "missing values")
})

test_that("demean resolves weights given as a column name", {
  dm1 <- xhdfe_demean(y ~ x1 | f1, rd, weights = "w")
  dm2 <- xhdfe_demean(y ~ x1 | f1, rd, weights = ~w)
  expect_equal(dm1$demeaned, dm2$demeaned)
  expect_error(xhdfe_demean(y ~ x1 | f1, rd, weights = rd$w[1:5]),
               "one entry per observation")
})

test_that("weighted fits use the effective N in ll and r2_a", {
  m_fw <- xhdfe(y ~ x1 | f1, rd, weights = ~round(w * 2),
                weights_type = "frequency")
  idx <- rep(seq_len(n), times = round(rd$w * 2))
  m_rep <- xhdfe(y ~ x1 | f1, rd[idx, ])
  expect_equal(m_fw$ll, m_rep$ll, tolerance = 1e-8)
  expect_equal(m_fw$r2_a, m_rep$r2_a, tolerance = 1e-8)
})

test_that("group FE decomposition honours forwarded fit options", {
  gd <- data.frame(pat = rep(1:100, each = 3))
  gd$inv <- ((gd$pat + seq_len(nrow(gd))) %% 25) + 1
  gd$fund <- gd$pat %% 2
  gd$cit <- ave(5 + 0.4 * gd$fund + gd$pat / 50 + rnorm(nrow(gd)) * 0.01,
                gd$pat)
  dec1 <- xhdfe_group_fes(gd$cit, cbind(fund = gd$fund), fes = list(gd$inv),
                          group = gd$pat, individual = gd$inv,
                          fit_intercept = FALSE, drop_singletons = FALSE,
                          tolerance_mode = "xhdfe-fast")
  expect_s3_class(dec1, "xhdfe_group_fes")
  expect_true(dec1$converged)
  # the fit options genuinely reach the core option parser
  expect_error(xhdfe_group_fes(gd$cit, cbind(fund = gd$fund),
                               fes = list(gd$inv), group = gd$pat,
                               individual = gd$inv,
                               tolerance_mode = "nope"),
               "Unknown tolerance mode")
  expect_error(xhdfe_group_fes(gd$cit, cbind(fund = gd$fund),
                               fes = list(gd$inv), group = gd$pat,
                               individual = gd$inv,
                               absorption_method = "nope"),
               "Unknown absorption method")
})

test_that("xhdfe_fit rejects individual without group (audit P0.2)", {
  ds <- data.frame(f1 = rep(1:30, 10), id = rep(1:10, 30))
  ds$x <- rnorm(300)
  ds$y <- ds$x + ds$f1 / 15 + rnorm(300)
  expect_error(xhdfe_fit(ds$y, cbind(x = ds$x), fes = list(ds$f1),
                         individual = ds$id),
               "requires `group`")
})

test_that("matrix interface rejects NAs in ids, weights and instruments (audit P1.3)", {
  ds <- data.frame(f1 = rep(1:30, 10))
  ds$x <- rnorm(300)
  ds$y <- ds$x + ds$f1 / 15 + rnorm(300)
  fe_chr <- c(NA, letters[(seq_len(299) %% 5) + 1])
  expect_error(xhdfe_fit(ds$y, cbind(x = ds$x), fes = list(fe_chr)),
               "missing values")
  cl <- ds$f1; cl[7] <- NA
  expect_error(xhdfe_fit(ds$y, cbind(x = ds$x), fes = list(ds$f1),
                         cluster = list(cl)),
               "missing values")
  w <- rep(1, 300); w[3] <- NA
  expect_error(xhdfe_fit(ds$y, cbind(x = ds$x), fes = list(ds$f1),
                         weights = w),
               "missing values")
  z <- cbind(rnorm(300)); z[9] <- NA
  expect_error(xhdfe_fit(ds$y, cbind(x = ds$x, en = rnorm(300)),
                         fes = list(ds$f1), instruments = z,
                         endogenous = 2),
               "missing values")
})
