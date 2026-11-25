#' Examples with public data
#'
#' Run each block independently. The Grunfeld blocks use the public
#' statsmodels edition. The connected-set and group examples use the
#' same public worker--firm teaching simulation and require \pkg{foreign}.
#'
#' @name xhdfe_examples
#' @seealso \code{\link{xhdfe}}, \code{\link{xhdfe_fit}}, \code{\link{xhdfe_demean}}
#' @examples
#' \dontrun{
#' # Absorption, clustered inference and saved effects
#' library(xhdfe)
#' data <- utils::read.csv(paste0(
#'   "https://raw.githubusercontent.com/statsmodels/statsmodels/main/",
#'   "statsmodels/datasets/grunfeld/grunfeld.csv"
#' ))
#' fit <- xhdfe(
#'   invest ~ value + capital | firm + year, data,
#'   vcov = ~ firm + year, save_fe = TRUE, backend = "cpu"
#' )
#' print(coeftable(fit))
#' print(length(predict(fit, type = "xbd")))
#'
#' # Analytic, frequency, importance and probability weights
#' library(xhdfe)
#' data <- utils::read.csv(paste0(
#'   "https://raw.githubusercontent.com/statsmodels/statsmodels/main/",
#'   "statsmodels/datasets/grunfeld/grunfeld.csv"
#' ))
#' # Illustrative weights for the four conventions.
#' data$w <- 1 + data$capital / 100
#' data$copies <- 1L + (data$year >= 1945)
#' model <- invest ~ value + capital | firm + year
#' analytic <- xhdfe(model, data, weights = ~ w)
#' frequency <- xhdfe(model, data, weights = ~ copies, weights_type = "frequency")
#' importance <- xhdfe(model, data, weights = ~ w, weights_type = "importance")
#' probability <- xhdfe(model, data, weights = ~ w, weights_type = "probability")
#' print(c(nobs(analytic), nobs(frequency), nobs(importance), nobs(probability)))
#'
#' # Heterogeneous slopes
#' library(xhdfe)
#' data <- utils::read.csv(paste0(
#'   "https://raw.githubusercontent.com/statsmodels/statsmodels/main/",
#'   "statsmodels/datasets/grunfeld/grunfeld.csv"
#' ))
#' # A firm intercept and a firm-specific slope on market value.
#' fit <- xhdfe(invest ~ capital | firm[value] + year, data, vcov = ~ firm)
#' print(coeftable(fit))
#'
#' # IV/2SLS
#' library(xhdfe)
#' path <- tempfile(fileext = ".csv.bz2")
#' utils::download.file(paste0(
#'   "https://raw.githubusercontent.com/bashtage/linearmodels/main/",
#'   "linearmodels/datasets/card/card.csv.bz2"
#' ), path, mode = "wb", quiet = TRUE)
#' data <- utils::read.csv(path)
#' fit <- xhdfe(
#'   lwage ~ exper + expersq | south66 | educ ~ nearc4,
#'   data, vcov = "robust"
#' )
#' print(coeftable(fit))
#'
#' # Group outcomes with individual effects
#' library(xhdfe)
#' path <- tempfile(fileext = ".dta")
#' utils::download.file(paste0(
#'   "https://raw.githubusercontent.com/reisportela/xhdfe-xfe/main/",
#'   "data/felsdvsimul.dta"
#' ), path, mode = "wb", quiet = TRUE)
#' data <- foreign::read.dta(path, convert.factors = FALSE)
#' data$firm_year <- interaction(data$j, data$t, drop = TRUE)
#' data$group_y <- ave(data$y, data$firm_year)
#' data$group_x <- ave(data$x1, data$firm_year)
#' fit <- xhdfe(
#'   group_y ~ group_x | i, data,
#'   group = ~ firm_year, individual = ~ i, aggregation = "mean", dof = "exact"
#' )
#' print(coeftable(fit))
#'
#' # Partial out fixed effects
#' library(xhdfe)
#' data <- utils::read.csv(paste0(
#'   "https://raw.githubusercontent.com/statsmodels/statsmodels/main/",
#'   "statsmodels/datasets/grunfeld/grunfeld.csv"
#' ))
#' within <- xhdfe_demean(invest ~ value + capital | firm + year, data)
#' fit <- stats::lm.fit(within$demeaned[, c("value", "capital")],
#'                      within$demeaned[, "invest"])
#' print(fit$coefficients)
#'
#' # Connected sample and AKM/KSS
#' library(xhdfe)
#' path <- tempfile(fileext = ".dta")
#' utils::download.file(paste0(
#'   "https://raw.githubusercontent.com/reisportela/xhdfe-xfe/main/",
#'   "data/felsdvsimul.dta"
#' ), path, mode = "wb", quiet = TRUE)
#' data <- foreign::read.dta(path, convert.factors = FALSE)
#' sample <- xhdfe_akm_leave_out_set(data$i, data$j)
#' kept <- data[sample$keep, ]
#' fit <- xhdfe_akm_kss(
#'   kept$y, kept$i, kept$j, prune = FALSE, leverages = "exact"
#' )
#' print(c(sample$n_obs, fit$kss$var_psi))
#'
#' # Gelbach decomposition, table and waterfall data
#' library(xhdfe)
#' data <- utils::read.csv(paste0(
#'   "https://raw.githubusercontent.com/statsmodels/statsmodels/main/",
#'   "statsmodels/datasets/grunfeld/grunfeld.csv"
#' ))
#' fit <- xhdfe_gelbach(
#'   data$invest, data["value"],
#'   x2_groups = list(capital = data["capital"]),
#'   common_fes = list(year = data$year), fes = list(firm = data$firm),
#'   vce = "cluster", cluster = data$firm, focal = "value"
#' )
#' rows <- xhdfe_gelbach_tidy(fit, share = "movement")
#' capital_movement <- xhdfe_gelbach_contrast(fit, focal = "value", groups = "capital")
#' cat(xhdfe_gelbach_etable(fit, focal = "value", format = "markdown"))
#' waterfall <- xhdfe_gelbach_waterfall_data(fit, focal = "value")
#'
#' # Gelbach with an absorbed target
#' library(xhdfe)
#' data <- utils::read.csv(paste0(
#'   "https://raw.githubusercontent.com/statsmodels/statsmodels/main/",
#'   "statsmodels/datasets/grunfeld/grunfeld.csv"
#' ))
#' # Firm effects absorb this indicator. Read the point decomposition.
#' data$steel <- as.integer(data$firm == "US Steel")
#' fit <- xhdfe_gelbach(
#'   data$invest, data[c("steel", "value")],
#'   x2_groups = list(capital = data["capital"]),
#'   fes = list(firm = data$firm, year = data$year),
#'   absorbed_targets = "steel", focal = "steel"
#' )
#' print(fit$b_full_status[["steel"]])
#' print(fit$delta["steel", ])
#'
#' # Gelbach cluster-pairs bootstrap
#' library(xhdfe)
#' data <- utils::read.csv(paste0(
#'   "https://raw.githubusercontent.com/statsmodels/statsmodels/main/",
#'   "statsmodels/datasets/grunfeld/grunfeld.csv"
#' ))
#' # The 39 replications illustrate the API.
#' fit <- xhdfe_gelbach_bootstrap(
#'   data$invest, data["value"],
#'   x2_groups = list(capital = data["capital"]),
#'   common_fes = list(year = data$year), fes = list(firm = data$firm),
#'   method = "cluster_pairs", bootstrap_cluster = data$firm, reps = 39, seed = 42
#' )
#' cat(xhdfe_gelbach_etable(fit, panels = "levels", format = "markdown",
#'                          interval = "bootstrap"))
#' }
NULL
