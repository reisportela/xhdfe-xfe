# Deterministic dataset shared by the R<->Python parity harness.
# tools/gen_parity_fixture.R exports exactly this data for the Python
# reference run; the parity test regenerates it here from the same seed.

jsonlite_fromJSON_lite <- function(path) {
  jsonlite::fromJSON(path, simplifyVector = TRUE, simplifyDataFrame = FALSE,
                     simplifyMatrix = TRUE)
}

make_parity_data <- function() {
  set.seed(20260703)
  n <- 15000
  d <- data.frame(
    id1 = sample.int(300, n, replace = TRUE),
    id2 = sample.int(40, n, replace = TRUE),
    id3 = sample.int(8, n, replace = TRUE),
    id_hi = sample.int(8000, n, replace = TRUE),
    x1 = rnorm(n),
    x2 = rnorm(n),
    z = rnorm(n),
    ze1 = rnorm(n),
    ze2 = rnorm(n),
    w = runif(n, 0.5, 3),
    wf = sample.int(3, n, replace = TRUE)
  )
  d$xend <- 0.6 * d$x1 + 0.5 * d$ze1 - 0.4 * d$ze2 + rnorm(n) * 0.7
  d$y <- 1 + 0.5 * d$x1 - 0.3 * d$x2 + 0.7 * d$xend +
    d$id1 / 150 + d$id2 / 20 + d$id3 / 4 + 0.03 * d$z * d$id2 + rnorm(n)
  # Group-level outcome block (600 groups x 4 rows, individuals nested).
  ng <- 600
  gsize <- 4
  gn <- ng * gsize
  g <- data.frame(
    grp = rep(seq_len(ng), each = gsize),
    ind = ((rep(seq_len(ng), each = gsize) * 7 +
              rep(seq_len(gsize), times = ng) * 13) %% 120) + 1
  )
  g$year <- 2000 + g$grp %% 5
  g$xg <- (g$grp %% 7) / 3
  yg <- 2 + 0.8 * g$xg + (g$grp %% 11) / 10 + rnorm(ng)[g$grp] * 0.1
  g$yg <- yg
  list(main = d, grouped = g)
}
