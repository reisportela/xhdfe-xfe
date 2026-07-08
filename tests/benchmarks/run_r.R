#!/usr/bin/env Rscript
# Run the public xhdfe core-23 replication benchmarks (R front-end).
#
# Reads registry.json, loads each dataset (.dta via haven / .parquet via
# arrow), fits the spec with xhdfe(), and appends timings + coefficients to
# output/r_runs.csv and output/r_coefficients.csv.
#
# Environment knobs (same semantics as run_python.py):
#   SPECS=credit,credit2   comma list of spec names (default: all available)
#   GROUPS=sergio          filter by group (sergio|simulated|pyfixest)
#   REPS=3                 timed repetitions per spec (default 1)
#   MODE=comparable|fast   tolerance mode (default comparable)
#   THREADS=8              OpenMP threads (0 = library default)
#   GPU=1                  request the CUDA backend (fail-closed)
#   OTHERS=1               also time fixest::feols on each spec, if installed
#
# Usage:  Rscript run_r.R

suppressMessages({
  library(jsonlite)
  library(xhdfe)
})

here <- normalizePath(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])))
out_dir <- file.path(here, "output")
dir.create(out_dir, showWarnings = FALSE)

registry <- fromJSON(file.path(here, "registry.json"), simplifyVector = FALSE)
want_specs <- strsplit(Sys.getenv("SPECS", ""), ",")[[1]]
want_groups <- strsplit(Sys.getenv("GROUPS", ""), ",")[[1]]
reps <- as.integer(Sys.getenv("REPS", "1"))
mode <- Sys.getenv("MODE", "comparable")
threads <- as.integer(Sys.getenv("THREADS", "0"))
use_gpu <- !Sys.getenv("GPU", "") %in% c("", "0")
others <- !Sys.getenv("OTHERS", "") %in% c("", "0")

tolmode <- if (mode == "fast") "xhdfe-fast" else "reghdfe-comparable"
backend <- if (use_gpu) "cuda" else "cpu"

runs_path <- file.path(out_dir, "r_runs.csv")
coef_path <- file.path(out_dir, "r_coefficients.csv")
if (!file.exists(runs_path)) {
  write.csv(data.frame(spec = character(), engine = character(), mode = character(),
                       gpu = integer(), rep = integer(), elapsed_seconds = numeric(),
                       n_obs = integer()),
            runs_path, row.names = FALSE)
}
if (!file.exists(coef_path)) {
  write.csv(data.frame(spec = character(), engine = character(), mode = character(),
                       variable = character(), coef = numeric(), se = numeric()),
            coef_path, row.names = FALSE)
}
append_csv <- function(path, df) {
  write.table(df, path, sep = ",", append = TRUE,
              col.names = FALSE, row.names = FALSE)
}

load_data <- function(path) {
  if (grepl("\\.parquet$", path)) {
    arrow::read_parquet(path)
  } else {
    haven::read_dta(path)
  }
}

for (spec in registry$specs) {
  name <- spec$name
  if (length(want_specs) && !(name %in% want_specs)) next
  if (length(want_groups) && !(spec$group %in% want_groups)) next
  path <- file.path(here, spec$file)
  if (!file.exists(path)) {
    cat(sprintf("skip %s: %s not found (generate/download it first)\n", name, spec$file))
    next
  }

  cat(sprintf("=== %s ===\n", name))
  d <- as.data.frame(load_data(path))
  fes <- paste(unlist(spec$absorb), collapse = " + ")
  xs <- paste(unlist(spec$regressors), collapse = " + ")
  fml <- as.formula(sprintf("%s ~ %s | %s", spec$depvar, xs, fes))
  clf <- as.formula(paste("~", spec$cluster))

  for (rep in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]
    m <- xhdfe(fml, data = d, cluster = clf, backend = backend,
               tolerance_mode = tolmode, threads = threads)
    dt <- proc.time()[["elapsed"]] - t0
    append_csv(runs_path, data.frame(name, "xhdfe_r", mode, as.integer(use_gpu),
                                     rep, round(dt, 4), m$nobs))
    cat(sprintf("  xhdfe rep%d: %.2fs\n", rep, dt))
    if (rep == 1) {
      cf <- coef(m)
      se <- m$se
      append_csv(coef_path, data.frame(name, "xhdfe_r", mode,
                                       names(cf), unname(cf), unname(se)))
    }
  }

  if (others && requireNamespace("fixest", quietly = TRUE)) {
    for (rep in seq_len(reps)) {
      t0 <- proc.time()[["elapsed"]]
      mf <- fixest::feols(fml, data = d, cluster = clf)
      dt <- proc.time()[["elapsed"]] - t0
      append_csv(runs_path, data.frame(name, "fixest", mode, 0L, rep,
                                       round(dt, 4), mf$nobs))
      cat(sprintf("  fixest rep%d: %.2fs\n", rep, dt))
      if (rep == 1) {
        cf <- coef(mf)
        append_csv(coef_path, data.frame(name, "fixest", mode,
                                         names(cf), unname(cf), NA_real_))
      }
    }
  }
}

cat(sprintf("done -> %s / %s\n", runs_path, coef_path))
