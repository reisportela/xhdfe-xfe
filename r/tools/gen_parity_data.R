#!/usr/bin/env Rscript
# Export the parity dataset as raw little-endian doubles so the Python
# reference run sees bit-identical inputs. Run from anywhere:
#   Rscript r/tools/gen_parity_data.R <outdir>

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[[1]] else "r/tools/parity_work"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

root <- normalizePath(file.path(dirname(sub("--file=", "",
  grep("--file=", commandArgs(), value = TRUE))), ".."))
source(file.path(root, "xhdfe", "tests", "testthat", "helper-parity.R"))

dat <- make_parity_data()

dump_df <- function(df, prefix) {
  for (nm in names(df)) {
    con <- file(file.path(outdir, paste0(prefix, "_", nm, ".bin")), "wb")
    writeBin(as.numeric(df[[nm]]), con, size = 8, endian = "little")
    close(con)
  }
  writeLines(paste(names(df), collapse = ","),
             file.path(outdir, paste0(prefix, "_columns.txt")))
  writeLines(as.character(nrow(df)),
             file.path(outdir, paste0(prefix, "_nrow.txt")))
}

dump_df(dat$main, "main")
dump_df(dat$grouped, "grouped")
cat("parity data written to", normalizePath(outdir), "\n")
