#!/usr/bin/env Rscript
# Drive the Bioconductor `decontX` package on a small synthetic two/three-
# population contaminated count matrix so the Python port (pydecontx) can
# be checked against it.
#
# Usage:
#   Rscript r_reference_driver.R <out_dir>
#
# The synthetic data is generated *in Python* (tests/conftest.py writes it
# to <out_dir>/sim_*.csv) so that BOTH sides see the identical integer
# count matrix and cluster labels -- this isolates the EM algorithm from
# any RNG differences in the simulator.
#
# Outputs (CSV files in out_dir):
#   r_contamination.csv     decontX() per-cell contamination fraction
#   r_decontx_counts.csv    decontX() decontaminated (native) count matrix
#   r_theta.csv             final per-cell theta (native proportion)
#   r_bg_contamination.csv  decontX() contamination with a background matrix

suppressPackageStartupMessages({
  library(decontX)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "R_out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --- load the shared synthetic input -------------------------------------
counts <- as.matrix(read.csv(file.path(out_dir, "sim_counts.csv"),
                             row.names = 1, check.names = FALSE))
storage.mode(counts) <- "integer"
z <- read.csv(file.path(out_dir, "sim_z.csv"))[[1]]

# --- run decontX (no background) -----------------------------------------
res <- decontX(x = counts, z = z, seed = 12345, verbose = FALSE)

write.csv(data.frame(contamination = res$contamination),
          file.path(out_dir, "r_contamination.csv"), row.names = FALSE)

dec <- as.matrix(res$decontXcounts)
write.csv(dec, file.path(out_dir, "r_decontx_counts.csv"))

theta <- res$estimates[["all_cells"]]$theta
write.csv(data.frame(theta = theta),
          file.path(out_dir, "r_theta.csv"), row.names = FALSE)

# --- run decontX with a background (empty-droplet) matrix ----------------
bg_file <- file.path(out_dir, "sim_background.csv")
if (file.exists(bg_file)) {
  bg <- as.matrix(read.csv(bg_file, row.names = 1, check.names = FALSE))
  storage.mode(bg) <- "integer"
  res_bg <- decontX(x = counts, z = z, background = bg,
                    seed = 12345, verbose = FALSE)
  write.csv(data.frame(contamination = res_bg$contamination),
            file.path(out_dir, "r_bg_contamination.csv"), row.names = FALSE)
}

cat("decontX R reference done\n")
