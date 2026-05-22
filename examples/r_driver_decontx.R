#!/usr/bin/env Rscript
# Run the original Bioconductor DecontX end-to-end on an external 10X-style
# counts matrix (genes x cells TSV) plus a per-cell cluster-label TSV, both
# dumped by the Python side via pandas.to_csv. Outputs: the per-cell
# contamination fraction, the decontaminated (native) count matrix, the
# final per-cell theta, and a small JSON of parameters -- everything the
# comparison notebook needs to overlay on the Python-computed results.
#
# Usage:
#   Rscript r_driver_decontx.R <counts_tsv> <z_tsv> <outdir> [maxIter] [seed]
#
# <counts_tsv> : genes x cells, tab-separated, gene names in column 1.
# <z_tsv>      : one column 'z' of cluster labels, one row per cell.

suppressPackageStartupMessages({
  library(decontX)
  library(Matrix)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
counts_path <- args[[1]]
z_path      <- args[[2]]
outdir      <- args[[3]]
max_iter <- if (length(args) >= 4) as.integer(args[[4]]) else 500L
seed     <- if (length(args) >= 5) as.integer(args[[5]]) else 12345L

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("[R] reading %s\n", counts_path))
counts <- as.matrix(read.table(counts_path, sep = "\t", header = TRUE,
                               row.names = 1, check.names = FALSE))
storage.mode(counts) <- "integer"
z <- read.table(z_path, sep = "\t", header = TRUE,
                check.names = FALSE)[["z"]]

cat(sprintf("[R] decontX: %d genes x %d cells, %d clusters\n",
            nrow(counts), ncol(counts), length(unique(z))))

# Run DecontX (ANY-method: a plain counts matrix + cluster labels z).
res <- decontX(x = counts, z = z, maxIter = max_iter,
               seed = seed, verbose = FALSE)

write.table(
  data.frame(cell = colnames(counts), contamination = res$contamination),
  file = file.path(outdir, "r_contamination.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)

dec <- as.matrix(res$decontXcounts)
write.table(dec, file = file.path(outdir, "r_decontx_counts.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

theta <- res$estimates[["all_cells"]]$theta
write.table(
  data.frame(cell = colnames(counts), theta = theta),
  file = file.path(outdir, "r_theta.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)

write(toJSON(list(maxIter = max_iter, seed = seed,
                  n_cells = ncol(counts), n_genes = nrow(counts),
                  n_clusters = length(unique(z)),
                  mean_contamination = mean(res$contamination)),
             auto_unbox = TRUE),
      file = file.path(outdir, "meta.json"))

cat(sprintf("[R] wrote outputs to %s (mean contamination=%.4f)\n",
            outdir, mean(res$contamination)))
