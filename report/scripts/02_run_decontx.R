# Reference run: decontX 1.8.0 on one task with the shared clusters (default arguments, seed 12345).
# Usage: Rscript 02_run_decontx.R <task_dir> <out_dir>
suppressMessages({library(Matrix); library(decontX); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
td <- args[1]; od <- args[2]
dir.create(od, recursive = TRUE, showWarnings = FALSE)
tod <- as(readMM(gzfile(file.path(td, "tod.mtx.gz"))), "CsparseMatrix")
genes <- read.delim(file.path(td, "genes.tsv"))
rownames(tod) <- genes$id; colnames(tod) <- readLines(file.path(td, "droplets.tsv"))
cl <- read.delim(file.path(td, "clusters.tsv"), colClasses = "character")
counts <- tod[, cl$cell]
t0 <- proc.time()[["elapsed"]]
res <- decontX(counts, z = cl$cluster, verbose = FALSE)
elapsed <- proc.time()[["elapsed"]] - t0
m <- as(res$decontXcounts, "CsparseMatrix")
writeBin(m@p, file.path(od, "adj_p.bin"), size = 4); writeBin(m@i, file.path(od, "adj_i.bin"), size = 4)
writeBin(m@x, file.path(od, "adj_x.bin"), size = 8)
writeBin(as.numeric(res$contamination), file.path(od, "contamination.bin"), size = 8)
e <- res$estimates$all_cells
write_json(list(engine = "decontX", version = as.character(packageVersion("decontX")), R = R.version.string,
                iterations = e$iteration, delta = e$delta, seconds = elapsed, dim = dim(m)),
           file.path(od, "result.json"), auto_unbox = TRUE, digits = NA)
cat(basename(td), sprintf("iter=%d %.1fs", e$iteration, elapsed), "\n")
