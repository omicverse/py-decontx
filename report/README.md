# Benchmark: pydecontx vs R decontX 1.8.0

Five 10x Genomics human–mouse mixtures (hgmm 1k/5k/10k v3, 6k/12k v2). The tasks are those of the
py-soupx benchmark (`py-soupx/report/scripts/01_prepare.py`, copied here): Cell Ranger's cells, one
scanpy/Leiden clustering given to both packages, and species labels. Counts of the other species
give each cell's ground-truth contamination.

```
python  scripts/01_prepare.py <dir with extracted 10x archives> <tasks_dir>
python  scripts/run_all.py <tasks_dir> <runs_dir> <Rscript> <python> 10 1,2    # idle cores, 1 thread each
bash    scripts/04_threads.sh <tasks_dir> <runs_dir> <python>                 # pydecontx, 8 threads
python  scripts/05_figures.py <runs_dir> <tasks_dir> figures
```

Both run `decontX(counts, z = clusters)` with defaults (seed 12345). R: decontX 1.8.0 on R 4.5.3 (bioconda).
`figures/comparison_per_dataset.tsv` holds the per-dataset numbers.
