"""pydecontx run on one task, mirroring 02_run_decontx.R.

Usage: python 03_run_pydecontx.py <task_dir> <out_dir> [repeat] [n_jobs]
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import pydecontx


def main():
    td, od = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    n_jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    od.mkdir(parents=True, exist_ok=True)
    tod = sp.csc_matrix(scipy.io.mmread(gzip.open(td / "tod.mtx.gz")), dtype=np.float64)
    droplets = (td / "droplets.tsv").read_text().split()
    cl = pd.read_csv(td / "clusters.tsv", sep="\t", dtype=str)
    counts = tod[:, pd.Index(droplets).get_indexer(cl.cell)].tocsc()
    t0 = time.perf_counter()
    res = pydecontx.decontx(counts, z=cl.cluster.to_numpy(), n_jobs=n_jobs)
    elapsed = time.perf_counter() - t0
    out = sp.csc_matrix(res.decontx_counts)
    out.sort_indices()
    out.indptr.astype(np.int32).tofile(od / "adj_p.bin")
    out.indices.astype(np.int32).tofile(od / "adj_i.bin")
    out.data.astype(np.float64).tofile(od / "adj_x.bin")
    np.asarray(res.contamination, dtype=np.float64).tofile(od / "contamination.bin")
    e = next(iter(res.estimates.values()))
    (od / "result.json").write_text(json.dumps({
        "engine": "pydecontx", "version": pydecontx.__version__, "python": sys.version.split()[0],
        "n_jobs": n_jobs, "iterations": int(e["iteration"]), "delta": [float(v) for v in e["delta"]], "seconds": elapsed,
        "dim": list(out.shape)}))
    print(td.name, f"iter={int(e['iteration'])} {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
