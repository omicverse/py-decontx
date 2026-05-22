"""Minimal end-to-end example — drop this into a Jupyter cell or run as a script.

Demonstrates the standalone DecontX pipeline two ways:

1. On a synthetic contaminated dataset with known ground truth.
2. On a real scRNA-seq dataset (scanpy's pbmc3k), where DecontX needs a
   broad clustering of cell types as its ``z`` argument.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

import pydecontx as dx


def synthetic_demo() -> None:
    """Run DecontX on a simulated dataset and score against ground truth."""
    sim = dx.simulate_contamination(C=300, G=100, K=3, delta=(1, 10),
                                    seed=12345)
    print(f"simulated {sim['observed_counts'].shape[0]} genes x "
          f"{sim['observed_counts'].shape[1]} cells, "
          f"{len(np.unique(sim['z']))} populations")

    res = dx.decontx(sim["observed_counts"], z=sim["z"], seed=12345)
    print(res)
    r = pearsonr(res.contamination, sim["contamination"])[0]
    print(f"Pearson(estimated, true contamination) = {r:.4f}")


def real_data_demo() -> None:
    """Run DecontX on scanpy's pbmc3k — a real filtered cell matrix."""
    import scanpy as sc

    adata = sc.datasets.pbmc3k()           # raw counts in .X
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # DecontX needs a broad clustering of cell types -> quick Leiden.
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)
    adata.X = adata.layers["counts"]       # restore raw counts for DecontX

    out = dx.decontx(adata, z="leiden", seed=12345, copy=True)
    cont = out.obs["decontX_contamination"]
    print(f"pbmc3k: {out.n_obs} cells, mean contamination = "
          f"{cont.mean():.4f}")
    print(out.obs.groupby("leiden")["decontX_contamination"].mean())


def main() -> None:
    synthetic_demo()
    print()
    try:
        real_data_demo()
    except ImportError:
        print("(install scanpy to run the real-data demo)")


if __name__ == "__main__":
    main()
