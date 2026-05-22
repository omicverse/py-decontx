"""Quick-start example for pydecontx.

Simulates a contaminated three-population single-cell dataset, runs
DecontX, and compares the estimated contamination fraction against the
known ground truth.

Run with:  python examples/quickstart.py
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

import pydecontx as dx


def main() -> None:
    # 1. Simulate a contaminated dataset (genes x cells) with ground truth.
    sim = dx.simulate_contamination(C=300, G=100, K=3, delta=(1, 10),
                                    seed=12345)
    print(f"simulated {sim['observed_counts'].shape[0]} genes x "
          f"{sim['observed_counts'].shape[1]} cells, "
          f"{len(np.unique(sim['z']))} populations")
    print(f"mean true contamination: {sim['contamination'].mean():.3f}")

    # 2. Run DecontX with the known cluster labels.
    res = dx.decontx(sim["observed_counts"], z=sim["z"], seed=12345)
    print(res)

    # 3. Compare the estimate against the ground truth.
    r = pearsonr(res.contamination, sim["contamination"])[0]
    print(f"Pearson(estimated, true contamination) = {r:.4f}")

    # 4. The decontaminated count matrix.
    native = res.decontaminated_counts()  # integer-rounded
    r_counts = pearsonr(
        np.asarray(native.todense()).ravel(),
        sim["native_counts"].astype(float).ravel())[0]
    print(f"Pearson(decontaminated, true native counts) = {r_counts:.4f}")


if __name__ == "__main__":
    main()
