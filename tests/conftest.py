"""Shared fixtures and the R-reference build hook for pydecontx tests.

The synthetic count matrix is generated once on the Python side and
written to ``tests/R_out`` as CSV so that the R ``decontX`` driver and
``pydecontx`` consume the *identical* integer counts and cluster labels.
This isolates the variational-EM algorithm from any simulator RNG.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
R_OUT = os.path.join(HERE, "R_out")
R_DRIVER = os.path.join(HERE, "r_reference_driver.R")

# CMAP path-based R environment (override with PYDECONTX_RSCRIPT).
_RSCRIPT = os.environ.get(
    "PYDECONTX_RSCRIPT", "/scratch/users/steorra/env/CMAP/bin/Rscript")


def _make_synthetic():
    """Generate the shared synthetic dataset (deterministic)."""
    from pydecontx import simulate_contamination

    sim = simulate_contamination(C=200, G=120, K=3, n_range=(800, 1500),
                                 beta=0.1, delta=(1.0, 10.0),
                                 num_markers=4, seed=2024)
    # an independent empty-droplet "background" matrix
    rng = np.random.default_rng(99)
    # ambient pool ~ overall gene frequency, sampled into 400 empty droplets
    pool = sim["observed_counts"].sum(axis=1).astype(float)
    pool = pool / pool.sum()
    bg = np.zeros((sim["observed_counts"].shape[0], 400), dtype=int)
    for j in range(400):
        bg[:, j] = rng.multinomial(rng.integers(20, 80), pool)
    sim["background"] = bg
    return sim


@pytest.fixture(scope="session")
def synthetic():
    """The shared synthetic dataset, written to disk for the R driver."""
    os.makedirs(R_OUT, exist_ok=True)
    sim = _make_synthetic()
    genes = sim["gene_names"]
    cells = sim["cell_names"]

    counts_path = os.path.join(R_OUT, "sim_counts.csv")
    if not os.path.exists(counts_path):
        pd.DataFrame(sim["observed_counts"], index=genes,
                     columns=cells).to_csv(counts_path)
        pd.DataFrame({"z": sim["z"]}).to_csv(
            os.path.join(R_OUT, "sim_z.csv"), index=False)
        bg_cols = [f"Drop_{i + 1}" for i in range(sim["background"].shape[1])]
        pd.DataFrame(sim["background"], index=genes,
                     columns=bg_cols).to_csv(
            os.path.join(R_OUT, "sim_background.csv"))
    return sim


def _r_outputs_present() -> bool:
    needed = ["r_contamination.csv", "r_decontx_counts.csv", "r_theta.csv"]
    return all(os.path.exists(os.path.join(R_OUT, f)) for f in needed)


def _build_r_reference() -> bool:
    """Run the decontX R driver if R is available; return success."""
    if not shutil.which(_RSCRIPT) and not os.path.exists(_RSCRIPT):
        return False
    env = dict(os.environ)
    gcc = "/share/software/user/open/gcc/14.2.0/bin"
    if os.path.isdir(gcc):
        env["PATH"] = gcc + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = (
            "/share/software/user/open/gcc/14.2.0/lib64" + os.pathsep
            + env.get("LD_LIBRARY_PATH", ""))
    try:
        subprocess.run([_RSCRIPT, R_DRIVER, R_OUT], check=True, env=env,
                       capture_output=True, timeout=1800)
    except Exception:
        return False
    return _r_outputs_present()


@pytest.fixture(scope="session")
def r_reference(synthetic):
    """Directory of decontX reference outputs; skips if R unavailable."""
    if not _r_outputs_present():
        if not _build_r_reference():
            pytest.skip("decontX R reference unavailable")
    return R_OUT
