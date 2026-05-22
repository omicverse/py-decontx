"""R-parity tests: pydecontx vs Bioconductor decontX.

Both sides run the variational-EM of DecontX on the *identical* small
synthetic two/three-population contaminated count matrix (written by
``conftest.py``) with the identical cluster labels and seed.

The contamination-fraction estimate and the decontaminated count matrix
are deterministic given the data, but the R initialisation of ``theta``
(``stats::rbeta`` under ``withr::with_seed``) uses R's Mersenne-Twister
RNG, which differs from numpy's PCG64. The EM is therefore checked for
high agreement (Pearson > 0.99 -- the level the DecontX paper reports
against ground truth) rather than bit-exactness.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import pydecontx as dx

TOL_CORR = 0.99


def _read(r_reference, name, **kw):
    return pd.read_csv(os.path.join(r_reference, name), **kw)


@pytest.fixture(scope="module")
def py_result(synthetic):
    """pydecontx run on the shared synthetic data."""
    return dx.decontx(synthetic["observed_counts"], z=synthetic["z"],
                      seed=12345, max_iter=500)


# ----------------------------------------------------------------------
# contamination fraction
# ----------------------------------------------------------------------
def test_contamination_matches_r(r_reference, py_result):
    r = _read(r_reference, "r_contamination.csv")["contamination"].values
    p = py_result.contamination
    assert len(r) == len(p)
    corr = pearsonr(r, p)[0]
    assert corr > TOL_CORR, f"contamination Pearson r={corr:.4f}"
    # absolute agreement: mean contamination within a few percent
    assert abs(r.mean() - p.mean()) < 0.03


def test_contamination_matches_ground_truth(r_reference, py_result,
                                            synthetic):
    """Both R and pydecontx should recover the simulated contamination."""
    r = _read(r_reference, "r_contamination.csv")["contamination"].values
    truth = synthetic["contamination"]
    assert pearsonr(py_result.contamination, truth)[0] > TOL_CORR
    assert pearsonr(r, truth)[0] > TOL_CORR


# ----------------------------------------------------------------------
# decontaminated count matrix
# ----------------------------------------------------------------------
def test_decontx_counts_match_r(r_reference, py_result):
    r = _read(r_reference, "r_decontx_counts.csv", index_col=0)
    p = np.asarray(py_result.decontx_counts.todense())
    assert r.shape == p.shape
    r_flat = r.values.ravel()
    p_flat = p.ravel()
    corr = pearsonr(r_flat, p_flat)[0]
    assert corr > TOL_CORR, f"decontX counts Pearson r={corr:.4f}"
    # per-cell library sizes of the native matrix agree closely
    r_cs = r.values.sum(axis=0)
    p_cs = p.sum(axis=0)
    assert pearsonr(r_cs, p_cs)[0] > TOL_CORR


def test_theta_matches_r(r_reference, py_result):
    r = _read(r_reference, "r_theta.csv")["theta"].values
    p = py_result.estimates["all_cells"]["theta"]
    corr = pearsonr(r, p)[0]
    assert corr > TOL_CORR, f"theta Pearson r={corr:.4f}"


# ----------------------------------------------------------------------
# background (empty-droplet) mode
# ----------------------------------------------------------------------
def test_background_mode_matches_r(r_reference, synthetic):
    bg_file = os.path.join(r_reference, "r_bg_contamination.csv")
    if not os.path.exists(bg_file):
        pytest.skip("background R reference not generated")
    r = pd.read_csv(bg_file)["contamination"].values
    p = dx.decontx(synthetic["observed_counts"], z=synthetic["z"],
                   background=synthetic["background"], seed=12345,
                   max_iter=500).contamination
    corr = pearsonr(r, p)[0]
    assert corr > TOL_CORR, f"background contamination Pearson r={corr:.4f}"
