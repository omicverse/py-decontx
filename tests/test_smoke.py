"""Smoke / unit tests for pydecontx -- no R required.

These exercise every public entry point, the AnnData and background
modes, the result object and the lower-level EM building blocks, and
check that DecontX recovers a known ground-truth contamination level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pydecontx as dx


# ----------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------
def test_simulate_shapes_and_keys():
    sim = dx.simulate_contamination(C=100, G=60, K=3, seed=1)
    assert sim["observed_counts"].shape == (60, 100)
    assert sim["native_counts"].shape == (60, 100)
    # observed = native + contamination, so observed >= native everywhere
    assert (sim["observed_counts"] >= sim["native_counts"]).all()
    assert len(sim["z"]) == 100
    assert sim["phi"].shape == (60, 3)
    assert 0.0 <= sim["contamination"].mean() <= 1.0


def test_simulate_is_deterministic():
    a = dx.simulate_contamination(C=80, G=50, K=3, seed=42)
    b = dx.simulate_contamination(C=80, G=50, K=3, seed=42)
    assert np.array_equal(a["observed_counts"], b["observed_counts"])
    assert np.array_equal(a["z"], b["z"])


# ----------------------------------------------------------------------
# decontx main function
# ----------------------------------------------------------------------
def test_decontx_recovers_ground_truth():
    sim = dx.simulate_contamination(C=200, G=100, K=3, delta=(1, 10),
                                    seed=2024)
    res = dx.decontx(sim["observed_counts"], z=sim["z"], seed=12345,
                     max_iter=200)
    from scipy.stats import pearsonr
    r = pearsonr(res.contamination, sim["contamination"])[0]
    # The DecontX paper reports R~0.99 vs ground truth.
    assert r > 0.97
    assert np.all(res.contamination >= 0) and np.all(res.contamination <= 1)


def test_decontx_is_deterministic():
    sim = dx.simulate_contamination(C=120, G=70, K=3, seed=5)
    r1 = dx.decontx(sim["observed_counts"], z=sim["z"], seed=12345)
    r2 = dx.decontx(sim["observed_counts"], z=sim["z"], seed=12345)
    assert np.allclose(r1.contamination, r2.contamination)


def test_decontx_decontaminated_matrix():
    sim = dx.simulate_contamination(C=120, G=70, K=3, seed=8)
    res = dx.decontx(sim["observed_counts"], z=sim["z"], seed=12345)
    dec = res.decontx_counts
    assert dec.shape == sim["observed_counts"].shape
    # the decontaminated matrix never exceeds the observed counts
    obs = sp.csc_matrix(sim["observed_counts"].astype(float))
    assert (dec.toarray() <= obs.toarray() + 1e-6).all()
    # rounded native counts should correlate with the true native matrix
    from scipy.stats import pearsonr
    r = pearsonr(np.asarray(res.decontaminated_counts().todense()).ravel(),
                 sim["native_counts"].astype(float).ravel())[0]
    assert r > 0.95


def test_decontx_dataframe_input():
    sim = dx.simulate_contamination(C=90, G=50, K=3, seed=3)
    df = pd.DataFrame(sim["observed_counts"], index=sim["gene_names"],
                      columns=sim["cell_names"])
    res = dx.decontx(df, z=sim["z"], seed=12345)
    assert res.gene_names == sim["gene_names"]
    assert res.cell_names == sim["cell_names"]
    tab = res.to_dataframe()
    assert "decontX_contamination" in tab.columns
    assert len(tab) == 90


def test_decontx_sparse_input():
    sim = dx.simulate_contamination(C=90, G=50, K=3, seed=4)
    res = dx.decontx(sp.csr_matrix(sim["observed_counts"].astype(float)),
                     z=sim["z"], seed=12345)
    assert len(res.contamination) == 90


def test_decontx_anndata_mode():
    import anndata as ad

    sim = dx.simulate_contamination(C=150, G=80, K=3, seed=7)
    adata = ad.AnnData(
        X=sim["observed_counts"].T.astype(float),
        obs=pd.DataFrame({"cluster": sim["z"].astype(str)},
                         index=[f"c{i}" for i in range(150)]),
        var=pd.DataFrame(index=sim["gene_names"]),
    )
    out = dx.decontx(adata, z="cluster", seed=12345, copy=True)
    assert "decontX_contamination" in out.obs
    assert "decontX_clusters" in out.obs
    assert out.layers["decontX_counts"].shape == (150, 80)
    # returning a DecontXResult when copy=False
    res = dx.decontx(adata, z="cluster", seed=12345, copy=False)
    assert isinstance(res, dx.DecontXResult)


def test_decontx_background_mode():
    sim = dx.simulate_contamination(C=150, G=80, K=3, seed=11)
    rng = np.random.default_rng(0)
    bg = rng.poisson(0.4, size=(80, 300))
    res = dx.decontx(sim["observed_counts"], z=sim["z"], background=bg,
                     seed=12345)
    assert len(res.contamination) == 150
    # eta is anchored and not re-estimated -> the eta columns are equal
    eta = res.estimates["all_cells"]["eta"]
    assert np.allclose(eta[:, 0], eta[:, 1])


def test_decontx_batch_mode():
    sim = dx.simulate_contamination(C=180, G=80, K=3, seed=13)
    batch = np.array(["A"] * 90 + ["B"] * 90)
    res = dx.decontx(sim["observed_counts"], z=sim["z"], batch=batch,
                     seed=12345)
    assert set(res.estimates) == {"A", "B"}
    assert all("-" in str(zz) for zz in res.z)


def test_decontx_requires_z():
    sim = dx.simulate_contamination(C=60, G=40, K=3, seed=1)
    with pytest.raises(ValueError):
        dx.decontx(sim["observed_counts"])


def test_decontx_rejects_single_cluster():
    sim = dx.simulate_contamination(C=60, G=40, K=3, seed=1)
    with pytest.raises(ValueError):
        dx.decontx(sim["observed_counts"], z=np.ones(60, dtype=int))


# ----------------------------------------------------------------------
# core EM building blocks
# ----------------------------------------------------------------------
def test_decontx_initialize_normalised():
    sim = dx.simulate_contamination(C=80, G=50, K=3, seed=2)
    counts = sp.csc_matrix(sim["observed_counts"].astype(float))
    theta = np.full(80, 0.7)
    init = dx.decontx_initialize(counts, theta, sim["z"])
    assert init["phi"].shape == (50, 3)
    # columns are proper probability distributions
    assert np.allclose(init["phi"].sum(axis=0), 1.0)
    assert np.allclose(init["eta"].sum(axis=0), 1.0)


def test_decontx_em_step():
    sim = dx.simulate_contamination(C=80, G=50, K=3, seed=2)
    counts = sp.csc_matrix(sim["observed_counts"].astype(float))
    theta = np.full(80, 0.7)
    init = dx.decontx_initialize(counts, theta, sim["z"])
    colsums = np.asarray(counts.sum(axis=0)).ravel()
    step = dx.decontx_em(counts, colsums, theta, init["eta"], init["phi"],
                         sim["z"])
    assert np.allclose(step["phi"].sum(axis=0), 1.0)
    assert step["theta"].shape == (80,)
    assert np.all(step["contamination"] >= 0)
    assert np.all(step["contamination"] <= 1)


def test_loglik_increases():
    """The EM log-likelihood is non-decreasing across iterations."""
    sim = dx.simulate_contamination(C=120, G=60, K=3, seed=6)
    res = dx.decontx(sim["observed_counts"], z=sim["z"], seed=12345,
                     iter_log_lik=1, max_iter=30)
    ll = np.array(res.log_likelihood)
    # allow tiny numerical noise
    assert np.all(np.diff(ll) > -1e-3)


def test_fit_dirichlet_recovers_alpha():
    """fit_dirichlet should recover a known Dirichlet concentration."""
    rng = np.random.default_rng(0)
    true_alpha = np.array([8.0, 3.0])
    x = rng.dirichlet(true_alpha, size=5000)
    est = dx.fit_dirichlet(x)["alpha"]
    # method-of-moments + fixed-point -> close but not exact
    assert np.allclose(est, true_alpha, rtol=0.25)
