"""Simulation of contaminated single-cell count matrices.

Pure-Python port of ``decontX::simulateContamination`` -- generates a
native (true) expression matrix plus an ambient-RNA contamination
matrix, mixing populations exactly as the R package does so the result
can be used to benchmark :func:`pydecontx.decontx`.
"""
from __future__ import annotations

import numpy as np

__all__ = ["simulate_contamination"]


def _rdirichlet(rng: np.random.Generator, n: int, alpha) -> np.ndarray:
    """Draw ``n`` Dirichlet samples (one per row) with shape ``alpha``."""
    alpha = np.asarray(alpha, dtype=float)
    x = rng.gamma(shape=np.broadcast_to(alpha, (n, len(alpha))))
    is_zero = x.sum(axis=1) == 0
    if np.any(is_zero):
        cols = rng.integers(0, len(alpha), size=int(is_zero.sum()))
        x[np.where(is_zero)[0], cols] = 1.0
    return x / x.sum(axis=1, keepdims=True)


def simulate_contamination(C: int = 300, G: int = 100, K: int = 3,
                           n_range=(500, 1000), beta: float = 0.1,
                           delta=(1.0, 10.0), num_markers: int = 3,
                           seed: int | None = 12345) -> dict:
    """Simulate a contaminated gene-by-cell count matrix.

    Parameters
    ----------
    C, G, K : int
        Number of cells, genes and cell populations.
    n_range : (int, int)
        Lower/upper bounds for the total counts per cell.
    beta : float
        Concentration parameter for the native distributions ``phi``.
    delta : float or (float, float)
        Beta parameters for the per-cell contamination proportion. A
        scalar gives a symmetric Beta; a pair gives ``(shape1, shape2)``.
    num_markers : int
        Number of exclusive marker genes per population.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with ``native_counts`` (G x C), ``observed_counts`` (G x C,
    native + contamination), ``z`` (1-based cluster labels), ``phi``
    (G x K native distributions), ``eta`` (G x K contamination
    distributions), ``contamination`` (per-cell ground-truth fraction),
    ``markers`` and ``num_markers``.
    """
    rng = np.random.default_rng(seed)
    delta = np.atleast_1d(np.asarray(delta, dtype=float))
    if delta.size == 1:
        cp_by_c = rng.beta(delta[0], delta[0], size=C)
    else:
        cp_by_c = rng.beta(delta[0], delta[1], size=C)

    z = rng.integers(1, K + 1, size=C)
    uniq = np.unique(z)
    if uniq.size < K:
        K = uniq.size
        remap = {old: new for new, old in enumerate(uniq, start=1)}
        z = np.array([remap[v] for v in z])

    n_by_c = rng.integers(min(n_range), max(n_range) + 1, size=C)
    c_n_by_c = np.array([rng.binomial(n_by_c[i], cp_by_c[i])
                         for i in range(C)])
    r_n_by_c = n_by_c - c_n_by_c

    # Native gene distributions, one Dirichlet draw per population.
    phi = _rdirichlet(rng, K, np.full(G, beta))  # K x G

    if K * num_markers > G:
        raise ValueError("numMarkers * K cannot exceed the number of genes.")
    marker_k_index = np.repeat(np.arange(K), num_markers)
    marker_row_index = rng.choice(G, size=num_markers * K, replace=False)
    for i in range(K):
        ix = marker_row_index[marker_k_index == i]
        phi[i, ix] = phi[i].max()
        for j in range(K):
            if j != i:
                phi[j, ix] = 0.0
    phi = phi / phi.sum(axis=1, keepdims=True)

    # Sample the native expression matrix.
    native = np.zeros((G, C), dtype=int)
    for i in range(C):
        native[:, i] = rng.multinomial(r_n_by_c[i], phi[z[i] - 1])

    gene_names = [f"Gene_{i + 1}" for i in range(G)]
    markers = {}
    for i in range(K):
        rows = marker_row_index[marker_k_index == i]
        markers[f"CellType_{i + 1}_Markers"] = [gene_names[r] for r in rows]

    # Contamination distribution: each population's eta = sum of every
    # other population's native counts, normalised.
    n_g_by_k = np.zeros((G, K), dtype=float)
    for k in range(1, K + 1):
        n_g_by_k[:, k - 1] = native[:, z == k].sum(axis=1)
    eta = n_g_by_k.sum(axis=1, keepdims=True) - n_g_by_k
    eta = eta / eta.sum(axis=0, keepdims=True)

    contam = np.zeros((G, C), dtype=int)
    for i in range(C):
        contam[:, i] = rng.multinomial(c_n_by_c[i], eta[:, z[i] - 1])

    observed = native + contam
    with np.errstate(divide="ignore", invalid="ignore"):
        contamination = contam.sum(axis=0) / observed.sum(axis=0)
    contamination = np.nan_to_num(contamination)

    return {
        "native_counts": native,
        "observed_counts": observed,
        "n_by_c": n_by_c,
        "z": z,
        "eta": eta,
        "phi": phi.T,
        "markers": markers,
        "num_markers": num_markers,
        "contamination": contamination,
        "gene_names": gene_names,
        "cell_names": [f"Cell_{i + 1}" for i in range(C)],
    }
