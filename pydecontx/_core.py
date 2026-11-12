"""Core variational-EM routines for DecontX.

Pure-Python / numpy / scipy port of the C++ inner loops of the
Bioconductor ``decontX`` package (``src/DecontX.cpp``):

* :func:`decontx_initialize` -- initial native (``phi``) and contamination
  (``eta``) gene distributions from a random ``theta``.
* :func:`decontx_em` -- one variational-EM step updating ``phi``, ``eta``,
  ``theta`` and (optionally) the Dirichlet hyper-parameter ``delta``.
* :func:`decontx_loglik` -- the two-component multinomial log-likelihood.
* :func:`calculate_native_matrix` -- the decontaminated count matrix.

All matrices follow the R convention: ``counts`` is genes-by-cells
(rows = genes, columns = cells); ``phi`` / ``eta`` are genes-by-clusters.
``z`` holds 1-based integer cluster labels (one per cell).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._dirichlet import fit_dirichlet

__all__ = [
    "decontx_initialize",
    "decontx_em",
    "decontx_loglik",
    "calculate_native_matrix",
]


def _as_csc(counts) -> sp.csc_matrix:
    """Return ``counts`` as a float CSC sparse matrix (genes x cells)."""
    if sp.issparse(counts):
        return counts.tocsc().astype(float, copy=False)
    return sp.csc_matrix(np.asarray(counts, dtype=float))


def _entries(counts, z, entries=None):
    """Nonzero entries of a genes x cells CSC matrix in column order: gene, cell, cluster, value
    and the flat index ``gene * K + cluster`` into a C-ordered genes x clusters array.
    ``entries`` returns a precomputed tuple (the EM loop computes it once per batch)."""
    if entries is not None:
        return entries
    z = np.asarray(z, dtype=int)
    cols = np.repeat(np.arange(counts.shape[1]), np.diff(counts.indptr))
    k = z[cols] - 1
    rows64 = counts.indices.astype(np.int64)
    return (counts.indices, cols, k, counts.data, rows64 * int(z.max()) + k, np.diff(counts.indptr),
            k.astype(np.int64) * counts.shape[0] + rows64)       # F-order index: one cluster column per cell


_BLOCK = 1 << 18                                                  # entries per block: temporaries stay in cache


def _for_blocks(n, fn, n_jobs=1):
    """Run ``fn(start, stop)`` on consecutive blocks of ``n`` entries, on ``n_jobs`` threads
    (NumPy releases the GIL inside the per-block array operations)."""
    spans = [(b0, min(b0 + _BLOCK, n)) for b0 in range(0, n, _BLOCK)]
    if n_jobs is None or n_jobs <= 1 or len(spans) < 2:
        return [fn(a, b) for a, b in spans]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(n_jobs) as ex:
        return list(ex.map(lambda ab: fn(*ab), spans))


def decontx_initialize(counts, theta, z, pseudocount: float = 1e-20):
    """Initialise the native (``phi``) and contamination (``eta``) matrices.

    Port of the C++ ``decontXInitialize``. ``phi[:, k]`` accumulates
    ``theta_j * counts`` over all cells ``j`` assigned to cluster ``k``;
    ``eta`` is the row-sum complement (every *other* cluster's signal),
    and both are column-normalised to proportions.

    Parameters
    ----------
    counts : (G, C) array or sparse matrix
        Gene-by-cell UMI counts.
    theta : (C,) array
        Initial native proportion for each cell.
    z : (C,) int array
        1-based cluster label for each cell.
    pseudocount : float
        Added to every cell of ``phi``/``eta`` before normalising.

    Returns
    -------
    dict with keys ``phi`` and ``eta`` -- (G, K) numpy arrays.
    """
    counts = _as_csc(counts)
    theta = np.asarray(theta, dtype=float)
    z = np.asarray(z, dtype=int)
    G, C = counts.shape
    K = int(z.max())

    rows, cols, k, x, f, per_cell, f_col = _entries(counts, z)
    phi = pseudocount + np.bincount(f, x * theta[cols], minlength=G * K).reshape(G, K)
    phi_rowsum = phi.sum(axis=1)
    eta = phi_rowsum[:, None] - phi

    phi = phi / phi.sum(axis=0, keepdims=True)
    eta = eta / eta.sum(axis=0, keepdims=True)
    return {"phi": phi, "eta": eta}


def decontx_em(counts, counts_colsums, theta, eta, phi, z,
               estimate_eta: bool = True, estimate_delta: bool = True,
               delta=(10.0, 10.0), pseudocount: float = 1e-20, entries=None, n_jobs: int = 1):
    """One variational-EM update of the DecontX model.

    Port of the C++ ``decontXEM``. For every observed transcript the
    variational native/contaminant responsibility is

    ``p_native  = (phi[i,k] + pc) * (theta[j] + pc)``
    ``p_contam  = (eta[i,k] + pc) * (1 - theta[j] + pc)``
    ``normp     = p_native / (p_native + p_contam)``

    (the non-log form -- exact for a two-component mixture and what the
    C++ code uses). The native mass ``normp * x`` is accumulated into the
    new ``phi`` by cluster; ``eta`` is the row-sum complement. ``theta``
    is then the posterior mean of a Beta/Dirichlet with concentration
    ``delta``, which is itself re-estimated by :func:`fit_dirichlet`.

    Returns a dict with the updated ``phi``, ``eta``, ``theta``,
    ``delta`` and the per-cell ``contamination`` fraction.
    """
    counts = _as_csc(counts)
    theta = np.asarray(theta, dtype=float)
    counts_colsums = np.asarray(counts_colsums, dtype=float)
    phi = np.asarray(phi, dtype=float)
    eta = np.asarray(eta, dtype=float)
    z = np.asarray(z, dtype=int)
    delta = np.asarray(delta, dtype=float)

    G, C = counts.shape
    K = phi.shape[1]

    # every nonzero entry at once, summed in the order of the C++ loop (column by column)
    rows, cols, k, x, f, per_cell, f_col = _entries(counts, z, entries)
    # normp = pnative / (pcontamin + pnative) = 1 / (1 + q[i, k] * s[j]) with the per gene-and-cluster
    # ratio q = (eta + pc) / (phi + pc) and the per-cell ratio s = (1 - theta + pc) / (theta + pc):
    # five passes over the entries instead of nine, in cache-sized blocks; q is column-major, so the
    # entries of a cell read one cluster column
    q = ((np.asfortranarray(eta) + pseudocount) / (np.asfortranarray(phi) + pseudocount)).ravel(order="F")
    s_cell = (1.0 - theta + pseudocount) / (theta + pseudocount)
    px = np.empty(x.size)

    def block(b0, b1):
        d = q.take(f_col[b0:b1])
        d *= s_cell.take(cols[b0:b1])
        d += 1.0
        np.divide(x[b0:b1], d, out=d)
        px[b0:b1] = d

    _for_blocks(x.size, block, n_jobs)
    new_phi = np.bincount(f, px, minlength=G * K).reshape(G, K)
    native_total = np.bincount(cols, px, minlength=C)
    if estimate_eta:
        phi_rowsum = new_phi.sum(axis=1)
        new_eta = phi_rowsum[:, None] - new_phi
    else:
        new_eta = eta

    new_phi = new_phi / new_phi.sum(axis=0, keepdims=True)
    if estimate_eta:
        new_eta = new_eta / new_eta.sum(axis=0, keepdims=True)

    # Update theta (and optionally its Dirichlet hyper-parameter delta).
    contamination_prop = (counts_colsums - native_total) / counts_colsums
    native_prop = 1.0 - contamination_prop
    new_delta = delta
    if estimate_delta:
        theta_raw = np.column_stack([native_prop, contamination_prop])
        new_delta = fit_dirichlet(theta_raw)["alpha"]

    new_theta = (native_total + new_delta[0]) / (counts_colsums
                                                 + np.sum(new_delta))

    return {
        "phi": new_phi,
        "eta": new_eta,
        "theta": new_theta,
        "delta": new_delta,
        "contamination": contamination_prop,
    }


def decontx_loglik(counts, theta, eta, phi, z, pseudocount: float = 1e-20, entries=None, n_jobs: int = 1):
    """Two-component multinomial log-likelihood of the DecontX model.

    Port of the C++ ``decontXLogLik``:
    ``ll = sum_{i,j} x_{ij} * log(phi*theta + eta*(1-theta) + pc)``.
    """
    counts = _as_csc(counts)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    eta = np.asarray(eta, dtype=float)
    z = np.asarray(z, dtype=int)

    rows, cols, k, x, f, per_cell, f_col = _entries(counts, z, entries)
    phi_c = np.asfortranarray(phi).ravel(order="F")
    eta_c = np.asfortranarray(eta).ravel(order="F")

    def block(b0, b1):
        th = theta.take(cols[b0:b1])
        mix = phi_c.take(f_col[b0:b1]) * th
        mix += eta_c.take(f_col[b0:b1]) * (1.0 - th)
        mix += pseudocount
        return float(np.dot(x[b0:b1], np.log(mix)))

    loglik = float(sum(_for_blocks(x.size, block, n_jobs)))
    return loglik


def calculate_native_matrix(counts, theta, eta, phi, z,
                            pseudocount: float = 1e-20) -> sp.csc_matrix:
    """Return the decontaminated (native) count matrix.

    Port of the C++ ``calculateNativeMatrix``: each observed entry is
    scaled by its variational native responsibility ``normp``. Values
    may be non-integer; round for integer counts.
    """
    counts = _as_csc(counts).copy()
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    eta = np.asarray(eta, dtype=float)
    z = np.asarray(z, dtype=int)

    rows, cols, k, x, f, per_cell, f_col = _entries(counts, z)
    p_native = np.log(phi[rows, k] + pseudocount) + np.log(theta[cols] + pseudocount)
    p_contam = np.log(eta[rows, k] + pseudocount) + np.log(1.0 - theta[cols] + pseudocount)
    out = x * (np.exp(p_native) / (np.exp(p_contam) + np.exp(p_native)))
    indices, indptr = counts.indices, counts.indptr
    native = sp.csc_matrix((out, indices.copy(), indptr.copy()),
                           shape=counts.shape)
    return native
