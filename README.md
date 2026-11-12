# pydecontx

A **pure-Python re-implementation of [DecontX](https://github.com/campbio/decontX)** (Yang et al., *Genome Biology* 2020) for decontamination of ambient / cross-contaminating RNA in droplet single-cell RNA-seq data.

- AnnData-native — drop-in for the scanpy ecosystem
- **No `rpy2`**, no R install — the variational-EM, the native/contamination multinomial mixture, and the Minka fixed-point Dirichlet MLE are all implemented directly in NumPy/SciPy
- Same function surface as the R workflow (`decontX` → per-cell contamination fraction + decontaminated count matrix)
- Same results as Bioconductor decontX 1.8.0 for the same seed: the initial `theta` is drawn with R's random number generator, and contamination, `theta` and the decontaminated matrix agree to rounding (see [Relationship to R decontX](#relationship-to-r-decontx))

> omicverse uses this package for `ov.pp.ambient.remove_ambient(adata, method='decontx')`.

## Install

```bash
pip install pydecontx
```

Dependencies: `numpy`, `scipy`, `pandas`, `anndata`.

## Quick-start (function API)

```python
import pydecontx as dx

# 1) a synthetic contaminated dataset (genes x cells) with ground truth
sim = dx.simulate_contamination(C=300, G=100, K=3, delta=(1, 10))

# 2) run DecontX with cluster labels
res = dx.decontx(sim["observed_counts"], z=sim["z"])

res.contamination            # per-cell contamination fraction (0-1)
res.decontx_counts           # decontaminated sparse count matrix
res.decontaminated_counts()  # ... integer-rounded
res.to_dataframe()           # per-cell summary table
```

## Quick-start (AnnData)

```python
import scanpy as sc, pydecontx as dx

adata = sc.read_10x_h5("filtered.h5")          # cells × genes, raw counts in .X
sc.pp.pca(adata); sc.pp.neighbors(adata); sc.tl.leiden(adata)

# write results back into the AnnData
adata = dx.decontx(adata, z="leiden", copy=True)
adata.obs["decontX_contamination"]      # per-cell contamination
adata.layers["decontX_counts"]          # decontaminated counts
```

`x` may be an `AnnData` (cells × genes), a `pandas.DataFrame`, a NumPy array or a SciPy sparse matrix (genes × cells). `z` is required — DecontX needs a broad clustering of cell types; when `x` is an `AnnData` it may name a column of `.obs`. Empty droplets are *not* required, but an optional `background` matrix of raw empty droplets can be supplied to anchor the contamination distribution.

## The model

DecontX models each cell's observed counts as a Bayesian **two-component multinomial mixture**:

* a **native** distribution `phi_k` — the gene probabilities of the cell's own population `k`;
* a **contamination** distribution `eta_k` — a weighted blend of every *other* population's native distribution;
* a per-cell latent `theta_j ~ Beta` giving the proportion of native counts, with a Bernoulli native/contaminant label per transcript.

Inference is **variational EM**: variational distributions over `theta` and the transcript labels maximise the ELBO; `phi`, `eta` and the Dirichlet hyper-parameter `delta` are re-estimated each iteration (the latter by a Minka fixed-point Dirichlet MLE — a port of `MCMCprecision::fit_dirichlet`). The output is a per-cell **contamination fraction** and a **decontaminated (native) count matrix**.

## Low-level functional API (mirrors R one-to-one)

```python
from pydecontx import (
    decontx, DecontXResult,
    decontx_initialize, decontx_em, decontx_loglik,
    calculate_native_matrix, fit_dirichlet,
)

# variational-EM building blocks
init = decontx_initialize(counts, theta, z)          # phi / eta initialisation
step = decontx_em(counts, colsums, theta, eta, phi, z)
ll   = decontx_loglik(counts, theta, eta, phi, z)

# decontaminated-matrix computation
native = calculate_native_matrix(counts, theta, eta, phi, z)

# Minka fixed-point Dirichlet MLE (= MCMCprecision::fit_dirichlet)
alpha = fit_dirichlet(proportions)["alpha"]
```

## What's included

| Python | R counterpart | Purpose |
|---|---|---|
| `decontx` | `decontX` | run DecontX on a count matrix / AnnData |
| `DecontXResult` | — | result object (`contamination`, `decontx_counts`, `estimates`, …) |
| `simulate_contamination` | `simulateContamination` | simulate a contaminated dataset with ground truth |
| `decontx_initialize` | internal | `phi` / `eta` initialisation |
| `decontx_em` | `decontXEM` (C++) | one variational-EM step |
| `decontx_loglik` | `decontXLogLik` (C++) | ELBO / log-likelihood |
| `calculate_native_matrix` | `calculateNativeMatrix` (C++) | decontaminated-matrix computation |
| `fit_dirichlet` | `MCMCprecision::fit_dirichlet` | Minka fixed-point Dirichlet MLE |

## Relationship to R decontX

pydecontx follows decontX 1.8.0. decontX draws the initial native proportions with
`rbeta(nCells, delta[1], delta[2])` inside `withr::with_seed(seed, ...)`; pydecontx draws them with a
port of R's Mersenne-Twister and of `rbeta` (Cheng's algorithms; `pydecontx.rrng`), so the same `seed`
gives the same start. The variational EM then follows the C++ loops of decontX: on five 10x
human–mouse mixtures (`report/`) both packages stop after the same number of iterations, and the
per-cell contamination agrees to ~1e-14 and the decontaminated counts to ~1e-11.

The per-transcript updates are vectorised and computed in cache-sized blocks; `n_jobs` spreads the
blocks over threads without changing the result.

`tests/test_r_parity.py` runs decontX through `Rscript` (set `PYDECONTX_RSCRIPT`) and checks
contamination, `theta`, the decontaminated matrix and the background mode to 1e-9;
`tests/test_r_reference.py` checks R's `rbeta` draws against stored values (no R needed).

## Use in omicverse

`ov.pp.ambient.remove_ambient(adata, method='decontx', cluster_key=...)` runs pydecontx on a
filtered, clustered AnnData.

## Citation

If you use this package, please cite the original DecontX paper:

> Yang, S. *et al.* **Decontamination of ambient RNA in single-cell RNA-seq with DecontX.** *Genome Biology* **21**, 57 (2020).

and acknowledge omicverse / this repo for the Python port.

## License

Apache-2.0. The upstream Bioconductor `decontX` / `celda` packages are MIT-licensed; `pydecontx` is an independent re-implementation from the published algorithm and the `decontX` source.
