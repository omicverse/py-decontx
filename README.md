# pydecontx

A **pure-Python re-implementation of [DecontX](https://github.com/campbio/decontX)** (Yang et al., *Genome Biology* 2020) for decontamination of ambient / cross-contaminating RNA in droplet single-cell RNA-seq data.

- AnnData-native — drop-in for the scanpy ecosystem
- **No `rpy2`**, no R install — the variational-EM, the native/contamination multinomial mixture, and the Minka fixed-point Dirichlet MLE are all implemented directly in NumPy/SciPy
- Same function surface as the R workflow (`decontX` → per-cell contamination fraction + decontaminated count matrix)
- Numerical reproducibility against the Bioconductor reference — contamination fraction, decontaminated matrix and per-cell `theta` agree at Pearson r > 0.99 (see `tests/test_r_parity.py`)

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse). All algorithmic work is developed upstream in omicverse and synced here for users who want DecontX without the full omicverse stack.

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

## Reproducing R results

`examples/compare_R_vs_Python.ipynb` runs Bioconductor `decontX` (via `Rscript`) and `pydecontx` on the **same real PBMC 3k dataset** with identical cluster labels, and shows the per-cell contamination fraction, the decontaminated count matrix and the per-cell `theta` all agree at Pearson r > 0.99.

`tests/test_r_parity.py` runs the **same synthetic two/three-population contaminated count matrix** — with identical cluster labels — through Bioconductor `decontX` (R) and `pydecontx`, and asserts agreement on the contamination fraction, the decontaminated matrix, the final `theta`, and the background-anchored mode.

**Unavoidable difference.** The variational EM is deterministic *given its initial `theta`*, but `theta` is seeded by a Beta draw: R uses its Mersenne-Twister RNG (`stats::rbeta` under `withr::with_seed`), NumPy uses PCG64. The two initialisations differ, so the converged estimates agree to high correlation rather than bit-exactly. R also runs the EM inner loops in C++; `pydecontx` reproduces them in vectorised NumPy.

## Relationship to omicverse

Developed **upstream** in [`omicverse`](https://github.com/Starlitnightly/omicverse):

- Canonical implementation: synced to omicverse
- Standalone mirror (this repo): same code, same API, minus the omicverse packaging

## Citation

If you use this package, please cite the original DecontX paper:

> Yang, S. *et al.* **Decontamination of ambient RNA in single-cell RNA-seq with DecontX.** *Genome Biology* **21**, 57 (2020).

and acknowledge omicverse / this repo for the Python port.

## License

Apache-2.0. The upstream Bioconductor `decontX` / `celda` packages are MIT-licensed; `pydecontx` is an independent re-implementation from the published algorithm and the `decontX` source.
