# py-decontx

Pure-Python port of the Bioconductor package
**[decontX](https://bioconductor.org/packages/release/bioc/html/decontX.html)**
(`campbio/celda`) — **DecontX** (Yang *et al.*, *Genome Biology* 2020):
**decontamination of ambient RNA** in droplet single-cell RNA-seq.

`pydecontx` estimates, for every cell, the fraction of UMI counts that
originate from **ambient / cross-contaminating RNA** rather than the
cell's own transcriptome, and returns a **decontaminated count matrix**.
It is a standalone re-implementation that does **not** require R or
`rpy2`.

| | |
|---|---|
| PyPI / import name | `pydecontx` |
| Repository | `omicverse/py-decontx` |
| License | Apache-2.0 |
| Upstream | Bioconductor `decontX` / `celda` (MIT, R + C++) |
| Numerical parity | contamination fraction Pearson r > 0.99 vs R `decontX` |

## Install

```bash
pip install pydecontx              # once published
# or, from a checkout:
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas`, `anndata`.

## The model

DecontX models each cell's observed counts as a Bayesian **two-component
multinomial mixture**:

* a **native** distribution `phi_k` — the gene probabilities of the
  cell's own population `k`;
* a **contamination** distribution `eta_k` — a weighted blend of every
  *other* population's native distribution,
  `eta_k = sum_{k' != k} w_{k'} phi_{k'}`;
* a per-cell latent `theta_j ~ Beta` giving the proportion of native
  counts, with a Bernoulli native/contaminant label per transcript.

Inference is **variational EM**: variational distributions over `theta`
and the transcript labels are updated to maximise the ELBO; `phi`, `eta`
and the Dirichlet hyper-parameter `delta` are re-estimated each
iteration (the latter by a Minka fixed-point Dirichlet MLE — a port of
`MCMCprecision::fit_dirichlet`). The output is a per-cell
**contamination fraction** and a **decontaminated (native) count
matrix**.

The input is a raw UMI gene-by-cell matrix of **filtered** cells plus
**cell cluster labels** — empty droplets are *not* required. An optional
`background` matrix of raw empty droplets can be supplied; its empirical
transcript distribution then **anchors** the contamination distribution
`eta` instead of estimating it from the cells.

## Quick start

```python
import pydecontx as dx

# 1. a synthetic contaminated dataset (genes x cells)
sim = dx.simulate_contamination(C=300, G=100, K=3, delta=(1, 10))

# 2. run DecontX with cluster labels
res = dx.decontx(sim["observed_counts"], z=sim["z"])

res.contamination            # per-cell contamination fraction (0-1)
res.decontx_counts           # decontaminated sparse count matrix
res.decontaminated_counts()  # ... integer-rounded
res.to_dataframe()           # per-cell summary table
```

### AnnData

```python
import scanpy as sc, pydecontx as dx

adata = sc.read_10x_h5("filtered.h5")
sc.pp.pca(adata); sc.pp.neighbors(adata); sc.tl.leiden(adata)

# write results back into the AnnData
adata = dx.decontx(adata, z="leiden", copy=True)
adata.obs["decontX_contamination"]      # per-cell contamination
adata.layers["decontX_counts"]          # decontaminated counts
```

`x` may be an `AnnData` (cells × genes), a `pandas.DataFrame`, a NumPy
array or a SciPy sparse matrix (genes × cells). `z` is required — DecontX
needs a broad clustering of cell types; when `x` is an `AnnData` it may
name a column of `.obs`.

### Empty-droplet background

```python
res = dx.decontx(counts, z=clusters, background=empty_droplet_counts)
```

## API

| Object | Purpose |
|---|---|
| `decontx` | run DecontX on a count matrix / AnnData |
| `DecontXResult` | result object (`contamination`, `decontx_counts`, `estimates`, …) |
| `simulate_contamination` | simulate a contaminated dataset with ground truth |
| `decontx_initialize` / `decontx_em` / `decontx_loglik` | variational-EM building blocks |
| `calculate_native_matrix` | decontaminated-matrix computation |
| `fit_dirichlet` | Minka fixed-point Dirichlet MLE |

## R parity

`tests/` runs the **same synthetic two/three-population contaminated
count matrix** — with identical cluster labels — through Bioconductor
`decontX` (R) and `pydecontx`, and asserts agreement on

* the per-cell **contamination fraction** (Pearson r > 0.99),
* the **decontaminated count matrix** (Pearson r > 0.99),
* the final per-cell `theta`,
* the **background-anchored** mode.

Both also recover the known simulated contamination at r > 0.99 — the
level the DecontX paper reports against ground truth.

**Unavoidable difference.** The variational EM is deterministic *given
its initial `theta`*, but `theta` is seeded by a Beta draw: R uses its
Mersenne-Twister RNG (`stats::rbeta` under `withr::with_seed`), NumPy
uses PCG64. The two initialisations differ, so the converged estimates
agree to high correlation rather than bit-exactly. R also runs the EM
inner loops in C++; `pydecontx` reproduces them in vectorised NumPy.

## References

* Yang, S. *et al.* Decontamination of ambient RNA in single-cell
  RNA-seq with DecontX. *Genome Biology* **21**, 57 (2020).
* `decontX` / `celda`: <https://github.com/campbio/celda>
* Minka, T. Estimating a Dirichlet distribution. *Technical Report*
  (2000).

## License

Apache-2.0. The upstream Bioconductor `decontX` / `celda` packages are
MIT-licensed; `pydecontx` is an independent re-implementation from the
published algorithm and the `decontX` source.
