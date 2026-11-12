"""Figure, table and the numbers quoted in the pydecontx technical report.

Usage: python 05_figures.py <runs_dir> <tasks_dir> <out_dir>
Compares decontX 1.8.0 and pydecontx on five 10x Genomics human-mouse mixtures. Counts of mouse genes in
human cells (and of human genes in mouse cells) can only come from ambient RNA: relative to the share of the
other species in the ambient profile (droplets with fewer than 100 counts) they give each cell's ground-truth
contamination fraction.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
import scipy.io
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

RUNS, TASKS, OUT = map(pathlib.Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
C_R, C_PY, C_TRUE = "#eb6834", "#2a78d6", "#52514e"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})
ENG = {"R": "decontX", "py": "pydecontx"}
LABEL = {"hgmm_1k_v3": "hgmm 1k (v3)", "hgmm_5k_v3": "hgmm 5k (v3)", "hgmm_6k": "hgmm 6k (v2)",
         "hgmm_10k_v3": "hgmm 10k (v3)", "hgmm_12k": "hgmm 12k (v2)"}
EXAMPLE = "hgmm_12k"


def result(engine, task, run=1):
    d = RUNS / ENG[engine] / task / f"run{run}"
    r = json.load(open(d / "result.json"))
    adj = sp.csc_matrix((np.fromfile(d / "adj_x.bin"), np.fromfile(d / "adj_i.bin", np.int32),
                         np.fromfile(d / "adj_p.bin", np.int32)), shape=tuple(r["dim"]))
    return r, adj, np.fromfile(d / "contamination.bin")


rows, pooled, umap_frac = [], [], {}
for t in LABEL:
    td = TASKS / t
    genes = pd.read_csv(td / "genes.tsv", sep="\t")
    species = pd.read_csv(td / "species.tsv", sep="\t")
    droplets = (td / "droplets.tsv").read_text().split()
    tod = sp.csc_matrix(scipy.io.mmread(gzip.open(td / "tod.mtx.gz")))
    toc = tod[:, pd.Index(droplets).get_indexer(species.cell)].tocsc()
    mouse_gene = (genes.genome == "mouse").to_numpy()
    numi = np.asarray(tod.sum(axis=0)).ravel()
    empty = (numi > 0) & (numi < 100)                             # SoupX's default soup range
    soup = np.asarray(tod[:, empty].sum(axis=1)).ravel()
    soup_mouse = soup[mouse_gene].sum() / soup.sum()
    hum, mou = (species.species == "human").to_numpy(), (species.species == "mouse").to_numpy()
    pure = hum | mou
    res = {e: result(e, t) for e in ENG}

    def split(m):
        m_cnt = np.asarray(m[mouse_gene].sum(axis=0)).ravel()
        h_cnt = np.asarray(m[~mouse_gene].sum(axis=0)).ravel()
        return np.where(hum, m_cnt, np.where(mou, h_cnt, np.nan)), np.where(hum, h_cnt, np.where(mou, m_cnt, np.nan))

    o0, s0 = split(toc)
    n = o0 + s0
    expected = np.where(hum, soup_mouse, 1 - soup_mouse)
    true_cell = o0 / (n * expected)
    row = dict(dataset=t, label=LABEL[t], droplets=tod.shape[1], cells=toc.shape[1],
               true_rho=o0[pure].sum() / (n[pure] * expected[pure]).sum())
    for e in ENG:
        r, adj, cont = res[e]
        o1, s1 = split(adj)
        row[f"mean_{e}"] = float(np.sum(cont[pure] * n[pure]) / n[pure].sum())   # count-weighted, as the truth
        row[f"removed_{e}"] = 1 - o1[pure].sum() / o0[pure].sum()
        row[f"kept_{e}"] = s1[pure].sum() / s0[pure].sum()
        row[f"removed_human_{e}"] = 1 - o1[hum].sum() / o0[hum].sum()
        row[f"removed_mouse_{e}"] = 1 - o1[mou].sum() / o0[mou].sum()
        row[f"sec_{e}"] = min(json.load(open(RUNS / ENG[e] / t / f"run{k}" / "result.json"))["seconds"] for k in (1, 2))
        row[f"iter_{e}"] = r["iterations"]
        if t == EXAMPLE:
            umap_frac[e] = np.where(pure, o1 / (o1 + s1), np.nan)
    t8 = RUNS / "pydecontx_8t" / t
    row["sec_py8"] = min(json.load(open(t8 / f"run{k}" / "result.json"))["seconds"] for k in (1, 2)) if t8.exists() else np.nan
    cr, cp = res["R"][2], res["py"][2]
    row["cont_maxdiff"] = np.abs(cr - cp).max()
    row["adj_maxdiff"] = abs(res["py"][1] - res["R"][1]).max()
    row["entries"] = res["R"][1].nnz
    row["same_iter"] = res["R"][0]["iterations"] == res["py"][0]["iterations"]
    row["delta_reldiff"] = float(np.max(np.abs(np.array(res["py"][0]["delta"]) / np.array(res["R"][0]["delta"]) - 1)))
    row["cell_pearson"] = pearsonr(true_cell[pure], cr[pure])[0]
    row["cell_spearman"] = spearmanr(true_cell[pure], cr[pure])[0]
    row["truth_numi_spearman"] = spearmanr(true_cell[pure], n[pure])[0]
    pooled.append(np.column_stack([cr, cp]))
    if t == EXAMPLE:
        ex_true, ex_est = true_cell[pure], cr[pure]
        umap_frac["before"] = np.where(pure, o0 / n, np.nan)
        umap_species = species.species.to_numpy()
        ad = scanpy.AnnData(sp.csr_matrix(toc.T).astype(np.float32))
        scanpy.pp.normalize_total(ad, target_sum=1e4)
        scanpy.pp.log1p(ad)
        scanpy.pp.highly_variable_genes(ad, n_top_genes=2000, subset=True)
        scanpy.pp.pca(ad, n_comps=30, random_state=0)
        scanpy.pp.neighbors(ad, n_neighbors=15, random_state=0)
        scanpy.tl.umap(ad, random_state=0)
        xy = ad.obsm["X_umap"]
    rows.append(row)
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "comparison_per_dataset.tsv", sep="\t", index=False)
pooled = np.vstack(pooled)

# ======================= Figure 1 =======================
fig = plt.figure(figsize=(7.2, 6.5))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.95], hspace=0.55)


def panel(ax, s, dx=-0.36):
    ax.text(dx, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def crop(img):
    ink = (img[..., 3] > 0.04) & (img[..., :3].min(-1) < 0.96)
    ys, xs = np.where(ink)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


top = gs[0].subgridspec(1, 4, width_ratios=[0.62, 1, 1, 1], wspace=0.55)
logos = [crop(plt.imread(ASSETS / f)) for f in ["pydecontx_logo.png", "omicverse_logo.png"]]
x = np.arange(len(tab))

axa = fig.add_subplot(top[1])
lo = 1e-4
pc_ = np.clip(pooled, lo, 1)
axa.plot([lo, 1], [lo, 1], color=INK2, lw=0.6, zorder=0)
axa.scatter(pc_[:, 0], pc_[:, 1], s=0.5, color=C_PY, lw=0, rasterized=True)
axa.set_xscale("log")
axa.set_yscale("log")
axa.set_xlim(lo, 1)
axa.set_ylim(lo, 1)
axa.set_aspect("equal")
axa.set_xlabel("decontX (R)")
axa.set_ylabel("pydecontx")
axa.set_title("Contamination per cell")
axa.text(0.04, 0.96, f"{len(pooled):,} cells\nmax |difference| {tab.cont_maxdiff.max():.0e}", transform=axa.transAxes,
         va="top", fontsize=5, color=INK2)
axa.grid(color=GRID, lw=0.5)
axa.set_axisbelow(True)
panel(axa, "a")

axb = fig.add_subplot(top[2])
axb.scatter(x, tab.true_rho * 100, s=30, marker="_", color=C_TRUE, linewidth=1.2, label="cross-species truth", zorder=3)
axb.scatter(x - 0.12, tab.mean_R * 100, s=12, color=C_R, edgecolor="white", linewidth=0.3, label="decontX (R)")
axb.scatter(x + 0.12, tab.mean_py * 100, s=12, color=C_PY, edgecolor="white", linewidth=0.3, label="pydecontx")
axb.set_xticks(x, tab.label, rotation=40, ha="right", rotation_mode="anchor", fontsize=5.5)
axb.set_ylim(0, max(tab.mean_R.max(), tab.true_rho.max()) * 135)
axb.set_ylabel("contamination (%)")
axb.set_title("Contamination per mixture")
axb.grid(axis="y", color=GRID, lw=0.5)
axb.set_axisbelow(True)
axb.legend(fontsize=5, loc="upper left", handletextpad=0.1, borderaxespad=0.1)
panel(axb, "b")

axc = fig.add_subplot(top[3])
hi = np.nanpercentile(np.r_[ex_true, ex_est] * 100, 99.5)
axc.plot([0, hi], [0, hi], color=INK2, lw=0.6, zorder=0)
axc.scatter(ex_true * 100, ex_est * 100, s=0.8, color=C_PY, lw=0, rasterized=True)
axc.set_xlim(0, hi)
axc.set_ylim(0, hi)
axc.set_aspect("equal")
axc.set_xlabel("cross-species truth (%)")
axc.set_ylabel("decontX estimate (%)")
axc.set_title(f"Per cell ({LABEL[EXAMPLE].split(' (')[0]})")
ex = tab.set_index("dataset").loc[EXAMPLE]
axc.text(0.04, 0.96, f"Spearman {ex.cell_spearman:.2f}", transform=axc.transAxes, va="top", fontsize=5, color=INK2)
axc.grid(color=GRID, lw=0.5)
axc.set_axisbelow(True)
panel(axc, "c")

bot = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.3)
axd = fig.add_subplot(bot[0])
w = 0.36
axd.bar(x - w / 2, tab.removed_R * 100, w * 0.9, color=C_R, label="decontX (R)")
axd.bar(x + w / 2, tab.removed_py * 100, w * 0.9, color=C_PY, label="pydecontx")
axd.set_xticks(x, tab.label, rotation=25, ha="right", rotation_mode="anchor", fontsize=5.5)
axd.set_ylim(0, 112)
axd.set_yticks([0, 20, 40, 60, 80, 100])
axd.set_ylabel("removed (%)")
axd.set_title("Cross-species counts removed")
axd.grid(axis="y", color=GRID, lw=0.5)
axd.set_axisbelow(True)
axd.legend(fontsize=5.5, loc="upper center", ncol=2, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)
panel(axd, "d", dx=-0.2)

axe = fig.add_subplot(bot[1])
axe.scatter(tab.cells, tab.sec_R, s=12, color=C_R, edgecolor="white", linewidth=0.3, label="decontX 1.8.0 (R), 1 thread")
axe.scatter(tab.cells, tab.sec_py, s=12, color=C_PY, edgecolor="white", linewidth=0.3, label="pydecontx, 1 thread")
axe.scatter(tab.cells, tab.sec_py8, s=12, color="#9ec5f4", edgecolor="white", linewidth=0.3, label="pydecontx, 8 threads")
for _, r in tab.iterrows():
    axe.plot([r.cells] * 2, [r.sec_py, r.sec_R], color=GRID, lw=0.8, zorder=0)
axe.set_xscale("log")
axe.set_yscale("log")
axe.set_xlabel("cells")
axe.set_yticks([5, 10, 20, 50, 100, 200, 500], ["5", "10", "20", "50", "100", "200", "500"])
axe.minorticks_off()
axe.set_xticks([1000, 2000, 5000, 10000], ["1,000", "2,000", "5,000", "10,000"])
axe.set_ylabel("wall time (s)")
axe.set_title("Run time")
axe.grid(color=GRID, lw=0.5)
axe.set_axisbelow(True)
axe.legend(fontsize=5.5, loc="upper left", handletextpad=0.1)
panel(axe, "e", dx=-0.2)

row3 = gs[2].subgridspec(1, 4, wspace=0.06)


def umap_ax(k, title, letter):
    ax = fig.add_subplot(row3[k])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title(title)
    ax.set_aspect("equal")
    panel(ax, letter, dx=-0.04)
    return ax


ax = umap_ax(0, f"{LABEL[EXAMPLE]}: species", "f")
for name, col in [("human", "#2f7f6f"), ("mouse", "#b5a642"), ("mixed", "#2a2a2a")]:
    m = umap_species == name
    ax.scatter(xy[m, 0], xy[m, 1], s=0.6 if name != "mixed" else 1.2, c=col, lw=0, rasterized=True,
               label=f"{name} ({m.sum():,})")
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=5.5, handletextpad=0.1,
                columnspacing=0.6)
for h in leg.legend_handles:
    h.set_sizes([10])
vmax = np.nanpercentile(umap_frac["before"] * 100, 99)
for k, (key, title, letter) in enumerate([("before", "Before correction", "g"), ("R", "After, decontX (R)", "h"),
                                          ("py", "After, pydecontx", "i")]):
    ax = umap_ax(k + 1, title, letter)
    v = umap_frac[key] * 100
    nan = np.isnan(v)
    ax.scatter(xy[nan, 0], xy[nan, 1], s=0.6, c="#d9d8d3", lw=0, rasterized=True)
    o = np.flatnonzero(~nan)[np.argsort(v[~nan], kind="stable")]
    im = ax.scatter(xy[o, 0], xy[o, 1], c=v[o], s=0.6, cmap="magma_r", vmin=0, vmax=vmax, lw=0, rasterized=True)
cax = fig.axes[-2].inset_axes([0.05, -0.14, 0.9, 0.035])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.5, length=1.5)
cax.set_title("other-species counts (% of cell; mixed cells grey)", fontsize=5.5, color=INK2, pad=2)

pa, pd_ = axa.get_position(), axd.get_position()
x0, x1 = pd_.x0 - 0.2 * pd_.width, pa.x0 - 0.075
y0, y1 = pa.y0 - 0.02, pa.y1 + 0.035
h_ov = (x1 - x0) * 7.2 / 6.5 * logos[1].shape[0] / logos[1].shape[1]
for img, rect, anchor in [(logos[0], [x0, y0 + h_ov + 0.012, x1 - x0, y1 - y0 - h_ov - 0.012], "N"),
                          (logos[1], [x0, y0, x1 - x0, h_ov], "S")]:
    axl = fig.add_axes(rect)
    axl.imshow(img)
    axl.set_anchor(anchor)
    axl.set_axis_off()
fig.savefig(OUT / "fig1.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig1.png", bbox_inches="tight", pad_inches=0.02, dpi=200)

# ======================= numbers =======================
speed = tab.sec_R / tab.sec_py
big = tab.iloc[tab.cells.idxmax()]
macros = {
    "nDatasets": len(tab), "nCellsTotal": f"{int(tab.cells.sum()):,}",
    "nCellsMin": f"{int(tab.cells.min()):,}", "nCellsMax": f"{int(tab.cells.max()):,}",
    "nPooled": f"{len(pooled):,}", "contMaxDiff": f"{tab.cont_maxdiff.max():.0e}",
    "adjMaxDiff": f"{tab.adj_maxdiff.max():.0e}", "nSameIter": int(tab.same_iter.sum()),
    "deltaRelDiff": f"{tab.delta_reldiff.max():.0e}",
    "iterMin": int(tab.iter_R.min()), "iterMax": int(tab.iter_R.max()),
    "meanMin": f"{tab.mean_R.min() * 100:.1f}", "meanMax": f"{tab.mean_R.max() * 100:.1f}",
    "trueMin": f"{tab.true_rho.min() * 100:.1f}", "trueMax": f"{tab.true_rho.max() * 100:.1f}",
    "removedMin": f"{tab.removed_R.min() * 100:.0f}", "removedMax": f"{tab.removed_R.max() * 100:.0f}",
    "keptMin": f"{tab.kept_R.min() * 100:.1f}", "keptMax": f"{tab.kept_R.max() * 100:.1f}",
    "cellSpearMin": f"{tab.cell_spearman.min():.2f}", "cellSpearMax": f"{tab.cell_spearman.max():.2f}",
    "exCellSpear": f"{ex.cell_spearman:.2f}", "exRemHuman": f"{ex.removed_human_R * 100:.0f}",
    "exRemMouse": f"{ex.removed_mouse_R * 100:.0f}",
    "speedMin": f"{speed.min():.1f}", "speedMax": f"{speed.max():.1f}", "speedMed": f"{speed.median():.1f}",
    "tBigR": f"{big.sec_R:.0f}", "tBigPy": f"{big.sec_py:.0f}", "bigName": big.label,
    "tBigPyEight": f"{big.sec_py8:.0f}",
    "speedEightMin": f"{(tab.sec_R / tab.sec_py8).min():.1f}", "speedEightMax": f"{(tab.sec_R / tab.sec_py8).max():.1f}",
}
for key, name in [("OneK", "hgmm_1k_v3"), ("FiveK", "hgmm_5k_v3"), ("SixK", "hgmm_6k"), ("TenK", "hgmm_10k_v3"),
                  ("TwelveK", "hgmm_12k")]:
    r_ = tab.set_index("dataset").loc[name]
    macros[f"mean{key}"] = f"{r_.mean_R * 100:.1f}"
    macros[f"true{key}"] = f"{r_.true_rho * 100:.1f}"
    macros[f"removed{key}"] = f"{r_.removed_R * 100:.0f}"
with open(OUT / "numbers.tex", "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(OUT / "numbers.tex").read())
print(tab.drop(columns=["label"]).round(4).to_string())
