"""pydecontx -- pure-Python port of the Bioconductor ``decontX`` package.

DecontX (Yang et al., *Genome Biology* 2020) identifies and removes
ambient-RNA / cross-contamination from droplet single-cell RNA-seq.
Each cell's observed UMI counts are modelled as a Bayesian
two-component multinomial mixture of a *native* gene distribution
``phi`` and a *contamination* distribution ``eta`` (a weighted blend of
every other population). Inference is by variational EM and yields a
per-cell contamination fraction and a decontaminated count matrix.

Main entry points
-----------------
- :func:`decontx`              -- run DecontX on a count matrix / AnnData.
- :class:`DecontXResult`       -- the result object.
- :func:`simulate_contamination` -- simulate a contaminated dataset.

Lower-level routines (variational-EM building blocks) are available in
:mod:`pydecontx._core` and the Dirichlet MLE in
:mod:`pydecontx._dirichlet`.
"""
from __future__ import annotations

from ._core import (
    calculate_native_matrix,
    decontx_em,
    decontx_initialize,
    decontx_loglik,
)
from ._dirichlet import fit_dirichlet
from .decontx import DecontXResult, decontx
from .simulate import simulate_contamination

__version__ = "0.1.0"

__all__ = [
    "decontx",
    "DecontXResult",
    "simulate_contamination",
    "decontx_initialize",
    "decontx_em",
    "decontx_loglik",
    "calculate_native_matrix",
    "fit_dirichlet",
    "__version__",
]
