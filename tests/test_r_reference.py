"""R's random numbers for the decontX initialisation (values from R 4.5.3; no R needed)."""
from __future__ import annotations

import numpy as np

from pydecontx.rrng import RRandom


def test_rbeta_algorithm_bb_follows_r():
    # set.seed(12345); rbeta(5, 10, 10)  -- decontX's default delta
    ref = [float.fromhex(v) for v in
           "0x1.26200745830e6p-1 0x1.2e5bc8bab6f1ep-1 0x1.f1e028b57b85ap-2 0x1.c520d416608dep-2 0x1.2778de7f815e8p-1".split()]
    got = RRandom(12345).rbeta(5, 10, 10)
    np.testing.assert_allclose(got, ref, rtol=0, atol=0)


def test_rbeta_algorithm_bc_and_stream_follow_r():
    # set.seed(7); c(rbeta(4, 0.5, 3), rbeta(3, 3, 0.7), runif(1))
    ref = [2.096258366535085e-05, 0.014199115245910748, 0.19493452064606417, 0.092835960107919918,
           0.99945070538823966, 0.9066855203308698, 0.99960973110523399, 0.62703875848092139]
    r = RRandom(7)
    got = np.r_[r.rbeta(4, 0.5, 3), r.rbeta(3, 3, 0.7), r.unif_rand(1)]
    np.testing.assert_allclose(got, ref, rtol=1e-15)
