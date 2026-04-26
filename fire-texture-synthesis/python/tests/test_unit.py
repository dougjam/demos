# This file is a derivative work of the C++ implementation released
# alongside Chadwick and James, "Animating Fire with Sound," SIGGRAPH 2011.
# See https://www.cs.cornell.edu/projects/Sound/fire/ for the original.
#
# Original copyright notice (preserved per BSD 2-Clause):
#
# Copyright (c) 2011, Jeffrey Chadwick
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Tier 1 unit tests: deterministic kernels.

Each test loads a binary or JSON golden produced by
``python/tools/generate_goldens.py`` and re-runs the kernel from the
Python port. Tolerance ``atol=rtol=1e-13`` for the Gaussian filter
output; the JSON metadata pins exact floats for the CDF and KD-tree
checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "python"))

from cdf_match import sample_cdf, sample_inverse_cdf  # noqa: E402
from gaussian_pyramid import (  # noqa: E402
    GaussianPyramid,
    build_gaussian_level,
    pad_input_signal,
)
from deterministic_rng import DeterministicRng  # noqa: E402

GOLDEN = _HERE / "golden"
TOL = dict(atol=1e-13, rtol=1e-13)


def _load_f64(name: str) -> np.ndarray:
    return np.frombuffer((GOLDEN / name).read_bytes(), dtype="<f8")


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads((GOLDEN / "tier1_metadata.json").read_text())


def test_gaussian_level_reduction() -> None:
    rng = DeterministicRng(42)
    sig = np.array([rng.random() * 2.0 - 1.0 for _ in range(257)], dtype=np.float64)
    out = build_gaussian_level(sig)
    assert out.shape == (129,)
    expected = _load_f64("gaussian_level_257to129.f64")
    assert np.allclose(out, expected, **TOL)


def test_pyramid_shapes(metadata: dict) -> None:
    for case in metadata["pyramid_shapes"]:
        n = case["input_size"]
        p = GaussianPyramid(np.linspace(-1.0, 1.0, n), 6, reflect_boundaries=True)
        assert [int(lvl.size) for lvl in p.levels] == case["level_sizes"]
        assert [int(p.start_index(i)) for i in range(p.num_levels)] == case["start_indices"]
        assert [int(p.end_index(i)) for i in range(p.num_levels)] == case["end_indices"]


def test_pad_input_signal_pow2_plus_one() -> None:
    """Padded length is the smallest 2^k + 1 that fits the input."""
    for n in (1, 2, 100, 256, 257, 258, 1000):
        out, start, end = pad_input_signal(np.arange(n, dtype=np.float64))
        # extended_length must be >= n
        assert out.size >= n
        # extended_length is 2^k + 1 for some k
        m = out.size - 1
        assert m & (m - 1) == 0, f"size {out.size} is not 2^k + 1"
        # The original signal is centred and recoverable.
        assert end - start == n
        assert np.array_equal(out[start:end], np.arange(n, dtype=np.float64))


def test_cdf_sampling(metadata: dict) -> None:
    cdf = np.array(metadata["cdf"]["cdf"], dtype=np.float64)
    for entry in metadata["cdf"]["samples"]:
        v = sample_cdf(entry["fraction"], cdf)
        assert v == pytest.approx(entry["amplitude"], rel=1e-13, abs=1e-13)
    for entry in metadata["cdf"]["inverse_samples"]:
        f = sample_inverse_cdf(entry["amplitude"], cdf)
        assert f == pytest.approx(entry["fraction"], rel=1e-13, abs=1e-13)


def test_cdf_roundtrip() -> None:
    cdf = np.sort(np.abs(np.linspace(-3.0, 4.0, 200)))
    # Roundtrip must use amplitudes inside [cdf[0], cdf[-1]]; outside
    # this range, sample_inverse_cdf clamps to 0 or 1.
    for amp in (cdf[0] + 0.01, 0.5, 1.5, 2.7, 3.5):
        f = sample_inverse_cdf(amp, cdf)
        v = sample_cdf(f, cdf)
        assert v == pytest.approx(amp, rel=1e-10, abs=1e-12)


def test_kdtree_determinism(metadata: dict) -> None:
    from scipy.spatial import cKDTree
    rng = DeterministicRng(7)
    points = np.array([
        [rng.random() for _ in range(metadata["kdtree"]["n_dim"])]
        for _ in range(metadata["kdtree"]["n_points"])
    ], dtype=np.float64)
    tree = cKDTree(points)
    query = np.array(metadata["kdtree"]["query"])
    dist, idx = tree.query(query, k=1)
    assert int(idx) == metadata["kdtree"]["matched_index"]
    assert float(dist) == pytest.approx(
        metadata["kdtree"]["matched_distance"], rel=1e-13, abs=1e-13
    )


def test_pcg32_first_outputs(metadata: dict) -> None:
    rng = DeterministicRng(42)
    actual = [rng.random() for _ in range(8)]
    assert actual == pytest.approx(metadata["pcg32_seed42_first8"], rel=0, abs=0)


def test_pcg32_determinism() -> None:
    a = [DeterministicRng(99).random() for _ in range(50)]
    b = [DeterministicRng(99).random() for _ in range(50)]
    assert a == b
    c = [DeterministicRng(100).random() for _ in range(50)]
    assert a != c


def test_pcg32_uniform_range() -> None:
    rng = DeterministicRng(0)
    for _ in range(2000):
        v = rng.random()
        assert 0.0 <= v < 1.0
