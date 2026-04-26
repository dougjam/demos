# This file is a derivative work of the Matlab implementation released
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
"""Tier 2 phase-controlled full-pipeline tests.

For each (signal, alpha) pair, the test feeds the same fixed phase array
that produced the golden, runs ``extend_signal``, and compares the
extended waveform and per-window beta values to the cached `.npz` golden
under ``python/tests/golden/``.

The goldens are produced by the Python port itself (the canonical
implementation in this repo). Tier 2 passes when the Python port is
self-consistent with its committed reference. The same goldens are the
parity target the JavaScript port must reproduce.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "python"))

from bandwidth_extension import extend_signal  # noqa: E402
from deterministic_rng import DeterministicRng  # noqa: E402
from vector_io import read_vector  # noqa: E402

ROOT = _HERE.parents[1]
GOLDEN = _HERE / "golden"
META_PATH = GOLDEN / "tier2_metadata.json"
SRC_DIR = ROOT / "srcOrig" / "work"


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _load_metadata() -> dict | None:
    if not META_PATH.exists():
        return None
    return json.loads(META_PATH.read_text())


_META = _load_metadata()


def _cases():
    if _META is None:
        return []
    return [(c["name"], c["alpha"]) for c in _META["cases"]]


@pytest.mark.skipif(_META is None, reason="Tier 2 goldens not yet generated; run python/tools/generate_goldens.py --tier 2")
@pytest.mark.parametrize("name,alpha", _cases())
def test_phase_controlled_pipeline(name: str, alpha: float) -> None:
    seed = _META["phases_seed"]
    fs = _META["fs"]
    p = read_vector(SRC_DIR / f"{name}.vector")
    NFFT = _next_pow2(p.size)
    rng = DeterministicRng(seed)
    phases = 2.0 * np.pi * rng.random_array(NFFT)
    diag = extend_signal(
        p, fs=fs, alpha=alpha, phases_override=phases, return_diagnostics=True
    )
    golden = np.load(GOLDEN / f"{name}_alpha{alpha}_seed{seed}.npz")
    ext_g = golden["extended"]
    betas_g = golden["betas"]

    # Per spec section 3.2 tolerances.
    np.testing.assert_allclose(diag["betas"], betas_g, atol=1e-12, rtol=1e-10)

    ext = diag["extended"]
    diff = ext - ext_g
    peak_g = float(np.max(np.abs(ext_g))) or 1.0
    rms_g = float(np.sqrt(np.mean(ext_g**2))) or 1.0
    peak_rel = float(np.max(np.abs(diff))) / peak_g
    l2_rel = float(np.sqrt(np.mean(diff**2))) / rms_g
    assert peak_rel < 1e-10, f"{name} alpha={alpha}: peak rel error {peak_rel:.3e}"
    assert l2_rel < 1e-12, f"{name} alpha={alpha}: L2 rel error {l2_rel:.3e}"
