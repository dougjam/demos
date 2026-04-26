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
"""Tier 2 phase-controlled full-pipeline tests.

For each (base, training) example pair, re-run the Python port with the
parameters captured in tier2_metadata.json and compare against the
cached `.npz` golden. Outputs should be byte-identical (same Python +
same scipy + same input = same output).
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "python"))

from texture_synthesis import synthesize  # noqa: E402
from vector_io import read_vector  # noqa: E402

ROOT = _HERE.parents[1]
GOLDEN = _HERE / "golden"
META_PATH = GOLDEN / "tier2_metadata.json"
SRC_DIR = ROOT / "srcOrig" / "work"


def _load_metadata() -> dict | None:
    if not META_PATH.exists():
        return None
    return json.loads(META_PATH.read_text())


_META = _load_metadata()


def _cases():
    if _META is None:
        return []
    return [c["name"] for c in _META["cases"]]


def _params_from_xml(path: Path) -> dict:
    cfg = dict(ET.parse(path).getroot().attrib)
    base_dir = path.parent
    return {
        "fs": float(cfg.get("Fs", 44100)),
        "num_levels": int(cfg.get("numLevels", 6)),
        "window_hw": int(cfg.get("windowHW", 4)),
        "feature_hw": int(cfg.get("featureHW", 3)),
        "falloff": float(cfg.get("falloff", 0.0)),
        "scale_cdf": bool(int(cfg.get("scaleCDF", 1))),
        "scaling_alpha": float(cfg.get("scalingAlpha", 1.0)),
        "base_path": base_dir / cfg.get("basesignal", "input_data.vector"),
        "training_path": base_dir / cfg.get(
            "trainingsignal", "training_data.vector"
        ),
    }


@pytest.mark.skipif(
    _META is None,
    reason="Tier 2 goldens not yet generated; "
           "run python/tools/generate_goldens.py --tier 2",
)
@pytest.mark.parametrize("name", _cases())
def test_phase_controlled_pipeline(name: str) -> None:
    cfg_path = SRC_DIR / name / "default.xml"
    params = _params_from_xml(cfg_path)
    base = read_vector(params["base_path"])
    training = read_vector(params["training_path"])
    out = synthesize(
        base, training,
        fs=params["fs"],
        num_levels=params["num_levels"],
        window_hw=params["window_hw"],
        feature_hw=params["feature_hw"],
        falloff=params["falloff"],
        scale_cdf=params["scale_cdf"],
        scaling_alpha=params["scaling_alpha"],
    )
    golden = np.load(GOLDEN / f"{name}_synthesized.npz")
    expected = golden["output"]
    # Same Python + same input = byte-identical output.
    assert out.shape == expected.shape
    assert np.array_equal(out, expected), (
        f"{name}: output diverged. "
        f"max abs diff = {np.max(np.abs(out - expected)):.3e}"
    )
