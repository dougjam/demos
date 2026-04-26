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
"""Generate Tier 1 (kernel) and Tier 2 (full pipeline) golden fixtures.

Run from the project root:

    python python/tools/generate_goldens.py            # both tiers
    python python/tools/generate_goldens.py --tier 1   # just kernels (~1 s)
    python python/tools/generate_goldens.py --tier 2   # full pipeline (~5 min)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from cdf_match import sample_cdf, sample_inverse_cdf  # noqa: E402
from gaussian_pyramid import (  # noqa: E402
    GaussianPyramid,
    build_gaussian_level,
    pad_input_signal,
)
from texture_synthesis import synthesize  # noqa: E402
from deterministic_rng import DeterministicRng  # noqa: E402
from vector_io import read_vector  # noqa: E402

ROOT = _HERE.parents[1]
GOLDEN_DIR = ROOT / "python" / "tests" / "golden"

EXAMPLES = ["burning_brick", "candle", "dragon", "flame_jet", "torch"]


def _save_f64(arr: np.ndarray, path: Path) -> None:
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    path.write_bytes(arr.astype("<f8").tobytes())


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)


def generate_tier1() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # Pyramid level reduction on a deterministic synthetic input.
    rng = DeterministicRng(42)
    sig = np.array(
        [rng.random() * 2.0 - 1.0 for _ in range(257)], dtype=np.float64
    )
    next_level = build_gaussian_level(sig)
    _save_f64(next_level, GOLDEN_DIR / "gaussian_level_257to129.f64")

    # Pyramid construction: shapes + start/end indices for a few sizes.
    shapes_meta = []
    for n in (200, 1000, 4096):
        p = GaussianPyramid(np.linspace(-1.0, 1.0, n), 6, reflect_boundaries=True)
        shapes_meta.append({
            "input_size": n,
            "level_sizes": [int(lvl.size) for lvl in p.levels],
            "start_indices": [int(p.start_index(i)) for i in range(p.num_levels)],
            "end_indices": [int(p.end_index(i)) for i in range(p.num_levels)],
        })

    # CDF sample / inverse on a hand-picked sorted array.
    cdf = np.array([0.0, 0.5, 1.5, 3.0, 5.0, 7.5, 10.0])
    cdf_meta = {
        "cdf": cdf.tolist(),
        "samples": [
            {"fraction": 0.0,  "amplitude": sample_cdf(0.0,  cdf)},
            {"fraction": 0.25, "amplitude": sample_cdf(0.25, cdf)},
            {"fraction": 0.5,  "amplitude": sample_cdf(0.5,  cdf)},
            {"fraction": 0.75, "amplitude": sample_cdf(0.75, cdf)},
            {"fraction": 1.0,  "amplitude": sample_cdf(1.0,  cdf)},
        ],
        "inverse_samples": [
            {"amplitude": 0.0,  "fraction": sample_inverse_cdf(0.0,  cdf)},
            {"amplitude": 0.5,  "fraction": sample_inverse_cdf(0.5,  cdf)},
            {"amplitude": 2.25, "fraction": sample_inverse_cdf(2.25, cdf)},
            {"amplitude": 5.0,  "fraction": sample_inverse_cdf(5.0,  cdf)},
            {"amplitude": 10.0, "fraction": sample_inverse_cdf(10.0, cdf)},
            {"amplitude": 50.0, "fraction": sample_inverse_cdf(50.0, cdf)},  # clamped
            {"amplitude": -5.0, "fraction": sample_inverse_cdf(-5.0, cdf)},  # clamped
        ],
    }

    # KD-tree determinism: build a tree on a fixed point set and query
    # for the matched index for a fixed query.
    from scipy.spatial import cKDTree
    rng2 = DeterministicRng(7)
    points = np.array([
        [rng2.random() for _ in range(8)] for _ in range(64)
    ], dtype=np.float64)
    tree = cKDTree(points)
    query = np.array([0.5] * 8)
    dist, idx = tree.query(query, k=1)
    kdtree_meta = {
        "n_points": int(points.shape[0]),
        "n_dim": int(points.shape[1]),
        "query": query.tolist(),
        "matched_index": int(idx),
        "matched_distance": float(dist),
    }

    # PCG32 first 8 outputs at seed 42 (re-used from sibling demo's pin).
    rng3 = DeterministicRng(42)
    rng_first8 = [rng3.random() for _ in range(8)]

    metadata = {
        "tier": 1,
        "regenerated_with": "python",
        "pyramid_shapes": shapes_meta,
        "cdf": cdf_meta,
        "kdtree": kdtree_meta,
        "pcg32_seed42_first8": rng_first8,
    }
    (GOLDEN_DIR / "tier1_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        "tier1: wrote gaussian_level_257to129, plus shapes / CDF / KD-tree / "
        "PCG32 metadata"
    )


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
        "base_path": str(base_dir / cfg.get("basesignal", "input_data.vector")),
        "training_path": str(
            base_dir / cfg.get("trainingsignal", "training_data.vector")
        ),
    }


def generate_tier2() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    src = ROOT / "srcOrig" / "work"
    if not src.is_dir():
        print(f"missing {src}", file=sys.stderr)
        return

    metadata = {
        "tier": 2,
        "regenerated_with": "python",
        "cases": [],
    }

    for name in EXAMPLES:
        cfg_path = src / name / "default.xml"
        if not cfg_path.exists():
            print(f"skipping {name}: missing {cfg_path}")
            continue
        params = _params_from_xml(cfg_path)
        base = read_vector(params["base_path"])
        training = read_vector(params["training_path"])

        t0 = time.time()
        diag = synthesize(
            base, training,
            fs=params["fs"],
            num_levels=params["num_levels"],
            window_hw=params["window_hw"],
            feature_hw=params["feature_hw"],
            falloff=params["falloff"],
            scale_cdf=params["scale_cdf"],
            scaling_alpha=params["scaling_alpha"],
            return_diagnostics=True,
        )
        dt = time.time() - t0

        out_path = GOLDEN_DIR / f"{name}_synthesized.npz"
        _save_npz(
            out_path,
            output=diag["output"],
            scaling_per_window=diag["scaling_per_window"],
        )
        n_windows = int(diag["scaling_per_window"].size)
        print(
            f"tier2: {name}: base={base.size} samples, training={training.size} "
            f"samples, {n_windows} windows, took {dt:.1f}s, wrote {out_path.name}"
        )
        metadata["cases"].append({
            "name": name,
            "base_samples": int(base.size),
            "training_samples": int(training.size),
            "n_windows": n_windows,
            "elapsed_s": dt,
            "params": {k: v for k, v in params.items()
                        if k not in ("base_path", "training_path")},
            "file": out_path.name,
        })

    (GOLDEN_DIR / "tier2_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier", type=int, choices=[1, 2], default=None,
        help="generate only tier 1 or tier 2 (default: both)",
    )
    args = parser.parse_args(argv)
    if args.tier is None or args.tier == 1:
        generate_tier1()
    if args.tier is None or args.tier == 2:
        generate_tier2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
