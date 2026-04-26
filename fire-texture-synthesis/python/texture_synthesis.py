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
"""Top-level sound texture synthesis orchestrator.

Faithful port of ``srcOrig/texture/WindowSynthesizer.{h,cpp}``. Differences
are documented in ``docs/ALGORITHM.md``; the most consequential is the
use of exact nearest-neighbour search (``scipy.spatial.cKDTree``)
instead of the C++ default of ANN's (1+epsilon)-approximate search.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, List
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from gaussian_pyramid import GaussianPyramid
from deterministic_rng import DeterministicRng
from vector_io import read_vector, write_vector


# --------------------------------------------------------------------------- #
# Per-window blend                                                            #
# --------------------------------------------------------------------------- #

def blend_window_data(
    input_signal: np.ndarray,
    input_window: int,
    output_signal: np.ndarray,
    output_window: int,
    window_hw: int,
    scale: float = 1.0,
) -> None:
    """Mirror ``WindowSynthesizer::blendWindowData``.

    Adds the matched training window into the output via a triangular
    hat function of half-width ``window_hw``. The input/output window
    indices follow the C++ convention ``window_start = window_hw * (idx - 1)``,
    which means the leftmost half of window 0 is OOB and skipped.
    """
    window_start_input = window_hw * (input_window - 1)
    window_start_output = window_hw * (output_window - 1)
    window_size = 2 * window_hw + 1
    inv_hw = 1.0 / float(window_hw)
    for i in range(window_size):
        si = window_start_input + i
        so = window_start_output + i
        if so < 0 or so >= output_signal.size or si < 0 or si >= input_signal.size:
            continue
        blend = (window_hw - abs(window_hw - i)) * inv_hw * scale
        output_signal[so] += blend * input_signal[si]


# --------------------------------------------------------------------------- #
# Top-level                                                                   #
# --------------------------------------------------------------------------- #

def synthesize(
    base: np.ndarray,
    training: np.ndarray,
    fs: float = 44100.0,
    num_levels: int = 6,
    window_hw: int = 4,
    feature_hw: int = 3,
    falloff: float = 0.0,
    scale_cdf: bool = True,
    scaling_alpha: float = 1.0,
    rng: DeterministicRng | None = None,
    return_diagnostics: bool = False,
    progress_callback=None,
    eps: float = 0.0,
) -> np.ndarray | dict[str, Any]:
    """Synthesise a fire-sound texture by stitching training-signal windows.

    Mirrors the structure of ``WindowSynthesizer::synthesizeSignal`` in
    ``srcOrig/texture/WindowSynthesizer.cpp``.

    Parameters
    ----------
    base
        Low-frequency, physically based base signal `p_S(t)`.
    training
        Real fire-audio recording `p_T(t)` (any duration).
    fs
        Sample rate. Stored for the diagnostics dict; the algorithm
        itself is sample-rate-agnostic.
    num_levels
        Pyramid depth (default 6, matching all bundled examples).
    window_hw
        Synthesis-window half-width in samples (h in the paper).
    feature_hw
        Feature context half-width (h_f in the paper).
    falloff
        Optional exponential weight on feature dimensions:
        ``exp(-falloff * |distance|)``. 0 = uniform.
    scale_cdf
        Enable the dynamic-range mapping of section 5.3.
    scaling_alpha
        Strength of the CDF scaling: ``(1 - alpha) + alpha * scaling``.
    rng
        Reserved for future tie-breaking; currently unused since
        ``cKDTree.query`` is deterministic.
    return_diagnostics
        If True, return a dict with intermediate signals and per-window
        scaling; otherwise return just the synthesised waveform.
    progress_callback
        Optional ``callable(level, fraction)``: called once per ~10
        windows during the synthesis loop.
    """
    base = np.asarray(base, dtype=np.float64).ravel()
    training = np.asarray(training, dtype=np.float64).ravel()

    training_pyr = GaussianPyramid(training, num_levels, reflect_boundaries=True)
    output_pyr = GaussianPyramid(base, num_levels, reflect_boundaries=False)

    if scale_cdf:
        training_pyr.init_cdf()
        output_pyr.init_cdf()

    window_hw_arr = [window_hw] * num_levels
    feature_hw_arr = [feature_hw] * num_levels

    # ---- Build the per-level training-feature dictionaries.
    # We synthesise levels [0, num_levels - 2]; the top level
    # (num_levels - 1) is preserved from the input. So we build trees
    # for the synthesised levels only.
    level_features: List[np.ndarray | None] = [None] * num_levels
    level_indices: List[np.ndarray | None] = [None] * num_levels
    level_trees: List[cKDTree | None] = [None] * num_levels

    for level in range(num_levels - 1):
        n_windows = training_pyr.num_windows(window_hw, level)
        feats = []
        idxs = []
        for w in range(n_windows):
            feature, all_inside, _ = training_pyr.compute_window_feature(
                window_hw_arr, feature_hw_arr, level, w, falloff
            )
            if all_inside:
                feats.append(feature)
                idxs.append(w)
        if not feats:
            continue
        feats = np.ascontiguousarray(np.stack(feats), dtype=np.float64)
        idxs = np.asarray(idxs, dtype=np.int64)
        level_features[level] = feats
        level_indices[level] = idxs
        level_trees[level] = cKDTree(feats)

    # ---- Initialise the output pyramid: zero everything except the top
    # level. (numBaseLevels = 1 in every bundled example.)
    num_base_levels = 1
    for level in range(num_levels - num_base_levels):
        output_pyr.zero_level(level)

    # ---- Coarse-to-fine synthesis.
    scaling_per_window: List[float] = []
    for level in range(num_levels - 1 - num_base_levels, -1, -1):
        tree = level_trees[level]
        idxs = level_indices[level]
        if tree is None or idxs is None:
            # No usable training features at this level; nothing to do.
            continue
        level_data = output_pyr.levels[level]
        training_data = training_pyr.levels[level]
        n_windows_out = output_pyr.num_windows(window_hw, level)

        for w in range(n_windows_out):
            if (
                progress_callback is not None
                and w % max(1, n_windows_out // 50) == 0
            ):
                progress_callback(level, w / n_windows_out)

            input_cdf = training_pyr.cdf if scale_cdf else None
            output_cdf = output_pyr.cdf if scale_cdf else None
            feature, _, scale = output_pyr.compute_window_feature(
                window_hw_arr,
                feature_hw_arr,
                level,
                w,
                falloff,
                input_cdf=input_cdf,
                output_cdf=output_cdf,
                scaling_alpha=scaling_alpha,
            )
            if math.isnan(scale) or math.isinf(scale):
                scale = 1.0
            scaling_per_window.append(scale if scale_cdf else 1.0)

            _, neighbour_idx_in_features = tree.query(feature, k=1, eps=eps)
            training_idx = int(idxs[int(neighbour_idx_in_features)])

            blend_window_data(
                training_data, training_idx, level_data, w, window_hw, scale=scale
            )

    if progress_callback is not None:
        progress_callback(0, 1.0)

    output = output_pyr.reconstruct_signal()

    if return_diagnostics:
        return {
            "output": output,
            "fs": fs,
            "output_pyramid": [lvl.copy() for lvl in output_pyr.levels],
            "training_pyramid": [lvl.copy() for lvl in training_pyr.levels],
            "input_cdf": (
                training_pyr.cdf.copy() if training_pyr.cdf is not None else None
            ),
            "output_cdf": (
                output_pyr.cdf.copy() if output_pyr.cdf is not None else None
            ),
            "scaling_per_window": np.asarray(scaling_per_window, dtype=np.float64),
        }
    return output


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _load_input(path: Path, fs_default: float) -> tuple[np.ndarray, float]:
    suffix = path.suffix.lower()
    if suffix == ".vector":
        return read_vector(path), fs_default
    if suffix == ".wav":
        from scipy.io import wavfile
        fs_in, raw = wavfile.read(path)
        data = np.asarray(raw)
        if data.ndim > 1:
            data = data[:, 0]
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            scale_factor = max(abs(info.min), info.max)
            data = data.astype(np.float64) / scale_factor
        else:
            data = data.astype(np.float64)
        return data, float(fs_in)
    raise ValueError(f"Unsupported input format: {suffix}")


def _save_output(p: np.ndarray, fs: float, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".vector":
        write_vector(p, path)
        return
    if suffix == ".wav":
        from scipy.io import wavfile
        peak = float(np.max(np.abs(p))) if p.size else 0.0
        if peak > 0.0:
            normalized = p / (peak * 1.01)
        else:
            normalized = p
        pcm16 = np.clip(normalized * 32767.0, -32768.0, 32767.0).astype(np.int16)
        wavfile.write(str(path), int(round(fs)), pcm16)
        return
    raise ValueError(f"Unsupported output format: {suffix}")


def _read_xml_config(path: Path) -> dict[str, Any]:
    """Parse a ``default.xml`` like the ones in srcOrig/work/{name}/."""
    tree = ET.parse(path)
    root = tree.getroot()
    return dict(root.attrib)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sound texture synthesis (Chadwick & James, SIGGRAPH 2011, "
            "section 5). Faithful port of "
            "srcOrig/texture/WindowSynthesizer.cpp."
        )
    )
    parser.add_argument("--input", type=Path, help="base signal (.wav or .vector)")
    parser.add_argument(
        "--training", type=Path, help="training signal (.wav or .vector)"
    )
    parser.add_argument("--output", type=Path, required=True, help="output (.wav or .vector)")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional XML config (mirrors srcOrig/work/<name>/default.xml).",
    )
    parser.add_argument(
        "--base-dir", type=Path, default=None,
        help="If --config is given and references basesignal/trainingsignal, "
             "resolve those paths relative to this directory.",
    )
    parser.add_argument("--fs", type=float, default=44100.0)
    parser.add_argument("--num-levels", type=int, default=6)
    parser.add_argument("--window-hw", type=int, default=4)
    parser.add_argument("--feature-hw", type=int, default=3)
    parser.add_argument("--falloff", type=float, default=0.0)
    parser.add_argument("--no-scale-cdf", action="store_true")
    parser.add_argument("--scaling-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    # Apply XML config first, then CLI flags override.
    fs = args.fs
    num_levels = args.num_levels
    window_hw = args.window_hw
    feature_hw = args.feature_hw
    falloff = args.falloff
    scale_cdf = not args.no_scale_cdf
    scaling_alpha = args.scaling_alpha
    input_path = args.input
    training_path = args.training

    if args.config is not None:
        cfg = _read_xml_config(args.config)
        base_dir = args.base_dir or args.config.parent
        if "basesignal" in cfg and input_path is None:
            input_path = base_dir / cfg["basesignal"]
        if "trainingsignal" in cfg and training_path is None:
            training_path = base_dir / cfg["trainingsignal"]
        if "Fs" in cfg:
            fs = float(cfg["Fs"])
        if "numLevels" in cfg:
            num_levels = int(cfg["numLevels"])
        if "windowHW" in cfg:
            window_hw = int(cfg["windowHW"])
        if "featureHW" in cfg:
            feature_hw = int(cfg["featureHW"])
        if "falloff" in cfg:
            falloff = float(cfg["falloff"])
        if "scaleCDF" in cfg:
            scale_cdf = bool(int(cfg["scaleCDF"]))
        if "scalingAlpha" in cfg:
            scaling_alpha = float(cfg["scalingAlpha"])

    if input_path is None or training_path is None:
        parser.error(
            "must provide --input and --training (or --config with basesignal/"
            "trainingsignal attributes)."
        )

    base, _ = _load_input(input_path, fs)
    training, _ = _load_input(training_path, fs)

    rng = DeterministicRng(args.seed)
    if not args.quiet:
        def cb(level, frac):
            sys.stdout.write(
                f"\rsynthesising level {level}: {100*frac:5.1f}% "
            )
            sys.stdout.flush()
    else:
        cb = None
    out = synthesize(
        base, training,
        fs=fs, num_levels=num_levels,
        window_hw=window_hw, feature_hw=feature_hw,
        falloff=falloff, scale_cdf=scale_cdf, scaling_alpha=scaling_alpha,
        rng=rng, progress_callback=cb,
    )
    if not args.quiet:
        sys.stdout.write("\n")
    _save_output(np.asarray(out), fs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
