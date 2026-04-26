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
"""Generate Tier 1 (kernel) and Tier 2 (full pipeline) golden fixtures.

Run from the project root:

    python python/tools/generate_goldens.py            # both tiers
    python python/tools/generate_goldens.py --tier 1   # just kernels
    python python/tools/generate_goldens.py --tier 2   # just full pipeline

Tier 1 goldens are cheap and regenerated on every run; Tier 2 goldens take
several minutes per signal because the Matlab-faithful path uses a
length-NFFT FFT per window. Existing files are overwritten.

The Python port is the source of truth: these fixtures pin the Python port
against regression and act as the parity target the JavaScript port must
match. See ``python/tests/README.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from bandwidth_extension import (  # noqa: E402
    build_blending_function_linear,
    build_blurring_function_gaussian,
    build_powerlaw_spectrum,
    build_window_function_linear,
    extend_signal,
    fit_dual_power_spectra,
    lowpass_filter,
)
from deterministic_rng import DeterministicRng  # noqa: E402
from vector_io import read_vector  # noqa: E402

ROOT = _HERE.parents[1]
GOLDEN_DIR = ROOT / "python" / "tests" / "golden"

# Tier 1 fixed inputs
TIER1_FS = 44100.0
TIER1_NFFT = 1024
TIER1_HALF_WIDTH = 500
TIER1_L = 1024
TIER1_PHASES_SEED = 42

# Tier 2 inputs
TIER2_PHASES_SEED = 42
TIER2_ALPHAS = (2.5, 3.0, 3.5)
TIER2_SIGNALS = ("burning_brick", "candle", "dragon", "flame_jet", "torch")


def _save_f64(arr: np.ndarray, path: Path) -> None:
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    path.write_bytes(arr.astype("<f8").tobytes())


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def generate_tier1() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    fs = TIER1_FS
    NFFT = TIER1_NFFT
    x = fs * np.linspace(0.0, 1.0, NFFT)

    # Triangular window
    w = build_window_function_linear(TIER1_HALF_WIDTH, TIER1_L)
    _save_f64(w, GOLDEN_DIR / f"window_{TIER1_HALF_WIDTH}_L{TIER1_L}.f64")

    # Blend filters
    b1, b2 = build_blending_function_linear(x, NFFT, 165.0, 195.0)
    _save_f64(b1, GOLDEN_DIR / "blend1.f64")
    _save_f64(b2, GOLDEN_DIR / "blend2.f64")

    # Gaussian weight (sigma = 30/3 = 10 Hz)
    g = build_blurring_function_gaussian(x, 180.0, 30.0, NFFT)
    _save_f64(g, GOLDEN_DIR / "gaussian.f64")

    # Powerlaw magnitudes (phases drawn from PCG32 seed 42 for reproducibility;
    # only magnitudes are pinned because phases are RNG-dependent).
    rng = DeterministicRng(TIER1_PHASES_SEED)
    Z = build_powerlaw_spectrum(-1.25, x, NFFT, rng=rng)
    _save_f64(np.abs(Z), GOLDEN_DIR / "powerlaw_mag.f64")

    # Lowpass filter on a fixed synthetic input
    L_lp = 4410  # 100 ms at 44.1 kHz
    t = np.arange(L_lp) / fs
    p_synth = (
        0.6 * np.sin(2.0 * np.pi * 100.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 800.0 * t)
        + 0.1 * np.sin(2.0 * np.pi * 4500.0 * t)
    )
    p_lp = lowpass_filter(p_synth, fs, 180.0)
    _save_f64(p_lp, GOLDEN_DIR / "lowpass.f64")

    # Beta quadratic on a hand-constructed input
    NFFT_b = 256
    x_b = fs * np.linspace(0.0, 1.0, NFFT_b)
    f_blur = build_blurring_function_gaussian(x_b, 180.0, 30.0, NFFT_b)
    rng_b = DeterministicRng(7)
    # Synthesize Y_S as the FFT of a damped sinusoid; Y_L is its lowpassed
    # spectrum, Y_N is a power-law random-phase spectrum.
    t_b = np.arange(NFFT_b) / fs
    p_b = np.exp(-30.0 * t_b) * np.sin(2.0 * np.pi * 120.0 * t_b)
    Y_S = np.fft.fft(p_b, n=NFFT_b)
    b1_b, b2_b = build_blending_function_linear(x_b, NFFT_b, 165.0, 195.0)
    Y_L = Y_S * b1_b
    Y_N = build_powerlaw_spectrum(-1.25, x_b, NFFT_b, rng=rng_b) * b2_b
    beta = fit_dual_power_spectra(x_b, Y_L, Y_N, Y_S, f_blur)

    metadata = {
        "tier": 1,
        "regenerated_with": "python",
        "fs": fs,
        "NFFT": NFFT,
        "half_width": TIER1_HALF_WIDTH,
        "L": TIER1_L,
        "phases_seed": TIER1_PHASES_SEED,
        "blend_start": 165.0,
        "blend_end": 195.0,
        "fit_center": 180.0,
        "fit_width": 30.0,
        "powerlaw_exponent": -1.25,
        "lowpass_input": "0.6 sin(2*pi*100 t) + 0.3 sin(2*pi*800 t) + 0.1 sin(2*pi*4500 t), L=4410",
        "lowpass_cutoff": 180.0,
        "beta_NFFT": NFFT_b,
        "beta_input": "Y_S = fft(exp(-30 t) sin(2 pi 120 t)); Y_L = Y_S * blend1; Y_N = powerlaw(-1.25, seed=7) * blend2",
        "beta_value": beta,
    }
    (GOLDEN_DIR / "tier1_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        "tier1: wrote window, blend1/2, gaussian, powerlaw_mag, lowpass, "
        f"and beta scalar = {beta:.12g}"
    )


def generate_tier2() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    src = ROOT / "srcOrig" / "work"
    if not src.is_dir():
        print(f"missing {src}", file=sys.stderr)
        return

    metadata = {
        "tier": 2,
        "regenerated_with": "python",
        "phases_seed": TIER2_PHASES_SEED,
        "fs": 44100.0,
        "alphas": list(TIER2_ALPHAS),
        "cases": [],
    }

    for name in TIER2_SIGNALS:
        vf = src / f"{name}.vector"
        if not vf.exists():
            print(f"skipping missing {vf}")
            continue
        p = read_vector(vf)
        L = p.size
        NFFT = _next_pow2(L)

        # Build phases from the shared PCG32 (matches what the JS port will do).
        rng = DeterministicRng(TIER2_PHASES_SEED)
        phases = 2.0 * np.pi * rng.random_array(NFFT)

        for alpha in TIER2_ALPHAS:
            t0 = time.time()
            diag = extend_signal(
                p,
                fs=44100.0,
                alpha=alpha,
                phases_override=phases,
                return_diagnostics=True,
            )
            dt = time.time() - t0
            ext = diag["extended"]
            betas = diag["betas"]
            tag = f"{name}_alpha{alpha}_seed{TIER2_PHASES_SEED}"
            out_path = GOLDEN_DIR / f"{tag}.npz"
            _save_npz(out_path, extended=ext, betas=betas)
            print(
                f"tier2: {tag}: L={L}, NFFT={NFFT}, "
                f"betas={betas.size}, took {dt:.1f}s, wrote {out_path.name}"
            )
            metadata["cases"].append(
                {
                    "name": name,
                    "alpha": alpha,
                    "L": int(L),
                    "NFFT": int(NFFT),
                    "n_betas": int(betas.size),
                    "elapsed_s": dt,
                    "file": out_path.name,
                }
            )
    (GOLDEN_DIR / "tier2_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2],
        default=None,
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
