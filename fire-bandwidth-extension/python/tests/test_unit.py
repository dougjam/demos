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
"""Tier 1 unit tests: deterministic kernels.

Each test loads a binary golden (``.f64`` little-endian Float64) produced by
``python/tools/generate_goldens.py`` and re-runs the kernel from the Python
port. The tolerance ``atol=rtol=1e-13`` pins the kernel to its current
behavior; any unintended numerical change will trip the test.

The goldens are produced by the Python port (the canonical implementation
in this repo). See ``python/tests/README.md`` for the rationale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "python"))

from bandwidth_extension import (  # noqa: E402
    build_blending_function_linear,
    build_blurring_function_gaussian,
    build_powerlaw_spectrum,
    build_window_function_linear,
    fit_dual_power_spectra,
    lowpass_filter,
)
from deterministic_rng import DeterministicRng  # noqa: E402

GOLDEN = _HERE / "golden"
TOL = dict(atol=1e-13, rtol=1e-13)


def _load_f64(name: str) -> np.ndarray:
    return np.frombuffer((GOLDEN / name).read_bytes(), dtype="<f8")


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads((GOLDEN / "tier1_metadata.json").read_text())


def test_window_triangular(metadata: dict) -> None:
    half_width = metadata["half_width"]
    L = metadata["L"]
    w = build_window_function_linear(half_width, L)
    expected = _load_f64(f"window_{half_width}_L{L}.f64")
    assert w.shape == expected.shape
    assert np.allclose(w, expected, **TOL)


def test_blend_filters(metadata: dict) -> None:
    fs, NFFT = metadata["fs"], metadata["NFFT"]
    x = fs * np.linspace(0.0, 1.0, NFFT)
    b1, b2 = build_blending_function_linear(
        x, NFFT, metadata["blend_start"], metadata["blend_end"]
    )
    assert np.allclose(b1, _load_f64("blend1.f64"), **TOL)
    assert np.allclose(b2, _load_f64("blend2.f64"), **TOL)
    # Sanity: spectrum and noise weights complement to 1 everywhere
    assert np.allclose(b1 + b2, 1.0, atol=1e-15)


def test_gaussian_weight(metadata: dict) -> None:
    fs, NFFT = metadata["fs"], metadata["NFFT"]
    x = fs * np.linspace(0.0, 1.0, NFFT)
    g = build_blurring_function_gaussian(
        x, metadata["fit_center"], metadata["fit_width"], NFFT
    )
    assert np.allclose(g, _load_f64("gaussian.f64"), **TOL)


def test_powerlaw_magnitudes(metadata: dict) -> None:
    fs, NFFT = metadata["fs"], metadata["NFFT"]
    x = fs * np.linspace(0.0, 1.0, NFFT)
    rng = DeterministicRng(metadata["phases_seed"])
    Z = build_powerlaw_spectrum(metadata["powerlaw_exponent"], x, NFFT, rng=rng)
    assert np.allclose(np.abs(Z), _load_f64("powerlaw_mag.f64"), **TOL)


def test_lowpass_filter() -> None:
    fs = 44100.0
    L = 4410
    t = np.arange(L) / fs
    p = (
        0.6 * np.sin(2.0 * np.pi * 100.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 800.0 * t)
        + 0.1 * np.sin(2.0 * np.pi * 4500.0 * t)
    )
    out = lowpass_filter(p, fs, 180.0)
    assert np.allclose(out, _load_f64("lowpass.f64"), **TOL)


def test_beta_quadratic(metadata: dict) -> None:
    fs = metadata["fs"]
    NFFT_b = metadata["beta_NFFT"]
    x_b = fs * np.linspace(0.0, 1.0, NFFT_b)
    f_blur = build_blurring_function_gaussian(x_b, 180.0, 30.0, NFFT_b)
    rng_b = DeterministicRng(7)
    t_b = np.arange(NFFT_b) / fs
    p_b = np.exp(-30.0 * t_b) * np.sin(2.0 * np.pi * 120.0 * t_b)
    Y_S = np.fft.fft(p_b, n=NFFT_b)
    b1_b, b2_b = build_blending_function_linear(x_b, NFFT_b, 165.0, 195.0)
    Y_L = Y_S * b1_b
    Y_N = build_powerlaw_spectrum(-1.25, x_b, NFFT_b, rng=rng_b) * b2_b
    beta = fit_dual_power_spectra(x_b, Y_L, Y_N, Y_S, f_blur)
    assert beta == pytest.approx(metadata["beta_value"], rel=1e-13, abs=1e-13)


def test_rng_first_outputs() -> None:
    """Pin the first 8 PCG32 outputs at seed 42; the JS port must match these."""
    rng = DeterministicRng(42)
    expected = [
        0.4421448060311377,
        0.23623736714944243,
        0.9536763080395758,
        0.14759166655130684,
        0.2652577902190387,
        0.56567323487252,
        0.3210757712367922,
        0.3276327084749937,
    ]
    actual = [rng.random() for _ in range(len(expected))]
    assert actual == pytest.approx(expected, rel=0, abs=0)


def test_rng_determinism() -> None:
    """Same seed reproduces the exact same stream."""
    a = [DeterministicRng(123).random() for _ in range(50)]
    b = [DeterministicRng(123).random() for _ in range(50)]
    assert a == b
    c = [DeterministicRng(124).random() for _ in range(50)]
    assert a != c


def test_rng_uniform_range() -> None:
    """Outputs are in [0, 1)."""
    rng = DeterministicRng(0)
    for _ in range(2000):
        v = rng.random()
        assert 0.0 <= v < 1.0
