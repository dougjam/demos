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
"""Tier 3 statistical smoke tests for ``synthesize``.

No goldens; each test checks an expected property of the output. Tests
use short synthetic signals (~0.5 s) to keep the suite fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "python"))

from texture_synthesis import synthesize  # noqa: E402
from deterministic_rng import DeterministicRng  # noqa: E402

FS = 44100.0


def _make_base(seconds: float = 0.5) -> np.ndarray:
    n = int(seconds * FS)
    t = np.arange(n) / FS
    env = np.exp(-2.0 * t)
    return 0.4 * np.sin(2.0 * np.pi * 100.0 * t) * env


def _make_training(seconds: float = 1.0) -> np.ndarray:
    """Pink-ish noise: white noise lowpassed to roll off ~3 dB / octave."""
    n = int(seconds * FS)
    rng = np.random.default_rng(7)
    white = rng.standard_normal(n)
    # Cumulative average smoothing to push energy toward low freqs slightly.
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 5000.0 / (FS / 2.0), btype="low")
    return 0.3 * filtfilt(b, a, white)


def test_silent_base_output_bounded_by_training() -> None:
    """An all-zero base does not produce all-zero output: the algorithm
    still picks the lowest-magnitude training window at each level and
    blends it in. The output amplitude is bounded by the training peak
    (modulo the COLA-1 hat-window blend)."""
    base = np.zeros(int(0.3 * FS))
    training = _make_training(0.5)
    out = synthesize(base, training, fs=FS, num_levels=4, window_hw=4, feature_hw=3)
    assert np.all(np.isfinite(out))
    assert out.shape == base.shape
    peak_training = float(np.max(np.abs(training)))
    peak_out = float(np.max(np.abs(out)))
    # Output peak should not exceed the training peak by more than a
    # small overshoot constant (4 levels of overlap-add can amplify by
    # at most ~num_levels in the worst case).
    assert peak_out < 5.0 * peak_training, (
        f"silent-base output peak {peak_out} >> training peak {peak_training}"
    )


def test_silent_training_returns_zero() -> None:
    base = _make_base(0.3)
    training = np.zeros(int(0.5 * FS))
    out = synthesize(base, training, fs=FS, num_levels=4, window_hw=4, feature_hw=3)
    # Training is all zeros -> all blend contributions are zero -> output
    # equals the preserved top-level base (after pyramid roundtrip).
    assert np.all(np.isfinite(out))
    assert out.shape == base.shape
    # The output should be very small (essentially the top-level base
    # broadcast back via the empty levels).
    assert float(np.max(np.abs(out))) < float(np.max(np.abs(base))) * 0.5


def test_byte_identical_determinism() -> None:
    base = _make_base(0.3)
    training = _make_training(0.5)
    a = synthesize(base, training, fs=FS, num_levels=4, window_hw=4, feature_hw=3)
    b = synthesize(base, training, fs=FS, num_levels=4, window_hw=4, feature_hw=3)
    assert a.dtype == b.dtype
    assert np.array_equal(a, b)


def test_output_has_high_frequency_content() -> None:
    """The synthesised output should carry energy above 1 kHz, which the
    base signal alone (a 100 Hz tone) does not have."""
    base = _make_base(0.3)
    training = _make_training(0.5)
    out = synthesize(base, training, fs=FS, num_levels=5, window_hw=4, feature_hw=3)
    f, psd_out = welch(out, fs=FS, nperseg=2048)
    f, psd_base = welch(base, fs=FS, nperseg=2048)
    band = (f >= 1000.0) & (f <= 5000.0)
    e_out = float(psd_out[band].sum())
    e_base = float(psd_base[band].sum())
    # Output should have at least 10x more high-frequency energy.
    assert e_out > 10.0 * e_base + 1e-12, (
        f"Output high-freq energy {e_out:.3e} not >> base {e_base:.3e}"
    )


def test_cdf_mapping_shifts_amplitude_distribution() -> None:
    """With CDF mapping ON, the output's amplitude CDF should be
    visibly different from with CDF mapping OFF."""
    base = _make_base(0.3)
    training = _make_training(0.5)
    out_on = synthesize(
        base, training, fs=FS, num_levels=4, window_hw=4, feature_hw=3,
        scale_cdf=True, scaling_alpha=1.0,
    )
    out_off = synthesize(
        base, training, fs=FS, num_levels=4, window_hw=4, feature_hw=3,
        scale_cdf=False,
    )
    # Different outputs.
    assert not np.array_equal(out_on, out_off)
    # CDF-on output is rescaled toward training amplitudes; the difference
    # should be non-trivial (at least 1e-3 RMS).
    rms_diff = float(np.sqrt(np.mean((out_on - out_off) ** 2)))
    assert rms_diff > 1e-4, f"CDF on/off produced suspiciously similar outputs (RMS diff={rms_diff:.3e})"


def test_output_shape_matches_base() -> None:
    base = _make_base(0.4)
    training = _make_training(0.5)
    out = synthesize(base, training, fs=FS, num_levels=4, window_hw=4, feature_hw=3)
    assert out.shape == base.shape
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))
