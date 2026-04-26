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
"""Tier 3 statistical smoke tests for ``extend_signal``.

No Matlab/Octave dependency. Each test checks an expected output property
of the Python port using shorter signals (~0.5 s) so the suite stays fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, filtfilt, hilbert, welch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "python"))

from bandwidth_extension import extend_signal  # noqa: E402
from deterministic_rng import DeterministicRng  # noqa: E402

FS = 44100.0


def _disable_blends() -> dict:
    """Parameter overrides that pin blend1=1, blend2=0 across the whole frequency axis.

    The Matlab/Python frequency axis runs ``[0, fs]`` (two-sided). Setting
    ``blend_start = blend_end >= fs`` makes ``blend1`` identically 1 and
    ``blend2`` identically 0 everywhere, so the noise contribution vanishes
    and the per-window output is just ``ifft(blend1 * Y) = ifft(Y) = psub``.
    """
    return dict(blend_start=FS, blend_end=FS)


def test_cola_identity_when_noise_disabled() -> None:
    """With blend2 forced to zero, COLA-1 reconstruction returns the input."""
    rng = DeterministicRng(42)
    L = int(0.2 * FS)
    t = np.arange(L) / FS
    p = (
        0.5 * np.sin(2.0 * np.pi * 80.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 130.0 * t)
    )
    out = extend_signal(p, fs=FS, alpha=2.5, rng=rng, **_disable_blends())
    rms = float(np.sqrt(np.mean(p**2)))
    err = float(np.sqrt(np.mean((out - p) ** 2)))
    assert err / rms < 1e-10, f"COLA error {err:.3e} exceeds tolerance (rms={rms:.3e})"


def test_silent_input_returns_zero() -> None:
    rng = DeterministicRng(42)
    L = int(0.1 * FS)
    p = np.zeros(L)
    out = extend_signal(p, fs=FS, alpha=2.5, rng=rng)
    assert np.array_equal(out, np.zeros_like(p))


def test_sub_cutoff_sine_preserved() -> None:
    """A 100 Hz tone (below the 180 Hz cutoff) should be preserved at half
    amplitude (-6.02 dB) within 0.5 dB.

    The Matlab reference applies a one-sided ``blend1`` lowpass to the
    two-sided FFT, which zeros the negative-frequency mirror. After the final
    ``real(ifft(...))`` step this halves the in-band magnitude. The output is
    then peak-normalized for playback, so the perceptual effect is invisible
    in the demo. See ``docs/ALGORITHM.md``.
    """
    rng = DeterministicRng(42)
    L = int(1.0 * FS)
    t = np.arange(L) / FS
    f0 = 100.0
    p = np.sin(2.0 * np.pi * f0 * t)
    out = extend_signal(p, fs=FS, alpha=2.5, rng=rng)

    freqs = np.fft.rfftfreq(L, d=1.0 / FS)
    P_in = np.abs(np.fft.rfft(p))
    P_out = np.abs(np.fft.rfft(out))
    bin_in = int(np.argmin(np.abs(freqs - f0)))
    db_in = 20.0 * np.log10(P_in[bin_in])
    db_out = 20.0 * np.log10(P_out[bin_in])
    expected_loss_db = -6.0206  # 20*log10(0.5)
    assert abs((db_out - db_in) - expected_loss_db) < 0.5, (
        f"sub-cutoff tone shifted by {db_out - db_in:.3f} dB "
        f"(expected approximately {expected_loss_db:.3f} dB)"
    )


@pytest.mark.parametrize("alpha", [2.0, 2.5, 3.0, 3.5])
def test_powerlaw_slope_above_cutoff(alpha: float) -> None:
    """Welch PSD log-log slope of the extended output between 500 Hz and 10 kHz
    should match the requested ``-alpha`` within 0.3."""
    rng_input = np.random.default_rng(7)
    L = int(0.5 * FS)
    p_white = rng_input.standard_normal(L)
    b, a = butter(4, 150.0 / (FS / 2.0), btype="low")
    p = filtfilt(b, a, p_white)
    p /= np.max(np.abs(p))

    rng = DeterministicRng(42)
    out = extend_signal(p, fs=FS, alpha=alpha, rng=rng)
    f, psd = welch(out, fs=FS, nperseg=4096)
    band = (f >= 500.0) & (f <= 10000.0) & (psd > 0)
    log_f = np.log10(f[band])
    log_p = np.log10(psd[band])
    slope, _ = np.polyfit(log_f, log_p, 1)
    assert abs(slope - (-alpha)) < 0.3, (
        f"alpha={alpha}: PSD slope {slope:.3f} (expected ~{-alpha})"
    )


def test_envelope_synchronization() -> None:
    """High-frequency Hilbert envelope of the output should track the input on/off envelope."""
    rng = DeterministicRng(42)
    L = int(1.0 * FS)
    t = np.arange(L) / FS
    # On-off square envelope at 1 Hz, smoothed slightly to avoid bandwidth issues
    env = (np.sin(2.0 * np.pi * 1.0 * t) > 0).astype(np.float64)
    b_env, a_env = butter(2, 20.0 / (FS / 2.0), btype="low")
    env_smooth = filtfilt(b_env, a_env, env)
    carrier = np.sin(2.0 * np.pi * 60.0 * t)
    p = env_smooth * carrier

    out = extend_signal(p, fs=FS, alpha=2.5, rng=rng)

    # High-pass the output above 500 Hz, take Hilbert magnitude envelope
    bh, ah = butter(4, 500.0 / (FS / 2.0), btype="high")
    out_hp = filtfilt(bh, ah, out)
    out_env = np.abs(hilbert(out_hp))
    # Smooth output envelope to compare on the slow timescale
    out_env_smooth = filtfilt(b_env, a_env, out_env)

    # Drop the first/last 50 ms to avoid filter edge effects
    margin = int(0.05 * FS)
    a_window = env_smooth[margin:-margin]
    b_window = out_env_smooth[margin:-margin]
    a_window = a_window - a_window.mean()
    b_window = b_window - b_window.mean()
    corr = float(
        (a_window @ b_window)
        / (np.linalg.norm(a_window) * np.linalg.norm(b_window) + 1e-30)
    )
    assert corr > 0.7, f"envelope sync correlation {corr:.3f} below 0.7"


def test_byte_identical_determinism() -> None:
    L = int(0.2 * FS)
    t = np.arange(L) / FS
    p = 0.4 * np.sin(2.0 * np.pi * 90.0 * t) * np.exp(-2.0 * t)
    a = extend_signal(p, fs=FS, alpha=2.5, rng=DeterministicRng(42))
    b = extend_signal(p, fs=FS, alpha=2.5, rng=DeterministicRng(42))
    assert a.dtype == b.dtype
    assert np.array_equal(a, b)
