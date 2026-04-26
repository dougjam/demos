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
"""Spectral bandwidth extension (Algorithm 1 of Chadwick & James, SIGGRAPH 2011).

Faithful Python port of srcOrig/matlab/. Function names and structure mirror
the Matlab source so the diff is auditable. Indexing follows the spec rule
of mirroring the Matlab 1-based scheme internally and converting to 0-based
only at slice boundaries.

Public entry point: ``extend_signal``.

Spec notes (see SPEC_SpectralBandwidthExtension.md and docs/ALGORITHM.md):
 - Two-sided FFTs (``np.fft.fft`` / ``np.fft.ifft``).
 - ``NFFT = 2 ** ceil(log2(L))`` (Matlab ``nextpow2``).
 - Frequency axis ``x = fs * np.linspace(0, 1, NFFT)``.
 - Order-2 Butterworth + ``filtfilt`` lowpass.
 - Triangular window with peak 1, COLA-1 with halfWidth stride.
 - Power-law noise generated once with ``blend2`` pre-applied; reused per window.
 - Envelope multiplier is ``|p_lowpass| * w``, not ``|p * w|``.
 - ``Y_S`` in the beta quadratic is the unfiltered windowed signal.
 - Stable Vieta branch on sign of ``b``; pick the positive root.
 - Powerlaw magnitude uses ``x^exponent`` (matches the released code's
   resolution of the ``% FIXME`` comment in ``build_powerlaw_spectrum.m``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

from deterministic_rng import DeterministicRng
from vector_io import read_vector, write_vector


# --------------------------------------------------------------------------- #
# Kernels                                                                     #
# --------------------------------------------------------------------------- #

def lowpass_filter(p: np.ndarray, fs: float, f_cutoff: float) -> np.ndarray:
    """Order-2 Butterworth lowpass with zero-phase filtfilt. Mirrors lowpass_filter.m."""
    b, a = butter(2, f_cutoff / (fs / 2.0), btype="low")
    return filtfilt(b, a, p)


def build_window_function(window_func, L: int) -> np.ndarray:
    """Discretize an arbitrary window over indices [-2L, 2L]. Mirrors build_window_function.m."""
    sz = 4 * L + 1
    w = np.empty(sz, dtype=np.float64)
    center = 2 * L  # 0-indexed equivalent of Matlab's centerPoint = 2L+1
    for i in range(sz):
        w[i] = window_func(i - center)
    return w


def build_window_function_linear(half_width: int, L: int) -> np.ndarray:
    """Triangular window with peak 1 at the center, support [-halfWidth, halfWidth].

    Vectorized equivalent of build_window_function with a triangular kernel.
    Adjacent windows separated by halfWidth COLA to 1.
    """
    sz = 4 * L + 1
    offsets = np.arange(sz, dtype=np.int64) - (2 * L)
    w = np.zeros(sz, dtype=np.float64)
    left = (offsets >= -half_width) & (offsets <= 0)
    right = (offsets > 0) & (offsets <= half_width)
    w[left] = (offsets[left] + half_width) / float(half_width)
    w[right] = (half_width - offsets[right]) / float(half_width)
    return w


def build_blurring_function(x: np.ndarray, blur_func, NFFT: int) -> np.ndarray:
    """Discretize an arbitrary blurring function over the frequency axis."""
    out = np.empty(NFFT, dtype=np.float64)
    for i in range(NFFT):
        out[i] = blur_func(x[i])
    return out


def build_blurring_function_gaussian(
    x: np.ndarray, f_middle: float, f_width: float, NFFT: int
) -> np.ndarray:
    """Gaussian centered at f_middle with sigma = f_width/3. Mirrors build_blurring_function_gaussian.m."""
    sigma = f_width / 3.0
    return np.exp(-((x - f_middle) ** 2) / (2.0 * sigma * sigma)).astype(np.float64)


def build_blending_function_linear(
    x: np.ndarray, NFFT: int, f1: float, f2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linear ramp between f1 and f2. Returns (spectrum_weight, noise_weight).

    Mirrors build_blending_function_linear.m. ``spectrum_weight`` is 1 below f1
    and ramps linearly to 0 at f2; ``noise_weight`` is the complement.
    """
    if f2 == f1:
        b1 = (x <= f1).astype(np.float64)
        b2 = 1.0 - b1
        return b1, b2
    t = np.clip((x - f1) / (f2 - f1), 0.0, 1.0)
    b1 = 1.0 - t
    b2 = t.copy()
    return b1.astype(np.float64), b2.astype(np.float64)


def build_powerlaw_spectrum(
    exponent: float,
    x: np.ndarray,
    NFFT: int,
    rng: DeterministicRng | None = None,
    phases_override: np.ndarray | None = None,
) -> np.ndarray:
    """Random-phase power-law spectrum with magnitude ``|x|^exponent``.

    Mirrors build_powerlaw_spectrum.m. The Matlab ``rand()`` call is **not**
    invoked at bins where ``x == 0``; this port preserves that consumption
    pattern so that the shared PCG32 RNG advances identically in Python and
    JavaScript. ``phases_override`` (length NFFT) substitutes for the RNG and
    is consulted at the same bins (entries at zero-frequency bins are unused).

    The constant ``2*pi`` in the released Matlab's alternative magnitude
    formula ``(2*pi*x)^exponent`` would cancel out of the per-window beta
    solve; this port therefore uses ``x^exponent`` to match the released code.
    """
    Z = np.zeros(NFFT, dtype=np.complex128)
    if phases_override is not None:
        phases = np.asarray(phases_override, dtype=np.float64).ravel()
        if phases.size != NFFT:
            raise ValueError(
                f"phases_override length {phases.size} != NFFT {NFFT}"
            )
        for j in range(NFFT):
            xj = x[j]
            if xj == 0.0:
                continue
            Z[j] = (xj**exponent) * np.exp(1j * phases[j])
        return Z

    if rng is None:
        raise ValueError("Either rng or phases_override must be provided")
    for j in range(NFFT):
        xj = x[j]
        if xj == 0.0:
            continue
        Z[j] = (xj**exponent) * np.exp(1j * rng.random() * 2.0 * np.pi)
    return Z


def get_windowed_spectra(Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-bin multiplication. Mirrors get_windowed_spectra.m."""
    return Y * w


def fit_dual_power_spectra(
    x: np.ndarray,
    Y_L: np.ndarray,
    Y_N: np.ndarray,
    Y_S: np.ndarray,
    f: np.ndarray,
) -> float:
    """Solve a quadratic for the noise scaling beta.

    Mirrors fit_dual_power_spectra.m. The integrals are trapezoidal over the
    full two-sided frequency axis ``x``, weighted by ``f`` (typically the
    Gaussian fit window). The quadratic ``a beta^2 + b beta + c = 0`` uses the
    Vieta-branched stable form to avoid catastrophic cancellation when ``b``
    has the same sign as ``sqrt(disc)``. When ``c < 0`` the two roots have
    opposite sign and ``max(beta1, beta2)`` selects the positive one.
    """
    L = Y_L.size
    if Y_N.size != L or Y_S.size != L or f.size != L:
        raise ValueError(
            f"Size mismatch: Y_L={L}, Y_N={Y_N.size}, Y_S={Y_S.size}, f={f.size}"
        )
    x = np.asarray(x, dtype=np.float64).ravel()

    X1 = x[:-1]
    X2 = x[1:]
    f1 = f[:-1]
    f2 = f[1:]

    Y_L_r_1, Y_L_r_2 = Y_L[:-1].real, Y_L[1:].real
    Y_L_i_1, Y_L_i_2 = Y_L[:-1].imag, Y_L[1:].imag
    Y_N_r_1, Y_N_r_2 = Y_N[:-1].real, Y_N[1:].real
    Y_N_i_1, Y_N_i_2 = Y_N[:-1].imag, Y_N[1:].imag
    Y_S_s_1 = (Y_S[:-1] * np.conj(Y_S[:-1])).real
    Y_S_s_2 = (Y_S[1:] * np.conj(Y_S[1:])).real

    z1_1, z1_2 = Y_L_r_1 * Y_L_r_1 * f1, Y_L_r_2 * Y_L_r_2 * f2
    z2_1, z2_2 = Y_N_r_1 * Y_N_r_1 * f1, Y_N_r_2 * Y_N_r_2 * f2
    z3_1, z3_2 = 2.0 * Y_L_r_1 * Y_N_r_1 * f1, 2.0 * Y_L_r_2 * Y_N_r_2 * f2
    z4_1, z4_2 = Y_L_i_1 * Y_L_i_1 * f1, Y_L_i_2 * Y_L_i_2 * f2
    z5_1, z5_2 = Y_N_i_1 * Y_N_i_1 * f1, Y_N_i_2 * Y_N_i_2 * f2
    z6_1, z6_2 = 2.0 * Y_L_i_1 * Y_N_i_1 * f1, 2.0 * Y_L_i_2 * Y_N_i_2 * f2
    zs_1, zs_2 = Y_S_s_1 * f1, Y_S_s_2 * f2

    def trap(za: np.ndarray, zb: np.ndarray) -> float:
        return 0.5 * float(
            np.dot(X2, za) - np.dot(X1, zb) + np.dot(X2, zb) - np.dot(X1, za)
        )

    I1 = trap(z1_1, z1_2)
    I2 = trap(z2_1, z2_2)
    I3 = trap(z3_1, z3_2)
    I4 = trap(z4_1, z4_2)
    I5 = trap(z5_1, z5_2)
    I6 = trap(z6_1, z6_2)
    Is = trap(zs_1, zs_2)

    a = I2 + I5
    b = I3 + I6
    c = I1 + I4 - Is

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        raise ValueError("Negative discriminant: complex roots found!")

    # Degenerate cases that arise when the noise spectrum is identically
    # zero (a == 0; e.g., the COLA test with blend2 = 0 everywhere) or when
    # the windowed input is silent (Y_S = Y_L = 0 makes b = 0 and c = 0,
    # which makes beta1 = 0 and would divide by zero in the Vieta step).
    # The Matlab reference does not handle these cases; this port returns
    # beta = 0 in both, which is the consistent choice (no noise to add).
    if a == 0.0:
        if b == 0.0:
            return 0.0
        return float(-c / b)

    sd = float(np.sqrt(disc))
    if b > 0.0:
        beta1 = (-b - sd) / (2.0 * a)
    else:
        beta1 = (-b + sd) / (2.0 * a)
    if beta1 == 0.0:
        return 0.0  # Vieta: beta1 * beta2 = c/a = 0
    beta2 = c / (beta1 * a)

    if beta1 > 0 and beta2 > 0:
        raise ValueError("Two positive alphas found")
    if beta1 < 0 and beta2 < 0:
        raise ValueError("Two negative alphas found")
    return float(max(beta1, beta2))


def extend_sub_signal_noise_source(
    psub: np.ndarray,
    pnoise: np.ndarray,
    NFFT: int,
    x: np.ndarray,
    f_blur: np.ndarray,
    blend1: np.ndarray,
    blend2: np.ndarray,
    noise_amplitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One-window blend of signal spectrum and envelope-shaped noise spectrum.

    Mirrors extend_sub_signal_noise_source.m. Returns
    ``(psub_out, p_noise_scaled, p_filtered, beta)`` where ``psub_out`` is the
    real time-domain blend (length L = psub.size) and ``p_noise_scaled`` and
    ``p_filtered`` are the complex per-window IFFT outputs (also length L)
    used for diagnostics, matching the Matlab return convention.
    """
    Y = np.fft.fft(psub, n=NFFT)
    Ynoise = np.fft.fft(pnoise, n=NFFT)
    L = psub.size

    spectrum_signal = get_windowed_spectra(Y, blend1)
    spectrum_noise = get_windowed_spectra(Ynoise, blend2)
    beta = fit_dual_power_spectra(x, spectrum_signal, spectrum_noise, Y, f_blur)

    p_extended = np.fft.ifft(
        spectrum_signal + beta * noise_amplitude * spectrum_noise, n=NFFT
    )
    p_noise_scaled = np.fft.ifft(beta * noise_amplitude * spectrum_noise, n=NFFT)
    p_filtered = np.fft.ifft(spectrum_signal, n=NFFT)

    return np.real(p_extended[:L]).astype(np.float64), p_noise_scaled[:L], p_filtered[:L], beta


# --------------------------------------------------------------------------- #
# Top-level                                                                   #
# --------------------------------------------------------------------------- #

def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def extend_signal(
    p: np.ndarray,
    fs: float = 44100.0,
    f_cutoff: float = 180.0,
    half_width: int = 500,
    fit_center: float = 180.0,
    fit_width: float = 30.0,
    blend_start: float = 165.0,
    blend_end: float = 195.0,
    noise_amplitude: float = 1.0,
    alpha: float = 2.5,
    rng: DeterministicRng | None = None,
    phases_override: np.ndarray | None = None,
    return_diagnostics: bool = False,
    nfft_per_window: int | None = None,
) -> np.ndarray | dict[str, Any]:
    """Spectrally bandwidth-extend a low-frequency signal. Mirrors extend_signal.m.

    Parameters mirror the Matlab call signature. With ``return_diagnostics=True``
    returns a dict with keys ``'extended', 'lowpass', 'noise_full', 'filtered',
    'betas'`` matching Matlab's ``[p_extended, p_lowpass, p_noiseFull,
    p_filtered, alphas]``. ``noise_full`` and ``filtered`` are complex
    (matching Matlab's auto-promoted output); call ``np.real`` if needed.

    Defaults reproduce the README example call.

    ``rng`` (a :class:`DeterministicRng`) supplies the random phases. To
    disable the RNG entirely and supply phases directly (used by Tier 2
    phase-controlled tests), set ``phases_override`` to a length-NFFT
    float64 array of phases in radians.

    ``nfft_per_window`` is reserved for the JavaScript demo's interactive
    speed path. The Python port currently raises ``NotImplementedError`` if
    a value other than ``None`` is supplied; see ``docs/ALGORITHM.md``.
    """
    if nfft_per_window is not None:
        raise NotImplementedError(
            "Small per-window FFT path is implemented in the JavaScript port "
            "only. The Python port runs Matlab-faithful at full NFFT."
        )

    p = np.asarray(p, dtype=np.float64).ravel()
    L = p.size
    if L == 0:
        empty = np.zeros(0, dtype=np.float64)
        if return_diagnostics:
            return {
                "extended": empty, "lowpass": empty,
                "noise_full": empty.astype(np.complex128),
                "filtered": empty.astype(np.complex128),
                "betas": np.zeros(0, dtype=np.float64),
            }
        return empty

    NFFT = _next_pow2(L)
    x = fs * np.linspace(0.0, 1.0, NFFT)

    powerlaw_exponent = -0.5 * alpha

    p_lowpass = lowpass_filter(p, fs, f_cutoff)
    abs_p_lowpass = np.abs(p_lowpass)

    f_blur = build_blurring_function_gaussian(x, fit_center, fit_width, NFFT)
    blend1, blend2 = build_blending_function_linear(x, NFFT, blend_start, blend_end)
    powerlaw = build_powerlaw_spectrum(
        powerlaw_exponent, x, NFFT, rng=rng, phases_override=phases_override
    )
    powerlaw = powerlaw * blend2

    window_function = build_window_function_linear(half_width, L)
    noise_unscaled = np.fft.ifft(powerlaw, n=NFFT)

    p_extended = np.zeros(L, dtype=np.float64)
    p_noise_full = np.zeros(L, dtype=np.complex128)
    p_filtered = np.zeros(L, dtype=np.complex128)
    betas: list[float] = []

    # Mirror Matlab indexing (1-based variables, 0-based slices).
    signal_center = 1
    signal_start = signal_center - half_width
    signal_end = signal_center + half_width
    window_start = 2 * L + 1
    window_end = 3 * L

    psub_lowpass_buf = np.zeros(NFFT, dtype=np.complex128)

    while signal_start <= L:
        w_slice = window_function[window_start - 1 : window_end]  # length L
        psub = p * w_slice

        psub_lowpass_buf.fill(0.0)
        psub_lowpass_buf[:L] = abs_p_lowpass * w_slice
        psub_lowpass_buf *= noise_unscaled

        psub_out, p_noise_scaled, p_filtered_window, beta = (
            extend_sub_signal_noise_source(
                psub, psub_lowpass_buf, NFFT, x, f_blur, blend1, blend2, noise_amplitude
            )
        )
        betas.append(beta)
        p_extended += psub_out
        p_noise_full += p_noise_scaled
        p_filtered += p_filtered_window

        signal_center += half_width
        signal_start = signal_center - half_width
        signal_end = signal_center + half_width
        window_start -= half_width
        window_end -= half_width

    if return_diagnostics:
        return {
            "extended": p_extended,
            "lowpass": p_lowpass,
            "noise_full": p_noise_full,
            "filtered": p_filtered,
            "betas": np.asarray(betas, dtype=np.float64),
        }
    return p_extended


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
            scale = max(abs(info.min), info.max)
            data = data.astype(np.float64) / scale
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Spectral bandwidth extension (Chadwick & James, SIGGRAPH 2011). "
            "Faithful port of srcOrig/matlab/extend_signal.m."
        )
    )
    parser.add_argument("input", type=Path, help="input .wav or .vector")
    parser.add_argument("output", type=Path, help="output .wav or .vector")
    parser.add_argument("--alpha", type=float, default=2.5, help="power-law exponent")
    parser.add_argument("--seed", type=int, default=42, help="PCG32 RNG seed")
    parser.add_argument("--fs", type=float, default=44100.0, help="sample rate (Hz) for .vector input")
    parser.add_argument("--cutoff", type=float, default=180.0, help="lowpass cutoff (Hz)")
    parser.add_argument("--half-width", type=int, default=500, help="window half-width (samples)")
    parser.add_argument("--fit-center", type=float, default=180.0, help="Gaussian fit center (Hz)")
    parser.add_argument("--fit-width", type=float, default=30.0, help="Gaussian fit width (Hz; sigma = width/3)")
    parser.add_argument("--blend-start", type=float, default=165.0, help="blend low-frequency edge (Hz)")
    parser.add_argument("--blend-end", type=float, default=195.0, help="blend high-frequency edge (Hz)")
    parser.add_argument("--noise-amplitude", type=float, default=1.0, help="extra scaling on noise contribution")
    args = parser.parse_args(argv)

    p, fs = _load_input(args.input, args.fs)
    rng = DeterministicRng(args.seed)
    extended = extend_signal(
        p,
        fs=fs,
        f_cutoff=args.cutoff,
        half_width=args.half_width,
        fit_center=args.fit_center,
        fit_width=args.fit_width,
        blend_start=args.blend_start,
        blend_end=args.blend_end,
        noise_amplitude=args.noise_amplitude,
        alpha=args.alpha,
        rng=rng,
    )
    _save_output(np.asarray(extended), fs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
