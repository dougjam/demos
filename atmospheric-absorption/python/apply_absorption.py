"""
Apply atmospheric absorption to an audio file (frequency-domain filter).

Given an input WAV at sample rate Fs, this script:
  1. Reads the audio, converts to mono float32 in [-1, 1].
  2. Computes the ISO 9613-1 absorption coefficient alpha(f) for the
     selected (T, RH, p), evaluated at every FFT bin frequency.
  3. Builds the linear-amplitude transfer function
                 H(f) = 10 ** (-alpha(f) * r / 20)
     (the linear-amplitude equivalent of the dB attenuation A = alpha * r).
  4. Multiplies the signal's spectrum by H(f) and inverse-FFTs.
  5. Optionally applies 1/r geometric spreading (off by default).
  6. Writes a mono 16-bit PCM WAV at the original sample rate.

Because alpha(f) is real, positive, and smooth, the filter H is real and
positive at every bin. The corresponding impulse response is therefore
real and zero-phase (symmetric about t=0). This is the cleanest way to
study the *magnitude* shape of atmospheric absorption; absolute time-of-
flight is not modelled.

Usage examples:

    # Defaults: 20 C, 50% RH, 1 atm, r = 1 km, no 1/r spreading.
    python apply_absorption.py in.wav out.wav

    # Distant thunder at 5 km, hot and humid:
    python apply_absorption.py thunder.wav far.wav \\
        --temperature 30 --humidity 80 --distance 5000

    # Include 1/r spreading and use the SPEC default conditions:
    python apply_absorption.py source.wav out.wav --distance 200 --spread

Dependencies: numpy, scipy. Both are standard scientific-Python packages.

The actual ISO 9613-1 formulas live in iso9613_reference.py; this script
just provides a numpy-vectorized alpha (so the per-bin evaluation is
fast for typical audio lengths) and asserts that it agrees with the
reference implementation at the boundary frequencies before running.

(c) 2026 Doug James, Stanford University. BSD-2-Clause.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# The pure-Python reference (scalar). We use it only to sanity-check our
# vectorized copy at a few frequencies, so a typo in either copy would be
# caught loudly on every run.
from iso9613_reference import (
    alpha_dB_per_m as alpha_scalar,
    T0, T01, P_REF,
)


# ---------------------------------------------------------------------------
# Vectorized alpha (same physics as iso9613_reference.alpha_dB_per_m, but
# accepts a numpy array of frequencies).
# ---------------------------------------------------------------------------
def alpha_dB_per_m_vec(f: np.ndarray, T_C: float, RH_pct: float, p_kPa: float) -> np.ndarray:
    """ISO 9613-1 absorption coefficient in dB/m, vectorized over `f`."""
    T = T_C + 273.15
    tr = T / T0           # T / T_ref
    pr = p_kPa / P_REF    # p / p_ref
    # Saturation vapour pressure / p_ref (Davis 1992):
    psat_pref = 10.0 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    # Molar concentration of water vapour, in percent:
    h = RH_pct * psat_pref / pr
    # Vibrational relaxation frequencies (Hz):
    fO = pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fN = pr * tr ** -0.5 * (
        9.0 + 280.0 * h * np.exp(-4.170 * (tr ** (-1.0 / 3.0) - 1.0))
    )
    # Three absorption mechanisms summed:
    classical = 1.84e-11 * (P_REF / p_kPa) * np.sqrt(tr)
    A_O = 0.01275 * np.exp(-2239.1 / T) * fO / (fO * fO + f * f)
    A_N = 0.1068  * np.exp(-3352.0 / T) * fN / (fN * fN + f * f)
    return 8.686 * f * f * (classical + tr ** -2.5 * (A_O + A_N))


def _self_check(T_C: float, RH_pct: float, p_kPa: float) -> None:
    """Cross-check vectorized alpha against the scalar reference at a few f."""
    for f in (100.0, 1_000.0, 10_000.0):
        scalar = alpha_scalar(f, T_C, RH_pct, p_kPa)
        vec = float(alpha_dB_per_m_vec(np.array([f]), T_C, RH_pct, p_kPa)[0])
        rel = abs(vec - scalar) / max(abs(scalar), 1e-30)
        if rel > 1e-12:
            raise RuntimeError(
                f"alpha_dB_per_m_vec disagrees with iso9613_reference at f={f}: "
                f"vec={vec}, scalar={scalar}, rel err {rel:.2e}"
            )


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------
def load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    """Read a WAV, convert to mono float32 in [-1, 1]. Supports the common
    integer PCM formats (8/16/24/32-bit) and float32."""
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        x = data.astype(np.float32)
    else:
        raise RuntimeError(f"unsupported WAV sample dtype: {data.dtype}")
    if x.ndim == 2:
        x = x.mean(axis=1)  # downmix to mono
    return sr, x


def write_wav_mono_16bit(path: Path, sr: int, x: np.ndarray) -> None:
    """Write mono 16-bit PCM. Clips into [-1, 1] before quantizing."""
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, pcm)


# ---------------------------------------------------------------------------
# The filter itself
# ---------------------------------------------------------------------------
def apply_absorption(
    x: np.ndarray,
    sr: int,
    T_C: float,
    RH_pct: float,
    p_kPa: float,
    r_m: float,
    include_spreading: bool = False,
) -> np.ndarray:
    """Return x filtered by atmospheric absorption at (T, RH, p) over r metres.

    Implementation: one-shot real FFT (rfft) of the entire signal. For audio
    files of up to a few minutes at 48 kHz this is the cleanest and fastest
    approach. For very long inputs (hours), switch to an STFT / overlap-add
    pipeline; the math is identical, only the buffering changes.
    """
    n = len(x)
    # We pad to the next power of two for FFT speed and to avoid circular
    # wrap-around (the filter's effective impulse response has support of
    # roughly 1 / smallest spectral feature, so a power-of-two pad beyond
    # the signal length is plenty in practice).
    n_fft = 1
    while n_fft < 2 * n:
        n_fft <<= 1
    # Bin frequencies for rfft of length n_fft:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    # alpha(f) in dB/m at every bin:
    alpha = alpha_dB_per_m_vec(freqs, T_C, RH_pct, p_kPa)
    # Linear-amplitude transfer function:  H(f) = 10 ** (-alpha * r / 20)
    H = np.power(10.0, -alpha * r_m / 20.0)
    # Filter in the frequency domain (real, positive H -> zero-phase filter):
    X = np.fft.rfft(x, n=n_fft)
    Y = X * H
    y = np.fft.irfft(Y, n=n_fft)[:n]
    if include_spreading:
        y *= 1.0 / max(1.0, r_m)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Apply ISO 9613-1 atmospheric absorption to a WAV file."
    )
    p.add_argument("input", type=Path, help="input WAV (mono or stereo)")
    p.add_argument("output", type=Path, help="output mono 16-bit WAV")
    p.add_argument("-T", "--temperature", type=float, default=20.0,
                   help="air temperature in degrees Celsius (default: 20)")
    p.add_argument("-H", "--humidity", type=float, default=50.0,
                   help="relative humidity in percent (default: 50)")
    p.add_argument("-P", "--pressure", type=float, default=101.325,
                   help="atmospheric pressure in kPa (default: 101.325)")
    p.add_argument("-r", "--distance", type=float, default=1000.0,
                   help="propagation distance in metres (default: 1000)")
    p.add_argument("--spread", action="store_true",
                   help="also apply 1/r geometric spreading (off by default; "
                        "the filter alone models absorption only)")
    args = p.parse_args(argv)

    _self_check(args.temperature, args.humidity, args.pressure)

    sr, x = load_wav_mono(args.input)
    print(f"loaded {args.input.name}: {len(x)} samples @ {sr} Hz "
          f"({len(x) / sr:.2f} s mono)")
    # Report what the filter is going to do at a few audible bands.
    for f in (250, 1000, 4000, 10000):
        a = float(alpha_dB_per_m_vec(np.array([f]), args.temperature,
                                     args.humidity, args.pressure)[0])
        att = a * args.distance
        print(f"  f = {f:>6d} Hz:  alpha = {a*1000:>7.2f} dB/km, "
              f"A(f, r={args.distance:g} m) = {att:>7.2f} dB")
    y = apply_absorption(x, sr, args.temperature, args.humidity,
                         args.pressure, args.distance, args.spread)
    # Don't auto-normalize the output: keeping the dB attenuation visible
    # is the whole point. The student can normalize downstream if desired.
    write_wav_mono_16bit(args.output, sr, y)
    print(f"wrote {args.output.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
