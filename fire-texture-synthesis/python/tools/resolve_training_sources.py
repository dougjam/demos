"""Resolve which Recordist WAV each preset's training.bin was excerpted from.

For each example in ``assets/manifest.json``, decode the bundled
``training.bin`` (Float32 LE, 44.1 kHz mono) and search for it inside each
``assets/training_audio/FIRE_*.wav`` via FFT-based normalised cross-
correlation. Print, for each preset, the best-matching WAV plus the peak
correlation, the offset (samples and seconds), and the runner-up so we can
sanity-check the gap.

Run from the project root:

    python python/tools/resolve_training_sources.py

Output is plain text; copy the resulting mapping into
``vectors_to_assets.py`` (the ``EXAMPLE_TRAINING_AUDIO`` dict it expects)
and re-run that script to regenerate ``assets/manifest.json`` with the
``training_audio_file`` field populated.
"""

from __future__ import annotations

import json
import struct
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"


def load_bin(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    return np.frombuffer(raw, dtype="<f4").astype(np.float64, copy=False)


def load_wav_mono(path: Path, target_sr: int = 44100) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        # 24-bit PCM: assemble into int32, sign-extend, scale.
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32)
                | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int32) << 16))
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float64) / (2 ** 23)
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / (2 ** 31)
    else:
        raise RuntimeError(f"unsupported sample width {sw} in {path}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != target_sr:
        # Linear resample: fine for the discovery purpose; we only need the
        # peak correlation to dominate, not bit-perfect alignment.
        ratio = sr / target_sr
        n_out = int(np.floor(len(data) / ratio))
        x = np.arange(n_out) * ratio
        lo = np.floor(x).astype(np.int64)
        hi = np.minimum(lo + 1, len(data) - 1)
        f = x - lo
        data = data[lo] * (1.0 - f) + data[hi] * f
    return data


def normxcorr(needle: np.ndarray, haystack: np.ndarray) -> tuple[int, float]:
    """Return (best_offset, best_correlation) of needle inside haystack
    using FFT cross-correlation, normalised by per-window energy."""
    nn, nh = len(needle), len(haystack)
    if nn > nh:
        return 0, 0.0
    # Numerator: cross-correlation via FFT (haystack * conj(reverse(needle))).
    n_fft = 1
    while n_fft < nh + nn:
        n_fft <<= 1
    a = np.fft.rfft(haystack, n_fft)
    b = np.fft.rfft(needle[::-1], n_fft)
    corr = np.fft.irfft(a * b, n_fft)[nn - 1: nn - 1 + (nh - nn + 1)]
    # Denominator: needle norm * sliding haystack norm.
    needle_norm = np.linalg.norm(needle)
    if needle_norm == 0:
        return 0, 0.0
    h2 = haystack * haystack
    csum = np.concatenate(([0.0], np.cumsum(h2)))
    win_e = csum[nn:] - csum[: nh - nn + 1]
    win_e = np.maximum(win_e, 1e-30)
    norm = needle_norm * np.sqrt(win_e)
    score = corr / norm
    idx = int(np.argmax(score))
    return idx, float(score[idx])


def main() -> int:
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    wavs = manifest["training_audio"]
    wav_data = {}
    for entry in wavs:
        path = ASSETS / entry["file"]
        print(f"[load] {entry['file']}", file=sys.stderr)
        wav_data[entry["file"]] = load_wav_mono(path)

    rows = []
    for ex in manifest["examples"]:
        bin_path = ASSETS / ex["training_file"]
        needle = load_bin(bin_path)
        # Sub-sample long needles to speed up the search; the peak survives
        # because cross-correlation is linear in the needle length.
        if len(needle) > 60000:
            needle = needle[:60000]
        scores = []
        for wav_file, hay in wav_data.items():
            off, score = normxcorr(needle, hay)
            scores.append((wav_file, off, score))
        scores.sort(key=lambda r: r[2], reverse=True)
        best, runner = scores[0], scores[1]
        rows.append({
            "name": ex["name"],
            "best_wav": best[0],
            "best_score": best[2],
            "best_offset_samples": best[1],
            "best_offset_seconds": best[1] / 44100.0,
            "runner_up_wav": runner[0],
            "runner_up_score": runner[2],
        })
        print(
            f"{ex['name']:14s}  -> {best[0]}  "
            f"score={best[2]:.4f}  offset={best[1]/44100:.3f}s  "
            f"(2nd: {runner[0]} {runner[2]:.4f})"
        )

    print()
    print("Suggested EXAMPLE_TRAINING_AUDIO mapping for vectors_to_assets.py:")
    print("EXAMPLE_TRAINING_AUDIO = {")
    for row in rows:
        wav_basename = Path(row["best_wav"]).name
        print(f'    "{row["name"]}": "training_audio/{wav_basename}",')
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
