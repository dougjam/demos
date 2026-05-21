"""
Generate the four source-preset WAV files for the atmospheric-absorption
demo from scratch. All clips are synthesized (no recordings used), so the
output is original work released as CC0 alongside the rest of the demo.

Each clip:
  - mono, 48 kHz, 16-bit PCM
  - peak-normalized to -6 dBFS
  - 3 to 8 seconds long

Run:
    python generate_audio.py
"""
from __future__ import annotations
import math
import struct
import wave
from pathlib import Path
from random import Random

SR = 48000
PEAK = 10 ** (-6 / 20)  # -6 dBFS


def write_wav(path: Path, samples: list[float]) -> None:
    m = max(abs(s) for s in samples) or 1.0
    norm = PEAK / m
    pcm = bytearray()
    for s in samples:
        v = max(-1.0, min(1.0, s * norm))
        pcm += struct.pack("<h", int(v * 32767))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(pcm))


# ---------- Filters ----------
def lowpass(x: list[float], fc: float) -> list[float]:
    """One-pole IIR low-pass."""
    a = math.exp(-2 * math.pi * fc / SR)
    y = [0.0] * len(x)
    z = 0.0
    for i, v in enumerate(x):
        z = (1 - a) * v + a * z
        y[i] = z
    return y


def highpass(x: list[float], fc: float) -> list[float]:
    """One-pole IIR high-pass."""
    a = math.exp(-2 * math.pi * fc / SR)
    y = [0.0] * len(x)
    prev_in = 0.0
    prev_out = 0.0
    for i, v in enumerate(x):
        out = a * (prev_out + v - prev_in)
        y[i] = out
        prev_in = v
        prev_out = out
    return y


# ---------- Thunder ----------
def make_thunder(seed: int = 1) -> list[float]:
    rng = Random(seed)
    dur = 6.0
    n = int(dur * SR)
    # Base: low-passed pink-ish noise with slow amplitude envelope.
    raw = [rng.gauss(0, 1) for _ in range(n)]
    rumble = lowpass(raw, 200.0)
    rumble = lowpass(rumble, 200.0)
    # Slow envelope: ramp up, plateau, decay.
    env = []
    for i in range(n):
        t = i / SR
        if t < 0.6:
            e = t / 0.6
        else:
            e = math.exp(-(t - 0.6) / 2.5)
        env.append(e)
    out = [r * e for r, e in zip(rumble, env)]
    # Add a few "cracks": sharp band-passed bursts.
    for _ in range(5):
        t0 = rng.uniform(0.3, 4.5)
        i0 = int(t0 * SR)
        burst_dur = int(0.08 * SR)
        burst = [rng.gauss(0, 1) for _ in range(burst_dur)]
        # Band-pass around 1-2 kHz: HP then LP.
        burst = highpass(burst, 800.0)
        burst = lowpass(burst, 2500.0)
        for k, b in enumerate(burst):
            env_k = math.exp(-k / (0.03 * SR))
            if i0 + k < n:
                out[i0 + k] += 0.5 * b * env_k
    return out


# ---------- Gunshot ----------
def make_gunshot(seed: int = 2) -> list[float]:
    rng = Random(seed)
    dur = 1.2
    n = int(dur * SR)
    # Sharp broadband transient with exponential decay.
    out = [0.0] * n
    for i in range(n):
        t = i / SR
        if t < 0.001:
            attack = t / 0.001
        else:
            attack = 1.0
        env = attack * math.exp(-t / 0.15)
        out[i] = env * rng.gauss(0, 1)
    # Add a low-frequency "thump" (boom).
    for i in range(n):
        t = i / SR
        thump = math.sin(2 * math.pi * 80 * t) * math.exp(-t / 0.08) * 0.4
        out[i] += thump
    # Tail: short reverb-like decay via low-passed noise.
    tail_start = int(0.05 * SR)
    for i in range(tail_start, n):
        t = (i - tail_start) / SR
        out[i] += rng.gauss(0, 1) * 0.2 * math.exp(-t / 0.25)
    return out


# ---------- Voice (synthesized vowels) ----------
def vowel(f0: float, formants: list[tuple[float, float]], dur: float, rng: Random) -> list[float]:
    """Source-filter vowel synthesis: glottal pulse train through formant filters."""
    n = int(dur * SR)
    # Glottal source: band-limited pulse train.
    src = [0.0] * n
    period = SR / f0
    next_pulse = 0.0
    while next_pulse < n:
        i0 = int(next_pulse)
        # Rosenberg-like pulse shape over ~5 ms.
        pw = int(0.005 * SR)
        for k in range(pw):
            if i0 + k < n:
                ph = k / pw
                src[i0 + k] += math.sin(math.pi * ph) if ph < 1 else 0.0
        next_pulse += period * (1 + 0.01 * rng.gauss(0, 1))
    # Apply formant resonators as cascaded biquads (simple analog approximation).
    out = src[:]
    for fc, bw in formants:
        r = math.exp(-math.pi * bw / SR)
        theta = 2 * math.pi * fc / SR
        a1 = -2 * r * math.cos(theta)
        a2 = r * r
        y1 = y2 = 0.0
        new = [0.0] * n
        for i in range(n):
            y = out[i] - a1 * y1 - a2 * y2
            new[i] = y
            y2, y1 = y1, y
        out = new
    return out


def make_voice(seed: int = 3) -> list[float]:
    rng = Random(seed)
    # Three vowels at falling pitches, total ~4 s.
    segments = [
        (vowel(140, [(700, 80), (1220, 90), (2600, 120)], 1.0, rng), 0.05),  # /a/
        (vowel(125, [(310, 70), (2020, 100), (2960, 120)], 1.0, rng), 0.10),  # /i/
        (vowel(110, [(440, 80), (1020, 90), (2240, 110)], 1.4, rng), 0.10),  # /o/
    ]
    out: list[float] = []
    silence = [0.0] * int(0.15 * SR)
    for seg, gap in segments:
        # Fade in/out per segment.
        n = len(seg)
        fade = int(0.04 * SR)
        for i in range(fade):
            seg[i] *= i / fade
            seg[n - 1 - i] *= i / fade
        out.extend(seg)
        out.extend([0.0] * int(gap * SR))
    return silence + out + silence


# ---------- Music (decaying tonal notes) ----------
def pluck(f0: float, dur: float, rng: Random) -> list[float]:
    """Mode-summation guitar-ish pluck."""
    n = int(dur * SR)
    out = [0.0] * n
    n_harm = 6
    for h in range(1, n_harm + 1):
        amp = 1.0 / h
        phase = rng.uniform(0, 2 * math.pi)
        decay = max(0.4, 1.5 / h)  # higher harmonics decay faster
        for i in range(n):
            t = i / SR
            out[i] += amp * math.sin(2 * math.pi * f0 * h * t + phase) * math.exp(-t / decay)
    # Plucking transient.
    pw = int(0.005 * SR)
    for k in range(pw):
        if k < n:
            out[k] += (rng.random() - 0.5) * 0.6 * (1 - k / pw)
    return out


def make_music(seed: int = 4) -> list[float]:
    rng = Random(seed)
    # A short melody, e.g. C E G C E G with eighth-note spacing.
    notes = [261.63, 329.63, 392.00, 523.25, 392.00, 329.63]
    spacing = 0.55
    note_dur = 1.8
    total = spacing * len(notes) + note_dur
    n = int(total * SR)
    out = [0.0] * n
    for k, f in enumerate(notes):
        seg = pluck(f, note_dur, rng)
        start = int(k * spacing * SR)
        for i, v in enumerate(seg):
            if start + i < n:
                out[start + i] += v
    return out


# ---------- Driver ----------
if __name__ == "__main__":
    # NOTE: voice.wav and music.wav now come from real public-domain
    # recordings via fetch_real_audio.py. This script only regenerates
    # the two synthetic presets (thunder, gunshot). To regenerate voice
    # or music, either re-run fetch_real_audio.py or temporarily
    # uncomment the entries below.
    out_dir = Path(__file__).parent
    presets = {
        # All four presets now come from real public-domain / CC0 / CC
        # BY-SA recordings via fetch_real_audio.py. This script is
        # kept as a reference (and a fallback if the real downloads
        # ever break). Uncomment any entry below to regenerate the
        # purely-synthetic version; doing so will overwrite the real
        # recording on disk.
        # "thunder.wav": make_thunder(),
        # "gunshot.wav": make_gunshot(),
        # "voice.wav":   make_voice(),
        # "music.wav":   make_music(),
    }
    for name, samples in presets.items():
        path = out_dir / name
        write_wav(path, samples)
        print(f"wrote {path}  ({len(samples) / SR:.2f} s, {len(samples)} samples)")
