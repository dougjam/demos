"""
Bubble Sound Bank  -  Python companion module
==============================================

A small, self-contained Python library for synthesising bubble sounds, written
as a companion to the interactive JavaScript "Bubble Sound Bank" demo:

    https://dougjam.github.io/demos/bubble-soundbank/

The goal is to give students a minimal but working Python implementation of the
same physics so that they can:

    1. Generate WAV files of individual bubbles at any radius.
    2. Synthesise a scripted sequence of bubbles with per-event parameters.
    3. Generate continuous textures (drip, rain, stream, waterfall) by using
       the built-in presets, which drive a Poisson process over a log-spaced
       bank of 50 bubble voices.
    4. Render a WAV file from a CSV file of bubble events - e.g. exported from
       Houdini, Blender, or any other simulator that tracks bubble creation
       times and radii.

The code uses only NumPy and the Python standard library.  Install once with:

    pip install numpy

and then run this file directly:

    python bubble_soundbank.py

It will create a folder called 'examples/' next to this file containing
several demonstration WAV files (individual bubbles, a hand-scripted drip
sequence, four of the built-in presets, and a CSV-import example).

--------------------------------------------------------------------------
Physics in one paragraph
--------------------------------------------------------------------------
In the Minnaert idealisation, a newly formed spherical air bubble in water
radiates a short, exponentially decaying sinusoid.  Its resonant frequency
is set by the bubble radius r through the Minnaert relation,

        f = (1 / (2 pi r)) * sqrt(3 gamma_air P0 / rho_water)

which for water at standard conditions evaluates to roughly 3.3 / r (r in
meters).  Van den Doel (2005) uses the further rounded approximation
f = 3/r (Eq. 2).  The decay rate ("damping") grows quickly with frequency,

        d = 0.13 / r + 0.0072 / r**1.5

so small high-pitched bubbles decay in a few milliseconds while a 10 mm
bubble rings for tens of milliseconds.  Bubbles that form close to the
air-water interface and rise toward it can exhibit a rising pitch.  The
accepted explanation (van den Doel 2005, p. 538, following the Minnaert
model) is that the spherical shell of water surrounding a deeply submerged
bubble acts as an effective mass; as the bubble approaches the free
surface this effective mass is reduced, raising the resonant frequency up
to a factor of about sqrt(2) just below the surface.  We follow van den
Doel's empirical chirp model,

        f(t) = f0 * (1 + sigma t),    sigma = rise_factor * d

which is a practical curve fit, not a derivation from first principles.

Complex liquid sounds (rain, streams, waterfalls) emerge from the
stochastic superposition of many bubbles drawn from a distribution of
sizes.

--------------------------------------------------------------------------
Scope and limitations
--------------------------------------------------------------------------
This module is a pedagogical simplification of a research model, not a
substitute for it.  In particular:

  * The Minnaert model assumes a radially (volume-mode) oscillating
    spherical bubble.  Larger bubbles often depart substantially from
    being spherical and can excite shape modes that this model does not
    represent [van den Doel 2005, Sec. 7].
  * Bubble excitation is modelled as impulsive (instantaneous onset,
    exponential decay).  Real bubbles have a characteristic, non-impulsive
    onset profile that this code ignores.
  * The damping formula is valid for r > ~0.15 mm, where viscous losses
    are small compared with thermal and radiative losses
    [van den Doel 2005, Eq. 3].  Outside that range, results degrade.
  * The amplitude law here is a_k proportional to r_k**alpha (paper
    Eq. 6), but omits the random depth factor D that the paper uses to
    attenuate deep bubbles.  See `synthesize_bubble_rain` for details.
  * The Minnaert relation ignores surface tension and liquid viscosity,
    which matter for very small bubbles, and assumes an infinite body of
    water.  The sqrt(2) surface correction above is likewise an
    idealisation; actual surface/boundary effects are geometry-dependent.

These are teaching-appropriate approximations.  For research use,
consult the primary literature listed at the bottom of this file.

--------------------------------------------------------------------------
Attribution
--------------------------------------------------------------------------
Written by Doug James (Stanford) using Claude Code (Anthropic) as a
companion to CS 448Z "Physically Based Sound for Computer Graphics,
Virtual Reality, and Interactive Systems", Spring 2026.

Inspired by Kees van den Doel's "Complex Liquid Sound Simulator" Java
applet (University of British Columbia, 2005).  See the references at the
bottom of this file.

Version: 1.0
Date:    2026-04-16

--------------------------------------------------------------------------
Disclaimer
--------------------------------------------------------------------------
This demo was created by Doug James using Claude Code (Anthropic).  As
AI-assisted work, it warrants independent review and may contain errors,
inaccuracies, or oversimplifications.  It is provided "as is" for
educational purposes only, with no warranty of any kind, express or
implied.  It does not represent an official Stanford University product or
endorsement.  Do not rely on it for safety-critical applications,
engineering decisions, medical or legal guidance, or any purpose where
inaccuracies could cause harm.

You are free to share, modify, and redistribute this file for educational
purposes, provided this disclaimer and attribution are retained.  No
guarantee is made regarding correctness, completeness, or fitness for any
particular purpose.

--------------------------------------------------------------------------
References
--------------------------------------------------------------------------
[1] Van den Doel, K. (2005).  "Physically based models for liquid sounds."
    ACM Transactions on Applied Perception 2(4), 534-546.
    DOI: https://doi.org/10.1145/1101530.1101554
[2] Minnaert, M. (1933).  "XVI. On musical air-bubbles and the sounds of
    running water."  Philosophical Magazine, Series 7, 16(104), 235-248.
    DOI: https://doi.org/10.1080/14786443309462277
"""

import csv
import math
import os
import random
import wave

import numpy as np


# ============================================================================
# 1. Physical constants
# ============================================================================

GAMMA_AIR = 1.4         # ratio of specific heats for air (dimensionless)
P0 = 101_325.0          # atmospheric pressure (Pa)
RHO_WATER = 998.0       # density of water (kg/m^3)

DEFAULT_SAMPLE_RATE = 44100

# Convenience: typical bubble radius range for the 50-voice bank, in meters.
DEFAULT_R_MIN = 0.0002   # 0.2 mm   -> about 16 kHz
DEFAULT_R_MAX = 0.050    # 50  mm   -> about 65 Hz
DEFAULT_N_VOICES = 50


# ============================================================================
# 2. Minnaert frequency and damping coefficient
# ============================================================================

def minnaert_frequency(radius_m):
    """Resonant frequency (Hz) of a spherical air bubble of radius r (m).

    Derived from the Minnaert relation assuming adiabatic gas behaviour
    and ignoring surface tension and liquid viscosity.  For water at the
    standard conditions encoded in the module-level constants above, the
    coefficient evaluates to approximately 3.29, so f ~ 3.29 / r.  Using
    this formula:

        r = 1 mm    -> f ~ 3.3 kHz
        r = 10 mm   -> f ~ 330 Hz
        r = 0.2 mm  -> f ~ 16 kHz
        r = 50 mm   -> f ~ 66 Hz

    Van den Doel (2005, Eq. 2) rounds this to the convenient
    approximation f = 3 / r.
    """
    return (1.0 / (2.0 * math.pi * radius_m)) * math.sqrt(
        3.0 * GAMMA_AIR * P0 / RHO_WATER
    )


def damping_coefficient(radius_m):
    """Exponential damping rate d (1/s) for a bubble of radius r (m).

    From van den Doel (2005, Eq. 3).  The formula sums thermal and
    radiative contributions and is stated to be valid for bubbles larger
    than about 0.15 mm, for which viscous losses are not a significant
    contribution to energy dissipation.

    The impulse-response envelope is exp(-d t), so 1/d is the 1/e
    amplitude-decay time constant.  The amplitude drops by 60 dB after
    ln(1000)/d ~ 6.91/d seconds.
    """
    return 0.13 / radius_m + 0.0072 * radius_m ** (-1.5)


# ============================================================================
# 3. Single-bubble synthesis
# ============================================================================

def synthesize_bubble(radius_m,
                      amplitude=1.0,
                      rise_factor=0.0,
                      duration_s=None,
                      sample_rate=DEFAULT_SAMPLE_RATE):
    """Return a NumPy array containing one bubble "blip".

    Parameters
    ----------
    radius_m : float
        Bubble radius in meters.  Try values between 0.0002 (0.2 mm, a tiny
        high-pitched "tink") and 0.05 (50 mm, a deep "plunk").
    amplitude : float, default 1.0
        Peak amplitude of the (un-damped) sinusoid, in [0, 1].
    rise_factor : float, default 0.0
        If > 0, the bubble chirps upward with frequency
        f(t) = f0 * (1 + rise_factor * d * t).  Van den Doel (2005) uses
        this symbol ξ.  Try ~0.1 for the subtle chirp of a drop-formed
        bubble, and larger values (their Fig. 5 uses ξ ~ 10) for bubbles
        forced through a straw or nozzle close to the surface.  Values
        much above ~1 can sound unnatural for isolated drop-like bubbles.
    duration_s : float or None, default None
        Length of the output in seconds.  If None, we use min(0.3, 6/d)
        seconds: enough for the envelope to decay to about exp(-6) ~ 0.25%
        for small and medium bubbles, but hard-capped at 300 ms so that
        very large low-pitched bubbles do not produce arbitrarily long
        buffers.  For bubbles larger than about 10 mm the cap truncates
        the ring tail (the envelope at 300 ms has not yet decayed to
        0.25%); pass an explicit `duration_s` if you need the full decay.
    sample_rate : int, default 44100

    Returns
    -------
    signal : 1-D NumPy float array of length int(ceil(duration_s * sample_rate)).
    """
    f0 = minnaert_frequency(radius_m)
    d = damping_coefficient(radius_m)
    if duration_s is None:
        duration_s = min(0.3, 6.0 / d)
    n = int(math.ceil(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    envelope = np.exp(-d * t)
    if rise_factor > 0.0:
        sigma = rise_factor * d
        # Integrating f(t) = f0 (1 + sigma t) gives phase = 2 pi f0 (t + sigma t^2 / 2)
        phase = 2.0 * math.pi * f0 * (t + 0.5 * sigma * t * t)
    else:
        phase = 2.0 * math.pi * f0 * t
    return amplitude * np.sin(phase) * envelope


# ============================================================================
# 4. WAV file I/O
# ============================================================================

def write_wav(path, audio, sample_rate=DEFAULT_SAMPLE_RATE,
              normalize=True, peak=0.95):
    """Write a mono 16-bit PCM WAV file.

    Parameters
    ----------
    path : str
        Destination file path.  Any parent directories must already exist.
    audio : 1-D array-like of float
        Audio samples.  Values outside [-1, 1] are clipped after optional
        normalisation.
    sample_rate : int, default 44100
    normalize : bool, default True
        If True, scale the audio so its peak absolute value equals `peak`.
    peak : float, default 0.95
        Target peak (just below 1.0 to avoid integer-saturation clicks).
    """
    audio = np.asarray(audio, dtype=np.float64)
    if normalize:
        p = float(np.max(np.abs(audio))) if audio.size else 0.0
        if p > 0.0:
            audio = audio * (peak / p)
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(samples.tobytes())


# ============================================================================
# 5. Scripted sequence of bubble events
# ============================================================================

def synthesize_sequence(events,
                        total_duration_s=None,
                        sample_rate=DEFAULT_SAMPLE_RATE):
    """Mix a list of bubble events into a single audio buffer.

    Each event is a dict with keys:

        time         : onset time in seconds (required)
        radius_m     : bubble radius in meters (required)
        amplitude    : optional, default 1.0
        rise_factor  : optional, default 0.0

    Example
    -------
    >>> events = [
    ...     {"time": 0.0, "radius_m": 0.003},
    ...     {"time": 0.4, "radius_m": 0.006, "rise_factor": 0.2},
    ...     {"time": 0.9, "radius_m": 0.010, "amplitude": 0.6},
    ... ]
    >>> audio = synthesize_sequence(events)
    """
    events = list(events)
    rendered = []
    latest_end = 0.0
    for e in events:
        r = float(e["radius_m"])
        t0 = float(e.get("time", 0.0))
        amp = float(e.get("amplitude", 1.0))
        rise = float(e.get("rise_factor", 0.0))
        buf = synthesize_bubble(r,
                                amplitude=amp,
                                rise_factor=rise,
                                sample_rate=sample_rate)
        rendered.append((t0, buf))
        latest_end = max(latest_end, t0 + len(buf) / sample_rate)

    if total_duration_s is None:
        total_duration_s = latest_end + 0.05  # small tail pad

    n = int(math.ceil(total_duration_s * sample_rate))
    out = np.zeros(n, dtype=np.float64)
    for t0, buf in rendered:
        start = int(round(t0 * sample_rate))
        if start >= n or start + len(buf) <= 0:
            continue
        # clip the buffer to fit into the output range
        src_start = max(0, -start)
        dst_start = max(0, start)
        length = min(len(buf) - src_start, n - dst_start)
        if length > 0:
            out[dst_start:dst_start + length] += buf[src_start:src_start + length]
    return out


# ============================================================================
# 6. Bubble rain: Poisson process over a log-spaced voice bank
# ============================================================================

def log_spaced_radii(r_min_m=DEFAULT_R_MIN,
                     r_max_m=DEFAULT_R_MAX,
                     n_voices=DEFAULT_N_VOICES):
    """Return n_voices radii, log-spaced from r_min_m to r_max_m (meters)."""
    if n_voices < 2:
        return [r_min_m]
    log_min = math.log(r_min_m)
    log_max = math.log(r_max_m)
    return [math.exp(log_min + i / (n_voices - 1) * (log_max - log_min))
            for i in range(n_voices)]


def synthesize_bubble_rain(duration_s,
                           bubbles_per_sec,
                           gamma=2.0,
                           alpha=1.0,
                           r_min_m=DEFAULT_R_MIN,
                           r_max_m=DEFAULT_R_MAX,
                           rise_factor=0.0,
                           rise_probability=0.0,
                           n_voices=DEFAULT_N_VOICES,
                           sample_rate=DEFAULT_SAMPLE_RATE,
                           seed=None):
    """Stochastic bubble texture using N independent Poisson-triggered voices.

    This mirrors the JavaScript demo's core algorithm:

      * N log-spaced radii from r_min_m to r_max_m.
      * Voice k fires at Poisson rate
            lambda_k = (bubbles_per_sec / Z) * (1 / r_k) ** gamma
        where Z normalises the rates so that the total expected rate equals
        `bubbles_per_sec`.  Larger gamma biases the cloud toward small
        bubbles.
      * Voice k's bubbles have amplitude proportional to r_k ** alpha.
        This is a simplification of van den Doel (2005, Eq. 6), which is
        `a_k = D * r_k ** alpha` with D a random per-bubble depth factor
        (their Eq. 7, `D = rnd ** beta`) that attenuates deep bubbles.
        We omit D, so every bubble in a given voice has the same amplitude;
        the paper notes that the depth factor only has an audible effect
        on sparse sounds where individual bubbles can be heard.  Low
        alpha leaves small bubbles relatively loud (brighter spectrum);
        high alpha makes large bubbles dominate (darker spectrum).
      * A fraction `rise_probability` of bubbles chirp upward using
        `rise_factor`.  In the paper this fraction is derived from a
        depth threshold (only bubbles close enough to the surface chirp);
        here and in the JS demo we expose it as an explicit probability.

    Returns
    -------
    1-D float NumPy array of length int(ceil(duration_s * sample_rate)).
    """
    rng = random.Random(seed)
    radii = log_spaced_radii(r_min_m, r_max_m, n_voices)

    # --- per-voice rates (Poisson intensity)
    weights = [(1.0 / r) ** gamma for r in radii]
    Z = sum(weights) or 1.0
    rates = [bubbles_per_sec * w / Z for w in weights]

    # --- per-voice amplitudes, normalized so the loudest voice has amp 1
    amps = [r ** alpha for r in radii]
    max_amp = max(amps) or 1.0
    amps = [a / max_amp for a in amps]

    # --- pre-synthesise one flat buffer per voice, and a chirp buffer if needed
    flat_buffers = [
        synthesize_bubble(r, sample_rate=sample_rate) for r in radii
    ]
    chirp_buffers = None
    if rise_factor > 0.0 and rise_probability > 0.0:
        chirp_buffers = [
            synthesize_bubble(r, rise_factor=rise_factor, sample_rate=sample_rate)
            for r in radii
        ]

    n = int(math.ceil(duration_s * sample_rate))
    out = np.zeros(n, dtype=np.float64)

    for k in range(n_voices):
        rate = rates[k]
        if rate <= 0.0:
            continue
        amp = amps[k]
        flat = flat_buffers[k]
        chirp = chirp_buffers[k] if chirp_buffers is not None else None

        # draw Poisson events by cumulative exponential gaps
        t = 0.0
        while True:
            u = rng.random()
            if u <= 0.0:
                u = 1e-12
            t += -math.log(u) / rate
            if t >= duration_s:
                break
            start = int(t * sample_rate)
            if start >= n:
                break
            use_chirp = (chirp is not None) and (rng.random() < rise_probability)
            buf = chirp if use_chirp else flat
            end = min(start + len(buf), n)
            out[start:end] += buf[:end - start] * amp

    return out


# ============================================================================
# 7. CSV import (for Houdini / Blender / custom simulation exports)
# ============================================================================

def load_events_csv(path,
                    radius_units="m",
                    default_amplitude=1.0,
                    default_rise_factor=0.0):
    """Parse a CSV file of bubble events.

    The CSV is expected to have a header row.  Recognised column names
    (case-insensitive) are:

        time        | t | time_s          -> event time in seconds (required)
        radius_m    | radius | r          -> bubble radius (uses `radius_units`)
        radius_mm                         -> bubble radius in millimeters
        amplitude   | amp                 -> optional, default `default_amplitude`
        rise_factor | xi | chirp          -> optional, default `default_rise_factor`

    Extra columns are ignored.  If a `radius_mm` column is present it takes
    precedence over `radius` regardless of `radius_units`.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    radius_units : {"m", "mm"}, default "m"
        Units of the generic `radius` / `radius_m` / `r` column.  Ignored
        when a `radius_mm` column is present.

    Returns
    -------
    List of event dicts ready to pass to `synthesize_sequence`.
    """
    if radius_units not in ("m", "mm"):
        raise ValueError("radius_units must be 'm' or 'mm'")
    scale = 1.0 if radius_units == "m" else 1e-3

    events = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        # normalise header names to lowercase for forgiving lookup
        def pick(row, *names, default=None):
            for name in names:
                for k, v in row.items():
                    if k is not None and k.strip().lower() == name:
                        if v is None or v == "":
                            return default
                        return v
            return default

        for row in reader:
            time_val = pick(row, "time", "t", "time_s")
            if time_val is None:
                raise ValueError(
                    f"CSV row missing a 'time' column: {row!r}"
                )
            time_s = float(time_val)

            r_mm = pick(row, "radius_mm")
            if r_mm is not None:
                radius_m = float(r_mm) * 1e-3
            else:
                r_generic = pick(row, "radius_m", "radius", "r")
                if r_generic is None:
                    raise ValueError(
                        f"CSV row missing a radius column: {row!r}"
                    )
                radius_m = float(r_generic) * scale

            amp = float(pick(row, "amplitude", "amp",
                             default=default_amplitude))
            rise = float(pick(row, "rise_factor", "xi", "chirp",
                              default=default_rise_factor))

            events.append({
                "time": time_s,
                "radius_m": radius_m,
                "amplitude": amp,
                "rise_factor": rise,
            })
    return events


def render_csv_to_wav(csv_path, wav_path,
                      radius_units="m",
                      sample_rate=DEFAULT_SAMPLE_RATE,
                      total_duration_s=None):
    """Convenience: read a CSV of events and write a WAV file.

    Uses all the defaults of `load_events_csv` and `synthesize_sequence`.
    """
    events = load_events_csv(csv_path, radius_units=radius_units)
    audio = synthesize_sequence(events,
                                total_duration_s=total_duration_s,
                                sample_rate=sample_rate)
    write_wav(wav_path, audio, sample_rate=sample_rate)
    return audio


# ============================================================================
# 8. Presets (same values as the interactive JS demo)
# ============================================================================

PRESETS = {
    "dripping": {
        "bubbles_per_sec": 5,
        "gamma": 2.0, "alpha": 1.0,
        "rise_factor": 0.1, "rise_probability": 0.3,
        "r_min_m": 0.001,  "r_max_m": 0.010,
    },
    "light_rain": {
        "bubbles_per_sec": 5000,
        "gamma": 6.0, "alpha": 1.5,
        "rise_factor": 0.0, "rise_probability": 0.0,
        "r_min_m": 0.0002, "r_max_m": 0.005,
    },
    "stream": {
        "bubbles_per_sec": 2000,
        "gamma": 1.5, "alpha": 1.0,
        "rise_factor": 0.05, "rise_probability": 0.2,
        "r_min_m": 0.0002, "r_max_m": 0.020,
    },
    "waterfall": {
        "bubbles_per_sec": 20000,
        "gamma": 1.0, "alpha": 0.5,
        "rise_factor": 0.0, "rise_probability": 0.0,
        "r_min_m": 0.0002, "r_max_m": 0.050,
    },
    "straw_bubbles": {
        "bubbles_per_sec": 20,
        "gamma": 0.0, "alpha": 1.0,
        "rise_factor": 0.8, "rise_probability": 0.9,
        "r_min_m": 0.005,  "r_max_m": 0.030,
    },
}


def synthesize_preset(name, duration_s,
                      sample_rate=DEFAULT_SAMPLE_RATE,
                      seed=None):
    """Render `duration_s` seconds of a named preset."""
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Available: {sorted(PRESETS)}"
        )
    p = PRESETS[name]
    return synthesize_bubble_rain(
        duration_s=duration_s,
        sample_rate=sample_rate,
        seed=seed,
        **p,
    )


# ============================================================================
# 9. Main demo: run this file directly to generate example WAVs
# ============================================================================

def _demo():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "examples")
    os.makedirs(out_dir, exist_ok=True)

    sr = DEFAULT_SAMPLE_RATE
    pad = np.zeros(int(0.05 * sr))  # 50 ms of silence padding for single bubbles

    def save(name, audio):
        path = os.path.join(out_dir, name)
        write_wav(path, audio, sample_rate=sr)
        print(f"  wrote {path}  ({len(audio) / sr:.2f} s)")

    print("Bubble Sound Bank - generating example WAV files ...")

    # --- (1) single small bubble, 1 mm -> around 3.3 kHz, very short "tink"
    b = synthesize_bubble(radius_m=0.001)
    save("01_bubble_small_1mm.wav", np.concatenate([pad, b, pad]))

    # --- (2) single large bubble, 20 mm -> around 165 Hz, low "plunk"
    b = synthesize_bubble(radius_m=0.020)
    save("02_bubble_large_20mm.wav", np.concatenate([pad, b, pad]))

    # --- (3) rising bubble: mid-size with a strong upward chirp, "blooink"
    b = synthesize_bubble(radius_m=0.006, rise_factor=0.5)
    save("03_rising_bubble_chirp.wav", np.concatenate([pad, b, pad]))

    # --- (4) a hand-scripted drip sequence of five bubbles
    drip_events = [
        {"time": 0.00, "radius_m": 0.004, "amplitude": 1.0, "rise_factor": 0.15},
        {"time": 0.40, "radius_m": 0.006, "amplitude": 0.9, "rise_factor": 0.20},
        {"time": 0.95, "radius_m": 0.003, "amplitude": 0.8, "rise_factor": 0.10},
        {"time": 1.50, "radius_m": 0.008, "amplitude": 1.0, "rise_factor": 0.25},
        {"time": 2.00, "radius_m": 0.002, "amplitude": 0.7, "rise_factor": 0.00},
    ]
    save("04_drip_sequence.wav",
         synthesize_sequence(drip_events, total_duration_s=2.5))

    # --- (5-8) four presets, each 4 seconds
    for i, name in enumerate(("dripping", "light_rain",
                              "stream", "straw_bubbles"), start=5):
        save(f"{i:02d}_preset_{name}.wav",
             synthesize_preset(name, duration_s=4.0, seed=i))

    # --- (9) CSV-import example: write a toy CSV, then render it.
    # This is the format students would produce from Houdini / Blender /
    # a custom simulator: one row per bubble, with creation time and radius.
    csv_path = os.path.join(out_dir, "example_events.csv")
    rng = random.Random(42)
    with open(csv_path, "w", newline="") as f:
        f.write("time,radius_m,amplitude,rise_factor\n")
        # five opening "drip" events
        for t in (0.10, 0.35, 0.60, 0.95, 1.40):
            r = rng.uniform(0.002, 0.008)
            a = rng.uniform(0.5, 1.0)
            x = rng.uniform(0.0, 0.3)
            f.write(f"{t:.3f},{r:.5f},{a:.3f},{x:.3f}\n")
        # followed by a 0.8 s burst of 40 smaller bubbles
        for _ in range(40):
            t = 1.8 + rng.random() * 0.8
            r = rng.uniform(0.0002, 0.006)
            a = rng.uniform(0.3, 1.0)
            f.write(f"{t:.3f},{r:.5f},{a:.3f},0.000\n")
    print(f"  wrote {csv_path}")

    render_csv_to_wav(csv_path,
                      os.path.join(out_dir, "09_csv_import.wav"))
    print(f"  wrote {os.path.join(out_dir, '09_csv_import.wav')}")

    print(f"\nDone. All files are in: {out_dir}")


if __name__ == "__main__":
    _demo()
