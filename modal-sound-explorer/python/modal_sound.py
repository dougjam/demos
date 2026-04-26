"""Modal sound synthesis for CS 448Z (Physically Based Animation and Sound).

This module implements modal sound synthesis following the van den Doel & Pai
framework. It provides four analytical vibration models (string, beam, membrane,
plate) with a consistent API for computing modal frequencies, evaluating mode
shapes, applying Rayleigh damping, synthesizing audio, and writing WAV files.

The core synthesis formula (van den Doel & Pai 1998, Eq. 1):

    y_k(t) = sum_n  a[n,k] * exp(-d[n] * t) * sin(2*pi*f[n] * t)

where f[n] are modal frequencies, d[n] are decay rates (Rayleigh damping),
and a[n,k] are gains (mode shapes evaluated at excitation point k).

Usage Examples
--------------
Example 1: Single impact on a steel plate::

    import modal_sound as ms

    model = ms.make_model("plate", "steel", num_modes=20,
                           strike_positions=(0.5, 0.5))
    audio = ms.synthesize_impact(model, amplitude=1.0, duration=3.0)
    ms.write_wav("steel_plate_center.wav", audio)

Example 2: Multiple impacts with spatial variation::

    import modal_sound as ms

    positions = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    model = ms.make_model("plate", "glass", num_modes=30,
                           strike_positions=positions)
    events = [
        {"time": 0.0,  "amplitude": 1.0, "position": 0},
        {"time": 0.3,  "amplitude": 0.7, "position": 1},
        {"time": 0.5,  "amplitude": 0.5, "position": 2},
        {"time": 0.8,  "amplitude": 0.9, "position": 3},
        {"time": 1.1,  "amplitude": 0.4, "position": 4},
    ]
    audio = ms.synthesize_contact_events(model, events)
    ms.write_wav("glass_plate_impacts.wav", audio)

Example 3: Loading events from a Houdini CSV export::

    import modal_sound as ms

    model = ms.make_model("beam", "wood", num_modes=16,
                           strike_positions=[0.1, 0.3, 0.5, 0.7, 0.9])
    events = ms.load_contact_events("houdini_impacts.csv",
                                     time_col="time",
                                     amplitude_col="impulse",
                                     position_col="position")
    audio = ms.synthesize_contact_events(model, events)
    ms.write_wav("wood_beam_sim.wav", audio)

Example 4: Overriding f1 for a specific object::

    import modal_sound as ms

    model = ms.make_model("plate", "steel", num_modes=24,
                           f1_override=120.0, aspect_ratio=1.5,
                           strike_positions=[(0.3, 0.5)])
    audio = ms.synthesize_impact(model, duration=4.0)
    ms.write_wav("big_steel_plate.wav", audio)

Example 5: Comparing materials on the same object::

    import modal_sound as ms

    for material in ["steel", "aluminum", "glass", "wood", "ceramic", "rubber"]:
        model = ms.make_model("beam", material, num_modes=12,
                               strike_positions=0.3)
        audio = ms.synthesize_impact(model, duration=2.0)
        ms.write_wav(f"beam_{material}.wav", audio)

References
----------
1. K. van den Doel and D. K. Pai, "The Sounds of Physical Shapes,"
   Presence, 7(4), 1998.
2. K. van den Doel, P. G. Kry, and D. K. Pai, "FoleyAutomatic:
   Physically-Based Sound Effects for Interactive Simulation and Animation,"
   SIGGRAPH 2001.
3. D. L. James, T. R. Langlois, R. Mehra, and C. Zheng, "Physically Based
   Sound for Computer Animation and Virtual Environments," SIGGRAPH 2016.
4. R. D. Blevins, Formulas for Natural Frequency and Mode Shape, 1979.
5. A. W. Leissa, Vibration of Plates, NASA SP-160, 1969.
6. CS 448Z Modal Sound Explorer: https://dougjam.github.io/demos/modal-sound-explorer/

Authorship & Disclaimer
-----------------------
Created by Doug James using Claude Code (Anthropic) for CS 448Z
(Physically Based Animation and Sound) at Stanford University, Spring 2026.
As AI-assisted work, this module warrants independent review and may contain
errors, inaccuracies, or oversimplifications. It is provided "as is" for
educational purposes only, with no warranty of any kind, express or implied.
It does not represent an official Stanford University product or endorsement.
Do not rely on it for safety-critical applications, engineering decisions,
or any purpose where inaccuracies could cause harm.

You are free to share, modify, and redistribute this module for educational
purposes, provided this disclaimer and attribution are retained. No guarantee
is made regarding correctness, completeness, or fitness for any particular
purpose.

Full disclaimer: https://github.com/dougjam/demos/blob/master/DEMO_DISCLAIMER.md
"""

__version__ = "1.0.0"
__date__ = "2026-04-16"
__author__ = "Doug James, with Claude Code (Anthropic)"
__license__ = "Educational use; provided 'as is' with no warranty. See module docstring."

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.io import wavfile
from scipy.special import jn_zeros, jv
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Material presets (matching the Modal Sound Explorer demo)
# ---------------------------------------------------------------------------

MATERIALS = {
    "steel":    {"f1": 440.0, "alpha": 1.0,   "beta": 2e-6},
    "aluminum": {"f1": 520.0, "alpha": 2.0,   "beta": 4e-6},
    "glass":    {"f1": 880.0, "alpha": 0.5,   "beta": 1e-6},
    "wood":     {"f1": 220.0, "alpha": 40.0,  "beta": 5e-5},
    "ceramic":  {"f1": 660.0, "alpha": 5.0,   "beta": 1e-5},
    "rubber":   {"f1": 110.0, "alpha": 200.0, "beta": 1e-3},
}

# ---------------------------------------------------------------------------
# Modal model dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModalModel:
    """The van den Doel modal model M = {f, d, A}.

    The impulse response at excitation point k is:

        y_k(t) = sum_n  a[n,k] * exp(-d[n] * t) * sin(2*pi*f[n] * t)

    Reference: van den Doel & Pai, "The Sounds of Physical Shapes," 1998, Eq. 1.
    Reference: van den Doel, Kry & Pai, "FoleyAutomatic," SIGGRAPH 2001, Eq. 1.

    Attributes
    ----------
    frequencies : np.ndarray, shape (N,)
        Modal frequencies in Hz.
    decay_rates : np.ndarray, shape (N,)
        Exponential decay rates in 1/s.
    gains : np.ndarray, shape (N, K)
        Mode-shape gains at K excitation points.
    name : str
        Optional label (e.g., "steel plate 1:1").
    """
    frequencies: np.ndarray
    decay_rates: np.ndarray
    gains: np.ndarray
    name: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rayleigh_damping(frequencies, alpha, beta):
    """Compute Rayleigh damping decay rates.

    d_n = alpha / 2 + beta * omega_n^2 / 2

    where omega_n = 2 * pi * f_n.

    Parameters
    ----------
    frequencies : np.ndarray
        Modal frequencies in Hz.
    alpha : float
        Mass-proportional Rayleigh damping (1/s).
    beta : float
        Stiffness-proportional Rayleigh damping (s).

    Returns
    -------
    np.ndarray
        Decay rates in 1/s.
    """
    omega = 2.0 * np.pi * frequencies
    return alpha / 2.0 + beta * omega ** 2 / 2.0


def _beam_frequency_eq(x):
    """Frequency equation for free-free beam: cos(x)*cosh(x) - 1 = 0."""
    return np.cos(x) * np.cosh(x) - 1.0


# Precompute the first 20 free-free beam eigenvalues (lambda_n).
_BEAM_EIGENVALUES = []
for _i in range(1, 21):
    _lo = (2 * _i + 1) * np.pi / 2 - np.pi / 2
    _hi = (2 * _i + 1) * np.pi / 2 + np.pi / 2
    # Tighten bracket around the known approximate root
    _lo = max(_lo, 4.0 if _i == 1 else (2 * _i + 1) * np.pi / 2 - 1.0)
    _hi = min(_hi, 5.5 if _i == 1 else (2 * _i + 1) * np.pi / 2 + 1.0)
    _BEAM_EIGENVALUES.append(brentq(_beam_frequency_eq, _lo, _hi))
_BEAM_EIGENVALUES = np.array(_BEAM_EIGENVALUES)


def _get_beam_eigenvalue(n):
    """Return the n-th free-free beam eigenvalue (1-indexed).

    Uses precomputed values for n <= 20 and the asymptotic approximation
    lambda_n ~ (2n+1)*pi/2 for higher modes.

    Parameters
    ----------
    n : int
        Mode number (1-indexed).

    Returns
    -------
    float
        The eigenvalue lambda_n.
    """
    if n <= len(_BEAM_EIGENVALUES):
        return _BEAM_EIGENVALUES[n - 1]
    return (2 * n + 1) * np.pi / 2.0


def _normalize_strike_1d(strike_positions):
    """Normalize 1D strike positions to a 1D numpy array.

    Accepts a single float, a list of floats, or an ndarray.
    Clamps to [0, 1] with a warning.
    """
    pos = np.atleast_1d(np.asarray(strike_positions, dtype=float)).ravel()
    if np.any(pos < 0) or np.any(pos > 1):
        warnings.warn("Strike positions clamped to [0, 1].")
        pos = np.clip(pos, 0.0, 1.0)
    return pos


def _normalize_strike_2d(strike_positions):
    """Normalize 2D strike positions to an (K, 2) numpy array.

    Accepts a single tuple (x, y), or a list of tuples.
    """
    if isinstance(strike_positions, tuple) and len(strike_positions) == 2:
        if not isinstance(strike_positions[0], (list, tuple, np.ndarray)):
            strike_positions = [strike_positions]
    pos = np.atleast_2d(np.asarray(strike_positions, dtype=float))
    if pos.shape[1] != 2:
        raise ValueError("2D strike positions must have shape (K, 2).")
    return pos


def _validate_f1(f1):
    """Warn if f1 is outside the audible range."""
    if f1 < 20.0:
        warnings.warn(f"f1 = {f1} Hz is below 20 Hz (infrasonic).")
    if f1 > 20000.0:
        warnings.warn(f"f1 = {f1} Hz is above 20 kHz (ultrasonic).")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def make_string_model(f1=440.0, alpha=1.0, beta=2e-6, num_modes=12,
                      strike_positions=0.5):
    """Build a modal model for a fixed-fixed string.

    Frequencies (harmonic):
        f_n = n * f1,    n = 1, 2, 3, ...

    Mode shapes:
        psi_n(x) = sin(n * pi * x / L)

    Reference: van den Doel & Pai 1998, Section 6, item 1 (taut string).

    Parameters
    ----------
    f1 : float
        Fundamental frequency in Hz.
    alpha : float
        Mass-proportional Rayleigh damping (1/s).
    beta : float
        Stiffness-proportional Rayleigh damping (s).
    num_modes : int
        Number of modes to include.
    strike_positions : float or array-like
        Normalized position(s) along the string, in [0, 1].

    Returns
    -------
    ModalModel
    """
    _validate_f1(f1)
    pos = _normalize_strike_1d(strike_positions)

    n = np.arange(1, num_modes + 1)
    freqs = n * f1
    decays = _rayleigh_damping(freqs, alpha, beta)

    # Gains: psi_n(x) = sin(n * pi * x), shape (num_modes, K)
    gains = np.sin(n[:, None] * np.pi * pos[None, :])

    return ModalModel(frequencies=freqs, decay_rates=decays, gains=gains,
                      name=f"string f1={f1}")


def make_beam_model(f1=440.0, alpha=1.0, beta=2e-6, num_modes=12,
                    strike_positions=0.5):
    """Build a modal model for a free-free Euler-Bernoulli beam.

    Frequencies (inharmonic):
        f_n = f1 * (lambda_n / lambda_1)^2

    where lambda_n are solutions to cos(x)*cosh(x) = 1.

    Mode shapes (free-free):
        psi_n(x) = cosh(lam*x) + cos(lam*x)
                   - sigma_n * [sinh(lam*x) + sin(lam*x)]
        sigma_n = [cosh(lam) - cos(lam)] / [sinh(lam) - sin(lam)]

    Reference: Blevins, Formulas for Natural Frequency and Mode Shape, 1979.

    Parameters
    ----------
    f1 : float
        Fundamental frequency in Hz.
    alpha : float
        Mass-proportional Rayleigh damping (1/s).
    beta : float
        Stiffness-proportional Rayleigh damping (s).
    num_modes : int
        Number of modes to include.
    strike_positions : float or array-like
        Normalized position(s) along the beam, in [0, 1].

    Returns
    -------
    ModalModel
    """
    _validate_f1(f1)
    pos = _normalize_strike_1d(strike_positions)

    lam1 = _get_beam_eigenvalue(1)
    freqs = np.empty(num_modes)
    gains = np.empty((num_modes, len(pos)))

    for i in range(num_modes):
        n = i + 1
        lam = _get_beam_eigenvalue(n)
        freqs[i] = f1 * (lam / lam1) ** 2

        # Mode shape coefficients
        sigma = (np.cosh(lam) - np.cos(lam)) / (np.sinh(lam) - np.sin(lam))

        # Evaluate mode shape at each strike position (x in [0,1], so L=1)
        lx = lam * pos
        psi = (np.cosh(lx) + np.cos(lx)
               - sigma * (np.sinh(lx) + np.sin(lx)))
        gains[i, :] = psi

    decays = _rayleigh_damping(freqs, alpha, beta)
    return ModalModel(frequencies=freqs, decay_rates=decays, gains=gains,
                      name=f"beam f1={f1}")


def make_membrane_model(f1=440.0, alpha=1.0, beta=2e-6, num_modes=20,
                        strike_positions=(0.3, 0.0)):
    """Build a modal model for a circular membrane with fixed edge.

    Modes labeled (m, n): azimuthal order m, radial order n.

    Frequencies (inharmonic):
        f_{mn} = f1 * z_{mn} / z_{01}

    where z_{mn} is the n-th zero of Bessel function J_m.

    Mode shapes (polar coords):
        psi_{mn}(r, theta) = J_m(z_{mn} * r / a) * cos(m * theta)

    Reference: van den Doel & Pai 1998, Section 6, item 4.

    Parameters
    ----------
    f1 : float
        Fundamental frequency in Hz.
    alpha : float
        Mass-proportional Rayleigh damping (1/s).
    beta : float
        Stiffness-proportional Rayleigh damping (s).
    num_modes : int
        Number of modes to include (sorted by frequency).
    strike_positions : tuple or list of tuples
        (r_norm, theta) with r_norm in [0, 1] and theta in radians.

    Returns
    -------
    ModalModel
    """
    _validate_f1(f1)
    pos = _normalize_strike_2d(strike_positions)  # (K, 2): r_norm, theta

    # Clamp r_norm to [0, 1]
    if np.any(pos[:, 0] < 0) or np.any(pos[:, 0] > 1):
        warnings.warn("Membrane r_norm clamped to [0, 1].")
        pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)

    # Enumerate (m, n) pairs and their Bessel zeros
    M_MAX, N_MAX = 10, 10
    mode_list = []  # (z_mn, m, n)
    for m in range(M_MAX + 1):
        zeros = jn_zeros(m, N_MAX)  # n-th zeros of J_m
        for idx, z in enumerate(zeros):
            mode_list.append((z, m, idx + 1))

    # Sort by Bessel zero (proportional to frequency)
    mode_list.sort(key=lambda x: x[0])

    z01 = jn_zeros(0, 1)[0]  # fundamental: z_{01} = 2.4048...

    # Take first num_modes
    modes_used = mode_list[:num_modes]

    freqs = np.empty(len(modes_used))
    gains = np.empty((len(modes_used), len(pos)))

    for i, (z_mn, m, n) in enumerate(modes_used):
        freqs[i] = f1 * z_mn / z01
        # psi_{mn}(r, theta) = J_m(z_mn * r_norm) * cos(m * theta)
        for k in range(len(pos)):
            r_norm, theta = pos[k]
            gains[i, k] = jv(m, z_mn * r_norm) * np.cos(m * theta)

    decays = _rayleigh_damping(freqs, alpha, beta)
    return ModalModel(frequencies=freqs, decay_rates=decays, gains=gains,
                      name=f"membrane f1={f1}")


def make_plate_model(f1=440.0, alpha=1.0, beta=2e-6, num_modes=20,
                     aspect_ratio=1.0, strike_positions=(0.5, 0.5)):
    """Build a modal model for a simply-supported rectangular plate.

    Frequencies (inharmonic):
        f_{mn} = f1 * [m^2 + n^2 * R^2] / [1 + R^2]

    where R = a/b is the aspect ratio, ensuring f_{11} = f1.

    Mode shapes:
        psi_{mn}(x, y) = sin(m * pi * x_norm) * sin(n * pi * y_norm)

    Reference: Leissa, Vibration of Plates, NASA SP-160, 1969.

    Parameters
    ----------
    f1 : float
        Fundamental frequency in Hz.
    alpha : float
        Mass-proportional Rayleigh damping (1/s).
    beta : float
        Stiffness-proportional Rayleigh damping (s).
    num_modes : int
        Number of modes to include (sorted by frequency).
    aspect_ratio : float
        Ratio a/b of plate dimensions. 1.0 = square.
    strike_positions : tuple or list of tuples
        (x_norm, y_norm) with both in [0, 1].

    Returns
    -------
    ModalModel
    """
    _validate_f1(f1)
    pos = _normalize_strike_2d(strike_positions)  # (K, 2): x_norm, y_norm

    # Clamp to [0, 1]
    if np.any(pos < 0) or np.any(pos > 1):
        warnings.warn("Plate strike positions clamped to [0, 1].")
        pos = np.clip(pos, 0.0, 1.0)

    R = aspect_ratio
    denom = 1.0 + R ** 2

    # Enumerate (m, n) pairs
    MAX_MN = 15
    mode_list = []  # (freq_ratio, m, n)
    for m in range(1, MAX_MN + 1):
        for n in range(1, MAX_MN + 1):
            ratio = (m ** 2 + n ** 2 * R ** 2) / denom
            mode_list.append((ratio, m, n))

    mode_list.sort(key=lambda x: x[0])
    modes_used = mode_list[:num_modes]

    freqs = np.empty(len(modes_used))
    gains = np.empty((len(modes_used), len(pos)))

    for i, (ratio, m, n) in enumerate(modes_used):
        freqs[i] = f1 * ratio
        for k in range(len(pos)):
            x_norm, y_norm = pos[k]
            gains[i, k] = (np.sin(m * np.pi * x_norm)
                           * np.sin(n * np.pi * y_norm))

    decays = _rayleigh_damping(freqs, alpha, beta)
    return ModalModel(frequencies=freqs, decay_rates=decays, gains=gains,
                      name=f"plate {aspect_ratio:.2f} f1={f1}")


def make_model(shape, material="steel", num_modes=12, strike_positions=0.5,
               aspect_ratio=1.0, f1_override=None):
    """Build a modal model from a shape name and material preset.

    This is the easiest entry point. Combines a shape type with a material
    preset from the MATERIALS dictionary.

    Parameters
    ----------
    shape : str
        One of "string", "beam", "membrane", "plate".
    material : str
        Key into MATERIALS dict (e.g., "steel", "glass", "wood").
    num_modes : int
        Number of modes to include.
    strike_positions : float, tuple, or list
        Strike position(s). Format depends on shape:
        - string/beam: float or list of floats in [0, 1]
        - membrane: (r_norm, theta) or list of such
        - plate: (x_norm, y_norm) or list of such
    aspect_ratio : float
        Plate aspect ratio a/b (only used for "plate").
    f1_override : float or None
        Override the material preset's f1 with a custom value.

    Returns
    -------
    ModalModel
    """
    mat = MATERIALS.get(material.lower())
    if mat is None:
        raise ValueError(
            f"Unknown material '{material}'. "
            f"Choose from: {', '.join(MATERIALS.keys())}")

    f1 = f1_override if f1_override is not None else mat["f1"]
    alpha, beta = mat["alpha"], mat["beta"]

    shape_lower = shape.lower()
    if shape_lower == "string":
        return make_string_model(f1, alpha, beta, num_modes, strike_positions)
    elif shape_lower == "beam":
        return make_beam_model(f1, alpha, beta, num_modes, strike_positions)
    elif shape_lower == "membrane":
        return make_membrane_model(f1, alpha, beta, num_modes,
                                   strike_positions)
    elif shape_lower == "plate":
        return make_plate_model(f1, alpha, beta, num_modes, aspect_ratio,
                                strike_positions)
    else:
        raise ValueError(
            f"Unknown shape '{shape}'. "
            f"Choose from: string, beam, membrane, plate")


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def synthesize_impact(model, strike_index=0, amplitude=1.0, duration=2.0,
                      sample_rate=44100):
    """Synthesize the sound of a single impact.

    Implements the van den Doel & Pai (1998) modal synthesis formula:

        y_k(t) = sum_n  a[n,k] * exp(-d[n] * t) * sin(2*pi*f[n] * t)

    Parameters
    ----------
    model : ModalModel
    strike_index : int
        Index into the gains matrix (which pre-sampled strike point).
    amplitude : float
        Overall amplitude scaling (proportional to impact force/velocity).
    duration : float
        Length of audio to generate, in seconds.
    sample_rate : int
        Audio sample rate in Hz.

    Returns
    -------
    np.ndarray
        1D array of audio samples (float64).
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate  # (T,)

    freqs = model.frequencies
    decays = model.decay_rates
    gains = model.gains[:, strike_index]  # (N,)

    # Filter out modes above Nyquist
    nyquist = sample_rate / 2.0
    mask = freqs < nyquist
    if not np.all(mask):
        n_skipped = np.sum(~mask)
        warnings.warn(
            f"Skipping {n_skipped} mode(s) above Nyquist ({nyquist} Hz).")
        freqs = freqs[mask]
        decays = decays[mask]
        gains = gains[mask]

    # Vectorized synthesis: broadcast (N,1) over (1,T)
    audio = np.sum(
        gains[:, None]
        * np.exp(-decays[:, None] * t[None, :])
        * np.sin(2.0 * np.pi * freqs[:, None] * t[None, :]),
        axis=0
    )

    return audio * amplitude


def synthesize_contact_events(model, events, duration=None,
                              sample_rate=44100, fade_out=0.01):
    """Synthesize audio from a sequence of contact events.

    Each event is a dict with keys:
        "time":      float  -- onset time in seconds
        "amplitude": float  -- impact amplitude
        "position":  int    -- index into model.gains columns

    Parameters
    ----------
    model : ModalModel
    events : list of dict or np.ndarray
        Contact events. If ndarray, columns are [time, amplitude, position].
    duration : float or None
        Total audio length. If None, extends to last event + 2s.
    fade_out : float
        Fade-out duration at the end, in seconds.
    sample_rate : int
        Audio sample rate in Hz.

    Returns
    -------
    np.ndarray
        Mono audio signal, float64.
    """
    # Parse events
    if isinstance(events, np.ndarray):
        event_list = []
        for row in events:
            event_list.append({
                "time": float(row[0]),
                "amplitude": float(row[1]),
                "position": int(row[2]),
            })
        events = event_list

    if not events:
        warnings.warn("No contact events provided.")
        return np.zeros(int(2.0 * sample_rate))

    # Determine total duration
    last_time = max(e["time"] for e in events)
    if duration is None:
        duration = last_time + 2.0

    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples)

    for event in events:
        t_onset = event["time"]
        amp = event.get("amplitude", 1.0)
        pos = event.get("position", 0)

        # Convert position to strike index
        if isinstance(pos, (int, np.integer)):
            strike_idx = int(pos)
        else:
            strike_idx = 0

        sample_offset = int(t_onset * sample_rate)
        if sample_offset >= num_samples:
            continue

        remaining = num_samples - sample_offset
        remaining_dur = remaining / sample_rate
        impact = synthesize_impact(model, strike_index=strike_idx,
                                   amplitude=amp, duration=remaining_dur,
                                   sample_rate=sample_rate)
        audio[sample_offset:sample_offset + len(impact)] += impact

    # Apply fade-out
    if fade_out > 0:
        fade_samples = min(int(fade_out * sample_rate), num_samples)
        if fade_samples > 0:
            fade = np.linspace(1.0, 0.0, fade_samples)
            audio[-fade_samples:] *= fade

    return audio


# ---------------------------------------------------------------------------
# Contact event I/O
# ---------------------------------------------------------------------------


def load_contact_events(filepath, time_col=0, amplitude_col=1,
                        position_col=2, delimiter=",", skip_header=True):
    """Load contact events from a CSV or text file.

    Expected format (CSV with header):
        time, amplitude, position
        0.10, 0.8, 0.25
        0.35, 0.5, 0.75

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    time_col : str or int
        Column name or index for onset time.
    amplitude_col : str or int
        Column name or index for impact amplitude.
    position_col : str or int or None
        Column name or index for strike position index.
        If None, all events use position 0.
    delimiter : str
        Column delimiter.
    skip_header : bool
        If True, skip the first row as a header.

    Returns
    -------
    list of dict
        Event dicts with keys "time", "amplitude", "position".
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    if not lines:
        return []

    # Parse header if column names are given as strings
    header = None
    start = 0
    if skip_header and lines:
        header = [h.strip() for h in lines[0].split(delimiter)]
        start = 1

    def _col_index(col, header):
        if isinstance(col, str) and header is not None:
            return header.index(col)
        return int(col)

    t_idx = _col_index(time_col, header)
    a_idx = _col_index(amplitude_col, header)
    p_idx = _col_index(position_col, header) if position_col is not None else None

    events = []
    for line in lines[start:]:
        parts = line.strip().split(delimiter)
        if not parts or not parts[0]:
            continue
        event = {
            "time": float(parts[t_idx]),
            "amplitude": float(parts[a_idx]),
            "position": int(float(parts[p_idx])) if p_idx is not None else 0,
        }
        events.append(event)

    return events


# ---------------------------------------------------------------------------
# WAV output
# ---------------------------------------------------------------------------


def write_wav(filepath, audio, sample_rate=44100, normalize=True,
              peak_db=-3.0):
    """Write audio to a WAV file (16-bit PCM).

    Parameters
    ----------
    filepath : str
        Output file path.
    audio : np.ndarray
        1D audio signal (float64).
    sample_rate : int
        Audio sample rate in Hz.
    normalize : bool
        If True, normalize peak amplitude to peak_db.
    peak_db : float
        Target peak level in dB (default -3.0 dB).
    """
    signal = np.copy(audio)
    peak = np.max(np.abs(signal))

    if normalize and peak > 0:
        target = 10.0 ** (peak_db / 20.0)
        signal = signal * (target / peak)

    # Convert to 16-bit PCM
    signal = np.clip(signal, -1.0, 1.0)
    pcm = (signal * 32767).astype(np.int16)
    wavfile.write(filepath, sample_rate, pcm)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def mix_audio(*signals, offsets=None, sample_rate=44100):
    """Mix multiple audio signals together, optionally with time offsets.

    Parameters
    ----------
    *signals : np.ndarray
        Audio signals to mix.
    offsets : list of float or None
        Time offsets in seconds for each signal. If None, all start at t=0.
    sample_rate : int
        Audio sample rate in Hz.

    Returns
    -------
    np.ndarray
        Mixed audio signal.
    """
    if offsets is None:
        offsets = [0.0] * len(signals)

    # Determine total length
    lengths = []
    for sig, off in zip(signals, offsets):
        start_sample = int(off * sample_rate)
        lengths.append(start_sample + len(sig))
    total = max(lengths)

    mixed = np.zeros(total)
    for sig, off in zip(signals, offsets):
        start = int(off * sample_rate)
        mixed[start:start + len(sig)] += sig

    return mixed


def apply_gain_db(audio, gain_db):
    """Apply a gain in dB to an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    gain_db : float
        Gain in decibels.

    Returns
    -------
    np.ndarray
        Scaled audio signal.
    """
    return audio * 10.0 ** (gain_db / 20.0)


def compute_t60(model, mode_index=0):
    """Compute the T60 (60 dB decay time) for a given mode.

    T60 = 6 * ln(10) / d_n  ~  13.8155 / d_n

    Parameters
    ----------
    model : ModalModel
    mode_index : int
        Index of the mode.

    Returns
    -------
    float
        T60 in seconds.
    """
    d = model.decay_rates[mode_index]
    if d <= 0:
        return float("inf")
    return 6.0 * np.log(10.0) / d


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("modal_sound.py -- self-test / demo")
    print("=" * 60)

    shapes = ["string", "beam", "membrane", "plate"]
    files_generated = []

    # 1. Single impact for each shape with steel preset
    for shape in shapes:
        if shape in ("membrane", "plate"):
            sp = (0.3, 0.0) if shape == "membrane" else (0.3, 0.7)
            nm = 20
        else:
            sp = 0.3
            nm = 12

        model = make_model(shape, "steel", num_modes=nm,
                           strike_positions=sp)
        audio = synthesize_impact(model, duration=2.0)
        fname = f"{shape}_steel.wav"
        write_wav(fname, audio)
        files_generated.append(fname)

        t60 = compute_t60(model, 0)
        print(f"\n{shape.upper()} (steel):")
        print(f"  Modes: {len(model.frequencies)}")
        print(f"  Frequencies: {model.frequencies[:6].round(1)} ...")
        print(f"  T60 (mode 0): {t60:.3f} s")

    # 2. Multi-impact glass plate demo
    print("\n" + "-" * 60)
    print("Multi-impact glass plate demo")
    positions = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    glass_plate = make_model("plate", "glass", num_modes=30,
                             strike_positions=positions)
    events = [
        {"time": 0.0,  "amplitude": 1.0, "position": 0},
        {"time": 0.3,  "amplitude": 0.7, "position": 1},
        {"time": 0.5,  "amplitude": 0.5, "position": 2},
        {"time": 0.8,  "amplitude": 0.9, "position": 3},
        {"time": 1.1,  "amplitude": 0.4, "position": 4},
    ]
    audio = synthesize_contact_events(glass_plate, events)
    fname = "glass_plate_impacts.wav"
    write_wav(fname, audio)
    files_generated.append(fname)

    t60 = compute_t60(glass_plate, 0)
    print(f"  Modes: {len(glass_plate.frequencies)}")
    print(f"  Frequencies: {glass_plate.frequencies[:6].round(1)} ...")
    print(f"  T60 (mode 0): {t60:.3f} s")
    print(f"  Events: {len(events)}")

    # Summary
    print("\n" + "=" * 60)
    print("Generated files:")
    for f in files_generated:
        print(f"  {f}")
    print("=" * 60)
    print("Done.")
