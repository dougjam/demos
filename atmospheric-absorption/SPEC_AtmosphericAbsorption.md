# SPEC: Atmospheric Absorption Demo

**Target directory:** `demos/atmospheric-absorption/`
**Framework:** Vanilla HTML/JS/CSS, Web Audio API, CDN libraries only (no build pipeline). Follows the conventions of `bubble-soundbank/`, `fire-bandwidth-extension/`, and other existing demos in the `dougjam/demos` repo per the root `CLAUDE.md`.
**Pedagogical context:** CS 448Z (Physically Based Animation and Sound), Spring 2026, lecture on Atmospheric and Underwater Sound (Thursday May 21, 2026). Accompanies Section 2.2 of `LECTURE_AtmosphericAndUnderwaterSound.md`.

---

## 0. Implementation deltas (post-spec)

Items in this spec that diverged from what shipped. The numerical model
(Sections 2, 5.1, 5.2, 5.5) is unchanged; the differences are in audio
implementation choice, source clips, and UI surface.

- **Audio filter (§4):** the FIR-via-IFFT convolver in §4.1 was prototyped
  but produced an audible click on every `ConvolverNode.buffer` swap
  (true even with double-buffered crossfade between two convolvers; the
  swap glitch is browser-internal). The shipping implementation is the
  §4.2 fallback: a 10-band peaking-EQ biquad cascade (31.5 Hz to 16 kHz,
  *Q* ≈ 1.4), per-band gains driven by `setTargetAtTime` so continuous
  slider drag is naturally smooth.
- **Source presets (§3.4, §7):** the file-based clips are now
  `thunder.wav`, `gunshot.wav`, `voice.wav`, `music.wav`, and
  `fan.wav` (added 7th preset: a real close-mic'd data-center recording
  for the "Data center" button). All five are real PD / CC0 / CC BY-SA
  recordings, sourced and rendered by `audio/fetch_real_audio.py`. See
  `ATTRIBUTION.md` for sources and licenses. `audio/generate_audio.py`
  is retained as a fallback only.
- **UI additions:** the plot is also click-and-drag interactive — drag
  the blue *A*(*f*, *r*) curve up/down to change *r* (the slider
  follows). The distance slider is labeled "Distance (*r*)" to match
  the variable used in the legend. The collapsible section is titled
  "About this sound model" rather than "About and references".
- **Numerical-accuracy test (§5.1):** the shipping `test.html` extends
  the spec's 14 expected rows with the §5.1 peak-RH sanity check
  (≈ 4% / 10% / 19% at 1 / 4 / 10 kHz) and the §5.2 relaxation
  frequencies — all expressed as tolerance-checked rows. The §5.4
  numbers ("0.4 dB at 50 Hz, ..., 527 dB at 8 kHz" at *r* = 5 km) are
  exact for the underlying *α*(*f*) · *r* and are reproduced by the
  plot; the biquad cascade approximates this curve to within ~2 dB per
  band in the audible range.

---

## 1. Purpose

Let the student hear and see what atmospheric absorption does to a broadband sound as a function of distance, temperature, humidity, and pressure. The student should be able to:

1. See the **absorption coefficient curve** alpha(f) in dB/km from 10 Hz to 100 kHz at any chosen (T, RH, p).
2. See the **total attenuation curve** A(f, r) = alpha(f) * r in dB for the current range r, overlaid on the alpha curve.
3. Hear an A/B comparison between a source clip played dry and the same clip played after frequency-dependent attenuation at the chosen (T, RH, p, r).
4. Switch between several source presets: thunder, gunshot, music, voice, white noise, pink noise.
5. Sweep distance smoothly with a slider and hear the timbre roll off as range grows. This is the central pedagogical experience.

The demo should make obvious that thunder at 5 km loses essentially all of its high-frequency content even at moderate humidity, that distance acts like a frequency-dependent low-pass filter, and that the rolloff is *not* a sharp cutoff but a gentle smoothly-varying absorption that gets steep above a few kHz.

---

## 2. Mathematical model

### 2.1 Standard

Implement ISO 9613-1:1993 *Acoustics: Attenuation of sound during propagation outdoors, Part 1*, equivalently Bass, Sutherland, Zuckerwar, Blackstock, Hester (1995) *Atmospheric absorption of sound: Further developments*, J. Acoust. Soc. Am. 97(1), 680-683, DOI 10.1121/1.412989. ANSI/ASA S1.26-2014 (R2019) is technically identical with one additional informative annex.

### 2.2 Formulas

Given frequency f (Hz), temperature T (Kelvin), pressure p (kPa), relative humidity RH (percent), the attenuation coefficient in dB/m is:

```
alpha(f) = 8.686 * f^2 * [
    A_classical +
    (T/T0)^(-5/2) * (A_O + A_N)
]
```

where

```
A_classical = 1.84e-11 * (p_ref / p) * sqrt(T / T0)
A_O = 0.01275 * exp(-2239.1 / T) * f_rO / (f_rO^2 + f^2)
A_N = 0.1068  * exp(-3352.0 / T) * f_rN / (f_rN^2 + f^2)
```

Constants:

- `T0  = 293.15` K  (reference temperature, 20 C)
- `T01 = 273.16` K  (triple point of water)
- `p_ref = 101.325` kPa  (reference pressure, 1 atm)

Saturation vapor pressure (Bass 1995 form, Davis 1992):

```
log10(psat / p_ref) = -6.8346 * (T01 / T)^1.261 + 4.6151
```

Molar concentration of water vapor (percent mole fraction):

```
h = RH * (psat / p_ref) / (p / p_ref)
```

Relaxation frequencies:

```
f_rO = (p / p_ref) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
f_rN = (p / p_ref) * (T / T0)^(-0.5)
              * (9 + 280 * h * exp(-4.170 * ((T / T0)^(-1/3) - 1)))
```

Valid (per ISO 9613-1) for -20 C <= T <= 50 C, 0.005% to >5% molar concentration of water vapor (roughly 0.2% to 100% RH at typical conditions), and frequency-to-pressure ratios from 4×10^-4 to 10 Hz/Pa (~40 Hz to ~1 MHz at 1 atm; ~20 Hz to ~500 kHz at 0.5 atm). The demo's slider ranges of 1 m to 100 km, -20 to 50 C, 1-100% RH, 50-110 kPa, and the displayed alpha curve from 10 Hz to 100 kHz all sit within the formula's validated range. The standard's stated accuracy depends on the regime; at typical surface-atmosphere conditions (within the ranges in Annex C tables) it is several percent, with larger uncertainty toward the edges of the validity range.

### 2.3 Application to audio

For a source spectrum S(f) propagated over range r (meters), the received spectrum is:

```
R(f) = S(f) * 10^(-alpha(f) * r / 20)
```

Note the factor of 20 (not 10): alpha is in dB per unit distance (a pure dB number), and the corresponding linear amplitude factor over distance r is 10^(-alpha*r/20), per the standard dB convention that an amplitude factor a corresponds to 20*log10(a) dB. (If we wanted a power factor we would use /10 in the exponent instead.)

**Geometric spreading (1/r) is NOT included by default**, since the focus is absorption, not propagation. Include a checkbox "include 1/r spreading" that multiplies output by 1/max(1, r) when checked.

### 2.4 Reference implementation

A verified Python reference must be included as `python/iso9613_reference.py`. It must reproduce the python-acoustics library output and the standard ISO 9613-1 Annex C tables. Sample test values at 20 C, 50% RH, 1 atm:

| f (Hz) | alpha (dB/km) |
|---|---|
| 100 | 0.29 |
| 1000 | 4.67 |
| 10000 | 159 |
| 100000 | 3280 (= 3.28 dB/m) |

Relaxation frequencies at 20 C, 50% RH, 1 atm: f_rO = 35.4 kHz, f_rN = 332 Hz. At 70% RH: f_rO = 53.2 kHz, f_rN = 461 Hz.

The JavaScript implementation must reproduce these values to within 1% (rounding errors only).

---

## 3. UI design

### 3.1 Layout

```
+--- Header ----------------------------------------------------+
| Atmospheric Absorption                                        |
| <- All demos . Source code . Python reference                 |
+---------------------------------------------------------------+

+--- Source presets --------------------------------------------+
| [Thunder] [Gunshot] [Voice] [Music] [White noise] [Pink noise]|
+---------------------------------------------------------------+

+--- Plot ------------------------------------------------------+
|   log-log plot of:                                            |
|     - alpha(f) in dB/km          (gray dashed, fixed)         |
|     - total attenuation A(f, r)  (solid colored, follows r)   |
|   x: 10 Hz - 100 kHz                                          |
|   y: 0.001 - 1000 dB (auto-clip beyond)                       |
|   light vertical guides at 20 Hz, 1 kHz, 20 kHz               |
+---------------------------------------------------------------+

+--- Environment ----+ +--- Propagation ----+ +--- Output ---------+
| Temperature    [-] | | Distance    [-]    | | A/B                |
|  -20 to 50 C       | | 1 m to 100 km      | |  ( ) Dry           |
| RH             [-] | |   (LOG slider)     | |  (*) Absorbed      |
|  1 to 100 %        | |                    | | Master volume [-]  |
| Pressure       [-] | | [ ] include 1/r    | | Loop          [x]  |
|  50 to 110 kPa     | +--------------------+ +--------------------+
+--------------------+

[ Start / Stop ]  [ Reload source ]
```

### 3.2 Sliders and ranges

| Parameter | Range | Default | Scale | Step |
|---|---|---|---|---|
| Temperature | -20 to +50 C | 20 C | linear | 1 C |
| Relative humidity | 1 to 100 % | 50 % | linear | 1 % |
| Pressure | 50 to 110 kPa | 101.325 kPa | linear | 0.1 kPa |
| Distance | 1 m to 100 km | 100 m | **logarithmic**, 200 steps | (log) |
| Master volume | 0 to 1 | 0.5 | linear | 0.01 |

Numeric value displayed next to each slider. Distance display should auto-switch units: m below 1000, km above (e.g., "850 m", "1.2 km", "45 km").

### 3.3 Plot

Use Chart.js loaded from CDN. Log-log axes. Two curves always visible:

- Thin gray dashed: alpha(f) in dB/km. Updates when (T, RH, p) change. Does NOT depend on distance.
- Thick colored solid (use the same accent color as the existing demos): total attenuation alpha(f) * r in dB at the current distance. Updates when any slider moves.

If the total attenuation curve exceeds 200 dB, clip visually and add a faint annotation "beyond audibility" at the right edge rather than letting the line escape.

Compute curves at 256 logarithmically-spaced frequency points from 10 Hz to 100 kHz. This is enough for a smooth plot and cheap enough to recompute on every slider movement (target 16 ms per redraw).

### 3.4 Presets

Six source clips, ~3-8 seconds each, mono 48 kHz WAV, peak-normalized to -6 dBFS. Hosted in `demos/atmospheric-absorption/audio/`. Suggested:

- `thunder.wav` - distant thunder clip (broadband, naturally low-pass, audible 1-5 kHz crackle)
- `gunshot.wav` - sharp transient with broadband content (rifle, door slam, or hand clap)
- `voice.wav` - speech (band-limited, good for showing speech-band absorption at km ranges)
- `music.wav` - solo instrument with substantial HF content (acoustic guitar, piano)
- white noise - generated live in Web Audio (no file needed)
- pink noise - generated live in Web Audio (no file needed)

**Licensing requirement:** all audio files MUST be CC0, public-domain, or own-recorded. Cite source and license for each file in `ATTRIBUTION.md`. Freesound (under CC0 / CC-BY licenses), the Internet Archive, and the Library of Congress National Jukebox have suitable thunder, gunshot/transient, voice, and music clips for academic use; check each file's license before downloading.

---

## 4. Audio implementation

### 4.1 Approach: FIR filter via inverse FFT

The absorption curve is smooth and monotonically increasing with frequency at all conditions inside the formula's validity range, well-suited to FIR filter approximation. A linear-phase FIR filter computed from the target frequency response is the cleanest implementation (the constant group delay of a linear-phase filter is acceptable here; the demo cares about magnitude shape, not absolute time-of-flight):

1. On parameter change, sample H(f) = 10^(-alpha(f) * r / 20) at N = 2048 frequency bins from 0 to Nyquist (24 kHz at 48 kHz sample rate). Use cosine mirror to ensure the impulse response is real.
2. IFFT to obtain a 2048-tap FIR impulse response.
3. Shift to center, apply a Hann window to suppress edge artifacts.
4. Load into a Web Audio `ConvolverNode` as a single-channel impulse response.
5. Route audio: source -> convolver -> gainNode -> destination.

Recompute the IR (steps 1-4) whenever any environment parameter or distance changes. Debounce slider input to at most 30 Hz update rate to avoid audio glitches; for parameters that change continuously (distance slider drag), use a short cross-fade between the old and new convolver paths to avoid clicks.

### 4.2 Alternative: time-domain biquad cascade

If the FIR convolver approach causes glitches or excess CPU on lower-end machines, fall back to a cascade of 4-6 Web Audio `BiquadFilterNode`s configured as a low-pass / shelf / peak chain that approximates alpha(f). Less accurate but cheaper and avoids the recompute-on-change problem. The FIR approach is preferred; this fallback is for performance regressions.

### 4.3 A/B switch

Two parallel signal paths from the source: one direct (dry) and one through the convolver (absorbed). The A/B radio buttons switch which path connects to the master gain. Use short (~10 ms) cross-fades on the gain to avoid clicks. The dry path should still respect the master volume; the only difference is whether the absorption convolver is in the chain.

### 4.4 White and pink noise generation

Use a Web Audio `AudioWorkletNode` (preferred) or `ScriptProcessorNode` (fallback) to generate noise live:

- White: Gaussian samples (Box-Muller), or uniform samples in [-1, 1], scaled to peak around -12 dBFS.
- Pink: filter white through a Voss-McCartney algorithm (3 octaves of summed random walks) or a Paul Kellet IIR filter.

These avoid distributing additional audio files.

---

## 5. Verification plan

The demo must pass all of the following tests before it ships. Add a small automated test harness at `demos/atmospheric-absorption/test.html` that runs these checks and reports pass/fail.

### 5.1 Numerical accuracy of alpha(f)

Test against the values produced by the Python reference (`python/iso9613_reference.py`). At each of the conditions in the table below, sample alpha at the listed frequencies and assert that the JavaScript implementation agrees with the Python reference to within 1% relative error. **The numbers below were computed directly from the ISO 9613-1 / Bass 1995 formula by the verified Python reference, not transcribed from a textbook**; the JS port must reproduce them to within rounding.

| T (C) | RH (%) | p (kPa) | f (Hz) | alpha (dB/km) | Notes |
|---|---|---|---|---|---|
| 20 | 50 | 101.325 | 100    | 0.294  | default conditions |
| 20 | 50 | 101.325 | 1000   | 4.66   | default |
| 20 | 50 | 101.325 | 10000  | 158.8  | default |
| 20 | 50 | 101.325 | 100000 | 3280   | default (3.28 dB/m) |
| 20 | 70 | 101.325 | 1000   | 4.98   | humid |
| 20 | 70 | 101.325 | 10000  | 117.5  | humid |
| 0  | 50 | 101.325 | 1000   | 6.83   | cold; modestly higher than 20 C at 1 kHz |
| 0  | 50 | 101.325 | 10000  | 172.1  | cold |
| 30 | 80 | 101.325 | 1000   | 7.41   | hot/humid |
| 30 | 80 | 101.325 | 10000  | 80.7   | hot/humid; *lower* than 20 C at 10 kHz |
| 20 | 50 | 80      | 1000   | 4.62   | reduced pressure (~altitude 2 km); barely changed from 1 atm |
| 20 | 5  | 101.325 | 1000   | 26.51  | very dry; near peak in RH at this frequency |
| 20 | 5  | 101.325 | 4000   | 64.48  | very dry, mid frequency |
| -20 | 50 | 101.325 | 1000  | 9.14   | very cold |

The Python reference is checked into the repo at `python/iso9613_reference.py` and produces the table above when run. The JavaScript test loads these expected values inline and verifies the JS port matches them to within 1% relative error.

Additionally, the peak absorption frequency depends on RH in a way the implementation should reproduce: at T = 20 C, 1 atm, sweeping RH from 1% to 100% with f = 1 kHz fixed, the peak absorption is reached at **RH ≈ 4%** (about 27 dB/km), not at very low humidity or at high humidity. The peak shifts to higher RH at higher frequencies (around 10% RH at 4 kHz, around 19% RH at 10 kHz). Verify this peak-finder behavior as a sanity check.

### 5.2 Relaxation frequencies

At 20 C, 50% RH, 1 atm, the JavaScript implementation must compute f_rO = 35.4 kHz (plus or minus 0.5 kHz) and f_rN = 332 Hz (plus or minus 5 Hz). At 70% RH, f_rO = 53.2 kHz and f_rN = 461 Hz.

### 5.3 Plot correctness

At default settings (20 C, 50% RH, 1 atm, r = 100 m):

- The gray alpha(f) curve should pass through approximately (1 kHz, 4.7 dB/km), (10 kHz, 159 dB/km).
- The colored attenuation curve A(f, r) = alpha(f) * r, with r = 100 m = 0.1 km, sits 10 dB below the alpha-per-km curve at every frequency (since multiplying by 0.1 corresponds to subtracting 10 dB). At r = 10 km, the attenuation curve sits 10 dB above. Document this offset in the legend.

### 5.4 Audio sanity

- Switch A/B with white noise at r = 5 km, 20 C, 50% RH, 1 atm. The "absorbed" version should sound dramatically muffled compared to dry. Numerically, the attenuation at 5 km is 0.4 dB at 50 Hz, 1.5 dB at 100 Hz, 14 dB at 500 Hz, 23 dB at 1 kHz, 49 dB at 2 kHz, 148 dB at 4 kHz, 527 dB at 8 kHz: low frequencies (below ~300 Hz) come through mostly intact, mid frequencies (around 1-2 kHz) are heavily attenuated, and high frequencies (above 4 kHz) are essentially gone.
- With r = 1 m, "absorbed" should sound essentially identical to "dry" (negligible absorption over 1 m). Difference at 1 kHz over 1 m is ~0.005 dB.
- Sweep distance from 1 m to 100 km with continuous noise input. No clicks or pops. Smooth audible rolloff.
- Switch source presets while audio plays. Clean transition.

### 5.5 Edge cases

- **RH at 5% (very dry, near peak):** At T = 20 C, 1 atm, alpha at 1 kHz is about 26.5 dB/km, roughly six times the 4.66 dB/km at 50% RH. At 10 kHz, it is about 83 dB/km, roughly *half* of the 159 dB/km at 50% RH. The frequency dependence reverses sign because the peak absorption frequency depends on humidity: the curve as a function of RH at fixed f has a peak in the middle.
- **RH at 1%:** At T = 20 C, 1 atm, alpha at 1 kHz is about 6.7 dB/km, only ~50% above 50% RH; the peak at 1 kHz is near 4%, so dropping further to 1% reduces absorption again.
- **T at -20 C with 50% RH:** alpha at 1 kHz is about 9.1 dB/km, about double the value at 20 C. Cold air does absorb more at audio frequencies, though not by orders of magnitude.
- **T at 30 C with 80% RH:** alpha at 10 kHz is about 81 dB/km, *lower* than the 159 dB/km at 20 C, 50% RH. Hot humid air can absorb *less* than the "default" conditions at high frequencies.
- **Pressure dependence is weak at fixed RH:** dropping from 101.325 kPa to 80 kPa at 20 C, 50% RH changes alpha at 1 kHz by less than 1% (4.66 to 4.62 dB/km). This is because lowering pressure at fixed RH raises the molar concentration of water vapor in a compensating way. The pressure slider is most interesting when combined with a low RH setting, simulating a dry high-altitude atmosphere where the relaxation frequencies shift substantially.

### 5.6 Cross-check against published references

Plot at 20 C, 50% RH and visually compare to Daniel Russell's interactive CDF demo "Absorption and Attenuation of Sound in Air" (linked from <https://www.acs.psu.edu/drussell/demos.html>). Russell's demo also uses the ISO 9613-1 formulas, so the curves should agree closely; visually any difference should be no more than a few dB anywhere in the audible band.

---

## 6. About-and-references section (page footer)

Include a section at the bottom of the page, matching the style of `bubble-soundbank/`. The text below is the canonical content; light wordsmithing is acceptable but do not change the science or the numbers.

> ### About
>
> Sound is absorbed by air through several mechanisms, parameterized in ISO 9613-1 as a sum of three terms: a classical term (viscous + thermal-conductive losses plus rotational relaxation, all approximately proportional to f² at audio frequencies); the **vibrational relaxation of O₂**; and the **vibrational relaxation of N₂**. The two vibrational terms have distinct relaxation frequencies and Arrhenius factors; they dominate above about 1 kHz and depend strongly on humidity because water vapor catalyzes both transitions, shifting their relaxation frequencies up into the ultrasonic.
>
> The result is a smooth absorption coefficient alpha(f) that grows with frequency: about 5 dB/km at 1 kHz, 160 dB/km at 10 kHz, and 3 dB/*meter* at 100 kHz under typical conditions (20 C, 50% RH, 1 atm). Over a few kilometers, all of the high-frequency content of a sound is stripped away: this is why distant thunder is a rumble, not a crack.
>
> Adjust temperature, humidity, pressure, and distance below and listen to the change. Move the distance slider continuously while a sound plays to hear the timbre roll off.
>
> ### Slider parameters
>
> **Temperature:** Air temperature. The vibrational relaxation frequencies depend on T via the Arrhenius factors exp(-Theta/T) where Theta = 2239 K (O2) and 3352 K (N2). Cold air generally absorbs more at audio frequencies than warm air, though the dependence is not large (about a factor of two between -20 C and +20 C at 1 kHz).
>
> **Relative humidity:** Water vapor catalyzes vibrational relaxation. The absorption at a given frequency does *not* increase monotonically with dryness; it peaks at a specific moderate humidity that depends on frequency (about 4% RH for 1 kHz, about 10% for 4 kHz, about 19% for 10 kHz at 20 C, 1 atm). Both much-drier and much-wetter air absorb less than the peak. This non-monotonic dependence is one of the more counterintuitive features of atmospheric acoustics.
>
> **Pressure:** Lowering pressure at fixed RH has a surprisingly weak effect on absorption (less than 1% change at 1 kHz between 80 and 101 kPa). This is because lowering pressure at fixed RH also raises the molar concentration of water vapor in a compensating way. The pressure slider becomes interesting when combined with a low RH: the dry, low-pressure stratosphere is a much more transparent acoustic medium than dense low-altitude air with the same RH, for example.
>
> **Distance:** Range from source to receiver. The total attenuation is alpha(f) * r, with alpha in dB/m or dB/km and r in matching units. The 1/r geometric spreading from a point source is *not* applied by default; toggle the checkbox to include it.
>
> ### References
>
> 1. **ISO 9613-1:1993**, *Acoustics: Attenuation of sound during propagation outdoors, Part 1: Calculation of the absorption of sound by the atmosphere*. International Organization for Standardization.
> 2. **ANSI/ASA S1.26-2014 (R2019)**, *Methods for Calculation of the Absorption of Sound by the Atmosphere*. Acoustical Society of America. Technical content identical to ISO 9613-1 with an additional approximate fractional-octave-band method.
> 3. **Bass, H. E., L. C. Sutherland, A. J. Zuckerwar, D. T. Blackstock, and D. M. Hester** (1995). "Atmospheric absorption of sound: Further developments." *J. Acoust. Soc. Am.* 97(1), 680-683. [doi:10.1121/1.412989](https://doi.org/10.1121/1.412989)
> 4. **Russell, D. A.** "Acoustics and Vibration Animations" — interactive Mathematica/CDF demos. Gallery: <https://www.acs.psu.edu/drussell/demos.html>. The relevant entry is "Absorption and Attenuation of Sound in Air".
> 5. **python-acoustics library**, ISO 9613-1 module: <https://github.com/python-acoustics/python-acoustics> (BSD 3-Clause). The `acoustics/standards/iso_9613_1_1993.py` module is a verified reference implementation; our Python reference is a self-contained port checked against it.

---

## 7. File layout

Per `CLAUDE.md`, single-file is preferred. Split only if `index.html` exceeds about 1000 lines.

```
demos/atmospheric-absorption/
├── index.html              the demo
├── ATTRIBUTION.md          audio sources, licensing
├── DEMO_DISCLAIMER.md      boilerplate, copy from existing demos
├── audio/
│   ├── thunder.wav
│   ├── gunshot.wav
│   ├── voice.wav
│   └── music.wav
├── python/
│   ├── iso9613_reference.py    50-line verified Python implementation
│   └── README.md               how to run it, expected output table
└── test.html               automated verification harness
```

---

## 8. Attribution requirements

1. **All audio source clips** must be CC0, public-domain, or own-recorded. Cite source and license for each file in `ATTRIBUTION.md`.
2. **The mathematical formulation** is from ISO 9613-1 and Bass 1995; cite both in the page footer as references 1 and 3 above.
3. **Code license:** BSD-2-Clause, consistent with the rest of `dougjam/demos`. Include a one-line attribution at the top of `index.html`: `<!-- (c) 2026 Doug James, Stanford University. BSD-2-Clause. -->`.

---

## 9. Acceptance criteria

The demo is done when all of the following pass:

1. All 14 entries in the numerical accuracy table (Section 5.1) pass in `test.html`.
2. Relaxation frequencies at the two conditions in Section 5.2 are reproduced to within tolerance.
3. The alpha(f) plot at default settings visually matches Russell's Penn State demo (Section 5.6).
4. All six source presets load, play, and can be A/B-switched cleanly.
5. Distance slider sweeps from 1 m to 100 km without audio artifacts.
6. The page works in current Chrome, Firefox, and Safari at desktop laptop resolutions (1280x800 minimum).
7. The page footer "About and references" section is present and accurate.
8. The Python reference in `python/` runs and produces the test table.
9. ATTRIBUTION.md cites every audio file's source and license.
10. The root `index.html` of the demos gallery has been updated with an entry pointing to this demo.

---

## 10. Suggested implementation order

1. Port the Python ISO 9613-1 code to JavaScript. Get the alpha(f) function right first.
2. Build the test harness (`test.html`) and verify all 14 numerical-accuracy entries pass. Do not move on until this passes.
3. Build the Chart.js plot. Verify it matches Russell's demo visually.
4. Add the sliders and wire them to the plot. Test that the plot updates smoothly.
5. Add white-noise generation as the first audio source. Wire up the convolver. Verify the FIR coefficients computed from the current alpha(f) sound right when applied to the noise.
6. Add A/B switching with cross-fades. Verify no clicks.
7. Add the WAV file presets.
8. Add edge-case testing (cold dry air, hot humid air, high altitude).
9. Write the About-and-references footer.
10. Cross-browser test.
11. Update the gallery index.

Stop after each step and re-run the test harness to catch regressions early.
