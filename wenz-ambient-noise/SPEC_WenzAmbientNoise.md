# SPEC: Wenz Curves Ambient Noise Generator

**Target directory:** `demos/wenz-ambient-noise/`
**Framework:** Vanilla HTML/JS/CSS, Web Audio API, CDN libraries only (no build pipeline). Follows the conventions of `bubble-soundbank/`, `fire-bandwidth-extension/`, and other existing demos in the `dougjam/demos` repo per the root `CLAUDE.md`.
**Pedagogical context:** CS 448Z (Physically Based Animation and Sound), Spring 2026, lecture on Atmospheric and Underwater Sound (Thursday May 21, 2026). Accompanies Section 3.6 of `LECTURE_AtmosphericAndUnderwaterSound.md`.

---

## 1. Purpose

Synthesize physically-meaningful ocean ambient noise directly from the Wenz (1962) source-decomposition model, with sliders for the major physical drivers. The student should be able to:

1. See the **Wenz spectrum** as a sum of four canonical components (turbulent pressure, distant shipping, wind/surface, thermal), each on the same log-log plot, plus their sum.
2. Hear the resulting audio in real time.
3. Adjust **wind speed** (or equivalently Beaufort scale) and hear the broadband hiss change in level and color.
4. Adjust **shipping density** and hear the low-frequency rumble appear and intensify.
5. Solo individual components (mute the other three) to hear what each mechanism sounds like in isolation. This is the central pedagogical experience.
6. Optionally mix in **bubble bursts** from the existing `bubble-soundbank` demo to add the natural texture of breaking waves at higher wind speeds.

By the end of two minutes with the demo, a student should viscerally understand that ocean ambient noise is not a single mysterious "ocean sound," but a sum of identifiable physical mechanisms each with a characteristic spectrum and source.

---

## 2. Mathematical model

### 2.1 Standard

The reference for the underlying empirical decomposition is Wenz (1962), *Acoustic ambient noise in the ocean: spectra and sources*, J. Acoust. Soc. Am. 34(12), 1936-1956, DOI 10.1121/1.1909155, Figure 13 in particular. Modern empirical fits and updated values come from Hildebrand, Frasier, Baumann-Pickering, Wiggins (2021), *An empirical model for wind-generated ocean noise*, J. Acoust. Soc. Am. 149(6), 4516-4533, DOI 10.1121/10.0005430. The JASA Reflections retrospective is Deane (2025), *The Wenz curves for underwater ambient sound*, J. Acoust. Soc. Am. 157(5), R9-R10, DOI 10.1121/10.0036690.

The closed-form component PSD parameterization used by this demo is the widely-cited Coates 1989 / Stojanovic 2007 family of formulas, established in Coates, R. F. W., *Underwater Acoustic Systems*, Macmillan New Electronics Series, 1989, ISBN 978-1-349-20508-0, and used in Stojanovic, M. (2007), *On the relationship between capacity and distance in an underwater acoustic channel*, ACM SIGMOBILE Mobile Computing and Communications Review 11(4), 34-43. These formulas are derived as analytic fits to Wenz 1962 Figure 13 and Mellen 1952; they reproduce the qualitative shape and rough levels of Wenz curves while being trivially evaluable in closed form. They are not exact reproductions of Wenz Figure 13, but they are the standard pedagogical and engineering parameterization used in the underwater acoustic communications literature.

All spectrum levels below are in dB re 1 µPa^2/Hz unless noted.

### 2.2 Total spectrum

The total ambient noise spectrum is the **incoherent sum of four components**, summed in linear PSD (µPa^2/Hz) and converted back to dB:

```
NL_total(f) = 10 * log10(
    P_t(f) +
    P_s(f, s) +
    P_w(f, U) +
    P_th(f)
)
```

where each `P_X(f) = 10^(NL_X(f) / 10)` is the linear PSD of the corresponding component, `s` in [0, 1] is the shipping activity factor (0 = low, 1 = heavy), and `U` is the 10 m wind speed in m/s.

### 2.3 Component spectra (Coates 1989 / Stojanovic 2007)

The four standard parameterizations are below. **f is in kHz**, U is in m/s, s is in [0, 1].

**Turbulent pressure** (Wenz 1962 low-frequency tail; pseudo-sound):

```
NL_t(f_kHz) = 17 - 30 * log10(f_kHz)        [dB re 1 µPa^2/Hz]
```

This is hydrodynamic ("pseudo-sound") rather than propagating acoustic at the lowest frequencies; it appears on Wenz Figure 13 below ~10 Hz and falls steeply. At 10 Hz (f = 0.01 kHz): NL_t = 77 dB. At 100 Hz: 47 dB. Above ~100 Hz it is negligible compared to wind and shipping.

**Distant shipping** (peaked around 30 Hz, function of traffic):

```
NL_s(f_kHz, s) = 40 + 20*(s - 0.5) + 26*log10(f_kHz) - 60*log10(f_kHz + 0.03)
```

At s = 1 (heavy), 30 Hz (f = 0.03 kHz): NL_s ≈ 84 dB. At s = 0 (low), same f: NL_s ≈ 64 dB.

To allow the user to silence shipping completely (a "None" preset that is not just "low" but truly off), implement an additional **mute toggle** independent of the s slider: when shipping is muted, P_s is set to zero in the linear sum, removing it entirely. The s slider controls the level when shipping is unmuted.

**Wind / surface noise** (dominant 100 Hz to 30 kHz; family-of-curves originally Knudsen et al. 1948 and Wenz 1962, closed-form Coates/Stojanovic):

```
NL_w(f_kHz, U) = 50 + 7.5 * sqrt(max(U, 0)) + 20*log10(f_kHz) - 40*log10(f_kHz + 0.4)
```

At U = 5 m/s, f = 1 kHz: NL_w ≈ 61 dB. At U = 12 m/s, f = 1 kHz: NL_w ≈ 70 dB. At U = 0, the wind component still contributes a small baseline (the constant 50 dB and the f-dependent shape); to silence wind, use a separate mute toggle the same way as for shipping.

**Thermal noise** (Mellen 1952):

```
NL_th(f_kHz) = -15 + 20*log10(f_kHz)        [dB re 1 µPa^2/Hz]
```

Source: Mellen, R. H. (1952), "The thermal-noise limit in the detection of underwater acoustic signals," J. Acoust. Soc. Am. 24(5), 478-480, DOI 10.1121/1.1906924. This is the theoretical thermal-noise floor set by equipartition at kT temperature; it rises 20 dB/decade with frequency and is the absolute floor for any underwater hydrophone above ~50 kHz. At 100 kHz: NL_th = 25 dB. At 1 kHz: -15 dB (essentially inaudible).

### 2.4 Calibration against Wenz Figure 13

The Coates/Stojanovic formulas reproduce the qualitative shape of Wenz 1962 Figure 13 across the audio band but are not exact digitizations. Typical agreement is within about plus or minus 5-10 dB at most frequencies. For example, the formulas give:

- Calm + no traffic, 1 kHz: total ≈ 50 dB (Wenz Fig 13 shows about 35-45 dB for sea state 0)
- Gentle breeze (U = 5 m/s), 1 kHz: total ≈ 61 dB (Wenz: about 50-55 dB)
- Strong wind (U = 12 m/s), 1 kHz: total ≈ 70 dB (Wenz: about 60-70 dB)
- Heavy shipping (s = 1), 30 Hz, U = 5: total ≈ 84 dB (Wenz: about 80-95 dB for heavy)

These are pedagogically adequate: shapes and slider responses match Wenz, even though specific dB values differ by a few dB from any single digitization of the original figure. The demo's verification in Section 5.1 tests against the formula values themselves, with an optional visual cross-check against Wenz Figure 13 as a secondary sanity test in Section 5.2.

### 2.5 Audio synthesis from a PSD

Given a target PSD `S(f)` in µPa^2/Hz, generate a continuous noise stream with that PSD as follows. (We work in normalized "audio units" not absolute Pa; the absolute scaling is set by the master volume.)

**Approach: filtered white noise via FIR.**

1. Generate a unit-variance white-Gaussian noise stream (mean 0, sigma 1).
2. Design a linear-phase FIR filter h_k whose magnitude response is `sqrt(S(f) / S_ref)`, where S_ref is a normalization constant (e.g., max(S) so the peak magnitude is 1).
3. Convolve the white noise with h_k.
4. Apply a gain g such that the output RMS matches the target total noise power, normalized to a comfortable listening level (e.g., total output RMS = 0.1 in floating-point amplitude).

For the demo we use **one filter per component** so the user can solo each one and adjust its level independently:

```
audio_out = g_master * (
    g_turb * (white_1 * h_turb) +
    g_ship * (white_2 * h_ship(s)) +
    g_wind * (white_3 * h_wind(U)) +
    g_thermal * (white_4 * h_thermal)
)
```

where each `white_k` is an independent noise stream (so the four components are uncorrelated, as they are physically), and `h_*` are FIR filters whose magnitude responses match the corresponding component PSDs. `g_*` are gains derived from the integrated PSDs to give the right relative levels.

### 2.6 Auditioning at 48 kHz

The Wenz spectrum extends to 200 kHz but audio playback is limited to Nyquist at the audio context sample rate (typically 24 kHz at 48 kHz sample rate, or 22.05 kHz at 44.1 kHz). The thermal component is essentially inaudible in the audio band: NL_th(1 kHz) = -15 dB (per Section 2.3); NL_th(10 kHz) = 5 dB; NL_th(20 kHz) = 11 dB. By comparison, even at U=0 the wind component is 44 dB at 1 kHz, so thermal contributes <1% of the linear power in the audible band. Include thermal in the plot (which can go up to 200 kHz where it dominates) but its contribution to the audio synthesis is negligible and can be omitted if performance matters.

---

## 3. UI design

### 3.1 Layout

```
+--- Header ----------------------------------------------------+
| Wenz Curves Ambient Noise Generator                           |
| <- All demos . Source code . Python reference                 |
+---------------------------------------------------------------+

+--- Plot ------------------------------------------------------+
|   log-log plot of spectrum level dB re 1 µPa^2/Hz vs freq:    |
|     turbulence       (purple, dashed)                          |
|     shipping         (orange, dashed)                          |
|     wind/surface     (blue,   dashed)                          |
|     thermal          (red,    dashed)                          |
|     SUM (theoretical) (black, solid)                           |
|     SUM (measured live from output, via AnalyserNode)          |
|                       (gray,  thin, real-time)                 |
|   x: 1 Hz to 200 kHz                                          |
|   y: -20 to 120 dB re 1 µPa^2/Hz                              |
+---------------------------------------------------------------+

+--- Sea state ----------+ +--- Shipping ----------+ +--- Output ----+
| Wind speed (m/s)  [-]  | | Density       [-]     | | Master vol[-] |
|  0 to 25 m/s           | |  None / Low / Med /Hvy| | Start  [ ▶ ]  |
| Beaufort:  3 (gentle)  | | Continuous 0..1       | |               |
+-----------------------+  +-----------------------+ +---------------+

+--- Component solo / mute ------------------------+
|  Turbulence    [x] on   [Solo]                   |
|  Shipping      [x] on   [Solo]                   |
|  Wind/surface  [x] on   [Solo]                   |
|  Thermal       [x] on   [Solo]                   |
|  Bubble bursts [ ] on   (optional, from bank)    |
+---------------------------------------------------+

+--- Hydrophone depth (optional advanced) ---+
|  Depth     [-]    1 to 5000 m              |
|  (modifies low-frequency shipping by 6 dB)  |
+---------------------------------------------+

[ Save spectrum (CSV) ]  [ Reset to defaults ]
```

### 3.2 Sliders and presets

| Parameter | Range | Default | Scale | Step |
|---|---|---|---|---|
| Wind speed | 0 to 25 m/s | 5 m/s | linear | 0.1 m/s |
| Shipping density | 0 to 1 | 0.3 | linear, with named ticks at 0 / 0.5 / 1 | 0.01 |
| Hydrophone depth | 1 to 5000 m | 100 m | logarithmic | (log) |
| Master volume | 0 to 1 | 0.3 | linear | 0.01 |

Display Beaufort number alongside the wind speed slider (e.g., "5.0 m/s = Beaufort 3, gentle breeze"). Use the WMO 1100 standard midpoint conversion for each band, in m/s (1 knot = 0.5144 m/s):

| Beaufort | Range (knots) | Range (m/s) | Midpoint (m/s) | Description |
|---|---|---|---|---|
| 0 | <1 | <0.5 | 0.3 | calm |
| 1 | 1-3 | 0.5-1.5 | 1.0 | light air |
| 2 | 4-6 | 2.0-3.1 | 2.6 | light breeze |
| 3 | 7-10 | 3.6-5.1 | 4.4 | gentle breeze |
| 4 | 11-16 | 5.7-8.2 | 7.0 | moderate breeze |
| 5 | 17-21 | 8.7-10.8 | 9.8 | fresh breeze |
| 6 | 22-27 | 11.3-13.9 | 12.6 | strong breeze |
| 7 | 28-33 | 14.4-17.0 | 15.7 | near gale |
| 8 | 34-40 | 17.5-20.6 | 19.0 | gale |
| 9 | 41-47 | 21.1-24.2 | 22.6 | strong gale |
| 10 | 48-55 | 24.7-28.3 | 26.5 | storm |
| 11 | 56-63 | 28.8-32.4 | 30.6 | violent storm |
| 12 | >=64 | >=32.9 | (open) | hurricane |

**Preset buttons** (above the sliders, like `bubble-soundbank`):

- **Calm Open Ocean** (U = 1 m/s, s = 0)
- **Typical Weather, Light Shipping** (U = 5 m/s, s = 0.3)
- **Storm at Sea** (U = 18 m/s, s = 0.1)
- **Busy Harbor** (U = 3 m/s, s = 1.0)
- **Deep, Quiet** (U = 2 m/s, s = 0, depth = 3000 m)

Clicking a preset animates the sliders to the new values over ~500 ms.

### 3.3 Plot

Use Chart.js loaded from CDN, log-log axes. Six curves:

- Four dashed component curves (turbulence, shipping, wind, thermal), each in a distinct color.
- One solid black curve for the theoretical sum (Eq. 2.2).
- One thin gray curve for the live measured spectrum from the audio output (AnalyserNode FFT, ~1 Hz update rate, time-averaged over the last 2 seconds).

The measured and theoretical sums should overlay closely (within ~3 dB above 30 Hz; below 30 Hz, FFT resolution limits comparison).

Plot range:
- x: 1 Hz to 200 kHz (component curves extend that far; live measurement only to fs/2)
- y: -20 to 120 dB

Light vertical guides at 100 Hz, 1 kHz, 10 kHz to anchor the eye.

When a component is muted or soloed, dim or hide its curve correspondingly. Soloing one component should remove the other three from the sum.

### 3.4 Optional bubble bursts

If the bubble checkbox is enabled, sparsely trigger bubble events from the bubble bank model (van den Doel 2005), with rate and size distribution depending on wind speed (more wind = more bubbles, larger sizes). This is a "Goldilocks" sweetener that makes the wind component sound like real breaking waves rather than just spectrally shaped noise.

Suggested implementation: when `bubbles_on && U > 3 m/s`, generate Poisson-distributed bubble events at rate `lambda = max(0, 5 * (U - 3))` events per second, with a size distribution biased toward small radii (r in 0.2-2 mm, gamma exponent ~3). Use the same damped-sinusoid bubble model from `bubble-soundbank/`; copy the code or refactor into a shared `bubbles.js` if convenient. Keep the bubble contribution small (~5-10% of total power) so it adds texture without dominating.

---

## 4. Audio implementation

### 4.1 Architecture

```
  white-noise-1 ─► turbulence FIR ────► gain_turb ────┐
  white-noise-2 ─► shipping FIR ──────► gain_ship ────┤
  white-noise-3 ─► wind FIR ──────────► gain_wind ────┤
  white-noise-4 ─► thermal FIR ───────► gain_thermal ─┤
  (bubbles)     ─► gain_bubbles ──────────────────────┤
                                                       ▼
                                                    Mixer ── master gain ── destination
                                                                            │
                                                                       AnalyserNode (for plot)
```

Each component is a separate signal path so the user can solo or mute individually with no audible coupling.

### 4.2 FIR filter design per component

For each component k:

1. Sample its target spectrum `NL_k(f)` at N = 2048 frequency bins from 0 to fs/2.
2. Convert to linear amplitude: `H_k(f) = 10^(NL_k(f) / 20)`. (Use 20, not 10, because we are designing an amplitude-response filter that will be applied to a unit-variance noise source.)
3. Normalize: divide by `sqrt(integrate(H_k^2 over Nyquist))` so the output is unit variance after filtering. Actual level is then set by `gain_k`.
4. IFFT (with cosine-mirror to ensure real impulse response).
5. Shift to center, apply a Hann window.
6. Truncate to a length that gives ~50 dB stopband attenuation. For thermal (steep), use 2048 taps. For others, 1024 or 512 may be enough.
7. Load each h_k into a Web Audio `ConvolverNode`.

Recompute h_ship whenever `s` changes; recompute h_wind whenever `U` changes. Turbulence and thermal are independent of all sliders so their FIRs are computed once at startup.

### 4.3 Optional: Hildebrand 2021 wind model

Replace the simple Coates/Stojanovic form for `NL_wind` with the Hildebrand 2021 per-Beaufort parameterization (Table III of Hildebrand, Frasier, Baumann-Pickering, Wiggins 2021):

```
NL_wind(f, U, depth) = A_n(f) + B_n(f) * log10(U) + depth_correction
```

where `A_n` and `B_n` are tabulated per Beaufort number and frequency, and `depth_correction` is a small depth correction (~+/-3 dB across 100-1000 m). This is more accurate against modern long-term ocean recordings but is parameter-heavy. Use it as a secondary mode behind a toggle "Hildebrand 2021 fit", with Coates/Stojanovic as the default for closed-form simplicity. The Hildebrand 2021 companion MATLAB code at <https://github.com/jahildebrand/WindNoise> can be ported to JavaScript for this mode.

### 4.4 Gain calibration

Set the per-component gains so that:
- With default sliders (U = 5 m/s, s = 0.3, all components on), the combined output sits at a comfortable listening level (RMS around -20 dBFS).
- The relative levels of the four components, as measured by the live AnalyserNode, match the theoretical Wenz components at the same settings to within 1 dB across the audible band.

Calibration procedure: in `test.html`, render 10 seconds of each component in isolation at unit gain, measure the RMS, compute the per-component normalization, and bake those numbers into the synthesis code.

---

## 5. Verification plan

`test.html` must implement and report pass/fail for the following.

### 5.1 Theoretical-curve numerical agreement

The plotted theoretical NL_total(f) at the conditions below should match the table values to within 1 dB. **The numbers below were computed directly from the Coates/Stojanovic formulas in Section 2.3 by the included Python reference, not transcribed from a textbook**; the JS port must reproduce them to within 0.5 dB. The "dominant" column indicates which component dominates the linear sum at that frequency, and the implementation should agree.

| U (m/s) | s | f (Hz) | NL_total (dB re 1 µPa^2/Hz) | Dominant component |
|---|---|---|---|---|
| 0.5  | 0.0 | 30     | 66.3  | ship |
| 0.5  | 0.0 | 100    | 58.0  | ship |
| 0.5  | 0.0 | 1000   | 49.5  | wind |
| 0.5  | 0.0 | 10000  | 34.6  | wind |
| 0.5  | 0.0 | 100000 | 25.4  | thermal |
| 5.0  | 0.5 | 30     | 74.1  | ship |
| 5.0  | 0.5 | 100    | 67.8  | ship |
| 5.0  | 0.5 | 1000   | 61.0  | wind |
| 5.0  | 0.5 | 10000  | 46.1  | wind |
| 12.0 | 0.0 | 30     | 67.2  | ship |
| 12.0 | 0.0 | 1000   | 70.1  | wind |
| 12.0 | 1.0 | 30     | 83.8  | ship |
| 12.0 | 1.0 | 1000   | 70.2  | wind |
| 1.0  | 1.0 | 30     | 83.8  | ship |
| 1.0  | 1.0 | 1000   | 53.6  | wind |

Note: at low wind speeds and low s values, the Coates/Stojanovic shipping term contributes substantial level even at s = 0 (this is a feature of the formula, representing "minimal ambient shipping noise" rather than "no shipping"). For a true "no shipping" scenario, mute the shipping component entirely via the mute toggle and recompute the table — without shipping the totals at 30 Hz drop substantially.

### 5.2 Visual match to Wenz Figure 13

Plot all components and the sum at U = 0.5, 2, 5, 8, 12 m/s with s = 0 (wind-only conditions), and at s = 0.3, 1.0 with U = 0.5 m/s (shipping-only conditions). Compare visually to Wenz 1962 Figure 13, reproduced in Deane 2025. The component curves and sum should overlay within ~5 dB across the band.

### 5.3 Live measured-vs-theoretical spectrum

Generate 30 seconds of output at default settings and measure the time-averaged PSD via the AnalyserNode FFT. Compare to the theoretical NL_total(f). They should agree to within 3 dB across 30 Hz to 20 kHz.

Below 30 Hz, the AnalyserNode FFT resolution is too coarse for a fair comparison; do not test there.

### 5.4 Solo and mute behavior

For each component:
- Solo it: measure the output PSD, confirm only that component is present.
- Mute it: confirm it disappears from the measured PSD.

The measured PSD with one component soloed should match that component's theoretical curve to within 3 dB.

### 5.5 Slider response

Sweep U from 0 to 25 m/s with s=0 (the Coates/Stojanovic "minimum traffic" baseline, not the muted state) while audio plays. The measured PSD level at 1 kHz should rise smoothly from ~44 dB (at U=0) to ~82 dB (at U=25) without clicks or glitches; the rise scales as 7.5 * sqrt(U) per the wind formula.

Sweep shipping density from s=0 to s=1 with U=5 m/s while audio plays. The PSD level at 30 Hz should rise smoothly from ~66 dB (at s=0) to ~84 dB (at s=1); the s=0 baseline includes both the Stojanovic shipping floor and the wind/turbulence contribution.

Also verify that muting shipping with the mute toggle (rather than just setting s=0) drops the 30 Hz level substantially compared to s=0: with shipping muted and U=5, the 30 Hz total drops from ~66 dB to whatever wind and turbulence contribute alone (~63 dB at U=5, 30 Hz, dominated by turbulence), demonstrating the difference between "low traffic" and "no traffic".

### 5.6 Cross-check against published spectrum levels

The Coates/Stojanovic formulas should reproduce these published sanity values to within ~5-10 dB:

| Condition | f | Expected (published) | Coates/Stojanovic |
|---|---|---|---|
| Sea state ≈ 0 (very calm, U=0.5 m/s, s=0), 1 kHz | 1 kHz | ~30-40 dB (Wenz Fig 13) | 50 dB (overshoots) |
| Sea state ≈ 3 (gentle breeze, U=5 m/s, s=0), 1 kHz | 1 kHz | ~50-55 dB | 61 dB |
| Sea state ≈ 6 (strong breeze, U=12 m/s, s=0), 1 kHz | 1 kHz | ~60-70 dB | 70 dB |
| Heavy shipping peak (s=1.0), 30 Hz | 30 Hz | ~80-95 dB | 84 dB |
| Thermal floor, 100 kHz | 100 kHz | ~25 dB (Mellen 1952) | 25 dB |

The Coates/Stojanovic fit overshoots Wenz Figure 13 by 5-10 dB at the lowest wind speeds and 1 kHz; this is a known characteristic of the formulas, which were calibrated for engineering use in underwater communications rather than as exact reproductions of Wenz. The shape and slider responses are pedagogically correct, and the absolute levels are consistent with published modern measurements (e.g., Hildebrand 2021 Figure 5 shows somewhat higher levels than Wenz Figure 13 at low Beaufort numbers).

---

## 6. About-and-references section (page footer)

Include this section at the bottom of the page, matching `bubble-soundbank/` style.

> ### About
>
> Ocean ambient noise is not a single sound. It is a sum of identifiable physical mechanisms, each with a distinct frequency dependence and physical driver. Wenz (1962) compiled the first systematic decomposition, dividing prevailing noise into four broad bands: turbulent pressure (very low frequency, hydrodynamic in origin), distant shipping (10-300 Hz, anthropogenic), wind and surface processes (100 Hz to 30 kHz, driven by breaking waves and bubble injection), and thermal noise (above 50 kHz, the equipartition floor set by the temperature of the water).
>
> This demo synthesizes each component using the Coates 1989 / Stojanovic 2007 closed-form parameterization, a standard engineering fit to Wenz's Figure 13 widely used in the underwater communications literature. The formulas reproduce the shapes and qualitative slider responses of the Wenz curves; they are within about 5-10 dB of any specific digitization of Wenz Figure 13 at the lowest wind speeds (modern measurements such as Hildebrand 2021 give somewhat higher levels than the original Wenz figure at low Beaufort numbers, partially closing the gap).
>
> Mute or solo individual components to hear what each one sounds like in isolation. With shipping muted and wind speed below 1 m/s, you hear essentially only the thermal floor (high-frequency hiss at the limit of equipartition). Turn shipping back on and crank to "Heavy" and the rumble of distant traffic emerges at 30-80 Hz. Crank the wind to 15 m/s and broadband noise from breaking surface waves takes over the entire mid-band.
>
> Optionally mix in bubble bursts from the bubble bank model (van den Doel 2005) to give the wind-driven component the natural texture of breaking surface bubbles, rather than the slightly artificial sound of pure spectrally-shaped noise.
>
> ### Slider parameters
>
> **Wind speed:** Speed of the wind blowing over the ocean surface, in m/s (or equivalently Beaufort number). Drives the surface noise band (100 Hz - 30 kHz). Breaking waves and the bubbles they inject under the surface are the primary radiation mechanism (Wenz 1962; Knudsen, Alford, Emling 1948; Carey and Evans 2011). The Coates/Stojanovic formula has the wind 1 kHz spectrum level scaling roughly as 7.5 * sqrt(U), giving a ~9 dB increase between gentle breeze (U=5 m/s) and strong wind (U=12 m/s).
>
> **Shipping density:** Approximate density of distant commercial shipping, parameterized by the Coates/Stojanovic shipping activity factor s in [0, 1]. Drives the 10-300 Hz band. Anthropogenic; has measurably increased in the global ocean since the 1960s (Andrew et al. 2002, McDonald et al. 2006). At s=1 with heavy traffic in busy lanes, the rumble can dominate the spectrum below 200 Hz. The slider does not go below s=0; to silence shipping entirely (rather than reduce to "low traffic" levels), use the mute toggle.
>
> **Hydrophone depth** (optional): Depth of an idealized hydrophone. Shallow water sees more wind and surface activity; deep water sees relatively more shipping (which propagates well in the SOFAR channel) and less wind. Implemented in this demo as a coarse low-frequency level adjustment, not a propagation model.
>
> **Component solo / mute:** Use these to isolate the contribution of each physical mechanism. Especially illuminating: solo "Thermal" at default master volume and turn up the volume until you can hear the high-frequency hiss; this is the absolute floor of any underwater hydrophone, set by water temperature alone.
>
> ### References
>
> 1. **Wenz, G. M.** (1962). "Acoustic ambient noise in the ocean: spectra and sources." *J. Acoust. Soc. Am.* 34(12), 1936-1956. [doi:10.1121/1.1909155](https://doi.org/10.1121/1.1909155)
> 2. **Deane, G. B.** (2025). "The Wenz curves for underwater ambient sound." *J. Acoust. Soc. Am.* 157(5), R9-R10. [doi:10.1121/10.0036690](https://doi.org/10.1121/10.0036690)
> 3. **Hildebrand, J. A., K. E. Frasier, S. Baumann-Pickering, S. M. Wiggins** (2021). "An empirical model for wind-generated ocean noise." *J. Acoust. Soc. Am.* 149(6), 4516-4533. [doi:10.1121/10.0005430](https://doi.org/10.1121/10.0005430). Companion code: <https://github.com/jahildebrand/WindNoise>.
> 4. **Knudsen, V. O., R. S. Alford, J. W. Emling** (1948). "Underwater ambient noise." *J. Marine Res.* 7, 410. (Yale University Press.)
> 5. **Mellen, R. H.** (1952). "The thermal-noise limit in the detection of underwater acoustic signals." *J. Acoust. Soc. Am.* 24(5), 478-480. [doi:10.1121/1.1906924](https://doi.org/10.1121/1.1906924)
> 6. **Coates, R. F. W.** (1989). *Underwater Acoustic Systems*. Macmillan New Electronics Series. ISBN 978-1-349-20508-0. The closed-form ambient-noise component parameterization compiled here.
> 7. **Stojanovic, M.** (2007). "On the relationship between capacity and distance in an underwater acoustic channel." *ACM SIGMOBILE Mobile Computing and Communications Review* 11(4), 34-43. The widely-cited reference for the closed-form Coates/Stojanovic ambient noise PSD formulas used in this demo.
> 8. **Ainslie, M. A.** (2010). *Principles of Sonar Performance Modelling*. Springer-Praxis. ISBN 978-3-540-87661-8 (hardcover), [doi:10.1007/978-3-540-87662-5](https://doi.org/10.1007/978-3-540-87662-5).
> 9. **van den Doel, K.** (2005). "Physically Based Models for Liquid Sounds." *ACM Trans. Applied Perception* 2(4), 534-546. [doi:10.1145/1101530.1101554](https://doi.org/10.1145/1101530.1101554) (Bubble model used in the optional bubble layer.)

---

## 7. File layout

```
demos/wenz-ambient-noise/
├── index.html              the demo
├── ATTRIBUTION.md          source acknowledgements
├── DEMO_DISCLAIMER.md      boilerplate, copy from existing demos
├── python/
│   ├── wenz_reference.py   verified Python implementation of the four
│   │                       component PSDs and the sum
│   └── README.md           how to run, expected test table
├── data/
│   └── wenz_fig13.png      (optional) digitized Wenz 1962 Fig 13 for
│                            on-page visual comparison; CC-BY or own digitization
└── test.html               automated verification harness
```

Single-file preferred per `CLAUDE.md`; split only if `index.html` exceeds about 1000 lines.

---

## 8. Attribution requirements

1. **Mathematical formulation** is from Wenz 1962, Knudsen 1948, Mellen 1952, Coates 1989, Stojanovic 2007, and Hildebrand 2021; cite all in the page footer (references 1, 4, 5, 6, 7, 3 above).
2. **The bubble layer** uses code from `bubble-soundbank/` which is in turn after van den Doel 2005. If you copy bubble code, preserve van den Doel's BSD-2-Clause license and attribution, and cite reference 9 above.
3. **Any digitized Wenz Figure 13** must be marked clearly as "digitized from Wenz 1962" with a note that the original figure is copyright Acoustical Society of America 1962. Reprinted in Deane 2025 with permission. If unsure, omit the digitized figure and use a pure-text reference; the figure does not need to appear on the page for the demo to work.
4. **Code license:** BSD-2-Clause, consistent with the rest of `dougjam/demos`. Top of `index.html`: `<!-- (c) 2026 Doug James, Stanford University. BSD-2-Clause. -->`.

---

## 9. Acceptance criteria

Done when:

1. All 15 entries in the numerical accuracy table (Section 5.1) pass in `test.html` within 1 dB.
2. Visual overlay of components and theoretical sum against Wenz Figure 13 at five wind speeds and two shipping densities is presented; Coates/Stojanovic curves are within about 10 dB of any single digitization of Wenz Figure 13, with consistent shapes.
3. Live measured PSD matches theoretical sum within 3 dB across 30 Hz - 20 kHz (Section 5.3).
4. Solo and mute behavior works correctly for all four components (Section 5.4).
5. Wind and shipping sliders sweep smoothly with no audio artifacts (Section 5.5).
6. Five presets load and animate.
7. Optional bubble layer works when enabled and is clearly absent when disabled.
8. The page works in current Chrome, Firefox, and Safari at desktop laptop resolutions (1280x800 minimum).
9. The page footer About and references section is present and accurate.
10. The Python reference in `python/` runs and produces the test table.
11. The root `index.html` of the demos gallery is updated with an entry pointing to this demo.

---

## 10. Suggested implementation order

1. Port Section 2.3 PSD formulas to JavaScript. Get the four `NL_*(f, ...)` functions right.
2. Build `test.html` and verify all 15 entries pass against the Python reference. Do not move on until this passes.
3. Build the Chart.js plot with the four component curves plus theoretical sum. Visually verify against Wenz Figure 13.
4. Add the sliders and presets; wire them to the plot. Test smooth updates.
5. Implement the FIR filter design (Section 4.2) and verify the impulse responses give the right frequency response when tested in isolation.
6. Wire up the audio graph: white noise sources, ConvolverNodes, gains, mixer, master.
7. Add the AnalyserNode and overlay the live measured spectrum on the plot. Verify it tracks the theoretical sum within 3 dB.
8. Add solo and mute toggles. Verify they work cleanly in audio.
9. Add slider response testing: sweep wind and shipping while audio plays.
10. Add optional bubble layer (if time permits; not blocking).
11. Write the About-and-references footer.
12. Cross-browser test.
13. Update the gallery index.

Stop after each step and re-run the test harness to catch regressions early.

---

## 11. Known limitations to document on the page

- The Coates/Stojanovic wind fit is approximate and overshoots the original Wenz 1962 Figure 13 at the lowest wind speeds by 5-10 dB. Modern measurements (Hildebrand 2021) give somewhat higher levels than Wenz Figure 13 at low Beaufort numbers, partially closing the gap. The optional Hildebrand 2021 mode in Section 4.3 gives more accurate values against modern data; the default Coates/Stojanovic mode is for closed-form simplicity and is the standard fit in the underwater communications literature.
- The shipping spectrum varies enormously by location, time of day, and traffic geometry. The fit here is an average; real recordings show individual ship spectra peaks, harmonic structure from propellers, and time variation that this demo does not model. For ship-specific modeling, see Wales and Heitmeyer 2002.
- The demo does not model marine mammal vocalizations, ice cracking, snapping shrimp, rain, or other intermittent or local sources from Wenz's Figure 14. Adding any of these as a "Rare events" layer is a possible extension.
- Hydrophone depth is implemented as a simple low-frequency adjustment, not a proper propagation model. For accurate depth dependence, a normal-mode or PE solver would be needed (see Section 3.4 of `LECTURE_AtmosphericAndUnderwaterSound.md`).
- Geographic and depth-of-water variation in the Wenz curves is real and substantial; this demo synthesizes an "open mid-latitude ocean" average.
