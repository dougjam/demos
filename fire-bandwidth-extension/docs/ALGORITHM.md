# Algorithm notes

A one-page summary of Algorithm 1 (spectral bandwidth extension) from
&sect;4 of Chadwick and James, &ldquo;Animating Fire with Sound&rdquo;
(SIGGRAPH 2011), plus the design choices and known port differences in
this implementation.

Canonical reference: <https://www.cs.cornell.edu/projects/Sound/fire/>

## Goal

Given a low-frequency physically based pressure signal `p(t)` (under
the simplifying assumptions of paper &sect;3 it is the time derivative
of the velocity flux through the moving flame front, equivalently the
time derivative of the integrated heat release rate; band-limited to
the flame solver's Nyquist, &asymp; 180 Hz at a 360 Hz time-stepping
rate), synthesize a perceptually plausible broadband signal by adding
mid- and high-frequency power-law noise that tracks the input envelope.

The empirical justification for the `f^{-alpha}` power law over fire and
turbulent-premixed-flame combustion noise comes from:

- Clavin and Siggia (1991): theoretical `f^{-5/2}`.
- Rajaram and Lieuwen (2009): measured `alpha` in `[2.1, 3.4]`.

So `alpha = 2.5` is the physically motivated default.

## Pipeline (per spec &sect;2.1, mirroring `srcOrig/matlab/extend_signal.m`)

1. **Lowpass `p_lowpass = lowpass_filter(p, fs, f_cutoff)`.**
   Order-2 Butterworth at `f_cutoff` (default 180 Hz), zero-phase via
   `filtfilt`. This is the slow envelope used to amplitude-modulate the
   noise so that bursts in the input drive bursts in the high frequencies.

2. **Build a single global noise field.**
   - `powerlaw[k] = (k * fs / NFFT)^{-alpha/2} * exp(i * 2*pi*phi[k])`
     with random phases `phi[k] ~ U[0,1)`. (Per the released code's
     `% FIXME` comment, the magnitude uses `x^exponent`, not
     `(2*pi*x)^exponent`. The constant cancels in the per-window
     `beta` solve, so audible output is identical.)
   - Multiply by the high-pass blend filter `blend2`.
   - `noise_unscaled = ifft(powerlaw * blend2)`.

   This noise is generated **once** before the window loop. Reusing it
   across windows is what produces the coherent overlap-add output;
   independent draws per window would smear the spectrum.

3. **Per-window loop**, with a triangular window of half-width
   `half_width` (default 500 samples), 50% overlap, COLA-to-1.

   For window `i` covering samples `[c_i - h, c_i + h]`:
   1. `psub = p .* w_i`: windowed input.
   2. `psub_lowpass = |p_lowpass| * w_i`: envelope, zero-padded to NFFT.
   3. `psub_noise = psub_lowpass * noise_unscaled`: envelope-shaped noise.
   4. `Y = fft(psub)`, `Y_noise = fft(psub_noise)`.
   5. `Y_L = Y * blend1`, `Y_N = Y_noise * blend2`: the lowpass and
      highpass branches; `Y_S = Y` (unfiltered) is the target spectrum.
   6. `beta_i = fit_dual_power_spectra(x, Y_L, Y_N, Y_S, f_blur)` solves
      a quadratic
      &nbsp;&nbsp;`a beta^2 + b beta + c = 0`
      &nbsp;&nbsp;`a = I[Y_N_r^2 + Y_N_i^2]`,
      &nbsp;&nbsp;`b = 2 I[Y_L_r Y_N_r + Y_L_i Y_N_i]`,
      &nbsp;&nbsp;`c = I[|Y_L|^2 - |Y_S|^2]`,
      where `I[.]` is the trapezoidal integral over the full two-sided
      frequency axis weighted by a Gaussian `f_blur` centered at
      `fit_center` (default 180 Hz, sigma = `fit_width/3` = 10 Hz). The
      stable Vieta branch is used (`alpha1 = (-b - sgn(b) sqrt(disc))/(2a)`,
      `alpha2 = c / (alpha1 a)`); when `c < 0` the two roots have opposite
      sign and `beta = max(alpha1, alpha2)` picks the positive one.
   7. `psub_extended = real(ifft(Y_L + beta_i * noise_amplitude * Y_N))`.
   8. Overlap-add `psub_extended` into the running `p_extended`.

## Design rules followed in this port (spec &sect;2.1)

The Python port follows these to remain bit-comparable with the Matlab
reference. The JavaScript port follows them too, with one optional
relaxation (the per-window FFT size; see below).

1. Read every `srcOrig/matlab/*.m` first; translate function-by-function.
2. Two-sided FFTs (`np.fft.fft` / `np.fft.ifft`), not real-to-half.
3. `NFFT = 2 ** ceil(log2(L))` (Matlab `nextpow2`).
4. Frequency axis `x = fs * linspace(0, 1, NFFT)` (Matlab convention).
5. Order-2 Butterworth + `filtfilt` for the lowpass.
6. Triangular window with peak 1; adjacent windows COLA-to-1; no final divide.
7. Power-law noise generated once, multiplied by `blend2` in the
   frequency domain, IFFT'd before the window loop.
8. Envelope multiplier is `|p_lowpass| * w_i`, not `|p * w_i|`.
9. `Y_S` in the quadratic is the unfiltered windowed signal.
10. Stable Vieta branch on the sign of `b`; `beta = max(beta1, beta2)`.
11. Powerlaw magnitude uses `x^exponent`, matching the released code
    (the alternative `(2*pi*x)^exponent` is an identical-output choice
    because the constant cancels in `beta`).

## In-band magnitude is halved (Matlab-faithful)

The Matlab applies the one-sided ``blend1`` lowpass to the **two-sided**
FFT of ``psub`` (so ``blend1`` is 1 for ``x in [0, blend_start]`` and 0 for
``x in [blend_end, fs]``). This zeros the negative-frequency mirror of the
in-band signal. After the final ``real(ifft(...))`` step, the in-band
magnitude is halved (a 6.02 dB drop). The Matlab reference exhibits the
same behavior; it is masked in practice because the released
``extend_signal`` example call peak-normalizes the output before saving:

```matlab
M = max( abs( p_extended ) );
p_extended = p_extended / ( M * 1.01 );
```

This port preserves the behavior. The Tier 3 ``test_sub_cutoff_sine_preserved``
test asserts the &minus;6.02 dB drop within 0.5 dB rather than asserting
unit-gain preservation.

## Degenerate beta-quadratic cases

The released Matlab ``fit_dual_power_spectra`` does not handle the case
where the noise spectrum is identically zero (``a == 0``; e.g., the COLA
identity check with ``blend2`` forced to zero) or where the windowed input
is silent (which also makes ``beta1 = 0`` and divides by zero in the Vieta
step). This port returns ``beta = 0`` in both cases, which is the
consistent choice (no noise to add). The Matlab would simply error out.

## Port differences from the released Matlab

The Python port's defaults exactly reproduce the README example call.

The Python port adds two parameters that do not affect the default behavior:

- `phases_override: np.ndarray | None = None`. When supplied, the
  `build_powerlaw_spectrum` call uses these phases verbatim instead of
  drawing from the RNG. Used by the Tier 2 phase-controlled tests so
  Python and JavaScript can produce bit-comparable output from the same
  fixed-phase array.
- `nfft_per_window: int | None = None`. **Reserved for the JS port only**;
  the Python port currently raises `NotImplementedError` if this is not
  `None`. The Python implementation always runs the Matlab-faithful path
  (`NFFT = next_pow2(L)`) so that Tier 1/2/3 tests are pinned to the
  reference behavior. Tier 3 statistical tests use shorter signals
  (~0.5 s) instead of a smaller FFT to keep runtime reasonable.

The JavaScript port implements the small per-window FFT optimization
internally. By default the worker uses
`nfft_per_window = next_pow2(8 * halfWidth)` (typically 4096 or 8192) so
that 10-second clips process in well under 5 s. The browser checkbox
labelled "match Matlab reference (slow on long clips)" falls back to the
full-signal NFFT (`next_pow2(L)`); the in-browser parity test
(`tests/validate.html`) uses that mode so the JS output can be compared
bit-for-bit with the Python-generated golden.

The two paths differ only in the discretization of the trapezoidal
integral inside `fit_dual_power_spectra`: the small-NFFT path evaluates
the integral on a coarser frequency grid. On all tested signals the
audible difference is well below the noise floor.

## Reference

- Cornell project page: <https://www.cs.cornell.edu/projects/Sound/fire/>
- Local copy of the paper: [`2011 Chadwick - Fire Sound_rekt.pdf`](2011%20Chadwick%20-%20Fire%20Sound_rekt.pdf)
- Canonical paper PDF: <https://www.cs.cornell.edu/projects/Sound/fire/FireSound2011.pdf>
- This spec: [`../SPEC_SpectralBandwidthExtension.md`](../SPEC_SpectralBandwidthExtension.md)
