# Algorithm notes

A summary of the sound texture synthesis algorithm (Algorithm 2,
&sect;5) of Chadwick and James, &ldquo;Animating Fire with Sound&rdquo;
(SIGGRAPH 2011), plus the per-rule fidelity log of this Python and
JavaScript port against the released C++ reference.

Canonical reference: <https://www.cs.cornell.edu/projects/Sound/fire/>

## Goal

Given a low-frequency physically based pressure signal `p_S(t)` (under
the simplifying assumptions of paper &sect;3 it is the time derivative
of the velocity flux through the moving flame front, equivalently the
time derivative of the integrated heat release rate; band-limited to
the flame solver's Nyquist, &asymp; 180 Hz at a 360 Hz time-stepping
rate) and a real fire-audio recording `p_T(t)` (the &ldquo;training&rdquo;
signal), synthesize a broadband output that:

- preserves the slow, low-frequency dynamics of `p_S`: bursts in
  the simulation appear as bursts in the output;
- exhibits the perceptual texture (spectral shape, bandwidth, micro-
  structure) of `p_T`;
- avoids spectral discontinuities at window boundaries.

The approach is a 1-D adaptation of the multi-resolution texture
synthesis method of Wei and Levoy (2000): build Gaussian pyramids of
both signals, then synthesize the output coarse-to-fine, choosing each
output window via a nearest-neighbour lookup into a per-level dictionary
built from the training pyramid.

## Pipeline (mirrors `srcOrig/texture/WindowSynthesizer.cpp`)

Inputs: `p_S` (base signal), `p_T` (training signal). Parameters:
`numLevels` (default 6), `windowHW` (default 4 samples), `featureHW`
(default 3 samples), `falloff` (default 0), `scaleCDF` (default true),
`scalingAlpha` (default 1).

1. **Padding and pyramids.** Pad each signal to the smallest
   `(2^k + 1)` length that contains it (centered, with the training
   signal's flanks reflected and the base signal's flanks zero-padded).
   Build a `numLevels`-level Gaussian pyramid of each: filter with the
   5-tap kernel `[0.05, 0.25, 0.40, 0.25, 0.05]` and decimate 2:1, so
   level `&ell;+1` has `(N&minus;1)/2 + 1` samples.

2. **Optional CDF init (&sect;5.3).** If `scaleCDF` is on, compute the
   sorted absolute amplitudes of the top (coarsest) pyramid level for
   both signals: these are the "input" and "output" CDFs used during
   per-window matching for dynamic-range mapping.

3. **Build training-feature dictionaries.** For each level, build a
   list of feature vectors over all training-signal windows; build a
   KD-tree for fast nearest-neighbour lookup.

4. **Initialise the output pyramid.** Copy `p_S`'s pyramid into the
   output pyramid, then zero every level except the top
   `numBaseLevels` (which always equals 1 in the demo, so only the
   coarsest level is preserved from the input).

5. **Coarse-to-fine synthesis.** For
   `&ell; = numLevels - 2, numLevels - 3, &hellip;, 0`:
   for each window `i = 0, &hellip;, numWindows-1` at level `&ell;`:

   1. Build a **feature vector** for the output's window at
      `(level=&ell;, window=i)`. The vector concatenates:
      - **Causal context from level &ell;**: the
        `windowHW &times; (featureHW + 1)` samples to the **left** of
        the window's centre (already-synthesised at coarser levels via
        the EXPAND-then-blend, or already-stitched at this level by
        previous windows);
      - **Coarser-level context from level &ell;+1**:
        `2 &middot; windowHW' &middot; (featureHW' + 1) + 1` symmetric
        samples around the window's centre (linearly interpolated since
        the coarser level is downsampled by 2);
      - All entries optionally weighted by an exponential
        `exp(&minus;falloff &middot; |distance|)`.
   2. **CDF rescale (only when `&ell; + 1` is the top level and
      `scaleCDF` is on)**: compute the average magnitude of the
      coarser-level entries, look up its percentile in the output CDF,
      look up the matching amplitude in the input (training) CDF, and
      take the ratio as `scaling`. Blend with `scalingAlpha`. Multiply
      the coarser-level feature entries by `scaling` (so the
      nearest-neighbour query is performed in a normalised space) and
      remember `1/scaling` to apply when blending the matched window.
   3. **Nearest-neighbour query**: find the training window whose
      feature vector minimises Euclidean distance to the query.
   4. **Hat-window blend**: add the matched training window into
      the output level via a triangular blend of half-width `windowHW`,
      multiplied by the saved `1/scaling` factor.

6. **Reconstruction.** The output signal is the trimmed bottom level
   of the output pyramid, restricted to the original (pre-pad) sample
   range.

## Parameter cheat sheet

From every bundled `srcOrig/work/{name}/default.xml`:

| Parameter | Default | Meaning |
|---|---|---|
| `Fs` | 44100 Hz | Sample rate (all signals are 44.1 kHz mono). |
| `numLevels` | 6 | Pyramid depth (6 levels &rArr; coarsest level has `Fs / 32` &asymp; 1.4 kHz Nyquist). |
| `windowHW` | 4 samples | Output-window half-width at each level; stride between windows. |
| `featureHW` | 3 samples | Feature half-width: each level contributes `windowHW &middot; (featureHW + 1)` samples to the feature. |
| `epsANN` | 1.0 | Approximation tolerance for ANN search in the original C++. **This port uses exact NN; the parameter is documented but ignored.** |
| `falloff` | 0.0 | Exponential weight on feature dimensions: `exp(&minus;falloff &middot; |i &minus; centre|)`. 0 means uniform. |
| `scaleCDF` | 1 | Enable dynamic-range mapping (see &sect;5.3). |
| `scalingAlpha` | 1.0 | Blend factor for the CDF scaling: `(1 - &alpha;) + &alpha; &middot; scaling`. |

## Port differences from the released C++

The Python and JavaScript ports follow the C++ reference function-for-
function with the following intentional differences. Each is logged so
that future readers can tell what changed and why.

### Exact nearest-neighbour search

The released C++ defaults to ANN's `(1+&epsilon;)`-approximate search
with `&epsilon; = 1.0` (very loose). This is inherently
non-deterministic across implementations, which would block any
bit-level Python &harr; JavaScript parity guarantee. Both ports here
use **exact** nearest-neighbour search:

- Python: `scipy.spatial.cKDTree.query(k=1)` (exact, deterministic).
- JavaScript: a hand-port of the bundled
  `srcOrig/datastructure/KDTreeMulti.cpp` template.

The C++ also bundles `KDTreeMulti.cpp` as the exact-NN alternative path;
this port is faithful to that path, just made the default. The audible
difference between exact NN and `&epsilon;=1.0` ANN is small (both find
nearby training windows), but the two are not bit-equivalent.

### Off-by-one in `buildGaussianLevel`

The C++ filter loop in
`srcOrig/texture/GaussianPyramid.cpp::buildGaussianLevel` (line 876)
uses `for (j = 0; j <= num_components; j++)` rather than `< num_components`.
That iterates one step past the end of both the filter stencil and the
clipped data slice, reading whatever happens to be in adjacent memory.
On a contiguous `std::vector<double>` the data read is the next sample
of the input (which gets multiplied by an undefined value: whatever
sits one past the end of the stencil array). It produces a small
deterministic perturbation per pyramid level on the C++ build, but is
unportable.

This port fixes the off-by-one (`< num_components`). Pyramid levels
therefore differ from a C++ run at the level of the filter's tail
contribution. Audible difference is well below the noise floor; the
Python port becomes the new reference and the test suite pins it.

### Dynamic-range mapping (`scaleCDF`)

Faithful to the C++ behaviour, with one defensive guard added:

- `&sect;5.3` algorithm: at the top pyramid level only, compute the
  average magnitude of the coarser-level feature entries. Find its
  percentile in the *output* CDF; look up the matching amplitude in the
  *input* CDF; take the ratio as `scaling`; rescale the
  coarser-level feature entries by `scaling`; remember `1/scaling` for
  the matched-window blend.
- Defensive guard: when `averageMagnitude` is below
  `AMPLITUDE_CUTOFF = 1e-4`, skip the rescale entirely (`scale = 1`).
  The C++ does the same (see `computeWindowFeature` line 509).
- The `scalingVector` debug array exposed by the C++ for a per-window
  scaling trace is reproduced verbatim under the same name in the
  Python diagnostics dict.

### Per-example demo override: torch and candle's `scalingAlpha`

The browser demo loads `torch` and `candle` with `scalingAlpha = 0.5`
instead of the C++ canonical `1.0`. This is a perceptual fix for a documented failure
mode of the &sect;5.3 dynamic-range mapping (Chadwick &amp; James &sect;6:
&ldquo;in some instances the method still has difficulty producing a
suitable, temporally coherent output sound. This can occur in cases
when the low-frequency input has a very wide dynamic range, while the
training data has a small range&rdquo;).

Torch's base signal has peak 0.76 and RMS 0.12 (wide range); its
training audio has peak 0.74 but RMS 0.047 (mostly quiet, rare bursts).
At `scalingAlpha = 1.0`, the per-window matched training windows get
amplified by similar large factors across many consecutive windows,
and the triangular overlap-add reinforces them into an audible tonal
buzz at the windowHW stride frequency. `scalingAlpha = 0.5` blends in
half the rescale and avoids the artefact while preserving the
qualitative effect.

The override lives in `python/tools/vectors_to_assets.py` (in the
`DEMO_OVERRIDES` dict, with rationale in a `_note` field that survives
into the per-example `default.json`). Tier 2 goldens use the canonical
`scalingAlpha = 1.0` because they're algorithm regression tests, not
perceptual references.

### Boundary handling in pyramid padding

The C++ reflects the boundaries of the **training** signal (so its
texture wraps cleanly) and zero-pads the **base** signal (so its
silent flanks remain silent). This port preserves both choices.

### Feature-vector dimensions

For level `&ell;`:

- causal context contributes `windowHW[&ell;] &middot; (featureHW[&ell;] + 1)` dims;
- if `&ell; < numLevels - 1`, coarser context adds
  `2 &middot; windowHW[&ell;+1] &middot; (featureHW[&ell;+1] + 1) + 1` dims.

Defaults (`windowHW=4`, `featureHW=3`) give 16 + 33 = 49 dimensions per
feature on intermediate levels and 16 dimensions at the top level.

### Performance notes (Python)

`scipy.spatial.cKDTree` is implemented in C and very fast. Expected
timings on the bundled examples (Python, default parameters):

- Pyramid construction: < 100 ms total per signal.
- KD-tree build per level: ~50 ms for the largest level (~180&times;10^3 features).
- Nearest-neighbour queries: ~5&micro;s each, &times; ~50&times;10^3 windows
  total &asymp; 250 ms.
- Total wall time per example: 1&ndash;5 s.

The C++ reference reports 20&ndash;86 s per example using ANN approximate
search; cKDTree's exact search is faster on these dimensions because
NumPy's tight inner loops avoid the per-call overhead of the C++
template indirection.

## In-band magnitude is *not* halved

Unlike the spectral bandwidth extension demo at
`../fire-bandwidth-extension/`, the texture synthesis pipeline never
applies a one-sided lowpass to a two-sided FFT, so it does not exhibit
the &minus;6.02 dB in-band drop documented there. Output amplitudes match
the input on average modulo the CDF scaling.

## Reference

- Cornell project page: <https://www.cs.cornell.edu/projects/Sound/fire/>
- Local copy of the paper: [`2011 Chadwick - Fire Sound_rekt.pdf`](2011%20Chadwick%20-%20Fire%20Sound_rekt.pdf)
- Canonical paper PDF: <https://www.cs.cornell.edu/projects/Sound/fire/FireSound2011.pdf>
- This spec: [`../SPEC_FireTextureSynthesis.md`](../SPEC_FireTextureSynthesis.md)
- Sibling demo (&sect;4 spectral bandwidth extension):
  <https://dougjam.github.io/demos/fire-bandwidth-extension/>
