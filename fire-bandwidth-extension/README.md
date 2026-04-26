# Spectral Bandwidth Extension: Fire Sound Demo

A faithful Python port and a self-contained interactive browser demo of
the spectral bandwidth extension algorithm (Algorithm 1, &sect;4) from:

> **Jeffrey N. Chadwick and Doug L. James.** *Animating Fire with Sound.*
> ACM Transactions on Graphics (SIGGRAPH 2011), 30(4), August 2011.
> Project page: <https://www.cs.cornell.edu/projects/Sound/fire/>

The algorithm extends a low-frequency physically based pressure signal
(the time derivative of the velocity flux through the moving flame
front, under the simplifying assumptions of paper &sect;3; band-limited
to the flame solver's Nyquist, &asymp; 180 Hz at a 360 Hz time-stepping
rate) into a broadband, perceptually fire-like signal by adding
amplitude-modulated power-law noise (`f^{-alpha}` with
`alpha &asymp; 2.5`) above the lowpass cutoff.

## Citation

```bibtex
@article{ChadwickJames2011,
  author  = {Chadwick, Jeffrey N. and James, Doug L.},
  title   = {Animating Fire with Sound},
  journal = {ACM Transactions on Graphics (SIGGRAPH 2011)},
  volume  = {30},
  number  = {4},
  year    = {2011},
  month   = {aug}
}
```

## Quickstart

### Browser demo

Browsers block `fetch()` and Web Workers when a page is opened from a
`file://` URL, so the demo needs a local HTTP server. From this directory:

```bash
python serve.py 8765
```

then visit <http://127.0.0.1:8765/> (or use the live deployment at the
parent demos repo's GitHub Pages URL). The five released flame-simulation
signals (`burning_brick`, `candle`, `dragon`, `flame_jet`, `torch`) are
bundled as Float32 binaries under `assets/` and appear as buttons; a
built-in **Synthetic burst** preset is also available so the page is
interactive immediately. You can also load your own WAV file.

Slider controls: `alpha` (power-law exponent), lowpass cutoff
`f_cutoff`, per-window half-width, Gaussian fit width, blend transition
half-width, noise amplitude, RNG seed, plus a "match Matlab reference"
checkbox that swaps the small per-window FFT for the full-signal FFT
(slower, exact-Matlab equivalent). Optional **auto-process** checkbox
re-runs the algorithm 250 ms after every slider change. **Download**
buttons save either the original or the extended signal as a 16-bit WAV.

The page also shows a live spectrum plot of the input signal vs. the
power-law extension (with the &beta; matching scale recomputed live), a
per-window &beta; trace, and stacked input/extended spectrograms. The
"About this sound model" disclosure at the bottom of the page summarizes
the algorithm and its assumptions in student-friendly terms.

### Python port

From the project root:

```bash
python python/bandwidth_extension.py srcOrig/work/dragon.vector dragon_ext.wav --alpha 2.5 --seed 42
python python/bandwidth_extension.py input.wav output.wav --alpha 2.5
```

The CLI accepts both `.wav` and `.vector` (Matlab single-column
double-precision with an `int32` row-count prefix; see
`python/vector_io.py`) for input and output.

### Tests

```bash
pip install numpy scipy pytest
pytest python/tests/
```

Three tiers (per `SPEC_SpectralBandwidthExtension.md`):

- **Tier 1**: deterministic kernel goldens (`atol=rtol=1e-13`).
- **Tier 2**: full-pipeline phase-controlled goldens for 5 signals
  &times; 3 alphas, used as the JS-vs-Python parity target.
- **Tier 3**: statistical smoke tests (COLA identity, silent
  input, sub-cutoff sine preservation, log-log slope check, envelope
  sync via Hilbert correlation, byte-identical determinism).

The goldens in `python/tests/golden/` are produced by the Python port
itself (`python/tools/generate_goldens.py`); they pin the Python port
against regression and act as the parity target the JavaScript port must
match. The original Matlab source under `srcOrig/matlab/` is kept
unmodified as a reference for the line-by-line port.

## Layout

```
fire-bandwidth-extension/
├── index.html, main.js, bandwidth_extension.js, deterministic_rng.js,
│   fft.js, style.css      browser demo (served at this directory)
├── worker.js              standalone Web Worker entry point (the demo
│                          itself bundles the same code via a Blob URL
│                          to bypass browser caching, but worker.js is a
│                          standalone reference for the message protocol)
├── serve.py               no-cache HTTP server for local development
├── thumb.jpg              gallery thumbnail
├── assets/                bundled Float32 .bin signals + manifest.json
├── tests/                 browser-side parity validation
│                          (validate.html + golden_synthetic.json)
├── python/
│   ├── bandwidth_extension.py     the port (ground truth)
│   ├── deterministic_rng.py       PCG32 (shared with deterministic_rng.js)
│   ├── vector_io.py               .vector reader/writer
│   ├── tools/                     golden generator, asset packer
│   └── tests/                     three-tier test suite
├── srcOrig/                       provided Matlab source (do not modify)
│   ├── README, matlab/, work/
├── docs/
│   ├── ALGORITHM.md               algorithm + port-fidelity notes
│   ├── DEMO_DISCLAIMER.md         course-wide demo disclaimer
│   └── 2011 Chadwick - Fire Sound_rekt.pdf
├── ATTRIBUTION.md, ATTRIBUTION.html
├── LICENSE                        BSD 2-Clause (Chadwick 2011 + ports)
├── SPEC_SpectralBandwidthExtension.md
└── README.md
```

## License

BSD 2-Clause. See [`LICENSE`](LICENSE) and [`ATTRIBUTION.md`](ATTRIBUTION.md).

## Acknowledgments

This work is a derivative of the Matlab reference implementation
released by **Jeffrey N. Chadwick** and **Doug L. James** at
<https://www.cs.cornell.edu/projects/Sound/fire/>. Both the algorithm
and the five bundled test signals are theirs.

The Python and JavaScript ports and the browser demo were developed
with Claude Code for **CS 448Z (Physically Based Animation and Sound),
Stanford University, Spring 2026**, taught by Doug L. James.

## References

1. Chadwick, J. N., and James, D. L. (2011). *Animating Fire with Sound.*
   ACM Transactions on Graphics (SIGGRAPH 2011), 30(4), August 2011.
   <https://www.cs.cornell.edu/projects/Sound/fire/>
2. Chadwick, J. N. (2011). *Reference Matlab implementation of spectral
   bandwidth extension.*
   <https://www.cs.cornell.edu/projects/Sound/fire/code/bandwidth_extension.zip>
3. Rajaram, R., and Lieuwen, T. (2009). *Acoustic radiation from
   turbulent premixed flames.* Journal of Fluid Mechanics, 637, 357&ndash;385.
4. Clavin, P., and Siggia, E. D. (1991). *Turbulent premixed flames and
   sound generation.* Combustion Science and Technology, 78(1&ndash;3),
   147&ndash;155.
5. Z&ouml;lzer, U., and Amatriain, X. (eds.) (2002). *DAFX: Digital
   Audio Effects.* Wiley.
