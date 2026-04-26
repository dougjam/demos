# Fire Sound Texture Synthesis

A faithful Python port and a self-contained interactive browser demo of
the data-driven sound-texture synthesis algorithm (Algorithm 2,
&sect;5) from:

> **Jeffrey N. Chadwick and Doug L. James.** *Animating Fire with Sound.*
> ACM Transactions on Graphics (SIGGRAPH 2011), 30(4), August 2011.
> Project page: <https://www.cs.cornell.edu/projects/Sound/fire/>

The algorithm extends a low-frequency, physically based pressure signal
(the time derivative of the velocity flux through the moving flame front,
under the simplifying assumptions of paper &sect;3; band-limited to the
flame solver's Nyquist, &asymp; 180 Hz at a 360 Hz time-stepping rate)
into a broadband, perceptually fire-like signal by **stitching together
windows of a real fire-audio recording**, chosen at every pyramid level
by a nearest-neighbour query that matches the simulation's local
context. This is the data-driven counterpart to the spectral bandwidth
extension demo at [`../fire-bandwidth-extension/`](../fire-bandwidth-extension/),
which adds synthetic power-law noise instead.

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

then visit <http://127.0.0.1:8765/>. Or use the live deployment at the
parent demos repo's GitHub Pages URL.

The five released flame-simulation signals (`burning_brick`, `candle`,
`dragon`, `flame_jet`, `torch`) are bundled as Float32 binaries under
`assets/`, paired with their hand-picked training audio. Each click of a
preset button fully re-runs the synthesis for the selected (base,
training) pair. You can also load your own base WAV or training WAV
through the file pickers.

Slider controls: `numLevels` (pyramid depth), `windowHW` (output-window
half-width in samples), `featureHW` (feature context half-width),
`falloff` (exponential weight on feature dimensions), `scalingAlpha`
(strength of the CDF dynamic-range mapping), plus a `scaleCDF` checkbox
and an RNG seed (used only for tie-breaking in the rare event of
identical feature vectors). Optional **auto-process** re-runs the
algorithm 250 ms after every slider change; **Download** buttons save
either the original or the synthesized signal as a 16-bit WAV.

The page also shows pyramid-level waveform stacks of both pyramids
(input and output), spectrograms (input vs synthesized), and a live
spectrum plot overlaying the input PSD, training PSD, and output PSD.
The "About this sound model" disclosure at the bottom of the page
summarizes the algorithm and its assumptions for students.

### Python port

From the project root:

```bash
python python/texture_synthesis.py \
    --input  srcOrig/work/dragon/input_data.vector \
    --training srcOrig/work/dragon/training_data_1.vector \
    --output dragon_synth.wav

python python/texture_synthesis.py \
    --config srcOrig/work/dragon/default.xml \
    --base-dir srcOrig/work/dragon \
    --output dragon_synth.wav
```

The CLI accepts both `.wav` and `.vector` (Matlab single-column
double-precision with an `int32` row-count prefix; see
`python/vector_io.py`) for input and output. With `--config` it reads
the same XML configuration files used by the original C++ binary.

### Tests

```bash
pip install numpy scipy pytest
pytest python/tests/
```

Three tiers (per `SPEC_FireTextureSynthesis.md`):

- **Tier 1**: deterministic kernel goldens (Gaussian level reduction,
  CDF sample / inverse-sample, KD-tree determinism, PCG32 PRNG outputs).
  `atol=rtol=1e-13`.
- **Tier 2**: full-pipeline goldens for 5 (base, training) example
  pairs. The Python port itself produces the goldens; the JavaScript
  port must reproduce them within FFT-library round-off.
- **Tier 3**: statistical smoke tests (low-frequency content
  preservation, training-PSD shape preservation, CDF-mapping benefit,
  byte-identical determinism).

The goldens in `python/tests/golden/` are produced by the Python port
itself (`python/tools/generate_goldens.py`). The original C++ source
under `srcOrig/` is kept unmodified as the line-by-line porting
reference; see `docs/ALGORITHM.md` for the per-rule fidelity log.

## Layout

```
fire-texture-synthesis/
├── index.html, main.js, style.css      browser demo
├── texture_synthesis.js                JS port of the algorithm
├── kdtree.js                            hand-port of KDTreeMulti.cpp
├── deterministic_rng.js                 PCG32 (shared with python/)
├── fft.js                               vendored (MIT, indutny/fft.js)
├── worker.js                            standalone worker reference
├── serve.py                             no-cache local HTTP server
├── thumb.jpg                            gallery thumbnail
├── assets/                              5 examples + raw Recordist WAVs
├── tests/                               browser-side parity validation
├── python/
│   ├── texture_synthesis.py             the port (ground truth)
│   ├── gaussian_pyramid.py              1D Gaussian pyramid + features
│   ├── cdf_match.py                     §5.3 dynamic-range mapping
│   ├── deterministic_rng.py             PCG32 (shared with deterministic_rng.js)
│   ├── vector_io.py                     .vector reader/writer
│   ├── tools/                           golden generator, asset packer
│   └── tests/                           three-tier test suite
├── srcOrig/                             provided C++ source (do not modify)
├── docs/
│   ├── ALGORITHM.md                     algorithm + port-fidelity notes
│   ├── DEMO_DISCLAIMER.md               course-wide demo disclaimer
│   └── 2011 Chadwick - Fire Sound_rekt.pdf
├── ATTRIBUTION.md, ATTRIBUTION.html
├── LICENSE                              BSD 2-Clause (Chadwick 2011 + ports)
├── SPEC_FireTextureSynthesis.md
└── README.md
```

## License

BSD 2-Clause. See [`LICENSE`](LICENSE) and [`ATTRIBUTION.md`](ATTRIBUTION.md).
The bundled training audio under `assets/training_audio/` is from
**The Recordist's** *Ultimate Fire* sound library
(<https://www.therecordist.com/>), redistributed under the original
release's permission.

## Acknowledgments

This work is a derivative of the C++ reference implementation released
by **Jeffrey N. Chadwick** and **Doug L. James** at
<https://www.cs.cornell.edu/projects/Sound/fire/>. Both the algorithm
and the bundled flame-simulation signals + training-audio clips are
theirs.

The Python and JavaScript ports and the browser demo were developed
with Claude Code for **CS 448Z (Physically Based Animation and Sound),
Stanford University, Spring 2026**, taught by Doug L. James.

## References

1. Chadwick, J. N., and James, D. L. (2011). *Animating Fire with Sound.*
   ACM Transactions on Graphics (SIGGRAPH 2011), 30(4), August 2011.
   <https://www.cs.cornell.edu/projects/Sound/fire/>
2. Chadwick, J. N. (2011). *Reference C++ implementation of sound
   texture synthesis.*
   <https://www.cs.cornell.edu/projects/Sound/fire/code/sound_texture_synthesis.zip>
3. Wei, L.-Y., and Levoy, M. (2000). *Fast texture synthesis using
   tree-structured vector quantization.* Proc. SIGGRAPH 2000, 479&ndash;488.
4. Burt, P. J., and Adelson, E. H. (1983). *A multiresolution spline
   with application to image mosaics.* ACM Transactions on Graphics,
   2(4), 217&ndash;236.
5. Heeger, D. J., and Bergen, J. R. (1995). *Pyramid-based texture
   analysis/synthesis.* Proc. SIGGRAPH 1995, 229&ndash;238.
   *(Source of the histogram-matching technique adapted in &sect;5.3.)*
