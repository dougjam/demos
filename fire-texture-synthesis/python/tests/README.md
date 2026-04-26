# Test suite

Three tiers, per `SPEC_FireTextureSynthesis.md` section 3.

## Running

```bash
pip install numpy scipy pytest
cd <project root>
pytest python/tests/                          # all tests
pytest python/tests/test_unit.py              # tier 1 only (~1 s)
pytest python/tests/test_phase_controlled.py  # tier 2 only (~5 min)
pytest python/tests/test_statistical.py       # tier 3 only (~30 s)
```

## Tiers

### Tier 1 &mdash; deterministic kernels

`test_unit.py`. Pyramid level reduction (binary `.f64` golden,
`atol=rtol=1e-13`), pyramid shapes for various input sizes, padded-input
power-of-2 invariant, CDF sample / inverse-sample on a hand-built
sorted array, KD-tree determinism, PCG32 first-N outputs, range and
same-seed determinism.

### Tier 2 &mdash; full pipeline

`test_phase_controlled.py`. For each of the 5 example (base, training)
pairs, re-run the Python port with the original `default.xml`
parameters and compare to the cached `.npz` golden. Same Python + same
input &rArr; byte-identical output (`np.array_equal`).

This tier is slow because the synthesis loop runs sequentially window
by window. ~30 s per smaller case (`burning_brick`, `torch`), ~1&ndash;2 min
per larger case (`candle`, `dragon`, `flame_jet`). Total ~5&ndash;7 min.

### Tier 3 &mdash; statistical smoke tests

`test_statistical.py`. Properties that must hold regardless of fixture
data:

- `silent base` &rArr; output is very small.
- `silent training` &rArr; output amplitude is small (no contributions to blend).
- `same input + same seed` &rArr; byte-identical output (determinism).
- Output has substantial energy in the 1&ndash;5 kHz band that the base
  alone (a 100 Hz tone) does not.
- Toggling `scale_cdf` produces different outputs (CDF mapping is
  actually doing something).
- Output shape and dtype match the base.

## Goldens

`golden/` contains binary fixtures plus `tier1_metadata.json` /
`tier2_metadata.json` describing the inputs that produced each. They
are produced by **the Python port itself**
(`python/tools/generate_goldens.py`):

1. The released C++ is the porting reference (see `srcOrig/` and
   `docs/ALGORITHM.md`), but it relies on ANN approximate NN search
   that this port replaces with exact NN, and contains a documented
   off-by-one in `buildGaussianLevel` that this port fixes. So
   bit-equivalence with the C++ is not the goal.
2. The goldens here pin the **Python port** against accidental
   regression. Any change to the Python kernels or synthesis loop that
   perturbs output by more than the documented tolerance trips Tier 1
   (kernels) or Tier 2 (full pipeline) immediately.
3. The Tier 2 goldens are also the **parity target** that the
   JavaScript port must reproduce. The JS port consumes the same
   inputs and the same exact-NN logic; the in-browser parity check at
   `tests/validate.html` reports the achieved tolerance on a small
   synthetic case.

If the Python port&apos;s output ever needs to be re-pinned (e.g.,
deliberate algorithmic change), regenerate with:

```bash
python python/tools/generate_goldens.py            # both tiers
python python/tools/generate_goldens.py --tier 1   # just kernels
```

Tier 2 goldens are committed; Tier 1 goldens are tiny and committed
too. Storage is ~25&ndash;40 MB (output `.npz`s for the 5 examples).
