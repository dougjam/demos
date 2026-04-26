# Tools

Helper scripts. Run all from the project root.

| Script | Purpose |
|---|---|
| `vectors_to_assets.py` | Convert `srcOrig/work/*.vector` to Float32 LE binaries under `assets/` plus a `manifest.json`. Run once after each change to the source vectors (in practice, just once when the demo is set up). |
| `generate_goldens.py`  | Regenerate the Tier 1/2 binary goldens under `python/tests/golden/`. The goldens are produced by the Python port itself; they pin the port against regression and act as the parity target the JavaScript port must match. See `python/tests/README.md` for the tolerance log. |

## Notes

The original Octave/Matlab reference implementation lives under
`srcOrig/matlab/` and is **not** required to run these tools or the test
suite. The new Python and JavaScript implementations are self-contained.
The Matlab source remains in the tree as the canonical reference for the
line-by-line port (see `docs/ALGORITHM.md`).
