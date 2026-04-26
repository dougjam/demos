# Tools

Helper scripts. Run all from the project root.

| Script | Purpose |
|---|---|
| `vectors_to_assets.py` | Convert `srcOrig/work/{name}/{input,training}_data.vector` to Float32 LE binaries under `assets/`, generate `assets/manifest.json`, and copy the 6 raw Recordist WAVs into `assets/training_audio/`. Run once after the source is extracted. |
| `generate_goldens.py` | Regenerate the Tier 1 (kernel) and Tier 2 (full pipeline) goldens under `python/tests/golden/`. Tier 1 is fast (~1 s); Tier 2 takes ~7 min for all five examples. |
| `generate_synthetic_golden.py` | Regenerate `tests/golden_synthetic.json`, the small fixed-seed reference used by `tests/validate.html` for in-browser parity verification. |

## Notes

The original C++ reference implementation lives under `srcOrig/` and is
**not** required to run these tools or the test suite. The new Python
and JavaScript implementations are self-contained.
