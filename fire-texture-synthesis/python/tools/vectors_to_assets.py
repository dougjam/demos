# This file is a derivative work of the C++ implementation released
# alongside Chadwick and James, "Animating Fire with Sound," SIGGRAPH 2011.
# See https://www.cs.cornell.edu/projects/Sound/fire/ for the original.
#
# Original copyright notice (preserved per BSD 2-Clause):
#
# Copyright (c) 2011, Jeffrey Chadwick
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Convert srcOrig/work/{name}/{input,training}_data.vector files to
Float32 LE binaries for the JS demo, plus copy the raw Recordist WAVs
into assets/training_audio/.

Run from the project root:

    python python/tools/vectors_to_assets.py
"""

from __future__ import annotations

import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from vector_io import read_vector  # noqa: E402

ROOT = _HERE.parents[1]
SRC_DIR = ROOT / "srcOrig" / "work"
DST_DIR = ROOT / "assets"

EXAMPLES = ["burning_brick", "candle", "dragon", "flame_jet", "torch"]
SAMPLE_RATE = 44100

# Per-example demo tunings that intentionally deviate from the canonical
# C++ default.xml values. Anything in here is merged on top of the XML
# config when writing assets/{name}/default.json. The demo's slider
# initializes from the merged value; users can drag back to the canonical
# value to hear the original behaviour.
DEMO_OVERRIDES = {
    # Both torch and candle hit the wide-input / narrow-training failure
    # mode of the section 5.3 dynamic-range mapping. At
    # scaling_alpha=1.0 the per-window matched training windows get
    # amplified by similar large factors across many consecutive windows
    # and the triangular overlap-add reinforces them constructively into
    # a tonal buzz at the windowHW stride frequency. The paper itself
    # flags this in section 6: "in some instances the method still has
    # difficulty producing a suitable, temporally coherent output sound.
    # This can occur in cases when the low-frequency input has a very
    # wide dynamic range, while the training data has a small range."
    # scaling_alpha=0.5 blends in only half the rescale and avoids the
    # artefact while keeping the partial dynamic-range matching the
    # algorithm wants. Confirmed by ear on both signals.
    "torch": {
        "scaling_alpha": 0.5,
        "_demo_overrides": {
            "scaling_alpha_canonical": 1.0,
            "_note": "scaling_alpha lowered from C++ canonical 1.0 to 0.5 to "
                     "avoid CDF-mapping amplification artefact on this signal.",
        },
    },
    "candle": {
        "scaling_alpha": 0.5,
        "_demo_overrides": {
            "scaling_alpha_canonical": 1.0,
            "_note": "scaling_alpha lowered from C++ canonical 1.0 to 0.5 to "
                     "avoid CDF-mapping amplification artefact on this signal.",
        },
    },
}


def _params_from_xml(path: Path) -> dict:
    cfg = dict(ET.parse(path).getroot().attrib)
    return {
        "fs": int(cfg.get("Fs", 44100)),
        "num_levels": int(cfg.get("numLevels", 6)),
        "window_hw": int(cfg.get("windowHW", 4)),
        "feature_hw": int(cfg.get("featureHW", 3)),
        "eps_ann": float(cfg.get("epsANN", 1.0)),
        "falloff": float(cfg.get("falloff", 0.0)),
        "scale_cdf": bool(int(cfg.get("scaleCDF", 1))),
        "scaling_alpha": float(cfg.get("scalingAlpha", 1.0)),
        "basesignal": cfg.get("basesignal", "input_data.vector"),
        "trainingsignal": cfg.get("trainingsignal", "training_data.vector"),
    }


def main() -> int:
    if not SRC_DIR.is_dir():
        print(f"missing {SRC_DIR}", file=sys.stderr)
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)
    audio_out = DST_DIR / "training_audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "sample_rate": SAMPLE_RATE,
        "format": "float32_le",
        "examples": [],
    }
    for name in EXAMPLES:
        in_dir = SRC_DIR / name
        out_dir = DST_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        params = _params_from_xml(in_dir / "default.xml")
        # Save params (with the resolved filenames stripped, since the JS
        # only needs the algorithm parameters). Merge any per-example
        # demo override on top of the canonical XML values.
        params_for_demo = {k: v for k, v in params.items()
                            if k not in ("basesignal", "trainingsignal")}
        if name in DEMO_OVERRIDES:
            params_for_demo.update(DEMO_OVERRIDES[name])
        (out_dir / "default.json").write_text(
            json.dumps(params_for_demo, indent=2) + "\n"
        )

        base_path = in_dir / params["basesignal"]
        training_path = in_dir / params["trainingsignal"]
        if not base_path.exists():
            print(f"skipping {name}: missing {base_path}", file=sys.stderr)
            continue
        if not training_path.exists():
            print(f"skipping {name}: missing {training_path}", file=sys.stderr)
            continue

        base = read_vector(base_path).astype(np.float32, copy=False)
        training = read_vector(training_path).astype(np.float32, copy=False)
        (out_dir / "input.bin").write_bytes(base.tobytes())
        (out_dir / "training.bin").write_bytes(training.tobytes())

        manifest["examples"].append({
            "name": name,
            "input_file": f"{name}/input.bin",
            "training_file": f"{name}/training.bin",
            "params_file": f"{name}/default.json",
            "input_samples": int(base.size),
            "training_samples": int(training.size),
            "input_duration_s": float(base.size / SAMPLE_RATE),
            "training_duration_s": float(training.size / SAMPLE_RATE),
            "training_basename": params["trainingsignal"],
        })
        print(
            f"wrote {name}: input={base.size} samples, "
            f"training={training.size} samples (from {params['trainingsignal']})"
        )

    # Copy the raw Recordist WAVs.
    recordist = sorted(SRC_DIR.glob("FIRE *.wav"))
    audio_manifest = []
    for wav in recordist:
        # Use a URL-safe name for the asset (preserve original in metadata).
        slug = wav.stem.replace(" ", "_")
        out = audio_out / (slug + ".wav")
        shutil.copy2(wav, out)
        audio_manifest.append({
            "file": f"training_audio/{out.name}",
            "original_name": wav.name,
        })
        print(f"copied {wav.name} -> {out.name}")
    manifest["training_audio"] = audio_manifest
    manifest["training_audio_credit"] = (
        "Source: The Recordist 'Ultimate Fire' sound library "
        "(https://www.therecordist.com/). Used with permission inherited from "
        "the original release at https://www.cs.cornell.edu/projects/Sound/fire/. "
        "All copyright in the audio remains with The Recordist."
    )

    (DST_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {(DST_DIR / 'manifest.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
