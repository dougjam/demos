# This file is a derivative work of the Matlab implementation released
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
"""Convert srcOrig/work/*.vector files to Float32 LE .bin assets for the demo.

Run from the project root:

    python python/tools/vectors_to_assets.py

Produces ``assets/{name}.bin`` for each ``.vector`` file plus
``assets/manifest.json`` with sample counts and durations. See
SPEC_SpectralBandwidthExtension.md section 4.3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from vector_io import read_vector  # noqa: E402

ROOT = _HERE.parents[1]
SRC_DIR = ROOT / "srcOrig" / "work"
DST_DIR = ROOT / "assets"
SAMPLE_RATE = 44100


def main() -> int:
    if not SRC_DIR.is_dir():
        print(f"missing source directory: {SRC_DIR}", file=sys.stderr)
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "sample_rate": SAMPLE_RATE,
        "format": "float32_le",
        "examples": [],
    }
    examples: list[dict[str, object]] = []
    for vf in sorted(SRC_DIR.glob("*.vector")):
        data = read_vector(vf).astype(np.float32, copy=False)
        out_path = DST_DIR / f"{vf.stem}.bin"
        out_path.write_bytes(data.tobytes())
        examples.append(
            {
                "name": vf.stem,
                "file": f"{vf.stem}.bin",
                "samples": int(data.size),
                "duration_s": float(data.size / SAMPLE_RATE),
            }
        )
        print(f"wrote {out_path.relative_to(ROOT)} ({data.size} samples)")
    manifest["examples"] = examples

    (DST_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {(DST_DIR / 'manifest.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
