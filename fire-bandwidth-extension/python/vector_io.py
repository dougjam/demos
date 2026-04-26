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
"""Read and write Matlab-style .vector files.

The .vector files in srcOrig/work/ are produced by srcOrig/matlab/write_vector.m
and consumed by srcOrig/matlab/read_vector.m. The on-disk layout is:

    int32   n_row              (little-endian, signed, 4 bytes)
    float64 samples[n_row]     (little-endian, IEEE 754 double, 8 bytes each)

The data is a single-column vector. This module's read_vector / write_vector
match the Matlab functions byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_vector(path: str | Path) -> np.ndarray:
    """Load a .vector file as a 1D float64 numpy array (mirrors read_vector.m)."""
    raw = Path(path).read_bytes()
    n_row = int(np.frombuffer(raw, dtype="<i4", count=1)[0])
    data = np.frombuffer(raw, dtype="<f8", count=n_row, offset=4)
    return np.ascontiguousarray(data, dtype=np.float64)


def write_vector(data: np.ndarray, path: str | Path) -> None:
    """Save a 1D array to a .vector file (mirrors write_vector.m)."""
    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float64).ravel())
    n_row = np.int32(arr.size)
    with open(path, "wb") as f:
        f.write(n_row.astype("<i4").tobytes())
        f.write(arr.astype("<f8").tobytes())
