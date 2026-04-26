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
"""Sorted-amplitude CDF helpers used by the dynamic-range mapping in
section 5.3 of Chadwick & James (2011).

A "CDF" here is a 1-D float64 array of absolute amplitudes from one
pyramid level, sorted ascending. ``sample_cdf(fraction)`` returns the
amplitude at the given percentile; ``sample_inverse_cdf(amplitude)``
returns the percentile of the given amplitude. Both mirror the static
methods of the same names on ``GaussianPyramid`` in the released C++
(``srcOrig/texture/GaussianPyramid.cpp`` lines 761-840).
"""

from __future__ import annotations

import numpy as np


def init_cdf_from_level(level_data: np.ndarray, start: int, end: int) -> np.ndarray:
    """Mirror ``GaussianPyramid::initCDF``.

    Returns a float64 array of ``|level_data[start:end]|`` sorted ascending.
    """
    return np.sort(np.abs(level_data[start:end].astype(np.float64, copy=False)))


def sample_cdf(fraction: float, cdf: np.ndarray) -> float:
    """Mirror ``GaussianPyramid::sampleCDF``.

    Linearly interpolates the sorted CDF at index ``fraction * (N-1)``.
    """
    n = cdf.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(cdf[0])
    idx_real = fraction * (n - 1)
    idx1 = int(idx_real)  # truncation toward 0; fraction is non-negative
    idx2 = idx1 + 1
    idx1_real = float(idx1)
    idx2_real = float(idx2)
    idx1_clamped = max(0, min(n - 1, idx1))
    idx2_clamped = max(0, min(n - 1, idx2))
    blend1 = idx2_real - idx_real
    blend2 = 1.0 - blend1
    return float(blend1 * cdf[idx1_clamped] + blend2 * cdf[idx2_clamped])


def sample_inverse_cdf(amplitude: float, cdf: np.ndarray) -> float:
    """Mirror ``GaussianPyramid::sampleInverseCDF``.

    Binary-searches the sorted CDF and linearly interpolates the
    percentile of ``amplitude``. Returns a float in ``[0, 1]``.
    """
    n = cdf.size
    if n == 0:
        return 0.0
    if amplitude < cdf[0]:
        return 0.0
    if amplitude >= cdf[n - 1]:
        return 1.0
    idx1 = _binary_search(cdf, amplitude, 0, n - 1)
    idx1 = max(0, min(n - 2, idx1))
    idx2 = idx1 + 1
    a1 = float(cdf[idx1])
    a2 = float(cdf[idx2])
    diff = a2 - a1
    if diff == 0.0:
        # Repeated amplitude; pick the lower index's percentile.
        return idx1 / float(n - 1)
    blend1 = (a2 - amplitude) / diff
    blend2 = 1.0 - blend1
    idx1_real = idx1 / float(n - 1)
    idx2_real = idx2 / float(n - 1)
    return float(blend1 * idx1_real + blend2 * idx2_real)


def _binary_search(data: np.ndarray, value: float, start: int, end: int) -> int:
    """Find the index ``i`` such that ``data[i] <= value < data[i+1]``.

    Mirrors ``GaussianPyramid::binarySearch``.
    """
    while start <= end:
        mid = (start + end) // 2
        if mid + 1 < data.size and data[mid] <= value < data[mid + 1]:
            return mid
        elif value >= data[mid]:
            start = mid + 1
        else:
            end = mid - 1
    return start
