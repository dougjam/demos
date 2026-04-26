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
"""1-D Gaussian pyramid + per-window feature extraction.

Faithful Python port of ``srcOrig/texture/GaussianPyramid.{h,cpp}``. The
function and method names mirror the C++ one-for-one so the diff is
auditable against the reference. See ``docs/ALGORITHM.md`` for the
per-rule fidelity log.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from cdf_match import init_cdf_from_level, sample_cdf, sample_inverse_cdf


# 5-tap Gaussian filter. From ``GaussianPyramid::GAUSSIAN_STENCIL`` (line 51).
GAUSSIAN_STENCIL = np.array([0.05, 0.25, 0.40, 0.25, 0.05], dtype=np.float64)
GAUSSIAN_STENCIL_HALF = (GAUSSIAN_STENCIL.size - 1) // 2  # 2

# Threshold below which CDF rescaling is skipped (mirrors AMPLITUDE_CUTOFF).
AMPLITUDE_CUTOFF = 1e-4


def pad_input_signal(
    signal: np.ndarray, reflect_boundaries: bool = False
) -> tuple[np.ndarray, int, int]:
    """Mirror ``GaussianPyramid::padInputSignal``.

    Pads ``signal`` to the smallest ``2^k + 1`` length that contains it,
    centring the original samples and either zero-padding or reflecting
    the flanks. Returns ``(padded, start_idx, end_idx)`` so the caller
    can recover the original range with ``padded[start_idx:end_idx]``.
    """
    n = signal.size
    extended_length = 1
    while extended_length + 1 < n:
        extended_length *= 2
    extended_length += 1

    out = np.zeros(extended_length, dtype=np.float64)
    start_idx = (extended_length - n) // 2
    end_idx = start_idx + n
    out[start_idx:end_idx] = signal

    if reflect_boundaries:
        # Reflect the left flank.  C++ writes out[i] = signal[start_idx - i]
        # for i in [0, start_idx), so out[0] = signal[start_idx] (the
        # first real sample reflected onto itself), out[1] = signal[start_idx - 1], etc.
        for i in range(start_idx):
            src = start_idx - i
            if 0 <= src < n:
                out[i] = signal[src]
        # Reflect the right flank symmetrically: C++ writes
        # out[i] = out[end_idx - (i - end_idx) - 2] for i in [end_idx, extended).
        for i in range(end_idx, extended_length):
            src = end_idx - (i - end_idx) - 2
            if 0 <= src < extended_length:
                out[i] = out[src]
    return out, start_idx, end_idx


def build_gaussian_level(base_level: np.ndarray) -> np.ndarray:
    """Mirror ``GaussianPyramid::buildGaussianLevel``.

    Filters ``base_level`` with the 5-tap Gaussian stencil and
    decimates 2:1, returning a level of size ``(N-1)//2 + 1``. Boundary
    samples are computed by clipping the filter's footprint to the
    available data (no reflection or zero extension).

    Bug fix vs C++: the released loop uses ``j <= num_components`` and
    reads one past the end of both the data slice and the stencil
    array. This port uses ``j < num_components`` so the filter is
    applied exactly to the in-range samples.
    """
    if (base_level.size - 1) % 2 != 0:
        raise ValueError(
            f"Base level size must be 2^k + 1 for some k; got {base_level.size}"
        )
    next_size = (base_level.size - 1) // 2 + 1
    next_level = np.zeros(next_size, dtype=np.float64)
    half_width = GAUSSIAN_STENCIL_HALF  # 2
    for i in range(next_size):
        middle_idx = i * 2
        start_idx = max(0, middle_idx - half_width)
        end_idx = min(base_level.size - 1, middle_idx + half_width)
        filter_start_idx = start_idx - middle_idx + half_width
        num_components = end_idx - start_idx + 1
        # FIXED: C++ uses `<=`, which reads one past the end of both
        # arrays. Use `<` for the in-range filter.
        for j in range(num_components):
            next_level[i] += (
                base_level[start_idx + j] * GAUSSIAN_STENCIL[filter_start_idx + j]
            )
    return next_level


def sample_signal(signal: np.ndarray, index: float) -> tuple[float, bool]:
    """Mirror ``GaussianPyramid::sampleSignal``.

    Linearly interpolates ``signal`` at non-integer ``index``. Returns
    ``(value, inside)`` where ``inside`` is True iff both bracketing
    integer indices are inside ``[0, signal.size)``.
    """
    n = signal.size
    floor_idx = math.floor(index)
    int_idx1 = int(floor_idx)
    int_idx2 = int_idx1 + 1
    v1 = signal[int_idx1] if 0 <= int_idx1 < n else 0.0
    v2 = signal[int_idx2] if 0 <= int_idx2 < n else 0.0
    inside = (0 <= int_idx1 < n) and (0 <= int_idx2 < n)
    frac = index - floor_idx
    return float((1.0 - frac) * v1 + frac * v2), inside


class GaussianPyramid:
    """1-D Gaussian pyramid with optional reflected boundary padding.

    Mirrors ``GaussianPyramid`` in
    ``srcOrig/texture/GaussianPyramid.{h,cpp}``. Stores ``num_levels``
    pyramid levels and per-level ``(start, end)`` indices for the
    original (pre-pad) sample range.
    """

    def __init__(
        self, signal: np.ndarray, num_levels: int, reflect_boundaries: bool = False
    ) -> None:
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        signal = np.asarray(signal, dtype=np.float64).ravel()
        padded, start, end = pad_input_signal(signal, reflect_boundaries)
        self._levels: List[np.ndarray] = [padded]
        self._start_index: List[int] = [start]
        self._end_index: List[int] = [end]
        for _ in range(1, num_levels):
            prev = self._levels[-1]
            self._levels.append(build_gaussian_level(prev))
            prev_start = self._start_index[-1]
            prev_end = self._end_index[-1]
            new_start = prev_start // 2 + (1 if prev_start % 2 != 0 else 0)
            new_end = prev_end // 2 + (1 if prev_end % 2 != 0 else 0)
            self._start_index.append(new_start)
            self._end_index.append(new_end)
        self._cdf: np.ndarray | None = None

    # ---- accessors ----
    @property
    def levels(self) -> List[np.ndarray]:
        return self._levels

    def start_index(self, level: int) -> int:
        return self._start_index[level]

    def end_index(self, level: int) -> int:
        return self._end_index[level]

    @property
    def cdf(self) -> np.ndarray | None:
        return self._cdf

    @property
    def num_levels(self) -> int:
        return len(self._levels)

    # ---- mutation ----
    def zero_level(self, level: int) -> None:
        self._levels[level][:] = 0.0

    def init_cdf(self) -> None:
        """Mirror ``GaussianPyramid::initCDF``."""
        top = self._levels[-1]
        self._cdf = init_cdf_from_level(
            top, self._start_index[-1], self._end_index[-1]
        )

    def reconstruct_signal(self) -> np.ndarray:
        """Mirror ``GaussianPyramid::reconstructSignal``.

        Returns the bottom level trimmed to the original (pre-pad) range.
        """
        return self._levels[0][self._start_index[0] : self._end_index[0]].copy()

    # ---- feature extraction ----
    def num_windows(self, window_hw: int, level: int) -> int:
        """Mirror ``numWindows = level_size / windowHW + 1`` from extendLevel."""
        return self._levels[level].size // window_hw + 1

    def compute_window_feature(
        self,
        window_half_widths: List[int],
        feature_half_widths: List[int],
        level: int,
        window_idx: int,
        falloff: float = 0.0,
        input_cdf: np.ndarray | None = None,
        output_cdf: np.ndarray | None = None,
        scaling_alpha: float = 1.0,
    ) -> tuple[np.ndarray, bool, float]:
        """Mirror ``GaussianPyramid::computeWindowFeature``.

        Builds the feature vector for output window ``window_idx`` at
        level ``level``. Two regions: ``window_hw * feature_hw + 1``
        causal samples from the current level immediately to the left
        of the window centre, then (if ``level + 1 < num_levels``)
        ``2 * window_hw' * (feature_hw' + 1) + 1`` symmetric, linearly
        interpolated samples from the next coarser level.

        With ``input_cdf``, ``output_cdf`` provided AND ``level + 1``
        is the top level, the coarser-level entries are rescaled by
        the dynamic-range mapping of section 5.3. Returns the inverse
        scaling as ``scale`` so the caller can apply it when blending
        the matched window into the output.

        Returns ``(feature, all_inside, scale)``.
        """
        num_levels = self.num_levels
        wh = window_half_widths[level]
        fh = feature_half_widths[level]
        window_middle = window_idx * wh

        # Causal region (current level)
        feature_start = window_middle - wh * (fh + 1)
        feature_end = window_middle - wh
        n_causal = feature_end - feature_start + 1

        if level < num_levels - 1:
            wh_c = window_half_widths[level + 1]
            fh_c = feature_half_widths[level + 1]
            n_coarser = 2 * wh_c * (fh_c + 1) + 1
        else:
            n_coarser = 0

        feature = np.zeros(n_causal + n_coarser, dtype=np.float64)
        feat_idx = 0
        all_inside = True
        scale = 1.0

        level_data = self._levels[level]
        start = self._start_index[level]
        end = self._end_index[level]
        for i in range(feature_start, feature_end + 1):
            in_range = start <= i < end
            all_inside = all_inside and in_range
            if 0 <= i < level_data.size:
                v = level_data[i]
            else:
                v = 0.0
            weight = math.exp(-falloff * abs(i - window_middle)) if falloff > 0.0 else 1.0
            feature[feat_idx] = v * weight
            feat_idx += 1

        if level == num_levels - 1:
            return feature, all_inside, scale

        # Coarser-level region (next level)
        next_data = self._levels[level + 1]
        wh_c = window_half_widths[level + 1]
        fh_c = feature_half_widths[level + 1]
        window_middle_real = window_middle / 2.0
        feature_start_real = window_middle_real - wh_c * (fh_c + 1)
        feature_length = 2 * wh_c * (fh_c + 1) + 1

        coarser_start_in_feature = feat_idx
        avg_magnitude = 0.0

        for i in range(feature_length):
            sample_real = feature_start_real + i
            v, inside = sample_signal(next_data, sample_real)
            all_inside = all_inside and inside
            if falloff > 0.0:
                weight = math.exp(-falloff * abs(sample_real - window_middle_real))
                v *= weight
            feature[feat_idx] = v
            avg_magnitude += abs(v)
            feat_idx += 1

        avg_magnitude /= feature_length

        # Dynamic-range mapping (section 5.3): only when the next level is
        # the top level and CDFs were supplied.
        if (
            level + 1 == num_levels - 1
            and input_cdf is not None
            and output_cdf is not None
        ):
            if avg_magnitude > AMPLITUDE_CUTOFF:
                output_fraction = sample_inverse_cdf(avg_magnitude, output_cdf)
                input_magnitude = sample_cdf(output_fraction, input_cdf)
                scaling = input_magnitude / avg_magnitude
                scaling = (1.0 - scaling_alpha) + scaling_alpha * scaling
                if scaling != 0.0 and not math.isnan(scaling) and not math.isinf(scaling):
                    scale = 1.0 / scaling
                    feature[
                        coarser_start_in_feature : coarser_start_in_feature
                        + feature_length
                    ] *= scaling
            # else: small-amplitude window; leave feature alone, scale = 1.

        return feature, all_inside, scale
