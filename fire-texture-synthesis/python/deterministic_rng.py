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
"""PCG32-XSH-RR PRNG, ported byte-for-byte to JavaScript in deterministic_rng.js.

Reference: https://www.pcg-random.org/

The generator state is two 64-bit integers (state, inc). Each call to
``next_u32`` advances state with the LCG ``state = state * MULT + inc`` and
returns a 32-bit output formed by an XSH (xorshift-high) followed by an RR
(random rotation) step. The output is interpreted as a uniform draw from
``[0, 1)`` by dividing by ``2**32``.

Seeding mirrors the canonical ``pcg32_srandom_r`` initializer with a fixed
``initseq`` so a single integer ``seed`` reproduces the same stream in both
languages. The JavaScript port uses the same constants; the matching unit
test compares the first 1000 outputs.
"""

from __future__ import annotations

import numpy as np

_MASK64 = (1 << 64) - 1
_MASK32 = (1 << 32) - 1
_MULT = 6364136223846793005
_INITSEQ = 0xDA3E39CB94B95BDB
_TWO_TO_32 = 4294967296.0


class DeterministicRng:
    """PCG32-XSH-RR. Same seed, same stream, in Python and JavaScript."""

    __slots__ = ("state", "inc")

    def __init__(self, seed: int) -> None:
        self.inc = ((_INITSEQ << 1) | 1) & _MASK64
        self.state = 0
        self._step()
        self.state = (self.state + (int(seed) & _MASK64)) & _MASK64
        self._step()

    def _step(self) -> int:
        oldstate = self.state
        self.state = (oldstate * _MULT + self.inc) & _MASK64
        xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & _MASK32
        rot = (oldstate >> 59) & 31
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & _MASK32

    def next_u32(self) -> int:
        """Return the next 32-bit unsigned output."""
        return self._step()

    def random(self) -> float:
        """Return a uniform float in [0, 1)."""
        return self._step() / _TWO_TO_32

    def random_array(self, n: int) -> np.ndarray:
        """Return ``n`` uniform float64 draws in [0, 1)."""
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self._step() / _TWO_TO_32
        return out
