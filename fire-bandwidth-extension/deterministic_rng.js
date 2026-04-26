// This file is a derivative work of the Matlab implementation released
// alongside Chadwick and James, "Animating Fire with Sound," SIGGRAPH 2011.
// See https://www.cs.cornell.edu/projects/Sound/fire/ for the original.
//
// Original copyright notice (preserved per BSD 2-Clause):
//
// Copyright (c) 2011, Jeffrey Chadwick
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// PCG32-XSH-RR. Mirrors python/deterministic_rng.py byte-for-byte.
// Same seed produces the same uniform [0, 1) stream in both languages.
// Verified by python/tests/test_unit.py::test_rng_first_outputs and
// tests/validate.html::testRng.

'use strict';

(function (global) {
  const PCG_MULT = 6364136223846793005n;
  const PCG_INITSEQ = 0xda3e39cb94b95bdbn;
  const MASK64 = (1n << 64n) - 1n;
  const MASK32 = (1n << 32n) - 1n;
  const TWO_TO_32 = 4294967296.0;

  class DeterministicRng {
    constructor(seed) {
      this.inc = ((PCG_INITSEQ << 1n) | 1n) & MASK64;
      this.state = 0n;
      this._step();
      this.state = (this.state + (BigInt(seed) & MASK64)) & MASK64;
      this._step();
    }

    _step() {
      const oldstate = this.state;
      this.state = (oldstate * PCG_MULT + this.inc) & MASK64;
      const xorshifted = (((oldstate >> 18n) ^ oldstate) >> 27n) & MASK32;
      const rot = (oldstate >> 59n) & 31n;
      const leftShift = (32n - rot) & 31n;
      const rotated =
        ((xorshifted >> rot) | (xorshifted << leftShift)) & MASK32;
      return Number(rotated);
    }

    nextUint32() {
      return this._step();
    }

    random() {
      return this._step() / TWO_TO_32;
    }

    randomArray(n) {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = this._step() / TWO_TO_32;
      return out;
    }
  }

  global.DeterministicRng = DeterministicRng;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = DeterministicRng;
  }
})(typeof self !== 'undefined' ? self : globalThis);
