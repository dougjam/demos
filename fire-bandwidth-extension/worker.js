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

// Standalone Web Worker entry point for the bandwidth extension. The live
// browser demo (main.js) does NOT load this file directly; instead it
// fetches fft.js / deterministic_rng.js / bandwidth_extension.js with
// cache: 'no-store' and bundles them with an inline message handler into a
// Blob URL Worker, to bypass aggressive browser caching that has caused
// "Script error." failures in the past. This file is preserved as a
// reference implementation of the worker's message protocol; if you ever
// want a "regular" Worker (e.g., from a build step that fingerprints
// filenames so caching is no longer a problem) just construct
// `new Worker('worker.js')`.

'use strict';

importScripts('fft.js?v=15', 'deterministic_rng.js?v=15', 'bandwidth_extension.js?v=15');

self.addEventListener('message', (event) => {
  const { id, input, params } = event.data;
  try {
    const opts = Object.assign({}, params);
    if (opts.seed !== undefined && opts.seed !== null) {
      opts.rng = new DeterministicRng(opts.seed);
      delete opts.seed;
    }
    opts.onProgress = (value) => {
      self.postMessage({ id, type: 'progress', value });
    };
    const out = BandwidthExtension.extendSignal(new Float64Array(input), opts);
    const extendedF32 = Float32Array.from(out.extended);
    const betas = Float32Array.from(out.betas);
    self.postMessage(
      {
        id,
        type: 'done',
        extended: extendedF32,
        betas,
        nfftPerWindow: out.nfftPerWindow,
        nfftFull: out.nfftFull,
      },
      [extendedF32.buffer, betas.buffer]
    );
  } catch (err) {
    self.postMessage({
      id,
      type: 'error',
      message: (err && err.message) || String(err),
    });
  }
});
