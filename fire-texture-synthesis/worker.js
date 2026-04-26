// This file is a derivative work of the C++ implementation released
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
// Standalone Web Worker entry point for sound texture synthesis. The live
// browser demo (main.js) does NOT load this file directly; instead it
// fetches kdtree.js / deterministic_rng.js / texture_synthesis.js with
// cache: 'no-store' and bundles them with an inline message handler into
// a Blob URL Worker (mirroring the sibling fire-bandwidth-extension demo)
// to bypass aggressive browser caching that has caused "Script error."
// failures in the past. This file is preserved as a reference
// implementation of the worker's message protocol.

'use strict';

importScripts('kdtree.js?v=12', 'deterministic_rng.js?v=12', 'texture_synthesis.js?v=12');

self.addEventListener('message', (event) => {
  const { id, base, training, params } = event.data;
  try {
    const opts = Object.assign({}, params);
    opts.onProgress = (info) => {
      self.postMessage({ id: id, type: 'progress', info: info });
    };
    const out = TextureSynthesis.synthesize(
      new Float64Array(base),
      new Float64Array(training),
      opts
    );
    const outF32 = Float32Array.from(out.output);
    const scalingF32 = Float32Array.from(out.scalingPerWindow);
    self.postMessage(
      {
        id: id,
        type: 'done',
        output: outF32,
        scalingPerWindow: scalingF32,
      },
      [outF32.buffer, scalingF32.buffer]
    );
  } catch (err) {
    self.postMessage({
      id: id,
      type: 'error',
      message: (err && err.message) || String(err),
    });
  }
});
