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
// JS port of python/texture_synthesis.py, python/gaussian_pyramid.py, and
// python/cdf_match.py. Mirrors the Python module structure inline so the
// diff is auditable. Depends on KDTreeMulti (kdtree.js) and DeterministicRng
// (deterministic_rng.js) being defined on the same global scope.

'use strict';

(function (global) {
  const KDT = global.KDTreeMulti
    || (typeof require !== 'undefined' ? require('./kdtree.js') : null);
  if (!KDT) throw new Error('kdtree.js must be loaded before texture_synthesis.js');

  // ---- 5-tap Gaussian filter constants (mirrors GaussianPyramid::GAUSSIAN_STENCIL).
  const GAUSSIAN_STENCIL = [0.05, 0.25, 0.40, 0.25, 0.05];
  const GAUSSIAN_STENCIL_HALF = 2;
  const AMPLITUDE_CUTOFF = 1e-4;

  // ---------- Pad / pyramid level construction ----------

  function padInputSignal(signal, reflectBoundaries) {
    const n = signal.length;
    let extendedLength = 1;
    while (extendedLength + 1 < n) extendedLength *= 2;
    extendedLength += 1;
    const out = new Float64Array(extendedLength);
    const start = ((extendedLength - n) / 2) | 0;
    const end = start + n;
    for (let i = 0; i < n; i++) out[start + i] = signal[i];
    if (reflectBoundaries) {
      for (let i = 0; i < start; i++) {
        const src = start - i;
        if (src >= 0 && src < n) out[i] = signal[src];
      }
      for (let i = end; i < extendedLength; i++) {
        const src = end - (i - end) - 2;
        if (src >= 0 && src < extendedLength) out[i] = out[src];
      }
    }
    return { padded: out, start: start, end: end };
  }

  function buildGaussianLevel(base) {
    if ((base.length - 1) % 2 !== 0) {
      throw new Error('Base level size must be 2^k + 1; got ' + base.length);
    }
    const nextSize = ((base.length - 1) >> 1) + 1;
    const next = new Float64Array(nextSize);
    const halfWidth = GAUSSIAN_STENCIL_HALF;
    const baseLen = base.length;
    for (let i = 0; i < nextSize; i++) {
      const middle = i * 2;
      const startIdx = Math.max(0, middle - halfWidth);
      const endIdx = Math.min(baseLen - 1, middle + halfWidth);
      const filterStart = startIdx - middle + halfWidth;
      const numComponents = endIdx - startIdx + 1;
      let acc = 0;
      // FIXED off-by-one vs C++ (loop uses < num_components, not <=).
      for (let j = 0; j < numComponents; j++) {
        acc += base[startIdx + j] * GAUSSIAN_STENCIL[filterStart + j];
      }
      next[i] = acc;
    }
    return next;
  }

  function sampleSignal(signal, index) {
    const n = signal.length;
    const floorIdx = Math.floor(index);
    const i1 = floorIdx;
    const i2 = i1 + 1;
    const v1 = (i1 >= 0 && i1 < n) ? signal[i1] : 0.0;
    const v2 = (i2 >= 0 && i2 < n) ? signal[i2] : 0.0;
    const inside = (i1 >= 0 && i1 < n) && (i2 >= 0 && i2 < n);
    const frac = index - floorIdx;
    return { value: (1 - frac) * v1 + frac * v2, inside: inside };
  }

  // ---------- CDF helpers ----------

  function initCdfFromLevel(level, start, end) {
    const out = new Float64Array(end - start);
    for (let i = 0; i < out.length; i++) out[i] = Math.abs(level[start + i]);
    out.sort(); // numerical sort for TypedArray
    return out;
  }

  function sampleCdf(fraction, cdf) {
    const n = cdf.length;
    if (n === 0) return 0;
    if (n === 1) return cdf[0];
    const idxReal = fraction * (n - 1);
    const idx1 = idxReal | 0;
    const idx2 = idx1 + 1;
    const idx1c = Math.max(0, Math.min(n - 1, idx1));
    const idx2c = Math.max(0, Math.min(n - 1, idx2));
    const blend1 = idx2 - idxReal;
    const blend2 = 1 - blend1;
    return blend1 * cdf[idx1c] + blend2 * cdf[idx2c];
  }

  function sampleInverseCdf(amplitude, cdf) {
    const n = cdf.length;
    if (n === 0) return 0;
    if (amplitude < cdf[0]) return 0;
    if (amplitude >= cdf[n - 1]) return 1;
    let idx1 = binarySearch(cdf, amplitude, 0, n - 1);
    idx1 = Math.max(0, Math.min(n - 2, idx1));
    const idx2 = idx1 + 1;
    const a1 = cdf[idx1];
    const a2 = cdf[idx2];
    const diff = a2 - a1;
    if (diff === 0) return idx1 / (n - 1);
    const blend1 = (a2 - amplitude) / diff;
    const blend2 = 1 - blend1;
    return blend1 * (idx1 / (n - 1)) + blend2 * (idx2 / (n - 1));
  }

  function binarySearch(data, value, start, end) {
    while (start <= end) {
      const mid = (start + end) >> 1;
      if (mid + 1 < data.length && data[mid] <= value && value < data[mid + 1]) {
        return mid;
      } else if (value >= data[mid]) {
        start = mid + 1;
      } else {
        end = mid - 1;
      }
    }
    return start;
  }

  // ---------- GaussianPyramid ----------

  class GaussianPyramid {
    constructor(signal, numLevels, reflectBoundaries) {
      const padded = padInputSignal(signal, !!reflectBoundaries);
      this._levels = [padded.padded];
      this._startIndex = [padded.start];
      this._endIndex = [padded.end];
      for (let i = 1; i < numLevels; i++) {
        this._levels.push(buildGaussianLevel(this._levels[i - 1]));
        const ps = this._startIndex[i - 1];
        const pe = this._endIndex[i - 1];
        this._startIndex.push((ps >> 1) + (ps % 2 !== 0 ? 1 : 0));
        this._endIndex.push((pe >> 1) + (pe % 2 !== 0 ? 1 : 0));
      }
      this._cdf = null;
    }

    get levels() { return this._levels; }
    get cdf() { return this._cdf; }
    get numLevels() { return this._levels.length; }
    startIndex(level) { return this._startIndex[level]; }
    endIndex(level) { return this._endIndex[level]; }

    zeroLevel(level) {
      const lvl = this._levels[level];
      for (let i = 0; i < lvl.length; i++) lvl[i] = 0;
    }

    initCdf() {
      const top = this._levels[this._levels.length - 1];
      this._cdf = initCdfFromLevel(
        top,
        this._startIndex[this._startIndex.length - 1],
        this._endIndex[this._endIndex.length - 1]
      );
    }

    reconstructSignal() {
      const lvl0 = this._levels[0];
      const start = this._startIndex[0];
      const end = this._endIndex[0];
      const out = new Float64Array(end - start);
      for (let i = 0; i < out.length; i++) out[i] = lvl0[start + i];
      return out;
    }

    numWindows(windowHw, level) {
      return ((this._levels[level].length / windowHw) | 0) + 1;
    }

    computeWindowFeature(
      windowHWArr, featureHWArr, level, windowIdx,
      falloff, inputCdf, outputCdf, scalingAlpha
    ) {
      const numLevels = this._levels.length;
      const wh = windowHWArr[level];
      const fh = featureHWArr[level];
      const windowMiddle = windowIdx * wh;
      const featureStart = windowMiddle - wh * (fh + 1);
      const featureEnd = windowMiddle - wh;
      const nCausal = featureEnd - featureStart + 1;
      let nCoarser = 0;
      let whC = 0, fhC = 0;
      if (level < numLevels - 1) {
        whC = windowHWArr[level + 1];
        fhC = featureHWArr[level + 1];
        nCoarser = 2 * whC * (fhC + 1) + 1;
      }
      const feature = new Float64Array(nCausal + nCoarser);
      let featIdx = 0;
      let allInside = true;
      let scale = 1.0;

      const levelData = this._levels[level];
      const levelLen = levelData.length;
      const startVal = this._startIndex[level];
      const endVal = this._endIndex[level];
      for (let i = featureStart; i <= featureEnd; i++) {
        const inRange = (i >= startVal && i < endVal);
        allInside = allInside && inRange;
        const v = (i >= 0 && i < levelLen) ? levelData[i] : 0.0;
        const w = (falloff > 0)
          ? Math.exp(-falloff * Math.abs(i - windowMiddle))
          : 1.0;
        feature[featIdx++] = v * w;
      }

      if (level === numLevels - 1) {
        return { feature: feature, allInside: allInside, scale: scale };
      }

      const nextData = this._levels[level + 1];
      const windowMiddleReal = windowMiddle / 2.0;
      const featureStartReal = windowMiddleReal - whC * (fhC + 1);
      const featureLength = 2 * whC * (fhC + 1) + 1;

      const coarserStartInFeature = featIdx;
      let avgMagnitude = 0;

      for (let i = 0; i < featureLength; i++) {
        const sampleReal = featureStartReal + i;
        const s = sampleSignal(nextData, sampleReal);
        allInside = allInside && s.inside;
        let v = s.value;
        if (falloff > 0) {
          v *= Math.exp(-falloff * Math.abs(sampleReal - windowMiddleReal));
        }
        feature[featIdx] = v;
        avgMagnitude += Math.abs(v);
        featIdx++;
      }

      avgMagnitude /= featureLength;

      if (
        level + 1 === numLevels - 1
        && inputCdf !== null && inputCdf !== undefined
        && outputCdf !== null && outputCdf !== undefined
      ) {
        if (avgMagnitude > AMPLITUDE_CUTOFF) {
          const outputFraction = sampleInverseCdf(avgMagnitude, outputCdf);
          const inputMagnitude = sampleCdf(outputFraction, inputCdf);
          let scaling = inputMagnitude / avgMagnitude;
          scaling = (1 - scalingAlpha) + scalingAlpha * scaling;
          if (scaling !== 0 && !isNaN(scaling) && isFinite(scaling)) {
            scale = 1.0 / scaling;
            for (let i = 0; i < featureLength; i++) {
              feature[coarserStartInFeature + i] *= scaling;
            }
          }
        }
      }

      return { feature: feature, allInside: allInside, scale: scale };
    }
  }

  // ---------- Per-window blend ----------

  function blendWindowData(input, inputWindow, output, outputWindow, windowHw, scale) {
    const wsi = windowHw * (inputWindow - 1);
    const wso = windowHw * (outputWindow - 1);
    const windowSize = 2 * windowHw + 1;
    const invHw = 1 / windowHw;
    const outLen = output.length;
    const inLen = input.length;
    for (let i = 0; i < windowSize; i++) {
      const si = wsi + i;
      const so = wso + i;
      if (so < 0 || so >= outLen || si < 0 || si >= inLen) continue;
      const blend = (windowHw - Math.abs(windowHw - i)) * invHw * scale;
      output[so] += blend * input[si];
    }
  }

  // ---------- Top-level synthesis ----------

  // Build the per-level training pyramid + feature dictionaries. Returned
  // value can be passed back into synthesize() as opts.trainingCache to
  // skip the (expensive) feature-extraction and KD-tree-build phase when
  // the user re-runs synthesis with the same training signal and the
  // same numLevels/windowHw/featureHw/falloff parameters. Caching does
  // NOT depend on scaleCdf / scalingAlpha / epsilon (those are applied
  // per-window during synthesis, after the dictionary is built).
  function buildTrainingDict(training, opts) {
    opts = opts || {};
    const numLevels = (opts.numLevels === undefined) ? 6 : opts.numLevels;
    const windowHw = (opts.windowHw === undefined) ? 4 : opts.windowHw;
    const featureHw = (opts.featureHw === undefined) ? 3 : opts.featureHw;
    const falloff = (opts.falloff === undefined) ? 0.0 : opts.falloff;
    const onProgress = opts.onProgress || null;
    const profile = opts.profile || null;
    const now = () => (typeof performance !== 'undefined') ? performance.now() : Date.now();

    const trainingF64 = training instanceof Float64Array ? training : Float64Array.from(training);

    let t0 = now();
    const trainingPyr = new GaussianPyramid(trainingF64, numLevels, true);
    if (profile) profile.training_pyramid_ms = now() - t0;

    const windowHWArr = new Array(numLevels);
    const featureHWArr = new Array(numLevels);
    for (let i = 0; i < numLevels; i++) {
      windowHWArr[i] = windowHw;
      featureHWArr[i] = featureHw;
    }

    const levelTrees = new Array(numLevels);
    for (let i = 0; i < numLevels; i++) levelTrees[i] = null;

    let nFeaturesBuilt = 0;

    for (let level = 0; level < numLevels - 1; level++) {
      const tBuild = now();
      if (onProgress) {
        onProgress({ stage: 'build_feat', level: level, progress: 0,
                      n_levels: numLevels - 1 });
      }
      const nWindows = trainingPyr.numWindows(windowHw, level);
      const features = [];
      const indices = [];
      let dim = 0;
      // Emit intra-level progress every ~256 windows (only meaningful at
      // level 0 where the count is in the tens of thousands).
      const reportEvery = Math.max(256, (nWindows / 32) | 0);
      for (let w = 0; w < nWindows; w++) {
        if (onProgress && level === 0 && (w % reportEvery) === 0 && w > 0) {
          onProgress({ stage: 'build_feat', level: level,
                        progress: w / nWindows, n_levels: numLevels - 1 });
        }
        const r = trainingPyr.computeWindowFeature(
          windowHWArr, featureHWArr, level, w, falloff, null, null, 1.0);
        if (r.allInside) {
          features.push(r.feature);
          indices.push(w);
          dim = r.feature.length;
        }
      }
      nFeaturesBuilt += features.length;
      if (features.length === 0) {
        if (profile) profile.build_per_level_ms.push(now() - tBuild);
        continue;
      }
      if (onProgress) {
        onProgress({ stage: 'build_tree', level: level, progress: 0,
                      n_levels: numLevels - 1 });
      }
      const flat = new Float64Array(features.length * dim);
      for (let i = 0; i < features.length; i++) {
        flat.set(features[i], i * dim);
      }
      const idxArr = new Int32Array(indices);
      levelTrees[level] = new KDT(flat, dim, idxArr);
      const dt = now() - tBuild;
      if (profile) {
        profile.build_per_level_ms.push(dt);
        profile.build_ms += dt;
      }
      if (onProgress) {
        onProgress({ stage: 'build_done', level: level,
                      progress: (level + 1) / numLevels,
                      n_levels: numLevels - 1 });
      }
    }
    if (profile) profile.n_features_built = nFeaturesBuilt;
    return {
      trainingPyramid: trainingPyr,
      levelTrees: levelTrees,
      // Capture the params that this dictionary depends on, so the
      // cache layer can verify a hit.
      params: { numLevels, windowHw, featureHw, falloff },
    };
  }

  function synthesize(base, training, opts) {
    opts = opts || {};
    const numLevels = (opts.numLevels === undefined) ? 6 : opts.numLevels;
    const windowHw = (opts.windowHw === undefined) ? 4 : opts.windowHw;
    const featureHw = (opts.featureHw === undefined) ? 3 : opts.featureHw;
    const falloff = (opts.falloff === undefined) ? 0.0 : opts.falloff;
    const scaleCdf = (opts.scaleCdf === undefined) ? true : opts.scaleCdf;
    const scalingAlpha = (opts.scalingAlpha === undefined) ? 1.0 : opts.scalingAlpha;
    const epsilon = (opts.epsilon === undefined) ? 0.0 : opts.epsilon;
    const onProgress = opts.onProgress || null;
    const trainingCache = opts.trainingCache || null;

    const profile = {
      pyramid_ms: 0, training_pyramid_ms: 0, cdf_ms: 0, build_ms: 0, synth_ms: 0,
      feature_ms: 0, query_ms: 0, blend_ms: 0,
      build_per_level_ms: [], synth_per_level_ms: [],
      n_queries: 0, n_features_built: 0, total_ms: 0,
      cache_hit: false,
    };
    const tStart = (typeof performance !== 'undefined') ? performance.now() : Date.now();
    const now = () => (typeof performance !== 'undefined') ? performance.now() : Date.now();

    const baseF64 = base instanceof Float64Array ? base : Float64Array.from(base);

    // Output pyramid is always built fresh from the base.
    let t0 = now();
    if (onProgress) onProgress({ stage: 'pyramid', level: -1, progress: 0 });
    const outputPyr = new GaussianPyramid(baseF64, numLevels, false);
    profile.pyramid_ms = now() - t0;

    // Reuse cached training dictionary if it matches.
    let dict = null;
    if (trainingCache) {
      const p = trainingCache.params;
      if (p.numLevels === numLevels && p.windowHw === windowHw
          && p.featureHw === featureHw && p.falloff === falloff) {
        dict = trainingCache;
        profile.cache_hit = true;
      }
    }
    if (!dict) {
      dict = buildTrainingDict(training, {
        numLevels, windowHw, featureHw, falloff,
        onProgress: onProgress, profile: profile,
      });
    }
    const trainingPyr = dict.trainingPyramid;
    const levelTrees = dict.levelTrees;

    t0 = now();
    if (scaleCdf) {
      // Recompute the training CDF on the cached pyramid only if it hasn't
      // already been computed (CDF of the top level is content-dependent
      // but doesn't depend on the per-window scaling parameters).
      if (trainingPyr.cdf === null) trainingPyr.initCdf();
      outputPyr.initCdf();
    }
    profile.cdf_ms = now() - t0;

    const windowHWArr = new Array(numLevels);
    const featureHWArr = new Array(numLevels);
    for (let i = 0; i < numLevels; i++) {
      windowHWArr[i] = windowHw;
      featureHWArr[i] = featureHw;
    }

    // Zero output below numBaseLevels = 1 (matches the C++ default).
    const numBaseLevels = 1;
    for (let level = 0; level < numLevels - numBaseLevels; level++) {
      outputPyr.zeroLevel(level);
    }

    const scalingPerWindow = [];
    for (let level = numLevels - 1 - numBaseLevels; level >= 0; level--) {
      const tLevel = now();
      const tree = levelTrees[level];
      if (!tree) {
        profile.synth_per_level_ms.push(0);
        continue;
      }
      const trainingData = trainingPyr.levels[level];
      const outputData = outputPyr.levels[level];
      const nWindowsOut = outputPyr.numWindows(windowHw, level);
      const inputCdf = scaleCdf ? trainingPyr.cdf : null;
      const outputCdf = scaleCdf ? outputPyr.cdf : null;
      let featAccum = 0, queryAccum = 0, blendAccum = 0;

      for (let w = 0; w < nWindowsOut; w++) {
        if (onProgress && (w & 31) === 0) {
          onProgress({ stage: 'synth', level: level, progress: w / nWindowsOut });
        }
        const tFeat = now();
        const r = outputPyr.computeWindowFeature(
          windowHWArr, featureHWArr, level, w, falloff,
          inputCdf, outputCdf, scalingAlpha);
        let scale = r.scale;
        if (isNaN(scale) || !isFinite(scale)) scale = 1.0;
        scalingPerWindow.push(scaleCdf ? scale : 1.0);
        const tQuery = now();
        featAccum += tQuery - tFeat;
        const nn = tree.nearestNeighbour(r.feature, epsilon);
        const tBlend = now();
        queryAccum += tBlend - tQuery;
        if (nn.index >= 0) {
          blendWindowData(trainingData, nn.index, outputData, w, windowHw, scale);
        }
        blendAccum += now() - tBlend;
        profile.n_queries += 1;
      }
      profile.feature_ms += featAccum;
      profile.query_ms += queryAccum;
      profile.blend_ms += blendAccum;
      const dt = now() - tLevel;
      profile.synth_per_level_ms.push(dt);
      profile.synth_ms += dt;
    }

    if (onProgress) onProgress({ stage: 'done', level: 0, progress: 1.0 });

    profile.total_ms = now() - tStart;

    return {
      output: outputPyr.reconstructSignal(),
      outputPyramid: outputPyr.levels,
      trainingPyramid: trainingPyr.levels,
      inputCdf: trainingPyr.cdf,
      outputCdf: outputPyr.cdf,
      scalingPerWindow: Float64Array.from(scalingPerWindow),
      profile: profile,
      // Caller can stash this and pass it back via opts.trainingCache
      // on the next call to skip the build phase.
      trainingCache: dict,
    };
  }

  global.TextureSynthesis = {
    synthesize: synthesize,
    buildTrainingDict: buildTrainingDict,
    GaussianPyramid: GaussianPyramid,
    buildGaussianLevel: buildGaussianLevel,
    padInputSignal: padInputSignal,
    sampleCdf: sampleCdf,
    sampleInverseCdf: sampleInverseCdf,
    initCdfFromLevel: initCdfFromLevel,
    blendWindowData: blendWindowData,
  };
})(typeof self !== 'undefined' ? self : globalThis);
