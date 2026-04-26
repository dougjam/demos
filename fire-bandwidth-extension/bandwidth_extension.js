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
// JS port of python/bandwidth_extension.py. Function-for-function mirror so
// the diff is auditable. Defaults match Matlab. The optional small
// per-window FFT path (spec section 4.5) is enabled by default for
// interactivity; pass {nfftPerWindow: null} to fall back to the full-NFFT
// Matlab-faithful path used by the parity test.

'use strict';

(function (global) {
  const FFT = global.FFT || (typeof require !== 'undefined' ? require('./fft.js') : null);
  if (!FFT) throw new Error('fft.js must be loaded before bandwidth_extension.js');

  // ---------- numerical helpers ----------

  function nextPow2(n) {
    if (n <= 1) return 1;
    return 1 << (32 - Math.clz32(n - 1));
  }

  function frequencyAxis(fs, NFFT) {
    const x = new Float64Array(NFFT);
    const inv = 1.0 / (NFFT - 1);
    for (let i = 0; i < NFFT; i++) x[i] = fs * i * inv;
    return x;
  }

  // ---------- lowpass filter (order-2 Butterworth + filtfilt) ----------

  function butterLowpass2(fc, fs) {
    // Bilinear transform with frequency pre-warping. Matches scipy.signal.butter(2, fc/(fs/2), 'low').
    const Wn = Math.tan(Math.PI * fc / fs);
    const Wn2 = Wn * Wn;
    const sqrt2Wn = Math.SQRT2 * Wn;
    const norm = 1 + sqrt2Wn + Wn2;
    return {
      b: [Wn2 / norm, 2 * Wn2 / norm, Wn2 / norm],
      a: [1, 2 * (Wn2 - 1) / norm, (1 - sqrt2Wn + Wn2) / norm],
    };
  }

  function lfilterZi(b, a) {
    // Steady-state initial conditions for direct-form-II-transposed order-2 filter, a[0] = 1.
    // Matches scipy.signal.lfilter_zi.
    const a1 = a[1], a2 = a[2];
    const b0 = b[0], b1 = b[1], b2 = b[2];
    const d0 = b1 - a1 * b0;
    const d1 = b2 - a2 * b0;
    const det = 1 + a1 + a2;
    const z0 = (d0 + d1) / det;
    const z1 = d1 - a2 * z0;
    return [z0, z1];
  }

  function lfilterDF2T(b, a, x, zi0, zi1) {
    // Direct form II transposed with optional initial state.
    const n = x.length;
    const y = new Float64Array(n);
    let z0 = zi0, z1 = zi1;
    const b0 = b[0], b1 = b[1], b2 = b[2];
    const a1 = a[1], a2 = a[2];
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      const yi = b0 * xi + z0;
      z0 = b1 * xi - a1 * yi + z1;
      z1 = b2 * xi - a2 * yi;
      y[i] = yi;
    }
    return y;
  }

  function lowpassFilter(p, fs, fCutoff) {
    const { b, a } = butterLowpass2(fCutoff, fs);
    // scipy.signal.filtfilt uses padlen = 3 * max(len(a), len(b)) by default;
    // for our order-2 Butterworth that's 3 * 3 = 9. Using 3 * order = 6 is a
    // common mistake that pushes the boundary error from 1e-13 to 1e-2.
    const padlen = 3 * Math.max(b.length, a.length);
    const n = p.length;
    if (n <= padlen + 1) {
      // Plain forward+backward without padding.
      const fwd = lfilterDF2T(b, a, p, 0, 0);
      const rev = new Float64Array(n);
      for (let i = 0; i < n; i++) rev[i] = fwd[n - 1 - i];
      const bwd = lfilterDF2T(b, a, rev, 0, 0);
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = bwd[n - 1 - i];
      return out;
    }
    // Odd extension on both sides (matches scipy default padtype).
    const m = n + 2 * padlen;
    const ext = new Float64Array(m);
    for (let i = 0; i < padlen; i++) ext[i] = 2 * p[0] - p[padlen - i];
    for (let i = 0; i < n; i++) ext[padlen + i] = p[i];
    for (let i = 0; i < padlen; i++) ext[padlen + n + i] = 2 * p[n - 1] - p[n - 2 - i];

    const zi = lfilterZi(b, a);
    // Forward pass scaled by first sample of input.
    const fwd = lfilterDF2T(b, a, ext, zi[0] * ext[0], zi[1] * ext[0]);
    // Reverse, scale init by first sample of reversed signal, backward pass.
    const rev = new Float64Array(m);
    for (let i = 0; i < m; i++) rev[i] = fwd[m - 1 - i];
    const bwd = lfilterDF2T(b, a, rev, zi[0] * rev[0], zi[1] * rev[0]);
    // Reverse and trim padding.
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = bwd[m - 1 - padlen - i];
    return out;
  }

  // ---------- window / blend / gaussian / powerlaw kernels ----------

  function buildWindowFunctionLinear(halfWidth, L) {
    const sz = 4 * L + 1;
    const w = new Float64Array(sz);
    const center = 2 * L;
    const inv = 1.0 / halfWidth;
    for (let i = 0; i < sz; i++) {
      const off = i - center;
      if (off < -halfWidth || off > halfWidth) continue;
      if (off <= 0) w[i] = (off + halfWidth) * inv;
      else w[i] = (halfWidth - off) * inv;
    }
    return w;
  }

  function buildBlendingFunctionLinear(x, NFFT, f1, f2) {
    const b1 = new Float64Array(NFFT);
    const b2 = new Float64Array(NFFT);
    if (f2 === f1) {
      for (let i = 0; i < NFFT; i++) {
        b1[i] = x[i] <= f1 ? 1.0 : 0.0;
        b2[i] = 1.0 - b1[i];
      }
      return [b1, b2];
    }
    const denom = f2 - f1;
    for (let i = 0; i < NFFT; i++) {
      let t = (x[i] - f1) / denom;
      if (t < 0) t = 0;
      else if (t > 1) t = 1;
      b1[i] = 1 - t;
      b2[i] = t;
    }
    return [b1, b2];
  }

  function buildBlurringFunctionGaussian(x, fMiddle, fWidth, NFFT) {
    const sigma = fWidth / 3.0;
    const denom = 2.0 * sigma * sigma;
    const g = new Float64Array(NFFT);
    for (let i = 0; i < NFFT; i++) {
      const d = x[i] - fMiddle;
      g[i] = Math.exp(-d * d / denom);
    }
    return g;
  }

  // Powerlaw spectrum stored as fft.js complex array (length 2*NFFT, interleaved).
  function buildPowerlawSpectrum(exponent, x, NFFT, opts) {
    const Z = new Float64Array(2 * NFFT);
    const phasesOverride = opts.phasesOverride || null;
    const rng = opts.rng || null;
    if (phasesOverride) {
      if (phasesOverride.length !== NFFT) {
        throw new Error(
          `phasesOverride length ${phasesOverride.length} != NFFT ${NFFT}`
        );
      }
      for (let j = 0; j < NFFT; j++) {
        const xj = x[j];
        if (xj === 0.0) continue;
        const mag = Math.pow(xj, exponent);
        const phi = phasesOverride[j];
        Z[2 * j] = mag * Math.cos(phi);
        Z[2 * j + 1] = mag * Math.sin(phi);
      }
      return Z;
    }
    if (!rng) throw new Error('Either rng or phasesOverride must be provided');
    const TWO_PI = 2.0 * Math.PI;
    for (let j = 0; j < NFFT; j++) {
      const xj = x[j];
      if (xj === 0.0) continue;
      const mag = Math.pow(xj, exponent);
      const phi = rng.random() * TWO_PI;
      Z[2 * j] = mag * Math.cos(phi);
      Z[2 * j + 1] = mag * Math.sin(phi);
    }
    return Z;
  }

  // ---------- complex array helpers (operate on interleaved [re, im] buffers) ----------

  function multiplyComplexByReal(C, R, NFFT) {
    // C[k] *= R[k] elementwise.
    for (let k = 0, j = 0; k < NFFT; k++, j += 2) {
      const r = R[k];
      C[j] *= r;
      C[j + 1] *= r;
    }
  }

  function multiplyComplexByRealInto(out, C, R, NFFT) {
    for (let k = 0, j = 0; k < NFFT; k++, j += 2) {
      const r = R[k];
      out[j] = C[j] * r;
      out[j + 1] = C[j + 1] * r;
    }
  }

  function addScaledComplexInto(out, A, B, scale, NFFT) {
    // out[k] = A[k] + scale * B[k] elementwise on complex arrays.
    const len = 2 * NFFT;
    for (let i = 0; i < len; i++) out[i] = A[i] + scale * B[i];
  }

  // ---------- beta quadratic ----------

  function fitDualPowerSpectra(x, Y_L, Y_N, Y_S, f, NFFT) {
    // All complex inputs are interleaved Float64 of length 2*NFFT.
    // Trapezoidal integral over the two-sided frequency axis x.
    // Quadratic a*beta^2 + b*beta + c = 0 with stable Vieta branch.
    let I1 = 0, I2 = 0, I3 = 0, I4 = 0, I5 = 0, I6 = 0, Is = 0;
    const Lm1 = NFFT - 1;
    let ylr_a = Y_L[0], yli_a = Y_L[1];
    let ynr_a = Y_N[0], yni_a = Y_N[1];
    let ysr_a = Y_S[0], ysi_a = Y_S[1];
    let f_a = f[0];
    let x_a = x[0];
    let yss_a = ysr_a * ysr_a + ysi_a * ysi_a;
    for (let k = 0; k < Lm1; k++) {
      const j2 = 2 * (k + 1);
      const ylr_b = Y_L[j2], yli_b = Y_L[j2 + 1];
      const ynr_b = Y_N[j2], yni_b = Y_N[j2 + 1];
      const ysr_b = Y_S[j2], ysi_b = Y_S[j2 + 1];
      const f_b = f[k + 1];
      const x_b = x[k + 1];
      const yss_b = ysr_b * ysr_b + ysi_b * ysi_b;

      const z1_1 = ylr_a * ylr_a * f_a, z1_2 = ylr_b * ylr_b * f_b;
      const z2_1 = ynr_a * ynr_a * f_a, z2_2 = ynr_b * ynr_b * f_b;
      const z3_1 = 2 * ylr_a * ynr_a * f_a, z3_2 = 2 * ylr_b * ynr_b * f_b;
      const z4_1 = yli_a * yli_a * f_a, z4_2 = yli_b * yli_b * f_b;
      const z5_1 = yni_a * yni_a * f_a, z5_2 = yni_b * yni_b * f_b;
      const z6_1 = 2 * yli_a * yni_a * f_a, z6_2 = 2 * yli_b * yni_b * f_b;
      const zs_1 = yss_a * f_a, zs_2 = yss_b * f_b;

      // 0.5 * (X2 z1 - X1 z2 + X2 z2 - X1 z1) = 0.5 * (X2 - X1) * (z1 + z2)
      const w = 0.5 * (x_b - x_a);
      I1 += w * (z1_1 + z1_2);
      I2 += w * (z2_1 + z2_2);
      I3 += w * (z3_1 + z3_2);
      I4 += w * (z4_1 + z4_2);
      I5 += w * (z5_1 + z5_2);
      I6 += w * (z6_1 + z6_2);
      Is += w * (zs_1 + zs_2);

      // Roll
      ylr_a = ylr_b; yli_a = yli_b;
      ynr_a = ynr_b; yni_a = yni_b;
      ysr_a = ysr_b; ysi_a = ysi_b;
      f_a = f_b; x_a = x_b; yss_a = yss_b;
    }
    const a = I2 + I5;
    const b = I3 + I6;
    const c = I1 + I4 - Is;

    const disc = b * b - 4 * a * c;
    if (disc < 0) throw new Error('Negative discriminant: complex roots found');

    if (a === 0) {
      if (b === 0) return 0;
      return -c / b;
    }
    const sd = Math.sqrt(disc);
    let beta1;
    if (b > 0) beta1 = (-b - sd) / (2 * a);
    else beta1 = (-b + sd) / (2 * a);
    if (beta1 === 0) return 0;
    const beta2 = c / (beta1 * a);
    if (beta1 > 0 && beta2 > 0) throw new Error('Two positive alphas');
    if (beta1 < 0 && beta2 < 0) throw new Error('Two negative alphas');
    return Math.max(beta1, beta2);
  }

  // ---------- top-level ----------

  function defaults() {
    return {
      fs: 44100.0,
      fCutoff: 180.0,
      halfWidth: 500,
      fitCenter: 180.0,
      fitWidth: 30.0,
      blendStart: 165.0,
      blendEnd: 195.0,
      noiseAmplitude: 1.0,
      alpha: 2.5,
      // null = Matlab-faithful (slow, used by the parity test).
      // 'auto' = use nextPow2(8 * halfWidth) for interactive speed (default).
      nfftPerWindow: 'auto',
      rng: null,
      phasesOverride: null,
      onProgress: null,
    };
  }

  function extendSignal(p, opts) {
    opts = Object.assign(defaults(), opts || {});
    const {
      fs, fCutoff, halfWidth, fitCenter, fitWidth, blendStart, blendEnd,
      noiseAmplitude, alpha, rng, phasesOverride, onProgress,
    } = opts;
    const L = p.length;
    if (L === 0) return { extended: new Float64Array(0), betas: new Float64Array(0) };

    const NFFTfull = nextPow2(L);
    let NFFTwin;
    if (opts.nfftPerWindow === null) {
      NFFTwin = NFFTfull;
    } else if (opts.nfftPerWindow === 'auto') {
      NFFTwin = Math.min(NFFTfull, Math.max(64, nextPow2(8 * halfWidth)));
    } else {
      NFFTwin = nextPow2(opts.nfftPerWindow | 0);
    }
    const useExtractPath = NFFTwin < NFFTfull;

    const xFull = frequencyAxis(fs, NFFTfull);
    const xWin = useExtractPath ? frequencyAxis(fs, NFFTwin) : xFull;

    const powerlawExponent = -0.5 * alpha;

    const pLowpass = lowpassFilter(p, fs, fCutoff);
    const absLowpass = new Float64Array(L);
    for (let i = 0; i < L; i++) absLowpass[i] = Math.abs(pLowpass[i]);

    // Per-window-NFFT versions of the spectral kernels.
    const fBlur = buildBlurringFunctionGaussian(xWin, fitCenter, fitWidth, NFFTwin);
    const [blend1, blend2] = buildBlendingFunctionLinear(xWin, NFFTwin, blendStart, blendEnd);

    // Noise generated at FULL NFFT so that per-window slices come from
    // different positions of the same noise pattern (preserves spec rule 7
    // even on the small-NFFT path).
    const blend2Full = useExtractPath
      ? buildBlendingFunctionLinear(xFull, NFFTfull, blendStart, blendEnd)[1]
      : blend2;
    const powerlawFull = buildPowerlawSpectrum(powerlawExponent, xFull, NFFTfull, {
      rng, phasesOverride,
    });
    multiplyComplexByReal(powerlawFull, blend2Full, NFFTfull);
    const noiseUnscaledFull = new Float64Array(2 * NFFTfull);
    const fftFull = new FFT(NFFTfull);
    fftFull.inverseTransform(noiseUnscaledFull, powerlawFull);

    const windowFunction = buildWindowFunctionLinear(halfWidth, L);

    const fftWin = (useExtractPath || NFFTwin !== NFFTfull) ? new FFT(NFFTwin) : fftFull;

    const pExtended = new Float64Array(L);
    const betas = [];

    // Per-window scratch buffers.
    const psubReal = new Float64Array(NFFTwin);
    const psubLowpass = new Float64Array(2 * NFFTwin);
    const Y = new Float64Array(2 * NFFTwin);
    const Ynoise = new Float64Array(2 * NFFTwin);
    const spectrumSignal = new Float64Array(2 * NFFTwin);
    const spectrumNoise = new Float64Array(2 * NFFTwin);
    const blended = new Float64Array(2 * NFFTwin);
    const blendedTime = new Float64Array(2 * NFFTwin);

    // Mirror Matlab indexing.
    let signalCenter = 1;
    let signalStart = signalCenter - halfWidth;
    let windowStart = 2 * L + 1; // 1-indexed into windowFunction
    let windowEnd = 3 * L;

    // Estimate progress total
    let windowsTotal = Math.max(1, Math.ceil(L / halfWidth) + 1);
    let windowIdx = 0;

    while (signalStart <= L) {
      // signal_end (inclusive in Matlab 1-index) = signal_center + half_width
      const signalEnd = signalCenter + halfWidth;
      const clipStart = Math.max(1, signalStart);    // 1-indexed
      const clipEnd = Math.min(L, signalEnd);        // 1-indexed inclusive
      const regionLen = clipEnd - clipStart + 1;

      // Re-zero scratch.
      psubReal.fill(0);
      psubLowpass.fill(0);

      // Window slice (full L) for either path.
      // window_function[window_start - 1 + (k - 1)] for k in 1..L
      // = window_function[window_start - 2 + k]
      // For the extract path, we use only the in-region portion.
      if (useExtractPath) {
        // Local window covers signal indices [clipStart, clipEnd].
        // Full L slice begins at windowFunction index (windowStart - 1) and runs L samples.
        // For signal-1-indexed k=clipStart..clipEnd, the corresponding windowFunction index is
        //   (windowStart - 1) + (k - 1) = windowStart - 2 + k.
        const wfBase = windowStart - 2;
        for (let r = 0; r < regionLen; r++) {
          const k1 = clipStart + r;
          const sigIdx = k1 - 1;
          const w = windowFunction[wfBase + k1];
          psubReal[r] = p[sigIdx] * w;
          // psub_lowpass = abs(p_lowpass) * w * noise_unscaled[sigIdx]
          const env = absLowpass[sigIdx] * w;
          const nIdx = 2 * sigIdx;
          psubLowpass[2 * r]     = env * noiseUnscaledFull[nIdx];
          psubLowpass[2 * r + 1] = env * noiseUnscaledFull[nIdx + 1];
        }
      } else {
        // Full-NFFT path: psubReal is length L padded to NFFTwin.
        const wfBase = windowStart - 1;
        for (let k = 0; k < L; k++) {
          const w = windowFunction[wfBase + k];
          psubReal[k] = p[k] * w;
          const env = absLowpass[k] * w;
          const nIdx = 2 * k;
          psubLowpass[2 * k]     = env * noiseUnscaledFull[nIdx];
          psubLowpass[2 * k + 1] = env * noiseUnscaledFull[nIdx + 1];
        }
      }

      // FFT of the windowed signal (real input → use realTransform + completeSpectrum).
      fftWin.realTransform(Y, psubReal);
      fftWin.completeSpectrum(Y);
      // FFT of the complex envelope-shaped noise.
      fftWin.transform(Ynoise, psubLowpass);

      // Spectrum_signal = Y * blend1, spectrum_noise = Ynoise * blend2 (per-bin real * complex).
      multiplyComplexByRealInto(spectrumSignal, Y, blend1, NFFTwin);
      multiplyComplexByRealInto(spectrumNoise, Ynoise, blend2, NFFTwin);

      const beta = fitDualPowerSpectra(xWin, spectrumSignal, spectrumNoise, Y, fBlur, NFFTwin);
      betas.push(beta);

      // Blended spectrum and IFFT.
      addScaledComplexInto(blended, spectrumSignal, spectrumNoise, beta * noiseAmplitude, NFFTwin);
      fftWin.inverseTransform(blendedTime, blended);

      // Overlap-add (real part only).
      if (useExtractPath) {
        for (let r = 0; r < regionLen; r++) {
          const sigIdx = (clipStart - 1) + r;
          pExtended[sigIdx] += blendedTime[2 * r];
        }
      } else {
        for (let k = 0; k < L; k++) pExtended[k] += blendedTime[2 * k];
      }

      windowIdx += 1;
      if (onProgress && (windowIdx & 7) === 0) {
        onProgress(Math.min(1.0, windowIdx / windowsTotal));
      }

      signalCenter += halfWidth;
      signalStart = signalCenter - halfWidth;
      windowStart -= halfWidth;
      windowEnd -= halfWidth;
    }

    if (onProgress) onProgress(1.0);

    return {
      extended: pExtended,
      betas: new Float64Array(betas),
      lowpass: pLowpass,
      nfftPerWindow: NFFTwin,
      nfftFull: NFFTfull,
    };
  }

  global.BandwidthExtension = {
    extendSignal,
    nextPow2,
    frequencyAxis,
    lowpassFilter,
    buildWindowFunctionLinear,
    buildBlendingFunctionLinear,
    buildBlurringFunctionGaussian,
    buildPowerlawSpectrum,
    fitDualPowerSpectra,
    butterLowpass2,
    lfilterZi,
    defaults,
  };
})(typeof self !== 'undefined' ? self : globalThis);
