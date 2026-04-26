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

'use strict';

// ---------- defaults / paper preset ----------
const DEFAULTS = {
  alpha: 2.5,
  f_cutoff: 180.0,
  half_width: 500,
  fit_width: 30.0,
  blend_half_width: 15.0,
  noise_amplitude: 1.0,
  seed: 42,
};

// ---------- audio context & playback ----------
let audioCtx = null;
let manifest = null;
let currentName = null;
let originalBuffer = null;     // Float32Array (signal samples at 44100 Hz)
let originalDuration = 0;
let extendedBuffer = null;     // Float32Array (most recent processing output)
let extendedBetas = null;
let worker = null;
let nextRequestId = 0;
let pendingRequest = null;

const playState = {
  original: { node: null, startedAt: 0, offset: 0, raf: null },
  extended: { node: null, startedAt: 0, offset: 0, raf: null },
};

function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') audioCtx.resume();
}

function makeAudioBuffer(samples) {
  ensureAudio();
  const buf = audioCtx.createBuffer(1, samples.length, 44100);
  buf.copyToChannel(samples, 0);
  return buf;
}

function makePlayer(which, getSamples) {
  const root = document.getElementById('player-' + which);
  const playBtn = root.querySelector('.play');
  const stopBtn = root.querySelector('.stop');
  const seek = root.querySelector('input[type="range"]');
  const meta = root.querySelector('.meta');
  const state = playState[which];

  function tick() {
    const samples = getSamples();
    if (!samples) return;
    if (!state.node) return;
    const elapsed = audioCtx.currentTime - state.startedAt + state.offset;
    const dur = samples.length / 44100;
    seek.value = String(Math.min(1, elapsed / dur));
    meta.textContent = `${elapsed.toFixed(2)} / ${dur.toFixed(2)} s   (${samples.length} samples)`;
    if (elapsed < dur) state.raf = requestAnimationFrame(tick);
    else stopPlayer(which);
  }

  function startPlayer(fromOffset) {
    const samples = getSamples();
    if (!samples) return;
    stopPlayer(which);
    ensureAudio();
    const audioBuf = makeAudioBuffer(samples);
    const node = audioCtx.createBufferSource();
    node.buffer = audioBuf;
    node.connect(audioCtx.destination);
    state.offset = Math.min(audioBuf.duration - 0.01, Math.max(0, fromOffset));
    state.startedAt = audioCtx.currentTime;
    state.node = node;
    node.onended = () => { if (state.node === node) stopPlayer(which); };
    node.start(0, state.offset);
    state.raf = requestAnimationFrame(tick);
    playBtn.textContent = 'Pause';
  }

  function stopPlayer() {
    if (state.node) {
      try { state.node.stop(); } catch (_) {}
      state.node.disconnect();
      state.node = null;
    }
    if (state.raf) { cancelAnimationFrame(state.raf); state.raf = null; }
    playBtn.textContent = 'Play';
  }

  playBtn.addEventListener('click', () => {
    if (state.node) {
      const samples = getSamples();
      const dur = samples.length / 44100;
      const elapsed = audioCtx.currentTime - state.startedAt + state.offset;
      // Race guard: if the clip already finished but neither tick() nor
      // onended has cleared state.node yet, treat the click as
      // "restart from the beginning" rather than as a pause-at-end.
      if (elapsed >= dur - 0.02) {
        stopPlayer();
        state.offset = 0;
        seek.value = '0';
        startPlayer(0);
      } else {
        stopPlayer();
        state.offset = Math.min(dur - 0.01, elapsed);
        meta.textContent = `paused at ${state.offset.toFixed(2)} s`;
      }
    } else {
      const samples = getSamples();
      if (!samples) return;
      const dur = samples.length / 44100;
      let seekFrac = parseFloat(seek.value) || 0;
      // If the previous playback ran to the end (or essentially the end),
      // rewind to the start instead of trying to play the last 10 ms.
      if (seekFrac >= 0.999 || state.offset >= dur - 0.02) {
        seekFrac = 0;
        state.offset = 0;
        seek.value = '0';
      }
      const seekTime = Math.min(dur - 0.01, seekFrac * dur);
      startPlayer(seekTime);
    }
  });
  stopBtn.addEventListener('click', () => {
    stopPlayer();
    seek.value = '0';
    state.offset = 0;
    const samples = getSamples();
    if (samples) meta.textContent = `${(0).toFixed(2)} / ${(samples.length/44100).toFixed(2)} s`;
  });
  seek.addEventListener('input', () => {
    if (state.node) return;
    const samples = getSamples();
    if (!samples) return;
    state.offset = parseFloat(seek.value) * (samples.length / 44100);
    meta.textContent = `${state.offset.toFixed(2)} / ${(samples.length/44100).toFixed(2)} s`;
  });

  return {
    refresh: () => {
      const samples = getSamples();
      if (samples) {
        seek.disabled = false;
        playBtn.disabled = false;
        stopBtn.disabled = false;
        meta.textContent = `${(0).toFixed(2)} / ${(samples.length/44100).toFixed(2)} s`;
      } else {
        seek.disabled = true;
        playBtn.disabled = true;
        stopBtn.disabled = true;
        meta.textContent = '(no signal loaded)';
      }
      state.offset = 0;
      seek.value = '0';
      stopPlayer();
    },
  };
}

// ---------- viridis colormap (interpolated 11-stop table) ----------
const VIRIDIS = [
  [68,   1,  84], [71,  44, 122], [59,  81, 139], [44, 113, 142],
  [33, 144, 141], [39, 173, 129], [92, 200, 100], [170, 220, 50],
  [253, 231, 36],
];
function viridis(t, out) {
  if (!(t > 0)) t = 0;
  else if (t > 1) t = 1;
  const idx = t * (VIRIDIS.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(VIRIDIS.length - 1, lo + 1);
  const f = idx - lo;
  const a = VIRIDIS[lo], b = VIRIDIS[hi];
  out[0] = a[0] + (b[0] - a[0]) * f;
  out[1] = a[1] + (b[1] - a[1]) * f;
  out[2] = a[2] + (b[2] - a[2]) * f;
}

// ---------- spectrogram ----------
const SPEC_NFFT = 1024;
const SPEC_HOP = 256;
const SPEC_FS = 44100;
const SPEC_F_MIN = 20;
const SPEC_F_MAX = SPEC_FS / 2;
const SPEC_DB_MIN = -90;
const SPEC_DB_MAX = 0;

let specFFT = null;
let specHann = null;
let specComplexIn = null;
let specOut = null;

function ensureSpecHelpers() {
  if (specFFT) return;
  specFFT = new FFT(SPEC_NFFT);
  specHann = new Float64Array(SPEC_NFFT);
  for (let i = 0; i < SPEC_NFFT; i++) {
    specHann[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (SPEC_NFFT - 1)));
  }
  specComplexIn = new Float64Array(SPEC_NFFT);
  specOut = specFFT.createComplexArray();
}

function renderSpectrogram(canvas, samples) {
  ensureSpecHelpers();
  const ctx = canvas.getContext('2d');
  // Set internal pixel size: width matches frame count, height fixed.
  const pxHeight = 220;
  const numFrames = Math.max(1, Math.floor((samples.length - SPEC_NFFT) / SPEC_HOP) + 1);
  const pxWidth = Math.min(2400, Math.max(numFrames, 100));
  canvas.width = pxWidth;
  canvas.height = pxHeight;

  if (samples.length < SPEC_NFFT) {
    ctx.fillStyle = '#1c1d2b';
    ctx.fillRect(0, 0, pxWidth, pxHeight);
    return;
  }

  // Precompute pixel-row -> bin interpolation table (log frequency axis).
  const binsHalf = SPEC_NFFT / 2; // up to Nyquist
  const rowToBin = new Float32Array(pxHeight);
  const logMin = Math.log(SPEC_F_MIN);
  const logMax = Math.log(SPEC_F_MAX);
  for (let py = 0; py < pxHeight; py++) {
    // py = 0 is TOP = high freq; py = pxHeight-1 = bottom = low freq
    const tFreq = (pxHeight - 1 - py) / (pxHeight - 1);
    const freq = Math.exp(logMin + (logMax - logMin) * tFreq);
    rowToBin[py] = Math.min(binsHalf, freq * SPEC_NFFT / SPEC_FS);
  }

  // Pass 1: compute all frame magnitudes and the global peak (so quiet
  // sections look quiet, not normalized up to peak brightness).
  const binsPlus1 = binsHalf + 1;
  const frameMags = new Float32Array(pxWidth * binsPlus1);
  let globalMax = 1e-12;
  for (let col = 0; col < pxWidth; col++) {
    const frame = Math.floor(col * Math.max(0, numFrames - 1) / Math.max(1, pxWidth - 1));
    const start = frame * SPEC_HOP;
    for (let i = 0; i < SPEC_NFFT; i++) {
      const k = start + i;
      specComplexIn[i] = (k < samples.length ? samples[k] : 0) * specHann[i];
    }
    specFFT.realTransform(specOut, specComplexIn);
    const base = col * binsPlus1;
    for (let k = 0; k < binsPlus1; k++) {
      const re = specOut[2 * k];
      const im = specOut[2 * k + 1];
      const m = Math.sqrt(re * re + im * im);
      frameMags[base + k] = m;
      if (m > globalMax) globalMax = m;
    }
  }

  // Pass 2: render with global normalization.
  const img = ctx.createImageData(pxWidth, pxHeight);
  const data = img.data;
  const colBuf = [0, 0, 0];
  const invRef = 1.0 / globalMax;
  for (let col = 0; col < pxWidth; col++) {
    const base = col * binsPlus1;
    for (let py = 0; py < pxHeight; py++) {
      const fbin = rowToBin[py];
      const lo = Math.floor(fbin);
      const hi = Math.min(binsHalf, lo + 1);
      const ff = fbin - lo;
      const m = frameMags[base + lo] * (1 - ff) + frameMags[base + hi] * ff;
      const db = 20 * Math.log10(m * invRef + 1e-12);
      const t = (db - SPEC_DB_MIN) / (SPEC_DB_MAX - SPEC_DB_MIN);
      viridis(t, colBuf);
      const off = (py * pxWidth + col) * 4;
      data[off]     = colBuf[0];
      data[off + 1] = colBuf[1];
      data[off + 2] = colBuf[2];
      data[off + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  // Frequency gridlines and labels
  const wrap = canvas.parentElement;
  Array.from(wrap.querySelectorAll('.axis-label')).forEach(el => el.remove());
  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.lineWidth = 1;
  for (const f of [100, 1000, 10000]) {
    const tFreq = (Math.log(f) - logMin) / (logMax - logMin);
    const py = Math.round((1 - tFreq) * (pxHeight - 1));
    ctx.beginPath();
    ctx.moveTo(0, py); ctx.lineTo(pxWidth, py); ctx.stroke();
    const label = document.createElement('div');
    label.className = 'axis-label';
    label.style.right = '4px';
    label.style.top = ((py / pxHeight) * 100) + '%';
    label.style.transform = 'translateY(-50%)';
    label.textContent = f >= 1000 ? (f / 1000) + ' kHz' : f + ' Hz';
    wrap.appendChild(label);
  }
  // Time labels at start/end
  const dur = samples.length / SPEC_FS;
  const t0 = document.createElement('div');
  t0.className = 'axis-label';
  t0.style.left = '4px'; t0.style.bottom = '4px';
  t0.textContent = '0 s';
  wrap.appendChild(t0);
  const t1 = document.createElement('div');
  t1.className = 'axis-label';
  t1.style.right = '4px'; t1.style.bottom = '4px';
  t1.textContent = dur.toFixed(2) + ' s';
  wrap.appendChild(t1);
}

// ---------- input spectrum + power-law extension plot ----------
let inputSpectrum = null; // { freqs: Float64Array, mags: Float64Array }
const SPECPLOT_NFFT = 4096;
let _specPlotFFT = null;
let _specPlotHann = null;
let _specPlotBuf = null;
let _specPlotOut = null;

function ensureSpecPlotHelpers() {
  if (_specPlotFFT) return;
  _specPlotFFT = new FFT(SPECPLOT_NFFT);
  _specPlotHann = new Float64Array(SPECPLOT_NFFT);
  for (let i = 0; i < SPECPLOT_NFFT; i++) {
    _specPlotHann[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (SPECPLOT_NFFT - 1)));
  }
  _specPlotBuf = new Float64Array(SPECPLOT_NFFT);
  _specPlotOut = _specPlotFFT.createComplexArray();
}

function computeInputSpectrum(samples) {
  ensureSpecPlotHelpers();
  const N = SPECPLOT_NFFT;
  const hop = N / 2;
  const halfBins = N / 2 + 1;
  const accum = new Float64Array(halfBins);
  let count = 0;
  if (samples.length >= N) {
    for (let start = 0; start + N <= samples.length; start += hop) {
      for (let i = 0; i < N; i++) _specPlotBuf[i] = samples[start + i] * _specPlotHann[i];
      _specPlotFFT.realTransform(_specPlotOut, _specPlotBuf);
      for (let k = 0; k < halfBins; k++) {
        const re = _specPlotOut[2 * k], im = _specPlotOut[2 * k + 1];
        accum[k] += re * re + im * im;
      }
      count++;
    }
  } else {
    _specPlotBuf.fill(0);
    for (let i = 0; i < samples.length; i++) _specPlotBuf[i] = samples[i] * _specPlotHann[i];
    _specPlotFFT.realTransform(_specPlotOut, _specPlotBuf);
    for (let k = 0; k < halfBins; k++) {
      const re = _specPlotOut[2 * k], im = _specPlotOut[2 * k + 1];
      accum[k] = re * re + im * im;
    }
    count = 1;
  }
  const fs = 44100;
  const freqs = new Float64Array(halfBins);
  const mags = new Float64Array(halfBins);
  for (let k = 0; k < halfBins; k++) {
    freqs[k] = k * fs / N;
    mags[k] = Math.sqrt(accum[k] / count);
  }
  inputSpectrum = { freqs, mags };
}

function renderSpectrumPlot() {
  const canvas = document.getElementById('spectrum-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width = Math.max(50, Math.floor(canvas.clientWidth * dpr));
  const H = canvas.height = Math.max(50, Math.floor(canvas.clientHeight * dpr));
  ctx.fillStyle = '#1c1d2b';
  ctx.fillRect(0, 0, W, H);

  const params = readParams();
  const fc = params.f_cutoff;
  const alpha = params.alpha;
  const halfBlend = params.blend_half_width;
  const blendStart = Math.max(1, fc - halfBlend);
  const blendEnd = fc + halfBlend;
  const fitCenter = fc;
  const sigma = params.fit_width / 3.0;

  const fMin = 10, fMax = 22050;
  const dbMin = -90, dbMax = 10;
  const padL = 50 * dpr, padR = 12 * dpr, padT = 10 * dpr, padB = 28 * dpr;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const lf = (f) => padL + (Math.log10(Math.max(f, fMin)) - Math.log10(fMin))
    / (Math.log10(fMax) - Math.log10(fMin)) * plotW;
  const ldb = (db) => padT + (1 - (Math.max(dbMin, Math.min(dbMax, db)) - dbMin)
    / (dbMax - dbMin)) * plotH;

  // Grid + axis labels
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.font = (10 * dpr) + 'px ui-monospace, Menlo, monospace';
  for (const f of [10, 100, 1000, 10000]) {
    const x = lf(f);
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
    const lbl = f >= 1000 ? (f / 1000) + ' kHz' : f + ' Hz';
    ctx.fillText(lbl, x + 3 * dpr, padT + plotH + 12 * dpr);
  }
  for (const db of [-80, -60, -40, -20, 0]) {
    const y = ldb(db);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
    ctx.fillText(db + ' dB', 4 * dpr, y + 4 * dpr);
  }

  // Vertical bands (drawn under the curves)
  ctx.fillStyle = 'rgba(251, 191, 36, 0.18)';  // blend region (yellow)
  ctx.fillRect(lf(blendStart), padT, lf(blendEnd) - lf(blendStart), plotH);
  ctx.fillStyle = 'rgba(34, 197, 94, 0.20)';   // gaussian fit region (green)
  const fitLeft = Math.max(fMin, fitCenter - params.fit_width / 2);
  const fitRight = fitCenter + params.fit_width / 2;
  ctx.fillRect(lf(fitLeft), padT, Math.max(2 * dpr, lf(fitRight) - lf(fitLeft)), plotH);

  // Vertical line at fc
  ctx.strokeStyle = '#fbbf24';
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath(); ctx.moveTo(lf(fc), padT); ctx.lineTo(lf(fc), padT + plotH); ctx.stroke();

  if (!inputSpectrum) {
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.fillText('(load a signal to see its spectrum)', padL + 8 * dpr, padT + 18 * dpr);
    return;
  }

  // Beta-for-display: match input vs power-law on the Gaussian fit region.
  const exponent = -alpha / 2;
  let inputE = 0, powerE = 0;
  for (let k = 1; k < inputSpectrum.freqs.length; k++) {
    const f = inputSpectrum.freqs[k];
    const w = Math.exp(-((f - fitCenter) * (f - fitCenter)) / (2 * sigma * sigma));
    if (w < 1e-6) continue;
    const inMag = inputSpectrum.mags[k];
    const pMag = Math.pow(f, exponent);
    inputE += inMag * inMag * w;
    powerE += pMag * pMag * w;
  }
  const betaVis = powerE > 0 ? Math.sqrt(inputE / powerE) : 1.0;

  // Reference for dB normalization: peak of input PSD.
  let peakIn = 1e-12;
  for (let k = 0; k < inputSpectrum.mags.length; k++) {
    if (inputSpectrum.mags[k] > peakIn) peakIn = inputSpectrum.mags[k];
  }
  const inDb = (m) => 20 * Math.log10(Math.max(m, 1e-12) / peakIn);

  // Blend weights (for the green "predicted blended target" curve).
  const blendWeight1 = (f) => {
    if (f <= blendStart) return 1;
    if (f >= blendEnd) return 0;
    return 1 - (f - blendStart) / (blendEnd - blendStart);
  };
  const blendWeight2 = (f) => 1 - blendWeight1(f);

  // (1) Input PSD line.
  ctx.strokeStyle = '#60a5fa';
  ctx.lineWidth = 1.6 * dpr;
  ctx.beginPath();
  let drawn = false;
  for (let k = 1; k < inputSpectrum.freqs.length; k++) {
    const f = inputSpectrum.freqs[k];
    if (f < fMin || f > fMax) continue;
    const x = lf(f), y = ldb(inDb(inputSpectrum.mags[k]));
    if (!drawn) { ctx.moveTo(x, y); drawn = true; }
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // (2) Power-law extension β·f^(-α/2) (dashed, log-spaced).
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 1.6 * dpr;
  ctx.setLineDash([6 * dpr, 4 * dpr]);
  ctx.beginPath();
  drawn = false;
  for (let f = fMin; f <= fMax; f *= 1.02) {
    const pMag = betaVis * Math.pow(f, exponent);
    const x = lf(f), y = ldb(inDb(pMag));
    if (!drawn) { ctx.moveTo(x, y); drawn = true; }
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // (3) Blended target = input·blend1 + β·powerlaw·blend2 (magnitude-summed
  // approximation, ignoring complex phase). Solid green.
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 1.4 * dpr;
  ctx.beginPath();
  drawn = false;
  for (let k = 1; k < inputSpectrum.freqs.length; k++) {
    const f = inputSpectrum.freqs[k];
    if (f < fMin || f > fMax) continue;
    const w1 = blendWeight1(f), w2 = blendWeight2(f);
    const inMag = inputSpectrum.mags[k] * w1;
    const pMag = betaVis * Math.pow(f, exponent) * w2;
    // RMS-like sum (uncorrelated assumption)
    const total = Math.sqrt(inMag * inMag + pMag * pMag);
    const x = lf(f), y = ldb(inDb(total));
    if (!drawn) { ctx.moveTo(x, y); drawn = true; }
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Axis title + reading-out for fc
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  ctx.font = (11 * dpr) + 'px sans-serif';
  ctx.fillText('Frequency', padL + plotW / 2 - 30 * dpr, H - 6 * dpr);
  ctx.fillStyle = '#fbbf24';
  ctx.fillText('f' + 'c=' + fc.toFixed(0) + ' Hz', lf(fc) + 4 * dpr, padT + 12 * dpr);
}

// ---------- beta trace ----------
function renderBetaTrace(canvas, betas, signalDuration) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = canvas.clientHeight * (window.devicePixelRatio || 1);
  ctx.fillStyle = '#1c1d2b';
  ctx.fillRect(0, 0, W, H);
  if (!betas || betas.length === 0) return;

  // Log-y axis from min to max
  let logMin = Infinity, logMax = -Infinity;
  for (let i = 0; i < betas.length; i++) {
    const b = betas[i];
    if (b > 0) {
      const lb = Math.log10(b);
      if (lb < logMin) logMin = lb;
      if (lb > logMax) logMax = lb;
    }
  }
  if (!isFinite(logMin) || !isFinite(logMax)) return;
  if (logMax === logMin) { logMax = logMin + 1; }

  const padTop = 4, padBot = 4;
  const yFor = (b) => {
    if (!(b > 0)) return H - padBot;
    const lb = Math.log10(b);
    const t = (lb - logMin) / (logMax - logMin);
    return padTop + (1 - t) * (H - padTop - padBot);
  };

  ctx.strokeStyle = '#fde047';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < betas.length; i++) {
    const x = (betas.length === 1 ? W / 2 : (i / (betas.length - 1)) * W);
    const y = yFor(betas[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // y-axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.font = (10 * (window.devicePixelRatio || 1)) + 'px ui-monospace, Menlo, monospace';
  ctx.fillText('beta=' + Math.pow(10, logMax).toExponential(1), 4, padTop + 10);
  ctx.fillText('beta=' + Math.pow(10, logMin).toExponential(1), 4, H - padBot - 2);
}

// ---------- worker plumbing ----------
// Build the Worker from a Blob URL containing freshly-fetched copies of
// fft.js + deterministic_rng.js + bandwidth_extension.js + an inline message
// handler. This sidesteps any browser/script caching of the dep files
// (which has caused "Script error." failures in the past) and avoids the
// separate worker.js fetch entirely.
const WORKER_HANDLER = `
self.addEventListener('message', (event) => {
  const { id, input, params } = event.data;
  try {
    const opts = Object.assign({}, params);
    if (opts.seed !== undefined && opts.seed !== null) {
      opts.rng = new DeterministicRng(opts.seed);
      delete opts.seed;
    }
    opts.onProgress = (value) => self.postMessage({ id, type: 'progress', value });
    const out = BandwidthExtension.extendSignal(new Float64Array(input), opts);
    const extendedF32 = Float32Array.from(out.extended);
    const betas = Float32Array.from(out.betas);
    self.postMessage(
      { id, type: 'done', extended: extendedF32, betas,
        nfftPerWindow: out.nfftPerWindow, nfftFull: out.nfftFull },
      [extendedF32.buffer, betas.buffer]
    );
  } catch (err) {
    self.postMessage({ id, type: 'error', message: (err && err.message) || String(err) });
  }
});
`;

let workerInitPromise = null;

async function ensureWorker() {
  if (worker) return worker;
  if (workerInitPromise) return workerInitPromise;
  workerInitPromise = (async () => {
    const fetchFresh = async (path) => {
      const url = new URL(path, location.href).href;
      try {
        const r = await fetch(url, { cache: 'no-store' });
        if (!r.ok) throw new Error(url + ' HTTP ' + r.status);
        return r.text();
      } catch (err) {
        const proto = location.protocol;
        const hint = proto === 'file:'
          ? ' (you appear to be loading the page over file://; browsers block fetch() under file://. Start a local HTTP server with `python serve.py 8765` from this directory, then visit http://127.0.0.1:8765/)'
          : '';
        throw new Error('failed to fetch ' + url + ': ' + (err.message || err) + hint);
      }
    };
    const [fftSrc, rngSrc, bweSrc] = await Promise.all([
      fetchFresh('fft.js?v=15'),
      fetchFresh('deterministic_rng.js?v=15'),
      fetchFresh('bandwidth_extension.js?v=15'),
    ]);
    const blob = new Blob(
      [fftSrc, '\n;\n', rngSrc, '\n;\n', bweSrc, '\n;\n', WORKER_HANDLER],
      { type: 'application/javascript' }
    );
    const w = new Worker(URL.createObjectURL(blob));
    w.addEventListener('message', onWorkerMessage);
    w.addEventListener('error', onWorkerError);
    worker = w;
    return w;
  })();
  return workerInitPromise;
}

function onWorkerMessage(e) {
  const msg = e.data;
  const req = pendingRequest;
  if (!req || msg.id !== req.id) return;
  if (msg.type === 'progress') {
    document.getElementById('progress-bar').style.width = (100 * msg.value) + '%';
    document.getElementById('process-status').textContent =
      'processing … ' + Math.round(100 * msg.value) + '%';
  } else if (msg.type === 'done') {
    extendedBuffer = msg.extended;
    extendedBetas = msg.betas;
    document.getElementById('progress-bar').style.width = '100%';
    document.getElementById('process-status').textContent =
      `done in ${(performance.now() - req.startedAt).toFixed(0)} ms ` +
      `(NFFT_win=${msg.nfftPerWindow}, NFFT_full=${msg.nfftFull}, betas=${msg.betas.length})`;
    document.getElementById('process-status').classList.remove('warn');
    document.getElementById('process-btn').disabled = false;
    pendingRequest = null;
    refreshExtended();
  } else if (msg.type === 'error') {
    document.getElementById('process-status').textContent = 'error: ' + msg.message;
    document.getElementById('process-status').classList.add('warn');
    document.getElementById('process-btn').disabled = false;
    pendingRequest = null;
  }
}

function onWorkerError(e) {
  const msg = (e && (e.message || (e.error && e.error.message))) || 'unknown';
  document.getElementById('process-status').textContent = 'worker error: ' + msg;
  document.getElementById('process-status').classList.add('warn');
  document.getElementById('process-btn').disabled = false;
  pendingRequest = null;
  showError('Worker reported an error: ' + msg);
}

async function startProcess() {
  if (!originalBuffer) return;
  document.getElementById('process-btn').disabled = true;
  document.getElementById('progress-bar').style.width = '0%';
  document.getElementById('process-status').textContent = 'processing … 0%';
  document.getElementById('process-status').classList.remove('warn');

  let w;
  try {
    w = await ensureWorker();
  } catch (err) {
    document.getElementById('process-status').textContent = 'worker setup failed: ' + err;
    document.getElementById('process-status').classList.add('warn');
    document.getElementById('process-btn').disabled = false;
    showError('Worker setup failed: ' + err);
    return;
  }

  const params = readParams();
  const nfftMode = document.getElementById('faithful-fft').checked ? null : 'auto';
  const id = ++nextRequestId;
  const inputCopy = new Float64Array(originalBuffer); // copy so worker takes ownership
  pendingRequest = { id, startedAt: performance.now() };
  w.postMessage({
    id, input: inputCopy,
    params: {
      fs: 44100.0,
      alpha: params.alpha,
      halfWidth: params.half_width | 0,
      fCutoff: params.f_cutoff,
      fitCenter: params.f_cutoff,
      fitWidth: params.fit_width,
      blendStart: params.f_cutoff - params.blend_half_width,
      blendEnd: params.f_cutoff + params.blend_half_width,
      noiseAmplitude: params.noise_amplitude,
      seed: params.seed | 0,
      nfftPerWindow: nfftMode,
    },
  }, [inputCopy.buffer]);
}

// ---------- controls ----------
const sliderSpecs = {
  alpha:        { min: 1.5, max: 4.0, step: 0.01, scale: 'lin', fmt: v => v.toFixed(2) },
  f_cutoff:     { min: 60,  max: 500, step: 0.01, scale: 'log', fmt: v => v.toFixed(1) + ' Hz' },
  half_width:   { min: 64,  max: 2048, step: 1, scale: 'log',  fmt: v => Math.round(v) + ' smp' },
  fit_width:    { min: 5,   max: 50,  step: 0.1, scale: 'lin', fmt: v => v.toFixed(1) + ' Hz' },
  blend_half_width: { min: 1, max: 50, step: 0.1, scale: 'lin', fmt: v => v.toFixed(1) + ' Hz' },
  noise_amplitude:  { min: 0, max: 4,  step: 0.01, scale: 'lin', fmt: v => v.toFixed(2) },
  seed:         { min: 0, max: 999999, step: 1, scale: 'lin', fmt: v => Math.round(v).toString() },
};

function setupSliders() {
  for (const [key, spec] of Object.entries(sliderSpecs)) {
    const el = document.getElementById('slider-' + key);
    el.min = spec.scale === 'log' ? Math.log(spec.min) : spec.min;
    el.max = spec.scale === 'log' ? Math.log(spec.max) : spec.max;
    el.step = spec.scale === 'log' ? (Math.log(spec.max) - Math.log(spec.min)) / 1000 : spec.step;
    el.value = spec.scale === 'log' ? Math.log(DEFAULTS[key]) : DEFAULTS[key];
    const valEl = document.getElementById('val-' + key);
    const update = () => {
      const raw = parseFloat(el.value);
      const v = spec.scale === 'log' ? Math.exp(raw) : raw;
      valEl.textContent = spec.fmt(v);
      // Live spectrum plot reacts to alpha / fc / fit_width / blend_half_width.
      if (key === 'alpha' || key === 'f_cutoff' ||
          key === 'fit_width' || key === 'blend_half_width') {
        safeRenderSpectrumPlot();
      }
      scheduleAutoProcess();
    };
    update();
    el.addEventListener('input', update);
  }
  document.getElementById('reset-btn').addEventListener('click', () => {
    for (const [key, spec] of Object.entries(sliderSpecs)) {
      const el = document.getElementById('slider-' + key);
      el.value = spec.scale === 'log' ? Math.log(DEFAULTS[key]) : DEFAULTS[key];
      el.dispatchEvent(new Event('input'));
    }
    safeRenderSpectrumPlot();
  });
  // Re-draw the spectrum plot on window resize so the canvas keeps a sensible
  // aspect ratio.
  window.addEventListener('resize', () => safeRenderSpectrumPlot());
}

let _autoProcessTimer = null;
function scheduleAutoProcess() {
  const cb = document.getElementById('auto-process');
  if (!cb || !cb.checked) return;
  if (_autoProcessTimer) clearTimeout(_autoProcessTimer);
  _autoProcessTimer = setTimeout(() => {
    _autoProcessTimer = null;
    startProcess();
  }, 250);
}

function readParams() {
  const out = {};
  for (const [key, spec] of Object.entries(sliderSpecs)) {
    const raw = parseFloat(document.getElementById('slider-' + key).value);
    out[key] = spec.scale === 'log' ? Math.exp(raw) : raw;
  }
  return out;
}

// ---------- error / banner display ----------
function showBanner(html, isError) {
  const el = document.getElementById('banner');
  if (!el) return;
  el.hidden = false;
  el.classList.toggle('error', !!isError);
  el.innerHTML = html;
}

function showError(msg) {
  console.error(msg);
  showBanner(String(msg), true);
}

window.addEventListener('error', (e) => {
  showError('JS error: ' + (e.message || e.error) +
    '. Try a hard refresh (Ctrl+Shift+R) to clear cached scripts.');
});
window.addEventListener('unhandledrejection', (e) => {
  showError('promise rejection: ' + (e.reason && e.reason.message || e.reason));
});

// ---------- example loading (manifest + built-in synthetic) ----------
function prettyName(name) {
  return name.split('_').map(w => w[0].toUpperCase() + w.slice(1)).join(' ');
}

function makeSyntheticSignal() {
  // ~1.5 s decaying tone burst with envelope variation. Below the cutoff so
  // the bandwidth-extension path produces audible high-frequency content.
  const fs = 44100;
  const dur = 1.5;
  const N = Math.floor(fs * dur);
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const t = i / fs;
    const env = Math.exp(-2.5 * t) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 4.0 * t));
    out[i] = (
      0.55 * Math.sin(2 * Math.PI * 80.0 * t) +
      0.30 * Math.sin(2 * Math.PI * 130.0 * t) +
      0.15 * Math.sin(2 * Math.PI * 170.0 * t)
    ) * env;
  }
  return out;
}

async function loadManifest() {
  try {
    const resp = await fetch('assets/manifest.json');
    if (!resp.ok) throw new Error('manifest.json HTTP ' + resp.status);
    const m = await resp.json();
    manifest = m;
    const row = document.getElementById('example-row');
    const labelEl = row.querySelector('label[for="wav-input"]');
    for (const ex of m.examples) {
      const btn = document.createElement('button');
      btn.textContent = prettyName(ex.name);
      btn.dataset.name = ex.name;
      btn.addEventListener('click', () => loadExample(ex.name));
      row.insertBefore(btn, labelEl);
    }
    if (m.examples.length) await loadExample(m.examples[0].name);
  } catch (err) {
    showBanner(
      'Could not load the bundled flame examples (' + err + '). ' +
      'You can still use the <strong>Synthetic burst</strong> preset, load your own WAV, or run from a local HTTP server (e.g. <code>python -m http.server 8765</code>).',
      false
    );
    // Fallback: load the built-in synthetic so the page is interactive.
    loadSynthetic();
  }
}

async function loadExample(name) {
  if (!manifest) return loadSynthetic();
  const ex = manifest.examples.find(e => e.name === name);
  if (!ex) return;
  try {
    const buf = await (await fetch('assets/' + ex.file)).arrayBuffer();
    const samples = new Float32Array(buf);
    setOriginal(samples, name);
    markActive(name);
    startProcess();
  } catch (err) {
    showError('Failed to fetch assets/' + ex.file + ': ' + err);
  }
}

function loadSynthetic() {
  const samples = makeSyntheticSignal();
  setOriginal(samples, 'synthetic');
  markActive('__synthetic__');
  startProcess();
}

function markActive(name) {
  document.querySelectorAll('#example-row button').forEach(b => {
    const key = b.dataset.builtin === 'synthetic' ? '__synthetic__' : b.dataset.name;
    b.classList.toggle('active', key === name || (name === 'synthetic' && key === '__synthetic__'));
  });
}

async function loadWav(file) {
  try {
    ensureAudio();
    const buf = await file.arrayBuffer();
    const audio = await audioCtx.decodeAudioData(buf);
    const ch = audio.getChannelData(0);
    let samples;
    if (audio.sampleRate === 44100) {
      samples = new Float32Array(ch);
    } else {
      const ratio = audio.sampleRate / 44100;
      const newLen = Math.floor(ch.length / ratio);
      samples = new Float32Array(newLen);
      for (let i = 0; i < newLen; i++) {
        const src = i * ratio;
        const lo = Math.floor(src);
        const hi = Math.min(ch.length - 1, lo + 1);
        const f = src - lo;
        samples[i] = ch[lo] * (1 - f) + ch[hi] * f;
      }
    }
    setOriginal(samples, file.name.replace(/\.[^.]+$/, ''));
    markActive(null);
    startProcess();
  } catch (err) {
    showError('Could not decode WAV: ' + err);
  }
}

function setOriginal(samples, name) {
  originalBuffer = samples;
  originalDuration = samples.length / 44100;
  currentName = name;
  extendedBuffer = null;
  extendedBetas = null;
  document.getElementById('current-name').textContent =
    `${prettyName(name)}  ·  ${samples.length} samples  ·  ${originalDuration.toFixed(2)} s`;
  try { computeInputSpectrum(samples); } catch (err) { console.warn('input PSD failed:', err); inputSpectrum = null; }
  safeRenderSpectrumPlot();
  safeRenderSpectrogram('spec-input', samples);
  safeRenderSpectrogram('spec-extended', new Float32Array(samples.length));
  safeRenderBetaTrace(new Float32Array(0));
  origPlayer.refresh();
  extPlayer.refresh();
}

function safeRenderSpectrumPlot() {
  try { renderSpectrumPlot(); } catch (err) { console.warn('spectrum plot failed:', err); }
}

function refreshExtended() {
  if (extendedBuffer) {
    safeRenderSpectrogram('spec-extended', extendedBuffer);
    safeRenderBetaTrace(extendedBetas);
  }
  extPlayer.refresh();
}

// Spectrogram and beta-trace failures should never kill the page.
function safeRenderSpectrogram(canvasId, samples) {
  try { renderSpectrogram(document.getElementById(canvasId), samples); }
  catch (err) { console.warn('spectrogram failed:', err); }
}
function safeRenderBetaTrace(betas) {
  try { renderBetaTrace(document.getElementById('beta-canvas'), betas, originalDuration); }
  catch (err) { console.warn('beta trace failed:', err); }
}

// ---------- WAV download ----------
function encodeWav(samples, sampleRate) {
  let peak = 1e-12;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    if (a > peak) peak = a;
  }
  const norm = 1 / (peak * 1.01);
  const bytes = 44 + samples.length * 2;
  const buf = new ArrayBuffer(bytes);
  const dv = new DataView(buf);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  dv.setUint32(4, bytes - 8, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  dv.setUint32(16, 16, true);          // PCM chunk size
  dv.setUint16(20, 1, true);           // PCM format
  dv.setUint16(22, 1, true);           // 1 channel
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * 2, true); // byte rate
  dv.setUint16(32, 2, true);           // block align
  dv.setUint16(34, 16, true);          // bits per sample
  writeStr(36, 'data');
  dv.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    let s = samples[i] * norm;
    if (s > 1) s = 1; else if (s < -1) s = -1;
    dv.setInt16(44 + 2 * i, Math.round(s * 32767), true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

function downloadWav(samples, baseName) {
  if (!samples || !samples.length) return;
  const blob = encodeWav(samples, 44100);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = baseName + '.wav';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
}

// ---------- player init (deferred until DOMContentLoaded) ----------
let origPlayer = null;
let extPlayer = null;

function wireDownloadButtons() {
  document.querySelector('#player-original .download').addEventListener('click', () => {
    downloadWav(originalBuffer, (currentName || 'original') + '_original');
  });
  document.querySelector('#player-extended .download').addEventListener('click', () => {
    if (!extendedBuffer) {
      showBanner('Click <strong>Process</strong> first to generate the extended signal.', false);
      return;
    }
    const params = readParams();
    const tag = `_alpha${params.alpha.toFixed(2)}_seed${params.seed | 0}`;
    downloadWav(extendedBuffer, (currentName || 'signal') + '_extended' + tag);
  });
}

// ---------- bootstrap ----------
function bootstrap() {
  setupSliders();
  origPlayer = makePlayer('original', () => originalBuffer);
  extPlayer = makePlayer('extended', () => extendedBuffer);
  wireDownloadButtons();
  origPlayer.refresh();
  extPlayer.refresh();

  document.getElementById('process-btn').addEventListener('click', startProcess);
  document.getElementById('faithful-fft').addEventListener('change', () => {});
  document.getElementById('wav-input').addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (f) loadWav(f);
  });
  document.querySelector('#example-row button[data-builtin="synthetic"]')
    .addEventListener('click', () => loadSynthetic());

  // file:// protocol detection. Workers and fetch generally don't work
  // when the page is opened directly from disk (no http(s) origin), so
  // make the situation extremely obvious and disable Process.
  if (location.protocol === 'file:') {
    showBanner(
      '<strong>This page is loaded from <code>file://</code>.</strong> ' +
      'Browsers block <code>fetch()</code> and Web Workers under file://, ' +
      'so the bandwidth extension can&rsquo;t run. Open a terminal in ' +
      '<code>fire-bandwidth-extension/</code> and run ' +
      '<code>python serve.py 8765</code>, then visit ' +
      '<a href="http://127.0.0.1:8765/" style="color:inherit;text-decoration:underline;">' +
      'http://127.0.0.1:8765/</a>. (You can still load and play the synthetic ' +
      'preset to confirm audio works, but Process and the bundled examples won&rsquo;t.)',
      true
    );
    document.getElementById('process-btn').disabled = true;
    document.getElementById('process-status').textContent = 'disabled (file:// protocol)';
    document.getElementById('process-status').classList.add('warn');
    // Still load the synthetic so playback at least works.
    loadSynthetic();
    return;
  }

  // Always start with a usable signal: synthesize first, then try to swap to
  // the bundled example if the manifest loads.
  loadSynthetic();
  loadManifest();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrap);
} else {
  bootstrap();
}
