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

'use strict';

// ---------- defaults ----------
const DEFAULTS = {
  num_levels: 6,
  window_hw: 4,
  feature_hw: 3,
  falloff: 0.0,
  scaling_alpha: 1.0,
  seed: 42,
  eps_ann: 5.0,  // (1+eps)-approximate NN. The original C++ release defaults
                 // to 1.0; this demo uses 5.0 because for these signals the
                 // perceptual difference vs exact NN is below the noise floor
                 // (and exact NN actually produces tonal artefacts on a couple
                 // of the bundled examples; see the About section). 5.0 is
                 // ~5-10x faster than exact. Set to 0 for bit-exact parity
                 // with the Python golden.
};

// ---------- globals ----------
let audioCtx = null;
let manifest = null;
let currentName = null;
let baseBuffer = null;          // Float32Array, base signal
let trainingBuffer = null;      // Float32Array, training signal
let extendedBuffer = null;      // Float32Array, synthesized output
let outputPyramid = null;       // [Float32Array per level], bottom-most last
let trainingId = null;          // stable identifier for the current training signal (for worker cache)
let _uploadCounter = 0;         // bumped per WAV upload so caches invalidate
let worker = null;
let workerInitPromise = null;
let nextRequestId = 0;
let pendingRequest = null;

const playState = {
  original: { node: null, startedAt: 0, offset: 0, raf: null },
  training: { node: null, startedAt: 0, offset: 0, raf: null },
  extended: { node: null, startedAt: 0, offset: 0, raf: null },
};

let origPlayer = null;
let trainPlayer = null;
let extPlayer = null;

// ---------- error / banner ----------
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
  showError('JS error: ' + (e.message || e.error)
    + '. Try a hard refresh (Ctrl+Shift+R) to clear cached scripts.');
});
window.addEventListener('unhandledrejection', (e) => {
  showError('promise rejection: ' + (e.reason && e.reason.message || e.reason));
});

// ---------- audio ----------
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

function makePlayer(which, getSamples, defaultLabel) {
  const root = document.getElementById('player-' + which);
  const playBtn = root.querySelector('.play');
  const stopBtn = root.querySelector('.stop');
  const seek = root.querySelector('input[type="range"]');
  const meta = root.querySelector('.meta');
  const state = playState[which];

  function tick() {
    const samples = getSamples();
    if (!samples || !state.node) return;
    const elapsed = audioCtx.currentTime - state.startedAt + state.offset;
    const dur = samples.length / 44100;
    seek.value = String(Math.min(1, elapsed / dur));
    meta.textContent = `${elapsed.toFixed(2)} / ${dur.toFixed(2)} s`;
    if (elapsed < dur) state.raf = requestAnimationFrame(tick);
    else stopPlayer();
  }
  function startPlayer(fromOffset) {
    const samples = getSamples();
    if (!samples) return;
    stopPlayer();
    ensureAudio();
    const audioBuf = makeAudioBuffer(samples);
    const node = audioCtx.createBufferSource();
    node.buffer = audioBuf;
    node.connect(audioCtx.destination);
    state.offset = Math.min(audioBuf.duration - 0.01, Math.max(0, fromOffset));
    state.startedAt = audioCtx.currentTime;
    state.node = node;
    node.onended = () => { if (state.node === node) stopPlayer(); };
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
      // Race guard: if the clip already finished playing but neither
      // tick() nor onended has cleared state.node yet, treat the click
      // as "restart from the beginning" rather than as a pause-at-end.
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
    if (samples) meta.textContent = `0.00 / ${(samples.length/44100).toFixed(2)} s`;
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
        meta.textContent = `0.00 / ${(samples.length/44100).toFixed(2)} s`;
      } else {
        seek.disabled = true;
        playBtn.disabled = true;
        stopBtn.disabled = true;
        meta.textContent = defaultLabel || '(no signal loaded)';
      }
      state.offset = 0;
      seek.value = '0';
      stopPlayer();
    },
  };
}

// ---------- viridis colormap ----------
const VIRIDIS = [
  [68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142],
  [33, 144, 141], [39, 173, 129], [92, 200, 100], [170, 220, 50],
  [253, 231, 36],
];
function viridis(t, out) {
  if (!(t > 0)) t = 0; else if (t > 1) t = 1;
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
let specFFT = null, specHann = null, specBuf = null, specOut = null;

function ensureSpecHelpers() {
  if (specFFT) return;
  specFFT = new FFT(SPEC_NFFT);
  specHann = new Float64Array(SPEC_NFFT);
  for (let i = 0; i < SPEC_NFFT; i++) {
    specHann[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (SPEC_NFFT - 1)));
  }
  specBuf = new Float64Array(SPEC_NFFT);
  specOut = specFFT.createComplexArray();
}

function renderSpectrogram(canvas, samples) {
  ensureSpecHelpers();
  const ctx = canvas.getContext('2d');
  const pxHeight = 200;
  const numFrames = Math.max(1, Math.floor((samples.length - SPEC_NFFT) / SPEC_HOP) + 1);
  const pxWidth = Math.min(2400, Math.max(numFrames, 100));
  canvas.width = pxWidth;
  canvas.height = pxHeight;

  if (samples.length < SPEC_NFFT) {
    ctx.fillStyle = '#1c1d2b';
    ctx.fillRect(0, 0, pxWidth, pxHeight);
    return;
  }

  const binsHalf = SPEC_NFFT / 2;
  const rowToBin = new Float32Array(pxHeight);
  const logMin = Math.log(SPEC_F_MIN);
  const logMax = Math.log(SPEC_F_MAX);
  for (let py = 0; py < pxHeight; py++) {
    const tFreq = (pxHeight - 1 - py) / (pxHeight - 1);
    const freq = Math.exp(logMin + (logMax - logMin) * tFreq);
    rowToBin[py] = Math.min(binsHalf, freq * SPEC_NFFT / SPEC_FS);
  }

  const binsPlus1 = binsHalf + 1;
  const frameMags = new Float32Array(pxWidth * binsPlus1);
  let globalMax = 1e-12;
  for (let col = 0; col < pxWidth; col++) {
    const frame = Math.floor(col * Math.max(0, numFrames - 1) / Math.max(1, pxWidth - 1));
    const start = frame * SPEC_HOP;
    for (let i = 0; i < SPEC_NFFT; i++) {
      const k = start + i;
      specBuf[i] = (k < samples.length ? samples[k] : 0) * specHann[i];
    }
    specFFT.realTransform(specOut, specBuf);
    const base = col * binsPlus1;
    for (let k = 0; k < binsPlus1; k++) {
      const re = specOut[2 * k], im = specOut[2 * k + 1];
      const m = Math.sqrt(re * re + im * im);
      frameMags[base + k] = m;
      if (m > globalMax) globalMax = m;
    }
  }

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
      data[off] = colBuf[0];
      data[off + 1] = colBuf[1];
      data[off + 2] = colBuf[2];
      data[off + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  // Frequency gridlines
  const wrap = canvas.parentElement;
  Array.from(wrap.querySelectorAll('.axis-label')).forEach(el => el.remove());
  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  for (const f of [100, 1000, 10000]) {
    const tFreq = (Math.log(f) - logMin) / (logMax - logMin);
    const py = Math.round((1 - tFreq) * (pxHeight - 1));
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(pxWidth, py); ctx.stroke();
    const label = document.createElement('div');
    label.className = 'axis-label';
    label.style.right = '4px';
    label.style.top = ((py / pxHeight) * 100) + '%';
    label.style.transform = 'translateY(-50%)';
    label.textContent = f >= 1000 ? (f / 1000) + ' kHz' : f + ' Hz';
    wrap.appendChild(label);
  }
  const dur = samples.length / SPEC_FS;
  const t0 = document.createElement('div');
  t0.className = 'axis-label';
  t0.style.left = '4px'; t0.style.bottom = '4px'; t0.textContent = '0 s';
  wrap.appendChild(t0);
  const t1 = document.createElement('div');
  t1.className = 'axis-label';
  t1.style.right = '4px'; t1.style.bottom = '4px';
  t1.textContent = dur.toFixed(2) + ' s';
  wrap.appendChild(t1);
}

function safeRenderSpec(canvasId, samples) {
  try {
    if (samples && samples.length > 0) {
      renderSpectrogram(document.getElementById(canvasId), samples);
    } else {
      const c = document.getElementById(canvasId);
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#1c1d2b';
      ctx.fillRect(0, 0, c.width, c.height);
    }
  } catch (err) { console.warn('spectrogram failed:', err); }
}

// ---------- spectrum plot (3 PSD lines) ----------
const PSD_NFFT = 4096;
let psdFFT = null, psdHann = null, psdBuf = null, psdOut = null;
function ensurePsdHelpers() {
  if (psdFFT) return;
  psdFFT = new FFT(PSD_NFFT);
  psdHann = new Float64Array(PSD_NFFT);
  for (let i = 0; i < PSD_NFFT; i++) {
    psdHann[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (PSD_NFFT - 1)));
  }
  psdBuf = new Float64Array(PSD_NFFT);
  psdOut = psdFFT.createComplexArray();
}

function computePSD(samples) {
  ensurePsdHelpers();
  const N = PSD_NFFT;
  const hop = N / 2;
  const halfBins = N / 2 + 1;
  const accum = new Float64Array(halfBins);
  let count = 0;
  if (samples.length >= N) {
    for (let start = 0; start + N <= samples.length; start += hop) {
      for (let i = 0; i < N; i++) psdBuf[i] = samples[start + i] * psdHann[i];
      psdFFT.realTransform(psdOut, psdBuf);
      for (let k = 0; k < halfBins; k++) {
        const re = psdOut[2*k], im = psdOut[2*k+1];
        accum[k] += re*re + im*im;
      }
      count++;
    }
  } else {
    psdBuf.fill(0);
    for (let i = 0; i < samples.length; i++) psdBuf[i] = samples[i] * psdHann[i];
    psdFFT.realTransform(psdOut, psdBuf);
    for (let k = 0; k < halfBins; k++) {
      const re = psdOut[2*k], im = psdOut[2*k+1];
      accum[k] = re*re + im*im;
    }
    count = 1;
  }
  const freqs = new Float64Array(halfBins);
  const mags = new Float64Array(halfBins);
  for (let k = 0; k < halfBins; k++) {
    freqs[k] = k * 44100 / N;
    mags[k] = Math.sqrt(accum[k] / count);
  }
  return { freqs, mags };
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

  const fMin = 20, fMax = 22050;
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
  for (const f of [100, 1000, 10000]) {
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

  // Per-pyramid Nyquist gridlines
  if (outputPyramid && outputPyramid.length) {
    let fs = 44100;
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    for (let lvl = 0; lvl < outputPyramid.length; lvl++) {
      const nyq = fs / 2;
      const x = lf(nyq);
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
      fs /= 2;
    }
  }

  function plotLine(samples, color, dpr) {
    if (!samples || samples.length === 0) return;
    const psd = computePSD(samples);
    let peak = 1e-12;
    for (let k = 0; k < psd.mags.length; k++) {
      if (psd.mags[k] > peak) peak = psd.mags[k];
    }
    const inDb = (m) => 20 * Math.log10(Math.max(m, 1e-12) / peak);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6 * dpr;
    ctx.beginPath();
    let drawn = false;
    for (let k = 1; k < psd.freqs.length; k++) {
      const f = psd.freqs[k];
      if (f < fMin || f > fMax) continue;
      const x = lf(f), y = ldb(inDb(psd.mags[k]));
      if (!drawn) { ctx.moveTo(x, y); drawn = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Plot in order: training (background), base (mid), output (foreground)
  if (trainingBuffer) plotLine(trainingBuffer, '#f59e0b', dpr);
  if (baseBuffer) plotLine(baseBuffer, '#60a5fa', dpr);
  if (extendedBuffer) plotLine(extendedBuffer, '#22c55e', dpr);

  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  ctx.font = (11 * dpr) + 'px sans-serif';
  ctx.fillText('Frequency', padL + plotW / 2 - 30 * dpr, H - 6 * dpr);
}

function safeRenderSpectrumPlot() {
  try { renderSpectrumPlot(); } catch (err) { console.warn('spectrum plot failed:', err); }
}

// ---------- pyramid stack ----------
function renderPyramidStack(pyramid) {
  const container = document.getElementById('pyramid-rows');
  container.innerHTML = '';
  if (!pyramid || pyramid.length === 0) return;
  const dpr = window.devicePixelRatio || 1;
  // Render coarsest level at top.
  for (let i = pyramid.length - 1; i >= 0; i--) {
    const lvl = pyramid[i];
    const row = document.createElement('div');
    row.className = 'level-row';
    const lbl = document.createElement('div');
    lbl.className = 'level-label';
    lbl.textContent = 'L' + i + ' (' + lvl.length + ')';
    const canvas = document.createElement('canvas');
    row.appendChild(lbl);
    row.appendChild(canvas);
    container.appendChild(row);
    // Defer rendering until DOM has laid out (so clientWidth is right).
    requestAnimationFrame(() => {
      const W = canvas.width = Math.max(50, Math.floor(canvas.clientWidth * dpr));
      const H = canvas.height = Math.max(20, Math.floor(canvas.clientHeight * dpr));
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#1c1d2b';
      ctx.fillRect(0, 0, W, H);
      let peak = 1e-12;
      for (let k = 0; k < lvl.length; k++) {
        const v = Math.abs(lvl[k]);
        if (v > peak) peak = v;
      }
      const invPeak = 1 / peak;
      ctx.strokeStyle = '#60a5fa';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = 0; k < lvl.length; k++) {
        const x = (k / Math.max(1, lvl.length - 1)) * W;
        const y = H / 2 - lvl[k] * invPeak * (H / 2 - 1);
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });
  }
}

function safeRenderPyramidStack(pyramid) {
  try { renderPyramidStack(pyramid); }
  catch (err) { console.warn('pyramid stack failed:', err); }
}

// ---------- worker ----------
// LRU cache of training dictionaries lives inside the worker. Key is the
// (training-identity, numLevels, windowHw, featureHw, falloff) tuple sent
// from main.js as a string. On cache hit, the build phase (~30-50% of
// total runtime) is skipped entirely.
const WORKER_HANDLER = `
const TRAINING_CACHE = new Map();
const TRAINING_CACHE_MAX = 3;

function cacheGet(key) {
  if (!TRAINING_CACHE.has(key)) return null;
  // Move to end (most-recently-used) by re-inserting.
  const v = TRAINING_CACHE.get(key);
  TRAINING_CACHE.delete(key);
  TRAINING_CACHE.set(key, v);
  return v;
}

function cachePut(key, value) {
  if (TRAINING_CACHE.has(key)) TRAINING_CACHE.delete(key);
  TRAINING_CACHE.set(key, value);
  while (TRAINING_CACHE.size > TRAINING_CACHE_MAX) {
    const oldest = TRAINING_CACHE.keys().next().value;
    TRAINING_CACHE.delete(oldest);
  }
}

self.addEventListener('message', (event) => {
  const { id, base, training, params, cacheKey } = event.data;
  try {
    const opts = Object.assign({}, params);
    opts.onProgress = (info) => self.postMessage({ id, type: 'progress', info });
    if (cacheKey) opts.trainingCache = cacheGet(cacheKey);
    const out = TextureSynthesis.synthesize(
      new Float64Array(base), new Float64Array(training), opts);
    if (cacheKey && out.trainingCache) cachePut(cacheKey, out.trainingCache);
    const outF32 = Float32Array.from(out.output);
    const scalingF32 = Float32Array.from(out.scalingPerWindow);
    const pyramidF32 = out.outputPyramid.map(l => Float32Array.from(l));
    self.postMessage(
      {
        id, type: 'done',
        output: outF32, scalingPerWindow: scalingF32, outputPyramid: pyramidF32,
        profile: out.profile,
        cacheSize: TRAINING_CACHE.size,
      },
      [outF32.buffer, scalingF32.buffer, ...pyramidF32.map(p => p.buffer)]
    );
  } catch (err) {
    self.postMessage({ id, type: 'error', message: (err && err.message) || String(err) });
  }
});
`;

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
          ? ' (loaded over file://; browsers block fetch under file://. Run `python serve.py 8765` and visit http://127.0.0.1:8765/)'
          : '';
        throw new Error('failed to fetch ' + url + ': ' + (err.message || err) + hint);
      }
    };
    const [kdSrc, rngSrc, tsSrc] = await Promise.all([
      fetchFresh('kdtree.js?v=12'),
      fetchFresh('deterministic_rng.js?v=12'),
      fetchFresh('texture_synthesis.js?v=12'),
    ]);
    const blob = new Blob(
      [kdSrc, '\n;\n', rngSrc, '\n;\n', tsSrc, '\n;\n', WORKER_HANDLER],
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
    const info = msg.info;
    let frac = 0, label = '';
    const nLevels = info.n_levels || 5;
    if (info.stage === 'pyramid') {
      frac = 0.02; label = 'building pyramids…';
    } else if (info.stage === 'build_feat') {
      // 5% of bar per level for feature extraction; level 0 spans 5..30%.
      const base = 0.05 + (info.level / nLevels) * 0.30;
      frac = base + 0.04 * (info.progress || 0);
      const pct = info.progress > 0 ? ` (${(100*info.progress).toFixed(0)}%)` : '';
      label = `extracting level ${info.level} features${pct}…`;
    } else if (info.stage === 'build_tree') {
      frac = 0.05 + ((info.level + 0.5) / nLevels) * 0.30;
      label = `building level ${info.level} KD-tree…`;
    } else if (info.stage === 'build_done') {
      frac = 0.05 + ((info.level + 1) / nLevels) * 0.30;
      label = `level ${info.level} dictionary built`;
    } else if (info.stage === 'synth') {
      frac = 0.40 + 0.55 * (info.progress || 0);
      label = `synthesising level ${info.level}…`;
    } else if (info.stage === 'done') {
      frac = 1.0; label = 'done';
    } else {
      label = info.stage;
    }
    document.getElementById('progress-bar').style.width = (100 * frac) + '%';
    document.getElementById('process-status').textContent = label;
  } else if (msg.type === 'done') {
    extendedBuffer = msg.output;
    outputPyramid = msg.outputPyramid;
    document.getElementById('progress-bar').style.width = '100%';
    const cacheTag = msg.profile && msg.profile.cache_hit ? ' · dict cache HIT' : '';
    document.getElementById('process-status').textContent =
      `done in ${(performance.now() - req.startedAt).toFixed(0)} ms ` +
      `(${msg.scalingPerWindow.length} windows${cacheTag})`;
    document.getElementById('process-status').classList.remove('warn');
    document.getElementById('process-btn').disabled = false;
    pendingRequest = null;
    if (msg.profile) {
      const p = msg.profile;
      console.group('Synthesis profile (ms)');
      console.log('total          :', p.total_ms.toFixed(0));
      console.log('  pyramid build:', p.pyramid_ms.toFixed(0));
      console.log('  CDF init     :', p.cdf_ms.toFixed(0));
      console.log('  KD-tree build:', p.build_ms.toFixed(0),
                   '  per level:', p.build_per_level_ms.map(v => v.toFixed(0)).join(' / '));
      console.log('  synthesis    :', p.synth_ms.toFixed(0),
                   '  per level:', p.synth_per_level_ms.map(v => v.toFixed(0)).join(' / '));
      console.log('    feature gen:', p.feature_ms.toFixed(0));
      console.log('    NN queries :', p.query_ms.toFixed(0),
                   '  (', p.n_queries, 'queries,',
                   (p.query_ms * 1000 / Math.max(1, p.n_queries)).toFixed(2), 'us each)');
      console.log('    blend      :', p.blend_ms.toFixed(0));
      console.log('  features built (training):', p.n_features_built);
      console.groupEnd();
    }
    refreshExtended();
  } else if (msg.type === 'error') {
    document.getElementById('process-status').textContent = 'error: ' + msg.message;
    document.getElementById('process-status').classList.add('warn');
    document.getElementById('process-btn').disabled = false;
    pendingRequest = null;
    showError('synthesis worker error: ' + msg.message);
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
  if (!baseBuffer || !trainingBuffer) {
    document.getElementById('process-status').textContent =
      'load both a base and training signal first';
    document.getElementById('process-status').classList.add('warn');
    return;
  }
  document.getElementById('process-btn').disabled = true;
  document.getElementById('progress-bar').style.width = '0%';
  document.getElementById('process-status').textContent = 'starting…';
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
  const id = ++nextRequestId;
  const baseCopy = new Float64Array(baseBuffer);
  const trainingCopy = new Float64Array(trainingBuffer);
  pendingRequest = { id, startedAt: performance.now() };
  // Cache key for the training dictionary. Anything that changes the
  // training pyramid or the per-level features must be in this key.
  const cacheKey = trainingId
    ? `${trainingId}:${params.num_levels|0}:${params.window_hw|0}:${params.feature_hw|0}:${params.falloff.toFixed(4)}`
    : null;
  w.postMessage({
    id, base: baseCopy, training: trainingCopy, cacheKey,
    params: {
      numLevels: params.num_levels | 0,
      windowHw: params.window_hw | 0,
      featureHw: params.feature_hw | 0,
      falloff: params.falloff,
      scaleCdf: document.getElementById('scale-cdf').checked,
      scalingAlpha: params.scaling_alpha,
      epsilon: params.eps_ann,
    },
  }, [baseCopy.buffer, trainingCopy.buffer]);
}

// ---------- sliders ----------
const sliderSpecs = {
  num_levels:    { min: 3, max: 8, step: 1, scale: 'lin', fmt: v => Math.round(v) + '' },
  window_hw:     { min: 1, max: 16, step: 1, scale: 'lin', fmt: v => Math.round(v) + '' },
  feature_hw:    { min: 1, max: 8, step: 1, scale: 'lin', fmt: v => Math.round(v) + '' },
  falloff:       { min: 0, max: 0.5, step: 0.005, scale: 'lin', fmt: v => v.toFixed(3) },
  scaling_alpha: { min: 0, max: 2, step: 0.01, scale: 'lin', fmt: v => v.toFixed(2) },
  seed:          { min: 0, max: 999999, step: 1, scale: 'lin', fmt: v => Math.round(v) + '' },
  eps_ann:       { min: 0, max: 5, step: 0.05, scale: 'lin', fmt: v => v.toFixed(2) },
};

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

function setupSliders() {
  for (const [key, spec] of Object.entries(sliderSpecs)) {
    const el = document.getElementById('slider-' + key);
    el.min = spec.scale === 'log' ? Math.log(spec.min) : spec.min;
    el.max = spec.scale === 'log' ? Math.log(spec.max) : spec.max;
    el.step = spec.step;
    el.value = spec.scale === 'log' ? Math.log(DEFAULTS[key]) : DEFAULTS[key];
    const valEl = document.getElementById('val-' + key);
    const update = () => {
      const raw = parseFloat(el.value);
      const v = spec.scale === 'log' ? Math.exp(raw) : raw;
      valEl.textContent = spec.fmt(v);
      scheduleAutoProcess();
    };
    update();
    el.addEventListener('input', update);
  }
  document.getElementById('scale-cdf').addEventListener('change', () => scheduleAutoProcess());
  document.getElementById('reset-btn').addEventListener('click', () => {
    for (const [key, spec] of Object.entries(sliderSpecs)) {
      const el = document.getElementById('slider-' + key);
      el.value = spec.scale === 'log' ? Math.log(DEFAULTS[key]) : DEFAULTS[key];
      el.dispatchEvent(new Event('input'));
    }
    document.getElementById('scale-cdf').checked = true;
    document.getElementById('scale-cdf').dispatchEvent(new Event('change'));
  });
  window.addEventListener('resize', () => safeRenderSpectrumPlot());
}

function readParams() {
  const out = {};
  for (const [key, spec] of Object.entries(sliderSpecs)) {
    const raw = parseFloat(document.getElementById('slider-' + key).value);
    out[key] = spec.scale === 'log' ? Math.exp(raw) : raw;
  }
  return out;
}

// ---------- example loading ----------
function prettyName(name) {
  return name.split('_').map(w => w[0].toUpperCase() + w.slice(1)).join(' ');
}

function makeSyntheticBase() {
  const fs = 44100, dur = 1.0, N = Math.floor(fs * dur);
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const t = i / fs;
    const env = Math.exp(-2.0 * t) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 4 * t));
    out[i] = (
      0.55 * Math.sin(2 * Math.PI * 80 * t) +
      0.30 * Math.sin(2 * Math.PI * 130 * t) +
      0.15 * Math.sin(2 * Math.PI * 170 * t)
    ) * env;
  }
  return out;
}

function makeSyntheticTraining() {
  // Pink-ish noise via simple cumulative-sum smoothing on white noise.
  const fs = 44100, dur = 2.0, N = Math.floor(fs * dur);
  const out = new Float32Array(N);
  let acc = 0;
  for (let i = 0; i < N; i++) {
    const w = (Math.random() * 2 - 1) * 0.5;
    acc = 0.97 * acc + w;
    out[i] = acc * 0.05;
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
    const labelEl = row.querySelector('label[for="base-input"]');
    for (const ex of m.examples) {
      const btn = document.createElement('button');
      btn.textContent = prettyName(ex.name);
      btn.title = `base: ${ex.input_samples} samples · training: ${ex.training_samples} samples (${ex.training_basename})`;
      btn.dataset.name = ex.name;
      btn.addEventListener('click', () => loadExample(ex.name));
      row.insertBefore(btn, labelEl);
    }
    if (m.examples.length) await loadExample(m.examples[0].name);
  } catch (err) {
    showBanner(
      'Could not load the bundled examples (' + err + '). The synthetic preset still works.',
      false
    );
    loadSynthetic();
  }
}

async function loadExample(name) {
  if (!manifest) return loadSynthetic();
  const ex = manifest.examples.find(e => e.name === name);
  if (!ex) return;
  try {
    const [baseBuf, trainBuf, paramsResp] = await Promise.all([
      fetch('assets/' + ex.input_file).then(r => r.arrayBuffer()),
      fetch('assets/' + ex.training_file).then(r => r.arrayBuffer()),
      fetch('assets/' + ex.params_file).then(r => r.json()),
    ]);
    const base = new Float32Array(baseBuf);
    const training = new Float32Array(trainBuf);
    setSignals(base, training, name, { trainingId: 'example:' + name });
    applyParamsFromExample(paramsResp);
    markActive(name);
    startProcess();
  } catch (err) {
    showError('Failed to load example ' + name + ': ' + err);
  }
}

function applyParamsFromExample(p) {
  if (!p) return;
  // Apply only the paper-spec algorithm parameters from the per-example
  // default.json. eps_ann is omitted intentionally: it's a performance /
  // approximation knob (the C++ release default of 1.0 is in the JSON,
  // but the demo's own default of 2.0 yields a noticeably faster live
  // synthesis with no audible change). The slider value persists across
  // example switches.
  const map = {
    num_levels: p.num_levels,
    window_hw: p.window_hw,
    feature_hw: p.feature_hw,
    falloff: p.falloff,
    scaling_alpha: p.scaling_alpha,
  };
  for (const [k, v] of Object.entries(map)) {
    if (v === undefined) continue;
    const el = document.getElementById('slider-' + k);
    if (!el) continue;
    el.value = v;
    el.dispatchEvent(new Event('input'));
  }
  if (p.scale_cdf !== undefined) {
    document.getElementById('scale-cdf').checked = !!p.scale_cdf;
  }
}

function loadSynthetic() {
  // Synthetic training uses Math.random per click, so it differs each time;
  // use an incrementing id so the cache invalidates correctly.
  setSignals(makeSyntheticBase(), makeSyntheticTraining(), 'synthetic',
              { trainingId: 'synthetic:' + (++_uploadCounter) });
  markActive('__synthetic__');
  startProcess();
}

function markActive(name) {
  document.querySelectorAll('#example-row button').forEach(b => {
    const key = b.dataset.builtin === 'synthetic' ? '__synthetic__' : b.dataset.name;
    b.classList.toggle('active', key === name || (name === 'synthetic' && key === '__synthetic__'));
  });
}

async function loadWavInto(file, slot) {
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
    if (slot === 'base') {
      // Only the base changed; training (and its cached dictionary) is reusable.
      setSignals(samples, trainingBuffer, file.name.replace(/\.[^.]+$/, ''),
                  { trainingChanged: false });
    } else {
      // Training changed; assign a fresh id so the cache invalidates.
      setSignals(baseBuffer, samples,
                  currentName + ' / ' + file.name.replace(/\.[^.]+$/, ''),
                  { trainingChanged: true,
                    trainingId: 'upload:' + (++_uploadCounter) });
    }
    markActive(null);
    if (baseBuffer && trainingBuffer) startProcess();
  } catch (err) {
    showError('Could not decode WAV: ' + err);
  }
}

function setSignals(base, training, name, opts) {
  opts = opts || {};
  baseBuffer = base;
  // If trainingChanged is unspecified, infer from object identity.
  const trainingChanged = (opts.trainingChanged !== undefined)
    ? opts.trainingChanged
    : (training !== trainingBuffer);
  trainingBuffer = training;
  extendedBuffer = null;
  outputPyramid = null;
  currentName = name;
  if (trainingChanged) {
    trainingId = opts.trainingId || ('tid_' + (++_uploadCounter));
  }
  const baseDur = base ? (base.length / 44100).toFixed(2) : '?';
  const trainDur = training ? (training.length / 44100).toFixed(2) : '?';
  document.getElementById('current-name').textContent =
    `${prettyName(name)}  ·  base ${baseDur}s  ·  training ${trainDur}s`;
  safeRenderSpec('spec-input', base);
  safeRenderSpec('spec-training', training);
  safeRenderSpec('spec-output', null);
  safeRenderPyramidStack(null);
  safeRenderSpectrumPlot();
  origPlayer.refresh();
  trainPlayer.refresh();
  extPlayer.refresh();
}

function refreshExtended() {
  safeRenderSpec('spec-output', extendedBuffer);
  safeRenderPyramidStack(outputPyramid);
  safeRenderSpectrumPlot();
  extPlayer.refresh();
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
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
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
  a.href = url; a.download = baseName + '.wav';
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
}

function wireDownloadButtons() {
  document.querySelector('#player-original .download').addEventListener('click', () => {
    downloadWav(baseBuffer, (currentName || 'base') + '_base');
  });
  document.querySelector('#player-training .download').addEventListener('click', () => {
    downloadWav(trainingBuffer, (currentName || 'training') + '_training');
  });
  document.querySelector('#player-extended .download').addEventListener('click', () => {
    if (!extendedBuffer) {
      showBanner('Click <strong>Synthesize</strong> first to generate the output signal.', false);
      return;
    }
    downloadWav(extendedBuffer, (currentName || 'signal') + '_synthesized');
  });
}

// ---------- bootstrap ----------
function bootstrap() {
  setupSliders();
  origPlayer = makePlayer('original', () => baseBuffer);
  trainPlayer = makePlayer('training', () => trainingBuffer);
  extPlayer = makePlayer('extended', () => extendedBuffer, '(not yet synthesized)');
  wireDownloadButtons();
  origPlayer.refresh();
  trainPlayer.refresh();
  extPlayer.refresh();

  document.getElementById('process-btn').addEventListener('click', startProcess);
  document.getElementById('base-input').addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (f) loadWavInto(f, 'base');
  });
  document.getElementById('training-input').addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (f) loadWavInto(f, 'training');
  });
  document.querySelector('#example-row button[data-builtin="synthetic"]')
    .addEventListener('click', () => loadSynthetic());

  if (location.protocol === 'file:') {
    showBanner(
      '<strong>This page is loaded from <code>file://</code>.</strong> ' +
      'Browsers block <code>fetch()</code> and Web Workers under file://, ' +
      'so the synthesis can&rsquo;t run. Open a terminal in ' +
      '<code>fire-texture-synthesis/</code> and run ' +
      '<code>python serve.py 8765</code>, then visit ' +
      '<a href="http://127.0.0.1:8765/" style="color:inherit;text-decoration:underline;">' +
      'http://127.0.0.1:8765/</a>.',
      true
    );
    document.getElementById('process-btn').disabled = true;
    document.getElementById('process-status').textContent = 'disabled (file:// protocol)';
    document.getElementById('process-status').classList.add('warn');
    loadSynthetic();
    return;
  }

  loadSynthetic();
  loadManifest();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrap);
} else {
  bootstrap();
}
