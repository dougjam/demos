// Atmospheric Absorption demo - main app.
// (c) 2026 Doug James, Stanford University. BSD-2-Clause.

window.addEventListener("error", (ev) => {
  const div = document.createElement("div");
  div.style.cssText =
    "background:#fee;border:1px solid #b00;color:#700;padding:.5rem 1rem;" +
    "margin:.5rem 0;font:13px/1.4 ui-monospace,monospace;white-space:pre-wrap;";
  div.textContent = `JS error: ${ev.message}\n  at ${ev.filename}:${ev.lineno}`;
  document.body.insertBefore(div, document.body.firstChild);
});

if (!window.ISO9613) throw new Error("iso9613.js failed to load");
if (typeof Chart === "undefined") throw new Error("Chart.js failed to load (CDN blocked?)");

const { alphaDbPerKm, alphaDbPerM } = window.ISO9613;

// --------------------------------------------------------------------------
// State
// --------------------------------------------------------------------------
const state = {
  T_C: 20,
  RH: 50,
  p_kPa: 101.325,
  r_m: 100,
  vol: 0.5,
  preset: "white",
  ab: "absorbed",
  invR: false,
  loop: true,
  playing: false,
};

// Distance slider: 0..200 maps log-uniformly to 1 m..100000 m.
const rFromSlider = (s) => Math.pow(10, (s / 200) * 5);
const sliderFromR = (r) => Math.log10(r) / 5 * 200;

function formatDistance(r) {
  if (r < 1000) return `${r < 10 ? r.toFixed(2) : r < 100 ? r.toFixed(1) : Math.round(r)} m`;
  const km = r / 1000;
  return `${km < 10 ? km.toFixed(2) : km < 100 ? km.toFixed(1) : Math.round(km)} km`;
}

// --------------------------------------------------------------------------
// Plot
// --------------------------------------------------------------------------
const N_PLOT = 256;
const F_MIN = 10, F_MAX = 1e5;
const plotFreqs = new Array(N_PLOT);
for (let i = 0; i < N_PLOT; i++) {
  plotFreqs[i] = F_MIN * Math.pow(F_MAX / F_MIN, i / (N_PLOT - 1));
}

const Y_CLIP = 1e3;     // clip above this
const Y_FLOOR = 1e-3;   // clamp below
const clamp = (v) => Math.min(Y_CLIP * 1.001, Math.max(Y_FLOOR, v));

// Plugin: faint vertical guides at 20, 1k, 20k Hz; "beyond audibility" tag.
const guidesPlugin = {
  id: "guides",
  afterDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    const xs = scales.x;
    ctx.save();
    ctx.strokeStyle = "rgba(0,0,0,0.10)";
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    for (const f of [20, 1000, 20000]) {
      const x = xs.getPixelForValue(f);
      if (x < chartArea.left || x > chartArea.right) continue;
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.font = "10px system-ui";
      ctx.textAlign = "center";
      const lbl = f >= 1000 ? `${f / 1000} kHz` : `${f} Hz`;
      ctx.fillText(lbl, x, chartArea.top - 2);
      ctx.setLineDash([3, 3]);
    }
    ctx.restore();

    // "beyond audibility" tag if absorbed curve clipped.
    const ds = chart.data.datasets[1];
    const maxRaw = ds._maxRaw || 0;
    if (maxRaw > Y_CLIP) {
      ctx.save();
      ctx.fillStyle = "rgba(180, 0, 32, 0.7)";
      ctx.font = "11px system-ui";
      ctx.textAlign = "right";
      ctx.fillText("beyond audibility", chartArea.right - 4, chartArea.top + 12);
      ctx.restore();
    }
  },
};

function initialPoints() {
  return plotFreqs.map((f) => ({ x: f, y: Y_FLOOR }));
}

const ctx = document.getElementById("plotCanvas").getContext("2d");
const chart = new Chart(ctx, {
  type: "line",
  data: {
    datasets: [
      {
        label: "alpha(f), dB/km",
        data: initialPoints(),
        borderColor: "rgba(60,60,60,0.55)",
        backgroundColor: "transparent",
        borderWidth: 1.2,
        borderDash: [5, 4],
        pointRadius: 0,
        tension: 0,
      },
      {
        label: "A(f, r) = alpha * r, dB",
        data: initialPoints(),
        borderColor: "#2563eb",
        backgroundColor: "transparent",
        borderWidth: 2.4,
        pointRadius: 0,
        tension: 0,
      },
    ],
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    parsing: false,
    scales: {
      x: {
        type: "logarithmic",
        min: F_MIN,
        max: F_MAX,
        title: { display: true, text: "Frequency (Hz)" },
        ticks: {
          callback: (v) => {
            const l = Math.log10(v);
            if (Math.abs(l - Math.round(l)) > 1e-3) return "";
            return v >= 1000 ? `${v / 1000}k` : `${v}`;
          },
        },
      },
      y: {
        type: "logarithmic",
        min: Y_FLOOR,
        max: Y_CLIP,
        title: { display: true, text: "Attenuation (dB)" },
        ticks: {
          callback: (v) => {
            const l = Math.log10(v);
            if (Math.abs(l - Math.round(l)) > 1e-3) return "";
            return v < 1 ? v.toString() : v.toString();
          },
        },
      },
    },
    plugins: {
      legend: { position: "bottom", labels: { boxWidth: 24, font: { size: 11 } } },
      tooltip: { enabled: false },
    },
  },
  plugins: [guidesPlugin],
});

// Convert (freq, value) arrays into Chart.js {x, y} parsing:false format.
function setDataset(ds, ys) {
  const pts = ds.data;
  for (let i = 0; i < N_PLOT; i++) {
    pts[i] = { x: plotFreqs[i], y: clamp(ys[i]) };
  }
}

// Vertical drag on the plot moves r. The y-axis is logarithmic, so a
// pixel delta corresponds to a multiplicative factor on every curve
// value (and hence on r, since A = alpha * r and only r depends on
// the slider). This lets the user "grab" the blue curve and pull it
// up/down — the slider follows.
const plotCanvas = document.getElementById("plotCanvas");
plotCanvas.style.cursor = "ns-resize";
plotCanvas.style.touchAction = "none"; // suppress mobile vertical-scroll on drag
let dragStartPixelY = null;
let dragStartR = null;
let dragPixelsPerDecade = null;

plotCanvas.addEventListener("pointerdown", (ev) => {
  // Only respond to primary button / single-touch.
  if (ev.button !== undefined && ev.button !== 0) return;
  const yScale = chart.scales.y;
  if (!yScale) return;
  // pixels-per-decade: difference between two known values on the log axis.
  dragPixelsPerDecade = yScale.getPixelForValue(1) - yScale.getPixelForValue(10);
  if (!isFinite(dragPixelsPerDecade) || dragPixelsPerDecade === 0) return;
  dragStartPixelY = ev.clientY;
  dragStartR = state.r_m;
  plotCanvas.setPointerCapture(ev.pointerId);
  ev.preventDefault();
});

plotCanvas.addEventListener("pointermove", (ev) => {
  if (dragStartPixelY === null) return;
  // Moving up on screen (smaller clientY) increases value on log axis.
  const decades = (dragStartPixelY - ev.clientY) / dragPixelsPerDecade;
  // A doubling of r is +6 dB everywhere; on log10 y-axis that's
  // log10(2) decades = 0.301 per doubling. So r_new = r_start * 10^decades.
  let r = dragStartR * Math.pow(10, decades);
  r = Math.max(1, Math.min(100000, r));
  state.r_m = r;
  // Sync the slider position so the input feels coupled.
  rSlider.value = String(Math.round(sliderFromR(r)));
  paramChanged();
});

function endDrag(ev) {
  if (dragStartPixelY === null) return;
  dragStartPixelY = null;
  if (ev && ev.pointerId !== undefined) {
    try { plotCanvas.releasePointerCapture(ev.pointerId); } catch (_) {}
  }
}
plotCanvas.addEventListener("pointerup", endDrag);
plotCanvas.addEventListener("pointercancel", endDrag);

function updatePlot() {
  const alpha_km = new Array(N_PLOT);
  const A_dB     = new Array(N_PLOT);
  const r_km = state.r_m / 1000;
  let maxA = 0;
  for (let i = 0; i < N_PLOT; i++) {
    const a = alphaDbPerKm(plotFreqs[i], state.T_C, state.RH, state.p_kPa);
    alpha_km[i] = a;
    const A = a * r_km;
    A_dB[i] = A;
    if (A > maxA) maxA = A;
  }
  setDataset(chart.data.datasets[0], alpha_km);
  setDataset(chart.data.datasets[1], A_dB);
  chart.data.datasets[1]._maxRaw = maxA;
  chart.data.datasets[1].label =
    `A(f, r) at ${formatDistance(state.r_m)}, dB`;
  chart.update("none");
}

// --------------------------------------------------------------------------
// Sliders / inputs
// --------------------------------------------------------------------------
const tSlider  = document.getElementById("tSlider");
const rhSlider = document.getElementById("rhSlider");
const pSlider  = document.getElementById("pSlider");
const rSlider  = document.getElementById("rSlider");
const volSlider = document.getElementById("volSlider");
const tVal = document.getElementById("tVal");
const rhVal = document.getElementById("rhVal");
const pVal = document.getElementById("pVal");
const rVal = document.getElementById("rVal");
const volVal = document.getElementById("volVal");

// Initialize distance slider position from default state.r_m=100.
rSlider.value = String(Math.round(sliderFromR(state.r_m)));

function refreshLabels() {
  tVal.textContent  = `${state.T_C} °C`;
  rhVal.textContent = `${state.RH} %`;
  pVal.textContent  = `${state.p_kPa.toFixed(1)} kPa`;
  rVal.textContent  = formatDistance(state.r_m);
  volVal.textContent = state.vol.toFixed(2);
}

function paramChanged() {
  updatePlot();
  refreshLabels();
  updateBands();
  // If 1/r spreading is engaged, the dry/wet gain depends on r and
  // needs to track distance-slider drags too. applyAB is a no-op
  // before audio init.
  if (state.invR) applyAB();
}

tSlider.addEventListener("input", () => {
  state.T_C = +tSlider.value; paramChanged();
});
rhSlider.addEventListener("input", () => {
  state.RH = +rhSlider.value; paramChanged();
});
pSlider.addEventListener("input", () => {
  state.p_kPa = (+pSlider.value) / 10;
  paramChanged();
});
rSlider.addEventListener("input", () => {
  state.r_m = rFromSlider(+rSlider.value); paramChanged();
});
volSlider.addEventListener("input", () => {
  state.vol = (+volSlider.value) / 100;
  refreshLabels();
  if (masterGain) masterGain.gain.setTargetAtTime(state.vol, audioCtx.currentTime, 0.01);
});

document.getElementById("invROpt").addEventListener("change", (e) => {
  state.invR = e.target.checked;
  applyAB();
});
document.getElementById("loopOpt").addEventListener("change", (e) => {
  state.loop = e.target.checked;
  if (sourceNode && sourceNode.loop !== undefined) sourceNode.loop = state.loop;
});
document.getElementById("abDry").addEventListener("change", () => {
  state.ab = "dry"; applyAB();
});
document.getElementById("abAbs").addEventListener("change", () => {
  state.ab = "absorbed"; applyAB();
});

const presetBar = document.getElementById("presetBar");
presetBar.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-preset]");
  if (!btn) return;
  for (const b of presetBar.querySelectorAll("button")) b.classList.remove("active");
  btn.classList.add("active");
  state.preset = btn.dataset.preset;
  if (state.playing) startSource(true);
});

document.getElementById("playBtn").addEventListener("click", togglePlay);
document.getElementById("reloadBtn").addEventListener("click", () => {
  wavCache = {};
  if (state.playing) startSource(true);
});

refreshLabels();
updatePlot();

// --------------------------------------------------------------------------
// Audio
// --------------------------------------------------------------------------
let audioCtx = null;
let masterGain = null;
let dryGain, wetGain;
// Wet path: cascade of peaking biquads forming a graphic-EQ-style
// approximation of 10^(-alpha(f)*r/20). Each biquad's .gain is an
// AudioParam that we drive with setTargetAtTime, which gives smooth,
// glitch-free transitions during continuous slider drag — the FIR /
// convolver-buffer-swap approach inherently glitches in some browsers
// no matter how cleverly we crossfade. The spec calls this out as the
// preferred fallback for exactly this reason.
let bandFilters = [];
const BAND_FREQS = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
const BAND_Q = 1.41;       // ~1-octave width
const GAIN_FLOOR_DB = -80; // clip aggressive attenuation
const GAIN_TAU_S = 0.04;   // setTargetAtTime time constant for band-gain changes

let sourceNode = null;
let noiseNode = null;
let wavCache = {};
let workletReady = false;

const SR = 48000;

async function ensureAudio() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  masterGain = audioCtx.createGain();
  masterGain.gain.value = state.vol;
  dryGain = audioCtx.createGain();
  wetGain = audioCtx.createGain();
  dryGain.connect(masterGain);
  // Build the wet cascade: wetGain -> [peak f0] -> [peak f1] -> ... -> masterGain
  bandFilters = BAND_FREQS.map((f) => {
    const b = audioCtx.createBiquadFilter();
    b.type = "peaking";
    b.frequency.value = f;
    b.Q.value = BAND_Q;
    b.gain.value = 0;
    return b;
  });
  let prev = wetGain;
  for (const b of bandFilters) { prev.connect(b); prev = b; }
  prev.connect(masterGain);
  masterGain.connect(audioCtx.destination);
  applyAB();
  updateBands(/*immediate=*/true);

  try {
    const blob = new Blob([NOISE_WORKLET_SRC], { type: "application/javascript" });
    await audioCtx.audioWorklet.addModule(URL.createObjectURL(blob));
    workletReady = true;
  } catch (e) {
    workletReady = false;
    console.warn("AudioWorklet failed; falling back to ScriptProcessor.", e);
  }
}

function applyAB() {
  if (!audioCtx) return;
  const r = state.invR ? 1 / Math.max(1, state.r_m) : 1;
  const t = audioCtx.currentTime;
  const tau = 0.01;
  dryGain.gain.setTargetAtTime(state.ab === "dry" ? r : 0, t, tau);
  wetGain.gain.setTargetAtTime(state.ab === "absorbed" ? r : 0, t, tau);
}

// Recompute each band's gain from the current ISO 9613-1 alpha and
// distance, and schedule a smooth approach. Cheap; safe to call on
// every slider 'input' event — no debounce needed.
function updateBands(immediate = false) {
  if (!audioCtx) return;
  const t = audioCtx.currentTime;
  for (let i = 0; i < bandFilters.length; i++) {
    const f = BAND_FREQS[i];
    const alpha = alphaDbPerM(f, state.T_C, state.RH, state.p_kPa);
    const gainDb = Math.max(GAIN_FLOOR_DB, -alpha * state.r_m);
    const p = bandFilters[i].gain;
    if (immediate) {
      p.cancelScheduledValues(t);
      p.setValueAtTime(gainDb, t);
    } else {
      p.setTargetAtTime(gainDb, t, GAIN_TAU_S);
    }
  }
}

// --------------------------------------------------------------------------
// Noise AudioWorklet (white + pink, Paul Kellet IIR)
// --------------------------------------------------------------------------
const NOISE_WORKLET_SRC = `
class NoiseProc extends AudioWorkletProcessor {
  static get parameterDescriptors() { return []; }
  constructor(opts) {
    super();
    this.kind = (opts && opts.processorOptions && opts.processorOptions.kind) || "white";
    this.b0 = this.b1 = this.b2 = this.b3 = this.b4 = this.b5 = this.b6 = 0;
  }
  process(_in, out) {
    const ch = out[0][0];
    if (this.kind === "white") {
      for (let i = 0; i < ch.length; i++) {
        // Box-Muller via two uniforms, then scale to ~ -12 dBFS RMS.
        const u1 = Math.random() || 1e-9, u2 = Math.random();
        ch[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * 0.12;
      }
    } else {
      // Paul Kellet pink filter.
      for (let i = 0; i < ch.length; i++) {
        const w = (Math.random() * 2 - 1);
        this.b0 = 0.99886 * this.b0 + w * 0.0555179;
        this.b1 = 0.99332 * this.b1 + w * 0.0750759;
        this.b2 = 0.96900 * this.b2 + w * 0.1538520;
        this.b3 = 0.86650 * this.b3 + w * 0.3104856;
        this.b4 = 0.55000 * this.b4 + w * 0.5329522;
        this.b5 = -0.7616 * this.b5 - w * 0.0168980;
        const pink = this.b0 + this.b1 + this.b2 + this.b3 + this.b4 + this.b5
                    + this.b6 + w * 0.5362;
        this.b6 = w * 0.115926;
        ch[i] = pink * 0.11;
      }
    }
    return true;
  }
}
registerProcessor("noise-proc", NoiseProc);
`;

// --------------------------------------------------------------------------
// Source playback
// --------------------------------------------------------------------------
function stopSource() {
  if (sourceNode) {
    try { sourceNode.stop(); } catch (_) {}
    try { sourceNode.disconnect(); } catch (_) {}
    sourceNode = null;
  }
  if (noiseNode) {
    try { noiseNode.disconnect(); } catch (_) {}
    noiseNode = null;
  }
}

function connectSource(node) {
  node.connect(dryGain);
  node.connect(wetGain);
}

// Bumping AUDIO_VERSION forces browsers to re-fetch the WAVs even when
// they're already cached. Bump this any time you regenerate or replace
// a file in audio/.
const AUDIO_VERSION = 10;

async function startWav(name) {
  if (!wavCache[name]) {
    try {
      const resp = await fetch(`audio/${name}.wav?v=${AUDIO_VERSION}`, { cache: "no-cache" });
      if (!resp.ok) throw new Error(resp.status);
      const ab = await resp.arrayBuffer();
      wavCache[name] = await audioCtx.decodeAudioData(ab);
    } catch (e) {
      console.warn(`audio/${name}.wav missing or failed; falling back to white noise.`, e);
      return startNoise("white");
    }
  }
  const src = audioCtx.createBufferSource();
  src.buffer = wavCache[name];
  src.loop = state.loop;
  connectSource(src);
  src.start();
  sourceNode = src;
}

function startNoise(kind) {
  if (workletReady) {
    const n = new AudioWorkletNode(audioCtx, "noise-proc", {
      processorOptions: { kind },
    });
    connectSource(n);
    noiseNode = n;
  } else {
    // Fallback: ScriptProcessor.
    const proc = audioCtx.createScriptProcessor(2048, 0, 1);
    let b0=0,b1=0,b2=0,b3=0,b4=0,b5=0,b6=0;
    proc.onaudioprocess = (ev) => {
      const ch = ev.outputBuffer.getChannelData(0);
      if (kind === "white") {
        for (let i = 0; i < ch.length; i++) {
          const u1 = Math.random() || 1e-9, u2 = Math.random();
          ch[i] = Math.sqrt(-2*Math.log(u1))*Math.cos(2*Math.PI*u2)*0.12;
        }
      } else {
        for (let i = 0; i < ch.length; i++) {
          const w = Math.random()*2-1;
          b0 = 0.99886*b0 + w*0.0555179;
          b1 = 0.99332*b1 + w*0.0750759;
          b2 = 0.96900*b2 + w*0.1538520;
          b3 = 0.86650*b3 + w*0.3104856;
          b4 = 0.55000*b4 + w*0.5329522;
          b5 = -0.7616*b5 - w*0.0168980;
          const pink = b0+b1+b2+b3+b4+b5+b6 + w*0.5362;
          b6 = w*0.115926;
          ch[i] = pink * 0.11;
        }
      }
    };
    connectSource(proc);
    noiseNode = proc;
  }
}

async function startSource(restart=false) {
  await ensureAudio();
  if (audioCtx.state === "suspended") await audioCtx.resume();
  if (restart) stopSource();
  const p = state.preset;
  if (p === "white" || p === "pink") startNoise(p);
  else await startWav(p);
}

async function togglePlay() {
  await ensureAudio();
  const btn = document.getElementById("playBtn");
  if (state.playing) {
    stopSource();
    state.playing = false;
    btn.textContent = "Start";
  } else {
    await startSource(true);
    state.playing = true;
    btn.textContent = "Stop";
  }
}
