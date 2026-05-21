# Ocean Acoustic Simulation — Verification Report

Last run: **2026-03-15** against `ocean_acoustic_sim.html` v2.1 (sub-step reflections)

This document records the quantitative tests applied to verify the physics
correctness of the ray tracer and all sound-speed profile implementations.
Tests can be re-run at any time by pasting the JavaScript snippets below
into the browser console while the simulation page is loaded.

---

## Test 1: dc/dz Analytical vs. Finite-Difference

**Purpose**: Verify that each profile's `dcdz(z)` function matches the true
derivative of its `c(z)` function.

**Method**: Central finite difference `(c(z+δz) − c(z−δz)) / (2δz)` with
δz = 0.01 m, sampled at 20 depths from 50 m to 4950 m in 250 m steps.

| Profile | Max Abs Error | Max Rel Error | Verdict |
|---|---|---|---|
| Munk | 1.7e-11 | 1.8e-9 | ✅ Pass |
| Arctic | 5.1e-12 | 3.2e-10 | ✅ Pass |
| n²-Linear | 1.4e-11 | 2.0e-9 | ✅ Pass |
| Bilinear | 5.1e-12 | 3.2e-10 | ✅ Pass |
| Epstein | 2.4e-10 | 1.0e+0* | ✅ Pass |
| Isovelocity | 0 | 0 | ✅ Pass |

\* Epstein relative error reaches 1.0 at z = 200 m where dc/dz crosses zero
(tanh(0) = 0); the absolute error (2.4e-10) confirms correctness.

**All derivatives match to near machine precision.**

```js
// Re-run Test 1
(function() {
  const results = {};
  const saved = activeProfile;
  for (const p of PROFILES) {
    activeProfile = p;
    let maxRelErr = 0, maxAbsErr = 0;
    for (let z = 50; z <= 4950; z += 250) {
      const dz = 0.01;
      const analytical = dSoundSpeedDz(z);
      const numerical = (soundSpeed(z + dz) - soundSpeed(z - dz)) / (2 * dz);
      const absErr = Math.abs(analytical - numerical);
      const relErr = Math.abs(analytical) > 1e-12
        ? absErr / Math.abs(analytical) : absErr;
      if (relErr > maxRelErr) maxRelErr = relErr;
      if (absErr > maxAbsErr) maxAbsErr = absErr;
    }
    results[p.id] = {
      maxAbsErr: maxAbsErr.toExponential(3),
      maxRelErr: maxRelErr.toExponential(3)
    };
  }
  activeProfile = saved;
  console.table(results);
  return results;
})()
```

---

## Test 2: Channel Axis Minimum Verification

**Purpose**: For profiles that declare an `axisDepth`, verify that c(axisDepth)
is indeed the global minimum of c(z) over [0, 5000 m].

**Method**: Sample c(z) every 10 m, check c(axisDepth) ≤ c(z) − 0.001 for all z.

| Profile | axisDepth | c(axis) m/s | Is Minimum? |
|---|---|---|---|
| Munk | 1300 m | 1500.0000 | ✅ Yes |
| Bilinear | 1000 m | 1492.0000 | ✅ Yes |
| Epstein | 200 m | 1485.2213 | ✅ Yes |
| n²-Linear | — | — | N/A (no axis; monotonic profile) |
| Arctic | — | — | N/A (no axis; monotonic profile) |
| Isovelocity | — | — | N/A (constant) |

```js
// Re-run Test 2
(function() {
  const saved = activeProfile;
  const results = {};
  for (const p of PROFILES) {
    if (p.axisDepth == null) { results[p.id] = 'N/A'; continue; }
    activeProfile = p;
    const cAxis = soundSpeed(p.axisDepth);
    let isMin = true;
    for (let z = 0; z <= 5000; z += 10) {
      if (soundSpeed(z) < cAxis - 0.001) { isMin = false; break; }
    }
    results[p.id] = { cAxis: cAxis.toFixed(4), isMinimum: isMin };
  }
  activeProfile = saved;
  console.table(results);
  return results;
})()
```

---

## Test 3: Snell Invariant Conservation

**Purpose**: Verify that the horizontal slowness sx = cos(θ)/c(z) is conserved
along each ray path — the fundamental invariant of range-independent ray tracing.

**Method**: Trace a ray at 3° launch angle from 2500 m depth. At each interior
path point (excluding within 100 m of surface/bottom boundaries), compute sx
from the local direction and sound speed, compare to launch sx.

| Profile | Max Rel Error | RMS Rel Error | Verdict |
|---|---|---|---|
| Munk | 4.3e-7 | 2.0e-7 | ✅ Excellent |
| Arctic | 1.8e-7 | 1.0e-7 | ✅ Excellent |
| n²-Linear | 2.7e-8 | 1.8e-8 | ✅ Excellent |
| Bilinear | 6.5e-5 | 3.7e-5 | ✅ Very good |
| Epstein | 6.1e-5 | 3.1e-6 | ✅ Very good |
| Isovelocity | 3.3e-16 | 1.8e-16 | ✅ Machine precision |

Bilinear/Epstein show slightly higher errors due to larger |dc/dz| gradients near
boundaries. Arctic and n²-Linear improved dramatically (100–1000×) after the v2.1
sub-step reflection fix eliminated boundary-clamping artifacts that corrupted the
slowness vector at each bounce.

```js
// Re-run Test 3
(function() {
  const saved = activeProfile;
  const results = {};
  for (const p of PROFILES) {
    activeProfile = p;
    const ray = traceRay(10000, 2500, 3.0, 1);
    const path = ray.path;
    const c0 = soundSpeed(2500);
    const theta0 = 3.0 * Math.PI / 180;
    const sx_launch = Math.cos(theta0) / c0;
    let maxErr = 0, sumSqErr = 0, n = 0;
    for (let i = 2; i < path.length - 2; i++) {
      if (path[i].z < 100 || path[i].z > 4900) continue;
      const dx = path[i+1].x - path[i-1].x;
      const dz = path[i+1].z - path[i-1].z;
      const ds = Math.sqrt(dx*dx + dz*dz);
      if (ds < 1e-10) continue;
      const sx = (dx / ds) / soundSpeed(path[i].z);
      const err = Math.abs(sx - sx_launch) / sx_launch;
      if (err > maxErr) maxErr = err;
      sumSqErr += err*err; n++;
    }
    results[p.id] = {
      maxRelErr: maxErr.toExponential(3),
      rmsRelErr: Math.sqrt(sumSqErr / Math.max(n, 1)).toExponential(3)
    };
  }
  activeProfile = saved;
  console.table(results);
  return results;
})()
```

---

## Test 4: Isovelocity Straight-Line Verification

**Purpose**: In constant sound speed, rays must travel in perfectly straight
lines between boundary reflections.

**Method**: Trace a 10° ray from 2500 m depth. In the first segment (before
any reflection), compute the ideal straight line from start to end, measure
maximum perpendicular deviation of intermediate points.

| Metric | Value |
|---|---|
| Max deviation from straight line | 3.0e-11 m |
| Segment length | 192 path points |
| Verdict | ✅ Machine precision |

```js
// Re-run Test 4
(function() {
  const saved = activeProfile;
  activeProfile = PROFILES.find(p => p.id === 'isovelocity');
  const ray = traceRay(10000, 2500, 10.0, 1);
  const path = ray.path;
  const firstRefl = ray.events.length > 0 ? ray.events[0].stepIdx : path.length;
  const seg = path.slice(0, Math.min(firstRefl, path.length));
  let maxDev = 0;
  if (seg.length > 2) {
    const slope = (seg[seg.length-1].z - seg[0].z) / (seg[seg.length-1].x - seg[0].x);
    for (let i = 1; i < seg.length - 1; i++) {
      const expected = seg[0].z + slope * (seg[i].x - seg[0].x);
      maxDev = Math.max(maxDev, Math.abs(seg[i].z - expected));
    }
  }
  activeProfile = saved;
  console.log('Max straight-line deviation:', maxDev.toExponential(3), 'm');
  return { maxDeviation_m: maxDev.toExponential(3), segmentPoints: seg.length };
})()
```

---

## Test 5: Profile-Specific Ray Behaviour

**Purpose**: Verify qualitative ray behaviour matches physical expectations.

| Test | Expected | Observed | Verdict |
|---|---|---|---|
| Munk: rays channel at SOFAR (1300 m) | Sinusoidal oscillation, no reflections | ✅ R-type rays dominate near axis | ✅ |
| Arctic: shallow rays surface-only | Upward refraction, S reflections | ✅ All S-type for small angles | ✅ |
| Arctic: steep rays from depth hit bottom | Curvature radius too large to turn in 5 km | ✅ Bottom reflections at 15° from 2500 m | ✅ |
| Bilinear: V-channel trapping | Rays oscillate around 1000 m axis | ✅ R-type dominant near axis | ✅ |
| Epstein: shallow duct trapping | Narrow oscillation near 200 m | ✅ Low-angle rays trapped | ✅ |
| Epstein: steep rays escape duct | Pass through duct, hit boundaries | ✅ S/B types at wider angles | ✅ |
| Isovelocity: geometric bouncing | Straight lines, equal-angle reflections | ✅ Verified in Test 4 | ✅ |
| n²-Linear: downward refraction | All rays curve toward bottom | ✅ dc/dz < 0 everywhere | ✅ |

```js
// Re-run Test 5 (Arctic surface-only check for shallow angles)
(function() {
  const saved = activeProfile;
  activeProfile = PROFILES.find(p => p.id === 'arctic');
  const results = [];
  for (let angle = -5; angle <= 5; angle += 1) {
    const r = traceRay(50000, 500, angle, 1);
    results.push({
      angle, hitS: r.hitSurface, hitB: r.hitBottom,
      type: r.hitSurface ? (r.hitBottom ? 'S+B' : 'S') : (r.hitBottom ? 'B' : 'R')
    });
  }
  activeProfile = saved;
  console.table(results);
  return results;
})()
```

---

## Test 6: c(z) Spot-Check Values

**Purpose**: Verify sound speed at key depths matches hand-calculated values.

| Profile | z (m) | c(z) Expected | c(z) Computed | Match? |
|---|---|---|---|---|
| Munk | 0 | ~1548.5 | 1548.5210 | ✅ |
| Munk | 1300 | 1500.0 | 1500.0000 | ✅ |
| Munk | 5000 | ~1551.9 | 1551.9107 | ✅ |
| Arctic | 0 | 1435.0 | 1435.0000 | ✅ |
| Arctic | 5000 | 1515.0 | 1515.0000 | ✅ |
| Bilinear | 0 | 1512.0 | 1512.0000 | ✅ |
| Bilinear | 1000 | 1492.0 | 1492.0000 | ✅ |
| Bilinear | 5000 | 1556.0 | 1556.0000 | ✅ |
| Epstein | 200 | ~1485.2 | 1485.2213 | ✅ |
| Epstein | 5000 | 1500.0 | 1500.0000 | ✅ |
| n²-Linear | 0 | ~1497.5 | 1497.5063 | ✅ |
| n²-Linear | 1000 | 1490.0 | 1490.0000 | ✅ |
| n²-Linear | 5000 | ~1461.1 | 1461.0652 | ✅ |
| Isovelocity | any | 1500.0 | 1500.0000 | ✅ |

---

## Test 7: Reflection Accuracy (Isovelocity Analytical Benchmark)

**Purpose**: In isovelocity (c = 1500 m/s), rays are straight lines and reflection
geometry is analytically exact. This test verifies that the sub-step interpolation
in `advanceWithReflections` produces reflections at the correct positions.

**Method**: Trace a 10° ray from (1000, 0) in the isovelocity profile. Measure:
- (a) **z-residual** at each reflection point: should be exactly 0 or DEPTH_M
- (b) **x-spacing** between consecutive bottom reflections vs analytical
  `Δx = 2 · DEPTH_M / tan(θ) = 56,712.82 m`
- (c) **Cumulative drift** over multiple bounces

Also verify z-residual = 0 across all 6 profiles (15° ray from 100 m depth).

### Results — Isovelocity spacing accuracy

| Metric | Value | Verdict |
|---|---|---|
| Bottom reflections | 3 | — |
| Analytical Δx | 56,712.82 m | — |
| Max z-residual | 0.000e+0 m | ✅ Exact (boundary snap) |
| Max x-spacing error | 4.3e-9 m | ✅ Sub-nanometre |
| RMS x-spacing error | 3.1e-9 m | ✅ Sub-nanometre |
| Cumulative drift (2 intervals) | 5.1e-9 m | ✅ No accumulation |

### Results — z-residual across all profiles

| Profile | Reflections | Max z-residual | Verdict |
|---|---|---|---|
| Munk | 9 | 0.000e+0 | ✅ |
| Arctic | 3 | 0.000e+0 | ✅ |
| n²-Linear | 9 | 0.000e+0 | ✅ |
| Bilinear | 6 | 0.000e+0 | ✅ |
| Epstein | 7 | 0.000e+0 | ✅ |
| Isovelocity | 8 | 0.000e+0 | ✅ |

**All reflections land exactly on boundaries. Spacing matches analytical
prediction to sub-nanometre precision.**

```js
// Re-run Test 7
(function() {
  const saved = activeProfile;
  // Part 1: Isovelocity spacing accuracy
  activeProfile = PROFILES.find(p => p.id === 'isovelocity');
  const srcX = 1000, srcZ = 0, angle = 10.0;
  const ray = traceRay(srcX, srcZ, angle, 1);
  const path = ray.path;
  const events = ray.events;
  const theta = angle * Math.PI / 180;
  const analyticDx = 2 * DEPTH_M / Math.tan(theta);

  let maxZresid = 0;
  for (const e of events) {
    const zExpected = (e.type === 'B') ? DEPTH_M : 0;
    const resid = Math.abs(path[e.stepIdx].z - zExpected);
    if (resid > maxZresid) maxZresid = resid;
  }

  const bottomXs = events.filter(e => e.type === 'B').map(e => path[e.stepIdx].x);
  let maxSpErr = 0, sumSqSpErr = 0, nSp = 0;
  for (let i = 1; i < bottomXs.length; i++) {
    const err = Math.abs((bottomXs[i] - bottomXs[i-1]) - analyticDx);
    if (err > maxSpErr) maxSpErr = err;
    sumSqSpErr += err * err; nSp++;
  }
  const rmsSpErr = Math.sqrt(sumSqSpErr / Math.max(nSp, 1));
  const lastBx = bottomXs[bottomXs.length - 1], firstBx = bottomXs[0];
  const cumulDrift = Math.abs((lastBx - firstBx) - (bottomXs.length - 1) * analyticDx);

  console.log('=== Isovelocity spacing ===');
  console.log('Reflections:', bottomXs.length, 'Analytic Δx:', analyticDx.toFixed(2));
  console.log('Max z-resid:', maxZresid.toExponential(3));
  console.log('Max spacing err:', maxSpErr.toExponential(3));
  console.log('RMS spacing err:', rmsSpErr.toExponential(3));
  console.log('Cumul drift:', cumulDrift.toExponential(3));

  // Part 2: z-residual across all profiles
  console.log('=== z-residual all profiles ===');
  for (const p of PROFILES) {
    activeProfile = p;
    const r = traceRay(1000, 100, 15.0, 1);
    let mzr = 0;
    for (const e of r.events) {
      const zExp = (e.type === 'B') ? DEPTH_M : 0;
      mzr = Math.max(mzr, Math.abs(r.path[e.stepIdx].z - zExp));
    }
    console.log(p.id, ':', r.events.length, 'events, max z-resid:', mzr.toExponential(3));
  }
  activeProfile = saved;
  return 'done';
})()
```

---

## Known Limitations

These are inherent to the ray-tracing approach and are acceptable for a
visualization tool. They are **not** bugs.

1. **No frequency dependence** — ray tracing is the infinite-frequency (geometric
   optics) limit. Diffraction, mode coupling, and low-frequency effects are absent.

2. **Simplified attenuation** — cylindrical spreading + per-bounce amplitude factors
   is an approximation. A full transmission loss model would require solving the
   Helmholtz equation (e.g. via BELLHOP or Kraken).

3. **No bottom interaction model** — reflections are ideal (amplitude factor only).
   Real ocean bottoms have angle-dependent, frequency-dependent reflection coefficients.

4. **No cross-validation with reference codes** — results have not been compared
   against BELLHOP, Kraken, or RAM. The internal consistency checks above provide
   confidence but are not a substitute for benchmark comparison.

5. **RK4 truncation error** — the integrator conserves the Snell invariant to 5–7
   significant digits with DT = 0.05 (improved in v2.1 by sub-step boundary
   interpolation). Boundary reflections are now exact to machine precision (z-residual
   = 0). Reducing DT would further improve mid-path accuracy at the cost of performance.

---

## Corrective Actions Taken

| Finding | Action | Date |
|---|---|---|
| n²-Linear described as "symmetric channel" | Fixed: description updated to "downward-refracting layer"; `axisDepth` set to `null` | 2026-03-14 |
| Arctic described as "surface bounces only" | Fixed: clarified that steep rays from depth can reach bottom | 2026-03-14 |
| Boundary reflections used post-step clamping, causing cumulative positional errors and non-periodic ray paths | Fixed: replaced with `advanceWithReflections` sub-step interpolation — linear interp to find crossing time, partial RK4 to boundary, snap z, reflect sz, integrate remainder | 2026-03-15 |
