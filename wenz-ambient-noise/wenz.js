// (c) 2026 Doug James, Stanford University. BSD-2-Clause.
//
// Wenz ambient ocean noise: Coates 1989 / Stojanovic 2007 closed-form
// component PSDs (turbulent, distant shipping, wind/surface, Mellen
// thermal), plus a coarse Hildebrand-2021-flavored wind alternative.
//
// All NL_* are in dB re 1 uPa^2/Hz. Frequency arguments are in kHz
// (matching the formulas as published).

(function (global) {
  const log10 = (x) => Math.log10(x);

  function nlTurb(fKhz) {
    return 17 - 30 * log10(fKhz);
  }

  function nlShip(fKhz, s) {
    return 40 + 20 * (s - 0.5)
         + 26 * log10(fKhz)
         - 60 * log10(fKhz + 0.03);
  }

  function nlWindCoates(fKhz, U) {
    const u = Math.max(U, 0);
    return 50 + 7.5 * Math.sqrt(u)
         + 20 * log10(fKhz)
         - 40 * log10(fKhz + 0.4);
  }

  // Hildebrand 2021 (simplified). The published model (Hildebrand,
  // Frasier, Baumann-Pickering, Wiggins 2021, JASA 149) is a
  // per-Beaufort, piecewise-frequency parameterization fit to a decade
  // of multi-site ocean recordings; the full version is implemented in
  // the companion MATLAB code at https://github.com/jahildebrand/WindNoise.
  // Here we keep the Coates frequency shape but apply a per-Beaufort
  // additive offset and exponent that reproduce two salient features:
  //   - low-Beaufort levels are ~5-8 dB higher than Coates/Wenz, in
  //     line with modern measurements (Hildebrand 2021 Fig. 5);
  //   - high-Beaufort levels and the basic spectral shape match.
  // Offsets are linearly interpolated in Beaufort (so the curve
  // changes smoothly with the wind slider).
  const BEAUFORT_MIDS_MPS = [0.3, 1.0, 2.6, 4.4, 7.0, 9.8, 12.6, 15.7,
                             19.0, 22.6, 26.5, 30.6];
  const HILD_OFFSET_DB = [8, 8, 7, 6, 5, 3, 2, 1, 0.5, 0, 0, 0];

  function _beaufortInterp(U, table) {
    const u = Math.max(U, BEAUFORT_MIDS_MPS[0]);
    for (let i = 0; i < BEAUFORT_MIDS_MPS.length - 1; i++) {
      const lo = BEAUFORT_MIDS_MPS[i];
      const hi = BEAUFORT_MIDS_MPS[i + 1];
      if (u <= hi) {
        const t = (u - lo) / (hi - lo);
        return table[i] * (1 - t) + table[i + 1] * t;
      }
    }
    return table[table.length - 1];
  }

  function nlWindHildebrand(fKhz, U) {
    const offset = _beaufortInterp(U, HILD_OFFSET_DB);
    return nlWindCoates(fKhz, U) + offset;
  }

  function nlThermal(fKhz) {
    return -15 + 20 * log10(fKhz);
  }

  // Ainslie-McColm 1998 simplified seawater absorption coefficient,
  // in dB/km. Three terms: boric-acid relaxation (~1 kHz, dominant
  // 100 Hz - 5 kHz), magnesium-sulfate relaxation (~100 kHz, dominant
  // 10-100 kHz), pure-water viscosity (dominant above 100 kHz).
  //   Ainslie, M. A., J. G. McColm (1998). "A simplified formula for
  //   viscous and chemical absorption in sea water." J. Acoust. Soc.
  //   Am. 103(3), 1671-1672. doi:10.1121/1.421258
  // Defaults are typical mid-latitude open-ocean values; depth_km is
  // the depth at which the absorption is evaluated (the depth term in
  // MgSO4 / pure-water reflects compressibility / cavity-volume
  // effects, NOT path length).
  function absorptionDbPerKm(fKhz, T_C, S_psu, pH, depth_km) {
    if (T_C === undefined)     T_C = 10;
    if (S_psu === undefined)   S_psu = 35;
    if (pH === undefined)      pH = 8.0;
    if (depth_km === undefined) depth_km = 0;
    const f2 = fKhz * fKhz;
    const f1B = 0.78 * Math.sqrt(S_psu / 35) * Math.exp(T_C / 26);
    const f1M = 42 * Math.exp(T_C / 17);
    const aB = 0.106 * (f1B * f2) / (f2 + f1B * f1B)
                 * Math.exp((pH - 8) / 0.56);
    const aM = 0.52 * (1 + T_C / 43) * (S_psu / 35)
                 * (f1M * f2) / (f2 + f1M * f1M)
                 * Math.exp(-depth_km / 6);
    const aW = 0.00049 * f2 * Math.exp(-(T_C / 27 + depth_km / 17));
    return aB + aM + aW;
  }

  // mutes: optional Set/Array of {"turb","ship","wind","thermal"}.
  // mode: "coates" (default) or "hildebrand" for wind.
  function components(fHz, U, s, mutes, mode) {
    const muteSet = mutes instanceof Set ? mutes : new Set(mutes || []);
    const fKhz = fHz / 1000;
    const nlWind = (mode === "hildebrand") ? nlWindHildebrand : nlWindCoates;
    const out = {};
    if (!muteSet.has("turb"))    out.turb    = nlTurb(fKhz);
    if (!muteSet.has("ship"))    out.ship    = nlShip(fKhz, s);
    if (!muteSet.has("wind"))    out.wind    = nlWind(fKhz, U);
    if (!muteSet.has("thermal")) out.thermal = nlThermal(fKhz);
    return out;
  }

  function nlTotal(fHz, U, s, mutes, mode) {
    const parts = components(fHz, U, s, mutes, mode);
    let lin = 0;
    for (const k in parts) lin += Math.pow(10, parts[k] / 10);
    return lin > 0 ? 10 * log10(lin) : -Infinity;
  }

  function dominantComponent(fHz, U, s, mutes, mode) {
    const parts = components(fHz, U, s, mutes, mode);
    let best = null, bestLin = -Infinity;
    for (const k in parts) {
      const lin = Math.pow(10, parts[k] / 10);
      if (lin > bestLin) { bestLin = lin; best = k; }
    }
    return best;
  }

  // WMO Beaufort midpoints in m/s and short labels for slider readout.
  const BEAUFORT_TABLE = [
    [0,   0.5,  "calm"],
    [1,   1.5,  "light air"],
    [2,   3.1,  "light breeze"],
    [3,   5.1,  "gentle breeze"],
    [4,   8.2,  "moderate breeze"],
    [5,  10.8,  "fresh breeze"],
    [6,  13.9,  "strong breeze"],
    [7,  17.0,  "near gale"],
    [8,  20.6,  "gale"],
    [9,  24.2,  "strong gale"],
    [10, 28.3,  "storm"],
    [11, 32.4,  "violent storm"],
    [12, Infinity, "hurricane"],
  ];

  function beaufort(U) {
    for (const [B, upper, label] of BEAUFORT_TABLE) {
      if (U < upper) return { number: B, label };
    }
    return { number: 12, label: "hurricane" };
  }

  global.Wenz = {
    nlTurb, nlShip, nlWindCoates, nlWindHildebrand, nlThermal,
    components, nlTotal, dominantComponent,
    absorptionDbPerKm,
    beaufort, BEAUFORT_TABLE,
  };
})(typeof window !== "undefined" ? window : globalThis);
