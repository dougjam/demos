// ISO 9613-1:1993 atmospheric absorption coefficient.
// Self-contained JS port of python/iso9613_reference.py.
// (c) 2026 Doug James, Stanford University. BSD-2-Clause.

(function () {
  const T0    = 293.15;      // reference temperature (K)
  const T01   = 273.16;      // triple point of water (K)
  const P_REF = 101.325;     // reference pressure (kPa)

  function psatOverPref(T) {
    return Math.pow(10, -6.8346 * Math.pow(T01 / T, 1.261) + 4.6151);
  }

  function molarH(T, RH, p) {
    return RH * psatOverPref(T) / (p / P_REF);
  }

  function relaxationFrequencies(T, RH, p) {
    const h = molarH(T, RH, p);
    const pr = p / P_REF;
    const fRO = pr * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
    const fRN = pr * Math.pow(T / T0, -0.5)
      * (9 + 280 * h * Math.exp(-4.170 * (Math.pow(T / T0, -1 / 3) - 1)));
    return { fRO, fRN };
  }

  // Absorption coefficient in dB/m.
  function alphaDbPerM(f, T_C, RH_pct, p_kPa) {
    const T = T_C + 273.15;
    const { fRO, fRN } = relaxationFrequencies(T, RH_pct, p_kPa);
    const classical = 1.84e-11 * (P_REF / p_kPa) * Math.sqrt(T / T0);
    const A_O = 0.01275 * Math.exp(-2239.1 / T) * fRO / (fRO * fRO + f * f);
    const A_N = 0.1068  * Math.exp(-3352.0 / T) * fRN / (fRN * fRN + f * f);
    return 8.686 * f * f * (classical + Math.pow(T / T0, -2.5) * (A_O + A_N));
  }

  function alphaDbPerKm(f, T_C, RH_pct, p_kPa) {
    return 1000 * alphaDbPerM(f, T_C, RH_pct, p_kPa);
  }

  window.ISO9613 = {
    T0, T01, P_REF,
    psatOverPref, molarH, relaxationFrequencies,
    alphaDbPerM, alphaDbPerKm,
  };
})();
