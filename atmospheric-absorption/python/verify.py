"""
Verify python/iso9613_reference.py against:

 1. A second, independently-factored implementation of the same standard
    (so a typo in either copy would show up).
 2. The python-acoustics library's ISO 9613-1 module, if installed.
 3. Physical sanity checks (alpha(0) = 0, monotonicity, units of h, etc.).
 4. The SPEC §5.1 test table (already in iso9613_reference.run_tests).

Exits 0 if every check passes.

Run:
    python verify.py
"""
from __future__ import annotations
import math
import sys
from typing import Callable

from iso9613_reference import (
    alpha_dB_per_m,
    alpha_dB_per_km,
    relaxation_frequencies,
    psat_over_pref,
    molar_h,
    run_tests,
    T0, T01, P_REF,
)


# ---------------------------------------------------------------------------
# Independent re-implementation (different factoring of the same formulas)
# ---------------------------------------------------------------------------
def alpha_check(f: float, T_C: float, RH_pct: float, p_kPa: float) -> float:
    """Same physics as alpha_dB_per_m but algebraically refactored.

    Differences:
      - Uses (f_r + f*f/f_r)**(-1) form instead of f_r/(f_r**2 + f*f).
      - Computes T/T0 and p/p_ref once and reuses them.
      - Inlines molar_h instead of calling out.
    """
    T = T_C + 273.15
    tr = T / T0           # reduced temperature
    pr = p_kPa / P_REF    # reduced pressure
    # Saturation vapour pressure, fraction of p_ref:
    psat_pref = 10.0 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    # Molar concentration of water vapour, percent:
    h = RH_pct * psat_pref / pr
    # Relaxation frequencies (Hz):
    fO = pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fN = pr * tr ** -0.5 * (
        9.0 + 280.0 * h * math.exp(-4.170 * (tr ** (-1.0 / 3.0) - 1.0))
    )
    # Three absorption mechanisms, summed:
    classical = 1.84e-11 / pr * math.sqrt(tr)
    relax_O = 0.01275 * math.exp(-2239.1 / T) / (fO + f * f / fO)
    relax_N = 0.1068  * math.exp(-3352.0 / T) / (fN + f * f / fN)
    return 8.686 * f * f * (classical + tr ** -2.5 * (relax_O + relax_N))


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------
def grid(T_C_list, RH_list, p_list, f_list):
    for T_C in T_C_list:
        for RH in RH_list:
            for p in p_list:
                for f in f_list:
                    yield T_C, RH, p, f


def report(name: str, ok: bool, detail: str = "") -> bool:
    tag = "OK  " if ok else "FAIL"
    line = f"  [{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


# ---------------------------------------------------------------------------
# Verification suite
# ---------------------------------------------------------------------------
def check_independent_implementation() -> bool:
    """Reference and the independently-factored version must agree to within
    floating-point noise (1e-12 relative) over a wide grid."""
    print("\n[1] Independent re-implementation agreement (rel tol 1e-12):")
    Ts  = [-20, -10, 0, 10, 20, 30, 40, 50]
    RHs = [1, 5, 10, 20, 30, 50, 70, 90, 99]
    ps  = [50, 80, 101.325, 110]
    fs  = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
    worst = 0.0
    worst_at = None
    n = 0
    for T_C, RH, p, f in grid(Ts, RHs, ps, fs):
        a_ref = alpha_dB_per_m(f, T_C, RH, p)
        a_chk = alpha_check(f, T_C, RH, p)
        if a_ref == 0:
            rel = abs(a_chk)
        else:
            rel = abs(a_chk - a_ref) / abs(a_ref)
        if rel > worst:
            worst = rel
            worst_at = (T_C, RH, p, f)
        n += 1
    ok = worst < 1e-12
    return report(
        f"agreement over {n} (T, RH, p, f) points",
        ok,
        f"worst rel err {worst:.2e} at T_C, RH, p, f = {worst_at}",
    )


def check_python_acoustics() -> bool:
    """If python-acoustics is installed, cross-check against its ISO 9613-1
    implementation. Skip cleanly if not installed."""
    print("\n[2] python-acoustics cross-check:")
    try:
        from acoustics.standards.iso_9613_1_1993 import (
            attenuation_coefficient,
            relaxation_frequency_oxygen,
            relaxation_frequency_nitrogen,
            saturation_pressure,
            molar_concentration_water_vapour,
            REFERENCE_PRESSURE,
            REFERENCE_TEMPERATURE,
        )
    except ImportError:
        return report("python-acoustics not installed, skipped", True,
                      "install with `pip install python-acoustics` to enable")
    worst = 0.0; worst_at = None; n = 0
    Ts  = [-10, 0, 20, 40]
    RHs = [5, 20, 50, 80]
    ps  = [80, 101.325]
    fs  = [100, 1_000, 10_000, 50_000]
    for T_C, RH, p, f in grid(Ts, RHs, ps, fs):
        T = T_C + 273.15
        ps_sat = saturation_pressure(T)
        h = molar_concentration_water_vapour(RH, ps_sat, p)
        fO = relaxation_frequency_oxygen(p, h)
        fN = relaxation_frequency_nitrogen(p, T, h)
        a_pa = float(attenuation_coefficient(
            p, T, REFERENCE_PRESSURE, REFERENCE_TEMPERATURE, fN, fO, f
        ))
        a_ref = alpha_dB_per_m(f, T_C, RH, p)
        rel = abs(a_pa - a_ref) / max(abs(a_ref), 1e-30)
        if rel > worst:
            worst = rel; worst_at = (T_C, RH, p, f)
        n += 1
    ok = worst < 1e-9
    return report(
        f"agreement over {n} points (rel tol 1e-9)",
        ok,
        f"worst rel err {worst:.2e} at {worst_at}",
    )


def check_zero_at_dc() -> bool:
    """alpha(0) = 0 by construction (f^2 prefactor)."""
    print("\n[3] alpha(f=0) = 0:")
    a = alpha_dB_per_m(0.0, 20, 50, 101.325)
    return report("alpha(0, default) == 0", a == 0, f"got {a}")


def check_monotone_in_distance() -> bool:
    """For any (f, T, RH, p), total attenuation in dB scales linearly with r,
    so it's monotonically increasing in r whenever alpha(f) > 0."""
    print("\n[4] linearity in r:")
    f, T, RH, p = 4000, 20, 50, 101.325
    a = alpha_dB_per_m(f, T, RH, p)
    ok = True
    for r in [1, 10, 100, 1_000, 10_000]:
        expected = a * r
        # No call needed; the model is alpha * r by construction.
        # Just sanity-check the ratio with the dB-per-km wrapper.
        if r >= 1000:
            from_km = alpha_dB_per_km(f, T, RH, p) * (r / 1000)
            if abs(from_km - expected) / expected > 1e-12:
                ok = False
    return report("alpha_dB_per_km(f) * r/1000 == alpha_dB_per_m(f) * r", ok)


def check_units_of_h() -> bool:
    """h (molar water-vapour concentration) is in PERCENT, not fraction. At
    20 C, 100% RH, 1 atm, the saturation vapour pressure is roughly 2.34 kPa
    so h should be ~2.3 percent. If we'd returned a fraction, h would be
    around 0.023. SPEC §2.2 confirms h is a percentage."""
    print("\n[5] units of h (should be percent, ~2.3 at 20 C/100% RH/1 atm):")
    h = molar_h(20 + 273.15, 100, 101.325)
    ok = 2.0 < h < 2.6
    return report(f"molar_h ~= 2.3 percent", ok, f"got {h:.4f}")


def check_relaxation_freqs() -> bool:
    """Two SPEC §5.2 reference values."""
    print("\n[6] relaxation frequencies (SPEC §5.2):")
    cases = [
        ((20 + 273.15, 50, 101.325), (35413.9, 331.9), (50, 5)),
        ((20 + 273.15, 70, 101.325), (53174.0, 461.0), (50, 5)),
    ]
    ok_all = True
    for (T, RH, p), (eO, eN), (tolO, tolN) in cases:
        fO, fN = relaxation_frequencies(T, RH, p)
        ok = abs(fO - eO) <= tolO and abs(fN - eN) <= tolN
        ok_all &= report(
            f"T_K={T:.2f} RH={RH}% p={p} kPa  ->  fO~{eO:.1f} Hz  fN~{eN:.1f} Hz",
            ok,
            f"got fO={fO:.1f}, fN={fN:.1f}",
        )
    return ok_all


def check_peak_humidity() -> bool:
    """SPEC §5.1 last paragraph: at fixed f, alpha vs RH has a maximum at a
    moderate humidity (about 4% at 1 kHz, 10% at 4 kHz, 19% at 10 kHz)."""
    print("\n[7] peak humidity (SPEC §5.1):")
    targets = [(1000, 4), (4000, 10), (10000, 19)]
    ok_all = True
    for f, want_RH in targets:
        best_RH = 1; best_a = -math.inf
        RH = 1.0
        while RH <= 100.0:
            a = alpha_dB_per_m(f, 20, RH, 101.325)
            if a > best_a:
                best_a = a; best_RH = RH
            RH += 0.5
        ok = abs(best_RH - want_RH) <= 2.0
        ok_all &= report(
            f"f = {f} Hz: peak at RH ~ {want_RH}%",
            ok,
            f"got {best_RH:.1f}%",
        )
    return ok_all


def check_spec_table() -> bool:
    """All 14 SPEC §5.1 rows pass via the bundled run_tests()."""
    print("\n[8] SPEC §5.1 table (14 rows, tol 1%):")
    ok = run_tests(tol=0.01)
    print()  # run_tests prints its own table
    return report("14/14 rows within 1%", ok)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    print("Verifying iso9613_reference.py")
    checks: list[Callable[[], bool]] = [
        check_independent_implementation,
        check_python_acoustics,
        check_zero_at_dc,
        check_monotone_in_distance,
        check_units_of_h,
        check_relaxation_freqs,
        check_peak_humidity,
        check_spec_table,
    ]
    results = [c() for c in checks]
    n_fail = results.count(False)
    print(f"\n{'='*40}\n{len(results) - n_fail}/{len(results)} checks passed.\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
