"""
ISO 9613-1:1993 atmospheric absorption coefficient.

Self-contained Python reference for the JavaScript port in
../index.html. Verified against the SPEC §5.1 test table.

Formulas follow Bass, Sutherland, Zuckerwar, Blackstock, Hester (1995),
"Atmospheric absorption of sound: Further developments", JASA 97(1).
"""

from math import exp, log10, sqrt

T0 = 293.15      # reference temperature (K)
T01 = 273.16     # triple point of water (K)
P_REF = 101.325  # reference pressure (kPa)


def psat_over_pref(T):
    """Saturation vapour pressure / p_ref (Davis 1992)."""
    return 10.0 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)


def molar_h(T, RH, p):
    """Percent mole fraction of water vapour."""
    return RH * psat_over_pref(T) / (p / P_REF)


def relaxation_frequencies(T, RH, p):
    h = molar_h(T, RH, p)
    pr = p / P_REF
    f_rO = pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    f_rN = pr * (T / T0) ** -0.5 * (
        9.0 + 280.0 * h * exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0))
    )
    return f_rO, f_rN


def alpha_dB_per_m(f, T_C, RH_pct, p_kPa):
    """Absorption coefficient in dB/m."""
    T = T_C + 273.15
    f_rO, f_rN = relaxation_frequencies(T, RH_pct, p_kPa)
    classical = 1.84e-11 * (P_REF / p_kPa) * sqrt(T / T0)
    A_O = 0.01275 * exp(-2239.1 / T) * f_rO / (f_rO ** 2 + f ** 2)
    A_N = 0.1068 * exp(-3352.0 / T) * f_rN / (f_rN ** 2 + f ** 2)
    return 8.686 * f * f * (classical + (T / T0) ** -2.5 * (A_O + A_N))


def alpha_dB_per_km(f, T_C, RH_pct, p_kPa):
    return 1000.0 * alpha_dB_per_m(f, T_C, RH_pct, p_kPa)


# SPEC §5.1 reference table.
TESTS = [
    (20,   50, 101.325,    100,   0.294, "default"),
    (20,   50, 101.325,   1000,   4.66,  "default"),
    (20,   50, 101.325,  10000, 158.8,   "default"),
    (20,   50, 101.325, 100000, 3280.0,  "default (3.28 dB/m)"),
    (20,   70, 101.325,   1000,   4.98,  "humid"),
    (20,   70, 101.325,  10000, 117.5,   "humid"),
    ( 0,   50, 101.325,   1000,   6.83,  "cold"),
    ( 0,   50, 101.325,  10000, 172.1,   "cold"),
    (30,   80, 101.325,   1000,   7.41,  "hot/humid"),
    (30,   80, 101.325,  10000,  80.7,   "hot/humid"),
    (20,   50,  80.0,     1000,   4.62,  "reduced pressure"),
    (20,    5, 101.325,   1000,  26.51,  "very dry"),
    (20,    5, 101.325,   4000,  64.48,  "very dry, mid f"),
    (-20,  50, 101.325,   1000,   9.14,  "very cold"),
]


def _fmt(x):
    return f"{x:7.3f}" if x < 100 else f"{x:7.1f}"


def run_tests(tol=0.01):
    print(f"{'T(C)':>4} {'RH(%)':>5} {'p(kPa)':>7} {'f(Hz)':>7}  "
          f"{'expect':>8} {'got':>8}  {'err%':>6}  note")
    ok = True
    for T_C, RH, p, f, expected, note in TESTS:
        got = alpha_dB_per_km(f, T_C, RH, p)
        rel = abs(got - expected) / expected
        flag = "OK" if rel < tol else "FAIL"
        if rel >= tol:
            ok = False
        print(f"{T_C:>4} {RH:>5} {p:>7.3f} {f:>7}  "
              f"{_fmt(expected)} {_fmt(got)}  {100*rel:>5.2f}%  {flag}  {note}")
    print()
    for T_C, RH, p in [(20, 50, 101.325), (20, 70, 101.325)]:
        f_rO, f_rN = relaxation_frequencies(T_C + 273.15, RH, p)
        print(f"T={T_C}C RH={RH}% p={p}kPa: f_rO={f_rO:7.1f} Hz, f_rN={f_rN:6.1f} Hz")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
