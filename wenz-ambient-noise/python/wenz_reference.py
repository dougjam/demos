"""
Wenz ambient ocean noise: Coates/Stojanovic component PSDs and total.

Self-contained Python reference for the JavaScript port in
../index.html. Verified against the SPEC section 5.1 reference table.

Formulas are the widely cited closed-form fits to Wenz 1962 Figure 13
compiled in Coates, R. F. W. (1989), Underwater Acoustic Systems, and
republished in Stojanovic, M. (2007), "On the relationship between
capacity and distance in an underwater acoustic channel", ACM
SIGMOBILE Mobile Computing and Communications Review 11(4).

All NL values are in dB re 1 uPa^2/Hz; frequency arguments are in kHz.
"""

from math import log10, sqrt


def nl_turb(f_khz):
    """Turbulent (pseudo-sound) pressure spectrum level, dB re 1 uPa^2/Hz."""
    return 17.0 - 30.0 * log10(f_khz)


def nl_ship(f_khz, s):
    """Distant shipping spectrum level. s in [0, 1]: 0 low, 1 heavy."""
    return (40.0 + 20.0 * (s - 0.5)
            + 26.0 * log10(f_khz)
            - 60.0 * log10(f_khz + 0.03))


def nl_wind(f_khz, U_mps):
    """Wind/surface noise (Coates/Stojanovic). U in m/s."""
    U = max(U_mps, 0.0)
    return (50.0 + 7.5 * sqrt(U)
            + 20.0 * log10(f_khz)
            - 40.0 * log10(f_khz + 0.4))


def nl_thermal(f_khz):
    """Mellen 1952 thermal-noise floor, dB re 1 uPa^2/Hz."""
    return -15.0 + 20.0 * log10(f_khz)


def components(f_hz, U_mps, s, mutes=None):
    """Return dict of NL_k for each component at frequency f_hz.

    mutes is an optional set/iterable of component names to drop:
    {"turb", "ship", "wind", "thermal"}.
    """
    mutes = set(mutes or ())
    f_khz = f_hz / 1000.0
    out = {}
    if "turb" not in mutes:
        out["turb"] = nl_turb(f_khz)
    if "ship" not in mutes:
        out["ship"] = nl_ship(f_khz, s)
    if "wind" not in mutes:
        out["wind"] = nl_wind(f_khz, U_mps)
    if "thermal" not in mutes:
        out["thermal"] = nl_thermal(f_khz)
    return out


def nl_total(f_hz, U_mps, s, mutes=None):
    """Total NL = 10 log10(sum_k 10^(NL_k/10))."""
    parts = components(f_hz, U_mps, s, mutes=mutes)
    lin = sum(10.0 ** (nl / 10.0) for nl in parts.values())
    return 10.0 * log10(lin) if lin > 0 else float("-inf")


def dominant_component(f_hz, U_mps, s, mutes=None):
    """Name of the component with the largest linear PSD at f_hz."""
    parts = components(f_hz, U_mps, s, mutes=mutes)
    return max(parts, key=lambda k: parts[k])


# SPEC section 5.1 reference table. Computed directly from the formulas
# above; the JavaScript port must reproduce these to within 0.5 dB.
TESTS = [
    #  U     s     f (Hz)  expected NL  dominant
    (0.5,  0.0,       30,   66.3, "ship"),
    (0.5,  0.0,      100,   58.0, "ship"),
    (0.5,  0.0,     1000,   49.5, "wind"),
    (0.5,  0.0,    10000,   34.6, "wind"),
    (0.5,  0.0,   100000,   25.4, "thermal"),
    (5.0,  0.5,       30,   74.1, "ship"),
    (5.0,  0.5,      100,   67.8, "ship"),
    (5.0,  0.5,     1000,   61.0, "wind"),
    (5.0,  0.5,    10000,   46.1, "wind"),
    (12.0, 0.0,       30,   67.2, "ship"),
    (12.0, 0.0,     1000,   70.1, "wind"),
    (12.0, 1.0,       30,   83.8, "ship"),
    (12.0, 1.0,     1000,   70.2, "wind"),
    (1.0,  1.0,       30,   83.8, "ship"),
    (1.0,  1.0,     1000,   53.6, "wind"),
]


def run_tests(tol_db=0.5):
    header = (f"{'U(m/s)':>6} {'s':>4} {'f(Hz)':>7}  "
              f"{'expect':>7} {'got':>7}  {'err':>5}  "
              f"{'dom':>7} {'got':>7}  status")
    print(header)
    print("-" * len(header))
    ok = True
    for U, s, f_hz, expected, dom in TESTS:
        got = nl_total(f_hz, U, s)
        got_dom = dominant_component(f_hz, U, s)
        err = abs(got - expected)
        dom_ok = (got_dom == dom)
        pass_ = err <= tol_db and dom_ok
        if not pass_:
            ok = False
        print(f"{U:>6.1f} {s:>4.1f} {f_hz:>7d}  "
              f"{expected:>7.2f} {got:>7.2f}  {err:>5.2f}  "
              f"{dom:>7} {got_dom:>7}  "
              f"{'OK' if pass_ else 'FAIL'}")
    print()

    # Section 5.5 sanity: muted shipping at U=5 drops the 30 Hz total
    f0 = 30
    full = nl_total(f0, 5.0, 0.0)
    no_ship = nl_total(f0, 5.0, 0.0, mutes=["ship"])
    drop = full - no_ship
    print(f"At U=5 m/s, 30 Hz: s=0 total = {full:.2f} dB, "
          f"shipping-muted = {no_ship:.2f} dB, drop = {drop:.2f} dB.")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
