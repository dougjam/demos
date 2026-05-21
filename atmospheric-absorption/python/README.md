# ISO 9613-1 reference

Self-contained Python implementation of the atmospheric absorption
coefficient from ISO 9613-1:1993 / Bass et al. 1995. Used as the ground
truth for the JavaScript port in [`../iso9613.js`](../iso9613.js); the
browser-based test harness [`../test.html`](../test.html) asserts the JS
port agrees with the table below to within 1% relative error.

## Run

```
python iso9613_reference.py
```

Prints the SPEC §5.1 test table and the two SPEC §5.2 relaxation
frequencies. Exits 0 if every row is within 1% of the expected value.

## Expected output

```
T(C) RH(%)  p(kPa)   f(Hz)    expect      got    err%  note
  20    50 101.325     100    0.294   0.294   ...   OK  default
  20    50 101.325    1000    4.660   4.665   ...   OK  default
  20    50 101.325   10000    158.8   158.8   ...   OK  default
  20    50 101.325  100000   3280.0  3280.4   ...   OK  default (3.28 dB/m)
  20    70 101.325    1000    4.980   4.978   ...   OK  humid
  20    70 101.325   10000    117.5   117.5   ...   OK  humid
   0    50 101.325    1000    6.830   6.827   ...   OK  cold
   0    50 101.325   10000    172.1   172.1   ...   OK  cold
  30    80 101.325    1000    7.410   7.405   ...   OK  hot/humid
  30    80 101.325   10000   80.700  80.678   ...   OK  hot/humid
  20    50  80.000    1000    4.620   4.619   ...   OK  reduced pressure
  20     5 101.325    1000   26.510  26.511   ...   OK  very dry
  20     5 101.325    4000   64.480  64.480   ...   OK  very dry, mid f
 -20    50 101.325    1000    9.140   9.140   ...   OK  very cold

T=20C RH=50% p=101.325kPa: f_rO=35413.9 Hz, f_rN= 331.9 Hz
T=20C RH=70% p=101.325kPa: f_rO=53174.0 Hz, f_rN= 461.0 Hz
```

## API

```python
from iso9613_reference import alpha_dB_per_m, alpha_dB_per_km, relaxation_frequencies

alpha = alpha_dB_per_km(f_Hz, T_C, RH_pct, p_kPa)
f_rO, f_rN = relaxation_frequencies(T_K, RH_pct, p_kPa)
```

## References

1. ISO 9613-1:1993 *Acoustics: Attenuation of sound during propagation
   outdoors, Part 1*.
2. Bass, Sutherland, Zuckerwar, Blackstock, Hester (1995). "Atmospheric
   absorption of sound: Further developments." *JASA* 97(1), 680-683.
   doi:10.1121/1.412989

License: BSD-2-Clause.
