# ISO 9613-1 Python reference + audio-filter

Three scripts:

| Script                       | Purpose                                                              | Dependencies          |
|------------------------------|----------------------------------------------------------------------|-----------------------|
| `iso9613_reference.py`       | Pure-Python implementation of the ISO 9613-1 absorption coefficient. Ground truth for `../iso9613.js`. Runs the SPEC §5.1 test table on import. | stdlib only           |
| `verify.py`                  | Deeper verification: independently-factored re-implementation, optional `python-acoustics` cross-check, sanity checks, peak-RH check, full SPEC §5.1 table. | stdlib (+ optional `python-acoustics`) |
| `apply_absorption.py`        | Student-facing utility: read a WAV, apply the frequency-domain absorption filter at chosen (T, RH, p, r), write a WAV. | `numpy`, `scipy`      |

License throughout: BSD-2-Clause.

---

## 1. `iso9613_reference.py` — the reference

```
python iso9613_reference.py
```

Prints the SPEC §5.1 test table and the two SPEC §5.2 relaxation
frequencies. Exits 0 if every row is within 1% of the expected value.

### Expected output

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

### API

```python
from iso9613_reference import (
    alpha_dB_per_m,
    alpha_dB_per_km,
    relaxation_frequencies,
)

alpha = alpha_dB_per_km(f_Hz, T_C, RH_pct, p_kPa)   # scalar
f_rO, f_rN = relaxation_frequencies(T_K, RH_pct, p_kPa)
```

---

## 2. `verify.py` — extra verification

```
python verify.py
```

Runs eight check groups; exit code is 0 iff all pass:

1. Independent re-implementation agreement (>2500 (T, RH, p, f) points within 1e-12 relative).
2. `python-acoustics` cross-check (optional; skips cleanly if not installed).
3. *α*(*f* = 0) = 0.
4. Total attenuation is linear in *r*.
5. *h* (water-vapour molar concentration) is in **percent**, not fraction.
6. Relaxation frequencies match SPEC §5.2 reference values to within tolerance.
7. Peak humidity for fixed *f* matches SPEC §5.1 (~4% at 1 kHz, ~10% at 4 kHz, ~19% at 10 kHz).
8. All 14 SPEC §5.1 rows within 1%.

To enable the `python-acoustics` cross-check:

```
pip install python-acoustics
python verify.py
```

---

## 3. `apply_absorption.py` — filter a WAV file

```
python apply_absorption.py INPUT.wav OUTPUT.wav \
    [--temperature 20] [--humidity 50] [--pressure 101.325] \
    [--distance 1000] [--spread]
```

Examples:

```
# Defaults: 20 C, 50% RH, 1 atm, r = 1 km, no 1/r spreading.
python apply_absorption.py ../audio/voice.wav voice_far.wav

# Thunder heard from 5 km away in hot humid air:
python apply_absorption.py ../audio/thunder.wav thunder_far.wav \
    --temperature 30 --humidity 80 --distance 5000

# Include geometric 1/r spreading, dry desert air at 200 m:
python apply_absorption.py ../audio/music.wav music_dry.wav \
    --distance 200 --humidity 5 --spread
```

### How it works

1. Read the WAV, downmix to mono, convert to float in [−1, 1].
2. Real FFT of the (zero-padded) signal: `rfft(x, n_fft)`.
3. Evaluate the ISO 9613-1 absorption coefficient *α*(*f*) at every
   bin frequency (vectorized via numpy; cross-checked against the
   scalar reference at startup, so a typo in either implementation
   trips an assertion immediately).
4. Build the linear-amplitude transfer function
   *H*(*f*) = 10<sup>−*α*(*f*) · *r* / 20</sup>.
5. Multiply: `Y = X * H`. Inverse-FFT, trim back to the original length.
6. With `--spread`, multiply the time-domain output by 1/max(1, *r*).
7. Write a mono 16-bit PCM WAV (not auto-normalized — the dB
   attenuation is the whole point of the demo).

Because *H*(*f*) is real and positive everywhere, the filter is
**zero-phase**: only the magnitude shape of the spectrum changes;
absolute time-of-flight is not modelled. (For a causal physical
implementation, replace zero-phase with linear-phase via a Hilbert
factorization, or use a minimum-phase reconstruction. Out of scope
for the demo.)

### Sample console output

```
$ python apply_absorption.py ../audio/voice.wav voice_far.wav -r 2000
loaded voice.wav: 432000 samples @ 48000 Hz (9.00 s mono)
  f =    250 Hz:  alpha =    1.31 dB/km, A(f, r=2000 m) =    2.62 dB
  f =   1000 Hz:  alpha =    4.66 dB/km, A(f, r=2000 m) =    9.33 dB
  f =   4000 Hz:  alpha =   29.67 dB/km, A(f, r=2000 m) =   59.33 dB
  f =  10000 Hz:  alpha =  158.84 dB/km, A(f, r=2000 m) =  317.68 dB
wrote voice_far.wav
```

The 317 dB of attenuation at 10 kHz means the output at that frequency
is below the 16-bit quantization floor (~96 dB dynamic range) by some
220 dB. The output is effectively low-passed; this is the same effect
you hear when sweeping the distance slider in the browser demo.

---

## References

1. ISO 9613-1:1993 *Acoustics: Attenuation of sound during propagation
   outdoors, Part 1*.
2. Bass, Sutherland, Zuckerwar, Blackstock, Hester (1995). "Atmospheric
   absorption of sound: Further developments." *JASA* 97(1), 680-683.
   [doi:10.1121/1.412989](https://doi.org/10.1121/1.412989)
3. python-acoustics library, ISO 9613-1 module (BSD 3-Clause):
   <https://github.com/python-acoustics/python-acoustics>.
