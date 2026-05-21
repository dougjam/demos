# Wenz ambient noise reference

Self-contained Python implementation of the Coates 1989 / Stojanovic
2007 closed-form parameterization of the Wenz 1962 ocean ambient noise
components: turbulent (pseudo-sound), distant shipping, wind/surface,
and Mellen 1952 thermal. Used as the ground truth for the JavaScript
port in `../index.html`.

## Run

```
python wenz_reference.py
```

Prints the SPEC section 5.1 reference table. Exits 0 if every row is
within 0.5 dB of the expected value and the dominant component matches.

## Expected output

```
U(m/s)    s   f(Hz)   expect     got    err      dom     got  status
--------------------------------------------------------------------
   0.5  0.0      30    66.30   66.25   0.05     ship    ship  OK
   0.5  0.0     100    58.00   57.96   0.04     ship    ship  OK
   0.5  0.0    1000    49.50   49.50   0.00     wind    wind  OK
   0.5  0.0   10000    34.60   34.63   0.03     wind    wind  OK
   0.5  0.0  100000    25.40   25.44   0.04  thermal thermal  OK
   5.0  0.5      30    74.10   74.07   0.03     ship    ship  OK
   5.0  0.5     100    67.80   67.79   0.01     ship    ship  OK
   5.0  0.5    1000    61.00   60.95   0.05     wind    wind  OK
   5.0  0.5   10000    46.10   46.09   0.01     wind    wind  OK
  12.0  0.0      30    67.20   67.20   0.00     ship    ship  OK
  12.0  0.0    1000    70.10   70.14   0.04     wind    wind  OK
  12.0  1.0      30    83.80   83.77   0.03     ship    ship  OK
  12.0  1.0    1000    70.20   70.17   0.03     wind    wind  OK
   1.0  1.0      30    83.80   83.75   0.05     ship    ship  OK
   1.0  1.0    1000    53.60   53.62   0.02     wind    wind  OK

At U=5 m/s, 30 Hz: s=0 total = 66.37 dB, shipping-muted = 62.97 dB, drop = 3.40 dB.
```

## API

```python
from wenz_reference import (
    nl_turb, nl_ship, nl_wind, nl_thermal,
    nl_total, components, dominant_component,
)

nl = nl_total(f_hz=1000, U_mps=5.0, s=0.3)
parts = components(f_hz=1000, U_mps=5.0, s=0.3,
                   mutes=["thermal"])    # drop a component
dom = dominant_component(f_hz=30, U_mps=5.0, s=1.0)
```

## References

1. **Wenz, G. M.** (1962). "Acoustic ambient noise in the ocean: spectra
   and sources." *J. Acoust. Soc. Am.* 34(12), 1936-1956.
   [doi:10.1121/1.1909155](https://doi.org/10.1121/1.1909155)
2. **Mellen, R. H.** (1952). "The thermal-noise limit in the detection
   of underwater acoustic signals." *J. Acoust. Soc. Am.* 24(5),
   478-480. [doi:10.1121/1.1906924](https://doi.org/10.1121/1.1906924)
3. **Coates, R. F. W.** (1989). *Underwater Acoustic Systems*.
   Macmillan New Electronics Series.
4. **Stojanovic, M.** (2007). "On the relationship between capacity and
   distance in an underwater acoustic channel." *ACM SIGMOBILE Mobile
   Computing and Communications Review* 11(4), 34-43.

License: BSD-2-Clause.
