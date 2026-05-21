# Attribution

## Audio

All audio in this demo is **generated live in the browser** with the
Web Audio API. There are no sample files.

Each component is independent Gaussian-white noise (one stream per
mechanism, so the four are uncorrelated, as they are physically),
shaped by a per-component linear-phase FIR filter whose magnitude
response matches the corresponding Wenz spectrum component. The four
filtered streams are summed and fed to a master gain.

## Mathematical model

The four component spectra are the closed-form Coates 1989 /
Stojanovic 2007 parameterization of Wenz 1962 Figure 13, plus the
Mellen 1952 thermal-noise floor. The optional "Hildebrand 2021" wind
mode is a coarse Beaufort-interpolated additive offset to the Coates
wind shape, capturing the qualitative fact that modern measurements
(Hildebrand, Frasier, Baumann-Pickering, Wiggins 2021) give somewhat
higher levels than the original Wenz figure at low Beaufort numbers.
The faithful, per-Beaufort, piecewise-frequency Hildebrand model is in
the companion MATLAB code at
<https://github.com/jahildebrand/WindNoise>.

## Depth-listening absorption

The "Depth" slider attenuates surface-generated components by the
Ainslie-McColm 1998 simplified seawater absorption coefficient
*alpha*(*f*) evaluated at fixed mid-latitude open-ocean defaults
(T = 10 C, S = 35 psu, pH = 8.0). Per mechanism:

- **Wind / surface**: loss = *alpha*(*f*) * 1.4 * *d*. The 1.4
  approximates the Cron-Sherman 1962 dipole-cosine cone integral over
  the surface noise distribution.
- **Distant shipping**: loss = *alpha*(*f*) * *d* (direct vertical
  path). Optimistic by 3-10 dB near the SOFAR axis (~1000 m), where
  real shipping noise is boosted by sound-channel trapping. The demo
  does not model the SOFAR channel.
- **Turbulent pressure** and **thermal noise**: depth-independent
  (both are local to the hydrophone).

At depth = 0 the corrections are zero and the listener spectrum equals
the surface Wenz spectrum exactly.

Primary references (see also `python/README.md` and the page footer):

1. Wenz, G. M. (1962). "Acoustic ambient noise in the ocean: spectra
   and sources." *J. Acoust. Soc. Am.* 34(12), 1936-1956.
   doi:10.1121/1.1909155
2. Mellen, R. H. (1952). "The thermal-noise limit in the detection of
   underwater acoustic signals." *J. Acoust. Soc. Am.* 24(5), 478-480.
   doi:10.1121/1.1906924
3. Coates, R. F. W. (1989). *Underwater Acoustic Systems*. Macmillan
   New Electronics Series.
4. Stojanovic, M. (2007). "On the relationship between capacity and
   distance in an underwater acoustic communication channel." *ACM
   SIGMOBILE Mobile Computing and Communications Review* 11(4), 34-43.
   doi:10.1145/1347364.1347373
5. Hildebrand, J. A., K. E. Frasier, S. Baumann-Pickering, S. M.
   Wiggins (2021). "An empirical model for wind-generated ocean
   noise." *J. Acoust. Soc. Am.* 149(6), 4516-4533.
   doi:10.1121/10.0005430
6. Ainslie, M. A., J. G. McColm (1998). "A simplified formula for
   viscous and chemical absorption in sea water." *J. Acoust. Soc.
   Am.* 103(3), 1671-1672. doi:10.1121/1.421258
7. Cron, B. F., C. H. Sherman (1962). "Spatial-correlation functions
   for various noise models." *J. Acoust. Soc. Am.* 34(11), 1732-1736.
   doi:10.1121/1.1909110

## Code

The demo code and Python reference are (c) 2026 Doug James, Stanford
University, BSD-2-Clause.

The Chart.js library is loaded from a CDN under MIT license.
