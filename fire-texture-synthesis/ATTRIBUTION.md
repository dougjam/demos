# Attribution

## Original work

The sound texture synthesis algorithm implemented here is by:

- **Jeffrey N. Chadwick** (Cornell University, at the time of publication)
- **Doug L. James** (Cornell University, at the time of publication; now Stanford University)

Published in:

> Jeffrey N. Chadwick and Doug L. James, "Animating Fire with Sound,"
> ACM Transactions on Graphics (SIGGRAPH 2011), 30(4), August 2011.
> Section 5 ("Synchronized Sound Texture Synthesis").

Project page (canonical reference): https://www.cs.cornell.edu/projects/Sound/fire/

Paper PDF: https://www.cs.cornell.edu/projects/Sound/fire/FireSound2011.pdf

## Reference implementation

The Python and JavaScript code in this repository is a faithful port of
the C++ reference implementation released by the authors at the URL
above (`sound_texture_synthesis.zip` / `sound_texture_synthesis.tar.gz`).
All algorithm design choices, parameter defaults, and numerical details
follow that implementation. Differences from the released C++ are
documented in `docs/ALGORITHM.md`.

The original C++ release used the
[ANN](http://www.cs.umd.edu/~mount/ANN/) approximate-nearest-neighbour
library. This port uses **exact** nearest-neighbour search instead
(`scipy.spatial.cKDTree` in Python; a hand-port of the bundled
`KDTreeMulti` template in JavaScript) so that the two ports produce
deterministic, bit-comparable output. See `docs/ALGORITHM.md`.

## License

The original C++ implementation is distributed under the BSD 2-Clause
License (see `LICENSE`). All derivative source files in this repository
carry forward the original copyright notice and disclaimer.

Copyright (c) 2011, Jeffrey Chadwick. All rights reserved.

Additional code (the Python and JavaScript ports, the browser demo, the
test harness) is also licensed under the BSD 2-Clause License, with
attribution to the original authors required by the same terms.

## Training audio

The six WAV files under `assets/training_audio/` are from
**The Recordist's** *Ultimate Fire* sound library
(<https://www.therecordist.com/>). The original Cornell release states:

> "Permission to provide a few of these audio clips as examples in our
> source code package has been generously granted by The Recordist."

This demo redistributes them as a derivative work of the same source-code
package under the same permission. They are used unmodified (downmixed
to 44.1 kHz mono in the original release, as noted in the README of the
C++ archive). All copyright in the audio remains with The Recordist.

## Bundled flame-simulation signals

The `input_data.vector` files for the five examples (`burning_brick`,
`candle`, `dragon`, `flame_jet`, `torch`) are the low-frequency outputs
of physically based flame simulations produced by the original authors
and distributed with their code release. They are used here unmodified
(re-encoded as Float32 for browser delivery) for educational
demonstration. These are the same signals used by the sibling spectral
bandwidth-extension demo at `../fire-bandwidth-extension/`.

## Browser demo context

This browser demo was developed for the course CS 448Z (Physically Based
Animation and Sound) at Stanford University, Spring 2026, taught by
Doug L. James. It is intended as an educational exploration of the
algorithm and is not affiliated with or endorsed by the original
publication beyond the use of its released code and data.

## Vendored libraries

- `fft.js` by Fedor Indutny (https://github.com/indutny/fft.js), MIT License.
  See `fft.js` (at the project root) for the original license header.
