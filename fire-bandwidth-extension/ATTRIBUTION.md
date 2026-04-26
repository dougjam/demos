# Attribution

## Original work

The spectral bandwidth extension algorithm implemented here is by:

- **Jeffrey N. Chadwick** (Cornell University, at the time of publication)
- **Doug L. James** (Cornell University, at the time of publication; now Stanford University)

Published in:

> Jeffrey N. Chadwick and Doug L. James, "Animating Fire with Sound,"
> ACM Transactions on Graphics (SIGGRAPH 2011), 30(4), August 2011.

Project page (canonical reference): https://www.cs.cornell.edu/projects/Sound/fire/

Paper PDF: https://www.cs.cornell.edu/projects/Sound/fire/FireSound2011.pdf

## Reference implementation

The Python and JavaScript code in this repository is a faithful port of the
Matlab reference implementation released by the authors at the URL above
(`bandwidth_extension.zip` / `bandwidth_extension.tar.gz`). All algorithm
design choices, parameter defaults, and numerical details follow that
implementation. Differences from the released Matlab are documented in
`docs/ALGORITHM.md`.

## License

The original Matlab implementation is distributed under the BSD 2-Clause
License (see `LICENSE`). All derivative source files in this repository
carry forward the original copyright notice and disclaimer.

Copyright (c) 2011, Jeffrey Chadwick. All rights reserved.

Additional code (the Python and JavaScript ports, the browser demo, the
test harness) is also licensed under the BSD 2-Clause License, with
attribution to the original authors required by the same terms.

## Test signal data

The five `.vector` files in `srcOrig/work/` are the low-frequency outputs
of physically based flame simulations, produced by the original authors
and distributed with their code release. They are used here unmodified
(re-encoded as Float32 for browser delivery) for educational demonstration.

## Browser demo context

This browser demo was developed for the course CS 448Z (Physically Based
Animation and Sound) at Stanford University, Spring 2026, taught by
Doug L. James. It is intended as an educational exploration of the
algorithm and is not affiliated with or endorsed by the original
publication beyond the use of its released code and data.

## Vendored libraries

- `fft.js` by Fedor Indutny (https://github.com/indutny/fft.js), MIT License.
  See `fft.js` (at the project root) for the original license header.
