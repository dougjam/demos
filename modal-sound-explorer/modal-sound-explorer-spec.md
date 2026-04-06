# Modal Sound Explorer: Build Specification

## Purpose

An interactive HTML demo for Stanford CS448Z (Physically Based Animation and Sound) that lets students build intuition for modal vibration and sound synthesis. This is the central conceptual demo for the course: every subsequent topic (contact sound, fracture sound, bubble sound, shell sound) reduces to "which modes get excited, at what amplitudes, with what damping."

The demo uses **analytical mode models** for idealized geometries (string, bar, membrane, plate), following the approach of van den Doel & Pai ("The Sounds of Physical Shapes," 1998). This sits between the textbook derivation of the generalized eigenvalue problem and the numerical eigensolve on complex 3D meshes, giving students concrete physical objects whose mode structures they can see, hear, and manipulate.

## Core Pedagogical Goals

1. **Frequency ratios determine timbre.** Switching between object types (string, bar, membrane, plate) while keeping the fundamental frequency constant transforms the timbre. Students hear that it's the *ratios* between mode frequencies, not the fundamental alone, that give an object its sonic identity.

2. **Strike position selects modes.** Clicking different positions on the object changes which modes are excited, because the excitation amplitude of mode k is proportional to the mode shape value at the strike point. Striking at a node of mode k silences it; striking at an antinode maximizes it. This demonstrates the U^T f coupling from the modal analysis equations.

3. **Damping controls "ring" vs "thud."** Rayleigh damping (D = αM + βK) produces frequency-dependent decay rates and damping ratios. Students hear that β-damping kills high modes faster (makes things sound "dull" or "warm"), while α-damping gives all modes the same decay rate in seconds but causes low modes to ring through fewer oscillation cycles, producing an unnatural "tinny" quality. The balance produces recognizable material characters (metal vs wood vs rubber).

4. **Mode shapes are visible patterns.** For 1D objects (string, bar), mode shapes are transverse displacement curves. For 2D objects (membrane, plate), they produce nodal-line patterns (Chladni figures). Students see which parts of the object move for each mode and connect that to the strike-position dependence.

## Architecture

Single-file HTML/JS application. Use Web Audio API for sound synthesis. Use Canvas 2D for all visualization (no WebGL dependency). The demo should work on modern desktop browsers and degrade gracefully on mobile (touch = click for strikes).

Target: embeddable in an iframe on the course website, self-contained, no external dependencies except possibly a small JS math library if needed for Bessel function evaluation.

## Object Models

### 1. String (Fixed-Fixed, Ideal Flexible)

The harmonic reference case. Students know what harmonic sounds are from musical instruments.

**Frequency ratios:**
```
f_n = n * f_1,   n = 1, 2, 3, ...
```
Ratios: 1, 2, 3, 4, 5, 6, 7, 8, ...

**Mode shapes (1D):**
```
φ_n(x) = sin(n π x / L),   x ∈ [0, L]
```

**Strike coupling:** Striking at position x_s excites mode n with amplitude proportional to φ_n(x_s) = sin(n π x_s / L). Example: striking the center (x_s = L/2) excites only odd harmonics; striking at 1/3 suppresses every 3rd harmonic.

**Number of modes to use:** 12-16 modes (harmonics up to ~16x fundamental).

### 2. Bar (Free-Free, Euler-Bernoulli Beam)

The first inharmonic case. Bending stiffness causes higher modes to be sharper than harmonic, producing the characteristic "metallic" or "bell-like" quality of struck bars.

**Frequency ratios:**
The characteristic equation for a free-free beam is cos(βL) cosh(βL) = 1. The first several roots β_n L and resulting frequency ratios (f_n/f_1 = (β_n/β_1)^2) are:

| Mode | β_n L    | f_n / f_1 |
|------|----------|-----------|
| 1    | 4.73004  | 1.0000    |
| 2    | 7.85320  | 2.7565    |
| 3    | 10.9956  | 5.4039    |
| 4    | 14.1372  | 8.9330    |
| 5    | 17.2788  | 13.344    |
| 6    | 20.4204  | 18.644    |
| 7    | 23.5619  | 24.834    |
| 8    | 26.7035  | 31.913    |

For n ≥ 3 the approximation β_n L ≈ (2n+1)π/2 works well. For modes 1 and 2, use the exact values above.

**Mode shapes (1D):**
```
φ_n(x) = cosh(β_n x) + cos(β_n x) - σ_n [sinh(β_n x) + sin(β_n x)]
```
where
```
σ_n = [cosh(β_n L) - cos(β_n L)] / [sinh(β_n L) - sin(β_n L)]
```

Normalize each φ_n so that max|φ_n| = 1 for display purposes. Note: free-free bar mode shapes have free ends (nonzero displacement and slope at boundaries), unlike the string.

**Strike coupling:** Same principle: amplitude of mode n is proportional to φ_n(x_s). Free-free bar modes are symmetric/antisymmetric alternately about the center: odd-numbered modes are symmetric, even-numbered are antisymmetric. Striking the exact center excites only symmetric modes.

**Number of modes:** 8-10 (the rapid frequency growth means higher modes quickly exceed audibility).

### 3. Circular Membrane (Fixed Edge, Ideal Flexible)

A 2D vibrating surface with drum-like sound. The inharmonic spectrum is why drums have indefinite pitch.

**Frequency ratios:**
Frequencies are proportional to the zeros j_{mn} of Bessel functions J_m(x) = 0, where m is the number of nodal diameters and n is the number of nodal circles.

| Mode (m,n) | j_{mn}  | f/f_1   | Description           |
|------------|---------|--------|-----------------------|
| (0,1)      | 2.4048  | 1.000  | fundamental, 0 nodal diameters |
| (1,1)      | 3.8317  | 1.593  | 1 nodal diameter      |
| (2,1)      | 5.1356  | 2.136  | 2 nodal diameters     |
| (0,2)      | 5.5201  | 2.295  | 1 nodal circle        |
| (3,1)      | 6.3802  | 2.653  | 3 nodal diameters     |
| (1,2)      | 7.0156  | 2.917  | 1 diam + 1 circle     |
| (4,1)      | 7.5883  | 3.156  | 4 nodal diameters     |
| (2,2)      | 8.4172  | 3.500  | 2 diam + 1 circle     |
| (0,3)      | 8.6537  | 3.599  | 2 nodal circles       |

**Mode shapes (2D):**
```
Ψ_{mn}(r, θ) = J_m(j_{mn} r / R) cos(m θ)
```
For m > 0, there is a degenerate partner mode with sin(mθ) at the same frequency. For the demo, use only the cos(mθ) orientation (the sin partner just rotates the pattern and doesn't change the sound).

**Visualization:** Show a circular region colored by displacement (diverging colormap: blue-white-red or similar). Nodal lines (where Ψ = 0) should be clearly visible. These are the Chladni-figure-like patterns: concentric circles and radial lines.

**Strike coupling:** Click a point (r_s, θ_s) on the membrane. Mode (m,n) is excited with amplitude proportional to Ψ_{mn}(r_s, θ_s). Striking the center (r=0) excites only the (0,n) modes (axisymmetric, no nodal diameters). Striking off-center excites modes with nodal diameters.

**Bessel function evaluation:** J_m(x) for integer m can be computed via polynomial approximation or a small lookup table. Only need J_0 through J_4 evaluated at the known zeros for the mode shapes. Alternatively, use a simple series expansion: J_m(x) = Σ_{k=0}^{K} (-1)^k (x/2)^{m+2k} / (k! (m+k)!), converging rapidly for moderate x. K=15 terms is more than sufficient for the argument range needed.

**Number of modes:** 9-12 modes (the table above gives 9).

### 4. Rectangular Plate (Simply Supported, Kirchhoff Thin Plate)

Plate vibration with 2D mode shapes. Simply supported boundary conditions give clean analytical solutions.

**Frequency ratios:**
For a plate of dimensions a × b:
```
f_{mn} ∝ (m/a)^2 + (n/b)^2,    m, n = 1, 2, 3, ...
```

For a **square plate** (a = b), this simplifies to f_{mn} ∝ m^2 + n^2. Frequency ratios relative to the (1,1) fundamental:

| Mode (m,n) | m²+n² | f/f_1 | Degeneracy |
|------------|-------|-------|------------|
| (1,1)      | 2     | 1.000 | -          |
| (1,2)      | 5     | 2.500 | (2,1)      |
| (2,2)      | 8     | 4.000 | -          |
| (1,3)      | 10    | 5.000 | (3,1)      |
| (2,3)      | 13    | 6.500 | (3,2)      |
| (1,4)      | 17    | 8.500 | (4,1)      |
| (3,3)      | 18    | 9.000 | -          |
| (2,4)      | 20    | 10.00 | (4,2)      |

For degenerate pairs (m,n) and (n,m), include both as separate modes in the synthesis since they have different spatial patterns and are excited differently depending on strike position.

Support an **aspect ratio slider** (a/b from 0.5 to 2.0) that breaks the degeneracies and lets students hear how the spectrum shifts. At non-square ratios, the formerly degenerate pairs split into distinct frequencies, changing the timbre.

**Mode shapes (2D):**
```
Ψ_{mn}(x, y) = sin(m π x / a) sin(n π y / b)
```

**Visualization:** Rectangular region colored by displacement. Nodal lines form a grid pattern (straight lines at x = ka/m and y = kb/n).

**Strike coupling:** Click (x_s, y_s). Mode (m,n) excited with amplitude ∝ sin(mπx_s/a) sin(nπy_s/b). Striking the center of a square plate excites only modes where both m and n are odd.

**Number of modes:** 12-16 (include modes up to m,n = 4).

## Audio Synthesis

### Method
Use Web Audio API. On each strike event, compute an AudioBuffer containing the sum of decaying sinusoids:

```
p(t) = Σ_k  a_k  sin(2π f_k t)  exp(-α_k t)
```

where for each mode k:
- f_k = fundamental × (frequency ratio of mode k)
- a_k = (mode shape value at strike point) × (gain normalization)
- α_k = decay rate from Rayleigh damping (see below)

Buffer duration: 3 seconds at 44100 Hz sample rate (or until all modes have decayed below a threshold, whichever is shorter).

### Rayleigh Damping in Modal Domain
Given Rayleigh parameters α_R and β_R (using subscript R to distinguish from decay rate α_k):

For mode k with angular frequency ω_k = 2π f_k:
```
damping ratio:  ξ_k = α_R / (2 ω_k) + β_R ω_k / 2
decay rate:     α_k = ξ_k ω_k = α_R / 2 + β_R ω_k^2 / 2
```

Key insight for students: α_R contributes a constant decay rate α_R/2 to all modes (mass-proportional damping), while β_R contributes a decay rate β_R ω_k²/2 that grows with frequency squared (stiffness-proportional damping). This means:
- High α_R, low β_R: all modes decay at the same rate in seconds, but low-frequency modes complete fewer oscillation cycles before fading, so they sound relatively damped while high modes ring through many cycles (sounds "tinny")
- Low α_R, high β_R: low modes ring, high modes die fast (sounds "warm" or "dull")
- Balanced: the typical behavior of most real materials

### Material Presets

Each preset sets fundamental frequency, α_R, and β_R. These should be perceptually tuned to sound convincing, not necessarily derived from measured material data. Suggested starting points:

| Material | Fundamental (Hz) | α_R (1/s) | β_R (s)    | Character      |
|----------|-------------------|-----------|------------|----------------|
| Steel    | 440               | 1.0       | 0.000002   | bright, ringing |
| Aluminum | 520               | 2.0       | 0.000004   | moderate ring   |
| Glass    | 880               | 0.5       | 0.000001   | clear, bell-like|
| Wood     | 220               | 40.0      | 0.00005    | short, warm     |
| Ceramic  | 660               | 5.0       | 0.00001    | mid-ring        |
| Rubber   | 110               | 200.0     | 0.001      | dead thud       |

These will need tuning by ear during development. The β_R values in particular need care: too large and all high modes vanish instantly; too small and rubber rings like metal.

### Gain Normalization
Normalize mode amplitudes so the total peak amplitude of the synthesized buffer is approximately 0.8 (leaving headroom). After computing all a_k from strike position, scale the entire set so that the initial sample sum doesn't clip. This ensures consistent volume across different strike positions and object types.

## UI Layout

### Main Panel (Left/Center, ~65% Width)

**Object display area.** For 1D objects (string, bar): show a horizontal line representing the object. When struck, animate the transverse displacement (superposition of excited mode shapes) decaying over time. The animation rate should be slowed relative to audio playback so the shapes are visible (vibration frequency for display purposes can be ~2-5 Hz per mode, not the actual audio frequency).

For 2D objects (membrane, plate): show the object surface as a filled region with displacement mapped to a diverging colormap. Animate the modal superposition. Use the same temporal scaling approach for visibility.

**Strike interaction.** Click/tap anywhere on the object to strike. Show a brief visual indicator (a dot or ripple) at the strike point. Immediately synthesize and play the resulting sound.

**Mode shape gallery.** Below the object, show small thumbnails of the first N mode shapes, each labeled with its frequency ratio and absolute frequency. Highlight (border glow or opacity change) the modes that are currently active (excited above some threshold). Clicking a mode shape thumbnail should toggle it on/off for the next strike, allowing students to isolate individual modes or subsets.

### Control Panel (Right, ~35% Width)

**Object type selector:** String | Bar | Membrane | Plate. Switching objects preserves the current fundamental frequency so the pitch reference stays constant and only timbre changes.

**Fundamental frequency slider:** Range 50-2000 Hz. Default 440 Hz. Show the value. Consider also showing a piano keyboard reference (which note this corresponds to).

**Damping controls:** Two sliders.
- "Mass damping (α)": range 0 to 200. Maps to α_R. Label: "low modes decay faster →"
- "Stiffness damping (β)": range 0 to 0.001. Maps to β_R. Label: "high modes decay faster →"
- Consider log-scale or a nonlinear mapping for more perceptual uniformity.

**Material presets:** Dropdown or button row: Steel, Aluminum, Glass, Wood, Ceramic, Rubber. Selecting a preset sets α_R, β_R, and fundamental.

**Plate aspect ratio slider** (visible only when Plate is selected): range 0.5 to 2.0, default 1.0 (square). Label: "a/b ratio."

**Additional controls:**
- "Number of modes" slider (1 to max for current object). Starting at 1 and adding modes one at a time is a key learning experience.
- "Play all modes" button: strikes the object at a position that excites all modes roughly equally.
- "Play single mode" option: when a mode thumbnail is clicked with this active, play just that one mode in isolation.

### Spectrogram/Decay Display (Bottom)

After each strike, display a visualization of the modal decay. Two options (implement the simpler one first, the other as a stretch goal):

**Option A (simpler, recommended first): Mode energy bars.** A horizontal bar chart showing each active mode as a horizontal bar. Bar length represents current amplitude, decaying in real time after the strike. Each bar is labeled with the mode's frequency. This directly shows students which modes are present and how they decay at different rates. Color-code by frequency (low = warm colors, high = cool colors).

**Option B (stretch goal): Spectrogram.** Compute a short-time Fourier transform of the synthesized buffer and display as a time-frequency image. This shows the modes as horizontal lines decaying at different rates, which is how struck-object sounds look in any audio analysis tool. Students who go on to HW2 will use spectrograms to analyze their own modal synthesis results, so familiarity helps.

## Specific Interaction Scenarios to Support

These are the key "aha moment" interactions that the demo must enable cleanly:

### Scenario 1: "What does one mode sound like?"
Set number of modes to 1. Strike the object. Hear a single decaying sinusoid. Pure tone, simple decay. Now increase to 2 modes. Hear a more complex beating pattern. Continue adding modes and hear the timbre build up toward a recognizable struck-object sound.

### Scenario 2: "Why does a drum sound different from a guitar string?"
Set fundamental to 220 Hz. Select String. Strike the center. Hear a harmonic tone. Switch to Membrane. Same fundamental, same strike position. Hear an inharmonic, drum-like tone. The *only* difference is the frequency ratios.

### Scenario 3: "Why does striking different spots produce different tones?"
Select Bar. Strike the center. Note which modes are excited (symmetric modes only). Strike at 1/4 from the end. Note the different spectrum. Strike right at the end. Hear the difference. Look at the mode shape gallery to understand why.

### Scenario 4: "Why does wood sound different from metal?"
Select Bar. Choose Steel preset. Strike. Long ring, bright harmonics. Switch to Wood preset (same geometry, different damping). Strike same position. Short, warm thud. Toggle between them. See that the frequency ratios are identical; only the decay rates differ.

### Scenario 5: "What does Rayleigh damping actually do?"
Select Bar, Steel preset. Set α_R high, β_R low. Strike. All modes decay at the same rate, but low modes ring through very few cycles while high modes ring through many: an unnatural "tinny" sound. Now swap: α_R low, β_R high. Low modes ring, high modes vanish quickly: a "dull" or "muffled" sound. Find a natural balance. This directly demonstrates the Rayleigh damping model from lecture.

### Scenario 6: "Chladni patterns and mode selection" (2D objects)
Select Membrane. See the mode shapes in the gallery. Strike the center. Only axisymmetric (m=0) modes are excited: (0,1), (0,2), (0,3). The nodal-line patterns for those modes have circular symmetry. Now strike off-center. Modes with nodal diameters appear. The sound changes.

## Implementation Notes

### Bessel Functions
For the circular membrane, J_m(x) is needed for m = 0 through 4. A simple polynomial approximation or truncated power series works. Recommended: use the ascending series
```
J_m(x) = (x/2)^m  Σ_{k=0}^{K}  (-1)^k (x/2)^{2k} / (k! Γ(m+k+1))
```
with K = 20 terms, which converges for all arguments needed (x up to ~10). For integer m, Γ(m+k+1) = (m+k)!.

Alternatively, hardcode the mode shapes at a grid resolution (e.g., 100×100 for the membrane) during initialization, since the Bessel zeros j_{mn} are known constants and the shapes don't change.

### Free-Free Bar Mode Shapes
The mode shapes involve cosh, cos, sinh, sin with the β_n values. These are standard Math functions. Compute σ_n from the formula in the Object Models section. Evaluate φ_n on a 1D grid (e.g., 200 points) during initialization.

### Animation Timing
Audio frequencies (hundreds of Hz) are far too fast to visualize. For the mode shape animation, use "display frequencies" that are scaled down to 1-5 Hz per mode, preserving the *ratios* between modes. The animation should show 2-3 seconds of vibration decaying to rest, running at a comfortable visual rate. Synchronize the animation start with the audio playback start.

### Performance
The audio buffer computation (summing ~15 decaying sinusoids over ~130,000 samples) is fast and should complete in <50ms. The visualization animation should use requestAnimationFrame. The 2D mode shape rendering (colormapped grid) may need optimization if the grid is fine; 80×80 is sufficient for clear nodal-line patterns.

### Responsive Layout
The demo will be embedded on the course website. Target a minimum viewport of 900×600 px. On smaller screens, stack the control panel below the visualization rather than beside it. Touch events should map to click events for strike interaction.

## Files to Produce

A single `modal-sound-explorer.html` file containing all HTML, CSS, and JavaScript. No external dependencies. The file should be well-commented, particularly in the sections implementing:
- Mode frequency and shape calculations
- Audio synthesis
- Rayleigh damping computation
- Strike-position coupling

## What Not to Build

- No 3D visualization. All visualization is 2D (1D curves or 2D colormaps).
- No mesh import or FEM eigensolve. This demo uses analytical models only.
- No real-time audio streaming. Compute complete buffers on each strike event.
- No microphone input or recording.
- No attempt to model acoustic transfer or radiation patterns (that's a separate demo topic).
