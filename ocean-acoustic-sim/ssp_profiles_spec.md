# Sound Speed Profile (SSP) Models -- Implementation Specification

Reference document for implementing canonical SSP models in an ocean acoustics simulator.

## 1. Munk Canonical Profile (Deep Ocean)

**Use case:** Standard deep-ocean SOFAR channel. The workhorse benchmark for mid-latitude deep water.

**Equation:**

```
c(z) = c₁ * (1 + ε * (η + e^(-η) - 1))
```

where:
- `η = 2 * (z - z₁) / B` is the scaled depth
- `z` = depth (m)
- `z₁` = depth of the channel axis (m)
- `B` = thermocline scale depth (m)
- `c₁` = sound speed at the channel axis (m/s)
- `ε` = perturbation coefficient (dimensionless)

**Typical parameters:**
- `c₁ = 1492 m/s`
- `z₁ = 1300 m`
- `B = 1300 m`
- `ε = 0.00737`

**Behavior:** Minimum at `z = z₁` (the SOFAR axis), increasing above and below. Produces the classic deep sound channel with convergence-zone propagation.

**Reference:** Munk, W. (1974). "Sound channel in an exponentially stratified ocean, with application to SOFAR." JASA 55, 220--226.


## 2. Arctic Linear Profile (Polar / Under-Ice)

**Use case:** Central Arctic Ocean, where near-freezing surface water and increasing pressure produce a monotonically increasing SSP. Models the upward-refracting surface duct that channels sound against the ice canopy.

**Equation:**

```
c(z) = c₀ + g * z
```

where:
- `z` = depth (m)
- `c₀` = sound speed at the surface (m/s)
- `g` = sound speed gradient (s⁻¹)

**Typical parameters:**
- `c₀ = 1435 m/s` (near-freezing surface water)
- `g = 0.016 s⁻¹`

**Behavior:** Monotonically increasing with depth. All ray paths are upward-refracting circular arcs (exact analytic solution: rays in a constant-gradient medium follow circular arcs with radius R = c₀/g). Sound is repeatedly reflected from the surface (or ice), never reaching the bottom in the deep ocean case.

**Ray curvature:** In a linear profile, a ray launched at angle θ₀ from depth z₀ follows a circular arc with radius:
```
R = c(z₀) / (g * cos(θ₀))
```

**Notes:** For a richer Arctic model, consider a piecewise-linear variant: an isothermal mixed layer (constant c) of depth ~50 m, then a linear increase below. The western Arctic (Beaufort Sea) has additional structure from Pacific Water intrusion creating the "Beaufort Duct" around 50--200 m depth, but the simple linear captures the dominant behavior of the eastern/central Arctic.

**Reference:** Jensen, Kuperman, Porter & Schmidt, *Computational Ocean Acoustics*, Ch. 1. Ainslie, *Principles of Sonar Performance Modelling*, Ch. 4. Gradient value from Hurdle (1986), *The Nordic Seas*.


## 3. n²-Linear Profile (Airy Function Modes)

**Use case:** The textbook profile for which exact normal mode solutions exist in terms of Airy functions. Extremely valuable for validating numerical mode solvers and parabolic equation codes against an analytic ground truth.

**Equation:**

The index of refraction squared varies linearly with depth:

```
n²(z) = 1 + a * (z - z₁)
```

which gives a sound speed profile (to first order for small perturbations):

```
1/c²(z) = (1/c₁²) * (1 + a * (z - z₁))
```

or equivalently:

```
c(z) = c₁ / sqrt(1 + a * (z - z₁))
```

where:
- `z₁` = depth of the channel axis (m)
- `c₁` = sound speed at the channel axis (m/s)
- `a` = gradient parameter (m⁻¹), controls channel width

**Typical parameters (to mimic a deep ocean channel):**
- `c₁ = 1490 m/s`
- `z₁ = 1000 m`
- `a = 1.0e-5 m⁻¹` (a small value gives a wide channel)

**Behavior:** Has a minimum at `z = z₁`. Above and below the axis, c(z) increases (symmetric for small perturbations). The normal modes of the Helmholtz equation with this profile are Airy functions Ai(ξ), and the eigenvalues (horizontal wavenumbers) are determined by the zeros of the Airy function at the boundaries.

**Why it matters:** The Airy function solutions provide exact mode shapes and eigenvalues you can compare against any numerical mode solver, making this the gold standard for code verification.

**Reference:** Brekhovskikh, L.M. & Lysanov, Yu.P. (2003). *Fundamentals of Ocean Acoustics*, 3rd ed., Springer, Sec. 6.1. Also: Jensen et al., *Computational Ocean Acoustics*, Sec. 2.4.


## 4. Bilinear (Thermocline) Profile

**Use case:** Simple piecewise-linear model of a shallow-water or coastal sound channel. Two linear segments meeting at the channel axis, capturing the essential feature of the thermocline above and pressure-dominated increase below.

**Equation:**

```
c(z) = c₁ + g₁ * (z₁ - z)    for z < z₁  (above axis)
c(z) = c₁ + g₂ * (z - z₁)    for z >= z₁  (below axis)
```

where:
- `z₁` = depth of the channel axis (m)
- `c₁` = minimum sound speed at the axis (m/s)
- `g₁` = magnitude of the sound speed gradient above axis (s⁻¹), note sign convention: c increases upward from axis
- `g₂` = sound speed gradient below axis (s⁻¹)

**Typical parameters (shallow-water channel):**
- `c₁ = 1490 m/s`
- `z₁ = 100 m` (shallow thermocline)
- `g₁ = 0.05 s⁻¹` (strong thermocline effect above)
- `g₂ = 0.016 s⁻¹` (pressure-dominated below)

**Typical parameters (deep-water channel):**
- `c₁ = 1492 m/s`
- `z₁ = 1000 m`
- `g₁ = 0.02 s⁻¹`
- `g₂ = 0.016 s⁻¹`

**Behavior:** V-shaped profile with a minimum at z₁. The asymmetry (g₁ ≠ g₂) is physically motivated: the thermocline gradient is typically steeper than the deep pressure gradient. Ray paths are piecewise circular arcs (each linear segment has its own constant gradient). This makes ray tracing trivially analytic.

**Notes:** The bilinear profile is the simplest model that captures the qualitative behavior of a sound channel without the exponential math of Munk. In shallow water, the channel axis can be quite shallow (50--200 m). The profile can also be extended to piecewise-linear with more segments (e.g., mixed layer + thermocline + deep isothermal) for greater realism.

**Reference:** Etter, P.C. (2018). *Underwater Acoustic Modeling and Simulation*, 5th ed., CRC Press, Ch. 2.


## 5. Epstein Layer (Hyperbolic Secant-Squared Duct)

**Use case:** Models a single trapped duct (e.g., a surface duct or an internal duct from a warm water intrusion). Admits an exact analytic solution for the plane-wave reflection coefficient, which is valuable for validating reflection/transmission calculations.

**Equation:**

```
c(z) = c₁ / sqrt(1 + δc * sech²((z - z₁) / L))
```

where:
- `z₁` = center depth of the duct (m)
- `c₁` = background sound speed outside the duct (m/s)
- `δc` = fractional sound speed perturbation (dimensionless, positive for a duct/minimum)
- `L` = half-width scale of the duct (m)

Equivalently, in terms of the index of refraction:
```
n²(z) = 1 + δc * sech²((z - z₁) / L)
```

**Typical parameters (surface duct):**
- `c₁ = 1500 m/s`
- `z₁ = 50 m`
- `δc = 0.01` (a 1% duct)
- `L = 30 m`

**Behavior:** A localized dip in sound speed centered at z₁, with the profile returning to c₁ far from the duct center. Sound can be trapped in the low-speed region. The exact reflection coefficient for this profile involves Gamma functions, making it a classic test case for wave propagation codes.

**Notes:** The Epstein profile is the acoustic analog of the Poschl-Teller potential in quantum mechanics. The number of trapped modes depends on the duct strength parameter `δc * (k * L)²` where k = ω/c₁.

**Reference:** Brekhovskikh, L.M. & Lysanov, Yu.P. (2003). *Fundamentals of Ocean Acoustics*, 3rd ed., Springer, Sec. 5.4. Also: Epstein, P.S. (1930). "Reflection of waves in an inhomogeneous absorbing medium." PNAS 16, 627--637.


## 6. Isovelocity (Constant) / Pekeris Waveguide

**Use case:** The simplest possible ocean model. Constant sound speed throughout the water column with a reflecting surface and a penetrable (fluid) bottom. The Pekeris waveguide admits exact normal mode solutions (sinusoidal mode shapes), making it the most basic validation benchmark.

**Equation:**

```
c(z) = c₀    for 0 <= z <= D  (water column)
c(z) = c_b   for z > D        (bottom halfspace)
```

where:
- `c₀` = sound speed in water (m/s)
- `D` = water depth (m)
- `c_b` = sound speed in the bottom (m/s), must satisfy c_b > c₀ for modes to exist
- Bottom density `ρ_b` and attenuation `α_b` are also needed for the full Pekeris solution

**Typical parameters:**
- `c₀ = 1500 m/s`
- `D = 100 m` (shallow water) or `D = 5000 m` (deep water)
- `c_b = 1600 m/s`
- `ρ_b = 1.5 g/cm³`
- `α_b = 0.5 dB/wavelength`

**Behavior:** Rays are straight lines between surface and bottom bounces. Modes are (approximately) sinusoidal in depth. The number of trapped modes is:
```
N ≈ floor(D * f / c₀ * sqrt(1 - (c₀/c_b)²) * 2)
```
where f is frequency in Hz.

**Notes:** Although physically unrealistic, the Pekeris waveguide is the foundation of underwater acoustics pedagogy. Every propagation code should reproduce the Pekeris solution exactly before being trusted with more complex profiles.

**Reference:** Pekeris, C.L. (1948). "Theory of propagation of explosive sound in shallow water." GSA Memoir 27. Also: Jensen et al., *Computational Ocean Acoustics*, Sec. 2.1--2.3.


---

## Implementation Notes

**Depth convention:** z = 0 at the ocean surface, z increasing downward (positive into the ocean).

**Domain:** All profiles should be evaluable for z in [0, D] where D is the water depth. For profiles with a channel axis, ensure z₁ is within the domain. Handle edge cases (z < 0, z > D) gracefully.

**Units:** All speeds in m/s, depths in m, gradients in s⁻¹.

**Interface:** Each profile should be callable as `c = profile(z)` and ideally also provide `dc/dz = profile_gradient(z)` for ray tracing (Snell's law requires the gradient).

**Suggested API:**
```
class SSPProfile:
    name: str
    description: str
    def sound_speed(z: float) -> float
    def gradient(z: float) -> float
    def plot(z_max: float, dz: float) -> None
```
