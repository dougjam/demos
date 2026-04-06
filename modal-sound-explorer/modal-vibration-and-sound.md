# Modal Vibration and Sound Synthesis

*CS 448Z: Physically Based Animation and Sound, Spring 2026*
*Companion material for the Modal Sound Explorer demo*

## 1. The Sound of a Struck Object

When you strike a solid object, kinetic energy from the impact is converted into elastic vibrations that propagate through the body. The vibrating surface displaces the surrounding air, producing sound waves. The resulting sound depends on three things: the *frequencies* at which the object vibrates (determined by its geometry, material stiffness, and density), the *amplitudes* of those vibrations (determined by where and how the object is struck), and the *decay rates* of each vibration component (determined by internal energy dissipation in the material).

Modal analysis provides a mathematical framework that decomposes the complex vibration into a set of independent oscillators, each fully described by a frequency, a spatial pattern (the mode shape), and a decay rate. The total sound is the sum of contributions from all modes.

## 2. The Damped Harmonic Oscillator

Every mode of a vibrating object behaves as a damped harmonic oscillator:

$$m\ddot{x} + c\dot{x} + kx = f(t)$$

where $m$ is mass, $k$ is stiffness, $c$ is damping, and $f(t)$ is an external driving force. The undamped natural frequency is $\omega_0 = \sqrt{k/m}$, and the damping ratio is $\xi = c/(2m\omega_0)$.

The impulse response (response to an instantaneous unit tap) for an underdamped system ($\xi < 1$) is:

$$x(t) = \frac{1}{m\omega_d} e^{-\xi\omega_0 t} \sin(\omega_d t), \quad t \geq 0$$

where $\omega_d = \omega_0\sqrt{1 - \xi^2}$ is the damped natural frequency. For the small damping ratios typical of sounding objects ($\xi \sim 0.001$ to $0.05$), $\omega_d \approx \omega_0$ and the response is a slowly decaying sinusoid.

This single decaying sinusoid is the fundamental building block of modal sound.

## 3. A Unified Framework for Vibrating Bodies

### The governing equation

Following van den Doel and Pai [1998], the transverse vibration of a body is described by a partial differential equation of the form:

$$\mathcal{L}\,\phi(\mathbf{x}, t) + \rho\,\frac{\partial^2 \phi}{\partial t^2} = F(\mathbf{x}, t) \tag{1}$$

where $\phi(\mathbf{x}, t)$ is the transverse displacement at position $\mathbf{x}$ and time $t$, $\rho$ is the mass density (per unit length for 1D objects, per unit area for 2D objects), $F$ is the applied external force density, and $\mathcal{L}$ is a **spatial differential operator** that encodes the restoring force. The operator $\mathcal{L}$ depends on the physics of the object: what kind of internal forces resist deformation and attempt to restore the body to its equilibrium configuration.

There are two fundamentally different restoring mechanisms:

**Tension.** In a string or membrane, the restoring force comes from tension in the material. The displacement creates a curvature, and the tension acts along the curved surface to pull it back toward equilibrium. This produces a restoring force proportional to the *second spatial derivative* of the displacement (the Laplacian). The operator is:

$$\mathcal{L}_{\text{tension}} = -T\,\nabla^2$$

where $T$ is the tension (force per unit length for a string, force per unit length of boundary for a membrane).

**Bending stiffness.** In a bar or plate, the restoring force comes from the material's resistance to bending. A thicker or stiffer material resists curvature more strongly. The physics of elastic bending produces a restoring force proportional to the *fourth spatial derivative* of the displacement (the biharmonic operator). The operator is:

$$\mathcal{L}_{\text{bending}} = D\,\nabla^4$$

where $D$ is the flexural rigidity. For a bar, $D = EI$ (Young's modulus times the second moment of area of the cross-section). For a plate, $D = Eh^3/[12(1-\nu^2)]$ (involving the plate thickness $h$ and Poisson's ratio $\nu$).

### The four analytical models

Combining the two spatial dimensions (1D or 2D) with the two restoring mechanisms (tension or bending stiffness) gives four canonical vibrating objects:

|                    | **Tension (2nd order)**  | **Bending stiffness (4th order)** |
|--------------------|--------------------------|-----------------------------------|
| **1D**             | String                   | Bar (Euler-Bernoulli beam)       |
| **2D**             | Membrane                 | Plate (Kirchhoff thin plate)     |

Moving right across the table (tension to bending) changes the spatial order of the operator from 2 to 4, which has profound consequences for the frequency spectrum. Moving down (1D to 2D) adds a spatial dimension, which introduces new mode structure (nodal lines rather than nodal points) and denser spectra.

### Separation into modes

For all four cases, we seek solutions of the form $\phi(\mathbf{x}, t) = \psi(\mathbf{x})\, q(t)$, separating spatial and temporal dependence. Substituting into Equation (1) with $F = 0$ and dividing by $\psi q$ gives:

$$\frac{\mathcal{L}\,\psi(\mathbf{x})}{\rho\,\psi(\mathbf{x})} = -\frac{\ddot{q}(t)}{q(t)} = \omega^2 \tag{2}$$

The left side depends only on $\mathbf{x}$; the right side depends only on $t$. Both must equal the same constant $\omega^2$. This yields:

- **Spatial eigenvalue problem:** $\mathcal{L}\,\psi_n = \rho\,\omega_n^2\,\psi_n$, subject to boundary conditions. The solutions $\psi_n(\mathbf{x})$ are the **mode shapes** and the constants $\omega_n$ are the **natural angular frequencies**.

- **Temporal equation:** $\ddot{q}_n + \omega_n^2 q_n = 0$, giving sinusoidal oscillation at frequency $\omega_n$.

The general solution is a superposition over all modes:

$$\phi(\mathbf{x}, t) = \sum_n q_n(t)\,\psi_n(\mathbf{x}) = \sum_n \bigl[a_n \sin(\omega_n t) + b_n \cos(\omega_n t)\bigr]\,\psi_n(\mathbf{x}) \tag{3}$$

where $a_n$ and $b_n$ are determined by initial conditions.

### Excitation by impact

When the object is struck at a point $\mathbf{x}_s$ with an impulsive force (starting from rest), the initial conditions are $\phi(\mathbf{x}, 0) = 0$ and $\dot{\phi}(\mathbf{x}, 0) = \delta(\mathbf{x} - \mathbf{x}_s)$. Using the orthogonality of the mode shapes, the modal amplitudes become:

$$a_n = \frac{\psi_n(\mathbf{x}_s)}{\|\psi_n\|^2\,\omega_n}, \quad b_n = 0 \tag{4}$$

The amplitude of mode $n$ is proportional to $\psi_n(\mathbf{x}_s)$: the value of the mode shape at the strike point. **If the strike point lies on a nodal line of mode $n$ (where $\psi_n = 0$), that mode is not excited.** This is why striking an object at different locations produces different timbres, even though the set of natural frequencies is the same.

## 4. The String (1D, Tension)

### Physical setup

A flexible string of length $L$, uniform linear mass density $\rho_L$ (kg/m), held under tension $T$ (N), with both ends fixed. The string has negligible bending stiffness; the restoring force comes entirely from tension.

### Governing PDE

$$T\frac{\partial^2 \phi}{\partial x^2} = \rho_L\frac{\partial^2 \phi}{\partial t^2}, \quad x \in [0, L]$$

This is the classical **1D wave equation**. The operator is $\mathcal{L} = -T\,\partial^2/\partial x^2$, second order in space.

The wave speed is $c = \sqrt{T/\rho_L}$.

### Boundary conditions

Fixed (Dirichlet) at both ends: $\phi(0, t) = \phi(L, t) = 0$.

### Mode shapes and frequencies

The spatial eigenvalue problem $-T\,\psi'' = \rho_L\,\omega^2\,\psi$ with $\psi(0) = \psi(L) = 0$ has solutions:

$$\psi_n(x) = \sin\!\left(\frac{n\pi x}{L}\right), \quad n = 1, 2, 3, \ldots$$

$$\omega_n = \frac{n\pi c}{L} = \frac{n\pi}{L}\sqrt{\frac{T}{\rho_L}}$$

$$f_n = \frac{\omega_n}{2\pi} = \frac{n}{2L}\sqrt{\frac{T}{\rho_L}} = n\,f_1 \tag{5}$$

### Key physical insight

The frequencies form a **harmonic series**: $f_n/f_1 = n$. This is a direct consequence of the second-order wave equation on a 1D domain with fixed boundaries. The harmonic spectrum is why plucked and bowed strings produce sounds with definite musical pitch.

### Strike-position dependence

Striking at $x_s$ excites mode $n$ with amplitude $\propto \sin(n\pi x_s/L)$. Striking the center ($x_s = L/2$) gives $\sin(n\pi/2)$, which is zero for all even $n$: only odd harmonics are excited. Striking at $x_s = L/3$ suppresses every third harmonic. This is the physics behind the tonal variation a guitarist achieves by plucking at different points along the string.

## 5. The Bar (1D, Bending Stiffness)

### Physical setup

A uniform bar of length $L$, cross-sectional area $A$, second moment of area $I$, material density $\rho$ (kg/m³), and Young's modulus $E$ (Pa). The bar is thin enough that the Euler-Bernoulli beam theory applies: plane cross-sections remain plane and perpendicular to the neutral axis during bending. Both ends are free (unclamped, unsupported).

### Governing PDE

$$EI\frac{\partial^4 \phi}{\partial x^4} + \rho A\frac{\partial^2 \phi}{\partial t^2} = 0, \quad x \in [0, L]$$

This is the **Euler-Bernoulli beam equation**. The operator is $\mathcal{L} = EI\,\partial^4/\partial x^4$, fourth order in space. The critical difference from the string: the restoring force involves the fourth derivative, not the second. This arises because bending stiffness resists changes in *curvature* ($\partial^2\phi/\partial x^2$), and the elastic restoring force is proportional to the *derivative* of the internal bending moment, giving $\partial^4\phi/\partial x^4$.

### Boundary conditions (free-free)

At a free end, both the bending moment and the shear force vanish:

$$\frac{\partial^2\phi}{\partial x^2}\bigg|_{x=0,L} = 0, \quad \frac{\partial^3\phi}{\partial x^3}\bigg|_{x=0,L} = 0$$

These are "natural" boundary conditions for the free-free bar. They require four conditions total (two per end), matching the fourth-order operator. We use free-free conditions because they model a bar suspended loosely or resting on a soft surface, as in a xylophone or marimba. Van den Doel and Pai [1998] analyze a clamped-clamped bar (both ends rigidly fixed, $\psi = \psi' = 0$). Both cases yield the same characteristic equation and the same eigenfrequencies; only the mode shapes and their boundary values differ.

Note: the free-free bar also admits two zero-frequency rigid-body modes (uniform translation and uniform rotation) that satisfy the boundary conditions trivially. These do not radiate sound and are excluded from the tables below.

### Mode shapes and frequencies

Substituting $\psi(x) = C_1\cosh(\beta x) + C_2\sinh(\beta x) + C_3\cos(\beta x) + C_4\sin(\beta x)$ and applying the boundary conditions yields the **characteristic equation**:

$$\cos(\beta L)\cosh(\beta L) = 1 \tag{6}$$

The first several roots $\beta_n L$ and the resulting frequency ratios (since $\omega_n \propto \beta_n^2$) are:

| Mode $n$ | $\beta_n L$ | $f_n / f_1$ |
|----------|-------------|-------------|
| 1        | 4.73004     | 1.000       |
| 2        | 7.85320     | 2.757       |
| 3        | 10.9956     | 5.404       |
| 4        | 14.1372     | 8.933       |
| 5        | 17.2788     | 13.34       |
| 6        | 20.4204     | 18.64       |
| 7        | 23.5619     | 24.83       |
| 8        | 26.7035     | 31.91       |

The absolute fundamental frequency is:

$$f_1 = \frac{(\beta_1 L)^2}{2\pi L^2}\sqrt{\frac{EI}{\rho A}} = \frac{(4.73)^2}{2\pi L^2}\sqrt{\frac{EI}{\rho A}} \tag{7}$$

The mode shapes are:

$$\psi_n(x) = \cosh(\beta_n x) + \cos(\beta_n x) - \sigma_n\bigl[\sinh(\beta_n x) + \sin(\beta_n x)\bigr] \tag{8}$$

where $\sigma_n = [\cosh(\beta_n L) - \cos(\beta_n L)]/[\sinh(\beta_n L) - \sin(\beta_n L)]$.

These mode shapes have nonzero displacement at the free ends, alternating between symmetric (odd $n$) and antisymmetric (even $n$) about the bar's midpoint.

### Key physical insight

The frequency ratios grow as roughly $n^2$ (for large $n$, $\beta_n L \approx (2n+1)\pi/2$, so $f_n \propto \beta_n^2 \propto n^2$). This **superlinear growth** is the defining acoustic signature of bending vibration. Compare with the string's linear growth ($f_n = n f_1$). The bar's higher modes are "compressed upward" in frequency relative to the harmonic series.

Perceptually, this makes struck bars sound distinctly different from struck strings. The inharmonic spectrum produces a "metallic" or "bell-like" quality. This is the physics behind the sound of xylophones, chimes, and tuning forks.

### Why fourth order produces faster-than-harmonic growth

The physical reason is that higher modes involve tighter spatial oscillations (shorter wavelengths). For a string, the restoring force is proportional to curvature ($\partial^2\phi/\partial x^2 \sim k^2$, where $k$ is the wavenumber), and since $\omega^2 \propto k^2$ from the wave equation, $\omega \propto k \propto n$: linear growth. For a bar, the restoring force involves the fourth derivative ($\partial^4\phi/\partial x^4 \sim k^4$), and $\omega^2 \propto k^4$, giving $\omega \propto k^2 \propto n^2$: quadratic growth. The stiffer physics of bending pushes high-frequency modes to even higher frequencies.

## 6. The Circular Membrane (2D, Tension)

### Physical setup

A thin, flexible membrane of radius $R$, uniform surface mass density $\rho_S$ (kg/m²), held under uniform tension $T$ (N/m), with its edge clamped to a rigid circular frame. The membrane has negligible bending stiffness. This is the idealized model of a drumhead.

### Governing PDE

$$T\,\nabla^2\phi = \rho_S\frac{\partial^2\phi}{\partial t^2}$$

This is the **2D wave equation**. In polar coordinates $(r, \theta)$:

$$T\left(\frac{\partial^2\phi}{\partial r^2} + \frac{1}{r}\frac{\partial\phi}{\partial r} + \frac{1}{r^2}\frac{\partial^2\phi}{\partial\theta^2}\right) = \rho_S\frac{\partial^2\phi}{\partial t^2}$$

The operator is $\mathcal{L} = -T\,\nabla^2$, the 2D generalization of the string's operator.

### Boundary conditions

Fixed (Dirichlet) at the rim: $\phi(R, \theta, t) = 0$.

### Mode shapes and frequencies

Separation of variables in polar coordinates, $\psi(r,\theta) = R(r)\Theta(\theta)$, gives:

- Angular part: $\Theta(\theta) = \cos(m\theta)$ or $\sin(m\theta)$, $m = 0, 1, 2, \ldots$
- Radial part: $R(r) = J_m(k_{mn}\,r)$, where $J_m$ is the Bessel function of the first kind of order $m$

The boundary condition $\psi(R, \theta) = 0$ requires $J_m(k_{mn}R) = 0$, so $k_{mn}R = j_{mn}$, where $j_{mn}$ is the $n$-th positive zero of $J_m$. The mode shapes and frequencies are:

$$\psi_{mn}(r, \theta) = J_m\!\left(\frac{j_{mn}\,r}{R}\right)\cos(m\theta) \tag{9}$$

$$f_{mn} = \frac{j_{mn}}{2\pi R}\sqrt{\frac{T}{\rho_S}} = \frac{j_{mn}}{j_{01}}\,f_{01} \tag{10}$$

The integer $m$ counts **nodal diameters** (radial lines where $\psi = 0$) and $n$ counts **nodal circles** (concentric rings where $\psi = 0$). The first several zeros and frequency ratios are:

| Mode $(m,n)$ | $j_{mn}$  | $f/f_{01}$ | Nodal pattern     |
|-------------|---------|-----------|-------------------|
| (0,1)       | 2.4048  | 1.000     | no nodal lines    |
| (1,1)       | 3.8317  | 1.593     | 1 diameter        |
| (2,1)       | 5.1356  | 2.136     | 2 diameters       |
| (0,2)       | 5.5201  | 2.295     | 1 circle          |
| (3,1)       | 6.3802  | 2.653     | 3 diameters       |
| (1,2)       | 7.0156  | 2.917     | 1 diam + 1 circle |
| (4,1)       | 7.5883  | 3.156     | 4 diameters       |
| (2,2)       | 8.4172  | 3.500     | 2 diam + 1 circle |
| (0,3)       | 8.6537  | 3.599     | 2 circles         |

For $m > 0$, each frequency has a degenerate partner with $\sin(m\theta)$ instead of $\cos(m\theta)$, corresponding to the same pattern rotated by $\pi/(2m)$.

### Key physical insight

The membrane is the 2D analog of the string: the same operator ($-T\nabla^2$), but on a 2D domain. Like the string, it is tension-dominated and second-order. But unlike the string, its frequency ratios are **inharmonic**: they are determined by the zeros of Bessel functions, which do not fall in integer ratios.

This inharmonicity is why drums have no definite musical pitch (unlike strings). The ear cannot lock onto a clear harmonic pattern, so the perceived pitch is ambiguous.

### Strike-position dependence

Striking the center ($r_s = 0$) gives $\psi_{mn}(0, \theta) = J_m(0)\cos(m\theta)$. Since $J_m(0) = 0$ for all $m \geq 1$ and $J_0(0) = 1$, striking the center excites **only the axisymmetric modes** $(0,n)$. The resulting sound has fewer spectral components and sounds simpler ("duller") than striking off-center, which excites the modes with nodal diameters. This matches the well-known behavior of real drums.

## 7. The Rectangular Plate (2D, Bending Stiffness)

### Physical setup

A thin, flat plate of dimensions $a \times b$, thickness $h$, material density $\rho$, Young's modulus $E$, and Poisson's ratio $\nu$. The plate is simply supported on all four edges (the edges are held in place but free to rotate). The Kirchhoff thin-plate assumptions apply: the plate is thin ($h \ll a, b$), plane sections remain plane, and transverse shear deformation is negligible.

### Governing PDE

$$D\,\nabla^4\phi + \rho h\,\frac{\partial^2\phi}{\partial t^2} = 0$$

where $\nabla^4 = \nabla^2(\nabla^2)$ is the **biharmonic operator** and $D = Eh^3/[12(1-\nu^2)]$ is the flexural rigidity. In Cartesian coordinates:

$$D\left(\frac{\partial^4\phi}{\partial x^4} + 2\frac{\partial^4\phi}{\partial x^2\partial y^2} + \frac{\partial^4\phi}{\partial y^4}\right) + \rho h\,\frac{\partial^2\phi}{\partial t^2} = 0$$

This is the 2D analog of the Euler-Bernoulli beam equation. The operator $\mathcal{L} = D\,\nabla^4$ is fourth order in space (each spatial dimension contributing up to fourth derivatives). The plate is to the membrane what the bar is to the string: a bending-dominated counterpart of the tension-dominated model.

### Boundary conditions (simply supported)

At each edge, the displacement and the bending moment vanish. For example, at $x = 0$ and $x = a$:

$$\phi = 0, \quad \frac{\partial^2\phi}{\partial x^2} = 0$$

These conditions allow the plate to pivot freely at its edges while constraining it to remain in the support plane.

### Mode shapes and frequencies

The simply-supported boundary conditions admit a separable solution. The mode shapes and frequencies are:

$$\psi_{mn}(x, y) = \sin\!\left(\frac{m\pi x}{a}\right)\sin\!\left(\frac{n\pi y}{b}\right), \quad m, n = 1, 2, 3, \ldots \tag{11}$$

$$f_{mn} = \frac{\pi}{2}\sqrt{\frac{D}{\rho h}}\left[\left(\frac{m}{a}\right)^2 + \left(\frac{n}{b}\right)^2\right] \tag{12}$$

Relative to the fundamental $(1,1)$:

$$\frac{f_{mn}}{f_{11}} = \frac{(m/a)^2 + (n/b)^2}{(1/a)^2 + (1/b)^2} \tag{13}$$

For a **square plate** ($a = b$), this simplifies to $f_{mn}/f_{11} = (m^2 + n^2)/2$:

| Mode $(m,n)$ | $m^2+n^2$ | $f/f_{11}$ | Degeneracy    |
|-------------|---------|-----------|---------------|
| (1,1)       | 2       | 1.000     |               |
| (1,2)       | 5       | 2.500     | (2,1)         |
| (2,2)       | 8       | 4.000     |               |
| (1,3)       | 10      | 5.000     | (3,1)         |
| (2,3)       | 13      | 6.500     | (3,2)         |
| (1,4)       | 17      | 8.500     | (4,1)         |
| (3,3)       | 18      | 9.000     |               |
| (2,4)       | 20      | 10.000    | (4,2)         |

### Key physical insight

The plate combines 2D geometry (dense spectrum, nodal-line patterns like the membrane) with bending-stiffness physics (superlinear frequency growth like the bar). The individual factors $m^2$ and $n^2$ in the frequency formula reflect the same $\omega \propto k^2$ scaling as the 1D bar, applied independently in each spatial direction.

### Symmetry and degeneracy

For a square plate, modes $(m,n)$ and $(n,m)$ have the same frequency but different spatial patterns. This **degeneracy** is a consequence of the square symmetry: rotating the plate 90° maps one mode onto the other. Breaking the symmetry (making $a \neq b$) splits the degenerate pairs: their frequencies separate, and the timbre changes. This is audible in the demo when adjusting the aspect ratio.

### Strike-position dependence

Striking at $(x_s, y_s)$ excites mode $(m,n)$ with amplitude $\propto \sin(m\pi x_s/a)\sin(n\pi y_s/b)$. Striking the center of a square plate gives $\sin(m\pi/2)\sin(n\pi/2)$, which is nonzero only when both $m$ and $n$ are odd. This suppresses a large fraction of the modes, producing a distinctly simpler sound.

## 8. Comparing the Four Models

The table below summarizes the essential physics of each model. The "dispersion" column gives the relationship between angular frequency $\omega$ and wavenumber $k$ from the governing PDE. The "frequency formula" column shows how frequencies depend on mode indices in the analytical solution.

| Property        | String       | Bar           | Membrane      | Plate         |
|-----------------|--------------|---------------|---------------|---------------|
| Dimension       | 1D           | 1D            | 2D            | 2D            |
| Restoring force | Tension      | Bending       | Tension       | Bending       |
| Operator order  | 2nd          | 4th           | 2nd           | 4th           |
| Dispersion      | $\omega \propto k$  | $\omega \propto k^2$ | $\omega \propto |\mathbf{k}|$  | $\omega \propto |\mathbf{k}|^2$ |
| Spectrum        | Harmonic     | Inharmonic    | Inharmonic    | Inharmonic    |
| Frequency formula | $f_n = nf_1$ | $f_n \propto \beta_n^2$ | $f_{mn} \propto j_{mn}$ | $f_{mn} \propto (m/a)^2 + (n/b)^2$ |
| Mode shapes     | $\sin(n\pi x/L)$ | cosh/cos/sinh/sin | $J_m(j_{mn}r/R)\cos(m\theta)$ | $\sin\sin$ products |
| Sound character | Musical pitch | Metallic/bell | Drum-like     | Bright, complex |

The key distinction is the **order of the spatial operator**. Second-order (tension) gives $\omega \propto k$; fourth-order (bending) gives $\omega \propto k^2$. This single difference accounts for why strings sound harmonic and bars sound metallic.

## 9. Damping

Real vibrations dissipate energy, causing modes to decay over time. Two damping models are commonly used in physically based sound synthesis.

### The van den Doel & Pai internal friction model

Van den Doel and Pai [1998] use a material-based damping model where the decay time of each mode is:

$$\tau_n = \frac{1}{\pi f_n \tan\varphi}$$

where $\varphi$ is the **internal friction parameter**, an approximate material constant. This gives a decay rate $\alpha_n = 1/\tau_n = \pi f_n \tan\varphi = \omega_n\tan\varphi/2$ that is proportional to frequency. Equivalently, the damping ratio $\xi_n = \alpha_n/\omega_n = \tan\varphi/2$ is the **same for all modes**. Higher modes decay faster in absolute terms (larger $\alpha_n$), but all modes complete the same number of oscillation cycles before decaying by a given factor.

This model has a single parameter $\varphi$ per material, making it simple and physically motivated: $\varphi$ is small for metals (long ring times) and large for wood and rubber (short, dull sounds).

### Rayleigh damping

The FEM-based modal vibration framework (Zheng [2016]) uses **Rayleigh damping**, where the damping matrix is a linear combination of the mass and stiffness matrices: $\mathbf{D} = \alpha\mathbf{M} + \beta\mathbf{K}$. In the modal domain, the damping ratio of mode $k$ with angular frequency $\omega_k$ is:

$$\xi_k = \frac{\alpha}{2\omega_k} + \frac{\beta\omega_k}{2}$$

and the decay rate is $\alpha_k = \xi_k \omega_k = \alpha/2 + \beta\omega_k^2/2$.

This has two parameters:

- $\alpha$ (mass-proportional damping): contributes a **constant** decay rate $\alpha/2$ to all modes. The damping *ratio* $\xi_k = \alpha/(2\omega_k)$ decreases with frequency: low-frequency modes have higher damping ratios and complete fewer oscillation cycles before decaying, while high-frequency modes ring through many more cycles.
- $\beta$ (stiffness-proportional damping): contributes a decay rate that grows with $\omega_k^2$. High-frequency modes decay much faster, both in absolute terms and relative to their oscillation period.

### How the two models differ

The internal friction model and Rayleigh damping have fundamentally different frequency scaling and cannot be made equivalent:

- **Internal friction:** $\alpha_n \propto \omega$ (decay rate linear in frequency; constant damping ratio)
- **Rayleigh, $\alpha$-only:** $\alpha_k = \alpha/2$ (decay rate constant; damping ratio $\propto 1/\omega$)
- **Rayleigh, $\beta$-only:** $\alpha_k \propto \omega^2$ (decay rate quadratic in frequency; damping ratio $\propto \omega$)

No single choice of $\alpha$ and $\beta$ can reproduce the constant damping ratio of the internal friction model across all frequencies. However, by choosing $\alpha$ and $\beta$ so that $\xi(\omega) = \alpha/(2\omega) + \beta\omega/2$ passes through a target damping ratio at two chosen frequencies, one can approximate constant-$\xi$ behavior over a limited frequency band. This is standard practice in structural dynamics (see Chopra, *Dynamics of Structures*, §11.4).

Both models share the qualitative property that $\beta$-dominated Rayleigh damping preferentially attenuates high-frequency modes, which is the perceptually dominant effect in most struck-object sounds.

### Perceptual consequences

The demo lets you sweep $\alpha$ and $\beta$ independently:

- **High $\beta$, low $\alpha$:** high modes die quickly, low modes ring. The sound is "warm" or "muffled." Characteristic of wood and soft materials.
- **Low $\beta$, low $\alpha$:** all modes ring for a long time. The sound is "bell-like," "glassy," or "ringing." Characteristic of metal and glass.
- **High $\alpha$, low $\beta$:** all modes decay at the same rate in seconds, but low-frequency modes complete fewer oscillation cycles before fading, making them sound relatively damped while high-frequency modes ring through many cycles. The net effect is an unnatural "tinny" quality.
- **High $\alpha$, high $\beta$:** all modes die rapidly. The sound is a short "thud." Characteristic of rubber, heavy damped materials, or objects resting on a soft surface.

## 10. From Analytical Models to Numerical Computation

The four analytical models above apply to idealized geometries with simple boundary conditions. Real objects (coffee mugs, engine parts, wine glasses) have arbitrary shapes. For these, the same physics applies, but the eigenvalue problem must be solved numerically.

The procedure, following Zheng [2016] and the SIGGRAPH course notes:

1. **Mesh the object** with a volumetric mesh (typically tetrahedra). The surface geometry and interior volume are discretized into elements.

2. **Assemble the mass and stiffness matrices** ($\mathbf{M}$ and $\mathbf{K}$) from element contributions, using the finite element method with linear elasticity. This is the numerical analog of specifying the operator $\mathcal{L}$ and the domain.

3. **Solve the generalized eigenvalue problem** $\mathbf{K}\mathbf{U} = \mathbf{M}\mathbf{U}\boldsymbol{\Lambda}$, retaining only modes with frequencies below 20 kHz (the limit of human hearing). This replaces the analytical eigenfunctions and eigenvalues with numerically computed ones.

4. **Synthesize sound** by summing decaying sinusoids weighted by the mode shapes at the strike point, exactly as in the analytical case.

The physics is identical. The analytical models provide exact solutions for simple shapes; the numerical method provides approximate solutions for arbitrary shapes. Understanding the analytical cases builds the intuition needed to interpret numerical results.

## 11. The Full Sound Synthesis Formula

Combining excitation, modal decomposition, and damping, the sound pressure from a single impact at position $\mathbf{x}_s$, heard at a listener position $\mathbf{x}_\ell$, is:

$$p(t) = \sum_{n=1}^{N} a_n\,e^{-\alpha_n t}\sin(2\pi f_n t), \quad t \geq 0 \tag{14}$$

where:

- $f_n$ is the frequency of mode $n$
- $\alpha_n$ is the decay rate (from internal friction or Rayleigh damping)
- $a_n \propto \psi_n(\mathbf{x}_s)$ is the excitation amplitude, set by the mode shape value at the strike point

In a full pipeline (as in Zheng & James [2010] and the FoleyAutomatic system of van den Doel, Kry & Pai [2001]), the amplitude $a_n$ also includes an **acoustic transfer** factor $T_n(\mathbf{x}_\ell)$ that accounts for the directional radiation pattern of each mode. The demo omits acoustic transfer for clarity.

For ongoing contact forces $f(t)$ rather than instantaneous impulses, each mode responds via convolution with its impulse response, efficiently computed using a two-pole IIR digital filter (see Zheng [2016], Section 1).

## 12. Connections to Later Topics in the Course

Modal analysis is the foundation for most physically based sound methods:

- **Contact sound**: impacts excite modes via $\psi_n(\mathbf{x}_s)$, with force profiles from Hertz contact theory determining the spectral bandwidth of the excitation.
- **Fracture sound**: new fragments have new mode sets; fracture events create impulsive excitation across the new modes.
- **Bubble acoustics**: a bubble in water is a single-mode oscillator (the Minnaert resonance) with frequency $f = (1/2\pi R)\sqrt{3\gamma P_0/\rho}$. This is modal analysis with $N = 1$.
- **Thin shells**: nonlinear modal coupling transfers energy between modes, producing the rich evolving sound of gongs and cymbals.
- **Acoustic transfer**: the listener-direction-dependent radiation amplitude of each mode, computed via BEM or multipole expansion.

## References

- A.A. Shabana, *Theory of Vibration, Volume II: Discrete and Continuous Systems*, Springer-Verlag, 1990.
- P.M. Morse, *Vibration and Sound*, American Institute of Physics, 4th edition, 1976. Chapters 3-5.
- R.D. Blevins, *Formulas for Natural Frequency and Mode Shape*, Van Nostrand Reinhold, 1979. Table 8-1 (beam eigenvalues).
- A.W. Leissa, *Vibration of Plates*, NASA SP-160, 1969.
- K. van den Doel and D.K. Pai, "The Sounds of Physical Shapes," *Presence*, 7(4):382-395, 1998.
- K. van den Doel, P.G. Kry, and D.K. Pai, "FoleyAutomatic: Physically-Based Sound Effects for Interactive Simulation and Animation," *Proc. ACM SIGGRAPH*, 2001.
- C. Zheng and D.L. James, "Rigid-Body Fracture Sound with Precomputed Soundbanks," *ACM TOG (SIGGRAPH 2010)*, 29(3), 2010.
- C. Zheng, "Modal Vibration," in *Physically Based Sound for Computer Animation and Virtual Environments*, SIGGRAPH 2016 Course Notes.
