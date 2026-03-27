# SmokeVolume — Clean-Room Specification

> **Purpose** — WebGPU 3D Euler-fluid simulation implementing a Jos Stam
> *Stable Fluids* variant with MacCormack advection, vorticity confinement,
> Red-Black SOR pressure solve, and optional SDF boundary enforcement.
> All GPU work is dispatched via Three.js TSL compute nodes.

---

## 1. Public API

### 1.1 Constructor

```ts
new SmokeVolume(options?: SmokeVolumeOptions)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `simRes` | `int` | 96 | Resolution (cubed) of velocity, pressure, curl, and divergence grids |
| `dyeRes` | `int` | 96 | Resolution (cubed) of the density (dye) grid |
| `iterations` | `int` | 40 | Number of Jacobi/SOR pressure-solve iterations per step |
| `densityDissipation` | `float` | 0.995 | Multiplier applied to density each advection step |
| `velocityDissipation` | `float` | 0.98 | Multiplier applied to velocity each advection step |
| `pressureDissipation` | `float` | 0.98 | Multiplier applied to pressure at the start of the pressure-clear pass |
| `curlStrength` | `float` | 20 | Vorticity confinement strength |
| `pressureFactor` | `float` | 1/6 (≈0.1667) | Jacobi relaxation weight = 1/(2·dimensions) for 3D |
| `radius` | `float` | 0.2 | Gaussian radius for splat injection |
| `useBoundaries` | `bool` | true | Enable Neumann-style wall boundaries at domain edges in the divergence pass |
| `neighborStride` | `float` | 1 | Texel stride for finite-difference stencils (values > 1 give coarser/faster stencils) |
| `speedFactor` | `float` | 1 | Time-scale factor; adjusts iteration count inversely |
| `buoyancyStrength` | `float` | 0 | Upward buoyancy force proportional to density |
| `sdfVolumeConstraint` | `object \| null` | null | Optional SDF-based obstacle. Contains `.sdfGenerator` with `sdfTexture`, `inverseBoundsMatrix`, `boundsMatrix`, and `.threshold` |

### 1.2 Methods

| Method | Signature | Description |
|---|---|---|
| `step(renderer, dt?)` | `(WebGPURenderer, number?) → void` | Run one full simulation tick. If `dt` is provided it overrides `deltaTime`. |
| `addSplat(x, y, z, fx?, fy?, fz?)` | `(float, float, float, float=0, float=0, float=0) → this` | Queue a splat at normalized coords `(x,y,z)∈[0,1]³` with force vector `(fx,fy,fz)`. |
| `clearPressure(renderer)` | `(WebGPURenderer) → void` | Run one pressure-clear pass (multiply by `pressureDissipation`). |
| `setDebugPasses(passes)` | `(object) → this` | Merge into `debug` flags to enable/disable individual passes. |
| `setUseBoundaries(enabled)` | `(bool) → this` | Toggle wall boundary conditions at runtime. |
| `dispose()` | `() → void` | Release all GPU textures. |

### 1.3 Texture Accessors

All return `THREE.Storage3DTexture` (the *current read-side* of the ping-pong pair, or the single texture for non-ping-pong fields).

| Accessor | Resolution | Channels | Contents |
|---|---|---|---|
| `getDensityTexture3D()` | `dyeRes³` | RGBA (RGB used) | Smoke/dye density |
| `getVelocityTexture3D()` | `simRes³` | RGBA (RGB = xyz velocity) | Velocity field |
| `getPressureTexture3D()` | `simRes³` | RGBA (R = scalar pressure) | Pressure field |
| `getDivergenceTexture3D()` | `simRes³` | RGBA (R = scalar divergence) | Velocity divergence |
| `getCurlTexture3D()` | `simRes³` | RGBA (RGB = xyz curl) | Curl (vorticity) vector field |

### 1.4 Exposed Uniforms (TSL `uniform` nodes)

All are live-writable via `.value`:

- `radius` — splat Gaussian radius
- `curlStrength` — vorticity confinement magnitude
- `densityDissipation`, `velocityDissipation`, `pressureDissipation`
- `neighborStride` — finite-difference stencil stride
- `deltaTime` — simulation timestep (default 0.004)
- `buoyancyStrength` — buoyancy force magnitude
- `pressureFactor` — Jacobi weight
- `sorOmega` — SOR over-relaxation factor
- `point3D` — current splat position (`vec3`, default far off-screen)
- `force3D` — current splat force (`vec3`)

### 1.5 Debug Flags

`debug` object with boolean fields (all default `true` except `runAdvectVelocity` which defaults to `false`):

```
runCurl, runVorticity, runDivergence, runPressureClear,
runPressureJacobi, runProjection, runAdvectVelocity,
runAdvectDensity, runBuoyancy, autoStep
```

Setting a flag to `false` skips that pass in `step()`.

---

## 2. 3D Texture Layout

### 2.1 Ping-Pong Textures (`Storage3DTexturePingPong`)

Used for fields that are both read and written each step: **velocity**, **density**, **pressure**.

Each contains two `Storage3DTexture` instances (`read` and `write`) and a boolean `phase`. A `swap()` toggles `phase`; the getter returns the current-phase read side.

| Config | Value |
|---|---|
| Format | `RGBAFormat` |
| Type | `HalfFloatType` (16-bit float per channel) |
| Min/Mag Filter | `LinearFilter` |
| Wrap | `ClampToEdgeWrapping` on all three axes |
| Mipmaps | Disabled |

### 2.2 Single-Target Textures (`Storage3DTextureTarget`)

Used for read-only intermediate fields: **curl**, **divergence**, and the two MacCormack temp buffers.

| Config | Value |
|---|---|
| Format | `RGBAFormat` |
| Type | `HalfFloatType` |
| Min/Mag Filter | `LinearFilter` (overridden in constructor; default for targets is `NearestFilter` but SmokeVolume passes `LinearFilter`) |
| Wrap | `ClampToEdgeWrapping` |
| Mipmaps | Disabled |

### 2.3 Resolution Summary

| Texture | Width × Height × Depth |
|---|---|
| `_velocity3D` | `simRes × simRes × simRes` |
| `_pressure3D` | `simRes × simRes × simRes` |
| `_curl3D` | `simRes × simRes × simRes` |
| `_divergence3D` | `simRes × simRes × simRes` |
| `_macCormackTemp3D` | `simRes × simRes × simRes` |
| `_density3D` | `dyeRes × dyeRes × dyeRes` |
| `_macCormackTempDensity` | `dyeRes × dyeRes × dyeRes` |

---

## 3. Compute Dispatch Details

All compute passes use a **workgroup size of [4, 4, 4]**.

Dispatch dimensions for a grid of size `(W, H, D)`: `[⌈W/4⌉, ⌈H/4⌉, ⌈D/4⌉]`.

Every compute kernel begins with two early-exit guards:
1. `instanceIndex >= W*H*D` → return
2. `globalId.{x,y,z} >= gridDim.{x,y,z}` → return

This handles overrun from the padded dispatch.

For advection passes (forward/backward), threads are indexed by **instanceIndex** and manually decompose into 3D coordinates: `x = instanceIndex % W`, `y = (instanceIndex / W) % H`, `z = instanceIndex / (W*H)`.

---

## 4. Navier–Stokes Algorithm — Full Pipeline

The simulation implements a variant of **Jos Stam's Stable Fluids** (1999) extended to 3D with MacCormack advection and Red-Black SOR pressure solving.

### 4.0 Per-Step Preamble

Before the pipeline, all queued splats are injected (see §5).

### Pipeline Order (inside `step()`):

```
1. Splat injection              (queued splats)
2. Curl computation             (if debug.runCurl)
3. Vorticity confinement        (if debug.runVorticity)
4. Buoyancy                     (if debug.runBuoyancy)
5. Divergence computation       (if debug.runDivergence)
6. Pressure clear               (if debug.runPressureClear)
7. Pressure solve (Jacobi/SOR)  (if debug.runPressureJacobi) × iterations
8. Pressure projection          (if debug.runProjection)
9. [SDF boundary enforcement]   (if sdfVolumeConstraint set)
10. Velocity advection (MacCormack)  (if debug.runAdvectVelocity)
11. Density advection (MacCormack)   (if debug.runAdvectDensity)
```

Each pass that writes to a ping-pong texture reads from the current side and writes to the other, then calls `swap()`.

---

### 4.1 Curl Computation

**Reads:** velocity (current read side)  
**Writes:** curl texture (single target — always the same texture)  
**Grid:** simRes³

Computes the curl (vorticity) ω = ∇ × **v** using central finite differences with stride `s = max(neighborStride, 1)`:

```
For each voxel at position P:
  Sample velocity at 6 axis-aligned neighbors (clamped to grid bounds):
    vL = vel(P - s·x̂),  vR = vel(P + s·x̂)
    vD = vel(P - s·ŷ),  vU = vel(P + s·ŷ)
    vB = vel(P - s·ẑ),  vF = vel(P + s·ẑ)

  h = 0.5 / s

  ωx = (vF.y - vB.y) · h  −  (vU.z - vD.z) · h
  ωy = (vR.z - vL.z) · h  −  (vF.x - vB.x) · h   [wait — see actual]
```

Actual formula (matching the source exactly):
```
h = 0.5 / s

P_ = vF.y - vB.y     (∂vy/∂z)
L_ = vU.z - vD.z     (∂vz/∂y)
K_ = vR.z - vL.z     (∂vz/∂x)
tt = vF.x - vB.x     (∂vx/∂z)
st = vU.x - vD.x     (∂vx/∂y)
it = vR.y - vL.y     (∂vy/∂x)

curl = vec3(
  (L_ - P_) · h,     // = (∂vz/∂y - ∂vy/∂z) · h
  (tt - K_) · h,     // = (∂vx/∂z - ∂vz/∂x) · h
  (it - st) · h      // = (∂vy/∂x - ∂vx/∂y) · h
)
```

This is the standard 3D curl: **ω = ∇ × v**.

Store `vec4(curl, 1)` into the curl texture.

**Note:** The curl pass does **not** swap the velocity ping-pong — it only reads.

---

### 4.2 Vorticity Confinement

**Reads:** curl texture, velocity (current read side)  
**Writes:** velocity (write side) → then swap  
**Grid:** simRes³

Applies vorticity confinement to counteract numerical dissipation of rotational motion:

```
For each voxel at position P:
  Sample curl magnitude at 6 neighbors:
    |ω_L|, |ω_R|, |ω_D|, |ω_U|, |ω_B|, |ω_F|
  Also load ω_center (curl at P)

  Compute η (gradient of curl magnitude):
    η = vec3(
      |ω_U| - |ω_D|,
      |ω_R| - |ω_L|,
      |ω_F| - |ω_B|
    ) · (0.5 / s)

  Normalize: N̂ = η / (|η| + 1e-4)

  Confinement force: F_conf = cross(N̂, ω_center) · curlStrength · deltaTime

  New velocity = vel(P) + F_conf
```

Store `vec4(newVelocity, 1)`. Swap velocity.

---

### 4.3 Buoyancy

**Reads:** density (current read side), velocity (current read side)  
**Writes:** velocity (write side) → then swap  
**Grid:** simRes³

Applies a vertical buoyancy force proportional to density:

```
For each voxel:
  ρ = density.x (red channel only)
  v = velocity.xyz
  buoyancyForce = (ρ - 0.5) · buoyancyStrength · deltaTime
  newVelocity = v + vec3(0, buoyancyForce, 0)
```

Store `vec4(newVelocity, 1)`. Swap velocity.

**Note:** Density is loaded via `textureLoad` at integer coordinates from the density grid. When `buoyancyStrength = 0` (default), this pass produces no change but is still dispatched.

---

### 4.4 Divergence Computation

**Reads:** velocity (current read side)  
**Writes:** divergence texture (single target)  
**Grid:** simRes³

Computes the scalar divergence ∇·**v**:

```
For each voxel at P:
  s = max(neighborStride, 1)
  Load velocity components at 6 neighbors (clamped):
    vL.x = vel(P - s·x̂).x
    vR.x = vel(P + s·x̂).x
    vD.y = vel(P - s·ŷ).y
    vU.y = vel(P + s·ŷ).y
    vB.z = vel(P - s·ẑ).z
    vF.z = vel(P + s·ẑ).z

  // Load current velocity for boundary handling
  vC = vel(P).xyz

  if useBoundaries:
    // Neumann-style no-penetration at domain walls:
    if P.x < s:           vL.x = -vC.x
    if P.x + s > max.x:   vR.x = -vC.x
    if P.y + s > max.y:   vU.y = -vC.y
    if P.y < s:           vD.y = -vC.y
    if P.z + s > max.z:   vF.z = -vC.z
    if P.z < s:           vB.z = -vC.z

  div = 0.5 · (vR.x - vL.x + vU.y - vD.y + vF.z - vB.z) / s
```

Store `vec4(div, 0, 0, 1)`. Does **not** swap velocity.

---

### 4.5 Pressure Clear

**Reads:** pressure (current read side)  
**Writes:** pressure (write side) → then swap  
**Grid:** simRes³

Simply multiplies existing pressure by `pressureDissipation`:

```
newPressure = textureLoad(pressure, P) * pressureDissipation
```

This damps the pressure field before the iterative solve, preventing accumulation across frames.

---

### 4.6 Pressure Solve — Red-Black SOR Jacobi

**Reads:** divergence texture, pressure (current read side)  
**Writes:** pressure (write side) → then swap  
**Grid:** simRes³  
**Iterations:** `this.iterations` (default 40), with **2 dispatches per iteration** (red then black)

Uses a **Red-Black Successive Over-Relaxation (SOR)** scheme:

```
For each iteration i in [0, iterations):
  // Red pass (parity = 0)
  pressureParity = 0
  dispatch pressure compute → swap

  // Black pass (parity = 1)
  pressureParity = 1
  dispatch pressure compute → swap
```

Each dispatch does:

```
For each voxel at P:
  s = max(neighborStride, 1)
  h² = s²

  // Red-black checkerboard: parity = (P.x/s + P.y/s + P.z/s) & 1
  cellParity = (P.x/s + P.y/s + P.z/s) & 1
  targetParity = round(pressureParity)  // 0 or 1

  if cellParity ≠ targetParity:
    // Not our color — copy through unchanged
    output = textureLoad(pressure_read, P)
    store and return

  // Sample 6 pressure neighbors (clamped)
  pL = pressure(P - s·x̂).x
  pR = pressure(P + s·x̂).x
  pD = pressure(P - s·ŷ).x
  pU = pressure(P + s·ŷ).x
  pB = pressure(P - s·ẑ).x
  pF = pressure(P + s·ẑ).x

  div = divergence(P).x
  pC  = pressure(P).x   // current center value

  // Jacobi update
  jacobi = pressureFactor · (pL + pR + pD + pU + pB + pF - div · h²)

  // SOR relaxation
  result = pC + sorOmega · (jacobi - pC)
```

Store `vec4(result, 0, 0, 1)`.

**SOR omega** is computed at construction time as:
```
ω = min(1.99, 2 / (1 + sin(π / simRes)))
```
This is the optimal SOR relaxation factor for a Poisson problem on a regular grid.

**Total dispatches per step:** `2 × iterations` (e.g., 80 for 40 iterations).

---

### 4.7 Pressure Projection (Gradient Subtraction)

**Reads:** pressure (current read side), velocity (current read side)  
**Writes:** velocity (write side) → then swap  
**Grid:** simRes³

Subtracts the pressure gradient from velocity to enforce incompressibility (Helmholtz-Hodge projection):

```
For each voxel at P:
  s = max(neighborStride, 1)
  Sample pressure at 6 neighbors (clamped):
    pL, pR, pD, pU, pB, pF

  gradP = vec3(pR - pL, pU - pD, pF - pB) · (0.5 / s)
  newVelocity = velocity(P) - gradP
```

Store `vec4(newVelocity, 1)`. Swap velocity.

**Note:** There are four variants of this pass to handle all combinations of pressure and velocity ping-pong phase. The correct one is selected based on both `_pressure3D.phase` and `_velocity3D.phase`.

---

### 4.8 SDF Boundary Enforcement (Optional)

**Reads:** velocity (current read side), SDF 3D texture  
**Writes:** velocity (write side) → then swap  
**Grid:** simRes³

Only dispatched when `sdfVolumeConstraint` is set. Written in raw WGSL (not TSL):

```
For each voxel:
  Convert grid coord → normalized [0,1]³ UVW → world space → SDF texture space
  Sample SDF value at that point

  if sdfValue > threshold:
    // Outside the SDF boundary → zero velocity
    velocity = vec3(0)
  else if sdfValue > threshold - 0.1:
    // Near boundary → project velocity onto tangent plane
    Compute SDF gradient via central differences → surface normal N̂
    tangentVel = vel - N̂ · dot(vel, N̂)
    velocity = mix(tangentVel · tangentProjection, vel · (1 - tangentProjection))
  else:
    // Deep inside → keep velocity unchanged
```

---

### 4.9 Velocity Advection — MacCormack

**Condition:** Only runs if `debug.runAdvectVelocity === true` (default: **false**)

Uses a two-pass **MacCormack advection** scheme for higher-order accuracy:

#### Pass A — Forward Advection (`ft` kernel)

**Reads:** velocity (current read via trilinear `.sample()`)  
**Writes:** `_macCormackTemp3D` texture  

```
For each voxel at integer index → normalized coord p = (idx + 0.5) / gridDim:
  texelSize = 1.0 / gridDim
  vel = velocity.sample(p).xyz      // trilinear lookup
  backtraced = clamp(p - deltaTime · vel · texelSize, [0,1]³)
  result = velocity.sample(backtraced) · velocityDissipation
```

Store result into temp texture.

#### Pass B — MacCormack Correction (`mt` kernel)

**Reads:** velocity (original, current read), temp texture (forward result)  
**Writes:** velocity (write side) → then swap  

```
For each voxel at p:
  vel_fwd = tempTex.sample(p).xyz
  forward = clamp(p + deltaTime · vel_fwd · texelSize, [0,1]³)

  // Error correction
  error = (velocity_original.sample(p) - tempTex.sample(forward)) · 0.5
  corrected = (tempTex.sample(p) + error) · velocityDissipation

  // Clamp to neighborhood min/max (6 axis-aligned neighbors)
  neighborMin = min of velocity_original.sample() at ±texelSize in each axis
  neighborMax = max of velocity_original.sample() at ±texelSize in each axis
  result = clamp(corrected, neighborMin, neighborMax)
```

The neighborhood clamping prevents overshoot artifacts inherent to MacCormack.

---

### 4.10 Density Advection — MacCormack

**Condition:** Runs if `debug.runAdvectDensity === true` (default: **true**)

Two-pass MacCormack, similar to velocity advection but **cross-resolution**: velocity is sampled from `simRes³` while density lives at `dyeRes³`.

#### Pass A — Forward Density Advection (`zt` kernel)

**Reads:** velocity texture (at simRes, sampled at density UV coords), density (current read)  
**Writes:** `_macCormackTempDensity`  

```
For each density voxel at index → normalized p = (idx + 0.5) / dyeGridDim:
  velTexelSize = 1.0 / velGridDim    // ← uses velocity grid's texel size
  vel = velocityTex.sample(p).xyz     // cross-resolution trilinear sample
  backtraced = clamp(p - deltaTime · vel · velTexelSize, [0,1]³)
  result = densityTex.sample(backtraced) · densityDissipation
```

#### Pass B — MacCormack Correction (reuses `mt` kernel)

Same MacCormack error-correction and neighborhood clamping as §4.9 Pass B, but operating on density textures at `dyeRes³`.

Swaps density after completion.

---

## 5. Splat Injection

Splats are queued via `addSplat(x, y, z, fx, fy, fz)` and consumed in FIFO order at the start of each `step()` call, before any simulation passes.

### 5.1 Velocity Splat

**Reads:** velocity (current read side)  
**Writes:** velocity (write side) → then swap  
**Grid:** simRes³

```
For each voxel:
  gridPos = (globalId + 0.5) / gridDim   // normalized [0,1]³
  delta = gridPos - splatPosition         // distance from splat center
  gaussian = exp(-dot(delta, delta) / (radius / 100))
  newVelocity = oldVelocity + splatForce · gaussian
```

Store `vec4(newVelocity, 1)`.

### 5.2 Density Splat

**Reads:** density (current read side)  
**Writes:** density (write side) → then swap  
**Grid:** dyeRes³

Same Gaussian but adds uniform density (no directional force):

```
For each voxel:
  gridPos = (globalId + 0.5) / gridDim
  delta = gridPos - splatPosition
  gaussian = exp(-dot(delta, delta) / (radius / 100))
  newDensity = oldDensity + vec3(gaussian)   // adds to all RGB channels equally
```

Store `vec4(newDensity, 1)`.

### 5.3 Gaussian Shape

The effective Gaussian standard deviation is:
```
σ² = radius / 100
```
With `radius = 0.2`, `σ² = 0.002`, giving a tight splat. The Gaussian is evaluated in normalized `[0,1]³` space, so the physical extent scales with the domain.

### 5.4 Multiple Splats

Each splat requires **two dispatches** (velocity + density), each followed by a swap. Splats are processed sequentially from the queue.

---

## 6. Boundary Conditions

### 6.1 Domain Wall Boundaries (`useBoundaries`)

When enabled, the **divergence pass** enforces **no-penetration (free-slip)** at the 6 faces of the cubic domain:

- For voxels within `neighborStride` of a face, the velocity component normal to that face is **reflected** (negated) when computing divergence.
- Specifically, if a voxel is near the left wall (x < s), then `vL.x` is replaced by `-vCenter.x`. Similarly for all 6 faces/axes.

This creates a "mirrored" velocity at the boundary that results in zero normal flow through domain walls.

### 6.2 Texture Clamping

All `textureLoad` neighbor lookups are clamped to `[0, gridDim-1]` using `clamp(coord, ivec3(0), gridDim - 1)`. This provides implicit Neumann (zero-gradient) boundaries for all stencil operations.

All `.sample()` (trilinear) lookups are clamped to `[0, 1]` and textures use `ClampToEdgeWrapping`.

### 6.3 SDF Boundaries

See §4.8. These operate on velocity only and are independent of `useBoundaries`.

---

## 7. Dissipation

### 7.1 Velocity Dissipation

- **Applied in:** velocity advection (MacCormack forward and backward passes)
- **Mechanism:** After backtrace sampling, the result is multiplied by `velocityDissipation`
- **Typical range:** [0.8, 1.0]. Values < 1 cause velocity to decay over time.

### 7.2 Density Dissipation

- **Applied in:** density advection (MacCormack forward and backward passes)
- **Mechanism:** Same multiplicative damping after backtrace sampling
- **Typical range:** [0.8, 1.0]. Values < 1 cause density to fade.

### 7.3 Pressure Dissipation

- **Applied in:** pressure clear pass (§4.5), **before** the Jacobi solve
- **Mechanism:** `newPressure = oldPressure × pressureDissipation`
- **Purpose:** Prevents pressure buildup across frames. Without this, the pressure field would grow unbounded.
- **Typical range:** [0.8, 1.0].

---

## 8. Defaults and Parameter Ranges

| Parameter | Default | Consumer Override | Typical Range | Notes |
|---|---|---|---|---|
| `simRes` | 96 | 64 | 32–128 | Must be integer. Higher = more accurate, slower. |
| `dyeRes` | 96 | 96 | 32–256 | Can differ from simRes for higher-quality visuals. |
| `iterations` | 40 | 40 | 1–100 | More = more accurate pressure. Internally clamped ≥ 1. |
| `densityDissipation` | 0.995 | 0.995 | [0.8, 1.0] | 1.0 = no dissipation. |
| `velocityDissipation` | 0.98 | 0.985 | [0.8, 1.0] | |
| `pressureDissipation` | 0.98 | 0.98 | [0.8, 1.0] | |
| `curlStrength` | 20 | 6 | [0, 80] | 0 disables vorticity confinement (if pass runs). |
| `pressureFactor` | 1/6 ≈ 0.1667 | 1/6 | Fixed at 1/6 for 3D | Jacobi relaxation weight = 1/(2·ndims). |
| `radius` | 0.2 | 0.2 | [0.01, 0.8] | Gaussian σ² = radius/100. |
| `useBoundaries` | true | true | bool | |
| `neighborStride` | 1 | 1 | [0.5, 4] | Clamped to ≥ 1 inside shaders. |
| `speedFactor` | 1 | 1 | > 0 | Adjusts iterations: `iterations = round(baseIterations / speedFactor)` |
| `buoyancyStrength` | 0 | 0 | [0, ∞) | 0 = disabled. Force = (density - 0.5) · strength · dt. |
| `deltaTime` | 0.004 | *(via step dt)* | > 0 | Simulation sub-step. |
| `sorOmega` | `min(1.99, 2/(1+sin(π/simRes)))` | *(computed)* | [1, 1.99] | Optimal SOR factor. For simRes=64: ≈ 1.903. |

---

## 9. Ping-Pong Bookkeeping

Each ping-pong texture pair has a `phase` boolean:
- `phase = true` → read from `.read`, write to `.write`
- `phase = false` → read from `.write`, write to `.read`

The simulation pre-builds **two variants** of every compute pass (one for each phase). At dispatch time, the correct variant is selected based on the current phase of each involved texture.

For the projection pass, which reads both pressure and velocity, there are **four variants** covering all phase combinations.

The texture accessors (`getDensityTexture3D()`, etc.) return the current read-side texture (i.e., `phase ? .read : .write`).

---

## 10. Algorithm Summary — Single Step

```
step(renderer, dt?):
  1. if dt provided: deltaTime.value = dt
  2. Drain splat queue → for each splat:
       a. Set point3D, force3D uniforms
       b. Dispatch velocity splat → swap velocity
       c. Dispatch density splat → swap density
  3. Compute curl of velocity → write curl texture
  4. Apply vorticity confinement → swap velocity
  5. Apply buoyancy → swap velocity
  6. Compute divergence of velocity → write divergence texture
  7. Clear pressure (multiply by pressureDissipation) → swap pressure
  8. For i in [0, iterations):
       a. Set pressureParity = 0, dispatch pressure (red) → swap pressure
       b. Set pressureParity = 1, dispatch pressure (black) → swap pressure
  9. Subtract pressure gradient from velocity → swap velocity
  10. [If SDF constraint: enforce SDF boundaries → swap velocity]
  11. [If runAdvectVelocity: MacCormack advect velocity (2 dispatches) → swap velocity]
  12. MacCormack advect density (2 dispatches) → swap density
```

**Total GPU dispatches per step** (default config, no SDF, velocity advection off):
- Splats: 2 per queued splat
- Curl: 1
- Vorticity: 1
- Buoyancy: 1
- Divergence: 1
- Pressure clear: 1
- Pressure Jacobi: 2 × iterations = 80
- Projection: 1
- Density advection: 2
- **Total: 88 + 2·(number of splats)**
