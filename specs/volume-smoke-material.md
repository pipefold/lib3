# VolumeSmokeNodeMaterial — Clean-Room Specification

> **Purpose**: This document specifies the complete rendering algorithm of `VolumeSmokeNodeMaterial`, a Three.js WebGPU TSL (Three Shading Language) material that renders volumetric smoke via raymarching through a unit cube. An engineer with no access to the original source should be able to reimplement this material from this spec alone.

---

## 1. Public API

### 1.1 Class: `VolumeSmokeNodeMaterial` (extends `THREE.NodeMaterial`)

**Static type identifier**: `"VolumeSmokeNodeMaterial"`

**Material configuration** (set in constructor, not user-configurable):

| Property | Value |
|---|---|
| `isVolumeSmokeNodeMaterial` | `true` |
| `forceSinglePass` | `true` |
| `transparent` | `true` |
| `depthWrite` | `false` |
| `side` | `THREE.BackSide` |

> **Why BackSide?** The mesh is a unit box. By rendering back-faces, the fragment shader fires for pixels where the camera looks *through* the box, which is the entry condition for raymarching.

### 1.2 Constructor Parameters

All parameters are passed as a single options object. Parameters that are already TSL nodes (`.isNode === true`) are used directly; scalars/vectors are wrapped in `uniform()`.

#### Required — 3D Textures

These must be `THREE.Data3DTexture` or compatible. They are wrapped via `texture3D(tex, null, 0)` internally.

| Parameter | Type | Description |
|---|---|---|
| `densityTexture` | `Data3DTexture` | RGB density field from fluid sim |
| `velocityTexture` | `Data3DTexture` | XYZ velocity field |
| `curlTexture` | `Data3DTexture` | XYZ curl field |
| `pressureTexture` | `Data3DTexture` | Scalar pressure field (in `.x`) |
| `divergenceTexture` | `Data3DTexture` | Scalar divergence field (in `.x`) |

#### Optional — Uniforms

| Parameter | TSL Type | Default | Description |
|---|---|---|---|
| `dyeTexelSize` | `vec3` | `Vector3(1/128, 1/128, 1/128)` | Texel size of density texture; used for gradient computation |
| `steps` | `float` | `160` | Number of primary ray steps |
| `lightDir` | `vec3` | `Vector3(-0.35, 0.9, 0.4)` | Direction TO light (not normalized internally until use) |
| `baseColor` | `color` | `Color(0x1f1f8b)` = `rgb(31,31,139)` | Dark tint for unlit smoke |
| `highlightColor` | `color` | `Color(0x97a3b5)` = `rgb(151,163,181)` | Bright tint for lit smoke |
| `lightColor` | `color` | `Color(0xf4f7ff)` = `rgb(244,247,255)` | Color of the directional light contribution |
| `ambientLight` | `float` | `0.35` | Ambient light intensity |
| `lightStrength` | `float` | `1.5` | Directional light multiplier |
| `rimStrength` | `float` | `1.0` | Rim-light intensity |
| `densityBoost` | `float` | `5.0` | Multiplier on sampled density |
| `absorption` | `float` | `6.0` | Beer's law absorption coefficient |
| `curlInfluence` | `float` | `0.3` | How much curl magnitude affects lighting/color |
| `velocityInfluence` | `float` | `0.25` | How much velocity magnitude affects lighting/color |
| `pressureInfluence` | `float` | `0.15` | How much pressure affects lighting/color |
| `divergenceInfluence` | `float` | `0.15` | How much divergence affects lighting/color |
| `brightness` | `float` | `0.45` | Final color brightness multiplier |
| `alphaHash` | `bool` | `false` | Enable stochastic alpha hashing (discard-based) |
| `blueNoiseTexture` | `Texture` (2D) | 1×1 white pixel | Blue noise texture for ray jittering |
| `anisotropy` | `float` | `0.5` | Henyey-Greenstein asymmetry parameter *g* |
| `shadowSteps` | `float` | `4` | Number of steps for shadow ray marching |
| `shadowIntensity` | `float` | `0.4` | Shadow darkness multiplier |
| `adaptiveStepThreshold` | `float` | `0.05` | Density threshold below which step size shrinks adaptively |

### 1.3 Methods

| Method | Returns | Description |
|---|---|---|
| `getSmokeNode()` | TSL `vec4` node | Returns the smoke color+alpha node. Allows compositing with other effects before assigning to `outputNode`. |
| `useSmokeOutput()` | `void` | Sets `this.outputNode = this.smokeNode` and marks `needsUpdate`. |
| `setVolumeTextures({...})` | `void` | Hot-swaps any subset of the 5 volume textures. Triggers internal rebuild of the TSL graph. |

### 1.4 Intended Usage Pattern

```js
const material = new VolumeSmokeNodeMaterial({ densityTexture, ... });
const smokeNode = material.getSmokeNode(); // vec4(rgb, alpha)
material.outputNode = vec4(
  smokeNode.rgb.add(otherEffect.rgb),
  smokeNode.a
);

const mesh = new THREE.Mesh(new THREE.BoxGeometry(1,1,1), material);
mesh.scale.set(10, 10, 10);
```

The mesh MUST be a unit cube (−0.5 to +0.5 in local space). Scaling is applied via the world matrix. The material renders back-faces so the ray origin is computed correctly.

---

## 2. Raymarching Algorithm

### 2.1 Ray Setup (from `RaymarchingBox` helper)

The raymarching infrastructure is a reusable TSL function, not part of the material itself.

1. **Camera position in local space**: Transform the world-space camera position into the model's local space:
   ```
   localCamPos = modelWorldMatrixInverse * vec4(cameraPosition, 1)
   ```
   This is stored as a `varying`.

2. **Ray direction**: Compute from the current fragment's geometry position (which is a back-face vertex):
   ```
   rayDir = normalize(positionGeometry - localCamPos)
   ```
   Also stored as a `varying`.

3. **Ray–AABB intersection**: Intersect the ray with the unit box `[-0.5, +0.5]³`:
   ```
   invDir = 1.0 / rayDir
   t0 = (-0.5 - localCamPos) * invDir
   t1 = (+0.5 - localCamPos) * invDir
   tNear = max(min(t0, t1))  // component-wise min, then max of xyz
   tFar  = min(max(t0, t1))  // component-wise max, then min of xyz
   ```
   If `tNear > tFar`, **discard** the fragment (ray misses the box).
   Clamp: `tNear = max(tNear, 0)` (camera may be inside the box).

4. **Step size**: Based on the shortest axis-aligned traversal distance, divided by `steps`:
   ```
   axisLengths = abs(1.0 / rayDir)   // vec3
   baseStepSize = min(axisLengths.x, axisLengths.y, axisLengths.z) / steps
   ```
   This ensures the step size adapts to the ray angle.

5. **March**: Initialize `positionRay = localCamPos + tNear * rayDir`, then loop from `tNear` to `tFar` with a float loop, incrementing the parametric `t` by `baseStepSize` each iteration and advancing `positionRay += rayDir * baseStepSize`.

### 2.2 Blue Noise Jittering

Before the first sample, the ray start is jittered to eliminate banding:

1. Compute screen UV: `screenUV = positionWorld.xy * 0.5 + 0.5`
2. Sample blue noise: `jitter = blueNoiseTexture.sample(screenUV * 4).x`
3. On the **first step only** (`stepIndex == 0`), offset the sample position:
   ```
   jitteredPosition = positionRay + jitter * invSteps * 0.5
   ```
   Subsequent steps use `positionRay` unmodified.

### 2.3 Adaptive Step Size

The step size adapts based on local density to save work in empty regions:

```
adaptiveFactor = density > adaptiveStepThreshold
    ? 1.0
    : density / adaptiveStepThreshold
currentStepSize = invSteps * adaptiveFactor
```

Where `invSteps = 1.0 / steps`. When density is near zero, steps become very small (effectively skipping faster through empty space since the absorption contribution is negligible). When density is above the threshold, full-size steps are taken.

### 2.4 Early Termination

If accumulated transmittance drops below `0.02`, the ray breaks out of the loop. This means once the smoke is ~98% opaque, no further samples are taken.

---

## 3. Density Sampling

### 3.1 Base Density

At each ray step:

1. Convert ray position to UVW texture coordinates: `uvw = saturate(positionRay + 0.5)` (maps `[-0.5, +0.5]` → `[0, 1]`)
2. Sample the density 3D texture: `densitySample = densityTex.sample(uvw).rgb`
3. Convert to scalar via luminance: `densityScalar = saturate(dot(densitySample, vec3(0.299, 0.587, 0.114)) * densityBoost)`

The luminance vector `(0.299, 0.587, 0.114)` is the standard Rec.601 luma. The RGB density field is collapsed to a single scalar.

### 3.2 Density Threshold

If `densityScalar <= 0.0005`, the entire lighting computation is skipped for this step (empty space optimization).

### 3.3 Auxiliary Field Sampling

When density is above the threshold, four additional fields are sampled at the same UVW:

| Field | Sampling | Derived Scalar |
|---|---|---|
| Velocity | `velocityTex.sample(uvw).xyz` | `velocityMag = length(velocity)` |
| Curl | `curlTex.sample(uvw).xyz` | `curlMag = length(curl)` |
| Pressure | `pressureTex.sample(uvw).x` | `pressureAbs = abs(pressure)` |
| Divergence | `divergenceTex.sample(uvw).x` | `divergenceAbs = abs(divergence)` |

These do **not** modify the density value directly. Instead, they influence:
- Lighting intensity (added to the lighting sum)
- Color mixing (shift between base and highlight colors)
- A small absorption perturbation from curl

### 3.4 Density Gradient (Surface Normal Approximation)

A central-difference gradient of the density field is computed using `dyeTexelSize` as the offset:

```
gradOffset = dyeTexelSize * 1.0

gradient.x = luminance(density(uvw + (gradOffset.x, 0, 0))) - luminance(density(uvw - (gradOffset.x, 0, 0)))
gradient.y = luminance(density(uvw + (0, gradOffset.y, 0))) - luminance(density(uvw - (0, gradOffset.y, 0)))
gradient.z = luminance(density(uvw + (0, 0, gradOffset.z))) - luminance(density(uvw - (0, 0, gradOffset.z)))

normal = normalize(-gradient)
```

The negative gradient points from high density toward low density — i.e., outward from the smoke surface, like a surface normal.

---

## 4. Lighting Model

The lighting model has five components that are summed into a single `lighting` scalar per sample.

### 4.1 Directional Light with Shadows

```
lightDirN = normalize(lightDir)
directional = saturate(dot(normal, lightDirN))
shadowFactor = shadowRaymarch(positionRay)  // see §4.2
phaseFunction = HenyeyGreenstein(cosTheta, anisotropy)  // see §6

shadedDirectional = directional * shadowFactor * (phaseFunction * 4.0)
```

The `* 4.0` on the phase function is a fixed scaling factor to compensate for the normalization of Henyey-Greenstein.

### 4.2 Shadow Ray Marching

For each primary ray sample, a secondary ray is cast toward the light to estimate self-shadowing:

1. **Direction**: Along `lightDirN` (normalized light direction)
2. **Steps**: `shadowSteps` (default 4, typical range 2–8)
3. **Step size**: `shadowStep = 1.0 / shadowSteps`
4. **Accumulation**: Start with `shadow = 1.0`, then for each step `i`:
   ```
   marchUVW = saturate(positionRay + lightDirN * (i * shadowStep) + 0.5)
   shadowDensity = luminance(densityTex.sample(marchUVW).rgb) * densityBoost
   shadowAbs = shadowDensity * absorption * shadowIntensity * shadowStep
   shadow *= exp(-shadowAbs)
   ```
5. Return `saturate(shadow)`

This is Beer's law applied along the light ray. `shadowIntensity` controls how much density attenuates the light.

### 4.3 Ambient Light

```
lighting += ambientLight
```

A constant ambient term, added directly.

### 4.4 Rim Light

```
viewDir = normalize(-positionRay)
rim = pow(1.0 - abs(dot(normal, viewDir)), 2.0)

lighting += rim * rimStrength
```

This is a Fresnel-like rim effect: strongest where the density normal is perpendicular to the view direction (silhouette edges).

### 4.5 Flow-Field Lighting Contributions

The four auxiliary fields each contribute to lighting with their own influence scalars and smoothstep mappings:

| Field | Remapping | Contribution to `lighting` |
|---|---|---|
| Curl | `D = saturate(curlMag * 0.1) * curlInfluence` | `+D` |
| Velocity | `V = saturate(velocityMag * 0.5) * velocityInfluence` | `+V` |
| Pressure | `p = saturate(pressureAbs * 2.0)`, then Hermite smoothstep `p² * (3 - 2p)`, then `P = hermite * pressureInfluence` | `+P` |
| Divergence | `d = saturate(divergenceAbs * 2.0)`, then Hermite smoothstep `d² * (3 - 2d)`, then `Dv = hermite * divergenceInfluence` | `+Dv` |

Pressure and divergence use a smoothstep (Hermite) remap; curl and velocity use a linear remap.

### 4.6 Final Lighting Sum

```
lighting = ambientLight
         + shadedDirectional * lightStrength
         + rim * rimStrength
         + D   // curl contribution
         + V   // velocity contribution
         + P   // pressure contribution
         + Dv  // divergence contribution
```

---

## 5. Anisotropic Scattering — Henyey-Greenstein Phase Function

The material uses the Henyey-Greenstein (HG) phase function to model anisotropic light scattering through the volume:

```
cosTheta = dot(viewDir, -lightDirN)   // angle between view and light
g = anisotropy                         // asymmetry parameter
g2 = g * g

denom = (1 + g2 - 2*g*cosTheta) ^ 1.5

HG = (1 / (4π)) * (1 - g2) / (denom + 0.0001)
```

Concretely in the code:
- The `1/(4π)` constant is approximated as `0.0795774715` (which is `1/(4π) ≈ 0.07958`)
- A small epsilon `1e-4` is added to the denominator to prevent division by zero
- The result is multiplied by `4.0` when applied (see §4.1), effectively making the normalization `1/π`

**Parameter behavior:**
- `g = 0`: Isotropic scattering (equal in all directions)
- `g > 0` (e.g., 0.5–0.8): **Forward scattering** — smoke glows when backlit (silver lining effect)
- `g < 0`: Back-scattering — smoke glows when front-lit
- Typical value: `0.5`–`0.6` for realistic cloud/smoke forward scattering

---

## 6. Blue Noise Dithering

### 6.1 Purpose

Blue noise jittering breaks up the visible banding artifacts inherent in fixed-step raymarching.

### 6.2 Texture

A 2D blue noise texture (typically 128×128, generated by `ComputeMipAwareBlueNoise`). If none is provided, a 1×1 white canvas texture is created as fallback (effectively no jittering).

### 6.3 Application

The noise is sampled once per pixel (not per step):

```
screenUV = positionWorld.xy * 0.5 + 0.5
jitter = blueNoiseTex.sample(screenUV * 4.0).x    // tiled 4× across screen
```

The `* 4.0` tiles the noise texture across the screen so the pattern is fine-grained.

This jitter value is applied only to the **first step** of the primary ray:
```
if (stepIndex == 0):
    samplePos = positionRay + jitter * (1/steps) * 0.5
else:
    samplePos = positionRay
```

The jitter shifts the ray start by up to half a step size, randomized per pixel. This is sufficient to decorrelate banding across neighboring pixels.

---

## 7. Color Computation

Color is computed per-sample through a multi-stage blending process.

### 7.1 Base-to-Highlight Color Blend

A `directionalTint` scalar determines the mix between `baseColor` and `highlightColor`:

```
directionalTint = saturate(shadedDirectional * 0.6 + D * 0.4)
blendedColor = mix(baseColor, highlightColor, directionalTint)
```

Where `D` is the curl lighting contribution. Well-lit areas with high curl shift toward `highlightColor`.

### 7.2 Light Color Overlay

A separate `lightOverlay` scalar controls how much `lightColor` is added:

```
flowScalar = saturate(D * 0.6 + V * 0.4 + P * 0.2 + Dv * 0.15)

lightOverlay = saturate(
    rim * (rimStrength * 0.6)
  + shadedDirectional * (lightStrength * 0.25)
  + flowScalar * 0.35
)

detailTint = saturate((blendedColor + lightColor * lightOverlay) * brightness)
```

### 7.3 Final Sample Color

The final color for each sample is modulated by the lighting sum:

```
sampleColor = detailTint * lighting
```

This is then accumulated into the output (see §8).

---

## 8. Alpha / Transmittance — Beer's Law Absorption Model

### 8.1 Per-Step Transmittance

The material uses Beer-Lambert law for physically-based light absorption through participating media.

For each step with non-zero density:

```
attenuation = exp(-(densityScalar * absorption * currentStepSize + D * 0.05))
```

Where:
- `densityScalar` is the boosted scalar density
- `absorption` is the global absorption coefficient
- `currentStepSize` is the (possibly adaptive) step size (`1/steps * adaptiveFactor`)
- `D * 0.05` is a small curl-based absorption perturbation (curl makes smoke slightly more opaque)

### 8.2 Front-to-Back Compositing

Accumulation uses standard front-to-back alpha compositing:

```
// Initialize before loop:
accumColor = vec3(0)
transmittance = 1.0

// Per step:
alpha = saturate(1.0 - attenuation)          // opacity of this slab
weight = transmittance * alpha                 // how much this slab contributes
accumColor += weight * sampleColor             // color weighted by remaining light
transmittance *= attenuation                   // reduce remaining transmittance
```

### 8.3 Final Output

```
finalAlpha = saturate(1.0 - transmittance)
output = vec4(accumColor, finalAlpha)
```

The RGB channels contain **pre-multiplied** color weighted by transmittance. The alpha channel is the total opacity.

### 8.4 Alpha Hash Mode (Optional)

When `alphaHash = true`, instead of transparent blending, the material uses stochastic alpha hashing for order-independent transparency:

```
if (alphaHash):
    if ((1.0 - transmittance) < alphaHashThreshold(lastPositionRay)):
        discard
```

The threshold function is a 3D alpha hash based on world position, using a multi-octave deterministic noise function with screen-space derivatives for anti-aliasing. This converts continuous opacity into a binary accept/reject, enabling the material to work with `transparent = false` and depth writes.

---

## 9. Parameter Ranges and Visual Effects

### 9.1 Density & Opacity

| Parameter | Practical Range | Low End | High End |
|---|---|---|---|
| `densityBoost` | 0.4 – 20 | Thin, ghostly smoke | Thick, opaque clouds |
| `absorption` | 0.4 – 20 | Translucent, light passes through easily | Dense, dark smoke; shadows are harsh |
| `steps` | 30 – 300 | Faster but banding artifacts visible | Smooth but expensive; 120–160 is typical |
| `adaptiveStepThreshold` | 0.01 – 0.2 | More steps in low-density regions (slower, smoother) | Aggressive skipping in sparse areas (faster, may miss wisps) |

### 9.2 Lighting

| Parameter | Practical Range | Low End | High End |
|---|---|---|---|
| `ambientLight` | 0.0 – 1.0 | Dark, dramatic (all contrast from directional light) | Flat, uniformly lit |
| `lightStrength` | 0.0 – 3.0 | Subtle directional shading | Blown-out highlights, strong light/dark contrast |
| `rimStrength` | 0.0 – 2.0 | No silhouette glow | Strong glowing edges |
| `lightDir` | unit vector | — | Direction the light is coming FROM (not toward); normalized internally |

### 9.3 Shadow Rays

| Parameter | Practical Range | Low End | High End |
|---|---|---|---|
| `shadowSteps` | 2 – 8 | Cheap, soft/inaccurate shadows | Expensive, sharper self-shadowing |
| `shadowIntensity` | 0.0 – 1.5 | No self-shadowing (flat lit) | Deep dark shadows in dense regions |

### 9.4 Scattering

| Parameter | Practical Range | Low End | High End |
|---|---|---|---|
| `anisotropy` | -0.5 – 0.9 | Back-scatter / isotropic | Strong forward scatter (silver lining when backlit) |

### 9.5 Flow-Field Influences

These parameters control how much the fluid simulation's auxiliary fields affect the visual result. They primarily modulate lighting intensity and color, not density.

| Parameter | Practical Range | Effect |
|---|---|---|
| `curlInfluence` | 0.0 – 2.0 | Adds brightness in high-curl (vortex) regions; also slightly increases opacity. Strongest visual impact of the four. |
| `velocityInfluence` | 0.0 – 1.0 | Adds brightness in high-velocity regions; emphasizes motion trails. |
| `pressureInfluence` | 0.0 – 1.0 | Smoothstepped highlights in high-pressure zones; subtle pressure ridges. |
| `divergenceInfluence` | 0.0 – 1.0 | Smoothstepped highlights in divergent/convergent zones; marks sources/sinks. |

### 9.6 Color

| Parameter | Type | Effect |
|---|---|---|
| `baseColor` | `Color` | The dark/shadowed smoke color. Dominates in unlit regions. |
| `highlightColor` | `Color` | The bright/lit smoke color. Dominates where directional light hits and curl is high. |
| `lightColor` | `Color` | Additional tint added based on rim light, directional light, and flow fields. |
| `brightness` | 0.1 – 2.0 | Global multiplier on the final per-sample color before compositing. |

### 9.7 Blue Noise

| Parameter | Type | Effect |
|---|---|---|
| `blueNoiseTexture` | `Texture` (2D) | Should be a high-quality blue noise texture (e.g., 128×128). Eliminates banding artifacts in the raymarching. If omitted, a 1×1 white fallback is used (no jitter). |

---

## 10. Complete Per-Sample Pseudocode

For reference, here is the complete per-step algorithm in pseudocode:

```
// --- Ray setup (done once per fragment) ---
localCam   = inverse(modelWorldMatrix) * vec4(cameraPosition, 1)
rayDir     = normalize(fragmentPosition - localCam)
(tNear, tFar) = intersectAABB(localCam, rayDir, box(-0.5, +0.5))
if tNear > tFar: discard
tNear = max(tNear, 0)
stepSize = min(|1/rayDir.x|, |1/rayDir.y|, |1/rayDir.z|) / steps
invSteps = 1.0 / steps
screenUV = worldPos.xy * 0.5 + 0.5
jitter = sampleBlueNoise(screenUV * 4).x

accumColor = vec3(0)
transmittance = 1.0
lightDirN = normalize(lightDir)
luma = vec3(0.299, 0.587, 0.114)
currentStepSize = invSteps

// --- March loop ---
pos = localCam + tNear * rayDir
for t = tNear to tFar step stepSize:
    samplePos = (t == first) ? pos + jitter * invSteps * 0.5 : pos
    uvw = saturate(samplePos + 0.5)
    
    raw = densityTex.sample(uvw).rgb
    density = saturate(dot(raw, luma) * densityBoost)
    
    // Adaptive step
    factor = density > adaptiveThreshold ? 1.0 : density / adaptiveThreshold
    currentStepSize = invSteps * factor
    
    if density > 0.0005:
        // Sample auxiliary fields
        velMag  = length(velocityTex.sample(uvw).xyz)
        curlMag = length(curlTex.sample(uvw).xyz)
        presAbs = abs(pressureTex.sample(uvw).x)
        divAbs  = abs(divergenceTex.sample(uvw).x)
        
        // Compute gradient / normal
        grad = centralDifferenceGradient(densityTex, uvw, dyeTexelSize)
        normal = -normalize(grad)
        
        // View direction
        viewDir = -normalize(samplePos)
        
        // Lighting components
        directional = saturate(dot(normal, lightDirN))
        rim = pow(1.0 - abs(dot(normal, viewDir)), 2.0)
        cosTheta = dot(viewDir, -lightDirN)
        phase = HenyeyGreenstein(cosTheta, anisotropy)
        shadow = shadowMarch(samplePos, lightDirN, shadowSteps, ...)
        
        // Flow contributions
        D  = saturate(curlMag * 0.1) * curlInfluence
        V  = saturate(velMag * 0.5) * velocityInfluence
        P  = smoothstep3(saturate(presAbs * 2.0)) * pressureInfluence
        Dv = smoothstep3(saturate(divAbs * 2.0)) * divergenceInfluence
        
        // Shaded directional
        G = directional * shadow * (phase * 4.0)
        
        // Total lighting
        lighting = ambientLight + G * lightStrength + rim * rimStrength + D + V + P + Dv
        
        // Color
        dt = saturate(G * 0.6 + D * 0.4)
        blended = mix(baseColor, highlightColor, dt)
        flowScalar = saturate(D * 0.6 + V * 0.4 + P * 0.2 + Dv * 0.15)
        overlay = saturate(rim * rimStrength * 0.6 + G * lightStrength * 0.25 + flowScalar * 0.35)
        tint = saturate((blended + lightColor * overlay) * brightness)
        
        // Beer's law absorption
        attenuation = exp(-(density * absorption * currentStepSize + D * 0.05))
        alpha = saturate(1.0 - attenuation)
        weight = transmittance * alpha
        
        // Accumulate
        accumColor += weight * (tint * lighting)
        transmittance *= attenuation
        
        // Early termination
        if transmittance < 0.02: break
    
    pos += rayDir * stepSize

finalAlpha = saturate(1.0 - transmittance)
return vec4(accumColor, finalAlpha)
```

---

## 11. Implementation Notes

### 11.1 Three.js TSL Specifics

- The material extends `THREE.NodeMaterial` (WebGPU path).
- All uniforms are created with `uniform()` from `three/tsl`.
- 3D textures are wrapped with `texture3D(tex, null, 0)` — the `0` is the LOD level.
- 2D textures are wrapped with `texture(tex)`.
- The main shader logic is a single `Fn(...)` call that returns `vec4`.
- The `RaymarchingBox` helper provides the outer loop; the material passes a callback that runs per step.
- The material's `smokeNode` stores the TSL node graph. It is rebuilt whenever `setVolumeTextures()` is called via `_rebuildSmokeOutput()`.

### 11.2 Loop Structure

`RaymarchingBox(steps, callback)` creates a TSL `Loop` with `type: "float"`, iterating from `tNear` to `tFar` with update step equal to the computed step size. The callback receives `{ positionRay, stepIndex }` where `positionRay` is a `vec3` and `stepIndex` is the float loop variable (parametric `t`).

### 11.3 Smoothstep Hermite

The "smoothstep" used for pressure and divergence is the classic Hermite polynomial, applied manually:
```
x_clamped = saturate(value * 2.0)
result = x² * (3 - 2x)
```
This is equivalent to `smoothstep(0, 1, value * 2.0)`.

### 11.4 Performance Considerations

- The shadow marching loop is nested inside the primary marching loop, making `shadowSteps` a critical performance knob.
- Adaptive stepping reduces cost in sparse regions.
- The `0.0005` density threshold skips all lighting math in empty space.
- Early termination at `transmittance < 0.02` avoids wasted work behind opaque smoke.
- Blue noise jittering allows fewer steps while maintaining perceived quality.
