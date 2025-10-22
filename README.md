# @pipefold/lib3

Composable TSL (Three.js Shading Language) nodes and utilities for Three.js
WebGPU. A collection of reusable shader building blocks for waves, morphing,
raymarching, and procedural effects.

## Features

- 🌊 **Wave Displacement** - Spherical wave displacement with simplex noise
  modulation
- 🔀 **Mesh Morphing** - Smooth position-based geometry morphing
- 🎯 **Raymarching** - Adaptive raymarching and volumetric rendering utilities
- 🧩 **Composable** - Pure TSL nodes that compose naturally with Three.js WebGPU
- 📦 **Tree-shakeable** - Modular exports for optimal bundle size
- 🎨 **Example Gallery** - 13+ interactive examples showcasing different
  techniques

## Installation

```bash
pnpm add @pipefold/lib3
```

Or with npm:

```bash
npm install @pipefold/lib3
```

**Requirements:**

- Three.js >= 0.180.0 with WebGPU support
- Modern browser with WebGPU support

## Quick Start

```javascript
import * as THREE from "three/webgpu";
import { sphericalWaveDisplacement } from "@pipefold/lib3/waves";
import { float, vec3 } from "three/tsl";

// Create material with wave displacement
const material = new THREE.MeshStandardNodeMaterial();
material.positionNode = sphericalWaveDisplacement({
  pos: positionGeometry,
  phase: float(time),
  waveAmplitude: float(0.2),
  noiseScale: float(1.0),
  noiseAmplitude: float(0.5),
  center: vec3(0, 0, 0),
});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);
```

## API Reference

### Waves (`@pipefold/lib3/waves`)

#### `sphericalWaveDisplacement(options)`

Creates spherical wave displacement with noise modulation.

**Parameters:**

- `pos` - vec3: Position in local/object space
- `phase` - float: Animation phase (typically time-based)
- `waveAmplitude` - float: Base wave amplitude (default: 0.2)
- `noiseScale` - float: Scale of noise modulation (default: 1.0)
- `noiseAmplitude` - float: Amplitude of noise effect (default: 0.5)
- `center` - vec3: Center point of spherical waves (default: vec3(0))

**Returns:** vec3 displacement

#### `displacedTexCoord(options)`

Compute displaced sampling coordinates for texture space [0,1]³.

**Parameters:**

- `texCoord` - vec3: Original texture coordinate
- `phase`, `waveAmplitude`, `noiseScale`, `noiseAmplitude` - Same as above
- `scale` - float: Displacement scale factor (default: 0.1)

**Returns:** vec3 displaced coordinate

#### `buildSphericalWaveCopyKernel(options)`

Build a compute shader kernel for 3D texture displacement.

**Parameters:**

- `width`, `height`, `depth` - numbers: Texture dimensions
- `storageTexture` - StorageTextureNode: Output texture
- `sourceTextureNode` - TextureNode: Input texture
- `waveAmplitude`, `noiseScale`, `noiseAmplitude`, `phase` - uniforms
- `intensityScale` - uniform: Intensity multiplier (default: 1.0)

#### `simplexNoise3(v)`

High-quality 3D simplex noise function.

**Parameters:**

- `v` - vec3: Input position

**Returns:** float in [-1, 1]

### Morphing (`@pipefold/lib3/knotMorph`)

#### `knotMorphPosition(options)`

Interpolate between geometry positions.

**Parameters:**

- `mixFactor` - float: Blend factor [0,1] (default: 0)

**Returns:** vec3 morphed position

**Usage:**

```javascript
// Add target positions as attribute
geometry.setAttribute(
  "targetPosition",
  new THREE.BufferAttribute(targetPositions, 3)
);

material.positionNode = knotMorphPosition({
  mixFactor: float(animationValue),
});
```

### Raymarching (`@pipefold/lib3`)

#### `adaptiveRaymarch(maxSteps, callback, threshold)`

Adaptive distance-field raymarching within a unit box [-0.5, 0.5]³.

**Parameters:**

- `maxSteps` - int: Maximum raymarch iterations
- `callback` - Function: `({ positionRay, maxStep }) => delta` distance function
- `threshold` - float: Hit detection threshold (default: 0.001)

**Returns:** Object with `{ positionRay, t, bounds, hit }`

#### `averageIntensityProjection(options)`

Average intensity projection for volumetric rendering.

**Parameters:**

- `texture` - Texture3DNode: 3D volume texture
- `steps` - int: Number of samples
- `intensityScale` - float: Intensity multiplier (default: 1.0)

**Returns:** vec4 color

## Examples

The package includes 13+ examples demonstrating various techniques:

- **hello-world** - Basic TSL function usage
- **knot-morph** - Geometry morphing between torus knots
- **raymarch-head** - 3D medical data visualization
- **raymarch-head-wave-displacement** - Volumetric waves
- **portal-door-transition** - Portal effects
- **cinematic-gallery** - Lighting and materials showcase
- **wispy-projector-beams** - Volumetric light beams
- **anisotropic-fbm-streaks** - Procedural noise patterns
- And more...

### Running Examples

```bash
# Development server with example gallery
pnpm dev

# Build examples for deployment
pnpm build:examples
```

Navigate to `http://localhost:5173` to see the example gallery.

## Development

```bash
# Install dependencies
pnpm install

# Build library
pnpm build

# Development mode (examples)
pnpm dev
```

## Project Structure

```
lib3/
├── src/
│   ├── index.js         # Main exports
│   ├── waves.js         # Wave displacement functions
│   ├── knotMorph.js     # Morphing utilities
│   └── raymarch.js      # Raymarching functions
├── examples/            # Interactive examples
├── dist/                # Built library (published)
└── package.json
```

## Exports

The package provides multiple entry points for tree-shaking:

```javascript
// Main bundle (all utilities)
import { sphericalWaveDisplacement, knotMorphPosition } from "@pipefold/lib3";

// Individual modules (smaller bundles)
import { sphericalWaveDisplacement } from "@pipefold/lib3/waves";
import { knotMorphPosition } from "@pipefold/lib3/knotMorph";
```

## WebGPU Compatibility

This library requires Three.js with WebGPU support. Import from `three/webgpu`:

```javascript
import * as THREE from "three/webgpu";
import /* TSL nodes */ "three/tsl";
```

Check browser compatibility: [WebGPU Status](https://caniuse.com/webgpu)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Repository

[github.com/pipefold/lib3](https://github.com/pipefold/lib3)
