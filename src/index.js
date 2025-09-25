// src/index.js
import { uniform, attribute, mix, Fn, positionGeometry } from "three/tsl"; // Example import
export { sphericalWaveDisplacement, displacedTexCoord, buildSphericalWaveCopyKernel } from "./waves.js";

// Example TSL function (add your own here)
export function exampleTSLFunction(input) {
  const myUniform = uniform(1.0, "float"); // This will get auto-GUI in demos via the plugin
  return input.mul(myUniform); // Simple example
}

// Expose mixFactor as a uniform so it can be updated externally (e.g., per frame)
export const knotMorphMixFactor = uniform(0);

// Knot morph position (lerp between geometry position and targetPosition attribute)
export const knotMorphPosition = Fn(() => {
  const targetPosition = attribute("targetPosition");
  return mix(positionGeometry, targetPosition, knotMorphMixFactor);
});

// Re-export from other files, e.g.:
// export * from './myOtherFunction.js';
