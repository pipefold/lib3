// src/index.js
import { uniform } from "three/tsl"; // Example import

// Example TSL function (add your own here)
export function exampleTSLFunction(input) {
  const myUniform = uniform(1.0, "float"); // This will get auto-GUI in demos via the plugin
  return input.mul(myUniform); // Simple example
}

// Re-export from other files, e.g.:
// export * from './myOtherFunction.js';
