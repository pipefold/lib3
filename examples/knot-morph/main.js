// examples/knot-morph/main.js
import * as THREE from "three/webgpu";
import { color } from "three/tsl";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { knotMorphPosition } from "../../src/knotMorph.js"; // Import from dedicated module

// Set up scene, camera, renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.z = 5; // Adjusted for better view of knot scale

const canvas = document.getElementById("canvas");
const renderer = new THREE.WebGPURenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);

// Controls for interaction
new OrbitControls(camera, renderer.domElement);

// Create starting geometry (p=2, q=3 for trefoil knot)
const startGeo = new THREE.TorusKnotGeometry(1, 0.4, 128, 32, 2, 3);

// Create target geometry (p=3, q=5)
const targetGeo = new THREE.TorusKnotGeometry(1, 0.4, 128, 32, 3, 5);

// Copy target positions as attribute to starting geo
const targetPositions = targetGeo.getAttribute("position").array;
startGeo.setAttribute(
  "targetPosition",
  new THREE.BufferAttribute(targetPositions, 3)
);

// TSL-based material (wireframe, with morph position and uniform color)
const material = new THREE.MeshBasicNodeMaterial({
  wireframe: true,
  transparent: true,
});
material.positionNode = knotMorphPosition(); // Use modular TSL function
material.colorNode = color(0x00ff00);

const mesh = new THREE.Mesh(startGeo, material);
scene.add(mesh);

// Animation loop
let time = 0;
function animate() {
  requestAnimationFrame(animate);
  time += 0.01;

  // Ping-pong mix factor for looping morph (0 to 1 and back)
  const mixFactor = Math.abs(Math.sin(time * 0.5));
  material.positionNode = knotMorphPosition({ mixFactor });

  // Dynamic rotation (ported from original)
  const quaternion = new THREE.Quaternion();

  const slowTime = time * 0.3;
  const fastTime = time * 1.2;

  // Y-axis rotation
  const ySpeedMultiplier = Math.sin(slowTime) * Math.sin(slowTime * 0.7);
  const yDirection = Math.sign(ySpeedMultiplier);
  const yRotationSpeed = Math.abs(ySpeedMultiplier) * 0.015;
  const yQuat = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 1, 0),
    yRotationSpeed * yDirection
  );

  // X-axis rotation
  const xSpeedMultiplier = Math.sin(slowTime * 1.3) * Math.cos(slowTime * 0.5);
  const xDirection = Math.sign(xSpeedMultiplier);
  const xRotationSpeed = Math.abs(xSpeedMultiplier) * 0.008;
  const xQuat = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(1, 0, 0),
    xRotationSpeed * xDirection
  );

  // Z-axis rotation
  const zSpeedMultiplier =
    Math.sin(fastTime * 0.8) * (1 - Math.abs(mixFactor - 0.5) * 2);
  const zDirection = Math.sign(zSpeedMultiplier);
  const zRotationSpeed = Math.abs(zSpeedMultiplier) * 0.006;
  const zQuat = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 0, 1),
    zRotationSpeed * zDirection
  );

  // Combine and apply rotations
  quaternion.multiply(yQuat).multiply(xQuat).multiply(zQuat);
  mesh.quaternion.multiply(quaternion);

  renderer.render(scene, camera);
}

// Initialize renderer and start animation
renderer.init().then(() => {
  animate();
});

// Handle resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
