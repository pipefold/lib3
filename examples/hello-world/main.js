// examples/main.js
import * as THREE from "three/webgpu";
import { float } from "three/tsl";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { exampleTSLFunction } from "../../src/index.js"; // Import locally from src/ for dev

// Set up scene, camera, renderer...
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
const renderer = new THREE.WebGPURenderer({
  canvas: document.getElementById("canvas"),
});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

new OrbitControls(camera, renderer.domElement);

// Use your TSL function in a material (uniforms will get auto-GUI)
const material = new THREE.MeshBasicNodeMaterial(); // Or whatever TSL material
material.colorNode = exampleTSLFunction(float(1.0));

const mesh = new THREE.Mesh(new THREE.BoxGeometry(), material);
scene.add(mesh);

// Animate/render loop
async function animate() {
  requestAnimationFrame(animate);
  await renderer.renderAsync(scene, camera);
}
animate();
