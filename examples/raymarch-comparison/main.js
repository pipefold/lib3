import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Inspector } from "three/addons/inspector/Inspector.js";
import { RaymarchingBox } from "three/addons/tsl/utils/Raymarching.js";
import { select, float, Fn, vec3, If, vec2, uniform } from "three/tsl";
import * as THREE from "three/webgpu";
import { adaptiveRaymarch } from "../../src/raymarch.js";

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.z = 3;
const renderer = new THREE.WebGPURenderer({
  canvas: document.getElementById("canvas"),
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.inspector = new Inspector();
new OrbitControls(camera, renderer.domElement);

// Torus SDF (thin ring shape to show precision differences)
const torusSDF = Fn(({ pos }) => {
  const q = vec2(pos.xz.length().sub(0.3), pos.y); // Torus with major radius 0.3, minor 0.08
  return q.length().sub(0.08);
});

// Adaptive material
const adaptiveMaterial = new THREE.MeshBasicNodeMaterial();
const adaptiveMaxSteps = uniform(100);
const adaptiveThreshold = uniform(0.001);
const adaptiveFn = Fn(() => {
  const { hit } = adaptiveRaymarch(
    adaptiveMaxSteps,
    ({ positionRay, maxStep }) => {
      return torusSDF({ pos: positionRay });
    },
    adaptiveThreshold
  );
  return select(hit, vec3(1, 0, 0), vec3(0));
});
adaptiveMaterial.colorNode = adaptiveFn();
const adaptiveMesh = new THREE.Mesh(
  new THREE.BoxGeometry(1, 1, 1),
  adaptiveMaterial
);
adaptiveMesh.position.x = -1;
scene.add(adaptiveMesh);

// Fixed material (reduced steps to show artifacts)
const fixedMaterial = new THREE.MeshBasicNodeMaterial();
const fixedMaxSteps = uniform(20);
const fixedThreshold = uniform(0.01);
const fixedFn = Fn(() => {
  const minDist = float(1e10).toVar();
  RaymarchingBox(fixedMaxSteps, ({ positionRay }) => {
    const dist = torusSDF({ pos: positionRay });
    If(dist.lessThan(minDist), () => {
      minDist.assign(dist);
    });
  });
  const hit = minDist.lessThan(fixedThreshold);
  return select(hit, vec3(0, 1, 0), vec3(0));
});
fixedMaterial.colorNode = fixedFn();
const fixedMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), fixedMaterial);
fixedMesh.position.x = 1;
scene.add(fixedMesh);

// Inspector GUI
const gui = renderer.inspector.createParameters("Raymarch Comparison");
const adaptiveFolder = gui.addFolder("Adaptive (Left - Red)");
adaptiveFolder.add(adaptiveMaxSteps, "value", 1, 200, 1).name("maxSteps");
adaptiveFolder
  .add(adaptiveThreshold, "value", 0.0001, 0.1, 0.0001)
  .name("threshold");
const fixedFolder = gui.addFolder("Fixed (Right - Green)");
fixedFolder.add(fixedMaxSteps, "value", 1, 100, 1).name("maxSteps");
fixedFolder.add(fixedThreshold, "value", 0.001, 0.1, 0.001).name("threshold");

async function animate() {
  requestAnimationFrame(animate);
  await renderer.renderAsync(scene, camera);
}
animate();
