/**
 * Thunder Lightning Clouds — lib3 example
 *
 * Minimal demo wiring up SmokeVolume + VolumeSmokeNodeMaterial + ComputeMipAwareBlueNoise
 * with the contained-thunder TSL node from the original experiment.
 * All @three-blocks/core dependencies replaced with clean-room lib3 implementations.
 */
import * as THREE from 'three/webgpu';
import { uniform, vec4 } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { SmokeVolume } from '../../src/fluidSim.js';
import { VolumeSmokeNodeMaterial } from '../../src/smokeMaterial.js';
import { ComputeMipAwareBlueNoise } from '../../src/blueNoise.js';
import { createThunderNode, createThunderStateMachine } from '../../src/thunder.js';

const container = document.getElementById('container');
const errorEl = document.getElementById('error');

try {
  await init();
} catch (e) {
  errorEl.style.display = 'block';
  errorEl.textContent = e.message + '\n\n' + e.stack;
  throw e;
}

async function init() {
  // Renderer
  const renderer = new THREE.WebGPURenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);
  await renderer.init();

  // Scene + camera
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 3, 15);

  const controls = new OrbitControls(camera, renderer.domElement);

  // Clock
  const clock = new THREE.Clock();

  // ---- Fluid sim ----
  const fluid = new SmokeVolume({
    simRes: 64,
    dyeRes: 96,
    iterations: 40,
    densityDissipation: 0.995,
    velocityDissipation: 0.985,
    pressureDissipation: 0.98,
    curlStrength: 6,
    pressureFactor: 1 / 6,
    radius: 0.2,
    useBoundaries: true,
    neighborStride: 1,
    speedFactor: 1,
    buoyancyStrength: 0,
  });

  // ---- Blue noise ----
  const blueNoise = new ComputeMipAwareBlueNoise(128, 128);
  const blueNoiseTex = blueNoise.init(renderer);

  // ---- Smoke material ----
  const dyeTexelSize = uniform(new THREE.Vector3(
    1 / fluid.dyeRes, 1 / fluid.dyeRes, 1 / fluid.dyeRes
  ), 'vec3');

  const material = new VolumeSmokeNodeMaterial({
    densityTexture: fluid.getDensityTexture3D(),
    velocityTexture: fluid.getVelocityTexture3D(),
    curlTexture: fluid.getCurlTexture3D(),
    pressureTexture: fluid.getPressureTexture3D(),
    divergenceTexture: fluid.getDivergenceTexture3D(),
    dyeTexelSize,
    steps: 120,
    lightDir: new THREE.Vector3(-0.35, 0.9, 0.4),
    baseColor: new THREE.Color(0x1f232b),
    highlightColor: new THREE.Color(0x97a3b5),
    lightColor: new THREE.Color(0xf4f7ff),
    ambientLight: 0.65,
    lightStrength: 1.45,
    rimStrength: 0.9,
    densityBoost: 6.65,
    absorption: 17.1,
    curlInfluence: 0.6,
    velocityInfluence: 0.6,
    pressureInfluence: 0.4,
    divergenceInfluence: 0.0,
    brightness: 0.35,
    blueNoiseTexture: blueNoiseTex,
    anisotropy: 0.6,
    shadowSteps: 6,
    shadowIntensity: 0.7,
    adaptiveStepThreshold: 0.05,
  });

  // ---- Thunder node ----
  const { node: thunderNode, uniforms: thunderUniforms } = createThunderNode({
    densityTexture: fluid.getDensityTexture3D(),
    pressureTexture: fluid.getPressureTexture3D(),
    curlTexture: fluid.getCurlTexture3D(),
  });

  const thunder = createThunderStateMachine(thunderUniforms);

  // ---- Composite smoke + thunder ----
  const smokeNode = material.getSmokeNode();
  material.outputNode = vec4(
    smokeNode.rgb.add(thunderNode.rgb.mul(thunderUniforms.outputGain)),
    smokeNode.a
  );
  material.needsUpdate = true;

  // ---- Mesh ----
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
  mesh.scale.set(10, 10, 10);
  mesh.frustumCulled = false;
  scene.add(mesh);

  // Wireframe box
  const wireGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(10, 10, 10));
  const wire = new THREE.LineSegments(wireGeo, new THREE.LineBasicMaterial({ color: 0xffffff }));
  scene.add(wire);

  // ---- Pointer interaction ----
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let prevPos = null;
  const tmpSplat = new THREE.Vector3();
  container.style.touchAction = 'none';

  const onPointerMove = (e) => {
    if (!e.isPrimary) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(mesh);
    if (hits.length) {
      const p = hits[0].point.clone();
      mesh.worldToLocal(p);
      tmpSplat.set(p.x + 0.5, p.y + 0.5, p.z + 0.5);
      if (prevPos) {
        const dir = new THREE.Vector3().subVectors(p, prevPos);
        if (dir.lengthSq() > 0.0001) {
          dir.normalize();
          fluid.addSplat(tmpSplat.x, tmpSplat.y, tmpSplat.z, dir.x * 1000, dir.y * 1000, dir.z * 1000);
        }
      }
      prevPos = p.clone();
    }
  };
  const onPointerStop = () => { prevPos = null; };
  renderer.domElement.addEventListener('pointermove', onPointerMove);
  renderer.domElement.addEventListener('pointerdown', (e) => { prevPos = null; onPointerMove(e); });
  renderer.domElement.addEventListener('pointerup', onPointerStop);
  renderer.domElement.addEventListener('pointerleave', onPointerStop);

  // ---- Resize ----
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // ---- Animate ----
  renderer.setAnimationLoop(() => {
    const dt = Math.min(0.1, clock.getDelta());
    thunder.update(dt);
    controls.update();
    fluid.step(renderer);
    renderer.render(scene, camera);
  });
}
