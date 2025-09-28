import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  Fn,
  Loop,
  abs,
  clamp,
  float,
  mix,
  step,
  time,
  uv,
  uniform,
  vec2,
  vec3,
  vec4,
} from "three/tsl";
import { simplexNoise3 } from "../../src/index.js";

// Scene setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 0, 2.2);

const renderer = new THREE.WebGPURenderer({
  canvas: document.getElementById("canvas"),
  antialias: true,
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000);
new OrbitControls(camera, renderer.domElement);

// TSL shader that generates anisotropic ridged fBm with vertical streak integration
const streakTextureTSL = Fn(() => {
  const uv0 = uv();

  // Aspect fix: scale X by aspect so isotropic base maps to screen space
  const aspect = float(window.innerWidth / window.innerHeight);

  // Controls
  // @range: { min: 0.1, max: 5.0, step: 0.1 }
  const scaleX = uniform(0.25);
  // @range: { min: 0.5, max: 10.0, step: 0.1 }
  const scaleY = uniform(4.0);
  // @range: { min: 1, max: 24, step: 1 }
  const integrateSamples = uniform(12);
  // @range: { min: 0.0, max: 1.0, step: 0.01 }
  const ridgedSharpness = uniform(0.85);
  // @range: { min: 0.0, max: 2.0, step: 0.01 }
  const warpStrength = uniform(0.15);
  // @range: { min: 0.1, max: 4.0, step: 0.1 }
  const fbmLacunarity = uniform(2.0);
  // @range: { min: 0.25, max: 1.0, step: 0.01 }
  const fbmGain = uniform(0.5);
  // @range: { min: 0, max: 1, step: 1 }
  const animate = uniform(1);

  // Prepare anisotropic domain: stretch Y, compress X
  const baseP = vec3(
    uv0.x.mul(scaleX).mul(aspect),
    uv0.y.mul(scaleY),
    time.mul(animate.mul(0.1))
  ).toVar();

  // Domain warp (mostly X) to introduce curvy drips and clumps
  const warpNoise = simplexNoise3({ v: baseP.mul(0.6) });
  baseP.x.addAssign(warpNoise.mul(warpStrength));

  // Ridged fBm helper
  const ridged = Fn(({ x }) => float(1.0).sub(abs(x).mul(2.0).sub(1.0).abs()));

  const fbmRidged = Fn(({ p }) => {
    const octaves = 5;
    const sum = float(0.0).toVar();
    const amp = float(1.0).toVar();
    const maxAmp = float(0.0).toVar();
    const freq = float(1.0).toVar();

    Loop(octaves, ({ i }) => {
      const n = simplexNoise3({ v: p.mul(freq) });
      const r = mix(abs(n), ridged({ x: n }), ridgedSharpness);
      sum.addAssign(r.mul(amp));
      maxAmp.addAssign(amp);
      amp.mulAssign(fbmGain);
      freq.mulAssign(fbmLacunarity);
    });

    return sum.div(maxAmp);
  });

  // Integrate upwards along Y to create vertical streaks (one-direction blur)
  const accum = float(0.0).toVar();
  const weightSum = float(0.0).toVar();
  const invSamples = float(1.0).div(integrateSamples);

  Loop(integrateSamples, ({ i }) => {
    const t = float(i).mul(invSamples);
    const weight = float(1.0).sub(t); // heavier near current row
    const p = vec3(baseP.x, baseP.y.add(t.mul(0.35)), baseP.z);
    const v = fbmRidged({ p });
    accum.addAssign(v.mul(weight));
    weightSum.addAssign(weight);
  });

  const gray = clamp(accum.div(weightSum), 0, 1);
  // Contrast and slight lift for readability
  const outV = clamp(gray.mul(1.35).add(0.02), 0, 1);
  return vec4(outV, outV, outV, 1.0);
});

const material = new THREE.MeshBasicNodeMaterial();
material.colorNode = streakTextureTSL();

const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
scene.add(mesh);

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener("resize", onResize);

renderer.setAnimationLoop(() => {
  renderer.render(scene, camera);
});
