/**
 * Thunder Lightning Clouds — lib3 example
 *
 * Minimal demo wiring up SmokeVolume + VolumeSmokeNodeMaterial + ComputeMipAwareBlueNoise
 * with the contained-thunder TSL node from the original experiment.
 * All @three-blocks/core dependencies replaced with clean-room lib3 implementations.
 */
import * as THREE from 'three/webgpu';
import { texture3D, uniform, Fn, float, vec3, vec4, If, Break, smoothstep } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RaymarchingBox } from 'three/addons/tsl/utils/Raymarching.js';

import { SmokeVolume } from '../../src/fluidSim.js';
import { VolumeSmokeNodeMaterial } from '../../src/smokeMaterial.js';
import { ComputeMipAwareBlueNoise } from '../../src/blueNoise.js';

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
  const thunderTime = uniform(0);
  const thunderCharge = uniform(0.18);
  const thunderFlash = uniform(0);
  const thunderFlashCenter = uniform(new THREE.Vector3(0.5, 0.5, 0.5), 'vec3');
  const thunderFlashRadius = uniform(0.13);
  const thunderBaseGlow = uniform(0.72);
  const thunderFlashGain = uniform(4.6);
  const thunderOutputGain = uniform(1.28);
  const thunderConfinement = uniform(1.3);
  const thunderShellRadius = uniform(0.46);
  const thunderShellSoftness = uniform(0.08);
  const thunderFilamentScale = uniform(31.0);
  const thunderFilamentSharpness = uniform(0.84);
  const thunderArcDrift = uniform(0.5);
  const thunderAbsorption = uniform(12.4);
  const thunderSteps = uniform(128);
  const thunderColorCold = uniform(new THREE.Color(0x3569ff));
  const thunderColorHot = uniform(new THREE.Color(0xf2f8ff));
  const thunderDensityTexture = texture3D(fluid.getDensityTexture3D(), null, 0);
  const thunderPressureTexture = texture3D(fluid.getPressureTexture3D(), null, 0);
  const thunderCurlTexture = texture3D(fluid.getCurlTexture3D(), null, 0);

  const thunderNode = Fn(() => {
    const thunderAccum = vec3(0).toVar();
    const thunderAlpha = float(0).toVar();
    const transmittance = float(1).toVar();
    const invSteps = float(1).div(thunderSteps).toVar();
    const luma = vec3(0.299, 0.587, 0.114).toConst('thunderLuma');

    RaymarchingBox(thunderSteps, ({ positionRay }) => {
      const uvw = positionRay.add(0.5).saturate().toVar('thunderUVW');
      const density = thunderDensityTexture.sample(uvw).rgb.dot(luma).saturate().toVar('thunderDensity');

      If(density.greaterThan(0.002), () => {
        const pressure = thunderPressureTexture.sample(uvw).x.abs().saturate().toVar();
        const curl = thunderCurlTexture.sample(uvw).xyz.length().saturate().toVar();
        const radial = uvw.sub(0.5).length().toVar();

        const shellInner = thunderShellRadius.sub(thunderShellSoftness.mul(1.95));
        const cocoonMask = float(1).sub(
          smoothstep(shellInner, thunderShellRadius.add(thunderShellSoftness), radial)
        ).toVar();
        const shellBand = smoothstep(
          shellInner, thunderShellRadius.sub(thunderShellSoftness.mul(0.24)), radial
        ).mul(
          float(1).sub(smoothstep(
            thunderShellRadius.sub(thunderShellSoftness.mul(0.08)),
            thunderShellRadius.add(thunderShellSoftness), radial
          ))
        ).saturate().toVar();
        const coreMask = float(1).sub(
          smoothstep(thunderShellRadius.mul(0.32), thunderShellRadius.mul(0.88), radial)
        ).toVar();

        const linePhase = uvw.x.mul(thunderFilamentScale)
          .add(uvw.y.mul(thunderFilamentScale.mul(1.31)))
          .add(uvw.z.mul(thunderFilamentScale.mul(1.87)))
          .add(thunderTime.mul(thunderArcDrift.mul(6)))
          .add(pressure.mul(8))
          .add(curl.mul(5))
          .toVar();
        const lineField = linePhase.sin().mul(0.5).add(0.5).toVar();
        const filament = smoothstep(
          thunderFilamentSharpness, float(1),
          lineField.add(pressure.mul(0.55)).add(curl.mul(0.35)).saturate()
        ).toVar();

        const hotspotDist = uvw.sub(thunderFlashCenter).length().toVar();
        const hotspot = float(1).sub(
          smoothstep(thunderFlashRadius, thunderFlashRadius.add(thunderShellSoftness), hotspotDist)
        ).toVar();

        const potential = density.pow(1.34).mul(cocoonMask).mul(thunderConfinement).toVar();
        const tension = coreMask.mul(0.82).add(shellBand.mul(0.48)).add(pressure.mul(0.56)).add(0.2).toVar();
        const dischargeMask = shellBand.mul(0.72).add(coreMask.mul(0.2)).add(hotspot.mul(0.45)).saturate().toVar();
        const chargeGlow = potential.mul(thunderCharge).mul(thunderBaseGlow).mul(tension).toVar();
        const burstGlow = potential.mul(thunderFlash).mul(thunderFlashGain)
          .mul(filament.mul(0.82).add(hotspot.mul(1.25)))
          .mul(dischargeMask).toVar();
        const emission = chargeGlow.add(burstGlow).toVar();
        const emissionAlpha = emission.mul(invSteps).mul(transmittance).saturate().toVar();

        const flashWeight = hotspot.mul(thunderFlash).add(filament.mul(0.34)).add(shellBand.mul(0.24)).saturate().toVar();
        const thunderTint = thunderColorCold.mix(thunderColorHot, flashWeight).toVar();

        thunderAccum.addAssign(emissionAlpha.mul(thunderTint));
        thunderAlpha.addAssign(emissionAlpha);

        const localAbsorption = density.mul(thunderAbsorption)
          .mul(cocoonMask.mul(0.75).add(0.25))
          .mul(invSteps)
          .negate()
          .exp()
          .toVar();
        transmittance.mulAssign(localAbsorption);

        If(thunderAlpha.greaterThan(0.985), () => { Break(); });
      });
    });

    return vec4(thunderAccum.saturate(), thunderAlpha.saturate());
  })();

  // ---- Composite smoke + thunder ----
  const smokeNode = material.getSmokeNode();
  material.outputNode = vec4(
    smokeNode.rgb.add(thunderNode.rgb.mul(thunderOutputGain)),
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

  // ---- Thunder state machine ----
  const thunderRuntime = {
    charge: 0.18, burst: 0, cooldown: 0,
    flickerFreq: 33, microFreq: 17, pulseFreq: 0.41,
    dir: new THREE.Vector3(),
  };
  const cfg = {
    chargeRate: 0.17, dischargeThreshold: 0.7, flashChance: 4.2,
    burstDecay: 6.6, cooldownMin: 0.06, cooldownMax: 0.26,
  };

  function triggerFlash() {
    const d = thunderRuntime.dir.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5);
    if (d.lengthSq() < 1e-4) d.set(1, 0, 0); else d.normalize();
    const inner = Math.max(0.07, thunderShellRadius.value - thunderShellSoftness.value * 2.3);
    const outer = Math.max(inner + 0.01, thunderShellRadius.value * 0.95);
    const r = inner + Math.random() * (outer - inner);
    thunderFlashCenter.value.set(
      THREE.MathUtils.clamp(0.5 + d.x * r, 0.03, 0.97),
      THREE.MathUtils.clamp(0.5 + d.y * r, 0.03, 0.97),
      THREE.MathUtils.clamp(0.5 + d.z * r, 0.03, 0.97)
    );
    thunderFlashRadius.value = 0.13 * (0.5 + Math.random() * 0.55);
    thunderRuntime.burst = 1.05 + Math.random() * 1.25;
    thunderRuntime.charge *= 0.34 + Math.random() * 0.22;
    thunderRuntime.flickerFreq = 26 + Math.random() * 22;
    thunderRuntime.microFreq = 10 + Math.random() * 20;
    thunderRuntime.cooldown = cfg.cooldownMin + Math.random() * (cfg.cooldownMax - cfg.cooldownMin);
  }

  function updateThunder(dt) {
    thunderTime.value += dt;
    const pulse = 0.5 + 0.5 * Math.sin(thunderTime.value * thunderRuntime.pulseFreq * Math.PI * 2);
    const brew = 0.62 + Math.pow(pulse, 1.7) * 0.75;
    thunderRuntime.charge = Math.min(1, thunderRuntime.charge + dt * cfg.chargeRate * brew);
    thunderRuntime.burst *= Math.exp(-dt * cfg.burstDecay);
    thunderRuntime.cooldown = Math.max(0, thunderRuntime.cooldown - dt);

    const threshold = Math.min(0.98, Math.max(0.05, cfg.dischargeThreshold));
    const range = Math.max(1e-4, 1 - threshold);
    const band = Math.max(0, thunderRuntime.charge - threshold) / range;
    const rate = band * band * cfg.flashChance * (0.55 + thunderRuntime.charge * 1.1);
    if (thunderRuntime.cooldown <= 0 && Math.random() < rate * dt) triggerFlash();

    const flicker = Math.max(
      Math.pow(Math.max(0, Math.sin(thunderTime.value * thunderRuntime.flickerFreq)), 9),
      Math.pow(Math.max(0, Math.sin(thunderTime.value * thunderRuntime.microFreq)), 6)
    );
    thunderFlash.value = thunderRuntime.burst * (0.18 + 0.82 * flicker);
    thunderCharge.value = Math.min(1,
      Math.pow(thunderRuntime.charge, 1.28) * (0.86 + 0.14 * brew) + thunderFlash.value * 0.22
    );
  }

  // ---- Resize ----
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // ---- Animate ----
  renderer.setAnimationLoop(() => {
    const dt = Math.min(0.1, clock.getDelta());
    updateThunder(dt);
    controls.update();
    fluid.step(renderer);
    renderer.render(scene, camera);
  });
}
