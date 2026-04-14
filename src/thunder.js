/**
 * @module thunder
 *
 * Contained-thunder effect for Three.js WebGPU / TSL.
 *
 * Provides a TSL raymarching node that renders volumetric lightning
 * filaments inside a unit cube, driven by fluid-simulation density,
 * pressure and curl textures, plus a JS state machine that animates
 * charge build-up, stochastic discharge flashes and flicker decay.
 *
 * @exports createThunderNode     — factory for the TSL raymarching node
 * @exports createThunderStateMachine — factory for the JS animation driver
 * @exports THUNDER_PRESETS        — named preset parameter sets
 */

import * as THREE from 'three/webgpu';
import { texture3D, uniform, Fn, float, vec3, vec4, If, Break, smoothstep } from 'three/tsl';
import { RaymarchingBox } from 'three/addons/tsl/utils/Raymarching.js';

// ─── Presets ─────────────────────────────────────────────────────────────────

/** All available thunder parameter presets. */
export const THUNDER_PRESETS = {
  'Contained Cocoon': {
    chargeRate: 0.17, dischargeThreshold: 0.7, flashChance: 4.2,
    burstDecay: 6.6, cooldownMin: 0.06, cooldownMax: 0.26,
    flashRadius: 0.13, baseGlow: 0.72, flashGain: 4.6, outputGain: 1.28,
    confinement: 1.3, shellRadius: 0.46, shellSoftness: 0.08,
    filamentScale: 31, filamentSharpness: 0.84, drift: 0.5,
    absorption: 12.4, steps: 128,
  },
  'Silent Pressure': {
    chargeRate: 0.12, dischargeThreshold: 0.79, flashChance: 2.6,
    burstDecay: 5.6, cooldownMin: 0.1, cooldownMax: 0.4,
    flashRadius: 0.1, baseGlow: 0.92, flashGain: 3.7, outputGain: 1.15,
    confinement: 1.58, shellRadius: 0.44, shellSoftness: 0.06,
    filamentScale: 36, filamentSharpness: 0.87, drift: 0.35,
    absorption: 13.6, steps: 136,
  },
  'Caged Arc Storm': {
    chargeRate: 0.28, dischargeThreshold: 0.62, flashChance: 8.4,
    burstDecay: 8.2, cooldownMin: 0.03, cooldownMax: 0.14,
    flashRadius: 0.15, baseGlow: 0.62, flashGain: 5.7, outputGain: 1.5,
    confinement: 1.15, shellRadius: 0.49, shellSoftness: 0.09,
    filamentScale: 34, filamentSharpness: 0.81, drift: 0.8,
    absorption: 11.2, steps: 124,
  },
  'Overcharged Core': {
    chargeRate: 0.22, dischargeThreshold: 0.76, flashChance: 6.1,
    burstDecay: 5.4, cooldownMin: 0.05, cooldownMax: 0.2,
    flashRadius: 0.11, baseGlow: 1.06, flashGain: 6.2, outputGain: 1.62,
    confinement: 1.9, shellRadius: 0.43, shellSoftness: 0.07,
    filamentScale: 41, filamentSharpness: 0.9, drift: 0.42,
    absorption: 14.5, steps: 148,
  },
};

// ─── TSL Node Factory ────────────────────────────────────────────────────────

/**
 * Create the TSL raymarching thunder node.
 *
 * @param {object} options
 * @param {THREE.Data3DTexture} options.densityTexture  — fluid density 3D texture
 * @param {THREE.Data3DTexture} options.pressureTexture — fluid pressure 3D texture
 * @param {THREE.Data3DTexture} options.curlTexture     — fluid curl 3D texture
 * @param {object}             [options.preset]         — initial uniform values (defaults to 'Contained Cocoon')
 * @returns {{ node: Node, uniforms: object }}
 */
export function createThunderNode({
  densityTexture,
  pressureTexture,
  curlTexture,
  preset = THUNDER_PRESETS['Contained Cocoon'],
} = {}) {
  // ── Uniforms ──
  const thunderTime            = uniform(0);
  const thunderCharge           = uniform(0.18);
  const thunderFlash            = uniform(0);
  const thunderFlashCenter      = uniform(new THREE.Vector3(0.5, 0.5, 0.5), 'vec3');
  const thunderFlashRadius      = uniform(preset.flashRadius);
  const thunderBaseGlow         = uniform(preset.baseGlow);
  const thunderFlashGain        = uniform(preset.flashGain);
  const thunderOutputGain       = uniform(preset.outputGain);
  const thunderConfinement      = uniform(preset.confinement);
  const thunderShellRadius      = uniform(preset.shellRadius);
  const thunderShellSoftness    = uniform(preset.shellSoftness);
  const thunderFilamentScale    = uniform(preset.filamentScale);
  const thunderFilamentSharpness = uniform(preset.filamentSharpness);
  const thunderArcDrift         = uniform(preset.drift);
  const thunderAbsorption       = uniform(preset.absorption);
  const thunderSteps            = uniform(preset.steps);
  const thunderColorCold        = uniform(new THREE.Color(0x3569ff));
  const thunderColorHot         = uniform(new THREE.Color(0xf2f8ff));

  // ── Texture nodes ──
  const thunderDensityTexture  = texture3D(densityTexture, null, 0);
  const thunderPressureTexture = texture3D(pressureTexture, null, 0);
  const thunderCurlTexture     = texture3D(curlTexture, null, 0);

  // ── TSL raymarching node ──
  const node = Fn(() => {
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

  // ── Collected uniforms ──
  const uniforms = {
    time: thunderTime,
    charge: thunderCharge,
    flash: thunderFlash,
    flashCenter: thunderFlashCenter,
    flashRadius: thunderFlashRadius,
    baseGlow: thunderBaseGlow,
    flashGain: thunderFlashGain,
    outputGain: thunderOutputGain,
    confinement: thunderConfinement,
    shellRadius: thunderShellRadius,
    shellSoftness: thunderShellSoftness,
    filamentScale: thunderFilamentScale,
    filamentSharpness: thunderFilamentSharpness,
    arcDrift: thunderArcDrift,
    absorption: thunderAbsorption,
    steps: thunderSteps,
    colorCold: thunderColorCold,
    colorHot: thunderColorHot,
  };

  return { node, uniforms };
}

// ─── State Machine Factory ───────────────────────────────────────────────────

/**
 * Create the JS state machine that drives the thunder charge / flash cycle.
 *
 * @param {object} uniforms — the uniforms object returned by `createThunderNode`
 * @param {object} [config] — initial config (defaults to 'Contained Cocoon' preset)
 * @returns {{ update(dt: number): void, triggerFlash(): void, setPreset(name: string): void }}
 */
export function createThunderStateMachine(uniforms, config) {
  const preset = config || THUNDER_PRESETS['Contained Cocoon'];

  const cfg = {
    chargeRate: preset.chargeRate,
    dischargeThreshold: preset.dischargeThreshold,
    flashChance: preset.flashChance,
    burstDecay: preset.burstDecay,
    cooldownMin: preset.cooldownMin,
    cooldownMax: preset.cooldownMax,
  };

  const runtime = {
    charge: 0.18,
    burst: 0,
    cooldown: 0,
    flickerFreq: 33,
    microFreq: 17,
    pulseFreq: 0.41,
    dir: new THREE.Vector3(),
  };

  function triggerFlash() {
    const d = runtime.dir.set(
      Math.random() - 0.5,
      Math.random() - 0.5,
      Math.random() - 0.5,
    );
    if (d.lengthSq() < 1e-4) d.set(1, 0, 0); else d.normalize();

    const inner = Math.max(0.07, uniforms.shellRadius.value - uniforms.shellSoftness.value * 2.3);
    const outer = Math.max(inner + 0.01, uniforms.shellRadius.value * 0.95);
    const r = inner + Math.random() * (outer - inner);

    uniforms.flashCenter.value.set(
      THREE.MathUtils.clamp(0.5 + d.x * r, 0.03, 0.97),
      THREE.MathUtils.clamp(0.5 + d.y * r, 0.03, 0.97),
      THREE.MathUtils.clamp(0.5 + d.z * r, 0.03, 0.97),
    );
    uniforms.flashRadius.value = 0.13 * (0.5 + Math.random() * 0.55);

    runtime.burst = 1.05 + Math.random() * 1.25;
    runtime.charge *= 0.34 + Math.random() * 0.22;
    runtime.flickerFreq = 26 + Math.random() * 22;
    runtime.microFreq = 10 + Math.random() * 20;
    runtime.cooldown = cfg.cooldownMin + Math.random() * (cfg.cooldownMax - cfg.cooldownMin);
  }

  function update(dt) {
    uniforms.time.value += dt;

    const t = uniforms.time.value;
    const pulse = 0.5 + 0.5 * Math.sin(t * runtime.pulseFreq * Math.PI * 2);
    const brew = 0.62 + Math.pow(pulse, 1.7) * 0.75;

    runtime.charge = Math.min(1, runtime.charge + dt * cfg.chargeRate * brew);
    runtime.burst *= Math.exp(-dt * cfg.burstDecay);
    runtime.cooldown = Math.max(0, runtime.cooldown - dt);

    const threshold = Math.min(0.98, Math.max(0.05, cfg.dischargeThreshold));
    const range = Math.max(1e-4, 1 - threshold);
    const band = Math.max(0, runtime.charge - threshold) / range;
    const rate = band * band * cfg.flashChance * (0.55 + runtime.charge * 1.1);

    if (runtime.cooldown <= 0 && Math.random() < rate * dt) triggerFlash();

    const flicker = Math.max(
      Math.pow(Math.max(0, Math.sin(t * runtime.flickerFreq)), 9),
      Math.pow(Math.max(0, Math.sin(t * runtime.microFreq)), 6),
    );
    uniforms.flash.value = runtime.burst * (0.18 + 0.82 * flicker);
    uniforms.charge.value = Math.min(
      1,
      Math.pow(runtime.charge, 1.28) * (0.86 + 0.14 * brew) + uniforms.flash.value * 0.22,
    );
  }

  /**
   * Apply a named preset. Updates both the state-machine config and the
   * visual uniforms that the preset controls.
   *
   * @param {string} name — one of the keys in THUNDER_PRESETS
   */
  function setPreset(name) {
    const p = THUNDER_PRESETS[name];
    if (!p) throw new Error(`Unknown thunder preset: "${name}"`);

    // State-machine config
    cfg.chargeRate = p.chargeRate;
    cfg.dischargeThreshold = p.dischargeThreshold;
    cfg.flashChance = p.flashChance;
    cfg.burstDecay = p.burstDecay;
    cfg.cooldownMin = p.cooldownMin;
    cfg.cooldownMax = p.cooldownMax;

    // Visual uniforms
    uniforms.flashRadius.value = p.flashRadius;
    uniforms.baseGlow.value = p.baseGlow;
    uniforms.flashGain.value = p.flashGain;
    uniforms.outputGain.value = p.outputGain;
    uniforms.confinement.value = p.confinement;
    uniforms.shellRadius.value = p.shellRadius;
    uniforms.shellSoftness.value = p.shellSoftness;
    uniforms.filamentScale.value = p.filamentScale;
    uniforms.filamentSharpness.value = p.filamentSharpness;
    uniforms.arcDrift.value = p.drift;
    uniforms.absorption.value = p.absorption;
    uniforms.steps.value = p.steps;
  }

  return { update, triggerFlash, setPreset };
}
