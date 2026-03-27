/**
 * @module VolumeSmokeNodeMaterial
 *
 * Volumetric smoke raymarching material for Three.js WebGPU / TSL.
 *
 * Renders smoke by raymarching through a unit box (−0.5 to +0.5) using
 * Beer-Lambert absorption, directional shadow rays, Henyey-Greenstein
 * phase scattering, rim lighting, and fluid-simulation field influences
 * (curl, velocity, pressure, divergence).
 *
 * The mesh MUST be a unit cube; scaling is applied via the world matrix.
 * The material renders back-faces so the ray origin is computed correctly.
 */

import * as THREE from "three/webgpu";
import {
  uniform,
  varying,
  vec2,
  vec3,
  vec4,
  float,
  int,
  Fn,
  Loop,
  Break,
  If,
  texture,
  texture3D,
  positionGeometry,
  positionWorld,
  modelWorldMatrixInverse,
  cameraPosition,
  normalize,
  abs,
  min,
  max,
  dot,
  exp,
  pow,
  length,
  mix,
  saturate,
  select,
  PI,
} from "three/tsl";

// ─── Constants ───────────────────────────────────────────────────────────────

const LUMA = vec3(0.299, 0.587, 0.114);
const HG_NORM = float(0.0795774715); // 1 / (4π)
const DENSITY_THRESHOLD = float(0.0005);
const TRANSMITTANCE_CUTOFF = float(0.02);

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Ray–AABB intersection for the unit box [−0.5, +0.5]³.
 * Returns vec2(tNear, tFar).
 */
const hitBox = /*@__PURE__*/ Fn(({ orig, dir }) => {
  const invDir = dir.reciprocal();
  const t0 = vec3(-0.5).sub(orig).mul(invDir);
  const t1 = vec3(0.5).sub(orig).mul(invDir);
  const tMin = min(t0, t1);
  const tMax = max(t0, t1);
  const tNear = max(tMin.x, max(tMin.y, tMin.z));
  const tFar = min(tMax.x, min(tMax.y, tMax.z));
  return vec2(tNear, tFar);
});

/**
 * Luminance of an RGB sample using Rec.601 luma weights.
 */
const luminance = /*@__PURE__*/ Fn(({ rgb }) => {
  return dot(rgb, LUMA);
});

/**
 * Hermite smoothstep polynomial: x² · (3 − 2x), with x already clamped to [0,1].
 */
const hermite = /*@__PURE__*/ Fn(({ x }) => {
  return x.mul(x).mul(float(3.0).sub(x.mul(2.0)));
});

/**
 * Henyey-Greenstein phase function.
 *
 * @param {Node<float>} cosTheta - cos(angle between view and light)
 * @param {Node<float>} g        - asymmetry parameter
 * @returns {Node<float>}
 */
const henyeyGreenstein = /*@__PURE__*/ Fn(({ cosTheta, g }) => {
  const g2 = g.mul(g);
  const num = float(1.0).sub(g2);
  const base = float(1.0).add(g2).sub(g.mul(2.0).mul(cosTheta));
  const denom = pow(base, float(1.5)).add(0.0001);
  return HG_NORM.mul(num).div(denom);
});

// ─── Material ────────────────────────────────────────────────────────────────

/**
 * Volumetric smoke node material.
 *
 * Extends `THREE.NodeMaterial` and renders on `BackSide` with transparency.
 * All parameters are passed as a single options object. Scalars / vectors
 * that are not already TSL nodes are automatically wrapped in `uniform()`.
 */
export class VolumeSmokeNodeMaterial extends THREE.NodeMaterial {
  /** @returns {"VolumeSmokeNodeMaterial"} */
  static get type() {
    return "VolumeSmokeNodeMaterial";
  }

  /**
   * @param {Object} opts
   * @param {THREE.Data3DTexture} opts.densityTexture     - RGB density field
   * @param {THREE.Data3DTexture} opts.velocityTexture    - XYZ velocity field
   * @param {THREE.Data3DTexture} opts.curlTexture        - XYZ curl field
   * @param {THREE.Data3DTexture} opts.pressureTexture    - Scalar pressure (.x)
   * @param {THREE.Data3DTexture} opts.divergenceTexture  - Scalar divergence (.x)
   * @param {Object}              [opts.*]                - See spec §1.2 for all optional params
   */
  constructor(opts = {}) {
    super();

    /** @type {true} */
    this.isVolumeSmokeNodeMaterial = true;
    this.forceSinglePass = true;
    this.transparent = true;
    this.depthWrite = false;
    this.side = THREE.BackSide;

    // ── 3D texture nodes ─────────────────────────────────────────────────
    this._densityTex = texture3D(opts.densityTexture, null, 0);
    this._velocityTex = texture3D(opts.velocityTexture, null, 0);
    this._curlTex = texture3D(opts.curlTexture, null, 0);
    this._pressureTex = texture3D(opts.pressureTexture, null, 0);
    this._divergenceTex = texture3D(opts.divergenceTexture, null, 0);

    // ── Blue noise (2D) ──────────────────────────────────────────────────
    const blueNoiseFallback = VolumeSmokeNodeMaterial._makeWhitePixel();
    this._blueNoiseTex = texture(opts.blueNoiseTexture || blueNoiseFallback);

    // ── Scalar / vector uniforms ─────────────────────────────────────────
    const u = (v, fallback) =>
      v != null && v.isNode ? v : uniform(v != null ? v : fallback);

    this._dyeTexelSize = u(
      opts.dyeTexelSize,
      new THREE.Vector3(1 / 128, 1 / 128, 1 / 128)
    );
    this._steps = u(opts.steps, 160);
    this._lightDir = u(
      opts.lightDir,
      new THREE.Vector3(-0.35, 0.9, 0.4)
    );
    this._baseColor = u(opts.baseColor, new THREE.Color(0x1f1f8b));
    this._highlightColor = u(opts.highlightColor, new THREE.Color(0x97a3b5));
    this._lightColor = u(opts.lightColor, new THREE.Color(0xf4f7ff));
    this._ambientLight = u(opts.ambientLight, 0.35);
    this._lightStrength = u(opts.lightStrength, 1.5);
    this._rimStrength = u(opts.rimStrength, 1.0);
    this._densityBoost = u(opts.densityBoost, 5.0);
    this._absorption = u(opts.absorption, 6.0);
    this._curlInfluence = u(opts.curlInfluence, 0.3);
    this._velocityInfluence = u(opts.velocityInfluence, 0.25);
    this._pressureInfluence = u(opts.pressureInfluence, 0.15);
    this._divergenceInfluence = u(opts.divergenceInfluence, 0.15);
    this._brightness = u(opts.brightness, 0.45);
    this._anisotropy = u(opts.anisotropy, 0.5);
    this._shadowSteps = u(opts.shadowSteps, 4);
    this._shadowIntensity = u(opts.shadowIntensity, 0.4);
    this._adaptiveStepThreshold = u(opts.adaptiveStepThreshold, 0.05);

    this._alphaHash = opts.alphaHash === true;
    if (this._alphaHash) {
      this.alphaHash = true;
    }

    // ── Build TSL graph ──────────────────────────────────────────────────
    /** @type {Node<vec4>} */
    this.smokeNode = this._buildSmokeNode();
    this.outputNode = this.smokeNode;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Returns the smoke color+alpha TSL node for compositing.
   * @returns {Node<vec4>}
   */
  getSmokeNode() {
    return this.smokeNode;
  }

  /**
   * Resets `outputNode` to the smoke node and flags a rebuild.
   */
  useSmokeOutput() {
    this.outputNode = this.smokeNode;
    this.needsUpdate = true;
  }

  /**
   * Hot-swap any subset of the five volume textures.
   * Triggers an internal rebuild of the TSL graph.
   *
   * @param {Object} textures
   * @param {THREE.Data3DTexture} [textures.densityTexture]
   * @param {THREE.Data3DTexture} [textures.velocityTexture]
   * @param {THREE.Data3DTexture} [textures.curlTexture]
   * @param {THREE.Data3DTexture} [textures.pressureTexture]
   * @param {THREE.Data3DTexture} [textures.divergenceTexture]
   */
  setVolumeTextures(textures) {
    if (textures.densityTexture)
      this._densityTex = texture3D(textures.densityTexture, null, 0);
    if (textures.velocityTexture)
      this._velocityTex = texture3D(textures.velocityTexture, null, 0);
    if (textures.curlTexture)
      this._curlTex = texture3D(textures.curlTexture, null, 0);
    if (textures.pressureTexture)
      this._pressureTex = texture3D(textures.pressureTexture, null, 0);
    if (textures.divergenceTexture)
      this._divergenceTex = texture3D(textures.divergenceTexture, null, 0);

    this._rebuildSmokeOutput();
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  /** Rebuild the TSL graph after texture hot-swap. */
  _rebuildSmokeOutput() {
    this.smokeNode = this._buildSmokeNode();
    this.outputNode = this.smokeNode;
    this.needsUpdate = true;
  }

  /**
   * Creates a 1×1 white DataTexture used as blue noise fallback.
   * @returns {THREE.DataTexture}
   */
  static _makeWhitePixel() {
    const data = new Uint8Array([255, 255, 255, 255]);
    const tex = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat);
    tex.needsUpdate = true;
    return tex;
  }

  /**
   * Builds the complete TSL smoke raymarching node graph.
   * @returns {Node<vec4>}
   */
  _buildSmokeNode() {
    // Alias uniforms for closure capture
    const densityTex = this._densityTex;
    const velocityTex = this._velocityTex;
    const curlTex = this._curlTex;
    const pressureTex = this._pressureTex;
    const divergenceTex = this._divergenceTex;
    const blueNoiseTex = this._blueNoiseTex;

    const uSteps = this._steps;
    const uLightDir = this._lightDir;
    const uBaseColor = this._baseColor;
    const uHighlightColor = this._highlightColor;
    const uLightColor = this._lightColor;
    const uAmbientLight = this._ambientLight;
    const uLightStrength = this._lightStrength;
    const uRimStrength = this._rimStrength;
    const uDensityBoost = this._densityBoost;
    const uAbsorption = this._absorption;
    const uCurlInfluence = this._curlInfluence;
    const uVelocityInfluence = this._velocityInfluence;
    const uPressureInfluence = this._pressureInfluence;
    const uDivergenceInfluence = this._divergenceInfluence;
    const uBrightness = this._brightness;
    const uAnisotropy = this._anisotropy;
    const uShadowSteps = this._shadowSteps;
    const uShadowIntensity = this._shadowIntensity;
    const uAdaptiveStepThreshold = this._adaptiveStepThreshold;
    const uDyeTexelSize = this._dyeTexelSize;

    // ── Shadow ray marching function ───────────────────────────────────
    const shadowMarch = /*@__PURE__*/ Fn(({ samplePos, lightDirN }) => {
      const shadow = float(1.0).toVar();
      const shadowStep = float(1.0).div(uShadowSteps);

      Loop(int(8), ({ i }) => {
        // Only execute for i < shadowSteps
        If(float(i).greaterThanEqual(uShadowSteps), () => {
          Break();
        });

        const marchOffset = lightDirN.mul(float(i).mul(shadowStep));
        const marchUVW = saturate(samplePos.add(marchOffset).add(0.5));
        const shadowDensity = dot(
          densityTex.sample(marchUVW).rgb,
          LUMA
        ).mul(uDensityBoost);
        const shadowAbs = shadowDensity
          .mul(uAbsorption)
          .mul(uShadowIntensity)
          .mul(shadowStep);
        shadow.mulAssign(exp(shadowAbs.negate()));
      });

      return saturate(shadow);
    });

    // ── Density gradient (central difference) ─────────────────────────
    const computeGradient = /*@__PURE__*/ Fn(({ uvw }) => {
      const off = uDyeTexelSize;

      const gx = luminance({
        rgb: densityTex.sample(uvw.add(vec3(off.x, 0, 0))).rgb,
      }).sub(
        luminance({
          rgb: densityTex.sample(uvw.sub(vec3(off.x, 0, 0))).rgb,
        })
      );

      const gy = luminance({
        rgb: densityTex.sample(uvw.add(vec3(0, off.y, 0))).rgb,
      }).sub(
        luminance({
          rgb: densityTex.sample(uvw.sub(vec3(0, off.y, 0))).rgb,
        })
      );

      const gz = luminance({
        rgb: densityTex.sample(uvw.add(vec3(0, 0, off.z))).rgb,
      }).sub(
        luminance({
          rgb: densityTex.sample(uvw.sub(vec3(0, 0, off.z))).rgb,
        })
      );

      return vec3(gx, gy, gz);
    });

    // ── Main smoke node ────────────────────────────────────────────────
    const smokeNode = Fn(() => {
      // Ray setup
      const localCamPos = varying(
        vec3(modelWorldMatrixInverse.mul(vec4(cameraPosition, 1.0)))
      );
      const vDirection = varying(positionGeometry.sub(localCamPos));
      const rayDir = normalize(vDirection);

      // AABB intersection
      const bounds = vec2(hitBox({ orig: localCamPos, dir: rayDir })).toVar();
      bounds.x.greaterThan(bounds.y).discard();
      bounds.assign(vec2(max(bounds.x, 0.0), bounds.y));

      // Step size computation
      const axisLengths = abs(rayDir.reciprocal());
      const baseStepSize = min(axisLengths.x, min(axisLengths.y, axisLengths.z)).div(uSteps);
      const invSteps = float(1.0).div(uSteps);

      // Blue noise jitter
      const screenUV = positionWorld.xy.mul(0.5).add(0.5);
      const jitter = blueNoiseTex.sample(screenUV.mul(4.0)).x;

      // Accumulation state
      const accumColor = vec3(0, 0, 0).toVar();
      const transmittance = float(1.0).toVar();

      // Pre-normalise light direction
      const lightDirN = normalize(uLightDir);

      // Ray position
      const positionRay = vec3(
        localCamPos.add(rayDir.mul(bounds.x))
      ).toVar();

      // First-step flag
      const isFirstStep = int(1).toVar();

      // March loop (float type, from tNear to tFar, step = baseStepSize)
      Loop(
        { type: "float", start: bounds.x, end: bounds.y, update: baseStepSize },
        () => {
          // Apply jitter on first step only
          const samplePos = vec3(positionRay).toVar();
          If(isFirstStep.equal(int(1)), () => {
            samplePos.addAssign(rayDir.mul(jitter.mul(invSteps).mul(0.5)));
            isFirstStep.assign(int(0));
          });

          // UVW coordinates
          const uvw = saturate(samplePos.add(0.5));

          // Sample density
          const rawDensity = densityTex.sample(uvw).rgb;
          const densityScalar = saturate(
            dot(rawDensity, LUMA).mul(uDensityBoost)
          );

          // Adaptive step size
          const adaptiveFactor = select(
            densityScalar.greaterThan(uAdaptiveStepThreshold),
            float(1.0),
            densityScalar.div(uAdaptiveStepThreshold)
          );
          const currentStepSize = invSteps.mul(adaptiveFactor);

          // Skip empty space
          If(densityScalar.greaterThan(DENSITY_THRESHOLD), () => {
            // ── Auxiliary field sampling ───────────────────────────────
            const velocity = velocityTex.sample(uvw).xyz;
            const velocityMag = length(velocity);
            const curl = curlTex.sample(uvw).xyz;
            const curlMag = length(curl);
            const pressureAbs = abs(pressureTex.sample(uvw).x);
            const divergenceAbs = abs(divergenceTex.sample(uvw).x);

            // ── Gradient / normal ─────────────────────────────────────
            const grad = computeGradient({ uvw });
            const normal = normalize(grad.negate());

            // ── View direction ────────────────────────────────────────
            const viewDir = normalize(samplePos.negate());

            // ── Directional light + shadow ────────────────────────────
            const directional = saturate(dot(normal, lightDirN));
            const shadowFactor = shadowMarch({ samplePos, lightDirN });
            const cosTheta = dot(viewDir, lightDirN.negate());
            const phase = henyeyGreenstein({ cosTheta, g: uAnisotropy });
            const shadedDirectional = directional
              .mul(shadowFactor)
              .mul(phase.mul(4.0));

            // ── Rim light ─────────────────────────────────────────────
            const rim = pow(
              float(1.0).sub(abs(dot(normal, viewDir))),
              float(2.0)
            );

            // ── Flow-field lighting contributions ─────────────────────
            const D = saturate(curlMag.mul(0.1)).mul(uCurlInfluence);
            const V = saturate(velocityMag.mul(0.5)).mul(uVelocityInfluence);
            const pClamped = saturate(pressureAbs.mul(2.0));
            const P = hermite({ x: pClamped }).mul(uPressureInfluence);
            const dClamped = saturate(divergenceAbs.mul(2.0));
            const Dv = hermite({ x: dClamped }).mul(uDivergenceInfluence);

            // ── Total lighting ────────────────────────────────────────
            const lighting = uAmbientLight
              .add(shadedDirectional.mul(uLightStrength))
              .add(rim.mul(uRimStrength))
              .add(D)
              .add(V)
              .add(P)
              .add(Dv);

            // ── Color blending ────────────────────────────────────────
            const directionalTint = saturate(
              shadedDirectional.mul(0.6).add(D.mul(0.4))
            );
            const blendedColor = mix(uBaseColor, uHighlightColor, directionalTint);

            const flowScalar = saturate(
              D.mul(0.6).add(V.mul(0.4)).add(P.mul(0.2)).add(Dv.mul(0.15))
            );
            const lightOverlay = saturate(
              rim
                .mul(uRimStrength.mul(0.6))
                .add(shadedDirectional.mul(uLightStrength.mul(0.25)))
                .add(flowScalar.mul(0.35))
            );
            const detailTint = saturate(
              blendedColor.add(uLightColor.mul(lightOverlay)).mul(uBrightness)
            );

            const sampleColor = detailTint.mul(lighting);

            // ── Beer's law absorption ─────────────────────────────────
            const attenuation = exp(
              densityScalar
                .mul(uAbsorption)
                .mul(currentStepSize)
                .add(D.mul(0.05))
                .negate()
            );
            const alpha = saturate(float(1.0).sub(attenuation));
            const weight = transmittance.mul(alpha);

            accumColor.addAssign(sampleColor.mul(weight));
            transmittance.mulAssign(attenuation);

            // ── Early termination ─────────────────────────────────────
            If(transmittance.lessThan(TRANSMITTANCE_CUTOFF), () => {
              Break();
            });
          });

          // Advance ray
          positionRay.addAssign(rayDir.mul(baseStepSize));
        }
      );

      const finalAlpha = saturate(float(1.0).sub(transmittance));
      return vec4(accumColor, finalAlpha);
    })();

    return smokeNode;
  }
}
