import {
  Fn,
  vec2,
  vec3,
  vec4,
  float,
  int,
  uniform,
  sin,
  cos,
  atan,
  acos,
  length,
  max,
  min,
  clamp,
  instanceIndex,
  textureStore,
} from "three/tsl";

// Pseudo 3D noise: cheap, fast, stable (-1..1). Not simplex, but good enough for modulation.
// Uses a fract(sin(dot())) hash pattern.
export const pseudoNoise3 = /*@__PURE__*/ Fn(({ p }) => {
  const dot1 = p.dot(vec3(127.1, 311.7, 74.7));
  const dot2 = p.dot(vec3(269.5, 183.3, 246.1));
  const dot3 = p.dot(vec3(113.5, 271.9, 124.6));
  const n = sin(dot1).mul(43758.5453)
    .add(sin(dot2).mul(24634.6345))
    .add(sin(dot3).mul(31578.9182));
  // Remap to [-1, 1]
  return n.fract().mul(2.0).sub(1.0);
});

// Spherical wave displacement adapted from astro project's shaders.ts
// Returns a vec3 displacement to apply to a position in object/local space.
// Inputs:
// - pos: vec3 (usually in [-0.5, 0.5] cube or any local/object space)
// - time: float (seconds)
// - waveAmplitude: float
// - waveSpeed: float
// - noiseScale: float
// - noiseAmplitude: float
// - center: vec3 (origin of the spherical waves)
export const sphericalWaveDisplacement = /*@__PURE__*/ Fn(
  ({
    pos,
    time = float(0),
    waveAmplitude = float(0.2),
    waveSpeed = float(1.0),
    noiseScale = float(1.0),
    noiseAmplitude = float(0.5),
    center = vec3(0.0),
  }) => {
    const p = pos.sub(center);
    const r = length(p);
    const eps = float(0.0001);

    // Guard small radius to avoid NaNs
    const safeR = max(r, eps);

    const theta = acos(p.z.div(safeR));
    const phi = atan(p.y, p.x);

    // Modulate amplitude and speed using cheap pseudo noise
    const noisePosAmp = vec3(
      p.x.mul(noiseScale).mul(0.5).add(time.mul(0.05)),
      p.y.mul(noiseScale).mul(0.5).add(time.mul(0.07)),
      p.z.mul(noiseScale).mul(0.5).add(time.mul(0.06))
    );
    const noisePosSpd = vec3(
      p.x.mul(noiseScale).mul(0.3).sub(time.mul(0.03)),
      p.y.mul(noiseScale).mul(0.3).sub(time.mul(0.04)),
      p.z.mul(noiseScale).mul(0.3).sub(time.mul(0.05))
    );

    // [-1,1] -> [0,1]
    const amplitudeNoise = pseudoNoise3({ p: noisePosAmp }).mul(0.5).add(0.5);
    const speedNoise = pseudoNoise3({ p: noisePosSpd }).mul(0.5).add(0.5);

    const localAmplitude = waveAmplitude.mul(float(1.0).add(amplitudeNoise.mul(noiseAmplitude)));
    const localSpeed = waveSpeed.mul(float(1.0).add(speedNoise.mul(noiseAmplitude)));

    // Wave term; divide by radius to emphasize near the origin, clamp denominator
    const wave = r.mul(10.0).sub(time.mul(localSpeed)).sin().div(max(r, float(0.1)));

    const disp = vec3(
      localAmplitude.mul(phi.cos()).mul(theta.sin()).mul(wave),
      localAmplitude.mul(phi.sin()).mul(theta.sin()).mul(wave),
      localAmplitude.mul(theta.cos()).mul(wave)
    );

    // Zero out near the center to avoid exploding displacements
    const mask = r.greaterThan(float(0.01)).select(float(1.0), float(0.0));
    return disp.mul(mask);
  }
);

// Helper to compute displaced sampling coordinate inside a [0,1]^3 texture space
// texCoord: vec3 in [0,1]
export const displacedTexCoord = /*@__PURE__*/ Fn(
  ({ texCoord, time, waveAmplitude, waveSpeed, noiseScale, noiseAmplitude, scale = float(0.1) }) => {
    const local = texCoord.sub(0.5);
    const offset = sphericalWaveDisplacement({
      pos: local,
      time,
      waveAmplitude,
      waveSpeed,
      noiseScale,
      noiseAmplitude,
      center: vec3(0.0),
    });
    const samplePos = texCoord.add(offset.mul(scale));
    const clamped = clamp(samplePos, vec3(0.0), vec3(1.0));
    return clamped;
  }
);

// Build a compute kernel that copies from a 3D source texture to a 3D storage texture,
// sampling the source at a spherical-wave displaced coordinate.
// width/height/depth are JS numbers used for dispatch sizing and indexing math.
export function buildSphericalWaveCopyKernel({
  width,
  height,
  depth,
  storageTexture,
  sourceTextureNode,
  waveAmplitude = uniform(0.1),
  waveSpeed = uniform(1.0),
  noiseScale = uniform(1.0),
  noiseAmplitude = uniform(0.5),
  time = uniform(0.0),
  intensityScale = uniform(1.0),
}) {
  return Fn(() => {
    const id = instanceIndex;

    const x = id.mod(width);
    const y = id.div(width).mod(height);
    const z = id.div(width * height);

    const fx = float(x).add(0.5).div(width);
    const fy = float(y).add(0.5).div(height);
    const fz = float(z).add(0.5).div(depth);

    const uvw = vec3(fx, fy, fz);
    const displaced = displacedTexCoord({
      texCoord: uvw,
      time,
      waveAmplitude,
      waveSpeed,
      noiseScale,
      noiseAmplitude,
    });

    const s = sourceTextureNode.sample(displaced).r.mul(intensityScale);

    textureStore(
      storageTexture,
      vec3(x, y, z),
      vec4(s, 0.0, 0.0, 1.0)
    );
  });
}


