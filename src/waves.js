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
  floor,
  step,
  abs,
  instanceIndex,
  textureStore,
} from "three/tsl";

// Pseudo 3D noise: cheap, fast, stable (-1..1). Not simplex, but good enough for modulation.
// Uses a fract(sin(dot())) hash pattern.
export const pseudoNoise3 = /*@__PURE__*/ Fn(({ p }) => {
  const dot1 = p.dot(vec3(127.1, 311.7, 74.7));
  const dot2 = p.dot(vec3(269.5, 183.3, 246.1));
  const dot3 = p.dot(vec3(113.5, 271.9, 124.6));
  const n = sin(dot1)
    .mul(43758.5453)
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

    // Use normalized direction instead of spherical angles to avoid acos/atan
    const dir = p.div(safeR);

    // Modulate amplitude and speed using simplex noise (higher quality)
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
    const amplitudeNoise = simplexNoise3({ v: noisePosAmp }).mul(0.5).add(0.5);
    const speedNoise = simplexNoise3({ v: noisePosSpd }).mul(0.5).add(0.5);

    const localAmplitude = waveAmplitude.mul(
      float(1.0).add(amplitudeNoise.mul(noiseAmplitude))
    );
    const localSpeed = waveSpeed.mul(
      float(1.0).add(speedNoise.mul(noiseAmplitude))
    );

    // Wave term; divide by radius to emphasize near the origin, clamp denominator
    const wave = r
      .mul(10.0)
      .sub(time.mul(localSpeed))
      .sin()
      .div(max(r, float(0.1)));

    const disp = dir.mul(localAmplitude).mul(wave);

    // Zero out near the center to avoid exploding displacements
    const mask = r.greaterThan(float(0.01)).select(float(1.0), float(0.0));
    return disp.mul(mask);
  }
);

// Helper to compute displaced sampling coordinate inside a [0,1]^3 texture space
// texCoord: vec3 in [0,1]
export const displacedTexCoord = /*@__PURE__*/ Fn(
  ({
    texCoord,
    time,
    waveAmplitude,
    waveSpeed,
    noiseScale,
    noiseAmplitude,
    scale = float(0.1),
  }) => {
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

    textureStore(storageTexture, vec3(x, y, z), vec4(s, 0.0, 0.0, 1.0));
  });
}

// --- High-quality 3D simplex noise implementation (float in [-1, 1]) ---

const mod289v3 = /*@__PURE__*/ Fn(({ x }) => {
  return x.sub(floor(x.mul(1.0 / 289.0)).mul(289.0));
});

const mod289v4 = /*@__PURE__*/ Fn(({ x }) => {
  return x.sub(floor(x.mul(1.0 / 289.0)).mul(289.0));
});

const permute4 = /*@__PURE__*/ Fn(({ x }) => {
  return mod289v4({ x: x.mul(34.0).add(1.0).mul(x) });
});

const taylorInvSqrt4 = /*@__PURE__*/ Fn(({ r }) => {
  return vec4(1.79284291400159).sub(vec4(0.85373472095314).mul(r));
});

export const simplexNoise3 = /*@__PURE__*/ Fn(({ v }) => {
  const Cx = vec2(1.0 / 6.0, 1.0 / 3.0);
  const D = vec4(0.0, 0.5, 1.0, 2.0);

  // Skew factor and base corner
  const i = floor(v.add(vec3(v.dot(vec3(Cx.y))))).toVar();
  const x0 = v
    .sub(i)
    .add(vec3(i.dot(vec3(Cx.x))))
    .toVar();

  // Rank order for simplex corners
  const g = step(x0.yzx, x0.xyz);
  const l = vec3(1.0).sub(g);
  const i1 = vec3(min(g.xyz, l.zxy));
  const i2 = vec3(max(g.xyz, l.zxy));

  // Offsets for other corners
  const x1 = x0.sub(i1).add(vec3(Cx.x));
  const x2 = x0.sub(i2).add(vec3(Cx.y));
  const x3 = x0.sub(vec3(D.y));

  // Permutations
  i.assign(mod289v3({ x: i }));
  const p = permute4({
    x: permute4({
      x: permute4({ x: vec4(i.z).add(vec4(0.0, i1.z, i2.z, 1.0)) }).add(
        vec4(i.y).add(vec4(0.0, i1.y, i2.y, 1.0))
      ),
    }).add(vec4(i.x).add(vec4(0.0, i1.x, i2.x, 1.0))),
  });

  // Gradients
  const n_ = float(0.142857142857); // 1/7
  const ns = vec3(n_.mul(D.wyz)).sub(vec3(D.xzx));

  const j = p.sub(vec4(49.0).mul(floor(p.mul(ns.z).mul(ns.z))));

  const x_ = floor(j.mul(ns.z));
  const y_ = floor(j.sub(vec4(7.0).mul(x_)));

  const x = x_.mul(ns.x).add(ns.y);
  const y = y_.mul(ns.x).add(ns.y);
  const h = vec4(1.0).sub(abs(x)).sub(abs(y));

  const b0 = vec4(x.xy, y.xy);
  const b1 = vec4(x.zw, y.zw);

  const s0 = floor(b0).mul(2.0).add(1.0);
  const s1 = floor(b1).mul(2.0).add(1.0);
  const sh = step(h, vec4(0.0)).negate();

  const a0 = vec4(b0.xzyw).add(vec4(s0.xzyw).mul(vec4(sh.xxyy)));
  const a1 = vec4(b1.xzyw).add(vec4(s1.xzyw).mul(vec4(sh.zzww)));

  const p0 = vec3(a0.xy, h.x);
  const p1 = vec3(a0.zw, h.y);
  const p2 = vec3(a1.xy, h.z);
  const p3 = vec3(a1.zw, h.w);

  // Normalize gradients
  const norm = taylorInvSqrt4({
    r: vec4(p0.dot(p0), p1.dot(p1), p2.dot(p2), p3.dot(p3)),
  });
  p0.mulAssign(norm.x);
  p1.mulAssign(norm.y);
  p2.mulAssign(norm.z);
  p3.mulAssign(norm.w);

  // Contribution
  const m = max(
    vec4(0.6).sub(vec4(x0.dot(x0), x1.dot(x1), x2.dot(x2), x3.dot(x3))),
    vec4(0.0)
  ).toVar();
  m.mulAssign(m);

  const result = vec4(m.mul(m)).dot(
    vec4(vec3(p0).dot(x0), vec3(p1).dot(x1), vec3(p2).dot(x2), vec3(p3).dot(x3))
  );
  return float(42.0).mul(result);
});
