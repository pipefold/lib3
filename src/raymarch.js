import {
  varying,
  vec4,
  modelWorldMatrixInverse,
  cameraPosition,
  positionGeometry,
  float,
  Fn,
  Loop,
  max,
  min,
  vec2,
  vec3,
  Break,
  bool,
  int,
  If,
} from "three/tsl";
import { RaymarchingBox } from "three/addons/tsl/utils/Raymarching.js";

const hitBox = /*@__PURE__*/ Fn(({ orig, dir }) => {
  const box_min = vec3(-0.5);
  const box_max = vec3(0.5);

  const inv_dir = dir.reciprocal();

  const tmin_tmp = box_min.sub(orig).mul(inv_dir);
  const tmax_tmp = box_max.sub(orig).mul(inv_dir);

  const tmin = min(tmin_tmp, tmax_tmp);
  const tmax = max(tmin_tmp, tmax_tmp);

  const t0 = max(tmin.x, max(tmin.y, tmin.z));
  const t1 = min(tmax.x, min(tmax.y, tmax.z));

  return vec2(t0, t1);
});

export const adaptiveRaymarch = (
  maxSteps,
  callback,
  threshold = float(0.001)
) => {
  const vOrigin = varying(
    vec3(modelWorldMatrixInverse.mul(vec4(cameraPosition, 1.0)))
  );
  const vDirection = varying(positionGeometry.sub(vOrigin));

  const rayDir = vDirection.normalize();
  const bounds = vec2(hitBox({ orig: vOrigin, dir: rayDir })).toVar();

  bounds.x.greaterThan(bounds.y).discard();

  bounds.assign(vec2(max(bounds.x, 0.0), bounds.y));

  const t = float(max(bounds.x, 0.0)).toVar();
  const positionRay = vec3(vOrigin.add(rayDir.mul(t))).toVar();
  const hit = bool(false).toVar();

  Loop({ type: "int", start: int(0), end: int(maxSteps) }, () => {
    const remaining = bounds.y.sub(t);

    If(remaining.lessThanEqual(float(0)), () => {
      Break();
    });

    let delta = callback({ positionRay, maxStep: remaining });

    If(delta.lessThanEqual(threshold), () => {
      hit.assign(true);
      Break();
    });

    delta = min(delta, remaining);

    t.addAssign(delta);
    positionRay.addAssign(rayDir.mul(delta));
  });

  return { positionRay, t, bounds, hit };
};

// Average Intensity Projection over a unit box using fixed-step raymarch
export const averageIntensityProjection = /*@__PURE__*/ Fn(
  ({ texture, steps, intensityScale = float(1.0) }) => {
    const finalColor = vec4(0).toVar();
    const intensitySum = float(0).toVar();
    const sampleCount = float(0).toVar();

    RaymarchingBox(steps, ({ positionRay }) => {
      const samplePos = positionRay.add(0.5);
      const mapValue = texture.sample(samplePos).r;

      intensitySum.addAssign(mapValue);
      sampleCount.addAssign(1);
    });

    const averageIntensity = intensitySum.div(sampleCount);
    const scaledIntensity = averageIntensity.mul(intensityScale);

    finalColor.rgb.assign(vec3(scaledIntensity));
    finalColor.a.assign(1);

    return finalColor;
  }
);
