import { attribute, mix, Fn, positionGeometry, float } from "three/tsl";

export const knotMorphPosition = Fn(({ mixFactor = float(0) }) => {
  const targetPosition = attribute("targetPosition");
  return mix(positionGeometry, targetPosition, mixFactor);
});
