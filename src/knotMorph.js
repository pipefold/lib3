import { uniform, attribute, mix, Fn, positionGeometry } from "three/tsl";

export const knotMorphMixFactor = uniform(0);

export const knotMorphPosition = Fn(() => {
  const targetPosition = attribute("targetPosition");
  return mix(positionGeometry, targetPosition, knotMorphMixFactor);
});
