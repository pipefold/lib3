import {
  Fn,
  float,
  int,
  uint,
  ivec2,
  uvec2,
  vec4,
  instanceIndex,
  textureStore,
  Loop,
  If,
} from "three/tsl";
import {
  LinearFilter,
  LinearMipmapNearestFilter,
  LinearSRGBColorSpace,
  RepeatWrapping,
} from "three";
import StorageTexture from "three/src/renderers/common/StorageTexture.js";

/**
 * Hilbert-curve R1 blue noise value for a 2D integer coordinate.
 *
 * Maps (x, y) → 1D Hilbert index at the given recursion level,
 * then applies a Knuth multiplicative hash (golden-ratio R1 sequence)
 * to produce a value in [0, 1) with blue-noise spectral distribution.
 *
 * @param {Object} params
 * @param {Node<ivec2>} params.p - integer 2D coordinate
 * @param {Node<int>}   params.level - Hilbert curve recursion depth
 * @returns {Node<float>} blue noise value in [0, 1)
 */
const hilbertR1BlueNoise = /*@__PURE__*/ Fn(({ p, level }) => {
  // Working copies that we mutate through the loop
  const x = int(p.x).toVar();
  const y = int(p.y).toVar();
  const d = uint(0).toVar();

  // Iterate from the most-significant bit down to the least-significant bit.
  // At each step n = level - i - 1 we inspect bit n of x and y,
  // accumulate the Hilbert index contribution, then apply the
  // sub-quadrant transform (reflect + transpose).
  Loop({ start: int(0), end: level, type: "int", condition: "<" }, ({ i }) => {
    const n = int(level).sub(i).sub(1);
    const rx = int(x.shiftRight(n).bitAnd(1));
    const ry = int(y.shiftRight(n).bitAnd(1));

    // d += ((3 * rx) ^ ry) << (2 * n)
    const contribution = uint(int(3).mul(rx).bitXor(ry)).shiftLeft(
      uint(int(2).mul(n)),
    );
    d.addAssign(contribution);

    // Hilbert sub-quadrant transform when ry == 0
    If(ry.equal(0), () => {
      const mask = int(1).shiftLeft(n).sub(1);
      // Reflect both axes when rx == 1
      If(rx.equal(1), () => {
        x.assign(mask.sub(x));
        y.assign(mask.sub(y));
      });
      // Transpose: swap x ↔ y
      const tmp = int(x).toVar();
      x.assign(y);
      y.assign(tmp);
    });
  });

  // Knuth multiplicative hash (R1 sequence via golden ratio)
  // h(d) = 0x80000000 + 2654435789 * d   (wrapping u32 arithmetic)
  const h = uint(0x80000000).add(uint(2654435789).mul(d));

  // Normalize to [0, 1)
  return float(h).div(4294967296.0);
});

/**
 * Procedural mip-aware blue noise texture generator.
 *
 * Each mip level is independently computed with its own blue-noise pattern
 * via a Hilbert-curve R1 low-discrepancy sequence on the GPU — no precomputed
 * LUT required.  This preserves blue-noise spectral properties at every mip
 * level, unlike standard box-filter downsampling which destroys them.
 *
 * @example
 * const blueNoise = new ComputeMipAwareBlueNoise(128, 128, 1.0);
 * const texture = blueNoise.init(renderer);
 * // Sample in a material:
 * // jitter = texture.sample(screenUV * 4.0).x;
 */
export class ComputeMipAwareBlueNoise {
  /**
   * @param {number} [width=128]  - Texture width (should be power of two).
   * @param {number} [height=128] - Texture height (should be power of two).
   * @param {number} [mipScaleExponent=1.0] - Controls coordinate scaling across
   *   mip levels.  1.0 = no scaling; 0.5 = √2 per level; 0.0 = full compensation.
   */
  constructor(width = 128, height = 128, mipScaleExponent = 1.0) {
    /** @type {number} */ this.width = width;
    /** @type {number} */ this.height = height;
    /** @type {number} */ this.mipScaleExponent = mipScaleExponent;
    /** @type {StorageTexture|null} */ this.storageTexture = null;
  }

  /**
   * Dispatches compute shaders to fill every mip level with independent
   * blue noise.  Each mip level gets its own Hilbert-curve R1 pattern.
   *
   * @param {import('three').WebGPURenderer} renderer - A WebGPU-capable renderer.
   * @returns {StorageTexture} The fully-populated storage texture.
   */
  init(renderer) {
    const { width, height, mipScaleExponent } = this;

    // Full mip chain: 1 + floor(log2(max(width, height)))
    const mipCount =
      1 + Math.floor(Math.log2(Math.max(width, height)));

    // --- Create storage texture ---
    const tex = new StorageTexture(width, height);
    tex.colorSpace = LinearSRGBColorSpace;
    tex.minFilter = LinearMipmapNearestFilter;
    tex.magFilter = LinearFilter;
    tex.wrapS = RepeatWrapping;
    tex.wrapT = RepeatWrapping;
    // We write each mip explicitly; disable auto mipmap generation.
    tex.generateMipmaps = true;
    tex.mipmapsAutoUpdate = false;

    this.storageTexture = tex;

    // --- Dispatch a compute shader for each mip level ---
    for (let L = 0; L < mipCount; L++) {
      const mipWidth = Math.max(1, width >> L);
      const mipHeight = Math.max(1, height >> L);

      // scale = 2^((1 - mipScaleExponent) * L)
      const scale = Math.pow(2, (1 - mipScaleExponent) * L);

      // Hilbert recursion depth covers the effective coordinate range
      const effectiveSize = Math.max(mipWidth, mipHeight) * scale;
      const hilbertLevel = Math.max(
        1,
        Math.ceil(Math.log2(Math.max(1, effectiveSize))),
      );

      // Capture per-iteration constants for the Fn closure
      const mw = mipWidth;
      const mh = mipHeight;
      const sc = scale;
      const hl = hilbertLevel;
      const mipLevel = L;

      const computeFn = Fn(() => {
        const idx = instanceIndex;
        const px = int(idx.mod(mw));
        const py = int(idx.div(mw));

        // Scale pixel coords to Hilbert input coords
        const coord = ivec2(
          float(px).mul(sc).toInt(),
          float(py).mul(sc).toInt(),
        );

        const v = hilbertR1BlueNoise({ p: coord, level: int(hl) });

        // Write grayscale + alpha=1 into this mip level
        textureStore(tex, uvec2(uint(px), uint(py)), vec4(v, v, v, 1.0))
          .setMipLevel(mipLevel);
      });

      const computeNode = computeFn().compute(mw * mh);
      renderer.compute(computeNode);
    }

    return tex;
  }

  /**
   * Returns the generated texture, or `null` if `init()` has not been called.
   *
   * @returns {StorageTexture|null}
   */
  getTexture() {
    return this.storageTexture;
  }
}
