// 1D Euclidean Distance Transform (Felzenszwalb & Huttenlocher)
// Computes squared distances; caller takes sqrt if needed.
// Ported from crush repo: https://github.com/pipefold/crush

const INF = 1e20;

/**
 * 1D distance transform
 * @param {Float64Array} f - Input values
 * @param {Float64Array} d - Output distances
 * @param {Int32Array} v - Location of parabolas
 * @param {Float64Array} z - Locations of boundaries
 * @param {number} n - Array length
 */
function edt1d(f, d, v, z, n) {
  v[0] = 0;
  z[0] = -INF;
  z[1] = INF;
  let k = 0;

  for (let q = 1; q < n; q++) {
    let s;
    do {
      const r = v[k];
      s = (f[q] - f[r] + q * q - r * r) / (2 * q - 2 * r);
      if (s > z[k]) break;
      k--;
    } while (k >= 0);
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = INF;
  }

  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const dx = q - v[k];
    d[q] = dx * dx + f[v[k]];
  }
}

/**
 * Compute a signed distance field from a binary alpha image.
 * Returns Float64Array of signed distances (negative = inside).
 * @param {Uint8ClampedArray} imageData - RGBA image data
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @param {number} alphaThreshold - Threshold for determining inside/outside (default 128)
 * @returns {Float64Array} Signed distances
 */
export function computeSDF(imageData, width, height, alphaThreshold = 128) {
  const size = width * height;
  const outside = new Float64Array(size);
  const inside = new Float64Array(size);

  // Initialize: outside gets 0 where glyph, INF where background
  //             inside  gets INF where glyph, 0 where background
  for (let i = 0; i < size; i++) {
    const a = imageData[i * 4 + 3]; // alpha channel
    if (a >= alphaThreshold) {
      outside[i] = 0;
      inside[i] = INF;
    } else {
      outside[i] = INF;
      inside[i] = 0;
    }
  }

  edt2d(outside, width, height);
  edt2d(inside, width, height);

  const sdf = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    sdf[i] = Math.sqrt(outside[i]) - Math.sqrt(inside[i]);
  }
  return sdf;
}

/**
 * 2D Euclidean distance transform
 * @param {Float64Array} grid - Grid values (modified in place)
 * @param {number} width - Grid width
 * @param {number} height - Grid height
 */
function edt2d(grid, width, height) {
  const maxDim = Math.max(width, height);
  const f = new Float64Array(maxDim);
  const d = new Float64Array(maxDim);
  const v = new Int32Array(maxDim);
  const z = new Float64Array(maxDim + 1);

  // Transform columns
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) f[y] = grid[y * width + x];
    edt1d(f, d, v, z, height);
    for (let y = 0; y < height; y++) grid[y * width + x] = d[y];
  }

  // Transform rows
  for (let y = 0; y < height; y++) {
    const offset = y * width;
    for (let x = 0; x < width; x++) f[x] = grid[offset + x];
    edt1d(f, d, v, z, width);
    for (let x = 0; x < width; x++) grid[offset + x] = d[x];
  }
}

// Export constants for configuration
export const SDF_DEFAULTS = {
  GLYPH_SIZE: 64,
  SDF_SIZE: 32,
  SDF_PADDING: 4,
  MAX_DISTANCE: 8,
  ALPHA_THRESHOLD: 128,
};