/**
 * SmokeVolume — WebGPU 3D Euler-fluid simulation.
 *
 * Implements a Jos Stam "Stable Fluids" variant with MacCormack advection,
 * vorticity confinement, Red-Black SOR pressure solve, and optional SDF
 * boundary enforcement. All GPU work uses Three.js TSL compute nodes.
 *
 * @module fluidSim
 */

import {
  Fn,
  float,
  int,
  uint,
  vec3,
  vec4,
  ivec3,
  uvec3,
  uniform,
  globalId,
  textureStore,
  texture3D,
  If,
  Return,
  max as tslMax,
  min as tslMin,
  exp,
  dot,
  cross,
  length as tslLength,
  clamp,
  round as tslRound,
  bitAnd,
} from "three/tsl";

import {
  Storage3DTexture,
  RGBAFormat,
  HalfFloatType,
  LinearFilter,
  ClampToEdgeWrapping,
  Vector3,
} from "three/webgpu";

// ─── Constants ──────────────────────────────────────────────────────────────

const WG = [4, 4, 4];

// ─── Texture helpers ────────────────────────────────────────────────────────

/**
 * Create a Storage3DTexture configured for the fluid sim (RGBA half-float,
 * linear filter, clamp-to-edge, no mipmaps).
 */
function makeTex3D(w, h, d) {
  const t = new Storage3DTexture(w, h, d);
  t.format = RGBAFormat;
  t.type = HalfFloatType;
  t.minFilter = LinearFilter;
  t.magFilter = LinearFilter;
  t.wrapS = ClampToEdgeWrapping;
  t.wrapT = ClampToEdgeWrapping;
  t.wrapR = ClampToEdgeWrapping;
  t.generateMipmaps = false;
  return t;
}

/** Dispatch dimensions for a cubic grid. */
function dispatch3D(res) {
  const n = Math.ceil(res / WG[0]);
  return [n, n, n];
}

// ─── Ping-Pong pair ─────────────────────────────────────────────────────────

class PingPong3D {
  constructor(w, h, d) {
    this.read = makeTex3D(w, h, d);
    this.write = makeTex3D(w, h, d);
    this.phase = true;
  }
  /** Current read-side texture. */
  get current() { return this.phase ? this.read : this.write; }
  /** Current write-side texture. */
  get target()  { return this.phase ? this.write : this.read; }
  swap() { this.phase = !this.phase; }
  dispose() { this.read.dispose(); this.write.dispose(); }
}

// ─── SmokeVolume ────────────────────────────────────────────────────────────

/**
 * @typedef {Object} SmokeVolumeOptions
 * @property {number} [simRes=96]
 * @property {number} [dyeRes=96]
 * @property {number} [iterations=40]
 * @property {number} [densityDissipation=0.995]
 * @property {number} [velocityDissipation=0.98]
 * @property {number} [pressureDissipation=0.98]
 * @property {number} [curlStrength=20]
 * @property {number} [pressureFactor=0.16667]
 * @property {number} [radius=0.2]
 * @property {boolean} [useBoundaries=true]
 * @property {number} [neighborStride=1]
 * @property {number} [speedFactor=1]
 * @property {number} [buoyancyStrength=0]
 * @property {object|null} [sdfVolumeConstraint=null]
 */

export class SmokeVolume {
  /**
   * @param {SmokeVolumeOptions} [opts]
   */
  constructor(opts = {}) {
    const {
      simRes = 96,
      dyeRes = 96,
      iterations = 40,
      densityDissipation = 0.995,
      velocityDissipation = 0.98,
      pressureDissipation = 0.98,
      curlStrength = 20,
      pressureFactor = 1 / 6,
      radius = 0.2,
      useBoundaries = true,
      neighborStride = 1,
      speedFactor = 1,
      buoyancyStrength = 0,
      sdfVolumeConstraint = null,
    } = opts;

    this.simRes = simRes;
    this.dyeRes = dyeRes;
    this.iterations = iterations;
    this.speedFactor = speedFactor;
    this.sdfVolumeConstraint = sdfVolumeConstraint;

    // ── TSL uniforms (live-writable via .value) ──
    this.radius              = uniform(radius);
    this.curlStrength        = uniform(curlStrength);
    this.densityDissipation  = uniform(densityDissipation);
    this.velocityDissipation = uniform(velocityDissipation);
    this.pressureDissipation = uniform(pressureDissipation);
    this.neighborStride      = uniform(neighborStride);
    this.deltaTime           = uniform(0.004);
    this.buoyancyStrength    = uniform(buoyancyStrength);
    this.pressureFactor      = uniform(pressureFactor);
    this.sorOmega            = uniform(Math.min(1.99, 2 / (1 + Math.sin(Math.PI / simRes))));
    this.point3D             = uniform(new Vector3(999, 999, 999));
    this.force3D             = uniform(new Vector3(0, 0, 0));

    // Internal uniforms
    /** @private */ this._pressureParity  = uniform(0);
    /** @private */ this._useBoundariesU  = uniform(useBoundaries ? 1.0 : 0.0);

    // ── 3D Textures ──
    this._velocity3D            = new PingPong3D(simRes, simRes, simRes);
    this._pressure3D            = new PingPong3D(simRes, simRes, simRes);
    this._density3D             = new PingPong3D(dyeRes, dyeRes, dyeRes);
    this._curl3D                = makeTex3D(simRes, simRes, simRes);
    this._divergence3D          = makeTex3D(simRes, simRes, simRes);
    this._macCormackTemp3D      = makeTex3D(simRes, simRes, simRes);
    this._macCormackTempDensity = makeTex3D(dyeRes, dyeRes, dyeRes);

    // ── Debug flags ──
    this.debug = {
      runCurl: true,
      runVorticity: true,
      runDivergence: true,
      runPressureClear: true,
      runPressureJacobi: true,
      runProjection: true,
      runAdvectVelocity: false,
      runAdvectDensity: true,
      runBuoyancy: true,
      autoStep: true,
    };

    /** @private */ this._splatQueue = [];

    // Pre-build two variants (one per phase) of every pass
    this._buildAllPasses();
  }

  // ─── Texture accessors ──────────────────────────────────────────────────

  getDensityTexture3D()    { return this._density3D.current; }
  getVelocityTexture3D()   { return this._velocity3D.current; }
  getPressureTexture3D()   { return this._pressure3D.current; }
  getDivergenceTexture3D() { return this._divergence3D; }
  getCurlTexture3D()       { return this._curl3D; }

  // ─── Public methods ─────────────────────────────────────────────────────

  /**
   * Queue a splat at normalised coords [0,1]³ with optional force vector.
   */
  addSplat(x, y, z, fx = 0, fy = 0, fz = 0) {
    this._splatQueue.push({ x, y, z, fx, fy, fz });
    return this;
  }

  /** Run one pressure-clear pass. */
  clearPressure(renderer) {
    renderer.compute(this._pickPass(this._pressureClearPasses, this._pressure3D));
    this._pressure3D.swap();
  }

  /** Merge into debug flags. */
  setDebugPasses(passes) {
    Object.assign(this.debug, passes);
    return this;
  }

  /** Toggle wall boundary conditions at runtime. */
  setUseBoundaries(enabled) {
    this._useBoundariesU.value = enabled ? 1.0 : 0.0;
    return this;
  }

  /**
   * Run one full simulation tick.
   * @param {WebGPURenderer} renderer
   * @param {number} [dt] — override deltaTime
   */
  step(renderer, dt) {
    if (dt !== undefined) this.deltaTime.value = dt;

    // 1. Drain splat queue
    while (this._splatQueue.length) {
      const s = this._splatQueue.shift();
      this.point3D.value.set(s.x, s.y, s.z);
      this.force3D.value.set(s.fx, s.fy, s.fz);

      renderer.compute(this._pickPass(this._splatVelPasses, this._velocity3D));
      this._velocity3D.swap();

      renderer.compute(this._pickPass(this._splatDenPasses, this._density3D));
      this._density3D.swap();
    }
    this.point3D.value.set(999, 999, 999);

    // 2. Curl
    if (this.debug.runCurl) {
      renderer.compute(this._pickPass(this._curlPasses, this._velocity3D));
    }

    // 3. Vorticity confinement
    if (this.debug.runVorticity) {
      renderer.compute(this._pickPass(this._vorticityPasses, this._velocity3D));
      this._velocity3D.swap();
    }

    // 4. Buoyancy
    if (this.debug.runBuoyancy) {
      renderer.compute(this._pickBuoyancyPass());
      this._velocity3D.swap();
    }

    // 5. Divergence
    if (this.debug.runDivergence) {
      renderer.compute(this._pickPass(this._divergencePasses, this._velocity3D));
    }

    // 6. Pressure clear
    if (this.debug.runPressureClear) {
      renderer.compute(this._pickPass(this._pressureClearPasses, this._pressure3D));
      this._pressure3D.swap();
    }

    // 7. Pressure solve — Red-Black SOR
    if (this.debug.runPressureJacobi) {
      const iters = Math.max(1, Math.round(this.iterations / this.speedFactor));
      for (let i = 0; i < iters; i++) {
        this._pressureParity.value = 0;
        renderer.compute(this._pickPass(this._pressureSolvePasses, this._pressure3D));
        this._pressure3D.swap();

        this._pressureParity.value = 1;
        renderer.compute(this._pickPass(this._pressureSolvePasses, this._pressure3D));
        this._pressure3D.swap();
      }
    }

    // 8. Projection
    if (this.debug.runProjection) {
      renderer.compute(this._pickProjectionPass());
      this._velocity3D.swap();
    }

    // 9. (SDF boundary enforcement — placeholder)

    // 10. Velocity advection (MacCormack, 2 dispatches)
    if (this.debug.runAdvectVelocity) {
      renderer.compute(this._pickPass(this._advVelFwdPasses, this._velocity3D));
      renderer.compute(this._pickPass(this._advVelCorrPasses, this._velocity3D));
      this._velocity3D.swap();
    }

    // 11. Density advection (MacCormack, 2 dispatches)
    if (this.debug.runAdvectDensity) {
      renderer.compute(this._pickAdvDenFwdPass());
      renderer.compute(this._pickPass(this._advDenCorrPasses, this._density3D));
      this._density3D.swap();
    }
  }

  /** Release all GPU textures. */
  dispose() {
    this._velocity3D.dispose();
    this._pressure3D.dispose();
    this._density3D.dispose();
    this._curl3D.dispose();
    this._divergence3D.dispose();
    this._macCormackTemp3D.dispose();
    this._macCormackTempDensity.dispose();
  }

  // ─── Pass selection helpers ─────────────────────────────────────────────

  /** Pick variant 0 or 1 from a [false, true]-indexed pair. */
  _pickPass(arr, pp) { return arr[pp.phase ? 1 : 0]; }

  /** Buoyancy reads both velocity and density ping-pongs → 2×2 variants. */
  _pickBuoyancyPass() {
    return this._buoyancyPasses[this._velocity3D.phase ? 1 : 0][this._density3D.phase ? 1 : 0];
  }

  /** Projection reads both pressure and velocity → 2×2 variants. */
  _pickProjectionPass() {
    return this._projectionPasses[this._velocity3D.phase ? 1 : 0][this._pressure3D.phase ? 1 : 0];
  }

  /** Density advection forward reads velocity + density → 2×2 variants. */
  _pickAdvDenFwdPass() {
    return this._advDenFwdPasses[this._velocity3D.phase ? 1 : 0][this._density3D.phase ? 1 : 0];
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // BUILD ALL PASSES
  // ═══════════════════════════════════════════════════════════════════════════

  _buildAllPasses() {
    const S = this.simRes;
    const D = this.dyeRes;
    const sDisp = dispatch3D(S);
    const dDisp = dispatch3D(D);

    // Helper: for a ping-pong pair, return [phase=false tex, phase=true tex]
    const rw = (pp) => [
      { r: pp.write, w: pp.read  },   // phase=false → read=write, write=read
      { r: pp.read,  w: pp.write },   // phase=true  → read=read,  write=write
    ];

    const vel = rw(this._velocity3D);
    const prs = rw(this._pressure3D);
    const den = rw(this._density3D);

    // ── Curl ──
    this._curlPasses = [0, 1].map(i =>
      this._mkCurl(vel[i].r, this._curl3D, S)
        .compute(sDisp, WG)
    );

    // ── Vorticity ──
    this._vorticityPasses = [0, 1].map(i =>
      this._mkVorticity(this._curl3D, vel[i].r, vel[i].w, S)
        .compute(sDisp, WG)
    );

    // ── Buoyancy (vel × den → 2×2) ──
    this._buoyancyPasses = [0, 1].map(vi =>
      [0, 1].map(di =>
        this._mkBuoyancy(den[di].r, vel[vi].r, vel[vi].w, S, D)
          .compute(sDisp, WG)
      )
    );

    // ── Divergence ──
    this._divergencePasses = [0, 1].map(i =>
      this._mkDivergence(vel[i].r, this._divergence3D, S)
        .compute(sDisp, WG)
    );

    // ── Pressure clear ──
    this._pressureClearPasses = [0, 1].map(i =>
      this._mkPressureClear(prs[i].r, prs[i].w, S)
        .compute(sDisp, WG)
    );

    // ── Pressure solve ──
    this._pressureSolvePasses = [0, 1].map(i =>
      this._mkPressureSolve(this._divergence3D, prs[i].r, prs[i].w, S)
        .compute(sDisp, WG)
    );

    // ── Projection (vel × prs → 2×2) ──
    this._projectionPasses = [0, 1].map(vi =>
      [0, 1].map(pi =>
        this._mkProjection(prs[pi].r, vel[vi].r, vel[vi].w, S)
          .compute(sDisp, WG)
      )
    );

    // ── Splat velocity ──
    this._splatVelPasses = [0, 1].map(i =>
      this._mkSplatVel(vel[i].r, vel[i].w, S)
        .compute(sDisp, WG)
    );

    // ── Splat density ──
    this._splatDenPasses = [0, 1].map(i =>
      this._mkSplatDen(den[i].r, den[i].w, D)
        .compute(dDisp, WG)
    );

    // ── Advect velocity forward ──
    this._advVelFwdPasses = [0, 1].map(i =>
      this._mkAdvectForward(vel[i].r, this._macCormackTemp3D, S, S, this.velocityDissipation)
        .compute(sDisp, WG)
    );

    // ── Advect velocity MacCormack correction ──
    this._advVelCorrPasses = [0, 1].map(i =>
      this._mkMacCormackCorr(vel[i].r, this._macCormackTemp3D, vel[i].w, S, this.velocityDissipation)
        .compute(sDisp, WG)
    );

    // ── Advect density forward (vel × den → 2×2) ──
    this._advDenFwdPasses = [0, 1].map(vi =>
      [0, 1].map(di =>
        this._mkAdvectDenForward(vel[vi].r, den[di].r, this._macCormackTempDensity, S, D, this.densityDissipation)
          .compute(dDisp, WG)
      )
    );

    // ── Advect density MacCormack correction ──
    this._advDenCorrPasses = [0, 1].map(i =>
      this._mkMacCormackCorr(den[i].r, this._macCormackTempDensity, den[i].w, D, this.densityDissipation)
        .compute(dDisp, WG)
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // COMPUTE KERNELS
  // ═══════════════════════════════════════════════════════════════════════════

  // ── helper: early-exit guard ──

  /**
   * Returns a TSL Fn callback guard snippet.
   * Call at the top of every kernel to skip out-of-bounds threads.
   */
  static _guard(res) {
    const R = uint(res);
    const gid = uvec3(globalId);
    If(gid.x.greaterThanEqual(R).or(gid.y.greaterThanEqual(R)).or(gid.z.greaterThanEqual(R)), () => {
      Return();
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 4.1  Curl:  ω = ∇ × v
  // ─────────────────────────────────────────────────────────────────────────

  _mkCurl(velR, curlW, res) {
    const nsU = this.neighborStride;
    return Fn(() => {
      SmokeVolume._guard(res);
      const P = ivec3(globalId);
      const s = int(tslMax(nsU, 1.0));
      const mx = int(res - 1);
      const lo = ivec3(0, 0, 0);
      const hi = ivec3(mx, mx, mx);

      const v = texture3D(velR);
      const vL = v.load(clamp(P.sub(ivec3(s, 0, 0)), lo, hi)).xyz;
      const vR = v.load(clamp(P.add(ivec3(s, 0, 0)), lo, hi)).xyz;
      const vD = v.load(clamp(P.sub(ivec3(0, s, 0)), lo, hi)).xyz;
      const vU = v.load(clamp(P.add(ivec3(0, s, 0)), lo, hi)).xyz;
      const vB = v.load(clamp(P.sub(ivec3(0, 0, s)), lo, hi)).xyz;
      const vF = v.load(clamp(P.add(ivec3(0, 0, s)), lo, hi)).xyz;

      const h = float(0.5).div(float(s));

      //  ωx = (∂vz/∂y − ∂vy/∂z)·h
      //  ωy = (∂vx/∂z − ∂vz/∂x)·h
      //  ωz = (∂vy/∂x − ∂vx/∂y)·h
      const cx = vU.z.sub(vD.z).sub(vF.y.sub(vB.y)).mul(h);
      const cy = vF.x.sub(vB.x).sub(vR.z.sub(vL.z)).mul(h);
      const cz = vR.y.sub(vL.y).sub(vU.x.sub(vD.x)).mul(h);

      textureStore(curlW, P, vec4(cx, cy, cz, 1)).toWriteOnly();
    })();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 4.2  Vorticity confinement
  // ─────────────────────────────────────────────────────────────────────────

  _mkVorticity(curlTex, velR, velW, res) {
    const nsU = this.neighborStride;
    const csU = this.curlStrength;
    const dtU = this.deltaTime;

    return Fn(() => {
      SmokeVolume._guard(res);
      const P = ivec3(globalId);
      const s = int(tslMax(nsU, 1.0));
      const mx = int(res - 1);
      const lo = ivec3(0, 0, 0);
      const hi = ivec3(mx, mx, mx);

      const c = texture3D(curlTex);
      const cL = tslLength(c.load(clamp(P.sub(ivec3(s, 0, 0)), lo, hi)).xyz);
      const cR = tslLength(c.load(clamp(P.add(ivec3(s, 0, 0)), lo, hi)).xyz);
      const cD = tslLength(c.load(clamp(P.sub(ivec3(0, s, 0)), lo, hi)).xyz);
      const cU = tslLength(c.load(clamp(P.add(ivec3(0, s, 0)), lo, hi)).xyz);
      const cB = tslLength(c.load(clamp(P.sub(ivec3(0, 0, s)), lo, hi)).xyz);
      const cF = tslLength(c.load(clamp(P.add(ivec3(0, 0, s)), lo, hi)).xyz);
      const omega = c.load(P).xyz;

      const h = float(0.5).div(float(s));
      const eta = vec3(cU.sub(cD), cR.sub(cL), cF.sub(cB)).mul(h);
      const N = eta.div(tslLength(eta).add(1e-4));
      const conf = cross(N, omega).mul(csU).mul(dtU);

      const vel = texture3D(velR).load(P).xyz;
      textureStore(velW, P, vec4(vel.add(conf), 1)).toWriteOnly();
    })();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 4.3  Buoyancy
  // ─────────────────────────────────────────────────────────────────────────

  _mkBuoyancy(denR, velR, velW, simRes, dyeRes) {
    const bsU = this.buoyancyStrength;
    const dtU = this.deltaTime;

    return Fn(() => {
      SmokeVolume._guard(simRes);
      const P = ivec3(globalId);

      // Density is on a (possibly) different grid — textureLoad at integer coords
      const rho = texture3D(denR).load(P).x;
      const vel = texture3D(velR).load(P).xyz;
      const bf  = rho.sub(0.5).mul(bsU).mul(dtU);

      textureStore(velW, P, vec4(vel.add(vec3(0, bf, 0)), 1)).toWriteOnly();
    })();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 4.4  Divergence  ∇·v
  // ─────────────────────────────────────────────────────────────────────────

  _mkDivergence(velR, divW, res) {
    const nsU = this.neighborStride;
    const ubU = this._useBoundariesU;

    return Fn(() => {
      SmokeVolume._guard(res);
      const P = ivec3(globalId);
      const s = int(tslMax(nsU, 1.0));
      const mx = int(res - 1);
      const lo = ivec3(0, 0, 0);
      const hi = ivec3(mx, mx, mx);

      const v = texture3D(velR);
      const vC = v.load(P).xyz;

      const vLx = v.load(clamp(P.sub(ivec3(s, 0, 0)), lo, hi)).x.toVar("vLx");
      const vRx = v.load(clamp(P.add(ivec3(s, 0, 0)), lo, hi)).x.toVar("vRx");
      const vDy = v.load(clamp(P.sub(ivec3(0, s, 0)), lo, hi)).y.toVar("vDy");
      const vUy = v.load(clamp(P.add(ivec3(0, s, 0)), lo, hi)).y.toVar("vUy");
      const vBz = v.load(clamp(P.sub(ivec3(0, 0, s)), lo, hi)).z.toVar("vBz");
      const vFz = v.load(clamp(P.add(ivec3(0, 0, s)), lo, hi)).z.toVar("vFz");

      // Neumann no-penetration at domain walls
      If(ubU.greaterThan(0.5), () => {
        If(P.x.lessThan(s),              () => { vLx.assign(vC.x.negate()); });
        If(P.x.add(s).greaterThan(mx),   () => { vRx.assign(vC.x.negate()); });
        If(P.y.lessThan(s),              () => { vDy.assign(vC.y.negate()); });
        If(P.y.add(s).greaterThan(mx),   () => { vUy.assign(vC.y.negate()); });
        If(P.z.lessThan(s),              () => { vBz.assign(vC.z.negate()); });
        If(P.z.add(s).greaterThan(mx),   () => { vFz.assign(vC.z.negate()); });
      });

      const d = float(0.5).mul(
        vRx.sub(vLx).add(vUy).sub(vDy).add(vFz).sub(vBz)
      ).div(float(s));

      textureStore(divW, P, vec4(d, 0, 0, 1)).toWriteOnly();
    })();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 4.5  Pressure clear
  // ─────────────────────────────────────────────────────────────────────────

  _mkPressureClear(pR, pW, res) {
    const pdU = this.pressureDissipation;
    return Fn(() => {
      SmokeVolume._guard(res);
      const P = ivec3(globalId);
      const old = texture3D(pR).load(P);
      textureStore(pW, P, old.mul(pdU)).toWriteOnly();
    })();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 4.6  Pressure solve — Red-Black SOR Jacobi
  // ─────────────────────────────────────────────────────────────────────────

  _mkPressureSolve(divTex, pR, pW, res) {
    const nsU  = this.neighborStride;
    const pfU  = this.pressureFactor;
    const soU  = this.sorOmega;
    const parU = this._pressureParity;

    return Fn(() => {
      SmokeVolume._guard(res);
      const P = ivec3(globalId);
      const s = int(tslMax(nsU, 1.0));
      const mx = int(res - 1);
      const lo = ivec3(0, 0, 0);
      const hi = ivec3(mx, mx, mx);

      // Red-black: parity = (x/s + y/s + z/s) & 1
      const cellPar = bitAnd(P.x.div(s).add(P.y.div(s)).add(P.z.div(s)), int(1));
      const targetPar = int(tslRound(parU));

      const pr = texture3D(pR);

      // Not our colour → copy through
      If(cellPar.notEqual(targetPar), () => {
        textureStore(pW, P, pr.load(P)).toWriteOnly();
        Return();
      });

      const pL = pr.load(clamp(P.sub(ivec3(s, 0, 0)), lo, hi)).x;
      const pRt = pr.load(clamp(P.add(ivec3(s, 0, 0)), lo, hi)).x;
      const pD = pr.load(clamp(P.sub(ivec3(0, s, 0)), lo, hi)).x;
      const pU = pr.load(clamp(P.add(ivec3(0, s, 0)), lo, hi)).x;
      const pB = pr.load(clamp(P.sub(ivec3(0, 0, s)), lo, hi)).x;
      const pF = pr.load(clamp(P.add(ivec3(0, 0, s)), lo, hi)).x;

      const div = texture3D(divTex).load(P).x;
      const pC  = pr.load(P).x;
      const hSq = float(s).mul(float(s));

      const jacobi = pfU.mul(pL.add(pRt).add(pD).add(pU).add(pB).add(pF).sub(div.mul(hSq)));
      const result = pC.add(soU.mul(jacobi.sub(pC)));

      textureStore(pW, P, vec4(result, 0, 0, 1)).toWriteOnly();
    })();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 4.7  Projection (gradient subtraction)
  // ─────────────────────────────────────────────────────────────────────────

  _mkProjection(pR, velR, velW, res) {
    const nsU = this.neighborStride;

    return Fn(() => {
      SmokeVolume._guard(res);
      const P = ivec3(globalId);
      const s = int(tslMax(nsU, 1.0));
      const mx = int(res - 1);
      const lo = ivec3(0, 0, 0);
      const hi = ivec3(mx, mx, mx);

      const pr = texture3D(pR);
      const pL  = pr.load(clamp(P.sub(ivec3(s, 0, 0)), lo, hi)).x;
      const pRt = pr.load(clamp(P.add(ivec3(s, 0, 0)), lo, hi)).x;
      const pD  = pr.load(clamp(P.sub(ivec3(0, s, 0)), lo, hi)).x;
      const pU  = pr.load(clamp(P.add(ivec3(0, s, 0)), lo, hi)).x;
      const pB  = pr.load(clamp(P.sub(ivec3(0, 0, s)), lo, hi)).x;
      const pF  = pr.load(clamp(P.add(ivec3(0, 0, s)), lo, hi)).x;

      const h = float(0.5).div(float(s));
      const grad = vec3(
        pRt.sub(pL).mul(h),
        pU.sub(pD).mul(h),
        pF.sub(pB).mul(h)
      );

      const vel = texture3D(velR).load(P).xyz;
      textureStore(velW, P, vec4(vel.sub(grad), 1)).toWriteOnly();
    })();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 5.1  Splat velocity
  // ─────────────────────────────────────────────────────────────────────────

  _mkSplatVel(velR, velW, res) {
    const radU = this.radius;
    const ptU  = this.point3D;
    const fU   = this.force3D;

    return Fn(() => {
      SmokeVolume._guard(res);
      const P = ivec3(globalId);
      const gridDim = vec3(float(res));
      const gp = vec3(globalId).add(0.5).div(gridDim);
      const delta = gp.sub(ptU);
      const g = exp(dot(delta, delta).negate().div(radU.div(100.0)));

      const old = texture3D(velR).load(P).xyz;
      textureStore(velW, P, vec4(old.add(vec3(fU).mul(g)), 1)).toWriteOnly();
    })();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 5.2  Splat density
  // ─────────────────────────────────────────────────────────────────────────

  _mkSplatDen(denR, denW, res) {
    const radU = this.radius;
    const ptU  = this.point3D;

    return Fn(() => {
      SmokeVolume._guard(res);
      const P = ivec3(globalId);
      const gridDim = vec3(float(res));
      const gp = vec3(globalId).add(0.5).div(gridDim);
      const delta = gp.sub(ptU);
      const g = exp(dot(delta, delta).negate().div(radU.div(100.0)));

      const old = texture3D(denR).load(P).xyz;
      textureStore(denW, P, vec4(old.add(vec3(g)), 1)).toWriteOnly();
    })();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 4.9/4.10  Advection — forward pass (semi-Lagrangian backtrace)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Forward advection for a self-advecting field (velocity).
   * Uses trilinear .sample() with normalised coords.
   */
  _mkAdvectForward(fieldR, tempW, fieldRes, velRes, dissU) {
    const dtU = this.deltaTime;

    return Fn(() => {
      SmokeVolume._guard(fieldRes);
      const P = ivec3(globalId);
      const gridDim = vec3(float(fieldRes));
      const ts = vec3(1.0).div(gridDim);
      const p  = vec3(globalId).add(0.5).div(gridDim);

      const fld = texture3D(fieldR);
      const vel = fld.sample(p).xyz;                       // trilinear
      const bt  = clamp(p.sub(vel.mul(dtU).mul(ts)), vec3(0), vec3(1));
      const res = fld.sample(bt).mul(dissU);

      textureStore(tempW, P, res).toWriteOnly();
    })();
  }

  /**
   * Forward advection for density (cross-resolution: velocity at simRes,
   * density at dyeRes).
   */
  _mkAdvectDenForward(velR, denR, tempW, simRes, dyeRes, dissU) {
    const dtU = this.deltaTime;

    return Fn(() => {
      SmokeVolume._guard(dyeRes);
      const P = ivec3(globalId);
      const dyeDim = vec3(float(dyeRes));
      const velTs  = vec3(1.0).div(vec3(float(simRes)));   // velocity texel size
      const p      = vec3(globalId).add(0.5).div(dyeDim);

      const vel = texture3D(velR).sample(p).xyz;           // cross-res trilinear
      const bt  = clamp(p.sub(vel.mul(dtU).mul(velTs)), vec3(0), vec3(1));
      const res = texture3D(denR).sample(bt).mul(dissU);

      textureStore(tempW, P, res).toWriteOnly();
    })();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // 4.9/4.10  MacCormack correction pass
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Generic MacCormack error-correction + neighbourhood clamping.
   * Works for both velocity and density advection.
   */
  _mkMacCormackCorr(origR, tempR, outW, fieldRes, dissU) {
    const dtU = this.deltaTime;

    return Fn(() => {
      SmokeVolume._guard(fieldRes);
      const P = ivec3(globalId);
      const gridDim = vec3(float(fieldRes));
      const ts = vec3(1.0).div(gridDim);
      const p  = vec3(globalId).add(0.5).div(gridDim);

      const orig = texture3D(origR);
      const tmp  = texture3D(tempR);

      const velFwd = tmp.sample(p).xyz;

      // Forward trace
      const fwd = clamp(p.add(velFwd.mul(dtU).mul(ts)), vec3(0), vec3(1));

      // Error correction
      const err = orig.sample(p).sub(tmp.sample(fwd)).mul(0.5);
      const corrected = tmp.sample(p).add(err).mul(dissU);

      // Neighbourhood min/max clamping
      const oC = orig.sample(p);
      const oL = orig.sample(clamp(p.sub(vec3(ts.x, 0, 0)), vec3(0), vec3(1)));
      const oR = orig.sample(clamp(p.add(vec3(ts.x, 0, 0)), vec3(0), vec3(1)));
      const oD = orig.sample(clamp(p.sub(vec3(0, ts.y, 0)), vec3(0), vec3(1)));
      const oU = orig.sample(clamp(p.add(vec3(0, ts.y, 0)), vec3(0), vec3(1)));
      const oB = orig.sample(clamp(p.sub(vec3(0, 0, ts.z)), vec3(0), vec3(1)));
      const oF = orig.sample(clamp(p.add(vec3(0, 0, ts.z)), vec3(0), vec3(1)));

      const nMin = tslMin(oC, tslMin(oL, tslMin(oR, tslMin(oD, tslMin(oU, tslMin(oB, oF))))));
      const nMax = tslMax(oC, tslMax(oL, tslMax(oR, tslMax(oD, tslMax(oU, tslMax(oB, oF))))));

      const result = clamp(corrected, nMin, nMax);
      textureStore(outW, P, result).toWriteOnly();
    })();
  }
}
