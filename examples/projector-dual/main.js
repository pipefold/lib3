import * as THREE from "three/webgpu";
import {
  Fn,
  vec3,
  sin,
  abs,
  uniform,
  time,
  texture3D,
  pass,
  screenUV,
  screenCoordinate,
} from "three/tsl";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { Inspector } from "three/addons/inspector/Inspector.js";
import { bayer16 } from "three/addons/tsl/math/Bayer.js";
import { gaussianBlur } from "three/addons/tsl/display/GaussianBlurNode.js";

// Phase 1: No passes/compute/morphs

// --- Renderer / Scene / Camera ---
const canvas = document.getElementById("canvas");
const renderer = new THREE.WebGPURenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.inspector = new Inspector();

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0);

const camera = new THREE.PerspectiveCamera(
  50,
  window.innerWidth / window.innerHeight,
  0.1,
  100
);
camera.position.set(6, 3, 10);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 2, 0);
controls.update();

// --- Ground + Wall ---
const floorMat = new THREE.MeshStandardNodeMaterial({ color: 0x111111 });
const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), floorMat);
floor.rotation.x = -Math.PI / 2;
floor.position.y = 0;
floor.receiveShadow = true;
scene.add(floor);

// Wall at z=0 for projector visualization
const wallMat = new THREE.MeshStandardNodeMaterial({
  color: 0x808080,
  roughness: 1.0,
  metalness: 0.0,
});
const wall = new THREE.Mesh(new THREE.BoxGeometry(30, 15, 0.5), wallMat);
wall.position.set(0, 7.5, 0); // Position at z=0
wall.receiveShadow = true;
scene.add(wall);

// Helpers for spatial orientation
const grid = new THREE.GridHelper(40, 40, 0x666666, 0x333333);
grid.position.y = 0.001; // avoid z-fighting with floor
scene.add(grid);

const axes = new THREE.AxesHelper(2);
axes.position.y = 0.01;
scene.add(axes);

// --- Ambient + Fill light ---
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const key = new THREE.DirectionalLight(0xffffff, 0.5);
key.position.set(3, 6, 2);
key.castShadow = true;
scene.add(key);

// --- Projector Lights ---
const LAYER_VOLUMETRIC = 10;
function createProjector(name, colorHex, pos) {
  const root = new THREE.Group();
  root.name = name;

  const projector = new THREE.ProjectorLight(colorHex, 60);
  // Position the root so TransformControls attach at the correct place
  root.position.set(...pos);
  projector.angle = Math.PI / 8;
  projector.penumbra = 1;
  projector.decay = 2;
  projector.distance = 0; // infinite
  projector.castShadow = true;
  projector.shadow.mapSize.set(1024, 1024);
  projector.shadow.camera.near = 0.5;
  projector.shadow.camera.far = 30;
  projector.shadow.bias = -0.0015;
  projector.layers.enable(LAYER_VOLUMETRIC);

  // Helper mesh (cone) for picking + visualizing direction
  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(0.2, 0.5, 20),
    new THREE.MeshStandardMaterial({
      color: colorHex,
      emissive: colorHex,
      emissiveIntensity: 0.4,
    })
  );
  cone.rotation.x = Math.PI / 2; // -Z forward
  cone.position.z = 0.15;
  cone.castShadow = false;
  cone.receiveShadow = false;
  cone.name = `${name}-gizmo`;

  root.add(projector);
  projector.add(cone);

  scene.add(root);
  return { root, projector, cone };
}

const projA = createProjector("Projector A", 0xffcc88, [-6, 5, 6]);
const projB = createProjector("Projector B", 0x88ccff, [6, 5, 6]);

// Function to calculate closest point on wall surface
function getClosestWallPoint(position) {
  // Wall dimensions and position
  const wallCenter = new THREE.Vector3(0, 7.5, 0);
  const wallWidth = 30;
  const wallHeight = 15;

  // Calculate closest point on wall rectangle (front face at z=0)
  const closest = new THREE.Vector3(
    Math.max(wallCenter.x - wallWidth/2, Math.min(position.x, wallCenter.x + wallWidth/2)),
    Math.max(wallCenter.y - wallHeight/2, Math.min(position.y, wallCenter.y + wallHeight/2)),
    wallCenter.z // Always on the wall surface (z=0)
  );

  return closest;
}

// Setup targets for projectors
function updateProjectorTargets() {
  const projAPos = projA.projector.getWorldPosition(new THREE.Vector3());
  const projBPos = projB.projector.getWorldPosition(new THREE.Vector3());

  projA.projector.target.position.copy(getClosestWallPoint(projAPos));
  projB.projector.target.position.copy(getClosestWallPoint(projBPos));
}

// Initialize targets and add them to scene
updateProjectorTargets();
scene.add(projA.projector.target);
scene.add(projB.projector.target);

// --- Simple dynamic content sources (Phase 1) ---
function makeCanvasTexture(draw) {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 512;
  const ctx = c.getContext("2d");
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return { ctx, canvas: c, texture: tex, draw };
}

const canvasA = makeCanvasTexture((ctx, t) => {
  const w = ctx.canvas.width,
    h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);
  const g = ctx.createLinearGradient(0, 0, w, h);
  g.addColorStop(0, `hsl(${(t * 40) % 360}, 80%, 55%)`);
  g.addColorStop(1, `hsl(${(t * 40 + 180) % 360}, 80%, 45%)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  // moving stripes
  ctx.globalCompositeOperation = "overlay";
  ctx.fillStyle = "rgba(255,255,255,0.25)";
  const k = (t * 2) % 1;
  for (let i = -1; i < 16; i++) {
    const x = ((i + k) / 16) * w;
    ctx.fillRect(x, 0, 8, h);
  }
  ctx.globalCompositeOperation = "source-over";
});

const canvasB = makeCanvasTexture((ctx, t) => {
  const w = ctx.canvas.width,
    h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);
  // radial pulsating ring
  const cx = w / 2,
    cy = h / 2;
  const radius = Math.abs(Math.sin(t * 1.2)) * 0.4 + 0.2;
  const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.min(w, h) * 0.5);
  grd.addColorStop(0, "rgba(30,200,255,1)");
  grd.addColorStop(radius, "rgba(30,200,255,0.3)");
  grd.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, w, h);
});

function setProjectorContent(projector, type) {
  projector.colorNode = null;
  if (type === "canvasA") {
    projector.map = canvasA.texture;
  } else if (type === "canvasB") {
    projector.map = canvasB.texture;
  } else if (type === "procedural") {
    // simple animated stripes via TSL using projectorUV length
    projector.colorNode = Fn(([projectorUV]) => {
      const u = projectorUV.x.mul(20.0);
      const band = abs(sin(u));
      const c = vec3(band, band.mul(0.5), band.mul(0.1));
      return c;
    });
    projector.map = null;
  } else {
    projector.map = null;
  }
}

setProjectorContent(projA.projector, "canvasA");
setProjectorContent(projB.projector, "canvasB");

// --- TransformControls + selection ---
const tControls = new TransformControls(camera, renderer.domElement);
scene.add(tControls.getHelper());
tControls.setSpace("local");
tControls.setSize(0.9);

let selected = null;

tControls.addEventListener("dragging-changed", (e) => {
  controls.enabled = !e.value;
});

window.addEventListener("keydown", (e) => {
  if (e.code === "KeyW") tControls.setMode("translate");
  if (e.code === "KeyE") tControls.setMode("rotate");
  if (e.code === "KeyR") tControls.setMode("scale");
});

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function pick(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const candidates = [projA.projector, projB.projector].map(
    (p) => p.children[0]
  );
  const intersects = raycaster.intersectObjects(candidates, true);
  if (intersects.length) {
    const cone = intersects[0].object;
    const pr = cone.parent; // projector
    const root = pr.parent; // group
    selected = root;
    tControls.attach(root);
  }
}

renderer.domElement.addEventListener("pointerdown", pick);

// --- GUI via Inspector ---
const gui = renderer.inspector.createParameters("Projectors");

function addProjectorGUI(label, proj, initialType) {
  const folder = gui.addFolder(label);
  const params = {
    content: initialType,
    color: proj.projector.color.getHex(),
    intensity: proj.projector.intensity,
    distance: proj.projector.distance,
    angle: proj.projector.angle,
    penumbra: proj.projector.penumbra,
    decay: proj.projector.decay,
    focus: proj.projector.shadow.focus,
    shadows: true,
  };

  folder
    .add(params, "content", ["none", "canvasA", "canvasB", "procedural"])
    .onChange((v) => setProjectorContent(proj.projector, v));
  folder
    .addColor(params, "color")
    .onChange((v) => proj.projector.color.setHex(v));
  folder
    .add(params, "intensity", 0, 500)
    .onChange((v) => (proj.projector.intensity = v));
  folder
    .add(params, "distance", 0, 50)
    .onChange((v) => (proj.projector.distance = v));
  folder
    .add(params, "angle", 0, Math.PI / 3)
    .onChange((v) => (proj.projector.angle = v));
  folder
    .add(params, "penumbra", 0, 1)
    .onChange((v) => (proj.projector.penumbra = v));
  folder.add(params, "decay", 1, 2).onChange((v) => (proj.projector.decay = v));
  folder
    .add(params, "focus", 0, 1)
    .onChange((v) => (proj.projector.shadow.focus = v));
  folder.add(params, "shadows").onChange((v) => {
    renderer.shadowMap.enabled = v;
    scene.traverse((child) => {
      if (child.material) child.material.needsUpdate = true;
    });
  });
}

addProjectorGUI("Projector A", projA, "head");
addProjectorGUI("Projector B", projB, "knot");

// --- Volumetric Medium (16x16x16) ---
let postProcessing;

function createTexture3D_16() {
  const size = 16;
  const data = new Uint8Array(size * size * size);
  let i = 0;
  // Simple soft spherical falloff density centered in the volume
  for (let z = 0; z < size; z++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const nx = ((x + 0.5) / size) * 2 - 1;
        const ny = ((y + 0.5) / size) * 2 - 1;
        const nz = ((z + 0.5) / size) * 2 - 1;
        const r = Math.sqrt(nx * nx + ny * ny + nz * nz);
        const d = Math.max(0, 1 - r);
        data[i++] = Math.min(255, Math.floor(d * 255));
      }
    }
  }
  const tex = new THREE.Data3DTexture(data, size, size, size);
  tex.format = THREE.RedFormat;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

const volumetricLayer = new THREE.Layers();
volumetricLayer.disableAll();
volumetricLayer.enable(LAYER_VOLUMETRIC);

const densityTex3D = createTexture3D_16();

const volumetricMaterial = new THREE.VolumeNodeMaterial();
volumetricMaterial.steps = 12; // tweakable via GUI
volumetricMaterial.offsetNode = bayer16(screenCoordinate);

// Controls for density and tiling
const densityScale = uniform(1.0);
const tileScale = uniform(1.0);
volumetricMaterial.scatteringNode = Fn(({ positionRay }) => {
  // Map world position into [0,1] using a simple scale and modulo to stay in range
  const uvw = positionRay
    .mul(tileScale)
    .mul(1 / 16)
    .mod(1);
  const d = texture3D(densityTex3D, uvw, 0).r;
  return d.mul(densityScale);
});

const volumetricBox = new THREE.Mesh(
  new THREE.BoxGeometry(16, 16, 16),
  volumetricMaterial
);
volumetricBox.position.set(0, 8, -2);
volumetricBox.receiveShadow = true;
volumetricBox.layers.disableAll();
volumetricBox.layers.enable(LAYER_VOLUMETRIC);
scene.add(volumetricBox);

// Post-processing setup for volumetric lighting compositing
postProcessing = new THREE.PostProcessing(renderer);

const volumetricIntensity = uniform(1.0);
const scenePass = pass(scene, camera);
const sceneDepth = scenePass.getTextureNode("depth");
volumetricMaterial.depthNode = sceneDepth.sample(screenUV);

const volumetricPass = pass(scene, camera, { depthBuffer: false });
volumetricPass.name = "Volumetric Lighting";
volumetricPass.setLayers(volumetricLayer);
volumetricPass.setResolutionScale(0.5);

const denoiseStrength = uniform(0.6);
let blurredVolumetricPass = gaussianBlur(volumetricPass, denoiseStrength);

let scenePassColor = scenePass.add(
  blurredVolumetricPass.mul(volumetricIntensity)
);
postProcessing.outputNode = scenePassColor;

// GUI for volumetrics
const volFolder = renderer.inspector.createParameters("Volumetrics");
const qualityParams = {
  resolution: volumetricPass.getResolutionScale(),
  denoise: true,
};
volFolder
  .add(qualityParams, "resolution", 0.1, 1)
  .onChange((v) => volumetricPass.setResolutionScale(v));
volFolder.add(volumetricMaterial, "steps", 2, 32).name("step count");
volFolder.add(denoiseStrength, "value", 0, 1).name("denoise strength");
volFolder.add(qualityParams, "denoise").onChange((denoise) => {
  blurredVolumetricPass = gaussianBlur(volumetricPass, denoiseStrength);
  const volNode = denoise ? blurredVolumetricPass : volumetricPass;
  scenePassColor = scenePass.add(volNode.mul(volumetricIntensity));
  postProcessing.outputNode = scenePassColor;
  postProcessing.needsUpdate = true;
});
volFolder.add(volumetricIntensity, "value", 0, 5).name("intensity");
volFolder.add(densityScale, "value", 0, 5).name("density");
volFolder.add(tileScale, "value", 0.1, 4).name("tile scale");

// --- Resize ---
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- Animation Loop ---
let last = performance.now() * 0.001;

renderer.setAnimationLoop(() => {
  const now = performance.now() * 0.001;
  const dt = Math.min(0.1, now - last);
  last = now;

  // Update projector targets to follow closest wall points
  updateProjectorTargets();

  // Update canvas textures
  canvasA.draw(canvasA.ctx, now);
  canvasA.texture.needsUpdate = true;
  canvasB.draw(canvasB.ctx, now);
  canvasB.texture.needsUpdate = true;

  if (postProcessing) {
    postProcessing.render();
  } else {
    renderer.render(scene, camera);
  }
});
