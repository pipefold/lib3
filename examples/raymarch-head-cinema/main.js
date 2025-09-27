import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { unzipSync } from "three/addons/libs/fflate.module.js";
import { texture3D, uniform, pass, screenUV } from "three/tsl";
import {
  buildSphericalWaveCopyKernel,
  averageIntensityProjection,
} from "../../src/index.js";
import * as THREE from "three/webgpu";

// Parent (main) scene and camera
const parentScene = new THREE.Scene();
const parentCamera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
parentCamera.position.set(0, 1.25, 2.5);

// Child (offscreen) scene and camera
const childScene = new THREE.Scene();
const childCamera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.05,
  1000
);
childCamera.position.set(0.0, 0.0, 1.2);

// Renderer
const renderer = new THREE.WebGPURenderer({
  canvas: document.getElementById("canvas"),
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000);

// Parent controls (user controls)
const controls = new OrbitControls(parentCamera, renderer.domElement);
controls.target.set(0, 1.0, 0);
controls.update();

// Load volumetric dataset and set up compute + child scene volume render
new THREE.FileLoader()
  .setResponseType("arraybuffer")
  .load("../assets/head256x256x109.zip", async function (data) {
    const zip = unzipSync(new Uint8Array(data));
    const array = new Uint8Array(zip["head256x256x109"].buffer);

    const width = 256;
    const height = 256;
    const depth = 109;

    // Source texture
    const sourceTexture = new THREE.Data3DTexture(array, width, height, depth);
    sourceTexture.format = THREE.RedFormat;
    sourceTexture.minFilter = THREE.LinearFilter;
    sourceTexture.magFilter = THREE.LinearFilter;
    sourceTexture.unpackAlignment = 1;
    sourceTexture.needsUpdate = true;

    // Destination storage texture
    const storageTexture = new THREE.Storage3DTexture(width, height, depth);
    storageTexture.generateMipmaps = false;
    storageTexture.name = "headWave";

    // Uniforms for wave compute
    const waveAmplitude = uniform(0.5);
    const waveSpeed = uniform(2);
    const noiseScale = uniform(0.64);
    const noiseAmplitude = uniform(0.6);
    const intensityScale = uniform(0.25);
    const timeUniform = uniform(0.0);
    const phaseUniform = uniform(0.0);

    // Build compute kernel once, feed uniforms each frame
    const waveKernel = buildSphericalWaveCopyKernel({
      width,
      height,
      depth,
      storageTexture,
      sourceTextureNode: texture3D(sourceTexture, null, 0),
      waveAmplitude,
      noiseScale,
      noiseAmplitude,
      intensityScale,
      phase: phaseUniform,
    });

    const computeNode = waveKernel()
      .compute(width * height * depth)
      .setName("copyHead3DDisplaced");

    await renderer.init();
    await renderer.computeAsync(computeNode);

    // Raymarch material in child scene
    const steps = uniform(4);
    const intensityScaleView = uniform(0.2);

    const childMaterial = new THREE.NodeMaterial();
    childMaterial.colorNode = averageIntensityProjection({
      texture: texture3D(storageTexture, null, 0),
      steps,
      intensityScale: intensityScaleView,
    });
    childMaterial.side = THREE.BackSide;
    childMaterial.transparent = true;

    const childMesh = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      childMaterial
    );
    childMesh.scale.set(1, -1, depth / width);
    childScene.add(childMesh);

    // Screen in parent scene (16:9 plane) showing the child scene via pass(...)
    const screenWidth = 1.6;
    const screenHeight = 0.9;
    const screenGeo = new THREE.PlaneGeometry(screenWidth, screenHeight);
    const screenMat = new THREE.MeshBasicNodeMaterial();
    screenMat.colorNode = pass(childScene, childCamera).context({
      getUV: () => screenUV,
    });
    const screen = new THREE.Mesh(screenGeo, screenMat);
    screen.position.set(0, 1.0, 0);
    parentScene.add(screen);

    // Simple cinema-like stand for context
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(6, 6),
      new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 1 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0;
    parentScene.add(floor);

    const light = new THREE.DirectionalLight(0xffffff, 1.0);
    light.position.set(1, 2, 1);
    parentScene.add(light);
    parentScene.add(new THREE.AmbientLight(0xffffff, 0.2));

    // Animate child camera along a spiral around the head
    let lastTime = performance.now() * 0.001;
    let accum = 0;
    const computeInterval = 1 / 30; // 30 Hz compute

    renderer.setAnimationLoop(async () => {
      const now = performance.now() * 0.001;
      const dt = Math.min(0.1, now - lastTime);
      lastTime = now;
      timeUniform.value = now;

      // Integrate phase with real dt
      phaseUniform.value += waveSpeed.value * dt;

      // Throttle compute to ~30Hz
      accum += dt;
      if (accum >= computeInterval) {
        accum -= computeInterval;
        await renderer.computeAsync(computeNode);
      }

      // Spiral motion for child camera
      const radius = 1.25;
      const angularSpeed = 0.4; // rad/s
      const elevationSpeed = 0.15; // cycles/s
      const angle = now * angularSpeed * Math.PI * 2.0 * 0.15915494309189535; // keep scale modest
      const y = Math.sin(now * elevationSpeed) * 0.2;
      childCamera.position.set(
        Math.cos(angle) * radius,
        y,
        Math.sin(angle) * radius
      );
      childCamera.lookAt(0, 0, 0);

      // Render parent scene with user controls
      controls.update();
      renderer.render(parentScene, parentCamera);
    });

    // Resize handling for both cameras
    window.addEventListener("resize", () => {
      renderer.setSize(window.innerWidth, window.innerHeight);
      parentCamera.aspect = window.innerWidth / window.innerHeight;
      parentCamera.updateProjectionMatrix();
      childCamera.aspect = window.innerWidth / window.innerHeight;
      childCamera.updateProjectionMatrix();
    });
  });
