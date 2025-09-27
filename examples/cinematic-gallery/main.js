import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { unzipSync } from "three/addons/libs/fflate.module.js";
import { texture3D, uniform, pass, uv, color, screenUV } from "three/tsl";
import { knotMorphPosition } from "../../src/knotMorph.js";
import {
  buildSphericalWaveCopyKernel,
  averageIntensityProjection,
} from "../../src/index.js";
import * as THREE from "three/webgpu";

// Choose cinematic aspect based on viewport orientation: 16:9 (landscape) or 9:16 (portrait)
const CHILD_ASPECT = window.innerWidth >= window.innerHeight ? 16 / 9 : 9 / 16;

// Parent (main) scene and camera
const parentScene = new THREE.Scene();
const parentCamera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
parentCamera.position.set(0, 1.25, 2.5);

// Child A (offscreen) scene and camera - Raymarch Head
const childScene = new THREE.Scene();
const childCamera = new THREE.PerspectiveCamera(60, CHILD_ASPECT, 0.05, 1000);
childCamera.position.set(0.0, 0.0, 1.2);
childCamera.updateProjectionMatrix();

// Child B (offscreen) scene and camera - Knot Morph
const childScene2 = new THREE.Scene();
const childCamera2 = new THREE.PerspectiveCamera(60, CHILD_ASPECT, 0.05, 1000);
childCamera2.position.set(0, 0, 5);
childCamera2.updateProjectionMatrix();

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

    // Screen A in parent scene showing the head via pass(...)
    // Match plane aspect to chosen cinematic aspect (landscape 16:9 or portrait 9:16)
    const longSide = 1.6;
    const screenWidth = CHILD_ASPECT >= 1 ? longSide : longSide * CHILD_ASPECT;
    const screenHeight = CHILD_ASPECT >= 1 ? longSide / CHILD_ASPECT : longSide;
    const screenGeo = new THREE.PlaneGeometry(screenWidth, screenHeight);
    const screenMat = new THREE.MeshBasicNodeMaterial();
    // Portal/window mapping: sample using screenUV so it feels like a window
    screenMat.colorNode = pass(childScene, childCamera).context({
      getUV: () => screenUV,
    });
    // Solid material (no vignette mask)
    screenMat.transparent = false;
    const screen = new THREE.Mesh(screenGeo, screenMat);
    screen.material.side = THREE.DoubleSide;
    screen.renderOrder = 1;
    screen.position.set(0, 1.0, 0);
    parentScene.add(screen);

    // Simple 3D frame around portal A
    const frameDepth = 0.05;
    const frameThickness = 0.06;
    const frameMat = new THREE.MeshStandardMaterial({
      color: 0x222222,
      roughness: 1,
    });
    const frameGroup = new THREE.Group();
    const horizLen = screenWidth + frameThickness * 2;
    const vertLen = screenHeight + frameThickness * 2;
    const top = new THREE.Mesh(
      new THREE.BoxGeometry(horizLen, frameThickness, frameDepth),
      frameMat
    );
    const bot = new THREE.Mesh(
      new THREE.BoxGeometry(horizLen, frameThickness, frameDepth),
      frameMat
    );
    const left = new THREE.Mesh(
      new THREE.BoxGeometry(frameThickness, screenHeight, frameDepth),
      frameMat
    );
    const right = new THREE.Mesh(
      new THREE.BoxGeometry(frameThickness, screenHeight, frameDepth),
      frameMat
    );
    top.position.set(0, screenHeight / 2 + frameThickness / 2, frameDepth / 2);
    bot.position.set(0, -screenHeight / 2 - frameThickness / 2, frameDepth / 2);
    left.position.set(-screenWidth / 2 - frameThickness / 2, 0, frameDepth / 2);
    right.position.set(screenWidth / 2 + frameThickness / 2, 0, frameDepth / 2);
    frameGroup.add(top, bot, left, right);
    frameGroup.position.copy(screen.position);
    parentScene.add(frameGroup);

    // Screen B (knot morph) on the right
    const screen2Geo = new THREE.PlaneGeometry(screenWidth, screenHeight);
    const screen2Mat = new THREE.MeshBasicNodeMaterial();
    // Use mesh UV mapping so the content is view-independent
    screen2Mat.colorNode = pass(childScene2, childCamera2).getTextureNode();
    screen2Mat.transparent = false;
    const screen2 = new THREE.Mesh(screen2Geo, screen2Mat);
    screen2.material.side = THREE.DoubleSide;
    screen2.renderOrder = 1;
    screen2.position.set(2.0, 1.0, 0);
    parentScene.add(screen2);

    // Frame around portal B
    const frameGroup2 = new THREE.Group();
    const top2 = new THREE.Mesh(
      new THREE.BoxGeometry(horizLen, frameThickness, frameDepth),
      frameMat
    );
    const bot2 = new THREE.Mesh(
      new THREE.BoxGeometry(horizLen, frameThickness, frameDepth),
      frameMat
    );
    const left2 = new THREE.Mesh(
      new THREE.BoxGeometry(frameThickness, screenHeight, frameDepth),
      frameMat
    );
    const right2 = new THREE.Mesh(
      new THREE.BoxGeometry(frameThickness, screenHeight, frameDepth),
      frameMat
    );
    top2.position.set(0, screenHeight / 2 + frameThickness / 2, frameDepth / 2);
    bot2.position.set(
      0,
      -screenHeight / 2 - frameThickness / 2,
      frameDepth / 2
    );
    left2.position.set(
      -screenWidth / 2 - frameThickness / 2,
      0,
      frameDepth / 2
    );
    right2.position.set(
      screenWidth / 2 + frameThickness / 2,
      0,
      frameDepth / 2
    );
    frameGroup2.add(top2, bot2, left2, right2);
    frameGroup2.position.copy(screen2.position);
    parentScene.add(frameGroup2);

    // Camera state management (default, fit to screen A, fit to screen B)
    const defaultCamPos = parentCamera.position.clone();
    const defaultTarget = controls.target.clone();

    function fitToPlane(mesh, options = {}) {
      const { cover = false, overscan = 1.02 } = options; // cover=true = fill like CSS background-size: cover
      const params = mesh.geometry.parameters || { width: 1, height: 1 };
      const rectWidth = (params.width || 1) * (mesh.scale?.x || 1);
      const rectHeight = (params.height || 1) * (mesh.scale?.y || 1);

      const center = new THREE.Vector3();
      mesh.getWorldPosition(center);

      const normal = new THREE.Vector3(0, 0, 1);
      const q = new THREE.Quaternion();
      mesh.getWorldQuaternion(q);
      normal.applyQuaternion(q);

      const vFov = (parentCamera.fov * Math.PI) / 180;
      const fovH = 2 * Math.atan(Math.tan(vFov / 2) * parentCamera.aspect);
      const distH = rectHeight / 2 / Math.tan(vFov / 2);
      const distW = rectWidth / 2 / Math.tan(fovH / 2);
      // contain (fit): max; cover (fill): min
      let distance = cover ? Math.min(distH, distW) : Math.max(distH, distW);
      // For cover, move slightly closer to ensure no border shows; for contain, a tiny margin could be applied if desired
      distance = cover ? distance / overscan : distance;

      // Place the camera on the same side of the plane as it currently is to avoid flips
      const toCamera = new THREE.Vector3().subVectors(
        parentCamera.position,
        center
      );
      const side = Math.sign(toCamera.dot(normal)) || 1; // +1 if in front (along normal), -1 if behind
      const position = new THREE.Vector3()
        .copy(center)
        .addScaledVector(normal, side * distance);
      const target = center.clone();
      return { position, target };
    }

    let camAnim = null;
    // Easing function for camera transitions (snappy, natural feel)
    function easeOutCubic(x) {
      return 1 - Math.pow(1 - x, 3);
    }

    function animateCameraTo({ position, target }, duration = 0.5) {
      camAnim = {
        t: 0,
        duration,
        ease: easeOutCubic,
        fromPos: parentCamera.position.clone(),
        toPos: position.clone(),
        fromTarget: controls.target.clone(),
        toTarget: target.clone(),
      };
    }

    function setCameraState(index) {
      switch (index) {
        case 1: {
          const { position, target } = fitToPlane(screen, {
            cover: true,
            overscan: 1.06,
          });
          animateCameraTo({ position, target });
          break;
        }
        case 2: {
          const { position, target } = fitToPlane(screen2, {
            cover: true,
            overscan: 1.06,
          });
          animateCameraTo({ position, target });
          break;
        }
        default: {
          animateCameraTo({ position: defaultCamPos, target: defaultTarget });
        }
      }
    }

    let cameraStateIndex = 0; // 0 default, 1 screen A, 2 screen B
    function cycleCameraState() {
      cameraStateIndex = (cameraStateIndex + 1) % 3;
      setCameraState(cameraStateIndex);
    }
    function cycleCameraStateBackward() {
      cameraStateIndex = (cameraStateIndex + 3 - 1) % 3;
      setCameraState(cameraStateIndex);
    }

    // Input binding: spacebar to cycle; shift+space to cycle backward
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        if (e.shiftKey) cycleCameraStateBackward();
        else cycleCameraState();
      }
    });

    // Child B: Knot morph content
    const startGeo = new THREE.TorusKnotGeometry(1, 0.4, 128, 32, 2, 3);
    const targetGeo = new THREE.TorusKnotGeometry(1, 0.4, 128, 32, 3, 5);
    const targetPositions = targetGeo.getAttribute("position").array;
    startGeo.setAttribute(
      "targetPosition",
      new THREE.BufferAttribute(targetPositions, 3)
    );
    const knotMat = new THREE.MeshBasicNodeMaterial({
      wireframe: true,
      transparent: true,
    });
    knotMat.positionNode = knotMorphPosition();
    knotMat.colorNode = color(0x00ff00);
    const knotMesh = new THREE.Mesh(startGeo, knotMat);
    childScene2.add(knotMesh);

    // Fit knot inside childCamera2 frustum and center it
    startGeo.computeBoundingSphere();
    const radius = startGeo.boundingSphere?.radius || 1.5;
    const vFov = (childCamera2.fov * Math.PI) / 180;
    const fitHeightDistance = radius / Math.tan(vFov / 2);
    const fitWidthDistance = (radius * CHILD_ASPECT) / Math.tan(vFov / 2);
    const distance = Math.max(fitHeightDistance, fitWidthDistance) * 1.2; // margin
    childCamera2.position.set(0, 0, distance);
    childCamera2.lookAt(0, 0, 0);

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

      // Child B animation (knot morph)
      const mixFactor = Math.abs(Math.sin(now * 0.5));
      knotMat.positionNode = knotMorphPosition({ mixFactor });
      knotMesh.rotation.y += 0.01;

      // Camera animation LERP (position and target)
      if (camAnim) {
        camAnim.t += dt / camAnim.duration;
        const a = Math.min(1, camAnim.t);
        const e = camAnim.ease ? camAnim.ease(a) : a;
        parentCamera.position.lerpVectors(camAnim.fromPos, camAnim.toPos, e);
        controls.target.lerpVectors(camAnim.fromTarget, camAnim.toTarget, e);
        if (a >= 1) camAnim = null;
      }

      // Render parent scene with user controls
      controls.update();
      renderer.render(parentScene, parentCamera);
    });

    // Resize handling for both cameras
    window.addEventListener("resize", () => {
      renderer.setSize(window.innerWidth, window.innerHeight);
      parentCamera.aspect = window.innerWidth / window.innerHeight;
      parentCamera.updateProjectionMatrix();
      // Keep child cameras at the cinematic aspect (landscape 16:9 or portrait 9:16)
      childCamera.aspect = CHILD_ASPECT;
      childCamera.updateProjectionMatrix();
      childCamera2.aspect = CHILD_ASPECT;
      childCamera2.updateProjectionMatrix();
    });
  });
