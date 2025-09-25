import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { unzipSync } from "three/addons/libs/fflate.module.js";
import { RaymarchingBox } from "three/addons/tsl/utils/Raymarching.js";
import { Fn, texture3D, uniform, vec3, vec4, float } from "three/tsl";
import { buildSphericalWaveCopyKernel } from "../../src/index.js";
import * as THREE from "three/webgpu";

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.z = 3;
const renderer = new THREE.WebGPURenderer({
  canvas: document.getElementById("canvas"),
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000);
new OrbitControls(camera, renderer.domElement);

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
    // @range: { min: 0.0, max: 0.3, step: 0.005 }
    const waveAmplitude = uniform(0.08);
    // @range: { min: 0.0, max: 5.0, step: 0.05 }
    const waveSpeed = uniform(1.2);
    // @range: { min: 0.1, max: 8.0, step: 0.1 }
    const noiseScale = uniform(2.0);
    // @range: { min: 0.0, max: 2.0, step: 0.05 }
    const noiseAmplitude = uniform(0.6);
    // @range: { min: 0.25, max: 2.0, step: 0.05 }
    const intensityScale = uniform(1.0);
    const timeUniform = uniform(0.0);

    // Build compute kernel once, feed uniforms each frame
    const waveKernel = buildSphericalWaveCopyKernel({
      width,
      height,
      depth,
      storageTexture,
      sourceTextureNode: texture3D(sourceTexture, null, 0),
      waveAmplitude,
      waveSpeed,
      noiseScale,
      noiseAmplitude,
      intensityScale,
      time: timeUniform,
    });

    const computeNode = waveKernel()
      .compute(width * height * depth)
      .setName("copyHead3DDisplaced");

    await renderer.init();
    await renderer.computeAsync(computeNode);

    // Shader for Average Intensity Projection (AIP)
    const averageIntensityProjection = Fn(
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

    // @range: { min: 1, max: 15, step: 0.01 }
    const steps = uniform(1);
    // @range: { min: 0.1, max: 5.0, step: 0.1 }
    const intensityScaleView = uniform(2.0);

    const material = new THREE.NodeMaterial();
    material.colorNode = averageIntensityProjection({
      texture: texture3D(storageTexture, null, 0),
      steps,
      intensityScale: intensityScaleView,
    });
    material.side = THREE.BackSide;
    material.transparent = true;

    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    mesh.scale.set(1, -1, depth / width);
    scene.add(mesh);

    renderer.setAnimationLoop(async () => {
      timeUniform.value = performance.now() * 0.001;
      await renderer.computeAsync(computeNode);
      renderer.render(scene, camera);
    });
  });
