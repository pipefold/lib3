import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { unzipSync } from "three/addons/libs/fflate.module.js";
import { RaymarchingBox } from "three/addons/tsl/utils/Raymarching.js";
import { Fn, texture3D, uniform, vec3, vec4, float } from "three/tsl";
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
  .load("../assets/head256x256x109.zip", function (data) {
    const zip = unzipSync(new Uint8Array(data));
    const array = new Uint8Array(zip["head256x256x109"].buffer);

    const texture = new THREE.Data3DTexture(array, 256, 256, 109);
    texture.format = THREE.RedFormat;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.unpackAlignment = 1;
    texture.needsUpdate = true;

    // Shader for Average Intensity Projection (AIP)

    const averageIntensityProjection = Fn(
      ({ texture, steps, intensityScale = float(1.0) }) => {
        const finalColor = vec4(0).toVar();
        const intensitySum = float(0).toVar();
        const sampleCount = float(0).toVar();

        RaymarchingBox(steps, ({ positionRay }) => {
          // Sample the texture at the current position using trilinear interpolation
          const samplePos = positionRay.add(0.5);
          const mapValue = texture.sample(samplePos).r;

          // Accumulate intensity values
          intensitySum.addAssign(mapValue);
          sampleCount.addAssign(1);
        });

        // Compute the average intensity
        const averageIntensity = intensitySum.div(sampleCount);

        // Apply intensity scaling and map to visible range
        const scaledIntensity = averageIntensity.mul(intensityScale);

        // Set as final color
        finalColor.rgb.assign(vec3(scaledIntensity));
        finalColor.a.assign(1);

        return finalColor;
      }
    );

    // @range: { min: 1, max: 150, step: 1 }
    const steps = uniform(100);
    // @range: { min: 0.1, max: 5.0, step: 0.1 }
    const intensityScale = uniform(2.0);

    const material = new THREE.NodeMaterial();
    material.colorNode = averageIntensityProjection({
      texture: texture3D(texture, null, 0),
      steps,
      intensityScale,
    });
    material.side = THREE.BackSide;
    material.transparent = true;

    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    mesh.scale.set(1, -1, 109 / 256);
    scene.add(mesh);

    function animate() {
      renderer.render(scene, camera);
    }
    renderer.setAnimationLoop(animate);
  });
