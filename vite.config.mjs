// vite.config.js (root - for library build + examples serve/build)
import { defineConfig } from "vite";
import threeUniformGui from "tsl-uniform-ui-vite-plugin";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import glob from "fast-glob"; // For dynamic example inputs

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) => {
  const common = {
    // Shared config (e.g., resolve alias for examples to import from src/)
    resolve: {
      alias: {
        "@pipefold/lib3": resolve(__dirname, "src/index.js"),
      },
    },
  };

  const commonPlugins = [
    threeUniformGui({
      persistent: false, // Saves tweaks to localStorage
      devOnly: true, // Only in dev mode
    }),
  ];

  if (mode === "examples") {
    // Serve/build examples as multi-page app
    return {
      ...common,
      plugins: commonPlugins,
      build: {
        // Keep examples output separate from the library build
        outDir: "dist-examples",
        // Multi-page build config (dynamic inputs via glob)
        rollupOptions: {
          input: Object.fromEntries(
            glob.sync("examples/**/index.html").map((file) => [
              // Key: relative path without .html (e.g., 'examples/example1/index')
              // This preserves sub-directory structure in dist-examples/
              file.slice(0, -5),
              fileURLToPath(new URL(file, import.meta.url)),
            ])
          ),
        },
      },
    };
  } else {
    // Default: Library build mode (from src/)
    return {
      ...common,
      plugins: [
        // No dts generation for now (JS library). Add back later if needed.
      ],
      build: {
        lib: {
          entry: {
            index: resolve(__dirname, "src/index.js"),
            waves: resolve(__dirname, "src/waves.js"),
            knotMorph: resolve(__dirname, "src/knotMorph.js"),
          },
          formats: ["es"],
        },
        rollupOptions: {
          external: ["three", "three/tsl"],
        },
        sourcemap: true,
        emptyOutDir: true,
      },
    };
  }
});
