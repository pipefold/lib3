// vite.config.js (root - for library build + examples serve/build)
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";
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
        "@pipefold/tsl-lib": resolve(__dirname, "src/index.js"),
      },
    },
  };

  const commonPlugins = [
    threeUniformGui({
      persistent: false, // Saves tweaks to localStorage
      devOnly: true, // Only in dev mode
    }),
  ];

  if (mode === "lib") {
    // Library build mode (from src/)
    return {
      ...common,
      plugins: [
        dts({ rollupTypes: true }), // Types only needed for lib
        ...commonPlugins,
      ],
      build: {
        lib: {
          entry: "./src/index.js",
          name: "PipefoldTSLLib",
          fileName: (format) =>
            `pipefold-tsl-lib.${format === "es" ? "js" : "cjs"}`,
          formats: ["es", "cjs"],
        },
        rollupOptions: {
          external: ["three", "three/tsl"], // Don't bundle Three.js
          output: {
            globals: {
              three: "THREE",
            },
          },
        },
      },
    };
  } else {
    // Default mode: Serve/build examples as multi-page app
    return {
      ...common,
      plugins: commonPlugins,
      build: {
        // Multi-page build config (dynamic inputs via glob)
        rollupOptions: {
          input: Object.fromEntries(
            glob.sync("examples/**/index.html").map((file) => [
              // Key: relative path without .html (e.g., 'examples/example1/index')
              // This preserves sub-directory structure in dist/ (e.g., dist/examples/example1/index.html)
              file.slice(0, -5),
              fileURLToPath(new URL(file, import.meta.url)),
            ])
          ),
        },
      },
    };
  }
});
