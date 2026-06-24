import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `base: "./"` so the built assets load when FastAPI serves dist/ from "/".
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "dist", emptyOutDir: true },
});
