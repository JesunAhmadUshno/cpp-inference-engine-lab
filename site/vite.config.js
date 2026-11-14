import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// The site's data file (src/results.json) is generated at build time from the
// repo's committed raw harness outputs in ../results/, so every number on the
// page is generated from the record, never typed.
export default defineConfig({
  base: '/cpp-inference-engine-lab/',
  plugins: [react()],
  server: { fs: { allow: ['..'] } },
});
