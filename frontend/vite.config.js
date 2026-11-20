import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// In development the API runs on port 8000 (make stack, or uvicorn) and these paths are
// proxied to it. A deployed build sets VITE_API_URL, or VITE_RECORDED=1 for the static
// demo that replays recorded runs.
const api = 'http://localhost:8000';

export default defineConfig({
  // GitHub Pages serves the site under /<repository>/; the Pages workflow sets VITE_BASE.
  base: process.env.VITE_BASE || '/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/ask': api,
      '/runs': api,
      '/arms': api,
      '/questions': api,
      '/reports': api,
      '/healthz': api,
    },
  },
});
