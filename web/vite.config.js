import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API paths the front end talks to. In development they are proxied
// to the local FastAPI service; in production FastAPI serves the built
// app itself, so the calls are same origin and need no proxy.
const API_PATHS = ['/health', '/model', '/features', '/predict', '/explain', '/feedback']

export default defineConfig({
  plugins: [react()],
  // The API serves the built app under /app, so the asset paths have to
  // be absolute against that prefix.
  base: '/app/',
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_PATHS.map((path) => [
        path,
        { target: 'http://127.0.0.1:8000', changeOrigin: true },
      ]),
    ),
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
