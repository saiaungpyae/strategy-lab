import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Build lands in viewer/static/dist and is served by viewer/server.py at
// /dist/*, with /, /chart, /swarm and /evolution returning the SPA index.
// In dev, vite serves at / and proxies API + report routes to the Python
// server.
export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === 'build' ? '/dist/' : '/',
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8020',
      '/reports': 'http://127.0.0.1:8020',
    },
  },
}))
