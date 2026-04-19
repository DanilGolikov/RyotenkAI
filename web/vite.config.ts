import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// Dev: Vite serves the SPA and proxies /api (incl. WebSockets) to the backend.
// Override backend address via RYOTENKAI_API_HOST / RYOTENKAI_API_PORT env vars.
// Prod: `ryotenkai serve` mounts web/dist/ — no proxy needed.
const apiHost = process.env.RYOTENKAI_API_HOST ?? '127.0.0.1'
const apiPort = process.env.RYOTENKAI_API_PORT ?? '8000'
const webPort = Number(process.env.RYOTENKAI_WEB_PORT ?? '5173')

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    port: webPort,
    proxy: {
      '/api': {
        target: `http://${apiHost}:${apiPort}`,
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
