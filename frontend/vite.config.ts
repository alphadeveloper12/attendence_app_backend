import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Vite build is consumed by Django through django-vite.
// - `base: '/static/'` so hashed asset URLs resolve under Django's STATIC_URL.
// - dev server runs on a fixed port with an absolute `origin` so HMR assets
//   load correctly when the page is served by Django on :8000.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: '/static/',
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    origin: 'http://127.0.0.1:5173',
    cors: true,
  },
  build: {
    manifest: true,
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: 'src/main.tsx',
    },
  },
})
