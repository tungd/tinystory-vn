import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/models': 'http://localhost:8000',
      '/generate/stream': 'http://localhost:8000',
      '/evaluate': 'http://localhost:8000',
    },
  },
})
