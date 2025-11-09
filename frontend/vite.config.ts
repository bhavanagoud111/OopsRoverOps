import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      'figma:asset/3d02b908cd238433f2979cfd7e5e716f8472ec41.png': path.resolve(__dirname, './src/assets/3d02b908cd238433f2979cfd7e5e716f8472ec41.png'),
      'figma:asset/868b782cadc788c733928c2597034af2ebdd4d87.png': path.resolve(__dirname, './src/assets/868b782cadc788c733928c2597034af2ebdd4d87.png'),
    }
  },
  server: {
    port: 5174,
    host: true
  }
})
