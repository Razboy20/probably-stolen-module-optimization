import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import typegpu from 'unplugin-typegpu/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [typegpu(), react()],
    base: '/probably-stolen-module-optimization/',
})
