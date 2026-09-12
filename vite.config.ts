import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import typegpu from 'unplugin-typegpu/vite'
import { cloudflare } from '@cloudflare/vite-plugin'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  // The Cloudflare plugin answers every HTML request with the single-page app, which hides scripts/bench.html
  plugins: [typegpu(), react(), ...(mode === 'bench' ? [] : [cloudflare()])],
  base: mode === 'cloudflare' ? '/' : '/probably-stolen-module-optimization/',
}))
