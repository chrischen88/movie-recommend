import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // API_PORT lets a second copy of the app run alongside `make serve`.
    proxy: { "/api": `http://localhost:${process.env.API_PORT ?? 8000}` },
  },
});
