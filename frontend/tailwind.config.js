/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'mission-control': {
          dark: '#0a0e27',
          darker: '#050814',
          light: '#1a1f3a',
          accent: '#00d4ff',
          'accent-dark': '#0099cc',
        }
      }
    },
  },
  plugins: [],
}

