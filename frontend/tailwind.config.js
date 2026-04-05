/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'neuro': {
          900: '#0a0a1a',
          800: '#0d0d2b',
          700: '#12123d',
          600: '#1a1a4e',
          500: '#252560',
        },
        'neon': {
          blue: '#00d4ff',
          purple: '#b44aff',
          pink: '#ff2d95',
          green: '#00ff88',
          orange: '#ff6b35',
        },
        'glass': {
          light: 'rgba(255, 255, 255, 0.05)',
          medium: 'rgba(255, 255, 255, 0.08)',
          heavy: 'rgba(255, 255, 255, 0.12)',
        },
      },
      fontFamily: {
        'display': ['Orbitron', 'monospace'],
        'body': ['Inter', 'system-ui', 'sans-serif'],
        'mono': ['JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        'neon-blue': '0 0 20px rgba(0, 212, 255, 0.3), 0 0 60px rgba(0, 212, 255, 0.1)',
        'neon-purple': '0 0 20px rgba(180, 74, 255, 0.3), 0 0 60px rgba(180, 74, 255, 0.1)',
        'neon-pink': '0 0 20px rgba(255, 45, 149, 0.3)',
        'glass': '0 8px 32px rgba(0, 0, 0, 0.4)',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(var(--tw-gradient-stops))',
        'cyber-grid': 'linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px)',
      },
      animation: {
        'pulse-neon': 'pulseNeon 2s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'scan': 'scan 3s linear infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        pulseNeon: {
          '0%, 100%': { boxShadow: '0 0 5px rgba(0, 212, 255, 0.5), 0 0 20px rgba(0, 212, 255, 0.2)' },
          '50%': { boxShadow: '0 0 20px rgba(0, 212, 255, 0.8), 0 0 60px rgba(0, 212, 255, 0.4)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        glow: {
          '0%': { opacity: 0.5 },
          '100%': { opacity: 1 },
        },
      },
    },
  },
  plugins: [],
}
