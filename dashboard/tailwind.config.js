/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        danger: { DEFAULT: "#dc2626", light: "#fca5a5" },
        warning: { DEFAULT: "#f59e0b", light: "#fde68a" },
        success: { DEFAULT: "#16a34a", light: "#bbf7d0" },
        observe: { DEFAULT: "#3b82f6", light: "#93c5fd" },
      },
    },
  },
  plugins: [],
};
