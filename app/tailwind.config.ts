import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background: {
          base: '#0B0E14',
          surface: '#151922',
          elevated: '#1C2231',
          hover: '#232A3B',
        },
        primary: {
          DEFAULT: '#3B82F6',
          light: '#60A5FA',
          dark: '#2563EB',
          subtle: 'rgba(59,130,246,0.08)',
          glow: 'rgba(59,130,246,0.15)',
        },
        success: { DEFAULT: '#10B981', light: '#34D399', subtle: 'rgba(16,185,129,0.08)' },
        warning: { DEFAULT: '#F59E0B', light: '#FBBF24', subtle: 'rgba(245,158,11,0.08)' },
        danger: { DEFAULT: '#EF4444', light: '#F87171', subtle: 'rgba(239,68,68,0.08)' },
        text: {
          primary: '#F1F5F9',
          secondary: '#94A3B8',
          tertiary: '#64748B',
          inverse: '#0B0E14',
        },
        border: {
          DEFAULT: '#1E293B',
          subtle: '#162032',
          focus: '#3B82F6',
        },
      },
      boxShadow: {
        card: '0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2)',
        elevated: '0 4px 12px rgba(0,0,0,0.4)',
        glow: '0 0 20px rgba(59,130,246,0.1)',
      },
      borderRadius: {
        card: '12px',
        button: '8px',
        badge: '6px',
        full: '9999px',
      },
      fontSize: {
        metric: ['2.25rem', { lineHeight: '1', fontWeight: '700' }],
        'metric-sm': ['1.5rem', { lineHeight: '1', fontWeight: '600' }],
      },
    },
  },
  plugins: [],
}

export default config
