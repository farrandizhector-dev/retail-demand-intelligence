import { TrendingUp, TrendingDown } from 'lucide-react'

type Accent = 'blue' | 'green' | 'yellow' | 'red' | 'primary' | 'success' | 'warning' | 'danger'

interface KPICardProps {
  label: string
  value: string | number
  delta?: number
  deltaLabel?: string
  accent?: Accent
  icon?: React.ComponentType<{ size?: number; color?: string }>
  prefix?: string
  suffix?: string
}

const accentMap: Record<string, { bg: string; text: string; bar: string }> = {
  blue: { bg: 'rgba(59,130,246,0.08)', text: '#3B82F6', bar: '#3B82F6' },
  primary: { bg: 'rgba(59,130,246,0.08)', text: '#3B82F6', bar: '#3B82F6' },
  green: { bg: 'rgba(16,185,129,0.08)', text: '#10B981', bar: '#10B981' },
  success: { bg: 'rgba(16,185,129,0.08)', text: '#10B981', bar: '#10B981' },
  yellow: { bg: 'rgba(245,158,11,0.08)', text: '#F59E0B', bar: '#F59E0B' },
  warning: { bg: 'rgba(245,158,11,0.08)', text: '#F59E0B', bar: '#F59E0B' },
  red: { bg: 'rgba(239,68,68,0.08)', text: '#EF4444', bar: '#EF4444' },
  danger: { bg: 'rgba(239,68,68,0.08)', text: '#EF4444', bar: '#EF4444' },
}

export function KPICard({ label, value, delta, deltaLabel, accent = 'blue', icon: Icon, prefix, suffix }: KPICardProps) {
  const c = accentMap[accent] ?? accentMap.blue
  const isPositive = delta !== undefined && delta >= 0

  return (
    <div
      style={{
        position: 'relative',
        backgroundColor: '#151922',
        borderRadius: 12,
        padding: 20,
        border: '1px solid #1E293B',
        boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
        overflow: 'hidden',
        transition: 'box-shadow 200ms, border-color 200ms',
        cursor: 'default',
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLDivElement).style.boxShadow = '0 4px 12px rgba(0,0,0,0.4)'
        ;(e.currentTarget as HTMLDivElement).style.borderColor = '#3B82F6'
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLDivElement).style.boxShadow = '0 1px 3px rgba(0,0,0,0.3)'
        ;(e.currentTarget as HTMLDivElement).style.borderColor = '#1E293B'
      }}
    >
      {/* Left accent bar */}
      <div style={{
        position: 'absolute', left: 0, top: 0, bottom: 0, width: 4,
        backgroundColor: c.bar, borderRadius: '12px 0 0 12px',
      }} />

      {/* Icon */}
      {Icon && (
        <div style={{
          position: 'absolute', top: 16, right: 16,
          padding: 8, backgroundColor: c.bg, borderRadius: 8,
        }}>
          <Icon size={16} color={c.text} />
        </div>
      )}

      <div style={{ paddingLeft: 12 }}>
        {/* Label */}
        <div style={{
          fontSize: 11, fontWeight: 500, textTransform: 'uppercase',
          letterSpacing: '0.08em', color: '#64748B', marginBottom: 8,
        }}>
          {label}
        </div>

        {/* Value */}
        <div style={{ fontSize: 28, fontWeight: 700, color: '#F1F5F9', lineHeight: 1 }}>
          {prefix && <span style={{ fontSize: 18, marginRight: 2 }}>{prefix}</span>}
          {value}
          {suffix && <span style={{ fontSize: 18, marginLeft: 2 }}>{suffix}</span>}
        </div>

        {/* Delta */}
        {delta !== undefined && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8 }}>
            {isPositive
              ? <TrendingUp size={14} color="#10B981" />
              : <TrendingDown size={14} color="#EF4444" />
            }
            <span style={{ fontSize: 12, fontWeight: 600, color: isPositive ? '#10B981' : '#EF4444' }}>
              {isPositive ? '+' : ''}{delta.toFixed(1)}%
            </span>
            {deltaLabel && <span style={{ fontSize: 12, color: '#64748B' }}>{deltaLabel}</span>}
          </div>
        )}
      </div>
    </div>
  )
}
