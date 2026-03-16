import { ReactNode } from 'react'

interface SectionCardProps {
  title?: string
  subtitle?: string
  children: ReactNode
  className?: string
  action?: ReactNode
  noPadding?: boolean
}

export function SectionCard({ title, subtitle, children, className = '', action, noPadding }: SectionCardProps) {
  return (
    <div
      className={className}
      style={{
        backgroundColor: '#151922',
        borderRadius: 12,
        border: '1px solid #1E293B',
        boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
        overflow: 'hidden',
      }}
    >
      {title && (
        <div style={{
          padding: '16px 20px',
          borderBottom: '1px solid #162032',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#F1F5F9' }}>{title}</div>
            {subtitle && <div style={{ fontSize: 12, color: '#64748B', marginTop: 2 }}>{subtitle}</div>}
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      <div style={noPadding ? {} : { padding: 20 }}>{children}</div>
    </div>
  )
}
