import { ReactNode } from 'react'

interface HeaderProps {
  title: string
  actions?: ReactNode
}

export function Header({ title, actions }: HeaderProps) {
  return (
    <header style={{
      position: 'sticky',
      top: 0,
      zIndex: 40,
      height: 64,
      backgroundColor: 'rgba(11,14,20,0.9)',
      backdropFilter: 'blur(12px)',
      borderBottom: '1px solid #162032',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 24px',
    }}>
      <h1 style={{ fontSize: 18, fontWeight: 600, color: '#F1F5F9', margin: 0 }}>{title}</h1>
      {actions && <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>{actions}</div>}
    </header>
  )
}
