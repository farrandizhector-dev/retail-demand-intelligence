import { ReactNode } from 'react'
import { Sidebar } from './Sidebar'

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#0B0E14' }}>
      <Sidebar />
      <main style={{ marginLeft: '260px', minHeight: '100vh' }}>
        {children}
      </main>
    </div>
  )
}
