import { useLocation, Link } from 'react-router-dom'
import {
  Activity, LayoutDashboard, TrendingUp, ShieldAlert,
  Package, BarChart3, FlaskConical, BookOpen
} from 'lucide-react'

const links = [
  { to: '/overview', icon: LayoutDashboard, label: 'Overview' },
  { to: '/forecast', icon: TrendingUp, label: 'Forecast Lab' },
  { to: '/inventory', icon: ShieldAlert, label: 'Inventory Risk' },
  { to: '/product', icon: Package, label: 'Product Drilldown' },
  { to: '/quality', icon: BarChart3, label: 'Model Quality' },
  { to: '/scenarios', icon: FlaskConical, label: 'Scenario Lab' },
  { to: '/about', icon: BookOpen, label: 'About' },
]

export function Sidebar() {
  const location = useLocation()
  return (
    <aside
      style={{
        position: 'fixed',
        left: 0,
        top: 0,
        height: '100vh',
        width: '260px',
        backgroundColor: '#151922',
        borderRight: '1px solid #1E293B',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 50,
      }}
    >
      {/* Logo */}
      <div style={{ padding: '24px 20px 16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ padding: 8, backgroundColor: 'rgba(59,130,246,0.08)', borderRadius: 8 }}>
            <Activity size={20} color="#3B82F6" />
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#F1F5F9' }}>AI Supply Chain</div>
            <div style={{ fontSize: 10, color: '#64748B', letterSpacing: '0.1em', textTransform: 'uppercase' }}>Control Tower</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: '0 12px', overflowY: 'auto' }}>
        {links.map(({ to, icon: Icon, label }) => {
          const active = location.pathname === to || (to === '/overview' && location.pathname === '/')
          return (
            <Link
              key={to}
              to={to}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                height: 40,
                padding: active ? '0 12px 0 10px' : '0 12px',
                borderRadius: 8,
                marginBottom: 4,
                fontSize: 14,
                fontWeight: 500,
                textDecoration: 'none',
                backgroundColor: active ? 'rgba(59,130,246,0.08)' : 'transparent',
                color: active ? '#3B82F6' : '#94A3B8',
                borderLeft: active ? '2px solid #3B82F6' : '2px solid transparent',
                transition: 'all 150ms',
              }}
              onMouseEnter={e => {
                if (!active) {
                  (e.currentTarget as HTMLAnchorElement).style.backgroundColor = '#232A3B'
                  ;(e.currentTarget as HTMLAnchorElement).style.color = '#F1F5F9'
                }
              }}
              onMouseLeave={e => {
                if (!active) {
                  (e.currentTarget as HTMLAnchorElement).style.backgroundColor = 'transparent'
                  ;(e.currentTarget as HTMLAnchorElement).style.color = '#94A3B8'
                }
              }}
            >
              <Icon size={18} />
              <span>{label}</span>
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div style={{ padding: '16px 20px', borderTop: '1px solid #162032' }}>
        <div style={{ fontSize: 12, color: '#64748B', display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 6, height: 6, backgroundColor: '#10B981', borderRadius: '50%', display: 'inline-block' }} />
          v2.0 · Héctor Ferrándiz
        </div>
      </div>
    </aside>
  )
}
