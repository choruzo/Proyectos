import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'
import { ThemeToggle } from '../components/ThemeToggle'

function navClass({ isActive }: { isActive: boolean }) {
  return `tp-card ${isActive ? 'tp-card--active' : ''}`
}

export function AdminLayout() {
  const { logout, me } = useAuth()
  const nav = useNavigate()

  return (
    <div className="tp-shell">
      <div className="tp-topbar">
        <div className="tp-topbarLeft">
          <div className="tp-brand">
            <h1 style={{ margin: 0 }}>Task Platform</h1>
            <small>Administración</small>
          </div>

          <div style={{ display: 'flex', gap: 8, marginLeft: 12, alignItems: 'center' }}>
            <Link to="/projects">Proyectos</Link>
            <button
              onClick={async () => {
                await logout()
                nav('/login')
              }}
            >
              Salir
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          {me ? (
            <div className="tp-muted" style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
              {me.email}
            </div>
          ) : null}
          <ThemeToggle />
        </div>
      </div>

      <div className="tp-layout tp-layout--sidebar-open">
        <div className="tp-panel tp-sidebar">
          <div className="tp-sidebarHeader">
            <h3 style={{ margin: 0 }}>Panel</h3>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <NavLink to="/admin/users" className={navClass} style={{ padding: 10, textDecoration: 'none' }}>
              Usuarios
            </NavLink>
            <NavLink to="/admin/projects" className={navClass} style={{ padding: 10, textDecoration: 'none' }}>
              Proyectos
            </NavLink>
            <NavLink to="/admin/stats" className={navClass} style={{ padding: 10, textDecoration: 'none' }}>
              Estadísticas
            </NavLink>
          </div>
        </div>

        <div className="tp-main">
          <Outlet />
        </div>
      </div>
    </div>
  )
}
