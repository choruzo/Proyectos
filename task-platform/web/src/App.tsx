import { BrowserRouter, Navigate, Route, Routes, useNavigate } from 'react-router-dom'
import { ThemeProvider } from './theme'
import { AuthProvider, useAuth } from './auth'
import { ProjectsPage } from './pages/ProjectsPage'
import { LoginRoute } from './pages/LoginRoute'
import { AdminLayout } from './pages/AdminLayout'
import { AdminUsersPage } from './pages/AdminUsersPage'
import { AdminProjectsPage } from './pages/AdminProjectsPage'
import { AdminStatsPage } from './pages/AdminStatsPage'
import { ForbiddenPage } from './pages/ForbiddenPage'

function RequireAuth({ children }: { children: any }) {
  const { ready, me } = useAuth()
  if (!ready) return <div className="tp-shell">Cargando…</div>
  if (!me) return <Navigate to="/login" replace />
  return children
}

function RequireAdmin({ children }: { children: any }) {
  const { me } = useAuth()
  if (!me) return <Navigate to="/login" replace />
  if (!me.is_admin) return <ForbiddenPage />
  return children
}

function HomeRedirect() {
  const { ready, me } = useAuth()
  if (!ready) return <div className="tp-shell">Cargando…</div>
  if (!me) return <Navigate to="/login" replace />
  return <Navigate to={me.is_admin ? '/admin/users' : '/projects'} replace />
}

function ProjectsRoute() {
  const { me, logout } = useAuth()
  const nav = useNavigate()

  return (
    <ProjectsPage
      isAdmin={!!me?.is_admin}
      onGoAdmin={me?.is_admin ? () => nav('/admin/users') : undefined}
      onLogout={async () => {
        await logout()
        nav('/login', { replace: true })
      }}
    />
  )
}

export function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<HomeRedirect />} />
            <Route path="/login" element={<LoginRoute />} />
            <Route
              path="/projects"
              element={
                <RequireAuth>
                  <ProjectsRoute />
                </RequireAuth>
              }
            />

            <Route
              path="/admin"
              element={
                <RequireAuth>
                  <RequireAdmin>
                    <AdminLayout />
                  </RequireAdmin>
                </RequireAuth>
              }
            >
              <Route index element={<Navigate to="/admin/users" replace />} />
              <Route path="users" element={<AdminUsersPage />} />
              <Route path="projects" element={<AdminProjectsPage />} />
              <Route path="stats" element={<AdminStatsPage />} />
            </Route>

            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  )
}
