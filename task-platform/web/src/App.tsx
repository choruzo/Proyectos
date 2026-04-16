import { useEffect, useMemo, useState } from 'react'
import { apiFetch, tryRestoreSession } from './api/client'
import { ThemeProvider } from './theme'
import { AdminPage } from './pages/AdminPage'
import { LoginPage } from './pages/LoginPage'
import { ProjectsPage } from './pages/ProjectsPage'

type Me = { id: string; email: string; is_admin: boolean; is_active: boolean }

type AppView = 'projects' | 'admin'

export function App() {
  const [ready, setReady] = useState(false)
  const [authed, setAuthed] = useState(false)
  const [me, setMe] = useState<Me | null>(null)
  const [view, setView] = useState<AppView>('projects')

  async function loadMe(): Promise<Me | null> {
    try {
      return await apiFetch('/auth/me')
    } catch {
      return null
    }
  }

  async function bootstrapAfterAuth() {
    const m = await loadMe()
    if (!m) {
      setAuthed(false)
      setMe(null)
      return
    }

    setAuthed(true)
    setMe(m)
    setView(m.is_admin ? 'admin' : 'projects')
  }

  useEffect(() => {
    const onUnauthorized = () => {
      setAuthed(false)
      setMe(null)
    }
    window.addEventListener('tp:unauthorized', onUnauthorized)
    return () => window.removeEventListener('tp:unauthorized', onUnauthorized)
  }, [])

  useEffect(() => {
    ;(async () => {
      try {
        const ok = await tryRestoreSession()
        if (ok) await bootstrapAfterAuth()
        else setAuthed(false)
      } finally {
        setReady(true)
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const content = useMemo(() => {
    if (!ready) return <div className="tp-shell">Cargando…</div>

    if (!authed) {
      return <LoginPage onLoggedIn={bootstrapAfterAuth} />
    }

    const isAdmin = !!me?.is_admin

    if (isAdmin && view === 'admin') {
      return (
        <AdminPage
          onGoProjects={() => setView('projects')}
          onLogout={() => {
            setAuthed(false)
            setMe(null)
          }}
        />
      )
    }

    return (
      <ProjectsPage
        isAdmin={isAdmin}
        onGoAdmin={isAdmin ? () => setView('admin') : undefined}
        onLogout={() => {
          setAuthed(false)
          setMe(null)
        }}
      />
    )
  }, [ready, authed, me?.is_admin, view])

  return <ThemeProvider>{content}</ThemeProvider>
}
