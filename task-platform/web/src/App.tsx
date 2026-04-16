import { useEffect, useMemo, useState } from 'react'
import { tryRestoreSession } from './api/client'
import { ThemeProvider } from './theme'
import { LoginPage } from './pages/LoginPage'
import { ProjectsPage } from './pages/ProjectsPage'

export function App() {
  const [ready, setReady] = useState(false)
  const [authed, setAuthed] = useState(false)

  useEffect(() => {
    const onUnauthorized = () => setAuthed(false)
    window.addEventListener('tp:unauthorized', onUnauthorized)
    return () => window.removeEventListener('tp:unauthorized', onUnauthorized)
  }, [])

  useEffect(() => {
    ;(async () => {
      try {
        const ok = await tryRestoreSession()
        setAuthed(ok)
      } finally {
        setReady(true)
      }
    })()
  }, [])

  const content = useMemo(() => {
    if (!ready) return <div className="tp-shell">Cargando…</div>
    if (!authed) return <LoginPage onLoggedIn={() => setAuthed(true)} />
    return <ProjectsPage onLogout={() => setAuthed(false)} />
  }, [ready, authed])

  return <ThemeProvider>{content}</ThemeProvider>
}
