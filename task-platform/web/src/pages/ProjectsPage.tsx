import { useEffect, useState } from 'react'
import { apiFetch } from '../api/client'
import { ThemeToggle } from '../components/ThemeToggle'
import { BoardPage } from './BoardPage'

type Project = { id: string; name: string; description?: string | null }

export function ProjectsPage({
  onLogout,
  isAdmin,
  onGoAdmin,
}: {
  onLogout: () => void | Promise<void>
  isAdmin: boolean
  onGoAdmin?: () => void
}) {
  const [projects, setProjects] = useState<Project[]>([])
  const [selected, setSelected] = useState<Project | null>(null)
  const [name, setName] = useState('')
  const [editName, setEditName] = useState('')
  const [editDesc, setEditDesc] = useState('')

  const [sidebarOpen, setSidebarOpen] = useState(() => {
    if (typeof window === 'undefined') return true
    return !window.matchMedia('(max-width: 900px)').matches
  })

  useEffect(() => {
    if (!sidebarOpen) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSidebarOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [sidebarOpen])

  function toggleSidebar() {
    setSidebarOpen((v) => !v)
  }

  function selectProject(p: Project) {
    setSelected(p)
    if (typeof window !== 'undefined' && window.matchMedia('(max-width: 900px)').matches) setSidebarOpen(false)
  }

  async function load() {
    const data = await apiFetch('/projects')
    setProjects(data)
    if (selected) {
      const still = data.find((p: Project) => p.id === selected.id)
      setSelected(still ?? null)
      if (still) {
        setEditName(still.name)
        setEditDesc(still.description ?? '')
      }
    }
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (selected) {
      setEditName(selected.name)
      setEditDesc(selected.description ?? '')
    }
  }, [selected?.id])

  async function createProject() {
    if (!name.trim()) return
    await apiFetch('/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim() }),
    })
    setName('')
    await load()
  }

  async function exportJson() {
    const data = await apiFetch('/export')
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `taskplatform-export-${new Date().toISOString().slice(0, 10)}.json`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  async function saveProject() {
    if (!selected) return
    await apiFetch(`/projects/${selected.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: editName.trim() || selected.name, description: editDesc.trim() ? editDesc : null }),
    })
    await load()
  }

  async function deleteProject() {
    if (!selected) return
    if (!confirm(`¿Borrar proyecto "${selected.name}"? (Se borrarán también sus tareas, versiones y adjuntos)`)) return
    await apiFetch(`/projects/${selected.id}`, { method: 'DELETE' })
    setSelected(null)
    await load()
  }

  async function logout() {
    await onLogout()
  }

  return (
    <div className="tp-shell">
      <div className="tp-topbar">
        <div className="tp-topbarLeft">
          <button
            type="button"
            className="tp-btn--ghost tp-iconBtn"
            onClick={toggleSidebar}
            aria-label={sidebarOpen ? 'Ocultar menú de proyectos' : 'Mostrar menú de proyectos'}
            aria-expanded={sidebarOpen}
            title={sidebarOpen ? 'Ocultar menú' : 'Mostrar menú'}
          >
            <span aria-hidden="true">{sidebarOpen ? '⟨' : '☰'}</span>
          </button>

          <div className="tp-brand">
            <h1 style={{ margin: 0 }}>Task Platform</h1>
            <small>Proyectos · Versiones · Tareas</small>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {isAdmin && onGoAdmin ? <button onClick={onGoAdmin}>Administración</button> : null}
          <ThemeToggle />
        </div>
      </div>

      <div className={`tp-layout ${sidebarOpen ? 'tp-layout--sidebar-open' : 'tp-layout--sidebar-collapsed'}`}>
        <div className="tp-sidebarOverlay" aria-hidden="true" onClick={() => setSidebarOpen(false)} />

        <div className="tp-panel tp-sidebar">
          <div className="tp-sidebarHeader">
            <h3 style={{ margin: 0 }}>Proyectos</h3>
            <button
              type="button"
              className="tp-btn--ghost tp-iconBtn tp-sidebarClose"
              onClick={() => setSidebarOpen(false)}
              aria-label="Cerrar menú"
              title="Cerrar"
            >
              <span aria-hidden="true">×</span>
            </button>
          </div>
          {isAdmin ? (
            <div style={{ display: 'flex', gap: 8 }}>
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Nuevo proyecto" />
              <button className="tp-btn--primary" onClick={createProject}>
                +
              </button>
            </div>
          ) : null}

          <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {projects.length === 0 ? (
              <div className="tp-muted" style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                No tienes proyectos asignados
              </div>
            ) : null}
            {projects.map((p) => (
              <button
                key={p.id}
                onClick={() => selectProject(p)}
                className="tp-card"
                style={{
                  textAlign: 'left',
                  padding: 10,
                  background:
                    selected?.id === p.id
                      ? 'color-mix(in srgb, var(--accent) 12%, var(--surface))'
                      : 'var(--surface)',
                  borderColor:
                    selected?.id === p.id
                      ? 'color-mix(in srgb, var(--accent) 40%, var(--border))'
                      : 'var(--border)',
                }}
              >
                <div style={{ fontWeight: 650 }}>{p.name}</div>
                {p.description ? <div className="tp-muted" style={{ fontSize: 12 }}>{p.description}</div> : null}
              </button>
            ))}
          </div>

          <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {isAdmin ? <button onClick={exportJson}>Exportar JSON</button> : null}
            <button onClick={logout}>Salir</button>
          </div>
        </div>

        <div className="tp-main">
          {selected ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div className="tp-panel" style={{ padding: 12 }}>
                <h3 style={{ marginTop: 0 }}>Proyecto</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 10, maxWidth: 820 }}>
                  <div>
                    <label>Nombre</label>
                    <input
                      style={{ width: '100%' }}
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      disabled={!isAdmin}
                    />
                  </div>
                  <div>
                    <label>Descripción</label>
                    <textarea
                      style={{ width: '100%', minHeight: 90 }}
                      value={editDesc}
                      onChange={(e) => setEditDesc(e.target.value)}
                      disabled={!isAdmin}
                    />
                  </div>
                  {isAdmin ? (
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      <button className="tp-btn--primary" onClick={saveProject}>
                        Guardar proyecto
                      </button>
                      <button className="tp-btn--danger" onClick={deleteProject}>
                        Borrar proyecto
                      </button>
                    </div>
                  ) : null}
                </div>
              </div>

              <BoardPage project={selected} isAdmin={isAdmin} />
            </div>
          ) : (
            <div className="tp-panel" style={{ padding: 16 }}>
              <div className="tp-muted">Selecciona un proyecto</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
