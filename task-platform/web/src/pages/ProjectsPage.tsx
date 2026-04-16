import { useEffect, useState } from 'react'
import { apiFetch, getApiUrl, setAccessToken } from '../api/client'
import { ThemeToggle } from '../components/ThemeToggle'
import { BoardPage } from './BoardPage'

type Project = { id: string; name: string; description?: string | null }

export function ProjectsPage({ onLogout }: { onLogout: () => void }) {
  const [projects, setProjects] = useState<Project[]>([])
  const [selected, setSelected] = useState<Project | null>(null)
  const [name, setName] = useState('')
  const [editName, setEditName] = useState('')
  const [editDesc, setEditDesc] = useState('')

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
    try {
      await fetch(`${getApiUrl()}/auth/logout`, { method: 'POST', credentials: 'include' })
    } finally {
      setAccessToken(null)
      onLogout()
    }
  }

  return (
    <div className="tp-shell">
      <div className="tp-topbar">
        <div className="tp-brand">
          <h1 style={{ margin: 0 }}>Task Platform</h1>
          <small>Proyectos · Versiones · Tareas</small>
        </div>
        <ThemeToggle />
      </div>

      <div className="tp-layout">
        <div className="tp-panel tp-sidebar">
          <h3 style={{ marginTop: 0 }}>Proyectos</h3>
          <div style={{ display: 'flex', gap: 8 }}>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Nuevo proyecto" />
            <button className="tp-btn--primary" onClick={createProject}>
              +
            </button>
          </div>

          <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {projects.map((p) => (
              <button
                key={p.id}
                onClick={() => setSelected(p)}
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
            <button onClick={exportJson}>Exportar JSON</button>
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
                    <input style={{ width: '100%' }} value={editName} onChange={(e) => setEditName(e.target.value)} />
                  </div>
                  <div>
                    <label>Descripción</label>
                    <textarea
                      style={{ width: '100%', minHeight: 90 }}
                      value={editDesc}
                      onChange={(e) => setEditDesc(e.target.value)}
                    />
                  </div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <button className="tp-btn--primary" onClick={saveProject}>
                      Guardar proyecto
                    </button>
                    <button className="tp-btn--danger" onClick={deleteProject}>
                      Borrar proyecto
                    </button>
                  </div>
                </div>
              </div>

              <BoardPage project={selected} />
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
