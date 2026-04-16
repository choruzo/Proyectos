import { FormEvent, useEffect, useMemo, useState } from 'react'
import { apiFetch } from '../api/client'

type Project = { id: string; name: string; description?: string | null }
type User = { id: string; email: string; is_admin: boolean; is_active: boolean }
type Release = {
  id: string
  name: string
  project_id: string
  note?: string | null
  start_date?: string | null
  end_date?: string | null
  status: string
}

export function AdminProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([])
  const [users, setUsers] = useState<User[]>([])

  const [selectedProjectId, setSelectedProjectId] = useState('')
  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  )

  const [newName, setNewName] = useState('')
  const [editName, setEditName] = useState('')
  const [editDesc, setEditDesc] = useState('')

  const [userFilter, setUserFilter] = useState('')
  const filteredUsers = useMemo(() => {
    const q = userFilter.trim().toLowerCase()
    if (!q) return users
    return users.filter((u) => u.email.toLowerCase().includes(q))
  }, [users, userFilter])

  const [memberUserIds, setMemberUserIds] = useState<Set<string>>(new Set())
  const [releases, setReleases] = useState<Release[]>([])
  const [newReleaseName, setNewReleaseName] = useState('')

  async function load() {
    const [ps, us] = await Promise.all([apiFetch('/projects'), apiFetch('/admin/users')])
    setProjects(ps)
    setUsers(us)

    if (selectedProjectId && !ps.find((p: Project) => p.id === selectedProjectId)) {
      setSelectedProjectId('')
      setMemberUserIds(new Set())
      setReleases([])
    }
  }

  async function loadProjectDetail(projectId: string) {
    const [members, rels] = await Promise.all([
      apiFetch(`/admin/projects/${projectId}/users`),
      apiFetch(`/projects/${projectId}/releases`),
    ])
    setMemberUserIds(new Set(members.user_ids || []))
    setReleases(rels)
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!selectedProject) return
    setEditName(selectedProject.name)
    setEditDesc(selectedProject.description ?? '')
  }, [selectedProject?.id])

  async function createProject(e: FormEvent) {
    e.preventDefault()
    if (!newName.trim()) return
    await apiFetch('/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName.trim() }),
    })
    setNewName('')
    await load()
  }

  async function saveProject() {
    if (!selectedProject) return
    await apiFetch(`/projects/${selectedProject.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: editName.trim() || selectedProject.name, description: editDesc.trim() ? editDesc : null }),
    })
    await load()
  }

  async function deleteProject() {
    if (!selectedProject) return
    if (!confirm(`¿Borrar proyecto "${selectedProject.name}"?`)) return
    await apiFetch(`/projects/${selectedProject.id}`, { method: 'DELETE' })
    setSelectedProjectId('')
    await load()
  }

  async function saveMembers() {
    if (!selectedProject) return
    await apiFetch(`/admin/projects/${selectedProject.id}/users`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_ids: Array.from(memberUserIds) }),
    })
    alert('Accesos guardados')
  }

  async function createRelease(e: FormEvent) {
    e.preventDefault()
    if (!selectedProject) return
    if (!newReleaseName.trim()) return

    await apiFetch(`/projects/${selectedProject.id}/releases`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newReleaseName.trim() }),
    })
    setNewReleaseName('')
    await loadProjectDetail(selectedProject.id)
  }

  async function renameRelease(rel: Release) {
    if (!selectedProject) return
    const name = prompt('Nuevo nombre de versión:', rel.name) || ''
    if (!name.trim()) return
    await apiFetch(`/projects/${selectedProject.id}/releases/${rel.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim() }),
    })
    await loadProjectDetail(selectedProject.id)
  }

  async function deleteRelease(rel: Release) {
    if (!selectedProject) return
    if (!confirm(`¿Borrar versión "${rel.name}"? (Las tareas pasarán a backlog)`)) return
    await apiFetch(`/projects/${selectedProject.id}/releases/${rel.id}`, { method: 'DELETE' })
    await loadProjectDetail(selectedProject.id)
  }

  return (
    <div className="tp-panel" style={{ padding: 12 }}>
      <h2 style={{ marginTop: 0 }}>Proyectos</h2>

      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 12 }}>
        <div>
          <h3 style={{ marginTop: 0 }}>Crear proyecto</h3>
          <form onSubmit={createProject} style={{ display: 'flex', gap: 8 }}>
            <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Nombre" />
            <button className="tp-btn--primary" type="submit">
              +
            </button>
          </form>

          <h3 style={{ marginTop: 12 }}>Listado</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {projects.map((p) => (
              <button
                key={p.id}
                className="tp-card"
                onClick={async () => {
                  setSelectedProjectId(p.id)
                  await loadProjectDetail(p.id)
                }}
                style={{
                  textAlign: 'left',
                  padding: 10,
                  background:
                    selectedProjectId === p.id
                      ? 'color-mix(in srgb, var(--accent) 12%, var(--surface))'
                      : 'var(--surface)',
                }}
              >
                <div style={{ fontWeight: 650 }}>{p.name}</div>
                {p.description ? <div className="tp-muted" style={{ fontSize: 12 }}>{p.description}</div> : null}
              </button>
            ))}
          </div>
        </div>

        <div>
          {!selectedProject ? (
            <div className="tp-muted">Selecciona un proyecto</div>
          ) : (
            <div style={{ display: 'grid', gap: 12 }}>
              <div className="tp-panel" style={{ padding: 12 }}>
                <h3 style={{ marginTop: 0 }}>Detalle del proyecto</h3>
                <div style={{ display: 'grid', gap: 10, maxWidth: 900 }}>
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
                      Guardar
                    </button>
                    <button className="tp-btn--danger" onClick={deleteProject}>
                      Borrar
                    </button>
                    <button className="tp-btn--ghost" onClick={() => loadProjectDetail(selectedProject.id)}>
                      Refrescar
                    </button>
                  </div>
                </div>
              </div>

              <div className="tp-panel" style={{ padding: 12 }}>
                <h3 style={{ marginTop: 0 }}>Accesos al proyecto</h3>
                <input value={userFilter} onChange={(e) => setUserFilter(e.target.value)} placeholder="Buscar usuario…" />
                <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {filteredUsers.map((u) => (
                    <label key={u.id} className="tp-card" style={{ padding: 10, display: 'flex', gap: 10, opacity: u.is_active ? 1 : 0.6 }}>
                      <input
                        type="checkbox"
                        checked={memberUserIds.has(u.id)}
                        onChange={(e) => {
                          setMemberUserIds((prev) => {
                            const next = new Set(prev)
                            if (e.target.checked) next.add(u.id)
                            else next.delete(u.id)
                            return next
                          })
                        }}
                      />
                      <div>
                        <div style={{ fontWeight: 650 }}>{u.email}</div>
                        <div className="tp-muted" style={{ fontSize: 12 }}>
                          {u.is_admin ? 'admin' : 'usuario'} · {u.is_active ? 'activo' : 'inactivo'}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
                <div style={{ marginTop: 10 }}>
                  <button className="tp-btn--primary" onClick={saveMembers}>
                    Guardar accesos
                  </button>
                </div>
              </div>

              <div className="tp-panel" style={{ padding: 12 }}>
                <h3 style={{ marginTop: 0 }}>Versiones (releases)</h3>

                <form onSubmit={createRelease} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input value={newReleaseName} onChange={(e) => setNewReleaseName(e.target.value)} placeholder="Nueva versión" />
                  <button type="submit" className="tp-btn--primary">
                    Crear
                  </button>
                </form>

                <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {releases.map((r) => (
                    <div key={r.id} className="tp-card" style={{ padding: 10, display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                      <div>
                        <div style={{ fontWeight: 650 }}>{r.name}</div>
                        <div className="tp-muted" style={{ fontSize: 12 }}>{r.status}</div>
                      </div>
                      <div style={{ display: 'flex', gap: 8 }}>
                        <button onClick={() => renameRelease(r)}>Renombrar</button>
                        <button className="tp-btn--danger" onClick={() => deleteRelease(r)}>Borrar</button>
                      </div>
                    </div>
                  ))}
                  {releases.length === 0 ? <div className="tp-muted">Sin versiones</div> : null}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
