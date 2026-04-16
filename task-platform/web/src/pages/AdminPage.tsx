import { FormEvent, useEffect, useMemo, useState } from 'react'
import { apiFetch } from '../api/client'
import { ThemeToggle } from '../components/ThemeToggle'

type User = { id: string; email: string; is_admin: boolean; is_active: boolean }
type Project = { id: string; name: string; description?: string | null }

export function AdminPage({
  onLogout,
  onGoProjects,
}: {
  onLogout: () => void
  onGoProjects: () => void
}) {
  const [users, setUsers] = useState<User[]>([])
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedUserId, setSelectedUserId] = useState<string>('')
  const [assigned, setAssigned] = useState<Set<string>>(new Set())

  const [newEmail, setNewEmail] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [newIsAdmin, setNewIsAdmin] = useState(false)

  const selectedUser = useMemo(() => users.find((u) => u.id === selectedUserId) ?? null, [users, selectedUserId])

  async function loadAll() {
    const [us, ps] = await Promise.all([apiFetch('/admin/users'), apiFetch('/projects')])
    setUsers(us)
    setProjects(ps)

    if (selectedUserId && !us.find((u: User) => u.id === selectedUserId)) {
      setSelectedUserId('')
      setAssigned(new Set())
    }
  }

  async function loadAssignments(userId: string) {
    const data = await apiFetch(`/admin/users/${userId}/projects`)
    setAssigned(new Set(data.project_ids || []))
  }

  useEffect(() => {
    loadAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function createUser(e: FormEvent) {
    e.preventDefault()
    if (!newEmail.trim() || newPassword.length < 8) return
    await apiFetch('/admin/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: newEmail.trim(), password: newPassword, is_admin: newIsAdmin }),
    })
    setNewEmail('')
    setNewPassword('')
    setNewIsAdmin(false)
    await loadAll()
  }

  async function patchUser(userId: string, patch: any) {
    await apiFetch(`/admin/users/${userId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    })
    await loadAll()
  }

  async function resetPassword(userId: string) {
    const pw = prompt('Nueva contraseña (mín. 8 caracteres):') || ''
    if (pw.length < 8) return
    await apiFetch(`/admin/users/${userId}/reset-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_password: pw }),
    })
    alert('Contraseña actualizada')
  }

  async function deactivateUser(userId: string) {
    if (!confirm('¿Desactivar este usuario?')) return
    await apiFetch(`/admin/users/${userId}`, { method: 'DELETE' })
    await loadAll()
  }

  async function saveAssignments() {
    if (!selectedUserId) return
    await apiFetch(`/admin/users/${selectedUserId}/projects`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_ids: Array.from(assigned) }),
    })
    alert('Asignaciones guardadas')
  }

  return (
    <div className="tp-shell">
      <div className="tp-topbar">
        <div className="tp-topbarLeft">
          <div className="tp-brand">
            <h1 style={{ margin: 0 }}>Task Platform</h1>
            <small>Administración</small>
          </div>
          <div style={{ display: 'flex', gap: 8, marginLeft: 12 }}>
            <button onClick={onGoProjects}>Proyectos</button>
            <button onClick={onLogout}>Salir</button>
          </div>
        </div>
        <ThemeToggle />
      </div>

      <div className="tp-layout tp-layout--sidebar-open">
        <div className="tp-panel tp-sidebar">
          <div className="tp-sidebarHeader">
            <h3 style={{ margin: 0 }}>Usuarios</h3>
          </div>

          <form onSubmit={createUser} style={{ display: 'grid', gap: 8 }}>
            <input value={newEmail} onChange={(e) => setNewEmail(e.target.value)} placeholder="Email" />
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="Password (mín. 8)"
            />
            <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input type="checkbox" checked={newIsAdmin} onChange={(e) => setNewIsAdmin(e.target.checked)} />
              Admin
            </label>
            <button className="tp-btn--primary" type="submit">
              Crear usuario
            </button>
          </form>

          <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {users.map((u) => (
              <button
                key={u.id}
                className="tp-card"
                onClick={async () => {
                  setSelectedUserId(u.id)
                  await loadAssignments(u.id)
                }}
                style={{
                  textAlign: 'left',
                  padding: 10,
                  background:
                    selectedUserId === u.id
                      ? 'color-mix(in srgb, var(--accent) 12%, var(--surface))'
                      : 'var(--surface)',
                  opacity: u.is_active ? 1 : 0.6,
                }}
              >
                <div style={{ fontWeight: 650 }}>{u.email}</div>
                <div className="tp-muted" style={{ fontSize: 12 }}>
                  {u.is_admin ? 'admin' : 'usuario'} · {u.is_active ? 'activo' : 'inactivo'}
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="tp-main">
          <div className="tp-panel" style={{ padding: 12 }}>
            <h2 style={{ marginTop: 0 }}>Detalle</h2>

            {!selectedUser ? (
              <div className="tp-muted">Selecciona un usuario</div>
            ) : (
              <div style={{ display: 'grid', gap: 10, maxWidth: 900 }}>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                  <div style={{ fontFamily: 'var(--font-mono)' }}>{selectedUser.email}</div>
                  <button onClick={() => patchUser(selectedUser.id, { is_admin: !selectedUser.is_admin })}>
                    {selectedUser.is_admin ? 'Quitar admin' : 'Hacer admin'}
                  </button>
                  <button onClick={() => patchUser(selectedUser.id, { is_active: !selectedUser.is_active })}>
                    {selectedUser.is_active ? 'Desactivar' : 'Activar'}
                  </button>
                  <button onClick={() => resetPassword(selectedUser.id)}>Reset password</button>
                  <button className="tp-btn--danger" onClick={() => deactivateUser(selectedUser.id)}>
                    Delete (soft)
                  </button>
                  <button className="tp-btn--ghost" onClick={loadAll}>
                    Refrescar
                  </button>
                </div>

                <div>
                  <h3 style={{ marginTop: 10 }}>Proyectos asignados</h3>
                  {projects.length === 0 ? (
                    <div className="tp-muted">No hay proyectos</div>
                  ) : (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      {projects.map((p) => (
                        <label key={p.id} className="tp-card" style={{ padding: 10, display: 'flex', gap: 10 }}>
                          <input
                            type="checkbox"
                            checked={assigned.has(p.id)}
                            onChange={(e) => {
                              setAssigned((prev) => {
                                const next = new Set(prev)
                                if (e.target.checked) next.add(p.id)
                                else next.delete(p.id)
                                return next
                              })
                            }}
                          />
                          <div>
                            <div style={{ fontWeight: 650 }}>{p.name}</div>
                            {p.description ? (
                              <div className="tp-muted" style={{ fontSize: 12 }}>
                                {p.description}
                              </div>
                            ) : null}
                          </div>
                        </label>
                      ))}
                    </div>
                  )}

                  <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
                    <button className="tp-btn--primary" onClick={saveAssignments} disabled={!selectedUserId}>
                      Guardar asignaciones
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
