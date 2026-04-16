import { FormEvent, useEffect, useMemo, useState } from 'react'
import { apiFetch } from '../api/client'

type User = { id: string; email: string; is_admin: boolean; is_active: boolean }
type Project = { id: string; name: string; description?: string | null }

export function AdminUsersPage() {
  const [users, setUsers] = useState<User[]>([])
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedUserId, setSelectedUserId] = useState<string>('')
  const [assignedProjectIds, setAssignedProjectIds] = useState<Set<string>>(new Set())

  const [newEmail, setNewEmail] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [newIsAdmin, setNewIsAdmin] = useState(false)

  const selectedUser = useMemo(() => users.find((u) => u.id === selectedUserId) ?? null, [users, selectedUserId])

  async function loadUsers() {
    const us = await apiFetch('/admin/users')
    setUsers(us)

    if (selectedUserId && !us.find((u: User) => u.id === selectedUserId)) {
      setSelectedUserId('')
      setAssignedProjectIds(new Set())
    }
  }

  async function loadProjects() {
    const ps = await apiFetch('/projects')
    setProjects(ps)
  }

  async function loadAssignments(userId: string) {
    const data = await apiFetch(`/admin/users/${userId}/projects`)
    setAssignedProjectIds(new Set(data.project_ids || []))
  }

  async function saveAssignments() {
    if (!selectedUserId) return
    await apiFetch(`/admin/users/${selectedUserId}/projects`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_ids: Array.from(assignedProjectIds) }),
    })
    alert('Asignaciones guardadas')
  }

  useEffect(() => {
    Promise.all([loadUsers(), loadProjects()])
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
    await loadUsers()
  }

  async function patchUser(userId: string, patch: any) {
    await apiFetch(`/admin/users/${userId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    })
    await loadUsers()
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
    await loadUsers()
  }

  return (
    <div className="tp-panel" style={{ padding: 12 }}>
      <h2 style={{ marginTop: 0 }}>Usuarios</h2>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div>
          <h3 style={{ marginTop: 0 }}>Crear usuario</h3>
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
              Crear
            </button>
          </form>
        </div>

        <div>
          <h3 style={{ marginTop: 0 }}>Listado</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
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
      </div>

      <div style={{ marginTop: 12 }}>
        <h3>Detalle</h3>
        {!selectedUser ? (
          <div className="tp-muted">Selecciona un usuario</div>
        ) : (
          <div style={{ display: 'grid', gap: 12 }}>
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
              <button
                className="tp-btn--ghost"
                onClick={async () => {
                  await Promise.all([loadUsers(), loadProjects()])
                  await loadAssignments(selectedUser.id)
                }}
              >
                Refrescar
              </button>
            </div>

            <div className="tp-panel" style={{ padding: 12 }}>
              <h4 style={{ marginTop: 0 }}>Proyectos asignados</h4>
              {projects.length === 0 ? (
                <div className="tp-muted">No hay proyectos</div>
              ) : (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {projects.map((p) => (
                    <label key={p.id} className="tp-card" style={{ padding: 10, display: 'flex', gap: 10 }}>
                      <input
                        type="checkbox"
                        checked={assignedProjectIds.has(p.id)}
                        onChange={(e) => {
                          setAssignedProjectIds((prev) => {
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

              <div style={{ marginTop: 10 }}>
                <button className="tp-btn--primary" onClick={saveAssignments} disabled={!selectedUserId}>
                  Guardar asignaciones
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
