import { FormEvent, useEffect, useMemo, useState } from 'react'
import { apiFetch, getApiUrl } from '../api/client'

type Project = { id: string; name: string }
type Release = {
  id: string
  name: string
  project_id: string
  note?: string | null
  start_date?: string | null
  end_date?: string | null
  status: string
}
type Task = {
  id: string
  project_id: string
  release_id?: string | null
  title: string
  description?: string | null
  status: 'Backlog' | 'Todo' | 'Doing' | 'Done'
  priority: 'Low' | 'Medium' | 'High'
  due_date?: string | null
  estimate_hours?: number | null
  tags: string[]
}

type Attachment = {
  id: string
  original_name: string
  mime_type: string
  size_bytes: number
  created_at: string
}

type AuditLog = {
  id: string
  actor_user_id?: string | null
  project_id?: string | null
  task_id?: string | null
  entity_type: string
  entity_id: string
  action: string
  before?: any | null
  after?: any | null
  meta?: any | null
  created_at: string
}

const STATUSES: Task['status'][] = ['Backlog', 'Todo', 'Doing', 'Done']
const PRIORITIES: Task['priority'][] = ['Low', 'Medium', 'High']

type ViewMode = 'kanban' | 'list'

function tagsToText(tags: string[]) {
  return tags.join(', ')
}

function parseTags(text: string): string[] {
  const raw = text
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean)

  const seen = new Set<string>()
  const out: string[] = []
  for (const t of raw) {
    const key = t.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(t)
  }
  return out
}

export function BoardPage({ project }: { project: Project }) {
  const [releases, setReleases] = useState<Release[]>([])
  const [releaseId, setReleaseId] = useState<string>('backlog')
  const [tasks, setTasks] = useState<Task[]>([])
  const [newTitle, setNewTitle] = useState('')
  const [newReleaseName, setNewReleaseName] = useState('')

  const [viewMode, setViewMode] = useState<ViewMode>('kanban')
  const [q, setQ] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [priorityFilter, setPriorityFilter] = useState<string>('')
  const [tagFilter, setTagFilter] = useState<string>('')
  const [tagSuggestions, setTagSuggestions] = useState<string[]>([])

  const [selectedTask, setSelectedTask] = useState<Task | null>(null)
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([])

  const [releaseEditorOpen, setReleaseEditorOpen] = useState(false)

  type SavedView = { id: string; name: string; mode: ViewMode; config: any }
  const [views, setViews] = useState<SavedView[]>([])
  const [viewId, setViewId] = useState<string>('')
  const [saveViewName, setSaveViewName] = useState('')

  async function load() {
    const [rels, sug, vs] = await Promise.all([
      apiFetch(`/projects/${project.id}/releases`),
      apiFetch(`/projects/${project.id}/tags`),
      apiFetch(`/projects/${project.id}/views`),
    ])
    setReleases(rels)
    setTagSuggestions(sug)
    setViews(vs)
    if (viewId && !vs.find((x: any) => x.id === viewId)) setViewId('')

    const qs = new URLSearchParams({ project_id: project.id })
    if (releaseId === 'backlog') qs.set('backlog', 'true')
    else qs.set('release_id', releaseId)

    if (q.trim()) qs.set('q', q.trim())
    if (statusFilter) qs.set('status', statusFilter)
    if (priorityFilter) qs.set('priority', priorityFilter)
    if (tagFilter.trim()) qs.set('tag', tagFilter.trim())

    const ts = await apiFetch(`/tasks?${qs.toString()}`)
    setTasks(ts)
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id, releaseId, q, statusFilter, priorityFilter, tagFilter])

  const byStatus = useMemo(() => {
    const map: Record<string, Task[]> = {}
    for (const s of STATUSES) map[s] = []
    for (const t of tasks) map[t.status].push(t)
    return map
  }, [tasks])

  async function createTask(e: FormEvent) {
    e.preventDefault()
    if (!newTitle.trim()) return

    const payload: any = {
      project_id: project.id,
      release_id: releaseId === 'backlog' ? null : releaseId,
      title: newTitle.trim(),
      status: releaseId === 'backlog' ? 'Backlog' : 'Todo',
      priority: 'Medium',
      tags: [],
    }

    await apiFetch('/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    setNewTitle('')
    await load()
  }

  async function createRelease(e: FormEvent) {
    e.preventDefault()
    if (!newReleaseName.trim()) return

    const rel = await apiFetch(`/projects/${project.id}/releases`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newReleaseName.trim() }),
    })

    setNewReleaseName('')
    setReleaseId(rel.id)
  }

  function currentRelease(): Release | null {
    if (releaseId === 'backlog') return null
    return releases.find((r) => r.id === releaseId) ?? null
  }

  async function saveRelease(patch: any) {
    const rel = currentRelease()
    if (!rel) return
    await apiFetch(`/projects/${project.id}/releases/${rel.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    })
    await load()
  }

  async function deleteRelease() {
    const rel = currentRelease()
    if (!rel) return
    if (!confirm(`¿Borrar versión "${rel.name}"? (Las tareas pasarán a backlog)`)) return
    await apiFetch(`/projects/${project.id}/releases/${rel.id}`, { method: 'DELETE' })
    setReleaseId('backlog')
    await load()
  }

  function applyView(v: any) {
    const cfg = v.config || {}
    setViewMode((v.mode as ViewMode) || 'kanban')
    setReleaseId(cfg.releaseId ?? 'backlog')
    setQ(cfg.q ?? '')
    setStatusFilter(cfg.status ?? '')
    setPriorityFilter(cfg.priority ?? '')
    setTagFilter(cfg.tag ?? '')
  }

  function currentView() {
    return viewId ? views.find((v) => v.id === viewId) ?? null : null
  }

  function currentViewPayload() {
    return {
      mode: viewMode,
      config: { releaseId, q, status: statusFilter, priority: priorityFilter, tag: tagFilter },
    }
  }

  async function createView() {
    if (!saveViewName.trim()) return
    const payload = {
      name: saveViewName.trim(),
      ...currentViewPayload(),
    }
    await apiFetch(`/projects/${project.id}/views`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    setSaveViewName('')
    await load()
  }

  async function updateView() {
    if (!viewId) return
    await apiFetch(`/projects/${project.id}/views/${viewId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(currentViewPayload()),
    })
    await load()
  }

  async function renameView() {
    if (!viewId) return
    if (!saveViewName.trim()) return
    await apiFetch(`/projects/${project.id}/views/${viewId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: saveViewName.trim() }),
    })
    setSaveViewName('')
    await load()
  }

  async function deleteView() {
    if (!viewId) return
    const v = currentView()
    if (!confirm(`¿Borrar esta vista guardada${v ? ` ("${v.name}")` : ''}?`)) return
    await apiFetch(`/projects/${project.id}/views/${viewId}`, { method: 'DELETE' })
    setViewId('')
    await load()
  }

  async function moveTask(taskId: string, status: Task['status']) {
    await apiFetch(`/tasks/${taskId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    })
    await load()
  }

  async function openTask(t: Task) {
    setSelectedTask(t)
    const [list, audit] = await Promise.all([apiFetch(`/tasks/${t.id}/attachments`), apiFetch(`/tasks/${t.id}/audit`)])
    setAttachments(list)
    setAuditLogs(audit)
  }

  async function uploadAttachment(file: File) {
    if (!selectedTask) return
    const form = new FormData()
    form.append('file', file)
    await apiFetch(`/tasks/${selectedTask.id}/attachments`, { method: 'POST', body: form })
    const [list, audit] = await Promise.all([
      apiFetch(`/tasks/${selectedTask.id}/attachments`),
      apiFetch(`/tasks/${selectedTask.id}/audit`),
    ])
    setAttachments(list)
    setAuditLogs(audit)
  }

  async function deleteAttachment(attachmentId: string) {
    if (!selectedTask) return
    await apiFetch(`/attachments/${attachmentId}`, { method: 'DELETE' })
    const [list, audit] = await Promise.all([
      apiFetch(`/tasks/${selectedTask.id}/attachments`),
      apiFetch(`/tasks/${selectedTask.id}/audit`),
    ])
    setAttachments(list)
    setAuditLogs(audit)
  }

  async function saveTask(taskId: string, patch: any) {
    await apiFetch(`/tasks/${taskId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    })
    await load()
  }

  async function deleteTask(taskId: string) {
    if (!confirm('¿Borrar esta tarea? (Se eliminarán también sus adjuntos)')) return
    await apiFetch(`/tasks/${taskId}`, { method: 'DELETE' })
    setSelectedTask(null)
    setAttachments([])
    setAuditLogs([])
    await load()
  }

  return (
    <div className="tp-panel" style={{ padding: 12 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
        <h2 style={{ margin: 0 }}>{project.name}</h2>
        <div className="tp-muted" style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
          Kanban · Lista · Presets
        </div>
      </div>

      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, marginBottom: 12, flexWrap: 'wrap' }}>
        <label>Vista</label>
        <select value={viewMode} onChange={(e) => setViewMode(e.target.value as ViewMode)}>
          <option value="kanban">Kanban</option>
          <option value="list">Lista</option>
        </select>

        <label>Preset</label>
        <select
          value={viewId}
          onChange={(e) => {
            const id = e.target.value
            setViewId(id)
            const v = views.find((x) => x.id === id)
            if (v) applyView(v)
          }}
        >
          <option value="">(sin preset)</option>
          {views.map((v) => (
            <option key={v.id} value={v.id}>
              {v.name}
            </option>
          ))}
        </select>
        <input value={saveViewName} onChange={(e) => setSaveViewName(e.target.value)} placeholder="Nombre preset…" />
        <button type="button" className="tp-btn--primary" onClick={createView}>
          Guardar nuevo
        </button>
        <button type="button" onClick={updateView} disabled={!viewId}>
          Actualizar
        </button>
        <button type="button" onClick={renameView} disabled={!viewId || !saveViewName.trim()}>
          Renombrar
        </button>
        <button type="button" className="tp-btn--danger" onClick={deleteView} disabled={!viewId}>
          Borrar
        </button>

        <label>Versión</label>
        <select value={releaseId} onChange={(e) => setReleaseId(e.target.value)}>
          <option value="backlog">Backlog (sin versión)</option>
          {releases.map((r) => (
            <option key={r.id} value={r.id}>
              {r.name}
            </option>
          ))}
        </select>
        <button type="button" onClick={() => setReleaseEditorOpen(true)} disabled={releaseId === 'backlog'}>
          Editar versión
        </button>

        <form onSubmit={createRelease} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input
            value={newReleaseName}
            onChange={(e) => setNewReleaseName(e.target.value)}
            placeholder="Nueva versión"
          />
          <button type="submit" className="tp-btn--primary">Crear versión</button>
        </form>

        <form onSubmit={createTask} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input value={newTitle} onChange={(e) => setNewTitle(e.target.value)} placeholder="Nueva tarea" />
          <button type="submit" className="tp-btn--primary">Crear</button>
        </form>

        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Buscar…" />

        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
          <option value="">(Estado: todos)</option>
          {STATUSES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        <select value={priorityFilter} onChange={(e) => setPriorityFilter(e.target.value)}>
          <option value="">(Prioridad: todas)</option>
          {PRIORITIES.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>

        <input value={tagFilter} onChange={(e) => setTagFilter(e.target.value)} placeholder="Tag…" />

        <button type="button" className="tp-btn--ghost" onClick={load}>
          Refrescar
        </button>
      </div>

      {viewMode === 'kanban' ? (
        <div className="tp-kanbanGrid">
          {STATUSES.map((s) => (
            <KanbanColumn
              key={s}
              title={s}
              tasks={byStatus[s]}
              onDropTask={(id) => moveTask(id, s)}
              onOpenTask={openTask}
            />
          ))}
        </div>
      ) : (
        <TaskList tasks={tasks} onOpenTask={openTask} />
      )}

      {releaseEditorOpen && currentRelease() && (
        <ReleaseModal
          release={currentRelease()!}
          onClose={() => setReleaseEditorOpen(false)}
          onSave={async (patch) => {
            await saveRelease(patch)
            setReleaseEditorOpen(false)
          }}
          onDelete={async () => {
            await deleteRelease()
            setReleaseEditorOpen(false)
          }}
        />
      )}

      {selectedTask && (
        <TaskModal
          task={selectedTask}
          releases={releases}
          tagSuggestions={tagSuggestions}
          attachments={attachments}
          auditLogs={auditLogs}
          onUploadAttachment={uploadAttachment}
          onDeleteAttachment={deleteAttachment}
          onSave={async (patch) => {
            await saveTask(selectedTask.id, patch)
            setSelectedTask(null)
          }}
          onDelete={async () => deleteTask(selectedTask.id)}
          onClose={() => setSelectedTask(null)}
        />
      )}
    </div>
  )
}

function TaskList({ tasks, onOpenTask }: { tasks: Task[]; onOpenTask: (t: Task) => void }) {
  return (
    <div className="tp-tableWrap">
      <table className="tp-table">
      <thead>
        <tr>
          <th>Título</th>
          <th>Estado</th>
          <th>Prioridad</th>
          <th>Vence</th>
          <th>Est.</th>
          <th>Tags</th>
        </tr>
      </thead>
      <tbody>
        {tasks.map((t) => (
          <tr
            key={t.id}
            onClick={() => onOpenTask(t)}
            style={{ cursor: 'pointer' }}
            title="Abrir tarea"
          >
            <td>{t.title}</td>
            <td>{t.status}</td>
            <td>{t.priority}</td>
            <td>{t.due_date ?? ''}</td>
            <td>{t.estimate_hours ?? ''}</td>
            <td>{t.tags?.join(', ')}</td>
          </tr>
        ))}
      </tbody>
      </table>
    </div>
  )
}

function ReleaseModal({
  release,
  onSave,
  onDelete,
  onClose,
}: {
  release: Release
  onSave: (patch: any) => void
  onDelete: () => void
  onClose: () => void
}) {
  const [name, setName] = useState(release.name)
  const [note, setNote] = useState(release.note ?? '')
  const [startDate, setStartDate] = useState(release.start_date ?? '')
  const [endDate, setEndDate] = useState(release.end_date ?? '')
  const [status, setStatus] = useState(release.status ?? 'planned')

  useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
    }
  }, [])

  return (
    <div className="tp-modalOverlay" onClick={onClose}>
      <div className="tp-panel tp-modal" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
          <h3 style={{ margin: 0 }}>Editar versión</h3>
          <button type="button" className="tp-btn--ghost" onClick={onClose}>
            Cerrar
          </button>
        </div>

        <div className="tp-formGrid" style={{ marginTop: 12 }}>
          <div style={{ gridColumn: '1 / -1' }}>
            <label>Nombre</label>
            <input style={{ width: '100%' }} value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div>
            <label>Inicio</label>
            <input style={{ width: '100%' }} type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
          </div>
          <div>
            <label>Fin</label>
            <input style={{ width: '100%' }} type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
          </div>
          <div style={{ gridColumn: '1 / -1' }}>
            <label>Estado</label>
            <select style={{ width: '100%' }} value={status} onChange={(e) => setStatus(e.target.value)}>
              <option value="planned">planned</option>
              <option value="in_progress">in_progress</option>
              <option value="closed">closed</option>
            </select>
          </div>
          <div style={{ gridColumn: '1 / -1' }}>
            <label>Nota</label>
            <textarea style={{ width: '100%', minHeight: 120 }} value={note} onChange={(e) => setNote(e.target.value)} />
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 12, gap: 8, flexWrap: 'wrap' }}>
          <button className="tp-btn--danger" onClick={onDelete}>
            Borrar versión
          </button>
          <button className="tp-btn--primary"
            onClick={() =>
              onSave({
                name: name.trim() || release.name,
                note: note.trim() ? note : null,
                start_date: startDate || null,
                end_date: endDate || null,
                status,
              })
            }
          >
            Guardar
          </button>
        </div>
      </div>
    </div>
  )
}

function TaskModal({
  task,
  releases,
  tagSuggestions,
  attachments,
  auditLogs,
  onUploadAttachment,
  onDeleteAttachment,
  onSave,
  onDelete,
  onClose,
}: {
  task: Task
  releases: Release[]
  tagSuggestions: string[]
  attachments: Attachment[]
  auditLogs: AuditLog[]
  onUploadAttachment: (file: File) => void
  onDeleteAttachment: (id: string) => void
  onSave: (patch: any) => void
  onDelete: () => void
  onClose: () => void
}) {
  const [title, setTitle] = useState(task.title)
  const [description, setDescription] = useState(task.description ?? '')
  const [status, setStatus] = useState<Task['status']>(task.status)
  const [priority, setPriority] = useState<Task['priority']>(task.priority)
  const [dueDate, setDueDate] = useState(task.due_date ?? '')
  const [estimate, setEstimate] = useState(task.estimate_hours?.toString() ?? '')
  const [tagsText, setTagsText] = useState(tagsToText(task.tags ?? []))
  const [releaseSel, setReleaseSel] = useState<string>(task.release_id ?? 'backlog')

  useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
    }
  }, [])

  function addTag(name: string) {
    const current = parseTags(tagsText)
    if (current.map((t) => t.toLowerCase()).includes(name.toLowerCase())) return
    setTagsText(tagsToText([...current, name]))
  }

  return (
    <div className="tp-modalOverlay" onClick={onClose}>
      <div className="tp-panel tp-modal" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center' }}>
          <h3 style={{ margin: 0 }}>Tarea</h3>
          <button type="button" className="tp-btn--ghost" onClick={onClose}>
            Cerrar
          </button>
        </div>

        <div className="tp-formGrid" style={{ marginTop: 12 }}>
          <div>
            <label>Título</label>
            <input style={{ width: '100%' }} value={title} onChange={(e) => setTitle(e.target.value)} />
          </div>
          <div>
            <label>Versión</label>
            <select style={{ width: '100%' }} value={releaseSel} onChange={(e) => setReleaseSel(e.target.value)}>
              <option value="backlog">Backlog (sin versión)</option>
              {releases.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label>Estado</label>
            <select style={{ width: '100%' }} value={status} onChange={(e) => setStatus(e.target.value as any)}>
              {STATUSES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Prioridad</label>
            <select style={{ width: '100%' }} value={priority} onChange={(e) => setPriority(e.target.value as any)}>
              {PRIORITIES.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label>Vencimiento</label>
            <input style={{ width: '100%' }} type="date" value={dueDate} onChange={(e) => setDueDate(e.target.value)} />
          </div>
          <div>
            <label>Estimación (horas)</label>
            <input
              style={{ width: '100%' }}
              type="number"
              step="0.25"
              value={estimate}
              onChange={(e) => setEstimate(e.target.value)}
            />
          </div>

          <div style={{ gridColumn: '1 / -1' }}>
            <label>Descripción (Markdown)</label>
            <textarea
              style={{ width: '100%', minHeight: 120 }}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </div>

          <div style={{ gridColumn: '1 / -1' }}>
            <label>Tags (separadas por coma)</label>
            <input style={{ width: '100%' }} value={tagsText} onChange={(e) => setTagsText(e.target.value)} />
            {tagSuggestions.length > 0 && (
              <div style={{ marginTop: 6, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {tagSuggestions.slice(0, 12).map((t) => (
                  <button key={t} type="button" className="tp-btn--ghost" onClick={() => addTag(t)} style={{ fontSize: 12 }}>
                    + {t}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div style={{ gridColumn: '1 / -1' }}>
            <div style={{ marginBottom: 8 }}>
              <input type="file" onChange={(e) => e.target.files && onUploadAttachment(e.target.files[0])} />
            </div>
            <div>
              <h4>Adjuntos</h4>
              {attachments.length === 0 && <div>Sin adjuntos</div>}
              <ul>
                {attachments.map((a) => (
                  <li key={a.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <a href={`${getApiUrl()}/attachments/${a.id}/download`} target="_blank" rel="noreferrer">
                      {a.original_name}
                    </a>
                    <button type="button" onClick={() => onDeleteAttachment(a.id)}>
                      Borrar
                    </button>
                  </li>
                ))}
              </ul>
            </div>

            <div style={{ marginTop: 12 }}>
              <h4>Historial</h4>
              {auditLogs.length === 0 && <div>Sin eventos</div>}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {auditLogs.map((e) => (
                  <details key={e.id} className="tp-card" style={{ padding: 10 }}>
                    <summary style={{ cursor: 'pointer' }}>
                      <span className="tp-kbd">{new Date(e.created_at).toLocaleString()}</span>
                      {' · '}
                      <span className="tp-kbd">{e.action}</span>
                      {e.meta?.changed_fields ? ` · ${e.meta.changed_fields.join(', ')}` : ''}
                    </summary>
                    <pre
                      className="tp-kbd"
                      style={{
                        whiteSpace: 'pre-wrap',
                        marginTop: 8,
                        background: 'var(--surface-2)',
                        border: '1px solid var(--border)',
                        borderRadius: 12,
                        padding: 10,
                        overflow: 'auto',
                      }}
                    >
                      {JSON.stringify({ before: e.before, after: e.after, meta: e.meta }, null, 2)}
                    </pre>
                  </details>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, marginTop: 12, justifyContent: 'space-between', flexWrap: 'wrap' }}>
          <button className="tp-btn--danger" onClick={onDelete}>
            Borrar tarea
          </button>
          <button className="tp-btn--primary"
            onClick={() =>
              onSave({
                title: title.trim() || task.title,
                description: description.trim() ? description : null,
                status,
                priority,
                due_date: dueDate || null,
                estimate_hours: estimate.trim() ? Number(estimate) : null,
                tags: parseTags(tagsText),
                release_id: releaseSel === 'backlog' ? null : releaseSel,
              })
            }
          >
            Guardar
          </button>
        </div>
      </div>
    </div>
  )
}

function KanbanColumn({
  title,
  tasks,
  onDropTask,
  onOpenTask,
}: {
  title: string
  tasks: Task[]
  onDropTask: (taskId: string) => void
  onOpenTask: (t: Task) => void
}) {
  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault()
        const id = e.dataTransfer.getData('text/plain')
        if (id) onDropTask(id)
      }}
      className="tp-card"
      style={{
        borderRadius: 18,
        padding: 10,
        minHeight: 420,
        background: 'var(--surface-2)',
      }}
    >
      <div style={{ fontWeight: 750, marginBottom: 10, fontFamily: 'var(--font-mono)', fontSize: 12, letterSpacing: '0.06em' }}>
        {title}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {tasks.map((t) => (
          <div
            key={t.id}
            draggable
            onDragStart={(e) => e.dataTransfer.setData('text/plain', t.id)}
            onClick={() => onOpenTask(t)}
            className="tp-card"
            style={{
              borderRadius: 14,
              padding: 10,
              cursor: 'pointer',
              background: 'var(--surface)',
            }}
          >
            <div style={{ fontWeight: 650 }}>{t.title}</div>
            <div style={{ fontSize: 12, color: 'var(--muted)', fontFamily: 'var(--font-mono)' }}>
              {t.priority}
              {t.due_date ? ` · vence ${t.due_date}` : ''}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
