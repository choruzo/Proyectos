import { useEffect, useMemo, useState } from 'react'
import { apiFetch } from '../api/client'

type ProjectStats = {
  project_id: string
  project_name: string
  tasks_total: number
  backlog_total: number
  tasks_by_status: Record<string, number>
  tasks_by_priority: Record<string, number>
  attachments_total: number
  attachments_bytes: number
  audit_total: number
  audit_last_7d: number
}

type AdminStatsResponse = {
  generated_at: string
  projects: ProjectStats[]
}

type Segment = { label: string; value: number; color: string; title?: string }

const STATUS_ORDER = ['Backlog', 'Todo', 'Doing', 'Done'] as const
const STATUS_LABEL: Record<string, string> = {
  Backlog: 'Backlog',
  Todo: 'Por hacer',
  Doing: 'En desarrollo',
  Done: 'Hecha',
}

function fmtBytes(n: number) {
  if (!Number.isFinite(n) || n <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

function sumRecord(rec: Record<string, number> | undefined): number {
  if (!rec) return 0
  let t = 0
  for (const k of Object.keys(rec)) t += Number(rec[k] || 0)
  return t
}

function StackedBar({ segments, height = 12 }: { segments: Segment[]; height?: number }) {
  const total = segments.reduce((a, s) => a + (s.value || 0), 0)

  return (
    <div
      style={{
        height,
        borderRadius: 999,
        overflow: 'hidden',
        border: '1px solid var(--border)',
        background: 'var(--surface-2)',
        display: 'flex',
      }}
      aria-label="Gráfico de barras apiladas"
      title={total ? `Total: ${total}` : 'Sin datos'}
    >
      {total === 0 ? (
        <div style={{ width: '100%' }} />
      ) : (
        segments
          .filter((s) => (s.value || 0) > 0)
          .map((s, i, arr) => {
            const pct = (s.value / total) * 100
            const t = s.title ?? `${s.label}: ${s.value} (${pct.toFixed(0)}%)`
            return (
              <div
                key={s.label}
                title={t}
                style={{
                  width: `${pct}%`,
                  background: s.color,
                  borderRight:
                    i === arr.length - 1
                      ? undefined
                      : '1px solid color-mix(in srgb, var(--border) 70%, transparent)',
                }}
              />
            )
          })
      )}
    </div>
  )
}

function Legend({ segments }: { segments: Segment[] }) {
  return (
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
      {segments.map((s) => (
        <div key={s.label} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span
            aria-hidden="true"
            style={{ width: 10, height: 10, borderRadius: 3, background: s.color, border: '1px solid var(--border)' }}
          />
          <span className="tp-muted" style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
            {s.label}: {s.value}
          </span>
        </div>
      ))}
    </div>
  )
}

function statusSegments(p: ProjectStats) {
  const c = p.tasks_by_status || {}

  const backlog = Number(c.Backlog || 0)
  const todo = Number(c.Todo || 0)
  const doing = Number(c.Doing || 0)
  const done = Number(c.Done || 0)

  const grouped: Segment[] = [
    {
      label: 'Pendiente',
      value: backlog + todo,
      color: 'color-mix(in srgb, var(--accent) 78%, var(--surface-3))',
      title: `Pendiente = Backlog(${backlog}) + Por hacer(${todo})`,
    },
    {
      label: 'En desarrollo',
      value: doing,
      color: 'color-mix(in srgb, var(--accent-2) 75%, var(--surface-3))',
    },
    {
      label: 'Hecha',
      value: done,
      color: 'color-mix(in srgb, var(--sage) 80%, var(--surface-3))',
    },
  ]

  const detailed: Segment[] = STATUS_ORDER.map((k) => {
    const value = Number(c[k] || 0)
    const color =
      k === 'Backlog'
        ? 'color-mix(in srgb, var(--muted-2) 90%, var(--surface-3))'
        : k === 'Todo'
          ? 'color-mix(in srgb, var(--accent) 75%, var(--surface-3))'
          : k === 'Doing'
            ? 'color-mix(in srgb, var(--accent-2) 70%, var(--surface-3))'
            : 'color-mix(in srgb, var(--sage) 80%, var(--surface-3))'
    return { label: STATUS_LABEL[k] ?? k, value, color }
  })

  // Include any unexpected status keys at the end
  for (const k of Object.keys(c)) {
    if (STATUS_ORDER.includes(k as any)) continue
    detailed.push({ label: k, value: Number(c[k] || 0), color: 'color-mix(in srgb, var(--border) 60%, var(--surface-3))' })
  }

  return { grouped, detailed }
}

function prioritySegments(p: ProjectStats): Segment[] {
  const c = p.tasks_by_priority || {}
  const order = ['High', 'Medium', 'Low']
  const labels: Record<string, string> = { High: 'Alta', Medium: 'Media', Low: 'Baja' }

  const colors: Record<string, string> = {
    High: 'color-mix(in srgb, var(--danger) 70%, var(--surface-3))',
    Medium: 'color-mix(in srgb, var(--accent) 75%, var(--surface-3))',
    Low: 'color-mix(in srgb, var(--sage) 70%, var(--surface-3))',
  }

  const out: Segment[] = []
  for (const k of order) out.push({ label: labels[k] ?? k, value: Number(c[k] || 0), color: colors[k] ?? 'var(--border)' })

  for (const k of Object.keys(c)) {
    if (order.includes(k)) continue
    out.push({ label: k, value: Number(c[k] || 0), color: 'color-mix(in srgb, var(--border) 60%, var(--surface-3))' })
  }

  return out
}

function aggregateProjects(list: ProjectStats[]): ProjectStats {
  const out: ProjectStats = {
    project_id: 'global',
    project_name: 'Global',
    tasks_total: 0,
    backlog_total: 0,
    tasks_by_status: {},
    tasks_by_priority: {},
    attachments_total: 0,
    attachments_bytes: 0,
    audit_total: 0,
    audit_last_7d: 0,
  }

  for (const p of list) {
    out.tasks_total += Number(p.tasks_total || 0)
    out.backlog_total += Number(p.backlog_total || 0)
    out.attachments_total += Number(p.attachments_total || 0)
    out.attachments_bytes += Number(p.attachments_bytes || 0)
    out.audit_total += Number(p.audit_total || 0)
    out.audit_last_7d += Number(p.audit_last_7d || 0)

    for (const k of Object.keys(p.tasks_by_status || {})) {
      out.tasks_by_status[k] = Number(out.tasks_by_status[k] || 0) + Number(p.tasks_by_status[k] || 0)
    }
    for (const k of Object.keys(p.tasks_by_priority || {})) {
      out.tasks_by_priority[k] = Number(out.tasks_by_priority[k] || 0) + Number(p.tasks_by_priority[k] || 0)
    }
  }

  // If tasks_total is not reliable, derive it
  if (!out.tasks_total) out.tasks_total = sumRecord(out.tasks_by_status)
  return out
}

export function AdminStatsPage() {
  const [data, setData] = useState<AdminStatsResponse | null>(null)
  const [filter, setFilter] = useState('')

  async function load() {
    const d = await apiFetch('/admin/stats')
    setData(d)
  }

  useEffect(() => {
    load()
  }, [])

  const rows = useMemo(() => {
    const q = filter.trim().toLowerCase()
    const list = data?.projects ?? []
    if (!q) return list
    return list.filter((p) => p.project_name.toLowerCase().includes(q))
  }, [data, filter])

  const global = useMemo(() => aggregateProjects(rows), [rows])

  return (
    <div className="tp-panel" style={{ padding: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
        <h2 style={{ marginTop: 0 }}>Estadísticas</h2>
        <button className="tp-btn--ghost" onClick={load}>
          Refrescar
        </button>
      </div>

      <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filtrar por proyecto…" />

      {!data ? (
        <div className="tp-muted" style={{ marginTop: 12 }}>
          Cargando…
        </div>
      ) : (
        <div style={{ marginTop: 12, display: 'grid', gap: 10 }}>
          <div className="tp-card" style={{ padding: 12 }}>
            <div style={{ fontWeight: 750 }}>Global (filtrado)</div>
            <div className="tp-muted" style={{ fontSize: 12, fontFamily: 'var(--font-mono)' }}>
              tasks={global.tasks_total} · backlog={global.backlog_total} · adjuntos={global.attachments_total} ({fmtBytes(global.attachments_bytes)}) · audit={global.audit_total} (7d={global.audit_last_7d})
            </div>
            <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div>
                <div className="tp-muted" style={{ fontSize: 12, marginBottom: 6 }}>Por estado</div>
                <StackedBar segments={statusSegments(global).detailed} />
                <div style={{ marginTop: 8 }}>
                  <Legend segments={statusSegments(global).detailed} />
                </div>
              </div>
              <div>
                <div className="tp-muted" style={{ fontSize: 12, marginBottom: 6 }}>Por prioridad</div>
                <StackedBar segments={prioritySegments(global)} />
                <div style={{ marginTop: 8 }}>
                  <Legend segments={prioritySegments(global)} />
                </div>
              </div>
            </div>
          </div>

          {rows.map((p) => {
            const st = statusSegments(p)
            const pr = prioritySegments(p)
            const pending = (p.tasks_by_status.Backlog || 0) + (p.tasks_by_status.Todo || 0)
            const doing = p.tasks_by_status.Doing || 0
            const done = p.tasks_by_status.Done || 0

            return (
              <div key={p.project_id} className="tp-card" style={{ padding: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                  <div>
                    <div style={{ fontWeight: 750 }}>{p.project_name}</div>
                    <div className="tp-muted" style={{ fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                      tasks={p.tasks_total} · pendiente={pending} · doing={doing} · done={done} · adjuntos={p.attachments_total} ({fmtBytes(p.attachments_bytes)}) · audit={p.audit_total} (7d={p.audit_last_7d})
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div>
                    <div className="tp-muted" style={{ fontSize: 12, marginBottom: 6 }}>Backlog / Por hacer / En desarrollo / Hecha</div>
                    <StackedBar segments={st.detailed} />
                    <div style={{ marginTop: 8 }}>
                      <Legend segments={st.detailed} />
                    </div>
                  </div>

                  <div>
                    <div className="tp-muted" style={{ fontSize: 12, marginBottom: 6 }}>Prioridad</div>
                    <StackedBar segments={pr} />
                    <div style={{ marginTop: 8 }}>
                      <Legend segments={pr} />
                    </div>
                  </div>
                </div>
              </div>
            )
          })}

          {rows.length === 0 ? <div className="tp-muted">Sin resultados</div> : null}
        </div>
      )}

      {data ? (
        <div className="tp-muted" style={{ marginTop: 12, fontFamily: 'var(--font-mono)', fontSize: 12 }}>
          generated_at: {data.generated_at}
        </div>
      ) : null}
    </div>
  )
}
