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
        <div style={{ marginTop: 12, display: 'grid', gap: 8 }}>
          {rows.map((p) => (
            <div key={p.project_id} className="tp-card" style={{ padding: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                <div>
                  <div style={{ fontWeight: 750 }}>{p.project_name}</div>
                  <div className="tp-muted" style={{ fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                    tasks={p.tasks_total} · backlog={p.backlog_total} · adjuntos={p.attachments_total} ({fmtBytes(p.attachments_bytes)}) · audit={p.audit_total} (7d={p.audit_last_7d})
                  </div>
                </div>
              </div>

              <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                <div>
                  <div className="tp-muted" style={{ fontSize: 12 }}>Por estado</div>
                  <pre style={{ margin: 0, fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                    {JSON.stringify(p.tasks_by_status, null, 2)}
                  </pre>
                </div>
                <div>
                  <div className="tp-muted" style={{ fontSize: 12 }}>Por prioridad</div>
                  <pre style={{ margin: 0, fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                    {JSON.stringify(p.tasks_by_priority, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          ))}
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
