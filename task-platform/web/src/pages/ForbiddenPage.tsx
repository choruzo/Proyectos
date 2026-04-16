import { Link } from 'react-router-dom'

export function ForbiddenPage() {
  return (
    <div className="tp-shell">
      <div className="tp-panel" style={{ padding: 16, maxWidth: 720 }}>
        <h2 style={{ marginTop: 0 }}>403 · No autorizado</h2>
        <div className="tp-muted">No tienes permisos para acceder a esta sección.</div>
        <div style={{ marginTop: 12 }}>
          <Link to="/projects">Volver a proyectos</Link>
        </div>
      </div>
    </div>
  )
}
