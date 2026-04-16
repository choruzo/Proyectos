import { FormEvent, useState } from 'react'
import { ThemeToggle } from '../components/ThemeToggle'

export function LoginPage({
  onLoggedIn,
}: {
  onLoggedIn: (email: string, password: string) => Promise<void>
}) {
  const [email, setEmail] = useState('admin@example.com')
  const [password, setPassword] = useState('admin123')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await onLoggedIn(email, password)
    } catch {
      setError('Credenciales incorrectas')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="tp-shell">
      <div className="tp-topbar">
        <div className="tp-brand">
          <h1 style={{ margin: 0 }}>Task Platform</h1>
          <small>Duna Serena</small>
        </div>
        <ThemeToggle />
      </div>

      <div className="tp-panel" style={{ maxWidth: 520, padding: 18 }}>
        <form onSubmit={onSubmit} style={{ display: 'grid', gap: 10 }}>
          <div>
            <label>Email</label>
            <input style={{ width: '100%' }} value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>
          <div>
            <label>Password</label>
            <input
              style={{ width: '100%' }}
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <div style={{ display: 'flex', gap: 10, alignItems: 'center', justifyContent: 'space-between', marginTop: 6 }}>
            <div className="tp-muted" style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
              API · cookie refresh · access in-memory
            </div>
            <button className="tp-btn--primary" disabled={loading}>
              Entrar
            </button>
          </div>
        </form>

        {error && (
          <div style={{ color: 'var(--danger)', marginTop: 12, fontFamily: 'var(--font-mono)' }}>{error}</div>
        )}
      </div>
    </div>
  )
}
