const API_URL = (import.meta as any).env.VITE_API_URL ?? 'http://localhost:8000'

let accessToken: string | null = null

export function setAccessToken(token: string | null) {
  accessToken = token
}

async function refresh(): Promise<string | null> {
  try {
    const res = await fetch(`${API_URL}/auth/refresh`, {
      method: 'POST',
      credentials: 'include',
    })
    if (!res.ok) return null
    const data = await res.json()
    setAccessToken(data.access_token)
    return data.access_token
  } catch {
    return null
  }
}

export async function apiFetch(path: string, init: RequestInit = {}, retry = true): Promise<any> {
  const headers = new Headers(init.headers)
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`)

  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers,
    credentials: 'include',
  })

  if (res.status === 401 && retry) {
    const t = await refresh()
    if (t) return apiFetch(path, init, false)
  }

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }

  const ct = res.headers.get('content-type') || ''
  if (ct.includes('application/json')) return res.json()
  return res.text()
}

export async function login(email: string, password: string): Promise<void> {
  const res = await fetch(`${API_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email, password }),
  })
  if (!res.ok) throw new Error('Login failed')
  const data = await res.json()
  setAccessToken(data.access_token)
}

export async function tryRestoreSession(): Promise<boolean> {
  const t = await refresh()
  return !!t
}

export function getApiUrl(): string {
  return API_URL
}
