import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from 'react'
import { apiFetch, login as apiLogin, setAccessToken, tryRestoreSession } from './api/client'

export type Me = { id: string; email: string; is_admin: boolean; is_active: boolean }

type AuthCtx = {
  ready: boolean
  me: Me | null
  isAdmin: boolean
  bootstrap: () => Promise<void>
  login: (email: string, password: string) => Promise<Me>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthCtx | null>(null)

async function loadMe(): Promise<Me | null> {
  try {
    return await apiFetch('/auth/me')
  } catch {
    return null
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false)
  const [me, setMe] = useState<Me | null>(null)

  const isAdmin = !!me?.is_admin

  async function bootstrap() {
    try {
      const ok = await tryRestoreSession()
      if (!ok) {
        setMe(null)
        return
      }
      const m = await loadMe()
      setMe(m)
    } finally {
      setReady(true)
    }
  }

  useEffect(() => {
    const onUnauthorized = () => {
      setAccessToken(null)
      setMe(null)
      setReady(true)
    }
    window.addEventListener('tp:unauthorized', onUnauthorized)
    return () => window.removeEventListener('tp:unauthorized', onUnauthorized)
  }, [])

  useEffect(() => {
    bootstrap()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const value = useMemo<AuthCtx>(() => {
    return {
      ready,
      me,
      isAdmin,
      bootstrap,
      login: async (email: string, password: string) => {
        await apiLogin(email, password)
        const m = await loadMe()
        if (!m) throw new Error('Failed to load session')
        setMe(m)
        return m
      },
      logout: async () => {
        try {
          await fetch(`${(import.meta as any).env.VITE_API_URL ?? 'http://localhost:8000'}/auth/logout`, {
            method: 'POST',
            credentials: 'include',
          })
        } finally {
          setAccessToken(null)
          setMe(null)
        }
      },
    }
  }, [ready, me, isAdmin])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthCtx {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
