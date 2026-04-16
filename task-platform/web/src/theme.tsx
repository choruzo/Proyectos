import { createContext, ReactNode, useContext, useEffect, useMemo, useState } from 'react'

export type Theme = 'light' | 'dark'

type ThemeCtx = {
  theme: Theme
  setTheme: (t: Theme) => void
  toggleTheme: () => void
}

const ThemeContext = createContext<ThemeCtx | null>(null)

const STORAGE_KEY = 'tp-theme'

function getSystemTheme(): Theme {
  if (typeof window === 'undefined') return 'light'
  return window.matchMedia?.('(prefers-color-scheme: dark)')?.matches ? 'dark' : 'light'
}

function getInitialTheme(): { theme: Theme; explicit: boolean } {
  if (typeof window === 'undefined') return { theme: 'light', explicit: false }

  const saved = window.localStorage.getItem(STORAGE_KEY)
  if (saved === 'light' || saved === 'dark') return { theme: saved, explicit: true }
  return { theme: getSystemTheme(), explicit: false }
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const init = useMemo(() => getInitialTheme(), [])
  const [theme, setTheme] = useState<Theme>(init.theme)
  const [explicit, setExplicit] = useState<boolean>(init.explicit)

  // Apply theme to the document
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    if (explicit) window.localStorage.setItem(STORAGE_KEY, theme)
    else window.localStorage.removeItem(STORAGE_KEY)
  }, [theme, explicit])

  // If user never picked explicitly, track OS changes
  useEffect(() => {
    if (explicit) return
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)')
    if (!mq) return

    const onChange = () => setTheme(mq.matches ? 'dark' : 'light')

    mq.addEventListener?.('change', onChange)
    return () => mq.removeEventListener?.('change', onChange)
  }, [explicit])

  const value = useMemo<ThemeCtx>(() => {
    return {
      theme,
      setTheme: (t) => {
        setExplicit(true)
        setTheme(t)
      },
      toggleTheme: () => {
        setExplicit(true)
        setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
      },
    }
  }, [theme])

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme(): ThemeCtx {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider')
  return ctx
}
