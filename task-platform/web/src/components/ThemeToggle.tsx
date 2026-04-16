import { useTheme } from '../theme'

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  const checked = theme === 'dark'

  return (
    <button
      type="button"
      className="tp-toggle"
      role="switch"
      aria-checked={checked}
      aria-label="Cambiar tema"
      title={checked ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro'}
      onClick={toggleTheme}
    >
      <span className="tp-toggleTrack" aria-hidden="true">
        <span className="tp-toggleKnob" aria-hidden="true" />
      </span>
      <span className="tp-toggleHint" aria-hidden="true">
        {checked ? 'oscuro' : 'claro'}
      </span>
    </button>
  )
}
