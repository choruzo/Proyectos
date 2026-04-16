import { useEffect, useId, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import { createPortal } from 'react-dom'

export type TpOption = {
  value: string
  label: string
  disabled?: boolean
}

export function TpSelect({
  value,
  options,
  onChange,
  ariaLabel,
  width,
}: {
  value: string
  options: TpOption[]
  onChange: (value: string) => void
  ariaLabel: string
  width?: number | string
}) {
  const listboxId = useId()
  const [open, setOpen] = useState(false)
  const [highlightedIndex, setHighlightedIndex] = useState<number>(-1)
  const [pos, setPos] = useState<{ left: number; top: number; width: number } | null>(null)

  const buttonRef = useRef<HTMLButtonElement | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)

  const selectedIndex = useMemo(() => options.findIndex((o) => o.value === value), [options, value])
  const selected = selectedIndex >= 0 ? options[selectedIndex] : null

  function close() {
    setOpen(false)
    setHighlightedIndex(-1)
  }

  function openMenu() {
    const btn = buttonRef.current
    if (!btn) return

    const rect = btn.getBoundingClientRect()
    const w = rect.width
    const maxH = 280

    let top = rect.bottom + 6
    const wouldOverflow = top + maxH > window.innerHeight - 8
    if (wouldOverflow) top = Math.max(8, rect.top - maxH - 6)

    setPos({ left: rect.left, top, width: w })
    setOpen(true)
    setHighlightedIndex(selectedIndex >= 0 ? selectedIndex : 0)
  }

  useEffect(() => {
    if (!open) return

    const onDocMouseDown = (e: MouseEvent) => {
      const t = e.target as Node
      if (buttonRef.current && buttonRef.current.contains(t)) return
      if (listRef.current && listRef.current.contains(t)) return
      close()
    }

    const onReposition = () => {
      const btn = buttonRef.current
      if (!btn) return
      const rect = btn.getBoundingClientRect()
      const maxH = 280
      let top = rect.bottom + 6
      const wouldOverflow = top + maxH > window.innerHeight - 8
      if (wouldOverflow) top = Math.max(8, rect.top - maxH - 6)
      setPos((p) => (p ? { ...p, left: rect.left, top, width: rect.width } : null))
    }

    document.addEventListener('mousedown', onDocMouseDown)
    window.addEventListener('resize', onReposition)
    window.addEventListener('scroll', onReposition, true)

    return () => {
      document.removeEventListener('mousedown', onDocMouseDown)
      window.removeEventListener('resize', onReposition)
      window.removeEventListener('scroll', onReposition, true)
    }
  }, [open])

  function move(delta: number) {
    if (!options.length) return

    let i = highlightedIndex
    if (i < 0) i = 0

    for (let step = 0; step < options.length; step++) {
      i = (i + delta + options.length) % options.length
      if (!options[i].disabled) {
        setHighlightedIndex(i)
        return
      }
    }
  }

  function commit(i: number) {
    const opt = options[i]
    if (!opt || opt.disabled) return
    onChange(opt.value)
    close()
    buttonRef.current?.focus()
  }

  function onKeyDown(e: ReactKeyboardEvent) {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (!open) return openMenu()
      move(+1)
      return
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      if (!open) return openMenu()
      move(-1)
      return
    }
    if (e.key === 'Home') {
      e.preventDefault()
      if (!open) openMenu()
      setHighlightedIndex(0)
      return
    }
    if (e.key === 'End') {
      e.preventDefault()
      if (!open) openMenu()
      setHighlightedIndex(options.length - 1)
      return
    }
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      if (!open) return openMenu()
      if (highlightedIndex >= 0) commit(highlightedIndex)
      return
    }
    if (e.key === 'Escape') {
      e.preventDefault()
      close()
    }
  }

  const activeDesc =
    open && highlightedIndex >= 0 ? `${listboxId}-opt-${highlightedIndex}` : undefined

  return (
    <span className="tp-select" style={{ width: width ?? undefined }}>
      <button
        ref={buttonRef}
        type="button"
        className="tp-selectButton"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listboxId}
        aria-activedescendant={activeDesc}
        onClick={() => (open ? close() : openMenu())}
        onKeyDown={onKeyDown}
      >
        <span className={selected ? 'tp-selectValue' : 'tp-selectValue tp-muted'}>
          {selected ? selected.label : '—'}
        </span>
        <span className="tp-selectCaret" aria-hidden="true">
          ▾
        </span>
      </button>

      {open && pos
        ? createPortal(
            <div
              ref={listRef}
              className="tp-selectPopover"
              style={{ left: pos.left, top: pos.top, width: pos.width }}
              role="listbox"
              id={listboxId}
              aria-label={ariaLabel}
              onKeyDown={onKeyDown}
            >
              {options.map((o, i) => {
                const isSelected = o.value === value
                const isActive = i === highlightedIndex
                return (
                  <div
                    key={`${o.value}-${i}`}
                    id={`${listboxId}-opt-${i}`}
                    className={
                      'tp-selectOption' +
                      (isSelected ? ' is-selected' : '') +
                      (isActive ? ' is-active' : '') +
                      (o.disabled ? ' is-disabled' : '')
                    }
                    role="option"
                    aria-selected={isSelected}
                    onMouseEnter={() => !o.disabled && setHighlightedIndex(i)}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => commit(i)}
                  >
                    <span className="tp-selectOptionLabel">{o.label}</span>
                    {isSelected ? <span className="tp-selectCheck">✓</span> : null}
                  </div>
                )
              })}
            </div>,
            document.body,
          )
        : null}
    </span>
  )
}
