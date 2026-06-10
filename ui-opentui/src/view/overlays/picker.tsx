/**
 * Picker — the generic fuzzy picker overlay (spec §2b; Epic 7 model picker v2).
 * Powers /model and /skills: one query line filters live across label AND
 * group AND extra haystacks (provider slug / lab name — `son4` finds
 * claude-sonnet-4, `oai` finds openai models); results render GROUPED with
 * non-selectable headers, and ↑↓ traverse the flat item order seamlessly
 * ACROSS group boundaries. Enter picks, Esc/Ctrl+C closes (keymap layer),
 * typing/backspace edits the query (maskedPrompt's own-buffer pattern — no
 * focused `<input>` feedback loops). Replaces the composer while open.
 *
 * Everything heavy is memoized off (query, items): score → group → window, so
 * keystrokes re-score at most once and unrelated store updates don't.
 */
import type { BoxRenderable } from '@opentui/core'
import { useKeyboard } from '@opentui/solid'
import { createEffect, createMemo, createSignal, For, on, onMount, Show } from 'solid-js'

import { buildPickerRows, fuzzyFilter, visibleRows, type FuzzyField } from '../../logic/fuzzy.ts'
import type { PickerItem } from '../../logic/store.ts'
import { useCloseLayer } from '../keymap.tsx'
import { useTheme } from '../theme.tsx'

/** Max visible rows (headers + items) before the window scrolls. */
const MAX_ROWS = 12

/** The fuzzy haystacks of a row: label ×2 (opencode's title weighting), then
 *  group (lab name), description and any extra haystacks (provider slug). */
function fieldsOf(item: PickerItem): FuzzyField[] {
  const fields: FuzzyField[] = [{ text: item.label, weight: 2 }]
  if (item.group) fields.push({ text: item.group })
  if (item.description) fields.push({ text: item.description })
  for (const h of item.haystacks ?? []) fields.push({ text: h })
  return fields
}

export function Picker(props: {
  title: string
  items: PickerItem[]
  onPick: (value: string) => void
  onClose: () => void
}) {
  const theme = useTheme()
  let rootRef: BoxRenderable | undefined
  // Esc/Ctrl+C close via the native keymap; the root box is focused on mount so
  // the focus-within layer is active (the list/query are not focusable).
  onMount(() => rootRef?.focus())
  useCloseLayer(
    () => rootRef,
    () => props.onClose()
  )

  const [query, setQuery] = createSignal('')
  // score → group → window, all memoized: typing re-scores once; nothing else does.
  const filtered = createMemo(() => fuzzyFilter(query(), props.items, fieldsOf))
  const grouped = createMemo(() => buildPickerRows(filtered(), it => it.group))

  // Start on the current (✓) item; reset to the top match whenever the filter changes.
  const [sel, setSel] = createSignal(
    Math.max(
      0,
      grouped().flat.findIndex(it => it.current)
    )
  )
  createEffect(on(filtered, () => setSel(0), { defer: true }))

  const win = createMemo(() => visibleRows(grouped().rows, sel(), MAX_ROWS))

  const pick = (item: PickerItem | undefined) => {
    if (item) props.onPick(item.value)
  }

  useKeyboard(key => {
    // Esc/Ctrl+C also close via the keymap layer above; handling them here too
    // keeps close working even when focus never landed (maskedPrompt pattern).
    if (key.name === 'escape' || (key.ctrl && key.name === 'c')) return props.onClose()
    const count = grouped().flat.length
    if (key.name === 'return') return pick(grouped().flat[sel()])
    if (key.name === 'up' || (key.ctrl && key.name === 'p')) {
      if (count) setSel(s => (s - 1 + count) % count)
      return
    }
    if (key.name === 'down' || (key.ctrl && key.name === 'n')) {
      if (count) setSel(s => (s + 1) % count)
      return
    }
    if (key.name === 'backspace') return setQuery(q => q.slice(0, -1))
    // printable → refine the query
    const ch = key.sequence
    if (ch.length === 1 && !key.ctrl && !key.meta && ch >= ' ') setQuery(q => q + ch)
  })

  return (
    <box
      ref={el => (rootRef = el)}
      focusable
      style={{ borderColor: theme().color.border, flexDirection: 'column', flexShrink: 0, marginTop: 1, padding: 1 }}
      border
    >
      <box style={{ flexDirection: 'row' }}>
        <text fg={theme().color.accent}>
          <b>{props.title}</b>
        </text>
        <text fg={theme().color.label}>{'  '}</text>
        <text fg={theme().color.prompt}>{'> '}</text>
        <text fg={theme().color.text}>{query()}</text>
        <text fg={theme().color.accent}>▍</text>
        <Show when={!query()}>
          <text fg={theme().color.muted}>type to filter</text>
        </Show>
      </box>
      <Show when={win().above > 0}>
        <text fg={theme().color.muted}>{`  ↑ ${win().above} more`}</text>
      </Show>
      <For each={win().rows}>
        {row =>
          row.kind === 'header' ? (
            <text fg={theme().color.label}>
              <b>{row.label}</b>
            </text>
          ) : (
            <text
              bg={row.index === sel() ? theme().color.selectionBg : 'transparent'}
              onMouseDown={() => pick(row.item)}
            >
              <span style={{ fg: row.index === sel() ? theme().color.text : theme().color.muted }}>
                {row.index === sel() ? '› ' : '  '}
              </span>
              <span style={{ fg: theme().color.text }}>{row.item.label}</span>
              <Show when={row.item.current}>
                <span style={{ fg: theme().color.ok }}> ✓</span>
              </Show>
              <Show when={row.item.description}>
                <span style={{ fg: theme().color.muted }}> {row.item.description}</span>
              </Show>
            </text>
          )
        }
      </For>
      <Show when={grouped().flat.length === 0}>
        <text fg={theme().color.muted}> (no matches)</text>
      </Show>
      <Show when={win().below > 0}>
        <text fg={theme().color.muted}>{`  ↓ ${win().below} more`}</text>
      </Show>
      <text fg={theme().color.muted}>↑↓ move · Enter choose · Esc cancel · type to filter</text>
    </box>
  )
}
