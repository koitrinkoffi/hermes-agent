/**
 * Picker overlay tests (Epic 7 model picker v2) — headless frames with a
 * simulated keyboard through the REAL component: provider group headers
 * render, typing filters live (fuzzy, incl. provider-field matches), arrows
 * traverse the flat item order ACROSS group boundaries (headers skipped),
 * Enter picks the highlighted value (cross-provider values carry
 * `--provider`), Esc closes, and a no-match query shows the empty state.
 */
import { describe, expect, test } from 'vitest'

import type { PickerItem } from '../logic/store.ts'
import { DEFAULT_THEME } from '../logic/theme.ts'
import { Picker } from '../view/overlays/picker.tsx'
import { ThemeProvider } from '../view/theme.tsx'
import { renderProbe, type RenderProbe } from './lib/render.ts'

/** A grouped model catalog: current = claude-sonnet-4 under Anthropic. */
const ITEMS: PickerItem[] = [
  {
    current: true,
    group: 'Anthropic',
    haystacks: ['anthropic', 'Anthropic'],
    label: 'claude-sonnet-4',
    value: 'claude-sonnet-4 --provider anthropic'
  },
  {
    group: 'Anthropic',
    haystacks: ['anthropic', 'Anthropic'],
    label: 'claude-opus-4',
    value: 'claude-opus-4 --provider anthropic'
  },
  { group: 'OpenAI', haystacks: ['openai', 'OpenAI'], label: 'gpt-5', value: 'gpt-5 --provider openai' },
  {
    group: 'Nous Research',
    haystacks: ['nous', 'Nous Research'],
    label: 'hermes-4-405b',
    value: 'hermes-4-405b --provider nous'
  }
]

interface Harness {
  probe: RenderProbe
  picked: string[]
  closed: { value: boolean }
}

async function mountPicker(items: PickerItem[] = ITEMS): Promise<Harness> {
  const picked: string[] = []
  const closed = { value: false }
  const probe = await renderProbe(
    () => (
      <ThemeProvider theme={() => DEFAULT_THEME}>
        <Picker title="Switch model" items={items} onPick={v => picked.push(v)} onClose={() => (closed.value = true)} />
      </ThemeProvider>
    ),
    // kitty keyboard so a SIMULATED lone Esc parses (see lib/render.ts)
    { height: 24, kittyKeyboard: true, width: 70 }
  )
  return { closed, picked, probe }
}

describe('Picker — grouped render', () => {
  test('group headers + items render; the current model carries the ✓', async () => {
    const h = await mountPicker()
    try {
      const frame = h.probe.frame()
      expect(frame).toContain('Anthropic')
      expect(frame).toContain('OpenAI')
      expect(frame).toContain('Nous Research')
      expect(frame).toContain('claude-sonnet-4 ✓')
      expect(frame).toContain('hermes-4-405b')
      // initial selection sits on the CURRENT model
      expect(frame).toContain('› claude-sonnet-4')
    } finally {
      h.probe.destroy()
    }
  })
})

describe('Picker — fuzzy filtering', () => {
  test('typing filters live; a provider-field query (oai) keeps only that group', async () => {
    const h = await mountPicker()
    try {
      await h.probe.keys.typeText('oai')
      await h.probe.settle()
      const frame = await h.probe.waitForFrame(f => !f.includes('claude-sonnet-4'))
      expect(frame).toContain('gpt-5')
      expect(frame).toContain('OpenAI') // its group header survives
      expect(frame).not.toContain('hermes-4-405b')
      expect(frame).toContain('› gpt-5') // selection reset to the top match
    } finally {
      h.probe.destroy()
    }
  })

  test('son4 finds claude-sonnet-4; backspace widens the filter again', async () => {
    const h = await mountPicker()
    try {
      await h.probe.keys.typeText('son4')
      await h.probe.settle()
      let frame = await h.probe.waitForFrame(f => !f.includes('gpt-5'))
      expect(frame).toContain('claude-sonnet-4')
      for (let i = 0; i < 4; i++) h.probe.keys.pressBackspace()
      await h.probe.settle()
      frame = await h.probe.waitForFrame(f => f.includes('gpt-5'))
      expect(frame).toContain('hermes-4-405b')
    } finally {
      h.probe.destroy()
    }
  })

  test('a no-match query shows the empty state; Enter is a no-op', async () => {
    const h = await mountPicker()
    try {
      await h.probe.keys.typeText('zzzz')
      await h.probe.settle()
      const frame = await h.probe.waitForFrame(f => f.includes('(no matches)'))
      expect(frame).not.toContain('claude-sonnet-4')
      h.probe.keys.pressEnter()
      await h.probe.settle()
      expect(h.picked).toEqual([])
    } finally {
      h.probe.destroy()
    }
  })
})

describe('Picker — traversal across groups + pick + close', () => {
  test('↓↓ from the current item crosses the Anthropic→OpenAI boundary (header skipped); Enter picks cross-provider', async () => {
    const h = await mountPicker()
    try {
      // start: claude-sonnet-4 (flat 0) → ↓ claude-opus-4 (flat 1) → ↓ gpt-5
      // (flat 2 — FIRST item of the next group; the header row is not a stop)
      h.probe.keys.pressArrow('down')
      h.probe.keys.pressArrow('down')
      await h.probe.settle()
      expect(h.probe.frame()).toContain('› gpt-5')
      h.probe.keys.pressEnter()
      await h.probe.settle()
      expect(h.picked).toEqual(['gpt-5 --provider openai']) // provider+model switch
    } finally {
      h.probe.destroy()
    }
  })

  test('↑ from the top wraps to the LAST item (across all groups)', async () => {
    const h = await mountPicker()
    try {
      // selection starts on the current item (flat 0)
      h.probe.keys.pressArrow('up')
      await h.probe.settle()
      expect(h.probe.frame()).toContain('› hermes-4-405b')
    } finally {
      h.probe.destroy()
    }
  })

  test('Esc closes without picking', async () => {
    const h = await mountPicker()
    try {
      h.probe.keys.pressEscape()
      await h.probe.settle()
      await h.probe.settle()
      expect(h.closed.value).toBe(true)
      expect(h.picked).toEqual([])
    } finally {
      h.probe.destroy()
    }
  })
})
