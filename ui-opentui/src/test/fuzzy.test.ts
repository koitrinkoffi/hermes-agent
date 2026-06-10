/**
 * fuzzy.ts tests (Epic 7) — the pure scorer + filter + grouped-rows helpers
 * behind the model picker v2: subsequence matching, ranking (prefix >
 * word-boundary > scattered), multi-field (provider/model/lab), empty query =
 * catalog order, no-match = empty, header rows non-selectable, and the
 * flat arrow-traversal order across groups.
 */
import { describe, expect, test } from 'vitest'

import { buildPickerRows, fuzzyFilter, scoreFields, scoreTerm, visibleRows, type FuzzyField } from '../logic/fuzzy.ts'

describe('scoreTerm — subsequence matching', () => {
  test('matches subsequences (case-insensitive), null when not a subsequence', () => {
    expect(scoreTerm('son', 'claude-sonnet-4')).not.toBeNull()
    expect(scoreTerm('son4', 'claude-sonnet-4')).not.toBeNull() // the complaint's example
    expect(scoreTerm('SON', 'claude-sonnet-4')).not.toBeNull()
    expect(scoreTerm('xyz', 'claude-sonnet-4')).toBeNull()
    expect(scoreTerm('sonn5', 'claude-sonnet-4')).toBeNull() // 5 not present after sonn
    expect(scoreTerm('', 'anything')).toBe(0) // empty term matches everything
  })

  test('ranking: prefix > word-boundary > scattered', () => {
    const prefix = scoreTerm('son', 'sonnet')!
    const boundary = scoreTerm('son', 'claude-sonnet')!
    const scattered = scoreTerm('son', 'meson')!
    expect(prefix).toBeGreaterThan(boundary)
    expect(boundary).toBeGreaterThan(scattered)
  })

  test('anchors at the BEST occurrence, not greedily at the first', () => {
    // greedy-from-first-char would match s@0 then o/n far away; the boundary
    // anchor at the second `s` (start of "sonnet") must win.
    expect(scoreTerm('son', 'saturn-sonnet')!).toBeGreaterThanOrEqual(scoreTerm('son', 'claude-sonnet')!)
  })
})

describe('scoreFields — multi-field, multi-term', () => {
  const fields: FuzzyField[] = [
    { text: 'claude-sonnet-4', weight: 2 }, // model id (label ×2)
    { text: 'anthropic' }, // provider slug
    { text: 'Anthropic' } // lab/display name
  ]

  test('a term may match ANY field (provider/model/lab)', () => {
    expect(scoreFields('son4', fields)).not.toBeNull() // via the model id
    expect(scoreFields('anthro', fields)).not.toBeNull() // via the provider
    expect(scoreFields('nope', fields)).toBeNull()
  })

  test('every whitespace term must match some field (anthropic son works)', () => {
    expect(scoreFields('anthropic son', fields)).not.toBeNull()
    expect(scoreFields('anthropic zzz', fields)).toBeNull()
  })

  test('label matches outrank same-quality group matches (weight 2×)', () => {
    const labelHit = scoreFields('claude', fields)!
    const providerHit = scoreFields('claude', [{ text: 'other-model', weight: 2 }, { text: 'claude' }])
    expect(providerHit).not.toBeNull()
    expect(labelHit).toBeGreaterThan(providerHit!)
  })
})

interface Row {
  label: string
  provider: string
  lab: string
}
const CATALOG: Row[] = [
  { lab: 'Anthropic', label: 'claude-sonnet-4', provider: 'anthropic' },
  { lab: 'Anthropic', label: 'claude-opus-4', provider: 'anthropic' },
  { lab: 'OpenAI', label: 'gpt-5', provider: 'openai' },
  { lab: 'Nous Research', label: 'hermes-4-405b', provider: 'nous' }
]
const rowFields = (r: Row): FuzzyField[] => [{ text: r.label, weight: 2 }, { text: r.provider }, { text: r.lab }]

describe('fuzzyFilter', () => {
  test('empty/blank query → catalog order, untouched', () => {
    expect(fuzzyFilter('', CATALOG, rowFields)).toEqual(CATALOG)
    expect(fuzzyFilter('   ', CATALOG, rowFields)).toEqual(CATALOG)
  })

  test('no match → empty', () => {
    expect(fuzzyFilter('qqqq', CATALOG, rowFields)).toEqual([])
  })

  test('son4 finds claude-sonnet-4 (under anthropic) first', () => {
    expect(fuzzyFilter('son4', CATALOG, rowFields)[0]?.label).toBe('claude-sonnet-4')
  })

  test('oai matches the openai-provider model via the provider field', () => {
    const hits = fuzzyFilter('oai', CATALOG, rowFields)
    expect(hits.map(h => h.label)).toContain('gpt-5')
  })

  test('ties keep catalog order (stable)', () => {
    const hits = fuzzyFilter('claude', CATALOG, rowFields)
    expect(hits.map(h => h.label)).toEqual(['claude-sonnet-4', 'claude-opus-4'])
  })
})

describe('buildPickerRows — grouping + traversal order', () => {
  test('items group by provider with headers; flat traversal crosses groups', () => {
    const { flat, rows } = buildPickerRows(CATALOG, r => r.lab)
    expect(rows.map(r => (r.kind === 'header' ? `# ${r.label}` : r.item.label))).toEqual([
      '# Anthropic',
      'claude-sonnet-4',
      'claude-opus-4',
      '# OpenAI',
      'gpt-5',
      '# Nous Research',
      'hermes-4-405b'
    ])
    // the flat ARROW order is exactly the item rows in render order — so ↓ from
    // claude-opus-4 lands on gpt-5 (next group) and headers are never selectable.
    expect(flat.map(f => f.label)).toEqual(['claude-sonnet-4', 'claude-opus-4', 'gpt-5', 'hermes-4-405b'])
    expect(rows.flatMap(r => (r.kind === 'item' ? [r.index] : []))).toEqual([0, 1, 2, 3])
  })

  test('ungrouped items render headerless (flat list)', () => {
    const { rows } = buildPickerRows(CATALOG, () => undefined)
    expect(rows.every(r => r.kind === 'item')).toBe(true)
  })

  test('group order = first appearance (score-sorted input → best group first)', () => {
    const sorted = [CATALOG[2]!, CATALOG[0]!, CATALOG[1]!] // gpt-5 scored best
    const { rows } = buildPickerRows(sorted, r => r.lab)
    expect(rows[0]).toEqual({ kind: 'header', label: 'OpenAI' })
  })
})

describe('visibleRows — selection-following window', () => {
  const { rows } = buildPickerRows(CATALOG, r => r.lab) // 7 rows

  test('no slicing when everything fits', () => {
    const w = visibleRows(rows, 0, 12)
    expect(w.rows).toHaveLength(7)
    expect(w.above).toBe(0)
    expect(w.below).toBe(0)
  })

  test('keeps the selected item in view and reports hidden counts', () => {
    const w = visibleRows(rows, 3, 4) // last item selected, window of 4
    expect(w.rows.some(r => r.kind === 'item' && r.index === 3)).toBe(true)
    expect(w.above + w.below + w.rows.length).toBe(7)
    expect(w.above).toBeGreaterThan(0)
  })
})
