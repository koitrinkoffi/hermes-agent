/**
 * fuzzy.ts — pure fuzzy filtering + grouped presentation for picker overlays
 * (Epic 7 model picker v2). No deps: a ~subsequence scorer in the spirit of
 * opencode's fuzzysort usage (multi-key scoring, title weighted 2×, grouped
 * headers) at the scale we need (≤ a few hundred catalog rows).
 *
 * Scoring model (per term, per field): a case-insensitive subsequence match
 * where consecutive runs, string-prefix and word-boundary starts score high and
 * gaps/late starts are penalized — so for `son`: `sonnet` (prefix) >
 * `claude-sonnet` (boundary) > `meson` (scattered). A query is split on
 * whitespace; EVERY term must match at least one field (best field wins, its
 * weight applied), so `anthropic son` works across provider+model fields.
 */

/** One searchable field of an item (e.g. model id ×2, provider slug, lab name). */
export interface FuzzyField {
  text: string
  /** Score multiplier (default 1). The primary label is conventionally 2. */
  weight?: number
}

/** Word-boundary characters inside catalog-ish ids/names. */
const BOUNDARY = new Set([' ', '-', '_', '.', '/', ':', '@', '(', ')'])

/** Cap on alternative start positions tried for the first term char. */
const MAX_STARTS = 8

/** Greedy subsequence score from a fixed start index; null when it can't match. */
function scoreFrom(term: string, hay: string, start: number): number | null {
  let score = 0
  let prev = -1
  let from = start
  for (let qi = 0; qi < term.length; qi++) {
    const idx = hay.indexOf(term.charAt(qi), from)
    if (idx === -1) return null
    let charScore = 1
    if (prev !== -1 && idx === prev + 1) charScore += 3 // consecutive run
    if (idx === 0)
      charScore += 6 // string prefix
    else if (BOUNDARY.has(hay.charAt(idx - 1))) charScore += 4 // word boundary
    if (prev !== -1 && idx > prev + 1) charScore -= Math.min(idx - prev - 1, 3) // gap
    if (prev === -1) charScore -= Math.min(idx, 4) // late start
    score += charScore
    prev = idx
    from = idx + 1
  }
  return score
}

/**
 * Score one term against one text. Null = the term is not a subsequence.
 * Greedy from the first occurrence is order-sensitive (`son` in `saturn-sonnet`
 * must anchor at the second `s`), so try each occurrence of the first term char
 * (capped) and keep the best.
 */
export function scoreTerm(term: string, text: string): number | null {
  const needle = term.toLowerCase()
  if (!needle) return 0
  const hay = text.toLowerCase()
  let best: number | null = null
  let start = hay.indexOf(needle.charAt(0))
  for (let tries = 0; start !== -1 && tries < MAX_STARTS; tries++) {
    const s = scoreFrom(needle, hay, start)
    if (s !== null && (best === null || s > best)) best = s
    start = hay.indexOf(needle.charAt(0), start + 1)
  }
  return best
}

/**
 * Score a whitespace-split query against an item's fields. Every term must
 * match at least one field; each term contributes its best weighted field
 * score. Empty/blank query scores 0 (matches everything — catalog order).
 */
export function scoreFields(query: string, fields: readonly FuzzyField[]): number | null {
  const terms = query.trim().split(/\s+/).filter(Boolean)
  if (!terms.length) return 0
  let total = 0
  for (const term of terms) {
    let best: number | null = null
    for (const field of fields) {
      const s = scoreTerm(term, field.text)
      if (s === null) continue
      const weighted = s * (field.weight ?? 1)
      if (best === null || weighted > best) best = weighted
    }
    if (best === null) return null
    total += best
  }
  return total
}

/**
 * Filter + rank items by query. Empty query → the items in catalog order;
 * otherwise matches sorted by score (descending), ties keeping catalog order.
 */
export function fuzzyFilter<T>(query: string, items: readonly T[], fieldsOf: (item: T) => FuzzyField[]): T[] {
  if (!query.trim()) return [...items]
  const scored: Array<{ item: T; score: number; at: number }> = []
  for (let i = 0; i < items.length; i++) {
    const item = items[i] as T
    const score = scoreFields(query, fieldsOf(item))
    if (score !== null) scored.push({ at: i, item, score })
  }
  scored.sort((a, b) => b.score - a.score || a.at - b.at)
  return scored.map(s => s.item)
}

/** A render row of a grouped picker: a non-selectable group header or an item.
 *  `index` is the item's position in the flat ARROW-TRAVERSAL order. */
export type PickerRow<T> = { kind: 'header'; label: string } | { kind: 'item'; item: T; index: number }

/**
 * Group items for display (group order = first appearance, so a score-sorted
 * input puts the best group first). Returns the header+item render rows and
 * the flat selectable list in traversal order — arrows walk `flat` and thus
 * cross group boundaries seamlessly; headers are never selectable. Items
 * without a group render headerless (e.g. the skills picker).
 */
export function buildPickerRows<T>(
  items: readonly T[],
  groupOf: (item: T) => string | undefined
): { rows: PickerRow<T>[]; flat: T[] } {
  const order: string[] = []
  const buckets = new Map<string, T[]>()
  for (const item of items) {
    const group = groupOf(item) ?? ''
    let bucket = buckets.get(group)
    if (!bucket) {
      bucket = []
      buckets.set(group, bucket)
      order.push(group)
    }
    bucket.push(item)
  }
  const rows: PickerRow<T>[] = []
  const flat: T[] = []
  for (const group of order) {
    if (group) rows.push({ kind: 'header', label: group })
    for (const item of buckets.get(group) ?? []) {
      rows.push({ index: flat.length, item, kind: 'item' })
      flat.push(item)
    }
  }
  return { flat, rows }
}

/**
 * Slice rows to a visible window of at most `cap` rows that keeps the selected
 * item in view (centered when possible). `above`/`below` are the hidden row
 * counts for the ↑/↓ "more" indicators.
 */
export function visibleRows<T>(
  rows: readonly PickerRow<T>[],
  selected: number,
  cap: number
): { rows: PickerRow<T>[]; above: number; below: number } {
  if (rows.length <= cap) return { above: 0, below: 0, rows: [...rows] }
  const selRow = rows.findIndex(r => r.kind === 'item' && r.index === selected)
  const anchor = selRow === -1 ? 0 : selRow
  const start = Math.max(0, Math.min(anchor - Math.floor(cap / 2), rows.length - cap))
  return { above: start, below: rows.length - (start + cap), rows: rows.slice(start, start + cap) }
}
