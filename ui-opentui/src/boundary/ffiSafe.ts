/**
 * Node-FFI coordinate safety shim for @opentui/core 0.4.0.
 *
 * Root cause (live crash, ERR_INVALID_ARG_VALUE looping every frame): several
 * OptimizedBuffer methods marshal x/y/width/height as **u32** in the FFI table
 * (zig.ts: `bufferFillRect: ["u32","u32","u32","u32","u32","ptr"]`, same for
 * `bufferDrawText` / `bufferSetCell*` / `bufferDrawChar`), while renderables
 * pass RAW SCREEN COORDINATES — which go NEGATIVE inside a <scrollbox> when an
 * element is partially scrolled above the viewport. Concretely:
 * `LineNumberRenderable.renderSelf` does `buffer.fillRect(this.x + gutterWidth,
 * this.y + i, …)` for diff added/removed line backgrounds, so expanding a tall
 * `<diff showLineNumbers>` pinned to the scrollbox bottom rendered with
 * `this.y < 0` and threw out of `CliRenderer.loop` on EVERY frame (frozen UI,
 * console error spam) until a resize forced a fresh layout.
 *
 * Upstream-on-Bun this never throws: Bun's FFI silently WRAPS negatives to
 * huge u32s and the native side bounds-checks them into a no-op. Node's
 * experimental FFI (node:ffi) instead REJECTS the argument. Other draw entry
 * points (`bufferDrawBox`, `bufferDrawTextBufferView`) already use i32 — which
 * is why ordinary text/boxes scroll fine and only the diff gutter path crashed.
 *
 * Fix at the seam we own: clamp/skip BEFORE the FFI call.
 *  - fillRect: clip the rect to the non-negative quadrant (the native side
 *    already clips right/bottom against the buffer + scissor) and skip empties.
 *  - drawText/setCell/setCellWithAlphaBlending/drawChar: skip when the origin
 *    is negative (Bun-parity: those cells/rows are off-screen anyway).
 *
 * TODO(upstream): file/track an OpenTUI issue to widen these FFI params to i32
 * (or clamp in core) — then this shim can be deleted.
 */
import { OptimizedBuffer, TextBufferView } from '@opentui/core'

let installed = false

/** Patch OptimizedBuffer's u32-coordinate methods to tolerate negative coords. Idempotent. */
export function installFfiCoordSafety(): void {
  if (installed) return
  installed = true

  const proto = OptimizedBuffer.prototype

  // Prototype monkey-patching: extracting the original methods unbound is the
  // point — they're re-invoked with `.call(this, …)` on the correct instance.
  /* eslint-disable @typescript-eslint/unbound-method */

  const origFillRect = proto.fillRect
  proto.fillRect = function (this: OptimizedBuffer, x, y, width, height, bg) {
    let x2 = Math.trunc(x)
    let y2 = Math.trunc(y)
    let w = Math.trunc(width)
    let h = Math.trunc(height)
    if (x2 < 0) {
      w += x2
      x2 = 0
    }
    if (y2 < 0) {
      h += y2
      y2 = 0
    }
    if (w <= 0 || h <= 0) return
    origFillRect.call(this, x2, y2, w, h, bg)
  }

  const origDrawText = proto.drawText
  proto.drawText = function (this: OptimizedBuffer, text, x, y, ...rest) {
    if (x < 0 || y < 0) return
    origDrawText.call(this, text, x, y, ...rest)
  }

  const origSetCell = proto.setCell
  proto.setCell = function (this: OptimizedBuffer, x, y, ...rest) {
    if (x < 0 || y < 0) return
    origSetCell.call(this, x, y, ...rest)
  }

  const origSetCellAlpha = proto.setCellWithAlphaBlending
  proto.setCellWithAlphaBlending = function (this: OptimizedBuffer, x, y, ...rest) {
    if (x < 0 || y < 0) return
    origSetCellAlpha.call(this, x, y, ...rest)
  }

  const origDrawChar = proto.drawChar
  proto.drawChar = function (this: OptimizedBuffer, char, x, y, ...rest) {
    if (x < 0 || y < 0) return
    origDrawChar.call(this, char, x, y, ...rest)
  }

  // Same u32 marshaling on a different entry point: `textBufferViewSetViewport`
  // takes x/y/width/height as u32, but `TextRenderable.onResize` feeds it the
  // RAW transient layout size — observed NON-u32 (negative/NaN) mid-relayout
  // while a shrinking list (the fuzzy picker filtering rows away) reflows. Bun
  // wraps/coerces, node:ffi throws (`Argument 3 must be a uint32`). Coerce into
  // the valid quadrant; a zero-sized viewport is the native side's own no-op.
  const u32 = (v: number) => (Number.isFinite(v) ? Math.max(0, Math.trunc(v)) : 0)
  const viewProto = TextBufferView.prototype
  const origSetViewport = viewProto.setViewport
  viewProto.setViewport = function (this: TextBufferView, x, y, width, height) {
    origSetViewport.call(this, u32(x), u32(y), u32(width), u32(height))
  }
}
