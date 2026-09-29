"""hermes-mods browser tools and behaviour layered on top of ``browser_tool``.

Everything here is specific to this fork, kept out of ``browser_tool.py`` so a
future upstream merge (upstream split that file into ~12 modules) only has to
re-home one self-contained module.

What it adds, and the measurement that motivated it (Flash-Next, 1,403
browser calls, 2026-09-08 → 09-29):

* ``browser_read``  — agent-readable text of the page or of one element.
  72 ``browser_eval`` calls were ``document.body.innerText`` and 32% of
  snapshots hit the 15k cap, cutting the bottom of the page.
* ``browser_find``  — act on an element by role/text/label/placeholder
  without a snapshot round trip.
* ``browser_wait``  — wait for text / URL / element / load state / JS
  condition instead of retrying (7 navigate + 5 snapshot timeouts).
* ``browser_frame`` — scope snapshot/click/type/read/eval to an iframe.
  42% of evals were contentDocument hacks into iframes.
* post-action observation on click/type/press/find: url + a snapshot DELTA,
  because ~60% of successful actions were immediately followed by a
  re-observation call (a whole model turn each).
* a progress guard: repeated failures, identical repeated calls and very
  long browser episodes get a ``guidance`` field telling the model to stop
  and report (one session burned 694 calls / 3h20 on an impossible target).
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.registry import registry
from tools import browser_tool as bt

logger = logging.getLogger(__name__)

# ── agent-browser capabilities ────────────────────────────────────────────────

_AB_VERSION: Optional[Tuple[Tuple[int, int, int], str]] = None
_AB_LOCK = threading.Lock()


def agent_browser_version() -> Tuple[Tuple[int, int, int], str]:
    """(numeric version, raw string) of the agent-browser CLI Hermes runs.

    ``--version`` returns before any daemon/connect logic, so this never
    touches the browser. Cached per process; ((0, 0, 0), "") when unknown.
    """
    global _AB_VERSION
    with _AB_LOCK:
        if _AB_VERSION is not None:
            return _AB_VERSION
        raw = ""
        try:
            cmd = bt._find_agent_browser()
            if cmd and not bt._is_npx_agent_browser_sentinel(cmd):
                out = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=10)
                raw = (out.stdout or out.stderr or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("agent-browser --version failed: %s", exc)
        m = re.search(r"(\d+)\.(\d+)\.(\d+)", raw)
        ver = tuple(int(x) for x in m.groups()) if m else (0, 0, 0)
        _AB_VERSION = (ver, raw)  # type: ignore[assignment]
        return _AB_VERSION  # type: ignore[return-value]


def ab_supports_delta() -> bool:
    """``snapshot --delta`` / persistent refs arrived in agent-browser 0.38.0."""
    return agent_browser_version()[0] >= (0, 38, 0)


def ab_is_hermes_fork() -> bool:
    """The local fork adds --prune, --frame-depth, frame-aware eval, download fixes."""
    return "hermes" in agent_browser_version()[1]


def snapshot_flags(*, compact: bool = True, baseline: bool = False) -> List[str]:
    """Flags every Hermes snapshot shares, so delta baselines stay comparable."""
    flags: List[str] = ["-c"] if compact else []
    if ab_is_hermes_fork():
        flags.append("--prune")
    if baseline and ab_supports_delta():
        flags += ["--delta", "--full"]
    return flags


def normalize_snapshot_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten 0.38's ``{"snapshot": {"kind", "tree", "refs"}}`` to the old shape."""
    snap = data.get("snapshot")
    if isinstance(snap, dict):
        out = dict(data)
        out["snapshot_kind"] = snap.get("kind")
        out["snapshot"] = snap.get("tree", "")
        out["refs"] = snap.get("refs", data.get("refs", {}))
        if snap.get("kind") == "delta":
            out["delta"] = snap
        return out
    return data


# ── per-task state: selected frame ────────────────────────────────────────────

_FRAME_SELECTED: Dict[str, str] = {}


def frame_selected(task_id: str) -> Optional[str]:
    return _FRAME_SELECTED.get(bt._last_session_key(task_id or "default"))


def _clear_frame(task_id: Optional[str]) -> None:
    _FRAME_SELECTED.pop(bt._last_session_key(task_id or "default"), None)


# ── helpers ───────────────────────────────────────────────────────────────────

def _json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _target(ref: Optional[str], selector: Optional[str]) -> Optional[str]:
    if ref:
        return bt._normalize_ref(ref)
    if selector:
        return selector.strip()
    return None


def _spill_long_text(text: str, max_chars: int, label: str) -> Tuple[str, Optional[str]]:
    """Cut ``text`` at a line boundary; store the full text for read_file paging."""
    if len(text) <= max_chars:
        return text, None
    path = bt._store_full_snapshot(text)
    cut = text[:max_chars]
    nl = cut.rfind("\n")
    if nl > max_chars * 0.6:
        cut = cut[:nl]
    shown_lines = cut.count("\n") + 1
    note = f"\n\n[... {label} truncated at {len(cut):,} of {len(text):,} chars"
    if path:
        note += f" — full text: read_file path=\"{path}\" offset={shown_lines + 1} limit=200"
    note += "]"
    return cut + note, path


# ── post-action observation ───────────────────────────────────────────────────

OBSERVATION_MAX_CHARS = 4000


def observe_after_action(task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """What changed after an action: url + a snapshot delta (agent-browser >= 0.38).

    Returns None when deltas are unsupported (the model then snapshots itself,
    as before). Costs one CLI call (~0.1-0.3 s) and saves a model turn in the
    common click → snapshot pattern.
    """
    if not ab_supports_delta() or bt._is_camofox_mode():
        return None
    tid = bt._last_session_key(task_id or "default")
    res = bt._run_browser_command(tid, "snapshot", snapshot_flags() + ["--delta"], timeout=20)
    if not res.get("success"):
        return None
    data = res.get("data", {}) or {}
    snap = data.get("snapshot")
    obs: Dict[str, Any] = {"url": data.get("origin", "")}
    if not isinstance(snap, dict):
        return None
    kind = snap.get("kind")
    if kind == "unchanged":
        obs["page"] = "unchanged (the accessibility tree is identical)"
        return obs
    if kind == "delta":
        tc = snap.get("treeChange") or {}
        lines = tc.get("lines") or []
        removed = [c.get("ref") for c in snap.get("changes", []) if c.get("op") == "remove"]
        text = "\n".join(lines)
        obs["page"] = "changed"
        obs["changed_lines"] = bt._redact_browser_output(
            _spill_long_text(text, OBSERVATION_MAX_CHARS, "change")[0]) if text else ""
        if removed:
            obs["removed_refs"] = removed[:40]
        return obs
    # full: new document or a large change
    tree = snap.get("tree", "")
    obs["page"] = "new page or large change - compact snapshot below"
    obs["snapshot"] = bt._redact_browser_output(_spill_long_text(tree, OBSERVATION_MAX_CHARS, "snapshot")[0])
    return obs


# ── keep the acted-on tab visible ─────────────────────────────────────────────
#
# When agent-browser attaches over CDP to an already-open browser it takes the
# FIRST tab as its active tab without bringing it to front, while the window
# shows another (restored) tab. Chromium ignores synthetic input sent to a
# hidden tab: clicks "succeed" and nothing happens (measured 2026-09-29:
# document.visibilityState == "hidden" on the tab being clicked). Only an
# explicit `tab <id>` switch calls Page.bringToFront, so do that once per
# session before the first input action and after every navigation.

_FRONTED: set = set()
_INPUT_TOOLS = frozenset({
    "browser_click", "browser_type", "browser_press", "browser_find", "browser_mouse",
    "browser_drag", "browser_scroll", "browser_upload", "browser_dropzone_upload",
    "browser_download", "browser_mouse_wheel",
})


def bring_active_tab_to_front(task_id: Optional[str]) -> bool:
    if bt._is_camofox_mode():
        return False
    tid = bt._last_session_key(task_id or "default")
    res = bt._run_browser_command(tid, "tab", ["list"], timeout=10)
    if not res.get("success"):
        return False
    tabs, _active = bt._normalize_tab_payload(res.get("data", {}))
    active = next((t for t in tabs if t.get("active")), None)
    if not active or not active.get("id"):
        return False
    ok = bool(bt._run_browser_command(tid, "tab", [active["id"]], timeout=10).get("success"))
    if ok:
        _FRONTED.add(tid)
    return ok


# ── progress guard (stop rule) ────────────────────────────────────────────────

EPISODE_IDLE_RESET_S = 600
VOLUME_THRESHOLDS = (40, 80, 120, 160, 200)


class _Progress:
    __slots__ = ("calls", "fail_streak", "last_sig", "same_streak", "last_t", "said")

    def __init__(self) -> None:
        self.calls = 0
        self.fail_streak = 0
        self.last_sig: Optional[str] = None
        self.same_streak = 0
        self.last_t = 0.0
        self.said: set = set()


_PROGRESS: Dict[str, _Progress] = {}
_PROGRESS_LOCK = threading.Lock()


def _result_failed(result: str) -> bool:
    try:
        data = json.loads(result)
    except Exception:
        return False
    return isinstance(data, dict) and data.get("success") is False


def progress_guidance(task_id: Optional[str], tool: str, args: Dict[str, Any], result: str) -> Optional[str]:
    """Return a stop/report nudge when the browser episode stops converging."""
    key = task_id or "default"
    now = time.time()
    with _PROGRESS_LOCK:
        p = _PROGRESS.get(key)
        if p is None or now - p.last_t > EPISODE_IDLE_RESET_S:
            p = _Progress()
            _PROGRESS[key] = p
        p.last_t = now
        p.calls += 1
        sig = tool + json.dumps(args, sort_keys=True, default=str)
        p.same_streak = p.same_streak + 1 if sig == p.last_sig else 1
        p.last_sig = sig
        p.fail_streak = p.fail_streak + 1 if _result_failed(result) else 0

        if p.fail_streak >= 3 and ("fail", p.fail_streak // 3) not in p.said:
            p.said.add(("fail", p.fail_streak // 3))
            return (f"{p.fail_streak} browser actions in a row have failed. Do not try another "
                    "variation of the same approach. Look at the page once (browser_snapshot or "
                    "browser_vision); if the obstacle is still there, stop and tell the user what "
                    "blocks you and what you suggest.")
        if p.same_streak >= 3 and ("same", sig) not in p.said:
            p.said.add(("same", sig))
            return ("You have sent this exact same browser call 3 times in a row. Repeating it will "
                    "not change the result: change approach or stop and report to the user.")
        for threshold in VOLUME_THRESHOLDS:
            if p.calls == threshold:
                return (f"You have made {p.calls} browser calls in this task. If you are not clearly "
                        "converging, stop now and give the user a short status: what is done, what "
                        "blocks you, what you propose. Do not keep grinding on an obstacle.")
    return None


def _with_guidance(result: str, guidance: str) -> str:
    try:
        data = json.loads(result)
    except Exception:
        return f"{result}\n\n[guidance] {guidance}"
    if isinstance(data, dict):
        data["guidance"] = guidance
        return _json(data)
    return f"{result}\n\n[guidance] {guidance}"


_OBSERVE_TOOLS = frozenset({"browser_click", "browser_type", "browser_press", "browser_find"})
_FRAME_RESET_TOOLS = frozenset({"browser_navigate", "browser_back", "browser_tab"})


def _wrap_handler(name: str, handler: Callable) -> Callable:
    def wrapped(args, **kw):
        task_id = kw.get("task_id")
        if name in _INPUT_TOOLS:
            try:
                if bt._last_session_key(task_id or "default") not in _FRONTED:
                    bring_active_tab_to_front(task_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("bring-to-front preflight failed: %s", exc)
        result = handler(args, **kw)
        if not isinstance(result, str):
            return result
        try:
            if name == "browser_navigate" and not _result_failed(result):
                bring_active_tab_to_front(task_id)
            elif name == "browser_tab" and not _result_failed(result):
                _FRONTED.add(bt._last_session_key(task_id or "default"))
            if name in _FRAME_RESET_TOOLS:
                _clear_frame(task_id)
            if name in _OBSERVE_TOOLS and not _result_failed(result):
                obs = observe_after_action(task_id)
                if obs is not None:
                    data = json.loads(result)
                    if isinstance(data, dict):
                        data["observation"] = obs
                        result = _json(data)
            guidance = progress_guidance(task_id, name, args if isinstance(args, dict) else {}, result)
            if guidance:
                result = _with_guidance(result, guidance)
        except Exception as exc:  # noqa: BLE001 — never break the tool over bookkeeping
            logger.debug("browser post-processing failed for %s: %s", name, exc)
        return result

    wrapped.__wrapped__ = handler  # type: ignore[attr-defined]
    return wrapped


# ── new tools ─────────────────────────────────────────────────────────────────

def browser_read(selector: Optional[str] = None, ref: Optional[str] = None, filter: Optional[str] = None,
                 outline: bool = False, max_chars: Optional[int] = None, task_id: Optional[str] = None) -> str:
    if bt._is_camofox_mode():
        return bt._browser_tool_unsupported_in_camofox("browser_read")
    tid = bt._last_session_key(task_id or "default")
    target = _target(ref, selector)
    limit = int(max_chars) if max_chars else bt.get_browser_snapshot_threshold()
    limit = max(1000, min(limit, 60000))
    if target:
        res = bt._run_browser_command(tid, "get", ["text", target])
        if not res.get("success"):
            return _json({"success": False, "error": res.get("error", f"Could not read text of {target}")})
        data = res.get("data", {}) or {}
        text = data.get("text") if isinstance(data, dict) else data
        text = "" if text is None else str(text)
    else:
        args: List[str] = []
        if filter:
            args += ["--filter", filter]
        if outline:
            args.append("--outline")
        res = bt._run_browser_command(tid, "read", args, timeout=40)
        if not res.get("success"):
            return _json({"success": False, "error": res.get("error", "Could not read the page")})
        data = res.get("data", {}) or {}
        text = ""
        if isinstance(data, dict):
            for key in ("content", "text", "markdown", "body"):
                if isinstance(data.get(key), str):
                    text = data[key]
                    break
        elif isinstance(data, str):
            text = data
    shown, path = _spill_long_text(text, limit, "text")
    out = {"success": True, "chars": len(text), "text": bt._redact_browser_output(shown)}
    if target:
        out["element"] = target
    if frame_selected(task_id or "default"):
        out["frame"] = frame_selected(task_id or "default")
    return _json(out)


_FIND_BY = ("role", "text", "label", "placeholder", "alt", "title", "testid")
_FIND_ACTIONS = ("click", "fill", "type", "hover", "focus", "check", "uncheck", "text")


def browser_find(by: str, value: str, action: str = "click", text: Optional[str] = None,
                 name: Optional[str] = None, exact: bool = False, task_id: Optional[str] = None) -> str:
    if bt._is_camofox_mode():
        return bt._browser_tool_unsupported_in_camofox("browser_find")
    by = (by or "").strip().lower()
    action = (action or "click").strip().lower()
    if by not in _FIND_BY:
        return _json({"success": False, "error": f"`by` must be one of {', '.join(_FIND_BY)}"})
    if action not in _FIND_ACTIONS:
        return _json({"success": False, "error": f"`action` must be one of {', '.join(_FIND_ACTIONS)}"})
    if action in ("fill", "type") and text is None:
        return _json({"success": False, "error": f"action '{action}' needs `text`"})
    tid = bt._last_session_key(task_id or "default")
    args = [by, value, action]
    if text is not None and action in ("fill", "type"):
        args.append(text)
    if name:
        args += ["--name", name]
    if exact:
        args.append("--exact")
    res = bt._run_browser_command(tid, "find", args)
    if not res.get("success"):
        return _json({"success": False, "error": res.get("error", f"No element found by {by}={value!r}")})
    data = res.get("data", {}) or {}
    out: Dict[str, Any] = {"success": True, "found_by": f"{by}={value}", "action": action}
    if action == "text":
        val = data.get("text") if isinstance(data, dict) else data
        out["text"] = bt._redact_browser_output(val)
    if action in ("fill", "type"):
        out["typed_chars"] = len(text or "")
    return _json(out)


def browser_wait(text: Optional[str] = None, url: Optional[str] = None, selector: Optional[str] = None,
                 ref: Optional[str] = None, load: Optional[str] = None, js: Optional[str] = None,
                 hidden: bool = False, ms: Optional[int] = None, timeout: Optional[float] = None,
                 task_id: Optional[str] = None) -> str:
    if bt._is_camofox_mode():
        return bt._browser_tool_unsupported_in_camofox("browser_wait")
    target = _target(ref, selector)
    given = [k for k, v in (("text", text), ("url", url), ("selector", target), ("load", load),
                            ("js", js), ("ms", ms)) if v]
    if len(given) != 1:
        return _json({"success": False, "error": "Give exactly one of: text, url, selector/ref, load, js, ms."})
    wait_s = float(timeout) if timeout else 15.0
    wait_s = max(1.0, min(wait_s, 120.0))
    if text:
        args = ["--text", text]
    elif url:
        args = ["--url", url]
    elif target:
        args = [target] + (["--state", "hidden"] if hidden else [])
    elif load:
        if load not in ("load", "domcontentloaded", "networkidle"):
            return _json({"success": False, "error": "load must be load, domcontentloaded or networkidle"})
        args = ["--load", load]
    elif js:
        args = ["--fn", js]
    else:
        ms_i = max(1, min(int(ms), 60000))
        args = [str(ms_i)]
        wait_s = ms_i / 1000.0
    if not ms:
        args += ["--timeout", str(int(wait_s * 1000))]
    tid = bt._last_session_key(task_id or "default")
    t0 = time.time()
    res = bt._run_browser_command(tid, "wait", args, timeout=int(wait_s) + 15)
    waited = round(time.time() - t0, 1)
    if not res.get("success"):
        err = res.get("error", "wait failed")
        if re.search(r"time(d)? ?out", str(err), re.I):
            err = f"Condition not met within {wait_s:g}s ({given[0]}). " + str(err)[:300]
        return _json({"success": False, "error": err, "waited_s": waited})
    return _json({"success": True, "condition": given[0], "waited_s": waited})


def browser_frame(ref: Optional[str] = None, selector: Optional[str] = None, main: bool = False,
                  task_id: Optional[str] = None) -> str:
    if bt._is_camofox_mode():
        return bt._browser_tool_unsupported_in_camofox("browser_frame")
    tid = bt._last_session_key(task_id or "default")
    target = "main" if main else _target(ref, selector)
    if not target:
        return _json({"success": False, "error": "Give the iframe's ref (from browser_snapshot), a CSS selector, or main=true."})
    res = bt._run_browser_command(tid, "frame", [target])
    if not res.get("success"):
        return _json({"success": False, "error": res.get("error", f"Could not select frame {target}")})
    if target == "main":
        _FRAME_SELECTED.pop(tid, None)
        return _json({"success": True, "frame": "main",
                      "note": "Back in the top document."})
    _FRAME_SELECTED[tid] = target
    return _json({
        "success": True,
        "frame": target,
        "note": ("browser_snapshot, browser_click, browser_type, browser_read and browser_eval now act "
                 "inside this iframe. Call browser_frame(main=true) to return to the page; navigating "
                 "or switching tab also resets it."),
    })


# ── schemas & registration ────────────────────────────────────────────────────

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "browser_read": {
        "name": "browser_read",
        "description": (
            "Read the page as clean text (headings, paragraphs, lists, tables) - the right tool to READ "
            "content, including long pages whose snapshot is truncated. Give a ref or CSS selector to read "
            "only one element (e.g. a table or article). `filter` keeps only sections mentioning a word; "
            "outline=true returns just the heading outline. Use browser_snapshot instead when you need "
            "refs to click or type."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Element ref from a snapshot (e.g. '@e12') to read only that element"},
                "selector": {"type": "string", "description": "CSS selector of the element to read (alternative to ref)"},
                "filter": {"type": "string", "description": "Only keep page sections that mention this text"},
                "outline": {"type": "boolean", "description": "Return only the heading outline of the page"},
                "max_chars": {"type": "integer", "description": "Max characters returned (default 15000); the rest is saved to a file you can page with read_file"},
            },
            "required": [],
        },
    },
    "browser_find": {
        "name": "browser_find",
        "description": (
            "Find ONE element by what a person sees (its role and name, visible text, label, placeholder...) "
            "and act on it in one call - no snapshot needed first. Examples: by='role', value='button', "
            "name='Sign in'; by='label', value='Email', action='fill', text='me@x.com'; by='text', "
            "value='Accept all'. The result includes what changed on the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "by": {"type": "string", "enum": list(_FIND_BY), "description": "How to locate the element"},
                "value": {"type": "string", "description": "The role (button, link, textbox, checkbox...), text, label, placeholder, alt, title or test id"},
                "name": {"type": "string", "description": "With by='role': the accessible name to match (e.g. the button's text)"},
                "action": {"type": "string", "enum": list(_FIND_ACTIONS), "description": "What to do with it (default click; 'text' returns its text)"},
                "text": {"type": "string", "description": "Text for action fill/type"},
                "exact": {"type": "boolean", "description": "Require an exact (not substring) match"},
            },
            "required": ["by", "value"],
        },
    },
    "browser_wait": {
        "name": "browser_wait",
        "description": (
            "Wait until the page is ready instead of retrying: until some text appears, the URL matches "
            "(glob like '**/dashboard'), an element appears (or disappears with hidden=true), a load state "
            "is reached, or a JavaScript condition is true. Give exactly one condition."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Wait until this text is visible"},
                "url": {"type": "string", "description": "Wait until the URL matches this glob pattern"},
                "ref": {"type": "string", "description": "Wait for this element ref"},
                "selector": {"type": "string", "description": "Wait for this CSS selector"},
                "hidden": {"type": "boolean", "description": "With ref/selector: wait until it disappears"},
                "load": {"type": "string", "enum": ["load", "domcontentloaded", "networkidle"], "description": "Wait for a page load state (networkidle only for pages that go quiet)"},
                "js": {"type": "string", "description": "Wait until this JavaScript expression is truthy"},
                "ms": {"type": "integer", "description": "Just wait this many milliseconds (max 60000)"},
                "timeout": {"type": "number", "description": "Max seconds to wait (default 15, max 120)"},
            },
            "required": [],
        },
    },
    "browser_frame": {
        "name": "browser_frame",
        "description": (
            "Work inside an iframe (embedded player, payment form, e-learning module, cross-origin widget): "
            "select it by its ref from browser_snapshot (the 'Iframe' node) or a CSS selector, then "
            "browser_snapshot / click / type / read / eval act inside it. main=true returns to the page. "
            "Use this instead of reaching into iframes with JavaScript (contentDocument is blocked for "
            "cross-origin frames)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Ref of the iframe element in the snapshot (e.g. '@e7')"},
                "selector": {"type": "string", "description": "CSS selector of the iframe (alternative to ref)"},
                "main": {"type": "boolean", "description": "Return to the top-level page"},
            },
            "required": [],
        },
    },
}

_EMOJI = {"browser_read": "📖", "browser_find": "🎯", "browser_wait": "⏳", "browser_frame": "🪟"}

registry.register(
    name="browser_read", toolset="browser", schema=SCHEMAS["browser_read"],
    handler=lambda args, **kw: browser_read(
        selector=args.get("selector"), ref=args.get("ref"), filter=args.get("filter"),
        outline=bool(args.get("outline")), max_chars=args.get("max_chars"), task_id=kw.get("task_id")),
    check_fn=bt.check_browser_requirements, emoji=_EMOJI["browser_read"],
)
registry.register(
    name="browser_find", toolset="browser", schema=SCHEMAS["browser_find"],
    handler=lambda args, **kw: browser_find(
        by=args.get("by", ""), value=args.get("value", ""), action=args.get("action", "click"),
        text=args.get("text"), name=args.get("name"), exact=bool(args.get("exact")), task_id=kw.get("task_id")),
    check_fn=bt.check_browser_requirements, emoji=_EMOJI["browser_find"],
)
registry.register(
    name="browser_wait", toolset="browser", schema=SCHEMAS["browser_wait"],
    handler=lambda args, **kw: browser_wait(
        text=args.get("text"), url=args.get("url"), selector=args.get("selector"), ref=args.get("ref"),
        load=args.get("load"), js=args.get("js"), hidden=bool(args.get("hidden")), ms=args.get("ms"),
        timeout=args.get("timeout"), task_id=kw.get("task_id")),
    check_fn=bt.check_browser_requirements, emoji=_EMOJI["browser_wait"],
)
registry.register(
    name="browser_frame", toolset="browser", schema=SCHEMAS["browser_frame"],
    handler=lambda args, **kw: browser_frame(
        ref=args.get("ref"), selector=args.get("selector"), main=bool(args.get("main")), task_id=kw.get("task_id")),
    check_fn=bt.check_browser_requirements, emoji=_EMOJI["browser_frame"],
)


def _install_wrappers() -> None:
    """Wrap every registered browser_* handler with observation + progress guard."""
    for name in list(getattr(registry, "_tools", {}).keys()):
        if not name.startswith("browser_"):
            continue
        entry = registry.get_entry(name)
        if entry is None or getattr(entry.handler, "__wrapped__", None) is not None or entry.is_async:
            continue
        entry.handler = _wrap_handler(name, entry.handler)


_install_wrappers()
