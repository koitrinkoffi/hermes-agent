"""hermes-mods browser tools and behaviour layered on top of ``browser_tool``.

Everything here is specific to this fork, kept out of ``browser_tool.py`` so a
future upstream merge (upstream split that file into ~12 modules) only has to
re-home one self-contained module. ``browser_tool.py`` keeps upstream's tools
plus the managed-browser lifecycle (browser_start / browser_close); the
fork-only tools live here: browser_tab, browser_upload,
browser_dropzone_upload, browser_download, browser_eval, browser_pdf,
browser_mouse, browser_mouse_wheel, browser_drag, and the ones below.

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
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

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


DEFAULT_FRAME_DEPTH = 3


def snapshot_flags(*, compact: bool = True, baseline: bool = False) -> List[str]:
    """Flags every Hermes snapshot shares, so delta baselines stay comparable."""
    flags: List[str] = ["-c"] if compact else []
    if ab_is_hermes_fork():
        # Nested iframes (e-learning players, payment widgets) are common;
        # upstream only inlines one level.
        flags += ["--prune", "--frame-depth", str(DEFAULT_FRAME_DEPTH)]
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


SETTLE_S = 0.3
SETTLE_LOAD_TIMEOUT_MS = 8000


def _settle(tid: str) -> None:
    """Let a click/submit start its navigation, then wait for the load to end.

    Without it the delta was taken before the navigation began and showed the
    old page (measured on a login form). Costs ~0.3 s when nothing navigates.
    """
    time.sleep(SETTLE_S)
    bt._run_browser_command(tid, "wait", ["--load", "load", "--timeout", str(SETTLE_LOAD_TIMEOUT_MS)],
                            timeout=SETTLE_LOAD_TIMEOUT_MS // 1000 + 5)


def observe_after_action(task_id: Optional[str], settle: bool = False) -> Optional[Dict[str, Any]]:
    """What changed after an action: url + a snapshot delta (agent-browser >= 0.38).

    Returns None when deltas are unsupported (the model then snapshots itself,
    as before). Costs one CLI call (~0.1-0.3 s) and saves a model turn in the
    common click → snapshot pattern.
    """
    if not ab_supports_delta() or bt._is_camofox_mode():
        return None
    tid = bt._last_session_key(task_id or "default")
    if settle:
        _settle(tid)
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
    "browser_download", "browser_mouse_wheel", "browser_fill_form",
})
# Tools whose arguments carry snapshot refs: switching tab clears agent-browser's
# ref map, so after a bring-to-front the refs are rebuilt with a silent snapshot.
_REF_TOOLS = frozenset({
    "browser_click", "browser_type", "browser_mouse", "browser_drag", "browser_upload",
    "browser_download", "browser_frame", "browser_read", "browser_fill_form",
})
_OBSERVE_FIRST_TOOLS = _INPUT_TOOLS | {"browser_snapshot", "browser_read", "browser_vision"}


def active_tab_hidden(task_id: Optional[str]) -> bool:
    """True when the tab agent-browser acts on is not the one shown in the window."""
    try:
        out = json.loads(bt._browser_eval("document.visibilityState", task_id, timeout=3))
    except Exception:
        return False
    return bool(out.get("success")) and out.get("result") == "hidden"


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


def _front_preflight(name: str, args: Dict[str, Any], task_id: Optional[str]) -> None:
    tid = bt._last_session_key(task_id or "default")
    if name == "browser_navigate":
        # The navigation replaces the page and its refs anyway: always safe.
        bring_active_tab_to_front(task_id)
        return
    if name not in _OBSERVE_FIRST_TOOLS:
        return
    if not active_tab_hidden(task_id):
        _FRONTED.add(tid)
        return
    if bring_active_tab_to_front(task_id) and name in _REF_TOOLS and name != "browser_snapshot":
        bt._run_browser_command(tid, "snapshot", snapshot_flags(baseline=True), timeout=20)


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


_OBSERVE_TOOLS = frozenset({"browser_click", "browser_type", "browser_press", "browser_find", "browser_fill_form"})
_FRAME_RESET_TOOLS = frozenset({"browser_navigate", "browser_back", "browser_tab"})


def _wrap_handler(name: str, handler: Callable) -> Callable:
    def wrapped(args, **kw):
        task_id = kw.get("task_id")
        try:
            _front_preflight(name, args if isinstance(args, dict) else {}, task_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("bring-to-front preflight failed: %s", exc)
        result = handler(args, **kw)
        if not isinstance(result, str):
            return result
        # agent-browser's own errors name its CLI verbs; point at the Hermes tool.
        if "`dialog accept` or `dialog dismiss`" in result:
            result = result.replace(
                "`dialog accept` or `dialog dismiss`",
                "browser_dialog(action='accept', prompt_text=...) or browser_dialog(action='dismiss')")
        try:
            _key = bt._last_session_key(task_id or "default")
            if _key in bt.RELAUNCH_NOTICES:
                bt.RELAUNCH_NOTICES.discard(_key)
                _FRONTED.discard(_key)
                data = json.loads(result)
                if isinstance(data, dict):
                    data["browser_relaunched"] = (
                        "The browser window had been closed; Hermes reopened it and it restored its "
                        "previous tabs. Check browser_tab(action='list') before relying on the current page.")
                    result = _json(data)
            if (name == "browser_navigate" and isinstance(args, dict) and args.get("read")
                    and not _result_failed(result)):
                page = json.loads(browser_read(max_chars=NAVIGATE_TEXT_MAX_CHARS, task_id=task_id))
                data = json.loads(result)
                if isinstance(data, dict) and page.get("success"):
                    data["text"] = page.get("text", "")
                    data["text_chars"] = page.get("chars")
                    result = _json(data)
            if name == "browser_tab" and not _result_failed(result):
                _FRONTED.add(bt._last_session_key(task_id or "default"))
            if name in _FRAME_RESET_TOOLS:
                _clear_frame(task_id)
            if name in _OBSERVE_TOOLS and not _result_failed(result) and '"dialog_opened": true' not in result:
                # Typing into a field never navigates; clicks, keys and submits may.
                navigating = name != "browser_type" and not (
                    name == "browser_fill_form" and '"submitted"' not in result)
                obs = observe_after_action(task_id, settle=navigating)
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

    wrapped._hermes_browser_wrapped = True  # type: ignore[attr-defined]
    wrapped._hermes_inner = handler  # type: ignore[attr-defined]
    return wrapped


# ── new tools ─────────────────────────────────────────────────────────────────


def browser_read(selector: Optional[str] = None, ref: Optional[str] = None, filter: Optional[str] = None,
                 outline: bool = False, max_chars: Optional[int] = None, task_id: Optional[str] = None,
                 frames: bool = False) -> str:
    if bt._is_camofox_mode():
        return bt._browser_tool_unsupported_in_camofox("browser_read")
    if frames:
        return browser_read_frames(task_id=task_id, max_chars=max_chars)
    tid = bt._last_session_key(task_id or "default")
    target = _target(ref, selector)
    if not target and frame_selected(task_id or "default") and not (filter or outline):
        # agent-browser's `read` always reads the top document; element text
        # reads honour the frame selected with browser_frame.
        target = "body"
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
    if isinstance(data, dict) and data.get("dialogOpened"):
        out["dialog_opened"] = True
        out["next"] = ("A JavaScript dialog is now open and blocks the page: answer it with "
                       "browser_dialog(action='accept', prompt_text=...) or browser_dialog(action='dismiss').")
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


def browser_fill_form(fields: List[Dict[str, Any]], submit_ref: Optional[str] = None,
                      submit_text: Optional[str] = None, press_enter: bool = False,
                      task_id: Optional[str] = None) -> str:
    """Fill several fields (and optionally submit) in ONE tool call.

    A login used to take 3-8 model turns (type, type, click, snapshot...);
    every turn costs ~4.7 s on Flash-Next while the browser work itself is
    ~0.1 s per field.
    """
    if bt._is_camofox_mode():
        return bt._browser_tool_unsupported_in_camofox("browser_fill_form")
    if not isinstance(fields, list) or not fields:
        return _json({"success": False, "error": "`fields` must be a non-empty list of {ref|selector|label|placeholder, value}."})
    tid = bt._last_session_key(task_id or "default")
    report: List[Dict[str, Any]] = []
    all_ok = True
    for i, f in enumerate(fields):
        if not isinstance(f, dict) or "value" not in f:
            report.append({"field": i, "ok": False, "error": "each field needs a locator and a value"})
            all_ok = False
            continue
        value = f["value"]
        target = _target(f.get("ref"), f.get("selector"))
        label = f.get("label") or f.get("placeholder")
        by = "label" if f.get("label") else "placeholder"
        where = target or f"{by}={label}"
        if target:
            if isinstance(value, bool):
                res = bt._run_browser_command(tid, "check" if value else "uncheck", [target])
            else:
                res = bt._run_browser_command(tid, "fill", [target, str(value)])
                if not res.get("success"):
                    # <select> elements refuse fill: pick the option instead.
                    alt = bt._run_browser_command(tid, "select", [target, str(value)])
                    if alt.get("success"):
                        res = alt
        elif label:
            action = ("check" if value else "uncheck") if isinstance(value, bool) else "fill"
            args = [by, str(label), action] + ([] if isinstance(value, bool) else [str(value)])
            res = bt._run_browser_command(tid, "find", args)
        else:
            res = {"success": False, "error": "no ref, selector, label or placeholder"}
        ok = bool(res.get("success"))
        entry: Dict[str, Any] = {"field": where, "ok": ok}
        if not ok:
            entry["error"] = str(res.get("error", "failed"))[:300]
            all_ok = False
        report.append(entry)
    out: Dict[str, Any] = {"success": all_ok, "fields": report}
    if not all_ok:
        out["error"] = "Some fields could not be filled; nothing was submitted."
        return _json(out)
    submitted = None
    if submit_ref:
        res = bt._run_browser_command(tid, "click", [bt._normalize_ref(submit_ref)])
        submitted = f"click {bt._normalize_ref(submit_ref)}"
    elif submit_text:
        res = bt._run_browser_command(tid, "find", ["role", "button", "click", "--name", submit_text])
        if not res.get("success"):
            res = bt._run_browser_command(tid, "find", ["text", submit_text, "click"])
        submitted = f"click '{submit_text}'"
    elif press_enter:
        res = bt._run_browser_command(tid, "press", ["Enter"])
        submitted = "press Enter"
    else:
        res = {"success": True}
    if submitted:
        out["submitted"] = submitted
        if not res.get("success"):
            out["success"] = False
            out["error"] = f"Fields filled but submit failed: {str(res.get('error', ''))[:300]}"
        data = res.get("data") or {}
        if isinstance(data, dict) and data.get("dialogOpened"):
            out["dialog_opened"] = True
    return _json(out)


def _frame_texts(tree: str) -> List[Dict[str, Any]]:
    """Group the visible text of every (nested) iframe of a full snapshot tree."""
    lines = tree.split("\n")
    frames: List[Dict[str, Any]] = []
    stack: List[Tuple[int, Dict[str, Any]]] = []  # (indent, frame)
    for line in lines:
        stripped = line.lstrip(" ")
        if not stripped.startswith("- "):
            continue
        indent = len(line) - len(stripped)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        body = stripped[2:]
        if body.startswith("Iframe"):
            name = re.search(r'Iframe "([^"]*)"', body)
            ref = re.search(r"\[ref=(e\d+)", body)
            frame = {"frame": f"@{ref.group(1)}" if ref else None,
                     "name": name.group(1) if name else "",
                     "inside": stack[-1][1]["frame"] if stack else None,
                     "_text": []}
            frames.append(frame)
            stack.append((indent, frame))
            continue
        if not stack:
            continue
        m = re.search(r'"((?:[^"\\]|\\.)*)"', body)
        if m and m.group(1).strip():
            text = m.group(1).strip()
            owner = stack[-1][1]["_text"]
            if not owner or owner[-1] != text:
                owner.append(text)
    for f in frames:
        f["text"] = "\n".join(f.pop("_text"))
    return frames


def browser_read_frames(task_id: Optional[str] = None, max_chars: Optional[int] = None) -> str:
    """Text of every iframe (nested ones included) in one call."""
    tid = bt._last_session_key(task_id or "default")
    res = bt._run_browser_command(tid, "snapshot", snapshot_flags(compact=False), timeout=30)
    if not res.get("success"):
        return _json({"success": False, "error": res.get("error", "Could not snapshot the page")})
    tree = normalize_snapshot_data(res.get("data", {}) or {}).get("snapshot", "") or ""
    frames = _frame_texts(tree)
    limit = max(1000, min(int(max_chars) if max_chars else bt.get_browser_snapshot_threshold(), 60000))
    per = max(500, limit // max(1, len(frames)))
    for f in frames:
        f["text"] = bt._redact_browser_output(_spill_long_text(f["text"], per, "frame text")[0])
    return _json({"success": True, "frames": frames, "count": len(frames),
                  "note": "Select one with browser_frame(ref=...) to act inside it." if frames
                  else "This page has no iframe."})


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
                "frames": {"type": "boolean", "description": "Return the text of EVERY iframe on the page (nested ones included), grouped per frame with its ref - one call instead of selecting frames one by one"},
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
        outline=bool(args.get("outline")), max_chars=args.get("max_chars"), task_id=kw.get("task_id"),
        frames=bool(args.get("frames"))),
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


# ── hermes-mods tools moved out of browser_tool.py (2026-09-29) ─────────────


SCHEMAS.update({
    "browser_tab": {
        "name": "browser_tab",
        "description": "Manage browser tabs: 'list' shows every open tab (index, id, title, url, active); 'switch' makes a tab the active one (all other browser tools act on the active tab); 'new' opens a tab (optionally at a URL) and makes it active; 'close' closes a tab (the active one when no index/tab_id is given). Links that open a new tab switch to it automatically - call 'list' if you are unsure which tab is active.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["new", "list", "switch", "close"],
                    "description": "Tab action to perform"
                },
                "index": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based position of the tab as shown by action='list' (for switch or close)"
                },
                "tab_id": {
                    "type": "string",
                    "description": "Alternative to index: the tab's id from action='list' (e.g. 't2')"
                },
                "url": {
                    "type": "string",
                    "description": "Optional URL to open when creating a new tab"
                }
            },
            "required": ["action"]
        }
    },
    "browser_upload": {
        "name": "browser_upload",
        "description": "Upload one or more local files through a file input element identified by its ref ID. Accepts either a single path or multiple paths. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The file input element reference from the snapshot (e.g., '@e3')"
                },
                "path": {
                    "type": "string",
                    "description": "Single local file path to upload"
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiple local file paths to upload"
                }
            },
            "required": ["ref"]
        }
    },
    "browser_dropzone_upload": {
        "name": "browser_dropzone_upload",
        "description": "Attach one or more local files to a drag-and-drop uploader (Dropzone.js and similar 'drop files here' zones) where browser_upload fails because the file input is hidden/JS-managed. Provide the dropzone element's CSS selector. Requires browser_navigate first.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the dropzone element (default '.dropzone')"
                },
                "path": {
                    "type": "string",
                    "description": "Single local file path to upload"
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiple local file paths to upload"
                }
            },
            "required": []
        }
    },
    "browser_download": {
        "name": "browser_download",
        "description": "Download a file by clicking its link or button (ref from the snapshot) and save it locally. Also works when the click opens a PDF in the browser viewer instead of downloading it. Returns the saved path and size. If `path` is omitted, Hermes picks a persistent default location.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., '@e5', '@e12')"
                },
                "path": {
                    "type": "string",
                    "description": "Optional destination file path. If omitted, Hermes chooses a persistent default path."
                },
                "timeout": {
                    "type": "number",
                    "description": "Seconds to wait for the file (default 60, max 300)"
                }
            },
            "required": ["ref"]
        }
    },
    "browser_eval": {
        "name": "browser_eval",
        "description": "Run a JavaScript expression in the active tab (inside the iframe selected with browser_frame, if any) and return its value. Best for READING structured data in one call: collect a list of items as JSON, pull window.__NEXT_DATA__ / JSON-LD, count elements. Top-level await works. Do NOT use it to click, type, scroll or navigate (use browser_click / browser_type / browser_find / browser_navigate), to read plain page text (use browser_read), or to reach into iframes via contentDocument (use browser_frame).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate (e.g., \"document.querySelectorAll('.price').length\"). Async/await is supported."
                },
                "timeout": {
                    "type": "number",
                    "description": "Seconds before giving up (default 10, max 60) - raise it for slow async work"
                }
            },
            "required": ["expression"]
        }
    },
    "browser_pdf": {
        "name": "browser_pdf",
        "description": "Save the current page as a PDF file (useful for archiving listings, receipts, articles). If no path is provided, Hermes saves it to the persistent default downloads directory and returns the absolute path. Requires browser_navigate first.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional destination .pdf file path. If omitted, Hermes chooses a persistent default path."
                }
            },
            "required": []
        }
    },
    "browser_mouse": {
        "name": "browser_mouse",
        "description": "Low-level mouse: move / down / up / hover, by ref, CSS selector or viewport x/y. Only for gestures the other tools cannot do (sliders, drag handles, canvas, hover menus). For ordinary clicks use browser_click or browser_find - clicking by computed coordinates is fragile.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["move", "down", "up", "hover"],
                    "description": "Mouse action: 'move' cursor, 'down' press button, 'up' release button, 'hover' an element"
                },
                "ref": {
                    "type": "string",
                    "description": "Element reference from the snapshot (e.g., '@e5') for move/hover targets"
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for move/hover targets (alternative to ref)"
                },
                "x": {
                    "type": "number",
                    "description": "Viewport x coordinate (alternative to ref/selector, for 'move')"
                },
                "y": {
                    "type": "number",
                    "description": "Viewport y coordinate (alternative to ref/selector, for 'move')"
                },
                "button": {
                    "type": "string",
                    "enum": ["left", "middle", "right"],
                    "description": "Mouse button for down/up (default 'left')"
                }
            },
            "required": ["action"]
        }
    },
    "browser_mouse_wheel": {
        "name": "browser_mouse_wheel",
        "description": (
            "Mouse-wheel scroll a nested scrollable container (virtualised feed, "
            "chat list, dropdown, map) that browser_scroll doesn't move. Target an "
            "element ref, x/y, or the viewport centre; positive delta_y scrolls down. "
            "Requires browser_navigate first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delta_y": {"type": "number", "description": "Vertical wheel delta in pixels (positive = down, negative = up). Default 0."},
                "delta_x": {"type": "number", "description": "Horizontal wheel delta in pixels. Default 0."},
                "ref": {"type": "string", "description": "Element ref (e.g. 'e22') to hover before wheeling; the wheel fires at its centre. Preferred over x/y."},
                "x": {"type": "number", "description": "Explicit x viewport coordinate (ignored if ref is set)."},
                "y": {"type": "number", "description": "Explicit y viewport coordinate (ignored if ref is set)."}
            },
            "required": []
        }
    },
    "browser_drag": {
        "name": "browser_drag",
        "description": (
            "Real mouse drag (press -> move -> release) for drag-and-drop, sliders, "
            "slider/puzzle CAPTCHAs, and canvas widgets. Start and end points each given "
            "as a snapshot ref, CSS selector, or viewport x/y coords. Optional waypoints "
            "for curved paths; humanize for eased motion. Requires browser_navigate first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_ref": {"type": "string", "description": "Start element ref from the snapshot (e.g. '@e5')."},
                "to_ref": {"type": "string", "description": "End/drop element ref from the snapshot (e.g. '@e9')."},
                "from_selector": {"type": "string", "description": "Start element CSS selector (dragged from its center)."},
                "to_selector": {"type": "string", "description": "End/drop element CSS selector (dragged to its center)."},
                "from_x": {"type": "number", "description": "Start X in viewport pixels (use with from_y)."},
                "from_y": {"type": "number", "description": "Start Y in viewport pixels (use with from_x)."},
                "to_x": {"type": "number", "description": "End X in viewport pixels (use with to_y)."},
                "to_y": {"type": "number", "description": "End Y in viewport pixels (use with to_x)."},
                "waypoints": {
                    "type": "array",
                    "description": "Optional explicit path between press and release, as a list of {x,y} points.",
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    },
                },
                "steps": {"type": "integer", "description": "Interpolation steps per move segment (default 20)."},
                "hold_ms": {"type": "integer", "description": "Pause after pressing the mouse, ms (default 120)."},
                "release_delay_ms": {"type": "integer", "description": "Pause before releasing, ms (default 120)."},
                "humanize": {"type": "boolean", "description": "Eased motion with slight jitter (default false)."},
                "button": {"type": "string", "enum": ["left", "middle", "right"], "description": "Mouse button (default left)."}
            },
            "required": []
        }
    },
})


def _fmt_coord(value: float) -> str:
    """Format a coordinate/delta for agent-browser mouse commands.

    agent-browser's `mouse move`/`mouse wheel` argument parser rejects
    floating-point strings ("100.0" -> 'missing arguments'); it only accepts
    integers.  Round to the nearest whole pixel.
    """
    return str(int(round(float(value))))


def _resolve_viewport_point(
    task_id: str,
    ref: Optional[str] = None,
    selector: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
) -> tuple:
    """Resolve a target to viewport coordinates (center of the element box).

    Returns (x, y, None) on success or (None, None, error_message) on failure.
    Explicit coordinates win over ref/selector.
    """
    if x is not None and y is not None:
        return float(x), float(y), None
    target = bt._normalize_ref(ref) if ref else (selector or "").strip()
    if not target:
        return None, None, "Provide a ref, a selector, or x/y coordinates"
    result = bt._run_browser_command(task_id, "get", ["box", target])
    if not result.get("success"):
        return None, None, result.get("error", f"Could not resolve position of {target}")
    box = result.get("data", {}) or {}
    try:
        cx = float(box["x"]) + float(box.get("width", 0)) / 2
        cy = float(box["y"]) + float(box.get("height", 0)) / 2
    except (KeyError, TypeError, ValueError):
        return None, None, f"Element {target} returned no usable bounding box"
    return round(cx, 1), round(cy, 1), None


def browser_drag(
    from_ref: Optional[str] = None,
    to_ref: Optional[str] = None,
    from_selector: Optional[str] = None,
    to_selector: Optional[str] = None,
    from_x: Optional[float] = None,
    from_y: Optional[float] = None,
    to_x: Optional[float] = None,
    to_y: Optional[float] = None,
    waypoints: Optional[list] = None,
    steps: Optional[int] = None,
    hold_ms: Optional[int] = None,
    release_delay_ms: Optional[int] = None,
    humanize: Optional[bool] = None,
    button: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:
    """
    Perform a real mouse drag (press -> move -> release).

    General-purpose: drag-and-drop, sliders, slider/puzzle CAPTCHAs, and canvas
    widgets. Provide a start and an end point, each as a ref, a CSS selector, or
    absolute viewport pixel coordinates.

    Returns:
        JSON string with the drag result (resolved from/to coordinates).
    """
    if bt._is_camofox_mode():
        from tools.browser_camofox import camofox_drag
        return camofox_drag(
            from_ref=from_ref, to_ref=to_ref,
            from_selector=from_selector, to_selector=to_selector,
            from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y,
            waypoints=waypoints, steps=steps, hold_ms=hold_ms,
            release_delay_ms=release_delay_ms, humanize=humanize,
            button=button, task_id=task_id,
        )

    effective_task_id = bt._last_session_key(task_id or "default")
    blocked = bt._blocked_private_page_action(effective_task_id, "drag")
    if blocked is not None:
        return blocked

    # Element-to-element drag with no coordinate/path tuning maps straight
    # onto the native agent-browser `drag <src> <dst>` command.
    coords_given = any(v is not None for v in (from_x, from_y, to_x, to_y))
    if not coords_given and not waypoints:
        src = bt._normalize_ref(from_ref) if from_ref else from_selector
        dst = bt._normalize_ref(to_ref) if to_ref else to_selector
        if src and dst:
            result = bt._run_browser_command(effective_task_id, "drag", [src, dst])
            if result.get("success"):
                return json.dumps({"success": True, "from": src, "to": dst},
                                  ensure_ascii=False)
            return json.dumps({
                "success": False,
                "error": result.get("error", f"Failed to drag {src} to {dst}"),
            }, ensure_ascii=False)

    # Coordinate path (sliders, canvas, CAPTCHAs): synthesize the drag with
    # raw mouse events — move, press, interpolated moves, release.
    fx, fy, err = _resolve_viewport_point(
        effective_task_id, ref=from_ref, selector=from_selector, x=from_x, y=from_y)
    if err:
        return json.dumps({"success": False, "error": f"drag start: {err}"},
                          ensure_ascii=False)
    tx, ty, err = _resolve_viewport_point(
        effective_task_id, ref=to_ref, selector=to_selector, x=to_x, y=to_y)
    if err:
        return json.dumps({"success": False, "error": f"drag end: {err}"},
                          ensure_ascii=False)

    anchors = [(fx, fy)]
    for wp in (waypoints or []):
        try:
            if isinstance(wp, dict):
                anchors.append((float(wp["x"]), float(wp["y"])))
            else:
                anchors.append((float(wp[0]), float(wp[1])))
        except (KeyError, IndexError, TypeError, ValueError):
            return json.dumps({
                "success": False,
                "error": f"Invalid waypoint {wp!r} (want {{'x': .., 'y': ..}})",
            }, ensure_ascii=False)
    anchors.append((tx, ty))

    n_steps = max(2, int(steps) if steps else 12)
    btn_args = [button] if button else []

    def _mouse(action: str, extra: List[str]) -> Dict[str, Any]:
        return bt._run_browser_command(effective_task_id, "mouse", [action] + extra)

    import time as _time
    sequence_error = None
    result = _mouse("move", [_fmt_coord(fx), _fmt_coord(fy)])
    if result.get("success"):
        result = _mouse("down", btn_args)
    if result.get("success"):
        if hold_ms:
            _time.sleep(min(int(hold_ms), 5000) / 1000)
        total_legs = len(anchors) - 1
        steps_per_leg = max(1, n_steps // total_legs)
        for i in range(total_legs):
            ax, ay = anchors[i]
            bx, by = anchors[i + 1]
            for s in range(1, steps_per_leg + 1):
                t = s / steps_per_leg
                mx, my = ax + (bx - ax) * t, ay + (by - ay) * t
                result = _mouse("move", [_fmt_coord(mx), _fmt_coord(my)])
                if not result.get("success"):
                    sequence_error = result.get("error")
                    break
            if sequence_error:
                break
        if not sequence_error and release_delay_ms:
            _time.sleep(min(int(release_delay_ms), 5000) / 1000)
    else:
        sequence_error = result.get("error")
    # Always release the button, even after a mid-drag failure — a stuck
    # pressed button corrupts every subsequent click in the session.
    up_result = _mouse("up", btn_args)
    if sequence_error is None and not result.get("success"):
        sequence_error = result.get("error")
    if sequence_error is None and not up_result.get("success"):
        sequence_error = up_result.get("error")

    if sequence_error:
        return json.dumps({"success": False, "error": f"Drag failed: {sequence_error}"},
                          ensure_ascii=False)
    return json.dumps({
        "success": True,
        "from": {"x": fx, "y": fy},
        "to": {"x": tx, "y": ty},
        "steps": n_steps,
    }, ensure_ascii=False)


def browser_mouse_wheel(
    delta_y: Optional[float] = None,
    delta_x: Optional[float] = None,
    ref: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    task_id: Optional[str] = None,
) -> str:
    """
    Scroll a nested/inner scrollable container with a real mouse wheel event.

    Targets an element ref (preferred), explicit x/y viewport coordinates, or the
    viewport centre. Use for virtualised feeds, chat/message lists, dropdowns and
    maps that ignore page-level browser_scroll.

    Returns:
        JSON string with the resolved wheel coordinates.
    """
    if bt._is_camofox_mode():
        from tools.browser_camofox import camofox_mouse_wheel
        return camofox_mouse_wheel(
            delta_y=delta_y or 0, delta_x=delta_x or 0,
            ref=ref, x=x, y=y, task_id=task_id,
        )

    effective_task_id = bt._last_session_key(task_id or "default")
    blocked = bt._blocked_private_page_action(effective_task_id, "scroll")
    if blocked is not None:
        return blocked

    # Position the cursor over the target first — wheel events land at the
    # current mouse position, and inner scrollables only react when hovered.
    px, py = None, None
    if ref or (x is not None and y is not None):
        px, py, err = _resolve_viewport_point(effective_task_id, ref=ref, x=x, y=y)
        if err:
            return json.dumps({"success": False, "error": err}, ensure_ascii=False)
        move_result = bt._run_browser_command(
            effective_task_id, "mouse", ["move", _fmt_coord(px), _fmt_coord(py)])
        if not move_result.get("success"):
            return json.dumps({
                "success": False,
                "error": move_result.get("error", "Failed to move mouse to target"),
            }, ensure_ascii=False)

    result = bt._run_browser_command(
        effective_task_id, "mouse",
        ["wheel", _fmt_coord(delta_y or 0), _fmt_coord(delta_x or 0)])
    if result.get("success"):
        response = {"success": True, "delta_y": delta_y or 0, "delta_x": delta_x or 0}
        if px is not None:
            response["at"] = {"x": px, "y": py}
        return json.dumps(response, ensure_ascii=False)
    return json.dumps({
        "success": False,
        "error": result.get("error", "Mouse wheel failed"),
    }, ensure_ascii=False)


def _normalize_upload_target(target: str) -> str:
    """Accept either an @eN ref or a raw selector for browser_upload."""
    stripped = (target or "").strip()
    if not stripped:
        return stripped
    if stripped.startswith("@"):
        return stripped
    if re.fullmatch(r"e\d+", stripped):
        return f"@{stripped}"
    return stripped


def _normalize_upload_paths(path: Optional[str] = None, paths: Optional[List[str]] = None) -> tuple[list[str], Optional[str]]:
    """Validate and normalize upload file paths."""
    raw_paths: list[str] = []
    if path:
        raw_paths.append(path)
    if paths:
        raw_paths.extend(paths)

    if not raw_paths:
        return [], "No files provided. Pass 'path' or 'paths'."

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_paths:
        if not isinstance(raw, str) or not raw.strip():
            return [], "Invalid file path: empty path entries are not allowed."
        candidate = str(Path(raw).expanduser().resolve())
        if not os.path.exists(candidate):
            return [], f"Upload file not found: {candidate}"
        if not os.path.isfile(candidate):
            return [], f"Upload path is not a file: {candidate}"
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)

    if not normalized:
        return [], "No valid files provided for upload."
    return normalized, None


def _default_browser_download_path() -> Path:
    """Return a persistent default file path for browser downloads."""
    import uuid

    downloads_dir = get_hermes_home() / "browser_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    return downloads_dir / f"download_{int(time.time())}_{uuid.uuid4().hex[:8]}.bin"


def _normalize_download_path(path: Optional[str]) -> Path:
    """Resolve an explicit or default download destination to an absolute path."""
    target = Path(path).expanduser() if path else _default_browser_download_path()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _camofox_tab_list(task_id: Optional[str]) -> tuple[list[dict[str, Any]], Optional[int], dict[str, Any]]:
    """Return normalized Camofox tabs plus active index and session info."""
    from tools.browser_camofox import _get, _get_session

    session = _get_session(task_id or "default")
    data = _get("/tabs", params={"userId": session["user_id"]})
    raw_tabs = data.get("tabs", []) if isinstance(data, dict) else []

    tabs: list[dict[str, Any]] = []
    active_index: Optional[int] = None
    active_tab_id = session.get("tab_id")
    for pos, item in enumerate(raw_tabs, start=1):
        tab_id = item.get("tabId") if isinstance(item, dict) else None
        is_active = bool(active_tab_id and tab_id == active_tab_id)
        tabs.append({
            "index": pos,
            "title": item.get("title", "") if isinstance(item, dict) else str(item),
            "url": item.get("url", "") if isinstance(item, dict) else "",
            "active": is_active,
            "tab_id": tab_id,
        })
        if is_active:
            active_index = pos

    return tabs, active_index, session


def _default_browser_download_dir() -> Path:
    downloads_dir = get_hermes_home() / "browser_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    return downloads_dir


def _choose_download_target(path: Optional[str], suggested_filename: Optional[str] = None) -> Path:
    if path:
        return _normalize_download_path(path)
    filename = (suggested_filename or "download.bin").strip() or "download.bin"
    filename = filename.replace("/", "_").replace("\\", "_")
    target = _default_browser_download_dir() / filename
    if not target.exists():
        return target.resolve()
    stem = target.stem
    suffix = target.suffix
    for i in range(1, 1000):
        candidate = target.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate.resolve()
    return _normalize_download_path(None)


def _camofox_browser_tab(action: str, index: Optional[int], url: Optional[str], task_id: Optional[str]) -> str:
    from tools.browser_camofox import _delete, _get_session, _post

    validated_index = int(index) if index is not None else None
    if action == "list":
        tabs, active_index, _session = _camofox_tab_list(task_id)
        for tab in tabs:
            tab.pop("tab_id", None)
        return json.dumps({"success": True, "tabs": tabs, "active_index": active_index}, ensure_ascii=False)

    session = _get_session(task_id or "default")

    if action == "new":
        data = _post(
            "/tabs",
            {
                "userId": session["user_id"],
                "sessionKey": session["session_key"],
                "url": url or "about:blank",
            },
            timeout=max(bt._get_command_timeout(), 60),
        )
        if isinstance(data, dict) and data.get("tabId"):
            session["tab_id"] = data["tabId"]
        response = {"success": True, "action": "new"}
        if url:
            response["url"] = url
        return json.dumps(response, ensure_ascii=False)

    tabs, active_index, session = _camofox_tab_list(task_id)
    if action == "switch":
        if validated_index is None or validated_index < 1 or validated_index > len(tabs):
            return json.dumps({"success": False, "error": "Tab index out of range."}, ensure_ascii=False)
        session["tab_id"] = tabs[validated_index - 1].get("tab_id")
        return json.dumps({"success": True, "action": "switch", "active_index": validated_index}, ensure_ascii=False)

    target_tab = None
    if validated_index is not None:
        if validated_index < 1 or validated_index > len(tabs):
            return json.dumps({"success": False, "error": "Tab index out of range."}, ensure_ascii=False)
        target_tab = tabs[validated_index - 1]
    else:
        target_tab = next((tab for tab in tabs if tab.get("active")), tabs[-1] if tabs else None)

    if not target_tab or not target_tab.get("tab_id"):
        return json.dumps({"success": False, "error": "No tab available to close."}, ensure_ascii=False)

    _delete(f"/tabs/{target_tab['tab_id']}", body={"userId": session["user_id"]}, timeout=max(bt._get_command_timeout(), 60))
    remaining_tabs, remaining_active_index, session = _camofox_tab_list(task_id)
    if session.get("tab_id") == target_tab.get("tab_id"):
        session["tab_id"] = remaining_tabs[-1].get("tab_id") if remaining_tabs else None
        for pos, item in enumerate(remaining_tabs, start=1):
            item["active"] = item.get("tab_id") == session.get("tab_id")
            if item["active"]:
                remaining_active_index = pos

    response = {"success": True, "action": "close"}
    if validated_index is not None:
        response["closed_index"] = validated_index
    else:
        response["closed_active"] = True
    response["active_index"] = remaining_active_index
    return json.dumps(response, ensure_ascii=False)


def _camofox_browser_download(ref: str, path: Optional[str], task_id: Optional[str]) -> str:
    import base64
    from tools.browser_camofox import _ensure_tab, _get, _post

    session = _ensure_tab(task_id or "default")
    tab_id = session.get("tab_id")
    normalized_ref = bt._normalize_ref(ref).lstrip("@")

    _post(
        f"/tabs/{tab_id}/click",
        {"userId": session["user_id"], "ref": normalized_ref},
        timeout=max(bt._get_command_timeout(), 60),
    )

    deadline = time.time() + 20
    last_downloads: list[dict[str, Any]] = []
    while time.time() < deadline:
        data = _get(
            f"/tabs/{tab_id}/downloads",
            params={"userId": session["user_id"], "includeData": "true", "consume": "true"},
            timeout=max(bt._get_command_timeout(), 60),
        )
        last_downloads = data.get("downloads", []) if isinstance(data, dict) else []
        if last_downloads:
            break
        time.sleep(0.5)

    if not last_downloads:
        return json.dumps({"success": False, "error": "No download was captured after clicking the target element."}, ensure_ascii=False)

    first = last_downloads[0]
    if first.get("failure"):
        return json.dumps({"success": False, "error": f"Download failed: {first['failure']}"}, ensure_ascii=False)
    raw_b64 = first.get("dataBase64")
    if not raw_b64:
        return json.dumps({
            "success": False,
            "error": "Download metadata was captured, but file bytes were unavailable from Camofox.",
            "downloads": last_downloads,
        }, ensure_ascii=False)

    target_path = _choose_download_target(path, first.get("suggestedFilename"))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(base64.b64decode(raw_b64))
    return json.dumps({
        "success": True,
        "path": str(target_path),
        "exists": True,
        "element": bt._normalize_ref(ref),
    }, ensure_ascii=False)


def browser_tab(action: str, index: Optional[int] = None, url: Optional[str] = None, task_id: Optional[str] = None,
                tab_id: Optional[str] = None) -> str:
    """Manage browser tabs using a single multiplexed tool."""
    if bt._is_camofox_mode():
        return _camofox_browser_tab(action=action, index=index, url=url, task_id=task_id)

    action = (action or "").strip().lower()
    if action not in {"new", "list", "switch", "close"}:
        return json.dumps({"success": False, "error": f"Invalid action '{action}'."}, ensure_ascii=False)

    validated_index: Optional[int] = None
    if index is not None:
        # Models sometimes pass the id ("t2") in `index`; accept it there too.
        if isinstance(index, str) and not index.strip().isdigit():
            tab_id = tab_id or index.strip()
        else:
            try:
                validated_index = int(index)
            except (TypeError, ValueError):
                return json.dumps({"success": False, "error": f"Invalid tab index {index!r}."}, ensure_ascii=False)
    if action == "switch" and tab_id is None and (validated_index is None or validated_index < 1):
        return json.dumps({"success": False, "error": "browser_tab(action='switch') requires a 1-based index (see action='list')."}, ensure_ascii=False)
    if action == "close" and validated_index is not None and validated_index < 1:
        return json.dumps({"success": False, "error": "browser_tab(action='close') index must be >= 1."}, ensure_ascii=False)

    effective_task_id = bt._last_session_key(task_id or "default")

    # agent-browser >= 0.26 addresses tabs by stable ids ("t2") or labels and
    # rejects bare integers ("positional integers are not accepted").  The
    # model keeps the simple 1-based position shown by action='list'; it is
    # translated to the tab id from a fresh listing right before use, so a
    # tab opened or closed in between cannot shift the target silently.
    def _listing() -> tuple[Optional[list], Optional[int], Dict[str, Any]]:
        res = bt._run_browser_command(effective_task_id, "tab", ["list"])
        if not res.get("success"):
            return None, None, res
        tabs_, active_ = bt._normalize_tab_payload(res.get("data", {}))
        return tabs_, active_, res

    def _resolve(tabs_: list) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        if tab_id:
            wanted = str(tab_id).strip()
            for t in tabs_:
                if wanted in (t.get("id"), t.get("label")):
                    return t, None
            return None, f"No tab with id '{wanted}'. Open tabs: " + ", ".join(
                f"{t['index']}={t.get('id')}" for t in tabs_)
        if validated_index is None:
            return None, None
        if validated_index > len(tabs_):
            return None, f"Tab index {validated_index} is out of range: {len(tabs_)} tab(s) open."
        return tabs_[validated_index - 1], None

    if action == "list":
        tabs, active_index, result = _listing()
        if tabs is None:
            response = {"success": False, "error": result.get("error", "Failed to list tabs")}
            return json.dumps(bt._copy_fallback_warning(response, result), ensure_ascii=False)
        response = {"success": True, "tabs": tabs, "active_index": active_index}
        return json.dumps(bt._copy_fallback_warning(response, result), ensure_ascii=False)

    if action == "new":
        create_result = bt._run_browser_command(effective_task_id, "tab", ["new"] + ([url] if url else []),
                                             timeout=max(bt._get_command_timeout(), 60) if url else None)
        if not create_result.get("success"):
            response = {"success": False, "error": create_result.get("error", "Failed to create new tab")}
            return json.dumps(bt._copy_fallback_warning(response, create_result), ensure_ascii=False)
        tabs, active_index, _ = _listing()
        response = {"success": True, "action": "new"}
        if url:
            response["url"] = url
        if tabs:
            response["active_index"] = active_index
            response["tab_count"] = len(tabs)
        return json.dumps(bt._copy_fallback_warning(response, create_result), ensure_ascii=False)

    tabs, active_index, list_result = _listing()
    if tabs is None:
        response = {"success": False, "error": list_result.get("error", "Failed to list tabs")}
        return json.dumps(bt._copy_fallback_warning(response, list_result), ensure_ascii=False)
    target, err = _resolve(tabs)
    if err:
        return json.dumps({"success": False, "error": err, "tabs": tabs}, ensure_ascii=False)

    if action == "switch":
        result = bt._run_browser_command(effective_task_id, "tab", [target["id"]])
        if not result.get("success"):
            response = {"success": False, "error": result.get("error", f"Failed to switch to tab {target['index']}")}
            return json.dumps(bt._copy_fallback_warning(response, result), ensure_ascii=False)
        response = {"success": True, "action": "switch", "active_index": target["index"],
                    "title": target.get("title", ""), "url": target.get("url", "")}
        return json.dumps(bt._copy_fallback_warning(response, result), ensure_ascii=False)

    close_args = ["close"]
    if target is not None:
        close_args.append(target["id"])
    result = bt._run_browser_command(effective_task_id, "tab", close_args)
    if not result.get("success"):
        response = {"success": False, "error": result.get("error", "Failed to close tab")}
        return json.dumps(bt._copy_fallback_warning(response, result), ensure_ascii=False)
    response = {"success": True, "action": "close"}
    if target is not None:
        response["closed_index"] = target["index"]
        response["closed_title"] = target.get("title", "")
    else:
        response["closed_active"] = True
    remaining, remaining_active, _ = _listing()
    if remaining is not None:
        response["active_index"] = remaining_active
        response["tabs"] = remaining
    return json.dumps(bt._copy_fallback_warning(response, result), ensure_ascii=False)


def browser_upload(ref: str, path: Optional[str] = None, paths: Optional[List[str]] = None, task_id: Optional[str] = None) -> str:
    """Upload one or more local files via a file input element."""
    # Validate paths/target before dispatching so both backends share identical
    # validation and the LLM-facing response shape is backend-independent.
    normalized_paths, error = _normalize_upload_paths(path=path, paths=paths)
    if error:
        return json.dumps({"success": False, "error": error}, ensure_ascii=False)

    normalized_target = _normalize_upload_target(ref)

    if bt._is_camofox_mode():
        from tools.browser_camofox import camofox_upload
        return camofox_upload(normalized_target, normalized_paths, task_id=task_id)

    effective_task_id = bt._last_session_key(task_id or "default")
    result = bt._run_browser_command(
        effective_task_id,
        "upload",
        [normalized_target, *normalized_paths],
        timeout=max(bt._get_command_timeout(), 60),
    )

    if result.get("success"):
        response = {
            "success": True,
            "element": normalized_target,
            "uploaded_paths": normalized_paths,
        }
        return json.dumps(bt._copy_fallback_warning(response, result), ensure_ascii=False)

    response = {
        "success": False,
        "error": result.get("error", f"Failed to upload files to {normalized_target}"),
    }
    return json.dumps(bt._copy_fallback_warning(response, result), ensure_ascii=False)


def browser_dropzone_upload(selector: Optional[str] = None, path: Optional[str] = None,
                            paths: Optional[List[str]] = None, task_id: Optional[str] = None) -> str:
    """Attach local files to a drag-and-drop uploader (Dropzone.js and similar).

    Use this instead of browser_upload when the target is a "drop files here"
    zone with no usable file input (the file input is hidden/managed by JS).
    ``selector`` is the dropzone element selector (default ``.dropzone``).
    """
    normalized_paths, error = _normalize_upload_paths(path=path, paths=paths)
    if error:
        return json.dumps({"success": False, "error": error}, ensure_ascii=False)

    drop_selector = (selector or ".dropzone").strip() or ".dropzone"

    if bt._is_camofox_mode():
        from tools.browser_camofox import camofox_dropzone_upload
        return camofox_dropzone_upload(drop_selector, normalized_paths, task_id=task_id)

    effective_task_id = bt._last_session_key(task_id or "default")

    # Dropzone.js path (first): a real Dropzone widget keeps its <input
    # type=file> (class ``dz-hidden-input``) on document.body — NOT inside the
    # dropzone element — and ignores synthetic drop events, so the scoped-input
    # and synthetic-drop paths below both miss it. Detect the Dropzone instance
    # for the selector (the element itself, a nested ``.dropzone``, or
    # ``el.dropzone``), tag ITS hidden input, and upload to that. Verified
    # against Portad's justificatif dropzone.
    tag_js = (
        "(() => {"
        "if (typeof window.Dropzone === 'undefined') return JSON.stringify({ok:false, reason:'no-lib'});"
        f"const root = document.querySelector({json.dumps(drop_selector)});"
        "if (!root) return JSON.stringify({ok:false, reason:'no-selector'});"
        "let el = (root.classList && root.classList.contains('dropzone')) ? root : root.querySelector('.dropzone');"
        "let dz = null;"
        "try { if (el && window.Dropzone.forElement) dz = window.Dropzone.forElement(el); } catch(e){}"
        "if (!dz && root.dropzone) dz = root.dropzone;"
        "if (!dz && el && el.dropzone) dz = el.dropzone;"
        "if (!dz) return JSON.stringify({ok:false, reason:'no-instance'});"
        "const inp = dz.hiddenFileInput;"
        "if (!inp) return JSON.stringify({ok:false, reason:'no-hidden-input'});"
        "const prev = document.getElementById('hermes_dz_upload_target');"
        "if (prev && prev !== inp) prev.removeAttribute('id');"
        "inp.id = 'hermes_dz_upload_target';"
        "inp.style.cssText = 'display:block;visibility:visible;opacity:0;position:fixed;left:0;top:0;width:1px;height:1px';"
        "return JSON.stringify({ok:true});"
        "})()"
    )
    tag_result = bt._run_browser_command(effective_task_id, "eval", [tag_js])
    if tag_result.get("success"):
        try:
            tag_outcome = json.loads((tag_result.get("data", {}) or {}).get("result") or "{}")
        except (json.JSONDecodeError, TypeError):
            tag_outcome = {}
        if tag_outcome.get("ok"):
            up = bt._run_browser_command(
                effective_task_id, "upload",
                ["#hermes_dz_upload_target", *normalized_paths],
                timeout=max(bt._get_command_timeout(), 60),
            )
            if up.get("success"):
                # Give Dropzone a moment to run its change handler (accept +,
                # if autoProcessQueue, upload). Report the last file's status.
                status_js = (
                    "(() => {"
                    f"const root = document.querySelector({json.dumps(drop_selector)});"
                    "let el = (root && root.classList && root.classList.contains('dropzone')) ? root : (root && root.querySelector('.dropzone'));"
                    "let dz = null; try { dz = window.Dropzone.forElement(el); } catch(e){}"
                    "if (!dz && root && root.dropzone) dz = root.dropzone;"
                    "const f = dz && dz.files && dz.files[dz.files.length-1];"
                    "return JSON.stringify({n: dz? dz.files.length:0, name: f?f.name:null, status: f?f.status:null});"
                    "})()"
                )
                import time as _t
                dz_status = {}
                for _ in range(10):
                    _t.sleep(0.6)
                    sres = bt._run_browser_command(effective_task_id, "eval", [status_js])
                    try:
                        dz_status = json.loads((sres.get("data", {}) or {}).get("result") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        dz_status = {}
                    if dz_status.get("status") in ("success", "error"):
                        break
                return json.dumps({
                    "success": dz_status.get("status") != "error",
                    "element": drop_selector,
                    "uploaded_paths": normalized_paths,
                    "method": "dropzone_js",
                    "dropzone_status": dz_status.get("status"),
                    "dropzone_file": dz_status.get("name"),
                }, ensure_ascii=False)

    # Dropzone widgets sometimes wrap a hidden <input type=file>. Try that
    # input, scoped to the dropzone so we never upload to an unrelated
    # file input elsewhere on the page. If the selector IS itself a file input,
    # target it directly. No page-wide fallback — a wrong-target upload is
    # worse than falling through to the synthetic drop below.
    candidates = [f"{drop_selector} input[type=file]"]
    if "input" in drop_selector.lower():
        candidates.append(drop_selector)
    upload_errors = []
    for candidate in candidates:
        result = bt._run_browser_command(
            effective_task_id, "upload", [candidate, *normalized_paths],
            timeout=max(bt._get_command_timeout(), 60),
        )
        if result.get("success"):
            return json.dumps({
                "success": True,
                "element": candidate,
                "uploaded_paths": normalized_paths,
                "method": "file_input",
            }, ensure_ascii=False)
        upload_errors.append(f"{candidate}: {result.get('error', 'failed')}")

    # Last resort: synthesize a real drop event with the file bytes embedded
    # as base64.  argv size limits (MAX_ARG_STRLEN ≈ 128 KiB per argument on
    # Linux) cap this path to small files.
    _SYNTH_DROP_MAX_BYTES = 90 * 1024
    total_size = 0
    for p in normalized_paths:
        try:
            total_size += os.path.getsize(p)
        except OSError as e:
            return json.dumps({"success": False, "error": f"Cannot read {p}: {e}"},
                              ensure_ascii=False)
    if total_size > _SYNTH_DROP_MAX_BYTES:
        return json.dumps({
            "success": False,
            "error": (
                "No usable file input found for the dropzone and the files are "
                f"too large ({total_size} bytes) for a synthetic drop event "
                f"(limit {_SYNTH_DROP_MAX_BYTES}). Tried: "
                + "; ".join(upload_errors)
            ),
        }, ensure_ascii=False)

    import base64
    import mimetypes
    files_payload = []
    for p in normalized_paths:
        with open(p, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        mime = mimetypes.guess_type(p)[0] or "application/octet-stream"
        files_payload.append({"name": os.path.basename(p), "mime": mime, "b64": b64})

    drop_js = (
        "(() => {"
        f"const el = document.querySelector({json.dumps(drop_selector)});"
        "if (!el) return JSON.stringify({ok:false, error:'dropzone selector not found'});"
        f"const files = {json.dumps(files_payload)};"
        "const dt = new DataTransfer();"
        "for (const f of files) {"
        "  const bytes = Uint8Array.from(atob(f.b64), c => c.charCodeAt(0));"
        "  dt.items.add(new File([bytes], f.name, {type: f.mime}));"
        "}"
        "for (const type of ['dragenter', 'dragover', 'drop']) {"
        "  el.dispatchEvent(new DragEvent(type, {bubbles: true, cancelable: true, dataTransfer: dt}));"
        "}"
        "return JSON.stringify({ok: true});"
        "})()"
    )
    result = bt._run_browser_command(effective_task_id, "eval", [drop_js])
    if result.get("success"):
        try:
            outcome = json.loads((result.get("data", {}) or {}).get("result") or "{}")
        except (json.JSONDecodeError, TypeError):
            outcome = {}
        if outcome.get("ok"):
            return json.dumps({
                "success": True,
                "element": drop_selector,
                "uploaded_paths": normalized_paths,
                "method": "synthetic_drop",
            }, ensure_ascii=False)
        return json.dumps({
            "success": False,
            "error": outcome.get("error", "Synthetic drop was dispatched but not confirmed"),
        }, ensure_ascii=False)
    return json.dumps({
        "success": False,
        "error": result.get("error", "Synthetic drop failed. Tried: " + "; ".join(upload_errors)),
    }, ensure_ascii=False)


def browser_download(ref: str, path: Optional[str] = None, task_id: Optional[str] = None,
                     timeout: Optional[float] = None) -> str:
    """Download a file by clicking an element and saving it locally."""
    if bt._is_camofox_mode():
        return _camofox_browser_download(ref=ref, path=path, task_id=task_id)

    effective_task_id = bt._last_session_key(task_id or "default")
    normalized_ref = bt._normalize_ref(ref)
    target_path = _normalize_download_path(path)
    try:
        wait_s = float(timeout) if timeout else 60.0
    except (TypeError, ValueError):
        wait_s = 60.0
    wait_s = max(5.0, min(wait_s, 300.0))
    args = [normalized_ref, str(target_path)]
    try:
        from tools.browser_tool_hermes import ab_is_hermes_fork
        if ab_is_hermes_fork():
            # Stock agent-browser hard-codes a 30 s download timeout: all 6
            # Flash-Next download attempts died at exactly 30.1 s.
            args += ["--timeout", str(int(wait_s * 1000))]
    except Exception:
        pass

    result = bt._run_browser_command(
        effective_task_id,
        "download",
        args,
        timeout=int(max(bt._get_command_timeout(), wait_s + 15)),
    )

    if not result.get("success"):
        response = {"success": False, "error": result.get("error", f"Failed to download from {normalized_ref}")}
        return json.dumps(bt._copy_fallback_warning(response, result), ensure_ascii=False)

    if not target_path.exists() or not target_path.is_file():
        response = {
            "success": False,
            "error": f"Download reported success but file was not found at {target_path}",
            "path": str(target_path),
        }
        return json.dumps(bt._copy_fallback_warning(response, result), ensure_ascii=False)

    response = {
        "success": True,
        "path": str(target_path),
        "exists": True,
        "size_bytes": target_path.stat().st_size,
        "element": normalized_ref,
    }
    _data = result.get("data") or {}
    if isinstance(_data, dict) and _data.get("mode"):
        # "download" or "inline_pdf" (a PDF the click opened in the viewer).
        response["mode"] = _data["mode"]
    return json.dumps(bt._copy_fallback_warning(response, result), ensure_ascii=False)


def browser_eval(expression: str, task_id: Optional[str] = None, timeout: Optional[float] = None) -> str:
    """Evaluate JavaScript in the page (SSRF-guarded via _browser_eval)."""
    if not expression or not expression.strip():
        return json.dumps({"success": False, "error": "Empty expression"},
                          ensure_ascii=False)
    # Opt-in denylist (browser.restrict_evaluate); off by default.
    policy_error = bt._enforce_browser_eval_policy(expression)
    if policy_error:
        return json.dumps({"success": False, "error": policy_error}, ensure_ascii=False)
    return bt._browser_eval(expression, task_id, timeout=timeout)


def browser_pdf(path: Optional[str] = None, task_id: Optional[str] = None) -> str:
    """Save the current page as a PDF file."""
    if bt._is_camofox_mode():
        return json.dumps({
            "success": False,
            "error": "browser_pdf is not supported on the Camofox backend.",
        }, ensure_ascii=False)

    effective_task_id = bt._last_session_key(task_id or "default")
    target_path = _normalize_download_path(path)
    if target_path.suffix.lower() != ".pdf":
        target_path = target_path.with_suffix(".pdf")

    result = bt._run_browser_command(
        effective_task_id, "pdf", [str(target_path)],
        timeout=max(bt._get_command_timeout(), 60),
    )
    if not result.get("success"):
        return json.dumps({
            "success": False,
            "error": result.get("error", "Failed to save page as PDF"),
        }, ensure_ascii=False)
    if not target_path.exists() or not target_path.is_file():
        return json.dumps({
            "success": False,
            "error": f"PDF reported success but file was not found at {target_path}",
        }, ensure_ascii=False)
    return json.dumps({
        "success": True,
        "path": str(target_path),
    }, ensure_ascii=False)


def browser_mouse(action: str, ref: Optional[str] = None,
                  selector: Optional[str] = None,
                  x: Optional[float] = None, y: Optional[float] = None,
                  button: Optional[str] = None,
                  task_id: Optional[str] = None) -> str:
    """Low-level mouse control: move, down, up, hover."""
    if bt._is_camofox_mode():
        return json.dumps({
            "success": False,
            "error": "browser_mouse is not supported on the Camofox backend.",
        }, ensure_ascii=False)

    effective_task_id = bt._last_session_key(task_id or "default")
    blocked = bt._blocked_private_page_action(effective_task_id, "mouse")
    if blocked is not None:
        return blocked

    action = (action or "").strip().lower()
    if action == "hover":
        target = bt._normalize_ref(ref) if ref else (selector or "").strip()
        if not target:
            return json.dumps({
                "success": False,
                "error": "hover requires a ref or a selector",
            }, ensure_ascii=False)
        result = bt._run_browser_command(effective_task_id, "hover", [target])
        if result.get("success"):
            return json.dumps({"success": True, "action": "hover", "element": target},
                              ensure_ascii=False)
        return json.dumps({
            "success": False,
            "error": result.get("error", f"Failed to hover {target}"),
        }, ensure_ascii=False)

    if action == "move":
        px, py, err = _resolve_viewport_point(
            effective_task_id, ref=ref, selector=selector, x=x, y=y)
        if err:
            return json.dumps({"success": False, "error": err}, ensure_ascii=False)
        result = bt._run_browser_command(
            effective_task_id, "mouse", ["move", _fmt_coord(px), _fmt_coord(py)])
        if result.get("success"):
            return json.dumps({"success": True, "action": "move", "x": px, "y": py},
                              ensure_ascii=False)
        return json.dumps({
            "success": False,
            "error": result.get("error", "Mouse move failed"),
        }, ensure_ascii=False)

    if action in ("down", "up"):
        args = [action] + ([button] if button else [])
        result = bt._run_browser_command(effective_task_id, "mouse", args)
        if result.get("success"):
            return json.dumps({"success": True, "action": action,
                               "button": button or "left"}, ensure_ascii=False)
        return json.dumps({
            "success": False,
            "error": result.get("error", f"Mouse {action} failed"),
        }, ensure_ascii=False)

    return json.dumps({
        "success": False,
        "error": f"Unknown mouse action {action!r} (want move, down, up, or hover)",
    }, ensure_ascii=False)


registry.register(
    name="browser_drag",
    toolset="browser",
    schema=SCHEMAS["browser_drag"],
    handler=lambda args, **kw: browser_drag(
        from_ref=args.get("from_ref"), to_ref=args.get("to_ref"),
        from_selector=args.get("from_selector"), to_selector=args.get("to_selector"),
        from_x=args.get("from_x"), from_y=args.get("from_y"),
        to_x=args.get("to_x"), to_y=args.get("to_y"),
        waypoints=args.get("waypoints"), steps=args.get("steps"),
        hold_ms=args.get("hold_ms"), release_delay_ms=args.get("release_delay_ms"),
        humanize=args.get("humanize"), button=args.get("button"),
        task_id=kw.get("task_id"),
    ),
    check_fn=bt.check_browser_requirements,
    emoji="🖐️",
)


registry.register(
    name="browser_mouse_wheel",
    toolset="browser",
    schema=SCHEMAS["browser_mouse_wheel"],
    handler=lambda args, **kw: browser_mouse_wheel(
        delta_y=args.get("delta_y"), delta_x=args.get("delta_x"),
        ref=args.get("ref"), x=args.get("x"), y=args.get("y"),
        task_id=kw.get("task_id"),
    ),
    check_fn=bt.check_browser_requirements,
    emoji="🖱️",
)


registry.register(
    name="browser_tab",
    toolset="browser",
    schema=SCHEMAS["browser_tab"],
    handler=lambda args, **kw: browser_tab(
        action=args.get("action", ""),
        index=args.get("index"),
        url=args.get("url"),
        task_id=kw.get("task_id"),
        tab_id=args.get("tab_id"),
    ),
    check_fn=bt.check_browser_requirements,
    emoji="🗂️",
)


registry.register(
    name="browser_upload",
    toolset="browser",
    schema=SCHEMAS["browser_upload"],
    handler=lambda args, **kw: browser_upload(
        ref=args.get("ref", ""),
        path=args.get("path"),
        paths=args.get("paths"),
        task_id=kw.get("task_id"),
    ),
    check_fn=bt.check_browser_requirements,
    emoji="📤",
)


registry.register(
    name="browser_dropzone_upload",
    toolset="browser",
    schema=SCHEMAS["browser_dropzone_upload"],
    handler=lambda args, **kw: browser_dropzone_upload(
        selector=args.get("selector"),
        path=args.get("path"),
        paths=args.get("paths"),
        task_id=kw.get("task_id"),
    ),
    check_fn=bt.check_browser_requirements,
    emoji="📤",
)


registry.register(
    name="browser_download",
    toolset="browser",
    schema=SCHEMAS["browser_download"],
    handler=lambda args, **kw: browser_download(
        ref=args.get("ref", ""),
        path=args.get("path"),
        task_id=kw.get("task_id"),
        timeout=args.get("timeout"),
    ),
    check_fn=bt.check_browser_requirements,
    emoji="📥",
)


registry.register(
    name="browser_eval",
    toolset="browser",
    schema=SCHEMAS["browser_eval"],
    handler=lambda args, **kw: browser_eval(
        expression=args.get("expression", ""),
        task_id=kw.get("task_id"),
        timeout=args.get("timeout"),
    ),
    check_fn=bt.check_browser_requirements,
    emoji="🧪",
)


registry.register(
    name="browser_pdf",
    toolset="browser",
    schema=SCHEMAS["browser_pdf"],
    handler=lambda args, **kw: browser_pdf(
        path=args.get("path"),
        task_id=kw.get("task_id"),
    ),
    check_fn=bt.check_browser_requirements,
    emoji="📄",
)


registry.register(
    name="browser_mouse",
    toolset="browser",
    schema=SCHEMAS["browser_mouse"],
    handler=lambda args, **kw: browser_mouse(
        action=args.get("action", ""),
        ref=args.get("ref"),
        selector=args.get("selector"),
        x=args.get("x"),
        y=args.get("y"),
        button=args.get("button"),
        task_id=kw.get("task_id"),
    ),
    check_fn=bt.check_browser_requirements,
    emoji="🖱️",
)


SCHEMAS["browser_fill_form"] = {
    "name": "browser_fill_form",
    "description": (
        "Fill a whole form in ONE call, then optionally submit it: each field by its ref (from the snapshot), "
        "CSS selector, visible label or placeholder, with the text to enter (true/false ticks a checkbox; a "
        "<select> gets the matching option). Submit with submit_ref (the button's ref), submit_text (the "
        "button's text) or press_enter. Nothing is submitted if a field fails. The result shows what changed. "
        "Prefer this to several browser_type calls - each extra call is a whole extra turn."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "description": "Fields to fill, in order",
                "items": {
                    "type": "object",
                    "properties": {
                        "ref": {"type": "string", "description": "Field ref from the snapshot (e.g. '@e6')"},
                        "selector": {"type": "string", "description": "CSS selector of the field"},
                        "label": {"type": "string", "description": "Visible label of the field"},
                        "placeholder": {"type": "string", "description": "Placeholder text of the field"},
                        "value": {"description": "Text to enter, or true/false for a checkbox"},
                    },
                    "required": ["value"],
                },
            },
            "submit_ref": {"type": "string", "description": "Ref of the submit button to click after filling"},
            "submit_text": {"type": "string", "description": "Text of the submit button to click after filling"},
            "press_enter": {"type": "boolean", "description": "Press Enter after filling (submits most forms)"},
        },
        "required": ["fields"],
    },
}

registry.register(
    name="browser_fill_form", toolset="browser", schema=SCHEMAS["browser_fill_form"],
    handler=lambda args, **kw: browser_fill_form(
        fields=args.get("fields") or [], submit_ref=args.get("submit_ref"), submit_text=args.get("submit_text"),
        press_enter=bool(args.get("press_enter")), task_id=kw.get("task_id")),
    check_fn=bt.check_browser_requirements, emoji="📝",
)


def _extend_navigate_schema() -> None:
    """browser_navigate(read=true) also returns the page text (saves a browser_read turn)."""
    entry = registry.get_entry("browser_navigate")
    if entry is None:
        return
    props = entry.schema.setdefault("parameters", {}).setdefault("properties", {})
    props.setdefault("read", {
        "type": "boolean",
        "description": ("Also return the page as clean text (headings, paragraphs, tables) in `text` - use it "
                        "when the task is to READ the page, instead of a separate browser_read call"),
    })


NAVIGATE_TEXT_MAX_CHARS = 12000


def _install_wrappers() -> None:
    """Wrap every registered browser_* handler with observation + progress guard."""
    for name in list(getattr(registry, "_tools", {}).keys()):
        if not name.startswith("browser_"):
            continue
        entry = registry.get_entry(name)
        # Other layers (the registry itself) already set __wrapped__ through
        # functools.wraps, so mark our own wrapper explicitly.
        if entry is None or getattr(entry.handler, "_hermes_browser_wrapped", False) or entry.is_async:
            continue
        entry.handler = _wrap_handler(name, entry.handler)


_extend_navigate_schema()
_install_wrappers()
