"""Agent-facing tool: respond to a native JS dialog captured by the CDP supervisor.

This tool is response-only — the agent first reads ``pending_dialogs`` from
``browser_snapshot`` output, then calls ``browser_dialog(action=...)`` to
accept or dismiss.

Gated on the same ``_browser_cdp_check`` as ``browser_cdp`` so it only
appears when a CDP endpoint is reachable (Browserbase with a
``connectUrl``, local Chromium-family browser via ``/browser connect``, or
``browser.cdp_url`` set in config).

See ``website/docs/developer-guide/browser-supervisor.md`` for the full
design.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from tools.browser_supervisor import SUPERVISOR_REGISTRY
from tools.registry import registry

logger = logging.getLogger(__name__)


BROWSER_DIALOG_SCHEMA: Dict[str, Any] = {
    "name": "browser_dialog",
    "description": (
        "Respond to a native JavaScript dialog (alert / confirm / prompt / "
        "beforeunload) that is currently blocking the page.\n\n"
        "**Workflow:** call ``browser_snapshot`` first — if a dialog is open, "
        "it appears in the ``pending_dialogs`` field with ``id``, ``type``, "
        "and ``message``. Then call this tool with ``action='accept'`` or "
        "``action='dismiss'``.\n\n"
        "**Prompt dialogs:** pass ``prompt_text`` to supply the response "
        "string. Ignored for alert/confirm/beforeunload.\n\n"
        "**Multiple dialogs:** if more than one dialog is queued (rare — "
        "happens when a second dialog fires while the first is still open), "
        "pass ``dialog_id`` from the snapshot to disambiguate.\n\n"
        "**Availability:** only present when a CDP-capable backend is "
        "attached — Browserbase sessions, local Chromium-family browser via "
        "``/browser connect``, or ``browser.cdp_url`` in config.yaml. "
        "Not available on Camofox (REST-only) or the default Playwright "
        "local browser (CDP port is hidden)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["accept", "dismiss"],
                "description": (
                    "'accept' clicks OK / returns the prompt text. "
                    "'dismiss' clicks Cancel / returns null from prompt(). "
                    "For ``beforeunload`` dialogs: 'accept' allows the "
                    "navigation, 'dismiss' keeps the page."
                ),
            },
            "prompt_text": {
                "type": "string",
                "description": (
                    "Response string for a ``prompt()`` dialog. Ignored for "
                    "other dialog types. Defaults to empty string."
                ),
            },
            "dialog_id": {
                "type": "string",
                "description": (
                    "Specific dialog to respond to, from "
                    "``browser_snapshot.pending_dialogs[].id``. Required "
                    "only when multiple dialogs are queued."
                ),
            },
        },
        "required": ["action"],
    },
}


def _agent_browser_dialog(action: str, prompt_text: Optional[str], task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Answer the dialog through agent-browser (``dialog accept [text]`` / ``dialog dismiss``).

    hermes-mods: the CDP supervisor listens on ONE tab (the first one it
    attached to), so a dialog opened in any other tab is invisible to it,
    while agent-browser sees dialogs on the tab it acts on.
    """
    try:
        from tools import browser_tool as bt
    except Exception:
        return None
    verb = "accept" if action == "accept" else "dismiss"
    args = [verb] + ([prompt_text] if verb == "accept" and prompt_text is not None else [])
    res = bt._run_browser_command(bt._last_session_key(task_id or "default"), "dialog", args, timeout=15)
    if res.get("success"):
        data = dict(res.get("data") or {})
        data.pop("lifecycle", None)
        return {"success": True, "action": verb, "via": "agent-browser", "dialog": data}
    return {"success": False, "error": res.get("error", "dialog command failed")}


def browser_dialog(
    action: str,
    prompt_text: Optional[str] = None,
    dialog_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:
    """Respond to a pending dialog (CDP supervisor first, agent-browser otherwise)."""
    effective_task_id = task_id or "default"
    supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
    if supervisor is None:
        try:
            from tools import browser_tool as bt
            supervisor = SUPERVISOR_REGISTRY.get(bt._last_session_key(effective_task_id))
        except Exception:
            supervisor = None
    sup_error = None
    if supervisor is not None:
        result = supervisor.respond_to_dialog(
            action=action,
            prompt_text=prompt_text,
            dialog_id=dialog_id,
        )
        if result.get("ok"):
            return json.dumps(
                {
                    "success": True,
                    "action": action,
                    "dialog": result.get("dialog", {}),
                }
            )
        sup_error = result.get("error", "unknown error")
    fallback = _agent_browser_dialog(action, prompt_text, task_id)
    if fallback is not None and fallback.get("success"):
        return json.dumps(fallback, ensure_ascii=False)
    error = (fallback or {}).get("error") or sup_error or (
        "No dialog is open and no browser session exists yet. Call browser_navigate first."
    )
    return json.dumps({"success": False, "error": error}, ensure_ascii=False)


def _browser_dialog_check() -> bool:
    """Gate: same as ``browser_cdp`` — only offered when CDP is reachable.

    Kept identical so the two tools appear and disappear together. The
    supervisor itself is started lazily by ``browser_navigate`` /
    ``/browser connect`` / Browserbase session creation, so a reachable
    CDP URL is enough to commit to showing the tool.
    """
    try:
        from tools.browser_cdp_tool import _browser_cdp_check  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("browser_dialog check: browser_cdp_tool import failed: %s", exc)
        return False
    return _browser_cdp_check()


def _browser_dialog_check_or_browser() -> bool:
    if _browser_dialog_check():
        return True
    try:
        from tools.browser_tool import check_browser_requirements
        return bool(check_browser_requirements())
    except Exception:
        return False


registry.register(
    name="browser_dialog",
    toolset="browser-cdp",
    schema=BROWSER_DIALOG_SCHEMA,
    handler=lambda args, **kw: browser_dialog(
        action=args.get("action", ""),
        prompt_text=args.get("prompt_text"),
        dialog_id=args.get("dialog_id"),
        task_id=kw.get("task_id"),
    ),
    # hermes-mods: offered whenever the browser tools are - the agent-browser
    # fallback answers dialogs without a reachable CDP endpoint at schema time.
    check_fn=_browser_dialog_check_or_browser,
    emoji="💬",
)
