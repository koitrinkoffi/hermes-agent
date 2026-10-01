"""Local mod (hermes-mods): agent types with `fallback: inherit`.

A typed child pinned to a provider that fails for a provider-side reason
(402/429/5xx/timeout/auth) is re-run once on the session model, and the
result says so. Task-level failures and types without the key are left
alone.
"""

import json
from unittest.mock import MagicMock

import pytest

import tools.delegate_tool as dt


PINNED = {
    "model": "nemotron-3-super", "provider": "ollama-cloud", "base_url": None,
    "api_key": None, "api_mode": None, "command": None, "args": None,
}
SESSION = {
    "model": None, "provider": None, "base_url": None, "api_key": None,
    "api_mode": None, "command": None, "args": None,
}


def _fake_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    return parent


def _setup(monkeypatch, *, fallback, first_failure_reason):
    built = []

    def fake_build_child(**kw):
        child = MagicMock()
        child._delegate_role = "leaf"
        child._subagent_id = f"s{len(built)}"
        child._built_provider = kw.get("override_provider")
        built.append(kw.get("override_provider"))
        return child

    def fake_run(task_index, goal, child=None, parent_agent=None, **kw):
        if child._built_provider == "ollama-cloud":
            entry = {
                "task_index": task_index, "status": "failed", "summary": None,
                "error": "HTTP 402", "api_calls": 1, "duration_seconds": 0.1,
                "exit_reason": "error",
            }
            if first_failure_reason:
                entry["failure_reason"] = first_failure_reason
            return entry
        return {
            "task_index": task_index, "status": "completed",
            "summary": "- fact (https://example.com)", "api_calls": 2,
            "duration_seconds": 0.2, "exit_reason": "completed",
        }

    type_def = {
        "name": "researcher", "tools": ["web"], "sync": True,
        "model": "ollama-cloud:nemotron-3-super", "fallback": fallback,
        "prompt": "You research.",
    }
    monkeypatch.setattr(dt, "_build_child_agent", fake_build_child)
    monkeypatch.setattr(dt, "_run_single_child", fake_run)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: dict(SESSION))
    monkeypatch.setattr(dt, "_resolve_model_override", lambda *a, **k: (dict(PINNED), None))
    monkeypatch.setattr(dt, "_load_agent_type", lambda name: dict(type_def))
    return built


def _delegate():
    out = dt.delegate_task(
        goal="Research one axis", agent_type="researcher", parent_agent=_fake_parent(),
    )
    return json.loads(out)


def test_billing_failure_reruns_on_session_model(monkeypatch):
    built = _setup(monkeypatch, fallback="inherit", first_failure_reason="billing")
    parsed = _delegate()
    result = parsed["results"][0]
    assert built == ["ollama-cloud", None]
    assert result["status"] == "completed"
    assert result["summary"].startswith(
        "[nemotron-3-super unavailable (billing); this task was re-run locally"
    )
    assert "- fact (https://example.com)" in result["summary"]


def test_rate_limit_failure_reruns(monkeypatch):
    built = _setup(monkeypatch, fallback="inherit", first_failure_reason="rate_limit")
    assert _delegate()["results"][0]["status"] == "completed"
    assert built == ["ollama-cloud", None]


def test_without_fallback_key_fails_loudly(monkeypatch):
    built = _setup(monkeypatch, fallback=None, first_failure_reason="billing")
    assert _delegate()["results"][0]["status"] == "failed"
    assert built == ["ollama-cloud"]


def test_task_level_failure_is_not_retried(monkeypatch):
    built = _setup(monkeypatch, fallback="inherit", first_failure_reason=None)
    assert _delegate()["results"][0]["status"] == "failed"
    assert built == ["ollama-cloud"]


def test_load_agent_type_parses_fallback(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "r.md").write_text(
        "---\ntools: web\nmodel: ollama-cloud:x\nfallback: inherit\n---\nbody\n"
    )
    (tmp_path / "agents" / "bad.md").write_text("---\nfallback: elsewhere\n---\nbody\n")
    assert dt._load_agent_type("r")["fallback"] == "inherit"
    with pytest.raises(ValueError, match="'fallback' must be 'inherit'"):
        dt._load_agent_type("bad")
