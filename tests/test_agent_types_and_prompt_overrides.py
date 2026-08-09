"""Local-mod tests: per-block prompt overrides + named subagent types.

Covers the two hermes-mods features added 2026-08-09:
  * ~/.hermes/prompts/<block>.md overrides in agent/system_prompt.py
  * delegate_task(agent_type=...) definitions in ~/.hermes/agents/<name>.md
"""

import types

import pytest


# ── Prompt block overrides ──────────────────────────────────────────────


def _patch_home(monkeypatch, tmp_path):
    from agent import system_prompt as sp

    monkeypatch.setattr(sp, "get_hermes_home", lambda: tmp_path)
    return sp


def test_prompt_override_missing_file_returns_default(tmp_path, monkeypatch):
    sp = _patch_home(monkeypatch, tmp_path)
    assert sp._prompt_override("task_completion", "DEFAULT") == "DEFAULT"


def test_prompt_override_file_replaces_default(tmp_path, monkeypatch):
    sp = _patch_home(monkeypatch, tmp_path)
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "task_completion.md").write_text("# Mine\ncustom text\n", encoding="utf-8")
    assert sp._prompt_override("task_completion", "DEFAULT") == "# Mine\ncustom text"


def test_prompt_override_empty_file_drops_block(tmp_path, monkeypatch):
    sp = _patch_home(monkeypatch, tmp_path)
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "google_guidance.md").write_text("", encoding="utf-8")
    assert sp._prompt_override("google_guidance", "DEFAULT") == ""


def test_prompt_override_supports_none_default(tmp_path, monkeypatch):
    sp = _patch_home(monkeypatch, tmp_path)
    assert sp._prompt_override("skills_index", None) is None


def test_bare_prompt_agent_gets_empty_tiers():
    from agent import system_prompt as sp

    agent = types.SimpleNamespace(_bare_prompt=True)
    parts = sp.build_system_prompt_parts(agent)
    assert parts == {"stable": "", "context": "", "volatile": ""}


# ── Agent-type definitions ──────────────────────────────────────────────


def _write_type(tmp_path, name, content):
    d = tmp_path / "agents"
    d.mkdir(exist_ok=True)
    (d / f"{name}.md").write_text(content, encoding="utf-8")


def _patch_constants_home(monkeypatch, tmp_path):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)


def test_load_agent_type_parses_none_tools_and_sync(tmp_path, monkeypatch):
    _patch_constants_home(monkeypatch, tmp_path)
    _write_type(tmp_path, "writer", "---\ntools: none\nsync: true\n---\nBody text.\n")
    from tools.delegate_tool import _load_agent_type

    t = _load_agent_type("writer")
    assert t["tools"] == "none"
    assert t["sync"] is True
    assert t["model"] is None
    assert t["prompt"] == "Body text."


def test_load_agent_type_parses_tool_list_and_model(tmp_path, monkeypatch):
    _patch_constants_home(monkeypatch, tmp_path)
    _write_type(
        tmp_path,
        "reader",
        "---\ntools: file, web\nsync: true\nmodel: Qwen3.5-4B-MTP\n---\nRead stuff.\n",
    )
    from tools.delegate_tool import _load_agent_type

    t = _load_agent_type("reader")
    assert t["tools"] == ["file", "web"]
    assert t["model"] == "Qwen3.5-4B-MTP"


def test_load_agent_type_unknown_name_lists_available(tmp_path, monkeypatch):
    _patch_constants_home(monkeypatch, tmp_path)
    _write_type(tmp_path, "writer", "---\ntools: none\n---\nBody.\n")
    from tools.delegate_tool import _load_agent_type

    with pytest.raises(ValueError) as exc:
        _load_agent_type("nope")
    assert "writer" in str(exc.value)


def test_load_agent_type_rejects_bad_tools_value(tmp_path, monkeypatch):
    _patch_constants_home(monkeypatch, tmp_path)
    _write_type(tmp_path, "bad", "---\ntools: 42\n---\nBody.\n")
    from tools.delegate_tool import _load_agent_type

    with pytest.raises(ValueError):
        _load_agent_type("bad")


def test_typed_child_prompt_is_bare():
    from tools.delegate_tool import _build_child_system_prompt

    p = _build_child_system_prompt("G", "C", type_prompt="TYPE BODY")
    assert p.startswith("TYPE BODY")
    assert "YOUR TASK:\nG" in p
    assert "CONTEXT:\nC" in p
    # None of the generic scaffold leaks into a typed child prompt.
    assert "summary" not in p.lower()
    assert "workspace" not in p.lower()


def test_untyped_child_prompt_unchanged():
    from tools.delegate_tool import _build_child_system_prompt

    p = _build_child_system_prompt("G", "C")
    assert "focused subagent working on a specific delegated task" in p
    assert "provide a clear, concise summary" in p
