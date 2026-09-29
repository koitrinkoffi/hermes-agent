"""Unit tests for tools/browser_tool_hermes.py (no browser, no daemon)."""
from __future__ import annotations

import json

import pytest

import tools.browser_tool as bt
import tools.browser_tool_hermes as bh


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(bh, "bring_active_tab_to_front", lambda task_id: True)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_last_session_key", lambda task_id: f"session::{task_id}")
    bh._PROGRESS.clear()
    bh._FRAME_SELECTED.clear()
    bh._FRONTED.clear()
    yield


def _recorder(monkeypatch, responses=None):
    calls = []

    def fake_run(task_id, command, args=None, timeout=None, **kw):
        calls.append((command, list(args or [])))
        if responses and command in responses:
            r = responses[command]
            return r(args) if callable(r) else r
        return {"success": True, "data": {}}

    monkeypatch.setattr(bt, "_run_browser_command", fake_run)
    return calls


# ── snapshot shape ────────────────────────────────────────────────────────────

def test_normalize_snapshot_data_flattens_038_shape():
    data = {"snapshot": {"kind": "full", "revision": 3, "tree": "- button [ref=e1]", "refs": {"e1": {}}},
            "origin": "https://x/"}
    out = bh.normalize_snapshot_data(data)
    assert out["snapshot"] == "- button [ref=e1]"
    assert out["refs"] == {"e1": {}}
    assert out["snapshot_kind"] == "full"


def test_normalize_snapshot_data_keeps_old_shape():
    data = {"snapshot": "- a", "refs": {"e1": {}}}
    assert bh.normalize_snapshot_data(data) is data


def test_snapshot_flags(monkeypatch):
    monkeypatch.setattr(bh, "agent_browser_version", lambda: ((0, 38, 1), "agent-browser 0.38.1-hermes.1"))
    assert bh.snapshot_flags(baseline=True) == ["-c", "--prune", "--delta", "--full"]
    monkeypatch.setattr(bh, "agent_browser_version", lambda: ((0, 35, 1), "agent-browser 0.35.1"))
    assert bh.snapshot_flags(baseline=True) == ["-c"]


# ── observation ───────────────────────────────────────────────────────────────

def test_observation_unsupported_returns_none(monkeypatch):
    monkeypatch.setattr(bh, "agent_browser_version", lambda: ((0, 35, 1), "agent-browser 0.35.1"))
    assert bh.observe_after_action("t") is None


@pytest.mark.parametrize("snap,expect", [
    ({"kind": "unchanged"}, "unchanged"),
    ({"kind": "delta", "changes": [{"op": "remove", "ref": "@e3"}],
      "treeChange": {"lines": ["- heading \"Done\"", "- button \"Next\" [ref=e9]"]}}, "changed"),
    ({"kind": "full", "tree": "- main\n  - heading \"New page\""}, "new page"),
])
def test_observation_kinds(monkeypatch, snap, expect):
    monkeypatch.setattr(bh, "agent_browser_version", lambda: ((0, 38, 1), "agent-browser 0.38.1"))
    calls = _recorder(monkeypatch, {"snapshot": {"success": True,
                                                 "data": {"snapshot": snap, "origin": "https://x/p"}}})
    obs = bh.observe_after_action("t")
    assert obs["url"] == "https://x/p"
    assert obs["page"].startswith(expect)
    assert calls == [("snapshot", ["-c", "--delta"])]
    if expect == "changed":
        assert "button \"Next\"" in obs["changed_lines"]
        assert obs["removed_refs"] == ["@e3"]


def test_observation_attached_to_click(monkeypatch):
    monkeypatch.setattr(bh, "observe_after_action", lambda task_id: {"url": "https://x/", "page": "changed"})
    handler = bh._wrap_handler("browser_click", lambda args, **kw: json.dumps({"success": True, "clicked": "@e1"}))
    out = json.loads(handler({"ref": "@e1"}, task_id="t"))
    assert out["observation"]["page"] == "changed"


def test_no_observation_on_failure(monkeypatch):
    monkeypatch.setattr(bh, "observe_after_action", lambda task_id: pytest.fail("must not observe after a failure"))
    handler = bh._wrap_handler("browser_click", lambda args, **kw: json.dumps({"success": False, "error": "x"}))
    assert "observation" not in json.loads(handler({"ref": "@e1"}, task_id="t"))


# ── progress guard ────────────────────────────────────────────────────────────

def test_guard_failure_streak():
    fail = json.dumps({"success": False, "error": "nope"})
    outs = [bh.progress_guidance("t", "browser_click", {"ref": f"@e{i}"}, fail) for i in range(3)]
    assert outs[0] is None and outs[1] is None
    assert "in a row have failed" in outs[2]


def test_guard_identical_repeats():
    ok = json.dumps({"success": True})
    outs = [bh.progress_guidance("t", "browser_eval", {"expression": "1"}, ok) for _ in range(3)]
    assert outs[2] and "exact same" in outs[2]
    # said once per signature
    assert bh.progress_guidance("t", "browser_eval", {"expression": "1"}, ok) is None


def test_guard_volume_threshold():
    ok = json.dumps({"success": True})
    said = [bh.progress_guidance("t", "browser_click", {"ref": f"@e{i}"}, ok) for i in range(40)]
    assert all(s is None for s in said[:39])
    assert "40 browser calls" in said[39]


def test_guard_resets_after_idle(monkeypatch):
    ok = json.dumps({"success": True})
    clock = [1000.0]
    monkeypatch.setattr(bh.time, "time", lambda: clock[0])
    for i in range(39):
        bh.progress_guidance("t", "browser_click", {"ref": f"@e{i}"}, ok)
    clock[0] += bh.EPISODE_IDLE_RESET_S + 1
    assert bh.progress_guidance("t", "browser_click", {"ref": "@e99"}, ok) is None


def test_guidance_injected_into_json_result():
    fail = json.dumps({"success": False, "error": "x"})
    handler = bh._wrap_handler("browser_type", lambda args, **kw: fail)
    for i in range(2):
        assert "guidance" not in json.loads(handler({"ref": f"@e{i}", "text": "a"}, task_id="g"))
    assert "guidance" in json.loads(handler({"ref": "@e9", "text": "a"}, task_id="g"))


# ── new tools: argv construction ──────────────────────────────────────────────

def test_read_page_and_element(monkeypatch):
    calls = _recorder(monkeypatch, {
        "read": {"success": True, "data": {"content": "# Title\nbody"}},
        "get": {"success": True, "data": {"text": "cell"}},
    })
    out = json.loads(bh.browser_read(filter="Pricing", task_id="t"))
    assert out["text"].startswith("# Title") and calls[-1] == ("read", ["--filter", "Pricing"])
    out = json.loads(bh.browser_read(ref="e12", task_id="t"))
    assert out["text"] == "cell" and calls[-1] == ("get", ["text", "@e12"])


def test_read_truncates_long_text(monkeypatch):
    long = "\n".join(f"line {i} " + "x" * 50 for i in range(2000))
    _recorder(monkeypatch, {"read": {"success": True, "data": {"content": long}}})
    monkeypatch.setattr(bt, "_store_full_snapshot", lambda text: "/tmp/full.txt")
    out = json.loads(bh.browser_read(max_chars=2000, task_id="t"))
    assert out["chars"] == len(long)
    assert "read_file path=\"/tmp/full.txt\"" in out["text"]
    assert len(out["text"]) < 2400


def test_find_argv(monkeypatch):
    calls = _recorder(monkeypatch)
    json.loads(bh.browser_find(by="role", value="button", name="Sign in", task_id="t"))
    assert calls[-1] == ("find", ["role", "button", "click", "--name", "Sign in"])
    json.loads(bh.browser_find(by="label", value="Email", action="fill", text="a@b.c", exact=True, task_id="t"))
    assert calls[-1] == ("find", ["label", "Email", "fill", "a@b.c", "--exact"])
    out = json.loads(bh.browser_find(by="label", value="Email", action="fill", task_id="t"))
    assert out["success"] is False


def test_wait_argv(monkeypatch):
    calls = _recorder(monkeypatch)
    json.loads(bh.browser_wait(text="Hello World!", timeout=20, task_id="t"))
    assert calls[-1] == ("wait", ["--text", "Hello World!", "--timeout", "20000"])
    json.loads(bh.browser_wait(ref="e4", hidden=True, task_id="t"))
    assert calls[-1] == ("wait", ["@e4", "--state", "hidden", "--timeout", "15000"])
    json.loads(bh.browser_wait(ms=1500, task_id="t"))
    assert calls[-1] == ("wait", ["1500"])
    out = json.loads(bh.browser_wait(text="a", url="b", task_id="t"))
    assert out["success"] is False


def test_frame_tracks_selection(monkeypatch):
    calls = _recorder(monkeypatch)
    out = json.loads(bh.browser_frame(ref="e7", task_id="t"))
    assert out["frame"] == "@e7" and calls[-1] == ("frame", ["@e7"])
    assert bh.frame_selected("t") == "@e7"
    # navigating resets it through the wrapper
    nav = bh._wrap_handler("browser_navigate", lambda args, **kw: json.dumps({"success": True}))
    nav({"url": "https://x"}, task_id="t")
    assert bh.frame_selected("t") is None
    json.loads(bh.browser_frame(ref="e7", task_id="t"))
    json.loads(bh.browser_frame(main=True, task_id="t"))
    assert bh.frame_selected("t") is None and calls[-1] == ("frame", ["main"])
