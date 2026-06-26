import json
from pathlib import Path

import pytest

import tools.browser_tool as bt


@pytest.fixture
def non_camofox(monkeypatch):
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_last_session_key", lambda task_id: f"session::{task_id}")
    monkeypatch.setattr(bt, "_get_command_timeout", lambda: 30)


def test_browser_tab_list_normalizes_rich_response(non_camofox, monkeypatch):
    monkeypatch.setattr(
        bt,
        "_run_browser_command",
        lambda task_id, command, args, timeout=None, **kwargs: {
            "success": True,
            "data": {
                "tabs": [
                    {"title": "One", "url": "https://one.example", "active": False},
                    {"title": "Two", "url": "https://two.example", "active": True},
                ]
            },
        },
    )

    result = json.loads(bt.browser_tab(action="list", task_id="t1"))

    assert result["success"] is True
    assert result["active_index"] == 2
    assert result["tabs"] == [
        {"index": 1, "title": "One", "url": "https://one.example", "active": False},
        {"index": 2, "title": "Two", "url": "https://two.example", "active": True},
    ]


def test_browser_tab_new_with_url_creates_then_navigates(non_camofox, monkeypatch):
    calls = []

    def fake_run(task_id, command, args, timeout=None, **kwargs):
        calls.append((task_id, command, args, timeout))
        return {"success": True, "data": {}}

    monkeypatch.setattr(bt, "_run_browser_command", fake_run)

    result = json.loads(bt.browser_tab(action="new", url="https://example.com", task_id="t1"))

    assert result == {"success": True, "action": "new", "url": "https://example.com"}
    assert calls == [
        ("session::t1", "tab", ["new"], None),
        ("session::t1", "open", ["https://example.com"], 60),
    ]


def test_browser_tab_switch_requires_positive_1_based_index(non_camofox):
    result = json.loads(bt.browser_tab(action="switch", index=0, task_id="t1"))
    assert result["success"] is False
    assert "1-based index" in result["error"]


def test_browser_upload_validates_and_normalizes_paths(non_camofox, monkeypatch, tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    calls = []

    def fake_run(task_id, command, args, timeout=None, **kwargs):
        calls.append((task_id, command, args, timeout))
        return {"success": True, "data": {}}

    monkeypatch.setattr(bt, "_run_browser_command", fake_run)

    result = json.loads(
        bt.browser_upload(ref="e9", path=str(first), paths=[str(second)], task_id="t1")
    )

    assert result == {
        "success": True,
        "element": "@e9",
        "uploaded_paths": [str(first.resolve()), str(second.resolve())],
    }
    assert calls == [
        (
            "session::t1",
            "upload",
            ["@e9", str(first.resolve()), str(second.resolve())],
            60,
        )
    ]


def test_browser_upload_missing_path_fails_fast(non_camofox, monkeypatch, tmp_path):
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        return {"success": True}

    monkeypatch.setattr(bt, "_run_browser_command", fake_run)

    missing = tmp_path / "missing.txt"
    result = json.loads(bt.browser_upload(ref="@e1", path=str(missing), task_id="t1"))

    assert result["success"] is False
    assert str(missing.resolve()) in result["error"]
    assert called is False


def test_browser_upload_accepts_selector_passthrough(non_camofox, monkeypatch, tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")
    captured = {}

    def fake_run(task_id, command, args, timeout=None, **kwargs):
        captured["args"] = args
        return {"success": True, "data": {}}

    monkeypatch.setattr(bt, "_run_browser_command", fake_run)

    result = json.loads(bt.browser_upload(ref="input[type=file]", path=str(file_path), task_id="t1"))

    assert result["success"] is True
    assert captured["args"][0] == "input[type=file]"


def test_browser_download_uses_default_path_and_verifies_file(non_camofox, monkeypatch, tmp_path):
    monkeypatch.setattr(bt, "get_hermes_home", lambda: tmp_path)

    def fake_run(task_id, command, args, timeout=None, **kwargs):
        assert task_id == "session::t1"
        assert command == "download"
        assert args[0] == "@e4"
        target = Path(args[1])
        target.write_text("downloaded", encoding="utf-8")
        return {"success": True, "data": {}}

    monkeypatch.setattr(bt, "_run_browser_command", fake_run)

    result = json.loads(bt.browser_download(ref="e4", task_id="t1"))

    assert result["success"] is True
    assert result["exists"] is True
    assert result["element"] == "@e4"
    assert result["path"].startswith(str(tmp_path / "browser_downloads"))
    assert Path(result["path"]).read_text(encoding="utf-8") == "downloaded"


def test_browser_download_reports_missing_file_after_success(non_camofox, monkeypatch, tmp_path):
    monkeypatch.setattr(bt, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        bt,
        "_run_browser_command",
        lambda task_id, command, args, timeout=None, **kwargs: {"success": True, "data": {}},
    )

    result = json.loads(bt.browser_download(ref="@e2", task_id="t1"))

    assert result["success"] is False
    assert "file was not found" in result["error"]


@pytest.fixture
def camofox(monkeypatch):
    """Run browser_upload through the Camofox backend with HTTP mocked out."""
    import tools.browser_camofox as bcfox

    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: True)
    monkeypatch.setattr(
        bcfox, "_get_session", lambda task_id: {"tab_id": "tab123", "user_id": "user123"}
    )
    calls = []

    def fake_post(path, body, timeout=None):
        calls.append({"path": path, "body": body, "timeout": timeout})
        return {"ok": True, "uploaded": len(body.get("paths", []))}

    monkeypatch.setattr(bcfox, "_post", fake_post)
    return calls


def test_browser_upload_camofox_ref_routes_and_matches_shape(camofox, tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    result = json.loads(
        bt.browser_upload(ref="e9", path=str(first), paths=[str(second)], task_id="t1")
    )

    # Response shape is identical to the agent-browser backend.
    assert result == {
        "success": True,
        "element": "@e9",
        "uploaded_paths": [str(first.resolve()), str(second.resolve())],
    }
    assert len(camofox) == 1
    call = camofox[0]
    assert call["path"] == "/tabs/tab123/upload"
    # @eN refs go in the "ref" field (stripped), never "selector".
    assert call["body"]["ref"] == "e9"
    assert "selector" not in call["body"]
    assert call["body"]["paths"] == [str(first.resolve()), str(second.resolve())]
    assert call["body"]["userId"] == "user123"


def test_browser_upload_camofox_routes_selector(camofox, tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = json.loads(
        bt.browser_upload(ref="input[type=file]", path=str(file_path), task_id="t1")
    )

    assert result["success"] is True
    call = camofox[0]
    # Raw selectors go in the "selector" field, never "ref".
    assert call["body"]["selector"] == "input[type=file]"
    assert "ref" not in call["body"]


def test_browser_upload_camofox_validates_before_dispatch(camofox, tmp_path):
    missing = tmp_path / "missing.txt"

    result = json.loads(bt.browser_upload(ref="e1", path=str(missing), task_id="t1"))

    assert result["success"] is False
    assert str(missing.resolve()) in result["error"]
    # Validation failed, so no HTTP request was issued.
    assert camofox == []


def test_chromium_installed_accepts_agent_browser_chrome_cache(monkeypatch, tmp_path):
    cache_root = tmp_path / "browsers"
    cache_root.mkdir()
    (cache_root / "chrome-150.0.7871.24").mkdir()

    monkeypatch.setattr(bt, "_cached_chromium_installed", None)
    monkeypatch.setattr(bt, "_chromium_search_roots", lambda: [str(cache_root)])
    monkeypatch.setattr(bt.shutil, "which", lambda name: None)
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)

    assert bt._chromium_installed() is True
