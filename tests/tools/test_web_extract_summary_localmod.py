"""Local mod (hermes-mods): main-agent web_extract summarization.

Covers the summarize path in web_extract_tool, its raw fallback, the retry
policy (402 fails at once, 429 retries once), and the handler gating
(kill switch, subagent callers, explicit char_limit).
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.web_tools as wt


BIG = "\n".join(f"para {i} " + "y" * 80 for i in range(300))  # ~26k chars
SMALL = "# Title\n\nshort body\n"


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        message = SimpleNamespace(content=outcome)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_client(outcomes):
    completions = _FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


class _FakeProvider:
    name = "fake"
    display_name = "Fake"

    def __init__(self, content):
        self.content = content

    def supports_extract(self):
        return True

    async def extract(self, urls, **kwargs):
        return [{"url": u, "title": "Page", "content": self.content,
                 "raw_content": self.content, "metadata": {}} for u in urls]


async def _no_sleep(*_a, **_k):
    return None


class _AsyncTrue:
    async def __call__(self, *a, **k):
        return True


def _run_extract(tmp_path, monkeypatch, content, outcomes, *, urls=None, summarize=True):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    client, completions = _fake_client(outcomes)
    with patch("tools.web_tools._ensure_web_plugins_loaded"), \
         patch("tools.web_tools._get_extract_backend", return_value="fake"), \
         patch("tools.web_tools.async_is_safe_url", new=_AsyncTrue()), \
         patch("agent.web_search_registry.get_provider", return_value=_FakeProvider(content)), \
         patch("agent.auxiliary_client.get_async_text_auxiliary_client",
               return_value=(client, "gemma4:31b")), \
         patch("agent.auxiliary_client._get_task_timeout", return_value=5.0), \
         patch("tools.web_tools.asyncio.sleep", new=_no_sleep):
        result = json.loads(asyncio.new_event_loop().run_until_complete(
            wt.web_extract_tool(urls or ["https://example.com/big"], summarize=summarize)
        ))
    return result, completions


class TestSummarizePath:
    def test_long_page_is_replaced_by_summary_with_stored_full_text(self, tmp_path, monkeypatch):
        result, completions = _run_extract(tmp_path, monkeypatch, BIG, ["- digest fact"])
        content = result["results"][0]["content"]
        assert content.startswith("- digest fact")
        assert "[SUMMARIZED]" in content
        assert "Summary by gemma4:31b" in content
        path_line = next(ln for ln in content.splitlines() if "Full text saved to:" in ln)
        stored = path_line.split("Full text saved to:", 1)[1].strip()
        assert "para 150 " in open(stored, encoding="utf-8").read()
        assert completions.calls == 1

    def test_short_page_is_returned_whole_without_a_call(self, tmp_path, monkeypatch):
        result, completions = _run_extract(tmp_path, monkeypatch, SMALL, [])
        assert result["results"][0]["content"] == SMALL
        assert completions.calls == 0

    def test_summarize_false_keeps_raw_path(self, tmp_path, monkeypatch):
        result, completions = _run_extract(tmp_path, monkeypatch, BIG, [], summarize=False)
        assert "[SUMMARIZED]" not in result["results"][0]["content"]
        assert completions.calls == 0

    def test_402_falls_back_to_raw_at_once(self, tmp_path, monkeypatch):
        result, completions = _run_extract(tmp_path, monkeypatch, BIG, [_StatusError(402)])
        content = result["results"][0]["content"]
        assert "[TRUNCATED]" in content
        assert "para 0 " in content
        assert content.rstrip().endswith("[summary unavailable (HTTP 402); raw page returned]")
        assert completions.calls == 1

    def test_429_retries_once_then_succeeds(self, tmp_path, monkeypatch):
        result, completions = _run_extract(
            tmp_path, monkeypatch, BIG, [_StatusError(429), "- second try"]
        )
        assert result["results"][0]["content"].startswith("- second try")
        assert completions.calls == 2

    def test_two_429s_fall_back(self, tmp_path, monkeypatch):
        result, completions = _run_extract(
            tmp_path, monkeypatch, BIG, [_StatusError(429), _StatusError(429)]
        )
        assert "[summary unavailable (HTTP 429)" in result["results"][0]["content"]
        assert completions.calls == 2

    def test_several_urls_are_summarized_one_by_one(self, tmp_path, monkeypatch):
        result, completions = _run_extract(
            tmp_path, monkeypatch, BIG, ["- a", "- b"],
            urls=["https://example.com/a", "https://example.com/b"],
        )
        contents = [r["content"] for r in result["results"]]
        assert contents[0].startswith("- a") and contents[1].startswith("- b")
        assert completions.calls == 2


class TestHandlerGating:
    def _handler_summarize(self, args, task_id, enabled=True, subagent=False):
        captured = {}

        async def fake_tool(urls, fmt, char_limit=None, summarize=False):
            captured["summarize"] = summarize
            return "{}"

        entry = wt.registry.get_entry("web_extract")
        with patch("tools.web_tools.web_extract_tool", new=fake_tool), \
             patch("tools.web_tools._load_web_config",
                   return_value={"extract_summarize": enabled}), \
             patch("tools.delegate_tool.get_subagent_attribution",
                   return_value=({"subagent_id": task_id} if subagent else None)):
            asyncio.new_event_loop().run_until_complete(entry.handler(args, task_id=task_id))
        return captured["summarize"]

    def test_main_agent_gets_summaries(self):
        assert self._handler_summarize({"urls": ["https://e.com"]}, "main") is True

    def test_kill_switch_off(self):
        assert self._handler_summarize({"urls": ["https://e.com"]}, "main", enabled=False) is False

    def test_subagent_keeps_raw(self):
        assert self._handler_summarize({"urls": ["https://e.com"]}, "sa-1", subagent=True) is False

    def test_explicit_char_limit_means_raw(self):
        args = {"urls": ["https://e.com"], "char_limit": 40000}
        assert self._handler_summarize(args, "main") is False
