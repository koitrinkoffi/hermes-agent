"""Tests for the per-call subagent model override on ``delegate_task``.

``delegation.model`` / ``delegation.provider`` pin every subagent to one
model. The ``model`` parameter (top-level, or per task inside ``tasks``)
narrows that to a single call so a mechanical task can run on a small model
while a review runs on a large one.

Covered here:
  - "provider:model" parsing against colon-bearing model ids
  - validation against the catalog of the endpoint the child will hit
  - precedence: per-task model > top-level model > delegation config
  - each distinct override resolved once per call, not once per task
"""

import pytest

import tools.delegate_tool as dt


# --------------------------------------------------------------------------
# provider:model parsing
# --------------------------------------------------------------------------


class TestSplitProviderModel:
    @pytest.fixture(autouse=True)
    def _known_providers(self, monkeypatch):
        monkeypatch.setattr(
            dt,
            "_known_provider_slugs",
            lambda: {"openrouter", "ollama-cloud", "custom", "lemonade", "custom:lemonade"},
        )

    def test_bare_model_has_no_provider(self):
        assert dt._split_provider_model("Qwen3.5-4B-MTP") == (None, "Qwen3.5-4B-MTP")

    def test_provider_prefix_is_split_off(self):
        assert dt._split_provider_model("openrouter:z-ai/glm-5") == (
            "openrouter",
            "z-ai/glm-5",
        )

    def test_colon_in_model_id_is_not_a_provider(self):
        # Ollama-style ids carry a colon; "gemma4" is not a provider.
        assert dt._split_provider_model("gemma4:31b") == (None, "gemma4:31b")

    def test_provider_prefix_before_colon_bearing_model(self):
        assert dt._split_provider_model("ollama-cloud:gemma4:31b") == (
            "ollama-cloud",
            "gemma4:31b",
        )

    def test_longest_provider_prefix_wins(self):
        # "custom" also matches, but the two-segment custom slug is the
        # specific one and must be preferred.
        assert dt._split_provider_model("custom:lemonade:Qwen3.5-4B") == (
            "custom:lemonade",
            "Qwen3.5-4B",
        )

    def test_whitespace_is_trimmed(self):
        assert dt._split_provider_model("  Qwen3.5-4B-MTP  ") == (
            None,
            "Qwen3.5-4B-MTP",
        )


# --------------------------------------------------------------------------
# override resolution / validation
# --------------------------------------------------------------------------


class _FakeParent:
    """Minimal stand-in for the parent AIAgent attributes we read."""

    def __init__(self, base_url="http://localhost:13305/v1", api_key="k", model="Big"):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.api_mode = "chat_completions"
        self.provider = "custom"


BASE_CREDS = {
    "model": None,
    "provider": None,
    "base_url": None,
    "api_key": None,
    "api_mode": None,
    "request_overrides": None,
    "max_output_tokens": None,
}


@pytest.fixture
def catalog(monkeypatch):
    """Endpoint advertises two models; nothing is declared in config."""
    monkeypatch.setattr(dt, "_declared_models_for_provider", lambda provider: [])
    monkeypatch.setattr(
        dt,
        "_endpoint_model_ids",
        lambda base_url, api_key, api_mode, allow_fetch=True: [
            "Qwen3.6-35B-A3B-MTP",
            "Qwen3.5-4B-MTP",
        ],
    )
    monkeypatch.setattr(dt, "_configured_runtime_provider", lambda: "custom:lemonade")


class TestResolveModelOverride:
    def test_known_model_swaps_only_the_model_id(self, catalog):
        creds, err = dt._resolve_model_override(
            "Qwen3.5-4B-MTP", {}, BASE_CREDS, _FakeParent()
        )

        assert err is None
        assert creds["model"] == "Qwen3.5-4B-MTP"
        # Endpoint untouched — the child still inherits the parent's.
        assert creds["base_url"] is None
        assert creds["provider"] is None

    def test_unknown_model_is_rejected_with_the_valid_list(self, catalog):
        creds, err = dt._resolve_model_override(
            "Qwen4-99B", {}, BASE_CREDS, _FakeParent()
        )

        assert creds is None
        assert "Qwen4-99B" in err
        assert "Qwen3.5-4B-MTP" in err  # tells the model what it CAN pick

    def test_model_id_is_canonicalized_from_the_catalog(self, catalog):
        creds, err = dt._resolve_model_override(
            "qwen3.5-4b-mtp", {}, BASE_CREDS, _FakeParent()
        )

        assert err is None
        assert creds["model"] == "Qwen3.5-4B-MTP"

    def test_empty_catalog_accepts_the_override(self, monkeypatch):
        # Discovery failure (endpoint down, provider with no /models route)
        # must not disable the feature.
        monkeypatch.setattr(dt, "_declared_models_for_provider", lambda provider: [])
        monkeypatch.setattr(
            dt,
            "_endpoint_model_ids",
            lambda base_url, api_key, api_mode, allow_fetch=True: [],
        )

        creds, err = dt._resolve_model_override(
            "anything-at-all", {}, BASE_CREDS, _FakeParent()
        )

        assert err is None
        assert creds["model"] == "anything-at-all"

    def test_declared_config_models_count_as_known(self, monkeypatch):
        monkeypatch.setattr(
            dt, "_declared_models_for_provider", lambda provider: ["Declared-7B"]
        )
        monkeypatch.setattr(
            dt,
            "_endpoint_model_ids",
            lambda base_url, api_key, api_mode, allow_fetch=True: [],
        )

        creds, err = dt._resolve_model_override(
            "Declared-7B", {}, BASE_CREDS, _FakeParent()
        )

        assert err is None
        assert creds["model"] == "Declared-7B"

    def test_provider_prefix_resolves_full_credentials(self, monkeypatch):
        monkeypatch.setattr(dt, "_declared_models_for_provider", lambda provider: [])
        monkeypatch.setattr(
            dt,
            "_endpoint_model_ids",
            lambda base_url, api_key, api_mode, allow_fetch=True: ["gemma4:31b"],
        )
        monkeypatch.setattr(
            dt, "_known_provider_slugs", lambda: {"ollama-cloud"}
        )
        fake_runtime = {
            "provider": "ollama-cloud",
            "base_url": "https://ollama.com/v1",
            "api_key": "secret",
            "api_mode": "chat_completions",
            "request_overrides": {"foo": 1},
            "max_output_tokens": 4096,
        }
        import hermes_cli.runtime_provider as rp

        monkeypatch.setattr(
            rp, "resolve_runtime_provider", lambda **kw: dict(fake_runtime)
        )

        creds, err = dt._resolve_model_override(
            "ollama-cloud:gemma4:31b", {}, BASE_CREDS, _FakeParent()
        )

        assert err is None
        assert creds["model"] == "gemma4:31b"
        assert creds["provider"] == "ollama-cloud"
        assert creds["base_url"] == "https://ollama.com/v1"
        assert creds["api_key"] == "secret"
        assert creds["request_overrides"] == {"foo": 1}

    def test_provider_without_api_key_is_rejected(self, monkeypatch):
        monkeypatch.setattr(dt, "_known_provider_slugs", lambda: {"openrouter"})
        import hermes_cli.runtime_provider as rp

        monkeypatch.setattr(
            rp,
            "resolve_runtime_provider",
            lambda **kw: {"provider": "openrouter", "api_key": ""},
        )

        creds, err = dt._resolve_model_override(
            "openrouter:z-ai/glm-5", {}, BASE_CREDS, _FakeParent()
        )

        assert creds is None
        assert "no API key" in err

    def test_unresolvable_provider_reports_how_to_recover(self, monkeypatch):
        monkeypatch.setattr(dt, "_known_provider_slugs", lambda: {"openrouter"})
        import hermes_cli.runtime_provider as rp

        def _boom(**kw):
            raise ValueError("not configured")

        monkeypatch.setattr(rp, "resolve_runtime_provider", _boom)

        creds, err = dt._resolve_model_override(
            "openrouter:z-ai/glm-5", {}, BASE_CREDS, _FakeParent()
        )

        assert creds is None
        assert "not configured" in err
        assert "Drop the provider prefix" in err


# --------------------------------------------------------------------------
# schema surface
# --------------------------------------------------------------------------


class TestSchema:
    def test_model_is_exposed_top_level_and_per_task(self):
        props = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]

        assert props["model"]["type"] == "string"
        assert props["tasks"]["items"]["properties"]["model"]["type"] == "string"

    def test_dynamic_overrides_fill_both_descriptions(self, monkeypatch):
        monkeypatch.setattr(
            dt, "_schema_model_candidates", lambda limit=24: ["Small-4B", "Big-35B"]
        )

        overrides = dt._build_dynamic_schema_overrides()
        props = overrides["parameters"]["properties"]

        assert "Small-4B" in props["model"]["description"]
        assert "Small-4B" in props["tasks"]["items"]["properties"]["model"]["description"]
        assert "selectable per call" in overrides["description"]

    def test_dynamic_overrides_do_not_mutate_the_static_schema(self, monkeypatch):
        monkeypatch.setattr(
            dt, "_schema_model_candidates", lambda limit=24: ["Small-4B"]
        )

        dt._build_dynamic_schema_overrides()

        static_items = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
        assert (
            static_items["properties"]["model"]["description"]
            == "(rebuilt at get_definitions() time)"
        )

    def test_advertised_candidates_exclude_non_chat_models(self, monkeypatch):
        monkeypatch.setattr(dt, "_load_config", lambda: {})
        monkeypatch.setattr(dt, "_configured_runtime_provider", lambda: "custom:x")
        monkeypatch.setattr(
            dt,
            "_known_model_ids_for",
            lambda *a, **kw: [
                "Qwen3.6-35B-A3B-MTP",
                "Qwen3-Embedding-0.6B-GGUF",
                "Qwen3-Reranker-0.6B",
                "Qwen3-ASR-1.7B",
                "GLM-OCR",
            ],
        )

        assert dt._schema_model_candidates() == ["Qwen3.6-35B-A3B-MTP"]


# --------------------------------------------------------------------------
# precedence + wiring through delegate_task
# --------------------------------------------------------------------------


class _RecordingBuild:
    """Captures the credential kwargs each child would be built with."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)

        class _Child:
            tool_progress_callback = None

        return _Child()


@pytest.fixture
def delegate_harness(monkeypatch):
    """Run delegate_task with stub children so child construction is observable.

    The delegation itself runs synchronously (``background`` defaults to False
    for direct Python callers) and each stub child fails immediately inside the
    executor, which is fine — the assertions are about how the children were
    BUILT, not what they returned.
    """
    builder = _RecordingBuild()
    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", builder)
    monkeypatch.setattr(dt, "is_spawn_paused", lambda: False)
    monkeypatch.setattr(dt, "_get_max_spawn_depth", lambda: 2)
    monkeypatch.setattr(dt, "_get_max_concurrent_children", lambda: 5)
    monkeypatch.setattr(dt, "_load_config", lambda: {"max_iterations": 10})
    monkeypatch.setattr(
        dt, "_resolve_delegation_credentials", lambda cfg, parent: dict(BASE_CREDS)
    )
    monkeypatch.setattr(dt, "_declared_models_for_provider", lambda provider: [])
    monkeypatch.setattr(
        dt,
        "_endpoint_model_ids",
        lambda base_url, api_key, api_mode, allow_fetch=True: ["Small-4B", "Big-35B"],
    )

    # No live-transcript files: create_live_transcripts is imported inside
    # delegate_task, so patch it at its source module.
    import tools.delegation_live_log as live_log

    monkeypatch.setattr(live_log, "create_live_transcripts", lambda *a, **kw: ("", [], []))
    return builder


class TestModelPrecedence:
    def _run(self, builder, parent, **kwargs):
        dt.delegate_task(parent_agent=parent, **kwargs)
        return builder.calls

    def test_top_level_model_applies_to_every_task(self, delegate_harness):
        calls = self._run(
            delegate_harness,
            _FakeParent(),
            tasks=[{"goal": "task alpha"}, {"goal": "task bravo"}],
            model="Small-4B",
        )

        assert [c["model"] for c in calls] == ["Small-4B", "Small-4B"]

    def test_per_task_model_beats_top_level(self, delegate_harness):
        calls = self._run(
            delegate_harness,
            _FakeParent(),
            tasks=[
                {"goal": "task alpha", "model": "Big-35B"},
                {"goal": "task bravo"},
            ],
            model="Small-4B",
        )

        assert [c["model"] for c in calls] == ["Big-35B", "Small-4B"]

    def test_no_override_keeps_delegation_credentials(self, delegate_harness):
        calls = self._run(delegate_harness, _FakeParent(), goal="a")

        assert calls[0]["model"] is None  # inherit from parent

    def test_unknown_model_fails_before_any_child_is_built(self, delegate_harness):
        builder = delegate_harness

        result = dt.delegate_task(
            parent_agent=_FakeParent(), goal="a", model="Nope-1B"
        )

        assert "Nope-1B" in result
        assert builder.calls == []

    def test_repeated_override_is_resolved_once(self, delegate_harness, monkeypatch):
        probes = []
        monkeypatch.setattr(
            dt,
            "_endpoint_model_ids",
            lambda base_url, api_key, api_mode, allow_fetch=True: (
                probes.append(base_url) or ["Small-4B"]
            ),
        )

        self._run(
            delegate_harness,
            _FakeParent(),
            tasks=[
                {"goal": "task alpha", "model": "Small-4B"},
                {"goal": "task bravo", "model": "Small-4B"},
                {"goal": "task charlie", "model": "Small-4B"},
            ],
        )

        assert len(probes) == 1


class TestBatchModelLabel:
    def test_single_model_reports_that_model(self):
        label = dt._batch_model_label([{"model": "A"}, {"model": "A"}], {"model": "Z"})
        assert label == "A"

    def test_mixed_models_report_all_of_them(self):
        label = dt._batch_model_label([{"model": "A"}, {"model": "B"}], {"model": "Z"})
        assert label == "A, B"

    def test_no_models_falls_back_to_base_creds(self):
        label = dt._batch_model_label([{"model": None}], {"model": "Z"})
        assert label == "Z"
