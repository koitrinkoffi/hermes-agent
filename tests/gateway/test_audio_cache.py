"""
Tests for audio cache cleanup in gateway/platforms/base.py.

Covers: cleanup_audio_cache (inbound voice-note pruning), get_audio_cache_dir.
"""

import os
import time

import pytest

from gateway.platforms.base import (
    cleanup_audio_cache,
    get_audio_cache_dir,
)

# ---------------------------------------------------------------------------
# Fixture: redirect AUDIO_CACHE_DIR to a temp directory for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point the module-level AUDIO_CACHE_DIR to a fresh tmp_path."""
    monkeypatch.setattr(
        "gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio_cache"
    )


# ---------------------------------------------------------------------------
# TestCleanupAudioCache
# ---------------------------------------------------------------------------

class TestCleanupAudioCache:
    def test_removes_old_files(self, tmp_path):
        cache_dir = get_audio_cache_dir()
        old_file = cache_dir / "audio_old.ogg"
        old_file.write_bytes(b"old")
        # Set modification time to 48 hours ago
        old_mtime = time.time() - 48 * 3600
        os.utime(old_file, (old_mtime, old_mtime))

        removed = cleanup_audio_cache(max_age_hours=24)
        assert removed == 1
        assert not old_file.exists()

    def test_keeps_recent_files(self):
        cache_dir = get_audio_cache_dir()
        recent = cache_dir / "audio_recent.ogg"
        recent.write_bytes(b"fresh")

        removed = cleanup_audio_cache(max_age_hours=24)
        assert removed == 0
        assert recent.exists()

    def test_returns_removed_count(self):
        cache_dir = get_audio_cache_dir()
        old_time = time.time() - 48 * 3600
        for i in range(3):
            f = cache_dir / f"audio_old_{i}.ogg"
            f.write_bytes(b"x")
            os.utime(f, (old_time, old_time))

        assert cleanup_audio_cache(max_age_hours=24) == 3

    def test_empty_cache_dir(self):
        assert cleanup_audio_cache(max_age_hours=24) == 0
