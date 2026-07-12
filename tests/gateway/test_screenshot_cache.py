"""
Tests for browser-screenshot cache cleanup in gateway/platforms/base.py.

Covers: cleanup_screenshot_cache (timer-based safety net for the trigger-based
cleanup in tools/browser_tool.py), get_screenshot_cache_dir.
"""

import os
import time

import pytest

from gateway.platforms.base import (
    cleanup_screenshot_cache,
    get_screenshot_cache_dir,
)

# ---------------------------------------------------------------------------
# Fixture: redirect SCREENSHOT_CACHE_DIR to a temp directory for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point the module-level SCREENSHOT_CACHE_DIR to a fresh tmp_path."""
    monkeypatch.setattr(
        "gateway.platforms.base.SCREENSHOT_CACHE_DIR", tmp_path / "screenshot_cache"
    )


# ---------------------------------------------------------------------------
# TestCleanupScreenshotCache
# ---------------------------------------------------------------------------

class TestCleanupScreenshotCache:
    def test_removes_old_files(self, tmp_path):
        cache_dir = get_screenshot_cache_dir()
        old_file = cache_dir / "browser_screenshot_old.png"
        old_file.write_bytes(b"old")
        old_mtime = time.time() - 48 * 3600
        os.utime(old_file, (old_mtime, old_mtime))

        removed = cleanup_screenshot_cache(max_age_hours=24)
        assert removed == 1
        assert not old_file.exists()

    def test_keeps_recent_files(self):
        cache_dir = get_screenshot_cache_dir()
        recent = cache_dir / "browser_screenshot_recent.png"
        recent.write_bytes(b"fresh")

        removed = cleanup_screenshot_cache(max_age_hours=24)
        assert removed == 0
        assert recent.exists()

    def test_returns_removed_count(self):
        cache_dir = get_screenshot_cache_dir()
        old_time = time.time() - 48 * 3600
        for i in range(3):
            f = cache_dir / f"browser_screenshot_old_{i}.png"
            f.write_bytes(b"x")
            os.utime(f, (old_time, old_time))

        assert cleanup_screenshot_cache(max_age_hours=24) == 3

    def test_empty_cache_dir(self):
        assert cleanup_screenshot_cache(max_age_hours=24) == 0
