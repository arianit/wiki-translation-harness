import pytest


@pytest.fixture(autouse=True)
def _isolated_lock_dir(tmp_path, monkeypatch):
    """Machine-local locks live under the home dir by default; keep tests out of it."""
    monkeypatch.setenv("WIKI_HARNESS_LOCK_DIR", str(tmp_path / "locks"))
