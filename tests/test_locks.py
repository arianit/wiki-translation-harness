"""Machine-local locks: two harness processes (different providers) on one
computer must not take the same queue line or translate the same article."""

import os
import socket
import threading
import time
from types import SimpleNamespace

import wiki_translation_harness.queue_runner as runner
from wiki_translation_harness import locks


def test_article_lock_is_exclusive_until_released():
    first = locks.try_lock_article("sq", "Albert_Einstein")
    assert first is not None
    assert locks.try_lock_article("sq", "Albert_Einstein") is None
    assert locks.try_lock_article("sq", "Cat") is not None  # other articles unaffected
    locks.release(first)
    again = locks.try_lock_article("sq", "Albert_Einstein")
    assert again is not None
    locks.release(again)


def test_queue_lock_serialises_claims(tmp_path):
    inside = []
    overlap = []

    def worker():
        with locks.queue_lock(tmp_path):
            inside.append(1)
            if len(inside) > 1:
                overlap.append(True)
            time.sleep(0.05)
            inside.pop()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not overlap


def test_worker_id_names_host_pid_and_provider():
    assert locks.worker_id("codex") == f"{socket.gethostname()}:{os.getpid()}:codex"


def test_claimant_alive_only_for_live_local_harness_process(monkeypatch):
    host = socket.gethostname()
    monkeypatch.setattr(locks.Path, "read_bytes", lambda self: b"python\0wiki-translation-harness\0queue")
    assert locks.claimant_alive(f"{host}:{os.getpid()}:codex")
    assert not locks.claimant_alive(f"other-host:{os.getpid()}:codex")
    assert not locks.claimant_alive(host)  # old bare-hostname claim: fall back to age
    assert not locks.claimant_alive(f"{host}:999999999:codex")  # no such pid
    monkeypatch.setattr(locks.Path, "read_bytes", lambda self: b"bash\0")
    assert not locks.claimant_alive(f"{host}:{os.getpid()}:codex")  # pid reused by something else


def test_bound_queue_lib_marks_claims_and_keeps_live_claims(monkeypatch):
    monkeypatch.setattr(locks.Path, "read_bytes", lambda self: b"wiki-translation-harness")
    queue = SimpleNamespace(
        claim_marker=lambda hostname=None: f"CLAIMED\t{hostname}\tts",
        is_stale_claim=lambda extra, stale_hours=3.0: True,
    )
    runner.bind_worker(queue, SimpleNamespace(provider="codex"))
    assert queue.claim_marker() == f"CLAIMED\t{locks.worker_id('codex')}\tts"
    mine = locks.worker_id("codex")
    other = f"{socket.gethostname()}:999999999:claude_code"
    assert not queue.is_stale_claim([mine, "2020-01-01T00:00:00+00:00"], 3.0)  # live local claimant: never stale
    assert queue.is_stale_claim([other, "2020-01-01T00:00:00+00:00"], 3.0)


def test_article_lock_excludes_another_process(tmp_path):
    import subprocess
    import sys

    held = locks.try_lock_article("sq", "Mars")
    code = (
        "from wiki_translation_harness import locks; import sys;"
        "fd = locks.try_lock_article('sq', 'Mars'); sys.exit(0 if fd is None else 1)"
    )
    env = {**os.environ, "WIKI_HARNESS_LOCK_DIR": str(locks.lock_dir())}
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0
    locks.release(held)
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 1
