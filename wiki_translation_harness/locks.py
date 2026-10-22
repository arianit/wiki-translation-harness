"""Machine-local locks so concurrent harness processes don't collide.

Several `wiki-translation-harness` processes (different providers/models) can
run on one computer. They share the queue clone and the output directories,
and git's push check can't see a race between processes that commit to the
same local clone. These flock-based locks cover that case; the kernel drops
them when a process exits or crashes, so there are no stale lock files to
clean up. Linux/macOS only (fcntl)."""
from __future__ import annotations

import fcntl
import hashlib
import os
import re
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOCK_DIR_ENV = "WIKI_HARNESS_LOCK_DIR"
DEFAULT_LOCK_DIR = Path("~/.cache/wiki-translation-harness/locks")


def lock_dir() -> Path:
    path = Path(os.environ.get(LOCK_DIR_ENV) or DEFAULT_LOCK_DIR).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _open_lock(path: Path, *, blocking: bool) -> int | None:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def release(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def queue_lock(queue_repo_dir: Path) -> Iterator[None]:
    """Serialises every read-modify-push of the shared queue clone (claim,
    finish, review/manual marks) across local processes. Held across the
    git round trip, which is a few seconds."""
    key = hashlib.sha1(os.path.realpath(str(queue_repo_dir)).encode()).hexdigest()[:10]
    fd = _open_lock(lock_dir() / f"queue-{key}.lock", blocking=True)
    try:
        yield
    finally:
        release(fd)


def try_lock_article(target_lang: str, name: str) -> int | None:
    """Non-blocking exclusive lock on one article (`name` is the sanitised
    output stem). Returns the fd to pass to release(), or None when another
    process is already translating it."""
    safe = re.sub(r"[^\w.-]", "_", f"{target_lang}-{name}")[:150]
    return _open_lock(lock_dir() / f"article-{safe}.lock", blocking=False)


def worker_id(provider: str) -> str:
    """`host:pid:provider` -- goes in the CLAIMED line's host field so a claim
    names the exact process that holds it (and which engine it runs)."""
    return f"{socket.gethostname()}:{os.getpid()}:{provider}"


def claimant_alive(claim_host_field: str) -> bool:
    """True if a `host:pid:provider` claim was made by a process that is
    still running on this machine. Anything else (another host, the old
    bare-hostname format) returns False, leaving it to the age check."""
    parts = claim_host_field.split(":")
    if len(parts) < 3 or parts[0] != socket.gethostname() or not parts[1].isdigit():
        return False
    pid = int(parts[1])
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True
    # Guards against pid reuse by an unrelated process after a crash.
    return b"wiki_translation_harness" in cmdline or b"wiki-translation-harness" in cmdline
