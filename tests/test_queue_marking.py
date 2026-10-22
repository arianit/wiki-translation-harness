"""mark_title_results records a manual --title run on the shared queue.

queue_lib lives in the queue repo, so these tests give it a small in-memory
stand-in with the same parse_queue/set_line_status contract (and a git layer
that just records commits)."""

from types import SimpleNamespace
from unittest.mock import patch

import wiki_translation_harness.queue_runner as runner

DONE = "DONE\tgpt-6-luna@codex\treview=gpt-6-luna@codex"
FAILED = "FAILED\tgpt-6-luna@codex"


class FakeQueue(SimpleNamespace):
    def __init__(self, lines):
        super().__init__(lines=lines, commits=[], QueueSyncError=RuntimeError)

    def git_pull(self, repo_dir):
        pass

    def git_commit_push(self, repo_dir, message):
        self.commits.append(message)
        return True

    def git_reset_hard_to_remote(self, repo_dir):
        pass

    def parse_queue(self, repo_dir):
        rows = []
        for number, raw in enumerate(self.lines, start=1):
            url, *rest = raw.split("\t")
            status = rest[0] if rest and rest[0] else None
            rows.append((number, url, status, rest[1:]))
        return rows

    def set_line_status(self, repo_dir, line_no, status):
        url = self.lines[line_no - 1].split("\t")[0]
        self.lines[line_no - 1] = f"{url}\t{status}"


def mark(lines, outcomes):
    queue = FakeQueue(lines)
    with patch.object(runner, "load_queue_lib", return_value=queue):
        updated = runner.mark_title_results(None, outcomes, DONE, FAILED, default_lang="en")
    return queue, updated


def test_success_marks_a_failed_line_done():
    queue, updated = mark(
        ["https://en.wikipedia.org/wiki/Cluj-Napoca\tFAILED\tgpt-6-luna@codex"],
        [("en", "Cluj-Napoca", True)],
    )
    assert queue.lines == [f"https://en.wikipedia.org/wiki/Cluj-Napoca\t{DONE}"]
    assert updated == ["https://en.wikipedia.org/wiki/Cluj-Napoca"]
    assert len(queue.commits) == 1


def test_success_marks_a_pending_line_and_matches_title_spelling_and_default_lang():
    queue, _ = mark(
        ["https://en.wikipedia.org/wiki/Mount_Everest", "https://en.wikipedia.org/wiki/Other"],
        [(None, "Mount Everest", True)],
    )
    assert queue.lines[0].endswith(DONE)
    assert queue.lines[1] == "https://en.wikipedia.org/wiki/Other"


def test_failure_never_downgrades_done_or_takes_a_pending_line():
    done = "https://en.wikipedia.org/wiki/A\tDONE\tx@y"
    pending = "https://en.wikipedia.org/wiki/B"
    queue, updated = mark([done, pending], [("en", "A", False), ("en", "B", False)])
    assert queue.lines == [done, pending]
    assert updated == [] and queue.commits == []


def test_failure_updates_a_claimed_line():
    queue, _ = mark(
        ["https://en.wikipedia.org/wiki/A\tCLAIMED\tpop-os\t2026-10-17T08:49:47+02:00"],
        [("en", "A", False)],
    )
    assert queue.lines == [f"https://en.wikipedia.org/wiki/A\t{FAILED}"]


def test_other_language_editions_do_not_match():
    line = "https://sq.wikipedia.org/wiki/Kosova\tFAILED\tx@y"
    queue, updated = mark([line], [("en", "Kosova", True)])
    assert queue.lines == [line] and updated == []
