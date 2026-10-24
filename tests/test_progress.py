"""ProgressReporter's on_event hook — the mechanism queue_runner.py uses to
get a per-article text trail into logs/run.log for non-interactive runs,
independent of the Rich Live view (which is never entered here, same as
queue_runner.py's usage -- confirms refresh() no-ops without a Live) — and
the per-article rows that view is drawn from."""

from io import StringIO

from rich.console import Console

from wiki_translation_harness.models import RunStats
from wiki_translation_harness.progress import ProgressReporter


def _reporter(on_event=None) -> ProgressReporter:
    return ProgressReporter(RunStats(), workers=2, on_event=on_event)


def test_on_chunk_done_emits_before_clearing_slot():
    events: list[str] = []
    reporter = _reporter(on_event=events.append)
    reporter.on_chunk_start(0, "Mars", "Geography")
    events.clear()
    reporter.on_chunk_done(0)
    assert len(events) == 1
    # slot.article/section must still be in the message, even though
    # on_chunk_done clears them on the slot right after
    assert "Mars" in events[0]
    assert "Geography" in events[0]
    assert reporter.slots[0].article == ""  # confirms the clear still happens


def test_on_chunk_done_includes_cumulative_stats():
    stats = RunStats(sections_translated=3, cache_hits=1, estimated_cost_usd=0.05)
    events: list[str] = []
    reporter = ProgressReporter(stats, workers=1, on_event=events.append)
    reporter.on_chunk_start(0, "Mars", "Geography")
    reporter.on_chunk_done(0)
    assert "3 translated" in events[-1]
    assert "1 cache hits" in events[-1]
    assert "0.0500" in events[-1]


def test_on_article_done_emits_title_and_outcome():
    events: list[str] = []
    reporter = _reporter(on_event=events.append)
    reporter.on_article_done("Neptune", "failed")
    assert "Neptune" in events[0]
    assert "failed" in events[0]




def _run_article(reporter, title: str, sections: int) -> None:
    for i in range(sections):
        reporter.on_chunk_start(0, title, f"Section {i}")
        reporter.on_chunk_done(0)


def test_text_trail_is_one_start_line_a_line_per_tenth_and_a_finish_line():
    events: list[str] = []
    reporter = _reporter(on_event=events.append)
    reporter.set_plan(1, 40, 0.0, {"Nigeria": 40})
    _run_article(reporter, "Nigeria", 40)
    reporter.on_article_done("Nigeria", "completed")
    assert events[0] == "▶ 'Nigeria': starting, 40 sections"
    progress = [e for e in events if "sections (" in e]
    assert len(progress) == 10  # 4, 8, ... 40
    assert "'Nigeria': 40/40 sections (100%)" in progress[-1]
    assert events[-1].startswith("article 'Nigeria' completed in ")
    assert events[-1].endswith("(1/1)")


def test_text_trail_logs_every_section_when_the_size_is_unknown():
    events: list[str] = []
    reporter = _reporter(on_event=events.append)
    _run_article(reporter, "Mars", 3)
    assert len([e for e in events if "last:" in e]) == 3
    assert "'Mars': 3 done" in events[-1]


def test_article_status_follows_the_run():
    reporter = _reporter()
    reporter.set_plan(2, 3, 0.0, {"A": 2, "B": 1})
    assert [a.status for a in reporter.articles.values()] == ["queued", "queued"]
    reporter.on_chunk_start(0, "A", "Intro")
    assert reporter.articles["A"].status == "running"
    assert reporter.articles["A"].active == ["Intro"]
    reporter.on_chunk_done(0)
    _run_article(reporter, "A", 1)
    assert reporter.articles["A"].status == "finishing"
    reporter.on_article_done("A", "needs_human_review")
    assert reporter.articles["A"].status == "needs_human_review"
    assert reporter.articles["B"].status == "queued"


def _rendered(reporter) -> str:
    console = Console(file=StringIO(), width=120, force_terminal=False, color_system=None)
    console.print(reporter._render())
    return console.file.getvalue()


def test_render_shows_a_row_per_article_with_counts_and_state():
    reporter = _reporter()
    reporter.set_plan(3, 10, 0.0, {"Nigeria": 4, "Los Angeles": 4, "Mount Everest": 2})
    _run_article(reporter, "Nigeria", 4)
    reporter.on_article_done("Nigeria", "completed")
    _run_article(reporter, "Los Angeles", 1)
    reporter.on_chunk_start(0, "Los Angeles", "Economy")
    out = _rendered(reporter)
    assert "Nigeria" in out and "4/4" in out and "done" in out
    assert "Los Angeles" in out and "1/4" in out and "25%" in out and "now: Economy" in out
    assert "Mount Everest" in out and "queued" in out


def test_render_folds_long_finished_and_queued_lists():
    reporter = _reporter()
    titles = [f"Article {i}" for i in range(12)]
    reporter.set_plan(12, 12, 0.0, {t: 1 for t in titles})
    for title in titles[:6]:
        _run_article(reporter, title, 1)
        reporter.on_article_done(title, "completed")
    out = _rendered(reporter)
    assert "… 2 earlier finished" in out
    assert "Article 0" not in out and "Article 5" in out
    assert "… 3 more queued" in out
