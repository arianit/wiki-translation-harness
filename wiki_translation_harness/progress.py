"""Rich live progress display: one line per article (bar, sections done, time,
what is in flight), plus run-level ETA and cost. Without a live display the
same state feeds a compact per-article text trail for logs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from rich.console import Console, Group
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from wiki_translation_harness.models import RunStats


@dataclass
class WorkerSlot:
    slot_id: int
    article: str = ""
    section: str = ""
    action: str = "idle"  # idle | translating | repairing


@dataclass
class ArticleProgress:
    title: str
    total: int = 0  # 0 until the plan (or the first chunk) says how many sections
    done: int = 0
    started: float | None = None
    finished: float | None = None
    outcome: str = ""  # "" while running; completed | failed | needs_human_review
    active: list[str] = field(default_factory=list)  # sections being translated now

    @property
    def status(self) -> str:
        if self.outcome:
            return self.outcome
        if self.started is None:
            return "queued"
        if self.total and self.done >= self.total:
            return "finishing"  # every section is in; assembling, validating, reviewing
        return "running"

    def elapsed(self, now: float) -> float | None:
        if self.started is None:
            return None
        return (self.finished or now) - self.started


# How many finished and queued articles the live view lists before folding
# the rest into a "+N" line, so a long queue keeps the screen short.
_MAX_FINISHED_SHOWN = 4
_MAX_QUEUED_SHOWN = 3
_LOG_STEPS = 10  # a progress line per article roughly every 1/_LOG_STEPS of its sections

_OUTCOME_STYLE = {
    "completed": ("✔", "green", "done"),
    "failed": ("✖", "red", "failed"),
    "needs_human_review": ("⚠", "yellow", "needs human review"),
}


@dataclass
class ProgressState:
    total_articles: int = 0
    articles_done: int = 0
    total_chunks: int = 0
    chunks_done: int = 0
    estimated_total_cost_usd: float = 0.0
    last_retry: str = ""
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def eta_seconds(self) -> float | None:
        if self.chunks_done == 0 or self.total_chunks == 0:
            return None
        elapsed = time.monotonic() - self._start_time
        rate = self.chunks_done / elapsed
        remaining = self.total_chunks - self.chunks_done
        return remaining / rate if rate > 0 else None


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class ProgressReporter:
    def __init__(
        self,
        stats: RunStats,
        workers: int,
        console: Console | None = None,
        on_event: Callable[[str], None] | None = None,
    ):
        self.stats = stats
        self.state = ProgressState()
        self.slots = [WorkerSlot(slot_id=i) for i in range(workers)]
        self.articles: dict[str, ArticleProgress] = {}
        self.console = console or Console()
        self._live: Live | None = None
        # Optional text sink, independent of the Rich Live view below — e.g.
        # queue_runner.py wires this to logger.info so non-interactive runs
        # (which never enter/render Live) still get a per-article trail in
        # logs/run.log: a start line, a progress line about every tenth of
        # the article, and a finish line.
        self.on_event = on_event

    def __enter__(self) -> "ProgressReporter":
        self._live = Live(self._render(), console=self.console, refresh_per_second=4, transient=False)
        self._live.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(*exc)

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    @property
    def is_live(self) -> bool:
        """True only for an interactive run with the Rich table actually
        rendering (cli.py's `with ProgressReporter(...)`) -- queue_runner.py
        constructs one without entering it as a context manager so its
        on_event=logger.info sink is used instead, which leaves _live None.
        pipeline.py uses this to decide whether a mid-run provider-fallback
        prompt can safely use console.input() (Live active) or must instead
        proceed non-interactively (unattended/logged only)."""
        return self._live is not None

    def pause(self) -> None:
        """Stop live rendering so a synchronous console.input() prompt
        doesn't fight the table's own periodic redraw. No-op if not live."""
        if self._live is not None:
            self._live.stop()

    def resume(self) -> None:
        if self._live is not None:
            self._live.start()

    # --- event hooks called by pipeline.py ---

    def _article(self, title: str) -> ArticleProgress:
        if title not in self.articles:
            self.articles[title] = ArticleProgress(title)
        return self.articles[title]

    def set_plan(
        self,
        total_articles: int,
        total_chunks: int,
        estimated_cost_usd: float,
        article_chunks: dict[str, int] | None = None,
    ) -> None:
        """`article_chunks` maps each planned article's title to its section
        count, in run order; it is what lets the view show queued articles
        and a bar per article."""
        self.state.total_articles = total_articles
        self.state.total_chunks = total_chunks
        self.state.estimated_total_cost_usd = estimated_cost_usd
        for title, count in (article_chunks or {}).items():
            self._article(title).total = count
        self.refresh()

    def on_chunk_start(self, slot_id: int, article: str, section: str) -> None:
        slot = self.slots[slot_id]
        slot.article, slot.section, slot.action = article, section, "translating"
        art = self._article(article)
        if art.started is None:
            art.started = time.monotonic()
            if self.on_event:
                count = f"{art.total} sections" if art.total else "sections unknown"
                self.on_event(f"▶ '{article}': starting, {count}")
        art.active.append(section)
        self.refresh()

    def on_repair_start(self, slot_id: int) -> None:
        self.slots[slot_id].action = "repairing"
        self.refresh()

    def on_chunk_done(self, slot_id: int) -> None:
        slot = self.slots[slot_id]
        art = self._article(slot.article) if slot.article else None
        if art is not None:
            art.done += 1
            if slot.section in art.active:
                art.active.remove(slot.section)
            if self.on_event and self._is_log_milestone(art):
                self.on_event(self._progress_line(art, slot.section))
        slot.action = "idle"
        slot.article, slot.section = "", ""
        self.state.chunks_done += 1
        self.refresh()

    def on_article_done(self, article: str, outcome: str) -> None:
        art = self._article(article)
        art.outcome = outcome
        art.finished = time.monotonic()
        if art.started is None:
            art.started = art.finished
        if self.on_event:
            took = _fmt_duration(art.elapsed(art.finished))
            self.on_event(
                f"article '{article}' {outcome} in {took} "
                f"({self.state.articles_done + 1}/{self.state.total_articles})"
            )
        self.state.articles_done += 1
        self.refresh()

    def on_retry(self, model: str, attempt: int, reason: str, delay: float) -> None:
        self.state.last_retry = f"retry {attempt} ({reason}) in {delay:.1f}s [{model}]"
        if self.on_event:
            self.on_event(self.state.last_retry)
        self.refresh()

    # --- text trail ---

    @staticmethod
    def _is_log_milestone(art: ArticleProgress) -> bool:
        """Every section while the size is unknown; otherwise about every
        tenth and always the last. A cache-warm article finishes all its
        sections in a few seconds and would otherwise log one line each."""
        if not art.total:
            return True
        step = max(1, art.total // _LOG_STEPS)
        return art.done >= art.total or art.done % step == 0

    def _progress_line(self, art: ArticleProgress, section: str) -> str:
        counts = f"{art.done}/{art.total} sections ({art.done * 100 // art.total}%)" if art.total else f"{art.done} done"
        took = _fmt_duration(art.elapsed(time.monotonic()))
        return (
            f"'{art.title}': {counts} · {took} · last: {section} | "
            f"run: {self.stats.sections_translated} translated, "
            f"{self.stats.cache_hits} cache hits, ${self.stats.estimated_cost_usd:.4f}"
        )

    # --- rendering ---

    def _article_row(self, art: ArticleProgress, now: float) -> tuple:
        status = art.status
        if status in _OUTCOME_STYLE:
            icon_text, style, label = _OUTCOME_STYLE[status]
            icon: object = Text(icon_text, style=style)
            detail = Text(label, style=style)
        elif status == "queued":
            icon, style = Text("○", style="dim"), "dim"
            detail = Text("queued", style="dim")
        elif status == "finishing":
            icon, style = Spinner("dots", style="cyan"), "cyan"
            detail = Text("assembling, validating, reviewing", style="cyan")
        else:
            icon, style = Spinner("dots", style="cyan"), "cyan"
            shown = "; ".join(art.active[:2])
            more = f" (+{len(art.active) - 2})" if len(art.active) > 2 else ""
            detail = Text(f"now: {shown}{more}" if shown else "starting")

        total = art.total or max(art.done, 1)
        bar = ProgressBar(total=total, completed=art.done, width=20, style="grey23", complete_style=style)
        count = f"{art.done}/{art.total}" if art.total else f"{art.done}"
        percent = f"{art.done * 100 // art.total}%" if art.total else ""
        if status == "queued":
            count = f"{art.total} sections" if art.total else ""
            percent = ""
        elapsed = art.elapsed(now)
        return (
            icon,
            Text(art.title, overflow="ellipsis", no_wrap=True),
            bar,
            Text(count, justify="right"),
            Text(percent, justify="right", style="dim"),
            Text(_fmt_duration(elapsed) if elapsed is not None else "", justify="right", style="dim"),
            detail,
        )

    def _render(self) -> Group:
        header = Table.grid(padding=(0, 2))
        for _ in range(4):
            header.add_column()
        header.add_row(
            f"Articles: {self.state.articles_done}/{self.state.total_articles}",
            f"Sections: {self.state.chunks_done}/{self.state.total_chunks}",
            f"ETA: {_fmt_duration(self.state.eta_seconds)}",
            f"Cost: ${self.stats.estimated_cost_usd:.4f} / ~${self.state.estimated_total_cost_usd:.4f} est.",
        )

        arts = list(self.articles.values())
        finished = [a for a in arts if a.outcome]
        queued = [a for a in arts if a.status == "queued"]
        running = [a for a in arts if a.status in ("running", "finishing")]
        shown_finished = finished[-_MAX_FINISHED_SHOWN:]
        shown_queued = queued[:_MAX_QUEUED_SHOWN]

        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=1)
        grid.add_column(max_width=28, no_wrap=True, overflow="ellipsis")
        grid.add_column(width=20)
        grid.add_column(justify="right", min_width=5)
        grid.add_column(justify="right", min_width=4)
        grid.add_column(justify="right", min_width=6)
        grid.add_column(no_wrap=True, overflow="ellipsis")
        now = time.monotonic()
        for art in [*shown_finished, *running, *shown_queued]:
            grid.add_row(*self._article_row(art, now))

        parts: list[object] = [header]
        if len(finished) > len(shown_finished):
            parts.append(Text(f"  … {len(finished) - len(shown_finished)} earlier finished", style="dim"))
        parts.append(grid)
        if len(queued) > len(shown_queued):
            parts.append(Text(f"  … {len(queued) - len(shown_queued)} more queued", style="dim"))
        if self.state.last_retry:
            parts.append(Text(self.state.last_retry, style="yellow"))
        return Group(*parts)
