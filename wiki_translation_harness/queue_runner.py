"""Queue mode: drain articles from the shared wiki-translation-queue repo
(github.com/arianit/wiki-translate-queue) instead of a manually-supplied
--title/--titles/--category. Delegates the actual translation to the same
run_pipeline() the CLI's normal modes use; this module only owns picking
articles off the shared queue and reporting results back to it.

See that repo's README for the claim/work/finish protocol this implements
via its queue_lib.py (dynamically imported from the cloned repo — this
package doesn't vendor a copy, to avoid two sources of truth).
"""
from __future__ import annotations

import functools
import importlib.util
import logging
from pathlib import Path

from wiki_translation_harness.config import Config
from wiki_translation_harness.locks import claimant_alive, queue_lock, worker_id
from wiki_translation_harness.progress import ProgressReporter
from wiki_translation_harness.sources import ArticleInput, parse_source_ref
from wiki_translation_harness.statistics import StatsTracker
from wiki_translation_harness.resume import unfinished_inputs

logger = logging.getLogger("wiki_translation_harness.queue_runner")

DEFAULT_QUEUE_REPO_DIR = Path("~/code/wiki-translation-queue").expanduser()

# queue_lib has no "unclaimed only" switch, so --fresh passes a stale threshold
# (100 years) that no real claim can exceed, which stops it reclaiming abandoned
# CLAIMED lines. (A CLAIMED line with a missing/unparseable timestamp still
# counts as stale there.)
NEVER_STALE_HOURS = 24 * 365 * 100


def load_queue_lib(repo_dir: Path):
    spec = importlib.util.spec_from_file_location("queue_lib", repo_dir / "queue_lib.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _under_queue_lock(fn):
    """Runs a queue-mutating function while holding the machine-wide queue
    lock. Every process on this computer edits the same clone; git can't
    detect a race between commits made in that one clone, and another
    process's `reset --hard` would wipe an uncommitted edit."""
    @functools.wraps(fn)
    def wrapper(queue_repo_dir, *args, **kwargs):
        with queue_lock(queue_repo_dir):
            return fn(queue_repo_dir, *args, **kwargs)
    return wrapper


def bind_worker(queue_lib, config: Config) -> None:
    """Makes this process's claims name it (`host:pid:provider`, not just the
    host) and stops other local runs reclaiming a claim whose process is
    still alive, however long its article takes (the age check alone would
    hand a slow article to a second worker after --stale-hours)."""
    marker, is_stale = queue_lib.claim_marker, queue_lib.is_stale_claim
    me = worker_id(config.provider)
    queue_lib.claim_marker = lambda hostname=None: marker(hostname or me)
    queue_lib.is_stale_claim = lambda extra, *args, **kwargs: (
        False if extra and claimant_alive(extra[0]) else is_stale(extra, *args, **kwargs)
    )


REVIEW_MARKER = "review="


def review_marker(model: str, provider: str) -> str:
    return f"{REVIEW_MARKER}{model}@{provider}"


def _status_with_engine(status: str, article_stats: StatsTracker, config: Config) -> str:
    """Appends `model@provider` to a DONE/FAILED status so
    totranslate.txt records which engine actually produced (or failed to
    produce) each result. Prefers the effective provider/model pipeline.py
    stamped on article_stats.stats (reflects a mid-run fallback-provider
    switch, see ensure_fallback_engine); falls back to the run's starting
    config if the article crashed before that was ever set.

    A DONE article that went through the semantic review pass also gets a
    third field, `review=model@provider`, naming the reviewer."""
    stats = article_stats.stats
    provider = stats.provider or config.provider
    model = stats.model or config.model
    line = f"{status}\t{model}@{provider}"
    if status == "DONE" and stats.review_model:
        line += f"\t{review_marker(stats.review_model, stats.review_provider or provider)}"
    return line


def unreviewed_done_lines(queue_repo_dir: Path, draft_engine: str) -> list[tuple[int, str]]:
    """[(line_no, url)] of DONE lines produced by `draft_engine` (a
    `model@provider` string) that carry no `review=` field yet."""
    queue_lib = load_queue_lib(queue_repo_dir)
    return [
        (line_no, url)
        for line_no, url, status, extra in queue_lib.parse_queue(queue_repo_dir)
        if status == "DONE" and extra and extra[0] == draft_engine
        and not any(field.startswith(REVIEW_MARKER) for field in extra[1:])
    ]


@_under_queue_lock
def mark_reviewed(
    queue_repo_dir: Path, url: str, review_model: str, review_provider: str, max_attempts: int = 5
) -> None:
    """Adds the `review=model@provider` field to a DONE line and pushes it.
    Pulls first and re-finds the line by URL, since other machines edit the
    same file; retries on a push conflict like queue_lib.finish_line."""
    queue_lib = load_queue_lib(queue_repo_dir)
    marker = review_marker(review_model, review_provider)
    for attempt in range(1, max_attempts + 1):
        queue_lib.git_pull(queue_repo_dir)
        for line_no, line_url, status, extra in queue_lib.parse_queue(queue_repo_dir):
            if line_url != url:
                continue
            if status != "DONE":
                raise ValueError(f"{url} is {status or 'pending'} in the queue, not DONE")
            kept = [field for field in extra if not field.startswith(REVIEW_MARKER)]
            queue_lib.set_line_status(queue_repo_dir, line_no, "\t".join(["DONE", *kept, marker]))
            break
        else:
            raise ValueError(f"{url} not found in the queue")
        try:
            queue_lib.git_commit_push(queue_repo_dir, f"review: {url} -> {marker}")
            return
        except queue_lib.QueueSyncError:
            if attempt == max_attempts:
                raise
            queue_lib.git_reset_hard_to_remote(queue_repo_dir)


def _queue_key(ref: str, default_lang: str) -> tuple[str, str]:
    lang, title = parse_source_ref(ref)
    return lang or default_lang, title.replace("_", " ").strip()


@_under_queue_lock
def mark_title_results(
    queue_repo_dir: Path,
    outcomes: list[tuple[str | None, str, bool]],
    done_status: str,
    failed_status: str,
    default_lang: str,
    max_attempts: int = 5,
) -> list[str]:
    """Records the result of a manual --title/--titles run on the shared queue.

    `outcomes` is [(source_lang, title, succeeded)]; done_status/failed_status
    are full status fields as _status_with_engine builds them. Only articles
    that have a queue line are touched. A success sets DONE on any such line
    (a manual run is as good as a claimed one, and stops another worker
    picking it up). A failure only updates a line that is already CLAIMED or
    FAILED: it never downgrades DONE, and never takes a still-pending line off
    the queue. Returns the URLs updated; pulls first and retries on a push
    conflict like mark_reviewed."""
    wanted = {(lang or default_lang, title.replace("_", " ").strip()): ok for lang, title, ok in outcomes}
    queue_lib = load_queue_lib(queue_repo_dir)
    for attempt in range(1, max_attempts + 1):
        queue_lib.git_pull(queue_repo_dir)
        updated: list[str] = []
        for line_no, url, status, _extra in queue_lib.parse_queue(queue_repo_dir):
            ok = wanted.get(_queue_key(url, default_lang))
            if ok is None or (not ok and status not in ("FAILED", "CLAIMED")):
                continue
            queue_lib.set_line_status(queue_repo_dir, line_no, done_status if ok else failed_status)
            updated.append(url)
        if not updated:
            return []
        try:
            queue_lib.git_commit_push(
                queue_repo_dir, f"result: manual run -> {', '.join(updated)}"[:200]
            )
            return updated
        except queue_lib.QueueSyncError:
            if attempt == max_attempts:
                raise
            queue_lib.git_reset_hard_to_remote(queue_repo_dir)
    return []


async def run_queue_mode(
    config: Config,
    queue_repo_dir: Path = DEFAULT_QUEUE_REPO_DIR,
    max_articles: int = 10,
    stale_hours: float = 3.0,
    fresh: bool = False,
) -> StatsTracker:
    """fresh=True starts only articles no agent has begun: the local
    partial_output/ backlog is skipped and stale-CLAIMED lines are not
    reclaimed (stale_hours is overridden with NEVER_STALE_HOURS), so every article this run translates is
    a fresh queue line. Otherwise (default) unfinished local drafts are
    retried first and abandoned claims are picked up as before."""
    from wiki_translation_harness.pipeline import run_pipeline  # local import: mirrors cli.py

    queue_lib = load_queue_lib(queue_repo_dir)
    bind_worker(queue_lib, config)
    queue_lib.sync_to_remote(queue_repo_dir)

    stats_tracker = StatsTracker()
    stats = stats_tracker.stats
    processed = 0

    # Retry the local backlog before claiming fresh shared-queue work. Disable
    # discovery below so a failed retry is attempted only once in this run.
    # --fresh skips this: a draft left by an earlier (or concurrent) agent is
    # not ours to finish, so we go straight to untouched queue lines.
    backlog = [] if fresh else unfinished_inputs(config, [])[:max_articles]
    if backlog:
        logger.info("Retrying %d unfinished local article(s) before claiming queue work", len(backlog))
        reporter = ProgressReporter(stats, config.workers, on_event=logger.info)
        await run_pipeline(
            config.model_copy(update={"sequential": True}), backlog,
            reporter=reporter, stats_tracker=stats_tracker, resume_unfinished=False,
        )
        processed += len(backlog)

    while processed < max_articles:
        with queue_lock(queue_repo_dir):
            claimed = queue_lib.claim_next_pending(
                queue_repo_dir, stale_hours=NEVER_STALE_HOURS if fresh else stale_hours
            )
        if claimed is None:
            logger.info("Shared queue has no pending articles.")
            break
        line_no, url = claimed
        lang, title = parse_source_ref(url)
        logger.info("Claimed queue line %d: %s", line_no, url)

        article_stats = StatsTracker()
        # Constructed but never entered as a context manager -- refresh()
        # no-ops whenever self._live is None (only set inside __enter__),
        # so this is a pure event emitter with zero Rich rendering
        # overhead. Gives queue-mode runs (this nightly-cron path has no
        # interactive terminal) the same per-chunk/per-article visibility
        # in logs/run.log that the interactive CLI gets from its Live table.
        reporter = ProgressReporter(article_stats.stats, config.workers, on_event=logger.info)
        try:
            await run_pipeline(
                config,
                [ArticleInput(title=title, source_lang=lang)],
                force=True,  # the queue is the source of truth for done/failed, not output_dir presence
                reporter=reporter,
                stats_tracker=article_stats,
                resume_unfinished=False,
            )
        except Exception as exc:  # noqa: BLE001 - must still record FAILED and move on to the next article
            logger.exception("Queue article %s crashed", url)
            status_line = _status_with_engine("FAILED", article_stats, config)
            try:
                with queue_lock(queue_repo_dir):
                    queue_lib.finish_line(queue_repo_dir, line_no, url, status_line, reason=str(exc)[:200])
            except queue_lib.QueueSyncError as sync_exc:
                logger.error("Could not push FAILED result for %s: %s", url, sync_exc)
            stats.articles_failed += 1
            # Whatever tokens/cost this article burned before crashing are
            # still real spend -- merge them in even on the failure path,
            # not just on success (see RunStats.merge_usage_from).
            stats.merge_usage_from(article_stats.stats)
            processed += 1
            continue

        if article_stats.stats.articles_skipped >= 1 and not (
            article_stats.stats.articles_completed or article_stats.stats.articles_failed
        ):
            # force=True means the only skip is the per-article lock: another
            # process on this machine is translating it (e.g. a manual
            # --title run), and it will record the result on this line itself.
            logger.info("Skipped %s: another local process is translating it; leaving the claim as is", url)
            processed += 1
            continue

        if article_stats.stats.articles_completed >= 1:
            status, reason = "DONE", None
            stats.articles_completed += 1
        else:
            status, reason = "FAILED", "translation did not complete (see run.log for this article)"
            stats.articles_failed += 1
        status_line = _status_with_engine(status, article_stats, config)
        try:
            with queue_lock(queue_repo_dir):
                queue_lib.finish_line(queue_repo_dir, line_no, url, status_line, reason=reason)
        except queue_lib.QueueSyncError as sync_exc:
            logger.error("Could not push %s result for %s: %s", status, url, sync_exc)
        # Each article gets its own fresh RunStats from run_pipeline (and
        # its own stats.json, overwritten per article) -- without this, the
        # queue-run-level stats_tracker returned below would never see any
        # article's tokens/cost/per-model usage at all, and the CLI's final
        # "Estimated cost: $X" line would always print $0.
        stats.merge_usage_from(article_stats.stats)
        processed += 1

    return stats_tracker
