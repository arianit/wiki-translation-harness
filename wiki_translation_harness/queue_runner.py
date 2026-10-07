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

import importlib.util
import logging
from pathlib import Path

from wiki_translation_harness.config import Config
from wiki_translation_harness.progress import ProgressReporter
from wiki_translation_harness.sources import ArticleInput, parse_source_ref
from wiki_translation_harness.statistics import StatsTracker
from wiki_translation_harness.resume import unfinished_inputs

logger = logging.getLogger("wiki_translation_harness.queue_runner")

DEFAULT_QUEUE_REPO_DIR = Path("~/code/wiki-translation-queue").expanduser()


def load_queue_lib(repo_dir: Path):
    spec = importlib.util.spec_from_file_location("queue_lib", repo_dir / "queue_lib.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _status_with_engine(status: str, article_stats: StatsTracker, config: Config) -> str:
    """Appends `model@provider` to a DONE/FAILED status so
    totranslate.txt records which engine actually produced (or failed to
    produce) each result. Prefers the effective provider/model pipeline.py
    stamped on article_stats.stats (reflects a mid-run fallback-provider
    switch, see ensure_fallback_engine); falls back to the run's starting
    config if the article crashed before that was ever set."""
    provider = article_stats.stats.provider or config.provider
    model = article_stats.stats.model or config.model
    return f"{status}\t{model}@{provider}"


async def run_queue_mode(
    config: Config,
    queue_repo_dir: Path = DEFAULT_QUEUE_REPO_DIR,
    max_articles: int = 10,
    stale_hours: float = 3.0,
) -> StatsTracker:
    from wiki_translation_harness.pipeline import run_pipeline  # local import: mirrors cli.py

    queue_lib = load_queue_lib(queue_repo_dir)
    queue_lib.sync_to_remote(queue_repo_dir)

    stats_tracker = StatsTracker()
    stats = stats_tracker.stats
    processed = 0

    # Retry the local backlog before claiming fresh shared-queue work. Disable
    # discovery below so a failed retry is attempted only once in this run.
    backlog = unfinished_inputs(config, [])[:max_articles]
    if backlog:
        logger.info("Retrying %d unfinished local article(s) before claiming queue work", len(backlog))
        reporter = ProgressReporter(stats, config.workers, on_event=logger.info)
        await run_pipeline(
            config.model_copy(update={"sequential": True}), backlog,
            reporter=reporter, stats_tracker=stats_tracker, resume_unfinished=False,
        )
        processed += len(backlog)

    while processed < max_articles:
        claimed = queue_lib.claim_next_pending(queue_repo_dir, stale_hours=stale_hours)
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

        if article_stats.stats.articles_completed >= 1:
            status, reason = "DONE", None
            stats.articles_completed += 1
        else:
            status, reason = "FAILED", "translation did not complete (see run.log for this article)"
            stats.articles_failed += 1
        status_line = _status_with_engine(status, article_stats, config)
        try:
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
