"""Run the semantic review pass over translations that are already saved.

The normal pipeline reviews an article right after drafting it. This is the
same pass (pipeline.run_review_pass) for output written without a review
model -- e.g. drafts from the cheap tier -- so a stronger model can read
them next to the English source afterwards.

The saved .wiki file is the starting point. It is split at its top-level
headings into stand-in chunks, so review findings are localized and repaired
section by section exactly as in the normal pipeline. The English source is
fetched at the revision recorded in the file's attribution block, so the
reviewer compares against the text that was actually translated.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from wiki_translation_harness.config import resolve_review_model, resolve_review_provider
from wiki_translation_harness.engines import build_client_pool
from wiki_translation_harness.mediawiki import MediaWikiClientPool
from wiki_translation_harness.models import Chunk, ChunkStatus, Config, ValidationIssue
from wiki_translation_harness.output import output_path_for
from wiki_translation_harness.report import build_review_section
from wiki_translation_harness.skill_loader import load_skill
from wiki_translation_harness.sources import ArticleInput
from wiki_translation_harness.statistics import StatsTracker
from wiki_translation_harness.cache import VerificationCache

logger = logging.getLogger("wiki_translation_harness.rereview")

_ATTRIBUTION_START = "<!--\nNUK ËSHTË PJESË"
_REVID_RE = re.compile(r"\{\{Përkthyer nga\|[^|}]*\|[^|}]*\|[^|}]*\|(\d+)\}\}")
_HEADING_RE = re.compile(r"^={2,4}[^=].*[^=]={2,4}\s*$")
_MAX_PIECE_CHARS = 12_000
_REPORT_SECTION_RE = re.compile(r"## Semantic review\n.*?(?=\n## |\n<!--\nNUK|\Z)", re.DOTALL)
PRE_REVIEW_DIRNAME = "pre-review"


@dataclass
class RereviewResult:
    title: str
    review_model: str
    review_provider: str
    findings: list[ValidationIssue] = field(default_factory=list)
    rounds: int = 0
    changed: bool = False


def split_attribution(text: str) -> tuple[str, str]:
    """(article body, trailing attribution block incl. its leading blank line)."""
    idx = text.rfind(_ATTRIBUTION_START)
    if idx == -1:
        return text, ""
    body = text[:idx].rstrip("\n")
    return body, text[len(body):]


def revid_from_attribution(attribution: str) -> int | None:
    match = _REVID_RE.search(attribution)
    return int(match.group(1)) if match else None


def split_into_stand_in_chunks(title: str, body: str, source_lang: str) -> list[Chunk]:
    """Cuts saved wikitext into pieces at headings, then cuts any piece over
    _MAX_PIECE_CHARS at blank lines outside templates and tables, so a
    repair call only rewrites a section-sized slice. Joining the pieces'
    translated_text reproduces `body` exactly."""
    sections: list[tuple[str, list[str]]] = [("(lead)", [])]
    for line in body.splitlines(keepends=True):
        if _HEADING_RE.match(line):
            sections.append((line.strip().strip("=").strip(), []))
        sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for heading, lines in sections:
        for text in _split_at_blank_lines(lines):
            chunks.append(Chunk(
                article_title=title, section_titles=[heading], order=len(chunks), text=text,
                token_estimate=0, source_lang=source_lang, status=ChunkStatus.TRANSLATED,
                translated_text=text,
            ))
    return chunks


def _split_at_blank_lines(lines: list[str]) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    depth = 0  # open "{{" templates plus "{|" tables
    for line in lines:
        depth += line.count("{{") - line.count("}}") + line.count("{|") - line.count("|}")
        current.append(line)
        size += len(line)
        if size >= _MAX_PIECE_CHARS and depth <= 0 and not line.strip():
            pieces.append("".join(current))
            current, size = [], 0
    if current:
        pieces.append("".join(current))
    return pieces


def _update_report(report_path: Path, review_model: str, findings: list[ValidationIssue]) -> None:
    if not report_path.exists():
        return
    text = report_path.read_text(encoding="utf-8")
    section = "\n".join(build_review_section(review_model, findings)) + "\n"
    if _REPORT_SECTION_RE.search(text):
        text = _REPORT_SECTION_RE.sub(lambda _: section, text, count=1)
    elif _ATTRIBUTION_START in text:
        idx = text.index(_ATTRIBUTION_START)
        text = f"{text[:idx]}{section}\n{text[idx:]}"
    else:
        text = f"{text.rstrip()}\n\n{section}"
    report_path.write_text(text, encoding="utf-8")


async def rereview_articles(
    config: Config,
    titles: list[str],
    on_done=None,
) -> tuple[list[RereviewResult], StatsTracker]:
    """Reviews each saved article in config.output_dir with the configured
    review model, rewrites the .wiki (original kept under
    output_dir/pre-review/) and updates its report. `on_done(result)` is
    called after each article is saved, so a caller can record progress
    even if a later article fails. A title that cannot be reviewed is
    logged and skipped."""
    # Imported here: pipeline imports a lot and this module is optional.
    from wiki_translation_harness.pipeline import _plan_article, run_review_pass

    review_model = resolve_review_model(config)
    if review_model is None:
        raise ValueError("No review model configured (set review_model/review_provider).")
    review_provider = resolve_review_provider(config)

    skill = load_skill(config.skill_path, config.include_skill_references, config.skill_git_ref)
    qa_skill = load_skill(config.qa_skill_path, config.include_skill_references, config.skill_git_ref)
    verification_cache = VerificationCache(config.verification_db_path) if config.verify_links else None
    mw_pool = MediaWikiClientPool(config.user_agent, config.source_lang, config.source_wiki_api)
    client_pool, config = build_client_pool(config)
    review_client = client_pool.get(review_provider)
    wikidata_client = (
        httpx.AsyncClient(headers={"User-Agent": config.user_agent}, timeout=config.wikidata_timeout_s)
        if config.verify_links else None
    )
    citation_client = (
        httpx.AsyncClient(headers={"User-Agent": config.user_agent}) if config.fill_citation_languages else None
    )
    stats_tracker = StatsTracker()
    stats = stats_tracker.stats
    fetch_sem = asyncio.Semaphore(1)
    results: list[RereviewResult] = []

    try:
        review_pricing = await review_client.get_pricing_for(review_model)
        target_mw_client = mw_pool.get(config.target_lang)

        for title in titles:
            path = output_path_for(config.output_dir, title)
            if not path.exists():
                logger.error("No saved translation for %r at %s", title, path)
                continue
            original = path.read_text(encoding="utf-8")
            body, attribution = split_attribution(original)
            revid = revid_from_attribution(attribution)
            if revid is None:
                logger.warning("No source revision found in %s; reviewing against the current revision", path)

            try:
                plan = await _plan_article(
                    mw_pool, ArticleInput(title=title, source_lang=config.source_lang, revid=revid),
                    config, fetch_sem, wikidata_client, verification_cache,
                )
                if plan is None:
                    continue
                source, _, facts = plan
                chunks = split_into_stand_in_chunks(source.title, body, config.source_lang)
                logger.info("Reviewing %r (%d section(s)) with %s", title, len(chunks), review_model)
                assembled, findings, rounds, _ = await run_review_pass(
                    chunks, source, config, body, review_client, review_model, skill, review_pricing,
                    target_mw_client, stats, facts, None, None, qa_skill, citation_client,
                    source_text=source.wikitext,
                )
            except Exception:
                logger.exception("Review of %r failed", title)
                continue

            result = RereviewResult(title, review_model, review_provider, findings, rounds, assembled != body)
            if result.changed:
                backup = config.output_dir / PRE_REVIEW_DIRNAME / path.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():  # keep the very first draft if reviewed twice
                    backup.write_text(original, encoding="utf-8")
                path.write_text(f"{assembled}{attribution}", encoding="utf-8")
            _update_report(path.with_suffix(".report.md"), review_model, findings)
            stats.articles_completed += 1
            results.append(result)
            logger.info(
                "Reviewed %r: %d unresolved finding(s), %d repair round(s), %s",
                title, len(findings), rounds, "text changed" if result.changed else "text unchanged",
            )
            if on_done is not None:
                on_done(result)
    finally:
        await mw_pool.aclose()
        for client in client_pool.clients.values():
            await client.aclose()
        if wikidata_client is not None:
            await wikidata_client.aclose()
        if citation_client is not None:
            await citation_client.aclose()
        if verification_cache is not None:
            verification_cache.close()
    return results, stats_tracker
