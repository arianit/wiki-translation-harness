"""Concurrent multi-article orchestration: fetch -> chunk -> translate -> assemble -> save.

Ties together mediawiki.py, parser.py, cache.py, translator.py, and
output.py. Contains no translation logic of its own.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re

import httpx

from wiki_translation_harness.cache import TranslationCache, VerificationCache
from wiki_translation_harness.resume import checkpoint_input, input_key, record_attempt, unfinished_inputs
from wiki_translation_harness.config import (
    default_model_for_provider,
    resolve_complex_provider,
    resolve_review_model,
    resolve_review_provider,
)
from wiki_translation_harness.engines import LLMEngineClient, build_client_pool, build_llm_client
from wiki_translation_harness.glossary import build_article_glossary
from wiki_translation_harness.comments import reconcile_chunk_comments
from wiki_translation_harness.citation_language import (
    dedupe_short_footnotes,
    fill_missing_citation_languages,
    fix_citation_param_names,
    convert_nested_ref_notes_to_refn,
    convert_nested_sfn_to_harvnb,
    fix_sfn_param_names,
    unwrap_redundant_sfn_ref,
)
from wiki_translation_harness.report import build_attribution_block
from wiki_translation_harness.live_validator import validate_wikitext_live
from wiki_translation_harness.mediawiki import MediaWikiClient, MediaWikiClientPool
from wiki_translation_harness.layout_fixes import (
    blank_line_before_headings,
    fix_archive_dates,
    replace_missing_excerpts,
    unnest_ill_in_wikilinks,
)
from wiki_translation_harness.hatnote_ill import add_ill_to_hatnotes
from wiki_translation_harness.ref_template import expand_r_definitions
from wiki_translation_harness.reference_fixes import (
    collapse_conflicting_ref_definitions,
    collapse_duplicate_list_defined_refs,
    drop_stray_reference_lists,
    drop_unsupported_ref_reuses,
    extend_references_block_over_trailing_defs,
    merge_split_references_blocks,
    restore_orphaned_ref_definitions,
    restore_ref_group_names,
    restore_ref_names,
)
from wiki_translation_harness.notes import expand_cref_notes, normalize_notelist_aliases
from wiki_translation_harness.models import (
    ArticleSource,
    Chunk,
    ChunkStatus,
    Config,
    EngineError,
    InsufficientCreditsError,
    ModelPricing,
    RunStats,
    TranslationResult,
    ValidationIssue,
)
from wiki_translation_harness.openrouter import RetryCallback
from wiki_translation_harness.locks import release as release_lock, try_lock_article
from wiki_translation_harness.output import (
    article_already_done,
    assemble_chunks,
    assemble_chunks_with_spans,
    discard_partial_article,
    output_path_for,
    save_article,
    save_partial_article,
)
from wiki_translation_harness.parser import build_chunks, split_into_sections
from wiki_translation_harness.progress import ProgressReporter
from wiki_translation_harness.qa import run_qa_checks
from wiki_translation_harness.report import ArticleReportData, build_article_report, save_report
from wiki_translation_harness.repair import repair_chunk, repair_chunk_semantic
from wiki_translation_harness.review import review_article
from wiki_translation_harness.review_queue import record_needs_human_review
from wiki_translation_harness.skill_loader import SkillContent, load_skill
from wiki_translation_harness.source_templates import expand_italian_data_templates
from wiki_translation_harness.sources import ArticleInput, load_article_source
from wiki_translation_harness.statistics import StatsTracker
from wiki_translation_harness.translator import chunk_cache_key, translate_chunk, translation_model
from wiki_translation_harness.validator import format_errors, validate_wikitext
from wiki_translation_harness.verification import (
    VerifiedFacts,
    build_verified_facts_block,
    existing_article_matches_by_length,
    verify_wikitext,
)

logger = logging.getLogger("wiki_translation_harness.pipeline")


class ArticleLimitExceeded(Exception):
    pass

# Bounds concurrent MediaWiki fetches during the planning phase. Not the same
# knob as `workers` (which bounds concurrent OpenRouter translation calls) —
# fetching is cheap and this just keeps us polite to the API.
_FETCH_CONCURRENCY = 8


def _chunk_for_line(spans: list[tuple[Chunk, int, int]], line_number: int | None) -> Chunk | None:
    if line_number is None:
        return None
    for chunk, start, end in spans:
        if start <= line_number <= end:
            return chunk
    return None


# MediaWiki's Cite extension doesn't echo the offending <ref name="X"> tag
# syntax into its (localized) error text — confirmed against real sqwiki
# output (see tests/test_live_validator.py): an orphaned ref reads "...asnjë
# tekst nuk u dha për refs e quajtura "x"" and English's own
# cite_error_references_duplicate_key reads 'name "$1" defined multiple
# times...' — in both, the ref name is simply quoted, with no "name="
# prefix and no consistent surrounding wording across languages/error
# kinds. The one thing that does hold: it's the only quoted text in the
# message. So match any quoted span rather than a literal "name=".
_REF_NAME_IN_MESSAGE_RE = re.compile(r"[\"'‘’“”]([^\"'‘’“”]+)[\"'‘’“”]")

# Only these live-validator kinds are Cite-extension ref errors whose
# message names the offending ref; extracting quoted spans from anything
# else (e.g. unexpanded_template's `Template 'X' does not exist`, which is
# also quoted) would misroute unrelated findings onto a coincidentally
# matching chunk.
_REF_NAMED_ISSUE_KINDS = frozenset({"orphaned_named_ref", "cite_error"})


def _ref_names_in_message(issue: ValidationIssue) -> list[str]:
    """Ref names named by a cite error message (an orphaned named ref, a
    ref "defined multiple times", ...) — empty list when the issue isn't a
    Cite ref error or its message doesn't name one."""
    if issue.kind not in _REF_NAMED_ISSUE_KINDS:
        return []
    return [html.unescape(m.group(1)) for m in _REF_NAME_IN_MESSAGE_RE.finditer(html.unescape(issue.message))]


def _chunk_mentions_ref(chunk: Chunk, ref_name: str) -> bool:
    """True if the chunk's wikitext contains a <ref name="{ref_name}"> or
    <ref name="{ref_name}"/> occurrence. Checks the translated text first
    (the text being repaired), falling back to the source text in case
    translation dropped/renamed the markup around the ref."""
    for text in (chunk.translated_text, chunk.text):
        if not text:
            continue
        text = html.unescape(text)
        if re.search(
            r"<ref\b[^>\n]*\bname\s*=\s*[\"']" + re.escape(ref_name) + r"[\"']",
            text,
            re.IGNORECASE,
        ):
            return True
    return False


async def _validate_assembled(
    text: str,
    target_mw_client: MediaWikiClient,
    article_title: str,
    live_validate: bool,
    live_validate_timeout_s: float,
) -> list[ValidationIssue]:
    """Static checks (validator.py) always run; the live parse-API checks
    (live_validator.py) are an optional enhancement, same treatment as
    verify_wikitext's Wikidata calls elsewhere in this module — a failure
    or timeout is logged and the article proceeds on static checks alone
    rather than blocking the whole run."""
    issues = list(validate_wikitext(text).issues)
    # Mandatory wikiqa checks (qa.py): never gated on config, and run on
    # every assembly/review round so repair sees them like any other finding.
    issues.extend(run_qa_checks(text))
    if live_validate:
        try:
            live_result = await asyncio.wait_for(
                validate_wikitext_live(target_mw_client, text, title=article_title),
                timeout=live_validate_timeout_s,
            )
            issues.extend(live_result.issues)
        except Exception as exc:
            logger.warning("Live validation failed for %r: %s", article_title, exc)
    return issues


_QUOTE_TEMPLATE_RE = re.compile(r"\{\{\s*([\'\"`]+(?:\s+[\'\"`]+)*)\s*\}\}")


def normalize_quote_templates(text: str) -> str:
    """Normalize English Wikipedia quote/spacing shortcut templates
    (e.g. {{'"}}, {{"'}}, {{' "}}, {{`}}) to their literal punctuation characters.
    These typography templates do not exist on other wikis (like sqwiki) and
    trigger unexpanded template warnings if left unexpanded.
    """
    return _QUOTE_TEMPLATE_RE.sub(r"\g<1>", text)


async def post_process_assembled(
    text: str,
    source_title: str,
    source_text: str,
    config: Config,
    citation_client: httpx.AsyncClient | None,
    target_mw_client: MediaWikiClient | None = None,
    source_lang: str = "en",
) -> tuple[str, dict[str, str] | None]:
    """Deterministic string-level fixes for an assembled article. Run on
    every fresh assembly — after each assembly-repair round and each
    review-repair round alike — since assembling from chunks drops them.
    Each fix is a no-op on already-fixed input, so repeating them is safe,
    just not free.

    Returns (patched_text, citation_languages_filled), the latter None when
    the citation-language fill didn't run or failed."""
    citation_languages_filled: dict[str, str] | None = None
    if citation_client is not None:
        try:
            citation_result = await asyncio.wait_for(
                fill_missing_citation_languages(
                    text,
                    citation_client,
                    max_url_fetches=config.max_citation_url_fetches,
                    concurrency=config.citation_fetch_concurrency,
                    fetch_timeout=config.citation_fetch_timeout_s,
                ),
                timeout=config.citation_fill_timeout_s,
            )
            text = citation_result.patched_wikitext
            citation_languages_filled = citation_result.filled
            if citation_result.filled:
                logger.info(
                    "Filled |language= for %d/%d citation(s) in %r",
                    len(citation_result.filled),
                    citation_result.attempted,
                    source_title,
                )
        except Exception as exc:
            logger.warning("Citation language fill failed for %r: %s", source_title, exc)

    if config.fix_citation_param_names:
        try:
            param_fix_result = fix_citation_param_names(text)
            text = param_fix_result.patched_wikitext
            if param_fix_result.renamed:
                logger.info(
                    "Renamed %d mistranslated citation parameter name(s) in %r: %s",
                    len(param_fix_result.renamed),
                    source_title,
                    param_fix_result.renamed,
                )
        except Exception as exc:
            logger.warning("Citation parameter name fix failed for %r: %s", source_title, exc)

    try:
        sfn_fix_result = fix_sfn_param_names(text)
        text = sfn_fix_result.patched_wikitext
        if sfn_fix_result.renamed:
            logger.info(
                "Renamed %d sfn parameter(s) in %r: %s",
                len(sfn_fix_result.renamed),
                source_title,
                sfn_fix_result.renamed,
            )
    except Exception as exc:
        logger.warning("Sfn parameter name fix failed for %r: %s", source_title, exc)

    # Before the sfn fixers below: a note's {{sfn}} is fine inside
    # {{refn}}, but convert_nested_sfn_to_harvnb would still see it as
    # nested in a <ref> and turn it into a bare harvnb link.
    try:
        refn_result = convert_nested_ref_notes_to_refn(text)
        text = refn_result.patched_wikitext
        if refn_result.converted:
            logger.info(
                "Rewrote %d note(s) with a <ref> nested inside a <ref> as {{refn}} in %r: %s",
                len(refn_result.converted),
                source_title,
                refn_result.converted,
            )
    except Exception as exc:
        logger.warning("Nested-ref note conversion failed for %r: %s", source_title, exc)

    try:
        unwrap_result = unwrap_redundant_sfn_ref(text)
        text = unwrap_result.patched_wikitext
        if unwrap_result.unwrapped:
            logger.info(
                "Unwrapped %d redundant <ref>{{sfn}}</ref> wrapper(s) in %r: %s",
                len(unwrap_result.unwrapped),
                source_title,
                unwrap_result.unwrapped,
            )
    except Exception as exc:
        logger.warning("Redundant sfn-ref unwrap failed for %r: %s", source_title, exc)

    try:
        nested_result = convert_nested_sfn_to_harvnb(text)
        text = nested_result.patched_wikitext
        if nested_result.converted:
            logger.info(
                "Converted %d {{sfn}}/{{sfnp}} call(s) nested inside <ref> to harvnb/harvp in %r",
                len(nested_result.converted),
                source_title,
            )
    except Exception as exc:
        logger.warning("Nested sfn-to-harvnb conversion failed for %r: %s", source_title, exc)

    if config.dedupe_short_footnotes:
        try:
            dedupe_result = dedupe_short_footnotes(text)
            text = dedupe_result.patched_wikitext
            if dedupe_result.canonicalized:
                logger.info(
                    "Reconciled %d short-footnote identity group(s) with diverging |ps= in %r",
                    len(dedupe_result.canonicalized),
                    source_title,
                )
        except Exception as exc:
            logger.warning("Short-footnote dedup failed for %r: %s", source_title, exc)

    text = normalize_quote_templates(text)

    # Group names first: drop_stray_reference_lists counts renderers
    # per source group name.
    text, renamed_groups = restore_ref_group_names(text, source_text)
    if renamed_groups:
        logger.info(
            "Restored translated ref group name(s) in %r: %s", source_title, renamed_groups
        )

    text, renamed_refs = restore_ref_names(text, source_text)
    if renamed_refs:
        logger.info(
            "Restored %d translated ref name(s) in %r: %s",
            len(renamed_refs), source_title, renamed_refs,
        )

    text, dropped_lists = drop_stray_reference_lists(text, source_text)
    if dropped_lists:
        logger.info(
            "Removed %d reference list(s) not in the source from %r", dropped_lists, source_title
        )

    if target_mw_client is not None:
        try:
            text, ill_titles = await add_ill_to_hatnotes(
                text, source_text, target_mw_client.existing_pages, source_lang
            )
            if ill_titles:
                logger.info(
                    "Gave %d hatnote target(s) missing on the target wiki an {{ill}} in %r: %s",
                    len(ill_titles), source_title, ill_titles,
                )
        except Exception as exc:
            logger.warning("Hatnote {{ill}} fix failed for %r: %s", source_title, exc)

    if target_mw_client is not None:
        try:
            text, missing_excerpts = await replace_missing_excerpts(
                text, target_mw_client.existing_pages, source_text, source_lang
            )
            if missing_excerpts:
                logger.warning(
                    "Replaced %d {{Excerpt}} with a missing target by an {{ill}} hatnote in %r: %s",
                    len(missing_excerpts), source_title, missing_excerpts,
                )
        except Exception as exc:
            logger.warning("Excerpt target check failed for %r: %s", source_title, exc)

    text, unnested = unnest_ill_in_wikilinks(text)
    if unnested:
        logger.info("Unnested %d {{ill}} from inside a wikilink in %r", unnested, source_title)
    try:
        text, dates_fixed = fix_archive_dates(text)
        if dates_fixed:
            logger.info("Fixed %d |archive-date= value(s) in %r", dates_fixed, source_title)
    except Exception as exc:
        logger.warning("archive-date fix failed for %r: %s", source_title, exc)
    text, headings_spaced = blank_line_before_headings(text)
    if headings_spaced:
        logger.info("Added a blank line before %d heading(s) in %r", headings_spaced, source_title)

    text, merged_refs_blocks = merge_split_references_blocks(text)
    if merged_refs_blocks:
        logger.info(
            "Merged %d <references> block(s) split across chunks in %r",
            merged_refs_blocks,
            source_title,
        )

    text, moved_defs = extend_references_block_over_trailing_defs(text)
    if moved_defs:
        logger.info(
            "Moved %d ref definition(s) left after </references> back into the block in %r",
            moved_defs,
            source_title,
        )

    text, collapsed_refs = collapse_duplicate_list_defined_refs(text, source_text)
    if collapsed_refs:
        logger.info(
            "Collapsed %d body definition(s) of list-defined ref(s) back to reuses in %r: %s",
            len(collapsed_refs), source_title, collapsed_refs,
        )

    text, restored_refs = restore_orphaned_ref_definitions(text, source_text)
    if restored_refs:
        logger.info(
            "Restored %d named-ref definition(s) dropped in translation of %r from the source: %s",
            len(restored_refs),
            source_title,
            restored_refs,
        )

    text, dropped_reuses = drop_unsupported_ref_reuses(text, source_text)
    if dropped_reuses:
        logger.info(
            "Removed reuses of %d ref name(s) that neither the source nor the translation defines in %r: %s",
            len(dropped_reuses), source_title, dropped_reuses,
        )

    text, collapsed_defs = collapse_conflicting_ref_definitions(text, source_text)
    if collapsed_defs:
        logger.info(
            "Collapsed repeated definitions of %d ref name(s) the model translated differently in %r: %s",
            len(collapsed_defs), source_title, collapsed_defs,
        )

    return text, citation_languages_filled


def _cache_repaired_chunk(
    cache: TranslationCache, model: str, target_lang: str, chunk: Chunk, skill: SkillContent, facts: VerifiedFacts | None
) -> None:
    """Re-cache a repaired chunk under the key translate_chunk would use for it with `model`."""
    facts_block = build_verified_facts_block(chunk.text, facts) if facts else ""
    key = chunk_cache_key(model, target_lang, chunk, skill, facts_block)
    cache.set(key, model, chunk.source_lang, target_lang, chunk.text, chunk.translated_text)


async def run_assembly_repair(
    chunks: list[Chunk],
    source: ArticleSource,
    config: Config,
    llm_client: LLMEngineClient,
    skill: SkillContent,
    pricing: ModelPricing | None,
    target_mw_client: MediaWikiClient,
    citation_client: httpx.AsyncClient | None,
    stats: RunStats,
    on_retry: RetryCallback | None = None,
    cache: TranslationCache | None = None,
    facts: VerifiedFacts | None = None,
    qa_skill: SkillContent | None = None,
) -> tuple[str, list[ValidationIssue], int, dict[str, str], list[str]]:
    """Post-processes and validates the assembled article, repairing only
    the chunks implicated by each round's findings, up to
    config.max_assembly_repair_rounds. Mutates chunk.translated_text in
    place for any chunk that gets repaired. This is the whole-article
    counterpart to translator.translate_chunk's per-chunk validate/repair
    loop — kept as a separate loop with its own round budget
    (config.max_assembly_repair_rounds) since these checks (live parse-API
    errors, table span mismatches, etc.) only make sense once every chunk
    is in place.

    Returns (assembled_text, remaining_issues, rounds_used,
    citation_languages_filled, removed_comments) — remaining_issues is empty iff the article
    is valid, whether immediately or after repair; the caller decides what
    "still has issues after the cap" means (needs_human_review)."""
    citation_languages_filled: dict[str, str] = {}
    source_text = "\n".join(chunk.text for chunk in chunks)

    async def _post_process(text: str) -> str:
        nonlocal citation_languages_filled
        text, filled = await post_process_assembled(
            text, source.title, source_text, config, citation_client, target_mw_client, source.source_lang
        )
        if filled is not None:
            citation_languages_filled = filled
        return text

    # Per chunk, before assembly: matching a comment to the source needs
    # the chunk's own source text. Re-run after every repair round, since a
    # repair call is where the model most often leaves a note of its own.
    removed_comments = reconcile_chunk_comments(chunks)
    assembled = await _post_process(assemble_chunks(chunks))
    combined_issues = await _validate_assembled(
        assembled, target_mw_client, source.title, config.live_validate, config.live_validate_timeout_s
    )

    rounds_used = 0
    while combined_issues and rounds_used < config.max_assembly_repair_rounds:
        rounds_used += 1
        _, spans = assemble_chunks_with_spans(chunks)

        localized: dict[int, list[str]] = {}
        unlocalized: list[tuple[ValidationIssue, str]] = []
        for issue in combined_issues:
            finding = issue.as_finding()
            text_line = f"{finding['severity']}: {finding['explanation']}"
            if finding["snippet"]:
                text_line += f" (near: {finding['snippet']})"
            chunk = _chunk_for_line(spans, issue.line_number)
            if chunk is not None:
                localized.setdefault(id(chunk), []).append(text_line)
            else:
                # Can't be pinned to one chunk by line number (e.g. most
                # live-API findings — a rendered error message doesn't
                # literally appear in the source wikitext to locate).
                unlocalized.append((issue, text_line))

        # An unlocated finding still can't be dropped: an issue no chunk
        # ever sees can never be fixed within the round cap. But most
        # unlocated findings are Cite-extension errors (an orphaned named
        # ref, a ref "defined multiple times") whose message names the
        # offending <ref name="..."> — those only make sense in a chunk
        # that actually mentions that ref: a chunk that never sees the ref
        # can't fix it (a broadcast just burns one call per round for
        # nothing), and letting several siblings each "fix" an orphaned ref
        # by inserting their own definition creates a new define-twice
        # finding next round, so the loop oscillates instead of converging.
        # So: hand a ref-named finding to the FIRST chunk (in order) that
        # mentions the ref — one repair target, exactly the chunk most
        # likely to own the definition — not to every match. Findings with
        # no extractable ref name keep the old all-chunk broadcast.
        targeted_unlocalized: dict[int, list[str]] = {}
        broadcast_unlocalized: list[str] = []
        targeted_counts: dict[tuple[str, int], int] = {}
        for issue, text_line in unlocalized:
            # Harvc's rendered Lua/missing-module errors describe the same
            # localized template defects already sent to their owners.
            if issue.kind in {"lua_script_error", "unexpanded_template"} and "harvc" in issue.message.lower():
                if any(i.kind == "harvc_used" and i.line_number is not None for i in combined_issues):
                    continue
            # A missing module that has no matching source invocation is
            # a wiki installation problem. Rewriting every chunk cannot
            # install it; retain the finding for the human-review report.
            if issue.kind == "unexpanded_template" and issue.line_number is None and "moduli:" in issue.message.lower():
                continue
            ref_names = _ref_names_in_message(issue)
            target = next(
                (
                    c
                    for c in chunks
                    if any(_chunk_mentions_ref(c, name) for name in ref_names)
                ),
                None,
            )
            if target is not None:
                lines_for_target = targeted_unlocalized.setdefault(id(target), [])
                if text_line not in lines_for_target:
                    lines_for_target.append(text_line)
                count_key = (issue.kind, target.order)
                targeted_counts[count_key] = targeted_counts.get(count_key, 0) + 1
            elif text_line not in broadcast_unlocalized:
                broadcast_unlocalized.append(text_line)
        for (kind, order), count in sorted(targeted_counts.items(), key=lambda kv: kv[0][1]):
            logger.info(
                "Targeted %d unlocalized %r finding(s) to chunk %d (mentions the named ref) instead of "
                "broadcasting to all %d chunk(s)",
                count, kind, order, len(chunks),
            )

        # Each chunk's repair only reads and writes that chunk, so a
        # round's repairs run in parallel (bounded by
        # assembly_repair_concurrency) instead of one after another.
        semaphore = asyncio.Semaphore(max(1, config.assembly_repair_concurrency))

        async def _repair(chunk: Chunk, errors_for_chunk: list[str]) -> None:
            async with semaphore:
                try:
                    stats.repair_attempts += 1
                    repair_result = await repair_chunk(
                        llm_client,
                        skill,
                        config.model,
                        config.temperature,
                        chunk.source_lang,
                        config.target_lang,
                        chunk.article_title,
                        chunk.section_title,
                        chunk.translated_text or "",
                        errors_for_chunk,
                        pricing,
                        on_retry=on_retry,
                        qa_skill=qa_skill,
                    )
                    stats.record_usage(config.model, repair_result)
                    chunk.translated_text = repair_result.text
                    if cache is not None:
                        # Without this, a fix made here is invisible to future
                        # reruns' cache lookups: translate_chunk's own cache.set
                        # already ran (pre-repair) when this chunk was first
                        # translated, so re-caching now with the same key
                        # overwrites that stale entry with the repaired text.
                        _cache_repaired_chunk(
                            cache, translation_model(config, chunk), config.target_lang, chunk, skill, facts
                        )
                except Exception as exc:
                    logger.warning(
                        "Assembly-level repair failed for %r chunk %d (round %d): %s",
                        source.title, chunk.order, rounds_used, exc,
                    )

        repair_targets = [
            (chunk, errors_for_chunk)
            for chunk in chunks
            if (errors_for_chunk := (
                localized.get(id(chunk), [])
                + targeted_unlocalized.get(id(chunk), [])
                + broadcast_unlocalized
            ))
        ]
        if not repair_targets:
            logger.warning("No actionable chunk repair targets for %r; retaining findings for human review", source.title)
            break
        await asyncio.gather(*(
            _repair(chunk, errors_for_chunk)
            for chunk, errors_for_chunk in repair_targets
        ))

        removed_comments.extend(reconcile_chunk_comments(chunks))
        assembled = await _post_process(assemble_chunks(chunks))
        combined_issues = await _validate_assembled(
            assembled, target_mw_client, source.title, config.live_validate, config.live_validate_timeout_s
        )

    return assembled, combined_issues, rounds_used, citation_languages_filled, removed_comments


def _localize_review_issues(
    issues: list[ValidationIssue], assembled: str, spans: list[tuple[Chunk, int, int]]
) -> tuple[dict[int, list[str]], list[str]]:
    """Same chunk-localization strategy as run_assembly_repair's own loop
    (via _chunk_for_line), except a review finding has no wikitext
    line_number to begin with (the review model doesn't see line numbers) —
    only a `snippet` it was asked to copy verbatim from the translated
    text. Locates it by searching the assembled text for that snippet and
    converting the match offset to a line number; a snippet that can't be
    found verbatim (the model paraphrased instead of copying) falls back to
    broadcasting to every chunk, same as an unlocalizable structural
    finding does."""
    localized: dict[int, list[str]] = {}
    broadcast: list[str] = []
    for issue in issues:
        finding = issue.as_finding()
        text_line = f"{finding['severity']}: {finding['explanation']}"
        if finding["snippet"]:
            text_line += f" (near: {finding['snippet']})"

        line_number = issue.line_number
        if line_number is None and issue.snippet:
            offset = assembled.find(issue.snippet)
            if offset != -1:
                line_number = assembled.count("\n", 0, offset) + 1

        chunk = _chunk_for_line(spans, line_number)
        if chunk is not None:
            localized.setdefault(id(chunk), []).append(text_line)
        else:
            broadcast.append(text_line)
    return localized, broadcast


_QUOTED_FORM_RE = re.compile(r"[“\"«]([^”\"»]{2,60})[”\"»]")


def _consistency_directives(
    issues: list[ValidationIssue], chunks: list[Chunk]
) -> dict[int, list[str]]:
    """Per-chunk instructions that settle an entity_consistency finding for
    the whole article.

    The reviewer names the competing spellings in quotes (Josephus is
    "Flavio Josefi" here and "Flavius Jozefi" later) but a finding reaches
    only the chunk its snippet sits in, whose model cannot see the other
    form: it may choose the opposite one, and the next review flags the
    reverse (Religion, Fundamentalism, Pakistan). Here the form used most
    across the chunks becomes canonical (the first one named on a tie) and
    every chunk using another form is told to switch to it."""
    directives: dict[int, list[str]] = {}
    for issue in issues:
        if issue.kind != "entity_consistency":
            continue
        forms: list[str] = []
        for form in _QUOTED_FORM_RE.findall(issue.message):
            form = form.strip().rstrip(".,;")
            if form and form not in forms:
                forms.append(form)
        counts = {f: sum((c.translated_text or "").count(f) for c in chunks) for f in forms}
        present = [f for f in forms if counts[f]]
        if len(present) < 2:
            continue
        canonical = max(present, key=lambda f: (counts[f], -forms.index(f)))
        for chunk in chunks:
            text = chunk.translated_text or ""
            others = [f for f in present if f != canonical and f in text]
            if others:
                directives.setdefault(id(chunk), []).append(
                    f"error: spell this consistently across the whole article: use “{canonical}” "
                    f"(the form used most) wherever this section has "
                    + ", ".join(f"“{f}”" for f in others)
                    + ", adjusting only the grammatical ending where the sentence needs it"
                )
    return directives


async def _repair_chunk_for_review(
    chunk: Chunk,
    findings: list[str],
    structural_errors: list[str],
    config: Config,
    review_client: LLMEngineClient,
    review_model: str,
    skill: SkillContent,
    review_pricing: ModelPricing | None,
    on_retry: RetryCallback | None,
    qa_skill: SkillContent | None,
) -> TranslationResult:
    """Semantic repair (source shown, rewording allowed) when the chunk has
    review findings; the syntax-only repair when it has only structural errors."""
    if findings:
        return await repair_chunk_semantic(
            review_client,
            skill,
            review_model,
            config.temperature,
            chunk.source_lang,
            config.target_lang,
            chunk.article_title,
            chunk.section_title,
            chunk.text,
            chunk.translated_text or "",
            findings,
            review_pricing,
            structural_errors=structural_errors,
            on_retry=on_retry,
            qa_skill=qa_skill,
        )
    return await repair_chunk(
        review_client,
        skill,
        review_model,
        config.temperature,
        chunk.source_lang,
        config.target_lang,
        chunk.article_title,
        chunk.section_title,
        chunk.translated_text or "",
        structural_errors,
        review_pricing,
        on_retry=on_retry,
        qa_skill=qa_skill,
    )


async def run_review_pass(
    chunks: list[Chunk],
    source: ArticleSource,
    config: Config,
    initial_assembled: str,
    review_client: LLMEngineClient,
    review_model: str,
    skill: SkillContent,
    review_pricing: ModelPricing | None,
    target_mw_client: MediaWikiClient,
    stats: RunStats,
    facts: VerifiedFacts,
    on_retry: RetryCallback | None = None,
    cache: TranslationCache | None = None,
    qa_skill: SkillContent | None = None,
    citation_client: httpx.AsyncClient | None = None,
    source_text: str | None = None,
) -> tuple[str, list[ValidationIssue], int, list[str]]:
    """Independent semantic-fidelity review of an already assembled,
    structurally valid article (see review.py) — the whole-article
    counterpart to translate_chunk's per-chunk generate/validate loop, run
    once assembly-level structural repair (run_assembly_repair above) has
    already passed clean, so a strong-model review call is never spent on
    wikitext that might still get rewritten by structural repair.

    Each round re-runs the review call (cheap: it only ever returns a
    findings list, not a full retranslation) plus a structural safety-net
    check (a review-driven edit could reintroduce a structural break — must
    not ship that silently), localizes findings to owning chunks, and
    repairs only those chunks, same as run_assembly_repair does for
    deterministic findings -- via repair_chunk_semantic() when the chunk has
    review findings (the model sees the source section and may reword) and
    via the syntax-only repair_chunk() when it has only structural errors.
    Capped at
    config.review_max_repair_attempts.

    After a repair round the article is reassembled from the chunks, so
    the comment reconciliation and post_process_assembled fixes that
    run_assembly_repair applied are applied again — otherwise the saved
    article would lose them.

    Returns (assembled_text, remaining_issues, rounds_used,
    removed_comments) — remaining_issues is empty iff the review pass converged (no findings, or every finding
    got resolved within the round cap); a non-empty result does NOT mean
    the caller should withhold the article (see report.build_review_section)."""

    async def _run_review_call(assembled_text: str) -> list[ValidationIssue]:
        findings, result = await review_article(
            review_client,
            review_model,
            config.temperature,
            source.wikitext,
            assembled_text,
            source.title,
            config.target_lang,
            facts.template_params,
            review_pricing,
            on_retry=on_retry,
        )
        stats.record_usage(review_model, result)
        stats.review_findings_total += len(findings)
        return findings

    if source_text is None:
        source_text = "\n".join(chunk.text for chunk in chunks)
    removed_comments: list[str] = []
    assembled = initial_assembled
    review_findings = await _run_review_call(assembled)
    structural_issues = await _validate_assembled(
        assembled, target_mw_client, source.title, config.live_validate, config.live_validate_timeout_s
    )
    combined = structural_issues + review_findings

    def _has_structural_error(issues: list[ValidationIssue]) -> bool:
        return any(i.severity == "error" for i in issues)

    # The last state without a structural error: a review-driven rewrite can
    # break the markup (Religion came out with one more "}}" than "{{"), and
    # shipping that with a flag is worse than shipping the unreviewed text.
    clean: tuple[list[str | None], str, list[ValidationIssue]] | None = None
    if not _has_structural_error(structural_issues):
        clean = ([c.translated_text for c in chunks], assembled, combined)

    rounds_used = 0
    while combined and rounds_used < config.review_max_repair_attempts:
        rounds_used += 1
        _, spans = assemble_chunks_with_spans(chunks)
        # Structural errors and review findings are localized separately: the
        # review findings are about wording, so they go to the semantic repair
        # prompt (source shown, rewording allowed); a chunk with only
        # structural errors keeps the syntax-only prompt that forbids rewording.
        structural_localized, structural_broadcast = _localize_review_issues(structural_issues, assembled, spans)
        review_localized, review_broadcast = _localize_review_issues(review_findings, assembled, spans)

        consistency = _consistency_directives(review_findings, chunks)
        for chunk in chunks:
            structural_for_chunk = structural_localized.get(id(chunk), []) + structural_broadcast
            findings_for_chunk = (
                review_localized.get(id(chunk), []) + review_broadcast + consistency.get(id(chunk), [])
            )
            if not structural_for_chunk and not findings_for_chunk:
                continue
            stats.review_attempts += 1
            try:
                repair_result = await _repair_chunk_for_review(
                    chunk, findings_for_chunk, structural_for_chunk, config, review_client, review_model,
                    skill, review_pricing, on_retry, qa_skill,
                )
                stats.record_usage(review_model, repair_result)
                stats.review_corrections_applied += 1
                chunk.translated_text = repair_result.text
                if cache is not None:
                    _cache_repaired_chunk(cache, review_model, config.target_lang, chunk, skill, facts)
            except InsufficientCreditsError:
                raise
            except Exception as exc:
                logger.warning(
                    "Review-driven repair failed for %r chunk %d (round %d): %s",
                    source.title, chunk.order, rounds_used, exc,
                )

        removed_comments.extend(reconcile_chunk_comments(chunks))
        assembled, _ = await post_process_assembled(
            assemble_chunks(chunks), source.title, source_text, config, citation_client,
            target_mw_client, source.source_lang,
        )
        structural_issues = await _validate_assembled(
            assembled, target_mw_client, source.title, config.live_validate, config.live_validate_timeout_s
        )
        review_findings = await _run_review_call(assembled)
        combined = structural_issues + review_findings
        if not _has_structural_error(structural_issues):
            clean = ([c.translated_text for c in chunks], assembled, combined)

    if _has_structural_error(structural_issues) and clean is not None:
        logger.warning(
            "Review-driven repair left a structural error in %r (%s); restoring the last text without one",
            source.title, "; ".join(i.message for i in structural_issues if i.severity == "error"),
        )
        for chunk, text in zip(chunks, clean[0]):
            chunk.translated_text = text
        assembled = clean[1]
        combined = list(clean[2])

    return assembled, combined, rounds_used, removed_comments


async def _plan_article(
    mw_pool: MediaWikiClientPool,
    item: ArticleInput,
    config: Config,
    fetch_sem: asyncio.Semaphore,
    wikidata_client: httpx.AsyncClient | None,
    verification_cache: VerificationCache | None,
) -> tuple[ArticleSource, list[Chunk], VerifiedFacts] | None:
    effective_lang = item.source_lang or config.source_lang
    async with fetch_sem:
        try:
            mw_client = mw_pool.get(effective_lang)
            source = await load_article_source(mw_client, item, effective_lang)
        except Exception as exc:
            logger.error("Failed to fetch %r (lang=%s): %s", item.title, effective_lang, exc)
            return None

        normalized_wikitext, aliased = normalize_notelist_aliases(source.wikitext)
        if aliased:
            source = source.model_copy(update={"wikitext": normalized_wikitext})
            logger.info("Rewrote %d note-list alias template(s) to {{notelist}} in %r", aliased, source.title)

        expanded_wikitext, r_converted, r_skipped = expand_r_definitions(source.wikitext)
        if r_converted or r_skipped:
            source = source.model_copy(update={"wikitext": expanded_wikitext})
            logger.info(
                "Rewrote %d {{R}} reference definition(s) as <ref> tags in %r%s",
                r_converted, source.title, f"; left {r_skipped} it could not model" if r_skipped else "",
            )

        # Before verification/chunking: a {{Cref2}} marker and its {{Cnote2}}
        # text live in different chunks, so they must be joined on the whole
        # source (see notes.py). Replaces `source` so the reviewer compares
        # the translation against the same text that was translated.
        try:
            note_result = expand_cref_notes(source.wikitext)
            if note_result.expanded:
                source = source.model_copy(update={"wikitext": note_result.patched_wikitext})
                logger.info(
                    "Expanded %d {{Cref2}} note marker(s) into <ref group> footnotes in %r%s%s",
                    note_result.expanded,
                    source.title,
                    f"; left {note_result.unresolved} unresolved" if note_result.unresolved else "",
                    f"; dropped unreferenced note(s) {note_result.unreferenced}" if note_result.unreferenced else "",
                )
        except Exception as exc:
            logger.warning("Cref2 note expansion failed for %r: %s", item.title, exc)

        if effective_lang == "it":
            try:
                data_result = await expand_italian_data_templates(source.wikitext, source.title, mw_client)
                if data_result.wikitext != source.wikitext:
                    source = source.model_copy(update={"wikitext": data_result.wikitext})
                    logger.info(
                        "Expanded itwiki data templates in %r: %d population, %d demography table(s), "
                        "%d infobox(es)%s",
                        source.title, data_result.population, data_result.demography, data_result.infobox,
                        f"; left {data_result.skipped} it could not expand" if data_result.skipped else "",
                    )
            except Exception as exc:
                logger.warning("itwiki data-template expansion failed for %r: %s", item.title, exc)

        facts = VerifiedFacts()
        if wikidata_client is not None:
            try:
                target_mw_client = mw_pool.get(config.target_lang)
                # Hard outer deadline, independent of httpx's own per-call
                # timeouts: verification is an optional enhancement (the
                # model still translates fine without it), so it must never
                # be able to stall an entire batch run indefinitely if a
                # network call hangs past its configured timeout for any
                # reason — confirmed to happen in practice.
                facts = await asyncio.wait_for(
                    verify_wikitext(
                        source.title,
                        source.wikitext,
                        effective_lang,
                        config.target_lang,
                        wikidata_client,
                        target_mw_client,
                        verification_cache,
                    ),
                    timeout=config.verification_timeout_s,
                )
            except Exception as exc:
                logger.warning("Link/template verification failed for %r: %s", item.title, exc)

    sections = split_into_sections(source.wikitext)
    chunks = build_chunks(
        source.title,
        sections,
        config.chunk_min_tokens,
        config.chunk_max_tokens,
        source_lang=effective_lang,
    )
    return source, chunks, facts


class _Engines:
    """The engines a run sends work to. Every worker shares one instance,
    because a credit-exhaustion failure can switch the primary engine to
    config.fallback_provider, or move complex/review work off an exhausted
    secondary account, part way through a run."""

    def __init__(self, config: Config, client_pool, reporter: ProgressReporter | None):
        self.config = config
        self.reporter = reporter
        self.llm_client = client_pool.get(config.provider)
        self.review_model = resolve_review_model(config)
        self.review_provider = resolve_review_provider(config) if self.review_model else None
        self.review_client = client_pool.get(self.review_provider) if self.review_provider else None
        self.complex_client = client_pool.get(resolve_complex_provider(config)) if config.complex_model else None
        self.pricing: ModelPricing | None = None
        self.complex_pricing: ModelPricing | None = None
        self.review_pricing: ModelPricing | None = None
        # Guards the one-time provider switch below: several workers can hit
        # InsufficientCreditsError for the same exhausted account at once,
        # but only the first should prompt/log and actually swap the client
        # — the rest just wait for that swap and retry against it too.
        self._fallback_lock = asyncio.Lock()
        self._fallback_switched = False
        self._fallback_declined = False

    async def load_pricing(self) -> None:
        config = self.config
        self.pricing = await self.llm_client.get_pricing_for(config.model)
        self.complex_pricing = (
            await self.complex_client.get_pricing_for(config.complex_model)
            if self.complex_client is not None else None
        )
        self.review_pricing = (
            await self.review_client.get_pricing_for(self.review_model)
            if self.review_client is not None else None
        )

    def review_on_primary(self) -> None:
        self.review_client, self.review_model = self.llm_client, self.config.model
        self.review_provider, self.review_pricing = self.config.provider, self.pricing

    async def use_primary_for_secondary(self, provider: str | None) -> bool:
        """Stop sending complex/review work to an exhausted secondary account."""
        async with self._fallback_lock:
            config = self.config
            if provider is None or provider == config.provider:
                return False
            updates = {}
            if resolve_complex_provider(config) == provider:
                updates.update(complex_provider=config.provider, complex_model=config.model)
                self.complex_client, self.complex_pricing = self.llm_client, self.pricing
            if self.review_provider == provider:
                self.review_on_primary()
                updates.update(review_provider=self.review_provider, review_model=self.review_model)
            if updates:
                logger.warning(
                    "Secondary provider %r hit its limit; continuing complex/review work on %r (%s).",
                    provider, config.provider, config.model,
                )
                self.config = config.model_copy(update=updates)
            # True even with nothing to move: the failed call went to an engine
            # that is no longer in use (another worker already moved the tier,
            # or the primary was switched), so a retry goes somewhere new.
            return True

    async def ensure_fallback_engine(self) -> bool:
        """On the first call, offers (or, non-interactively, just makes)
        a one-way switch from config.provider to config.fallback_provider
        after a credit-exhaustion failure. Returns True if chunks should
        now retry against the (possibly already, by another worker)
        switched engine, False if there's nothing to switch to or the
        user declined."""
        async with self._fallback_lock:
            if self._fallback_switched:
                return True
            if self._fallback_declined:
                return False
            config, reporter = self.config, self.reporter
            target = config.fallback_provider or (
                # claude_code's own default fallback is opencode_go, not
                # itself -- a separate binary/session, so it isn't
                # affected by claude_code hitting its own session/spend
                # limit (see claude_code_client.ClaudeCodeSessionLimitError).
                "opencode_go" if config.provider == "claude_code" else "claude_code"
            )
            target_model = config.fallback_model or default_model_for_provider(target)
            if not target or (target == config.provider and (
                not config.fallback_model or target_model == config.model
            )):
                return False

            if target == "openrouter" and not config.openrouter_api_key:
                logger.error("Cannot switch to OpenRouter: set OPENROUTER_API_KEY or openrouter_api_key.")
                self._fallback_declined = True
                return False

            interactive = reporter is not None and reporter.is_live
            proceed = True
            if interactive and not config.fallback_auto_switch:
                reporter.pause()
                try:
                    # markup=False: the prompt's own `[provider]`/`[Y/n]`
                    # brackets would otherwise be parsed as (invalid,
                    # silently-dropped) Rich style tags -- confirmed in
                    # practice that Console.input() swallows "[openrouter]"
                    # entirely rather than printing it literally.
                    answer = await asyncio.to_thread(
                        reporter.console.input,
                        f"\n{config.provider} ran out of credits. Switch to "
                        f"'{target}' ({target_model}) and continue this run? [Y/n] ",
                        markup=False,
                    )
                finally:
                    reporter.resume()
                proceed = answer.strip().lower() not in ("n", "no")
            else:
                # Unattended (e.g. `queue` mode has no Live table attached
                # — see ProgressReporter.is_live): can't block on stdin,
                # so switch automatically and just log it.
                logger.warning(
                    "Provider %r ran out of credits; auto-switching to fallback "
                    "provider %r for the rest of this run.",
                    config.provider, target,
                )

            if not proceed:
                self._fallback_declined = True
                return False

            new_config = config.model_copy(
                update={"provider": target, "model": target_model}
            )
            new_client, effective_model = build_llm_client(new_config)
            if effective_model != new_config.model:
                new_config = new_config.model_copy(update={"model": effective_model})
            new_pricing = await new_client.get_pricing_for(new_config.model)

            # Deliberately not closing the old client here: another
            # worker could still have an in-flight call against it right
            # up to this point (it only reaches this lock *after*
            # raising InsufficientCreditsError, not before starting the
            # call), and aclose()'ing out from under a live request is
            # worse than leaking one now-unreferenced client for the
            # rest of the process.
            old_client = self.llm_client
            self.llm_client, self.pricing, self.config = new_client, new_pricing, new_config
            self._fallback_switched = True
            # A complex/review tier that shared the exhausted primary's client
            # would otherwise keep calling it, so its retry fails the same way.
            if self.complex_client is old_client:
                self.complex_client, self.complex_pricing = new_client, new_pricing
                self.config = self.config.model_copy(
                    update={"complex_provider": target, "complex_model": new_config.model}
                )
            if self.review_client is old_client:
                self.review_on_primary()

            logger.warning(
                "Switched engine to provider=%r model=%r after credit exhaustion.",
                target, new_config.model,
            )
            if reporter is not None and reporter.on_event is not None:
                reporter.on_event(f"switched provider to {target!r} after credit exhaustion")
            return True


def _claim_pending_items(
    inputs: list[ArticleInput], config: Config, force: bool, stats: RunStats, article_locks: list[int]
) -> list[ArticleInput]:
    """Inputs still to translate. Takes this machine's per-article lock for
    each one, appending it to article_locks for the caller to release."""
    pending_items: list[ArticleInput] = []
    for item in inputs:
        if not force and article_already_done(config.output_dir, item.title):
            stats.articles_skipped += 1
            logger.info("Skipping %r: output already exists", item.title)
            continue
        # Another harness process on this machine (different provider/model)
        # may already be translating this article; two would overwrite each
        # other's partial_output and burn tokens on the same work.
        lock = try_lock_article(config.target_lang, output_path_for(config.partial_output_dir, item.title).stem)
        if lock is None:
            stats.articles_skipped += 1
            logger.info("Skipping %r: another local process is already translating it", item.title)
            continue
        article_locks.append(lock)
        if not force and article_already_done(config.output_dir, item.title):
            # finished by the other process between our first check and getting the lock
            stats.articles_skipped += 1
            logger.info("Skipping %r: output already exists", item.title)
            continue
        pending_items.append(item)
    return pending_items


def _articles_to_translate(
    pending_items: list[ArticleInput],
    planned: list[tuple[ArticleSource, list[Chunk], VerifiedFacts] | None],
    stats: RunStats,
) -> list[tuple[ArticleInput, ArticleSource, list[Chunk], VerifiedFacts]]:
    """Drops articles that failed to plan or already exist on the target wiki at a similar size."""
    article_plans: list[tuple[ArticleInput, ArticleSource, list[Chunk], VerifiedFacts]] = []
    for item, plan in zip(pending_items, planned):
        if plan is None:
            stats.articles_failed += 1
            stats.article_results[item.title] = "failed"
            continue
        source, chunks, facts = plan
        if existing_article_matches_by_length(source.wikitext, facts):
            stats.articles_skipped += 1
            stats.article_results[item.title] = "skipped"
            logger.info(
                "Skipping %r: target article %r already exists and is similar in size (%d vs %d bytes)",
                item.title,
                facts.existing_target_title,
                len(source.wikitext.encode("utf-8")),
                facts.existing_target_bytes,
            )
            continue
        article_plans.append((item, source, chunks, facts))
    return article_plans


def _estimate_cost(chunks: list[Chunk], pricing: ModelPricing | None) -> float:
    if pricing is None:
        return 0.0
    # Heuristic: assume completion length is roughly comparable to
    # prompt length for a translation task (source + target text
    # are similar order of magnitude). Actual cost uses real usage.
    return sum(
        chunk.token_estimate * (pricing.prompt_price_per_token + pricing.completion_price_per_token)
        for chunk in chunks
    )


def _fail_pending_chunks(chunks: list[Chunk]) -> None:
    for c in chunks:
        if c.status == ChunkStatus.PENDING:
            c.status = ChunkStatus.FAILED


class _ArticleRunner:
    """Translates, repairs, reviews and saves one article at a time, sharing
    the run's engines, caches, clients and worker slots across articles."""

    def __init__(
        self,
        engines: _Engines,
        skill: SkillContent,
        qa_skill: SkillContent,
        cache: TranslationCache | None,
        mw_pool: MediaWikiClientPool,
        citation_client: httpx.AsyncClient | None,
        reporter: ProgressReporter | None,
        stats_tracker: StatsTracker,
    ):
        self.engines = engines
        self.skill = skill
        self.qa_skill = qa_skill
        self.cache = cache
        self.mw_pool = mw_pool
        self.citation_client = citation_client
        self.reporter = reporter
        self.stats_tracker = stats_tracker
        self.stats = stats_tracker.stats
        self.slot_queue: asyncio.Queue[int] = asyncio.Queue()
        for i in range(engines.config.workers):
            self.slot_queue.put_nowait(i)

    @property
    def config(self) -> Config:
        # Read through every time: a provider switch replaces engines.config.
        return self.engines.config

    def on_retry(self, attempt: int, reason: str, delay: float) -> None:
        # Matches RetryCallback's (attempt, reason, delay) signature from
        # openrouter.py's _backoff — model isn't passed through that call
        # chain, so it's taken from the current config instead.
        self.stats.retries += 1
        logger.warning("Retry %d for %s: %s (sleeping %.1fs)", attempt, self.config.model, reason, delay)
        if self.reporter is not None:
            self.reporter.on_retry(self.config.model, attempt, reason, delay)

    def _article_done(self, title: str, status: str) -> None:
        if self.reporter is not None:
            self.reporter.on_article_done(title, status)

    async def process_article(
        self, original: ArticleInput, source: ArticleSource, chunks: list[Chunk], facts: VerifiedFacts
    ) -> None:
        stats = self.stats
        checkpoint_input(self.config, ArticleInput(
            title=source.title, local_path=original.local_path,
            source_lang=source.source_lang,
        ))
        record_attempt(self.config, source.title)

        if not await self._translate_chunks(source, chunks, facts):
            return

        # All chunks succeeded, proceed with assembly and post-processing
        engines = self.engines
        target_mw_client = self.mw_pool.get(self.config.target_lang)
        assembled, combined_issues, rounds_used, citation_languages_filled, removed_comments = await run_assembly_repair(
            chunks, source, self.config, engines.llm_client, self.skill, engines.pricing, target_mw_client,
            self.citation_client, stats, self.on_retry, self.cache, facts, self.qa_skill,
        )

        config = self.config  # read after repair: a provider switch replaces it
        warning_findings = (
            combined_issues
            if combined_issues
            and config.save_with_warnings
            and all(i.severity == "warning" for i in combined_issues)
            else []
        )
        if combined_issues and not warning_findings:
            stats.articles_needs_human_review += 1
            stats.article_results[source.title] = "needs_human_review"
            review_path = record_needs_human_review(config.output_dir, source.title, combined_issues, rounds_used)
            logger.error(
                "Article %r needs human review after %d assembly-level repair round(s) — see %s: %s",
                source.title,
                rounds_used,
                review_path,
                "; ".join(f"{i.kind}: {i.message}" for i in combined_issues),
            )
            self._article_done(source.title, "needs_human_review")
            return

        assembled, review_findings = await self._review(
            source, chunks, facts, assembled, target_mw_client, removed_comments
        )
        flagged = warning_findings + review_findings
        if flagged:
            logger.warning(
                "Article %r has %d unresolved finding(s) (%d warning(s) left after assembly repair, "
                "%d from the semantic review) — saved anyway: %s",
                source.title,
                len(flagged),
                len(warning_findings),
                len(review_findings),
                "; ".join(f"{i.kind}: {i.message}" for i in flagged),
            )

        # Final mandatory QA gate (qa.py): the review pass may have
        # rewritten chunks after the last assembly-level QA round, and
        # a review-flagged article is saved anyway -- a QA defect never is.
        qa_issues = run_qa_checks(assembled)
        if qa_issues:
            stats.articles_needs_human_review += 1
            stats.article_results[source.title] = "needs_human_review"
            review_path = record_needs_human_review(self.config.output_dir, source.title, qa_issues, rounds_used)
            logger.error(
                "Article %r failed the final QA gate — see %s: %s",
                source.title,
                review_path,
                "; ".join(f"{i.kind}: {i.message}" for i in qa_issues),
            )
            self._article_done(source.title, "needs_human_review")
            return

        self._save(source, chunks, facts, assembled, citation_languages_filled, removed_comments, review_findings)

    async def _translate_chunks(self, source: ArticleSource, chunks: list[Chunk], facts: VerifiedFacts) -> bool:
        """Translates every chunk; False (with the failure recorded) if any failed."""
        config, engines, stats = self.config, self.engines, self.stats
        article_stats = {
            'input_tokens': 0,
            'output_tokens': 0,
            'chunks_done': 0,
            'failed': False,
        }

        if config.glossary and len(chunks) > 1 and any(c.status == ChunkStatus.PENDING for c in chunks):
            try:
                await build_article_glossary(
                    chunks, facts, config, engines.llm_client, config.model, self.skill, self.cache,
                    engines.pricing, stats, self.on_retry,
                )
            except Exception as exc:
                logger.warning("Glossary for %r failed, translating without it: %s", source.title, exc)

        # Execute chunk translation with overall article timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*(self._translate_chunk(source, chunks, c, facts, article_stats) for c in chunks)),
                timeout=self.config.article_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Article %r translation timed out after %.1f seconds",
                source.title, self.config.article_timeout_s
            )
            _fail_pending_chunks(chunks)
            article_stats['failed'] = True
        except ArticleLimitExceeded:
            # Already logged, mark remaining chunks as failed
            _fail_pending_chunks(chunks)
            article_stats['failed'] = True
        except Exception as exc:
            logger.error("Unexpected error processing article %r: %s", source.title, exc)
            article_stats['failed'] = True
            _fail_pending_chunks(chunks)

        # Determine article outcome
        failed_chunks = [c for c in chunks if c.status == ChunkStatus.FAILED]
        if failed_chunks or article_stats['failed']:
            stats.articles_failed += 1
            stats.article_results[source.title] = "failed"
            logger.error(
                "Article %r failed: %d of %d sections failed",
                source.title, len(failed_chunks), len(chunks)
            )
            self._article_done(source.title, "failed")
            return False
        return True

    async def _translate_chunk(
        self,
        source: ArticleSource,
        chunks: list[Chunk],
        chunk: Chunk,
        facts: VerifiedFacts,
        article_stats: dict,
    ) -> None:
        engines, reporter = self.engines, self.reporter
        slot_id = await self.slot_queue.get()
        try:
            if reporter is not None:
                reporter.on_chunk_start(slot_id, chunk.article_title, chunk.section_title)
            outcome = None
            for engine_attempt in range(2):  # 1 retry, only after a fallback-provider switch
                config = self.config
                attempted_provider = (
                    resolve_complex_provider(config)
                    if config.complex_model and chunk.is_complex else config.provider
                )
                try:
                    outcome = await translate_chunk(
                        chunk,
                        config,
                        engines.llm_client,
                        self.skill,
                        self.cache,
                        engines.pricing,
                        self.stats,
                        on_retry=self.on_retry,
                        verified_facts=facts,
                        qa_skill=self.qa_skill,
                        complex_pricing=engines.complex_pricing,
                        complex_client=engines.complex_client,
                    )
                    break
                except EngineError as exc:
                    if (
                        engine_attempt == 0
                        and isinstance(exc, InsufficientCreditsError)
                        and (await engines.use_primary_for_secondary(attempted_provider)
                             or await engines.ensure_fallback_engine())
                    ):
                        continue
                    chunk.status = ChunkStatus.FAILED
                    logger.error(
                        "Engine call failed for %s chunk %d: %s", chunk.article_title, chunk.order, exc
                    )
                    article_stats['failed'] = True
                    return
            if not outcome.validation.valid:
                logger.error(
                    "Validation failed for %s chunk %d after repair attempts: %s",
                    chunk.article_title,
                    chunk.order,
                    "; ".join(format_errors(outcome.validation)),
                )
                article_stats['failed'] = True
                return
            # Update article stats
            article_stats['input_tokens'] += outcome.prompt_tokens
            article_stats['output_tokens'] += outcome.completion_tokens
            article_stats['chunks_done'] += 1
            logger.info(
                "Chunk done: %s — %s (%d/%d) [%s]",
                source.title, chunk.section_title, article_stats['chunks_done'], len(chunks), chunk.status.value,
            )
            self._check_article_limits(source, article_stats)
        finally:
            if reporter is not None:
                reporter.on_chunk_done(slot_id)
            self.slot_queue.put_nowait(slot_id)
            self.stats_tracker.write(self.config.stats_path)
            save_partial_article(self.config.partial_output_dir, source.title, chunks)

    def _check_article_limits(self, source: ArticleSource, article_stats: dict) -> None:
        config = self.config
        # Check token ratio limit
        if config.max_token_ratio > 0 and article_stats['input_tokens'] > 0:
            ratio = article_stats['output_tokens'] / article_stats['input_tokens']
            if ratio > config.max_token_ratio:
                logger.error(
                    "Article %r token ratio exceeded: output/input = %.2f > %.2f",
                    source.title, ratio, config.max_token_ratio
                )
                raise ArticleLimitExceeded(f"Token ratio {ratio:.2f} exceeds limit {config.max_token_ratio}")
        # Check total token limit
        if config.max_article_tokens > 0:
            total_tokens = article_stats['input_tokens'] + article_stats['output_tokens']
            if total_tokens > config.max_article_tokens:
                logger.error(
                    "Article %r total tokens exceeded: %d > %d",
                    source.title, total_tokens, config.max_article_tokens
                )
                raise ArticleLimitExceeded(f"Total tokens {total_tokens} exceeds limit {config.max_article_tokens}")

    async def _review(
        self,
        source: ArticleSource,
        chunks: list[Chunk],
        facts: VerifiedFacts,
        assembled: str,
        target_mw_client: MediaWikiClient,
        removed_comments: list[str],
    ) -> tuple[str, list[ValidationIssue]]:
        """Runs the semantic review pass when a review model is configured.
        Appends the comments it removes to removed_comments."""
        engines = self.engines
        review_findings: list[ValidationIssue] = []
        if engines.review_model is None:
            return assembled, review_findings
        for review_attempt in range(2):
            attempted_provider = engines.review_provider
            try:
                assembled, review_findings, _, review_removed_comments = await run_review_pass(
                    chunks, source, self.config, assembled, engines.review_client, engines.review_model, self.skill,
                    engines.review_pricing, target_mw_client, self.stats, facts, self.on_retry, self.cache,
                    self.qa_skill, self.citation_client,
                )
                break
            except InsufficientCreditsError:
                switched = False
                if review_attempt == 0:
                    switched = await engines.use_primary_for_secondary(attempted_provider)
                    if not switched and await engines.ensure_fallback_engine():
                        engines.review_on_primary()
                        switched = True
                if switched:
                    # A failed review may already have repaired chunks.
                    assembled, _ = await post_process_assembled(
                        assemble_chunks(chunks), source.title, "\n".join(c.text for c in chunks), self.config,
                        self.citation_client,
                        target_mw_client, source.source_lang,
                    )
                    continue
                raise
        removed_comments.extend(review_removed_comments)
        if review_findings:
            self.stats.articles_flagged_by_review += 1
        return assembled, review_findings

    def _save(
        self,
        source: ArticleSource,
        chunks: list[Chunk],
        facts: VerifiedFacts,
        assembled: str,
        citation_languages_filled: int,
        removed_comments: list[str],
        review_findings: list[ValidationIssue],
    ) -> None:
        config, stats = self.config, self.stats
        # Add attribution block as HTML comment at the bottom of the file
        attribution = build_attribution_block(source)
        if attribution:
            assembled = f"{assembled}\n\n{attribution}"

        path = save_article(config.output_dir, source.title, assembled)
        stats.articles_completed += 1
        stats.article_results[source.title] = "completed"
        logger.info("Saved %s", path)
        discard_partial_article(config.partial_output_dir, source.title)

        if config.generate_reports:
            report_text = build_article_report(
                ArticleReportData(
                    source=source,
                    chunks=chunks,
                    facts=facts,
                    citation_languages_filled=citation_languages_filled,
                    removed_comments=removed_comments,
                    review_model=self.engines.review_model,
                    review_findings=review_findings,
                ),
                assembled,
            )
            report_path = save_report(config.output_dir, source.title, report_text)
            logger.info("Saved %s", report_path)

        self._article_done(source.title, "completed")


async def run_pipeline(
    config: Config,
    inputs: list[ArticleInput],
    force: bool = False,
    reporter: ProgressReporter | None = None,
    stats_tracker: StatsTracker | None = None,
    resume_unfinished: bool = True,
    resume_requested_only: bool = False,
) -> StatsTracker:
    stats_tracker = stats_tracker if stats_tracker is not None else StatsTracker()
    stats = stats_tracker.stats

    if resume_unfinished:
        unfinished = unfinished_inputs(config, inputs, requested_only=resume_requested_only)
        if unfinished:
            logger.info("Retrying %d unfinished article(s) before starting new work", len(unfinished))
            await run_pipeline(
                config.model_copy(update={"sequential": True}), unfinished,
                reporter=reporter, stats_tracker=stats_tracker, resume_unfinished=False,
            )
            attempted = {input_key(item, config) for item in unfinished}
            inputs = [item for item in inputs if input_key(item, config) not in attempted]
        if not inputs:
            return stats_tracker

    skill = load_skill(config.skill_path, config.include_skill_references, config.skill_git_ref)
    qa_skill = load_skill(config.qa_skill_path, config.include_skill_references, config.skill_git_ref)
    cache = TranslationCache(config.cache_db_path) if config.cache else None
    verification_cache = VerificationCache(config.verification_db_path) if config.verify_links else None

    mw_pool = MediaWikiClientPool(config.user_agent, config.source_lang, config.source_wiki_api)
    client_pool, config = build_client_pool(config)
    engines = _Engines(config, client_pool, reporter)
    wikidata_client = (
        httpx.AsyncClient(headers={"User-Agent": config.user_agent}, timeout=config.wikidata_timeout_s)
        if config.verify_links
        else None
    )
    # Separate from wikidata_client: this one fetches arbitrary third-party
    # citation URLs, independently toggleable from verify_links.
    citation_client = (
        httpx.AsyncClient(headers={"User-Agent": config.user_agent}) if config.fill_citation_languages else None
    )

    article_locks: list[int] = []  # held until the run ends; the kernel also drops them if we crash
    try:
        await engines.load_pricing()

        pending_items = _claim_pending_items(inputs, config, force, stats, article_locks)
        fetch_sem = asyncio.Semaphore(_FETCH_CONCURRENCY)
        planned = await asyncio.gather(
            *(
                _plan_article(mw_pool, item, config, fetch_sem, wikidata_client, verification_cache)
                for item in pending_items
            )
        )
        article_plans = _articles_to_translate(pending_items, planned, stats)

        if reporter is not None:
            all_chunks = [chunk for _, _, chunks, _ in article_plans for chunk in chunks]
            reporter.set_plan(
                len(article_plans), len(all_chunks), _estimate_cost(all_chunks, engines.pricing),
                {source.title: len(chunks) for _, source, chunks, _ in article_plans},
            )

        runner = _ArticleRunner(engines, skill, qa_skill, cache, mw_pool, citation_client, reporter, stats_tracker)
        if config.sequential:
            for plan in article_plans:
                await runner.process_article(*plan)
        else:
            await asyncio.gather(*(runner.process_article(*plan) for plan in article_plans))

    finally:
        for lock in article_locks:
            release_lock(lock)
        await mw_pool.aclose()
        # Closes every client the pool actually built (complex/review's
        # own client, only when routed to a different provider than the
        # draft tier) plus llm_client itself -- which, after a mid-run
        # ensure_fallback_engine() switch, is a client build_llm_client
        # constructed directly and is NOT one of client_pool's entries (see
        # that function's own "deliberately not closing the old client"
        # comment: the pre-switch client is intentionally left for this
        # same final cleanup instead of being closed mid-run under a
        # possible in-flight call). Deduplicated by identity so a
        # single-provider run (the common case, llm_client IS the pool's
        # one entry) still closes exactly one client, not two.
        seen_clients: set[int] = set()
        for client in (engines.llm_client, *client_pool.clients.values()):
            if id(client) in seen_clients:
                continue
            seen_clients.add(id(client))
            await client.aclose()
        if wikidata_client is not None:
            await wikidata_client.aclose()
        if citation_client is not None:
            await citation_client.aclose()
        if cache is not None:
            cache.close()
        if verification_cache is not None:
            verification_cache.close()
        # config.provider/model reflect whichever engine was actually in
        # force when the run ended, including a mid-run fallback switch
        # (ensure_fallback_engine) -- callers like queue_runner.py want
        # that effective value, not just what the run started with.
        stats_tracker.stats.provider = engines.config.provider
        stats_tracker.stats.model = engines.config.model
        stats_tracker.stats.review_model = engines.review_model
        stats_tracker.stats.review_provider = engines.review_provider
        stats_tracker.write(engines.config.stats_path)

    return stats_tracker
