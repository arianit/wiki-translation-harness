"""Concurrent multi-article orchestration: fetch -> chunk -> translate -> assemble -> save.

Ties together mediawiki.py, parser.py, cache.py, translator.py, and
output.py. Contains no translation logic of its own.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time

import httpx

from wiki_translation_harness.cache import TranslationCache, VerificationCache, compute_key
from wiki_translation_harness.resume import checkpoint_input, unfinished_inputs, input_key
from wiki_translation_harness.config import (
    default_model_for_provider,
    resolve_complex_provider,
    resolve_review_model,
    resolve_review_provider,
)
from wiki_translation_harness.engines import LLMEngineClient, build_client_pool, build_llm_client
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
from wiki_translation_harness.notes import expand_cref_notes
from wiki_translation_harness.models import (
    ArticleSource,
    Chunk,
    ChunkStatus,
    Config,
    EngineError,
    InsufficientCreditsError,
    ModelPricing,
    RunStats,
    ValidationIssue,
)
from wiki_translation_harness.openrouter import RetryCallback
from wiki_translation_harness.output import (
    article_already_done,
    assemble_chunks,
    assemble_chunks_with_spans,
    discard_partial_article,
    save_article,
    save_partial_article,
)
from wiki_translation_harness.parser import build_chunks, split_into_sections
from wiki_translation_harness.progress import ProgressReporter
from wiki_translation_harness.report import ArticleReportData, build_article_report, save_report
from wiki_translation_harness.repair import repair_chunk
from wiki_translation_harness.review import review_article
from wiki_translation_harness.review_queue import record_needs_human_review, record_review_flags
from wiki_translation_harness.skill_loader import SkillContent, load_skill
from wiki_translation_harness.sources import ArticleInput, load_article_source
from wiki_translation_harness.statistics import StatsTracker
from wiki_translation_harness.translator import translate_chunk
from wiki_translation_harness.validator import format_errors, validate_wikitext
from wiki_translation_harness.verification import VerifiedFacts, build_verified_facts_block, verify_wikitext

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
    return [m.group(1) for m in _REF_NAME_IN_MESSAGE_RE.finditer(issue.message)]


def _chunk_mentions_ref(chunk: Chunk, ref_name: str) -> bool:
    """True if the chunk's wikitext contains a <ref name="{ref_name}"> or
    <ref name="{ref_name}"/> occurrence. Checks the translated text first
    (the text being repaired), falling back to the source text in case
    translation dropped/renamed the markup around the ref."""
    for text in (chunk.translated_text, chunk.text):
        if not text:
            continue
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


_REFERENCES_OPEN_RE = re.compile(r"<references\b[^>]*(?<!/)>", re.IGNORECASE)
_REFERENCES_CLOSE = "</references>"
_HEADING_LINE_RE = re.compile(r"^=+[^=].*=+\s*$")


def merge_split_references_blocks(text: str) -> tuple[str, int]:
    """Rejoin a list-defined `<references>` block that chunking split in two.

    A long "Citations" section is cut mid-block, so the first chunk's
    translation closes the `<references>` it opened and the second chunk
    (which has no opener) ends with the source's original `</references>`.
    Assembled, that is `<references> defs </references> more defs
    </references>`: the defs after the first closer sit outside the block,
    so Cite reports every ref used only by them as "no text given" and every
    ref defined inside as "not used in preceding text" — dozens of
    orphaned_named_ref findings that no per-chunk repair can fix, since each
    chunk looks fine alone.

    A closer with no matching opener is the second half's; the previous
    closer (same section, nothing but ref definitions between) is the
    spurious one and is dropped. Returns (patched_text, merged_count)."""
    lines = text.split("\n")
    is_open = False
    last_close: int | None = None
    drop: set[int] = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _HEADING_LINE_RE.match(stripped):
            last_close = None
        elif _REFERENCES_OPEN_RE.search(stripped):
            is_open = True
            last_close = None
        elif stripped.lower() == _REFERENCES_CLOSE:
            if is_open:
                is_open = False
                last_close = i
            elif last_close is not None:
                drop.add(last_close)
                last_close = None
    if not drop:
        return text, 0
    return "\n".join(line for i, line in enumerate(lines) if i not in drop), len(drop)


_GROUP_ATTR_RE = re.compile(
    r"(\bgroup\s*=\s*)(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'/>|}]+))", re.IGNORECASE
)


def _group_value(match: re.Match[str]) -> str:
    return next(g for g in match.groups()[1:] if g is not None).strip()


def restore_ref_group_names(text: str, source_text: str) -> tuple[str, dict[str, str]]:
    """Undo translation of a footnote group's name (group=note ->
    group="shënim").

    The group name never reaches the reader, but it has to match exactly:
    the model translates it in some chunks and keeps it in others, so the
    notes of a translated `<ref group="shënim">` find no list (the
    `{{Reflist|group=note|refs=...}}` that kept its name) and the model
    adds a `<references group="shënim"/>` of its own wherever it pleases.

    A translation-only group maps back when there's no doubt which source
    group it was: the source has just one group, or exactly one source
    group went missing in translation as exactly one new one appeared.
    Returns (patched_text, {translated_name: source_name})."""
    source_groups = {_group_value(m) for m in _GROUP_ATTR_RE.finditer(source_text)}
    text_groups = {_group_value(m) for m in _GROUP_ATTR_RE.finditer(text)}
    unknown = text_groups - source_groups
    if not unknown:
        return text, {}
    if len(source_groups) == 1:
        (target,) = source_groups
        renamed = {name: target for name in unknown}
    elif len(unknown) == 1 and len(source_groups - text_groups) == 1:
        renamed = {next(iter(unknown)): next(iter(source_groups - text_groups))}
    else:
        return text, {}

    def _rename(match: re.Match[str]) -> str:
        name = _group_value(match)
        if name not in renamed:
            return match.group(0)
        return f'{match.group(1)}"{renamed[name]}"'

    return _GROUP_ATTR_RE.sub(_rename, text), renamed


# A tag or template that renders a reference list and carries no
# definitions of its own: `<references />`, an empty `<references></references>`,
# or a one-line `{{Reflist}}`/`{{Notelist}}` without |refs=.
_REF_LIST_RENDERER_RE = re.compile(
    r"<references\b(?P<tag_attrs>[^>]*?)/>"
    r"|<references\b(?P<pair_attrs>[^>]*)>\s*</references\s*>"
    r"|\{\{\s*(?P<tpl>reflist|references|notelist|listë shënimesh)\s*(?P<tpl_args>\|[^{}\n]*)?\}\}",
    re.IGNORECASE,
)


def _ref_list_key(match: re.Match[str]) -> tuple[str, str] | None:
    """(kind, group) of a reference-list renderer, or None if it's a
    template that defines refs (|refs=), which is content, not a stray."""
    tpl = match.group("tpl")
    attrs = match.group("tag_attrs") or match.group("pair_attrs") or match.group("tpl_args") or ""
    if tpl and re.search(r"\|\s*refs\s*=", attrs, re.IGNORECASE):
        return None
    kind = "notelist" if tpl and tpl.lower() in ("notelist", "listë shënimesh") else "refs"
    group = _GROUP_ATTR_RE.search(attrs)
    return kind, _group_value(group) if group else ""


def drop_stray_reference_lists(text: str, source_text: str) -> tuple[str, int]:
    """Remove reference lists the model invented.

    When chunking splits a long list-defined `<references>` block, the
    model translating each later piece "completes" its fragment with a
    `<references />` (or `<references group="..."/>`, or an empty
    `<references></references>`) of its own — the Albert Einstein run had
    nine where the source had one. Each one flushes the refs before it, so
    the definitions after the first one land outside the real block and
    Cite reports every ref used only by them as having no text. The same
    habit drops `<references group="note" />` under random mid-article
    headings.

    For each (kind, group), keeps as many renderers as the source has
    — the last ones, since reference lists close an article — and removes
    the rest, whole line and all where the renderer stood alone.
    Returns (patched_text, removed_count)."""
    allowed: dict[tuple[str, str], int] = {}
    for m in _REF_LIST_RENDERER_RE.finditer(source_text):
        key = _ref_list_key(m)
        if key is not None:
            allowed[key] = allowed.get(key, 0) + 1

    by_key: dict[tuple[str, str], list[re.Match[str]]] = {}
    for m in _REF_LIST_RENDERER_RE.finditer(text):
        key = _ref_list_key(m)
        if key is not None:
            by_key.setdefault(key, []).append(m)

    spans: list[tuple[int, int]] = []
    for key, matches in by_key.items():
        excess = len(matches) - allowed.get(key, 0)
        for m in matches[: max(excess, 0)]:
            start, end = m.span()
            line_start = text.rfind("\n", 0, start) + 1
            line_end = text.find("\n", end)
            line_end = len(text) if line_end == -1 else line_end
            if not text[line_start:start].strip() and not text[end:line_end].strip():
                start, end = line_start, min(line_end + 1, len(text))
            spans.append((start, end))
    if not spans:
        return text, 0

    parts: list[str] = []
    pos = 0
    for start, end in sorted(spans):
        parts.append(text[pos:start])
        pos = end
    parts.append(text[pos:])
    return "".join(parts), len(spans)


_DEFS_AFTER_REFERENCES_RE = re.compile(
    # HTML comments may sit between the definitions (the model leaves its
    # own notes there); the run still has to end on a definition.
    r"</references\s*>((?:(?:\s*<!--.*?-->)*\s*<ref\b[^>]*?(?<!/)>(?:(?!<ref\b).)*?</ref\s*>)+)",
    re.IGNORECASE | re.DOTALL,
)


def extend_references_block_over_trailing_defs(text: str) -> tuple[str, int]:
    """Move a `</references>` down past the ref definitions that directly
    follow it.

    The other half of merge_split_references_blocks: there the later
    chunk kept the source's `</references>`; here the model dropped it (or
    drop_stray_reference_lists removed the `<references />` it wrote
    instead), so the block closes after the first chunk's definitions and
    the rest trail after it, outside any list. A named definition right
    after a reference list is never intended, so the run joins the block.
    Returns (patched_text, moved_definition_count)."""
    moved = 0

    def _extend(match: re.Match[str]) -> str:
        nonlocal moved
        defs = match.group(1)
        moved += len(re.findall(r"</ref\s*>", defs, re.IGNORECASE))
        # lstrip: the newline after the old closer would otherwise leave a
        # blank line where it stood.
        return defs.lstrip() + "\n</references>"

    return _DEFS_AFTER_REFERENCES_RE.sub(_extend, text), moved


_REF_NAME_ATTR = r"\bname\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s\"'/>]+))"
_REF_SELF_CLOSING_RE = re.compile(r"<ref\b[^>]*?" + _REF_NAME_ATTR + r"[^>]*?/>", re.IGNORECASE)
_REF_DEFINITION_RE = re.compile(
    # The body may not run past another <ref opener: an unclosed definition
    # (its </ref> lost in translation) would otherwise extend to the NEXT
    # definition's </ref>, hiding that ref from `defined` — and
    # restore_orphaned_ref_definitions would then add a duplicate
    # definition of a ref that was there all along.
    r"<ref\b[^>]*?" + _REF_NAME_ATTR + r"[^>]*?(?<!/)>((?:(?!<ref\b).)*?)</ref\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _ref_name(match: re.Match[str]) -> str:
    return (match.group(1) or match.group(2) or match.group(3)).strip()


def restore_orphaned_ref_definitions(text: str, source_text: str) -> tuple[str, list[str]]:
    """Copy back named-ref definitions that translation dropped.

    The model sometimes condenses away the sentence that carried a
    `<ref name="x">...</ref>` definition while keeping a later
    `<ref name="x" />` reuse — Cite then renders an orphaned-ref error.
    Handing that to the LLM repair loop rarely works: the chunk that has
    the reuse never saw the definition's content, so it either invents a
    citation or does nothing, and the loop burns its rounds. The source
    article still has the exact definition, so this rewrites the first
    orphaned reuse into a full definition using the source's ref body
    (citation bodies are kept untranslated anyway). Refs the source doesn't
    define either are left for the repair loop.

    Returns (patched_text, restored_names)."""
    defined = {_ref_name(m) for m in _REF_DEFINITION_RE.finditer(text)}
    source_defs: dict[str, str] = {}
    for m in _REF_DEFINITION_RE.finditer(source_text):
        source_defs.setdefault(_ref_name(m), m.group(4))

    restored: list[str] = []

    def _restore(match: re.Match[str]) -> str:
        name = _ref_name(match)
        if name in defined or name not in source_defs:
            return match.group(0)
        defined.add(name)
        restored.append(name)
        return f'<ref name="{name}">{source_defs[name]}</ref>'

    return _REF_SELF_CLOSING_RE.sub(_restore, text), restored


async def post_process_assembled(
    text: str,
    source_title: str,
    source_text: str,
    config: Config,
    citation_client: httpx.AsyncClient | None,
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

    text, dropped_lists = drop_stray_reference_lists(text, source_text)
    if dropped_lists:
        logger.info(
            "Removed %d reference list(s) not in the source from %r", dropped_lists, source_title
        )

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

    text, restored_refs = restore_orphaned_ref_definitions(text, source_text)
    if restored_refs:
        logger.info(
            "Restored %d named-ref definition(s) dropped in translation of %r from the source: %s",
            len(restored_refs),
            source_title,
            restored_refs,
        )

    return text, citation_languages_filled


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
        text, filled = await post_process_assembled(text, source.title, source_text, config, citation_client)
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
                        facts_block = build_verified_facts_block(chunk.text, facts) if facts else ""
                        facts_hash = hashlib.sha256(facts_block.encode("utf-8")).hexdigest()[:16] if facts_block else ""
                        key = compute_key(
                            config.model, chunk.source_lang, config.target_lang, chunk.text, skill.content_hash, facts_hash
                        )
                        cache.set(key, config.model, chunk.source_lang, config.target_lang, chunk.text, chunk.translated_text)
                except Exception as exc:
                    logger.warning(
                        "Assembly-level repair failed for %r chunk %d (round %d): %s",
                        source.title, chunk.order, rounds_used, exc,
                    )

        await asyncio.gather(*(
            _repair(chunk, errors_for_chunk)
            for chunk in chunks
            if (errors_for_chunk := (
                localized.get(id(chunk), [])
                + targeted_unlocalized.get(id(chunk), [])
                + broadcast_unlocalized
            ))
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
    repairs only those chunks via the existing repair_chunk(), same as
    run_assembly_repair does for deterministic findings. Capped at
    config.review_max_repair_attempts.

    After a repair round the article is reassembled from the chunks, so
    the comment reconciliation and post_process_assembled fixes that
    run_assembly_repair applied are applied again — otherwise the saved
    article would lose them.

    Returns (assembled_text, remaining_issues, rounds_used,
    removed_comments) — remaining_issues is empty iff the review pass converged (no findings, or every finding
    got resolved within the round cap); a non-empty result does NOT mean
    the caller should withhold the article (see review_queue.record_review_flags)."""

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

    source_text = "\n".join(chunk.text for chunk in chunks)
    removed_comments: list[str] = []
    assembled = initial_assembled
    review_findings = await _run_review_call(assembled)
    structural_issues = await _validate_assembled(
        assembled, target_mw_client, source.title, config.live_validate, config.live_validate_timeout_s
    )
    combined = structural_issues + review_findings

    rounds_used = 0
    while combined and rounds_used < config.review_max_repair_attempts:
        rounds_used += 1
        _, spans = assemble_chunks_with_spans(chunks)
        localized, broadcast = _localize_review_issues(combined, assembled, spans)

        for chunk in chunks:
            errors_for_chunk = localized.get(id(chunk), []) + broadcast
            if not errors_for_chunk:
                continue
            stats.review_attempts += 1
            try:
                repair_result = await repair_chunk(
                    review_client,
                    skill,
                    review_model,
                    config.temperature,
                    chunk.source_lang,
                    config.target_lang,
                    chunk.article_title,
                    chunk.section_title,
                    chunk.translated_text or "",
                    errors_for_chunk,
                    review_pricing,
                    on_retry=on_retry,
                    qa_skill=qa_skill,
                )
                stats.record_usage(review_model, repair_result)
                stats.review_corrections_applied += 1
                chunk.translated_text = repair_result.text
                if cache is not None:
                    facts_block = build_verified_facts_block(chunk.text, facts) if facts else ""
                    facts_hash = (
                        hashlib.sha256(facts_block.encode("utf-8")).hexdigest()[:16] if facts_block else ""
                    )
                    key = compute_key(
                        review_model, chunk.source_lang, config.target_lang, chunk.text, skill.content_hash, facts_hash
                    )
                    cache.set(
                        key, review_model, chunk.source_lang, config.target_lang, chunk.text, chunk.translated_text
                    )
            except Exception as exc:
                logger.warning(
                    "Review-driven repair failed for %r chunk %d (round %d): %s",
                    source.title, chunk.order, rounds_used, exc,
                )

        removed_comments.extend(reconcile_chunk_comments(chunks))
        assembled, _ = await post_process_assembled(
            assemble_chunks(chunks), source.title, source_text, config, citation_client
        )
        structural_issues = await _validate_assembled(
            assembled, target_mw_client, source.title, config.live_validate, config.live_validate_timeout_s
        )
        review_findings = await _run_review_call(assembled)
        combined = structural_issues + review_findings

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


async def run_pipeline(
    config: Config,
    inputs: list[ArticleInput],
    force: bool = False,
    reporter: ProgressReporter | None = None,
    stats_tracker: StatsTracker | None = None,
    resume_unfinished: bool = True,
) -> StatsTracker:
    stats_tracker = stats_tracker if stats_tracker is not None else StatsTracker()
    stats = stats_tracker.stats

    if resume_unfinished:
        unfinished = unfinished_inputs(config, inputs)
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
    qa_skill = (
        load_skill(config.qa_skill_path, config.include_skill_references, config.skill_git_ref)
        if config.qa_skill_path is not None
        else None
    )
    cache = TranslationCache(config.cache_db_path) if config.cache else None
    verification_cache = VerificationCache(config.verification_db_path) if config.verify_links else None

    mw_pool = MediaWikiClientPool(config.user_agent, config.source_lang, config.source_wiki_api)
    client_pool, config = build_client_pool(config)
    llm_client = client_pool.get(config.provider)
    review_model = resolve_review_model(config)
    review_provider = resolve_review_provider(config) if review_model else None
    review_client = client_pool.get(review_provider) if review_provider else None
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

    complex_client = client_pool.get(resolve_complex_provider(config)) if config.complex_model else None

    try:
        pricing = await llm_client.get_pricing_for(config.model)
        complex_pricing = (
            await complex_client.get_pricing_for(config.complex_model)
            if complex_client is not None else None
        )
        review_pricing = (
            await review_client.get_pricing_for(review_model)
            if review_client is not None else None
        )

        pending_items: list[ArticleInput] = []
        for item in inputs:
            if not force and article_already_done(config.output_dir, item.title):
                stats.articles_skipped += 1
                logger.info("Skipping %r: output already exists", item.title)
                continue
            pending_items.append(item)

        fetch_sem = asyncio.Semaphore(_FETCH_CONCURRENCY)
        planned = await asyncio.gather(
            *(
                _plan_article(mw_pool, item, config, fetch_sem, wikidata_client, verification_cache)
                for item in pending_items
            )
        )

        article_plans: list[tuple[ArticleSource, list[Chunk], VerifiedFacts]] = []
        for plan in planned:
            if plan is None:
                stats.articles_failed += 1
                continue
            article_plans.append(plan)

        total_chunks = sum(len(chunks) for _, chunks, _ in article_plans)
        estimated_cost = 0.0
        if pricing is not None:
            for _, chunks, _ in article_plans:
                for chunk in chunks:
                    # Heuristic: assume completion length is roughly comparable to
                    # prompt length for a translation task (source + target text
                    # are similar order of magnitude). Actual cost uses real usage.
                    estimated_cost += chunk.token_estimate * (
                        pricing.prompt_price_per_token + pricing.completion_price_per_token
                    )

        if reporter is not None:
            reporter.set_plan(len(article_plans), total_chunks, estimated_cost)

        slot_queue: asyncio.Queue[int] = asyncio.Queue()
        for i in range(config.workers):
            slot_queue.put_nowait(i)

        def on_retry(attempt: int, reason: str, delay: float) -> None:
            # Matches RetryCallback's (attempt, reason, delay) signature from
            # openrouter.py's _backoff — model isn't passed through that call
            # chain, so it's taken from the enclosing config instead.
            stats.retries += 1
            logger.warning("Retry %d for %s: %s (sleeping %.1fs)", attempt, config.model, reason, delay)
            if reporter is not None:
                reporter.on_retry(config.model, attempt, reason, delay)

        # Guards the one-time provider switch below: several workers can hit
        # InsufficientCreditsError for the same exhausted account at once,
        # but only the first should prompt/log and actually swap the client
        # — the rest just wait for that swap and retry against it too.
        fallback_lock = asyncio.Lock()
        fallback_switched = False
        fallback_declined = False

        async def ensure_fallback_engine() -> bool:
            """On the first call, offers (or, non-interactively, just makes)
            a one-way switch from config.provider to config.fallback_provider
            after a credit-exhaustion failure. Returns True if chunks should
            now retry against the (possibly already, by another worker)
            switched engine, False if there's nothing to switch to or the
            user declined."""
            nonlocal llm_client, pricing, config, fallback_switched, fallback_declined
            async with fallback_lock:
                if fallback_switched:
                    return True
                if fallback_declined:
                    return False
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
                    fallback_declined = True
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
                    fallback_declined = True
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
                llm_client, pricing, config = new_client, new_pricing, new_config
                fallback_switched = True

                logger.warning(
                    "Switched engine to provider=%r model=%r after credit exhaustion.",
                    target, new_config.model,
                )
                if reporter is not None and reporter.on_event is not None:
                    reporter.on_event(f"switched provider to {target!r} after credit exhaustion")
                return True

        async def process_article(source: ArticleSource, chunks: list[Chunk], facts: VerifiedFacts) -> None:
            original = next(
                (item for item, plan in zip(pending_items, planned)
                 if plan is not None and plan[0] is source),
                ArticleInput(title=source.title, source_lang=source.source_lang),
            )
            checkpoint_input(config, ArticleInput(
                title=source.title, local_path=original.local_path,
                source_lang=source.source_lang,
            ))
            article_stats = {
                'input_tokens': 0,
                'output_tokens': 0,
                'start_time': time.monotonic(),
                'chunks_done': 0,
                'failed': False,
                'limit_exceeded': False,
            }

            async def process_chunk(chunk: Chunk, facts: VerifiedFacts) -> None:
                slot_id = await slot_queue.get()
                try:
                    if reporter is not None:
                        reporter.on_chunk_start(slot_id, chunk.article_title, chunk.section_title)
                    outcome = None
                    for engine_attempt in range(2):  # 1 retry, only after a fallback-provider switch
                        try:
                            outcome = await translate_chunk(
                                chunk,
                                config,
                                llm_client,
                                skill,
                                cache,
                                pricing,
                                stats,
                                on_retry=on_retry,
                                verified_facts=facts,
                                qa_skill=qa_skill,
                                complex_pricing=complex_pricing,
                                complex_client=complex_client,
                            )
                            break
                        except EngineError as exc:
                            if (
                                engine_attempt == 0
                                and isinstance(exc, InsufficientCreditsError)
                                and await ensure_fallback_engine()
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
                    # Check token ratio limit
                    if config.max_token_ratio > 0 and article_stats['input_tokens'] > 0:
                        ratio = article_stats['output_tokens'] / article_stats['input_tokens']
                        if ratio > config.max_token_ratio:
                            logger.error(
                                "Article %r token ratio exceeded: output/input = %.2f > %.2f",
                                source.title, ratio, config.max_token_ratio
                            )
                            article_stats['limit_exceeded'] = True
                            raise ArticleLimitExceeded(f"Token ratio {ratio:.2f} exceeds limit {config.max_token_ratio}")
                    # Check total token limit
                    if config.max_article_tokens > 0:
                        total_tokens = article_stats['input_tokens'] + article_stats['output_tokens']
                        if total_tokens > config.max_article_tokens:
                            logger.error(
                                "Article %r total tokens exceeded: %d > %d",
                                source.title, total_tokens, config.max_article_tokens
                            )
                            article_stats['limit_exceeded'] = True
                            raise ArticleLimitExceeded(f"Total tokens {total_tokens} exceeds limit {config.max_article_tokens}")
                finally:
                    if reporter is not None:
                        reporter.on_chunk_done(slot_id)
                    slot_queue.put_nowait(slot_id)
                    stats_tracker.write(config.stats_path)
                    save_partial_article(config.partial_output_dir, source.title, chunks)

            # Execute chunk translation with overall article timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(process_chunk(c, facts) for c in chunks)),
                    timeout=config.article_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Article %r translation timed out after %.1f seconds",
                    source.title, config.article_timeout_s
                )
                # Mark remaining pending chunks as failed
                for c in chunks:
                    if c.status == ChunkStatus.PENDING:
                        c.status = ChunkStatus.FAILED
                article_stats['failed'] = True
            except ArticleLimitExceeded:
                # Already logged, mark remaining chunks as failed
                for c in chunks:
                    if c.status == ChunkStatus.PENDING:
                        c.status = ChunkStatus.FAILED
                article_stats['failed'] = True
            except Exception as exc:
                logger.error("Unexpected error processing article %r: %s", source.title, exc)
                article_stats['failed'] = True
                for c in chunks:
                    if c.status == ChunkStatus.PENDING:
                        c.status = ChunkStatus.FAILED

            # Determine article outcome
            failed_chunks = [c for c in chunks if c.status == ChunkStatus.FAILED]
            if failed_chunks or article_stats['failed']:
                stats.articles_failed += 1
                logger.error(
                    "Article %r failed: %d of %d sections failed",
                    source.title, len(failed_chunks), len(chunks)
                )
                if reporter is not None:
                    reporter.on_article_done(source.title, "failed")
                return

            # All chunks succeeded, proceed with assembly and post-processing
            target_mw_client = mw_pool.get(config.target_lang)
            assembled, combined_issues, rounds_used, citation_languages_filled, removed_comments = await run_assembly_repair(
                chunks, source, config, llm_client, skill, pricing, target_mw_client, citation_client, stats, on_retry,
                cache, facts, qa_skill,
            )

            if combined_issues:
                stats.articles_needs_human_review += 1
                review_path = record_needs_human_review(config.output_dir, source.title, combined_issues, rounds_used)
                logger.error(
                    "Article %r needs human review after %d assembly-level repair round(s) — see %s: %s",
                    source.title,
                    rounds_used,
                    review_path,
                    "; ".join(f"{i.kind}: {i.message}" for i in combined_issues),
                )
                if reporter is not None:
                    reporter.on_article_done(source.title, "needs_human_review")
                return

            review_findings: list[ValidationIssue] = []
            if review_model is not None:
                assembled, review_findings, review_rounds, review_removed_comments = await run_review_pass(
                    chunks, source, config, assembled, review_client, review_model, skill,
                    review_pricing, target_mw_client, stats, facts, on_retry, cache, qa_skill,
                    citation_client,
                )
                removed_comments.extend(review_removed_comments)
                if review_findings:
                    stats.articles_flagged_by_review += 1
                    flags_path = record_review_flags(config.output_dir, source.title, review_findings, review_rounds)
                    logger.warning(
                        "Article %r has %d unresolved semantic-review finding(s) after %d round(s) "
                        "— flagged at %s, saved anyway: %s",
                        source.title,
                        len(review_findings),
                        review_rounds,
                        flags_path,
                        "; ".join(f"{i.kind}: {i.message}" for i in review_findings),
                    )

            # Add attribution block as HTML comment at the bottom of the file
            attribution = build_attribution_block(source)
            if attribution:
                assembled = f"{assembled}\n\n{attribution}"

            path = save_article(config.output_dir, source.title, assembled)
            stats.articles_completed += 1
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
                        review_model=review_model,
                        review_findings=review_findings,
                    ),
                    assembled,
                )
                report_path = save_report(config.output_dir, source.title, report_text)
                logger.info("Saved %s", report_path)

            if reporter is not None:
                reporter.on_article_done(source.title, "completed")

        if config.sequential:
            for source, chunks, facts in article_plans:
                await process_article(source, chunks, facts)
        else:
            await asyncio.gather(
                *(process_article(source, chunks, facts) for source, chunks, facts in article_plans)
            )

    finally:
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
        for client in (llm_client, *client_pool.clients.values()):
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
        stats_tracker.stats.provider = config.provider
        stats_tracker.stats.model = config.model
        stats_tracker.stats.review_model = review_model
        stats_tracker.stats.review_provider = review_provider
        stats_tracker.write(config.stats_path)

    return stats_tracker
