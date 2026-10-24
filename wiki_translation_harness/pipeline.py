"""Concurrent multi-article orchestration: fetch -> chunk -> translate -> assemble -> save.

Ties together mediawiki.py, parser.py, cache.py, translator.py, and
output.py. Contains no translation logic of its own.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import re
import time

import httpx

from wiki_translation_harness.cache import TranslationCache, VerificationCache, compute_key
from wiki_translation_harness.resume import checkpoint_input, input_key, record_attempt, unfinished_inputs
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
from wiki_translation_harness.hatnote_ill import add_ill_to_hatnotes
from wiki_translation_harness.ref_template import expand_r_definitions
from wiki_translation_harness.notes import (
    IMPLICIT_NOTE_GROUP,
    NOTELIST_ALIASES,
    expand_cref_notes,
    normalize_notelist_aliases,
    uses_implicit_note_group,
)
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
    # {{Efn}} and {{Notelist}} put their notes in lower-alpha when the source
    # names no group, so that group exists in the source without being
    # written there. The model then invents one for the {{Efn}} it turns into
    # a <ref> (Mount Everest "note", Washington D.C. and Periodic table "sh"),
    # and sqwiki's {{Listë shënimesh}} lists lower-alpha only.
    if uses_implicit_note_group(source_text):
        source_groups.add(IMPLICIT_NOTE_GROUP)
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
    r"|\{\{\s*(?P<tpl>reflist|references|notelist|listë shënimesh"
    + "".join("|" + re.escape(a).replace(r"\ ", r"\s+") for a in NOTELIST_ALIASES)
    + r")\s*(?P<tpl_args>\|[^{}\n]*)?\}\}",
    re.IGNORECASE,
)


def _ref_list_key(match: re.Match[str]) -> tuple[str, str] | None:
    """(kind, group) of a reference-list renderer, or None if it's a
    template that defines refs (|refs=), which is content, not a stray."""
    tpl = match.group("tpl")
    attrs = match.group("tag_attrs") or match.group("pair_attrs") or match.group("tpl_args") or ""
    if tpl and re.search(r"\|\s*refs\s*=", attrs, re.IGNORECASE):
        return None
    group = _GROUP_ATTR_RE.search(attrs)
    group_name = _group_value(group) if group else ""
    if tpl and tpl.lower() not in ("reflist", "references"):
        # A note list is a reflist for the group its notes are in, so the
        # model writing {{Reflist|group=lower-alpha}} for {{notelist}}
        # counts as the same list.
        return "refs", group_name or IMPLICIT_NOTE_GROUP
    return "refs", group_name


_NOTES_HEADING_RE = re.compile(r"=+\s*(?:notes?|shënim\w*)\s*=+\s*", re.IGNORECASE)


def _under_notes_heading(text: str, pos: int) -> bool:
    """True when the last non-blank line before `pos` is a Notes/Shënime heading."""
    for line in reversed(text[:pos].splitlines()):
        if line.strip():
            return _NOTES_HEADING_RE.fullmatch(line.strip()) is not None
    return False


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
    — one directly under a Notes heading first, then the last ones, since
    reference lists close an article — and removes the rest, whole line
    and all where the renderer stood alone.
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
        if excess > 0:
            # A list directly under the Notes heading is where the source
            # has it; the model's other copies are strays even when one of
            # them is the last in the file.
            kept = sorted(
                range(len(matches)),
                key=lambda i: (not _under_notes_heading(text, matches[i].start()), -i),
            )[: len(matches) - excess]
            removed = [matches[i] for i in range(len(matches)) if i not in kept]
        else:
            removed = []
        for m in removed:
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


_REFERENCES_BLOCK_RE = re.compile(r"<references\b[^>/]*>(.*?)</references\s*>", re.IGNORECASE | re.DOTALL)


def _block_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _REFERENCES_BLOCK_RE.finditer(text)]


def collapse_duplicate_list_defined_refs(text: str, source_text: str) -> tuple[str, list[str]]:
    """Turn a body definition of a list-defined ref back into a plain reuse.

    The source keeps these refs' text only inside the `<references>` block;
    in the body they are `<ref name="x" />`. A chunk whose model "helpfully"
    writes the full text at the first use (or whose definition
    restore_orphaned_ref_definitions copied in from the source before the
    block's own copy was in place) leaves two definitions — often worded
    differently — and Cite reports the ref as defined multiple times with
    different content. The block's copy wins: it is the source's.

    Only names the source defines inside a block and nowhere else, and only
    refs without a group. Returns (patched_text, collapsed_names)."""
    def _names_in(block_texts: list[str]) -> set[str]:
        return {_ref_name(m) for chunk in block_texts for m in _REF_DEFINITION_RE.finditer(chunk)}

    source_blocks = _block_spans(source_text)
    in_source_blocks = _names_in([source_text[a:b] for a, b in source_blocks])
    source_outside = "".join(
        source_text[start:end]
        for start, end in _complement_spans(len(source_text), source_blocks)
    )
    list_defined = in_source_blocks - _names_in([source_outside])
    blocks = _block_spans(text)
    in_text_blocks = _names_in([text[a:b] for a, b in blocks])
    targets = list_defined & in_text_blocks
    if not targets:
        return text, []

    collapsed: list[str] = []
    out: list[str] = []
    for start, end in _complement_spans(len(text), blocks):
        def _collapse(match: re.Match[str]) -> str:
            name = _ref_name(match)
            opening = match.group(0)[: match.group(0).index(">") + 1]
            if name not in targets or re.search(r"\bgroup\s*=", opening, re.IGNORECASE):
                return match.group(0)
            collapsed.append(name)
            return f'<ref name="{name}" />'

        out.append((start, _REF_DEFINITION_RE.sub(_collapse, text[start:end])))
    pieces = sorted(out + [(a, text[a:b]) for a, b in blocks])
    return "".join(piece for _, piece in pieces), collapsed


def _complement_spans(length: int, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result, pos = [], 0
    for start, end in sorted(spans):
        if start > pos:
            result.append((pos, start))
        pos = end
    if pos < length:
        result.append((pos, length))
    return result


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


def drop_unsupported_ref_reuses(text: str, source_text: str) -> tuple[str, list[str]]:
    """Remove `<ref name="x" />` reuses of a name that nothing defines.

    The model sometimes invents a descriptive name for a citation
    (`corona-particle-density`, `Sun-Rotation-rotation` on the Sun run) and
    reuses it in several places without ever writing the definition. The
    source has no such name, so restore_orphaned_ref_definitions has nothing
    to copy back and the repair loop can only invent a citation; every use
    is a Cite error. A name that appears nowhere in the source and is
    defined nowhere in the translation is unsupported, so its reuses go.

    Returns (patched_text, dropped_names)."""
    defined = {_ref_name(m) for m in _REF_DEFINITION_RE.finditer(text)}
    dropped: list[str] = []

    def _drop(match: re.Match[str]) -> str:
        name = _ref_name(match)
        if name in defined or name in source_text:
            return match.group(0)
        if name not in dropped:
            dropped.append(name)
        return ""

    return _REF_SELF_CLOSING_RE.sub(_drop, text), dropped


def collapse_conflicting_ref_definitions(text: str, source_text: str) -> tuple[str, list[str]]:
    """Turn the later definitions of a name into reuses when the model
    translated identical source definitions differently.

    A source may repeat `<ref name="short">All numbers ... short scale</ref>`
    word for word in several sections (Sun does, twice). Each copy is
    translated in its own chunk, the wording differs, and Cite reports the
    name as defined multiple times with different content. Only names whose
    source definitions all agree are touched; the first definition stays.

    Returns (patched_text, collapsed_names)."""
    def _squash(body: str) -> str:
        return re.sub(r"\s+", " ", body).strip()

    source_bodies: dict[str, set[str]] = {}
    for m in _REF_DEFINITION_RE.finditer(source_text):
        source_bodies.setdefault(_ref_name(m), set()).add(_squash(m.group(4)))

    seen: dict[str, str] = {}
    conflicting: set[str] = set()
    for m in _REF_DEFINITION_RE.finditer(text):
        name, body = _ref_name(m), _squash(m.group(4))
        if name in seen and seen[name] != body and len(source_bodies.get(name, ())) == 1:
            conflicting.add(name)
        seen.setdefault(name, body)
    if not conflicting:
        return text, []

    first: set[str] = set()
    collapsed: list[str] = []

    def _collapse(match: re.Match[str]) -> str:
        name = _ref_name(match)
        if name not in conflicting:
            return match.group(0)
        if name not in first:
            first.add(name)
            return match.group(0)
        if name not in collapsed:
            collapsed.append(name)
        # Keep the opening tag's attributes: a reuse must name the same group.
        opening = re.match(r"<ref\b[^>]*>", match.group(0)).group(0)
        return opening[:-1].rstrip() + " />"

    return _REF_DEFINITION_RE.sub(_collapse, text), collapsed


_REF_TAG_RE = re.compile(
    r"<ref\b(?P<attrs>[^>]*?)(?:/>|(?<!/)>(?P<body>(?:(?!<ref\b).)*?)</ref\s*>)",
    re.IGNORECASE | re.DOTALL,
)
_REF_NAME_ATTR_RE = re.compile(_REF_NAME_ATTR.replace(r"\bname", r"\s*\bname", 1), re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s|}\]<]+")


def _ref_fingerprint(body: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """URLs and digit runs of a ref body: what survives translation of a
    citation (titles and month names change, addresses and numbers don't)."""
    return tuple(_URL_RE.findall(body)), tuple(re.findall(r"\d+", _URL_RE.sub("", body)))


def restore_ref_names(text: str, source_text: str) -> tuple[str, dict[str, list[str | None]]]:
    """Undo the model renaming refs (`Brubaker et al-4` -> `Brubaker et-4`).

    A ref name is an identifier, never prose, but the model sometimes
    rewrites it, or names a ref the source left unnamed — and once several
    different citations share one invented name Cite reports it as defined
    multiple times with different content (Cluj-Napoca: seven Brubaker
    citations under one name).

    Only names the source never uses are touched. Each definition under such
    a name is matched to the source definition with the same URLs and
    numbers; it takes that definition's name, or loses its name when the
    source ref was unnamed. A definition with no unambiguous match is left
    alone. Reuses of the name follow only when its matched definitions
    agree on exactly one source name.

    Returns (patched_text, {translated_name: [restored names; None = unnamed]})."""
    source_tags = list(_REF_TAG_RE.finditer(source_text))
    source_names: set[str] = set()
    by_fingerprint: dict[tuple, list[str | None]] = {}
    for tag in source_tags:
        name_match = _REF_NAME_ATTR_RE.search(tag.group("attrs"))
        name = _ref_name(name_match) if name_match else None
        if name:
            source_names.add(name)
        if tag.group("body") is not None:
            by_fingerprint.setdefault(_ref_fingerprint(tag.group("body")), []).append(name)

    tags = list(_REF_TAG_RE.finditer(text))
    definition_targets: dict[int, str | None] = {}
    named_targets: dict[str, set[str]] = {}
    for index, tag in enumerate(tags):
        name_match = _REF_NAME_ATTR_RE.search(tag.group("attrs"))
        if not name_match or tag.group("body") is None:
            continue
        name = _ref_name(name_match)
        if name in source_names:
            continue
        fingerprint = _ref_fingerprint(tag.group("body"))
        candidates = by_fingerprint.get(fingerprint, []) if any(fingerprint) else []
        if not candidates or len(set(candidates)) != 1:
            continue
        definition_targets[index] = candidates[0]
        if candidates[0] is not None:
            named_targets.setdefault(name, set()).add(candidates[0])

    changes: dict[str, list[str | None]] = {}
    edits: list[tuple[int, int, str]] = []
    for index, tag in enumerate(tags):
        name_match = _REF_NAME_ATTR_RE.search(tag.group("attrs"))
        if not name_match:
            continue
        name = _ref_name(name_match)
        if name in source_names:
            continue
        if tag.group("body") is not None:
            if index not in definition_targets:
                continue
            target = definition_targets[index]
        else:
            targets = named_targets.get(name, set())
            if len(targets) != 1:
                continue
            (target,) = targets
        start = tag.start("attrs") + name_match.start()
        end = tag.start("attrs") + name_match.end()
        edits.append((start, end, f' name="{target}"' if target else ""))
        changes.setdefault(name, []).append(target)

    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text, changes


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
        clean = ([c.translated_text for c in chunks], assembled, review_findings)

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
                if findings_for_chunk:
                    repair_result = await repair_chunk_semantic(
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
                        findings_for_chunk,
                        review_pricing,
                        structural_errors=structural_for_chunk,
                        on_retry=on_retry,
                        qa_skill=qa_skill,
                    )
                else:
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
                        structural_for_chunk,
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
            clean = ([c.translated_text for c in chunks], assembled, review_findings)

    if _has_structural_error(structural_issues) and clean is not None:
        logger.warning(
            "Review-driven repair left a structural error in %r (%s); restoring the last text without one",
            source.title, "; ".join(i.message for i in structural_issues if i.severity == "error"),
        )
        for chunk, text in zip(chunks, clean[0]):
            chunk.translated_text = text
        assembled = clean[1]
        combined = [i for i in clean[2]]

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

    article_locks: list[int] = []  # held until the run ends; the kernel also drops them if we crash
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

        fetch_sem = asyncio.Semaphore(_FETCH_CONCURRENCY)
        planned = await asyncio.gather(
            *(
                _plan_article(mw_pool, item, config, fetch_sem, wikidata_client, verification_cache)
                for item in pending_items
            )
        )

        article_plans: list[tuple[ArticleSource, list[Chunk], VerifiedFacts]] = []
        for item, plan in zip(pending_items, planned):
            if plan is None:
                stats.articles_failed += 1
                stats.article_results[item.title] = "failed"
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
            reporter.set_plan(
                len(article_plans), total_chunks, estimated_cost,
                {source.title: len(chunks) for source, chunks, _ in article_plans},
            )

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

        async def use_primary_for_secondary(provider: str | None) -> bool:
            """Stop sending complex/review work to an exhausted secondary account."""
            nonlocal config, complex_client, complex_pricing
            nonlocal review_client, review_model, review_provider, review_pricing
            async with fallback_lock:
                if provider is None or provider == config.provider:
                    return False
                logger.warning(
                    "Secondary provider %r hit its limit; continuing complex/review work on %r (%s).",
                    provider, config.provider, config.model,
                )
                updates = {}
                if resolve_complex_provider(config) == provider:
                    updates.update(complex_provider=config.provider, complex_model=config.model)
                    complex_client, complex_pricing = llm_client, pricing
                if review_provider == provider:
                    review_provider, review_model = config.provider, config.model
                    review_client, review_pricing = llm_client, pricing
                    updates.update(review_provider=review_provider, review_model=review_model)
                config = config.model_copy(update=updates)
                return True

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
            nonlocal review_client, review_model, review_provider, review_pricing
            original = next(
                (item for item, plan in zip(pending_items, planned)
                 if plan is not None and plan[0] is source),
                ArticleInput(title=source.title, source_lang=source.source_lang),
            )
            checkpoint_input(config, ArticleInput(
                title=source.title, local_path=original.local_path,
                source_lang=source.source_lang,
            ))
            record_attempt(config, source.title)
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
                        attempted_provider = (
                            resolve_complex_provider(config)
                            if config.complex_model and chunk.is_complex else config.provider
                        )
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
                                and (await use_primary_for_secondary(attempted_provider)
                                     or await ensure_fallback_engine())
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
                stats.article_results[source.title] = "failed"
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
                if reporter is not None:
                    reporter.on_article_done(source.title, "needs_human_review")
                return

            review_findings: list[ValidationIssue] = []
            if review_model is not None:
                for review_attempt in range(2):
                    attempted_provider = review_provider
                    try:
                        assembled, review_findings, review_rounds, review_removed_comments = await run_review_pass(
                            chunks, source, config, assembled, review_client, review_model, skill,
                            review_pricing, target_mw_client, stats, facts, on_retry, cache, qa_skill,
                            citation_client,
                        )
                        break
                    except InsufficientCreditsError:
                        switched = False
                        if review_attempt == 0:
                            switched = await use_primary_for_secondary(attempted_provider)
                            if not switched and await ensure_fallback_engine():
                                review_client, review_model = llm_client, config.model
                                review_provider, review_pricing = config.provider, pricing
                                switched = True
                        if switched:
                            # A failed review may already have repaired chunks.
                            assembled, _ = await post_process_assembled(
                                assemble_chunks(chunks), source.title, source.wikitext, config, citation_client,
                                target_mw_client, source.source_lang,
                            )
                            continue
                        raise
                removed_comments.extend(review_removed_comments)
                if review_findings:
                    stats.articles_flagged_by_review += 1
            else:
                review_rounds = 0
            flagged = warning_findings + review_findings
            if flagged:
                flags_path = record_review_flags(
                    config.output_dir, source.title, flagged, rounds_used + review_rounds
                )
                logger.warning(
                    "Article %r has %d unresolved finding(s) (%d warning(s) left after assembly repair, "
                    "%d from the semantic review) — flagged at %s, saved anyway: %s",
                    source.title,
                    len(flagged),
                    len(warning_findings),
                    len(review_findings),
                    flags_path,
                    "; ".join(f"{i.kind}: {i.message}" for i in flagged),
                )

            # Final mandatory QA gate (qa.py): the review pass may have
            # rewritten chunks after the last assembly-level QA round, and
            # a review-flagged article is saved anyway -- a QA defect never is.
            qa_issues = run_qa_checks(assembled)
            if qa_issues:
                stats.articles_needs_human_review += 1
                stats.article_results[source.title] = "needs_human_review"
                review_path = record_needs_human_review(config.output_dir, source.title, qa_issues, rounds_used)
                logger.error(
                    "Article %r failed the final QA gate — see %s: %s",
                    source.title,
                    review_path,
                    "; ".join(f"{i.kind}: {i.message}" for i in qa_issues),
                )
                if reporter is not None:
                    reporter.on_article_done(source.title, "needs_human_review")
                return

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
