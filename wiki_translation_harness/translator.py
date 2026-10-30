"""Per-chunk orchestration: cache check -> invoke skill -> validate -> repair -> cache store.

This module contains no translation judgment — it only sequences calls into
cache.py, skill_loader.py, openrouter.py, validator.py, and repair.py.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

import mwparserfromhell as mwp

from wiki_translation_harness.cache import TranslationCache, compute_key
from wiki_translation_harness.engines import LLMEngineClient
from wiki_translation_harness.models import Chunk, ChunkStatus, Config, ModelPricing, RunStats, ValidationResult
from wiki_translation_harness.openrouter import RetryCallback, run_completion
from wiki_translation_harness.repair import repair_chunk
from wiki_translation_harness.skill_loader import SkillContent, build_translation_messages
from wiki_translation_harness.validator import format_errors, validate_wikitext
from wiki_translation_harness.verification import VerifiedFacts, build_verified_facts_block


_REF_DEF_BLOCK_RE = re.compile(r"<ref\b[^>]*?(?<!/)>.*?</ref\s*>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REFERENCES_TAG_RE = re.compile(r"</?references\b[^>]*>", re.IGNORECASE)


def is_reference_definitions_only(text: str) -> bool:
    """True when a chunk holds nothing but `<ref>` definitions (plus
    comments and `<references>` tags): citation bodies are kept untranslated
    anyway, so the model has nothing to add — and given a long run of them
    it rewrites the block (bullets, dropped `<ref name>` wrappers, invented
    headings and `{{Reflist}}` calls) in ways no repair pass undoes. A chunk
    with any other content, a heading included, still goes to the model."""
    if not _REF_DEF_BLOCK_RE.search(text):
        return False
    rest = _REF_DEF_BLOCK_RE.sub("", text)
    rest = _HTML_COMMENT_RE.sub("", rest)
    rest = _REFERENCES_TAG_RE.sub("", rest)
    return not rest.strip()


_HEADING_LINE_RE = re.compile(r"^=+[^=\n].*=+[ \t]*$")


def split_heading_and_reference_definitions(text: str) -> tuple[str, str] | None:
    """(heading line, rest) when a chunk is a section heading followed by
    nothing but reference definitions (see is_reference_definitions_only),
    else None."""
    head, sep, rest = text.lstrip("\n").partition("\n")
    if not sep or not _HEADING_LINE_RE.match(head.strip()):
        return None
    if not is_reference_definitions_only(rest):
        return None
    return head, sep + rest


def keep_source_reference_block(source_text: str, translated_text: str) -> str:
    """For a heading + reference-definitions chunk, only the heading needs
    the model: it rewrites the definitions that follow (dropping the
    <references> opener among other things), so keep the translated
    heading and the source's block. Any other chunk is returned as is."""
    split = split_heading_and_reference_definitions(source_text)
    if split is None:
        return translated_text
    source_heading, source_rest = split
    translated_head = translated_text.lstrip("\n").partition("\n")[0]
    if not _HEADING_LINE_RE.match(translated_head.strip()):
        translated_head = source_heading
    return translated_head + source_rest


logger = logging.getLogger(__name__)

_INFOBOX_PREFIXES = ("infobox", "kuti")
_INFOBOX_ATTEMPTS = 3


def _top_level_infoboxes(text: str) -> list[str]:
    return [
        str(node)
        for node in mwp.parse(text).nodes
        if isinstance(node, mwp.nodes.Template) and str(node.name).strip().lower().startswith(_INFOBOX_PREFIXES)
    ]


def dropped_infoboxes(source_text: str, translated_text: str) -> list[str]:
    """Top-level infoboxes of the source that the translation lost.

    A big infobox is the first thing a model skips when it condenses a long
    chunk (Mars: every cached Lead translation had none), and nothing else
    notices — the refs only the infobox used then render as Cite errors.
    Compared by count, since the translation may rename the template."""
    source_boxes = _top_level_infoboxes(source_text)
    if not source_boxes:
        return []
    return source_boxes[len(_top_level_infoboxes(translated_text)):]


def insert_infoboxes(translated_text: str, boxes: list[str]) -> str:
    """Insert infoboxes after the translation's leading templates (short
    description, hatnotes, maintenance banners) — where the source keeps
    them too."""
    pos = offset = 0
    for node in mwp.parse(translated_text).nodes:
        text = str(node)
        offset += len(text)
        if isinstance(node, (mwp.nodes.Template, mwp.nodes.Comment)):
            pos = offset
        elif not (isinstance(node, mwp.nodes.Text) and not text.strip()):
            break
    return translated_text[:pos] + "\n" + "\n".join(boxes) + translated_text[pos:]


def restore_dropped_infoboxes(source_text: str, translated_text: str) -> tuple[str, int]:
    """Last resort: put the source's lost infoboxes back verbatim (still
    English). Returns (patched_text, restored_count)."""
    missing = dropped_infoboxes(source_text, translated_text)
    if not missing:
        return translated_text, 0
    return insert_infoboxes(translated_text, missing), len(missing)


@dataclass
class ChunkOutcome:
    chunk: Chunk
    validation: ValidationResult
    from_cache: bool
    repair_attempts: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


def is_complex_routed(config: Config, chunk: Chunk) -> bool:
    """Hybrid-model routing: complex chunks go to the complex tier
    (config.complex_model) when one is configured."""
    return bool(config.complex_model) and chunk.is_complex


def translation_model(config: Config, chunk: Chunk) -> str:
    """The model translate_chunk sends this chunk to."""
    return config.complex_model if is_complex_routed(config, chunk) else config.model


def chunk_cache_key(model: str, target_lang: str, chunk: Chunk, skill: SkillContent, facts_block: str) -> str:
    """The translation-cache key translate_chunk stores this chunk under."""
    facts_hash = hashlib.sha256(facts_block.encode("utf-8")).hexdigest()[:16] if facts_block else ""
    return compute_key(model, chunk.source_lang, target_lang, chunk.text, skill.content_hash, facts_hash)


async def translate_chunk(
    chunk: Chunk,
    config: Config,
    client: LLMEngineClient,
    skill: SkillContent,
    cache: TranslationCache | None,
    pricing: ModelPricing | None,
    stats: RunStats,
    on_retry: RetryCallback | None = None,
    verified_facts: VerifiedFacts | None = None,
    qa_skill: SkillContent | None = None,
    complex_pricing: ModelPricing | None = None,
    complex_client: LLMEngineClient | None = None,
) -> ChunkOutcome:
    if is_reference_definitions_only(chunk.text):
        chunk.translated_text = chunk.text
        chunk.status = ChunkStatus.TRANSLATED
        return ChunkOutcome(chunk=chunk, validation=ValidationResult(valid=True), from_cache=False, repair_attempts=0)

    # Hybrid-model routing: complex chunks use complex_model (and, when the
    # caller passed one, a distinct complex_client -- see
    # engines.build_client_pool/config.resolve_complex_provider) when configured.
    complex_routed = is_complex_routed(config, chunk)
    effective_model = translation_model(config, chunk)
    effective_pricing = complex_pricing if complex_routed else pricing
    effective_client = complex_client if (complex_routed and complex_client is not None) else client
    facts_block = build_verified_facts_block(chunk.text, verified_facts) if verified_facts else ""

    key = chunk_cache_key(effective_model, config.target_lang, chunk, skill, facts_block)
    chunk.cache_key = key

    cached_text = cache.get(key) if cache is not None else None
    if cached_text is not None:
        cached_text = keep_source_reference_block(chunk.text, cached_text)
        if dropped_infoboxes(chunk.text, cached_text):
            logger.warning(
                "Ignoring cached translation of %r chunk %d: it dropped the source infobox",
                chunk.article_title, chunk.order,
            )
            cached_text = None
    if cached_text is not None:
        stats.cache_hits += 1
        chunk.translated_text = cached_text
        chunk.status = ChunkStatus.CACHED
        validation = validate_wikitext(cached_text) if config.validate_output else ValidationResult(valid=True)
        if validation.valid:
            if verified_facts is not None:
                verified_facts.record_established_renderings(cached_text)
            return ChunkOutcome(chunk=chunk, validation=validation, from_cache=True, repair_attempts=0)

    total_prompt_tokens = total_completion_tokens = 0
    total_latency = 0.0
    translated_text = cached_text
    if cached_text is None:
        stats.cache_misses += 1
        messages = build_translation_messages(
            skill, chunk.source_lang, config.target_lang, chunk.article_title,
            chunk.section_title, chunk.text, verified_facts_block=facts_block,
        )
        result = await run_completion(effective_client, effective_model, messages, config.temperature, effective_pricing, on_retry=on_retry)
        stats.record_usage(effective_model, result)
        total_prompt_tokens = result.prompt_tokens
        total_completion_tokens = result.completion_tokens
        total_latency = result.latency_s
        translated_text = result.text
        missing_infoboxes = dropped_infoboxes(chunk.text, translated_text)
        if missing_infoboxes:
            # The model skipped it inside a long chunk; asked about the
            # infobox alone it rarely does, and the rest of the chunk is
            # already translated.
            logger.warning(
                "Translation of %r chunk %d dropped %d source infobox(es); translating them on their own",
                chunk.article_title, chunk.order, len(missing_infoboxes),
            )
            translated_boxes: list[str] = []
            for box in missing_infoboxes:
                box_messages = build_translation_messages(
                    skill, chunk.source_lang, config.target_lang, chunk.article_title,
                    "Infobox", box, verified_facts_block=facts_block,
                )
                for box_attempt in range(1, _INFOBOX_ATTEMPTS + 1):
                    box_result = await run_completion(effective_client, effective_model, box_messages, config.temperature, effective_pricing, on_retry=on_retry)
                    stats.record_usage(effective_model, box_result)
                    total_prompt_tokens += box_result.prompt_tokens
                    total_completion_tokens += box_result.completion_tokens
                    total_latency += box_result.latency_s
                    if len(_top_level_infoboxes(box_result.text)) == 1:
                        translated_boxes.append(box_result.text.strip())
                        break
                    logger.warning(
                        "Infobox-only translation of %r chunk %d (attempt %d/%d) has no infobox; starts: %r",
                        chunk.article_title, chunk.order, box_attempt, _INFOBOX_ATTEMPTS, box_result.text[:200],
                    )
            translated_text = insert_infoboxes(translated_text, translated_boxes)
            if dropped_infoboxes(chunk.text, translated_text):
                translated_text, restored = restore_dropped_infoboxes(chunk.text, translated_text)
                logger.warning(
                    "Restored %d source infobox(es) untranslated into %r chunk %d: the model dropped them again",
                    restored, chunk.article_title, chunk.order,
                )
    translated_text = keep_source_reference_block(chunk.text, translated_text)
    validation = ValidationResult(valid=True)
    repair_attempts = 0

    if config.validate_output:
        validation = validate_wikitext(translated_text)
        while not validation.valid and config.repair and repair_attempts < config.max_repair_attempts:
            repair_attempts += 1
            stats.repair_attempts += 1
            errors = format_errors(validation)
            repair_result = await repair_chunk(
                effective_client,
                skill,
                effective_model,
                config.temperature,
                chunk.source_lang,
                config.target_lang,
                chunk.article_title,
                chunk.section_title,
                translated_text,
                errors,
                effective_pricing,
                on_retry=on_retry,
                qa_skill=qa_skill,
            )
            stats.record_usage(effective_model, repair_result)
            total_prompt_tokens += repair_result.prompt_tokens
            total_completion_tokens += repair_result.completion_tokens
            total_latency += repair_result.latency_s
            translated_text = repair_result.text
            validation = validate_wikitext(translated_text)

        if repair_attempts > 0:
            if validation.valid:
                stats.repairs_succeeded += 1
            else:
                stats.repairs_failed += 1
        if not validation.valid:
            stats.validation_failures += 1

    chunk.translated_text = translated_text
    if not config.validate_output:
        chunk.status = ChunkStatus.TRANSLATED
    elif validation.valid:
        chunk.status = ChunkStatus.REPAIRED if repair_attempts > 0 else ChunkStatus.TRANSLATED
    else:
        chunk.status = ChunkStatus.FAILED

    if verified_facts is not None and chunk.status in (ChunkStatus.TRANSLATED, ChunkStatus.REPAIRED):
        # Relays this chunk's own {{ill}} choices forward to whichever of
        # this article's other chunks haven't started yet (or haven't built
        # their facts_block yet) — see VerifiedFacts.record_established_renderings.
        verified_facts.record_established_renderings(translated_text)

    if cache is not None and chunk.status != ChunkStatus.FAILED:
        cache.set(key, effective_model, chunk.source_lang, config.target_lang, chunk.text, translated_text)

    stats.sections_translated += 1

    return ChunkOutcome(chunk=chunk, validation=validation, from_cache=False, repair_attempts=repair_attempts, prompt_tokens=total_prompt_tokens, completion_tokens=total_completion_tokens, latency_s=total_latency)
