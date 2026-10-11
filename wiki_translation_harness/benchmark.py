"""Benchmark mode: translate the same article with several models (on
the configured provider) and record translation, runtime, token usage, and estimated cost for each,
under quality/ so outputs can be compared for translation quality."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import orjson

from wiki_translation_harness.cache import TranslationCache
from wiki_translation_harness.engines import build_llm_client
from wiki_translation_harness.mediawiki import MediaWikiClient, wiki_api_url_for_lang
from wiki_translation_harness.models import ChunkStatus, Config, EngineError
from wiki_translation_harness.output import assemble_chunks, sanitize_filename, save_article
from wiki_translation_harness.parser import build_chunks, split_into_sections
from wiki_translation_harness.skill_loader import load_skill
from wiki_translation_harness.sources import ArticleInput, load_article_source
from wiki_translation_harness.statistics import StatsTracker
from wiki_translation_harness.translator import translate_chunk
from wiki_translation_harness.validator import format_errors, validate_wikitext

logger = logging.getLogger("wiki_translation_harness.benchmark")


async def run_benchmark(
    base_config: Config,
    item: ArticleInput,
    models: list[str],
    output_dir: Path,
    judge_model: str | None = None,
) -> dict:
    """Run translation benchmark for multiple models.
    
    Returns a dict with keys:
    - results: dict model -> metrics
    - source_title: str
    - source_wikitext: str
    - translations: dict model -> translated wikitext (or None if failed)
    - evaluation: EvaluationResult if judge_model provided and evaluation succeeded, else None
    """
    skill = load_skill(base_config.skill_path, base_config.include_skill_references, base_config.skill_git_ref)
    qa_skill = load_skill(base_config.qa_skill_path, base_config.include_skill_references, base_config.skill_git_ref)
    cache = TranslationCache(base_config.cache_db_path) if base_config.cache else None

    effective_lang = item.source_lang or base_config.source_lang
    api_url = (
        base_config.source_wiki_api
        if effective_lang == base_config.source_lang and base_config.source_wiki_api
        else wiki_api_url_for_lang(effective_lang)
    )
    mw_client = MediaWikiClient(api_url, base_config.user_agent, effective_lang)
    try:
        source = await load_article_source(mw_client, item, effective_lang)
    finally:
        await mw_client.aclose()

    sections = split_into_sections(source.wikitext)
    results: dict[str, dict] = {}
    translations: dict[str, str | None] = {}

    for model in models:
        model_config = base_config.model_copy(update={"model": model})
        llm_client, _ = build_llm_client(model_config)
        try:
            pricing = await llm_client.get_pricing_for(model)
            chunks = build_chunks(
                source.title,
                sections,
                model_config.chunk_min_tokens,
                model_config.chunk_max_tokens,
                source_lang=effective_lang,
            )
            stats_tracker = StatsTracker()
            sem = asyncio.Semaphore(model_config.workers)

            async def worker(chunk):
                async with sem:
                    try:
                        await translate_chunk(
                            chunk, model_config, llm_client, skill, cache, pricing, stats_tracker.stats,
                            qa_skill=qa_skill,
                        )
                    except EngineError as exc:
                        chunk.status = ChunkStatus.FAILED
                        logger.error("Engine call failed for model %s: %s", model, exc)

            start = time.monotonic()
            await asyncio.gather(*(worker(c) for c in chunks))
            runtime_s = time.monotonic() - start

            failed = [c for c in chunks if c.status == ChunkStatus.FAILED]
            assembled = assemble_chunks(chunks)

            output_path = None
            if not failed:
                final_check = validate_wikitext(assembled)
                if final_check.valid:
                    model_dir = output_dir / sanitize_filename(model)
                    output_path = save_article(model_dir, source.title, assembled)
                else:
                    logger.error(
                        "Model %s: final assembly failed validation: %s",
                        model,
                        "; ".join(format_errors(final_check)),
                    )

            results[model] = {
                "runtime_s": runtime_s,
                "tokens_in": stats_tracker.stats.tokens_in,
                "tokens_out": stats_tracker.stats.tokens_out,
                "estimated_cost_usd": stats_tracker.stats.estimated_cost_usd,
                "cache_hits": stats_tracker.stats.cache_hits,
                "cache_misses": stats_tracker.stats.cache_misses,
                "repair_attempts": stats_tracker.stats.repair_attempts,
                "validation_failures": stats_tracker.stats.validation_failures,
                "failed_sections": len(failed),
                "output_path": str(output_path) if output_path else None,
            }
            translations[model] = assembled if not failed else None
        finally:
            await llm_client.aclose()

    if cache is not None:
        cache.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{sanitize_filename(source.title)}_benchmark.json"
    report_path.write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    logger.info("Benchmark report written to %s", report_path)

    evaluation_result = None
    if judge_model and judge_model not in models:
        # Filter out failed translations
        valid_translations = {m: t for m, t in translations.items() if t is not None}
        if len(valid_translations) >= 2:
            try:
                from wiki_translation_harness.evaluation import evaluate_translations
                eval_result, error = await evaluate_translations(
                    judge_model=judge_model,
                    source_english=source.wikitext,
                    translations=valid_translations,
                    openrouter_api_key=base_config.openrouter_api_key,
                    openrouter_base_url=base_config.openrouter_base_url,
                    user_agent=base_config.user_agent,
                    timeout=base_config.request_timeout_s,
                )
                if error:
                    logger.error("Evaluation failed: %s", error)
                else:
                    evaluation_result = eval_result
            except ImportError:
                logger.warning("Evaluation module not available")
            except Exception as e:
                logger.error("Evaluation error: %s", e)
        else:
            logger.warning("Not enough successful translations for evaluation (need at least 2)")

    return {
        "results": results,
        "source_title": source.title,
        "source_wikitext": source.wikitext,
        "translations": translations,
        "evaluation": evaluation_result,
    }
