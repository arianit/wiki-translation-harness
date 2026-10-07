"""wiki-translation-harness CLI entrypoint. Wires config, sources, and pipeline together."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from wiki_translation_harness.output import sanitize_filename
from wiki_translation_harness.evaluation import save_evaluation_results
from wiki_translation_harness.config import build_config
from wiki_translation_harness.logging_setup import setup_logging
from wiki_translation_harness.mediawiki import MediaWikiClient, wiki_api_url_for_lang
from wiki_translation_harness.progress import ProgressReporter
from wiki_translation_harness.sources import ArticleInput, parse_source_ref, resolve_static_inputs
from wiki_translation_harness.statistics import StatsTracker
from wiki_translation_harness.benchmark import run_benchmark
from wiki_translation_harness.queue_runner import DEFAULT_QUEUE_REPO_DIR

app = typer.Typer(
    add_completion=False,
    help="Batch-translate Wikipedia articles into a target-language wiki source, "
    "delegating all translation judgment to a configured Pi skill. Source language is "
    "per-title (a `lang:Title` prefix or full Wikipedia URL, e.g. `sq:Gjergj Arianiti`), "
    "falling back to config.yaml's source_lang; target language is a single run-wide setting.",
)

console = Console()


def _print_model_usage(stats) -> None:
    """Per-model token/cost breakdown (RunStats.model_usage) -- e.g.
    distinguishing a cheap draft model's spend from a stronger
    complex/review model's, when Config.complex_model/review_model differ
    from Config.model. Prints nothing when nothing was ever recorded (e.g.
    a run that hit zero real chat_completion calls)."""
    if not stats.model_usage:
        return
    console.print("Per-model usage:")
    for model, usage in sorted(stats.model_usage.items(), key=lambda kv: -kv[1].cost_usd):
        console.print(
            f"  {model}: {usage.tokens_in:,} in / {usage.tokens_out:,} out tokens, "
            f"${usage.cost_usd:.4f} ({usage.calls} call{'s' if usage.calls != 1 else ''})"
        )


def _build_overrides(
    model: Optional[str],
    complex_model: Optional[str] = None,
    complex_provider: Optional[str] = None,
    review_model: Optional[str] = None,
    review_provider: Optional[str] = None,
    workers: Optional[int] = None,
    temperature: Optional[float] = None,
    max_retries: Optional[int] = None,
    cache: Optional[bool] = None,
    validate: Optional[bool] = None,
    repair: Optional[bool] = None,
    live_validate: Optional[bool] = None,
    provider: Optional[str] = None,
    fallback_provider: Optional[str] = None,
    base_url: Optional[str] = None,
    sequential: Optional[bool] = None,
) -> dict:
    overrides = {
        "model": model,
        "complex_model": complex_model,
        "complex_provider": complex_provider,
        "review_model": review_model,
        "review_provider": review_provider,
        "workers": workers,
        "temperature": temperature,
        "max_retries": max_retries,
        "cache": cache,
        "validate": validate,
        "repair": repair,
        "live_validate": live_validate,
        "provider": provider,
        "fallback_provider": fallback_provider,
        "sequential": sequential,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if base_url is not None:
        # Applies to whichever provider ends up active for this run.
        effective_provider = provider or "openrouter"
        key = {
            "local": "local_base_url",
            "experiential": "experiential_base_url",
        }.get(effective_provider, "openrouter_base_url")
        overrides[key] = base_url
    return overrides


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    title: Optional[str] = typer.Option(
        None, "--title", help="Article title, optionally `lang:Title` or a full Wikipedia URL"
    ),
    titles: Optional[Path] = typer.Option(
        None, "--titles", help="Text file, one title per line (each may carry its own `lang:Title` prefix)"
    ),
    category: Optional[str] = typer.Option(
        None, "--category", help="Category name to expand into articles, optionally `lang:Category`"
    ),
    file: Optional[Path] = typer.Option(None, "--file", help="Local .wiki/.txt file with raw source wikitext"),
    directory: Optional[Path] = typer.Option(None, "--directory", help="Directory of local .wiki/.txt files"),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model id, e.g. deepseek/deepseek-v3.2 (OpenRouter) or a local model name"
    ),
    complex_model: Optional[str] = typer.Option(
        None, "--complex-model",
        help="When set, complex chunks (Infoboxes, tables, dense refs) are routed to this "
        "model instead of --model — a hybrid strategy for cost efficiency.",
    ),
    complex_provider: Optional[str] = typer.Option(
        None, "--complex-provider",
        help="Provider to run --complex-model on, when it should differ from --provider "
        "(e.g. a cheap draft tier on opencode_go, complex chunks on a stronger model via "
        "claude_code). Defaults to --provider.",
    ),
    review_model: Optional[str] = typer.Option(
        None, "--review-model",
        help="When set, an independent semantic-fidelity review pass re-reads the whole "
        "assembled article with this model after structural repair passes clean, looking "
        "for mistranslation, hallucinated/dropped facts, grammar errors, and "
        "cross-article transliteration inconsistency. Defaults to --complex-model when "
        "unset (and --complex-model is set).",
    ),
    review_provider: Optional[str] = typer.Option(
        None, "--review-provider",
        help="Provider to run --review-model on. Defaults to --complex-provider, then --provider.",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider",
        help="'claude_code' (default, uses your Claude Code CLI login, no API key), "
        "'openrouter', 'local' (any OpenAI-compatible server), 'experiential' "
        "(platform.experientiallabs.ai), or 'opencode_go' (runs the `opencode` CLI "
        "under its own separate login/session), or 'codex' (ChatGPT subscription login)",
    ),
    fallback_provider: Optional[str] = typer.Option(
        None, "--fallback-provider",
        help="Engine to switch to if --provider raises an insufficient-credits error "
        "(e.g. OpenRouter HTTP 402, or Claude Code hitting its own session/spend "
        "limit) mid-run -- interactively confirmed here, applied automatically (just "
        "logged) under `queue`. Defaults to 'opencode_go' when --provider is "
        "claude_code, otherwise 'claude_code'.",
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Override the active provider's base URL (pair with --provider local)"
    ),
    workers: Optional[int] = typer.Option(None, "--workers", help="Concurrent section-translation workers"),
    sequential: Optional[bool] = typer.Option(
        None, "--sequential/--no-sequential",
        help="Process articles one at a time (still up to --workers concurrent chunks within "
        "each) instead of starting every requested article concurrently. Use with multiple "
        "--titles to bound a mid-run failure (e.g. a provider rate limit) to one article.",
    ),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    max_retries: Optional[int] = typer.Option(None, "--max-retries"),
    cache: Optional[bool] = typer.Option(None, "--cache/--no-cache"),
    validate: Optional[bool] = typer.Option(None, "--validate/--no-validate"),
    repair: Optional[bool] = typer.Option(None, "--repair/--no-repair"),
    live_validate: Optional[bool] = typer.Option(
        None,
        "--live-validate/--no-live-validate",
        help="Render the assembled article via the target wiki's action=parse API to catch "
        "Lua/Cite errors, missing templates, and leaked flag-template text (see live_validator.py)",
    ),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Path to config.yaml"),
    force: bool = typer.Option(False, "--force", help="Re-translate even if output .wiki already exists"),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    if not any([title, titles, category, file, directory]):
        console.print(ctx.get_help())
        raise typer.Exit(code=1)

    overrides = _build_overrides(
        model=model,
        complex_model=complex_model,
        complex_provider=complex_provider,
        review_model=review_model,
        review_provider=review_provider,
        workers=workers,
        temperature=temperature,
        max_retries=max_retries,
        cache=cache,
        validate=validate,
        repair=repair,
        live_validate=live_validate,
        provider=provider,
        fallback_provider=fallback_provider,
        base_url=base_url,
        sequential=sequential,
    )
    cfg = build_config(config_path, overrides)

    logger = setup_logging(cfg.log_dir)

    async def _run() -> StatsTracker:
        from wiki_translation_harness.pipeline import run_pipeline  # local import: avoids import cost on --help

        inputs: list[ArticleInput] = resolve_static_inputs(title, titles, file, directory)

        if category:
            cat_lang, cat_name = parse_source_ref(category)
            effective_lang = cat_lang or cfg.source_lang
            api_url = (
                cfg.source_wiki_api
                if effective_lang == cfg.source_lang and cfg.source_wiki_api
                else wiki_api_url_for_lang(effective_lang)
            )
            mw_client = MediaWikiClient(api_url, cfg.user_agent, effective_lang)
            try:
                member_titles = await mw_client.fetch_category_members(cat_name)
            finally:
                await mw_client.aclose()
            inputs.extend(ArticleInput(title=t, source_lang=effective_lang) for t in member_titles)

        logger.info(
            "Starting run: %d article(s) requested, provider=%s, model=%s, workers=%d",
            len(inputs),
            cfg.provider,
            cfg.model,
            cfg.workers,
        )

        stats_tracker = StatsTracker()
        with ProgressReporter(stats_tracker.stats, cfg.workers, console=console) as reporter:
            result = await run_pipeline(
                cfg, inputs, force=force, reporter=reporter, stats_tracker=stats_tracker
            )
        return result

    stats_tracker = asyncio.run(_run())
    stats = stats_tracker.stats
    console.print(
        f"\nDone. Articles completed: {stats.articles_completed}, failed: {stats.articles_failed}, "
        f"skipped: {stats.articles_skipped}. Estimated cost: ${stats.estimated_cost_usd:.4f}. "
        f"Stats written to {cfg.stats_path}"
    )
    _print_model_usage(stats)


@app.command()
def benchmark(
    title: Optional[str] = typer.Option(
        None, "--title", help="Article title to benchmark, optionally `lang:Title` or a full Wikipedia URL"
    ),
    file: Optional[Path] = typer.Option(None, "--file", help="Local .wiki/.txt file instead of a title"),
    model: list[str] = typer.Option(
        ..., "--model", help="Model id to include (for the configured provider); repeat for multiple models"
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="Model id for evaluation (must not be among the evaluated models)"
    ),
    no_evaluation: bool = typer.Option(
        False, "--no-evaluation", help="Skip the evaluation step even if judge-model is provided"
    ),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Path to config.yaml"),
    output: Path = typer.Option(Path("quality"), "--output", help="Directory for benchmark outputs"),
) -> None:
    """Translate the same article with several models and compare
    translation, runtime, token usage, and estimated cost under quality/.
    
    If --judge-model is provided, a separate evaluation will be performed by that
    model (blind, randomized) to rank translations and provide quality scores.
    """
    if not title and not file:
        console.print("[red]Provide --title or --file[/red]")
        raise typer.Exit(code=1)

    cfg = build_config(config_path, {})
    setup_logging(cfg.log_dir)

    if title:
        title_lang, title_text = parse_source_ref(title)
    else:
        title_lang, title_text = None, file.stem.replace("_", " ") if file else ""
    item = ArticleInput(title=title_text, local_path=file, source_lang=title_lang)

    # Validate judge model
    do_evaluation = bool(judge_model) and not no_evaluation
    if do_evaluation and judge_model in model:
        console.print(f"[yellow]Judge model {judge_model} is also among evaluated models; skipping evaluation.[/yellow]")
        do_evaluation = False
    
    benchmark_data = asyncio.run(run_benchmark(cfg, item, model, output, judge_model if do_evaluation else None))
    
    results = benchmark_data["results"]
    source_title = benchmark_data["source_title"]
    source_wikitext = benchmark_data["source_wikitext"]
    translations = benchmark_data["translations"]
    evaluation_result = benchmark_data["evaluation"]
    
    console.print(f"\nBenchmark results for {source_title!r}:")
    for m, r in results.items():
        console.print(
            f"  {m}: {r['runtime_s']:.1f}s, in={r['tokens_in']} out={r['tokens_out']} "
            f"tokens, cost=${r['estimated_cost_usd']:.4f}, failed_sections={r['failed_sections']}"
        )
    console.print(f"Full report: {output / (source_title.replace(' ', '_') + '_benchmark.json')}")
    
    if evaluation_result:
        # Create article subdirectory
        article_dir = output / sanitize_filename(source_title)
        article_dir.mkdir(parents=True, exist_ok=True)
        
        # Save evaluation results and create comparison.md
        save_evaluation_results(
            result=evaluation_result,
            output_dir=article_dir,
            source_title=source_title,
            translations=translations,
            benchmark_results=results,
        )
        
        console.print(f"\nEvaluation saved to {article_dir}/")
        console.print(f"  - Translations: {article_dir}/translations/")
        console.print(f"  - Evaluation report: {article_dir}/evaluation/sonnet_judge.md")
        console.print(f"  - Comparison: {article_dir}/comparison.md")
        
        # Print ranking
        ranking = evaluation_result.ranking
        if ranking:
            ranked_models = [evaluation_result.label_to_model.get(l, l) for l in ranking]
            console.print(f"\nRanking by judge: {' > '.join(ranking)} ({' > '.join(ranked_models)})")
    elif do_evaluation:
        console.print("[yellow]Evaluation was requested but failed or not enough successful translations.[/yellow]")


@app.command()
def queue(
    queue_repo_dir: Path = typer.Option(
        DEFAULT_QUEUE_REPO_DIR, "--queue-repo-dir",
        help="Local clone of github.com/arianit/wiki-translate-queue",
    ),
    max_articles: int = typer.Option(10, "--max-articles", help="Stop after this many articles this run"),
    stale_hours: float = typer.Option(
        3.0, "--stale-hours", help="A CLAIMED line older than this is treated as abandoned and reclaimed"
    ),
    model: Optional[str] = typer.Option(None, "--model"),
    complex_model: Optional[str] = typer.Option(None, "--complex-model"),
    complex_provider: Optional[str] = typer.Option(None, "--complex-provider"),
    review_model: Optional[str] = typer.Option(None, "--review-model"),
    review_provider: Optional[str] = typer.Option(None, "--review-provider"),
    provider: Optional[str] = typer.Option(None, "--provider"),
    fallback_provider: Optional[str] = typer.Option(
        None, "--fallback-provider",
        help="Engine to auto-switch to (logged, not prompted -- queue mode has no "
        "interactive terminal) if --provider raises an insufficient-credits error. "
        "Defaults to 'opencode_go' when --provider is claude_code, otherwise "
        "'claude_code'.",
    ),
    base_url: Optional[str] = typer.Option(None, "--base-url"),
    workers: Optional[int] = typer.Option(None, "--workers"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    max_retries: Optional[int] = typer.Option(None, "--max-retries"),
    cache: Optional[bool] = typer.Option(None, "--cache/--no-cache"),
    validate: Optional[bool] = typer.Option(None, "--validate/--no-validate"),
    repair: Optional[bool] = typer.Option(None, "--repair/--no-repair"),
    live_validate: Optional[bool] = typer.Option(None, "--live-validate/--no-live-validate"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Path to config.yaml"),
) -> None:
    """Drain articles from the shared queue (github.com/arianit/wiki-translate-queue)
    instead of a manually-supplied --title/--titles/--category, so this
    harness can run unattended (e.g. from cron) against the same queue
    wikitranslateautorun's batch_controller.py uses."""
    from wiki_translation_harness.queue_runner import run_queue_mode  # local import: mirrors main()

    overrides = _build_overrides(
        model=model,
        complex_model=complex_model,
        complex_provider=complex_provider,
        review_model=review_model,
        review_provider=review_provider,
        workers=workers,
        temperature=temperature,
        max_retries=max_retries,
        cache=cache,
        validate=validate,
        repair=repair,
        live_validate=live_validate,
        provider=provider,
        fallback_provider=fallback_provider,
        base_url=base_url,
    )
    cfg = build_config(config_path, overrides)
    # INFO (not the module default of WARNING): queue mode has no Rich Live
    # table to show progress, so ProgressReporter's on_event=logger.info
    # (wired in queue_runner.py) is the only per-article/per-chunk trail
    # this run has. At the default WARNING level those INFO lines only ever
    # reached logs/run.log, invisible on an unattended run's console/log
    # capture until something errors -- see GitHub issue #3. The formatter
    # in logging_setup.py already prefixes every line with %(asctime)s, so
    # raising this one command's console level is enough to get a live,
    # timestamped trail without touching the interactive Live-table path.
    setup_logging(cfg.log_dir, console_level=logging.INFO)

    start = datetime.now()
    console.print(f"Queue run started at {start:%Y-%m-%d %H:%M:%S}")
    stats_tracker = asyncio.run(
        run_queue_mode(cfg, queue_repo_dir=queue_repo_dir, max_articles=max_articles, stale_hours=stale_hours)
    )
    end = datetime.now()
    stats = stats_tracker.stats
    console.print(
        f"\nQueue run done at {end:%Y-%m-%d %H:%M:%S} (started {start:%H:%M:%S}, "
        f"ran {end - start}). Completed: {stats.articles_completed}, failed: {stats.articles_failed}. "
        f"Estimated cost: ${stats.estimated_cost_usd:.4f}."
    )
    _print_model_usage(stats)


if __name__ == "__main__":
    app()
