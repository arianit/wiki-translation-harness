"""Discover unfinished articles and preserve their original input identity."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from wiki_translation_harness.models import Config
from wiki_translation_harness.output import article_already_done, output_path_for
from wiki_translation_harness.sources import ArticleInput

logger = logging.getLogger(__name__)


def checkpoint_input(config: Config, item: ArticleInput) -> None:
    path = output_path_for(config.partial_output_dir, item.title).with_suffix('.resume.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps({
        'title': item.title,
        'source_lang': item.source_lang or config.source_lang,
        'target_lang': config.target_lang,
        'local_path': str(item.local_path.resolve()) if item.local_path else None,
    }), encoding='utf-8')
    tmp.replace(path)


def _attempts_path(config: Config, title: str) -> Path:
    return output_path_for(config.partial_output_dir, title).with_suffix('.attempts')


def attempts_made(config: Config, title: str) -> int:
    try:
        return int(_attempts_path(config, title).read_text(encoding='utf-8').strip())
    except (OSError, ValueError):
        return 0


def record_attempt(config: Config, title: str) -> int:
    """Counts one more try at translating `title` from its partial. The counter
    sits beside the partial and is removed with it when the article finishes,
    so it only ever grows for articles that keep failing."""
    count = attempts_made(config, title) + 1
    path = _attempts_path(config, title)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(count), encoding='utf-8')
    return count


def unfinished_inputs(
    config: Config, requested: list[ArticleInput], *, requested_only: bool = False
) -> list[ArticleInput]:
    """Old snapshots lack metadata; prefer a matching requested input, then
    infer the source title from the filename using the configured language.
    Invalid metadata raises rather than silently bypassing unfinished work.
    """
    by_name = {output_path_for(config.partial_output_dir, i.title).stem: i for i in requested}
    requested_names = set(by_name)
    found: list[ArticleInput] = []
    stuck: list[str] = []
    names = {p.stem for p in config.partial_output_dir.glob('*.wiki')}
    names.update(p.name.removesuffix('.resume.json') for p in config.partial_output_dir.glob('*.resume.json'))
    for name in sorted(names):
        if requested_only and name not in requested_names:
            continue
        metadata = config.partial_output_dir / f'{name}.resume.json'
        if metadata.exists():
            data = json.loads(metadata.read_text(encoding='utf-8'))
            if data['target_lang'] != config.target_lang:
                continue
            item = ArticleInput(data['title'], Path(data['local_path']) if data['local_path'] else None, data['source_lang'])
        else:
            item = by_name.get(name, ArticleInput(name.replace('_', ' '), source_lang=config.source_lang))
        if article_already_done(config.output_dir, item.title):
            continue
        limit = config.max_resume_attempts
        if limit and name not in requested_names and attempts_made(config, item.title) >= limit:
            stuck.append(item.title)
            continue
        found.append(item)
    if stuck:
        logger.warning(
            "Skipping %d stuck article(s) after %d attempts: %s. Request one with --title, or raise "
            "--max-resume-attempts (0 = no limit) to retry them.",
            len(stuck), limit, ", ".join(stuck),
        )
    return found


def input_key(item: ArticleInput, config: Config) -> tuple[str, str]:
    return item.source_lang or config.source_lang, item.title.replace('_', ' ')
