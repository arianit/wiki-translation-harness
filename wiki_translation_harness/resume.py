"""Discover unfinished articles and preserve their original input identity."""
from __future__ import annotations

import json
from pathlib import Path

from wiki_translation_harness.models import Config
from wiki_translation_harness.output import article_already_done, output_path_for
from wiki_translation_harness.sources import ArticleInput


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


def unfinished_inputs(config: Config, requested: list[ArticleInput]) -> list[ArticleInput]:
    """Old snapshots lack metadata; prefer a matching requested input, then
    infer the source title from the filename using the configured language.
    Invalid metadata raises rather than silently bypassing unfinished work.
    """
    by_name = {output_path_for(config.partial_output_dir, i.title).stem: i for i in requested}
    found: list[ArticleInput] = []
    names = {p.stem for p in config.partial_output_dir.glob('*.wiki')}
    names.update(p.name.removesuffix('.resume.json') for p in config.partial_output_dir.glob('*.resume.json'))
    for name in sorted(names):
        metadata = config.partial_output_dir / f'{name}.resume.json'
        if metadata.exists():
            data = json.loads(metadata.read_text(encoding='utf-8'))
            if data['target_lang'] != config.target_lang:
                continue
            item = ArticleInput(data['title'], Path(data['local_path']) if data['local_path'] else None, data['source_lang'])
        else:
            item = by_name.get(name, ArticleInput(name.replace('_', ' '), source_lang=config.source_lang))
        if not article_already_done(config.output_dir, item.title):
            found.append(item)
    return found


def input_key(item: ArticleInput, config: Config) -> tuple[str, str]:
    return item.source_lang or config.source_lang, item.title.replace('_', ' ')
