"""Resolves --title / --titles / --category / --file / --directory into work items.

Source language is generic, not hardcoded to English: each raw title can
carry its own source wiki, detected from a full Wikipedia URL or a
`lang:Title` prefix (interwiki-link style, e.g. `sq:Gjergj Arianiti`).
Falls back to the run's configured default source_lang when neither is
present. Target language stays a single run-wide setting.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from wiki_translation_harness.mediawiki import MediaWikiClient
from wiki_translation_harness.models import ArticleSource

_WIKI_URL_RE = re.compile(
    r"^(?:https?://)?([a-z0-9-]+)\.(?:m\.)?wikipedia\.org/wiki/([^?#]+)(?:[?#].*)?$",
    re.IGNORECASE,
)
# A lowercase 2-3 letter prefix (optionally with a subtag, e.g. zh-hans) is
# treated as a language code, not a MediaWiki namespace: real namespaces
# (Category:, Template:, File:, Talk:...) are capitalized by convention, so
# this rarely collides with a genuine title in practice.
_LANG_PREFIX_RE = re.compile(r"^([a-z]{2,3}(?:-[a-z]+)?):(.+)$")


@dataclass
class ArticleInput:
    title: str
    local_path: Path | None = None  # set for --file/--directory: raw wikitext already on disk
    source_lang: str | None = None  # None = use the run's configured default source_lang
    revid: int | None = None  # fetch this exact revision instead of the current one


def parse_source_ref(raw: str) -> tuple[str | None, str]:
    """Detect (source_lang, title) from a full Wikipedia URL or a `lang:Title`
    prefix. Returns (None, raw) when neither pattern matches."""
    raw = raw.strip()

    url_match = _WIKI_URL_RE.match(raw)
    if url_match:
        lang = url_match.group(1).lower()
        title = urllib.parse.unquote(url_match.group(2)).replace("_", " ")
        return lang, title

    prefix_match = _LANG_PREFIX_RE.match(raw)
    if prefix_match:
        lang = prefix_match.group(1).lower()
        title = prefix_match.group(2).strip()
        if title:
            return lang, title

    return None, raw


def resolve_static_inputs(
    title: str | None,
    titles_file: Path | None,
    file: Path | None,
    directory: Path | None,
) -> list[ArticleInput]:
    inputs: list[ArticleInput] = []

    if title:
        lang, parsed_title = parse_source_ref(title)
        inputs.append(ArticleInput(title=parsed_title, source_lang=lang))

    if titles_file:
        for line in titles_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lang, parsed_title = parse_source_ref(line)
                inputs.append(ArticleInput(title=parsed_title, source_lang=lang))

    if file:
        inputs.append(ArticleInput(title=file.stem.replace("_", " "), local_path=file))

    if directory:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in (".wiki", ".txt"):
                inputs.append(ArticleInput(title=path.stem.replace("_", " "), local_path=path))

    return inputs


async def load_article_source(
    mw_client: MediaWikiClient, item: ArticleInput, source_lang: str
) -> ArticleSource:
    """Local file input skips the MediaWiki fetch entirely (no HTML scraping either way)."""
    if item.local_path is not None:
        text = item.local_path.read_text(encoding="utf-8")
        return ArticleSource(title=item.title, wikitext=text, source_lang=source_lang)
    return await mw_client.fetch_article(item.title, item.revid)
