"""Harness-side link/template verification against Wikidata + sqwiki.

The skill's "Verify link targets and templates" section describes a live
research workflow (batch Wikidata sitelink checks, curl-based template
existence/parameter checks) that a single tool-less OpenRouter completion
call cannot execute itself. This module does that work in the harness,
which *does* have network access, and hands the results to the model as
verified facts — mechanical fetching, not translation judgment. Results
persist in cache.py's SQLite store (see VerificationCache) so they only
need to be looked up once, ever, across the whole batch and future runs —
the harness's own, growing equivalent of the skill's sqwiki-verified.md.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

import httpx
import mwparserfromhell as mwp

from wiki_translation_harness.cache import VerificationCache
from wiki_translation_harness.citation_language import CITATION_TEMPLATE_NAMES
from wiki_translation_harness.mediawiki import MediaWikiClient, MediaWikiError

logger = logging.getLogger(__name__)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
_INFOBOX_NAME_RE = re.compile(r"^infobox\b", re.IGNORECASE)
_MAX_TEMPLATE_PARAMS_REPORTED = 60
_MAX_SIBLING_LINKS_REPORTED = 80
# Wikitext length varies between languages, but an existing target article in
# this band is usually the already-translated article rather than a coincidental
# same-title page.  The check is deliberately broad to avoid false negatives.
_SAME_ARTICLE_MIN_RATIO = 0.65
_SAME_ARTICLE_MAX_RATIO = 1.50
# Hard, independent backstop for every individual Wikidata call — httpx's
# own `timeout=` on the client has been confirmed, twice, to not reliably
# fire under real network conditions. Generous enough to never trip on a
# merely-slow-but-alive request; its only job is to guarantee eventual
# cancellation when a socket genuinely hangs.
_WIKIDATA_HARD_TIMEOUT = 60.0

# CS1/CS2 citation templates (cite web, cite book, citation, ...) are almost
# always thin wrappers around Module:Citation/CS1 (confirmed by fetching
# Stampa:Cite book on sqwiki: `{{#invoke:citation/CS1|citation|...}}`, no
# literal {{{param}}} mentions to discover via source inspection at all).
# Their parameter *names* are fixed/English on essentially every Wikipedia
# regardless of the wiki's language — only the rendered output labels are
# localized. This is a stable, well-documented software fact (not a
# translation judgment call), so it's injected statically rather than
# fetched, and it exists specifically because a real run mistranslated
# |title= / |publisher= / |year= into |titulli= / |botues= / |vit=, which
# Module:Citation/CS1 then silently ignored as unknown parameters.
_CITATION_PARAM_NAMES_NOTE = (
    "Citation templates ({{cite web}}, {{cite book}}, {{cite journal}}, "
    "{{citation}}, etc.) use fixed, English parameter NAMES on this wiki "
    "regardless of language — e.g. |title=, |publisher=, |author=, |date=, "
    "|access-date=, |url=, |year=. Only translate parameter VALUES; never "
    "translate or invent parameter names (do not write |titulli=, |botues=, "
    "|vit=, etc. — the citation module silently ignores unrecognized "
    "parameter names, which drops the field from the rendered citation)."
)

_LINK_TARGET_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
_BATCH_SIZE = 50

# File/Image links point at Commons, shared verbatim across every wiki (the
# skill already treats them as needing no translation); Category links need
# translation judgment, not an existence check, so a sitelink lookup isn't
# the right tool for them. Both are excluded from verification targets.
_SKIP_NAMESPACE_RE = re.compile(r"^(file|image|category)\s*:", re.IGNORECASE)


def extract_link_targets(wikitext: str) -> list[str]:
    """Unique wikilink targets worth verifying: excludes external links,
    File/Image links (Commons-shared), and Category links (translated, not
    existence-checked)."""
    seen: dict[str, None] = {}
    for m in _LINK_TARGET_RE.finditer(wikitext):
        target = m.group(1).strip()
        if not target or target.startswith(("http://", "https://")):
            continue
        if _SKIP_NAMESPACE_RE.match(target):
            continue
        seen.setdefault(target, None)
    return list(seen.keys())


def extract_template_names(wikitext: str) -> list[str]:
    """Unique template names (recursive), excluding parser functions (name
    starts with '#') and subst/safesubst/int/msg/raw magic prefixes."""
    code = mwp.parse(wikitext)
    seen: dict[str, None] = {}
    for tmpl in code.filter_templates(recursive=True):
        name = str(tmpl.name).strip()
        if not name or name.startswith("#"):
            continue
        prefix = name.split(":", 1)[0].strip().lower()
        if prefix in ("safesubst", "subst", "int", "msg", "msgnw", "raw"):
            continue
        seen.setdefault(name, None)
    return list(seen.keys())


_TEMPLATE_PARAM_RE = re.compile(r"\{\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*[|}]")


def extract_template_params(template_wikitext: str) -> list[str]:
    """Unique named {{{param}}} placeholders declared in a template's own
    source (first-seen order). Purely numeric positional params (1, 2, ...)
    are skipped since they're positional, not named fields to report."""
    seen: dict[str, None] = {}
    for m in _TEMPLATE_PARAM_RE.finditer(template_wikitext):
        name = m.group(1)
        if name.isdigit():
            continue
        seen.setdefault(name, None)
    return list(seen.keys())


def _extract_ill_renderings(translated_text: str) -> dict[str, str]:
    """{{ill|Display|en|Source Title}} (or the lt=-free shorthand
    {{ill|Display|en}}, where Display doubles as the source title) states,
    in the model's own output, exactly how it chose to render one specific
    unconfirmed source-language term — first-seen order, source title ->
    display text."""
    code = mwp.parse(translated_text)
    renderings: dict[str, str] = {}
    for tmpl in code.filter_templates(recursive=True):
        if str(tmpl.name).strip().lower() != "ill":
            continue
        if not tmpl.has(1) or not tmpl.has(2):
            continue
        display = str(tmpl.get(1).value).strip()
        lang = str(tmpl.get(2).value).strip().lower()
        if lang != "en" or not display:
            continue
        title = str(tmpl.get(3).value).strip() if tmpl.has(3) else ""
        title = title or display
        renderings.setdefault(title, display)
    return renderings


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _normalize_title_case(title: str) -> str:
    """MediaWiki auto-capitalizes only the first letter of a title (default
    config); Wikidata's sitelink API requires an exact match against that
    canonical form. Wikitext casually written in lowercase — a common
    convention for template calls like `{{reflist}}` — would otherwise query
    the wrong string and come back as a false 'not found'."""
    if not title:
        return title
    return title[0].upper() + title[1:]


class WikidataVerifier:
    """Batched enwiki<->sqwiki (or any lang pair) sitelink checks."""

    def __init__(self, http_client: httpx.AsyncClient, source_lang: str, target_lang: str):
        self._client = http_client
        self.source_site = f"{source_lang}wiki"
        self.target_site = f"{target_lang}wiki"

    async def check_sitelinks(self, titles: list[str]) -> dict[str, str | None]:
        """titles -> confirmed target-wiki title, or None if unresolved.
        Result keys match the exact casing passed in, even though lookups
        are normalized internally (see _normalize_title_case)."""
        unique_titles = list(dict.fromkeys(titles))
        normalized_to_original: dict[str, str] = {}
        for original in unique_titles:
            normalized_to_original.setdefault(_normalize_title_case(original), original)
        normalized_titles = list(normalized_to_original.keys())

        by_normalized: dict[str, str | None] = {}
        for batch in _chunked(normalized_titles, _BATCH_SIZE):
            if not batch:
                continue
            params = {
                "action": "wbgetentities",
                "sites": self.source_site,
                "titles": "|".join(batch),
                "props": "sitelinks",
                "sitefilter": f"{self.source_site}|{self.target_site}",
                "format": "json",
                "formatversion": "2",
            }
            resp = await asyncio.wait_for(
                self._client.get(WIKIDATA_API, params=params), timeout=_WIKIDATA_HARD_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            for entity in data.get("entities", {}).values():
                if "missing" in entity:
                    by_normalized[entity["title"]] = None
                    continue
                sitelinks = entity.get("sitelinks", {})
                source_link = sitelinks.get(self.source_site)
                if not source_link:
                    continue
                target_link = sitelinks.get(self.target_site)
                by_normalized[source_link["title"]] = target_link["title"] if target_link else None

        results: dict[str, str | None] = {}
        for original in unique_titles:
            norm = _normalize_title_case(original)
            if norm in by_normalized:
                results[original] = by_normalized[norm]
        return results

    async def check_template_sitelinks(self, template_names: list[str]) -> dict[str, str | None]:
        """Same as check_sitelinks but for the Template: namespace, keyed by
        the *bare* source template name. Values keep the target wiki's own
        namespace prefix (e.g. "Stampa:Infobox royalty" on sqwiki) since
        that's needed to fetch the template's source — strip it separately
        (see strip_namespace) only when it needs to be shown to the model,
        since `{{Name}}` in wikitext auto-resolves to the Template namespace."""
        # Normalize before prefixing: check_sitelinks only capitalizes the
        # first character of the *whole* string it's given, which is a
        # no-op on "Template:reflist" (already starts with capital T) and
        # would leave "reflist" itself lowercase for the Wikidata query.
        name_to_prefixed = {name: f"Template:{_normalize_title_case(name)}" for name in template_names}
        prefixed_results = await self.check_sitelinks(list(name_to_prefixed.values()))
        results: dict[str, str | None] = {}
        for name, prefixed in name_to_prefixed.items():
            results[name] = prefixed_results.get(prefixed)
        return results

    async def count_other_language_sitelinks(self, titles: list[str]) -> dict[str, int]:
        """titles -> how many Wikipedia language editions (any language,
        not just source/target) have some article for this Wikidata item.
        Unlike check_sitelinks, this queries without a sitefilter — a rough
        but genuinely mechanical proxy for "is this a distinct,
        cross-culturally notable topic" vs "too generic/minor to bother
        linking", used only for titles that already came back with no
        target-wiki sitelink (see verify_wikitext)."""
        unique_titles = list(dict.fromkeys(titles))
        normalized_to_original: dict[str, str] = {}
        for original in unique_titles:
            normalized_to_original.setdefault(_normalize_title_case(original), original)
        normalized_titles = list(normalized_to_original.keys())

        by_normalized: dict[str, int] = {}
        for batch in _chunked(normalized_titles, _BATCH_SIZE):
            if not batch:
                continue
            params = {
                "action": "wbgetentities",
                "sites": self.source_site,
                "titles": "|".join(batch),
                "props": "sitelinks",
                "format": "json",
                "formatversion": "2",
            }
            resp = await asyncio.wait_for(
                self._client.get(WIKIDATA_API, params=params), timeout=_WIKIDATA_HARD_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            for entity in data.get("entities", {}).values():
                if "missing" in entity:
                    by_normalized[entity["title"]] = 0
                    continue
                sitelinks = entity.get("sitelinks", {})
                source_link = sitelinks.get(self.source_site)
                if not source_link:
                    continue
                wiki_editions = sum(1 for site in sitelinks if site.endswith("wiki") and site != self.source_site)
                by_normalized[source_link["title"]] = wiki_editions

        results: dict[str, int] = {}
        for original in unique_titles:
            norm = _normalize_title_case(original)
            if norm in by_normalized:
                results[original] = by_normalized[norm]
        return results


def strip_namespace(title: str) -> str:
    """'Stampa:Infobox royalty' -> 'Infobox royalty'. Wikitext template calls
    resolve bare names into the Template namespace automatically, so this is
    what a model should actually write in `{{...}}`."""
    if ":" in title:
        return title.split(":", 1)[1]
    return title


_TRAILING_DIGITS_RE = re.compile(r"\d+$")


def _collapse_numbered_params(params: list[str], max_count: int) -> list[str]:
    """Infobox templates often support numbered variants of the same field
    (succession1, succession2, ...). Collapsing to the base name keeps the
    reported list focused and short instead of dozens of near-duplicates."""
    seen: dict[str, None] = {}
    for p in params:
        base = _TRAILING_DIGITS_RE.sub("", p)
        seen.setdefault(base, None)
        if len(seen) >= max_count:
            break
    return list(seen.keys())


@dataclass
class VerifiedFacts:
    """Article-level verification results, gathered once and reused across
    all of that article's chunks."""

    links: dict[str, str | None] = field(default_factory=dict)
    templates: dict[str, str | None] = field(default_factory=dict)
    template_params: dict[str, list[str]] = field(default_factory=dict)  # keyed by source template name
    # Set when the article itself already has a target-wiki sitelink — this
    # is a rewrite, not a first translation (see the skill's own "Check
    # whether the sq article already exists" section).
    existing_target_title: str | None = None
    existing_target_bytes: int | None = None
    # That existing article's own outgoing wikilink targets — already in
    # the target wiki's established form for this exact topic cluster, so
    # real terminology/naming to reuse rather than re-decide from scratch.
    sibling_links: list[str] = field(default_factory=list)
    # For links with no target-wiki sitelink: how many Wikipedia language
    # editions (any language) have some article for the concept — a
    # mechanical proxy for "distinct, cross-culturally notable topic" vs
    # "too generic to bother linking" (e.g. a common noun like "egg").
    not_found_link_language_counts: dict[str, int] = field(default_factory=dict)
    # Compact, growing article-level "translation decisions" registry: for a
    # source link target with no confirmed target-wiki sitelink, the model
    # itself still has to render it somehow (typically {{ill|display|en|
    # Title}}) — the first chunk to translate that term records its own
    # choice here (see record_established_renderings), and every later
    # chunk containing the same source title is told to reuse it. This is
    # not the harness inventing a translation — it only relays a decision
    # the model already made elsewhere in this same article, keyed by
    # source title, mutated in place as chunks complete so later chunks
    # (dispatched after an earlier one frees a worker slot) benefit from it.
    established_renderings: dict[str, str] = field(default_factory=dict)
    # Source term -> the one rendering every chunk must use, fixed before
    # translation starts (see glossary.build_article_glossary).
    glossary: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.links and not self.templates and not self.existing_target_title

    def record_established_renderings(self, translated_text: str) -> None:
        """Scans one chunk's own translated output for {{ill|display|en|
        Title}} calls and records the display text chosen for each source
        Title, first occurrence wins. Mechanical extraction only — it reads
        back what the model already wrote, the same way build_verified_facts_block
        only ever states what already exists."""
        for title, display in _extract_ill_renderings(translated_text).items():
            self.established_renderings.setdefault(title, display)


def existing_article_matches_by_length(source_wikitext: str, facts: VerifiedFacts) -> bool:
    """Return whether an existing target article is plausibly this article.

    This is a conservative duplicate guard, not a semantic comparison: it is
    only used when the target page is already identified by a Wikidata sitelink
    or a direct target-wiki title/redirect lookup.
    """
    if facts.existing_target_title is None or facts.existing_target_bytes is None:
        return False
    source_bytes = len(source_wikitext.encode("utf-8"))
    if source_bytes == 0:
        return False
    ratio = facts.existing_target_bytes / source_bytes
    return _SAME_ARTICLE_MIN_RATIO <= ratio <= _SAME_ARTICLE_MAX_RATIO


async def _resolve_cached_then_live(
    titles: list[str],
    kind: str,
    source_lang: str,
    target_lang: str,
    cache: VerificationCache | None,
    live_check,
) -> dict[str, str | None]:
    if not titles:
        return {}
    cached = cache.get_many(source_lang, target_lang, kind, titles) if cache else {}
    to_fetch = [t for t in titles if t not in cached]
    fresh: dict[str, str | None] = {}
    if to_fetch:
        fresh = await live_check(to_fetch)
        if cache:
            cache.set_many(source_lang, target_lang, kind, fresh)
    return {**cached, **fresh}


async def _check_existing_article(
    article_title: str,
    source_lang: str,
    target_lang: str,
    verifier: WikidataVerifier,
    target_mw_client: MediaWikiClient | None,
    cache: VerificationCache | None,
) -> tuple[str | None, list[str], int | None]:
    """Checks whether the article itself already has a target-wiki sitelink
    (a rewrite, not a first translation — see the skill's "Check whether the
    sq article already exists" section) and, if so, fetches that existing
    article's own wikilinks: real, already-established target-wiki
    terminology for this exact topic, not a guess."""
    cached = cache.get_sibling_links(source_lang, target_lang, article_title) if cache is not None else None
    if cached is not None:
        existing_target_title, sibling_links = cached
    else:
        existing_target_title = (await verifier.check_sitelinks([article_title])).get(article_title)
        sibling_links = []

    # Wikidata sitelinks are not complete.  Check the target wiki directly as
    # well, following redirects, so an existing same-title article is caught
    # even when its Wikidata item is missing or incomplete.
    if target_mw_client is not None:
        try:
            direct_target = await target_mw_client.fetch_article(article_title)
            existing_target_title = direct_target.title
        except MediaWikiError:
            pass

    existing_target_bytes: int | None = None
    if existing_target_title and target_mw_client is not None:
        try:
            existing_source = await target_mw_client.fetch_article(existing_target_title)
            existing_target_bytes = len(existing_source.wikitext.encode("utf-8"))
            sibling_links = extract_link_targets(existing_source.wikitext)[:_MAX_SIBLING_LINKS_REPORTED]
        except MediaWikiError:
            pass

    if cache is not None:
        cache.set_sibling_links(source_lang, target_lang, article_title, existing_target_title, sibling_links)
    return existing_target_title, sibling_links, existing_target_bytes


async def verify_wikitext(
    article_title: str,
    wikitext: str,
    source_lang: str,
    target_lang: str,
    wikidata_client: httpx.AsyncClient,
    target_mw_client: MediaWikiClient | None,
    cache: VerificationCache | None,
) -> VerifiedFacts:
    """Mechanical fact-gathering only: extracts link/template targets from
    source wikitext and resolves them against Wikidata + the target wiki,
    persisting results for reuse across the batch and future runs. No
    translation judgment happens here — only what already exists is checked."""
    link_targets = extract_link_targets(wikitext)
    template_names = extract_template_names(wikitext)

    verifier = WikidataVerifier(wikidata_client, source_lang, target_lang)

    links = await _resolve_cached_then_live(
        link_targets, "link", source_lang, target_lang, cache, verifier.check_sitelinks
    )
    templates = await _resolve_cached_then_live(
        template_names, "template", source_lang, target_lang, cache, verifier.check_template_sitelinks
    )
    # Wikidata only knows a template when its item has a sitelink to the
    # target wiki, but sqwiki carries many English-named templates (Stampa:
    # Infobox planet) without one. Told "NOT FOUND", the model drops or
    # rebuilds them as tables, so ask the target wiki itself.
    unresolved_templates = [name for name, target in templates.items() if target is None]
    if unresolved_templates and target_mw_client is not None:
        try:
            on_target = await target_mw_client.existing_templates(unresolved_templates)
        except Exception as exc:
            logger.warning("Target-wiki template lookup failed for %r: %s", article_title, exc)
            on_target = {}
        templates.update({name: target for name, target in on_target.items() if target})
    existing_target_title, sibling_links, existing_target_bytes = await _check_existing_article(
        article_title, source_lang, target_lang, verifier, target_mw_client, cache
    )

    not_found_links = [title for title, target in links.items() if target is None]
    not_found_counts: dict[str, int] = {}
    if not_found_links:
        cached_counts = cache.get_sitelink_counts(source_lang, not_found_links) if cache else {}
        to_check = [t for t in not_found_links if t not in cached_counts]
        fresh_counts: dict[str, int] = {}
        if to_check:
            fresh_counts = await verifier.count_other_language_sitelinks(to_check)
            if cache:
                cache.set_sitelink_counts(source_lang, fresh_counts)
        not_found_counts = {**cached_counts, **fresh_counts}

    template_params: dict[str, list[str]] = {}
    if target_mw_client is not None:
        for source_name, target_full_title in templates.items():
            if target_full_title is None or not _INFOBOX_NAME_RE.match(source_name):
                continue
            cached_params = (
                cache.get_template_params(source_lang, target_lang, source_name) if cache else None
            )
            if cached_params is not None:
                if cached_params:
                    template_params[source_name] = cached_params
                continue
            try:
                target_source = await target_mw_client.fetch_article(target_full_title)
            except MediaWikiError:
                if cache:
                    cache.set_template_params(source_lang, target_lang, source_name, target_full_title, [])
                continue
            params = _collapse_numbered_params(
                extract_template_params(target_source.wikitext), _MAX_TEMPLATE_PARAMS_REPORTED
            )
            if cache:
                cache.set_template_params(source_lang, target_lang, source_name, target_full_title, params)
            if params:
                template_params[source_name] = params

    return VerifiedFacts(
        links=links,
        templates=templates,
        template_params=template_params,
        existing_target_title=existing_target_title,
        existing_target_bytes=existing_target_bytes,
        sibling_links=sibling_links,
        not_found_link_language_counts=not_found_counts,
    )


def build_verified_facts_block(chunk_text: str, facts: VerifiedFacts) -> str:
    """Compact, chunk-scoped subset of the article's already-verified facts,
    formatted as plain data for the skill invocation's user message. This is
    data, not translation guidance — it states what does/doesn't exist on
    the target wiki, not how to phrase or grammar-check anything."""
    chunk_links = extract_link_targets(chunk_text)
    chunk_templates = extract_template_names(chunk_text)

    link_lines = []
    for title in chunk_links:
        if title not in facts.links:
            continue
        target = facts.links[title]
        if target:
            link_lines.append(f"- [[{title}]] -> confirmed as [[{target}]]")
        else:
            established = facts.established_renderings.get(title)
            if established:
                link_lines.append(
                    f"- [[{title}]] -> NOT FOUND on target wiki; already rendered elsewhere in "
                    f"this article as {{{{ill|{established}|en|{title}}}}} — reuse that exact "
                    "rendering here instead of choosing new wording"
                )
                continue
            count = facts.not_found_link_language_counts.get(title)
            if count is not None:
                note = (
                    f"has articles in {count} other Wikipedia language(s) — likely a real, "
                    "distinct topic"
                    if count > 0
                    else "has no article in any other Wikipedia language either — likely not "
                    "a distinct encyclopedic topic in this context"
                )
                link_lines.append(f"- [[{title}]] -> NOT FOUND on target wiki ({note})")
            else:
                link_lines.append(f"- [[{title}]] -> NOT FOUND on target wiki")

    template_lines = []
    has_citation_template = False
    for name in chunk_templates:
        if name.strip().lower() in CITATION_TEMPLATE_NAMES:
            has_citation_template = True
        if name not in facts.templates:
            continue
        target = facts.templates[name]
        if target:
            bare = strip_namespace(target)
            line = f"- {{{{{name}}}}} -> confirmed present as {{{{{bare}}}}}"
            params = facts.template_params.get(name)
            if params:
                line += (
                    "; confirmed real parameter names (use these exactly — do not "
                    "invent or translate them): " + ", ".join(params)
                )
            template_lines.append(line)
        else:
            template_lines.append(f"- {{{{{name}}}}} -> NOT FOUND on target wiki")

    parts: list[str] = []

    if facts.existing_target_title:
        sibling_note = (
            f"This article already exists on the target wiki as '{facts.existing_target_title}' "
            "— this is a rewrite, not a first translation. Use that established title/spelling "
            "rather than a fresh transliteration where they'd otherwise differ."
        )
        if facts.sibling_links:
            sibling_note += (
                " Wikilink targets already used in that existing article (real target-wiki "
                "terminology for this exact topic — reuse the matching ones instead of "
                "re-deciding a phrasing from scratch): " + ", ".join(facts.sibling_links)
            )
        parts.append(sibling_note)

    if has_citation_template:
        parts.append(_CITATION_PARAM_NAMES_NOTE)

    glossary_block = [f"- {term} -> {rendering}" for term, rendering in facts.glossary.items() if term in chunk_text]
    if glossary_block:
        parts.append(
            "Article glossary (names that recur across this article's sections; the rendering "
            "is fixed — use exactly this spelling every time, adding only grammatical endings):\n"
            + "\n".join(glossary_block)
        )

    if not link_lines and not template_lines and not parts:
        return ""

    if link_lines or template_lines:
        detail = [
            "Pre-verified facts about links/templates in this section (already "
            "checked by the harness against Wikidata/the target wiki — use these "
            "instead of guessing):"
        ]
        if link_lines:
            detail.append("Links:\n" + "\n".join(link_lines))
        if template_lines:
            detail.append("Templates:\n" + "\n".join(template_lines))
        parts.append("\n\n".join(detail))

    return "\n\n".join(parts)
