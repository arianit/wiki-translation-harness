"""Article-level glossary of recurring names, so concurrently translated
chunks render the same saint, place or organisation the same way.

Chunks are translated in parallel, so the {{ill}} relay in
VerifiedFacts.record_established_renderings only helps chunks dispatched
after another finished. The semantic review then flags the leftovers as
entity_consistency ("Shën Mërisë" vs "Shën Marisë", "Montalto-s" vs
"Montalto"). One cheap call before translation fixes a rendering for every
recurring name the target wiki has no article for; build_verified_facts_block
hands each chunk the entries it contains.
"""

from __future__ import annotations

import hashlib
import logging
import re

import orjson

from wiki_translation_harness.cache import TranslationCache, compute_key
from wiki_translation_harness.models import Chunk, Config, ModelPricing, RunStats
from wiki_translation_harness.openrouter import RetryCallback, run_completion
from wiki_translation_harness.skill_loader import SkillContent
from wiki_translation_harness.verification import VerifiedFacts, extract_link_targets

logger = logging.getLogger(__name__)

_MAX_TERMS = 40
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = (
    "You fix one canonical rendering per recurring name before an article is translated in "
    "separate parts. Follow the conventions of the translation skill below (target-wiki "
    "spelling of names, transliteration, saint and place-name forms). Give the base "
    "(nominative, indefinite) form of each name; grammatical endings are added in context. "
    "Reply with a single JSON object mapping each source term to its rendering, nothing else.\n\n---\n\n"
)


def recurring_terms(chunks: list[Chunk], facts: VerifiedFacts) -> list[str]:
    """Link targets that appear in at least two chunks and have no confirmed
    target-wiki article (a confirmed one is already fixed by its link target),
    most widespread first."""
    spread: dict[str, int] = {}
    for chunk in chunks:
        for target in extract_link_targets(chunk.text):
            spread[target] = spread.get(target, 0) + 1
    terms = [
        t for t, n in spread.items()
        if n >= 2 and not facts.links.get(t) and not t.isdigit() and not t.startswith("#")
    ]
    terms.sort(key=lambda t: -spread[t])
    return terms[:_MAX_TERMS]


def parse_glossary(reply: str, terms: list[str]) -> dict[str, str]:
    """Entries of the model's JSON reply for the asked-for terms; {} when the
    reply holds no usable object."""
    match = _JSON_OBJECT_RE.search(reply)
    if not match:
        return {}
    try:
        data = orjson.loads(match.group(0))
    except orjson.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    wanted = set(terms)
    return {
        k: v.strip() for k, v in data.items()
        if k in wanted and isinstance(v, str) and v.strip() and "\n" not in v
    }


async def build_article_glossary(
    chunks: list[Chunk],
    facts: VerifiedFacts,
    config: Config,
    client,
    model: str,
    skill: SkillContent,
    cache: TranslationCache | None,
    pricing: ModelPricing | None,
    stats: RunStats,
    on_retry: RetryCallback | None = None,
) -> int:
    """Fills facts.glossary and returns its size. Cached under the term list,
    so a resumed run reuses the glossary the finished chunks were translated
    with. A failed call leaves the glossary empty -- translation still works."""
    terms = recurring_terms(chunks, facts)
    if not terms:
        return 0
    article = chunks[0].article_title
    prompt = (
        f"Article: {article}\nSource language: {chunks[0].source_lang}\n"
        f"Target language: {config.target_lang}\n\nTerms:\n" + "\n".join(f"- {t}" for t in terms)
    )
    key = compute_key(model, chunks[0].source_lang, config.target_lang, prompt, skill.content_hash, "glossary")
    reply = cache.get(key) if cache is not None else None
    if reply is None:
        messages = [
            {"role": "system", "content": _SYSTEM + skill.combined},
            {"role": "user", "content": prompt},
        ]
        result = await run_completion(client, model, messages, config.temperature, pricing, on_retry=on_retry)
        stats.record_usage(model, result)
        reply = result.text
        if cache is not None and parse_glossary(reply, terms):
            cache.set(key, model, chunks[0].source_lang, config.target_lang, prompt, reply)
    facts.glossary = parse_glossary(reply, terms)
    logger.info("Glossary for %r: %d of %d recurring term(s)", article, len(facts.glossary), len(terms))
    return len(facts.glossary)


def glossary_lines(chunk_text: str, glossary: dict[str, str]) -> list[str]:
    return [f"- {term} -> {rendering}" for term, rendering in glossary.items() if term in chunk_text]
