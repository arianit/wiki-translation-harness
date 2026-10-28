from types import SimpleNamespace

import pytest

from wiki_translation_harness import glossary as glossary_mod
from wiki_translation_harness.glossary import (
    build_article_glossary,
    parse_glossary,
    recurring_terms,
)
from wiki_translation_harness.models import Chunk
from wiki_translation_harness.verification import VerifiedFacts, build_verified_facts_block


def _chunk(order: int, text: str) -> Chunk:
    return Chunk(article_title="Art", section_titles=[f"S{order}"], order=order, text=text, token_estimate=10, source_lang="it")


def test_recurring_terms_skips_single_use_confirmed_and_digits():
    chunks = [
        _chunk(0, "[[Santa Maria Assunta]] [[Roma]] [[1470]] [[Solo qui]]"),
        _chunk(1, "[[Santa Maria Assunta]] [[Roma]] [[1470]]"),
        _chunk(2, "[[Santa Maria Assunta]]"),
    ]
    facts = VerifiedFacts(links={"Roma": "Roma", "Santa Maria Assunta": None})
    assert recurring_terms(chunks, facts) == ["Santa Maria Assunta"]


def test_parse_glossary_keeps_only_asked_for_string_entries():
    reply = 'Here you go: {"A": "Ai", "B": "", "C": 3, "X": "nope"}'
    assert parse_glossary(reply, ["A", "B", "C"]) == {"A": "Ai"}
    assert parse_glossary("no json", ["A"]) == {}
    assert parse_glossary("{broken", ["A"]) == {}


def test_facts_block_lists_only_glossary_terms_present_in_chunk():
    facts = VerifiedFacts(glossary={"Santa Maria": "Shën Maria", "Vallja": "Vallja"})
    block = build_verified_facts_block("Text on [[Santa Maria]] only.", facts)
    assert "- Santa Maria -> Shën Maria" in block
    assert "Vallja" not in block


@pytest.mark.asyncio
async def test_build_article_glossary_fills_facts_and_caches(monkeypatch):
    calls = []

    async def fake_completion(client, model, messages, temperature, pricing, on_retry=None):
        calls.append(messages)
        return SimpleNamespace(text='{"Santa Maria": "Shën Maria"}')

    class FakeCache:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, model, src, tgt, source, translated):
            self.data[key] = translated

    monkeypatch.setattr(glossary_mod, "run_completion", fake_completion)
    chunks = [_chunk(0, "[[Santa Maria]]"), _chunk(1, "[[Santa Maria]]")]
    config = SimpleNamespace(target_lang="sq", temperature=0.0)
    skill = SimpleNamespace(combined="skill", content_hash="h")
    stats = SimpleNamespace(record_usage=lambda m, r: None)
    cache = FakeCache()

    facts = VerifiedFacts()
    assert await build_article_glossary(chunks, facts, config, None, "m", skill, cache, None, stats) == 1
    assert facts.glossary == {"Santa Maria": "Shën Maria"}

    again = VerifiedFacts()
    await build_article_glossary(chunks, again, config, None, "m", skill, cache, None, stats)
    assert again.glossary == facts.glossary and len(calls) == 1  # second run hit the cache
