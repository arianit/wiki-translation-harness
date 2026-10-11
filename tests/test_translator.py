from pathlib import Path

import pytest

from wiki_translation_harness.cache import TranslationCache
from wiki_translation_harness.models import Chunk, ChunkStatus, Config, RunStats
from wiki_translation_harness.skill_loader import SkillContent
from wiki_translation_harness.translator import translate_chunk
from wiki_translation_harness.verification import VerifiedFacts


class FakeOpenRouterClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat_completion(self, model, messages, temperature=0.0, on_retry=None, usage_out=None):
        self.calls.append(messages)
        text = self.responses.pop(0)
        return text, 100, 50


def _skill() -> SkillContent:
    return SkillContent(skill_md="Translate faithfully.", reference_texts={})


def _config(**overrides) -> Config:
    base = dict(
        model="test-model",
        source_lang="en",
        target_lang="sq",
        openrouter_api_key="sk-test",
        cache=True,
        validate=True,
        repair=True,
        max_repair_attempts=2,
    )
    base.update(overrides)
    return Config.model_validate(base)


def _chunk(text="'''Paris''' is a city.", source_lang="en") -> Chunk:
    return Chunk(
        article_title="Paris",
        section_titles=["Lead"],
        order=0,
        text=text,
        token_estimate=10,
        source_lang=source_lang,
    )


@pytest.mark.asyncio
async def test_translate_valid_output_no_repair(tmp_path: Path):
    client = FakeOpenRouterClient(["'''Parisi''' është qytet."])
    cache = TranslationCache(tmp_path / "c.sqlite3")
    stats = RunStats()
    outcome = await translate_chunk(_chunk(), _config(), client, _skill(), cache, None, stats)

    assert outcome.validation.valid
    assert outcome.repair_attempts == 0
    assert outcome.chunk.status == ChunkStatus.TRANSLATED
    assert stats.sections_translated == 1
    assert stats.cache_misses == 1
    assert stats.tokens_in == 100
    assert stats.tokens_out == 50
    assert stats.model_usage["test-model"].tokens_in == 100
    assert stats.model_usage["test-model"].tokens_out == 50
    assert stats.model_usage["test-model"].calls == 1
    cache.close()


@pytest.mark.asyncio
async def test_translate_cache_hit_skips_api_call(tmp_path: Path):
    cache = TranslationCache(tmp_path / "c.sqlite3")
    client = FakeOpenRouterClient(["'''Parisi''' është qytet."])
    stats = RunStats()
    chunk = _chunk()
    await translate_chunk(chunk, _config(), client, _skill(), cache, None, stats)

    # second call with identical input+model+langs must hit cache, not call the API again
    client2 = FakeOpenRouterClient([])  # would raise IndexError if called
    stats2 = RunStats()
    outcome2 = await translate_chunk(_chunk(), _config(), client2, _skill(), cache, None, stats2)

    assert outcome2.from_cache
    assert stats2.cache_hits == 1
    assert stats2.cache_misses == 0
    assert outcome2.chunk.translated_text == "'''Parisi''' është qytet."
    cache.close()


@pytest.mark.asyncio
async def test_invalid_output_triggers_repair_and_succeeds(tmp_path: Path):
    client = FakeOpenRouterClient(
        [
            "[[broken link translation",  # initial translation: invalid
            "'''Parisi''' është qytet me [[Lidhje|lidhje]].",  # repair: valid
        ]
    )
    cache = TranslationCache(tmp_path / "c.sqlite3")
    stats = RunStats()
    outcome = await translate_chunk(_chunk(), _config(), client, _skill(), cache, None, stats)

    assert outcome.validation.valid
    assert outcome.repair_attempts == 1
    assert outcome.chunk.status == ChunkStatus.REPAIRED
    assert stats.repair_attempts == 1
    assert stats.repairs_succeeded == 1
    # repair call's messages should carry the validation errors
    repair_call_content = client.calls[-1][-1]["content"]
    assert "link" in repair_call_content.lower()
    cache.close()


@pytest.mark.asyncio
async def test_repair_exhausted_marks_failed_and_does_not_cache(tmp_path: Path):
    client = FakeOpenRouterClient(
        [
            "[[broken 1",
            "[[broken 2",
            "[[broken 3",
        ]
    )
    cache = TranslationCache(tmp_path / "c.sqlite3")
    stats = RunStats()
    outcome = await translate_chunk(
        _chunk(), _config(max_repair_attempts=2), client, _skill(), cache, None, stats
    )

    assert not outcome.validation.valid
    assert outcome.chunk.status == ChunkStatus.FAILED
    assert stats.repairs_failed == 1
    assert stats.validation_failures == 1

    # a failed translation must not be cached, so a rerun retries it
    from wiki_translation_harness.cache import compute_key

    key = compute_key("test-model", "en", "sq", _chunk().text, _skill().content_hash)
    assert cache.get(key) is None
    cache.close()


@pytest.mark.asyncio
async def test_later_chunk_is_told_about_earlier_chunks_established_rendering(tmp_path: Path):
    cache = TranslationCache(tmp_path / "c.sqlite3")
    stats = RunStats()
    facts = VerifiedFacts(links={"Sustainable architecture": None})

    first_client = FakeOpenRouterClient(
        ["Shih {{ill|arkitektura e qëndrueshme|en|Sustainable architecture}}."]
    )
    await translate_chunk(
        _chunk(text="See [[Sustainable architecture]]."),
        _config(),
        first_client,
        _skill(),
        cache,
        None,
        stats,
        verified_facts=facts,
    )

    second_client = FakeOpenRouterClient(["ok"])
    await translate_chunk(
        _chunk(text="Also [[Sustainable architecture]] again."),
        _config(),
        second_client,
        _skill(),
        cache,
        None,
        stats,
        verified_facts=facts,
    )
    second_user_message = second_client.calls[-1][-1]["content"]
    assert "arkitektura e qëndrueshme" in second_user_message
    assert "reuse that exact rendering" in second_user_message
    cache.close()


@pytest.mark.asyncio
async def test_per_chunk_source_lang_isolates_cache(tmp_path: Path):
    cache = TranslationCache(tmp_path / "c.sqlite3")
    client_en = FakeOpenRouterClient(["English-sourced translation"])
    stats1 = RunStats()
    await translate_chunk(_chunk(source_lang="en"), _config(), client_en, _skill(), cache, None, stats1)

    # same text, same model, different source_lang -> must not hit the "en" cache entry
    client_sq = FakeOpenRouterClient(["Albanian-sourced translation"])
    stats2 = RunStats()
    outcome = await translate_chunk(
        _chunk(source_lang="sq"), _config(), client_sq, _skill(), cache, None, stats2
    )
    assert not outcome.from_cache
    assert outcome.chunk.translated_text == "Albanian-sourced translation"
    cache.close()


def _complex_chunk(text="{{Infobox settlement}}") -> Chunk:
    return Chunk(
        article_title="Paris",
        section_titles=["Infobox"],
        order=0,
        text=text,
        token_estimate=10,
        source_lang="en",
        is_complex=True,
    )


@pytest.mark.asyncio
async def test_complex_chunk_routes_to_distinct_complex_client_when_given():
    draft_client = FakeOpenRouterClient([])  # would raise IndexError if called
    complex_client = FakeOpenRouterClient(["{{Infobox settlement}}\n'''Parisi''' është qytet."])
    stats = RunStats()
    outcome = await translate_chunk(
        _complex_chunk(),
        _config(complex_model="strong-model"),
        draft_client,
        _skill(),
        None,
        None,
        stats,
        complex_client=complex_client,
    )
    assert outcome.validation.valid
    assert draft_client.calls == []
    assert len(complex_client.calls) == 1
    # model_usage is keyed by model id, not client -- confirms spend is
    # attributed to "strong-model" (config.complex_model), not "test-model"
    # (config.model), even though both share this test's draft_client type.
    assert "strong-model" in stats.model_usage
    assert "test-model" not in stats.model_usage


@pytest.mark.asyncio
async def test_complex_chunk_repair_also_uses_complex_client():
    draft_client = FakeOpenRouterClient([])  # would raise IndexError if called
    complex_client = FakeOpenRouterClient(
        # First response uses {{harvc}}, which validator.py flags unconditionally
        # (broken on sqwiki) -> triggers a repair round; second is clean.
        ["{{Infobox settlement}}\nBibliografia.\n{{harvc|last=Smith|c=Ch1}}\n", "{{Infobox settlement}}\nBibliografia.\n{{Cite book|last=Smith}}\n"]
    )
    stats = RunStats()
    outcome = await translate_chunk(
        _complex_chunk(),
        _config(complex_model="strong-model", max_repair_attempts=1),
        draft_client,
        _skill(),
        None,
        None,
        stats,
        complex_client=complex_client,
    )
    assert draft_client.calls == []
    assert len(complex_client.calls) == 2
    assert outcome.repair_attempts == 1
    assert outcome.validation.valid

@pytest.mark.asyncio
async def test_invalid_cache_is_repaired_and_replaced(tmp_path):
    from wiki_translation_harness.cache import compute_key
    with_cache = TranslationCache(tmp_path / 'c.sqlite3')
    chunk = _chunk()
    cfg, skill = _config(), _skill()
    key = compute_key(cfg.model, chunk.source_lang, cfg.target_lang, chunk.text, skill.content_hash, '')
    with_cache.set(key, cfg.model, chunk.source_lang, cfg.target_lang, chunk.text, '[[Broken')
    client = FakeOpenRouterClient(['[[Parisi]] është qytet.'])
    stats = RunStats()
    try:
        result = await translate_chunk(chunk, cfg, client, skill, with_cache, None, stats)
        assert result.validation.valid
        assert result.chunk.status == ChunkStatus.REPAIRED
        assert result.repair_attempts == 1
        assert stats.cache_hits == 1
        assert stats.cache_misses == 0
        assert len(client.calls) == 1
        assert with_cache.get(key) == '[[Parisi]] është qytet.'
        assert result.prompt_tokens == 100
    finally:
        with_cache.close()


def test_reference_definitions_only_detection():
    from wiki_translation_harness.translator import (
        is_reference_definitions_only,
        split_heading_and_reference_definitions,
    )

    defs = '\n<ref name="a">{{cite web |url=x}}</ref>\n<!-- note -->\n<ref name="b">\n{{cite book |title=y}}\n</ref>\n</references>\n'
    assert is_reference_definitions_only(defs)
    assert is_reference_definitions_only('<references>' + defs)
    assert not is_reference_definitions_only("Some prose.\n" + defs)
    assert not is_reference_definitions_only("== References ==\n" + defs)
    assert not is_reference_definitions_only("No refs here")

    heading, rest = split_heading_and_reference_definitions("== References ==\n<references>" + defs)
    assert heading == "== References =="
    assert rest.startswith("\n<references>")
    assert split_heading_and_reference_definitions("Prose\n" + defs) is None


def test_keep_source_reference_block_overrides_rewritten_body():
    from wiki_translation_harness.translator import keep_source_reference_block

    source = '== References ==\n<references>\n<ref name="a">{{cite web |url=x}}</ref>\n'
    mangled = "== Referime ==\n* {{cite web |url=x}}\n"
    assert keep_source_reference_block(source, mangled) == "== Referime ==\n<references>\n<ref name=\"a\">{{cite web |url=x}}</ref>\n"
    assert keep_source_reference_block(source, "no heading\nbody").startswith("== References ==\n<references>")
    assert keep_source_reference_block("Prose.", "Tekst.") == "Tekst."


def test_dropped_infobox_detected_and_restored_after_leading_templates():
    from wiki_translation_harness.translator import dropped_infoboxes, restore_dropped_infoboxes

    box = "{{Infobox planet\n| name = Mars\n| mass = {{val|6.4|e=23}}\n}}"
    source = "{{Short description|Planet}}\n" + box + "\n[[File:A.png|thumb|x]]\n'''Mars''' is a planet.\n"
    dropped = "{{Short description|Planeti}}\n\n[[File:A.png|thumb|x]]\n'''Marsi''' është planet.\n"
    kept = "{{Short description|Planeti}}\n{{Infobox planet\n| name = Marsi\n}}\n'''Marsi''' është planet.\n"

    assert dropped_infoboxes(source, dropped) == [box]
    assert dropped_infoboxes(source, kept) == []
    assert dropped_infoboxes("No infobox here.", dropped) == []

    restored, n = restore_dropped_infoboxes(source, dropped)
    assert n == 1
    assert restored.startswith("{{Short description|Planeti}}\n" + box)
    assert restored.index(box) < restored.index("[[File:A.png")
    assert dropped_infoboxes(source, restored) == []
    assert restore_dropped_infoboxes(source, kept) == (kept, 0)
