"""run_assembly_repair is pipeline.py's whole-article validate/repair loop
— the counterpart to translator.translate_chunk's per-chunk loop, but
operating on the assembled article and with its own round budget
(max_assembly_repair_rounds). Tested directly rather than through the full
run_pipeline(), which would require mocking fetch/plan/cache/OpenRouter
pricing lookups unrelated to this loop's own logic.

Static (validator.py) defects are used to drive most of these tests rather
than live-API ones, since the static checks need no network and are
already covered by test_validator.py — this file's job is proving the
orchestration (round counting, chunk-targeted repair, the cap, and
independence from the per-chunk loop), not re-testing either validator.
"""

import asyncio
from pathlib import Path

import pytest

from wiki_translation_harness.cache import TranslationCache, compute_key
from wiki_translation_harness.models import Chunk, Config, RunStats
from wiki_translation_harness.pipeline import (
    drop_stray_reference_lists,
    extend_references_block_over_trailing_defs,
    merge_split_references_blocks,
    restore_orphaned_ref_definitions,
    restore_ref_group_names,
    run_assembly_repair,
    run_review_pass,
)
from wiki_translation_harness.skill_loader import SkillContent
from wiki_translation_harness.verification import VerifiedFacts


class FakeOpenRouterClient:
    """Matches test_translator.py's fake — repair_chunk (via run_completion)
    calls client.chat_completion(model, messages, temperature, on_retry=...)."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat_completion(self, model, messages, temperature=0.0, on_retry=None, usage_out=None):
        self.calls.append(messages)
        text = self.responses.pop(0)
        return text, 100, 50


class FakeMediaWikiClient:
    """Only parse_wikitext is exercised by run_assembly_repair (via
    live_validator.validate_wikitext_live)."""

    def __init__(self, responses: list[dict] | None = None, raise_if_called: bool = False):
        self.responses = list(responses or [])
        self.raise_if_called = raise_if_called
        self.calls = 0

    async def parse_wikitext(self, text: str, title: str = "API") -> dict:
        self.calls += 1
        if self.raise_if_called:
            raise AssertionError("parse_wikitext should not have been called")
        if self.responses:
            return self.responses.pop(0)
        return {"text": "<p>clean</p>", "templates": []}


def _skill() -> SkillContent:
    return SkillContent(skill_md="Translate faithfully.", reference_texts={})


def _config(**overrides) -> Config:
    base = dict(
        model="test-model",
        source_lang="en",
        target_lang="sq",
        live_validate=False,
        max_assembly_repair_rounds=3,
    )
    base.update(overrides)
    return Config.model_validate(base)


def _chunk(text: str, order: int = 0) -> Chunk:
    return Chunk(
        article_title="Test Article",
        section_titles=[f"S{order}"],
        order=order,
        text=text,
        token_estimate=10,
        translated_text=text,
    )


class _FakeSource:
    title = "Test Article"
    wikitext = "English source text."


@pytest.mark.asyncio
async def test_no_issues_needs_no_repair():
    chunk = _chunk("Prozë krejt e pastër shqipe.")
    client = FakeOpenRouterClient([])  # would raise IndexError if a repair call happened
    stats = RunStats()

    assembled, issues, rounds, _, _ = await run_assembly_repair(
        [chunk], _FakeSource(), _config(), client, _skill(), None,
        FakeMediaWikiClient(raise_if_called=True), None, stats,
    )

    assert issues == []
    assert rounds == 0
    assert client.calls == []


@pytest.mark.asyncio
async def test_resolves_within_cap():
    # {{harvc}} is a static defect (validator.py) — no network needed to detect it.
    chunk = _chunk("Bibliografia.\n{{harvc|last=Smith|c=Ch1}}\n")
    client = FakeOpenRouterClient(["Bibliografia.\n{{Cite book|last=Smith}}\n"])
    stats = RunStats()

    assembled, issues, rounds, _, _ = await run_assembly_repair(
        [chunk], _FakeSource(), _config(max_assembly_repair_rounds=3), client, _skill(), None,
        FakeMediaWikiClient(raise_if_called=True), None, stats,
    )

    assert issues == []
    assert rounds == 1
    assert "harvc" not in assembled
    assert chunk.translated_text.strip() == "Bibliografia.\n{{Cite book|last=Smith}}"
    assert stats.repair_attempts == 1
    assert stats.model_usage["test-model"].calls == 1  # config.model, per _config()'s default


@pytest.mark.asyncio
async def test_exhausts_cap_returns_remaining_issues():
    chunk = _chunk("{{harvc|last=Smith}}")
    # Every repair attempt still contains {{harvc}} — never actually fixed.
    client = FakeOpenRouterClient(["{{harvc|last=Smith}} v2", "{{harvc|last=Smith}} v3"])
    stats = RunStats()

    assembled, issues, rounds, _, _ = await run_assembly_repair(
        [chunk], _FakeSource(), _config(max_assembly_repair_rounds=2), client, _skill(), None,
        FakeMediaWikiClient(raise_if_called=True), None, stats,
    )

    assert rounds == 2
    assert any(i.kind == "harvc_used" for i in issues)
    assert len(client.calls) == 2
    assert stats.repair_attempts == 2


@pytest.mark.asyncio
async def test_only_the_affected_chunk_gets_repaired():
    broken = _chunk("{{harvc|last=Smith}}", order=0)
    clean = _chunk("Prozë krejt e pastër.", order=1)
    client = FakeOpenRouterClient(["{{Cite book|last=Smith}}"])
    stats = RunStats()

    await run_assembly_repair(
        [broken, clean], _FakeSource(), _config(max_assembly_repair_rounds=3), client, _skill(), None,
        FakeMediaWikiClient(raise_if_called=True), None, stats,
    )

    assert len(client.calls) == 1  # only the broken chunk's repair call
    assert clean.translated_text == "Prozë krejt e pastër."  # untouched


@pytest.mark.asyncio
async def test_live_validate_enabled_issue_repairs_via_live_check():
    chunk = _chunk("{{NonExistentTemplateXYZ}}")
    # First parse: reports the template as missing (drives one repair round).
    # Second parse (post-repair): clean.
    mw_client = FakeMediaWikiClient(
        responses=[
            {"text": "<p>irrelevant</p>", "templates": [{"ns": 10, "title": "Stampa:NonExistentTemplateXYZ", "exists": False}]},
            {"text": "<p>clean</p>", "templates": [{"ns": 10, "title": "Stampa:Sfn", "exists": True}]},
        ]
    )
    client = FakeOpenRouterClient(["Fixed: {{Sfn|Smith|2020}}"])
    stats = RunStats()

    assembled, issues, rounds, _, _ = await run_assembly_repair(
        [chunk], _FakeSource(), _config(live_validate=True, max_assembly_repair_rounds=2),
        client, _skill(), None, mw_client, None, stats,
    )

    assert issues == []
    assert rounds == 1
    assert mw_client.calls == 2


@pytest.mark.asyncio
async def test_unlocalized_issue_reaches_every_chunk():
    # An orphaned-named-ref finding's line_number/snippet come back None
    # (see test_live_validator.py) — it can't be pinned to one chunk, so
    # every chunk should see it in its repair error list rather than the
    # issue being silently dropped from every chunk's prompt.
    a = _chunk("Chunk A text.", order=0)
    b = _chunk("Chunk B text.", order=1)
    orphaned_ref_html = (
        '<span class="error mw-ext-cite-error">Gabim citimi: Etiketë ref e pavlefshme</span>'
    )
    mw_client = FakeMediaWikiClient(
        responses=[
            {"text": orphaned_ref_html, "templates": []},
            {"text": "<p>clean</p>", "templates": []},
        ]
    )
    client = FakeOpenRouterClient(["Chunk A fixed.", "Chunk B fixed."])
    stats = RunStats()

    await run_assembly_repair(
        [a, b], _FakeSource(), _config(live_validate=True, max_assembly_repair_rounds=2),
        client, _skill(), None, mw_client, None, stats,
    )

    assert len(client.calls) == 2  # both chunks got a repair call
    for call in client.calls:
        user_message = call[-1]["content"]
        assert "Gabim citimi" in user_message


@pytest.mark.asyncio
async def test_unlocalized_ref_issue_only_reaches_chunk_mentioning_that_ref():
    # An orphaned named ref can't be pinned to a chunk by line number, but its
    # message names `<ref name="RefA"/>` — the finding should reach ONLY the
    # chunk whose text mentions that ref, not every chunk.
    clean = _chunk("Prozë krejt e pastër.", order=0)
    with_ref = _chunk("Vijazimi.\nKjo qe e dhëna.<ref name=\"RefA\"/>\n", order=1)
    orphaned_ref_html = (
        '<span class="error mw-ext-cite-error">Gabim citimi: Etiketë &lt;ref&gt; e '
        'pavlefshme;\nasnjë tekst nuk u dha për refs e quajtura "RefA"</span>'
    )
    mw_client = FakeMediaWikiClient(
        responses=[
            {"text": orphaned_ref_html, "templates": []},
            {"text": "<p>clean</p>", "templates": []},
        ]
    )
    client = FakeOpenRouterClient(["Chunk with ref fixed."])
    stats = RunStats()

    assembled, issues, rounds, _, _ = await run_assembly_repair(
        [clean, with_ref], _FakeSource(), _config(live_validate=True, max_assembly_repair_rounds=2),
        client, _skill(), None, mw_client, None, stats,
    )

    assert issues == []
    assert rounds == 1
    assert len(client.calls) == 1  # only the ref-holding chunk got a repair call
    assert with_ref.translated_text == "Chunk with ref fixed."
    assert clean.translated_text == "Prozë krejt e pastër."  # untouched


@pytest.mark.asyncio
async def test_repaired_chunk_is_re_cached(tmp_path: Path):
    # translate_chunk's own cache.set (translator.py) runs pre-repair, when
    # a chunk is first translated. Without re-caching here, a fix made by
    # this assembly-level loop is invisible to future reruns' cache lookups
    # — this proves the fix actually reaches the cache under the same key
    # translate_chunk/translator.py would look it up with.
    chunk = _chunk("{{harvc|last=Smith}}")
    client = FakeOpenRouterClient(["{{Cite book|last=Smith}}"])
    stats = RunStats()
    cache = TranslationCache(tmp_path / "cache.sqlite3")
    config = _config(max_assembly_repair_rounds=3)
    skill = _skill()

    try:
        await run_assembly_repair(
            [chunk], _FakeSource(), config, client, skill, None,
            FakeMediaWikiClient(raise_if_called=True), None, stats,
            cache=cache, facts=None,
        )

        key = compute_key(config.model, chunk.source_lang, config.target_lang, chunk.text, skill.content_hash, "")
        assert cache.get(key) == chunk.translated_text
        assert "harvc" not in cache.get(key)
    finally:
        cache.close()


# run_review_pass: the semantic-fidelity review pass (review.py), run once
# run_assembly_repair above has already passed clean. The same
# FakeOpenRouterClient drives both the review call (JSON findings) and any
# resulting repair_chunk() call, in call order, since review_client serves
# both roles here exactly as it does in the real pipeline (config.
# resolve_review_model/resolve_review_provider commonly resolve review to
# the same client as complex_model).


@pytest.mark.asyncio
async def test_review_pass_finds_and_fixes_issue():
    chunk = _chunk("Parisi eshte nje qytet i madh.")
    client = FakeOpenRouterClient(
        [
            '[{"kind": "grammar_case", "message": "Wrong case.", "snippet": "Parisi eshte"}]',
            "Parisi është një qytet i madh.",  # repair_chunk's fix
            "[]",  # re-check after repair: clean
        ]
    )
    stats = RunStats()

    assembled, issues, rounds, _ = await run_review_pass(
        [chunk], _FakeSource(), _config(review_max_repair_attempts=2), "Parisi eshte nje qytet i madh.",
        client, "review-model", _skill(), None, FakeMediaWikiClient(raise_if_called=True), stats, VerifiedFacts(),
    )

    assert issues == []
    assert rounds == 1
    assert assembled == "Parisi është një qytet i madh."
    assert chunk.translated_text == "Parisi është një qytet i madh."
    assert stats.review_attempts == 1
    assert stats.review_corrections_applied == 1
    assert stats.review_findings_total == 1
    # 3 chat_completion calls total (2 review calls + 1 repair), all on
    # "review-model" -- confirms review.py's calls and the review-driven
    # repair both get attributed to the same per-model breakdown entry.
    assert stats.model_usage["review-model"].calls == 3
    assert stats.model_usage["review-model"].tokens_in == 300


@pytest.mark.asyncio
async def test_review_pass_reapplies_post_processing_after_repair():
    # Regression: the review pass used to reassemble with plain
    # assemble_chunks(), so a review repair silently undid every
    # post-processing fix in the saved article.
    chunk = _chunk("Parisi eshte nje qytet.{{sfn|Smith|2020|f=5}}")
    chunk.text = "Paris is a city.{{sfn|Smith|2020|p=5}}"
    client = FakeOpenRouterClient(
        [
            '[{"kind": "grammar_case", "message": "Wrong case.", "snippet": "Parisi eshte"}]',
            # The repair keeps the mistranslated |f= and adds a note of its own.
            "Parisi është një qytet.{{sfn|Smith|2020|f=5}}<!-- rregullova rasën -->",
            "[]",
        ]
    )

    assembled, issues, _, removed_comments = await run_review_pass(
        [chunk], _FakeSource(), _config(review_max_repair_attempts=2), "Parisi eshte nje qytet.{{sfn|Smith|2020|p=5}}",
        client, "review-model", _skill(), None, FakeMediaWikiClient(raise_if_called=True), RunStats(), VerifiedFacts(),
    )

    assert issues == []
    assert assembled == "Parisi është një qytet.{{sfn|Smith|2020|p=5}}"
    assert removed_comments == ["rregullova rasën"]


@pytest.mark.asyncio
async def test_review_pass_localizes_finding_to_owning_chunk_only():
    a = _chunk("Chunk A has a problem here.", order=0)
    b = _chunk("Chunk B is perfectly fine.", order=1)
    client = FakeOpenRouterClient(
        [
            '[{"kind": "semantic_fidelity", "message": "Bad.", "snippet": "Chunk A has a problem here."}]',
            "Chunk A fixed.",
            "[]",
        ]
    )
    stats = RunStats()
    initial = a.translated_text + b.translated_text

    await run_review_pass(
        [a, b], _FakeSource(), _config(), initial, client, "review-model", _skill(), None,
        FakeMediaWikiClient(raise_if_called=True), stats, VerifiedFacts(),
    )

    assert a.translated_text == "Chunk A fixed."
    assert b.translated_text == "Chunk B is perfectly fine."  # untouched


@pytest.mark.asyncio
async def test_review_pass_structural_safety_net_catches_review_driven_regression():
    # The review call itself reports no findings, but the assembled text
    # already has a static defect ({{harvc}}) -- the structural safety-net
    # check (re-running _validate_assembled each round) must still catch
    # and repair it, independent of what the review model said.
    chunk = _chunk("{{harvc|last=Smith}}")
    client = FakeOpenRouterClient(["[]", "{{Cite book|last=Smith}}", "[]"])
    stats = RunStats()

    assembled, issues, rounds, _ = await run_review_pass(
        [chunk], _FakeSource(), _config(review_max_repair_attempts=2), "{{harvc|last=Smith}}",
        client, "review-model", _skill(), None, FakeMediaWikiClient(raise_if_called=True), stats, VerifiedFacts(),
    )

    assert issues == []
    assert rounds == 1
    assert "harvc" not in assembled


@pytest.mark.asyncio
async def test_review_pass_uses_review_model_not_config_model():
    chunk = _chunk("Prozë.")
    client = FakeOpenRouterClient(["[]"])
    stats = RunStats()

    calls_with_model = []
    original_chat_completion = client.chat_completion

    async def _tracking_chat_completion(model, *args, **kwargs):
        calls_with_model.append(model)
        return await original_chat_completion(model, *args, **kwargs)

    client.chat_completion = _tracking_chat_completion

    await run_review_pass(
        [chunk], _FakeSource(), _config(model="draft-model"), "Prozë.", client, "review-model-xyz",
        _skill(), None, FakeMediaWikiClient(raise_if_called=True), stats, VerifiedFacts(),
    )

    assert calls_with_model == ["review-model-xyz"]


def test_merge_split_references_blocks_drops_spurious_first_closer():
    text = (
        "=== Citime ===\n<references>\n<ref name=\"A\">a</ref>\n</references>\n"
        "<ref name=\"B\">b</ref>\n</references>\n=== Bibliografia ===\n"
    )
    patched, merged = merge_split_references_blocks(text)
    assert merged == 1
    assert patched == (
        "=== Citime ===\n<references>\n<ref name=\"A\">a</ref>\n"
        "<ref name=\"B\">b</ref>\n</references>\n=== Bibliografia ===\n"
    )


def test_merge_split_references_blocks_leaves_well_formed_blocks_alone():
    text = (
        "== Shënime ==\n<references group=\"sh\">\n<ref name=\"A\">a</ref>\n</references>\n"
        "=== Citime ===\n<references>\n<ref name=\"B\">b</ref>\n</references>\n"
        "{{Reflist}}\n<references />\n"
    )
    assert merge_split_references_blocks(text) == (text, 0)


def test_merge_split_references_blocks_does_not_cross_headings():
    # A stray closer in a later section is not the other half of the earlier block.
    text = "<references>\n</references>\n== Next ==\n</references>\n"
    assert merge_split_references_blocks(text) == (text, 0)


def test_restore_ref_group_names_maps_translated_group_back_to_source():
    # Regression (Albert Einstein): group=note became group="shënim" in some
    # chunks but not in {{Reflist|group=note|refs=...}}.
    source = 'x<ref group=note name=A/>\n{{reflist|group=note|refs=\n<ref name=A>a</ref>\n}}\n'
    text = 'x<ref group="shënim" name=A/>\n{{Reflist|group=note|refs=\n<ref name=A>a</ref>\n}}\n'
    patched, renamed = restore_ref_group_names(text, source)
    assert renamed == {"shënim": "note"}
    assert patched == source.replace("group=note name", 'group="note" name').replace("reflist", "Reflist")


def test_restore_ref_group_names_leaves_ambiguous_groups_alone():
    source = '<ref group="a">x</ref><ref group="b">y</ref>'
    text = '<ref group="ä">x</ref><ref group="bë">y</ref>'
    assert restore_ref_group_names(text, source) == (text, {})


def test_drop_stray_reference_lists_removes_lists_the_source_lacks():
    # Regression (Albert Einstein): the source's one <references> block
    # came back with a <references group=.../> or <references /> after
    # every chunk boundary, plus one under a random mid-article heading.
    source = (
        "== Life ==\nProse.\n== Notes ==\n{{reflist|group=note|refs=\n<ref name=N>n</ref>\n}}\n"
        "== References ==\n<references>\n<ref name=A>a</ref>\n<ref name=B>b</ref>\n</references>\n"
    )
    text = (
        '== Jeta ==\n<references group="note" />\nProzë.\n== Shënime ==\n'
        "{{Reflist|group=note|refs=\n<ref name=N>n</ref>\n}}\n"
        '== Referime ==\n<references>\n<ref name=A>a</ref>\n</references>\n<references group="note"/>\n'
        '<ref name=B>b</ref>\n<references group="note"></references>\n<references />\n'
    )
    patched, removed = drop_stray_reference_lists(text, source)
    assert removed == 4
    assert patched == (
        "== Jeta ==\nProzë.\n== Shënime ==\n{{Reflist|group=note|refs=\n<ref name=N>n</ref>\n}}\n"
        "== Referime ==\n<references>\n<ref name=A>a</ref>\n</references>\n<ref name=B>b</ref>\n"
    )


def test_drop_stray_reference_lists_keeps_lists_the_source_has():
    source = "a\n== Notes ==\n{{notelist}}\n== References ==\n{{reflist}}\n"
    text = "a\n== Shënime ==\n{{Notelist}}\n== Referime ==\n{{Reflist}}\n"
    assert drop_stray_reference_lists(text, source) == (text, 0)


def test_drop_stray_reference_lists_keeps_the_last_of_duplicates():
    source = "x\n{{reflist}}\n"
    text = "x\n{{Reflist}}\ny\n{{Reflist}}\n"
    assert drop_stray_reference_lists(text, source) == ("x\ny\n{{Reflist}}\n", 1)


def test_extend_references_block_over_trailing_defs():
    text = (
        "== Referime ==\n<references>\n<ref name=A>a</ref>\n</references>\n"
        "<!-- shënim i modelit -->\n<ref name=B>b</ref>\n\n<ref name=C>c</ref>\n=== Veprat ===\n"
    )
    patched, moved = extend_references_block_over_trailing_defs(text)
    assert moved == 2
    assert patched == (
        "== Referime ==\n<references>\n<ref name=A>a</ref>\n"
        "<!-- shënim i modelit -->\n<ref name=B>b</ref>\n\n<ref name=C>c</ref>\n</references>\n"
        "=== Veprat ===\n"
    )


def test_extend_references_block_leaves_prose_after_block_alone():
    text = "<references>\n<ref name=A>a</ref>\n</references>\nProzë.<ref name=B>b</ref>\n"
    assert extend_references_block_over_trailing_defs(text) == (text, 0)


def test_restore_orphaned_ref_definitions_copies_dropped_definition_from_source():
    # Translation condensed away the sentence carrying the definition but
    # kept the later reuse — the Soviet Union "Hanson" case.
    source = (
        'Second economy.<ref name="Hanson">Hanson, Philip. 2003.</ref>\n'
        'Later.<ref name="Gregory" /><ref name="Hanson" />\n'
    )
    text = 'Më vonë.<ref name="Gregory">G</ref><ref name="Hanson" /> Pastaj.<ref name=Hanson/>\n'
    patched, restored = restore_orphaned_ref_definitions(text, source)
    assert restored == ["Hanson"]
    assert patched == (
        'Më vonë.<ref name="Gregory">G</ref><ref name="Hanson">Hanson, Philip. 2003.</ref>'
        " Pastaj.<ref name=Hanson/>\n"
    )


def test_restore_orphaned_ref_definitions_leaves_defined_and_unknown_refs_alone():
    source = '<ref name="A">a</ref>'
    text = '<ref name="A" /><ref name=\'A\'>a</ref><ref name="Unknown" />'
    assert restore_orphaned_ref_definitions(text, source) == (text, [])


def test_restore_orphaned_ref_definitions_not_fooled_by_unclosed_definition():
    # Regression (Albert Einstein): an unclosed definition (its </ref> lost
    # in translation) used to extend to the next definition's </ref>,
    # hiding "B" from the defined set, so a duplicate "B" was restored.
    source = '<ref name="A">a, p. 23.</ref><ref name="B">b</ref>'
    text = 'Use.<ref name="B"/>\n<references>\n<ref name="A">a, p.\n23.\n<ref name="B">b</ref>\n</references>'
    assert restore_orphaned_ref_definitions(text, source) == (text, [])


@pytest.mark.asyncio
async def test_assembly_repairs_run_concurrently_up_to_the_limit():
    in_flight = 0
    peak = 0

    class SlowClient(FakeOpenRouterClient):
        async def chat_completion(self, model, messages, temperature=0.0, on_retry=None, usage_out=None):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return "Prozë e pastër.", 100, 50

    chunks = [_chunk("{{harvc|last=Smith}}", order=i) for i in range(5)]
    await run_assembly_repair(
        chunks, _FakeSource(), _config(max_assembly_repair_rounds=1, assembly_repair_concurrency=2),
        SlowClient([]), _skill(), None, FakeMediaWikiClient(raise_if_called=True), None, RunStats(),
    )

    assert peak == 2
    assert all(c.translated_text == "Prozë e pastër." for c in chunks)
