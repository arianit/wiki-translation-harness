import asyncio

from wiki_translation_harness.hatnote_ill import add_ill_to_hatnotes


def _run(text, source, present):
    async def exists(titles):
        return {t: t in present for t in titles}

    return asyncio.run(add_ill_to_hatnotes(text, source, exists))


def test_missing_further_targets_become_ill_in_a_hatnote():
    # Regression (Karl Marx): both targets redlinked with no interwiki.
    text = "==== A ====\n{{Shih|Kritika e ekonomisë politike|Ekonomia marksiste}}\nBody"
    source = "==== A ====\n{{further|Critique of political economy|Marxian economics}}\n"
    patched, converted = _run(text, source, set())
    assert patched == (
        "==== A ====\n{{Hatnote|Lexo më tepër: {{ill|Kritika e ekonomisë politike|en|Critique of political economy}}"
        " dhe {{ill|Ekonomia marksiste|en|Marxian economics}}}}\nBody"
    )
    assert converted == ["Kritika e ekonomisë politike", "Ekonomia marksiste"]


def test_existing_target_is_left_alone():
    text = "{{Kryesor|Londra}}"
    assert _run(text, "{{Main|London}}", {"Londra"}) == (text, [])


def test_mixed_call_links_the_existing_target_plainly():
    patched, _ = _run("{{Shih|A|B}}", "{{Further|X|Y}}", {"A"})
    assert patched == "{{Hatnote|Lexo më tepër: [[A]] dhe {{ill|B|en|Y}}}}"


def test_unpairable_call_is_left_alone():
    text = "{{Kryesor|A}}"
    assert _run(text, "{{Main|X|Y}}", set()) == (text, [])


def test_pairing_skips_source_hatnotes_the_model_already_rewrote():
    # Karl Marx: the model had turned the one-target {{Further}} into {{Hatnote}}.
    source = "{{Further|Human nature}}\n{{Further|Critique|Marxian economics}}"
    text = "{{Hatnote|Lexo më tepër: {{ill|Natyra|en|Human nature}}}}\n{{Shih|Kritika|Ekonomia}}"
    patched, converted = _run(text, source, set())
    assert patched.endswith("{{Hatnote|Lexo më tepër: {{ill|Kritika|en|Critique}} dhe {{ill|Ekonomia|en|Marxian economics}}}}")
    assert converted == ["Kritika", "Ekonomia"]
