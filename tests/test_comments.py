"""comments.reconcile_comments: label comments carried over from the
source, remove the ones the model wrote itself."""

from wiki_translation_harness.comments import COMMENT_MARKER, reconcile_chunk_comments, reconcile_comments
from wiki_translation_harness.models import Chunk


def test_translated_source_comment_is_labelled():
    source = "Intro.\n<!-- Please do not change this without consensus -->\nMore text here.\n"
    text = "Hyrje.\n<!-- Mos e ndryshoni pa konsensus -->\nMë shumë tekst.\n"
    patched, removed = reconcile_comments(source, text)
    assert removed == []
    assert patched == f"Hyrje.\n<!-- {COMMENT_MARKER} Mos e ndryshoni pa konsensus -->\nMë shumë tekst.\n"


def test_model_comment_is_removed_with_its_line():
    # Regression (Albert Einstein): a repair call that found nothing to fix
    # explained itself in a comment.
    source = '<ref name="A">a</ref>\n<ref name="B">b</ref>\n'
    text = (
        '<ref name="A">a</ref>\n'
        "<!-- Nuk u identifikuan me siguri ndryshimet e nevojshme për riparimin e sintaksës. -->\n"
        '<ref name="B">b</ref>\n'
    )
    patched, removed = reconcile_comments(source, text)
    assert removed == ["Nuk u identifikuan me siguri ndryshimet e nevojshme për riparimin e sintaksës."]
    assert patched == source


def test_extra_model_comment_among_source_comments():
    # Napoleon: the model added a note about content it left out, between
    # two editor comments it translated.
    source = (
        "<!-- Infobox agreed by consensus -->\n" + "x" * 400 + "\n"
        "== Early career ==\n" + "y" * 400 + "\n<!-- Do not change the name -->\nEnd.\n"
    )
    text = (
        "<!-- Infobox-i u ra dakord me konsensus -->\n" + "x" * 400 + "\n"
        "== Karriera e hershme ==\n<!-- Harta ({{OSM Location map}}) është lënë jashtë. -->\n"
        + "y" * 400 + "\n<!-- Mos e ndryshoni emrin -->\nFund.\n"
    )
    patched, removed = reconcile_comments(source, text)
    assert removed == ["Harta ({{OSM Location map}}) është lënë jashtë."]
    assert patched.count(COMMENT_MARKER) == 2
    assert "OSM Location map" not in patched


def test_untranslated_comment_matches_by_its_tokens_even_if_moved():
    source = "<!-- see footnote 4, https://example.org/a -->" + "x" * 1000
    text = "y" * 1000 + "<!--see footnote 4, https://example.org/a-->"
    patched, removed = reconcile_comments(source, text)
    assert removed == []
    assert patched.endswith(f"<!-- {COMMENT_MARKER} see footnote 4, https://example.org/a -->")


def test_inline_comment_removal_keeps_the_line():
    patched, removed = reconcile_comments("Teksti.", "Teksti.<!-- shënim i modelit --> Vazhdim.")
    assert removed == ["shënim i modelit"]
    assert patched == "Teksti. Vazhdim."


def test_reconcile_is_idempotent():
    source = "A.\n<!-- note -->\nB.\n"
    once, _ = reconcile_comments(source, "A.\n<!-- shënim -->\nB.\n")
    assert reconcile_comments(source, once) == (once, [])


def test_reconcile_chunk_comments_works_per_chunk():
    chunks = [
        Chunk(article_title="T", section_titles=["S0"], order=0, text="A <!-- c -->", token_estimate=1,
              translated_text="A <!-- k -->"),
        Chunk(article_title="T", section_titles=["S1"], order=1, text="B", token_estimate=1,
              translated_text="B <!-- shtuar -->"),
    ]
    assert reconcile_chunk_comments(chunks) == ["shtuar"]
    assert chunks[0].translated_text == f"A <!-- {COMMENT_MARKER} k -->"
    assert chunks[1].translated_text == "B "
