from wiki_translation_harness.notes import expand_cref_notes

_SOURCE = """Lead.{{Cref2|a}} Body.{{Cref2|b}} Again.{{Cref2|a}}

== Notes ==
{{Cnote2 Begin|liststyle=upper-alpha}}
{{Cnote2|a|First note.{{Sfn|Smith|2000|p=1}}}}
{{Cnote2|b|Second note.<ref name="x">{{harvnb|Jones|1999|p=2}}</ref>}}
{{Cnote2|z|Never cited.}}
{{Cnote2 End}}
{{notelist}}

== Citations ==
"""


def test_expand_cref_notes_inlines_markers_and_replaces_block():
    result = expand_cref_notes(_SOURCE)
    text = result.patched_wikitext
    assert result.expanded == 3
    assert result.unresolved == []
    assert result.unreferenced == ["z"]
    # First use carries the text (sfn swapped to harvnb so refs don't nest); reuse is self-closing.
    assert '<ref name="cnote-a" group="upper-alpha">First note.{{harvnb|Smith|2000|p=1}}</ref>' in text
    assert text.count('<ref name="cnote-a" group="upper-alpha" />') == 1
    # The note's own citation is moved out of the note ref, right after it.
    assert (
        '<ref name="cnote-b" group="upper-alpha">Second note.</ref>'
        '<ref name="x">{{harvnb|Jones|1999|p=2}}</ref>'
    ) in text
    # Block and notelist collapse to one Reflist; blank line before the next heading is kept.
    assert "== Notes ==\n{{Reflist|group=upper-alpha}}\n\n== Citations ==" in text
    assert "Cnote2" not in text and "Cref2" not in text and "notelist" not in text


def test_expand_cref_notes_uses_the_liststyle_group():
    text = "A.{{Cref2|a}}\n{{Cnote2 Begin|liststyle=lower-roman}}\n{{Cnote2|a|N.}}\n{{Cnote2 End}}\n"
    assert 'group="lower-roman"' in expand_cref_notes(text).patched_wikitext


def test_expand_cref_notes_leaves_unknown_labels_and_plain_articles_alone():
    plain = "No notes here.<ref>x</ref>"
    assert expand_cref_notes(plain).patched_wikitext == plain
    text = "A.{{Cref2|q}} B.{{Cref2|a}}\n{{Cnote2 Begin}}\n{{Cnote2|a|N.}}\n{{Cnote2 End}}\n"
    result = expand_cref_notes(text)
    assert result.unresolved == ["q"]
    assert "{{Cref2|q}}" in result.patched_wikitext
    assert result.expanded == 1


def test_expand_cref_notes_skips_a_marker_inside_a_ref():
    text = "A.<ref>See {{Cref2|a}}</ref>\n{{Cnote2 Begin}}\n{{Cnote2|a|N.}}\n{{Cnote2 End}}\n"
    result = expand_cref_notes(text)
    assert result.unresolved == ["a"]
    assert "{{Cref2|a}}" in result.patched_wikitext
