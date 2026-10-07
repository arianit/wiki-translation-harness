from wiki_translation_harness.parser import build_chunks, estimate_tokens, split_into_sections


def test_sections_roundtrip_lead_and_headings():
    text = "Lead text.\n\n== History ==\nHistory body.\n\n=== Sub ===\nSub body.\n"
    sections = split_into_sections(text)
    assert [s.title for s in sections] == ["Lead", "History", "Sub"]
    assert "".join(s.wikitext for s in sections) == text


def test_chunk_never_splits_template():
    big_template = "{{cite web|title=" + ("x" * 20000) + "}}"
    text = f"Some intro.\n\n== Refs ==\n{big_template}\n"
    sections = split_into_sections(text)
    chunks = build_chunks("Test", sections, chunk_min=10, chunk_max=50)
    combined = "".join(c.text for c in chunks)
    assert combined == text
    assert any(big_template in c.text for c in chunks)
    for c in chunks:
        assert c.text.count("{{") == c.text.count("}}")


def test_chunk_never_splits_table():
    table = "{| class=\"wikitable\"\n" + "\n".join(f"| row {i}" for i in range(200)) + "\n|}"
    text = f"Intro.\n\n== Data ==\n{table}\n"
    sections = split_into_sections(text)
    chunks = build_chunks("Test", sections, chunk_min=10, chunk_max=50)
    assert any(table in c.text for c in chunks)
    for c in chunks:
        assert c.text.count("{|") == c.text.count("|}")


def test_chunk_never_splits_reference():
    ref = "<ref>" + ("Citation detail. " * 500) + "</ref>"
    text = f"Body with a ref{ref} continues.\n"
    sections = split_into_sections(text)
    chunks = build_chunks("Test", sections, chunk_min=10, chunk_max=50)
    assert any(ref in c.text for c in chunks)
    for c in chunks:
        assert c.text.count("<ref>") == c.text.count("</ref>")


def test_small_sections_merge_up_to_max():
    text = "\n\n".join(f"== S{i} ==\nShort body {i}." for i in range(10))
    sections = split_into_sections(text)
    chunks = build_chunks("Test", sections, chunk_min=50, chunk_max=200)
    # small sections should merge into fewer chunks than sections
    assert len(chunks) < len(sections)
    assert "".join(c.text for c in chunks) == text


def test_build_chunks_propagates_source_lang():
    text = "Lead.\n\n== History ==\nBody.\n"
    sections = split_into_sections(text)
    chunks = build_chunks("Test", sections, chunk_min=1500, chunk_max=2500, source_lang="sq")
    assert all(c.source_lang == "sq" for c in chunks)


def test_oversized_table_trailing_remainder_merged_not_standalone():
    # Regression: a huge table (a protected, unsplittable unit) shipped as
    # its own oversized chunk, leaving a trailing near-empty fragment (in
    # the real case, a single newline -> 1 token) as its own final chunk.
    # Sent almost nothing to translate, the model didn't echo it back — it
    # fabricated a plausible-looking replacement section instead. No chunk
    # should ever be that close to empty when there's a previous chunk to
    # absorb it into.
    table_rows = "\n".join(f"| row {i} data here" for i in range(300))
    table = f"{{| class=\"wikitable\"\n{table_rows}\n|}}"
    text = f"Intro sentence.\n\n== Section ==\n{table}\n"
    sections = split_into_sections(text)
    chunks = build_chunks("Test", sections, chunk_min=100, chunk_max=300)

    assert "".join(c.text for c in chunks) == text
    # the trailing "\n" after the table must not become its own chunk
    for c in chunks:
        assert estimate_tokens(c.text) > 5 or c.text.strip()


def test_split_oversized_section_no_standalone_near_empty_chunk():
    from wiki_translation_harness.parser import _split_oversized_section

    table = "{| class=\"wikitable\"\n" + "\n".join(f"| {i}" for i in range(300)) + "\n|}"
    text = f"{table}\n"
    result = _split_oversized_section(text, chunk_min=100, chunk_max=300)
    assert "".join(result) == text
    # no chunk should be a bare near-empty remainder trailing the table
    assert not any(estimate_tokens(c) <= 2 for c in result)


def test_oversized_section_heading_merged_into_following_table_not_standalone():
    # Regression (real case: "Timeline of Albanian history", "== 21st
    # century ==" section): a section heading followed immediately by a
    # huge table (a protected, unsplittable unit) left the heading as its
    # own near-empty leading chunk, flushed right before the oversized
    # table unit. Sent a bare heading with nothing to translate, the model
    # fabricated a replacement instead of echoing it back. The heading must
    # be merged into the following oversized chunk instead.
    from wiki_translation_harness.parser import _split_oversized_section

    table_rows = "\n".join(f"| row {i} data here" for i in range(300))
    table = f"{{| class=\"wikitable\"\n{table_rows}\n|}}"
    text = f"== 21st century ==\n{table}\n"
    result = _split_oversized_section(text, chunk_min=100, chunk_max=300)

    assert "".join(result) == text
    assert not any(estimate_tokens(c) <= 5 for c in result)
    assert result[0].startswith("== 21st century ==\n")


def test_chunk_never_splits_ref_inside_references_block():
    # Regression (Albert Einstein): list-defined refs inside a large
    # <references> block were split as free text, so the sentence splitter
    # cut `... p. 23.</ref>` after "p." and the model dropped the orphaned
    # `</ref>` that opened the next chunk.
    defs = "".join(
        f'<ref name="r{i}">van Dongen, Jeroen (2010) Einstein\'s Unification. CUP, p. {i}.</ref>\n'
        for i in range(60)
    )
    text = f"== Citations ==\n<references>\n{defs}</references>\n"
    sections = split_into_sections(text)
    chunks = build_chunks("Test", sections, chunk_min=10, chunk_max=50)
    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == text
    for c in chunks:
        assert c.text.count("<ref ") == c.text.count("</ref>")
