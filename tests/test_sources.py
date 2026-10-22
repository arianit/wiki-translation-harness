from pathlib import Path

from wiki_translation_harness.sources import parse_source_ref, resolve_static_inputs


def test_parse_lang_prefix():
    lang, title = parse_source_ref("sq:Gjergj Arianiti")
    assert lang == "sq"
    assert title == "Gjergj Arianiti"


def test_parse_full_url_with_scheme():
    lang, title = parse_source_ref("https://sq.wikipedia.org/wiki/Gjergj_Arianiti")
    assert lang == "sq"
    assert title == "Gjergj Arianiti"


def test_parse_mobile_subdomain_url():
    lang, title = parse_source_ref("https://sq.m.wikipedia.org/wiki/Gjergj_Arianiti")
    assert lang == "sq"
    assert title == "Gjergj Arianiti"


def test_parse_url_decodes_percent_encoding():
    lang, title = parse_source_ref("https://sr.wikipedia.org/wiki/%D0%9D%D0%B8%D1%88")
    assert lang == "sr"
    assert title == "Ниш"


def test_namespace_prefix_not_mistaken_for_lang():
    # "Category" is capitalized -> real MediaWiki namespace, not a lang code
    lang, title = parse_source_ref("Category:Physics")
    assert lang is None
    assert title == "Category:Physics"


def test_resolve_static_inputs_mixed_titles_file(tmp_path: Path):
    titles_file = tmp_path / "titles.txt"
    titles_file.write_text("Paris\nsq:Gjergj Arianiti\n# a comment\nsr:Ниш\n")
    inputs = resolve_static_inputs(title=None, titles_file=titles_file, file=None, directory=None)
    assert [(i.title, i.source_lang) for i in inputs] == [
        ("Paris", None),
        ("Gjergj Arianiti", "sq"),
        ("Ниш", "sr"),
    ]


def test_resolve_static_inputs_repeated_titles():
    inputs = resolve_static_inputs(
        title=["Paris", "sq:Gjergj Arianiti", "https://sr.wikipedia.org/wiki/%D0%9D%D0%B8%D1%88"],
        titles_file=None,
        file=None,
        directory=None,
    )
    assert [(i.title, i.source_lang) for i in inputs] == [
        ("Paris", None),
        ("Gjergj Arianiti", "sq"),
        ("Ниш", "sr"),
    ]


