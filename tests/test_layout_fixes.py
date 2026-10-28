import asyncio

from wiki_translation_harness.layout_fixes import (
    blank_line_before_headings,
    fix_archive_dates,
    replace_missing_excerpts,
    unnest_ill_in_wikilinks,
)


def test_unnest_ill_with_label():
    t = "x [[{{ill|Pëlqimi|en|Consent of the governed}}|pëlqimi i të qeverisurve]] y"
    out, n = unnest_ill_in_wikilinks(t)
    assert out == "x {{ill|Pëlqimi|en|Consent of the governed|lt=pëlqimi i të qeverisurve}} y"
    assert n == 1


def test_unnest_ill_without_label():
    out, _ = unnest_ill_in_wikilinks("[[{{ill|Protektorati|en|the Protectorate}}]]")
    assert out == "{{ill|Protektorati|en|the Protectorate}}"


def test_heading_gets_blank_line():
    out, n = blank_line_before_headings("para\n====Deliberimi====\ntext\n==A==\n===B===\n\n==C==")
    assert out == "para\n\n====Deliberimi====\ntext\n\n==A==\n===B===\n\n==C=="
    assert n == 2


def test_archive_date_follows_wayback_timestamp():
    t = "{{cite web|archive-url=https://web.archive.org/web/20190905232450/https://x.org|archive-date=2015-12-08}}"
    out, n = fix_archive_dates(t)
    assert "archive-date=2019-09-05" in out and n == 1


def test_archive_date_english_format():
    out, _ = fix_archive_dates("{{cite web|archive-date=22 February 2024|url=u}}")
    assert "archive-date=2024-02-22" in out


def test_missing_excerpt_becomes_ill_hatnote():
    async def exists(titles):
        return {t: False for t in titles}

    out, rep = asyncio.run(
        replace_missing_excerpts(
            "{{Excerpt|Demokracia procedurale|paragraphs=1}}", exists, "{{Excerpt|Procedural democracy|paragraphs=1}}"
        )
    )
    assert out == "{{Hatnote|Artikulli kryesor: {{ill|Demokracia procedurale|en|Procedural democracy}}}}"
    assert rep == ["Procedural democracy"]


def test_english_dates_become_iso():
    t = "{{cite web|date=February 2019|access-date=27 November 2025|archive-date=Jan 5, 2020|url=u}}"
    out, n = fix_archive_dates(t)
    assert "date=2019-02|" in out and "access-date=2025-11-27|" in out and "archive-date=2020-01-05|" in out
    assert n == 3


def test_ill_inside_link_label():
    out, _ = unnest_ill_in_wikilinks("[[Steven Levitsky|{{ill|Steven Levitsky|en|Steven Levitsky}}]] thotë")
    assert out == "{{ill|Steven Levitsky|en|Steven Levitsky}} thotë"
