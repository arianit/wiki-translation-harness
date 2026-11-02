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
    assert "date=February 2019|" in out and "access-date=2025-11-27|" in out and "archive-date=2020-01-05|" in out
    assert n == 2


def test_ill_inside_link_label():
    out, _ = unnest_ill_in_wikilinks("[[Steven Levitsky|{{ill|Steven Levitsky|en|Steven Levitsky}}]] thotë")
    assert out == "{{ill|Steven Levitsky|en|Steven Levitsky}} thotë"


def test_albanian_month_only_date_restored_to_english():
    out, n = fix_archive_dates("{{cite web|url=u|date=Dhjetor 2021|access-date=Nëntor 2022}}")
    assert "date=December 2021|" in out and "access-date=November 2022}}" in out and n == 2


def test_bare_year_month_date_restored_to_english():
    out, n = fix_archive_dates("{{cite web|url=u|date=2008-03|access-date=2024-10}}")
    assert "date=March 2008" in out and "access-date=October 2024" in out
    assert n == 2
    out, _ = fix_archive_dates("{{cite web|url=u|date=2024-10-05}}")
    assert "date=2024-10-05" in out


def test_as_of_with_month_becomes_year_plus_alt():
    from wiki_translation_harness.layout_fixes import fix_as_of

    src = "Fjali.<ref>x</ref> {{As of|April 2021}} a. Dhe, {{As of|2021|April|lang=sq}} b ({{as of|2023|11|30}}) {{As of|2025|lc=y}}"
    out, n = fix_as_of(src)
    assert out == (
        "Fjali.<ref>x</ref> {{As of|2021|alt=Që nga prilli 2021}} a. Dhe, {{As of|2021|alt=që nga prilli 2021}} b "
        "({{As of|2023|alt=që nga 30 nëntori 2023}}) {{As of|2025|lc=y}}"
    )
    assert n == 3


def test_age_template_expanded_to_plain_text():
    from wiki_translation_harness.layout_fixes import expand_age_templates

    out, n = expand_age_templates(
        "|{{Age in years and days nts|1946|02|02|1952|11|10}}|x|{{Age in years and days nts|1945|10|24|1946|02|02}}|{{Age in years and days nts|2017|01|01}}|"
    )
    assert out == "|6 vjet, 282 ditë|x|101 ditë||" and n == 3


def test_broken_wayback_urls_are_rebuilt():
    from wiki_translation_harness.layout_fixes import fix_wayback_urls

    src = (
        "a=https://web.archive.org/web/2020/01/2020-01-29133126/https://web.archive.org/web/2020/01/2020-01-29133126/https://x.com/p "
        "b=https://web.archive.org/web/2020-05-16/052110/https://x.com/q c=https://web.archive.org/web/2020/05/11/150234/https://x.com/r "
        "d=https://web.archive.org/web/20200606054536/https://x.com/s"
    )
    out, n = fix_wayback_urls(src)
    assert out == (
        "a=https://web.archive.org/web/20200129133126/https://x.com/p "
        "b=https://web.archive.org/web/20200516052110/https://x.com/q c=https://web.archive.org/web/20200511150234/https://x.com/r "
        "d=https://web.archive.org/web/20200606054536/https://x.com/s"
    )
    assert n == 3


def test_agency_folded_into_publisher():
    from wiki_translation_harness.layout_fixes import fix_cs1_agency

    out, n = fix_cs1_agency("{{cite news|title=T|agency=AFP|date=2009}}{{cite web|publisher=chinaview.cn|agency=Xinhua}}")
    assert out == "{{cite news|title=T|publisher=AFP|date=2009}}{{cite web|publisher=Xinhua; chinaview.cn}}" and n == 2
