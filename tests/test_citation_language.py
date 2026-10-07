import httpx
import pytest
import respx

from wiki_translation_harness.citation_language import (
    dedupe_short_footnotes,
    detect_source_language,
    fill_missing_citation_languages,
    fix_citation_param_names,
    guess_language_from_title,
    convert_nested_ref_notes_to_refn,
    convert_nested_sfn_to_harvnb,
    unwrap_redundant_sfn_ref,
)


def test_guess_language_from_title_english():
    assert guess_language_from_title("This is a fairly normal English sentence used as a title") == "en"


@pytest.mark.asyncio
async def test_detect_source_language_from_content_language_header():
    with respx.mock() as mock:
        mock.get("https://example.com/page").mock(
            return_value=httpx.Response(200, headers={"content-language": "sr-Latn"}, text="hi")
        )
        async with httpx.AsyncClient() as client:
            lang = await detect_source_language(client, "https://example.com/page")
    assert lang == "sr"


@pytest.mark.asyncio
async def test_detect_source_language_falls_back_to_text_detection():
    html = (
        "<html><body><p>" + ("Dies ist ein deutscher Beispieltext mit vielen Wörtern. " * 10) + "</p></body></html>"
    )
    with respx.mock() as mock:
        mock.get("https://example.com/page").mock(
            return_value=httpx.Response(200, headers={"content-type": "text/html"}, text=html)
        )
        async with httpx.AsyncClient() as client:
            lang = await detect_source_language(client, "https://example.com/page")
    assert lang == "de"


@pytest.mark.asyncio
async def test_fill_missing_citation_languages_via_url():
    text = "{{Cite web|title=Some Page|url=https://example.com/page}}"
    with respx.mock() as mock:
        mock.get("https://example.com/page").mock(
            return_value=httpx.Response(200, headers={"content-language": "en"}, text="hi")
        )
        async with httpx.AsyncClient() as client:
            result = await fill_missing_citation_languages(text, client)
    assert "|language=en" in result.patched_wikitext
    assert result.filled == {"Some Page": "en"}
    assert result.fetched_urls == 1


@pytest.mark.asyncio
async def test_fill_missing_citation_languages_falls_back_to_title_when_no_url():
    text = "{{Cite book|title=This is quite clearly written in English prose}}"
    async with httpx.AsyncClient() as client:
        result = await fill_missing_citation_languages(text, client)
    assert "|language=en" in result.patched_wikitext
    assert result.fetched_urls == 0


@pytest.mark.asyncio
async def test_fill_missing_citation_languages_skips_already_tagged():
    text = "{{Cite web|title=X|url=https://example.com/page|language=fr}}"
    async with httpx.AsyncClient() as client:
        result = await fill_missing_citation_languages(text, client)
    assert result.attempted == 0
    assert result.filled == {}
    assert result.patched_wikitext == text


@pytest.mark.asyncio
async def test_fill_missing_citation_languages_respects_max_url_fetches():
    text = "".join(
        f"{{{{Cite web|title=Page {i}|url=https://example.com/{i}}}}}" for i in range(5)
    )
    with respx.mock() as mock:
        mock.get(url__regex=r"https://example\.com/\d+").mock(
            return_value=httpx.Response(200, headers={"content-language": "en"}, text="hi")
        )
        async with httpx.AsyncClient() as client:
            result = await fill_missing_citation_languages(text, client, max_url_fetches=2)
    assert result.fetched_urls == 2
    # the other 3 should still get a fallback attempt from their (too-short) titles, likely unfilled
    assert result.attempted == 5


@pytest.mark.asyncio
async def test_title_wins_over_url_when_they_disagree():
    # Regression: a Springer/JSTOR/Cambridge-style landing page reports its
    # own English UI language via <html lang>, not the cited work's actual
    # language — the title text is direct evidence about the work itself.
    german_title = "Sonderheft zum 60. Geburtstag von Herrn Professor mit vielen Wörtern"
    text = f"{{{{Cite book|title={german_title}|url=https://example.com/landing}}}}"
    with respx.mock() as mock:
        mock.get("https://example.com/landing").mock(
            return_value=httpx.Response(
                200, headers={"content-type": "text/html"}, text='<html lang="en"><body>x</body></html>'
            )
        )
        async with httpx.AsyncClient() as client:
            result = await fill_missing_citation_languages(text, client)
    assert result.filled == {german_title: "de"}
    assert "|language=de" in result.patched_wikitext


def test_fix_citation_param_names_real_world_case():
    # Exact case observed in a real run: every CS1 param name mistranslated.
    text = (
        "{{Cite web |data=3 mars 2025 |titulli=Some Title "
        "|url=https://example.com |faqja=BalkanWeb "
        "|data-e-përdorimit=22 qershor 2026|language=en }}"
    )
    result = fix_citation_param_names(text)
    assert "|date=3 mars 2025" in result.patched_wikitext
    assert "|title=Some Title" in result.patched_wikitext
    assert "|website=BalkanWeb" in result.patched_wikitext
    assert "|access-date=22 qershor 2026" in result.patched_wikitext
    assert result.renamed == {
        "data": "date",
        "titulli": "title",
        "faqja": "website",
        "data-e-përdorimit": "access-date",
    }


def test_fix_citation_param_names_skips_rename_if_english_already_present():
    # Both |title= and |titulli= present — ambiguous, not the harness's
    # call to resolve which one is authoritative, so leave both alone.
    text = "{{Cite web|title=Correct|titulli=Duplicate}}"
    result = fix_citation_param_names(text)
    assert result.patched_wikitext == text
    assert result.renamed == {}


def test_dedupe_short_footnotes_real_world_collision():
    text = (
        '{{sfn|Lane Fox|2011|p=342|ps=: "Quote version A"}}'
        " text in between "
        '{{sfn|Lane Fox|2011|p=342|ps=: "Quote version B, different"}}'
    )
    result = dedupe_short_footnotes(text)
    assert result.patched_wikitext.count('ps=: "Quote version A"') == 2
    assert "Quote version B" not in result.patched_wikitext
    assert len(result.canonicalized) == 1


def test_dedupe_short_footnotes_handles_harvnb():
    text = (
        '{{harvnb|Jones|1999|p=10|ps=First version}} '
        '{{harvnb|Jones|1999|p=10|ps=Second version}}'
    )
    result = dedupe_short_footnotes(text)
    assert result.patched_wikitext.count("First version") == 2
    assert "Second version" not in result.patched_wikitext


def test_unwrap_redundant_sfn_ref_strips_bare_wrapper():
    # Confirmed live against sq.wikipedia (Conservatism, 2026-09-22): {{sfn}}
    # already expands to its own <ref name="FOOTNOTE...">...</ref>, so an
    # extra outer <ref> around it corrupts Cite's usage tracking for that
    # auto-generated name.
    text = "A.<ref>{{sfn|Eccleshall|1990|p=83}}</ref> B."
    result = unwrap_redundant_sfn_ref(text)
    assert result.patched_wikitext == "A.{{sfn|Eccleshall|1990|p=83}} B."
    assert result.unwrapped == ["{{sfn|Eccleshall|1990|p=83}}"]


def test_unwrap_redundant_sfn_ref_leaves_named_ref_untouched():
    # A name= may be relied on for reuse elsewhere (<ref name="x" />) --
    # unwrapping could silently break that, so this is left for repair.
    text = '<ref name="x">{{sfn|Smith|2020|p=1}}</ref>'
    result = unwrap_redundant_sfn_ref(text)
    assert result.patched_wikitext == text
    assert result.unwrapped == []


def test_unwrap_redundant_sfn_ref_leaves_extra_prose_untouched():
    # Content beside the sfn call would be dropped by a naive unwrap.
    text = "<ref>{{sfn|Smith|2020|p=1}} see also the discussion above.</ref>"
    result = unwrap_redundant_sfn_ref(text)
    assert result.patched_wikitext == text
    assert result.unwrapped == []




def test_convert_nested_sfn_to_harvnb_renames_bundled_calls_inside_ref():
    text = "A.<ref>{{Sfn|Lane Fox|1980|pp=65–66}}, {{sfnp|Renault|2001|p=44}}</ref> B.{{sfn|Roisman|2010|p=1}}"
    result = convert_nested_sfn_to_harvnb(text)
    assert result.patched_wikitext == (
        "A.<ref>{{harvnb|Lane Fox|1980|pp=65–66}}, {{harvp|Renault|2001|p=44}}</ref> B.{{sfn|Roisman|2010|p=1}}"
    )
    assert len(result.converted) == 2


def test_convert_nested_sfn_to_harvnb_preserves_name_spacing_and_leaves_bare_sfn():
    text = '<ref name="x">{{Sfn |A|1999|p=2}} note</ref>{{Sfn|B|2000|p=3}}'
    result = convert_nested_sfn_to_harvnb(text)
    assert result.patched_wikitext == '<ref name="x">{{harvnb |A|1999|p=2}} note</ref>{{Sfn|B|2000|p=3}}'


def test_convert_nested_ref_notes_to_refn_restores_refn():
    # Albert Einstein / Vietnam War: {{refn|group=note|...<ref>..</ref>}}
    # came back as a <ref> nested inside a <ref>, which Cite can't parse.
    text = (
        'Spy;<ref group="A" name="start date">Claim | per [[a|b]].<ref>{{cite web|title=T}}</ref>'
        '<ref name="x" /></ref> next.<ref>{{cite web|title=U}}</ref>'
    )
    result = convert_nested_ref_notes_to_refn(text)
    assert result.converted == ['<ref group="A" name="start date">']
    assert result.patched_wikitext == (
        "Spy;{{refn|group=A|name=start date|1=Claim {{!}} per [[a|b]].<ref>{{cite web|title=T}}</ref>"
        '<ref name="x" />}} next.<ref>{{cite web|title=U}}</ref>'
    )


def test_convert_nested_ref_notes_to_refn_ignores_cut_definitions_and_plain_refs():
    # A ref definition cut by chunking (no closer on its line) must not be
    # taken for a note wrapping the next definition.
    text = (
        '<ref name="a">van Dongen, p.\n23.\n<ref name="b">{{cite book|title=B}}</ref>\n'
        "x.<ref>A</ref><ref>B</ref></ref>"
    )
    assert convert_nested_ref_notes_to_refn(text).patched_wikitext == text
    assert convert_nested_ref_notes_to_refn(text).converted == []
