"""find_live_issues is tested directly against canned action=parse response
dicts — no network. The HTML fragments below are real responses captured
from a live sq.wikipedia.org action=parse call during development (see
live_validator.py's module docstring), not hand-guessed, so the regexes are
tested against MediaWiki's actual output shape, including the
reference-text-wraps-the-error-span nesting that a naive first version of
the matcher missed.

A couple of respx-mocked tests at the bottom cover MediaWikiClient.parse_wikitext
and validate_wikitext_live end-to-end, matching test_mediawiki.py's convention.
"""

import httpx
import pytest
import respx

from wiki_translation_harness.live_validator import find_live_issues, validate_wikitext_live
from wiki_translation_harness.mediawiki import MediaWikiClient

API_URL = "https://sq.wikipedia.org/w/api.php"


def test_lua_script_error_detected():
    parse_result = {
        "text": (
            '<div class="mw-content-ltr mw-parser-output" lang="sq" dir="ltr"><p>'
            '<strong class="error"><span class="scribunto-error mw-scribunto-error-0ffbb6ca">'
            'Script error: No such module &quot;NonExistentModuleXYZ&quot;.</span></strong>'
            "</p></div>"
        ),
        "templates": [],
    }
    issues = find_live_issues(parse_result)
    assert len(issues) == 1
    assert issues[0].kind == "lua_script_error"
    assert issues[0].severity == "error"
    assert "NonExistentModuleXYZ" in issues[0].message


def test_orphaned_named_ref_detected_despite_wrapping_span():
    # Regression: the real Cite output wraps the error span in an outer
    # <span class="reference-text">...</span> with no error class of its
    # own and the SAME tag name — a naive non-greedy match pairs the
    # wrapper's open tag with the error span's own close tag and silently
    # swallows it as unrecognized content. This is the exact shape that bit
    # the first version of _ERROR_SPAN_RE.
    parse_result = {
        "text": (
            '<ol class="references">'
            '<li id="cite_note-x-1"><span class="mw-cite-backlink">^</span> '
            '<span class="reference-text"> <span class="error mw-ext-cite-error" lang="sq" dir="ltr">'
            "Gabim citimi: Etiketë <code>&lt;ref&gt;</code> e pavlefshme;\n"
            'asnjë tekst nuk u dha për refs e quajtura "x"</span></span>'
            "</li></ol>"
        ),
        "templates": [],
    }
    issues = find_live_issues(parse_result)
    assert len(issues) == 1
    assert issues[0].kind == "orphaned_named_ref"
    assert "Gabim citimi" in issues[0].message


def test_repeated_cite_error_reported_once():
    # Cite renders one error span per use of a broken named ref; a ref
    # reused several times must still produce a single finding.
    span = (
        '<span class="error mw-ext-cite-error" lang="sq" dir="ltr">'
        "Gabim citimi: Etiketë <code>&lt;ref&gt;</code> e pavlefshme; "
        'asnjë tekst nuk u dha për refs e quajtura "x"</span>'
    )
    parse_result = {"text": "<p>" + span * 9 + "</p>", "templates": []}
    issues = find_live_issues(parse_result)
    assert len(issues) == 1
    assert issues[0].kind == "orphaned_named_ref"


def test_unexpanded_template_locates_template_syntax_not_prose_word():
    """Regression test: a missing template named '\"' or 'Main' must match
    the actual {{'\"}} or {{Main}} template invocation, not an unrelated
    prose word or quote mark earlier in the article."""
    source_text = "Line 1 has 'quotes' here.\nLine 2 has {{'\"}} template.\nLine 3 is end."
    parse_result = {
        "text": "<p>irrelevant</p>",
        "templates": [{"ns": 10, "title": "Stampa:'\"", "exists": False}],
    }
    issues = find_live_issues(parse_result, source_text=source_text)
    assert len(issues) == 1
    assert issues[0].line_number == 2
    assert issues[0].snippet == "{{'\"}} template."



def test_known_harmless_missing_dependency_module_not_flagged():
    """Moduli:WikidataIB/i18n is confirmed missing on sq.wikipedia but is
    only a conditional dependency of the legitimate, existing
    {{Authority control}} template -- not a translation defect, and not
    fixable by re-prompting the model, so it must not block delivery."""
    parse_result = {
        "text": "<p>irrelevant</p>",
        "templates": [
            {"ns": 10, "title": "Stampa:Authority control", "exists": True},
            {"ns": 828, "title": "Moduli:WikidataIB/i18n", "exists": False},
        ],
    }
    assert find_live_issues(parse_result) == []


def test_country_data_leak_detected_once_not_per_html_occurrence():
    # Regression: MediaWiki's redlink markup repeats the same leaked title
    # in the href query string, the title="" attribute, and the anchor's
    # visible text — matching raw HTML (rather than stripped visible text)
    # flagged this three times for one real leak, and the href's
    # underscore-for-space encoding let \S+ run on into the query string.
    parse_result = {
        "text": (
            '<p><a href="/w/index.php?title=Stampa:Country_data_NonExistentCountryXYZ'
            '&amp;action=edit&amp;redlink=1" class="new" '
            'title="Stampa:Country data NonExistentCountryXYZ (nuk është shkruar akoma)">'
            "Stampa:Country data NonExistentCountryXYZ</a>\n</p>"
        ),
        "templates": [{"ns": 10, "title": "Stampa:Country data NonExistentCountryXYZ", "exists": False}],
    }
    issues = find_live_issues(parse_result)
    leak_issues = [i for i in issues if i.kind == "country_data_leak"]
    assert len(leak_issues) == 1
    assert "NonExistentCountryXYZ" in leak_issues[0].message


def test_clean_render_has_no_issues():
    parse_result = {
        "text": '<div class="mw-parser-output"><p>Kjo është prozë krejt normale shqipe.</p></div>',
        "templates": [{"ns": 10, "title": "Stampa:Sfn", "exists": True}],
    }
    assert find_live_issues(parse_result) == []


@pytest.mark.asyncio
async def test_parse_wikitext_and_validate_wikitext_live_end_to_end():
    client = MediaWikiClient(API_URL, "test-agent/1.0")
    try:
        with respx.mock(base_url=API_URL) as mock:
            mock.post(data__contains={"action": "parse"}).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "parse": {
                            "title": "API",
                            "text": (
                                '<strong class="error"><span class="scribunto-error">'
                                "Script error: No such module &quot;X&quot;.</span></strong>"
                            ),
                            "templates": [{"ns": 828, "title": "Moduli:X", "exists": False}],
                        }
                    },
                )
            )
            result = await validate_wikitext_live(client, "{{#invoke:X|foo}}")
    finally:
        await client.aclose()

    assert not result.valid
    kinds = {i.kind for i in result.issues}
    assert "lua_script_error" in kinds
    assert "unexpanded_template" in kinds


