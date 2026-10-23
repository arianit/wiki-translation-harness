from pathlib import Path

from wiki_translation_harness.validator import validate_wikitext

_FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def test_valid_wikitext_passes():
    text = "'''Bold''' [[Link|Display]] {{cite web|title=x}} <ref>foo</ref> {| |-\n| a\n|} <!-- c -->"
    result = validate_wikitext(text)
    assert result.valid
    assert result.issues == []


def test_unbalanced_template_detected():
    result = validate_wikitext("{{cite web|title=x")
    assert not result.valid
    assert any(i.kind == "template" for i in result.issues)


def test_unbalanced_table_detected():
    result = validate_wikitext("{| class=wikitable\n|-\n| a\n")
    assert not result.valid
    assert any(i.kind == "table" for i in result.issues)


def test_template_trailing_pipe_not_flagged_as_table():
    # Regression: a template whose last parameter is empty closes with ``|}}``,
    # whose ``|}`` substring was wrongly counted as a table close, making any
    # chunk containing e.g. {{val|12.6|u=km/s|}} permanently fail validation
    # (Jupiter infobox). {{val|...|}} mid-line and {{Infobox\n|p\n|}} on its
    # own line must both stay balanced.
    text = (
        "{{Short description|planet}}\n"
        "{{Infobox planet\n"
        "| rot_velocity = {{val|12.6|u=km/s|}}\n"
        "| axial_tilt = 3.13°\n"
        "|}}\n"
        "'''Jupiter''' is the fifth planet.\n"
    )
    result = validate_wikitext(text)
    assert result.valid, [i.message for i in result.issues]


def test_dropped_table_close_still_detected_with_templates_present():
    text = "{| class=wikitable\n| cell {{val|1|u=km/s|}}\n| a\n"
    result = validate_wikitext(text)
    assert not result.valid
    assert any(i.kind == "table" for i in result.issues)


def test_unbalanced_ref_detected():
    result = validate_wikitext("text <ref>unterminated")
    assert not result.valid
    assert any(i.kind == "reference" for i in result.issues)


def test_math_verbatim_not_counted_as_template_braces():
    # Regression: <math>\tfrac{...}{...}}</math> ends in "}}" from two single
    # closing braces, not a template close. The raw {{/}} count flagged
    # Neptune's "Physical characteristics" and "Moons" chunks (7 {{ vs 19 }}
    # in the source) as unbalanced templates, failing the whole article
    # even though the model translated correctly. Verbatim tags (math, code,
    # nowiki, …) must be stripped before the brace/bracket pair counts.
    text = (
        "'''Neptune''' has a mass of 1.024{{e|26}}&nbsp;kg.\n"
        ":<math>\\tfrac{M_\\text{Neptune}}{M_\\text{Earth}} = 17.15.</math>\n"
        "Values from {{cite web |title=Fact Sheet }}.\n"
    )
    result = validate_wikitext(text)
    assert result.valid, [i.message for i in result.issues]


def test_heading_merged_onto_prior_line_detected():
    # Regression: a chunk boundary that drops the newline before a heading
    # (e.g. trailing content ending in "-->" immediately followed by "==...==")
    # silently fails to render as a heading in real MediaWiki.
    result = validate_wikitext("some trailing text-->== Referime ==\n{{reflist}}\n")
    assert not result.valid
    assert any(i.kind == "heading" for i in result.issues)


def test_equals_in_template_param_not_flagged_as_heading():
    result = validate_wikitext("Text with {{infobox|param=value}} more text.\n")
    assert result.valid


def test_leaked_attribution_block_detected():
    text = (
        "Some translated prose.\n\n"
        "<!--\nNUK ËSHTË PJESË E ARTIKULLIT — mos e kopjo këtë bllok në faqen kryesore.\n\n"
        "== Për faqen e diskutimit (Talk) ==\n"
        "{{Përkthyer nga|en|Test|1 janar 2024|123}}\n-->"
    )
    result = validate_wikitext(text)
    assert not result.valid
    assert any(i.kind == "leaked_commentary" for i in result.issues)


def test_leaked_fabrication_refusal_detected():
    text = "Nuk ka tekst burimor për të përkthyer në këtë seksion."
    result = validate_wikitext(text)
    assert not result.valid
    assert any(i.kind == "leaked_commentary" for i in result.issues)


def test_degenerate_repetition_loop_detected():
    # Regression: a real chunk (Timeline of Albanian history, "20th
    # century") came back from the model as thousands of near-identical
    # repeated lines instead of a translation — no fixed phrase to match,
    # since the repeated content itself varies run to run, so this needs a
    # content-agnostic detector rather than another string in the marker
    # list. zlib compression ratio catches it: real wikitext (even
    # formatting-heavy tables) compresses to ~30-47% of raw size, while
    # this kind of pathological loop compresses to under 3%.
    text = ("- cite a book in the context (Albanopolis in the list of links in this section) " * 400)
    result = validate_wikitext(text)
    assert not result.valid
    assert any(i.kind == "degenerate_repetition" for i in result.issues)


def test_fixture_convert_malformed_detected():
    result = validate_wikitext(_load_fixture("convert_malformed.wiki"))
    assert not result.valid
    issue = next(i for i in result.issues if i.kind == "convert_malformed")
    assert issue.severity == "error"
    assert issue.line_number == 1
    assert "{{convert" in issue.snippet


def test_fixture_harvc_used_detected():
    result = validate_wikitext(_load_fixture("harvc_used.wiki"))
    assert not result.valid
    issue = next(i for i in result.issues if i.kind == "harvc_used")
    assert issue.severity == "error"
    assert issue.line_number == 3


def test_fixture_sfn_text_param_detected():
    result = validate_wikitext(_load_fixture("sfn_text_param.wiki"))
    assert not result.valid
    issue = next(i for i in result.issues if i.kind == "sfn_unsupported_param")
    assert issue.severity == "warning"
    assert issue.line_number == 1


def test_fixture_table_span_mismatch_simple_detected():
    result = validate_wikitext(_load_fixture("table_span_mismatch_simple.wiki"))
    assert not result.valid
    issue = next(i for i in result.issues if i.kind == "table_span_mismatch")
    assert issue.severity == "warning"
    assert "1990" in issue.snippet


def test_fixture_table_span_mismatch_rowspan_detected():
    # Row 3 ("| Beta") relies on the rowspan="2" cell above it for column 1
    # but is missing its own column-3 value — a mismatch a naive per-row
    # cell count wouldn't catch without honoring the active rowspan.
    result = validate_wikitext(_load_fixture("table_span_mismatch_rowspan.wiki"))
    assert not result.valid
    issue = next(i for i in result.issues if i.kind == "table_span_mismatch")
    assert "Beta" in issue.snippet


def test_fixture_table_span_ok_rowspan_not_flagged():
    # Same rowspan shape as the mismatch fixture, but every row's cell count
    # correctly accounts for the carried-down rowspan — must not false-positive.
    result = validate_wikitext(_load_fixture("table_span_ok_rowspan.wiki"))
    assert result.valid


def test_fixture_table_span_trailing_block_not_flagged():
    # A deliberately narrower trailing sub-block (a summary/notes row pair
    # that doesn't fill every column) spanning two consecutive rows — a real
    # Wikipedia authoring pattern, confirmed via Ashoka's "Edicts of Ashoka"
    # table, which triggers this exact shape on the pristine English source
    # itself (i.e. it's not something translation could have introduced or
    # could ever "fix"). Must not be flagged.
    result = validate_wikitext(_load_fixture("table_span_trailing_block_not_flagged.wiki"))
    assert result.valid


def test_fixture_table_span_first_row_without_delimiter_not_flagged():
    # The first row sits straight after "{|" with no "|-" before it. It used to
    # be dropped, so its rowspan never reached the second row and that row
    # looked one cell short — seen on the English "Periodic table" article's
    # Electron configurations table, where the pristine source tripped it.
    result = validate_wikitext(_load_fixture("table_span_first_row_without_delimiter_not_flagged.wiki"))
    assert result.valid


def test_fixture_clean_valid_has_no_issues():
    # Ordinary sqwiki prose using convert/Sfn/a table correctly — the four
    # new checks must not false-positive on legitimate usage.
    result = validate_wikitext(_load_fixture("clean_valid.wiki"))
    assert result.valid
    assert result.issues == []


def test_normal_long_prose_not_flagged_as_degenerate_repetition():
    # Varied content, including a formatting-heavy table (structurally
    # repetitive but with different data per row, like real wikitext) —
    # must stay well clear of the compression-ratio threshold.
    rows = "\n".join(f"| {2000 + i} || Ngjarje e rëndësishme numër {i} në histori || fq. {i * 3}" for i in range(80))
    text = (
        "'''Bardhyli''' ishte një mbret ilir i shekullit IV p.e.s. Ai qeverisi mbretërinë Dardane "
        "dhe u përfshi në luftëra kundër Maqedonisë antike nën udhëheqjen e Filipit II.\n\n"
        "== Ngjarjet ==\n{| class=\"wikitable\"\n" + rows + "\n|}\n"
    )
    result = validate_wikitext(text)
    assert result.valid


def test_balance_findings_preserve_lines_after_verbatim_blocks():
    result = validate_wikitext('<math>\n[[ignored]]\n</math>\nClean [[link]].\nextra]]\n')
    issue = next(i for i in result.issues if i.kind == 'link')
    assert issue.line_number == 5
    assert 'extra' not in issue.snippet
    assert issue.snippet.startswith(']]')


def test_unmatched_ref_reports_owning_line():
    result = validate_wikitext('Clean <ref>citation</ref>\nExtra </ref>')
    issue = next(i for i in result.issues if i.kind == 'reference')
    assert issue.line_number == 2
    assert issue.snippet == '</ref>'
