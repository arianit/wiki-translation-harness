from pathlib import Path

from wiki_translation_harness.models import ArticleSource, Chunk, ChunkStatus, ValidationIssue
from wiki_translation_harness.report import (
    ArticleReportData,
    build_article_report,
    build_attribution_block,
    report_path_for,
    save_report,
)
from wiki_translation_harness.verification import VerifiedFacts


def _source(**overrides) -> ArticleSource:
    base = dict(title="Andrea III Muzaka", wikitext="text", revid=12345, revision_timestamp="2024-09-20T00:00:00Z", source_lang="en")
    base.update(overrides)
    return ArticleSource(**base)


def _chunk(status=ChunkStatus.TRANSLATED) -> Chunk:
    return Chunk(
        article_title="Andrea III Muzaka",
        section_titles=["Lead"],
        order=0,
        text="x",
        token_estimate=1,
        status=status,
    )


def test_report_includes_source_metadata():
    data = ArticleReportData(source=_source(), chunks=[_chunk()], facts=VerifiedFacts())
    report = build_article_report(data, "assembled text")
    assert "Andrea III Muzaka" in report
    assert "12345" in report
    assert "2024-09-20T00:00:00Z" in report


def test_report_flags_rewrite_when_target_article_exists():
    facts = VerifiedFacts(existing_target_title="Bardhyli", sibling_links=["Ilirët", "Mbretëria e Ilirisë"])
    data = ArticleReportData(source=_source(title="Bardylis"), chunks=[_chunk()], facts=facts)
    report = build_article_report(data, "text")
    assert "REWRITE" in report
    assert "Bardhyli" in report
    assert "Ilirët" in report


def test_report_lists_confirmed_and_missing_links():
    facts = VerifiedFacts(links={"Muzaka family": "Muzakajt", "Nonexistent": None})
    data = ArticleReportData(source=_source(), chunks=[_chunk()], facts=facts)
    report = build_article_report(data, "text")
    assert "Muzaka family" in report
    assert "Muzakajt" in report
    assert "Nonexistent" in report
    assert "Not found" in report


def test_report_lists_template_params():
    facts = VerifiedFacts(
        templates={"Infobox royalty": "Stampa:Infobox royalty"},
        template_params={"Infobox royalty": ["name", "title"]},
    )
    data = ArticleReportData(source=_source(), chunks=[_chunk()], facts=facts)
    report = build_article_report(data, "text")
    assert "Infobox royalty" in report
    assert "`name`" in report
    assert "`title`" in report


def test_report_language_breakdown_from_citations():
    assembled = (
        "{{cite web|title=x|language=en}} {{cite book|title=y|language=sr}} "
        "{{cite web|title=z|language=en}}"
    )
    data = ArticleReportData(source=_source(), chunks=[_chunk()], facts=VerifiedFacts())
    report = build_article_report(data, assembled)
    assert "`en`: 2" in report
    assert "`sr`: 1" in report


def test_report_flags_repaired_sections():
    chunks = [_chunk(ChunkStatus.TRANSLATED), _chunk(ChunkStatus.REPAIRED)]
    data = ArticleReportData(source=_source(), chunks=chunks, facts=VerifiedFacts())
    report = build_article_report(data, "text")
    assert "Choices to review" in report
    assert "1 of 2 section(s)" in report


def test_report_lists_auto_filled_citation_languages():
    data = ArticleReportData(
        source=_source(),
        chunks=[_chunk()],
        facts=VerifiedFacts(),
        citation_languages_filled={"Some Page": "en", "Neka Strana": "sr"},
    )
    report = build_article_report(data, "text")
    assert "Citation languages auto-filled" in report
    assert "`en` — Some Page" in report
    assert "`sr` — Neka Strana" in report


def test_attribution_block_matches_skill_worked_example():
    source = _source(
        title="Archimedes", revid=1366588187, revision_timestamp="2026-07-28T18:48:49Z", source_lang="en"
    )
    block = build_attribution_block(source)
    assert block == (
        "<!--\n"
        "NUK ËSHTË PJESË E ARTIKULLIT — mos e kopjo këtë bllok në faqen kryesore.\n\n"
        "== Për faqen e diskutimit (Talk) ==\n"
        "{{Përkthyer nga|en|Archimedes|28 korrik 2026|1366588187}}\n\n"
        "== Përmbledhja e redaktimit ==\n"
        "Përkthyer nga anglishtja, sipas artikullit en:Archimedes, versioni i datës "
        "28 korrik 2026 (revizioni 1366588187)\n"
        "-->"
    )


def test_attribution_block_none_for_non_english_source():
    source = _source(source_lang="sq")
    assert build_attribution_block(source) is None


def test_attribution_block_all_months():
    # spot-check month boundaries, not just July
    jan = _source(revision_timestamp="2024-01-05T00:00:00Z")
    dec = _source(revision_timestamp="2024-12-25T00:00:00Z")
    assert "5 janar 2024" in build_attribution_block(jan)
    assert "25 dhjetor 2024" in build_attribution_block(dec)


def test_save_report_writes_file(tmp_path: Path):
    path = save_report(tmp_path, "Andrea III Muzaka", "report body")
    assert path == report_path_for(tmp_path, "Andrea III Muzaka")
    assert path.read_text(encoding="utf-8") == "report body"
    assert path.name == "Andrea_III_Muzaka.report.md"


def test_report_semantic_review_section_with_findings():
    issue = ValidationIssue(kind="grammar_case", message="Wrong case.", snippet="Parisi eshte")
    data = ArticleReportData(
        source=_source(),
        chunks=[_chunk()],
        facts=VerifiedFacts(),
        review_model="claude-sonnet-5",
        review_findings=[issue],
    )
    report = build_article_report(data, "text")
    assert "Semantic review" in report
    assert "claude-sonnet-5" in report
    assert "Wrong case." in report
    assert "Parisi eshte" in report
