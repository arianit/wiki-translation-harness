from wiki_translation_harness.qa import run_qa_checks

CLEAN = (
    "Text.<ref>{{cite web|url=https://x.org|title=T|language=en}}</ref>\n\n"
    "== Referimet ==\n{{Reflist}}\n"
)


def _kinds(text: str) -> set[str]:
    return {i.kind for i in run_qa_checks(text)}


def test_clean_article_has_no_findings():
    assert run_qa_checks(CLEAN) == []


def test_flags_grc_even_inside_a_language_list():
    assert "qa_language_grc" in _kinds(CLEAN + "{{cite web|url=u|language=en, grc}}")


def test_flags_bare_month_and_albanian_month_dates():
    assert "qa_bare_month_date" in _kinds(CLEAN + "{{cite web|url=u|date=2016-03}}")
    assert "qa_albanian_month_date" in _kinds(CLEAN + "{{cite web|url=u|date=Mars 2008}}")
    assert "qa_bare_month_date" not in _kinds(CLEAN + "{{cite web|url=u|date=2016-03-14}}")


def test_cite_web_without_url_but_chapter_url_alone_is_fine():
    assert "qa_cite_web_no_url" in _kinds(CLEAN + "{{cite web|title=T}}")
    assert "qa_cite_web_no_url" not in _kinds(CLEAN + "{{cite web|chapter-url=https://x|title=T}}")


def test_orphan_url_qualifier_is_not_satisfied_by_archive_url():
    text = CLEAN + "{{cite book|title=T|archive-url=https://web.archive.org/web/2020/https://x|archive-date=2020-01-01}}"
    assert "qa_cite_orphan_url_qualifier" in _kinds(text)
    assert "qa_cite_orphan_url_qualifier" not in _kinds(CLEAN + "{{cite book|url=https://x|access-date=2020-01-01}}")


def test_broken_templates_and_image_option():
    for snippet, kind in [
        ("{{transliteration|hbo|Mōše}}", "qa_broken_transliteration"),
        ("{{tlit|hbo|Mōše}}", "qa_broken_transliteration"),
        ("{{Native name|ar|x}}", "qa_broken_native_name"),
        ("{{cite tweet|user=a|number=1}}", "qa_cite_tweet"),
        ("[[Skeda:X.jpg|miniaturë|caption]]", "qa_miniature_option"),
        ("[[File:X.jpg|miniatura|caption]]", "qa_miniature_option"),
        ("[[File:X.jpg|mini|caption]]", "qa_miniature_option"),
        ("[[File:X.jpg|sinistra|miniatura|caption]]", "qa_miniature_option"),
        ("{{Shih edhe|Foo}}", "qa_shih_edhe_template"),
        ("{{Kryesor|[[Foo]]}}", "qa_hatnote_wrapped_param"),
    ]:
        assert kind in _kinds(CLEAN + snippet), snippet


def test_native_name_list_is_not_flagged():
    assert "qa_broken_native_name" not in _kinds(CLEAN + "{{native name list|tag1=ar|name1=x}}")


def test_sfnmp_duplicate_year_param():
    assert "qa_sfnmp_duplicate_param" in _kinds(CLEAN + "{{Sfnmp|Foo|2020|p=1|2=Smith|2y=Jones|2y=2019|2p=5}}")
    assert "qa_sfnmp_duplicate_param" not in _kinds(
        CLEAN + "{{Sfnmp|Foo|2020|p=1|2=Smith|2last2=Jones|2y=2019|2p=5}}"
    )


def test_missing_reflist_is_anchored_to_the_last_line():
    text = "Text.<ref>{{cite web|url=u|language=en}}</ref>\n\n== Burimet ==\nx"
    (issue,) = [i for i in run_qa_checks(text) if i.kind == "qa_missing_reflist"]
    assert issue.line_number == text.count("\n") + 1
    assert "qa_missing_reflist" not in _kinds("No citations at all.")
