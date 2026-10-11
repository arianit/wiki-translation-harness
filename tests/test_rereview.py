from types import SimpleNamespace

from wiki_translation_harness.output import assemble_chunks
from wiki_translation_harness.queue_runner import _status_with_engine
from wiki_translation_harness.rereview import (
    _REPORT_SECTION_RE,
    revid_from_attribution,
    split_attribution,
    split_into_stand_in_chunks,
)

ATTRIBUTION = (
    "\n\n<!--\nNUK ËSHTË PJESË E ARTIKULLIT — mos e kopjo këtë bllok në faqen kryesore.\n\n"
    "== Për faqen e diskutimit (Talk) ==\n{{Përkthyer nga|en|Venus|19 shtator 2026|1375690330}}\n-->"
)
BODY = "Lead.\n\n== A ==\nText {{x|\n\ny}}\n\n=== B ===\nMore.\n\n[[Kategoria:X]]"


def test_split_attribution_and_revid():
    body, attribution = split_attribution(BODY + ATTRIBUTION)
    assert body == BODY
    assert body + attribution == BODY + ATTRIBUTION
    assert revid_from_attribution(attribution) == 1375690330


def test_stand_in_chunks_round_trip_at_headings():
    chunks = split_into_stand_in_chunks("T", BODY, "en")
    assert assemble_chunks(chunks) == BODY
    assert [c.section_titles[0] for c in chunks] == ["(lead)", "A", "B"]


def test_report_section_regex_stops_before_attribution():
    text = "# R\n\n## Semantic review\n\nReviewed.\n\n<!--\nNUK ËSHTË PJESË"
    assert _REPORT_SECTION_RE.search(text).group(0).endswith("Reviewed.\n")


def test_status_marks_review_model_on_done_only():
    stats = SimpleNamespace(stats=SimpleNamespace(
        provider="codex", model="gpt-6-luna", review_model="claude-sonnet-5-5", review_provider="claude_code",
    ))
    config = SimpleNamespace(provider="codex", model="gpt-6-luna")
    assert _status_with_engine("DONE", stats, config) == "DONE\tgpt-6-luna@codex\treview=claude-sonnet-5-5@claude_code"
    assert _status_with_engine("FAILED", stats, config) == "FAILED\tgpt-6-luna@codex"
