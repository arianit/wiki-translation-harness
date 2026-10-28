from pathlib import Path

import orjson

from wiki_translation_harness.models import ValidationIssue
from wiki_translation_harness.review_queue import (
    record_needs_human_review,
    review_path_for,
)


def _issue(**kw) -> ValidationIssue:
    defaults = dict(kind="harvc_used", message="{{harvc}} is broken")
    defaults.update(kw)
    return ValidationIssue(**defaults)


def test_writes_review_markdown(tmp_path: Path):
    path = record_needs_human_review(tmp_path, "Test Article", [_issue(line_number=5, snippet="{{harvc|x}}")], 3)
    assert path == review_path_for(tmp_path, "Test Article")
    text = path.read_text(encoding="utf-8")
    assert "Test Article" in text
    assert "3 assembly-level repair round" in text
    assert "harvc" in text
    assert "5" in text


def test_index_upserts_by_title_not_duplicates(tmp_path: Path):
    record_needs_human_review(tmp_path, "Test Article", [_issue()], 3)
    record_needs_human_review(tmp_path, "Test Article", [_issue(message="different issue this time")], 2)
    index = orjson.loads((tmp_path / "needs_human_review.json").read_bytes())
    assert len(index) == 1
    assert index[0]["repair_rounds"] == 2
    assert index[0]["findings"][0]["explanation"] == "different issue this time"
