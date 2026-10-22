"""Mandatory pre-delivery QA gate: the deterministic half of the wikiqa skill.

wikiqa's checklist is mostly grep-able defect patterns -- known sqwiki bugs
that render as live errors but pass every structural check in validator.py.
Interactive use runs those greps by hand before delivery; this module runs
the same checks on every assembled article, unconditionally, so a defect
can't reach a saved file just because nobody (or no prompt) remembered to
look. Findings are ordinary ValidationIssues (kind prefixed ``qa_``), so
run_assembly_repair localizes and repairs them like any other finding.

Checks already covered elsewhere (``{{Harvc}}``, ``<ref>{{sfn}}</ref>``
double-wrap, orphaned named refs, missing ``|language=``) are not repeated
here -- validator.py / post_process_assembled own those.
"""

from __future__ import annotations

import re

from wiki_translation_harness.models import ValidationIssue
from wiki_translation_harness.validator import _line_number, _snippet

_ALBANIAN_MONTHS = "janar|shkurt|mars|prill|maj|qershor|korrik|gusht|shtator|tetor|nëntor|dhjetor"

_CITE_RE = re.compile(r"\{\{\s*[Cc]ite [a-z]+\s*\|[^{}]*\}\}|\{\{\s*[Cc]itation\s*\|[^{}]*\}\}")
_CITE_WEB_RE = re.compile(r"\{\{\s*[Cc]ite web\s*\|[^{}]*\}\}")
_URL_RE = re.compile(r"(?<![-a-zA-Z])url\s*=\s*\S|chapter-url\s*=\s*\S")
_URL_QUALIFIER_RE = re.compile(r"\b(access-date|archive-url|archive-date)\s*=")
_SFNMP_RE = re.compile(r"\{\{\s*[Ss]fnmp\s*\|[^{}]*\}\}")
_HAS_REFS_RE = re.compile(r"<ref\b|\{\{\s*(?:sfn[a-z]*|harvnb)\s*\|", re.IGNORECASE)
_HAS_REFLIST_RE = re.compile(r"\{\{\s*reflist\b|<references\b", re.IGNORECASE)

# (kind, pattern, message). Each match is one issue.
_PATTERN_CHECKS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "qa_language_grc",
        re.compile(r"language\s*=[a-zA-Z, -]*grc[a-zA-Z, -]*"),
        "|language= containing grc crashes sqwiki's CS1 module; remove the grc token (keep any other language)",
    ),
    (
        "qa_bare_month_date",
        re.compile(r"\b(?:date|archive-date|access-date)\s*=\s*\d{4}-\d{2}(?![-\d])"),
        "bare YYYY-MM date crashes CS1 date parsing; use the original English 'Month YYYY' form for month-only dates",
    ),
    (
        "qa_albanian_month_date",
        re.compile(
            rf"\|\s*(?:date|archive-date|access-date)\s*=\s*(?:{_ALBANIAN_MONTHS})\s+\d{{4}}\b", re.IGNORECASE
        ),
        "Albanian month name in a month-only date crashes CS1; keep the English 'Month YYYY' form",
    ),
    (
        "qa_shih_edhe_template",
        re.compile(r"\{\{\s*Shih edhe\s*\|"),
        "{{Shih edhe}} does not exist on sqwiki; use {{Shiko gjithashtu}}",
    ),
    (
        "qa_hatnote_wrapped_param",
        re.compile(r"\{\{\s*(?:Main|Kryesor|See also|Shiko gjithashtu)\s*\|[^{}\n]*(?:\[\[|\{\{\s*ill)", re.IGNORECASE),
        "hatnote parameter must be a bare title, not [[bracketed]] or a nested {{ill}}; "
        "use {{Hatnote|Artikulli kryesor: {{ill|...}}}} for a target that needs {{ill}}",
    ),
    (
        "qa_broken_transliteration",
        re.compile(r"\{\{\s*(?:transliteration|tlit)\s*\|", re.IGNORECASE),
        "{{transliteration}}/{{tlit}} are broken on sqwiki; use {{lang|code-Latn|text}}",
    ),
    (
        "qa_broken_native_name",
        re.compile(r"\{\{\s*native name\s*\|", re.IGNORECASE),
        "{{native name}} (singular) throws on every use; use {{lang|code|text}} "
        "({{native name list}} is fine)",
    ),
    (
        "qa_cite_tweet",
        re.compile(r"\{\{\s*cite tweet\b", re.IGNORECASE),
        "{{cite tweet}} is broken on sqwiki; convert to {{cite web}} with the reconstructed tweet URL",
    ),
    (
        "qa_miniature_option",
        re.compile(r"\|(?:miniaturë|miniatura|mini|sinistra)(?=\||\]\])"),
        "'miniaturë'/'miniatura'/'mini'/'sinistra' are not image option aliases on sqwiki and silently drop the thumbnail frame or alignment; use |thumb| and |left|",
    ),
]


def _scan(text: str, kind: str, pattern: re.Pattern[str], message: str) -> list[ValidationIssue]:
    return [
        ValidationIssue(
            kind=kind,
            message=message,
            line_number=_line_number(text, m.start()),
            snippet=_snippet(m.group(0)),
        )
        for m in pattern.finditer(text)
    ]


def run_qa_checks(text: str) -> list[ValidationIssue]:
    """Runs every deterministic wikiqa check over the assembled article."""
    issues: list[ValidationIssue] = []
    for kind, pattern, message in _PATTERN_CHECKS:
        issues.extend(_scan(text, kind, pattern, message))

    for m in _CITE_WEB_RE.finditer(text):
        if not _URL_RE.search(m.group(0)):
            issues.append(
                ValidationIssue(
                    kind="qa_cite_web_no_url",
                    message="{{cite web}} without |url= throws a live error; recover the url from the source "
                    "revision or the archive-url path",
                    line_number=_line_number(text, m.start()),
                    snippet=_snippet(m.group(0)),
                )
            )

    for m in _CITE_RE.finditer(text):
        call = m.group(0)
        if _URL_QUALIFIER_RE.search(call) and not _URL_RE.search(call):
            issues.append(
                ValidationIssue(
                    kind="qa_cite_orphan_url_qualifier",
                    message="citation has |access-date=/|archive-url=/|archive-date= but no |url= or "
                    "|chapter-url=; restore the url (it is the path after the timestamp in archive-url) "
                    "or drop the orphaned access-date",
                    line_number=_line_number(text, m.start()),
                    snippet=_snippet(call),
                )
            )

    for m in _SFNMP_RE.finditer(text):
        if len(re.findall(r"\|\s*\d*y\s*=", m.group(0))) != len(
            set(re.findall(r"\|\s*(\d*y)\s*=", m.group(0)))
        ):
            issues.append(
                ValidationIssue(
                    kind="qa_sfnmp_duplicate_param",
                    message="{{Sfnmp}} has a duplicated year parameter; a second author's surname belongs in "
                    "2last2=, a duplicate 2y= silently drops that author",
                    line_number=_line_number(text, m.start()),
                    snippet=_snippet(m.group(0)),
                )
            )

    if _HAS_REFS_RE.search(text) and not _HAS_REFLIST_RE.search(text):
        # Anchored to the last line so the repair loop hands it to the
        # final chunk, where end matter lives, instead of broadcasting.
        issues.append(
            ValidationIssue(
                kind="qa_missing_reflist",
                message="article has <ref>/{{sfn}} citations but no {{Reflist}}; add it in the references "
                "section at the end (after any Shënime section)",
                line_number=text.count("\n") + 1,
            )
        )

    return issues
