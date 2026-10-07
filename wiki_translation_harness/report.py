"""End-of-run Report, per the skill's own "Report at the end" section:
link targets checked, templates confirmed, citation language breakdown,
what needed repair, and an attribution reminder. Built entirely from data
the harness collected mechanically — no translation-quality judgment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from wiki_translation_harness.models import ArticleSource, Chunk, ChunkStatus, ValidationIssue
from wiki_translation_harness.output import sanitize_filename
from wiki_translation_harness.verification import VerifiedFacts

_CITATION_LANGUAGE_RE = re.compile(r"\|\s*language\s*=\s*([a-zA-Z, -]+)")

# Albanian month names, quoted verbatim from the skill's own "Attribution is
# required" section — a fixed calendar fact, not a translation judgment call.
_ALBANIAN_MONTHS = [
    "janar", "shkurt", "mars", "prill", "maj", "qershor",
    "korrik", "gusht", "shtator", "tetor", "nëntor", "dhjetor",
]


def _format_albanian_date(iso_timestamp: str) -> str | None:
    """'DD Month YYYY' per the skill's {{Përkthyer nga}} date parameter —
    a rendered sentence, not a citation date field, so no ISO/zero-padding."""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return f"{dt.day} {_ALBANIAN_MONTHS[dt.month - 1]} {dt.year}"


def build_attribution_block(source: ArticleSource) -> str | None:
    """Ready-to-paste block in the skill's own documented format (its
    "Attribution is required" section) — ready to copy onto the article's
    Talk page and into the edit summary. Only defined for source_lang "en",
    which is what this skill (and its Albanian prose here) is written for;
    a different source language would need the skill's own equivalent
    wording, not a harness-invented substitute.
    """
    if source.source_lang != "en" or not source.revid or not source.revision_timestamp:
        return None
    albanian_date = _format_albanian_date(source.revision_timestamp)
    if not albanian_date:
        return None
    return (
        "<!--\n"
        "NUK ËSHTË PJESË E ARTIKULLIT — mos e kopjo këtë bllok në faqen kryesore.\n\n"
        "== Për faqen e diskutimit (Talk) ==\n"
        f"{{{{Përkthyer nga|en|{source.title}|{albanian_date}|{source.revid}}}}}\n\n"
        "== Përmbledhja e redaktimit ==\n"
        f"Përkthyer nga anglishtja, sipas artikullit en:{source.title}, "
        f"versioni i datës {albanian_date} (revizioni {source.revid}).\n"
        "-->"
    )


@dataclass
class ArticleReportData:
    source: ArticleSource
    chunks: list[Chunk]
    facts: VerifiedFacts
    citation_languages_filled: dict[str, str] = field(default_factory=dict)
    # Comments the model wrote itself (comments.reconcile_comments), removed
    # from the article.
    removed_comments: list[str] = field(default_factory=list)
    # Set only when the run's semantic review pass (pipeline.run_review_pass)
    # was enabled for this article. review_findings is whatever remained
    # unresolved after its repair-attempt cap -- empty means either the
    # review pass found nothing, or every finding got fixed.
    review_model: str | None = None
    review_findings: list[ValidationIssue] = field(default_factory=list)


def _language_breakdown(translated_text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in _CITATION_LANGUAGE_RE.finditer(translated_text):
        for lang in m.group(1).split(","):
            lang = lang.strip()
            if lang:
                counts[lang] = counts.get(lang, 0) + 1
    return counts


def build_article_report(data: ArticleReportData, assembled_text: str) -> str:
    source = data.source
    facts = data.facts
    lines: list[str] = [f"# Report: {source.title}\n"]

    lines.append("## Source\n")
    lines.append(f"- Title: `{source.title}` (source language: `{source.source_lang}`)")
    if source.revid:
        rev_line = f"- Revision: `{source.revid}`"
        if source.revision_timestamp:
            rev_line += f" (`{source.revision_timestamp}`)"
        lines.append(rev_line)
    lines.append("")

    if facts.existing_target_title:
        lines.append("## ⚠ This is a REWRITE, not a first translation\n")
        lines.append(
            f"The target wiki already has an article at **`{facts.existing_target_title}`**. "
            "The saved output here is a fresh translation and does *not* merge with, or account "
            "for, the existing article's content, infobox, or categories — compare the two "
            "before replacing anything live.\n"
        )
        if facts.sibling_links:
            lines.append(
                "Wikilink targets already used in that existing article (passed to the "
                "translation as established terminology for this topic):\n"
            )
            lines.append(", ".join(f"`{link}`" for link in facts.sibling_links))
        lines.append("")

    lines.append("## Link targets checked\n")
    confirmed_links = {k: v for k, v in facts.links.items() if v}
    missing_links = sorted(k for k, v in facts.links.items() if v is None)
    if confirmed_links:
        lines.append("| Source title | Confirmed target title |")
        lines.append("|---|---|")
        for k, v in sorted(confirmed_links.items()):
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("_None confirmed._")
    lines.append("")
    if missing_links:
        lines.append(
            f"**Not found on the target wiki ({len(missing_links)}):** "
            + ", ".join(f"`{m}`" for m in missing_links)
        )
        lines.append("")

    lines.append("## Templates checked\n")
    confirmed_templates = {k: v for k, v in facts.templates.items() if v}
    missing_templates = sorted(k for k, v in facts.templates.items() if v is None)
    if confirmed_templates:
        lines.append("| Source template | Confirmed target template | Parameters confirmed |")
        lines.append("|---|---|---|")
        for k, v in sorted(confirmed_templates.items()):
            params = facts.template_params.get(k)
            param_note = f"{len(params)} (see below)" if params else "not fetched"
            lines.append(f"| {{{{{k}}}}} | {{{{{v}}}}} | {param_note} |")
    else:
        lines.append("_None confirmed._")
    lines.append("")
    if missing_templates:
        lines.append(
            f"**Not found on the target wiki ({len(missing_templates)}):** "
            + ", ".join(f"`{{{{{m}}}}}`" for m in missing_templates)
        )
        lines.append("")

    if facts.template_params:
        lines.append("## Confirmed template parameters\n")
        lines.append(
            "Use these exact names — several target-wiki infobox templates keep "
            "English parameter names with only internal labels localized; inventing "
            "translated parameter names silently drops the field on render.\n"
        )
        for name, params in sorted(facts.template_params.items()):
            lines.append(f"- `{{{{{name}}}}}`: " + ", ".join(f"`{p}`" for p in params))
        lines.append("")

    lines.append("## Citation language breakdown\n")
    breakdown = _language_breakdown(assembled_text)
    if breakdown:
        for lang, count in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{lang}`: {count}")
    else:
        lines.append("_No `|language=` tags found in citations._")
    lines.append("")

    if data.citation_languages_filled:
        lines.append("## Citation languages auto-filled\n")
        lines.append(
            "The translation left these citations without `|language=`; the harness "
            "filled them in by visiting the cited URL (or, failing that, guessing from "
            "the citation title) — worth a spot check, not a guaranteed-correct judgment call.\n"
        )
        for title, lang in sorted(data.citation_languages_filled.items()):
            lines.append(f"- `{lang}` — {title}")
        lines.append("")

    if data.removed_comments:
        lines.append("## Model notes removed from the article\n")
        lines.append(
            "The model left these as HTML comments; none matches a comment in the "
            "source, so they were removed. Some say what it left out or couldn't fix.\n"
        )
        for body in data.removed_comments:
            lines.append(f"- {' '.join(body.split())}")
        lines.append("")

    repaired = [c for c in data.chunks if c.status == ChunkStatus.REPAIRED]
    if repaired:
        lines.append("## Choices to review\n")
        lines.append(
            f"- {len(repaired)} of {len(data.chunks)} section(s) needed a syntax repair "
            "pass before validating cleanly — worth a manual skim of those sections."
        )
        lines.append("")

    if data.review_model:
        lines.append("## Semantic review\n")
        if data.review_findings:
            lines.append(
                f"Reviewed by `{data.review_model}`; {len(data.review_findings)} finding(s) "
                "remained unresolved after the review repair-attempt cap — spot-check before "
                "publishing.\n"
            )
            lines.append("| severity | explanation | snippet |")
            lines.append("|---|---|---|")
            for issue in data.review_findings:
                finding = issue.as_finding()
                snippet = (finding["snippet"] or "").replace("|", "\\|").replace("\n", " ")
                explanation = finding["explanation"].replace("|", "\\|")
                lines.append(f"| {finding['severity']} | {explanation} | `{snippet}` |")
        else:
            lines.append(f"Reviewed by `{data.review_model}` — no unresolved findings.")
        lines.append("")

    attribution_block = build_attribution_block(source)
    if attribution_block:
        lines.append(attribution_block)
    else:
        lines.append("## Attribution\n")
        lines.append(
            "_Could not build the attribution block (needs source_lang `en`, a "
            "revision ID, and a revision timestamp)._ Facts on hand:\n"
        )
        lines.append(f"- Source language: `{source.source_lang}`")
        lines.append(f"- Source title: `{source.title}`")
        if source.revid:
            lines.append(f"- Revision ID: `{source.revid}`")
        if source.revision_timestamp:
            lines.append(f"- Revision timestamp (ISO): `{source.revision_timestamp}`")
    lines.append("")

    return "\n".join(lines)


def report_path_for(output_dir: Path, title: str) -> Path:
    return output_dir / f"{sanitize_filename(title)}.report.md"


def save_report(output_dir: Path, title: str, report_text: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = report_path_for(output_dir, title)
    path.write_text(report_text, encoding="utf-8")
    return path
