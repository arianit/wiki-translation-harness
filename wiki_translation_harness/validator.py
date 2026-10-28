"""Post-translation wikitext syntax validation.

Mechanical structural checks only — never semantic/translation-quality
checks. Uses mwparserfromhell to catch hard parse failures, plus explicit
marker-balance counts for the categories the spec calls out (templates,
links, tables, references, comments), since mwparserfromhell's parser is
deliberately lenient and silently treats an unmatched brace/bracket as plain
text rather than raising — the raw counts are what actually surface a
translation that dropped or duplicated a delimiter.
"""

from __future__ import annotations

import re
import zlib

import mwparserfromhell as mwp

from wiki_translation_harness.models import ValidationIssue, ValidationResult

_REF_OPEN_RE = re.compile(r"<ref\b[^>]*>", re.IGNORECASE)
_REF_CLOSE_RE = re.compile(r"</ref\s*>", re.IGNORECASE)
_HEADING_LIKE_RE = re.compile(r"(={2,6})([^\n=]+)\1[ \t]*(?=\n|$)")

# Content inside these tags is verbatim/escaped wikitext (LaTeX math, chemical
# formulas, code samples) and legitimately contains brace substrings that are
# not template delimiters — e.g. ``<math>\tfrac{M_\text{Neptune}}{M_\text{Earth}}
# </math>`` ends in ``}}`` from two single closing braces, not one template
# close. Stripping these spans before the raw ``{{``/``}}``, ``[[``/``]]`` and
# ``<!--``/``-->`` counts prevents a false-positive ``Unbalanced template`` on
# any chunk containing math (confirmed on Neptune's "Physical characteristics"
# and "Moons" sections, which are brace-balanced in the source but produce
# 7 ``{{`` vs 19 ``}}`` because of ``\tfrac`` LaTeX). Not used for the table
# check, which already masks templates via mwparserfromhell and whose ``{|``/``|}``
# delimiters don't occur inside math/code.
_VERBATIM_TAG_RE = re.compile(
    r"<(math|chem|nowiki|code|pre|syntaxhighlight|source|tt)\b[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_verbatim_spans(text: str) -> str:
    # Preserve offsets and line numbers for localized repair findings.
    return _VERBATIM_TAG_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)

_SNIPPET_MAX = 120


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _snippet(s: str) -> str:
    s = s.strip().replace("\n", " ")
    return s if len(s) <= _SNIPPET_MAX else s[: _SNIPPET_MAX - 1] + "…"

# Confirmed in practice, twice: a chunk given little or no real source
# content (see the parser.py fix for the near-empty-trailing-chunk case
# that caused this) didn't just echo back what little it had — the model
# fabricated a plausible-looking replacement section, and separately kept
# leaking the skill's whole-article "end of file" attribution block into
# individual per-chunk output despite the invocation frame telling it not
# to (skill_loader._INVOCATION_FRAME). Both produce content with zero
# legitimate reason to appear in a chunk's translated wikitext, so they're
# checked as exact/near-exact known strings — mechanical detection, not a
# judgment call about translation quality.
_LEAKED_META_MARKERS = [
    "NUK ËSHTË PJESË E ARTIKULLIT",
    "Për faqen e diskutimit (Talk)",
    "{{Përkthyer nga|",
    "Nuk ka tekst burimor",
    "Ju lutem siguroni tekstin",
    "as an ai language model",
    "i cannot translate",
    "i don't have access to",
    "there is no source text",
    "please provide the text",
    "the provided wikitext appears to be",
    "i'll provide a clean version",
    "doesn't form meaningful content",
]


def _check_pair(text: str, open_tok: str, close_tok: str, kind: str, issues: list[ValidationIssue]) -> None:
    o = text.count(open_tok)
    c = text.count(close_tok)
    if o != c:
        stack = []
        unmatched = []
        for match in re.finditer(re.escape(open_tok) + "|" + re.escape(close_tok), text):
            if match.group() == open_tok:
                stack.append(match.start())
            elif stack:
                stack.pop()
            else:
                unmatched.append(match.start())
        offsets = unmatched + stack
        offset = min(offsets) if offsets else 0
        issues.append(
            ValidationIssue(
                kind=kind,
                message=f"Unbalanced {kind}: {o} occurrences of '{open_tok}' vs {c} of '{close_tok}'",
                line_number=_line_number(text, offset) if kind != "table" else None,
                snippet=_snippet(text[offset:offset + _SNIPPET_MAX]),
            )
        )


def _check_table_pair(text: str, issues: list[ValidationIssue]) -> None:
    """Count wikitext table delimiters ``{|`` / ``|}`` while ignoring the
    ``|}`` substring that a template close produces when its last parameter
    is empty (``|}}``), e.g. ``{{val|12.6|u=km/s|}}``.

    mwparserfromhell parses tables leniently — an unmatched ``{|`` silently
    becomes plain text rather than a Tag node, so counting table tags alone
    can't detect a dropped ``|}`` close. But it parses templates reliably, so
    we strip template spans first: a real ``|}`` table close is never inside
    a ``{{...}}`` span in source wikitext, while every false-positive
    ``|}}`` is. Templates nested inside a table cell are removed too (the
    recursive filter walks into Tag nodes), and a nested template whose
    parent was already removed is simply skipped."""
    try:
        code = mwp.parse(text)
        for tpl in code.filter_templates():
            try:
                code.remove(tpl)
            except ValueError:
                pass  # already removed together with its parent template
        masked = str(code)
    except Exception:
        masked = text  # parse failed outright; a parse_error issue is added elsewhere
    _check_pair(masked, "{|", "|}", "table", issues)


def _check_refs(text: str, issues: list[ValidationIssue]) -> None:
    opens = _REF_OPEN_RE.findall(text)
    non_self_closing = [o for o in opens if not o.rstrip().endswith("/>")]
    closes = _REF_CLOSE_RE.findall(text)
    if len(non_self_closing) != len(closes):
        stack = []
        unmatched = []
        for match in re.finditer(r"<ref\b[^>]*>|</ref\s*>", text, re.IGNORECASE):
            tag = match.group().lower()
            if tag.startswith("</"):
                if stack:
                    stack.pop()
                else:
                    unmatched.append(match.start())
            elif not tag.rstrip().endswith("/>"):
                stack.append(match.start())
        offset = min(unmatched + stack)
        issues.append(
            ValidationIssue(
                kind="reference",
                message=(
                    f"Unbalanced <ref> tags: {len(non_self_closing)} opening vs "
                    f"{len(closes)} closing </ref>"
                ),
                line_number=_line_number(text, offset),
                snippet=_snippet(text[offset:offset + _SNIPPET_MAX]),
            )
        )


def _check_headings_at_line_start(text: str, issues: list[ValidationIssue]) -> None:
    """A `==Heading==`-shaped run not at the start of a line silently fails to
    render as a heading — MediaWiki only recognizes it at line start, and so
    does mwparserfromhell (it just falls back to plain Text, so parsing
    alone won't surface this). Scanned directly on raw text rather than via
    mwparserfromhell's node tree, since a merged/broken heading is exactly
    the case where mwparserfromhell does *not* produce a Heading node."""
    for match in _HEADING_LIKE_RE.finditer(text):
        start = match.start()
        if start > 0 and text[start - 1] != "\n":
            issues.append(
                ValidationIssue(
                    kind="heading",
                    message=f"{match.group(0).strip()!r} does not start at the beginning of a line",
                )
            )


def _check_leaked_meta_commentary(text: str, issues: list[ValidationIssue]) -> None:
    lowered = text.lower()
    for marker in _LEAKED_META_MARKERS:
        if marker.lower() in lowered:
            issues.append(
                ValidationIssue(
                    kind="leaked_commentary",
                    message=f"Output contains {marker!r} — model commentary/meta-text instead of translated wikitext",
                )
            )


# A degenerate repetition loop (the model gets stuck echoing the same line
# or short phrase thousands of times) is a distinct failure mode from
# leaked commentary — no fixed phrase to match against, since the repeated
# content itself varies. Confirmed in practice on a real large chunk: real
# translated wikitext (including large, formatting-heavy tables) compresses
# to roughly 30-47% of its raw size, while two independently observed
# degenerate loops compressed to under 3%. zlib compression ratio is a
# language-agnostic, content-agnostic way to catch this mechanically.
_MIN_LEN_FOR_REPETITION_CHECK = 500
_MAX_COMPRESSION_RATIO = 0.15


def _check_degenerate_repetition(text: str, issues: list[ValidationIssue]) -> None:
    raw = text.encode("utf-8")
    if len(raw) < _MIN_LEN_FOR_REPETITION_CHECK:
        return
    ratio = len(zlib.compress(raw, level=9)) / len(raw)
    if ratio < _MAX_COMPRESSION_RATIO:
        issues.append(
            ValidationIssue(
                kind="degenerate_repetition",
                message=(
                    f"Output compresses to {ratio:.1%} of its raw size — "
                    "a pathological repetition loop, not real translated content"
                ),
            )
        )


_CONVERT_NUMERIC_RE = re.compile(r"^[\d.,\-–—/±\s]+$")
_SFN_FAMILY = {"sfn", "sfnp", "sfnm"}


def _check_template_based_issues(text: str, issues: list[ValidationIssue]) -> None:
    """One pass over the template tree for every mwparserfromhell-based
    check, sharing a single left-to-right search cursor so repeated,
    byte-identical templates each resolve to their own (not the first
    occurrence's) line number."""
    cursor = 0
    for template in mwp.parse(text).filter_templates():
        raw = str(template)
        offset = text.find(raw, cursor)
        if offset == -1:
            offset = text.find(raw)  # fallback: template inside a template, out of order
        if offset != -1:
            cursor = offset + 1
        line = _line_number(text, offset) if offset != -1 else None
        name = template.name.strip().lower()

        if name == "convert" and template.params:
            first_val = template.params[0].value.strip()
            if first_val and not _CONVERT_NUMERIC_RE.match(first_val):
                issues.append(
                    ValidationIssue(
                        kind="convert_malformed",
                        message=f"{{{{convert}}}} first argument {first_val!r} is not numeric",
                        line_number=line,
                        snippet=_snippet(raw),
                    )
                )

        elif name == "harvc":
            issues.append(
                ValidationIssue(
                    kind="harvc_used",
                    message="{{harvc}} is broken on sqwiki (Moduli:Harvc is missing) — "
                    "expand into a standalone {{Cite book}} instead",
                    line_number=line,
                    snippet=_snippet(raw),
                )
            )

        elif name in _SFN_FAMILY and template.has("text"):
            issues.append(
                ValidationIssue(
                    kind="sfn_unsupported_param",
                    message=f"{{{{{template.name.strip()}}}}} uses unsupported parameter |text= on sqwiki",
                    severity="warning",
                    line_number=line,
                    snippet=_snippet(raw),
                )
            )


# Rowspan/colspan validation from raw wikitext (no real MediaWiki renderer
# available here) is inherently heuristic. This deliberately favors false
# negatives over false positives: any table it can't parse with confidence
# (nested tables, cells it can't cleanly split) is skipped rather than
# guessed at.
_TABLE_RE = re.compile(r"^\{\|.*?^\|\}", re.DOTALL | re.MULTILINE)
_CELL_ATTR_RE = re.compile(r"^((?:[a-zA-Z-]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s|]+)\s*)+)\|(?!\|)")
_ROWSPAN_RE = re.compile(r"rowspan\s*=\s*\"?(\d+)\"?", re.IGNORECASE)
_COLSPAN_RE = re.compile(r"colspan\s*=\s*\"?(\d+)\"?", re.IGNORECASE)


def _cell_span(cell_text: str) -> tuple[int, int]:
    match = _CELL_ATTR_RE.match(cell_text.strip())
    if not match:
        return 1, 1
    attrs = match.group(1)
    rowspan = _ROWSPAN_RE.search(attrs)
    colspan = _COLSPAN_RE.search(attrs)
    return (int(rowspan.group(1)) if rowspan else 1, int(colspan.group(1)) if colspan else 1)


def _parse_table_rows(block: str) -> list[tuple[int, list[tuple[int, int]]]] | None:
    """Returns [(line_offset_within_block, [(rowspan, colspan), ...]), ...]
    for each data/header row, or None if the block isn't confidently
    parseable (e.g. contains a nested table)."""
    if block.count("{|") > 1:
        return None
    rows: list[tuple[int, list[tuple[int, int]]]] = []
    current_cells: list[tuple[int, int]] = []
    current_row_offset = 0
    row_offset_set = False  # points at the first content line, not the |- delimiter
    in_row = False
    implicit_first_row = False  # the row between "{|" and the first "|-"
    offset = 0
    for line in block.split("\n"):
        stripped = line.rstrip()
        if stripped.startswith("{|"):
            in_row = True
            implicit_first_row = True
        elif stripped.startswith("|-"):
            if in_row and not (implicit_first_row and not current_cells):
                rows.append((current_row_offset, current_cells))
            implicit_first_row = False
            current_cells = []
            current_row_offset = offset
            row_offset_set = False
            in_row = True
        elif stripped.startswith("|}"):
            if in_row:
                rows.append((current_row_offset, current_cells))
            in_row = False
        elif in_row and stripped and not stripped.startswith("|+"):
            if stripped.startswith("!"):
                segments = re.split(r"!!|\|\|", stripped[1:])
            elif stripped.startswith("|"):
                segments = re.split(r"\|\|", stripped[1:])
            else:
                segments = None
            if segments is not None:
                if not row_offset_set:
                    current_row_offset = offset
                    row_offset_set = True
                current_cells.extend(_cell_span(seg) for seg in segments)
        offset += len(line) + 1
    return rows


def _check_table_span_mismatches(text: str, issues: list[ValidationIssue]) -> None:
    for table_match in _TABLE_RE.finditer(text):
        block = table_match.group(0)
        rows = _parse_table_rows(block)
        if rows is None or len(rows) < 2:
            continue

        active_rowspans: list[tuple[int, int]] = []  # (remaining_rows, colspan)
        widths: list[int] = []
        for _, cells in rows:
            width = sum(colspan for _, colspan in active_rowspans) + sum(cs for _, cs in cells)
            widths.append(width)
            active_rowspans = [(remaining - 1, colspan) for remaining, colspan in active_rowspans if remaining - 1 > 0]
            active_rowspans.extend((rowspan - 1, colspan) for rowspan, colspan in cells if rowspan > 1)

        width_counts: dict[int, int] = {}
        for w in widths:
            width_counts[w] = width_counts.get(w, 0) + 1
        if len(width_counts) <= 1:
            continue
        expected_width = max(width_counts.items(), key=lambda kv: kv[1])[0]

        # Only flag *isolated* off-width rows (mismatch surrounded on both
        # sides by rows that do match the table's dominant width) — real
        # translation defects (a dropped colspan/rowspan) show up as a single
        # row breaking an otherwise-uniform table. A contiguous run of two or
        # more off-width rows is far more often a deliberately asymmetric
        # trailing sub-block (e.g. a summary/notes row group with fewer
        # populated columns than the rest of the table) that MediaWiki
        # renders fine as-is — confirmed via a real sq.wikipedia infobox-style
        # table (Ashoka's Edicts table) that triggers this exact shape on the
        # pristine, untranslated English source, i.e. before any translation
        # touched it at all.
        for idx, ((row_offset, _), width) in enumerate(zip(rows, widths)):
            if width == expected_width:
                continue
            prev_ok = idx == 0 or widths[idx - 1] == expected_width
            next_ok = idx == len(widths) - 1 or widths[idx + 1] == expected_width
            if not (prev_ok and next_ok):
                continue
            abs_offset = table_match.start() + row_offset
            line_end = block.find("\n", row_offset)
            row_text = block[row_offset : line_end if line_end != -1 else None]
            issues.append(
                ValidationIssue(
                    kind="table_span_mismatch",
                    message=(
                        f"Table row has effective width {width} (columns, honoring "
                        f"rowspan/colspan), expected {expected_width} to match the rest of the table"
                    ),
                    severity="warning",
                    line_number=_line_number(text, abs_offset),
                    snippet=_snippet(row_text),
                )
            )


def validate_wikitext(text: str) -> ValidationResult:
    issues: list[ValidationIssue] = []

    try:
        mwp.parse(text)
    except Exception as exc:  # mwparserfromhell.parser.ParserError and friends
        issues.append(
            ValidationIssue(kind="parse_error", message=f"mwparserfromhell failed to parse output: {exc}")
        )

    _check_leaked_meta_commentary(text, issues)
    _check_degenerate_repetition(text, issues)
    # Brace/bracket counts must ignore verbatim content (math, code, nowiki)
    # where stray ``}}``/``]]`` substrings are legitimate, not delimiters.
    masked = _strip_verbatim_spans(text)
    _check_pair(masked, "{{", "}}", "template", issues)
    _check_pair(masked, "[[", "]]", "link", issues)
    _check_table_pair(text, issues)
    _check_pair(masked, "<!--", "-->", "comment", issues)
    _check_refs(text, issues)
    _check_headings_at_line_start(text, issues)
    _check_template_based_issues(text, issues)
    _check_table_span_mismatches(text, issues)

    return ValidationResult(valid=len(issues) == 0, issues=issues)


def format_errors(result: ValidationResult) -> list[str]:
    return [f"{issue.kind}: {issue.message}" for issue in result.issues]
