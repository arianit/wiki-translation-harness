"""Deterministic fixes for defects the model keeps reintroducing.

* `[[{{ill|A|en|B}}|label]]` — an {{ill}} nested inside a wikilink renders as
  literal "Label [en]" text with a broken link, so it becomes
  `{{ill|A|en|B|lt=label}}`.
* A `==Heading==` glued to the paragraph above it (no blank line); MediaWiki
  renders it, but the source loses the spacing the English article has.
* `|archive-date=` that disagrees with the Wayback timestamp in
  `|archive-url=` (CS1 flags the mismatch), or is still in English format.
* `{{Excerpt|Title}}` whose target is missing on the target wiki renders
  "Page 'Title' not found".
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

import mwparserfromhell as mwp

_ILL_IN_LINK_RE = re.compile(r"\[\[(\{\{\s*ill\s*\|[^{}\[\]]*\}\})(?:\|([^\[\]{}]*))?\]\]", re.IGNORECASE)
_ILL_IN_LABEL_RE = re.compile(r"\[\[[^\[\]|{}]+\|(\{\{\s*ill\s*\|[^{}\[\]]*\}\})\]\]", re.IGNORECASE)
_HEADING_RE = re.compile(r"^=+[^=\n](?:.*[^=\n])?=+[ \t]*$")
_WAYBACK_RE = re.compile(r"web\.archive\.org/web/(\d{4})(\d{2})(\d{2})\d*")
_EN_DMY_RE = re.compile(r"^\s*(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})\s*$")
_EN_MDY_RE = re.compile(r"^\s*([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})\s*$")
_EN_MY_RE = re.compile(r"^\s*([A-Za-z]+)\.?,?\s+(\d{4})\s*$")
_EN_MONTHS = {
    m[:3]: i + 1
    for i, m in enumerate(
        "january february march april may june july august september october november december".split()
    )
}
_DATE_PARAMS = ("date", "access-date", "accessdate", "archive-date", "archivedate")


def _month(name: str) -> int | None:
    return _EN_MONTHS.get(name.lower()[:3])


def _iso_from_english(value: str) -> str | None:
    """'22 February 2024' / 'February 22, 2024' / 'Feb 2024' -> ISO, else None."""
    if m := _EN_DMY_RE.match(value):
        d, mon, y = m.groups()
        if _month(mon):
            return f"{y}-{_month(mon):02d}-{int(d):02d}"
    if m := _EN_MDY_RE.match(value):
        mon, d, y = m.groups()
        if _month(mon):
            return f"{y}-{_month(mon):02d}-{int(d):02d}"
    if m := _EN_MY_RE.match(value):
        mon, y = m.groups()
        if _month(mon):
            return f"{y}-{_month(mon):02d}"
    return None


def unnest_ill_in_wikilinks(text: str) -> tuple[str, int]:
    """`[[{{ill|A|en|B}}|label]]` / `[[{{ill|A|en|B}}]]` -> `{{ill|A|en|B|lt=label}}`;
    `[[X|{{ill|A|en|B}}]]` -> `{{ill|A|en|B}}`."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        count += 1
        ill, label = m.group(1), m.group(2)
        if not label or not label.strip():
            return ill
        label = label.strip()
        if re.search(r"\|\s*lt\s*=", ill):
            return ill
        inner = ill[2:-2]
        parts = inner.split("|")
        if len(parts) >= 2 and parts[1].strip() == label:
            return ill
        return "{{" + inner + "|lt=" + label + "}}"

    text = _ILL_IN_LINK_RE.sub(repl, text)
    text, n = _ILL_IN_LABEL_RE.subn(lambda m: m.group(1), text)
    return text, count + n


def blank_line_before_headings(text: str) -> tuple[str, int]:
    """Insert an empty line before a heading whose previous line is neither
    blank nor itself a heading. Skips <!-- comments --> and <pre>/<nowiki>."""
    lines = text.split("\n")
    out: list[str] = []
    added = 0
    in_block = False
    for line in lines:
        if re.search(r"<(pre|nowiki|syntaxhighlight|source)\b[^>]*>(?!.*</\1>)", line, re.IGNORECASE):
            in_block = True
        if not in_block and out and _HEADING_RE.match(line):
            prev = out[-1]
            if prev.strip() and not _HEADING_RE.match(prev) and not prev.rstrip().endswith("-->"):
                out.append("")
                added += 1
        if in_block and re.search(r"</(pre|nowiki|syntaxhighlight|source)>", line, re.IGNORECASE):
            in_block = False
        out.append(line)
    return "\n".join(out), added


def fix_archive_dates(text: str) -> tuple[str, int]:
    """In citation templates: set |archive-date= to the Wayback timestamp's
    date, and convert English-format |date=/|access-date=/|archive-date=
    values to ISO (an English month name makes sqwiki's CS1 flag the
    citation as having an automatically translated date)."""
    code = mwp.parse(text)
    fixed = 0
    for tmpl in code.filter_templates(recursive=True):
        name = str(tmpl.name).strip().lower()
        if not (name.startswith("cite") or name == "citation"):
            continue
        for pname in _DATE_PARAMS:
            if not tmpl.has(pname):
                continue
            param = tmpl.get(pname)
            raw = str(param.value)
            current = raw.strip()
            iso = None
            if pname in ("archive-date", "archivedate") and tmpl.has("archive-url"):
                wb = _WAYBACK_RE.search(str(tmpl.get("archive-url").value))
                if wb:
                    iso = "-".join(wb.groups())
            if iso is None:
                iso = _iso_from_english(current)
            if iso and iso != current:
                param.value = iso + raw[len(raw.rstrip()):]
                fixed += 1
    return str(code), fixed


_EXCERPT_RE = re.compile(r"\{\{\s*Excerpt\s*\|([^|{}]+)(?:\|[^{}]*)?\}\}", re.IGNORECASE)


async def replace_missing_excerpts(
    text: str,
    exists: Callable[[list[str]], Awaitable[dict[str, bool]]],
    source_text: str,
    source_lang: str = "en",
) -> tuple[str, list[str]]:
    """Replace `{{Excerpt|T}}` with a hatnote `{{ill}}` when T is missing on
    the target wiki. The model often 'translates' T into a title that does
    not exist, so the Nth Excerpt is first matched back to the Nth source
    Excerpt to recover the real source title for the {{ill}}."""
    source_titles = [m.group(1).strip() for m in _EXCERPT_RE.finditer(source_text)]
    matches = list(_EXCERPT_RE.finditer(text))
    if not matches:
        return text, []
    titles = sorted({m.group(1).strip() for m in matches})
    present = await exists(titles)
    replaced: list[str] = []
    for i in reversed(range(len(matches))):
        m = matches[i]
        target = m.group(1).strip()
        if present.get(target, False):
            continue
        original = source_titles[i] if len(source_titles) == len(matches) else target
        if present.get(original, False):
            new = m.group(0).replace(target, original, 1)
        else:
            new = f"{{{{Hatnote|Artikulli kryesor: {{{{ill|{target}|{source_lang}|{original}}}}}}}}}"
            replaced.append(original)
        text = text[: m.start()] + new + text[m.end():]
    return text, replaced
