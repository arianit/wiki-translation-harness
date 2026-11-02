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
_EN_MONTHS = {
    m[:3]: i + 1
    for i, m in enumerate(
        "january february march april may june july august september october november december".split()
    )
}
_ISO_MONTH_ONLY_RE = re.compile(r"^(\s*)(\d{4})-(0[1-9]|1[0-2])(\s*)$")
_EN_MONTH_NAMES = (
    "January February March April May June July August September October November December".split()
)
_SQ_MONTH_ONLY_RE = re.compile(r"^(\s*)([A-Za-zëË]+)(\s+\d{4})(\s*)$")
_SQ_MONTHS = {
    "janar": "January", "shkurt": "February", "mars": "March", "prill": "April", "maj": "May",
    "qershor": "June", "korrik": "July", "gusht": "August", "shtator": "September",
    "tetor": "October", "nëntor": "November", "dhjetor": "December",
}
_DATE_PARAMS = ("date", "access-date", "accessdate", "archive-date", "archivedate")


def _month(name: str) -> int | None:
    return _EN_MONTHS.get(name.lower()[:3])


def _iso_from_english(value: str) -> str | None:
    """'22 February 2024' / 'February 22, 2024' -> ISO, else None. Month-only
    dates ('Feb 2024') are left alone: a bare YYYY-MM crashes sqwiki's CS1."""
    if m := _EN_DMY_RE.match(value):
        d, mon, y = m.groups()
        if _month(mon):
            return f"{y}-{_month(mon):02d}-{int(d):02d}"
    if m := _EN_MDY_RE.match(value):
        mon, d, y = m.groups()
        if _month(mon):
            return f"{y}-{_month(mon):02d}-{int(d):02d}"
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
            if iso is None and (sq := _SQ_MONTH_ONLY_RE.match(raw)) and sq.group(2).lower() in _SQ_MONTHS:
                # An Albanian month name in a month-only date crashes CS1; restore the English form.
                param.value = f"{sq.group(1)}{_SQ_MONTHS[sq.group(2).lower()]}{sq.group(3)}{sq.group(4)}"
                fixed += 1
                continue
            if iso is None and (im := _ISO_MONTH_ONLY_RE.match(raw)):
                # A bare YYYY-MM crashes CS1; restore the English 'Month YYYY' form.
                param.value = f"{im.group(1)}{_EN_MONTH_NAMES[int(im.group(3)) - 1]} {im.group(2)}{im.group(4)}"
                fixed += 1
                continue
            if iso and iso != current:
                param.value = iso + raw[len(raw.rstrip()):]
                fixed += 1
    return str(code), fixed


_AS_OF_RE = re.compile(r"\{\{\s*as of\s*\|([^{}]*)\}\}", re.IGNORECASE)
_AS_OF_DATE_RE = re.compile(
    r"^\s*(?:(\d{1,2})\s+)?([A-Za-z]+)\.?(?:\s+(\d{1,2})\s*,)?\s+(\d{4})\s*$"
    r"|^\s*(\d{4})\s+([A-Za-z]+)\s*$"
)
_SQ_MONTH_NOM = "janari shkurti marsi prilli maji qershori korriku gushti shtatori tetori nëntori dhjetori".split()
_TRAILING_NOISE_RE = re.compile(
    r"(?:\s|<ref[^>]*/>|<ref[^>]*>(?:(?!</ref>).)*</ref>|\{\{\s*sfn\|[^{}]*\}\})+\Z", re.IGNORECASE | re.DOTALL
)


def fix_as_of(text: str) -> tuple[str, int]:
    """sqwiki's {{As of}} breaks whenever a month is given: its category
    branch runs #time on an Albanian month name, so every month (number,
    English or Albanian name) renders "Gabim shprehjeje: Operator i papritur
    <" and "Gabim: Nevojitet vit, muaj, datë e vlefshme"; `lang=` is also an
    unknown parameter. Year-only calls are fine, so a dated call becomes
    `{{As of|YEAR|alt=që nga prilli 2021}}`, keeping the month in the text."""
    fixed = 0

    def repl(m: re.Match) -> str:
        nonlocal fixed
        raw = m.group(1).split("|")
        pos = [x.strip() for x in raw if "=" not in x]
        named = [(x.split("=", 1)[0].strip(), x.split("=", 1)[1].strip()) for x in raw if "=" in x]
        year = day = mon = None
        if pos and pos[0].isdigit():
            year = pos[0]
            if len(pos) > 1 and pos[1]:
                mon = int(pos[1]) if pos[1].isdigit() else _month(pos[1])
            if len(pos) > 2 and pos[2].isdigit():
                day = int(pos[2])
        elif pos and (dm := _AS_OF_DATE_RE.match(pos[0])):
            if dm.group(5):
                year, mon = dm.group(5), _month(dm.group(6))
            else:
                year, mon = dm.group(4), _month(dm.group(2))
                day = int(dm.group(1) or dm.group(3) or 0) or None
        else:
            return m.group(0)
        if not mon or not 1 <= mon <= 12 or any(k == "alt" for k, _ in named):
            mon = None
        keep = [(k, v) for k, v in named if k != "lang"]
        if mon is None and len(keep) == len(named) and len(pos) == 1:
            return m.group(0)
        fixed += 1
        if mon is None:
            parts = [year] + [f"{k}={v}" for k, v in keep]
            return "{{As of|" + "|".join(parts) + "}}"
        before = _TRAILING_NOISE_RE.sub("", m.string[max(0, m.start() - 1500): m.start()])
        lower = any(k == "lc" and v.lower() in ("y", "yes", "on", "1", "true") for k, v in keep)
        start = (not before or before[-1] in ".!?\n:") and not lower
        label = ("Që" if start else "që") + " nga " + (f"{day} " if day else "") + f"{_SQ_MONTH_NOM[mon - 1]} {year}"
        parts = [year, f"alt={label}"] + [f"{k}={v}" for k, v in keep if k != "lc"]
        return "{{As of|" + "|".join(parts) + "}}"

    return _AS_OF_RE.sub(repl, text), fixed


_AGE_RE = re.compile(r"\{\{\s*age in years and days(?: nts)?\s*\|([^{}]*)\}\}", re.IGNORECASE)


def expand_age_templates(text: str) -> tuple[str, int]:
    """sqwiki's {{Age in years and days (nts)}} renders "Gabim: Nevojitet vit,
    muaj, datë e vlefshme" for every input, so a two-date call becomes the
    plain text "X vjet, Y ditë"; an open-ended one (still in office) is dropped."""
    import datetime as dt

    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        args = [a.strip() for a in m.group(1).split("|")]
        if not all(a.isdigit() for a in args):
            return m.group(0)
        nums = [int(a) for a in args]
        count += 1
        if len(nums) != 6:
            return ""
        a, b = dt.date(*nums[:3]), dt.date(*nums[3:])
        years = b.year - a.year - ((b.month, b.day) < (a.month, a.day))
        try:
            anniv = a.replace(year=a.year + years)
        except ValueError:  # 29 February
            anniv = a.replace(year=a.year + years, day=28)
        days = (b - anniv).days
        parts = ([f"{years} vjet"] if years else []) + ([f"{days} ditë"] if days or not years else [])
        return ", ".join(parts)

    return _AGE_RE.sub(repl, text), count


_BROKEN_WAYBACK_RE = re.compile(
    r"(?:https?://web\.archive\.org/web/(?P<y>\d{4})[/-](?P<m>\d{2})[/-](?:(?P=y)-(?P=m)-)?(?P<d>\d{2})/?(?P<t>\d{6})/)+"
    r"(?=https?://)"
)


def fix_wayback_urls(text: str) -> tuple[str, int]:
    """The model sometimes splits a Wayback timestamp into path segments
    (`web/2020/01/2020-01-29133126/`, `web/2020-05-16/052110/`) and repeats the
    prefix; CS1 then flags "archive-url is malformed: timestamp"."""
    return _BROKEN_WAYBACK_RE.subn(
        lambda m: f"https://web.archive.org/web/{m['y']}{m['m']}{m['d']}{m['t']}/", text
    )


def fix_cs1_agency(text: str) -> tuple[str, int]:
    """sqwiki's CS1 ignores |agency= ("Parametri i panjohur"); fold it into
    |publisher= so the wire service is still shown."""
    code = mwp.parse(text)
    fixed = 0
    for tmpl in code.filter_templates(recursive=True):
        name = str(tmpl.name).strip().lower()
        if not (name.startswith("cite") or name == "citation") or not tmpl.has("agency"):
            continue
        agency = str(tmpl.get("agency").value).strip()
        if tmpl.has("publisher") and str(tmpl.get("publisher").value).strip():
            pub = str(tmpl.get("publisher").value).strip()
            if agency and agency.lower() not in pub.lower():
                tmpl.get("publisher").value = f"{agency}; {pub}" + (" " if str(tmpl.get("publisher").value).endswith(" ") else "")
            tmpl.remove("agency")
        else:
            tmpl.get("agency").name = "publisher" if not tmpl.has("publisher") else "agency"
            if tmpl.has("agency"):
                tmpl.remove("agency")
        fixed += 1
    return str(code), fixed


_REDIRECT_HATNOTE_RE = re.compile(
    r"^[ \t]*\{\{\s*redirect[\w -]*\|(?:[^{}]|\{\{[^{}]*\}\})*\}\}[ \t]*\n?", re.IGNORECASE | re.MULTILINE
)


def remove_redirect_hatnotes(text: str) -> tuple[str, int]:
    """sqwiki does not want {{Redirect|UN|…|UN (kthjellim)}}-style hatnotes
    ({{Redirect}}, {{Redirect2}}, {{Redirect-distinguish}}, …); drop them."""
    return _REDIRECT_HATNOTE_RE.subn("", text)


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
