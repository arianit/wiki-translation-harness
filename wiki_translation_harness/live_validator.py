"""Live parse-API validation.

Checks here require a real MediaWiki render and can't be caught from raw
wikitext alone: Lua/Scribunto errors, Cite-extension errors (including an
orphaned named ref — a `<ref name=x/>` with no matching definition, which
Cite itself detects and renders as an error span), templates that don't
exist on the target wiki, and raw flag/origin `Country data ...` text
leaking into visible output when a `{{flagicon}}`/`{{flag}}`-style template
can't resolve its underlying data page.

Complements validator.py's static checks — see pipeline.py for how the two
combine at the assembly stage. Kept as a separate module/function
(validate_wikitext_live, not validate_wikitext) because the existing static
validate_wikitext(text) in validator.py is a different, already-wired, sync
function with its own call sites and tests; colliding the names would be a
footgun, not a merge.

Class names and message text below were confirmed against a live
sq.wikipedia.org action=parse call, not guessed: Scribunto errors render as
`<strong class="error"><span class="scribunto-error ...">Script error: ...`
(message text is NOT localized); Cite errors (including orphaned named
refs) render as `<span class="error mw-ext-cite-error" ...>` with a
LOCALIZED message ("Gabim citimi: ..." on sqwiki) — so error-kind
classification here matches on CSS class, never on English error text,
except for the Scribunto case where MediaWiki genuinely does not localize
"Script error"/"Lua error".
"""

from __future__ import annotations

import re

from wiki_translation_harness.mediawiki import MediaWikiClient
from wiki_translation_harness.models import ValidationIssue, ValidationResult

_SNIPPET_MAX = 120

# One rendered <strong>/<span class="...error...">...</tag> at a time, so
# each distinct occurrence becomes its own finding rather than one blob
# match. The class lookahead is required, not just a post-match filter:
# MediaWiki's Cite output wraps the actual error span in an outer
# <span class="reference-text">...</span> with the SAME tag name and no
# error class of its own — without the lookahead, a non-greedy (?P<msg>.*?)
# pairs the outer wrapper's opening tag with the FIRST </span> it meets
# (the inner error span's own close), silently swallowing the real error
# span as unrecognized "reference-text" content instead of matching it.
# Requiring the class to already contain an error marker at the match-start
# position means the outer wrapper is never chosen as a match start at all,
# so the scan lands on the inner error span instead. Confirmed against a
# live sq.wikipedia.org action=parse response, not guessed.
_ERROR_SPAN_RE = re.compile(
    r'<(?P<tag>strong|span)\b[^>]*\bclass="(?=[^"]*(?:scribunto-error|cite[-_]error))(?P<cls>[^"]*)"[^>]*>(?P<msg>.*?)</(?P=tag)>',
    re.IGNORECASE | re.DOTALL,
)
_COUNTRY_DATA_LEAK_RE = re.compile(r"Country[\s_-]data[\s_]\S+", re.IGNORECASE)

# Dependency modules confirmed missing on sq.wikipedia that a legitimate,
# EXISTING template can conditionally reach for, depending on which
# Wikidata properties the linked item happens to have -- not a translation
# defect, not fixable by rewriting wikitext, and not necessarily even a
# visible break on the rendered page. {{Authority control}} (Stampa:Authority
# control, which itself exists and renders correctly on real sq.wikipedia
# articles, e.g. "Pjetër Bogdani") reaches for "Moduli:WikidataIB/i18n" only
# for certain identifier types; confirmed missing via the API 2026-08-21.
# Flagging this every time it's hit would only ever produce a permanently
# unresolvable needs_human_review, since no amount of repair-round
# re-prompting can create a missing module on the live target wiki.
# Moduli:SST/registry: same treatment, confirmed embedded (via
# `embeddedin`) in dozens of long-standing, fine-rendering sq.wikipedia
# articles (e.g. Aristoteli, Skënderbeu, Abraham Lincoln, Batman) --
# reached internally by some widely-used template/module chain (still
# missing on sq.wikipedia itself), not something translating an article
# introduces or could ever fix. Confirmed missing via the API on Nagarjuna
# 2026-09-06.
_KNOWN_HARMLESS_MISSING_DEPENDENCY_MODULES = {"Moduli:WikidataIB/i18n", "Moduli:SST/registry"}

# Phantom template dependency: reported by action=parse's `templates` list but
# absent from the article wikitext, absent from every existing template's raw
# source, and producing zero occurrences in the rendered HTML (confirmed on
# Jupiter 2026-09-05). Some Lua/template path references it internally without
# ever rendering it, so it can't be fixed by editing the article and never
# breaks the visible page. Same treatment as the module entries above.
_KNOWN_HARMLESS_MISSING_DEPENDENCY_TEMPLATES = {"Stampa:Sec link image"}


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _locate(source_text: str, needle: str) -> tuple[int | None, str | None]:
    """Best-effort only: parsed HTML carries no wikitext line mapping, so a
    finding's line/snippet are approximated by searching the wikitext that
    was actually sent for the finding's identifying string (a template
    name, the leaked text itself, a fragment of the error message)."""
    if not needle:
        return None, None
    offset = source_text.find(needle)
    if offset == -1:
        return None, None
    line = source_text.count("\n", 0, offset) + 1
    line_end = source_text.find("\n", offset)
    snippet = source_text[offset : line_end if line_end != -1 else None].strip()
    if len(snippet) > _SNIPPET_MAX:
        snippet = snippet[: _SNIPPET_MAX - 1] + "…"
    return line, snippet


def _locate_template(source_text: str, bare_name: str) -> tuple[int | None, str | None]:
    """Locate a template invocation in wikitext (e.g. {{name}} or {{name|...).

    Matches {{bare_name or {{Template:bare_name rather than bare_name alone,
    preventing common-word or punctuation template names (e.g. '"', 'Main')
    from false-matching unrelated prose or punctuation in earlier chunks.
    """
    if not bare_name:
        return None, None
    pattern = re.compile(
        rf"\{{\{{\s*(?:[A-Za-z0-9_]+:)?{re.escape(bare_name)}\s*([\|}}])",
        re.IGNORECASE,
    )
    match = pattern.search(source_text)
    if match:
        offset = match.start()
        line = source_text.count("\n", 0, offset) + 1
        line_end = source_text.find("\n", offset)
        snippet = source_text[offset : line_end if line_end != -1 else None].strip()
        if len(snippet) > _SNIPPET_MAX:
            snippet = snippet[: _SNIPPET_MAX - 1] + "…"
        return line, snippet
    return _locate(source_text, bare_name)


def find_live_issues(parse_result: dict, source_text: str = "") -> list[ValidationIssue]:
    """Pure function over an action=parse response (see
    MediaWikiClient.parse_wikitext) — no network, easy to unit test with a
    canned dict. source_text, if given, is the wikitext that was sent, used
    only to approximate line_number/snippet (see _locate)."""
    issues: list[ValidationIssue] = []
    html = parse_result.get("text", "") or ""

    # Cite renders one error span per *use* of a broken named ref, so a ref
    # reused nine times yields nine identical messages. One finding per
    # distinct message is enough; duplicates only bloat the repair prompt.
    seen_cite_msgs: set[str] = set()
    for match in _ERROR_SPAN_RE.finditer(html):
        cls = match.group("cls").lower()
        msg = _strip_html(match.group("msg"))
        if not msg:
            continue

        if "scribunto-error" in cls or re.search(r"script error|lua error", msg, re.IGNORECASE):
            line, snippet = _locate(source_text, msg[:40])
            issues.append(
                ValidationIssue(
                    kind="lua_script_error",
                    message=f"Lua/Scribunto error rendered on the page: {msg}",
                    line_number=line,
                    snippet=snippet,
                )
            )
        elif re.search(r"cite[-_]error", cls):
            # MediaWiki's Cite extension renders a generic cite error and an
            # orphaned named ref (a `<ref name=x/>` with no definition)
            # through the same error-span mechanism, distinguished only by
            # localized message text — "ref" survives translation into
            # Albanian inside the literal `<ref>` tag mention, so this
            # heuristic holds there, but isn't guaranteed on every wiki.
            if msg in seen_cite_msgs:
                continue
            seen_cite_msgs.add(msg)
            kind = "orphaned_named_ref" if "ref" in msg.lower() else "cite_error"
            line, snippet = _locate(source_text, msg[:40])
            issues.append(
                ValidationIssue(
                    kind=kind,
                    message=f"Cite error rendered on the page: {msg}",
                    line_number=line,
                    snippet=snippet,
                )
            )

    # Visible text only, not raw HTML: MediaWiki's redlink markup repeats
    # the same leaked title in an href's query string, a title="" attribute,
    # and the anchor's inner text — matching raw HTML flags the same leak
    # two or three times, and the href copy uses underscores for spaces,
    # which the "\S+" tail can run on into the rest of the URL. Stripping
    # tags first collapses this to exactly one clean match per real leak.
    seen_leaks: set[str] = set()
    for country_match in _COUNTRY_DATA_LEAK_RE.finditer(_strip_html(html)):
        leaked = country_match.group(0)
        if leaked in seen_leaks:
            continue
        seen_leaks.add(leaked)
        line, snippet = _locate(source_text, leaked)
        issues.append(
            ValidationIssue(
                kind="country_data_leak",
                message=f"Raw flag/origin template text leaked into rendered output: {leaked!r}",
                line_number=line,
                snippet=snippet,
            )
        )

    for template in parse_result.get("templates", []) or []:
        if template.get("exists") is False:
            title = template.get("title", "?")
            if (
                title in _KNOWN_HARMLESS_MISSING_DEPENDENCY_MODULES
                or title in _KNOWN_HARMLESS_MISSING_DEPENDENCY_TEMPLATES
            ):
                continue
            # search by the bare name (without namespace prefix) since
            # that's what appears in the wikitext, not "Stampa:X"/"Template:X"
            bare_name = title.split(":", 1)[-1]
            line, snippet = _locate_template(source_text, bare_name)
            issues.append(
                ValidationIssue(
                    kind="unexpanded_template",
                    message=f"Template {title!r} does not exist on the target wiki",
                    severity="warning",
                    line_number=line,
                    snippet=snippet,
                )
            )

    return issues


async def validate_wikitext_live(
    client: MediaWikiClient, text: str, title: str = "API"
) -> ValidationResult:
    parse_result = await client.parse_wikitext(text, title=title)
    issues = find_live_issues(parse_result, source_text=text)
    return ValidationResult(valid=len(issues) == 0, issues=issues)
