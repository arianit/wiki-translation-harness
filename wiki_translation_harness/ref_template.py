"""Turns enwiki's {{R|n=name|r=text}} reference definitions into <ref> tags
before the article is chunked.

enwiki's {{R}} both reuses a named reference ({{R|name}}) and defines one
({{R|n=name|r=citation}}). sqwiki's copy of the template (Stampa:R, via
Stampa:R/ref) only does the first: it always emits an empty
<ref name="..."/>, whatever |r= says. Every article that defines its
references this way therefore ends up with reuses of references that have no
text, and Cite reports "no text was given for refs named X" for each one.
Confirmed live on sq.wikipedia, 2026-10-24: Roman Empire (55 definitions, 54
errors) and Pakistan (47 calls) were held for human review on exactly this.

Rewriting the definition as a plain <ref name="X">text</ref> on the whole
source gives the model ordinary markup to translate. Reuses ({{R|X}}) stay
as they are: sqwiki's template handles those.
"""

from __future__ import annotations

import re

import mwparserfromhell as mwp

# Keys of {{R}} that name the reference, hold its text, or carry its page.
_NAME_KEYS = {"n", "name", "n1", "name1", "1"}
_TEXT_KEYS = {"r", "ref", "reference", "r1", "ref1"}
_GROUP_KEYS = {"g", "group", "grp"}
_PAGE_KEYS = {"p", "page", "pp", "pages"}
_SAFE_NAME_RE = re.compile(r'^[^"<>{}|]+$')


def _inside_ref(node: mwp.nodes.Node, ancestors: list[mwp.nodes.Node]) -> bool:
    return any(isinstance(a, mwp.nodes.Tag) and str(a.tag).strip().lower() == "ref" for a in ancestors)


def _expand(tmpl: mwp.nodes.Template) -> str | None:
    """The <ref> markup for one {{R}} call that defines a reference, or None
    when the call is a plain reuse or has a shape this does not model (more
    than one name, a quote, a |loc=) and must be left alone."""
    name = text = group = page = None
    for param in tmpl.params:
        key = str(param.name).strip().lower()
        value = str(param.value).strip()
        if key in _NAME_KEYS:
            if name is not None:
                return None
            name = value
        elif key in _TEXT_KEYS:
            if text is not None:
                return None
            text = value
        elif key in _GROUP_KEYS:
            group = value
        elif key in _PAGE_KEYS:
            page = value
        elif value:
            return None
    if not text or not name or not _SAFE_NAME_RE.match(name):
        return None
    attrs = f' name="{name}"'
    if group:
        attrs += f' group="{group}"'
    markup = f"<ref{attrs}>{text}</ref>"
    if page:
        markup += "{{Rp|" + page + "}}"
    return markup


def expand_r_definitions(wikitext: str) -> tuple[str, int, int]:
    """Replaces each {{R}} call that defines a reference with a <ref> tag.
    Returns (patched_text, converted_count, left_alone_count); the last
    counts defining calls it could not model."""
    code = mwp.parse(wikitext)
    converted = skipped = 0
    for tmpl in code.filter_templates():
        if str(tmpl.name).strip().lower() != "r":
            continue
        if not any(str(p.name).strip().lower() in _TEXT_KEYS for p in tmpl.params):
            continue
        ancestors = code.get_ancestors(tmpl)
        markup = None if _inside_ref(tmpl, ancestors) else _expand(tmpl)
        if markup is None:
            skipped += 1
            continue
        code.replace(tmpl, markup)
        converted += 1
    return (str(code) if converted else wikitext), converted, skipped
