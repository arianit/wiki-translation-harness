"""Gives hatnotes whose targets sqwiki lacks an {{ill}} link.

sqwiki's {{Shih}}, {{Kryesor}} and {{Shiko gjithashtu}} wrap each parameter
in [[...]] themselves, so a target that has no sqwiki article renders as a
bare redlink with nothing pointing at the source article (Karl Marx:
"Lexo më tepër: Kritika e ekonomisë politike dhe Ekonomia marksiste").
Passing {{ill}} as the parameter double-wraps it, so such a hatnote has to
become {{Hatnote|Label: {{ill|Display|en|Title}}}} instead.

The model only sees the Albanian titles it chose, so the English titles come
from the source's own hatnote of the same kind and size, paired by order.
"""

from __future__ import annotations

from typing import Awaitable, Callable

import mwparserfromhell as mwp

# kind -> (source template names, translated template names, Albanian label)
_KINDS: dict[str, tuple[set[str], set[str], str]] = {
    "main": ({"main", "main article", "main2"}, {"kryesor", "main", "main article"}, "Artikulli kryesor"),
    "further": ({"further", "further information"}, {"shih", "further", "further information"}, "Lexo më tepër"),
    "see also": ({"see also", "seealso"}, {"shiko gjithashtu", "see also", "seealso"}, "Shih edhe"),
}

_SOURCE_KIND = {name: kind for kind, (names, _, _) in _KINDS.items() for name in names}
_TARGET_KIND = {name: kind for kind, (_, names, _) in _KINDS.items() for name in names}

ExistsFn = Callable[[list[str]], Awaitable[dict[str, bool]]]


def _positional(tmpl: mwp.nodes.Template) -> list[str]:
    return [str(p.value).strip() for p in tmpl.params if not p.showkey and str(p.value).strip()]


def _calls(text: str, kinds: dict[str, str]) -> list[tuple[str, mwp.nodes.Template]]:
    """Top-level hatnote calls, in order, tagged with their kind."""
    found = []
    for node in mwp.parse(text).nodes:
        if isinstance(node, mwp.nodes.Template):
            kind = kinds.get(str(node.name).strip().lower())
            if kind:
                found.append((kind, node))
    return found


def _join(links: list[str]) -> str:
    if len(links) < 2:
        return "".join(links)
    return ", ".join(links[:-1]) + " dhe " + links[-1]


async def add_ill_to_hatnotes(
    text: str, source_text: str, exists: ExistsFn, source_lang: str = "en"
) -> tuple[str, list[str]]:
    """Returns (patched_text, titles that were given an {{ill}})."""
    source_calls: dict[str, list[list[str]]] = {}
    for kind, tmpl in _calls(source_text, _SOURCE_KIND):
        source_calls.setdefault(kind, []).append(_positional(tmpl))

    calls = _calls(text, _TARGET_KIND)
    titles = sorted({t for _, tmpl in calls for t in _positional(tmpl)})
    if not titles:
        return text, []
    present = await exists(titles)

    cursor: dict[str, int] = {}
    edits: list[tuple[str, str]] = []
    converted: list[str] = []
    for kind, tmpl in calls:
        targets = _positional(tmpl)
        pool = source_calls.get(kind, [])
        # The model may already have rewritten some hatnotes as {{Hatnote}},
        # so the Nth call is not the Nth source call: take the next source
        # call of this kind with as many targets.
        start = cursor.get(kind, 0)
        match = next((i for i in range(start, len(pool)) if len(pool[i]) == len(targets)), None)
        if match is None:
            continue
        cursor[kind] = match + 1
        if all(present.get(t, True) for t in targets):
            continue
        originals = pool[match]
        links = []
        for target, original in zip(targets, originals):
            if present.get(target, True):
                links.append(f"[[{target}]]")
            else:
                links.append(f"{{{{ill|{target}|{source_lang}|{original}}}}}")
                converted.append(target)
        label = _KINDS[kind][2]
        edits.append((str(tmpl), f"{{{{Hatnote|{label}: {_join(links)}}}}}"))

    for old, new in edits:
        text = text.replace(old, new, 1)
    return text, converted

