"""Expands enwiki's {{Cref2}}/{{Cnote2}} note system into plain
<ref group=...> footnotes before the article is chunked.

enwiki keeps these notes in two places: {{Cref2|c}} markers scattered through
the body, and a {{Cnote2 Begin}} ... {{Cnote2|c|note text}} ... {{Cnote2 End}}
block in the Notes section holding the text. sqwiki has neither template, and
the two halves land in different chunks, so no single translation call sees
both. Confirmed on Alexander the Great (2026-10-04): the model rebuilt the
notes as a hand-made <ol>, dropped seven of the eight {{Cref2}} markers from
the body, and left one stray <ref group="sh"> with no <references group="sh">
(a live Cite error). Done mechanically on the whole source before chunking,
each marker becomes a self-contained <ref group="...">note text</ref> that
travels with its chunk, and the block becomes one {{Reflist|group=...}}.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import mwparserfromhell as mwp

from wiki_translation_harness.citation_language import convert_nested_sfn_to_harvnb

# Groups sqwiki's Cite renders as letters/numerals ("[A]") rather than the
# generic "[sh 1]" — verified live against sq.wikipedia, 2026-10-04.
_LETTER_GROUPS = {"upper-alpha", "lower-alpha", "upper-roman", "lower-roman", "lower-greek"}
_DEFAULT_GROUP = "upper-alpha"

_NOTE_NAME_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _norm(name: object) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).lower()


@dataclass
class NoteExpansionResult:
    patched_wikitext: str
    expanded: int = 0  # {{Cref2}} labels turned into <ref group=...> footnotes
    unresolved: list[str] = field(default_factory=list)  # labels left as {{Cref2}}
    unreferenced: list[str] = field(default_factory=list)  # {{Cnote2}} notes no marker pointed at


def _inside_ref(code: mwp.wikicode.Wikicode, node: mwp.nodes.Node) -> bool:
    try:
        ancestors = code.get_ancestors(node)
    except ValueError:
        return False
    return any(
        isinstance(a, mwp.nodes.Tag) and str(a.tag).strip().lower() == "ref" for a in ancestors
    )


def _split_out_inner_refs(note_text: str) -> tuple[str, str]:
    """A <ref> can't sit inside another <ref>, but enwiki notes routinely
    carry their own citations. Returns (note text without them, those refs
    concatenated) so the caller can put them right after the note marker,
    where they become ordinary citations of the same passage."""
    code = mwp.parse(note_text)
    hoisted = []
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).strip().lower() == "ref":
            hoisted.append(str(tag))
            code.remove(tag)
    return str(code).strip(), "".join(hoisted)


def expand_cref_notes(wikitext: str) -> NoteExpansionResult:
    """Replaces each {{Cref2|x}} with <ref name="cnote-x" group="G">text</ref>
    (later uses of the same label become a reuse <ref name="cnote-x" group="G" />),
    and the {{Cnote2 Begin}}...{{Cnote2 End}} block with {{Reflist|group=G}}.
    G comes from the block's |liststyle= when it names a letter/numeral group,
    else upper-alpha. A {{notelist}} directly after the block is dropped (the
    Reflist replaces it). Returns the text untouched when the article has no
    {{Cnote2}} notes."""
    if not re.search(r"\{\{\s*cnote2\b", wikitext, re.IGNORECASE):
        return NoteExpansionResult(wikitext)

    code = mwp.parse(wikitext)
    templates = code.filter_templates(recursive=True)

    notes: dict[str, str] = {}
    group = _DEFAULT_GROUP
    group_seen = False
    for tmpl in templates:
        name = _norm(tmpl.name)
        if name == "cnote2 begin" and not group_seen:
            group_seen = True
            if tmpl.has("liststyle"):
                style = str(tmpl.get("liststyle").value).strip().lower()
                if style in _LETTER_GROUPS:
                    group = style
        elif name == "cnote2" and tmpl.has("1") and tmpl.has("2"):
            label = str(tmpl.get("1").value).strip()
            notes.setdefault(label, str(tmpl.get("2").value).strip())
    if not notes:
        return NoteExpansionResult(wikitext)

    used: set[str] = set()
    unresolved: list[str] = []
    expanded = 0
    for tmpl in templates:
        if _norm(tmpl.name) != "cref2":
            continue
        labels = [str(p.value).strip() for p in tmpl.params if not p.showkey or str(p.name).strip().isdigit()]
        labels = [label for label in labels if label]
        if not labels or any(label not in notes for label in labels) or _inside_ref(code, tmpl):
            unresolved.extend(labels or ["?"])
            continue
        refs = []
        for label in labels:
            ref_name = "cnote-" + _NOTE_NAME_UNSAFE_RE.sub("", label)
            if label in used:
                refs.append(f'<ref name="{ref_name}" group="{group}" />')
            else:
                used.add(label)
                note_text, citations = _split_out_inner_refs(notes[label])
                # {{sfn}} self-wraps in a <ref>, so one inside a note would nest
                # refs; harvnb is the bare-link twin meant to live in one.
                ref = f'<ref name="{ref_name}" group="{group}">{note_text}</ref>'
                refs.append(convert_nested_sfn_to_harvnb(ref).patched_wikitext + citations)
            expanded += 1
        code.replace(tmpl, "".join(refs))

    reflist = "{{Reflist|group=" + group + "}}"
    placed = False
    for tmpl in code.filter_templates(recursive=True):
        if _norm(tmpl.name) in {"cnote2", "cnote2 begin", "cnote2 end"}:
            if not placed and _norm(tmpl.name) != "cnote2 end":
                code.replace(tmpl, reflist)
                placed = True
            else:
                code.remove(tmpl)
    text = str(code)
    if not placed:
        # An End with no Begin/entries to anchor on can't be the block's start.
        return NoteExpansionResult(wikitext)
    text = re.sub(
        re.escape(reflist) + r"\s*\n\{\{\s*notelist\s*(?:\|[^{}]*)?\}\}",
        reflist,
        text,
        flags=re.IGNORECASE,
    )
    # The removed entries leave a run of blank lines behind the Reflist.
    text = re.sub(re.escape(reflist) + r"\n{3,}", reflist + "\n\n", text)
    return NoteExpansionResult(
        patched_wikitext=text,
        expanded=expanded,
        unresolved=unresolved,
        unreferenced=[label for label in notes if label not in used],
    )
