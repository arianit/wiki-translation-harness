"""Deterministic repairs to <ref>/<references> markup in an assembled article.

Translation chunks are translated independently, so the model can split a
<references> block, rename or drop ref names and groups, duplicate or orphan
list-defined refs, and so on. Each function here compares the assembled
translation against the English source text and undoes one such defect.
Called from pipeline.post_process_assembled.
"""

from __future__ import annotations

import re

from wiki_translation_harness.notes import IMPLICIT_NOTE_GROUP, NOTELIST_ALIASES, uses_implicit_note_group

_REFERENCES_OPEN_RE = re.compile(r"<references\b[^>]*(?<!/)>", re.IGNORECASE)
_REFERENCES_CLOSE = "</references>"
_HEADING_LINE_RE = re.compile(r"^=+[^=].*=+\s*$")


def merge_split_references_blocks(text: str) -> tuple[str, int]:
    """Rejoin a list-defined `<references>` block that chunking split in two.

    A long "Citations" section is cut mid-block, so the first chunk's
    translation closes the `<references>` it opened and the second chunk
    (which has no opener) ends with the source's original `</references>`.
    Assembled, that is `<references> defs </references> more defs
    </references>`: the defs after the first closer sit outside the block,
    so Cite reports every ref used only by them as "no text given" and every
    ref defined inside as "not used in preceding text" — dozens of
    orphaned_named_ref findings that no per-chunk repair can fix, since each
    chunk looks fine alone.

    A closer with no matching opener is the second half's; the previous
    closer (same section, nothing but ref definitions between) is the
    spurious one and is dropped. Returns (patched_text, merged_count)."""
    lines = text.split("\n")
    is_open = False
    last_close: int | None = None
    drop: set[int] = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _HEADING_LINE_RE.match(stripped):
            last_close = None
        elif _REFERENCES_OPEN_RE.search(stripped):
            is_open = True
            last_close = None
        elif stripped.lower() == _REFERENCES_CLOSE:
            if is_open:
                is_open = False
                last_close = i
            elif last_close is not None:
                drop.add(last_close)
                last_close = None
    if not drop:
        return text, 0
    return "\n".join(line for i, line in enumerate(lines) if i not in drop), len(drop)


_GROUP_ATTR_RE = re.compile(
    r"(\bgroup\s*=\s*)(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'/>|}]+))", re.IGNORECASE
)


def _group_value(match: re.Match[str]) -> str:
    return next(g for g in match.groups()[1:] if g is not None).strip()


def restore_ref_group_names(text: str, source_text: str) -> tuple[str, dict[str, str]]:
    """Undo translation of a footnote group's name (group=note ->
    group="shënim").

    The group name never reaches the reader, but it has to match exactly:
    the model translates it in some chunks and keeps it in others, so the
    notes of a translated `<ref group="shënim">` find no list (the
    `{{Reflist|group=note|refs=...}}` that kept its name) and the model
    adds a `<references group="shënim"/>` of its own wherever it pleases.

    A translation-only group maps back when there's no doubt which source
    group it was: the source has just one group, or exactly one source
    group went missing in translation as exactly one new one appeared.
    Returns (patched_text, {translated_name: source_name})."""
    source_groups = {_group_value(m) for m in _GROUP_ATTR_RE.finditer(source_text)}
    # {{Efn}} and {{Notelist}} put their notes in lower-alpha when the source
    # names no group, so that group exists in the source without being
    # written there. The model then invents one for the {{Efn}} it turns into
    # a <ref> (Mount Everest "note", Washington D.C. and Periodic table "sh"),
    # and sqwiki's {{Listë shënimesh}} lists lower-alpha only.
    if uses_implicit_note_group(source_text):
        source_groups.add(IMPLICIT_NOTE_GROUP)
    text_groups = {_group_value(m) for m in _GROUP_ATTR_RE.finditer(text)}
    unknown = text_groups - source_groups
    if not unknown:
        return text, {}
    if len(source_groups) == 1:
        (target,) = source_groups
        renamed = {name: target for name in unknown}
    elif len(unknown) == 1 and len(source_groups - text_groups) == 1:
        renamed = {next(iter(unknown)): next(iter(source_groups - text_groups))}
    else:
        return text, {}

    def _rename(match: re.Match[str]) -> str:
        name = _group_value(match)
        if name not in renamed:
            return match.group(0)
        return f'{match.group(1)}"{renamed[name]}"'

    return _GROUP_ATTR_RE.sub(_rename, text), renamed


# A tag or template that renders a reference list and carries no
# definitions of its own: `<references />`, an empty `<references></references>`,
# or a one-line `{{Reflist}}`/`{{Notelist}}` without |refs=.
_REF_LIST_RENDERER_RE = re.compile(
    r"<references\b(?P<tag_attrs>[^>]*?)/>"
    r"|<references\b(?P<pair_attrs>[^>]*)>\s*</references\s*>"
    r"|\{\{\s*(?P<tpl>reflist|references|notelist|listë shënimesh"
    + "".join("|" + re.escape(a).replace(r"\ ", r"\s+") for a in NOTELIST_ALIASES)
    + r")\s*(?P<tpl_args>\|[^{}\n]*)?\}\}",
    re.IGNORECASE,
)


def _ref_list_key(match: re.Match[str]) -> tuple[str, str] | None:
    """(kind, group) of a reference-list renderer, or None if it's a
    template that defines refs (|refs=), which is content, not a stray."""
    tpl = match.group("tpl")
    attrs = match.group("tag_attrs") or match.group("pair_attrs") or match.group("tpl_args") or ""
    if tpl and re.search(r"\|\s*refs\s*=", attrs, re.IGNORECASE):
        return None
    group = _GROUP_ATTR_RE.search(attrs)
    group_name = _group_value(group) if group else ""
    if tpl and tpl.lower() not in ("reflist", "references"):
        # A note list is a reflist for the group its notes are in, so the
        # model writing {{Reflist|group=lower-alpha}} for {{notelist}}
        # counts as the same list.
        return "refs", group_name or IMPLICIT_NOTE_GROUP
    return "refs", group_name


_NOTES_HEADING_RE = re.compile(r"=+\s*(?:notes?|shënim\w*)\s*=+\s*", re.IGNORECASE)


def _under_notes_heading(text: str, pos: int) -> bool:
    """True when the last non-blank line before `pos` is a Notes/Shënime heading."""
    for line in reversed(text[:pos].splitlines()):
        if line.strip():
            return _NOTES_HEADING_RE.fullmatch(line.strip()) is not None
    return False


def drop_stray_reference_lists(text: str, source_text: str) -> tuple[str, int]:
    """Remove reference lists the model invented.

    When chunking splits a long list-defined `<references>` block, the
    model translating each later piece "completes" its fragment with a
    `<references />` (or `<references group="..."/>`, or an empty
    `<references></references>`) of its own — the Albert Einstein run had
    nine where the source had one. Each one flushes the refs before it, so
    the definitions after the first one land outside the real block and
    Cite reports every ref used only by them as having no text. The same
    habit drops `<references group="note" />` under random mid-article
    headings.

    For each (kind, group), keeps as many renderers as the source has
    — one directly under a Notes heading first, then the last ones, since
    reference lists close an article — and removes the rest, whole line
    and all where the renderer stood alone.
    Returns (patched_text, removed_count)."""
    allowed: dict[tuple[str, str], int] = {}
    for m in _REF_LIST_RENDERER_RE.finditer(source_text):
        key = _ref_list_key(m)
        if key is not None:
            allowed[key] = allowed.get(key, 0) + 1

    by_key: dict[tuple[str, str], list[re.Match[str]]] = {}
    for m in _REF_LIST_RENDERER_RE.finditer(text):
        key = _ref_list_key(m)
        if key is not None:
            by_key.setdefault(key, []).append(m)

    spans: list[tuple[int, int]] = []
    for key, matches in by_key.items():
        excess = len(matches) - allowed.get(key, 0)
        if excess > 0:
            # A list directly under the Notes heading is where the source
            # has it; the model's other copies are strays even when one of
            # them is the last in the file.
            kept = sorted(
                range(len(matches)),
                key=lambda i: (not _under_notes_heading(text, matches[i].start()), -i),
            )[: len(matches) - excess]
            removed = [matches[i] for i in range(len(matches)) if i not in kept]
        else:
            removed = []
        for m in removed:
            start, end = m.span()
            line_start = text.rfind("\n", 0, start) + 1
            line_end = text.find("\n", end)
            line_end = len(text) if line_end == -1 else line_end
            if not text[line_start:start].strip() and not text[end:line_end].strip():
                start, end = line_start, min(line_end + 1, len(text))
            spans.append((start, end))
    if not spans:
        return text, 0

    parts: list[str] = []
    pos = 0
    for start, end in sorted(spans):
        parts.append(text[pos:start])
        pos = end
    parts.append(text[pos:])
    return "".join(parts), len(spans)


_DEFS_AFTER_REFERENCES_RE = re.compile(
    # HTML comments may sit between the definitions (the model leaves its
    # own notes there); the run still has to end on a definition.
    r"</references\s*>((?:(?:\s*<!--.*?-->)*\s*<ref\b[^>]*?(?<!/)>(?:(?!<ref\b).)*?</ref\s*>)+)",
    re.IGNORECASE | re.DOTALL,
)


def extend_references_block_over_trailing_defs(text: str) -> tuple[str, int]:
    """Move a `</references>` down past the ref definitions that directly
    follow it.

    The other half of merge_split_references_blocks: there the later
    chunk kept the source's `</references>`; here the model dropped it (or
    drop_stray_reference_lists removed the `<references />` it wrote
    instead), so the block closes after the first chunk's definitions and
    the rest trail after it, outside any list. A named definition right
    after a reference list is never intended, so the run joins the block.
    Returns (patched_text, moved_definition_count)."""
    moved = 0

    def _extend(match: re.Match[str]) -> str:
        nonlocal moved
        defs = match.group(1)
        moved += len(re.findall(r"</ref\s*>", defs, re.IGNORECASE))
        # lstrip: the newline after the old closer would otherwise leave a
        # blank line where it stood.
        return defs.lstrip() + "\n</references>"

    return _DEFS_AFTER_REFERENCES_RE.sub(_extend, text), moved


_REF_NAME_ATTR = r"\bname\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s\"'/>]+))"
_REF_SELF_CLOSING_RE = re.compile(r"<ref\b[^>]*?" + _REF_NAME_ATTR + r"[^>]*?/>", re.IGNORECASE)
_REF_DEFINITION_RE = re.compile(
    # The body may not run past another <ref opener: an unclosed definition
    # (its </ref> lost in translation) would otherwise extend to the NEXT
    # definition's </ref>, hiding that ref from `defined` — and
    # restore_orphaned_ref_definitions would then add a duplicate
    # definition of a ref that was there all along.
    r"<ref\b[^>]*?" + _REF_NAME_ATTR + r"[^>]*?(?<!/)>((?:(?!<ref\b).)*?)</ref\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _ref_name(match: re.Match[str]) -> str:
    return (match.group(1) or match.group(2) or match.group(3)).strip()


_REFERENCES_BLOCK_RE = re.compile(r"<references\b[^>/]*>(.*?)</references\s*>", re.IGNORECASE | re.DOTALL)


def _block_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _REFERENCES_BLOCK_RE.finditer(text)]


def collapse_duplicate_list_defined_refs(text: str, source_text: str) -> tuple[str, list[str]]:
    """Turn a body definition of a list-defined ref back into a plain reuse.

    The source keeps these refs' text only inside the `<references>` block;
    in the body they are `<ref name="x" />`. A chunk whose model "helpfully"
    writes the full text at the first use (or whose definition
    restore_orphaned_ref_definitions copied in from the source before the
    block's own copy was in place) leaves two definitions — often worded
    differently — and Cite reports the ref as defined multiple times with
    different content. The block's copy wins: it is the source's.

    Only names the source defines inside a block and nowhere else, and only
    refs without a group. Returns (patched_text, collapsed_names)."""
    def _names_in(block_texts: list[str]) -> set[str]:
        return {_ref_name(m) for chunk in block_texts for m in _REF_DEFINITION_RE.finditer(chunk)}

    source_blocks = _block_spans(source_text)
    in_source_blocks = _names_in([source_text[a:b] for a, b in source_blocks])
    source_outside = "".join(
        source_text[start:end]
        for start, end in _complement_spans(len(source_text), source_blocks)
    )
    list_defined = in_source_blocks - _names_in([source_outside])
    blocks = _block_spans(text)
    in_text_blocks = _names_in([text[a:b] for a, b in blocks])
    targets = list_defined & in_text_blocks
    if not targets:
        return text, []

    collapsed: list[str] = []
    out: list[str] = []
    for start, end in _complement_spans(len(text), blocks):
        def _collapse(match: re.Match[str]) -> str:
            name = _ref_name(match)
            opening = match.group(0)[: match.group(0).index(">") + 1]
            if name not in targets or re.search(r"\bgroup\s*=", opening, re.IGNORECASE):
                return match.group(0)
            collapsed.append(name)
            return f'<ref name="{name}" />'

        out.append((start, _REF_DEFINITION_RE.sub(_collapse, text[start:end])))
    pieces = sorted(out + [(a, text[a:b]) for a, b in blocks])
    return "".join(piece for _, piece in pieces), collapsed


def _complement_spans(length: int, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result, pos = [], 0
    for start, end in sorted(spans):
        if start > pos:
            result.append((pos, start))
        pos = end
    if pos < length:
        result.append((pos, length))
    return result


def restore_orphaned_ref_definitions(text: str, source_text: str) -> tuple[str, list[str]]:
    """Copy back named-ref definitions that translation dropped.

    The model sometimes condenses away the sentence that carried a
    `<ref name="x">...</ref>` definition while keeping a later
    `<ref name="x" />` reuse — Cite then renders an orphaned-ref error.
    Handing that to the LLM repair loop rarely works: the chunk that has
    the reuse never saw the definition's content, so it either invents a
    citation or does nothing, and the loop burns its rounds. The source
    article still has the exact definition, so this rewrites the first
    orphaned reuse into a full definition using the source's ref body
    (citation bodies are kept untranslated anyway). Refs the source doesn't
    define either are left for the repair loop.

    Returns (patched_text, restored_names)."""
    defined = {_ref_name(m) for m in _REF_DEFINITION_RE.finditer(text)}
    source_defs: dict[str, str] = {}
    for m in _REF_DEFINITION_RE.finditer(source_text):
        source_defs.setdefault(_ref_name(m), m.group(4))

    restored: list[str] = []

    def _restore(match: re.Match[str]) -> str:
        name = _ref_name(match)
        if name in defined or name not in source_defs:
            return match.group(0)
        defined.add(name)
        restored.append(name)
        return f'<ref name="{name}">{source_defs[name]}</ref>'

    return _REF_SELF_CLOSING_RE.sub(_restore, text), restored


def drop_unsupported_ref_reuses(text: str, source_text: str) -> tuple[str, list[str]]:
    """Remove `<ref name="x" />` reuses of a name that nothing defines.

    The model sometimes invents a descriptive name for a citation
    (`corona-particle-density`, `Sun-Rotation-rotation` on the Sun run) and
    reuses it in several places without ever writing the definition. The
    source has no such name, so restore_orphaned_ref_definitions has nothing
    to copy back and the repair loop can only invent a citation; every use
    is a Cite error. A name that appears nowhere in the source and is
    defined nowhere in the translation is unsupported, so its reuses go.

    Returns (patched_text, dropped_names)."""
    defined = {_ref_name(m) for m in _REF_DEFINITION_RE.finditer(text)}
    dropped: list[str] = []

    def _drop(match: re.Match[str]) -> str:
        name = _ref_name(match)
        if name in defined or name in source_text:
            return match.group(0)
        if name not in dropped:
            dropped.append(name)
        return ""

    return _REF_SELF_CLOSING_RE.sub(_drop, text), dropped


def collapse_conflicting_ref_definitions(text: str, source_text: str) -> tuple[str, list[str]]:
    """Turn the later definitions of a name into reuses when the model
    translated identical source definitions differently.

    A source may repeat `<ref name="short">All numbers ... short scale</ref>`
    word for word in several sections (Sun does, twice). Each copy is
    translated in its own chunk, the wording differs, and Cite reports the
    name as defined multiple times with different content. Only names whose
    source definitions all agree are touched; the first definition stays.

    Returns (patched_text, collapsed_names)."""
    def _squash(body: str) -> str:
        return re.sub(r"\s+", " ", body).strip()

    source_bodies: dict[str, set[str]] = {}
    for m in _REF_DEFINITION_RE.finditer(source_text):
        source_bodies.setdefault(_ref_name(m), set()).add(_squash(m.group(4)))

    seen: dict[str, str] = {}
    conflicting: set[str] = set()
    for m in _REF_DEFINITION_RE.finditer(text):
        name, body = _ref_name(m), _squash(m.group(4))
        if name in seen and seen[name] != body and len(source_bodies.get(name, ())) == 1:
            conflicting.add(name)
        seen.setdefault(name, body)
    if not conflicting:
        return text, []

    first: set[str] = set()
    collapsed: list[str] = []

    def _collapse(match: re.Match[str]) -> str:
        name = _ref_name(match)
        if name not in conflicting:
            return match.group(0)
        if name not in first:
            first.add(name)
            return match.group(0)
        if name not in collapsed:
            collapsed.append(name)
        # Keep the opening tag's attributes: a reuse must name the same group.
        opening = re.match(r"<ref\b[^>]*>", match.group(0)).group(0)
        return opening[:-1].rstrip() + " />"

    return _REF_DEFINITION_RE.sub(_collapse, text), collapsed


_REF_TAG_RE = re.compile(
    r"<ref\b(?P<attrs>[^>]*?)(?:/>|(?<!/)>(?P<body>(?:(?!<ref\b).)*?)</ref\s*>)",
    re.IGNORECASE | re.DOTALL,
)
_REF_NAME_ATTR_RE = re.compile(_REF_NAME_ATTR.replace(r"\bname", r"\s*\bname", 1), re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s|}\]<]+")


def _ref_fingerprint(body: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """URLs and digit runs of a ref body: what survives translation of a
    citation (titles and month names change, addresses and numbers don't)."""
    return tuple(_URL_RE.findall(body)), tuple(re.findall(r"\d+", _URL_RE.sub("", body)))


def restore_ref_names(text: str, source_text: str) -> tuple[str, dict[str, list[str | None]]]:
    """Undo the model renaming refs (`Brubaker et al-4` -> `Brubaker et-4`).

    A ref name is an identifier, never prose, but the model sometimes
    rewrites it, or names a ref the source left unnamed — and once several
    different citations share one invented name Cite reports it as defined
    multiple times with different content (Cluj-Napoca: seven Brubaker
    citations under one name).

    Only names the source never uses are touched. Each definition under such
    a name is matched to the source definition with the same URLs and
    numbers; it takes that definition's name, or loses its name when the
    source ref was unnamed. A definition with no unambiguous match is left
    alone. Reuses of the name follow only when its matched definitions
    agree on exactly one source name.

    Returns (patched_text, {translated_name: [restored names; None = unnamed]})."""
    source_tags = list(_REF_TAG_RE.finditer(source_text))
    source_names: set[str] = set()
    by_fingerprint: dict[tuple, list[str | None]] = {}
    for tag in source_tags:
        name_match = _REF_NAME_ATTR_RE.search(tag.group("attrs"))
        name = _ref_name(name_match) if name_match else None
        if name:
            source_names.add(name)
        if tag.group("body") is not None:
            by_fingerprint.setdefault(_ref_fingerprint(tag.group("body")), []).append(name)

    tags = list(_REF_TAG_RE.finditer(text))
    definition_targets: dict[int, str | None] = {}
    named_targets: dict[str, set[str]] = {}
    for index, tag in enumerate(tags):
        name_match = _REF_NAME_ATTR_RE.search(tag.group("attrs"))
        if not name_match or tag.group("body") is None:
            continue
        name = _ref_name(name_match)
        if name in source_names:
            continue
        fingerprint = _ref_fingerprint(tag.group("body"))
        candidates = by_fingerprint.get(fingerprint, []) if any(fingerprint) else []
        if not candidates or len(set(candidates)) != 1:
            continue
        definition_targets[index] = candidates[0]
        if candidates[0] is not None:
            named_targets.setdefault(name, set()).add(candidates[0])

    changes: dict[str, list[str | None]] = {}
    edits: list[tuple[int, int, str]] = []
    for index, tag in enumerate(tags):
        name_match = _REF_NAME_ATTR_RE.search(tag.group("attrs"))
        if not name_match:
            continue
        name = _ref_name(name_match)
        if name in source_names:
            continue
        if tag.group("body") is not None:
            if index not in definition_targets:
                continue
            target = definition_targets[index]
        else:
            targets = named_targets.get(name, set())
            if len(targets) != 1:
                continue
            (target,) = targets
        start = tag.start("attrs") + name_match.start()
        end = tag.start("attrs") + name_match.end()
        edits.append((start, end, f' name="{target}"' if target else ""))
        changes.setdefault(name, []).append(target)

    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text, changes
