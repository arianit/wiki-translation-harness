"""Wikitext sectioning and token-budget chunking, via mwparserfromhell.

Rules (from spec): split at section boundaries first; never split a chunk
inside a template, table, reference, list, or parser function; target
~1500-2500 tokens per chunk; preserve ordering.
"""

from __future__ import annotations

import re

import mwparserfromhell as mwp
import tiktoken

from wiki_translation_harness.models import Chunk, Section

_ENCODING = tiktoken.get_encoding("cl100k_base")

_LIST_LINE_RE = re.compile(r"^[ \t]*[*#:;]+")
_BLANK_LINE_SPLIT_RE = re.compile(r"(\n[ \t]*\n)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(\s+)")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_ENCODING.encode(text))


def split_into_sections(wikitext: str) -> list[Section]:
    """Split article wikitext into ordered, non-overlapping sections.

    Each section's text includes its own heading line (if any) so the
    heading gets translated along with its body, in order.
    """
    code = mwp.parse(wikitext)
    raw_sections = code.get_sections(flat=True, include_lead=True, include_headings=True)

    sections: list[Section] = []
    order = 0
    for raw in raw_sections:
        text = str(raw)
        if not text.strip():
            continue
        headings = raw.filter_headings()
        if headings:
            title = str(headings[0].title).strip()
            level = headings[0].level
        else:
            title = "Lead"
            level = 0
        sections.append(Section(title=title, level=level, order=order, wikitext=text))
        order += 1
    return sections


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _find_protected_spans(text: str) -> list[tuple[int, int]]:
    """Char ranges that must never be split: templates/parser-functions,
    tables, and <ref> tags (via mwparserfromhell), plus contiguous list
    blocks (via line-based regex, since mwparserfromhell only tags the
    bullet marker itself, not the whole list line/run)."""
    spans: list[tuple[int, int]] = []

    code = mwp.parse(text)
    offset = 0
    for node in code.nodes:
        s = str(node)
        node_type = type(node).__name__
        if node_type == "Template":
            spans.append((offset, offset + len(s)))
        elif node_type == "Tag":
            tag_name = str(getattr(node, "tag", "")).lower()
            if tag_name in ("table", "ref"):
                spans.append((offset, offset + len(s)))
            elif tag_name == "references" and node.contents is not None:
                # A list-defined <references> block is one top-level node,
                # usually far too big to protect whole — but its inner
                # <ref> definitions must still never be cut. Unprotected,
                # the sentence splitter split `... p. 23.</ref>` after
                # "p.", the model dropped the stray `</ref>` opening the
                # next chunk, and the unclosed definition swallowed the
                # following one (Albert Einstein, ref "iJwuX").
                inner = str(node.contents)
                inner_offset = offset + s.find(inner) if inner else offset
                for child in node.contents.nodes:
                    cs = str(child)
                    if type(child).__name__ == "Template" or (
                        type(child).__name__ == "Tag"
                        and str(getattr(child, "tag", "")).lower() == "ref"
                    ):
                        spans.append((inner_offset, inner_offset + len(cs)))
                    inner_offset += len(cs)
        offset += len(s)

    spans.extend(_find_list_spans(text))
    return _merge_spans(spans)


def _find_list_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    end = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        is_list = bool(stripped.strip()) and bool(_LIST_LINE_RE.match(stripped))
        if is_list:
            if start is None:
                start = offset
            end = offset + len(line)
        else:
            if start is not None:
                spans.append((start, end))
                start = None
        offset += len(line)
    if start is not None:
        spans.append((start, end))
    return spans


def _split_free_text(text: str, chunk_max: int) -> list[str]:
    """Split unprotected text into paragraph units, falling back to
    sentence splitting only if a single paragraph alone exceeds chunk_max."""
    if not text:
        return []
    paragraphs = [p for p in _BLANK_LINE_SPLIT_RE.split(text) if p]
    units: list[str] = []
    for part in paragraphs:
        if _BLANK_LINE_SPLIT_RE.fullmatch(part):
            # a blank-line separator: attach to previous unit to preserve exact text
            if units:
                units[-1] += part
            else:
                units.append(part)
            continue
        if estimate_tokens(part) > chunk_max:
            pieces = _SENTENCE_SPLIT_RE.split(part)
            sentences: list[str] = []
            for piece in pieces:
                if not piece:
                    continue
                if _SENTENCE_SPLIT_RE.fullmatch(piece) and sentences:
                    sentences[-1] += piece
                else:
                    sentences.append(piece)
            units.extend(sentences)
        else:
            units.append(part)
    return units


def _split_oversized_section(text: str, chunk_min: int, chunk_max: int) -> list[str]:
    """Break one section's text into multiple chunk texts, never splitting
    a protected span (template/table/ref/list) even if it alone exceeds
    chunk_max."""
    # Re-derive units with paragraph-level fallback now sized against chunk_max.
    protected_spans = _find_protected_spans(text)
    units: list[tuple[str, bool]] = []
    pos = 0
    for start, end in protected_spans:
        if start > pos:
            units.extend((u, False) for u in _split_free_text(text[pos:start], chunk_max))
        units.append((text[start:end], True))
        pos = end
    if pos < len(text):
        units.extend((u, False) for u in _split_free_text(text[pos:], chunk_max))

    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            chunks.append("".join(buf))
            buf = []
            buf_tokens = 0

    for unit_text, is_protected in units:
        t = estimate_tokens(unit_text)
        if t > chunk_max:
            # oversized on its own (a huge template/table/ref/list, or a
            # paragraph that survived sentence-splitting still too large):
            # never split it further, ship it alone. If a small buffer is
            # already pending (e.g. just a section heading whose entire
            # body is this oversized table), merge it in rather than
            # flushing it as its own near-empty chunk first — confirmed in
            # practice: a lone "== 21st century ==" heading, with all its
            # body content pulled into the following oversized unit, was
            # sent to the model as a 6-token chunk with nothing to
            # translate, and it fabricated a replacement instead of just
            # echoing the heading back.
            if buf and buf_tokens < chunk_min:
                chunks.append("".join(buf) + unit_text)
                buf, buf_tokens = [], 0
            else:
                flush()
                chunks.append(unit_text)
            continue
        if buf_tokens + t > chunk_max and buf:
            flush()
        buf.append(unit_text)
        buf_tokens += t
        if buf_tokens >= chunk_min:
            flush()

    if buf:
        # A trailing remainder too small to reach chunk_min on its own —
        # confirmed in practice: a huge table got shipped as its own
        # oversized chunk, leaving a 1-token leftover (just trailing
        # whitespace) as a separate final chunk. Sent almost nothing to
        # translate, the model didn't just echo it back — it fabricated a
        # plausible-looking replacement section instead. Merging into the
        # previous chunk (already an accepted oversized exception, so a
        # few extra tokens is harmless) avoids ever shipping a
        # near-content-free chunk on its own.
        if chunks and buf_tokens < chunk_min:
            chunks[-1] += "".join(buf)
        else:
            flush()

    return chunks if chunks else [text]


_INFOBOX_RE = re.compile(r"\{\{\s*[Ii]nfobox")
_TABLE_ROW_RE = re.compile(r"\|-")
_REF_RE = re.compile(r"<ref", re.IGNORECASE)
_TEMPLATE_RE = re.compile(r"\{\{")
_NAVBOX_RE = re.compile(r"\{\{\s*[Nn]avbox")

# Thresholds for classifying a chunk as complex
_COMPLEX_MIN_TABLE_ROWS = 5  # tables with 5+ rows
_COMPLEX_REF_DENSITY = 15  # refs per 1000 tokens
_COMPLEX_TEMPLATE_DENSITY = 30  # templates per 1000 tokens


def classify_chunk_complexity(text: str) -> bool:
    """Heuristic classifier for chunk complexity.

    Returns True for chunks containing markup structures where
    structural integrity matters most: Infoboxes, large tables,
    dense reference lists, navboxes/sidebars, and template-heavy
    sections. Routed to ``complex_model`` when configured.
    """
    # 1. Infobox — the single most markup-critical structure
    if _INFOBOX_RE.search(text):
        return True

    # 2. Navboxes / sidebars — complex nested templates
    if _NAVBOX_RE.search(text):
        return True

    total_tokens = estimate_tokens(text)
    if total_tokens == 0:
        return False

    tokens_k = total_tokens / 1000.0

    # 3. Tables with many rows
    table_rows = len(_TABLE_ROW_RE.findall(text))
    if table_rows >= _COMPLEX_MIN_TABLE_ROWS:
        return True

    # 4. Dense reference lists
    ref_count = len(_REF_RE.findall(text))
    if tokens_k > 0 and ref_count / tokens_k > _COMPLEX_REF_DENSITY:
        return True

    # 5. Template-heavy content
    template_count = len(_TEMPLATE_RE.findall(text))
    if tokens_k > 0 and template_count / tokens_k > _COMPLEX_TEMPLATE_DENSITY:
        return True

    return False


def build_chunks(
    article_title: str,
    sections: list[Section],
    chunk_min: int = 1500,
    chunk_max: int = 2500,
    source_lang: str = "en",
) -> list[Chunk]:
    """Merge small sections and split oversized ones into ordered Chunks.

    Section boundaries are always chunk boundaries except when merging
    consecutive small sections to stay near the target token window
    (reduces per-request skill-prompt overhead / cost). A section is only
    split internally when it alone exceeds chunk_max.
    """
    chunks: list[Chunk] = []
    order = 0
    pending: list[Section] = []
    pending_tokens = 0

    def flush_pending() -> None:
        nonlocal pending, pending_tokens, order
        if not pending:
            return
        combined_text = "".join(s.wikitext for s in pending)
        chunks.append(
            Chunk(
                article_title=article_title,
                section_titles=[s.title for s in pending],
                order=order,
                text=combined_text,
                token_estimate=estimate_tokens(combined_text),
                source_lang=source_lang,
                is_complex=classify_chunk_complexity(combined_text),
            )
        )
        order += 1
        pending = []
        pending_tokens = 0

    for section in sections:
        sec_tokens = estimate_tokens(section.wikitext)

        if sec_tokens > chunk_max:
            flush_pending()
            for sub_text in _split_oversized_section(section.wikitext, chunk_min, chunk_max):
                chunks.append(
                    Chunk(
                        article_title=article_title,
                        section_titles=[section.title],
                        order=order,
                        text=sub_text,
                        token_estimate=estimate_tokens(sub_text),
                        source_lang=source_lang,
                        is_complex=classify_chunk_complexity(sub_text),
                    )
                )
                order += 1
            continue

        if pending_tokens + sec_tokens > chunk_max and pending:
            flush_pending()

        pending.append(section)
        pending_tokens += sec_tokens

        if pending_tokens >= chunk_min:
            flush_pending()

    flush_pending()
    return chunks
