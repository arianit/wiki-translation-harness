"""HTML comments in a translated article: label the ones carried over from
the source, remove the ones the model wrote itself.

A source comment is an English Wikipedia editor's note (`<!-- Please do not
change this without consensus on the talk page -->`). Translated and left
unlabelled, it reads as if sq.wikipedia editors had written it — about a
talk-page consensus that doesn't exist there. So each one is prefixed with
COMMENT_MARKER.

The model also writes comments of its own: in a repair call that found
nothing to fix ("Nuk u identifikuan me siguri ndryshimet..." — the repair
frame allows wikitext only, so a comment is the one place it can explain),
or to note content it left out. Those don't belong in the article; they
are removed and returned so the report can list them.

Matching is per chunk: a translated comment is the source's if it pairs
with one of that chunk's source comments in order, at about the same
relative position or sharing its untranslatable tokens (URLs, numbers,
names, markup). The text is translated, so it can't be matched exactly.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from wiki_translation_harness.models import Chunk

COMMENT_MARKER = "Nga en.wiki:"

_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)
# Tokens a translation leaves alone: URLs, numbers, wiki markup, and
# capitalized words (names).
_INVARIANT_TOKEN_RE = re.compile(r"https?://\S+|\d[\d.,]*|\[\[[^\]]+\]\]|\{\{[^}]+\}\}|\b[A-Z][\w-]+")
# Pairing thresholds: relative positions within the chunk this close, or
# invariant tokens this similar, count as the same comment.
_MAX_POSITION_GAP = 0.2
_MIN_TOKEN_SIMILARITY = 0.5


def _comments(text: str) -> list[tuple[float, re.Match[str]]]:
    length = max(len(text), 1)
    return [(m.start() / length, m) for m in _COMMENT_RE.finditer(text)]


def _is_marked(body: str) -> bool:
    return body.strip().startswith(COMMENT_MARKER)


def _pair_score(source: tuple[float, re.Match[str]], translated: tuple[float, re.Match[str]]) -> float:
    """0 when the two can't be the same comment, else higher is better."""
    (source_pos, source_match), (translated_pos, translated_match) = source, translated
    source_body = source_match.group(1).strip()
    translated_body = translated_match.group(1).strip()
    if source_body == translated_body:
        return 3.0
    if _is_marked(translated_body):
        # Labelled on an earlier pass: already known to be a source
        # comment, even if a repair since moved it.
        return 2.0
    source_tokens = _INVARIANT_TOKEN_RE.findall(source_body)
    token_similarity = (
        SequenceMatcher(None, source_tokens, _INVARIANT_TOKEN_RE.findall(translated_body)).ratio()
        if source_tokens
        else 0.0
    )
    position_closeness = 1 - abs(source_pos - translated_pos) / _MAX_POSITION_GAP
    if token_similarity < _MIN_TOKEN_SIMILARITY and position_closeness <= 0:
        return 0.0
    return 1.0 + token_similarity + max(position_closeness, 0.0)


def _align(source: list, translated: list) -> set[int]:
    """Indexes of the translated comments that pair with a source comment,
    in order (an order-preserving maximum-score matching)."""
    if len(source) == len(translated) and all(
        _pair_score(s, t) > 0 for s, t in zip(source, translated)
    ):
        return set(range(len(translated)))
    rows, cols = len(source), len(translated)
    best = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            score = _pair_score(source[i - 1], translated[j - 1])
            best[i][j] = max(
                best[i - 1][j],
                best[i][j - 1],
                best[i - 1][j - 1] + score if score > 0 else 0.0,
            )
    paired: set[int] = set()
    i, j = rows, cols
    while i > 0 and j > 0:
        score = _pair_score(source[i - 1], translated[j - 1])
        if score > 0 and best[i][j] == best[i - 1][j - 1] + score:
            paired.add(j - 1)
            i, j = i - 1, j - 1
        elif best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return paired


def reconcile_comments(source_text: str, translated_text: str) -> tuple[str, list[str]]:
    """Label the source's comments in one chunk's translation and remove
    the rest. Idempotent: an already-labelled comment keeps its label.
    Returns (patched_text, removed_comment_bodies)."""
    translated = _comments(translated_text)
    if not translated:
        return translated_text, []
    paired = _align(_comments(source_text), translated)

    removed: list[str] = []
    parts: list[str] = []
    pos = 0
    for index, (_, match) in enumerate(translated):
        start, end = match.span()
        body = match.group(1)
        if index in paired:
            replacement = match.group(0) if _is_marked(body) else f"<!-- {COMMENT_MARKER} {body.strip()} -->"
        else:
            removed.append(body.strip())
            replacement = ""
            # A comment alone on its line takes the line with it.
            line_start = translated_text.rfind("\n", 0, start) + 1
            line_end = translated_text.find("\n", end)
            line_end = len(translated_text) if line_end == -1 else line_end
            if (
                not translated_text[line_start:start].strip()
                and not translated_text[end:line_end].strip()
                and line_start >= pos
            ):
                start, end = line_start, min(line_end + 1, len(translated_text))
        parts.append(translated_text[pos:start])
        parts.append(replacement)
        pos = end
    parts.append(translated_text[pos:])
    return "".join(parts), removed


def reconcile_chunk_comments(chunks: list[Chunk]) -> list[str]:
    """reconcile_comments over every chunk, in place. Returns the comment
    bodies removed, in article order."""
    removed: list[str] = []
    for chunk in chunks:
        if not chunk.translated_text:
            continue
        chunk.translated_text, chunk_removed = reconcile_comments(chunk.text, chunk.translated_text)
        removed.extend(chunk_removed)
    return removed
