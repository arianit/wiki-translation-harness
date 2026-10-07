"""Saves final MediaWiki source files. No publishing — local .wiki files only."""

from __future__ import annotations

import re
from pathlib import Path

from wiki_translation_harness.models import Chunk, ChunkStatus

_UNSAFE_RE = re.compile(r'[\\/:*?"<>|]')


def sanitize_filename(title: str) -> str:
    """Article title -> filesystem-safe base name, MediaWiki-style (spaces -> underscores)."""
    name = title.strip().replace(" ", "_")
    name = _UNSAFE_RE.sub("_", name)
    return name


def _assemble_with_spans(chunks: list[Chunk]) -> tuple[str, list[tuple[int, int]]]:
    """Shared core for assemble_chunks/assemble_chunks_with_spans. Returns
    the joined text plus each chunk's 1-indexed (start_line, end_line)
    span in that text — end_line < start_line for a chunk that contributed
    no lines (empty translated_text)."""
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    current_line = 1
    for chunk in chunks:
        text = chunk.translated_text or ""
        if pieces and pieces[-1] and not pieces[-1].endswith("\n") and text and not text.startswith("\n"):
            pieces.append("\n")
            current_line += 1
        start_line = current_line
        pieces.append(text)
        if text:
            current_line += text.count("\n")
            end_line = current_line - 1 if text.endswith("\n") else current_line
        else:
            end_line = start_line - 1
        spans.append((start_line, end_line))
    return "".join(pieces), spans


def assemble_chunks(chunks: list[Chunk]) -> str:
    """Concatenate translated chunks in order, guaranteeing a newline at every
    chunk boundary. Models don't always faithfully preserve trailing
    whitespace, and a missing newline right before a `==` heading silently
    breaks it (MediaWiki only recognizes a heading at the start of a line) —
    this is a purely mechanical safety net, independent of prompting."""
    return _assemble_with_spans(chunks)[0]


def assemble_chunks_with_spans(chunks: list[Chunk]) -> tuple[str, list[tuple[Chunk, int, int]]]:
    """Like assemble_chunks, but also returns each chunk's 1-indexed
    (start_line, end_line) span in the assembled text — used by the
    assembly-level repair loop (pipeline.py) to map a validation finding's
    line_number back to the chunk that produced it."""
    text, spans = _assemble_with_spans(chunks)
    return text, [(chunk, start, end) for chunk, (start, end) in zip(chunks, spans)]


def output_path_for(output_dir: Path, title: str) -> Path:
    return output_dir / f"{sanitize_filename(title)}.wiki"


def article_already_done(output_dir: Path, title: str) -> bool:
    return output_path_for(output_dir, title).exists()


def save_article(output_dir: Path, title: str, wikitext: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_path_for(output_dir, title)
    path.write_text(wikitext, encoding="utf-8")
    return path


_DONE_CHUNK_STATUSES = {ChunkStatus.TRANSLATED, ChunkStatus.CACHED, ChunkStatus.REPAIRED}


def assemble_chunks_partial(chunks: list[Chunk]) -> str:
    """Like assemble_chunks, but a chunk that hasn't succeeded yet
    (PENDING/FAILED) is replaced with an inert HTML-comment placeholder
    naming its section instead of silently contributing nothing — so a
    partial snapshot never reads as if content were dropped. Only used for
    partial_output_dir; never for the real output_dir save."""
    shadow = [
        chunk if chunk.status in _DONE_CHUNK_STATUSES
        else chunk.model_copy(update={
            "translated_text": f"<!-- pending: {chunk.section_title} ({chunk.status.value}) -->"
        })
        for chunk in chunks
    ]
    return assemble_chunks(shadow)


def save_partial_article(partial_output_dir: Path, title: str, chunks: list[Chunk]) -> Path:
    """Kill-safe snapshot of an in-progress article, rewritten after every
    chunk. Atomic tmp-write + replace so a process killed mid-write can
    never leave a truncated partial file behind."""
    partial_output_dir.mkdir(parents=True, exist_ok=True)
    path = output_path_for(partial_output_dir, title)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(assemble_chunks_partial(chunks), encoding="utf-8")
    tmp_path.replace(path)
    return path


def discard_partial_article(partial_output_dir: Path, title: str) -> None:
    output_path_for(partial_output_dir, title).unlink(missing_ok=True)
    output_path_for(partial_output_dir, title).with_suffix('.resume.json').unlink(missing_ok=True)
