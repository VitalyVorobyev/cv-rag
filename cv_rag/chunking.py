from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cv_rag.tei_extract import Section


@dataclass(slots=True)
class Chunk:
    section_title: str
    text: str
    chunk_index: int
    char_start: int
    char_end: int


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _windowed_text(text: str, max_chars: int, overlap_chars: int) -> list[tuple[str, int, int]]:
    text = _normalize_whitespace(text)
    if not text:
        return []

    chunks: list[tuple[str, int, int]] = []
    cursor = 0
    text_len = len(text)

    while cursor < text_len:
        hard_end = min(cursor + max_chars, text_len)
        end = hard_end

        if hard_end < text_len:
            soft_break = text.rfind(" ", cursor + max_chars // 2, hard_end)
            if soft_break > cursor:
                end = soft_break

        snippet = text[cursor:end].strip()
        if snippet:
            chunks.append((snippet, cursor, end))

        if end >= text_len:
            break

        next_cursor = max(end - overlap_chars, cursor + 1)
        while next_cursor < text_len and text[next_cursor].isspace():
            next_cursor += 1
        cursor = next_cursor

    return chunks


def chunk_sections(
    sections: Iterable[Section],
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        windows = _windowed_text(
            text=section.text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        for snippet, start, end in windows:
            all_chunks.append(
                Chunk(
                    section_title=section.title.strip() or "Untitled",
                    text=snippet,
                    chunk_index=chunk_index,
                    char_start=start,
                    char_end=end,
                )
            )
            chunk_index += 1

    return all_chunks
