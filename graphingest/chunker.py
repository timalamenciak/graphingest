"""Section-aware chunking with trustworthy source offsets.

Vendored from RacoonLab/repos/camo_extract/src/chunker.py so the hackathon
packs are self-contained. Unmodified apart from this note.

Offsets are absolute into the original document, which is what lets the
extractor turn a model-quoted sentence back into a TextSpan with real
start_char/end_char rather than a bare string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    section: str = "unknown"
    page: Optional[int] = None
    line_range: Optional[tuple[int, int]] = None
    token_count: Optional[int] = None


class Chunker:
    def __init__(
        self,
        max_chunk_size: Optional[int] = 8000,
        overlap_size: int = 400,
        max_sentences: Optional[int] = None,
        max_characters: Optional[int] = 4000,
    ):
        character_limits = [
            limit for limit in (max_chunk_size, max_characters) if limit is not None
        ]
        self.max_characters = min(character_limits) if character_limits else None
        self.overlap_size = overlap_size
        self.max_sentences = max_sentences

    def chunk_text(
        self, text: str, section: str = "unknown", page: Optional[int] = None
    ) -> list[Chunk]:
        if not text or not text.strip():
            return []
        if self.max_characters is None and self.max_sentences is None:
            return [
                Chunk(
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    section="document",
                    line_range=(1, text.count("\n") + 1),
                    token_count=max(1, len(text) // 4),
                )
            ]
        result = []
        for section_chunk in self._split_by_sections(text, section, page):
            result.extend(self._chunk_section(section_chunk))
        return result

    def _split_by_sections(
        self, text: str, default_section: str, page: Optional[int]
    ) -> list[Chunk]:
        matches = list(re.finditer(r"(?m)^(?:#+\s+|\s*%section\s+)(.+)$", text))
        if not matches:
            return [Chunk(text, 0, len(text), default_section, page)]
        sections = []
        if matches[0].start() > 0 and text[: matches[0].start()].strip():
            sections.append(
                Chunk(
                    text[: matches[0].start()],
                    0,
                    matches[0].start(),
                    default_section,
                    page,
                )
            )
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            label = match.group(1).strip().lower()
            page_match = re.fullmatch(r"page\s+(\d+)", label)
            section_page = int(page_match.group(1)) if page_match else page
            sections.append(
                Chunk(
                    text[match.start() : end],
                    match.start(),
                    end,
                    label,
                    section_page,
                )
            )
        return sections

    def _chunk_section(self, section: Chunk) -> list[Chunk]:
        sentence_matches = list(
            re.finditer(r".*?(?:[.!?](?=\s|$)|$)", section.text, flags=re.DOTALL)
        )
        spans = [
            (match.start(), match.end())
            for match in sentence_matches
            if match.group().strip()
        ]
        if not spans:
            spans = [(0, len(section.text))]
        chunks: list[Chunk] = []
        start_index = 0
        while start_index < len(spans):
            first_start = spans[start_index][0]
            end_index = start_index
            while end_index + 1 < len(spans):
                candidate_end = spans[end_index + 1][1]
                exceeds_sentence_limit = (
                    self.max_sentences is not None
                    and end_index + 2 - start_index > self.max_sentences
                )
                exceeds_character_limit = (
                    self.max_characters is not None
                    and candidate_end - first_start > self.max_characters
                )
                if exceeds_sentence_limit or exceeds_character_limit:
                    break
                end_index += 1
            local_end = spans[end_index][1]
            local_start = first_start
            if chunks and self.overlap_size:
                local_start = max(0, local_start - self.overlap_size)
            value = section.text[local_start:local_end].strip()
            absolute_start = section.start_char + local_start
            absolute_end = section.start_char + local_end
            chunks.append(
                Chunk(
                    value,
                    absolute_start,
                    absolute_end,
                    section.section,
                    section.page,
                    (
                        section.text.count("\n", 0, local_start) + 1,
                        section.text.count("\n", 0, local_end) + 1,
                    ),
                    max(1, len(value) // 4),
                )
            )
            start_index = end_index + 1
        return chunks
