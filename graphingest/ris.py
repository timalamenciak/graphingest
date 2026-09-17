"""Read a folder of PDFs plus an RIS export into a corpus manifest.

This is the front door of the pipeline. Point it at a directory — a Zotero
"Export Items" folder, or any folder with an ``.ris`` file and PDFs somewhere
underneath — and it produces one manifest row per document, carrying the
bibliographic metadata that later becomes a ``SourceDocument`` in the graph.

    python -m graphingest.ris /path/to/corpus --out build/manifest.json

Three matching strategies run in order, and the manifest records which one
produced each row, because a citation attached to the wrong PDF is worse than
no citation at all:

1. **The RIS file link** (``L1``/``L2``/``L4``/``UR``). Zotero writes a path
   relative to the export root, which is exactly right, so it is tried first.
2. **Filename similarity.** Failing a link, a PDF is matched to the record
   whose title and author-year it most resembles — Zotero's own
   "Author et al. - 2022 - Title.pdf" convention makes this reliable, but the
   score is recorded so a weak match is visible.
3. **Nothing.** A PDF no record claims still gets ingested, with metadata
   parsed out of its filename and ``match: unmatched_pdf``. A record whose PDF
   is missing from disk is kept too, as ``match: no_pdf``, so the manifest
   remains a full account of the export rather than of the successes.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urlparse

from .cli import configure_logging, configure_stdio
from .graph_io import atomic_write_json

LOGGER = logging.getLogger("ingest.ris")

#: RIS tag -> the field it feeds. Several tags map to the same field; the first
#: one present wins for single-valued fields.
_TITLE_TAGS = ("TI", "T1", "CT", "BT")
_JOURNAL_TAGS = ("T2", "JO", "JF", "JA", "J2")
_AUTHOR_TAGS = ("AU", "A1", "A2", "A3", "A4")
_FILE_TAGS = ("L1", "L2", "L4", "LK")
_YEAR_TAGS = ("PY", "Y1", "DA")

#: A line in an RIS file: two-letter tag, two spaces, hyphen, space, value.
#: Some exporters emit a single space or no space before the hyphen.
_RIS_LINE = re.compile(r"^([A-Z][A-Z0-9])\s{0,2}-\s?(.*)$")

#: Zotero wraps taxon names in the title so citation styles leave their case
#: alone. It is markup, not content.
_HTML_TAG = re.compile(r"<[^>]+>")

#: Zotero's attachment filename convention: "Author et al. - 2022 - Title.pdf".
_ZOTERO_FILENAME = re.compile(
    r"^(?P<authors>.+?)\s+-\s+(?P<year>\d{4})\s+-\s+(?P<title>.+)$"
)

#: Below this title-similarity score a filename match is not trusted.
MIN_FILENAME_MATCH = 0.45


# ---------------------------------------------------------------------------
# RIS parsing
# ---------------------------------------------------------------------------


def parse_ris(text: str) -> list[dict[str, list[str]]]:
    """Parse RIS text into a list of records, each tag -> list of values.

    Everything is kept as a list because RIS repeats tags for authors and
    keywords; callers take ``[0]`` where a single value is meant. Continuation
    lines (an indented line with no tag) are appended to the previous value,
    which is how long abstracts are wrapped.
    """
    records: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    last_tag: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        match = _RIS_LINE.match(line)
        if match is None:
            # A continuation of the previous field.
            if last_tag and current.get(last_tag):
                current[last_tag][-1] = f"{current[last_tag][-1]} {line.strip()}"
            continue

        tag, value = match.group(1), match.group(2).strip()
        if tag == "TY":
            if current:
                records.append(current)
            current = {"TY": [value]}
            last_tag = "TY"
            continue
        if tag == "ER":
            if current:
                records.append(current)
            current, last_tag = {}, None
            continue
        current.setdefault(tag, []).append(value)
        last_tag = tag

    if current:
        records.append(current)
    return records


def find_ris_files(corpus_dir: Path) -> list[Path]:
    """Every ``.ris`` file under ``corpus_dir``, shallowest first."""
    candidates = [
        path
        for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".ris", ".txt.ris"}
    ]
    return sorted(candidates, key=lambda path: (len(path.parts), str(path).lower()))


def find_pdfs(corpus_dir: Path) -> list[Path]:
    """Every PDF under ``corpus_dir``, in a stable order."""
    return sorted(
        (path for path in corpus_dir.rglob("*.pdf") if path.is_file()),
        key=lambda path: str(path).lower(),
    )


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------


def clean_text(value: str) -> str:
    """Strip the markup exporters leave in titles and collapse whitespace."""
    return " ".join(_HTML_TAG.sub("", value or "").split())


def first(record: dict[str, list[str]], tags: Iterable[str]) -> Optional[str]:
    for tag in tags:
        values = record.get(tag)
        if values and values[0].strip():
            return values[0].strip()
    return None


def parse_year(record: dict[str, list[str]]) -> Optional[int]:
    """Pull a four-digit year out of whichever date tag the exporter used."""
    for tag in _YEAR_TAGS:
        for value in record.get(tag) or []:
            match = re.search(r"(1[5-9]\d{2}|20\d{2}|21\d{2})", value)
            if match:
                return int(match.group(1))
    return None


def parse_doi(record: dict[str, list[str]]) -> Optional[str]:
    """Normalize a DOI from ``DO``, or dig one out of a ``UR``/``M3`` field."""
    for value in [
        *(record.get("DO") or []),
        *(record.get("M3") or []),
        *(record.get("UR") or []),
        *(record.get("N1") or []),
    ]:
        match = re.search(r"10\.\d{4,9}/\S+", value)
        if match:
            return match.group(0).rstrip(" .,;)").lower()
    return None


def record_file_paths(record: dict[str, list[str]], base_dir: Path) -> list[Path]:
    """Resolve the PDF paths an RIS record links to, relative to the export."""
    resolved: list[Path] = []
    for tag in _FILE_TAGS:
        for value in record.get(tag) or []:
            candidate = _as_local_path(value)
            if candidate is None:
                continue
            path = candidate if candidate.is_absolute() else base_dir / candidate
            resolved.append(path)
    return resolved


def _as_local_path(value: str) -> Optional[Path]:
    """Turn an RIS link value into a local path, or None if it is a web URL."""
    value = value.strip()
    if not value:
        return None
    if value.lower().startswith("internal-pdf://"):
        value = value[len("internal-pdf://") :]
    elif value.lower().startswith("file:"):
        parsed = urlparse(value)
        value = unquote(parsed.path)
        # file:///C:/... parses to /C:/..., which is not a Windows path.
        if re.match(r"^/[A-Za-z]:", value):
            value = value[1:]
    elif re.match(r"^[a-z][a-z0-9+.-]*://", value, re.IGNORECASE):
        return None  # http(s) and friends: not a file on this disk
    else:
        value = unquote(value)
    if not value.lower().endswith(".pdf"):
        return None
    return Path(value.replace("\\", "/"))


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def slugify(value: str, max_length: int = 70) -> str:
    """A filesystem- and URL-safe slug, stable across runs and platforms."""
    text = unicodedata.normalize("NFKD", value or "")
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_length].strip("_") or "document"


def build_slug(authors: list[str], year: Optional[int], title: Optional[str]) -> str:
    """``surname_year_first_words_of_title`` — readable in a directory listing."""
    surname = ""
    if authors:
        surname = slugify(authors[0].split(",")[0], 24)
    title_part = "_".join(slugify(title or "", 60).split("_")[:6])
    parts = [part for part in (surname, str(year) if year else "", title_part) if part]
    return "_".join(parts) or "document"


def _unique(slug: str, taken: set[str]) -> str:
    """Disambiguate a slug collision by suffix, so no row overwrites another."""
    if slug not in taken:
        taken.add(slug)
        return slug
    for index in range(2, 100):
        candidate = f"{slug}_{index}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
    raise ValueError(f"could not disambiguate slug {slug!r}")


# ---------------------------------------------------------------------------
# Filename matching
# ---------------------------------------------------------------------------


def _tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", (value or "").lower()) if len(token) > 2}


def title_similarity(candidate: str, title: str) -> float:
    """Jaccard overlap of word tokens. Cheap, and good enough for filenames."""
    left, right = _tokens(candidate), _tokens(title)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def parse_filename(path: Path) -> dict:
    """Recover author/year/title from a Zotero-style attachment filename."""
    stem = path.stem
    match = _ZOTERO_FILENAME.match(stem)
    if match:
        authors = [
            name.strip()
            for name in re.split(r"\s+and\s+|,\s*", match.group("authors"))
            if name.strip() and name.strip().lower() not in {"et al", "et al."}
        ]
        return {
            "authors": authors,
            "year": int(match.group("year")),
            "title": clean_text(match.group("title")),
        }
    return {"authors": [], "year": None, "title": clean_text(stem)}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(
    corpus_dir: Path,
    ris_paths: Optional[list[Path]] = None,
    include_unmatched: bool = True,
) -> dict:
    """Assemble the corpus manifest for ``corpus_dir``."""
    corpus_dir = corpus_dir.resolve()
    ris_paths = ris_paths if ris_paths is not None else find_ris_files(corpus_dir)
    pdfs = find_pdfs(corpus_dir)
    warnings: list[str] = []

    records: list[tuple[Path, dict[str, list[str]]]] = []
    for ris_path in ris_paths:
        text = ris_path.read_text(encoding="utf-8-sig", errors="replace")
        parsed = parse_ris(text)
        LOGGER.info("Parsed %d record(s) from %s", len(parsed), ris_path.name)
        records.extend((ris_path, record) for record in parsed)
    if not records:
        warnings.append(f"no RIS records found under {corpus_dir}")

    remaining = {path.resolve(): path for path in pdfs}
    documents: list[dict] = []
    taken: set[str] = set()

    for ris_path, record in records:
        title = clean_text(first(record, _TITLE_TAGS) or "")
        authors = [clean_text(name) for name in record.get("AU") or record.get("A1") or []]
        if not authors:
            authors = [
                clean_text(name)
                for tag in _AUTHOR_TAGS
                for name in record.get(tag) or []
            ]
        year = parse_year(record)
        doi = parse_doi(record)

        pdf_path, match_kind, score = _match_pdf(
            record, ris_path.parent, title, remaining
        )
        if pdf_path is not None:
            remaining.pop(pdf_path.resolve(), None)
        else:
            warnings.append(f"no PDF found for: {title or '<untitled>'}")

        slug = _unique(build_slug(authors, year, title or (pdf_path.stem if pdf_path else "")), taken)
        documents.append(
            _document_row(
                slug=slug,
                doi=doi,
                title=title or None,
                authors=authors,
                year=year,
                journal=clean_text(first(record, _JOURNAL_TAGS) or "") or None,
                publication_type=first(record, ("TY",)),
                volume=first(record, ("VL",)),
                issue=first(record, ("IS",)),
                pages=_pages(record),
                abstract=clean_text(first(record, ("AB", "N2")) or "") or None,
                keywords=[clean_text(word) for word in record.get("KW") or []],
                pdf_path=pdf_path,
                corpus_dir=corpus_dir,
                ris_path=ris_path,
                match=match_kind,
                match_score=score,
            )
        )

    if include_unmatched:
        for path in sorted(remaining.values(), key=lambda item: str(item).lower()):
            parsed = parse_filename(path)
            slug = _unique(
                build_slug(parsed["authors"], parsed["year"], parsed["title"]), taken
            )
            warnings.append(f"PDF not listed in any RIS record: {path.name}")
            documents.append(
                _document_row(
                    slug=slug,
                    doi=None,
                    title=parsed["title"] or None,
                    authors=parsed["authors"],
                    year=parsed["year"],
                    journal=None,
                    publication_type=None,
                    volume=None,
                    issue=None,
                    pages=None,
                    abstract=None,
                    keywords=[],
                    pdf_path=path,
                    corpus_dir=corpus_dir,
                    ris_path=None,
                    match="unmatched_pdf",
                    match_score=None,
                )
            )

    return {
        "corpus_dir": str(corpus_dir),
        "generated": datetime.now(timezone.utc).isoformat(),
        "ris_files": [str(path) for path in ris_paths],
        "counts": {
            "records": len(records),
            "pdfs_found": len(pdfs),
            "documents": len(documents),
            "with_pdf": sum(1 for row in documents if row["pdf_path"]),
            "unmatched_pdfs": sum(
                1 for row in documents if row["match"] == "unmatched_pdf"
            ),
            "missing_pdfs": sum(1 for row in documents if row["match"] == "no_pdf"),
        },
        "warnings": warnings,
        "documents": documents,
    }


def _pages(record: dict[str, list[str]]) -> Optional[str]:
    start, end = first(record, ("SP",)), first(record, ("EP",))
    if start and end:
        return f"{start}-{end}"
    return start or end


def _document_row(
    *,
    slug: str,
    doi: Optional[str],
    title: Optional[str],
    authors: list[str],
    year: Optional[int],
    journal: Optional[str],
    publication_type: Optional[str],
    volume: Optional[str],
    issue: Optional[str],
    pages: Optional[str],
    abstract: Optional[str],
    keywords: list[str],
    pdf_path: Optional[Path],
    corpus_dir: Path,
    ris_path: Optional[Path],
    match: str,
    match_score: Optional[float],
) -> dict:
    """One manifest row. ``document_id`` is the DOI when there is one.

    Preferring the DOI means two ingests of the same paper from different
    folders merge into one source document downstream, which is the whole point
    of merging into an existing graph.
    """
    return {
        "slug": slug,
        "document_id": doi or f"doc:{slug}",
        "doi": doi,
        "title": title,
        "authors": authors,
        "year": year,
        "journal": journal,
        "publication_type": publication_type,
        "volume": volume,
        "issue": issue,
        "pages": pages,
        "abstract": abstract,
        "keywords": keywords,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "pdf_relative": (
            str(pdf_path.relative_to(corpus_dir))
            if pdf_path and _is_relative_to(pdf_path, corpus_dir)
            else None
        ),
        "ris_file": str(ris_path) if ris_path else None,
        "match": match,
        "match_score": match_score,
    }


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base)
        return True
    except ValueError:
        return False


def _match_pdf(
    record: dict[str, list[str]],
    base_dir: Path,
    title: str,
    remaining: dict[Path, Path],
) -> tuple[Optional[Path], str, Optional[float]]:
    """Find this record's PDF: by RIS link first, then by filename similarity."""
    for candidate in record_file_paths(record, base_dir):
        resolved = candidate.resolve()
        if resolved in remaining:
            return remaining[resolved], "ris_link", 1.0
        if candidate.exists():
            # Linked but outside the corpus directory: still the right file.
            return candidate, "ris_link_external", 1.0

    if not title:
        return None, "no_pdf", None

    best_path, best_score = None, 0.0
    for path in remaining.values():
        score = title_similarity(path.stem, title)
        if score > best_score:
            best_path, best_score = path, score
    if best_path is not None and best_score >= MIN_FILENAME_MATCH:
        return best_path, "filename", round(best_score, 3)
    return None, "no_pdf", None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("corpus_dir", type=Path, help="Folder of PDFs with an RIS file")
    parser.add_argument("--out", type=Path, required=True, help="Manifest JSON path")
    parser.add_argument(
        "--ris",
        type=Path,
        nargs="*",
        help="Explicit RIS file(s); default is every .ris under the corpus folder",
    )
    parser.add_argument(
        "--skip-unmatched-pdfs",
        action="store_true",
        help="Ignore PDFs no RIS record claims instead of ingesting them",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)

    configure_stdio()
    configure_logging(args.log_level)

    if not args.corpus_dir.is_dir():
        parser.error(f"Not a directory: {args.corpus_dir}")

    manifest = build_manifest(
        args.corpus_dir,
        ris_paths=list(args.ris) if args.ris else None,
        include_unmatched=not args.skip_unmatched_pdfs,
    )
    atomic_write_json(args.out, manifest)

    counts = manifest["counts"]
    print(f"\n{counts['documents']} document(s) from {args.corpus_dir}")
    print(f"    {counts['records']:5d}  RIS record(s)")
    print(f"    {counts['pdfs_found']:5d}  PDF(s) on disk")
    print(f"    {counts['with_pdf']:5d}  document(s) with a PDF")
    if counts["missing_pdfs"]:
        print(f"    {counts['missing_pdfs']:5d}  record(s) with no PDF")
    if counts["unmatched_pdfs"]:
        print(f"    {counts['unmatched_pdfs']:5d}  PDF(s) with no RIS record")
    for warning in manifest["warnings"][:10]:
        print(f"    - {warning}")
    if len(manifest["warnings"]) > 10:
        print(f"    ... and {len(manifest['warnings']) - 10} more")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
