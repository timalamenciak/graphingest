"""Read an external SQLite manifest without changing it.

The conversion stage's JSON manifest is intentionally writable: it records the
markdown and graph it produces.  A SQLite manifest owned by another pipeline is
different.  This adapter reads the document rows into that in-memory shape and
leaves the database untouched.
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

#: A drive-absolute Windows path, e.g. ``C:\foo`` or ``C:/foo``. ``Path`` only
#: recognises this as absolute on Windows itself; elsewhere it reads as a
#: relative path and gets silently (and wrongly) joined onto the db directory.
_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")


class SQLiteManifestError(ValueError):
    """The database does not expose enough information to annotate Markdown."""


_TABLE_PREFERENCE = ("documents", "document", "files", "file", "items", "manifest")
_PATH_COLUMNS = (
    "markdown_path", "markdown_file", "markdown_filepath", "md_path", "md_file",
    "relative_path", "local_path", "file_path", "filepath", "path", "file",
)
_METADATA_COLUMNS = {
    "document_id": ("document_id", "id", "uuid", "key"),
    "doi": ("doi", "doi_url"),
    "title": ("title", "name"),
    "authors": ("authors", "author"),
    "year": ("year", "publication_year", "published_year", "date"),
    "journal": ("journal", "publication", "container_title", "source"),
    "slug": ("slug", "filename", "stem"),
}


def load_sqlite_manifest(path: Path) -> list[dict[str, Any]]:
    """Return Markdown-bearing rows from a SQLite manifest.

    Supported layouts are deliberately modest and inspectable: a table with a
    path-like column (``markdown_path``, ``path``, ``file_path``, and common
    variants), optionally accompanied by normal bibliographic columns.  Paths
    relative to the database are resolved relative to the database directory.
    """
    if not path.is_file():
        raise FileNotFoundError(f"SQLite manifest not found: {path}")
    try:
        connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise SQLiteManifestError(f"Could not open SQLite manifest {path}: {error}") from error
    try:
        tables = _tables(connection)
        if "works" in {table.lower() for table in tables}:
            evidence_jam = _evidence_jam_rows(connection, path)
            if evidence_jam:
                return evidence_jam
        choices = []
        for table in tables:
            columns = _columns(connection, table)
            path_column = next((name for name in _PATH_COLUMNS if name in columns), None)
            if path_column:
                choices.append((table, columns, path_column))
        if not choices:
            listed = ", ".join(tables) or "no user tables"
            expected = ", ".join(_PATH_COLUMNS[:5])
            raise SQLiteManifestError(
                f"No document table with a Markdown path column was found in {path}. "
                f"Tables: {listed}. Expected a column such as {expected}."
            )
        order = {name: index for index, name in enumerate(_TABLE_PREFERENCE)}
        table, columns, path_column = min(choices, key=lambda item: order.get(item[0].lower(), len(order)))
        selected = [path_column]
        aliases: dict[str, str] = {}
        for output, candidates in _METADATA_COLUMNS.items():
            column = next((name for name in candidates if name in columns), None)
            if column and column not in selected:
                selected.append(column)
                aliases[column] = output
        quoted = ", ".join(_quote(column) for column in selected)
        records = connection.execute(f"SELECT {quoted} FROM {_quote(table)}").fetchall()
    except sqlite3.Error as error:
        raise SQLiteManifestError(f"Could not read SQLite manifest {path}: {error}") from error
    finally:
        connection.close()

    rows: list[dict[str, Any]] = []
    used_slugs: set[str] = set()
    for number, values in enumerate(records, 1):
        raw = dict(zip(selected, values))
        value = raw.get(path_column)
        if not isinstance(value, str) or not value.strip():
            continue
        if os.name != "nt" and _WINDOWS_ABS.match(value):
            raise SQLiteManifestError(
                f"{path} has a Windows-absolute Markdown path ({value!r}) that "
                "cannot be resolved on this host. Store manifest paths relative "
                "to the database (or as POSIX paths that exist here) instead."
            )
        # Normalise separators before splitting: a relative path written with
        # backslashes on Windows otherwise reads as one opaque filename on
        # POSIX, since Path there only treats "/" as a separator.
        markdown_path = Path(value.replace("\\", "/"))
        if markdown_path.suffix.lower() not in {".md", ".markdown", ".mdown"}:
            continue
        if not markdown_path.is_absolute():
            markdown_path = path.parent / markdown_path
        title = _value(raw, aliases, "title")
        identifier = _value(raw, aliases, "document_id") or _value(raw, aliases, "doi")
        slug = _unique_slug(_value(raw, aliases, "slug") or markdown_path.stem or title or f"document-{number}", used_slugs)
        rows.append({
            "slug": slug,
            "document_id": identifier or slug,
            "doi": _value(raw, aliases, "doi"),
            "title": title,
            "authors": _authors(_value(raw, aliases, "authors")),
            "year": _year(_value(raw, aliases, "year")),
            "journal": _value(raw, aliases, "journal"),
            "markdown_path": str(markdown_path),
        })
    if not rows:
        raise SQLiteManifestError(
            f"{path} has a path column in table {table!r}, but no rows pointing to Markdown files."
        )
    return rows


def _evidence_jam_rows(connection: sqlite3.Connection, manifest_path: Path) -> list[dict[str, Any]]:
    """Read EvidenceJam's ``works`` table and its sibling Markdown directory.

    Its ``content_path`` points at downloaded source bytes, while the rendered
    Markdown is deliberately kept separately as ``build/markdown/<OpenAlex
    id>.md``.  Treating the bytes as Markdown would silently feed PDFs and XML
    into the extractor, so only the rendered files are returned.
    """
    tables = {table.lower(): table for table in _tables(connection)}
    table = tables["works"]
    columns = _columns(connection, table)
    required = {"openalex_id", "title", "publication_year", "journal_name"}
    if not required.issubset(columns):
        return []
    markdown_dir = manifest_path.parent / "markdown"
    if not markdown_dir.is_dir():
        return []
    records = connection.execute(
        f"SELECT openalex_id, doi, title, publication_year, journal_name FROM {_quote(table)} "
        "WHERE openalex_id IS NOT NULL"
    ).fetchall()
    rows = []
    used_slugs: set[str] = set()
    for openalex_id, doi, title, year, journal in records:
        markdown_path = markdown_dir / f"{openalex_id}.md"
        if not markdown_path.is_file():
            continue
        rows.append({
            "slug": _unique_slug(openalex_id, used_slugs),
            "document_id": openalex_id,
            "doi": doi,
            "title": title,
            "authors": [],
            "year": _year(year),
            "journal": journal,
            "markdown_path": str(markdown_path),
        })
    return rows


def _tables(connection: sqlite3.Connection) -> list[str]:
    return [row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    )]


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1].lower() for row in connection.execute(f"PRAGMA table_info({_quote(table)})")}


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _value(raw: dict[str, Any], aliases: dict[str, str], name: str) -> Any:
    return next((raw[column] for column, output in aliases.items() if output == name), None)


def _authors(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"\s*;\s*|\s+and\s+", value) if part.strip()]
    return [str(value)]


def _year(value: Any) -> int | None:
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", str(value or ""))
    return int(match.group(1)) if match else None


def _unique_slug(value: Any, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "document"
    slug, suffix = base, 2
    while slug in used:
        slug = f"{base}-{suffix}"
        suffix += 1
    used.add(slug)
    return slug
