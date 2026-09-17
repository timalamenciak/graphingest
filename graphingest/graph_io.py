"""Read, write, and schema-validate CAMO causal graphs.

Writes are atomic (temp file + ``os.replace``) so an interrupted batch run
never leaves a half-written graph behind — the same discipline camo_extract
uses, kept because these pipelines are long enough to get killed mid-run.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from .schema import DEFAULT_SCHEMA, json_schema_for

LOGGER = logging.getLogger("camo.graph")


class GraphValidationError(ValueError):
    """Raised when a graph fails schema or referential-integrity checks."""

    def __init__(self, message: str, problems: list[str]):
        super().__init__(message)
        self.problems = problems


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_graph(path: str | Path) -> dict:
    """Load a CAMO graph from ``.yaml``/``.yml``/``.json``."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Graph not found: {resolved}")
    text = resolved.read_text(encoding="utf-8")
    if resolved.suffix.lower() == ".json":
        graph = json.loads(text)
    else:
        graph = yaml.safe_load(text)
    if not isinstance(graph, dict):
        raise ValueError(f"Graph root must be a mapping: {resolved}")
    return graph


def save_graph(graph: dict, path: str | Path) -> Path:
    """Write a graph atomically, choosing format from the suffix."""
    resolved = Path(path)
    if resolved.suffix.lower() == ".json":
        text = json.dumps(graph, indent=2, ensure_ascii=False) + "\n"
    else:
        text = yaml.safe_dump(graph, sort_keys=False, allow_unicode=True)
    return atomic_write(resolved, text)


def save_graph_both(graph: dict, directory: str | Path, stem: str = "causal_graph") -> tuple[Path, Path]:
    """Write both ``<stem>.yaml`` and ``<stem>.json`` into ``directory``."""
    base = Path(directory)
    return (
        save_graph(graph, base / f"{stem}.yaml"),
        save_graph(graph, base / f"{stem}.json"),
    )


def atomic_write(path: str | Path, text: str) -> Path:
    """Write text to ``path`` via a temp file in the same directory."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=resolved.parent, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, resolved)
    return resolved


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Outcome of validating one graph. Falsy when problems were found."""

    path: Optional[str]
    schema_version: str
    node_count: int
    edge_count: int
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def __bool__(self) -> bool:
        return self.ok

    def summary(self) -> str:
        status = "PASS" if self.ok else f"FAIL ({len(self.problems)} problem(s))"
        return (
            f"{status}  schema {self.schema_version}  "
            f"{self.node_count} node(s)  {self.edge_count} edge(s)"
        )


def validate_graph(
    graph: dict,
    schema_path: str | Path | None = None,
    top_class: str = "CausalGraph",
    path: Optional[str] = None,
) -> ValidationReport:
    """Validate a graph against the LinkML schema plus referential integrity.

    JSON Schema catches shape and enum violations; it cannot check that an
    edge's ``subject`` names a node that exists, so those checks run here.
    Returns a report rather than raising, so batch callers can tally failures.
    """
    problems: list[str] = []
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    import jsonschema

    schema = json_schema_for(schema_path or DEFAULT_SCHEMA, top_class)
    validator = jsonschema.Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(graph), key=lambda e: list(e.path)):
        location = "/".join(str(part) for part in error.path) or "<root>"
        problems.append(f"schema: {location}: {error.message}")

    problems.extend(check_referential_integrity(nodes, edges))

    return ValidationReport(
        path=path,
        schema_version=str(graph.get("schema_version", "unknown")),
        node_count=len(nodes),
        edge_count=len(edges),
        problems=problems,
    )


def check_referential_integrity(nodes: Iterable[dict], edges: Iterable[dict]) -> list[str]:
    """Check id uniqueness and that every edge reference resolves to a node."""
    problems: list[str] = []
    nodes = list(nodes)
    edges = list(edges)

    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        node_id = node.get("id")
        if not node_id:
            problems.append(f"nodes[{index}]: missing id")
            continue
        if node_id in node_ids:
            problems.append(f"nodes[{index}]: duplicate node id {node_id!r}")
        node_ids.add(node_id)

    edge_ids: set[str] = set()
    for index, edge in enumerate(edges):
        edge_id = edge.get("id")
        if not edge_id:
            problems.append(f"edges[{index}]: missing id")
        elif edge_id in edge_ids:
            problems.append(f"edges[{index}]: duplicate edge id {edge_id!r}")
        else:
            edge_ids.add(edge_id)

        label = edge_id or f"edges[{index}]"
        for role in ("subject", "object"):
            reference = _reference_id(edge.get(role))
            if reference is None:
                problems.append(f"{label}: missing {role}")
            elif reference not in node_ids:
                problems.append(f"{label}: {role} {reference!r} is not a known node id")

        # Annotation slots that point at nodes by id.
        for slot, key in (
            ("mediation", "mediator_node_ids"),
            ("moderation", "moderator_node_ids"),
        ):
            for reference in (edge.get(slot) or {}).get(key) or []:
                if reference not in node_ids:
                    problems.append(
                        f"{label}: {slot}.{key} {reference!r} is not a known node id"
                    )
        comparator_node = (edge.get("comparator") or {}).get("comparator_node_id")
        if comparator_node and comparator_node not in node_ids:
            problems.append(
                f"{label}: comparator.comparator_node_id {comparator_node!r} "
                "is not a known node id"
            )
    return problems


def _reference_id(value: Any) -> Optional[str]:
    """An edge endpoint may be a bare id or an inlined node object."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("id")
    return str(value)


# ---------------------------------------------------------------------------
# Convenience accessors used across the reasoning tools
# ---------------------------------------------------------------------------


def node_index(graph: dict) -> dict[str, dict]:
    """Map node id -> node object."""
    return {node["id"]: node for node in graph.get("nodes") or [] if node.get("id")}


def source_document_index(graph: dict) -> dict[str, dict]:
    """Map document_id -> SourceDocument, from the graph-level list."""
    documents = graph.get("source_documents") or []
    index = {
        document["document_id"]: document
        for document in documents
        if document.get("document_id")
    }
    single = graph.get("source_document")
    if isinstance(single, dict) and single.get("document_id"):
        index.setdefault(single["document_id"], single)
    return index


def edge_source_document(edge: dict, documents: dict[str, dict]) -> dict:
    """Resolve an edge's source document, whether referenced by id or inlined."""
    reference = edge.get("source_document")
    if isinstance(reference, dict):
        if reference.get("document_id") in documents:
            return documents[reference["document_id"]]
        return reference
    if isinstance(reference, str):
        return documents.get(reference, {"document_id": reference})
    return {}
