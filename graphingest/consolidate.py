"""Merge CAMO graphs, from chunks of one article or across a whole corpus.

Adapted from RacoonLab/repos/camo_extract/src/consolidator.py. Two changes
matter: schema version is read from the profile rather than pinned to 0.7.1,
and ``source_document`` is treated as a document_id *reference* into
graph-level ``source_documents``, which is how CAMO has worked since 0.7.8.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .normalize import stable_id

LOGGER = logging.getLogger("camo.consolidate")


@dataclass
class ConsolidationReport:
    graphs_merged: int = 0
    nodes_in: int = 0
    nodes_out: int = 0
    edges_in: int = 0
    edges_out: int = 0
    merged_node_groups: list[dict] = field(default_factory=list)
    #: Per input graph, in order: the id each node and edge had going in, to
    #: the id it has coming out. Not reported (it is as large as the graph);
    #: it is what lets anything keyed by the old ids follow along.
    node_id_maps: list[dict[str, str]] = field(default_factory=list)
    edge_id_maps: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "graphs_merged": self.graphs_merged,
            "nodes_in": self.nodes_in,
            "nodes_out": self.nodes_out,
            "nodes_merged": self.nodes_in - self.nodes_out,
            "edges_in": self.edges_in,
            "edges_out": self.edges_out,
            "edges_deduplicated": self.edges_in - self.edges_out,
            "merged_node_groups": self.merged_node_groups[:50],
        }


def node_identity(node: dict) -> tuple:
    """Semantic identity of a node: entity + attribute + qualifier + type.

    Two nodes naming the same state of the same thing are the same node even
    if two chunks phrased the label differently, which is what lets a
    per-chunk extraction reassemble into one article graph.
    """
    return (
        _norm(node.get("entity_term")),
        _norm(node.get("measured_attribute")),
        _norm(node.get("state_or_change_qualifier")),
        _norm(node.get("entity_type")),
        tuple(sorted(_norm(entry.get("entity_term")) for entry in node.get("applied_to") or [])),
    )


def edge_identity(edge: dict) -> tuple:
    return (
        edge.get("subject"),
        edge.get("predicate"),
        edge.get("object"),
        _norm(edge.get("original_sentence")),
        _document_ref(edge.get("source_document")),
    )


def _norm(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split()).lower()


def _document_ref(value: Any) -> Optional[str]:
    """Accept both a document_id reference and a legacy inlined object."""
    if isinstance(value, dict):
        return value.get("document_id") or value.get("doi")
    return None if value is None else str(value)


class Consolidator:
    """Merge many CAMO graphs into one, rewriting references as it goes."""

    def __init__(
        self,
        merge_duplicate_nodes: bool = True,
        deduplicate_edges: bool = True,
        schema_version: str = "0.7.9",
    ):
        self.merge_duplicate_nodes = merge_duplicate_nodes
        self.deduplicate_edges = deduplicate_edges
        self.schema_version = schema_version

    def consolidate(
        self, graphs: Iterable[dict], graph_id: Optional[str] = None
    ) -> tuple[dict, ConsolidationReport]:
        graphs = [graph for graph in graphs if graph]
        report = ConsolidationReport(graphs_merged=len(graphs))
        if not graphs:
            return self._empty(graph_id), report

        nodes_by_key: dict[tuple, dict] = {}
        source_ids: dict[tuple, list[str]] = {}
        edges: list[dict] = []
        edge_origins: list[tuple[int, str, dict]] = []
        documents: dict[str, dict] = {}

        for graph_index, graph in enumerate(graphs):
            id_map: dict[str, str] = {}
            report.node_id_maps.append(id_map)

            for original in graph.get("nodes") or []:
                report.nodes_in += 1
                node = deepcopy(original)
                key = (
                    node_identity(node)
                    if self.merge_duplicate_nodes
                    else (node.get("id"),)
                )
                if key in nodes_by_key:
                    self._merge_node(nodes_by_key[key], node)
                    source_ids[key].append(str(original.get("id")))
                else:
                    # Never trust the incoming id: per-document extractions
                    # number their nodes independently (n1, n2, ...), so the
                    # same literal id shows up in unrelated documents. Always
                    # derive it from the semantic identity instead, which is
                    # what keeps it both unique and stable across re-runs.
                    node["id"] = stable_id("camo:node_", *key)
                    nodes_by_key[key] = node
                    source_ids[key] = [str(original.get("id"))]
                id_map[str(original.get("id"))] = nodes_by_key[key]["id"]

            for original in graph.get("edges") or []:
                report.edges_in += 1
                edge = deepcopy(original)
                edge["subject"] = id_map.get(str(edge.get("subject")), edge.get("subject"))
                edge["object"] = id_map.get(str(edge.get("object")), edge.get("object"))
                for structure, key in (
                    ("mediation", "mediator_node_ids"),
                    ("moderation", "moderator_node_ids"),
                ):
                    block = edge.get(structure)
                    if isinstance(block, dict) and block.get(key):
                        block[key] = [
                            id_map.get(str(ref), ref) for ref in block[key]
                        ]
                comparator = edge.get("comparator")
                if isinstance(comparator, dict) and comparator.get("comparator_node_id"):
                    comparator["comparator_node_id"] = id_map.get(
                        str(comparator["comparator_node_id"]),
                        comparator["comparator_node_id"],
                    )
                edges.append(edge)
                edge_origins.append((graph_index, str(original.get("id")), edge))

            for document in graph.get("source_documents") or []:
                if document.get("document_id"):
                    merged = documents.setdefault(document["document_id"], deepcopy(document))
                    for field_name, value in document.items():
                        merged.setdefault(field_name, value)

        for key, ids in source_ids.items():
            if len(ids) > 1:
                report.merged_node_groups.append(
                    {"id": nodes_by_key[key]["id"], "merged_from": ids}
                )

        survivor: dict[int, dict] = {}
        if self.deduplicate_edges:
            unique: dict[tuple, dict] = {}
            for edge in edges:
                identity = edge_identity(edge)
                if identity in unique:
                    self._merge_edge(unique[identity], edge)
                else:
                    unique[identity] = edge
                survivor[id(edge)] = unique[identity]
            edges = list(unique.values())

        # Ids are only assigned now: they hash the final endpoint ids, so an
        # edge that survived node merging keeps a stable, content-derived id.
        # Never trust an incoming id here either, for the same reason as
        # nodes above: per-document extractions number their edges
        # independently (e1, e2, ...), so the same literal id turns up
        # across unrelated documents.
        for edge in edges:
            edge["id"] = stable_id(
                "camo:edge_",
                edge.get("subject"),
                edge.get("predicate"),
                edge.get("object"),
                edge.get("original_sentence"),
                _document_ref(edge.get("source_document")),
            )
        report.edge_id_maps = [{} for _ in graphs]
        for graph_index, original_id, edge in edge_origins:
            report.edge_id_maps[graph_index][original_id] = (
                survivor.get(id(edge), edge)["id"]
            )

        report.nodes_out = len(nodes_by_key)
        report.edges_out = len(edges)

        consolidated: dict[str, Any] = {
            "graph_id": graph_id
            or stable_id("camo:graph_", *(g.get("graph_id", "") for g in graphs)),
            "schema_version": self.schema_version,
            "provenance": deepcopy(graphs[0].get("provenance") or {}),
            "nodes": list(nodes_by_key.values()),
            "edges": edges,
        }
        if documents:
            consolidated["source_documents"] = list(documents.values())
        return consolidated, report

    def _empty(self, graph_id: Optional[str]) -> dict:
        return {
            "graph_id": graph_id or stable_id("camo:graph_", "empty"),
            "schema_version": self.schema_version,
            "provenance": {"ontology_framework": "ELMO + CAMO"},
            "nodes": [],
            "edges": [],
        }

    @staticmethod
    def _merge_node(existing: dict, incoming: dict) -> None:
        """Union text spans and fill gaps; never overwrite a settled value."""
        spans = existing.setdefault("source_spans", [])
        for span in incoming.get("source_spans") or []:
            if span not in spans:
                spans.append(span)
        if not spans:
            existing.pop("source_spans")
        applied = existing.setdefault("applied_to", [])
        for entry in incoming.get("applied_to") or []:
            if entry not in applied:
                applied.append(entry)
        if not applied:
            existing.pop("applied_to")
        for key, value in incoming.items():
            if key in {"id", "source_spans", "applied_to"}:
                continue
            if existing.get(key) in (None, "", []):
                existing[key] = value

    @staticmethod
    def _merge_edge(existing: dict, incoming: dict) -> None:
        spans = existing.setdefault("source_spans", [])
        for span in incoming.get("source_spans") or []:
            if span not in spans:
                spans.append(span)
        if not spans:
            existing.pop("source_spans")
        for key, value in incoming.items():
            if key in {"id", "source_spans"}:
                continue
            if existing.get(key) in (None, "", []):
                existing[key] = value
