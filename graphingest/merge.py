"""Merge per-document graphs into one, optionally into an existing graph.

    python -m graphingest.merge build/graphs --out build/causal_graph.yaml
    python -m graphingest.merge build/graphs --into ~/mosaic/causal_graph.yaml \\
        --out build/causal_graph.yaml --validate
    python -m graphingest.merge build/graphs --into ~/mosaic/causal_graph.yaml \\
        --in-place --validate

Nodes describing the same state of the same thing are merged across documents
(same entity_term + measured_attribute + qualifier + entity_type), which is
what turns a pile of per-paper graphs into a corpus-level evidence base: two
studies reporting "increased native richness" become one node with two incoming
edges rather than two disconnected islands.

``--into`` is what makes this an *ingest* rather than a rebuild. The existing
graph is merged in first, so its ``graph_id`` and provenance survive and the
new documents attach to nodes that are already there. It is read, never
written, unless you pass ``--in-place`` — and even then the previous version is
kept beside it as ``.bak`` first, because a merge that goes wrong should cost
you a rename and not a corpus.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .cli import configure_logging, configure_stdio
from .consolidate import Consolidator
from .graph_io import (
    atomic_write_json,
    load_graph,
    save_graph,
    validate_graph,
)
from .schema import DEFAULT_SCHEMA, load_schema, schema_version

LOGGER = logging.getLogger("ingest.merge")

GRAPH_SUFFIXES = ("*.yaml", "*.yml", "*.json")

#: Report files the pipeline writes next to the graphs. They are not graphs,
#: and picking them up would fail the load with a confusing error.
_NOT_GRAPHS = {"annotation_report.json", "conversion_report.json",
               "manifest.json", "merge_report.json", "validation.json"}


def collect_paths(inputs: list[Path]) -> list[Path]:
    """Expand directories into the graph files they contain."""
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            for suffix in GRAPH_SUFFIXES:
                paths.extend(
                    path
                    for path in sorted(item.glob(suffix))
                    if path.name not in _NOT_GRAPHS
                    and not path.name.endswith(".report.json")
                )
        elif item.exists():
            paths.append(item)
        else:
            LOGGER.warning("Input not found, skipping: %s", item)
    return paths


def merge_graphs(
    graphs: list[dict],
    schema_version_string: str,
    existing: Optional[dict] = None,
    graph_id: Optional[str] = None,
    merge_nodes: bool = True,
) -> tuple[dict, dict]:
    """Consolidate ``graphs``, with ``existing`` (if any) merged in first.

    Order matters: the Consolidator keeps the first graph's provenance and the
    first-seen value of any slot, so putting the existing graph first means an
    ingest adds to it rather than overwriting what is already recorded.
    """
    ordered = ([existing] if existing else []) + graphs
    consolidator = Consolidator(
        merge_duplicate_nodes=merge_nodes, schema_version=schema_version_string
    )
    merged, report = consolidator.consolidate(
        ordered, graph_id or (existing or {}).get("graph_id")
    )

    provenance = merged.setdefault("provenance", {})
    provenance.setdefault("ontology_framework", (existing or {}).get(
        "provenance", {}).get("ontology_framework", "CAMO"))
    # The Consolidator keeps the first graph's provenance, so a corpus merged
    # into an ungrounded existing graph would otherwise lose all record of what
    # grounded it. Carry the snapshots forward from whoever has them.
    snapshots = {
        snapshot
        for graph in ordered
        for snapshot in [(graph.get("provenance") or {}).get("ontology_snapshot_id")]
        if snapshot
    }
    if snapshots:
        provenance["ontology_snapshot_id"] = "; ".join(sorted(snapshots))
    provenance.update(
        {
            "causal_mosaic_version": schema_version_string,
            "created": datetime.now(timezone.utc).isoformat(),
            "source_corpus": (
                f"{len(graphs)} document graph(s)"
                + (" merged into an existing graph" if existing else "")
            ),
            "exporter_version": f"graphingest.merge/{_version()}",
        }
    )
    return merged, report.to_dict()


def _version() -> str:
    from . import __version__

    return __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("graphs", nargs="+", type=Path,
                        help="Graph files or directories of them")
    destination = parser.add_argument_group("output")
    destination.add_argument("--out", type=Path, help="Write the merged graph here")
    destination.add_argument("--out-dir", type=Path,
                             help="Write causal_graph.yaml/.json plus reports here")
    destination.add_argument(
        "--in-place", action="store_true",
        help="Write the result back over --into, keeping a .bak of the previous version",
    )

    parser.add_argument("--into", type=Path,
                        help="Existing graph to merge these into (read-only)")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--graph-id", help="Identifier for the merged graph")
    parser.add_argument(
        "--no-merge-nodes", action="store_true",
        help="Keep per-document nodes distinct instead of merging equivalent ones",
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_stdio()
    configure_logging(args.log_level)

    if not any((args.out, args.out_dir, args.in_place)):
        parser.error("give one of --out, --out-dir or --in-place")
    if args.in_place and not args.into:
        parser.error("--in-place needs --into: there is nothing to write back to")

    paths = collect_paths(args.graphs)
    if not paths:
        parser.error("no graph files found")
    LOGGER.info("Merging %d graph(s)", len(paths))

    graphs, sources = [], []
    for path in paths:
        try:
            graphs.append(load_graph(path))
            sources.append(str(path))
        except (OSError, ValueError) as error:
            LOGGER.error("Could not load %s: %s", path, error)

    existing = None
    if args.into:
        if args.into.exists():
            existing = load_graph(args.into)
            LOGGER.info(
                "Merging into %s (%d node(s), %d edge(s))",
                args.into.name,
                len(existing.get("nodes") or []),
                len(existing.get("edges") or []),
            )
        else:
            # Not an error: the first ingest into a new graph has nothing to
            # merge with, and should still produce one.
            LOGGER.warning("--into %s does not exist yet; creating it", args.into)

    version = schema_version(load_schema(args.schema))
    merged, report = merge_graphs(
        graphs, version, existing, args.graph_id, merge_nodes=not args.no_merge_nodes
    )

    written: list[Path] = []
    if args.out:
        written.append(save_graph(merged, args.out))
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        written.append(save_graph(merged, args.out_dir / "causal_graph.yaml"))
        written.append(save_graph(merged, args.out_dir / "causal_graph.json"))
    if args.in_place:
        if args.into.exists():
            backup = args.into.with_suffix(args.into.suffix + ".bak")
            shutil.copy2(args.into, backup)
            LOGGER.info("Previous graph kept as %s", backup.name)
        written.append(save_graph(merged, args.into))

    report_dir = args.out_dir or (args.out or written[0]).parent
    atomic_write_json(report_dir / "merge_report.json",
                      {"sources": sources, "into": str(args.into) if args.into else None,
                       **report})

    print(f"\nMerged {len(graphs)} graph(s)" + (f" into {args.into}" if args.into else ""))
    print(f"  nodes: {report['nodes_in']} in -> {report['nodes_out']} out "
          f"({report['nodes_merged']} merged)")
    print(f"  edges: {report['edges_in']} in -> {report['edges_out']} out "
          f"({report['edges_deduplicated']} deduplicated)")
    for path in written:
        print(f"  wrote {path}")

    if args.validate:
        result = validate_graph(merged, args.schema)
        atomic_write_json(
            report_dir / "validation.json",
            {"ok": result.ok, "problems": result.problems,
             "nodes": result.node_count, "edges": result.edge_count},
        )
        print(f"  validation: {result.summary()}")
        for problem in result.problems[:10]:
            print(f"      - {problem[:170]}")
        return 0 if result.ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
