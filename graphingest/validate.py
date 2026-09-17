"""Validate causal graphs against a LinkML schema.

Use it on anything the pipeline produces, or on a graph you edited by hand:

    python -m graphingest.validate graph.yaml
    python -m graphingest.validate out/*.yaml --schema schema/causalmosaic.yaml
    python -m graphingest.validate graph.yaml --json report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .cli import configure_stdio
from .graph_io import atomic_write_json, load_graph, validate_graph
from .schema import DEFAULT_SCHEMA

LOGGER = logging.getLogger("ingest.validate")


def validate_paths(
    paths: list[Path], schema_path: Path, top_class: str = "CausalGraph"
) -> list[dict]:
    """Validate each graph, returning one report dict per path."""
    reports = []
    for path in paths:
        try:
            graph = load_graph(path)
        except (OSError, ValueError) as error:
            reports.append(
                {
                    "path": str(path),
                    "ok": False,
                    "problems": [f"could not load: {error}"],
                    "node_count": 0,
                    "edge_count": 0,
                    "schema_version": "unknown",
                }
            )
            continue
        report = validate_graph(graph, schema_path, top_class, path=str(path))
        reports.append(
            {
                "path": str(path),
                "ok": report.ok,
                "problems": report.problems,
                "node_count": report.node_count,
                "edge_count": report.edge_count,
                "schema_version": report.schema_version,
            }
        )
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("graphs", nargs="+", type=Path, help="Graph YAML/JSON")
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help=f"LinkML schema (default: {DEFAULT_SCHEMA.name})",
    )
    parser.add_argument("--top-class", default="CausalGraph")
    parser.add_argument("--json", type=Path, help="Write a JSON report here")
    parser.add_argument(
        "--max-problems",
        type=int,
        default=20,
        help="Problems to print per graph (default: 20; 0 for all)",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print failures")
    args = parser.parse_args(argv)

    configure_stdio()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    reports = validate_paths(args.graphs, args.schema, args.top_class)

    failed = 0
    for report in reports:
        status = "PASS" if report["ok"] else "FAIL"
        if not report["ok"]:
            failed += 1
        if report["ok"] and args.quiet:
            continue
        print(
            f"{status}  {report['path']}  "
            f"(schema {report['schema_version']}, "
            f"{report['node_count']} nodes, {report['edge_count']} edges)"
        )
        problems = report["problems"]
        shown = problems if args.max_problems == 0 else problems[: args.max_problems]
        for problem in shown:
            print(f"    - {problem}")
        if len(problems) > len(shown):
            print(f"    ... and {len(problems) - len(shown)} more")

    if args.json:
        atomic_write_json(
            args.json,
            {
                "schema": str(args.schema),
                "passed": len(reports) - failed,
                "failed": failed,
                "reports": reports,
            },
        )
        print(f"\nJSON report written to {args.json}")

    print(f"\n{len(reports) - failed}/{len(reports)} graph(s) valid")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
