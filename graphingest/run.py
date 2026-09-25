"""Run the whole pipeline: a folder of PDFs in, a merged causal graph out.

    python -m graphingest.run "/path/to/mosquito_corpus" --out-dir build \\
        --into ~/mosaic/causal_graph.yaml \\
        --example examples/murphy_2005_camo_annotation_revised.yaml \\
        --domain "restoration ecology"

Four stages, each resumable and each with its own module if you want to drive
them one at a time:

    1. ris       find the RIS export and the PDFs, pair them up
    2. convert   PDF -> markdown
    3. annotate  markdown -> one causal graph per document, with every term
                 looked up in real ontologies -- ELMO first, then the public
                 ones -- in Python, never by the model, and every node matched
                 against the graph it is being added to
    4. merge     document graphs -> one graph, merged into any existing one

Everything lands under ``--out-dir``:

    build/manifest.json            what was found, and how each PDF was matched
    build/markdown/<slug>.md       converted articles
    build/graphs/<slug>.yaml       per-document graphs, one per article
    build/graphs/<slug>.report.json  what the model did, per article
    build/grounding_cache.json     every ontology lookup, reused across runs
    build/causal_graph.yaml/.json  the merged result
    build/merge_report.json        what merged with what
    build/validation.json          the validator's verdict

With ``--confidence`` (token logprobs from an OpenAI-compatible endpoint):

    build/graphs/<slug>.confidence.json  per node and edge: scores, the
                                   alternatives the model weighed, every token
    build/confidence_summary.md    for reviewers: buckets, what to check first
    build/confidence_report.json   the same, as data
    build/confidence_index.json    the sidecars re-keyed to the merged graph

With ``--ensemble`` (witness models; the graph stays the primary's):

    build/graphs/<slug>.agreement.json   per node and edge: agreed, partial,
                                   conflict or unsupported, and what differed
    build/graphs/witnesses/<name>/<slug>.yaml   each witness's own graph
    build/agreement_summary.md     for reviewers: flagged claims, possible omissions
    build/agreement_report.json    the same, as data
    build/agreement_index.json     the sidecars re-keyed to the merged graph

Rerunning skips work already done, so an interrupted run resumes and a corpus
that gained three new PDFs costs three conversions rather than fifty. Pass
``--force`` to redo everything, or ``--stop-after ris`` to look before you
spend an evening of GPU time.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from . import annotate as annotate_module
from . import convert as convert_module
from . import ris as ris_module
from .cli import configure_logging, configure_stdio
from .config import DEFAULT_LLM_CONFIG
from .graph_io import atomic_write_json, load_graph, save_graph, validate_graph
from .ground import grounder_from_config
from .llm_client import LLMClient, LLMSettings
from .merge import collect_paths, merge_graphs
from .reconcile import DEFAULT_MIN_SCORE as DEFAULT_RECONCILE_SCORE
from .reconcile import NodeReconciler
from .schema import DEFAULT_SCHEMA, build_extraction_profile
from .confidence import print_buckets, write_corpus_report, write_merged_index
from .ensemble import print_summary as print_agreement
from .ensemble import write_corpus_report as write_agreement_report

LOGGER = logging.getLogger("ingest.run")

STAGES = ("ris", "convert", "annotate", "merge")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("corpus_dir", type=Path,
                        help="Folder of PDFs with an RIS export")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--into", type=Path,
                        help="Existing graph to merge the corpus into (read-only)")
    parser.add_argument(
        "--in-place", action="store_true",
        help="Also write the merged graph back over --into, keeping a .bak",
    )

    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG)
    parser.add_argument("--example", type=Path,
                        help="Gold-standard annotation to use as a one-shot")
    parser.add_argument("--example-max-nodes", type=int)
    parser.add_argument("--example-max-edges", type=int)
    parser.add_argument("--domain",
                        help='Field of study, e.g. "restoration ecology"')
    parser.add_argument("--ris", type=Path, nargs="*",
                        help="Explicit RIS file(s) instead of searching the folder")

    parser.add_argument("--converter", choices=convert_module.CONVERTERS,
                        default="marker")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable Marker's LLM assist during conversion")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--min-words", type=int, default=200)

    grounding = parser.add_argument_group("ontology grounding")
    grounding.add_argument("--no-ground", action="store_true",
                           help="Skip ontology lookup; terms stay as the model wrote them")
    grounding.add_argument("--ground-backend",
                           help="One backend for every entity type, instead of the "
                                "routes in config/pipeline.yaml")
    grounding.add_argument("--ground-min-score", type=float)
    grounding.add_argument("--refresh-ontologies", action="store_true",
                           help="Re-download locally loaded ontologies (ELMO) and "
                                "rebuild their term indexes")
    grounding.add_argument(
        "--no-reconcile", action="store_true",
        help="Do not match extracted nodes against --into; every node is new",
    )
    grounding.add_argument("--reconcile-min-score", type=float,
                           default=DEFAULT_RECONCILE_SCORE,
                           help="How alike two measured attributes must be to "
                                "count as the same node (default: %(default)s)")

    parser.add_argument("--limit", type=int,
                        help="Process only the first N documents (a useful dry run)")
    parser.add_argument("--force", action="store_true",
                        help="Redo conversion and annotation even where output exists")
    parser.add_argument("--stop-after", choices=STAGES,
                        help="Run up to this stage and stop")
    parser.add_argument("--no-merge-nodes", action="store_true")
    parser.add_argument("--graph-id")

    annotate_module.add_model_arguments(parser)
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)

    configure_stdio()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_level, out_dir / "run.log")

    if not args.corpus_dir.is_dir():
        parser.error(f"Not a directory: {args.corpus_dir}")

    manifest_path = out_dir / "manifest.json"
    markdown_dir = out_dir / "markdown"
    graphs_dir = out_dir / "graphs"

    # -- 1. manifest -------------------------------------------------------
    _banner("1/4  reading the corpus")
    manifest = ris_module.build_manifest(
        args.corpus_dir, ris_paths=list(args.ris) if args.ris else None
    )
    # Carry forward markdown/graph paths from a previous run, so resuming does
    # not lose the record of what has already been done.
    if manifest_path.exists():
        _carry_forward(manifest, json.loads(manifest_path.read_text(encoding="utf-8")))
    atomic_write_json(manifest_path, manifest)

    counts = manifest["counts"]
    print(f"  {counts['documents']} document(s): {counts['with_pdf']} with a PDF, "
          f"{counts['missing_pdfs']} without, "
          f"{counts['unmatched_pdfs']} PDF(s) with no RIS record")
    for warning in manifest["warnings"][:5]:
        print(f"    - {warning}")
    if len(manifest["warnings"]) > 5:
        print(f"    ... and {len(manifest['warnings']) - 5} more (see manifest.json)")
    if args.stop_after == "ris":
        print(f"\nStopped after 'ris'. Manifest: {manifest_path}")
        return 0

    documents = [row for row in manifest["documents"] if row.get("pdf_path")]
    if args.limit:
        documents = documents[: args.limit]
    if not documents:
        print("\nNo PDFs to process.")
        return 1

    # -- 2. convert --------------------------------------------------------
    _banner(f"2/4  converting {len(documents)} PDF(s) with {args.converter}")
    llm_for_marker = None
    if args.converter == "marker" and not args.no_llm:
        llm_for_marker = LLMSettings.from_config(args.llm_config).resolved_marker_llm()
    try:
        converter = convert_module.build_converter(
            args.converter, llm_for_marker, args.force_ocr
        )
    except ImportError as error:
        print(f"\n{args.converter} is not installed ({error}).", file=sys.stderr)
        print("Install it, or rerun with --converter pymupdf.", file=sys.stderr)
        return 2

    conversion = convert_module.convert_documents(
        documents, markdown_dir, converter, args.converter,
        force=args.force, min_words=args.min_words,
    )
    atomic_write_json(out_dir / "conversion_report.json",
                      {"converter": args.converter, "total": len(conversion),
                       "by_status": convert_module.tally(conversion),
                       "results": conversion})
    atomic_write_json(manifest_path, manifest)
    _print_tally(convert_module.tally(conversion))
    if args.stop_after == "convert":
        print(f"\nStopped after 'convert'. Markdown in {markdown_dir}")
        return 0

    # -- 3. annotate -------------------------------------------------------
    ready = [row for row in documents if row.get("markdown_path")]
    _banner(f"3/4  annotating {len(ready)} article(s)")
    profile = build_extraction_profile(args.schema)
    grounder = (
        None
        if args.no_ground
        else grounder_from_config(
            args.llm_config,
            out_dir / "grounding_cache.json",
            args.ground_backend,
            args.ground_min_score,
            # The schema's prefix map is what lets a locally loaded ontology
            # mint the CURIEs the schema's own enums use.
            profile=profile,
            refresh_ontologies=args.refresh_ontologies,
        )
    )
    examples = (
        annotate_module.load_example(
            args.example, args.example_max_nodes, args.example_max_edges, grounder
        )
        if args.example
        else None
    )
    if examples is None:
        LOGGER.warning(
            "No --example given: extraction quality is markedly better with a "
            "gold-standard annotation as a one-shot"
        )
    client = LLMClient(
        LLMSettings.from_config(args.llm_config),
        **annotate_module.client_overrides(args),
    )
    confidence = annotate_module.confidence_settings(args.llm_config, client)
    ensemble = annotate_module.ensemble_settings(args, args.llm_config, client)
    LOGGER.info("Inference: %s @ %s", client.settings.model, client.settings.endpoint)

    # Extraction resolves against the graph it is being added to, so a node
    # the corpus already has is reused rather than duplicated. Loaded here,
    # once, and never written.
    reconciler = None
    if args.into and args.into.exists() and not args.no_reconcile:
        reconciler = NodeReconciler(load_graph(args.into), args.reconcile_min_score)
        print(f"  reconciling against {len(reconciler.nodes)} existing node(s) "
              f"in {args.into.name}")

    annotation = annotate_module.annotate_documents(
        ready, profile, client, graphs_dir,
        markdown_dir=markdown_dir, examples=examples, domain=args.domain,
        max_chunk_characters=args.max_chunk_characters, overlap=args.overlap,
        force=args.force, schema_path=args.schema, grounder=grounder,
        reconciler=reconciler, confidence=confidence, ensemble=ensemble,
    )
    atomic_write_json(out_dir / "annotation_report.json",
                      {"total": len(annotation),
                       "by_status": annotate_module.tally(annotation),
                       "results": annotation})
    atomic_write_json(manifest_path, manifest)
    _print_tally(annotate_module.tally(annotation))
    if confidence:
        print_buckets(write_corpus_report(graphs_dir, out_dir, confidence))
    if ensemble:
        print_agreement(write_agreement_report(graphs_dir, out_dir, ensemble))
    if args.stop_after == "annotate":
        print(f"\nStopped after 'annotate'. Graphs in {graphs_dir}")
        return 0

    # -- 4. merge ----------------------------------------------------------
    paths = collect_paths([graphs_dir])
    _banner(f"4/4  merging {len(paths)} document graph(s)")
    if not paths:
        print("  nothing to merge: no document graph was produced")
        return 1

    graphs, sources, loaded = [], [], []
    for path in paths:
        try:
            graphs.append(load_graph(path))
            sources.append(str(path))
            loaded.append(path)
        except (OSError, ValueError) as error:
            LOGGER.error("Could not load %s: %s", path, error)

    existing = _load_existing(args.into)
    id_maps: dict = {}
    merged, report = merge_graphs(
        graphs, profile.version, existing, args.graph_id,
        merge_nodes=not args.no_merge_nodes, id_maps=id_maps,
    )
    save_graph(merged, out_dir / "causal_graph.yaml")
    save_graph(merged, out_dir / "causal_graph.json")
    atomic_write_json(out_dir / "merge_report.json",
                      {"sources": sources,
                       "into": str(args.into) if args.into else None, **report})
    for index in write_merged_index(loaded, id_maps, existing is not None, out_dir):
        print(f"  re-keyed to the merged graph: {index}")

    print(f"  nodes: {report['nodes_in']} in -> {report['nodes_out']} out "
          f"({report['nodes_merged']} merged)")
    print(f"  edges: {report['edges_in']} in -> {report['edges_out']} out "
          f"({report['edges_deduplicated']} deduplicated)")

    result = validate_graph(merged, args.schema)
    atomic_write_json(out_dir / "validation.json",
                      {"ok": result.ok, "problems": result.problems,
                       "nodes": result.node_count, "edges": result.edge_count})
    print(f"  validation: {result.summary()}")
    for problem in result.problems[:10]:
        print(f"      - {problem[:170]}")

    if args.in_place and args.into:
        # Only overwrite the corpus graph once the merge validated: a graph
        # that fails the schema is exactly what you do not want in its place.
        if result.ok:
            import shutil

            if args.into.exists():
                backup = args.into.with_suffix(args.into.suffix + ".bak")
                shutil.copy2(args.into, backup)
                print(f"  previous graph kept as {backup.name}")
            save_graph(merged, args.into)
            print(f"  wrote {args.into}")
        else:
            print(f"  NOT written back to {args.into}: the merged graph does not "
                  f"validate. The result is in {out_dir / 'causal_graph.yaml'}.")

    print(f"\nDone. Merged graph: {out_dir / 'causal_graph.yaml'}")
    return 0 if result.ok else 1


def _load_existing(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    if not path.exists():
        LOGGER.warning("--into %s does not exist yet; creating it", path)
        return None
    existing = load_graph(path)
    print(f"  merging into {path.name}: {len(existing.get('nodes') or [])} node(s), "
          f"{len(existing.get('edges') or [])} edge(s)")
    return existing


def _carry_forward(manifest: dict, previous: dict) -> None:
    """Copy markdown/graph paths from a previous manifest onto a fresh scan."""
    by_slug = {row.get("slug"): row for row in previous.get("documents") or []}
    for row in manifest.get("documents") or []:
        old = by_slug.get(row.get("slug"))
        if not old:
            continue
        for key in ("markdown_path", "word_count", "graph_path"):
            if old.get(key) and not row.get(key):
                row[key] = old[key]


def _banner(text: str) -> None:
    print(f"\n=== {text} " + "=" * max(0, 66 - len(text)))


def _print_tally(counts: dict[str, int]) -> None:
    for status, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"    {count:5d}  {status}")


if __name__ == "__main__":
    sys.exit(main())
