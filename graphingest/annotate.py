"""Annotate markdown articles into per-document causal graphs.

    python -m graphingest.annotate --manifest build/manifest.json \\
        --markdown-dir build/markdown --out-dir build/graphs \\
        --example examples/murphy_2005_camo_annotation_revised.yaml
    python -m graphingest.annotate --article paper.md --out one.yaml --title "..."
    python -m graphingest.annotate --article paper.md --out one.yaml --dry-run

The schema is a parameter, not a hardcoded assumption: point ``--schema`` at
any LinkML file with CausalNode/CausalEdge-shaped classes and the prompt, the
structured-output constraint, the normalizer and the validator all retarget
together. Where inference runs is decided entirely by ``config/pipeline.yaml``.

The gold standard passed as ``--example`` is included verbatim as a one-shot:
a hand-checked annotation of a *different* article, in the exact output shape.
It is the single most effective lever on output quality here, because it shows
the model the house conventions — node granularity, how much of a sentence to
quote, when to use which qualifier — that no amount of schema prose conveys.

Running over a manifest is resumable: a document whose graph already exists is
skipped unless ``--force``, so an interrupted overnight run picks up where it
stopped.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from .chunker import Chunker
from .confidence import (
    ConfidenceSettings,
    ConfidenceTracker,
    print_buckets,
    write_corpus_report,
    write_sidecar,
)
from .cli import configure_logging, configure_stdio
from .config import DEFAULT_LLM_CONFIG
from .consolidate import Consolidator
from .ensemble import (
    EnsembleSettings,
    compare_graphs,
    ensemble_enabled_in_config,
    ensemble_from_config,
    print_summary as print_agreement,
    remap_after_reconciliation,
    witness_graph_path,
    write_corpus_report as write_agreement_report,
)
from .ensemble import finalize as finalize_agreement
from .ensemble import sidecar_path as agreement_sidecar_path
from .graph_io import atomic_write_json, load_graph, save_graph, validate_graph
from .llm_client import LLMClient, LLMError, LLMSettings
from .normalize import NormalizationReport, normalize_graph
from .reconcile import (
    DEFAULT_MIN_SCORE as DEFAULT_RECONCILE_SCORE,
    NodeReconciler,
    ReconciliationReport,
)
from .ground import (
    Grounder,
    GroundingReport,
    grounder_from_config,
    plainify,
    snapshot_id,
)
from .schema import DEFAULT_SCHEMA, ExtractionProfile, build_extraction_profile
from .schema_prompt import (
    build_extraction_json_schema,
    build_extraction_prompt,
    build_system_prompt,
)
from .sqlite_manifest import SQLiteManifestError, load_sqlite_manifest

LOGGER = logging.getLogger("ingest.annotate")


# ---------------------------------------------------------------------------
# Source document metadata
# ---------------------------------------------------------------------------


def source_document_from_row(row: dict, profile: ExtractionProfile) -> dict:
    """Build a SourceDocument from a manifest row, keeping only real slots.

    A manifest row carries more than the schema models (abstract, keywords,
    volume). Passing those straight through would fail validation, so the row
    is filtered against the schema's own SourceDocument definition rather than
    against a hand-maintained list that would drift.
    """
    try:
        permitted = {slot.name for slot in profile.get("SourceDocument").slots}
    except KeyError:  # a schema without SourceDocument: send the basics
        permitted = {"document_id", "doi", "title", "authors", "year", "journal"}

    candidate = {
        "document_id": row.get("document_id") or row.get("doi") or row.get("slug"),
        "doi": row.get("doi"),
        "title": row.get("title"),
        "authors": row.get("authors") or None,
        "year": row.get("year"),
        "journal": row.get("journal"),
    }
    return {
        key: value
        for key, value in candidate.items()
        if key in permitted and value not in (None, "", [])
    }


# ---------------------------------------------------------------------------
# The one-shot example
# ---------------------------------------------------------------------------


def load_example(
    path: Path,
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = None,
    grounder: Optional[Grounder] = None,
) -> str:
    """Read the gold-standard example, trimmed and de-grounded for the prompt.

    A full hand annotation can run to 70KB, which on a short article is more
    example than article. Trimming keeps the first ``max_nodes`` nodes and only
    the edges whose endpoints survive, so the fragment is still a *valid* graph
    rather than one with dangling references — a broken example teaches broken
    output.

    De-grounding matters more. A hand annotation has already been through
    ontology lookup, so it contains identifiers (``entity_term: Q30019``), and
    a one-shot is imitated rather than read: leave those in and the model
    answers in QIDs it invented, which is the one thing this pipeline asks it
    not to do. With a ``grounder`` those identifiers are turned back into the
    labels they denote.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        if max_nodes is not None or max_edges is not None:
            LOGGER.warning(
                "Cannot trim a non-graph example (%s); using it whole", path.name
            )
        return text
    if max_nodes is None and max_edges is None and grounder is None:
        return text

    graph = load_graph(path)
    if grounder is not None:
        graph = plainify(graph, grounder)
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if max_nodes is not None:
        nodes = nodes[:max_nodes]
    kept_ids = {node.get("id") for node in nodes}
    edges = [
        edge
        for edge in edges
        if edge.get("subject") in kept_ids and edge.get("object") in kept_ids
    ]
    if max_edges is not None:
        edges = edges[:max_edges]
    trimmed = {**graph, "nodes": nodes, "edges": edges}
    if max_nodes is not None or max_edges is not None:
        LOGGER.info(
            "Trimmed example %s to %d node(s) and %d edge(s)",
            path.name, len(nodes), len(edges),
        )
    return yaml.safe_dump(trimmed, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_from_markdown(
    text: str,
    profile: ExtractionProfile,
    client: LLMClient,
    source_document: Optional[dict] = None,
    max_chunk_characters: Optional[int] = None,
    overlap: int = 400,
    examples: Optional[str] = None,
    domain: Optional[str] = None,
    annotator: Optional[str] = None,
    confidence: Optional[ConfidenceTracker] = None,
) -> tuple[dict, dict]:
    """Extract a causal graph from article markdown.

    With ``max_chunk_characters`` unset the whole article goes in one request,
    which is what a long-context model should do: chunking splits claims whose
    cause and effect sit in different sections. Chunking exists for models that
    cannot hold a full paper.

    Returns ``(graph, run_report)``. With a ``confidence`` tracker, each
    reply's token logprobs are scored per item and followed through
    normalization and consolidation to the ids the returned graph carries.
    """
    started = time.time()
    system_prompt = build_system_prompt(domain)
    # Compact: the unpruned schema is ~65KB and servers choke on it.
    json_schema = build_extraction_json_schema(profile, compact=True)
    chunker = Chunker(
        max_chunk_size=max_chunk_characters,
        overlap_size=overlap,
        max_characters=max_chunk_characters,
    )
    chunks = chunker.chunk_text(text)
    LOGGER.info(
        "Prepared %d chunk(s) from %d characters (limit=%s)",
        len(chunks),
        len(text),
        max_chunk_characters or "none - whole document",
    )

    chunk_graphs: list[dict] = []
    normalization = NormalizationReport()
    chunk_details: list[dict] = []

    for number, chunk in enumerate(chunks, 1):
        LOGGER.info(
            "Chunk %d/%d (section=%s, %d chars): requesting extraction",
            number, len(chunks), chunk.section, len(chunk.text),
        )
        prompt = build_extraction_prompt(
            profile, chunk.text, source_document, examples=examples
        )
        raw = client.complete_json(
            system_prompt,
            prompt,
            json_schema,
            schema_name="causal_extraction",
            # The prompt already documents every class, slot and enum in prose.
            schema_in_prompt=False,
        )
        # Scored before anything touches ``raw``: the scorer re-parses the
        # reply text and checks it arrives at exactly this object.
        scored = (
            confidence.score_chunk(
                number, client.last_trace, raw, chunk.text,
                client.trace_unavailable or "the endpoint returned no logprobs",
            )
            if confidence else {}
        )
        _attach_spans(raw, chunk.text, chunk.start_char, chunk.section)
        graph, report = normalize_graph(raw, profile, source_document)
        _merge_reports(normalization, report)
        if confidence:
            confidence.bind_chunk(number, graph, scored, raw, chunk.start_char)
        chunk_graphs.append(graph)
        chunk_details.append(
            {
                "chunk": number,
                "section": chunk.section,
                "characters": len(chunk.text),
                "nodes": len(graph.get("nodes") or []),
                "edges": len(graph.get("edges") or []),
            }
        )
        LOGGER.info(
            "Chunk %d/%d: %d node(s), %d edge(s)",
            number, len(chunks), chunk_details[-1]["nodes"], chunk_details[-1]["edges"],
        )

    consolidator = Consolidator(schema_version=profile.version)
    graph, consolidation = consolidator.consolidate(chunk_graphs)
    if confidence:
        confidence.after_consolidation(
            consolidation.node_id_maps, consolidation.edge_id_maps
        )
    # Who annotated this. The model cannot be trusted to report its own name,
    # so it is stamped here rather than asked for.
    stamp = annotator or annotator_stamp(profile, client.settings.model)
    if stamp:
        for item in [*(graph.get("nodes") or []), *(graph.get("edges") or [])]:
            item.setdefault("annotator", stamp)
    graph.setdefault("provenance", {}).update(provenance_block(profile))
    if source_document:
        graph.setdefault("source_documents", [source_document])

    run_report = {
        "elapsed_seconds": round(time.time() - started, 2),
        "chunks": chunk_details,
        "normalization": normalization.to_dict(),
        "consolidation": consolidation.to_dict(),
        "llm": {
            "provider": client.settings.provider,
            "endpoint": client.settings.endpoint,
            "model": client.settings.model,
            "structured_output": client.settings.structured_output,
        },
        "schema": {"path": str(profile.schema_path), "version": profile.version},
    }
    return graph, run_report


def annotator_stamp(profile: ExtractionProfile, model: str) -> Optional[str]:
    """An ``annotator`` value naming the model, in a form the schema accepts.

    CAMO constrains the slot to an ORCID or a ``camo_agent:`` identifier, and
    "qwen3.6:35b" satisfies neither — the colon alone breaks the pattern. So
    the candidates are tried against the schema's own regex and the first that
    passes is used; if none does, nothing is stamped. Producing a graph that
    fails validation in order to record who made it is a bad trade.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model or "unknown").strip("_") or "unknown"
    try:
        slot = next(
            slot
            for slot in profile.get("CausalNode").slots
            if slot.name == "annotator"
        )
    except (KeyError, StopIteration):
        return None
    if not slot.pattern:
        return f"model:{safe}"
    for candidate in (f"camo_agent:{safe}", f"model:{safe}", safe):
        if re.fullmatch(slot.pattern, candidate):
            return candidate
    LOGGER.debug(
        "No annotator value for %r satisfies the schema pattern; leaving it unset",
        model,
    )
    return None


def provenance_block(profile: ExtractionProfile, exporter: str = "graphingest.annotate") -> dict:
    """Graph provenance, naming the schema actually used.

    Which ontologies the terms were resolved against is not asserted here: it
    is recorded in ``ontology_snapshot_id`` by the grounding step, which is the
    only part of the pipeline that knows.
    """
    from . import __version__

    return {
        "ontology_framework": profile.schema_name or profile.schema_path.stem,
        "causal_mosaic_version": profile.version,
        "created": datetime.now(timezone.utc).isoformat(),
        "exporter_version": f"{exporter}/{__version__}",
    }


def _attach_spans(raw: dict, chunk_text: str, offset: int, section: str) -> None:
    """Turn model-quoted sentences into TextSpans with real document offsets.

    A quote the model invented will not be found in the chunk, so it gets no
    offsets -- which is itself a useful signal when reviewing an extraction.
    """
    for item in [*(raw.get("nodes") or []), *(raw.get("edges") or [])]:
        if not isinstance(item, dict):
            continue
        spans = item.get("source_spans") or []
        if not spans and item.get("original_sentence"):
            spans = [{"text": item["original_sentence"]}]
            item["source_spans"] = spans
        for span in spans:
            if not isinstance(span, dict) or not span.get("text"):
                continue
            span.setdefault("section", section if section != "unknown" else None)
            if span.get("section") is None:
                span.pop("section")
            position = chunk_text.find(span["text"])
            if position >= 0:
                span.setdefault("start_char", offset + position)
                span.setdefault("end_char", offset + position + len(span["text"]))


def _merge_reports(target: NormalizationReport, source: NormalizationReport) -> None:
    target.coerced.update(source.coerced)
    target.defaults_applied.update(source.defaults_applied)
    target.dropped.extend(source.dropped)
    target.generated_ids += source.generated_ids


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def annotate_documents(
    documents: list[dict],
    profile: ExtractionProfile,
    client: LLMClient,
    out_dir: Path,
    markdown_dir: Optional[Path] = None,
    examples: Optional[str] = None,
    domain: Optional[str] = None,
    max_chunk_characters: Optional[int] = None,
    overlap: int = 400,
    force: bool = False,
    schema_path: Optional[Path] = None,
    grounder: Optional[Grounder] = None,
    reconciler: Optional[NodeReconciler] = None,
    confidence: Optional[ConfidenceSettings] = None,
    ensemble: Optional[EnsembleSettings] = None,
) -> list[dict]:
    """Annotate every document that has markdown, writing one graph each.

    Two resolution steps run here, per document, rather than on the merged
    graph. Grounding is first: the lookups are cached and shared across the
    corpus either way, but grounding first means the second step and the merge
    can both recognise that "Aedes dorsalis" in one paper and "Ae. dorsalis" in
    another are the same thing.

    Then reconciliation against the graph being added to, so a node that
    already exists is *that* node — extraction attaches new evidence to the
    corpus rather than beside it.

    With ``confidence`` settings, every graph gets a ``<slug>.confidence.json``
    beside it: per node and edge, how sure the model's tokens were, keyed by
    the ids in the saved graph.

    With ``ensemble`` settings, each witness model extracts the same article
    and ``<slug>.agreement.json`` records where it agrees with the primary,
    where it contradicts it, and what it found that the primary did not. The
    saved graph is the primary's either way. A document annotated on an
    earlier run is checked too, so witnesses can be added to a finished
    corpus without re-extracting it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    extraction = {"examples": examples, "domain": domain, "overlap": overlap,
                  "max_chunk_characters": max_chunk_characters}
    results: list[dict] = []

    for index, row in enumerate(documents, 1):
        slug = row["slug"]
        destination = out_dir / f"{slug}.yaml"
        record = {
            "slug": slug,
            "document_id": row.get("document_id"),
            "graph_path": str(destination),
        }

        markdown_path = _markdown_path(row, markdown_dir)
        if markdown_path is None:
            record.update({"status": "skipped_no_markdown"})
            results.append(record)
            LOGGER.warning("[%d/%d] %s: no markdown", index, len(documents), slug)
            continue

        if destination.exists() and not force:
            try:
                existing = load_graph(destination)
                record.update(
                    {
                        "status": "skipped_existing",
                        "nodes": len(existing.get("nodes") or []),
                        "edges": len(existing.get("edges") or []),
                    }
                )
                if ensemble and not agreement_sidecar_path(destination).exists():
                    LOGGER.info("[%d/%d] %s: already annotated; checking it against "
                                "the witnesses", index, len(documents), slug)
                    summary = check_agreement(
                        ensemble, slug,
                        markdown_path.read_text(encoding="utf-8", errors="replace"),
                        source_document_from_row(row, profile), existing, existing, [],
                        profile, out_dir, destination, grounder, extraction, force,
                        {"slug": slug, "document_id": row.get("document_id")},
                    )
                    _add_to_report(out_dir / f"{slug}.report.json", "agreement", summary)
                    record["agreement_flagged_edges"] = summary["flagged_edges"]
                results.append(record)
                LOGGER.info("[%d/%d] %s: already annotated", index, len(documents), slug)
                continue
            except (OSError, ValueError) as error:
                LOGGER.warning("Re-annotating %s: existing graph unreadable (%s)",
                               slug, error)

        text = markdown_path.read_text(encoding="utf-8", errors="replace")
        source_document = source_document_from_row(row, profile)
        LOGGER.info(
            "[%d/%d] %s: annotating %d characters",
            index, len(documents), slug, len(text),
        )
        tracker = ConfidenceTracker(confidence, profile) if confidence else None
        try:
            graph, run_report = extract_from_markdown(
                text, profile, client, source_document,
                max_chunk_characters=max_chunk_characters,
                overlap=overlap, examples=examples, domain=domain,
                confidence=tracker,
            )
        except LLMError as error:
            # One unanswerable article must not end a corpus run.
            record.update({"status": "failed", "error": str(error)})
            results.append(record)
            LOGGER.error("[%d/%d] %s: %s", index, len(documents), slug, error)
            continue

        grounding = GroundingReport()
        if grounder is not None:
            grounding = grounder.ground_graph(graph, profile)
            if grounding.grounded:
                graph.setdefault("provenance", {})["ontology_snapshot_id"] = (
                    snapshot_id(grounding, grounder.routes)
                )
            LOGGER.info(
                "[%d/%d] %s: grounded %d term(s), %d left as free text",
                index, len(documents), slug, grounding.grounded, grounding.unresolved,
            )

        # Witnesses are compared with the graph as the primary extracted it:
        # reconciliation is about to swap node wording for the corpus's,
        # which the witnesses never saw.
        as_extracted = copy.deepcopy(graph) if ensemble else None

        reconciliation = ReconciliationReport()
        if reconciler:
            reconciliation = reconciler.reconcile(graph)
            if reconciliation.matched:
                LOGGER.info(
                    "[%d/%d] %s: %d node(s) matched the existing graph, %d new",
                    index, len(documents), slug,
                    reconciliation.matched, reconciliation.unmatched,
                )

        if tracker:
            tracker.after_reconciliation(reconciliation.matches)

        validation = validate_graph(graph, schema_path or profile.schema_path)
        save_graph(graph, destination)
        extra: dict = {}
        if tracker:
            sidecar = tracker.finalize(
                graph, {"slug": slug, "document_id": row.get("document_id")}
            )
            write_sidecar(destination, sidecar)
            extra["confidence"] = {
                key: value for key, value in sidecar["summary"].items()
                if key != "review_first"
            }
            buckets = sidecar["summary"]["buckets"]["edges"]
            record["confidence_edges"] = {
                bucket: buckets[bucket] for bucket in ("high", "medium", "low")
            }
            LOGGER.info(
                "[%d/%d] %s: edge confidence high %d, medium %d, low %d",
                index, len(documents), slug,
                buckets["high"], buckets["medium"], buckets["low"],
            )
        if ensemble:
            # After the primary graph is safely on disk: witnesses are the
            # slow, optional part, and a failure there must not cost it.
            summary = check_agreement(
                ensemble, slug, text, source_document, as_extracted, graph,
                reconciliation.matches, profile, out_dir, destination, grounder,
                extraction, force,
                {"slug": slug, "document_id": row.get("document_id")},
            )
            extra["agreement"] = summary
            record["agreement_flagged_edges"] = summary["flagged_edges"]
        atomic_write_json(out_dir / f"{slug}.report.json",
                          {**run_report,
                           "grounding": grounding.to_dict(),
                           "reconciliation": reconciliation.to_dict(),
                           "validation": {
                              "ok": validation.ok, "problems": validation.problems},
                           **extra})
        record.update(
            {
                "status": "annotated" if validation.ok else "annotated_invalid",
                "nodes": validation.node_count,
                "edges": validation.edge_count,
                "seconds": run_report["elapsed_seconds"],
                "grounded": grounding.grounded,
                "ungrounded": grounding.unresolved,
                "reconciled": reconciliation.matched,
                "validation_problems": len(validation.problems),
            }
        )
        row["graph_path"] = str(destination)
        results.append(record)
        LOGGER.info(
            "[%d/%d] %s: %d node(s), %d edge(s), %s in %.1fs",
            index, len(documents), slug, validation.node_count, validation.edge_count,
            "valid" if validation.ok else f"{len(validation.problems)} problem(s)",
            run_report["elapsed_seconds"],
        )

    return results


def check_agreement(
    ensemble: EnsembleSettings,
    slug: str,
    text: str,
    source_document: dict,
    as_extracted: dict,
    graph: dict,
    reconciled: list[dict],
    profile: ExtractionProfile,
    out_dir: Path,
    destination: Path,
    grounder: Optional[Grounder],
    extraction: dict,
    force: bool,
    document: dict,
) -> dict:
    """Extract with every witness, compare with the primary, write the sidecar.

    Witness graphs are kept under ``<out-dir>/witnesses/<name>/`` and reused on
    a re-run, so adding a fourth witness costs one model's time, not four.
    They are grounded with the same grounder as the primary, so both sides
    are compared in the same CURIEs. They are not reconciled against
    ``--into``: nothing of theirs is saved into the corpus.
    """
    runs: list[dict] = []
    graphs: dict[str, dict] = {}
    for witness in ensemble.witnesses:
        path = witness_graph_path(out_dir, witness, slug)
        run = witness.describe()
        if path.exists() and not force:
            try:
                graphs[witness.name] = load_graph(path)
                runs.append({**run, "status": "reused"})
                continue
            except (OSError, ValueError) as error:
                LOGGER.warning("Re-extracting %s with %s: saved graph unreadable (%s)",
                               slug, witness.name, error)
        LOGGER.info("%s: extracting with witness %s (%s)",
                    slug, witness.name, witness.settings.model)
        try:
            witness_graph, report = extract_from_markdown(
                text, profile, witness.client(), source_document,
                max_chunk_characters=(witness.max_chunk_characters
                                      or extraction["max_chunk_characters"]),
                overlap=extraction["overlap"], examples=extraction["examples"],
                domain=extraction["domain"],
            )
        except LLMError as error:
            # A witness that cannot answer leaves the others to vote.
            runs.append({**run, "status": "failed", "error": str(error)})
            LOGGER.warning("%s: witness %s failed: %s", slug, witness.name, error)
            continue
        if grounder is not None:
            grounder.ground_graph(witness_graph, profile)
        save_graph(witness_graph, path)
        graphs[witness.name] = witness_graph
        runs.append({**run, "status": "extracted",
                     "nodes": len(witness_graph.get("nodes") or []),
                     "edges": len(witness_graph.get("edges") or []),
                     "seconds": report["elapsed_seconds"]})

    result = compare_graphs(as_extracted, graphs, ensemble)
    remap_after_reconciliation(result, reconciled)
    sidecar = finalize_agreement(result, graph, runs, ensemble, document)
    atomic_write_json(agreement_sidecar_path(destination), sidecar)
    summary = sidecar["summary"]
    LOGGER.info(
        "%s: %d of %d edge(s) flagged by %d witness(es), %d possible omission(s)",
        slug, summary["flagged_edges"], len(sidecar["edges"]),
        len(summary["checked_by"]), summary["possible_omissions"],
    )
    return summary


def _add_to_report(path: Path, key: str, value: dict) -> None:
    try:
        report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        report = {}
    report[key] = value
    atomic_write_json(path, report)


def _markdown_path(row: dict, markdown_dir: Optional[Path]) -> Optional[Path]:
    """Where this document's markdown lives: the manifest first, then by slug."""
    recorded = row.get("markdown_path")
    if recorded and Path(recorded).exists():
        return Path(recorded)
    if markdown_dir:
        candidate = markdown_dir / f"{row['slug']}.md"
        if candidate.exists():
            return candidate
    return None


def tally(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in results:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags that override config/pipeline.yaml for one run."""
    tuning = parser.add_argument_group("extraction tuning")
    tuning.add_argument(
        "--max-chunk-characters",
        type=int,
        help="Split articles into chunks of at most this size. Omit to send the "
        "whole document in one request (preferred for long-context models).",
    )
    tuning.add_argument("--overlap", type=int, default=400)
    tuning.add_argument("--model", help="Override the configured model")
    tuning.add_argument("--endpoint", help="Override the configured endpoint")
    tuning.add_argument(
        "--structured-output",
        choices=("json_object", "json_schema", "tool_use", "prompt_only"),
    )
    tuning.add_argument("--max-tokens", type=int, help="Override the generation budget")

    confidence = parser.add_argument_group(
        "confidence (token logprobs; large output)"
    )
    confidence.add_argument(
        "--confidence", action="store_true",
        help="Request token logprobs and write <slug>.confidence.json per graph, "
             "plus confidence_summary.md for reviewers. OpenAI-compatible "
             "endpoints only (vLLM, OpenAI, recent Ollama)",
    )
    confidence.add_argument(
        "--top-logprobs", type=int,
        help="Alternatives recorded per generated token (max 20; default from "
             "config/pipeline.yaml)",
    )
    confidence.add_argument(
        "--prompt-logprobs", type=int, nargs="?", const=0, metavar="K",
        help="vLLM only: also score the article itself (perplexity, the most "
             "surprising passages). Implies --confidence. Costly on long prompts",
    )

    ensemble = parser.add_argument_group(
        "ensemble (cross-model agreement; one extra extraction per witness)"
    )
    ensemble.add_argument(
        "--ensemble", action="store_true",
        help="Also extract each article with the witness models in "
             "ensemble.witnesses, and flag primary nodes and edges they disagree "
             "with. The saved graph is always the primary model's",
    )
    ensemble.add_argument(
        "--witness", action="append", metavar="NAME_OR_MODEL",
        help="Use this witness (repeatable; implies --ensemble). A configured "
             "witness name, or a model id served by the primary endpoint",
    )
    ensemble.add_argument(
        "--ensemble-min-score", type=float,
        help="How alike two models' nodes must be to count as the same "
             "variable (default: ensemble.match_min_score)",
    )


def client_overrides(args: argparse.Namespace) -> dict:
    """LLMClient overrides from the command line; ``None`` leaves config alone."""
    wants_confidence = args.confidence or args.prompt_logprobs is not None
    return {
        "model": args.model,
        "endpoint": args.endpoint,
        "structured_output": args.structured_output,
        "max_tokens": args.max_tokens,
        "logprobs": True if wants_confidence else None,
        "top_logprobs": args.top_logprobs,
        "prompt_logprobs": args.prompt_logprobs,
    }


def ensemble_settings(
    args: argparse.Namespace, llm_config: Path, client: LLMClient
) -> Optional[EnsembleSettings]:
    """Witnesses for this run, or ``None`` when the ensemble is off."""
    if not (args.ensemble or args.witness or ensemble_enabled_in_config(llm_config)):
        return None
    try:
        ensemble = ensemble_from_config(
            llm_config, client.settings, args.witness, args.ensemble_min_score
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    LOGGER.info("Ensemble: primary %s, witnesses %s", client.settings.model,
                ", ".join(f"{w.name} ({w.settings.model})" for w in ensemble.witnesses))
    return ensemble


def confidence_settings(
    llm_config: Path, client: LLMClient
) -> Optional[ConfidenceSettings]:
    """Confidence settings for this run, or ``None`` when it is off."""
    if not client.settings.logprobs:
        return None
    settings = ConfidenceSettings.from_config(llm_config)
    settings.enabled = True
    settings.top_logprobs = client.settings.top_logprobs
    settings.prompt_logprobs = client.settings.prompt_logprobs
    if client.settings.provider == "anthropic":
        LOGGER.warning(
            "--confidence: the Anthropic API does not return logprobs; every "
            "document will be reported as unscored"
        )
    return settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest", type=Path,
        help="Corpus manifest: graphingest JSON or a read-only SQLite Markdown manifest",
    )
    source.add_argument("--article", type=Path, help="A single markdown article")

    parser.add_argument("--out-dir", type=Path, help="Where per-document graphs go")
    parser.add_argument("--out", type=Path, help="Output path for a single --article")
    parser.add_argument("--markdown-dir", type=Path, help="Where the markdown lives")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG)
    parser.add_argument(
        "--example", type=Path, help="Gold-standard annotation to use as a one-shot"
    )
    parser.add_argument("--example-max-nodes", type=int,
                        help="Trim the example to this many nodes")
    parser.add_argument("--example-max-edges", type=int,
                        help="Trim the example to this many edges")
    parser.add_argument(
        "--domain",
        help='Field of study, e.g. "restoration ecology"; sharpens the system prompt',
    )
    grounding = parser.add_argument_group("ontology grounding")
    grounding.add_argument(
        "--no-ground", action="store_true",
        help="Skip ontology lookup; terms stay as the model wrote them",
    )
    grounding.add_argument(
        "--ground-backend",
        help="Use one backend for every entity type (ols:envo,go | wikidata | "
             "oaklib | none) instead of the routes in config/pipeline.yaml",
    )
    grounding.add_argument("--ground-min-score", type=float,
                           help="Reject a match below this label similarity")
    grounding.add_argument("--ground-cache", type=Path,
                           help="Lookup cache JSON (default: <out-dir>/grounding_cache.json)")
    grounding.add_argument("--refresh-ontologies", action="store_true",
                           help="Re-download locally loaded ontologies (ELMO) and "
                                "rebuild their term indexes")
    grounding.add_argument(
        "--against", type=Path,
        help="Existing graph to reconcile extracted nodes against, so a node "
             "that already exists is reused rather than duplicated (read-only)",
    )
    grounding.add_argument("--reconcile-min-score", type=float,
                           default=DEFAULT_RECONCILE_SCORE,
                           help="How alike two measured attributes must be to "
                                "count as the same node (default: %(default)s)")

    parser.add_argument("--limit", type=int, help="Annotate only the first N documents")
    parser.add_argument("--force", action="store_true", help="Re-annotate existing graphs")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the rendered prompt for the first document and exit",
    )

    metadata = parser.add_argument_group("single-article metadata")
    metadata.add_argument("--doi")
    metadata.add_argument("--document-id")
    metadata.add_argument("--title")
    metadata.add_argument("--journal")
    metadata.add_argument("--year", type=int)

    add_model_arguments(parser)
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)

    configure_stdio()
    configure_logging(args.log_level)

    profile = build_extraction_profile(args.schema)
    out_dir = args.out_dir or (args.out.parent if args.out else Path("."))
    grounder = (
        None
        if args.no_ground
        else grounder_from_config(
            args.llm_config,
            args.ground_cache or (out_dir / "grounding_cache.json"),
            args.ground_backend,
            args.ground_min_score,
            # The schema's prefix map is what lets a locally loaded ontology
            # mint the CURIEs the schema's own enums use.
            profile=profile,
            refresh_ontologies=args.refresh_ontologies,
        )
    )
    examples = (
        load_example(
            args.example, args.example_max_nodes, args.example_max_edges, grounder
        )
        if args.example
        else None
    )

    if args.article:
        rows = [
            {
                "slug": args.article.stem,
                "document_id": args.doi or args.document_id or args.article.stem,
                "doi": args.doi,
                "title": args.title,
                "journal": args.journal,
                "year": args.year,
                "authors": [],
                "markdown_path": str(args.article),
            }
        ]
        manifest, manifest_path = None, None
    else:
        manifest_path = args.manifest
        if manifest_path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
            try:
                rows = load_sqlite_manifest(manifest_path)
            except (FileNotFoundError, SQLiteManifestError) as error:
                parser.error(str(error))
            # An external SQLite manifest is never rewritten with pipeline state.
            manifest, manifest_path = None, None
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows = manifest.get("documents") or []

    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("Nothing to annotate.")
        return 0

    if args.dry_run:
        row = rows[0]
        markdown_path = _markdown_path(row, args.markdown_dir)
        if markdown_path is None:
            parser.error(f"No markdown for {row['slug']}; run graphingest.convert first")
        text = markdown_path.read_text(encoding="utf-8", errors="replace")
        system_prompt = build_system_prompt(args.domain)
        prompt = build_extraction_prompt(
            profile, text, source_document_from_row(row, profile), examples=examples
        )
        print(f"--- SYSTEM ({len(system_prompt)} chars) ---\n{system_prompt}\n")
        print(f"--- USER ({len(prompt)} chars, ~{len(prompt) // 4} tokens) ---")
        print(prompt)
        return 0

    if args.article and not args.out:
        parser.error("--article requires --out")
    if args.manifest and not args.out_dir:
        parser.error("--manifest requires --out-dir")

    settings = LLMSettings.from_config(args.llm_config)
    client = LLMClient(settings, **client_overrides(args))
    confidence = confidence_settings(args.llm_config, client)
    ensemble = ensemble_settings(args, args.llm_config, client)
    LOGGER.info(
        "Annotating %d document(s) with %s @ %s (schema %s v%s)",
        len(rows), client.settings.model, client.settings.endpoint,
        profile.schema_path.name, profile.version,
    )

    reconciler = None
    if args.against:
        if not args.against.exists():
            parser.error(f"No existing graph at {args.against}")
        reconciler = NodeReconciler(load_graph(args.against), args.reconcile_min_score)
        LOGGER.info("Reconciling against %d existing node(s) in %s",
                    len(reconciler.nodes), args.against.name)

    results = annotate_documents(
        rows, profile, client, out_dir,
        markdown_dir=args.markdown_dir, examples=examples, domain=args.domain,
        max_chunk_characters=args.max_chunk_characters, overlap=args.overlap,
        force=args.force, schema_path=args.schema, grounder=grounder,
        reconciler=reconciler, confidence=confidence, ensemble=ensemble,
    )

    if args.article and args.out:
        produced = out_dir / f"{rows[0]['slug']}.yaml"
        if produced.exists() and produced.resolve() != args.out.resolve():
            save_graph(load_graph(produced), args.out)

    counts = tally(results)
    atomic_write_json(out_dir / "annotation_report.json",
                      {"total": len(results), "by_status": counts, "results": results})
    if manifest is not None and manifest_path is not None:
        atomic_write_json(manifest_path, manifest)

    print(f"\nAnnotated {len(results)} document(s):")
    for status, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"    {count:5d}  {status}")
    print(f"  wrote {out_dir / 'annotation_report.json'}")
    if confidence:
        print_buckets(write_corpus_report(out_dir, out_dir, confidence))
    if ensemble:
        print_agreement(write_agreement_report(out_dir, out_dir, ensemble))
    return 0 if not counts.get("failed") else 1


if __name__ == "__main__":
    sys.exit(main())
