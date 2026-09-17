"""Match freshly extracted nodes against the graph they are being added to.

    python -m graphingest.reconcile build/graphs --against ~/mosaic/causal_graph.yaml
    python -m graphingest.reconcile doc.yaml --against corpus.yaml --report-only

Grounding resolves a term to an ontology. This resolves a *node* to one that
already exists: "increased larval abundance of Aedes dorsalis" extracted from a
new paper is the node three earlier papers already talk about, and it should be
that node, not a fourth copy of it.

The merge stage already unifies nodes whose identity tuples match exactly.
That is too strict on its own, because two annotators — or the same model on
two days — write ``measured_attribute`` differently: "larval abundance",
"abundance of larvae", "larval density". Reconciliation runs earlier, while the
document graph is still separate, and adopts the existing node's id wherever
the two are the same thing. The merge then joins them for free, and the graph
accumulates evidence on one node instead of growing near-duplicates.

**What is never merged.** Two nodes with different
``state_or_change_qualifier`` are different nodes, however alike their wording:
"increased salinity" and "decreased salinity" are opposite claims, and folding
them together would invert half the evidence. The same goes for ``entity_type``
and for a grounded entity term that differs from another grounded one — a CURIE
is an assertion, and two different CURIEs assert two different things.

Every decision is recorded with the score and the rule that produced it, so a
reconciliation you disagree with can be found and undone.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .cli import configure_logging, configure_stdio
from .consolidate import node_identity
from .graph_io import atomic_write_json, load_graph, save_graph
from .ground import looks_like_identifier, similarity

LOGGER = logging.getLogger("ingest.reconcile")

#: Below this, two differently-worded attributes are not the same measurement.
#: Higher than the grounding threshold on purpose: grounding a term loosely
#: costs one wrong CURIE, merging two nodes wrongly costs every edge on both.
DEFAULT_MIN_SCORE = 0.7


@dataclass
class Reconciliation:
    """One new node matched to an existing one."""

    new_id: str
    existing_id: str
    rule: str
    score: float
    new_name: str = ""
    existing_name: str = ""

    def to_dict(self) -> dict:
        return {
            "new_id": self.new_id,
            "existing_id": self.existing_id,
            "rule": self.rule,
            "score": round(self.score, 3),
            "new_name": self.new_name,
            "existing_name": self.existing_name,
        }


@dataclass
class ReconciliationReport:
    matched: int = 0
    unmatched: int = 0
    collapsed: int = 0
    matches: list[dict] = field(default_factory=list)
    by_rule: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "matched": self.matched,
            "unmatched": self.unmatched,
            "collapsed": self.collapsed,
            "by_rule": dict(sorted(self.by_rule.items())),
            "matches": self.matches,
        }

    def merge(self, other: "ReconciliationReport") -> None:
        self.matched += other.matched
        self.unmatched += other.unmatched
        self.collapsed += other.collapsed
        self.matches.extend(other.matches)
        for rule, count in other.by_rule.items():
            self.by_rule[rule] = self.by_rule.get(rule, 0) + count


#: A containment match is the same measurement said at more length, which is
#: slightly weaker evidence than the same words, so it is discounted.
CONTAINMENT_DISCOUNT = 0.9


def attribute_similarity(left: str, right: str) -> float:
    """How alike two ``measured_attribute`` strings are.

    Plain token overlap is too strict here, because annotators qualify the same
    measurement to different depths: "larval abundance" and "mosquito larval
    abundance" share two of three tokens and score 0.67, below any threshold
    worth having, while plainly being the same measurement. So one token set
    being contained in the other also counts, discounted.

    What this deliberately does *not* do is morphology. "larval abundance" and
    "abundance of larvae" do not match, and no amount of suffix-stripping makes
    that safe to guess at. They stay two nodes, the report says one was left
    unmatched, and a person can decide.
    """
    left_tokens, right_tokens = _token_set(left), _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    if left_tokens == right_tokens:
        return 1.0
    shared = len(left_tokens & right_tokens)
    if not shared:
        return 0.0
    jaccard = shared / len(left_tokens | right_tokens)
    containment = shared / min(len(left_tokens), len(right_tokens))
    return max(jaccard, containment * CONTAINMENT_DISCOUNT)


def _token_set(value: str) -> set[str]:
    # Reuses the grounder's tokenizer, so "salinity of soil" and "soil
    # salinity" are the same tokens in both halves of the pipeline.
    from .ground import _tokens

    return _tokens(value)


def _norm(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split()).lower()


def _facets(node: dict) -> tuple[str, str, str, str]:
    """The four fields a node's sameness is judged on."""
    return (
        _norm(node.get("entity_term")),
        _norm(node.get("measured_attribute")),
        _norm(node.get("state_or_change_qualifier")),
        _norm(node.get("entity_type")),
    )


class NodeReconciler:
    """Match nodes against an existing graph, by identity then by meaning."""

    def __init__(self, existing: Optional[dict] = None, min_score: float = DEFAULT_MIN_SCORE):
        self.min_score = min_score
        self.nodes: list[dict] = list((existing or {}).get("nodes") or [])
        # Exact identity, as the merge stage computes it: a free win.
        self._by_identity = {node_identity(node): node for node in self.nodes}
        self._by_id = {node["id"]: node for node in self.nodes if node.get("id")}
        # Grouped by the facets that must agree exactly, so a candidate search
        # compares attribute wording only against nodes it could possibly be.
        self._by_anchor: dict[tuple[str, str, str], list[dict]] = {}
        for node in self.nodes:
            entity, _, qualifier, entity_type = _facets(node)
            self._by_anchor.setdefault((entity, qualifier, entity_type), []).append(node)

    def __bool__(self) -> bool:
        return bool(self.nodes)

    # -- matching -----------------------------------------------------------

    def match(self, node: dict) -> Optional[Reconciliation]:
        """The existing node this one is, or None."""
        exact = self._by_identity.get(node_identity(node))
        if exact is not None and exact.get("id"):
            return self._result(node, exact, "identity", 1.0)

        entity, attribute, qualifier, entity_type = _facets(node)
        if not entity:
            return None

        # The qualifier and the entity type must agree exactly: "increased" and
        # "decreased" are opposite claims, not near-duplicates.
        candidates = self._by_anchor.get((entity, qualifier, entity_type)) or []
        if not candidates:
            return None

        # Judged on the raw term: the folded facet is lowercased, and "q13543883"
        # does not look like an identifier while "Q13543883" does.
        rule = "grounded" if looks_like_identifier(node.get("entity_term")) else "lexical"
        best, best_score = None, 0.0
        for candidate in candidates:
            score = self._attribute_score(attribute, _facets(candidate)[1])
            if score > best_score:
                best, best_score = candidate, score
        if best is None or best_score < self.min_score or not best.get("id"):
            return None
        return self._result(node, best, rule, best_score)

    def _adopt_identity(self, node: dict, existing_id: str) -> None:
        """Take on the existing node's identity fields, keeping our own evidence.

        Adopting the id alone is not enough. The merge stage groups by identity
        — entity term, attribute, qualifier, type — so a node that kept its own
        wording would merge as a *separate* node while carrying the same id,
        and the result would fail referential integrity with two nodes named
        alike. Being the same node means being the same node.

        The document's own wording is not lost: it stays in ``source_spans``
        and in the edges' ``original_sentence``, which is where the text a
        claim came from belongs.
        """
        existing = self._by_id.get(existing_id)
        if existing is None:
            return
        for slot in ("entity_term", "measured_attribute",
                     "state_or_change_qualifier", "entity_type", "name"):
            if existing.get(slot) is not None:
                node[slot] = existing[slot]

    def _attribute_score(self, left: str, right: str) -> float:
        """How alike two measured attributes are, with both-empty counting as agreement."""
        if left == right:
            return 1.0
        if not left or not right:
            # One side measures nothing in particular. That is weaker evidence
            # of sameness than matching wording, but the entity, qualifier and
            # type already agree, so it is not nothing.
            return self.min_score
        return attribute_similarity(left, right)

    @staticmethod
    def _result(node: dict, existing: dict, rule: str, score: float) -> Reconciliation:
        return Reconciliation(
            new_id=str(node.get("id")),
            existing_id=str(existing.get("id")),
            rule=rule,
            score=score,
            new_name=str(node.get("name") or ""),
            existing_name=str(existing.get("name") or ""),
        )

    # -- rewriting ----------------------------------------------------------

    def reconcile(self, graph: dict) -> ReconciliationReport:
        """Adopt existing node ids in ``graph``, in place."""
        report = ReconciliationReport()
        if not self.nodes:
            return report

        id_map: dict[str, str] = {}
        for node in graph.get("nodes") or []:
            if not isinstance(node, dict) or not node.get("id"):
                continue
            found = self.match(node)
            if found is None:
                report.unmatched += 1
                continue
            id_map[found.new_id] = found.existing_id
            node["id"] = found.existing_id
            self._adopt_identity(node, found.existing_id)
            report.matched += 1
            report.by_rule[found.rule] = report.by_rule.get(found.rule, 0) + 1
            report.matches.append(found.to_dict())

        if not id_map:
            return report

        _rewrite_references(graph, id_map)
        report.collapsed = _collapse_duplicate_ids(graph)
        return report


def _rewrite_references(graph: dict, id_map: dict[str, str]) -> None:
    """Point every edge endpoint at the id its node now has."""
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        for role in ("subject", "object"):
            reference = edge.get(role)
            if isinstance(reference, str) and reference in id_map:
                edge[role] = id_map[reference]
            elif isinstance(reference, dict) and reference.get("id") in id_map:
                reference["id"] = id_map[reference["id"]]
        for structure, key in (
            ("mediation", "mediator_node_ids"),
            ("moderation", "moderator_node_ids"),
        ):
            block = edge.get(structure)
            if isinstance(block, dict) and block.get(key):
                block[key] = [id_map.get(str(ref), ref) for ref in block[key]]
        comparator = edge.get("comparator")
        if isinstance(comparator, dict) and comparator.get("comparator_node_id"):
            comparator["comparator_node_id"] = id_map.get(
                str(comparator["comparator_node_id"]), comparator["comparator_node_id"]
            )


def _collapse_duplicate_ids(graph: dict) -> int:
    """Fold nodes that now share an id into one.

    Two nodes in the same document can match the same existing node — the model
    often extracts "larval abundance" and "abundance of larvae" from different
    sections of one paper. After adopting the existing id they would be two
    nodes with one id, which fails referential integrity, so they are merged
    here rather than left for the validator to reject.
    """
    seen: dict[str, dict] = {}
    kept: list[dict] = []
    collapsed = 0
    for node in graph.get("nodes") or []:
        identifier = node.get("id")
        if identifier in seen:
            _absorb(seen[identifier], node)
            collapsed += 1
            continue
        seen[identifier] = node
        kept.append(node)
    graph["nodes"] = kept
    return collapsed


def _absorb(target: dict, other: dict) -> None:
    """Union the evidence of a duplicate into the node that survives."""
    spans = target.setdefault("source_spans", [])
    for span in other.get("source_spans") or []:
        if span not in spans:
            spans.append(span)
    if not spans:
        target.pop("source_spans")
    applied = target.setdefault("applied_to", [])
    for entry in other.get("applied_to") or []:
        if entry not in applied:
            applied.append(entry)
    if not applied:
        target.pop("applied_to")
    for key, value in other.items():
        if key in {"id", "source_spans", "applied_to"}:
            continue
        if target.get(key) in (None, "", []):
            target[key] = value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def collect_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(
                path
                for suffix in ("*.yaml", "*.yml")
                for path in sorted(item.glob(suffix))
            )
        elif item.exists():
            paths.append(item)
        else:
            LOGGER.warning("Input not found, skipping: %s", item)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("graphs", nargs="+", type=Path,
                        help="Document graphs to reconcile")
    parser.add_argument("--against", type=Path, required=True,
                        help="The existing graph to match against (read-only)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-only", action="store_true",
                        help="Report the matches without rewriting anything")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)

    configure_stdio()
    configure_logging(args.log_level)

    if not args.against.exists():
        parser.error(f"No existing graph at {args.against}")
    existing = load_graph(args.against)
    reconciler = NodeReconciler(existing, args.min_score)
    LOGGER.info("Matching against %d existing node(s)", len(reconciler.nodes))

    paths = collect_paths(args.graphs)
    if not paths:
        parser.error("no graph files found")

    total = ReconciliationReport()
    for path in paths:
        try:
            graph = load_graph(path)
        except (OSError, ValueError) as error:
            LOGGER.error("Could not load %s: %s", path, error)
            continue
        report = reconciler.reconcile(graph)
        total.merge(report)
        LOGGER.info("%s: %d matched, %d new", path.name, report.matched, report.unmatched)
        if not args.report_only and report.matched:
            save_graph(graph, path)

    report_path = args.report or paths[0].parent / "reconciliation_report.json"
    atomic_write_json(report_path, {"against": str(args.against), **total.to_dict()})

    print(f"\nReconciled {len(paths)} graph(s) against {args.against.name}:")
    print(f"    {total.matched:5d}  node(s) matched an existing node")
    print(f"    {total.unmatched:5d}  node(s) are new")
    if total.collapsed:
        print(f"    {total.collapsed:5d}  duplicate(s) collapsed within a document")
    for rule, count in sorted(total.by_rule.items()):
        print(f"    {count:5d}  by {rule}")
    print(f"  wrote {report_path}")
    if args.report_only:
        print("  (--report-only: no graph was rewritten)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
