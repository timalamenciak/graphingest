"""Cross-model agreement: the same article, extracted by other models, compared.

Opt-in (``--ensemble``, or ``--witness MODEL``). The configured model is the
*primary*: its graph is the one saved, merged and trusted, exactly as without
this mode. One or more *witness* models extract the same article with the
same prompt and one-shot, their graphs are grounded the same way, and every
primary node and edge is checked against them:

* ``agreed``       every witness found it, and agreed on its core fields
* ``partial``      some witnesses found it, none contradicted it
* ``conflict``     a witness found it and disagreed: a different qualifier,
                   predicate, claim strength, or the arrow pointing backwards
* ``unsupported``  no witness found it: only the primary made this claim

Anything not ``agreed`` is flagged. The other direction is the recall signal:
a claim witnesses extracted that the primary did not is listed as a *possible
omission*, strongest when both its endpoints are nodes the primary has and it
is only the relation that is missing.

Matching is by meaning, not by string. Two models will not word a node alike,
so nodes are matched on entity term and measured attribute the way
reconciliation matches them (after grounding, so both sides speak in the same
CURIEs), but *without* requiring the qualifier or entity type to agree: those
are what is being compared, and "increased" against "decreased" is the
disagreement worth finding, not a reason to call them different nodes. Edges
match when their endpoints do.

Agreement is not correctness. Every witness reads the same prompt and the
same one-shot, which is the largest shared cause of shared mistakes, and a
weak witness disagrees for its own reasons. The per-witness agreement rate in
the summary is there to tell those apart.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import DEFAULT_LLM_CONFIG, load_yaml
from .graph_io import atomic_write, atomic_write_json
from .ground import looks_like_identifier
from .llm_client import LLMClient, LLMSettings
from .reconcile import DEFAULT_MIN_SCORE, attribute_similarity

LOGGER = logging.getLogger("ingest.ensemble")

STATUSES = ("agreed", "partial", "conflict", "unsupported", "unchecked")

DEFAULT_COMPARE_FIELDS = {
    "node": ["entity_type", "state_or_change_qualifier"],
    "edge": ["predicate", "claim_strength", "negated"],
}

#: Settings a witness may set; anything it leaves out is inherited from the
#: primary, as long as the two share a provider.
_SETTING_TYPES: dict[str, Any] = {
    "provider": str, "endpoint": str, "model": str, "api_key": str,
    "temperature": float, "max_tokens": int, "timeout": int,
    "structured_output": str, "max_repair_attempts": int, "extra_body": dict,
}
_WITNESS_KEYS = {"name", "max_chunk_characters"}

#: Booleans whose absence means ``false``: a model that did not write
#: ``negated`` did not negate the claim.
_FALSE_WHEN_ABSENT = {"negated"}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Witness:
    name: str
    settings: LLMSettings
    #: ``None`` chunks the way the primary run does. Set it for a witness
    #: with a smaller context window than the primary.
    max_chunk_characters: Optional[int] = None
    _client: Optional[LLMClient] = field(default=None, repr=False)

    def client(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient(self.settings)
        return self._client

    def describe(self) -> dict:
        described = {"name": self.name, "provider": self.settings.provider,
                     "model": self.settings.model}
        if self.settings.provider != "anthropic":
            described["endpoint"] = self.settings.endpoint
        return described


@dataclass
class EnsembleSettings:
    witnesses: list[Witness]
    match_min_score: float = DEFAULT_MIN_SCORE
    compare_fields: dict = field(default_factory=lambda: {
        kind: list(names) for kind, names in DEFAULT_COMPARE_FIELDS.items()
    })
    review_count: int = 40

    def describe(self) -> dict:
        return {
            "witnesses": [witness.describe() for witness in self.witnesses],
            "match_min_score": self.match_min_score,
            "compare_fields": self.compare_fields,
        }


def witness_settings(primary: LLMSettings, block: dict) -> LLMSettings:
    """A witness's LLM settings: its own block over the primary's.

    Sharing a provider, a witness inherits the endpoint and key, so a witness
    on the same LiteLLM or vLLM host is one line: its model. A witness on
    another provider inherits nothing but the generation budget and timeout —
    an OpenAI endpoint URL means nothing to the Anthropic SDK. Never inherited:
    logprobs (witnesses are compared, not scored) and ``extra_body``, which is
    model-specific.
    """
    provider = block.get("provider") or primary.provider
    settings = dataclasses.replace(
        primary, provider=provider, logprobs=False, prompt_logprobs=None,
        extra_body={}, marker_llm={},
    )
    if provider != primary.provider:
        settings.endpoint = "" if provider == "anthropic" else LLMSettings.endpoint
        settings.api_key = ""
        settings.structured_output = "tool_use" if provider == "anthropic" else "json_object"
    for key, cast in _SETTING_TYPES.items():
        if key != "provider" and block.get(key) is not None:
            setattr(settings, key, cast(block[key]))
    unknown = set(block) - set(_SETTING_TYPES) - _WITNESS_KEYS
    if unknown:
        LOGGER.warning("Witness %r: ignoring unknown setting(s) %s",
                       block.get("name"), sorted(unknown))
    settings.validate()
    return settings


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "witness"


def ensemble_from_config(
    path: str | Path | None,
    primary: LLMSettings,
    selected: Optional[list[str]] = None,
    min_score: Optional[float] = None,
) -> EnsembleSettings:
    """Witnesses from the ``ensemble:`` block, or named on the command line.

    A name that is not a configured witness is taken as a model id on the
    primary's endpoint, so ``--witness openai/models/Llama-4`` works without
    touching the config.
    """
    block = load_yaml(path or DEFAULT_LLM_CONFIG).get("ensemble") or {}
    configured = {
        str(entry["name"]): entry
        for entry in block.get("witnesses") or []
        if isinstance(entry, dict) and entry.get("name")
    }
    names = list(selected or configured)
    if not names:
        raise ValueError(
            "--ensemble needs at least one witness: list them under "
            "ensemble.witnesses in config/pipeline.yaml, or name a model with "
            "--witness"
        )
    witnesses: list[Witness] = []
    for name in names:
        entry = configured.get(name) or {"model": name}
        settings = witness_settings(primary, entry)
        if (settings.model, settings.endpoint) == (primary.model, primary.endpoint):
            LOGGER.warning(
                "Witness %r is the primary model on the primary endpoint: a "
                "model agreeing with itself is not a second opinion", name,
            )
        witnesses.append(Witness(
            name=_safe_name(name), settings=settings,
            max_chunk_characters=entry.get("max_chunk_characters"),
        ))
    compare = block.get("compare_fields") or {}
    ensemble = EnsembleSettings(
        witnesses=witnesses,
        match_min_score=float(
            min_score if min_score is not None
            else block.get("match_min_score", DEFAULT_MIN_SCORE)
        ),
        review_count=int(block.get("review_count", 40)),
    )
    for kind in ("node", "edge"):
        if compare.get(kind):
            ensemble.compare_fields[kind] = list(compare[kind])
    return ensemble


def ensemble_enabled_in_config(path: str | Path | None) -> bool:
    try:
        return bool((load_yaml(path or DEFAULT_LLM_CONFIG).get("ensemble") or {})
                    .get("enabled", False))
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _norm(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split()).lower()


def _term_score(left: Any, right: Any) -> float:
    if not left or not right:
        return 0.0
    if str(left).strip() == str(right).strip() or _norm(left) == _norm(right):
        return 1.0
    if looks_like_identifier(left) or looks_like_identifier(right):
        # Two different CURIEs assert two different things; a CURIE and free
        # text went through the same grounder and came out differently.
        return 0.0
    return attribute_similarity(str(left), str(right))


def node_similarity(primary: dict, witness: dict, floor: float = DEFAULT_MIN_SCORE) -> float:
    """How likely two nodes are the same variable, ignoring what is compared.

    Qualifier and entity type are deliberately not part of this: they are
    the fields being checked for agreement. Where one model folds the
    measurement into the entity term and the other splits it out ("soil
    salinity" / "" against "soil" / "salinity"), the two read together still
    match, at a discount.
    """
    left_term, right_term = primary.get("entity_term"), witness.get("entity_term")
    term = _term_score(left_term, right_term)
    left_attr = _norm(primary.get("measured_attribute"))
    right_attr = _norm(witness.get("measured_attribute"))
    if left_attr == right_attr:
        attribute = 1.0
    elif not left_attr or not right_attr:
        attribute = floor
    else:
        attribute = attribute_similarity(left_attr, right_attr)
    combined = 0.0
    if not looks_like_identifier(left_term) and not looks_like_identifier(right_term):
        combined = 0.9 * attribute_similarity(
            f"{_norm(left_term)} {left_attr}", f"{_norm(right_term)} {right_attr}"
        )
    return max(min(term, attribute), combined)


def _comparable(name: str, value: Any) -> Any:
    if name in _FALSE_WHEN_ABSENT:
        return bool(value)
    if isinstance(value, list):
        return sorted(_norm(item) for item in value)
    return value


def compare_fields(primary: dict, witness: dict, names: list[str], prefix: str = "") -> list[dict]:
    """Fields both sides filled in, and filled in differently.

    A field one model left out is not a disagreement: it is a model that said
    less, which the omission and support counts already capture.
    """
    conflicts = []
    for name in names:
        left, right = primary.get(name), witness.get(name)
        if name not in _FALSE_WHEN_ABSENT and (left is None or right is None):
            continue
        if _comparable(name, left) != _comparable(name, right):
            conflicts.append({"field": prefix + name, "primary": left, "witness": right})
    return conflicts


def map_nodes(
    primary_nodes: list[dict], witness_nodes: list[dict], settings: EnsembleSettings
) -> dict[str, tuple[str, float]]:
    """Each witness node's best primary counterpart, if it has one.

    Ties go to the counterpart that agrees on more compared fields, so where
    the primary has both "increased salinity" and "decreased salinity", the
    witness's "increased salinity" is matched to the one it agrees with.
    """
    fields_ = settings.compare_fields["node"]
    mapping: dict[str, tuple[str, float]] = {}
    for witness in witness_nodes:
        best: Optional[tuple[tuple[float, int], str]] = None
        for primary in primary_nodes:
            score = node_similarity(primary, witness, settings.match_min_score)
            if score < settings.match_min_score:
                continue
            key = (score, -len(compare_fields(primary, witness, fields_)))
            if best is None or key > best[0]:
                best = (key, primary["id"])
        if best is not None and witness.get("id"):
            mapping[str(witness["id"])] = (best[1], round(best[0][0], 3))
    return mapping


_RANK = {"agrees": 2, "conflict": 1, "missing": 0}


def _record(verdicts: dict, witness: str, verdict: dict) -> None:
    """Keep the best verdict a witness gave an item: agreement beats conflict."""
    current = verdicts.get(witness)
    if current is None or _RANK[verdict["verdict"]] > _RANK[current["verdict"]]:
        verdicts[witness] = verdict


def compare_graphs(
    primary: dict, witnesses: dict[str, dict], settings: EnsembleSettings
) -> dict:
    """Every primary item's verdict from each witness, and what only they found.

    Keyed by the primary graph's ids. The result holds verdicts, not
    statuses: statuses depend on how many witnesses answered, which is
    settled in :func:`finalize`.
    """
    nodes = [n for n in primary.get("nodes") or [] if isinstance(n, dict) and n.get("id")]
    edges = [e for e in primary.get("edges") or [] if isinstance(e, dict) and e.get("id")]
    node_by_id = {node["id"]: node for node in nodes}
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for edge in edges:
        by_pair.setdefault((edge.get("subject"), edge.get("object")), []).append(edge)

    node_verdicts: dict[str, dict[str, dict]] = {node["id"]: {} for node in nodes}
    edge_verdicts: dict[str, dict[str, dict]] = {edge["id"]: {} for edge in edges}
    omissions: dict[tuple, dict] = {}
    node_fields = settings.compare_fields["node"]
    edge_fields = settings.compare_fields["edge"]

    for name, graph in witnesses.items():
        w_nodes = [n for n in graph.get("nodes") or [] if isinstance(n, dict)]
        w_node_by_id = {str(node.get("id")): node for node in w_nodes}
        mapping = map_nodes(nodes, w_nodes, settings)

        for w_id, (p_id, score) in mapping.items():
            conflicts = compare_fields(node_by_id[p_id], w_node_by_id[w_id], node_fields)
            _record(node_verdicts[p_id], name, {
                "verdict": "conflict" if conflicts else "agrees",
                "matched": w_id, "score": score, "conflicts": conflicts,
            })

        for w_edge in graph.get("edges") or []:
            if not isinstance(w_edge, dict):
                continue
            w_subject, w_object = str(w_edge.get("subject")), str(w_edge.get("object"))
            subject = mapping.get(w_subject, (None,))[0]
            obj = mapping.get(w_object, (None,))[0]
            matched = False
            if subject and obj:
                for p_edge in by_pair.get((subject, obj)) or []:
                    matched = True
                    # The same arrow between variables in opposite states is a
                    # different claim, so the endpoints' fields count too.
                    conflicts = (
                        compare_fields(p_edge, w_edge, edge_fields)
                        + compare_fields(node_by_id[subject],
                                         w_node_by_id.get(w_subject, {}),
                                         node_fields, "subject.")
                        + compare_fields(node_by_id[obj],
                                         w_node_by_id.get(w_object, {}),
                                         node_fields, "object.")
                    )
                    _record(edge_verdicts[p_edge["id"]], name, {
                        "verdict": "conflict" if conflicts else "agrees",
                        "matched": w_edge.get("id"), "conflicts": conflicts,
                        "sentence": w_edge.get("original_sentence"),
                    })
                if not matched:
                    for p_edge in by_pair.get((obj, subject)) or []:
                        matched = True
                        _record(edge_verdicts[p_edge["id"]], name, {
                            "verdict": "conflict", "matched": w_edge.get("id"),
                            "conflicts": [{"field": "direction",
                                           "primary": "subject -> object",
                                           "witness": "object -> subject"}],
                            "sentence": w_edge.get("original_sentence"),
                        })
            if matched:
                continue
            _add_omission(omissions, name, w_edge, subject, obj,
                          w_node_by_id, node_by_id)

    return {"nodes": node_verdicts, "edges": edge_verdicts,
            "omissions": list(omissions.values())}


def _endpoint(primary_id: Optional[str], witness_node: dict) -> tuple[str, str]:
    if primary_id:
        return ("primary", primary_id)
    return ("new", "|".join(_norm(witness_node.get(key)) for key in (
        "entity_term", "measured_attribute", "state_or_change_qualifier")))


def _add_omission(
    omissions: dict, witness: str, w_edge: dict, subject: Optional[str],
    obj: Optional[str], w_nodes: dict, p_nodes: dict,
) -> None:
    w_subject = w_nodes.get(str(w_edge.get("subject"))) or {}
    w_object = w_nodes.get(str(w_edge.get("object"))) or {}
    key = (_endpoint(subject, w_subject), _endpoint(obj, w_object))

    def label(primary_id: Optional[str], node: dict) -> str:
        chosen = p_nodes.get(primary_id) if primary_id else node
        return str((chosen or {}).get("name") or (chosen or {}).get("entity_term") or "?")

    entry = omissions.setdefault(key, {
        "subject": key[0][1] if key[0][0] == "primary" else None,
        "object": key[1][1] if key[1][0] == "primary" else None,
        "label": f"{label(subject, w_subject)} --{w_edge.get('predicate')}--> "
                 f"{label(obj, w_object)}",
        "endpoints_in_primary": bool(subject and obj),
        "witnesses": [],
        "evidence": [],
    })
    if witness not in entry["witnesses"]:
        entry["witnesses"].append(witness)
    spans = w_edge.get("source_spans") or []
    entry["evidence"].append({
        "witness": witness,
        "predicate": w_edge.get("predicate"),
        "claim_strength": w_edge.get("claim_strength"),
        "sentence": w_edge.get("original_sentence"),
        "quote_found": bool(spans) and all(
            "start_char" in span for span in spans if isinstance(span, dict)
        ),
    })


def remap_after_reconciliation(result: dict, matches: list[dict]) -> None:
    """Follow primary nodes that reconciliation re-identified.

    Agreement is judged on the graph as extracted, before reconciliation
    swaps a node's wording for the existing corpus node's (which the
    witnesses never saw). The verdicts are then moved to the id the saved
    graph carries. Two nodes collapsed into one keep the better verdicts.
    """
    for match in matches:
        new, existing = match.get("new_id"), match.get("existing_id")
        if not new or not existing or new == existing or new not in result["nodes"]:
            continue
        moved = result["nodes"].pop(new)
        target = result["nodes"].setdefault(existing, {})
        for witness, verdict in moved.items():
            _record(target, witness, verdict)
        for omission in result["omissions"]:
            for role in ("subject", "object"):
                if omission.get(role) == new:
                    omission[role] = existing


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


def status_of(verdicts: dict[str, dict], answered: list[str]) -> tuple[str, int, int]:
    """(status, support, agreeing) over the witnesses that answered."""
    if not answered:
        return "unchecked", 0, 0
    given = [verdicts.get(name) or {"verdict": "missing"} for name in answered]
    support = sum(1 for verdict in given if verdict["verdict"] != "missing")
    agreeing = sum(1 for verdict in given if verdict["verdict"] == "agrees")
    if any(verdict["verdict"] == "conflict" for verdict in given):
        return "conflict", support, agreeing
    if agreeing == len(answered):
        return "agreed", support, agreeing
    return ("partial" if agreeing else "unsupported"), support, agreeing


def finalize(
    result: dict,
    graph: dict,
    witness_runs: list[dict],
    settings: EnsembleSettings,
    document: dict,
) -> dict:
    """The per-document sidecar, keyed by the ids the saved graph carries."""
    answered = [run["name"] for run in witness_runs if run["status"] != "failed"]
    names = {node.get("id"): node.get("name") or node.get("entity_term")
             for node in graph.get("nodes") or []}

    def entries(items: list[dict], verdicts: dict, kind: str) -> dict:
        out = {}
        for item in items:
            given = verdicts.get(item["id"], {})
            status, support, agreeing = status_of(given, answered)
            entry = {
                "status": status,
                "flagged": status not in ("agreed", "unchecked"),
                "support": f"{support}/{len(answered)}",
                "agreeing": agreeing,
                "witnesses": {name: given.get(name) or {"verdict": "missing"}
                              for name in answered},
            }
            if kind == "edge":
                subject = names.get(item.get("subject")) or item.get("subject")
                obj = names.get(item.get("object")) or item.get("object")
                negated = "NOT " if item.get("negated") else ""
                entry = {"label": f"{subject} --{negated}{item.get('predicate')}--> {obj}",
                         **entry}
            else:
                entry = {"name": names.get(item["id"]), **entry}
            out[item["id"]] = entry
        return out

    nodes = entries([n for n in graph.get("nodes") or [] if n.get("id")],
                    result["nodes"], "node")
    edges = entries([e for e in graph.get("edges") or [] if e.get("id")],
                    result["edges"], "edge")
    omissions = sorted(
        ({**omission, "support": f"{len(omission['witnesses'])}/{len(answered)}"}
         for omission in result["omissions"]),
        key=lambda o: (-len(o["witnesses"]), not o["endpoints_in_primary"], o["label"]),
    )
    return {
        "document": document,
        "settings": settings.describe(),
        "witnesses": witness_runs,
        "summary": summarize(nodes, edges, omissions, witness_runs, answered),
        "nodes": nodes,
        "edges": edges,
        "possible_omissions": omissions,
    }


def summarize(nodes: dict, edges: dict, omissions: list[dict],
              witness_runs: list[dict], answered: list[str]) -> dict:
    per_witness = {}
    for name in answered:
        found = sum(1 for e in edges.values() if e["witnesses"][name]["verdict"] != "missing")
        agreed = sum(1 for e in edges.values() if e["witnesses"][name]["verdict"] == "agrees")
        per_witness[name] = {
            "edges_found": found,
            "edges_agreed": agreed,
            "only_this_witness": sum(1 for o in omissions if o["witnesses"] == [name]),
        }
    return {
        "checked_by": answered,
        "failed": [run["name"] for run in witness_runs if run["status"] == "failed"],
        "nodes": {status: sum(1 for e in nodes.values() if e["status"] == status)
                  for status in STATUSES},
        "edges": {status: sum(1 for e in edges.values() if e["status"] == status)
                  for status in STATUSES},
        "flagged_nodes": sum(1 for e in nodes.values() if e["flagged"]),
        "flagged_edges": sum(1 for e in edges.values() if e["flagged"]),
        "possible_omissions": len(omissions),
        "possible_omissions_corroborated": sum(
            1 for o in omissions if len(o["witnesses"]) > 1
        ),
        "per_witness": per_witness,
    }


def sidecar_path(graph_path: Path) -> Path:
    return graph_path.with_name(f"{graph_path.stem}.agreement.json")


def witness_graph_path(out_dir: Path, witness: Witness, slug: str) -> Path:
    """Witness graphs live under their own directory, out of the merge's way."""
    return out_dir / "witnesses" / witness.name / f"{slug}.yaml"


# ---------------------------------------------------------------------------
# Corpus report
# ---------------------------------------------------------------------------


def _conflict_text(entry: dict) -> str:
    parts = []
    for name, verdict in entry["witnesses"].items():
        for conflict in verdict.get("conflicts") or []:
            parts.append(f"{conflict['field']}: {conflict['primary']} vs "
                         f"{name} {conflict['witness']}")
    return "; ".join(parts)


def corpus_report(graphs_dir: Path, settings: EnsembleSettings) -> Optional[dict]:
    """Roll every ``*.agreement.json`` under ``graphs_dir`` into one report."""
    totals = {kind: {status: 0 for status in STATUSES} for kind in ("nodes", "edges")}
    witnesses: dict[str, dict] = {}
    disputed, omissions, documents = [], [], []
    crosstab: dict[str, dict[str, int]] = {}
    for path in sorted(graphs_dir.glob("*.agreement.json")):
        try:
            sidecar = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            LOGGER.warning("Unreadable agreement sidecar %s: %s", path.name, error)
            continue
        slug = path.name[: -len(".agreement.json")]
        summary = sidecar.get("summary") or {}
        for kind in ("nodes", "edges"):
            for status, count in (summary.get(kind) or {}).items():
                totals[kind][status] = totals[kind].get(status, 0) + count
        for name, stats in (summary.get("per_witness") or {}).items():
            slot = witnesses.setdefault(name, {"documents": 0, "edges_found": 0,
                                               "edges_agreed": 0, "only_this_witness": 0,
                                               "primary_edges": 0})
            slot["documents"] += 1
            slot["primary_edges"] += len(sidecar.get("edges") or {})
            for key in ("edges_found", "edges_agreed", "only_this_witness"):
                slot[key] += stats.get(key, 0)

        buckets = _confidence_buckets(graphs_dir / f"{slug}.confidence.json")
        for identifier, entry in (sidecar.get("edges") or {}).items():
            bucket = buckets.get(identifier)
            if bucket:
                row = crosstab.setdefault(entry["status"], {})
                row[bucket] = row.get(bucket, 0) + 1
            if entry["flagged"]:
                disputed.append({"document": slug, "id": identifier,
                                 "label": entry["label"], "status": entry["status"],
                                 "support": entry["support"], "confidence": bucket,
                                 "conflicts": _conflict_text(entry)})
        for omission in sidecar.get("possible_omissions") or []:
            first = next((e for e in omission["evidence"] if e.get("sentence")), {})
            omissions.append({"document": slug, "label": omission["label"],
                              "support": omission["support"],
                              "witnesses": omission["witnesses"],
                              "endpoints_in_primary": omission["endpoints_in_primary"],
                              "sentence": first.get("sentence"),
                              "quote_found": first.get("quote_found")})
        documents.append({"document": slug, "checked_by": summary.get("checked_by"),
                          "failed": summary.get("failed"), "edges": summary.get("edges"),
                          "flagged_edges": summary.get("flagged_edges"),
                          "flagged_nodes": summary.get("flagged_nodes"),
                          "possible_omissions": summary.get("possible_omissions")})
    if not documents:
        return None
    order = {"conflict": 0, "unsupported": 1, "partial": 2}
    confidence_order = {"low": 0, "medium": 1, "high": 2, None: 3, "unscored": 3}
    disputed.sort(key=lambda d: (order.get(d["status"], 3),
                                 confidence_order.get(d["confidence"], 3)))
    omissions.sort(key=lambda o: (-len(o["witnesses"]), not o["endpoints_in_primary"]))
    return {
        "documents": len(documents),
        "witnesses": {
            name: {**stats, "agreement_rate": round(
                stats["edges_agreed"] / stats["primary_edges"], 3)
                if stats["primary_edges"] else None}
            for name, stats in witnesses.items()
        },
        "buckets": totals,
        "agreement_vs_confidence": crosstab,
        "disputed": disputed,
        "possible_omissions": omissions,
        "by_document": documents,
    }


def _confidence_buckets(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {identifier: entry.get("bucket")
            for identifier, entry in (sidecar.get("edges") or {}).items()}


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _counts(counts: dict) -> str:
    total = sum(counts.values()) or 1
    return " · ".join(f"{status} {counts.get(status, 0)} "
                      f"({100 * counts.get(status, 0) / total:.0f}%)"
                      for status in STATUSES if counts.get(status) or status != "unchecked")


def render_markdown(report: dict, limit: int = 40) -> str:
    """The reviewer's page: where the models disagree, and what the primary missed."""
    lines = [
        "# Cross-model agreement",
        "",
        f"{report['documents']} document(s). The graph is the primary model's; "
        "each witness extracted the same articles with the same prompt, and "
        "every primary claim was checked against them. **agreed**: every "
        "witness found it and agreed on its core fields. **partial**: some "
        "found it, none contradicted it. **conflict**: a witness found it and "
        "disagreed. **unsupported**: only the primary made it. Anything not "
        "agreed is flagged. Agreement is not correctness: every witness read "
        "the same prompt and one-shot.",
        "",
        f"- **Edges (claims):** {_counts(report['buckets']['edges'])}",
        f"- **Nodes:** {_counts(report['buckets']['nodes'])}",
        "",
        "## Witnesses",
        "",
        "How often each witness agreed with the primary. A witness that rarely "
        "agrees with it, and alone finds many claims no one else does, is "
        "telling you about itself rather than about the primary.",
        "",
        "| witness | docs | primary edges agreed | found, disagreed | claims only it found |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in report["witnesses"].items():
        rate = stats.get("agreement_rate")
        lines.append(
            f"| {_cell(name)} | {stats['documents']} "
            f"| {stats['edges_agreed']} ({'' if rate is None else f'{100 * rate:.0f}%'}) "
            f"| {stats['edges_found'] - stats['edges_agreed']} "
            f"| {stats['only_this_witness']} |"
        )
    lines.append("")

    if report["disputed"]:
        with_confidence = any(d["confidence"] for d in report["disputed"])
        lines += ["## Flagged claims", "",
                  "Conflicts first" + (", weakest token confidence first within each"
                                       if with_confidence else "") + ".", "",
                  "| doc | claim | status | support | "
                  + ("confidence | " if with_confidence else "") + "disagreement |",
                  "|---|---|---|---:|" + ("---|" if with_confidence else "") + "---|"]
        for entry in report["disputed"][:limit]:
            lines.append(
                f"| {_cell(entry['document'])} | {_cell(entry['label'])} "
                f"| {entry['status']} | {entry['support']} | "
                + (f"{_cell(entry['confidence'])} | " if with_confidence else "")
                + f"{_cell(entry['conflicts'])} |"
            )
        if len(report["disputed"]) > limit:
            lines.append(f"\n…and {len(report['disputed']) - limit} more in "
                         "agreement_report.json.")
        lines.append("")

    if report["possible_omissions"]:
        lines += ["## Possible omissions", "",
                  "Claims witnesses extracted and the primary did not. Most "
                  "telling when several witnesses agree, and when the primary "
                  "already has both nodes and only the relation is missing.", "",
                  "| doc | claim | found by | both nodes in primary | witness quote |",
                  "|---|---|---|---|---|"]
        for entry in report["possible_omissions"][:limit]:
            quote = entry.get("sentence") or ""
            if quote and entry.get("quote_found") is False:
                quote += " ⚠ not in article"
            lines.append(
                f"| {_cell(entry['document'])} | {_cell(entry['label'])} "
                f"| {_cell(', '.join(entry['witnesses']))} ({entry['support']}) "
                f"| {'yes' if entry['endpoints_in_primary'] else 'no'} "
                f"| {_cell(quote[:200])} |"
            )
        lines.append("")

    if report["agreement_vs_confidence"]:
        lines += ["## Agreement against token confidence", "",
                  "Edges by agreement status and by logprob bucket (from "
                  "--confidence). Where the two signals agree they reinforce "
                  "each other; conflicts the model was confident about are the "
                  "ones logprobs alone would have let through.", "",
                  "| status | high | medium | low | unscored |",
                  "|---|---:|---:|---:|---:|"]
        for status in STATUSES:
            row = report["agreement_vs_confidence"].get(status)
            if row:
                lines.append(f"| {status} | " + " | ".join(
                    str(row.get(bucket, 0))
                    for bucket in ("high", "medium", "low", "unscored")) + " |")
        lines.append("")

    lines += ["## By document", "",
              "| doc | checked by | edges agreed/partial/conflict/unsupported "
              "| flagged edges | flagged nodes | possible omissions |",
              "|---|---|---|---:|---:|---:|"]
    for doc in report["by_document"]:
        edges = doc.get("edges") or {}
        checked = ", ".join(doc.get("checked_by") or []) or "none"
        if doc.get("failed"):
            checked += f" (failed: {', '.join(doc['failed'])})"
        lines.append(
            f"| {_cell(doc['document'])} | {_cell(checked)} | "
            + "/".join(str(edges.get(s, 0)) for s in
                       ("agreed", "partial", "conflict", "unsupported"))
            + f" | {doc.get('flagged_edges')} | {doc.get('flagged_nodes')} "
            f"| {doc.get('possible_omissions')} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_corpus_report(
    graphs_dir: Path, out_dir: Path, settings: EnsembleSettings
) -> Optional[Path]:
    """``agreement_report.json`` and ``agreement_summary.md`` in ``out_dir``."""
    report = corpus_report(graphs_dir, settings)
    if report is None:
        return None
    atomic_write_json(out_dir / "agreement_report.json", report)
    return atomic_write(out_dir / "agreement_summary.md",
                        render_markdown(report, settings.review_count))


def print_summary(summary_path: Optional[Path]) -> None:
    if summary_path is None:
        return
    report = json.loads((summary_path.parent / "agreement_report.json")
                        .read_text(encoding="utf-8"))
    print(f"  agreement, edges: {_counts(report['buckets']['edges'])}")
    omitted = len(report["possible_omissions"])
    corroborated = sum(1 for o in report["possible_omissions"] if len(o["witnesses"]) > 1)
    print(f"  possible omissions: {omitted} ({corroborated} found by more than one witness)")
    print(f"  wrote {summary_path}")
