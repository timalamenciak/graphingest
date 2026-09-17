"""Coerce raw LLM output into a schema-valid CAMO graph.

Models return *nearly* right enum values: ``"Direct causal"``, ``"reduces"``,
``"not specified"``. The predecessor tool handled this with one hand-written
alias table per enum, which meant every schema bump needed a matching code
edit and drifted silently when it did not get one.

Here the coercion is driven by the schema itself: the ExtractionProfile knows
which slots are enums and what each enum permits, so most fixes fall out of
case/punctuation folding and token matching. A small curated table covers the
handful of mappings no amount of string similarity can derive -- ``correlation
-> associational`` is a modelling decision, not a typo.

Everything that could not be coerced is reported rather than silently dropped.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import ClassProfile, ExtractionProfile, SlotProfile

LOGGER = logging.getLogger("camo.normalize")

#: Mappings that genuinely cannot be derived from string similarity, because
#: the source and target words share no useful surface form. Keyed by enum name
#: so an alias for one enum cannot leak into another.
CURATED_ALIASES: dict[str, dict[str, str]] = {
    "ClaimStrengthEnum": {
        "correlation": "associational",
        "correlational": "associational",
        "association": "associational",
        "causal": "direct_causal",
        "direct": "direct_causal",
        "hedged": "uncertain_causal",
        "possible": "uncertain_causal",
        "none": "no_relationship",
        "null": "no_relationship",
        "null_result": "no_relationship",
    },
    "CausalPredicateEnum": {
        "leads_to": "causes",
        "results_in": "causes",
        "produces": "causes",
        "drives": "causes",
        "increases": "causes",
        "promotes": "enables",
        "facilitates": "enables",
        "decreases": "disrupts",
        "reduces": "disrupts",
        "suppresses": "disrupts",
        "inhibits": "prevents",
        "blocks": "prevents",
        "affects": "regulates",
        "modulates": "regulates",
        # 0.7.3 predicates. Polarity belongs on the node qualifier in 0.7.9;
        # see extract.migrate_schema for the qualifier-aware treatment.
        "positively_regulates": "regulates",
        "negatively_regulates": "regulates",
    },
    "PhilosophicalAccountEnum": {
        "intervention": "interventionist",
        "manipulation": "interventionist",
        "mechanism": "mechanistic",
        "difference_making": "probabilistic",
        "statistical": "probabilistic",
        "variation": "probabilistic",
        "inus": "inus_component",
        "constant_conjunction": "regularity",
    },
    "FeatureAssertionEnum": {
        "yes": "explicitly_asserted",
        "true": "explicitly_asserted",
        "asserted": "explicitly_asserted",
        "stated": "explicitly_asserted",
        "implicit": "implicitly_assumed",
        "implied": "implicitly_assumed",
        "assumed": "implicitly_assumed",
        "no": "explicitly_denied",
        "false": "explicitly_denied",
        "denied": "explicitly_denied",
        "unknown": "not_addressed",
        "unspecified": "not_addressed",
        "not_specified": "not_addressed",
        "not_reported": "not_addressed",
        "not_discussed": "not_addressed",
        "not_applicable": "not_addressed",
        "ambiguous": "not_addressed",
        "none": "not_addressed",
        "na": "not_addressed",
        "n_a": "not_addressed",
    },
    "StateOrChangeQualifierEnum": {
        "increase": "increased",
        "elevated": "increased",
        "higher": "increased",
        "enhanced": "increased",
        "improved": "increased",
        "recovered": "increased",
        "greater": "increased",
        "decrease": "decreased",
        "reduction": "decreased",
        "reduced": "decreased",
        "declined": "decreased",
        "lower": "decreased",
        "suppressed": "decreased",
        "stable": "unchanged",
        "maintained": "unchanged",
        "no_change": "unchanged",
        "constant": "unchanged",
        "established": "present",
        "colonized": "present",
        "lost": "absent",
        "eliminated": "absent",
        "extirpated": "removed",
        "started": "initiated",
        "began": "initiated",
        "ended": "terminated",
        "stopped": "terminated",
        "ceased": "terminated",
        "continuing": "ongoing",
    },
    "EntityTypeEnum": {
        "variable": "environmental_variable",
        "environmental_factor": "environmental_variable",
        "abiotic_factor": "environmental_variable",
        "measurement": "environmental_variable",
        "process": "environmental_process",
        "intervention": "management_intervention",
        "management_action": "management_intervention",
        "management_process": "management_intervention",
        "treatment": "management_intervention",
        "action": "management_intervention",
        "species": "taxon",
        "taxa": "taxon",
        "organism": "taxon",
    },
    "TokenTypeEnum": {
        "instance": "token",
        "singular": "token",
        "general": "type",
        "generic": "type",
        "unknown": "ambiguous",
    },
    "DeterminismEnum": {
        "deterministic": "deterministic_process",
        "indeterministic": "indeterministic_process",
        "stochastic": "indeterministic_process",
        "probabilistic": "indeterministic_process",
        "epistemic": "epistemic_probability_only",
        "unknown": "ambiguous",
    },
    "ContributingSoleEnum": {
        "sole": "sole_cause",
        "only": "sole_cause",
        "contributing": "contributing_cause",
        "partial": "contributing_cause",
        "multiple": "contributing_cause",
        "one_of_several": "contributing_cause",
    },
    "ProximateDistalEnum": {"both": "both_specified", "immediate": "proximate", "root": "distal"},
    "ReversibilityEnum": {
        "partial": "partially_reversible",
        "partially": "partially_reversible",
        "permanent": "irreversible",
    },
    "DirectionStatusEnum": {
        "explicit": "asserted",
        "unidirectional": "asserted",
        "stated": "asserted",
        "reciprocal": "bidirectional",
        "unclear": "uncertain",
    },
    "EvidenceObjectEnum": {
        "difference_making": "correlation",
        "statistical": "correlation",
        "production": "mechanism",
        "mechanistic": "mechanism",
    },
}


@dataclass
class NormalizationReport:
    """What normalization changed, and what it could not fix."""

    coerced: Counter = field(default_factory=Counter)
    dropped: list[dict] = field(default_factory=list)
    generated_ids: int = 0
    defaults_applied: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "coerced": dict(self.coerced),
            "dropped": self.dropped,
            "generated_ids": self.generated_ids,
            "defaults_applied": dict(self.defaults_applied),
        }


# ---------------------------------------------------------------------------
# Enum coercion
# ---------------------------------------------------------------------------


def _fold(value: str) -> str:
    """Lowercase, collapse punctuation and whitespace to single underscores."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def coerce_enum(
    value: Any, slot: SlotProfile
) -> tuple[Optional[str], Optional[str]]:
    """Coerce one value to a permissible value of ``slot``'s enum.

    Returns ``(coerced_value, how)``; ``how`` is ``None`` when the value was
    already valid, and ``coerced_value`` is ``None`` when nothing matched.
    """
    if value is None:
        return None, None
    permitted = {detail["name"] for detail in slot.enum_values}
    text = str(value).strip()
    if text in permitted:
        return text, None

    folded = _fold(text)
    if folded in permitted:
        return folded, "case/punctuation"

    by_fold = {_fold(name): name for name in permitted}
    if folded in by_fold:
        return by_fold[folded], "case/punctuation"

    curated = CURATED_ALIASES.get(slot.range or "", {})
    if folded in curated and curated[folded] in permitted:
        return curated[folded], "curated alias"

    # Token containment: "direct causal relationship" -> direct_causal
    wanted = set(folded.split("_"))
    best, best_score = None, 0.0
    for name in permitted:
        tokens = set(name.split("_"))
        overlap = len(wanted & tokens)
        if not overlap:
            continue
        score = overlap / len(wanted | tokens)
        if score > best_score:
            best, best_score = name, score
    if best and best_score >= 0.5:
        return best, f"token overlap {best_score:.2f}"

    return None, None


def normalize_enum_slot(
    container: dict, slot: SlotProfile, path: str, report: NormalizationReport
) -> None:
    """Coerce a slot's value(s) in place, dropping what cannot be coerced."""
    if slot.name not in container:
        return
    raw = container[slot.name]

    if slot.multivalued:
        values = raw if isinstance(raw, list) else [raw]
        kept = []
        for item in values:
            coerced, how = coerce_enum(item, slot)
            if coerced is None:
                report.dropped.append(
                    {"path": f"{path}.{slot.name}", "value": item, "enum": slot.range}
                )
                continue
            if how:
                report.coerced[f"{slot.range}: {how}"] += 1
            if coerced not in kept:
                kept.append(coerced)
        if kept:
            container[slot.name] = kept
        else:
            container.pop(slot.name)
        return

    coerced, how = coerce_enum(raw, slot)
    if coerced is None:
        report.dropped.append(
            {"path": f"{path}.{slot.name}", "value": raw, "enum": slot.range}
        )
        container.pop(slot.name)
        return
    if how:
        report.coerced[f"{slot.range}: {how}"] += 1
    container[slot.name] = coerced


# ---------------------------------------------------------------------------
# Recursive graph normalization
# ---------------------------------------------------------------------------


def normalize_instance(
    instance: dict,
    class_profile: ClassProfile,
    profile: ExtractionProfile,
    path: str,
    report: NormalizationReport,
    apply_defaults: bool = True,
) -> dict:
    """Normalize one object against its class profile, recursing into children."""
    if not isinstance(instance, dict):
        return instance

    slots_by_name = {slot.name: slot for slot in class_profile.slots}

    # Drop slots the schema does not define; they would fail validation and
    # usually mean the model invented a field.
    for key in list(instance):
        if key not in slots_by_name:
            report.dropped.append(
                {"path": f"{path}.{key}", "value": "<unknown slot>", "enum": None}
            )
            instance.pop(key)

    for name, slot in slots_by_name.items():
        if slot.is_enum:
            normalize_enum_slot(instance, slot, path, report)
        elif slot.inlined_class and slot.inlined_class in profile.classes:
            _normalize_child(instance, slot, profile, path, report, apply_defaults)

        if (
            apply_defaults
            and slot.default is not None
            and name not in instance
            and slot.is_enum
        ):
            instance[name] = slot.default
            report.defaults_applied[f"{class_profile.name}.{name}"] += 1

    return instance


def _normalize_child(
    instance: dict,
    slot: SlotProfile,
    profile: ExtractionProfile,
    path: str,
    report: NormalizationReport,
    apply_defaults: bool,
) -> None:
    """Recurse into an inlined child object or list of them.

    Endpoint slots (``subject``/``object``) have a class range but hold plain
    id references, so a bare string is left exactly as it is.
    """
    value = instance.get(slot.name)
    if value is None:
        return
    child_profile = profile.classes[slot.inlined_class]

    if slot.multivalued:
        if not isinstance(value, list):
            value = [value]
        instance[slot.name] = [
            normalize_instance(
                item, child_profile, profile,
                f"{path}.{slot.name}[{index}]", report, apply_defaults,
            )
            if isinstance(item, dict) else item
            for index, item in enumerate(value)
        ]
        return

    if isinstance(value, dict):
        instance[slot.name] = normalize_instance(
            value, child_profile, profile, f"{path}.{slot.name}", report, apply_defaults
        )


def stable_id(prefix: str, *parts: Any) -> str:
    """Deterministic id from content, so re-running produces the same graph."""
    digest = hashlib.sha256(
        "|".join("" if part is None else str(part) for part in parts).encode("utf-8")
    ).hexdigest()[:12]
    return f"{prefix}{digest}"


def normalize_graph(
    graph: dict,
    profile: ExtractionProfile,
    source_document: Optional[dict] = None,
    apply_defaults: bool = True,
) -> tuple[dict, NormalizationReport]:
    """Normalize a raw extraction into a schema-shaped CAMO graph.

    Assigns deterministic ids where the model omitted them, coerces enums,
    drops unknown slots, and wires edge endpoints to the node ids actually
    present so the result passes referential-integrity checks.
    """
    report = NormalizationReport()
    nodes = [node for node in (graph.get("nodes") or []) if isinstance(node, dict)]
    edges = [edge for edge in (graph.get("edges") or []) if isinstance(edge, dict)]

    node_profile = profile.get("CausalNode")
    edge_profile = profile.get("CausalEdge")

    # Nodes first: edges are rewired against the ids this pass settles on.
    id_map: dict[str, str] = {}
    for index, node in enumerate(nodes):
        original_id = node.get("id")
        normalize_instance(
            node, node_profile, profile, f"nodes[{index}]", report, apply_defaults
        )
        if not node.get("id"):
            node["id"] = stable_id(
                "camo:node_",
                node.get("entity_term"),
                node.get("measured_attribute"),
                node.get("state_or_change_qualifier"),
                node.get("name"),
            )
            report.generated_ids += 1
        if original_id:
            id_map[str(original_id)] = node["id"]
        id_map.setdefault(node["id"], node["id"])
        if node.get("name") and node.get("id"):
            id_map.setdefault(str(node["name"]), node["id"])

    known_ids = {node["id"] for node in nodes}
    kept_edges = []
    for index, edge in enumerate(edges):
        normalize_instance(
            edge, edge_profile, profile, f"edges[{index}]", report, apply_defaults
        )
        subject = _resolve(edge.get("subject"), id_map)
        obj = _resolve(edge.get("object"), id_map)
        if subject not in known_ids or obj not in known_ids:
            report.dropped.append(
                {
                    "path": f"edges[{index}]",
                    "value": f"endpoint(s) not found: subject={edge.get('subject')!r} "
                    f"object={edge.get('object')!r}",
                    "enum": None,
                }
            )
            continue
        edge["subject"], edge["object"] = subject, obj

        for structure, key in (
            ("mediation", "mediator_node_ids"),
            ("moderation", "moderator_node_ids"),
        ):
            block = edge.get(structure)
            if isinstance(block, dict) and block.get(key):
                resolved = [_resolve(ref, id_map) for ref in block[key]]
                block[key] = [ref for ref in resolved if ref in known_ids]

        comparator = edge.get("comparator")
        if isinstance(comparator, dict) and comparator.get("comparator_node_id"):
            resolved = _resolve(comparator["comparator_node_id"], id_map)
            if resolved in known_ids:
                comparator["comparator_node_id"] = resolved
            else:
                comparator.pop("comparator_node_id")

        if not edge.get("id"):
            edge["id"] = stable_id(
                "camo:edge_",
                edge.get("subject"),
                edge.get("predicate"),
                edge.get("object"),
                edge.get("original_sentence"),
            )
            report.generated_ids += 1
        kept_edges.append(edge)

    normalized: dict[str, Any] = {
        "graph_id": graph.get("graph_id")
        or stable_id("camo:graph_", (source_document or {}).get("document_id"), len(nodes)),
        "schema_version": profile.version,
        "provenance": graph.get("provenance") or {},
        "nodes": nodes,
        "edges": kept_edges,
    }
    if source_document:
        normalized["source_documents"] = [source_document]
        document_id = source_document.get("document_id")
        for edge in kept_edges:
            edge.setdefault("source_document", document_id)
    return normalized, report


def _resolve(value: Any, id_map: dict[str, str]) -> Optional[str]:
    if isinstance(value, dict):
        value = value.get("id")
    if value is None:
        return None
    return id_map.get(str(value), str(value))
