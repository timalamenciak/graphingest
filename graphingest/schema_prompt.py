"""Build extraction prompts from a LinkML schema.

The prompt is *rendered from the schema*, not hand-written, so bumping the
schema propagates on the next run with no edit. Nothing here knows the name of
any particular schema: point the pipeline at a different LinkML file and the
prompt, the output constraint, the normalizer and the validator retarget
together.

CAMO carries its own extraction guidance inside enum ``annotations``:
``linguistic_cues`` for philosophical accounts, ``exemplars`` for claim
strengths and comparator types, ``canonical_question`` for each account. Those
were written to instruct a human annotator; they work just as well on a model,
and they are the schema author's intent rather than ours.

What this prompt deliberately does **not** do is ask for ontology identifiers.
Several slots are described in the schema as holding "an ontology CURIE", and a
model asked for one will cheerfully produce ``ENVO:00002006`` for anything at
all — a plausible-looking identifier that resolves to the wrong term, or to
nothing. Those slots are re-described here as plain-language terms, and
``graphingest.ground`` resolves them against real ontologies afterwards, where
a miss is a recorded miss instead of a fabrication.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .schema import ClassProfile, ExtractionProfile, SlotProfile

#: Enums larger than this are listed compactly rather than inline with
#: per-value descriptions, to keep the prompt from being swamped by the
#: 110-value IUCN ecosystem typology.
LARGE_ENUM_THRESHOLD = 40

#: Annotation keys worth surfacing to the model, in the order they read best.
_HINT_KEYS = ("canonical_question", "linguistic_cues", "exemplars")

#: Slots that carry machine bookkeeping rather than anything readable from the
#: text. Asking a model to fill these produces confident nonsense.
_SUPPRESSED_SLOTS = {
    "embedding_text",
    "embedding_vector",
    "variable_key",
    "fcm_weight",
    "fcm_weight_source",
    "aggregate_fcm_weight",
    "aggregate_fcm_weight_source",
    "annotation_timestamp",
    # Set from the run's own settings after extraction; a model asked who
    # annotated something will answer with a plausible ORCID.
    "annotator",
    "exporter_version",
    "export_sha256",
    "exported_at",
    "ontology_snapshot_id",
    "start_char",
    "end_char",
    "sentence_id",
    "paragraph_id",
}

#: Used when no ``--domain`` is given. Deliberately broad: the schema, not this
#: sentence, is what tells the model what kind of claims to look for.
DEFAULT_DOMAIN = "scientific"

#: Appended to any slot whose schema description asks for an identifier. The
#: schema is written for an annotation tool that looks terms up; this prompt is
#: not that tool, and asking a model for an identifier invites invention.
_PLAIN_TERM_NOTE = (
    "Write a plain-language term here -- the common or scientific name of the "
    "thing itself, e.g. \"Aedes dorsalis\" or \"soil salinity\". Do NOT write an "
    "identifier, CURIE, accession or URL of any kind; identifiers are assigned "
    "afterwards by a lookup step, and one you invent here would be wrong."
)

#: Matches the identifier vocabulary a schema uses when a slot wants a term id.
_IDENTIFIER_WORDS = ("curie", "ontology term", "accession", "identifier", "uri")

_SYSTEM_TEMPLATE = """\
You are a careful evidence-synthesis annotator working on {domain} literature. \
You extract causal claims from articles into a strict labelled property graph \
whose vocabulary is grounded in published ontologies.

Four rules govern everything you do:

1. ONLY WHAT THE TEXT SAYS. Never infer a claim the authors did not make. If a \
paper reports a correlation, do not record causation. If it hedges, record the \
hedge. A null result is a finding, not an absence -- record it.

2. NODES ARE STATES OR CHANGES IN STATE, NOT THINGS. An entity does not cause \
anything by itself; the causally relevant unit is a state or change in an \
attribute of that entity. Every node decomposes into entity (what thing) + \
measured attribute (what property) + qualifier (whether or how it changed). \
"Increased abundance of Canis lupus", not "wolves".

3. GROUND EVERY CLAIM IN THE TEXT. Each node and edge must quote the verbatim \
sentence it came from. Do not paraphrase quotes.

4. USE THE VOCABULARY, AND PLAIN WORDS EVERYWHERE ELSE. Where a slot lists \
permissible values, choose the one the authors' own wording supports, and omit \
the slot when none of them does. Everywhere else write plain language, never an \
identifier, CURIE or accession number: ontology identifiers are looked up \
afterwards by software, and any you invent would be wrong.

Return a single JSON object and nothing else."""

#: Back-compatible default rendering, for callers that want no domain wording.
SYSTEM_PROMPT = _SYSTEM_TEMPLATE.format(domain=DEFAULT_DOMAIN)


def build_system_prompt(domain: Optional[str] = None) -> str:
    """The annotator system prompt, optionally narrowed to a field of study.

    ``--domain "restoration ecology"`` reads better to a model than the generic
    wording and costs nothing; leaving it unset is not a downgrade, because the
    rules and the vocabulary both come from the schema either way.
    """
    return _SYSTEM_TEMPLATE.format(domain=(domain or DEFAULT_DOMAIN).strip())


def wants_identifier(slot: SlotProfile) -> bool:
    """True when the schema describes this slot as holding a *term* identifier.

    Detected from the description rather than listed by name, so a schema that
    adds another CURIE-valued slot is covered without a code change. The range
    is no help here — CAMO gives ``id`` and ``entity_term`` the same
    ``uriorcurie`` range — so record identifiers are excluded by LinkML's own
    ``identifier: true`` flag, which is what actually distinguishes the slot
    that names a record from a slot naming a term in a vocabulary.
    """
    if slot.is_enum or slot.inlined_class or slot.identifier:
        return False
    description = (slot.description or "").lower()
    return any(word in description for word in _IDENTIFIER_WORDS)


def _slot_line(slot: SlotProfile, indent: str = "  ") -> list[str]:
    """Render one slot as prompt lines, including its enum vocabulary."""
    flags = []
    if slot.required:
        flags.append("REQUIRED")
    if slot.multivalued:
        flags.append("list")
    if slot.default:
        flags.append(f"default={slot.default}")
    suffix = f"  [{', '.join(flags)}]" if flags else ""

    lines = [f"{indent}- {slot.name}{suffix}"]
    if slot.description:
        lines.append(f"{indent}    {_trim(slot.description, 260)}")
    if wants_identifier(slot):
        lines.append(f"{indent}    {_PLAIN_TERM_NOTE}")

    if slot.is_enum and len(slot.enum_values) > LARGE_ENUM_THRESHOLD:
        # Rendering 110 ecosystem classes with descriptions here would swamp
        # the prompt; they are listed compactly in their own section instead.
        lines.append(
            f"{indent}    ({len(slot.enum_values)} permissible values -- see the "
            f"'Permissible values for ...{slot.name}' section below)"
        )
    elif slot.is_enum:
        for detail in slot.enum_values:
            hint = _hint(detail)
            description = _trim(detail.get("description") or "", 130)
            text = f"{indent}    * {detail['name']}"
            if description:
                text += f" -- {description}"
            lines.append(text)
            if hint:
                lines.append(f"{indent}        {hint}")
    elif slot.inlined_class:
        lines.append(f"{indent}    (object of type {slot.inlined_class})")
    return lines


def _hint(detail: dict) -> str:
    """Pull the schema's own annotator guidance out of an enum value."""
    annotations = detail.get("annotations") or {}
    parts = [
        f"{key.replace('_', ' ')}: {_trim(str(annotations[key]), 190)}"
        for key in _HINT_KEYS
        if annotations.get(key)
    ]
    return " | ".join(parts)


def _trim(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render_class(
    class_profile: ClassProfile,
    include: Optional[Iterable[str]] = None,
    enum_detail: bool = True,
) -> str:
    """Render a class and its slots as a prompt section."""
    wanted = set(include) if include is not None else None
    lines = [f"## {class_profile.name}"]
    if class_profile.description:
        lines.append(_trim(class_profile.description, 420))
    lines.append("")
    for slot in class_profile.slots:
        if slot.name in _SUPPRESSED_SLOTS:
            continue
        if wanted is not None and slot.name not in wanted:
            continue
        if not enum_detail and slot.is_enum:
            permitted = ", ".join(detail["name"] for detail in slot.enum_values)
            flags = " [REQUIRED]" if slot.required else ""
            lines.append(f"  - {slot.name}{flags}: one of {permitted}")
            continue
        lines.extend(_slot_line(slot))
    return "\n".join(lines)


def build_extraction_prompt(
    profile: ExtractionProfile,
    text: str,
    source_document: Optional[dict] = None,
    classes: Iterable[str] = ("CausalNode", "CausalEdge"),
    enum_detail: bool = True,
    examples: Optional[str] = None,
) -> str:
    """Build the user prompt for extracting a graph from ``text``."""
    sections: list[str] = [
        f"# Target schema: {profile.schema_name or profile.schema_path.stem} "
        f"v{profile.version}",
        "",
        "Populate the classes below. Slot names and enum values must be used "
        "exactly as written -- do not invent slots or values. Everything that "
        "is not an enum value is plain language: never an identifier.",
        "",
    ]
    for class_name in classes:
        sections.append(render_class(profile.get(class_name), enum_detail=enum_detail))
        sections.append("")

    # A handful of large enums exist purely for context tagging. Listing every
    # permissible value inline would dominate the prompt, so they are named
    # compactly at the end.
    for class_name in classes:
        for slot in profile.get(class_name).slots:
            if slot.is_enum and len(slot.enum_values) > LARGE_ENUM_THRESHOLD:
                sections.append(
                    f"### Permissible values for {class_name}.{slot.name} "
                    f"({len(slot.enum_values)} values)"
                )
                sections.append(
                    ", ".join(detail["name"] for detail in slot.enum_values)
                )
                sections.append("")

    if examples:
        sections.extend(
            [
                "# Worked example",
                "A hand-checked annotation of a different article, in the exact "
                "shape your answer must take. Follow its conventions; do not "
                "reuse its content.",
                "",
                examples,
                "",
            ]
        )

    if source_document:
        sections.extend(
            [
                "# Article metadata",
                "\n".join(
                    f"{key}: {_metadata_value(value)}"
                    for key, value in source_document.items()
                    if value not in (None, "", [])
                ),
                "",
            ]
        )

    sections.extend(
        [
            "# Article text",
            text,
            "",
            "# Task",
            "Extract every causal claim the article makes into a JSON object with "
            "two keys: `nodes` (a list of CausalNode objects) and `edges` (a list "
            "of CausalEdge objects).",
            "",
            "- Give each node a short stable `id` and reference it from an edge's "
            "`subject` and `object` by that exact id.",
            "- Put the verbatim source sentence in each edge's `original_sentence`, "
            "and quote supporting text in `source_spans[].text`.",
            "- Set `claim_strength` from the language the authors actually use, not "
            "from how convincing you find the result.",
            "- Include null results: an effect the authors tested and did not find "
            "is an edge with claim_strength `no_relationship` and `negated: true`.",
            "- Write every term in plain language. `entity_term` is the thing "
            "itself as the authors name it (\"Aedes dorsalis\", \"tidal "
            "flushing\", \"soil salinity\"); a later step looks those up in "
            "real ontologies, so an identifier here can only be wrong.",
            "- Omit any slot the text does not support. Omitting is always better "
            "than guessing.",
            "",
            "Return only the JSON object.",
        ]
    )
    return "\n".join(sections)


def _metadata_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    return str(value)


def _collect_refs(node: Any, found: set[str]) -> None:
    """Walk a JSON Schema fragment collecting ``$ref`` definition names."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and "/" in ref:
            found.add(ref.rsplit("/", 1)[-1])
        for value in node.values():
            _collect_refs(value, found)
    elif isinstance(node, list):
        for value in node:
            _collect_refs(value, found)


def _relax_large_enums(node: Any, limit: int) -> Any:
    """Replace oversized ``enum`` lists with a plain string constraint.

    A 110-value enum makes a server-side grammar enormous for no benefit: the
    prompt already lists the permissible values, and normalization plus final
    validation check them against the real schema either way. Relaxing here
    only affects what the *decoder* is constrained to.
    """
    if isinstance(node, dict):
        relaxed = {}
        for key, value in node.items():
            if key == "enum" and isinstance(value, list) and len(value) > limit:
                continue
            relaxed[key] = _relax_large_enums(value, limit)
        if "enum" not in relaxed and isinstance(node.get("enum"), list):
            relaxed.setdefault("type", "string")
        return relaxed
    if isinstance(node, list):
        return [_relax_large_enums(value, limit) for value in node]
    return node


def build_extraction_json_schema(
    profile: ExtractionProfile,
    compact: bool = True,
    enum_limit: int = LARGE_ENUM_THRESHOLD,
) -> dict:
    """A ``{nodes, edges}`` JSON Schema for constraining model output.

    The full graph JSON Schema demands graph-level slots (`graph_id`,
    `provenance`) that the model has no basis to invent and that normalization
    supplies afterwards, so extraction is constrained to just the two lists.

    With ``compact`` (the default) the result is additionally pruned to the
    definitions actually reachable from CausalNode and CausalEdge, and large
    enums are relaxed. This matters in practice: the unpruned CAMO schema is
    ~65KB across 44 definitions, and Ollama returns a 500 rather than compile a
    grammar that big. Correctness is unaffected -- the graph is still validated
    against the complete schema after extraction.
    """
    from .schema import json_schema_for

    full = json_schema_for(profile.schema_path, profile.top_class)
    defs_key = "$defs" if "$defs" in full else "definitions"
    definitions = full.get(defs_key, {})

    roots = {
        "nodes": definitions.get("CausalNode", {"type": "object"}),
        "edges": definitions.get("CausalEdge", {"type": "object"}),
    }

    if compact:
        needed: set[str] = set()
        _collect_refs(roots, needed)
        # Definitions can reference each other, so close over them.
        frontier = set(needed)
        while frontier:
            current = frontier.pop()
            body = definitions.get(current)
            if body is None:
                continue
            discovered: set[str] = set()
            _collect_refs(body, discovered)
            new = discovered - needed
            needed |= new
            frontier |= new
        definitions = {
            name: body for name, body in definitions.items() if name in needed
        }
        roots = _relax_large_enums(roots, enum_limit)
        definitions = _relax_large_enums(definitions, enum_limit)

    schema = {
        "type": "object",
        "properties": {
            "nodes": {"type": "array", "items": roots["nodes"]},
            "edges": {"type": "array", "items": roots["edges"]},
        },
        "required": ["nodes", "edges"],
    }
    if definitions:
        schema[defs_key] = definitions
    return schema
