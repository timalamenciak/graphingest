"""LinkML schema introspection, JSON-Schema generation, and extraction profiles.

This module is what makes the extractor *schema-driven* rather than
CAMO-specific: point it at any LinkML schema and it derives both the JSON
Schema used to constrain model output and the prose profile used to build the
extraction prompt. Bumping CAMO from 0.7.9 to 0.8.x should require no code
change here.

CAMO carries extraction hints inside its own enum ``annotations``
(``linguistic_cues``, ``exemplars``, ``canonical_question``). Those are
surfaced in the profile deliberately: they are the schema author's guidance to
an annotator, and they work just as well as guidance to a model.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

LOGGER = logging.getLogger("camo.schema")

#: Vendored canonical CAMO schema shipped with the toolchain.
DEFAULT_SCHEMA = Path(__file__).resolve().parent.parent / "schema" / "causalmosaic.yaml"


# ---------------------------------------------------------------------------
# Raw schema access
# ---------------------------------------------------------------------------


def load_schema(path: str | Path | None = None) -> dict:
    """Load a LinkML schema YAML as a plain dict."""
    resolved = Path(path or DEFAULT_SCHEMA)
    if not resolved.exists():
        raise FileNotFoundError(f"LinkML schema not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        schema = yaml.safe_load(handle) or {}
    if "classes" not in schema:
        raise ValueError(f"{resolved} does not look like a LinkML schema (no classes)")
    return schema


def schema_version(schema: dict) -> str:
    return str(schema.get("version", "unknown"))


def get_classes(schema: dict) -> list[str]:
    return list((schema.get("classes") or {}).keys())


def get_enums(schema: dict) -> list[str]:
    return list((schema.get("enums") or {}).keys())


def get_class_attributes(schema: dict, class_name: str) -> dict[str, dict]:
    """Attributes of a class, including any inherited via ``is_a``.

    LinkML also allows top-level ``slots`` referenced by ``slots:`` on a class;
    both forms are merged here so callers do not have to care which style a
    schema uses.
    """
    classes = schema.get("classes") or {}
    definition = classes.get(class_name)
    if definition is None:
        raise KeyError(f"Class {class_name!r} not in schema")

    merged: dict[str, dict] = {}
    parent = definition.get("is_a")
    if parent and parent in classes:
        merged.update(get_class_attributes(schema, parent))

    top_level_slots = schema.get("slots") or {}
    for slot_name in definition.get("slots") or []:
        if slot_name in top_level_slots:
            merged[slot_name] = dict(top_level_slots[slot_name])

    for name, body in (definition.get("attributes") or {}).items():
        merged[name] = dict(body or {})
    return merged


def get_required_attributes(schema: dict, class_name: str) -> list[str]:
    return [
        name
        for name, body in get_class_attributes(schema, class_name).items()
        if body.get("required")
    ]


def get_multivalued_attributes(schema: dict, class_name: str) -> list[str]:
    return [
        name
        for name, body in get_class_attributes(schema, class_name).items()
        if body.get("multivalued")
    ]


def get_attribute_range(schema: dict, class_name: str, attribute: str) -> Optional[str]:
    """Range of an attribute, resolving ``any_of`` to its first concrete range."""
    body = get_class_attributes(schema, class_name).get(attribute) or {}
    if body.get("range"):
        return body["range"]
    for option in body.get("any_of") or []:
        if isinstance(option, dict) and option.get("range"):
            return option["range"]
    return None


def get_enum_values(schema: dict, enum_name: str) -> list[str]:
    """Permissible value names for an enum.

    LinkML spells these ``permissible_values``. An earlier helper in
    camo_microreview read ``values`` instead and silently returned an empty
    list, which made every enum look unconstrained; both spellings are accepted
    here so that bug cannot recur.
    """
    definition = (schema.get("enums") or {}).get(enum_name)
    if definition is None:
        raise KeyError(f"Enum {enum_name!r} not in schema")
    values = definition.get("permissible_values")
    if values is None:
        values = definition.get("values") or []
    if isinstance(values, dict):
        return list(values.keys())
    return [item["name"] if isinstance(item, dict) else str(item) for item in values]


def get_enum_details(schema: dict, enum_name: str) -> list[dict]:
    """Permissible values with description and annotations, in schema order."""
    definition = (schema.get("enums") or {}).get(enum_name) or {}
    values = definition.get("permissible_values") or definition.get("values") or {}
    if not isinstance(values, dict):
        values = {
            (item["name"] if isinstance(item, dict) else str(item)): (
                item if isinstance(item, dict) else {}
            )
            for item in values
        }
    details = []
    for name, body in values.items():
        body = body or {}
        details.append(
            {
                "name": name,
                "description": _flatten(body.get("description")),
                "meaning": body.get("meaning"),
                "annotations": {
                    key: _flatten(value)
                    for key, value in (body.get("annotations") or {}).items()
                },
            }
        )
    return details


def validate_required_classes(schema: dict, required: Iterable[str]) -> None:
    """Raise if any of ``required`` is absent, naming all of them at once."""
    available = set(get_classes(schema))
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"Missing required classes in schema: {missing}")


def _flatten(value: Any) -> Any:
    """Collapse LinkML folded scalars to single-line strings."""
    if isinstance(value, str):
        return " ".join(value.split())
    return value


# ---------------------------------------------------------------------------
# JSON Schema generation
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _generate_json_schema(path_str: str, top_class: Optional[str]) -> dict:
    from linkml.generators.jsonschemagen import JsonSchemaGenerator

    generator = JsonSchemaGenerator(
        path_str,
        top_class=top_class,
        not_closed=True,
        include_range_class_descendants=True,
    )
    return generator.generate()


def json_schema_for(
    schema_path: str | Path | None = None, top_class: str = "CausalGraph"
) -> dict:
    """Generate a JSON Schema from a LinkML schema, rooted at ``top_class``.

    Results are cached per (path, class) because generation is slow enough to
    notice when extracting hundreds of articles.
    """
    resolved = Path(schema_path or DEFAULT_SCHEMA)
    return _generate_json_schema(str(resolved), top_class)


# ---------------------------------------------------------------------------
# Extraction profile
# ---------------------------------------------------------------------------


@dataclass
class SlotProfile:
    """One slot, flattened into what a prompt builder needs."""

    name: str
    range: Optional[str]
    required: bool
    multivalued: bool
    description: str
    enum_values: list[dict] = field(default_factory=list)
    inlined_class: Optional[str] = None
    default: Optional[str] = None
    #: LinkML ``identifier: true`` — the slot that names the record, as opposed
    #: to a vocabulary term. The pipeline assigns these; nothing asks for them.
    identifier: bool = False
    #: LinkML ``pattern``: a regex the value must match. Worth carrying because
    #: a slot the pipeline fills in itself has to satisfy it.
    pattern: Optional[str] = None

    @property
    def is_enum(self) -> bool:
        return bool(self.enum_values)


@dataclass
class ClassProfile:
    name: str
    description: str
    slots: list[SlotProfile]

    @property
    def required_slots(self) -> list[SlotProfile]:
        return [slot for slot in self.slots if slot.required]


@dataclass
class ExtractionProfile:
    """A schema reduced to what an extraction prompt and validator need."""

    schema_path: Path
    version: str
    top_class: str
    classes: dict[str, ClassProfile]
    #: The schema's own name/id, recorded in graph provenance.
    schema_name: str = ""
    schema_id: str = ""
    #: CURIE prefix map from the LinkML schema, shown to the model so the
    #: ontology terms behind each enum value are legible rather than opaque.
    prefixes: dict[str, str] = field(default_factory=dict)

    def get(self, class_name: str) -> ClassProfile:
        if class_name not in self.classes:
            raise KeyError(
                f"Class {class_name!r} not in profile; have {sorted(self.classes)}"
            )
        return self.classes[class_name]

    @property
    def class_names(self) -> list[str]:
        return list(self.classes)


def build_extraction_profile(
    schema_path: str | Path | None = None,
    classes: Optional[Iterable[str]] = None,
    top_class: str = "CausalGraph",
) -> ExtractionProfile:
    """Reduce a LinkML schema to the classes an extractor needs to populate.

    ``classes`` defaults to the transitive closure of classes reachable from
    ``top_class``, so adding a new inlined class to the schema pulls it into the
    profile automatically.
    """
    resolved = Path(schema_path or DEFAULT_SCHEMA)
    schema = load_schema(resolved)
    wanted = (
        list(classes)
        if classes is not None
        else _reachable_classes(schema, top_class)
    )
    validate_required_classes(schema, wanted)

    built: dict[str, ClassProfile] = {}
    for class_name in wanted:
        definition = (schema.get("classes") or {})[class_name]
        slots = []
        for slot_name, body in get_class_attributes(schema, class_name).items():
            slot_range = get_attribute_range(schema, class_name, slot_name)
            enum_values = (
                get_enum_details(schema, slot_range)
                if slot_range in (schema.get("enums") or {})
                else []
            )
            inlined = (
                slot_range
                if slot_range in (schema.get("classes") or {})
                else None
            )
            slots.append(
                SlotProfile(
                    name=slot_name,
                    range=slot_range,
                    required=bool(body.get("required")),
                    multivalued=bool(body.get("multivalued")),
                    description=_flatten(body.get("description")) or "",
                    enum_values=enum_values,
                    inlined_class=inlined,
                    default=_ifabsent(body.get("ifabsent")),
                    identifier=bool(body.get("identifier")),
                    pattern=body.get("pattern"),
                )
            )
        built[class_name] = ClassProfile(
            name=class_name,
            description=_flatten(definition.get("description")) or "",
            slots=slots,
        )

    return ExtractionProfile(
        schema_path=resolved,
        version=schema_version(schema),
        top_class=top_class,
        classes=built,
        schema_name=str(schema.get("name") or resolved.stem),
        schema_id=str(schema.get("id") or ""),
        prefixes=_prefix_map(schema),
    )


def _prefix_map(schema: dict) -> dict[str, str]:
    """Flatten LinkML ``prefixes``, which allows both a bare string and a map."""
    flattened: dict[str, str] = {}
    for name, body in (schema.get("prefixes") or {}).items():
        if isinstance(body, dict):
            reference = body.get("prefix_reference")
            if reference:
                flattened[name] = str(reference)
        elif body:
            flattened[name] = str(body)
    return flattened


def _ifabsent(raw: Any) -> Optional[str]:
    """Unwrap LinkML ``ifabsent`` forms like ``string(not_addressed)``."""
    if not isinstance(raw, str):
        return None
    if "(" in raw and raw.endswith(")"):
        return raw[raw.index("(") + 1 : -1]
    return raw


def _reachable_classes(schema: dict, top_class: str) -> list[str]:
    """Breadth-first closure of classes reachable from ``top_class`` via ranges."""
    classes = schema.get("classes") or {}
    if top_class not in classes:
        raise KeyError(f"Top class {top_class!r} not in schema")
    ordered: list[str] = []
    queue = [top_class]
    seen = {top_class}
    while queue:
        current = queue.pop(0)
        ordered.append(current)
        for slot_name in get_class_attributes(schema, current):
            slot_range = get_attribute_range(schema, current, slot_name)
            if slot_range in classes and slot_range not in seen:
                seen.add(slot_range)
                queue.append(slot_range)
    return ordered
