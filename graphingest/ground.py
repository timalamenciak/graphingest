"""Ground free-text terms against real ontologies, in Python.

    python -m graphingest.ground build/graphs --cache build/grounding_cache.json
    python -m graphingest.ground build/causal_graph.yaml --report-only
    python -m graphingest.ground build/graphs --backend none      # skip lookups

The model is never asked for an ontology identifier. Asked for one it will
produce something shaped exactly like a real CURIE — ``ENVO:00002006``,
``Q56987`` — that denotes the wrong thing, or nothing at all, and there is no
way to tell from the string which happened. So extraction asks for plain
language ("Aedes dorsalis", "soil salinity", "tidal flushing") and this module
looks those terms up in the ontologies themselves.

The difference that matters is not accuracy but *failure behaviour*. A lookup
that finds nothing records a miss, the term stays as the authors wrote it, and
the miss is counted in the report. A hallucinated identifier is indistinguishable
from a correct one until somebody dereferences it.

Routing is by entity type, because the right ontology depends on what kind of
thing the node is: taxa go to Wikidata (which is what CAMO's own
``AppliedToEntity`` asks for), environments and processes to ELMO and then
ENVO/GO, attributes to PATO. A route is a *list* of backends tried in order, so
a project's own ontology can be consulted before the public ones — which is the
point of having one. All of that is configuration, not code; see the
``grounding:`` block of ``config/pipeline.yaml``.

Three backends, for three situations. ``ols`` reaches the EBI Ontology Lookup
Service, which hosts the published ontologies. ``wikidata`` covers taxa.
``local`` loads an ontology file directly — a URL or a path — which is the only
way to ground against an ontology that is not published to a lookup service,
and the only way to ground at all on a machine with no outbound network.

Every lookup is cached to disk by (backend, ontologies, term), so a corpus that
mentions *Aedes dorsalis* in nine papers costs one request, and a re-run costs
none.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .cli import configure_logging, configure_stdio
from .config import DEFAULT_PIPELINE_CONFIG, load_yaml
from .graph_io import atomic_write_json, load_graph, save_graph
from .schema import DEFAULT_SCHEMA, ExtractionProfile, build_extraction_profile
from .schema_prompt import wants_identifier

LOGGER = logging.getLogger("ingest.ground")

OLS_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

#: Wikidata "taxon". A search hit that is not one is not a species, whatever
#: its label says — "Culex" is also a surname.
WIKIDATA_TAXON = "Q16521"
#: ``taxon name``: present on taxon items, and a cheaper signal than P31 alone.
WIKIDATA_TAXON_NAME = "P225"

#: A score at or above this is an outright name match, not a near one, and
#: ends the search: no later backend can beat it, so none is asked.
EXACT_MATCH = 0.999

#: Anything shaped like an identifier: ``ENVO:00002006``, ``Q30019``, a URL.
IDENTIFIER_PATTERN = re.compile(
    r"^(?:[A-Za-z][A-Za-z0-9_.]{0,30}:[A-Za-z0-9_.\-/#]+|[QP]\d+|https?://\S+)$"
)

#: Sent on every request. Wikidata asks for a descriptive agent, and being
#: identifiable is the price of using a free public service politely.
USER_AGENT = "graphingest/0.1.0 (causal graph ingest; ontology term lookup)"

#: Defaults used when config/pipeline.yaml has no ``grounding:`` block.
DEFAULT_GROUNDING = {
    "enabled": True,
    "min_score": 0.6,
    "timeout": 20,
    "pause_seconds": 0.1,
    "routes": {
        "taxon": "wikidata",
        "environmental_variable": ["local:elmo", "ols:envo,pato,chebi"],
        "environmental_process": ["local:elmo", "ols:envo,go"],
        "management_intervention": ["local:elmo", "ols:envo,go"],
        "default": ["local:elmo", "ols:envo,go,chebi,pato"],
    },
    # Ontology files loaded directly, by the name a "local:<name>" route uses.
    "ontologies": {
        "elmo": {
            "source": "https://raw.githubusercontent.com/timalamenciak/elmo"
                      "/refs/heads/main/elmo.owl",
            "prefix": "elmo",
        }
    },
    #: Where downloaded ontologies and their term indexes are kept.
    "ontology_dir": None,
    # Slots to ground, beyond the entity term. measured_attribute is off by
    # default: hand annotation keeps it as measured prose ("bioavailable
    # phosphorus concentration (< 0.03 mg/kg)"), which no ontology term carries.
    "slots": ["entity_term"],
    # Retry a phrase that resolves to nothing with its trailing words, which is
    # where English puts the head noun. Recorded distinctly and held to a
    # higher bar, because it discards part of what the authors wrote.
    "backoff": True,
    "backoff_penalty": 0.15,
}


#: Terms whose grounding is a judgement call rather than a lookup, kept beside
#: the config so each one can carry its reasoning. See load_overrides.
DEFAULT_OVERRIDES_PATH = DEFAULT_PIPELINE_CONFIG.parent / "grounding_overrides.yaml"

#: Where ontology files loaded by the ``local`` backend are kept, alongside the
#: term index built from each. Both are derived artefacts: delete and re-run.
DEFAULT_ONTOLOGY_DIR = DEFAULT_PIPELINE_CONFIG.parent.parent / "ontologies"


class GroundingError(RuntimeError):
    """Raised when a backend is misconfigured. Network failures are not fatal."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class Match:
    """One resolved term."""

    query: str
    curie: str
    label: str
    ontology: str
    backend: str
    score: float
    #: How this match was reached: "lookup", "backoff:<query>", or "override".
    via: str = "lookup"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroundingReport:
    grounded: int = 0
    unresolved: int = 0
    already_identifier: int = 0
    cache_hits: int = 0
    requests: int = 0
    errors: int = 0
    overridden: int = 0
    by_backoff: int = 0
    matches: list[dict] = field(default_factory=list)
    misses: list[dict] = field(default_factory=list)
    ontologies: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "overridden": self.overridden,
            "by_backoff": self.by_backoff,
            "unresolved": self.unresolved,
            "already_identifier": self.already_identifier,
            "cache_hits": self.cache_hits,
            "requests": self.requests,
            "errors": self.errors,
            "ontologies": dict(sorted(self.ontologies.items())),
            "matches": self.matches,
            "misses": self.misses,
        }

    def merge(self, other: "GroundingReport") -> None:
        self.overridden += other.overridden
        self.by_backoff += other.by_backoff
        self.grounded += other.grounded
        self.unresolved += other.unresolved
        self.already_identifier += other.already_identifier
        self.cache_hits += other.cache_hits
        self.requests += other.requests
        self.errors += other.errors
        self.matches.extend(other.matches)
        self.misses.extend(other.misses)
        for name, count in other.ontologies.items():
            self.ontologies[name] = self.ontologies.get(name, 0) + count


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


#: Function words carry no meaning in an ontology label, and counting them
#: makes "salinity of soil" look less like "soil salinity" than it is.
_STOPWORDS = frozenset(
    {"a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}
)


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", (value or "").lower())
        if token and token not in _STOPWORDS
    }


#: Generic head nouns an ontology appends to classify a term rather than name
#: it. ELMO ends 232 of its 652 class labels with "process", so "grubbing"
#: would otherwise miss "grubbing process" at 0.5 — the suffix is the ontology
#: saying what kind of thing it is, not part of what the authors called it.
_CLASSIFIER_SUFFIXES = (
    "process", "quality", "entity", "class", "role", "variable", "rate",
    "trait", "attribute", "characteristic",
)


def label_variants(label: str) -> list[str]:
    """A label, plus the same label without a trailing classifier noun."""
    variants = [label]
    words = (label or "").split()
    if len(words) > 1 and words[-1].lower() in _CLASSIFIER_SUFFIXES:
        variants.append(" ".join(words[:-1]))
    return variants


def best_similarity(query: str, labels: Iterable[str]) -> float:
    """The strongest agreement between the query and any of these labels."""
    return max(
        [0.0]
        + [
            similarity(query, variant)
            for label in labels
            if label
            for variant in label_variants(label)
        ]
    )


def similarity(query: str, label: str) -> float:
    """How much a candidate label agrees with the query, from 0 to 1.

    Token overlap rather than edit distance: "increased soil salinity" against
    "soil salinity" should score well, and "salinity of soil" identically,
    while "soil temperature" should not.
    """
    left, right = _tokens(query), _tokens(label)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return len(left & right) / len(left | right)


def looks_like_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(IDENTIFIER_PATTERN.match(value.strip()))


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _get_json(url: str, params: dict, timeout: int) -> dict:
    query = urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class OLSBackend:
    """EBI Ontology Lookup Service: ENVO, GO, CHEBI, PATO, NCBITaxon and more."""

    name = "ols"

    def __init__(self, timeout: int = 20, rows: int = 5):
        self.timeout = timeout
        self.rows = rows

    def search(self, term: str, ontologies: list[str]) -> list[Match]:
        params = {
            "q": term,
            "rows": self.rows,
            "fieldList": "iri,label,obo_id,short_form,ontology_name,synonym",
            "type": "class",
        }
        if ontologies:
            params["ontology"] = ",".join(ontologies)
        payload = _get_json(OLS_SEARCH, params, self.timeout)
        matches = []
        for doc in (payload.get("response") or {}).get("docs") or []:
            curie = doc.get("obo_id") or _curie_from_short_form(doc.get("short_form"))
            label = doc.get("label") or ""
            if not curie or not label:
                continue
            # Synonyms count: ENVO labels a saltmarsh "saline marsh" and lists
            # "salt marsh" as a synonym, and the authors write the synonym.
            # They are discounted slightly so an exact label wins a tie.
            by_label = best_similarity(term, [label])
            by_synonym = best_similarity(term, doc.get("synonym") or [])
            ontology = doc.get("ontology_name") or ""
            matches.append(
                Match(term, curie, label, ontology, self.name,
                      max(by_label, by_synonym * 0.98))
            )
        # Ties break on the order the route lists its ontologies: for ecology,
        # an ENVO term is a better answer than an equally-scoring CHEBI one.
        rank = {name.lower(): index for index, name in enumerate(ontologies)}
        return sorted(
            matches,
            key=lambda match: (-match.score, rank.get(match.ontology.lower(), 99)),
        )

    def label(self, curie: str) -> Optional[str]:
        """Reverse lookup, used to turn an identifier back into a term."""
        payload = _get_json(
            OLS_SEARCH,
            {"q": curie, "queryFields": "obo_id", "rows": 1, "fieldList": "label,obo_id"},
            self.timeout,
        )
        for doc in (payload.get("response") or {}).get("docs") or []:
            if (doc.get("obo_id") or "").lower() == curie.lower():
                return doc.get("label")
        return None


def _curie_from_short_form(short_form: Optional[str]) -> Optional[str]:
    """``ENVO_00002006`` -> ``ENVO:00002006``, for ontologies OLS leaves unprefixed."""
    if not short_form or "_" not in short_form:
        return None
    prefix, _, local = short_form.partition("_")
    return f"{prefix}:{local}"


class WikidataBackend:
    """Wikidata, filtered to actual taxa.

    CAMO asks for Wikidata QIDs for taxa, and its gold annotations use them.
    The filtering is the whole point: ``wbsearchentities`` for "Culex" returns
    the mosquito genus, but for "Parker" it returns a surname, and a graph that
    grounds an author to a person item is worse than one that grounds nothing.
    """

    name = "wikidata"

    def __init__(self, timeout: int = 20, limit: int = 5, taxa_only: bool = True):
        self.timeout = timeout
        self.limit = limit
        self.taxa_only = taxa_only

    def search(self, term: str, ontologies: list[str]) -> list[Match]:
        payload = _get_json(
            WIKIDATA_API,
            {
                "action": "wbsearchentities",
                "search": term,
                "language": "en",
                "uselang": "en",
                "format": "json",
                "type": "item",
                "limit": self.limit,
            },
            self.timeout,
        )
        candidates = [
            (hit["id"], hit.get("label") or "", hit.get("description") or "")
            for hit in payload.get("search") or []
            if hit.get("id")
        ]
        if not candidates:
            return []

        keep = candidates
        if self.taxa_only:
            taxa = self._taxon_ids([identifier for identifier, _, _ in candidates])
            keep = [item for item in candidates if item[0] in taxa]

        return sorted(
            (
                Match(term, identifier, label, "wikidata", self.name,
                      similarity(term, label))
                for identifier, label, _ in keep
            ),
            key=lambda match: -match.score,
        )

    def _taxon_ids(self, identifiers: list[str]) -> set[str]:
        """Which of these items are taxa, by P31=taxon or the presence of P225."""
        payload = _get_json(
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "ids": "|".join(identifiers[:20]),
                "props": "claims",
                "format": "json",
            },
            self.timeout,
        )
        taxa: set[str] = set()
        for identifier, entity in (payload.get("entities") or {}).items():
            claims = entity.get("claims") or {}
            if WIKIDATA_TAXON_NAME in claims:
                taxa.add(identifier)
                continue
            for statement in claims.get("P31") or []:
                value = (
                    ((statement.get("mainsnak") or {}).get("datavalue") or {}).get("value")
                    or {}
                )
                if isinstance(value, dict) and value.get("id") == WIKIDATA_TAXON:
                    taxa.add(identifier)
                    break
        return taxa

    def label(self, curie: str) -> Optional[str]:
        identifier = curie.split(":")[-1]
        payload = _get_json(
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "ids": identifier,
                "props": "labels",
                "languages": "en",
                "format": "json",
            },
            self.timeout,
        )
        entity = (payload.get("entities") or {}).get(identifier) or {}
        return ((entity.get("labels") or {}).get("en") or {}).get("value")


class LocalOntologyBackend:
    """An ontology file — ELMO, or any OWL/RDF the lookup services do not host.

    The file is fetched once into ``ontology_dir`` and reduced to a term index:
    CURIE, label, synonyms. Indexing rather than querying the graph each time
    matters because the alternative is re-parsing 9,000 triples per term; the
    index is rebuilt only when the source file's checksum changes.

    CURIEs are minted with the *schema's* prefix map, so an ELMO IRI becomes
    ``elmo:3622713`` — the same CURIE CAMO's own enums use — rather than a
    prefix this module invented, which would agree with nothing downstream.
    """

    name = "local"

    #: CURIE prefixes that name people and publications rather than terms. An
    #: ontology credits its authors with ORCIDs, and "Tim Alamenciak" is not a
    #: thing a causal claim is about.
    EXCLUDED_PREFIXES = ("orcid", "doi", "foaf", "dcterms", "dce")

    #: Predicates worth indexing besides rdfs:label. An ontology that lists
    #: "saltmarsh" as a synonym of "saline marsh" should match either.
    SYNONYM_PREDICATES = (
        "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
        "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
        "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
        "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym",
        "http://www.w3.org/2004/02/skos/core#altLabel",
        "http://www.w3.org/2004/02/skos/core#prefLabel",
    )

    def __init__(
        self,
        source: str,
        cache_dir: Path,
        prefix: str = "",
        prefixes: Optional[dict[str, str]] = None,
        ontology_id: str = "local",
        timeout: int = 30,
        refresh: bool = False,
    ):
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.prefix = prefix
        self.prefixes = prefixes or {}
        self.ontology_id = ontology_id
        self.timeout = timeout
        self.terms: list[dict] = []
        self._by_curie: dict[str, dict] = {}
        self._load(refresh)

    # -- index ----------------------------------------------------------

    def _load(self, refresh: bool) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        local = self._fetch(refresh)
        index_path = self.cache_dir / f"{self.ontology_id}.index.json"
        # The prefix map is part of the index's identity, not just the source:
        # the same OWL file minted against a different prefix map yields
        # different CURIEs, and a stale index would silently keep the old ones.
        checksum = f"{_checksum(local)}/{_digest(self.prefixes)}"

        if index_path.exists() and not refresh:
            try:
                cached = json.loads(index_path.read_text(encoding="utf-8"))
                if cached.get("checksum") == checksum:
                    self._adopt(cached["terms"])
                    LOGGER.info(
                        "Loaded %d %s term(s) from the cached index",
                        len(self.terms), self.ontology_id,
                    )
                    return
            except (OSError, ValueError, KeyError) as error:
                LOGGER.warning("Rebuilding %s index: %s", self.ontology_id, error)

        terms = self._build_index(local)
        atomic_write_json(
            index_path,
            {"source": self.source, "checksum": checksum,
             "prefix": self.prefix, "terms": terms},
        )
        self._adopt(terms)
        LOGGER.info(
            "Indexed %d %s term(s) from %s", len(self.terms), self.ontology_id, local.name
        )

    def _adopt(self, terms: list[dict]) -> None:
        self.terms = terms
        self._by_curie = {term["curie"]: term for term in terms}

    def _fetch(self, refresh: bool) -> Path:
        """Return a local copy of the ontology, downloading it if need be."""
        candidate = Path(self.source)
        if candidate.exists():
            return candidate

        suffix = "".join(Path(urllib.parse.urlparse(self.source).path).suffixes[-1:])
        local = self.cache_dir / f"{self.ontology_id}{suffix or '.owl'}"
        if local.exists() and not refresh:
            return local
        LOGGER.info("Downloading %s from %s", self.ontology_id, self.source)
        request = urllib.request.Request(
            self.source, headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if local.exists():
                # An outage is not a reason to lose a corpus: the copy already
                # on disk is what the last run used, and still grounds terms.
                LOGGER.warning(
                    "Could not refresh %s (%s); using the cached copy",
                    self.ontology_id, error,
                )
                return local
            raise GroundingError(
                f"Could not fetch ontology {self.ontology_id} from {self.source}: {error}"
            ) from error
        local.write_bytes(payload)
        return local

    def _build_index(self, path: Path) -> list[dict]:
        try:
            from rdflib import RDF, RDFS, OWL, Graph, URIRef
        except ImportError as error:  # pragma: no cover - dependency guard
            raise GroundingError(
                "The 'local' grounding backend needs rdflib: pip install rdflib"
            ) from error

        # rdflib logs a full traceback per malformed literal. ELMO has a few
        # dates typed as xsd:dateTime, which is a note for its maintainer and
        # not something this run can act on.
        noisy = logging.getLogger("rdflib.term")
        previous = noisy.level
        noisy.setLevel(logging.CRITICAL)
        try:
            graph = Graph()
            graph.parse(str(path))
        finally:
            noisy.setLevel(previous)

        # Classes and named individuals; properties are not things a node's
        # entity_term denotes.
        subjects = set(graph.subjects(RDF.type, OWL.Class)) | set(
            graph.subjects(RDF.type, OWL.NamedIndividual)
        )
        synonym_predicates = [URIRef(iri) for iri in self.SYNONYM_PREDICATES]

        terms: list[dict] = []
        for subject in subjects:
            if not isinstance(subject, URIRef):
                continue  # blank nodes: restrictions and other anonymous classes
            curie = self._curie(str(subject))
            if curie is None or curie.split(":")[0].lower() in self.EXCLUDED_PREFIXES:
                continue
            labels = [str(value) for value in graph.objects(subject, RDFS.label)]
            if not labels:
                continue
            synonyms = [
                str(value)
                for predicate in synonym_predicates
                for value in graph.objects(subject, predicate)
            ]
            terms.append(
                {
                    "curie": curie,
                    "label": labels[0],
                    "synonyms": sorted({*labels[1:], *synonyms}),
                }
            )
        return sorted(terms, key=lambda term: term["curie"])

    def _curie(self, iri: str) -> Optional[str]:
        """Mint a CURIE, preferring the longest matching schema prefix."""
        best_prefix, best_base = None, ""
        for prefix, base in self.prefixes.items():
            if iri.startswith(base) and len(base) > len(best_base):
                best_prefix, best_base = prefix, base
        if best_prefix:
            return f"{best_prefix}:{iri[len(best_base):]}"
        if not self.prefix:
            return None
        # No declared prefix covers it: fall back to the configured one and the
        # IRI's own last segment, which is what OBO-style IRIs encode anyway.
        local = re.split(r"[/#]", iri)[-1]
        if not local:
            return None
        return f"{self.prefix}:{local.split('_')[-1] if '_' in local else local}"

    # -- lookup ---------------------------------------------------------

    def search(self, term: str, ontologies: list[str]) -> list[Match]:
        matches = []
        for entry in self.terms:
            by_label = best_similarity(term, [entry["label"]])
            by_synonym = best_similarity(term, entry["synonyms"])
            score = max(by_label, by_synonym * 0.98)
            if score <= 0:
                continue
            matches.append(
                Match(
                    term,
                    entry["curie"],
                    entry["label"],
                    # An ontology that imports terms answers for them under
                    # their own prefix: an ENVO term reached through ELMO is
                    # still an ENVO term, and the report should say so.
                    entry["curie"].split(":")[0].lower(),
                    self.name,
                    score,
                )
            )
        return sorted(matches, key=lambda match: -match.score)[:5]

    def label(self, curie: str) -> Optional[str]:
        entry = self._by_curie.get(curie)
        return entry["label"] if entry else None


def _checksum(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _digest(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]


class OaklibBackend:
    """Optional: the LinkML ecosystem's own Ontology Access Kit.

    Worth having because it can point at a *local* ontology file, which is the
    only way to ground reproducibly against a pinned release, and the only way
    to ground at all on a machine with no outbound network. Not a dependency:
    it is used when installed and configured, and never otherwise.
    """

    name = "oaklib"

    def __init__(self, adapter: str = "ols:", timeout: int = 20):
        try:
            from oaklib import get_adapter
        except ImportError as error:  # pragma: no cover - optional dependency
            raise GroundingError(
                "backend 'oaklib' needs the oaklib package: pip install oaklib"
            ) from error
        self._adapter = get_adapter(adapter)
        self.timeout = timeout

    def search(self, term: str, ontologies: list[str]) -> list[Match]:
        matches = []
        for curie in list(self._adapter.basic_search(term))[:5]:
            label = self._adapter.label(curie) or ""
            prefix = curie.split(":")[0].lower()
            if ontologies and prefix not in {name.lower() for name in ontologies}:
                continue
            matches.append(
                Match(term, curie, label, prefix, self.name, similarity(term, label))
            )
        return sorted(matches, key=lambda match: -match.score)

    def label(self, curie: str) -> Optional[str]:
        return self._adapter.label(curie)


class NullBackend:
    """Resolves nothing. What ``--backend none`` selects, and the offline default."""

    name = "none"

    def search(self, term: str, ontologies: list[str]) -> list[Match]:
        return []

    def label(self, curie: str) -> Optional[str]:
        return None


def build_backend(
    spec: str,
    timeout: int = 20,
    registry: Optional[dict] = None,
    ontology_dir: Optional[Path] = None,
    prefixes: Optional[dict[str, str]] = None,
    refresh: bool = False,
) -> tuple[Any, list[str]]:
    """Parse a route like ``ols:envo,go`` or ``local:elmo`` into a backend.

    ``registry`` holds the ``ontologies:`` block from config, which is where a
    ``local:`` route's source URL and CURIE prefix are declared — a URL does
    not fit in a route string, and naming it once keeps a route readable.
    """
    name, _, rest = spec.partition(":")
    name = name.strip().lower()
    ontologies = [item.strip() for item in rest.split(",") if item.strip()]
    if name in {"", "none", "null"}:
        return NullBackend(), []
    if name == "ols":
        return OLSBackend(timeout=timeout), ontologies
    if name == "wikidata":
        return WikidataBackend(timeout=timeout), ontologies
    if name == "oaklib":
        return OaklibBackend(adapter=rest or "ols:", timeout=timeout), []
    if name in {"local", "owl", "file"}:
        identifier = (ontologies[0] if ontologies else "").strip()
        definition = (registry or {}).get(identifier)
        if definition is None:
            raise GroundingError(
                f"Route {spec!r} names no ontology declared under grounding.ontologies; "
                f"declared: {sorted((registry or {}))}"
            )
        if isinstance(definition, str):
            definition = {"source": definition}
        return (
            LocalOntologyBackend(
                source=definition["source"],
                cache_dir=ontology_dir or DEFAULT_ONTOLOGY_DIR,
                prefix=definition.get("prefix") or identifier,
                prefixes=prefixes or {},
                ontology_id=identifier,
                timeout=timeout,
                refresh=refresh,
            ),
            [],
        )
    raise GroundingError(
        f"Unknown grounding backend {name!r}; expected ols, wikidata, local, "
        f"oaklib or none"
    )


# ---------------------------------------------------------------------------
# Curated overrides
# ---------------------------------------------------------------------------


def load_overrides(path: Optional[Path] = None) -> dict[str, dict]:
    """Read term -> decided CURIE mappings, keyed by folded term.

    Some groundings are not lookups but judgements. "Phosphorus" in an ecology
    paper means the element as a nutrient; CHEBI's best lexical match for it is
    ``tetraphosphorus``, the P4 allotrope, which is a different claim about the
    world. No similarity threshold separates those two -- only a person can --
    so decided cases live in a file beside the config, each carrying its
    reasoning, and are consulted before any lookup runs.
    """
    resolved = Path(path or DEFAULT_OVERRIDES_PATH)
    if not resolved.exists():
        return {}
    data = load_yaml(resolved)
    overrides: dict[str, dict] = {}
    for term, body in (data.get("terms") or {}).items():
        if isinstance(body, str):
            body = {"curie": body}
        if not isinstance(body, dict) or not body.get("curie"):
            LOGGER.warning("Ignoring override for %r: no curie", term)
            continue
        overrides[_fold_term(term)] = {
            "curie": str(body["curie"]),
            "label": str(body.get("label") or term),
            "ontology": str(body["curie"]).split(":")[0].lower(),
        }
    if overrides:
        LOGGER.info(
            "Loaded %d curated grounding override(s) from %s",
            len(overrides), resolved.name,
        )
    return overrides


def _fold_term(value: str) -> str:
    return " ".join(str(value).lower().split())


# ---------------------------------------------------------------------------
# Grounder
# ---------------------------------------------------------------------------


class Grounder:
    """Resolve free-text terms to ontology CURIEs, with a persistent cache."""

    def __init__(
        self,
        routes: Optional[dict[str, Any]] = None,
        min_score: float = 0.6,
        timeout: int = 20,
        pause_seconds: float = 0.1,
        cache_path: Optional[Path] = None,
        slots: Optional[Iterable[str]] = None,
        backoff: bool = True,
        backoff_penalty: float = 0.15,
        overrides: Optional[dict[str, dict]] = None,
        ontologies: Optional[dict] = None,
        ontology_dir: Optional[Path] = None,
        prefixes: Optional[dict[str, str]] = None,
        refresh_ontologies: bool = False,
    ):
        self.routes = dict(routes or DEFAULT_GROUNDING["routes"])
        self.min_score = min_score
        self.timeout = timeout
        self.pause_seconds = pause_seconds
        self.slots = set(slots or DEFAULT_GROUNDING["slots"])
        self.backoff = backoff
        self.backoff_penalty = backoff_penalty
        self.overrides = overrides if overrides is not None else load_overrides()
        self.ontologies = dict(ontologies or {})
        self.ontology_dir = Path(ontology_dir) if ontology_dir else DEFAULT_ONTOLOGY_DIR
        self.prefixes = dict(prefixes or {})
        self.refresh_ontologies = refresh_ontologies
        self.cache_path = cache_path
        self.cache: dict[str, Optional[dict]] = {}
        if cache_path and cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
                LOGGER.info("Loaded %d cached lookup(s) from %s",
                            len(self.cache), cache_path.name)
            except (OSError, ValueError) as error:
                LOGGER.warning("Ignoring unreadable cache %s: %s", cache_path, error)
        self._backends: dict[str, tuple[Any, list[str]]] = {}

    # -- backends -----------------------------------------------------------

    def _route(self, entity_type: Optional[str]) -> list[tuple[Any, list[str], str]]:
        """The backends to try for this entity type, in the order configured.

        A route may name several: a project's own ontology first, then the
        published ones. First to meet the threshold wins, which is what makes
        the order a statement of preference rather than a tie-break.
        """
        configured = self.routes.get(entity_type or "") or self.routes.get(
            "default", "none"
        )
        specs = [configured] if isinstance(configured, str) else list(configured)
        return [self._backend(spec) for spec in specs]

    def _backend(self, spec: str) -> tuple[Any, list[str], str]:
        if spec not in self._backends:
            try:
                self._backends[spec] = build_backend(
                    spec,
                    self.timeout,
                    registry=self.ontologies,
                    ontology_dir=self.ontology_dir,
                    prefixes=self.prefixes,
                    refresh=self.refresh_ontologies,
                )
            except GroundingError as error:
                # A misconfigured or unreachable ontology must not take the
                # whole corpus down; the other backends in the route still work.
                LOGGER.error("Grounding backend %r unavailable: %s", spec, error)
                self._backends[spec] = (NullBackend(), [])
        backend, ontologies = self._backends[spec]
        return backend, ontologies, spec

    # -- lookup -------------------------------------------------------------

    def ground_term(
        self, term: str, entity_type: Optional[str], report: GroundingReport
    ) -> Optional[Match]:
        """Resolve one term, or return None and record why.

        A decided override wins outright. Otherwise the whole phrase is looked
        up, and only if that finds nothing are its trailing words tried -- held
        to a higher bar, since dropping "soil" from "soil salinity" throws away
        something the authors wrote.
        """
        decided = self.overrides.get(_fold_term(term))
        if decided:
            report.overridden += 1
            return Match(term, decided["curie"], decided["label"],
                         decided["ontology"], "override", 1.0, via="override")

        errors_before = report.errors
        match = self._lookup(term, entity_type, report)
        if match is not None or not self.backoff:
            return match
        if report.errors > errors_before:
            # The service is unreachable, not merely unhelpful. Retrying with a
            # shorter query would just fail again, once per word.
            return None

        threshold = min(0.95, self.min_score + self.backoff_penalty)
        for shorter in _backoff_queries(term):
            candidate = self._lookup(shorter, entity_type, report, threshold)
            if candidate is not None:
                report.by_backoff += 1
                candidate.via = f"backoff:{shorter}"
                candidate.query = term
                return candidate
        return None

    def _lookup(
        self,
        term: str,
        entity_type: Optional[str],
        report: GroundingReport,
        min_score: Optional[float] = None,
    ) -> Optional[Match]:
        """The best match across this route's backends, preferring earlier ones.

        Not first-past-the-post: ELMO offers "Inland Salt Marsh" for "salt
        marsh" at 0.67, and taking that because ELMO comes first would lose
        ENVO's "saline marsh" at 0.98 — a coastal corpus grounded to an inland
        ecosystem. Preference decides ties, not contests.

        An exact match in an earlier backend does short-circuit, so a term ELMO
        names outright never costs a request to a public service.
        """
        threshold = self.min_score if min_score is None else min_score
        best: Optional[Match] = None
        for backend, ontologies, spec in self._route(entity_type):
            candidate = self._search(backend, ontologies, spec, term, report)
            if candidate is None:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
            if best.score >= EXACT_MATCH:
                break
        return best if best is not None and best.score >= threshold else None

    def _search(
        self,
        backend: Any,
        ontologies: list[str],
        spec: str,
        term: str,
        report: GroundingReport,
    ) -> Optional[Match]:
        """One backend's best candidate, cached by (route spec, term).

        No threshold is applied here: the caller compares candidates across the
        whole route, and the cache keeps whatever was found so re-running with
        a different --min-score re-decides without re-querying.
        """
        if isinstance(backend, NullBackend):
            return None

        key = f"{spec}|{term.strip().lower()}"
        if key in self.cache:
            report.cache_hits += 1
            cached = self.cache[key]
            return Match(**cached) if cached else None

        try:
            report.requests += 1
            candidates = backend.search(term, ontologies)
            # A local index is in memory; only a remote service needs pacing.
            if self.pause_seconds and not isinstance(backend, LocalOntologyBackend):
                time.sleep(self.pause_seconds)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            # A lookup service being down is not a reason to lose an extraction.
            report.errors += 1
            LOGGER.warning("Lookup failed for %r (%s): %s", term, spec, error)
            return None

        best = candidates[0] if candidates else None
        self._remember(key, best.to_dict() if best else None)
        return best

    def label_for(self, curie: str) -> Optional[str]:
        """Reverse lookup: the label an identifier denotes, cached like the rest."""
        key = f"label|{curie.lower()}"
        if key in self.cache:
            cached = self.cache[key]
            return cached.get("label") if cached else None
        backend, _, _ = self._backend(self._reverse_spec(curie))
        try:
            label = backend.label(curie)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            LOGGER.warning("Reverse lookup failed for %s: %s", curie, error)
            return None
        self._remember(key, {"label": label} if label else None)
        return label

    def _reverse_spec(self, curie: str) -> str:
        """Which backend can name this identifier, from its prefix.

        A locally loaded ontology answers for its own prefix, which is what
        turns ``elmo:3622713`` back into "water table depth" when the one-shot
        example is de-grounded.
        """
        prefix = curie.split(":")[0].lower()
        if re.match(r"^[QP]\d+$", curie) or prefix == "wd":
            return "wikidata"
        for identifier, definition in self.ontologies.items():
            declared = (
                definition.get("prefix", identifier)
                if isinstance(definition, dict)
                else identifier
            )
            if prefix == str(declared).lower() or prefix == identifier.lower():
                return f"local:{identifier}"
        return "ols"

    def _remember(self, key: str, value: Optional[dict]) -> None:
        self.cache[key] = value
        if self.cache_path:
            atomic_write_json(self.cache_path, self.cache)

    # -- graphs -------------------------------------------------------------

    def ground_graph(self, graph: dict, profile: Optional[ExtractionProfile] = None) -> GroundingReport:
        """Ground every term slot in a graph, in place."""
        report = GroundingReport()
        slots = self.slots if profile is None else self._slots_for(profile)
        for node in graph.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            self._ground_object(node, node.get("entity_type"), slots, report,
                                context=node.get("id"))
            for applied in node.get("applied_to") or []:
                if isinstance(applied, dict):
                    self._ground_object(
                        applied, applied.get("entity_type") or "taxon", slots, report,
                        context=f"{node.get('id')}.applied_to",
                    )
        return report

    def _slots_for(self, profile: ExtractionProfile) -> set[str]:
        """Term slots according to the schema, intersected with the configured set.

        The schema knows which slots hold vocabulary terms — the same test the
        prompt uses to tell the model *not* to invent one. Reusing it means the
        two halves cannot drift: whatever was asked for as plain text is
        exactly what gets looked up.
        """
        from_schema = {
            slot.name
            for class_name in profile.class_names
            for slot in profile.get(class_name).slots
            if wants_identifier(slot)
        }
        return {name for name in self.slots if name in from_schema} or set(self.slots)

    def _ground_object(
        self,
        instance: dict,
        entity_type: Optional[str],
        slots: set[str],
        report: GroundingReport,
        context: Optional[str] = None,
    ) -> None:
        for slot in slots:
            value = instance.get(slot)
            if not isinstance(value, str) or not value.strip():
                continue
            if looks_like_identifier(value):
                # Either a previous run grounded it, or the model ignored the
                # instruction. Either way it is not ours to look up again.
                report.already_identifier += 1
                continue
            match = self.ground_term(value, entity_type, report)
            if match is None:
                report.unresolved += 1
                report.misses.append(
                    {"slot": slot, "term": value, "entity_type": entity_type,
                     "node": context}
                )
                continue
            instance[slot] = match.curie
            report.grounded += 1
            report.ontologies[match.ontology] = report.ontologies.get(match.ontology, 0) + 1
            report.matches.append(
                {**match.to_dict(), "slot": slot, "entity_type": entity_type,
                 "node": context}
            )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def grounding_settings(config_path: Optional[Path] = None) -> dict:
    """Read the ``grounding:`` block, filling in the defaults it omits."""
    try:
        data = load_yaml(config_path or DEFAULT_PIPELINE_CONFIG)
    except FileNotFoundError:
        return dict(DEFAULT_GROUNDING)
    block = data.get("grounding") or {}
    settings = {**DEFAULT_GROUNDING, **block}
    settings["routes"] = {**DEFAULT_GROUNDING["routes"], **(block.get("routes") or {})}
    settings["ontologies"] = {
        **DEFAULT_GROUNDING["ontologies"], **(block.get("ontologies") or {})
    }
    return settings


def grounder_from_config(
    config_path: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    backend_override: Optional[str] = None,
    min_score: Optional[float] = None,
    profile: Optional[ExtractionProfile] = None,
    refresh_ontologies: bool = False,
) -> Grounder:
    """Build a Grounder from config.

    ``profile`` supplies the schema's CURIE prefix map, which is what lets a
    locally loaded ontology mint the same CURIEs the schema's own enums use.
    """
    settings = grounding_settings(config_path)
    routes = settings["routes"]
    if backend_override:
        # One backend for everything, for a quick "--backend none" or a test.
        routes = {key: backend_override for key in routes}
    return Grounder(
        routes=routes,
        min_score=min_score if min_score is not None else float(settings["min_score"]),
        timeout=int(settings["timeout"]),
        pause_seconds=float(settings["pause_seconds"]),
        cache_path=cache_path,
        slots=settings["slots"],
        backoff=bool(settings.get("backoff", True)),
        backoff_penalty=float(settings.get("backoff_penalty", 0.15)),
        overrides=load_overrides(settings.get("overrides")),
        ontologies=settings.get("ontologies") or {},
        ontology_dir=settings.get("ontology_dir"),
        prefixes=(profile.prefixes if profile else None),
        refresh_ontologies=refresh_ontologies,
    )


def _backoff_queries(term: str) -> list[str]:
    """Progressively shorter tails of a phrase: English puts the head noun last."""
    words = [word for word in term.split() if word]
    if len(words) < 2:
        return []
    return [" ".join(words[index:]) for index in range(1, min(len(words), 4))]


def snapshot_id(report: GroundingReport, routes: dict[str, str]) -> str:
    """A one-line record of what did the grounding, for graph provenance."""
    from datetime import date

    specs = [
        spec
        for value in routes.values()
        for spec in ([value] if isinstance(value, str) else value)
        if spec
    ]
    backends = sorted({spec.split(":")[0] for spec in specs})
    ontologies = ",".join(sorted(report.ontologies)) or "none"
    return f"{'+'.join(backends)} {date.today().isoformat()} [{ontologies}]"


# ---------------------------------------------------------------------------
# The one-shot example
# ---------------------------------------------------------------------------


def plainify(graph: dict, grounder: Grounder, slots: Iterable[str] = ("entity_term",)) -> dict:
    """Turn identifiers in a worked example back into the terms they denote.

    Hand annotations carry grounded terms — CAMO's own gold files use Wikidata
    QIDs — and an example is imitated, not read. Leaving ``entity_term: Q30019``
    in the one-shot teaches the model to emit QIDs, which is exactly what this
    pipeline does not want. Where the label cannot be recovered the value is
    left alone and counted, because silently deleting half an example is worse
    than showing one identifier.
    """
    replaced = unresolved = 0

    def visit(value: Any) -> Any:
        nonlocal replaced, unresolved
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if key in slots and looks_like_identifier(item):
                    label = grounder.label_for(str(item))
                    if label:
                        value[key] = label
                        replaced += 1
                    else:
                        unresolved += 1
                else:
                    value[key] = visit(item)
            return value
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    visit(graph)
    if replaced or unresolved:
        LOGGER.info(
            "Example de-grounded: %d identifier(s) replaced with labels, %d left as-is",
            replaced, unresolved,
        )
    return graph


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def collect_graph_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(
                path
                for suffix in ("*.yaml", "*.yml", "*.json")
                for path in sorted(item.glob(suffix))
                if not path.name.endswith(".report.json")
                and path.name not in {"annotation_report.json", "grounding_report.json",
                                      "grounding_cache.json", "merge_report.json",
                                      "validation.json", "manifest.json",
                                      "conversion_report.json"}
            )
        elif item.exists():
            paths.append(item)
        else:
            LOGGER.warning("Input not found, skipping: %s", item)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("graphs", nargs="+", type=Path,
                        help="Graph files or directories of them")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--cache", type=Path, help="Lookup cache JSON (reused across runs)")
    parser.add_argument("--backend", help="Use this backend for every entity type "
                                          "(ols:envo,go | wikidata | oaklib | none)")
    parser.add_argument("--min-score", type=float,
                        help="Reject a match below this label similarity")
    parser.add_argument("--report", type=Path, help="Where to write the grounding report")
    parser.add_argument("--report-only", action="store_true",
                        help="Look terms up and report, but do not rewrite the graphs")
    parser.add_argument("--refresh-ontologies", action="store_true",
                        help="Re-download locally loaded ontologies and rebuild "
                             "their term indexes")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)

    configure_stdio()
    configure_logging(args.log_level)

    paths = collect_graph_paths(args.graphs)
    if not paths:
        parser.error("no graph files found")

    profile = build_extraction_profile(args.schema)
    grounder = grounder_from_config(
        args.config, args.cache, args.backend, args.min_score,
        profile=profile, refresh_ontologies=args.refresh_ontologies,
    )
    total = GroundingReport()

    for path in paths:
        try:
            graph = load_graph(path)
        except (OSError, ValueError) as error:
            LOGGER.error("Could not load %s: %s", path, error)
            continue
        report = grounder.ground_graph(graph, profile)
        total.merge(report)
        LOGGER.info(
            "%s: %d grounded, %d unresolved, %d already identified",
            path.name, report.grounded, report.unresolved, report.already_identifier,
        )
        if not args.report_only and report.grounded:
            graph.setdefault("provenance", {})["ontology_snapshot_id"] = snapshot_id(
                report, grounder.routes
            )
            save_graph(graph, path)

    report_path = args.report or (
        paths[0].parent / "grounding_report.json" if paths else Path("grounding_report.json")
    )
    atomic_write_json(report_path, {"routes": grounder.routes, **total.to_dict()})

    print(f"\nGrounded {len(paths)} graph(s):")
    print(f"    {total.grounded:5d}  term(s) grounded "
          f"({total.overridden} by override, {total.by_backoff} by backoff)")
    print(f"    {total.unresolved:5d}  term(s) left as free text")
    print(f"    {total.already_identifier:5d}  already an identifier")
    print(f"    {total.requests:5d}  lookup(s), {total.cache_hits} cache hit(s), "
          f"{total.errors} error(s)")
    for ontology, count in sorted(total.ontologies.items(), key=lambda item: -item[1]):
        print(f"    {count:5d}  {ontology}")
    print(f"  wrote {report_path}")
    if args.report_only:
        print("  (--report-only: no graph was rewritten)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
