"""Tests for the ingest pipeline.

Everything here runs offline. The two lookup backends are the only part that
touches the network, and they are exercised through a stub: what is worth
testing is the routing, the thresholds, the overrides and the failure
behaviour, none of which depend on EBI being up.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from graphingest import ground as ground_module
from graphingest import ris as ris_module
from graphingest.annotate import load_example, source_document_from_row
from graphingest.consolidate import Consolidator, node_identity
from graphingest.graph_io import load_graph, save_graph, validate_graph
from graphingest.ground import (
    Grounder,
    GroundingReport,
    Match,
    _backoff_queries,
    load_overrides,
    looks_like_identifier,
    plainify,
    similarity,
)
from graphingest.merge import collect_paths, merge_graphs
from graphingest.normalize import coerce_enum, normalize_graph
from graphingest.schema import DEFAULT_SCHEMA, build_extraction_profile
from graphingest.schema_prompt import (
    build_extraction_prompt,
    build_system_prompt,
    wants_identifier,
)

RIS_SAMPLE = """\
TY  - JOUR
TI  - Reduction of <span class="nocase">Aedes dorsalis</span> by enhancing tidal action
AU  - Kramer, V. L.
AU  - Collins, J. N.
T2  - Journal of the American Mosquito Control Association
PY  - 1995
VL  - 11
SP  - 389
EP  - 394
L1  - files/16247/Kramer et al. - 1995 - Reduction of Aedes dorsalis.pdf
ER  -

TY  - JOUR
TI  - Wetlands and mosquitoes: a review
AU  - Dale, P. E. R.
DA  - 2008///
DO  - 10.1007/s11273-008-9098-2
AB  - Wetlands are valued for many reasons
  and mosquitoes are one cost of that value.
ER  -
"""


@pytest.fixture(scope="session")
def profile():
    return build_extraction_profile(DEFAULT_SCHEMA)


# ---------------------------------------------------------------------------
# RIS
# ---------------------------------------------------------------------------


def test_parse_ris_splits_records_and_repeats_authors():
    records = ris_module.parse_ris(RIS_SAMPLE)
    assert len(records) == 2
    assert records[0]["AU"] == ["Kramer, V. L.", "Collins, J. N."]
    assert records[1]["DO"] == ["10.1007/s11273-008-9098-2"]


def test_parse_ris_joins_continuation_lines():
    records = ris_module.parse_ris(RIS_SAMPLE)
    assert records[1]["AB"][0].endswith("one cost of that value.")
    assert "\n" not in records[1]["AB"][0]


def test_clean_text_strips_exporter_markup():
    records = ris_module.parse_ris(RIS_SAMPLE)
    title = ris_module.clean_text(records[0]["TI"][0])
    assert title == "Reduction of Aedes dorsalis by enhancing tidal action"


def test_parse_year_reads_whichever_date_tag_is_present():
    records = ris_module.parse_ris(RIS_SAMPLE)
    assert ris_module.parse_year(records[0]) == 1995
    assert ris_module.parse_year(records[1]) == 2008  # from "DA  - 2008///"


def test_file_links_resolve_relative_to_the_ris_file(tmp_path):
    records = ris_module.parse_ris(RIS_SAMPLE)
    paths = ris_module.record_file_paths(records[0], tmp_path)
    assert paths == [tmp_path / "files/16247/Kramer et al. - 1995 - Reduction of Aedes dorsalis.pdf"]


def test_web_urls_are_not_mistaken_for_files(tmp_path):
    record = {"L1": ["https://example.org/paper.pdf"], "UR": ["https://doi.org/10.1/x"]}
    assert ris_module.record_file_paths(record, tmp_path) == []


def test_build_manifest_pairs_pdfs_and_reports_the_gaps(tmp_path):
    corpus = tmp_path / "corpus"
    linked = corpus / "files" / "16247"
    linked.mkdir(parents=True)
    (linked / "Kramer et al. - 1995 - Reduction of Aedes dorsalis.pdf").write_bytes(b"%PDF-")
    (corpus / "stray.pdf").write_bytes(b"%PDF-")
    (corpus / "export.ris").write_text(RIS_SAMPLE, encoding="utf-8")

    manifest = ris_module.build_manifest(corpus)
    by_match = {row["match"] for row in manifest["documents"]}
    assert by_match == {"ris_link", "no_pdf", "unmatched_pdf"}
    assert manifest["counts"]["records"] == 2
    # The record with a DOI is identified by it; the one without falls back.
    ids = {row["document_id"] for row in manifest["documents"]}
    assert "10.1007/s11273-008-9098-2" in ids


def test_manifest_slugs_stay_unique(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    duplicated = RIS_SAMPLE + RIS_SAMPLE
    (corpus / "export.ris").write_text(duplicated, encoding="utf-8")
    slugs = [row["slug"] for row in ris_module.build_manifest(corpus)["documents"]]
    assert len(slugs) == len(set(slugs))


def test_zotero_filename_gives_metadata_for_an_unlisted_pdf(tmp_path):
    parsed = ris_module.parse_filename(
        tmp_path / "Dale and Knight - 2008 - Wetlands and mosquitoes a review.pdf"
    )
    assert parsed["year"] == 2008
    assert parsed["title"] == "Wetlands and mosquitoes a review"
    assert "Dale" in parsed["authors"][0]


# ---------------------------------------------------------------------------
# The prompt asks for terms, not identifiers
# ---------------------------------------------------------------------------


def test_term_slots_are_asked_for_in_plain_language(profile):
    entity_term = next(
        slot for slot in profile.get("CausalNode").slots if slot.name == "entity_term"
    )
    assert wants_identifier(entity_term)


def test_record_identifiers_are_not_asked_for_as_terms(profile):
    node_id = next(slot for slot in profile.get("CausalNode").slots if slot.name == "id")
    # Same uriorcurie range as entity_term; only LinkML's identifier flag separates them.
    assert node_id.range == entity_range(profile)
    assert node_id.identifier
    assert not wants_identifier(node_id)


def entity_range(profile):
    return next(
        slot.range for slot in profile.get("CausalNode").slots if slot.name == "entity_term"
    )


def test_prompt_tells_the_model_not_to_invent_identifiers(profile):
    prompt = build_extraction_prompt(profile, "ARTICLE")
    assert "Do NOT write an identifier" in prompt
    assert "plain-language term here" in prompt


def test_prompt_carries_no_curie_vocabulary_for_the_model_to_copy(profile):
    prompt = build_extraction_prompt(profile, "ARTICLE")
    assert "# Ontology prefixes" not in prompt
    # An ELMO/ENVO identifier appearing in the prompt is one the model can echo.
    assert "ENVO:" not in prompt
    assert "elmo:" not in prompt


def test_system_prompt_names_the_domain_when_given_one():
    assert "restoration ecology literature" in build_system_prompt("restoration ecology")
    assert "scientific literature" in build_system_prompt(None)


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


class StubBackend:
    """A backend with a fixed vocabulary, so routing can be tested offline."""

    name = "stub"

    def __init__(self, vocabulary: dict[str, tuple[str, str]]):
        self.vocabulary = vocabulary
        self.queries: list[str] = []

    def search(self, term, ontologies):
        self.queries.append(term)
        hit = self.vocabulary.get(term.lower())
        if hit is None:
            return []
        curie, label = hit
        return [
            Match(term, curie, label, curie.split(":")[0].lower(), self.name,
                  similarity(term, label))
        ]

    def label(self, curie):
        for _, (candidate, label) in self.vocabulary.items():
            if candidate == curie:
                return label
        return None


@pytest.fixture
def stub_grounder(monkeypatch):
    backend = StubBackend(
        {
            "aedes dorsalis": ("Q13543883", "Aedes dorsalis"),
            "salinity": ("PATO:0085001", "salinity"),
            "saline marsh": ("ENVO:00000054", "saline marsh"),
        }
    )
    monkeypatch.setattr(
        ground_module, "build_backend", lambda spec, *args, **kwargs: (backend, [])
    )
    grounder = Grounder(routes={"default": "stub"}, min_score=0.6, pause_seconds=0,
                        overrides={})
    return grounder, backend


def test_identifier_shapes_are_recognised():
    assert looks_like_identifier("ENVO:00002006")
    assert looks_like_identifier("Q30019")
    assert looks_like_identifier("http://purl.obolibrary.org/obo/ENVO_00002006")
    assert not looks_like_identifier("soil salinity")
    assert not looks_like_identifier("Aedes dorsalis")


def test_a_term_that_resolves_is_replaced_by_its_curie(stub_grounder):
    grounder, _ = stub_grounder
    report = GroundingReport()
    match = grounder.ground_term("Aedes dorsalis", "taxon", report)
    assert match.curie == "Q13543883"
    assert match.via == "lookup"


def test_a_term_that_resolves_to_nothing_is_left_alone_and_counted(stub_grounder):
    grounder, _ = stub_grounder
    graph = {
        "nodes": [
            {"id": "n1", "entity_type": "management_intervention",
             "entity_term": "open marsh water management"}
        ]
    }
    report = grounder.ground_graph(graph)
    assert graph["nodes"][0]["entity_term"] == "open marsh water management"
    assert report.unresolved == 1
    assert report.misses[0]["term"] == "open marsh water management"


def test_backoff_falls_back_to_the_head_noun(stub_grounder):
    grounder, backend = stub_grounder
    report = GroundingReport()
    match = grounder.ground_term("soil salinity", "environmental_variable", report)
    assert match.curie == "PATO:0085001"
    assert match.via == "backoff:salinity"
    assert match.query == "soil salinity"   # the report keeps what the authors wrote
    assert backend.queries == ["soil salinity", "salinity"]
    assert report.by_backoff == 1


def test_backoff_can_be_switched_off(stub_grounder):
    grounder, _ = stub_grounder
    grounder.backoff = False
    assert grounder.ground_term("soil salinity", None, GroundingReport()) is None


def test_backoff_queries_are_progressively_shorter_tails():
    assert _backoff_queries("mosquito larval abundance") == [
        "larval abundance",
        "abundance",
    ]
    assert _backoff_queries("salinity") == []


def test_a_curated_override_beats_the_lookup(stub_grounder):
    grounder, backend = stub_grounder
    grounder.overrides = {"phosphorus": {"curie": "CHEBI:28659",
                                         "label": "phosphorus atom",
                                         "ontology": "chebi"}}
    report = GroundingReport()
    match = grounder.ground_term("Phosphorus", "environmental_variable", report)
    assert match.curie == "CHEBI:28659"
    assert match.via == "override"
    assert backend.queries == []  # decided cases never hit the service


def test_grounding_is_idempotent(stub_grounder):
    grounder, backend = stub_grounder
    graph = {"nodes": [{"id": "n1", "entity_type": "taxon",
                        "entity_term": "Aedes dorsalis"}]}
    grounder.ground_graph(graph)
    queries_after_first = len(backend.queries)
    second = grounder.ground_graph(graph)
    assert len(backend.queries) == queries_after_first
    assert second.already_identifier == 1
    assert second.grounded == 0


def test_applied_to_entities_are_grounded_too(stub_grounder):
    grounder, _ = stub_grounder
    graph = {
        "nodes": [
            {
                "id": "n1",
                "entity_type": "environmental_variable",
                "entity_term": "saline marsh",
                "applied_to": [{"entity_type": "taxon", "entity_term": "Aedes dorsalis"}],
            }
        ]
    }
    grounder.ground_graph(graph)
    assert graph["nodes"][0]["entity_term"] == "ENVO:00000054"
    assert graph["nodes"][0]["applied_to"][0]["entity_term"] == "Q13543883"


def test_a_backend_outage_leaves_the_graph_as_it_was(monkeypatch):
    class BrokenBackend:
        name = "broken"

        def search(self, term, ontologies):
            raise OSError("connection refused")

        def label(self, curie):
            raise OSError("connection refused")

    monkeypatch.setattr(
        ground_module, "build_backend", lambda spec, *args, **kwargs: (BrokenBackend(), [])
    )
    grounder = Grounder(routes={"default": "broken"}, pause_seconds=0, overrides={})
    graph = {"nodes": [{"id": "n1", "entity_term": "Aedes dorsalis"}]}
    report = grounder.ground_graph(graph)
    assert graph["nodes"][0]["entity_term"] == "Aedes dorsalis"
    assert report.errors == 1
    assert report.grounded == 0


def test_lookups_are_cached_between_runs(tmp_path, monkeypatch):
    backend = StubBackend({"aedes dorsalis": ("Q13543883", "Aedes dorsalis")})
    monkeypatch.setattr(
        ground_module, "build_backend", lambda spec, *args, **kwargs: (backend, [])
    )
    cache = tmp_path / "cache.json"
    first = Grounder(routes={"default": "stub"}, cache_path=cache, pause_seconds=0,
                     overrides={})
    first.ground_term("Aedes dorsalis", "taxon", GroundingReport())
    assert cache.exists()

    second = Grounder(routes={"default": "stub"}, cache_path=cache, pause_seconds=0,
                      overrides={})
    report = GroundingReport()
    match = second.ground_term("Aedes dorsalis", "taxon", report)
    assert match.curie == "Q13543883"
    assert report.cache_hits == 1
    assert len(backend.queries) == 1  # the second run asked nobody


def test_the_shipped_overrides_file_parses():
    overrides = load_overrides()
    assert overrides
    for term, body in overrides.items():
        assert ":" in body["curie"], f"{term} has no prefix in its CURIE"


def test_similarity_rewards_word_overlap_not_spelling():
    assert similarity("soil salinity", "salinity of soil") == 1.0
    assert similarity("salt marsh", "saline marsh") < 0.6
    assert similarity("soil temperature", "soil salinity") < 0.6


# ---------------------------------------------------------------------------
# The one-shot example
# ---------------------------------------------------------------------------


def test_example_identifiers_become_labels_again(tmp_path, stub_grounder):
    grounder, _ = stub_grounder
    example = {"nodes": [{"id": "x", "entity_term": "Q13543883",
                          "applied_to": [{"entity_term": "Q13543883"}]}]}
    plainify(example, grounder)
    assert example["nodes"][0]["entity_term"] == "Aedes dorsalis"
    assert example["nodes"][0]["applied_to"][0]["entity_term"] == "Aedes dorsalis"


def test_example_trimming_keeps_a_valid_graph(tmp_path):
    example = {
        "graph_id": "g", "schema_version": "0.7.9",
        "nodes": [{"id": f"n{index}", "entity_term": f"term {index}"} for index in range(5)],
        "edges": [
            {"id": "e1", "subject": "n0", "object": "n1"},
            {"id": "e2", "subject": "n0", "object": "n4"},   # endpoint gets trimmed away
        ],
    }
    path = tmp_path / "example.yaml"
    path.write_text(yaml.safe_dump(example), encoding="utf-8")

    trimmed = yaml.safe_load(load_example(path, max_nodes=2))
    kept = {node["id"] for node in trimmed["nodes"]}
    assert kept == {"n0", "n1"}
    # No edge may point at a node the trim removed.
    for edge in trimmed["edges"]:
        assert edge["subject"] in kept and edge["object"] in kept


def test_a_non_graph_example_is_passed_through(tmp_path):
    path = tmp_path / "example.md"
    path.write_text("# how to annotate\n", encoding="utf-8")
    assert load_example(path, max_nodes=2) == "# how to annotate\n"


# ---------------------------------------------------------------------------
# Metadata, normalization, merging
# ---------------------------------------------------------------------------


def test_source_document_drops_fields_the_schema_does_not_model(profile):
    row = {
        "slug": "kramer_1995", "document_id": "10.1/x", "doi": "10.1/x",
        "title": "A paper", "authors": ["Kramer, V. L."], "year": 1995,
        "journal": "JAMCA", "abstract": "...", "keywords": ["mosquito"],
        "volume": "11", "pdf_path": "/tmp/x.pdf",
    }
    document = source_document_from_row(row, profile)
    assert document["document_id"] == "10.1/x"
    assert document["authors"] == ["Kramer, V. L."]
    assert "abstract" not in document and "volume" not in document


def test_enum_coercion_folds_case_and_punctuation(profile):
    slot = next(
        slot for slot in profile.get("CausalEdge").slots if slot.name == "claim_strength"
    )
    assert coerce_enum("Direct Causal", slot)[0] == "direct_causal"
    assert coerce_enum("correlation", slot)[0] == "associational"
    assert coerce_enum("wildly invented value", slot)[0] is None


def test_normalization_drops_edges_with_unknown_endpoints(profile):
    raw = {
        "nodes": [{"id": "a", "name": "A", "entity_term": "thing a"}],
        "edges": [
            {"id": "e1", "subject": "a", "object": "ghost", "predicate": "causes"}
        ],
    }
    graph, report = normalize_graph(raw, profile)
    assert graph["edges"] == []
    assert any("endpoint" in drop["value"] for drop in report.dropped)


def test_merging_into_an_existing_graph_keeps_its_identity_and_unifies_nodes(profile):
    def node(identifier, term):
        return {"id": identifier, "name": term, "entity_type": "taxon",
                "entity_term": term, "measured_attribute": "abundance",
                "state_or_change_qualifier": "increased"}

    existing = {
        "graph_id": "camo:corpus", "schema_version": "0.7.9",
        "provenance": {"ontology_framework": "CAMO", "project": "mosaic"},
        "nodes": [node("old1", "Q13543883")], "edges": [],
    }
    incoming = {
        "graph_id": "camo:doc", "schema_version": "0.7.9", "provenance": {},
        "nodes": [node("new1", "Q13543883"), node("new2", "Q14570773")], "edges": [],
    }

    merged, report = merge_graphs([incoming], "0.7.9", existing=existing)
    assert merged["graph_id"] == "camo:corpus"
    assert merged["provenance"]["project"] == "mosaic"
    # The same grounded term in both graphs is one node, not two.
    assert report["nodes_in"] == 3 and report["nodes_out"] == 2


def test_grounding_before_merging_is_what_unifies_terms_across_papers():
    """Two papers naming the same taxon differently merge only once grounded."""
    def node(term):
        return {"id": f"n_{abs(hash(term)) % 1000}", "name": term, "entity_type": "taxon",
                "entity_term": term, "measured_attribute": "abundance",
                "state_or_change_qualifier": "increased"}

    ungrounded = [{"nodes": [node("Aedes dorsalis")], "edges": []},
                  {"nodes": [node("Ae. dorsalis")], "edges": []}]
    grounded = [{"nodes": [node("Q13543883")], "edges": []},
                {"nodes": [node("Q13543883")], "edges": []}]

    _, apart = Consolidator().consolidate(ungrounded)
    _, together = Consolidator().consolidate(grounded)
    assert apart.nodes_out == 2
    assert together.nodes_out == 1


def test_node_identity_ignores_the_composed_label():
    left = {"name": "Increased abundance of X", "entity_term": "Q1",
            "measured_attribute": "abundance", "state_or_change_qualifier": "increased",
            "entity_type": "taxon"}
    right = {**left, "name": "X abundance went up"}
    assert node_identity(left) == node_identity(right)


def test_merge_ignores_the_report_files_written_beside_the_graphs(tmp_path):
    (tmp_path / "doc.yaml").write_text("graph_id: g\nnodes: []\nedges: []\n", encoding="utf-8")
    (tmp_path / "doc.report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "annotation_report.json").write_text("{}", encoding="utf-8")
    assert [path.name for path in collect_paths([tmp_path])] == ["doc.yaml"]


def test_a_grounded_graph_still_validates(tmp_path, profile):
    graph = {
        "graph_id": "camo:test",
        "schema_version": profile.version,
        "provenance": {"ontology_framework": "causal-mosaic",
                       "ontology_snapshot_id": "ols+wikidata 2026-09-13 [envo]"},
        "source_documents": [{"document_id": "10.1/x", "title": "A paper", "year": 1995}],
        "nodes": [
            {"id": "camo:n1", "name": "Increased abundance of Aedes dorsalis",
             "entity_type": "taxon", "entity_term": "Q13543883",
             "measured_attribute": "abundance", "state_or_change_qualifier": "increased"},
            {"id": "camo:n2", "name": "Increased salinity", "entity_type":
             "environmental_variable", "entity_term": "PATO:0085001",
             "measured_attribute": "salinity", "state_or_change_qualifier": "increased"},
        ],
        "edges": [
            {"id": "camo:e1", "subject": "camo:n2", "object": "camo:n1",
             "predicate": "causes", "claim_strength": "direct_causal",
             "philosophical_accounts": ["interventionist"],
             "source_document": "10.1/x",
             "original_sentence": "Salinity increased Aedes dorsalis abundance."}
        ],
    }
    path = tmp_path / "graph.yaml"
    save_graph(graph, path)
    report = validate_graph(load_graph(path), DEFAULT_SCHEMA)
    assert report.ok, report.problems


def test_annotator_stamp_satisfies_the_schema_pattern(profile):
    from graphingest.annotate import annotator_stamp
    import re as _re

    slot = next(
        slot for slot in profile.get("CausalNode").slots if slot.name == "annotator"
    )
    stamp = annotator_stamp(profile, "qwen3.6:35b")
    assert stamp is not None
    # The colon in the model name is exactly what the raw name gets wrong.
    assert _re.fullmatch(slot.pattern, stamp)


def test_no_annotator_is_stamped_when_nothing_can_satisfy_the_pattern(profile):
    from copy import deepcopy
    from graphingest.annotate import annotator_stamp

    narrowed = deepcopy(profile)
    for slot in narrowed.get("CausalNode").slots:
        if slot.name == "annotator":
            slot.pattern = r"^orcid:\d{4}$"
    assert annotator_stamp(narrowed, "qwen3.6:35b") is None


def test_merging_keeps_the_record_of_what_grounded_the_terms():
    grounded = {
        "graph_id": "camo:doc", "schema_version": "0.7.9",
        "provenance": {"ontology_snapshot_id": "ols+wikidata 2026-09-13 [envo,pato]"},
        "nodes": [], "edges": [],
    }
    ungrounded_corpus = {
        "graph_id": "camo:corpus", "schema_version": "0.7.9",
        "provenance": {"ontology_framework": "CAMO"}, "nodes": [], "edges": [],
    }
    merged, _ = merge_graphs([grounded], "0.7.9", existing=ungrounded_corpus)
    assert merged["provenance"]["ontology_snapshot_id"] == (
        "ols+wikidata 2026-09-13 [envo,pato]"
    )


# ---------------------------------------------------------------------------
# Local ontologies (ELMO) and multi-backend routes
# ---------------------------------------------------------------------------

ELMO_FRAGMENT = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#">
  <owl:Class rdf:about="https://w3id.org/elmo/elmo_3620022">
    <rdfs:label>grubbing process</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="https://w3id.org/elmo/elmo_3621037">
    <rdfs:label>prescribed fire process</rdfs:label>
    <oboInOwl:hasExactSynonym>prescribed burning</oboInOwl:hasExactSynonym>
  </owl:Class>
  <owl:Class rdf:about="https://w3id.org/elmo/elmo_3620281">
    <rdfs:label>Inland Salt Marsh</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://purl.obolibrary.org/obo/ENVO_06105203">
    <rdfs:label>water table depth</rdfs:label>
  </owl:Class>
  <owl:NamedIndividual rdf:about="https://orcid.org/0000-0002-1825-0097">
    <rdfs:label>Josiah Carberry</rdfs:label>
  </owl:NamedIndividual>
</rdf:RDF>
"""

#: The prefix map CAMO declares; what makes an ELMO IRI mint as elmo:3620022.
SCHEMA_PREFIXES = {
    "elmo": "https://w3id.org/elmo/elmo_",
    "elmo_cas": "https://w3id.org/elmo/cas/",
    "ENVO": "http://purl.obolibrary.org/obo/ENVO_",
    "orcid": "https://orcid.org/",
}


@pytest.fixture
def elmo(tmp_path):
    """A local ontology backend over a fragment of real ELMO."""
    from graphingest.ground import LocalOntologyBackend

    source = tmp_path / "elmo.owl"
    source.write_text(ELMO_FRAGMENT, encoding="utf-8")
    return LocalOntologyBackend(
        source=str(source),
        cache_dir=tmp_path / "ontologies",
        prefix="elmo",
        prefixes=SCHEMA_PREFIXES,
        ontology_id="elmo",
    )


def test_local_ontology_mints_the_curies_the_schema_uses(elmo):
    by_label = {term["label"]: term["curie"] for term in elmo.terms}
    assert by_label["grubbing process"] == "elmo:3620022"


def test_an_imported_term_keeps_its_own_prefix(elmo):
    by_label = {term["label"]: term["curie"] for term in elmo.terms}
    # ELMO imports ENVO terms; reaching one through ELMO does not make it ELMO's.
    assert by_label["water table depth"] == "ENVO:06105203"
    match = elmo.search("water table depth", [])[0]
    assert match.ontology == "envo"


def test_people_are_not_indexed_as_vocabulary(elmo):
    assert all(not term["curie"].startswith("orcid") for term in elmo.terms)
    assert elmo.search("Josiah Carberry", []) == []


def test_a_classifier_suffix_does_not_block_a_match(elmo):
    # ELMO ends 232 of 652 labels with "process"; "grubbing" must still match.
    match = elmo.search("grubbing", [])[0]
    assert match.curie == "elmo:3620022"
    assert match.score == 1.0


def test_synonyms_are_indexed(elmo):
    match = elmo.search("prescribed burning", [])[0]
    assert match.curie == "elmo:3621037"
    assert match.score >= 0.9


def test_label_variants_strip_only_a_trailing_classifier():
    from graphingest.ground import label_variants

    assert label_variants("grubbing process") == ["grubbing process", "grubbing"]
    assert label_variants("process") == ["process"]          # nothing left to name
    assert label_variants("saline marsh") == ["saline marsh"]


def test_the_index_is_rebuilt_only_when_the_source_changes(tmp_path):
    from graphingest.ground import LocalOntologyBackend

    source = tmp_path / "elmo.owl"
    source.write_text(ELMO_FRAGMENT, encoding="utf-8")
    cache = tmp_path / "ontologies"

    def build():
        return LocalOntologyBackend(source=str(source), cache_dir=cache,
                                    prefixes=SCHEMA_PREFIXES, ontology_id="elmo")

    first = build()
    index = cache / "elmo.index.json"
    stamp = index.stat().st_mtime_ns
    build()
    assert index.stat().st_mtime_ns == stamp, "unchanged source should not reindex"

    source.write_text(
        ELMO_FRAGMENT.replace("grubbing process", "grubbing out process"),
        encoding="utf-8",
    )
    rebuilt = build()
    assert {term["label"] for term in rebuilt.terms} != {
        term["label"] for term in first.terms
    }


def test_a_route_prefers_the_project_ontology_but_not_at_any_price(monkeypatch):
    """ELMO first, yet a clearly better public match still wins."""
    project = StubBackend({"salt marsh": ("elmo:3620281", "Inland Salt Marsh"),
                           "ditch plugging": ("elmo:3620072", "ditch plugging process")})
    public = StubBackend({"salt marsh": ("ENVO:00000054", "salt marsh")})
    backends = {"local:elmo": (project, []), "ols:envo": (public, [])}
    monkeypatch.setattr(
        ground_module, "build_backend",
        lambda spec, *args, **kwargs: backends[spec],
    )
    grounder = Grounder(
        routes={"default": ["local:elmo", "ols:envo"]},
        min_score=0.6, pause_seconds=0, overrides={},
    )

    # A weak ELMO hit (0.67) loses to an exact ENVO one.
    salt = grounder.ground_term("salt marsh", None, GroundingReport())
    assert salt.curie == "ENVO:00000054"

    # A term only ELMO knows still comes from ELMO.
    ditch = grounder.ground_term("ditch plugging", None, GroundingReport())
    assert ditch.curie == "elmo:3620072"


def test_an_exact_match_in_the_preferred_ontology_asks_nobody_else(monkeypatch):
    project = StubBackend({"grubbing": ("elmo:3620022", "grubbing")})
    public = StubBackend({"grubbing": ("ENVO:9999999", "grubbing")})
    backends = {"local:elmo": (project, []), "ols:envo": (public, [])}
    monkeypatch.setattr(
        ground_module, "build_backend",
        lambda spec, *args, **kwargs: backends[spec],
    )
    grounder = Grounder(routes={"default": ["local:elmo", "ols:envo"]},
                        min_score=0.6, pause_seconds=0, overrides={})
    match = grounder.ground_term("grubbing", None, GroundingReport())
    assert match.curie == "elmo:3620022"
    assert public.queries == [], "the public service should not have been asked"


def test_an_unavailable_ontology_does_not_take_the_route_down(monkeypatch):
    from graphingest.ground import GroundingError

    public = StubBackend({"salt marsh": ("ENVO:00000054", "salt marsh")})

    def build(spec, *args, **kwargs):
        if spec.startswith("local:"):
            raise GroundingError("could not fetch elmo.owl")
        return (public, [])

    monkeypatch.setattr(ground_module, "build_backend", build)
    grounder = Grounder(routes={"default": ["local:elmo", "ols:envo"]},
                        min_score=0.6, pause_seconds=0, overrides={})
    match = grounder.ground_term("salt marsh", None, GroundingReport())
    assert match.curie == "ENVO:00000054"


def test_a_local_prefix_reverse_looks_up_through_its_own_ontology():
    grounder = Grounder(
        routes={"default": ["local:elmo", "ols:envo"]},
        ontologies={"elmo": {"source": "x.owl", "prefix": "elmo"}},
        overrides={}, pause_seconds=0,
    )
    # De-grounding a worked example has to know who can name elmo:3622713.
    assert grounder._reverse_spec("elmo:3622713") == "local:elmo"
    assert grounder._reverse_spec("Q30019") == "wikidata"
    assert grounder._reverse_spec("ENVO:00000054") == "ols"


def test_the_shipped_config_puts_elmo_ahead_of_the_public_services():
    from graphingest.ground import grounding_settings

    settings = grounding_settings()
    assert "elmo" in settings["ontologies"]
    interventions = settings["routes"]["management_intervention"]
    assert isinstance(interventions, list)
    assert interventions[0] == "local:elmo"


def test_the_index_is_rebuilt_when_the_prefix_map_changes(tmp_path):
    """The same OWL file under a different prefix map mints different CURIEs."""
    from graphingest.ground import LocalOntologyBackend

    source = tmp_path / "elmo.owl"
    source.write_text(ELMO_FRAGMENT, encoding="utf-8")
    cache = tmp_path / "ontologies"

    def curie_for(label, prefixes):
        backend = LocalOntologyBackend(
            source=str(source), cache_dir=cache, prefix="elmo",
            prefixes=prefixes, ontology_id="elmo",
        )
        return {term["label"]: term["curie"] for term in backend.terms}[label]

    # With the schema's map, an imported ENVO term keeps its ENVO identity.
    assert curie_for("water table depth", SCHEMA_PREFIXES) == "ENVO:06105203"
    # Without it, the fallback claims the term for ELMO — a different CURIE, so
    # a cache keyed only on the file would hand back the wrong one.
    assert curie_for("water table depth", {}) == "elmo:06105203"


# ---------------------------------------------------------------------------
# Reconciliation against the existing graph
# ---------------------------------------------------------------------------


def existing_node(identifier, term, attribute, qualifier="increased",
                  entity_type="taxon", name=None):
    return {
        "id": identifier,
        "name": name or f"{qualifier} {attribute} of {term}",
        "entity_type": entity_type,
        "entity_term": term,
        "measured_attribute": attribute,
        "state_or_change_qualifier": qualifier,
    }


CORPUS = {
    "graph_id": "camo:corpus",
    "schema_version": "0.7.9",
    "nodes": [
        existing_node("camo:n_larvae", "Q13543883", "larval abundance"),
        existing_node("camo:n_salinity", "PATO:0085001", "salinity",
                      entity_type="environmental_variable"),
        existing_node("camo:n_salinity_down", "PATO:0085001", "salinity",
                      qualifier="decreased", entity_type="environmental_variable"),
    ],
    "edges": [],
}


@pytest.fixture
def reconciler():
    from graphingest.reconcile import NodeReconciler

    return NodeReconciler(CORPUS, min_score=0.7)


def test_an_identical_node_is_the_existing_node(reconciler):
    found = reconciler.match(existing_node("doc:1", "Q13543883", "larval abundance"))
    assert found.existing_id == "camo:n_larvae"
    assert found.rule == "identity"
    assert found.score == 1.0


def test_the_same_measurement_qualified_differently_still_matches(reconciler):
    # The case exact-identity merging misses: same measurement, said at more
    # length. Token overlap alone scores this 0.67; containment carries it.
    found = reconciler.match(
        existing_node("doc:1", "Q13543883", "mosquito larval abundance")
    )
    assert found.existing_id == "camo:n_larvae"
    assert found.rule == "grounded"
    assert found.score >= 0.7


def test_a_morphological_variant_is_left_for_a_person(reconciler):
    """No suffix-stripping: "larvae" and "larval" stay two nodes, and it shows."""
    assert reconciler.match(
        existing_node("doc:1", "Q13543883", "abundance of larvae")
    ) is None


def test_a_different_measurement_is_a_different_node(reconciler):
    assert reconciler.match(
        existing_node("doc:1", "Q13543883", "wing length")
    ) is None


def test_opposite_qualifiers_are_never_merged(reconciler):
    """The polarity lives on the node: merging these would invert the evidence."""
    decreased = existing_node("doc:1", "Q13543883", "larval abundance",
                              qualifier="decreased")
    assert reconciler.match(decreased) is None


def test_a_different_entity_type_is_a_different_node(reconciler):
    assert reconciler.match(
        existing_node("doc:1", "Q13543883", "larval abundance",
                      entity_type="environmental_variable")
    ) is None


def test_two_different_grounded_terms_are_never_merged(reconciler):
    # A CURIE is an assertion; two of them assert two different things.
    assert reconciler.match(
        existing_node("doc:1", "Q14570773", "larval abundance")
    ) is None


def test_ungrounded_terms_still_reconcile_lexically():
    from graphingest.reconcile import NodeReconciler

    corpus = {"nodes": [existing_node("camo:n1", "sphagnum moss", "cover")], "edges": []}
    found = NodeReconciler(corpus).match(
        existing_node("doc:1", "Sphagnum moss", "percent cover")
    )  # "cover" is contained in "percent cover"
    assert found.existing_id == "camo:n1"
    assert found.rule == "lexical"


def test_reconciling_rewrites_the_edges_that_point_at_the_node(reconciler):
    document = {
        "nodes": [
            existing_node("doc:salinity", "PATO:0085001", "salinity",
                          entity_type="environmental_variable"),
            existing_node("doc:larvae", "Q13543883", "mosquito larval abundance"),
        ],
        "edges": [
            {"id": "doc:e1", "subject": "doc:salinity", "object": "doc:larvae",
             "predicate": "causes",
             "mediation": {"mediator_node_ids": ["doc:salinity"]},
             "comparator": {"comparator_node_id": "doc:larvae"}},
        ],
    }
    report = reconciler.reconcile(document)
    assert report.matched == 2
    edge = document["edges"][0]
    assert edge["subject"] == "camo:n_salinity"
    assert edge["object"] == "camo:n_larvae"
    assert edge["mediation"]["mediator_node_ids"] == ["camo:n_salinity"]
    assert edge["comparator"]["comparator_node_id"] == "camo:n_larvae"


def test_two_nodes_matching_one_existing_node_are_collapsed(reconciler):
    """One paper often names the same measurement twice; the ids must stay unique."""
    document = {
        "nodes": [
            dict(existing_node("doc:a", "Q13543883", "larval abundance"),
                 source_spans=[{"text": "first mention"}]),
            dict(existing_node("doc:b", "Q13543883", "mosquito larval abundance"),
                 source_spans=[{"text": "second mention"}]),
        ],
        "edges": [{"id": "doc:e1", "subject": "doc:a", "object": "doc:b",
                   "predicate": "causes"}],
    }
    report = reconciler.reconcile(document)
    assert report.collapsed == 1
    assert len(document["nodes"]) == 1
    assert document["nodes"][0]["id"] == "camo:n_larvae"
    # The evidence from both mentions survives on the surviving node.
    texts = {span["text"] for span in document["nodes"][0]["source_spans"]}
    assert texts == {"first mention", "second mention"}


def test_a_reconciled_document_merges_without_duplicating(profile):
    """The point of the whole step: one node afterwards, not two."""
    from graphingest.reconcile import NodeReconciler

    document = {
        "graph_id": "camo:doc", "schema_version": "0.7.9", "provenance": {},
        "nodes": [existing_node("doc:larvae", "Q13543883", "mosquito larval abundance")],
        "edges": [],
    }
    NodeReconciler(CORPUS).reconcile(document)
    merged, report = merge_graphs([document], "0.7.9", existing=CORPUS)
    assert report["nodes_out"] == len(CORPUS["nodes"])
    assert len({node["id"] for node in merged["nodes"]}) == len(merged["nodes"])


def test_without_reconciling_the_same_node_arrives_twice(profile):
    """Shows what reconciliation is for: identity merging alone does not catch it."""
    document = {
        "graph_id": "camo:doc", "schema_version": "0.7.9", "provenance": {},
        "nodes": [existing_node("doc:larvae", "Q13543883", "mosquito larval abundance")],
        "edges": [],
    }
    _, report = merge_graphs([document], "0.7.9", existing=CORPUS)
    assert report["nodes_out"] == len(CORPUS["nodes"]) + 1


def test_an_empty_corpus_reconciles_nothing_and_changes_nothing():
    from graphingest.reconcile import NodeReconciler

    reconciler = NodeReconciler({"nodes": []})
    assert not reconciler
    document = {"nodes": [existing_node("doc:1", "Q1", "abundance")], "edges": []}
    report = reconciler.reconcile(document)
    assert report.matched == 0
    assert document["nodes"][0]["id"] == "doc:1"


def test_attribute_similarity_accepts_qualification_but_not_a_different_measure():
    from graphingest.reconcile import attribute_similarity

    assert attribute_similarity("larval abundance", "mosquito larval abundance") >= 0.7
    assert attribute_similarity("salinity", "soil salinity") >= 0.7
    # abundance and density are different measurements of the same thing.
    assert attribute_similarity("larval abundance", "larval density") < 0.7
    assert attribute_similarity("abundance", "wing length") == 0.0


# ---------------------------------------------------------------------------
# The Claude (Anthropic) provider
# ---------------------------------------------------------------------------


class FakeAnthropicMessages:
    """Stands in for client.messages, recording what it was sent."""

    def __init__(self, reject_temperature: bool):
        self.reject_temperature = reject_temperature
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_temperature and "temperature" in kwargs:
            # The shape current Claude models return: sampling parameters were
            # removed, and naming one is a 400.
            raise RuntimeError(
                "Error code: 400 - {'type': 'invalid_request_error', 'message': "
                "'temperature: Extra inputs are not permitted'}"
            )

        class Block:
            type = "tool_use"
            input = {"nodes": [], "edges": []}

        class Response:
            content = [Block()]

        return Response()


class FakeAnthropicClient:
    def __init__(self, reject_temperature: bool = True):
        self.messages = FakeAnthropicMessages(reject_temperature)


def anthropic_client(monkeypatch, reject_temperature=True):
    """An LLMClient on the anthropic path, wired to a fake SDK client."""
    import sys
    import types

    from graphingest.llm_client import LLMClient, LLMSettings

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=object))
    settings = LLMSettings(
        provider="anthropic", model="claude-opus-5", api_key="sk-ant-test",
        structured_output="tool_use", max_tokens=32000, temperature=0.1,
    )
    client = LLMClient(settings)
    client._client = FakeAnthropicClient(reject_temperature)
    return client


def test_claude_request_carries_the_schema_as_a_forced_tool(monkeypatch):
    client = anthropic_client(monkeypatch, reject_temperature=False)
    client.complete_json("system", "user", {"type": "object"}, schema_in_prompt=False)

    sent = client._client.messages.calls[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["max_tokens"] == 32000
    assert sent["tools"][0]["input_schema"] == {"type": "object"}
    # Forced, not optional: the one call the model may make is the emitter.
    assert sent["tool_choice"]["type"] == "tool"
    assert sent["tool_choice"]["name"] == sent["tools"][0]["name"]


def test_temperature_is_dropped_when_the_model_refuses_it(monkeypatch):
    """claude-opus-5 removed the sampling parameters; sending one is a 400."""
    client = anthropic_client(monkeypatch)
    result = client.complete_json("system", "user", {"type": "object"},
                                  schema_in_prompt=False)

    calls = client._client.messages.calls
    assert len(calls) == 2, "should have retried once"
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
    assert result == {"nodes": [], "edges": []}


def test_the_refusal_is_remembered_for_the_rest_of_the_run(monkeypatch):
    client = anthropic_client(monkeypatch)
    client.complete_json("system", "user", {"type": "object"}, schema_in_prompt=False)
    client.complete_json("system", "another", {"type": "object"}, schema_in_prompt=False)

    calls = client._client.messages.calls
    # One wasted request in the whole run, not one per document.
    assert sum(1 for call in calls if "temperature" in call) == 1
    assert len(calls) == 3


def test_temperature_survives_where_the_model_accepts_it(monkeypatch):
    client = anthropic_client(monkeypatch, reject_temperature=False)
    client.complete_json("system", "user", {"type": "object"}, schema_in_prompt=False)

    calls = client._client.messages.calls
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.1


def test_an_unrelated_error_is_not_mistaken_for_a_sampling_refusal(monkeypatch):
    from graphingest.llm_client import LLMError

    client = anthropic_client(monkeypatch)
    client._client.messages.create = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("Error code: 529 - overloaded_error")
    )
    with pytest.raises(LLMError):
        client.complete_json("system", "user", {"type": "object"},
                             schema_in_prompt=False)
    assert client._sampling_refused is False


def test_sampling_rejection_detector():
    from graphingest.llm_client import _is_sampling_rejection

    assert _is_sampling_rejection(
        RuntimeError("temperature: Extra inputs are not permitted")
    )
    assert _is_sampling_rejection(
        RuntimeError("top_p is not supported for this model")
    )
    # A rate limit is not a parameter problem, and must not silently retry.
    assert not _is_sampling_rejection(RuntimeError("rate_limit_error: slow down"))
    # Nor is a model that merely mentions the word in passing.
    assert not _is_sampling_rejection(RuntimeError("temperature reading complete"))


def test_anthropic_without_a_key_fails_before_any_request():
    from graphingest.llm_client import LLMSettings

    with pytest.raises(ValueError, match="api_key"):
        LLMSettings(provider="anthropic", api_key="").validate()


def test_the_shipped_config_keeps_claude_commented_out():
    """The block is documentation until someone chooses it."""
    from graphingest.config import DEFAULT_PIPELINE_CONFIG
    from graphingest.llm_client import LLMSettings

    text = DEFAULT_PIPELINE_CONFIG.read_text(encoding="utf-8")
    assert "# provider: anthropic" in text, "the Claude block should be present"
    assert "claude-opus-5" in text
    assert LLMSettings.from_config().provider == "openai_compatible", "and not active"
