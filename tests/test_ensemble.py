"""Tests for cross-model agreement.

Offline: the primary and the witnesses are fakes that answer in the
OpenAI/vLLM shape (shared with the confidence tests). What is worth testing
is the matching (by meaning, qualifier compared rather than required), the
verdicts, that the saved graph stays the primary's, and that a witness that
fails or has already run costs nothing it should not.
"""

from __future__ import annotations

import copy
import json

import pytest

from graphingest.annotate import annotate_documents
from graphingest.confidence import ConfidenceSettings
from graphingest.ensemble import (
    EnsembleSettings,
    Witness,
    compare_graphs,
    corpus_report,
    ensemble_from_config,
    finalize,
    node_similarity,
    witness_settings,
    write_corpus_report,
)
from graphingest.graph_io import load_graph
from graphingest.llm_client import LLMSettings
from graphingest.merge import collect_paths, merge_graphs
from graphingest.reconcile import NodeReconciler
from graphingest.schema import DEFAULT_SCHEMA, build_extraction_profile
from tests.test_confidence import (
    ARTICLE,
    REPLIES,
    SALINITY,
    FakeCompletions,
    fake_client,
)


@pytest.fixture(scope="module")
def profile():
    return build_extraction_profile(DEFAULT_SCHEMA)


PRIMARY = LLMSettings(provider="openai_compatible", endpoint="http://brine/v1",
                      model="qwen", api_key="k", max_tokens=16384, logprobs=True,
                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_a_witness_on_the_same_host_needs_only_a_model():
    settings = witness_settings(PRIMARY, {"name": "llama", "model": "llama"})
    assert (settings.endpoint, settings.api_key, settings.max_tokens) == \
        ("http://brine/v1", "k", 16384)
    # Compared, not scored; and a Qwen switch means nothing to Llama.
    assert settings.logprobs is False and settings.extra_body == {}


def test_a_witness_on_another_provider_inherits_no_credentials():
    with pytest.raises(ValueError, match="api_key"):
        witness_settings(PRIMARY, {"provider": "anthropic", "model": "claude-sonnet-5"})
    settings = witness_settings(PRIMARY, {"provider": "anthropic", "api_key": "sk-ant",
                                          "model": "claude-sonnet-5"})
    assert settings.structured_output == "tool_use" and settings.api_key == "sk-ant"


def test_witnesses_come_from_config_or_straight_from_the_command_line(tmp_path):
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        "ensemble:\n  match_min_score: 0.8\n  witnesses:\n"
        "    - {name: llama, model: meta/llama, max_chunk_characters: 9000}\n",
        encoding="utf-8",
    )
    configured = ensemble_from_config(config, PRIMARY)
    assert [(w.name, w.settings.model, w.max_chunk_characters)
            for w in configured.witnesses] == [("llama", "meta/llama", 9000)]
    assert configured.match_min_score == 0.8
    ad_hoc = ensemble_from_config(config, PRIMARY, ["openai/models/Gemma-3"])
    assert ad_hoc.witnesses[0].name == "openai_models_Gemma-3"
    assert ad_hoc.witnesses[0].settings.endpoint == "http://brine/v1"

    config.write_text("ensemble:\n  witnesses: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one witness"):
        ensemble_from_config(config, PRIMARY)


def test_the_shipped_config_leaves_the_ensemble_off():
    from graphingest.ensemble import ensemble_enabled_in_config

    assert ensemble_enabled_in_config(None) is False


# ---------------------------------------------------------------------------
# Matching and verdicts
# ---------------------------------------------------------------------------


def node(identifier, term, attribute, qualifier, entity_type="environmental_variable"):
    return {"id": identifier, "name": f"{qualifier} {attribute or term}",
            "entity_term": term, "measured_attribute": attribute,
            "state_or_change_qualifier": qualifier, "entity_type": entity_type}


def edge(identifier, subject, obj, strength="direct_causal", sentence="s"):
    return {"id": identifier, "subject": subject, "object": obj, "predicate": "causes",
            "claim_strength": strength, "original_sentence": sentence}


PLUGGING = node("p1", "ditch plugging", None, "present", "management_intervention")
SALT = node("p2", "salinity", "salinity", "increased")
LARVAE = node("p3", "mosquito larvae", "larval abundance", "decreased")
PRIMARY_GRAPH = {"nodes": [PLUGGING, SALT, LARVAE],
                 "edges": [edge("e1", "p1", "p2"), edge("e2", "p2", "p3")]}


def witness_graph(**changes):
    graph = copy.deepcopy(PRIMARY_GRAPH)
    for n in graph["nodes"]:
        n["id"] = "w" + n["id"]
    for e in graph["edges"]:
        e["subject"], e["object"] = "w" + e["subject"], "w" + e["object"]
    for key, value in changes.items():
        kind, index, field = key.split("__")
        graph[kind][int(index)][field] = value
    return graph


def verdicts(result, answered, kind, identifier):
    graph_result = finalize(result, PRIMARY_GRAPH,
                            [{"name": n, "status": "extracted"} for n in answered],
                            EnsembleSettings(witnesses=[]), {})
    return graph_result[kind][identifier]


def test_matching_is_by_meaning_and_ignores_what_is_compared():
    # More words for the same measurement still matches...
    assert node_similarity(SALT, node("x", "salinity", "salinity levels", "increased")) >= 0.7
    # ...and so does the opposite qualifier: that is the disagreement to find.
    assert node_similarity(SALT, node("x", "salinity", "salinity", "decreased")) == 1.0
    # Folding the measurement into the term is the same variable.
    assert node_similarity(LARVAE, node("x", "mosquito larval abundance", None,
                                        "decreased")) >= 0.7
    # Two different CURIEs are two different things.
    assert node_similarity(node("a", "Q1", None, "present"),
                           node("b", "Q2", None, "present")) == 0.0


def test_an_identical_witness_agrees_with_everything():
    result = compare_graphs(PRIMARY_GRAPH, {"llama": witness_graph()},
                            EnsembleSettings(witnesses=[]))
    for kind, ids in (("nodes", ["p1", "p2", "p3"]), ("edges", ["e1", "e2"])):
        for identifier in ids:
            entry = verdicts(result, ["llama"], kind, identifier)
            assert entry["status"] == "agreed" and not entry["flagged"]


def test_an_opposite_qualifier_is_a_conflict_on_the_node_and_its_claims():
    result = compare_graphs(
        PRIMARY_GRAPH, {"gemma": witness_graph(nodes__1__state_or_change_qualifier="decreased")},
        EnsembleSettings(witnesses=[]),
    )
    salt = verdicts(result, ["gemma"], "nodes", "p2")
    assert salt["status"] == "conflict" and salt["flagged"]
    assert salt["witnesses"]["gemma"]["conflicts"] == [
        {"field": "state_or_change_qualifier", "primary": "increased", "witness": "decreased"}
    ]
    # Both claims touching salinity now say something different.
    for identifier, role in (("e1", "object"), ("e2", "subject")):
        conflicts = verdicts(result, ["gemma"], "edges", identifier)["witnesses"]["gemma"]["conflicts"]
        assert conflicts[0]["field"] == f"{role}.state_or_change_qualifier"


def test_claim_strength_and_direction_disagreements():
    reversed_ = witness_graph(edges__1__claim_strength="associational")
    reversed_["edges"][0]["subject"], reversed_["edges"][0]["object"] = "wp2", "wp1"
    result = compare_graphs(PRIMARY_GRAPH, {"llama": reversed_},
                            EnsembleSettings(witnesses=[]))
    e1 = verdicts(result, ["llama"], "edges", "e1")["witnesses"]["llama"]
    e2 = verdicts(result, ["llama"], "edges", "e2")["witnesses"]["llama"]
    assert e1["conflicts"][0]["field"] == "direction"
    assert e2["conflicts"] == [{"field": "claim_strength", "primary": "direct_causal",
                                "witness": "associational"}]


def test_missing_partial_and_unsupported():
    silent = {"nodes": [], "edges": []}
    result = compare_graphs(PRIMARY_GRAPH, {"llama": witness_graph(), "mute": silent},
                            EnsembleSettings(witnesses=[]))
    entry = verdicts(result, ["llama", "mute"], "edges", "e1")
    assert (entry["status"], entry["support"]) == ("partial", "1/2")
    assert verdicts(result, ["mute"], "edges", "e1")["status"] == "unsupported"
    # A witness that failed does not count against anything.
    assert verdicts(result, [], "edges", "e1")["status"] == "unchecked"


def test_what_the_witnesses_found_and_the_primary_did_not():
    extra = edge("w9", "wp1", "wp3", sentence="Plugging reduced larvae.")
    first, second = witness_graph(), witness_graph()
    first["edges"].append(extra)
    second["edges"].append(copy.deepcopy(extra))
    result = compare_graphs(PRIMARY_GRAPH, {"llama": first, "gemma": second},
                            EnsembleSettings(witnesses=[]))
    (omission,) = result["omissions"]
    assert sorted(omission["witnesses"]) == ["gemma", "llama"]
    # The primary has both nodes; only the relation is missing.
    assert omission["endpoints_in_primary"]
    assert (omission["subject"], omission["object"]) == ("p1", "p3")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def flipped_replies():
    """A witness that reads the second finding the other way round."""
    replies = copy.deepcopy(REPLIES)
    replies[1][1]["nodes"][1]["state_or_change_qualifier"] = "increased"
    return [(marker, reply, {}) for marker, reply, _ in replies]


class Unreachable:
    def create(self, **kwargs):
        raise RuntimeError("503 - service unavailable")


def witnesses(**completions) -> EnsembleSettings:
    made = []
    for name, fake in completions.items():
        witness = Witness(name=name, settings=LLMSettings(
            provider="openai_compatible", endpoint="http://fake/v1", model=name,
            structured_output="json_object"))
        witness._client = fake_client(fake, logprobs=False)
        made.append(witness)
    return EnsembleSettings(witnesses=made)


def annotate(tmp_path, profile, ensemble, confidence=None, reconciler=None, force=False):
    article = tmp_path / "kramer.md"
    article.write_text(ARTICLE, encoding="utf-8")
    primary = fake_client(FakeCompletions(REPLIES, reasoning=""),
                          logprobs=confidence is not None)
    return annotate_documents(
        [{"slug": "kramer", "document_id": "doi:10.1/kramer",
          "markdown_path": str(article)}],
        profile, primary, tmp_path / "graphs", max_chunk_characters=300, overlap=0,
        confidence=confidence, ensemble=ensemble, reconciler=reconciler, force=force,
    )


def sidecar(tmp_path):
    return json.loads((tmp_path / "graphs" / "kramer.agreement.json")
                      .read_text(encoding="utf-8"))


def test_the_graph_stays_the_primarys_and_disagreements_are_flagged(tmp_path, profile):
    ensemble = witnesses(same=FakeCompletions(REPLIES),
                         flipped=FakeCompletions(flipped_replies()),
                         down=Unreachable())
    (result,) = annotate(tmp_path, profile, ensemble)
    assert result["status"] == "annotated"

    alone = tmp_path / "alone"
    alone.mkdir()
    annotate(alone, profile, None)
    assert load_graph(tmp_path / "graphs" / "kramer.yaml")["nodes"] == \
        load_graph(alone / "graphs" / "kramer.yaml")["nodes"]

    graph = load_graph(tmp_path / "graphs" / "kramer.yaml")
    agreement = sidecar(tmp_path)
    assert set(agreement["nodes"]) == {n["id"] for n in graph["nodes"]}
    assert set(agreement["edges"]) == {e["id"] for e in graph["edges"]}
    assert agreement["summary"]["checked_by"] == ["same", "flipped"]
    assert agreement["summary"]["failed"] == ["down"]

    by_name = {entry["name"]: entry for entry in agreement["nodes"].values()}
    assert by_name["fewer larvae"]["status"] == "conflict"
    assert by_name["increased salinity"]["status"] == "agreed"
    edges = {entry["label"]: entry for entry in agreement["edges"].values()}
    assert edges["ditch plugging --causes--> increased salinity"]["status"] == "agreed"
    disputed = edges["increased salinity --causes--> fewer larvae"]
    assert disputed["status"] == "conflict" and disputed["flagged"]
    assert disputed["support"] == "2/2"
    assert result["agreement_flagged_edges"] == 1

    for name in ("same", "flipped"):
        assert (tmp_path / "graphs" / "witnesses" / name / "kramer.yaml").exists()
    report = json.loads((tmp_path / "graphs" / "kramer.report.json").read_text(encoding="utf-8"))
    assert report["agreement"]["flagged_edges"] == 1


def test_a_finished_corpus_can_be_checked_later_without_re_extracting(tmp_path, profile):
    annotate(tmp_path, profile, None)
    same = FakeCompletions(REPLIES)
    (result,) = annotate(tmp_path, profile, witnesses(same=same))
    assert result["status"] == "skipped_existing"
    assert sidecar(tmp_path)["summary"]["edges"]["agreed"] == 2
    calls = len(same.calls)

    # Checked already: nothing to do.
    annotate(tmp_path, profile, witnesses(same=same))
    assert len(same.calls) == calls
    # Sidecar gone, witness graph kept: rebuilt from disk, no model call.
    (tmp_path / "graphs" / "kramer.agreement.json").unlink()
    annotate(tmp_path, profile, witnesses(same=same))
    assert len(same.calls) == calls
    assert sidecar(tmp_path)["witnesses"][0]["status"] == "reused"


def test_verdicts_follow_nodes_that_reconciliation_renamed(tmp_path, profile):
    existing = {"nodes": [{"id": "camo:node_corpus_salinity", **SALINITY}], "edges": []}
    annotate(tmp_path, profile, witnesses(same=FakeCompletions(REPLIES)),
             reconciler=NodeReconciler(existing))
    graph = load_graph(tmp_path / "graphs" / "kramer.yaml")
    assert "camo:node_corpus_salinity" in {n["id"] for n in graph["nodes"]}
    assert sidecar(tmp_path)["nodes"]["camo:node_corpus_salinity"]["status"] == "agreed"


def test_the_reviewer_summary_combines_both_signals(tmp_path, profile):
    ensemble = witnesses(flipped=FakeCompletions(flipped_replies()))
    annotate(tmp_path, profile, ensemble, confidence=ConfidenceSettings(enabled=True))
    markdown = write_corpus_report(tmp_path / "graphs", tmp_path, ensemble) \
        .read_text(encoding="utf-8")
    assert "## Flagged claims" in markdown
    assert "object.state_or_change_qualifier: decreased vs flipped increased" in markdown
    assert "## Agreement against token confidence" in markdown
    report = corpus_report(tmp_path / "graphs", ensemble)
    assert report["witnesses"]["flipped"]["agreement_rate"] == 0.5
    assert report["disputed"][0]["confidence"] == "low"


def test_agreement_sidecars_and_witness_graphs_stay_out_of_the_merge(tmp_path, profile):
    from graphingest.confidence import write_merged_index

    annotate(tmp_path, profile, witnesses(same=FakeCompletions(REPLIES)))
    paths = collect_paths([tmp_path / "graphs"])
    assert [path.name for path in paths] == ["kramer.yaml"]
    maps: dict = {}
    merged, _ = merge_graphs([load_graph(paths[0])], profile.version, id_maps=maps)
    written = write_merged_index(paths, maps, False, tmp_path)
    assert [path.name for path in written] == ["agreement_index.json"]
    index = json.loads(written[0].read_text(encoding="utf-8"))
    assert set(index["edges"]) == {e["id"] for e in merged["edges"]}
    assert not any(entry["flagged"] for entry in index["edges"].values())
