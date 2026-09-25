"""Tests for token-level extraction confidence.

Offline, like the rest: the endpoint is a fake that speaks the OpenAI/vLLM
response shape, with a toy tokenizer. What is worth testing is that each
token lands on the right field, that the scores follow the items through the
id rewrites, and that an endpoint without logprobs degrades rather than fails.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from graphingest.annotate import annotate_documents
from graphingest.confidence import (
    ConfidenceSettings,
    GenerationTrace,
    TokenLogprob,
    align_tokens,
    corpus_report,
    enum_slot_names,
    locate_json_object,
    render_markdown,
    score_extraction,
    tokens_from_prompt_logprobs,
    write_corpus_report,
)
from graphingest.consolidate import Consolidator
from graphingest.graph_io import load_graph
from graphingest.llm_client import LLMClient, LLMSettings
from graphingest.merge import collect_paths, merge_graphs
from graphingest.schema import DEFAULT_SCHEMA, build_extraction_profile


@pytest.fixture(scope="module")
def profile():
    return build_extraction_profile(DEFAULT_SCHEMA)


# ---------------------------------------------------------------------------
# A fake vLLM
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|\s+|[^\w\s]", text)


def fake_tokens(text: str, doubts: dict | None = None, skip: int = 0) -> list:
    """OpenAI-shaped logprob entries; ``doubts`` makes named tokens uncertain.

    ``doubts`` maps a token to ``(p, [(alternative, p), ...])`` and applies to
    its first occurrence at or after character ``skip`` — so a word the model
    already wrote in its reasoning is not the one being doubted.
    """
    doubts = dict(doubts or {})
    entries, position = [], 0
    for piece in tokenize(text):
        p, alternatives = 0.99, []
        if position >= skip and piece in doubts:
            p, alternatives = doubts.pop(piece)
        top = [(piece, p)] + alternatives
        entries.append(SimpleNamespace(
            token=piece, logprob=math.log(p), bytes=list(piece.encode("utf-8")),
            top_logprobs=[SimpleNamespace(token=t, logprob=math.log(q)) for t, q in top],
        ))
        position += len(piece)
    return entries


def prompt_entries(prompt: str, strange: str = "") -> list:
    """vLLM's prompt_logprobs: None first, then {id: {logprob, rank, decoded}}."""
    entries: list = [None]
    for index, piece in enumerate(tokenize(prompt)[1:], 1):
        logprob = -9.0 if strange and piece in strange.split() else -1.0
        entries.append({str(1000 + index): {"logprob": logprob, "rank": 3,
                                            "decoded_token": piece},
                        "7": {"logprob": -0.1, "rank": 1, "decoded_token": "the"}})
    return entries


class FakeCompletions:
    def __init__(self, replies, reasoning="", refuse=None, strange=""):
        self.replies = replies  # [(marker in prompt, reply dict, doubts)]
        self.reasoning = reasoning
        self.refuse = refuse
        self.strange = strange
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.refuse and (self.refuse in kwargs
                            or self.refuse in (kwargs.get("extra_body") or {})):
            raise RuntimeError(f"400 - {self.refuse} is not supported by this server")
        user = kwargs["messages"][-1]["content"]
        reply, doubts = next((r, d) for marker, r, d in self.replies if marker in user)
        text = json.dumps(reply, indent=1)
        streamed = self.reasoning + text
        logprobs = (
            SimpleNamespace(content=fake_tokens(streamed, doubts, len(self.reasoning)))
            if kwargs.get("logprobs") else None
        )
        prompt = "".join(message["content"] for message in kwargs["messages"])
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason="stop", logprobs=logprobs,
            )],
            prompt_logprobs=(
                prompt_entries(prompt, self.strange)
                if "prompt_logprobs" in (kwargs.get("extra_body") or {}) else None
            ),
        )


def fake_client(completions, **settings) -> LLMClient:
    defaults = dict(provider="openai_compatible", endpoint="http://fake/v1",
                    model="qwen-test", structured_output="json_object",
                    logprobs=True, top_logprobs=5)
    client = LLMClient(LLMSettings(**{**defaults, **settings}))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client


ARTICLE = (
    "# Methods\n\n" + "Salinity rose after ditch plugging. " * 6
    + "\n\n# Results\n\n" + "Larval abundance fell as salinity rose. " * 6
)

SALINITY = {"name": "increased salinity", "entity_type": "environmental_variable",
            "entity_term": "salinity", "measured_attribute": "salinity",
            "state_or_change_qualifier": "increased"}

METHODS_REPLY = {
    "nodes": [
        {"id": "n1", "name": "ditch plugging", "entity_type": "management_intervention",
         "entity_term": "ditch plugging", "state_or_change_qualifier": "present"},
        {"id": "n2", **SALINITY},
    ],
    "edges": [
        {"id": "e1", "subject": "n1", "predicate": "causes", "object": "n2",
         "claim_strength": "direct_causal", "philosophical_accounts": ["interventionist"],
         "original_sentence": "Salinity rose after ditch plugging."},
    ],
}
RESULTS_REPLY = {
    "nodes": [
        {"id": "n1", **SALINITY},
        {"id": "n2", "name": "fewer larvae",
         "entity_type": "environmental_variable", "entity_term": "mosquito larvae",
         "measured_attribute": "larval abundance",
         "state_or_change_qualifier": "decreased"},
    ],
    "edges": [
        {"id": "e1", "subject": "n1", "predicate": "causes", "object": "n2",
         "claim_strength": "direct_causal", "philosophical_accounts": ["interventionist"],
         "original_sentence": "Larval abundance fell as salinity rose."},
    ],
}
REPLIES = [
    # The model is torn on how strong the first claim is...
    ("Salinity rose after ditch plugging.", METHODS_REPLY,
     {"direct_causal": (0.5, [("associational", 0.45)])}),
    # ...and on which way larval abundance went in the second.
    ("Larval abundance fell", RESULTS_REPLY,
     {"decreased": (0.55, [("increased", 0.4)])}),
]


def run_annotation(tmp_path, profile, completions, prompt_logprobs=None, **client):
    article = tmp_path / "kramer.md"
    article.write_text(ARTICLE, encoding="utf-8")
    out_dir = tmp_path / "graphs"
    llm = fake_client(completions, prompt_logprobs=prompt_logprobs, **client)
    settings = ConfidenceSettings(enabled=True, prompt_logprobs=prompt_logprobs)
    results = annotate_documents(
        [{"slug": "kramer", "document_id": "doi:10.1/kramer",
          "markdown_path": str(article)}],
        profile, llm, out_dir, max_chunk_characters=300, overlap=0,
        confidence=settings,
    )
    return results, out_dir, settings


# ---------------------------------------------------------------------------
# Locating the answer
# ---------------------------------------------------------------------------


def test_json_values_are_located_past_reasoning_and_fences():
    text = ('<think>draft: {"nodes": []}</think>\n```json\n'
            '{"nodes": [{"id": "n1", "entity_term": "salt marsh"}], "edges": []}\n```')
    value, spans = locate_json_object(text)
    assert value["nodes"][0]["entity_term"] == "salt marsh"
    start, end = spans[("nodes", 0, "entity_term")]
    assert text[start:end] == '"salt marsh"'


def test_bytes_keep_a_split_character_aligned():
    # "µ" is two bytes; a byte-level tokenizer can emit them as two tokens.
    tokens = [TokenLogprob("�", -1.0, raw=b"\xc2"),
              TokenLogprob("�", -1.0, raw=b"\xb5"),
              TokenLogprob("m", -0.1, raw=b"m"),
              TokenLogprob(" 5", -0.1, raw=b" 5")]
    spans = align_tokens(tokens, "µm 5")
    assert spans == [(0, 1), (0, 1), (1, 2), (2, 4)]


def test_without_bytes_the_aligner_resynchronises():
    text = "salinity µm rose after plugging, measured at the outflow weir"
    pieces = ["salinity", " ", "�", "�", "m", " rose", " after",
              " plugging", ",", " measured", " at", " the", " outflow", " weir"]
    tokens = [TokenLogprob(piece, -0.1) for piece in pieces]
    spans = align_tokens(tokens, text)
    assert text[slice(*spans[5])] == " rose"
    assert text[slice(*spans[-1])] == " weir"


def test_prompt_token_is_read_by_id_then_by_position():
    response = SimpleNamespace(
        prompt_logprobs=[None,
                         {"5": {"logprob": -0.2, "rank": 1, "decoded_token": "a"},
                          "9": {"logprob": -3.0, "rank": 4, "decoded_token": "b"}}],
        prompt_token_ids=[1, 9],
    )
    tokens = tokens_from_prompt_logprobs(response)
    assert tokens[1].token == "b" and tokens[1].logprob == -3.0
    response.prompt_token_ids = None
    assert tokens_from_prompt_logprobs(response)[1].token == "a"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_a_doubted_enum_names_its_alternative(profile):
    text = json.dumps(RESULTS_REPLY, indent=1)
    trace = GenerationTrace(text, [
        TokenLogprob(e.token, e.logprob, [(t.token, t.logprob) for t in e.top_logprobs],
                     bytes(e.bytes))
        for e in fake_tokens(text, {"decreased": (0.55, [("increased", 0.4)])})
    ])
    scored = score_extraction(trace, RESULTS_REPLY, enum_slot_names(profile),
                              ConfidenceSettings())
    assert scored["available"]
    doubted = scored["nodes"][1]
    assert doubted["weakest_field"] == "state_or_change_qualifier"
    assert doubted["score"] == pytest.approx(0.55, abs=0.01)
    qualifier = doubted["fields"]["state_or_change_qualifier"]
    assert qualifier["kind"] == "decision"
    assert [a["token"] for a in qualifier["pivot"]["alternatives"]] == ["decreased", "increased"]
    # Syntax is not scored: the id and the quotes around values are not fields.
    assert "id" not in doubted["fields"]
    assert scored["nodes"][0]["score"] > 0.9


def test_a_reply_that_does_not_match_its_tokens_is_not_scored(profile):
    trace = GenerationTrace('{"nodes": []}', [TokenLogprob('{"nodes": []}', -0.1)])
    scored = score_extraction(trace, {"nodes": [1]}, set(), ConfidenceSettings())
    assert not scored["available"]
    assert "located" in scored["reason"]


# ---------------------------------------------------------------------------
# End to end: through normalization, consolidation and onto disk
# ---------------------------------------------------------------------------


def test_scores_follow_items_to_the_ids_the_graph_carries(tmp_path, profile):
    completions = FakeCompletions(REPLIES, reasoning="<think>direct_causal, decreased</think>")
    results, out_dir, _ = run_annotation(tmp_path, profile, completions)
    assert results[0]["status"] == "annotated"

    graph = load_graph(out_dir / "kramer.yaml")
    sidecar = json.loads((out_dir / "kramer.confidence.json").read_text(encoding="utf-8"))
    assert set(sidecar["nodes"]) == {node["id"] for node in graph["nodes"]}
    assert set(sidecar["edges"]) == {edge["id"] for edge in graph["edges"]}
    # The graph itself is untouched by any of this.
    assert all("confidence" not in json.dumps(node) for node in graph["nodes"])

    by_name = {entry["name"]: entry for entry in sidecar["nodes"].values()}
    # Extracted from both chunks, merged into one node, both extractions kept.
    assert len(by_name["increased salinity"]["observations"]) == 2
    assert by_name["increased salinity"]["bucket"] == "high"
    assert by_name["fewer larvae"]["bucket"] == "low"

    edges = {entry["label"]: entry for entry in sidecar["edges"].values()}
    strength = edges["ditch plugging --causes--> increased salinity"]
    assert strength["bucket"] == "low" and strength["limited_by"] == "edge"
    # A confident arrow into an uncertain node is an uncertain claim.
    larvae = edges["increased salinity --causes--> fewer larvae"]
    assert larvae["edge_score"] > 0.9
    assert larvae["limited_by"] == "object" and larvae["bucket"] == "low"

    report = json.loads((out_dir / "kramer.report.json").read_text(encoding="utf-8"))
    assert report["confidence"]["buckets"]["edges"]["low"] == 2
    assert report["confidence"]["reasoning_tokens"] > 0
    assert completions.calls[0]["logprobs"] is True
    assert completions.calls[0]["top_logprobs"] == 5


def test_the_reviewer_summary(tmp_path, profile):
    _, out_dir, settings = run_annotation(tmp_path, profile, FakeCompletions(REPLIES))
    written = write_corpus_report(out_dir, tmp_path, settings)
    markdown = written.read_text(encoding="utf-8")
    assert "Review these claims first" in markdown
    # The second claim is held down by its object node, and says why.
    assert "`object.state_or_change_qualifier` | decreased (0.55)" in markdown
    assert "`increased` 0.40" in markdown

    report = corpus_report(out_dir, settings)
    assert report["buckets"]["edges"]["low"] == 2
    assert report["weakest_fields"]["node.state_or_change_qualifier"] == 1
    assert render_markdown(report) == markdown


def test_prompt_logprobs_find_the_strange_passage(tmp_path, profile):
    completions = FakeCompletions(REPLIES, strange="Larval abundance")
    _, out_dir, _ = run_annotation(tmp_path, profile, completions, prompt_logprobs=0)
    assert completions.calls[0]["extra_body"] == {"prompt_logprobs": 0}
    summary = json.loads((out_dir / "kramer.confidence.json")
                         .read_text(encoding="utf-8"))["summary"]
    assert summary["article_perplexity"] > 1
    assert "Larval" in summary["most_surprising_passages"][0]["text"]


def test_an_endpoint_that_refuses_logprobs_still_extracts(tmp_path, profile):
    completions = FakeCompletions(REPLIES, refuse="logprobs")
    results, out_dir, _ = run_annotation(tmp_path, profile, completions)
    assert results[0]["status"] == "annotated"
    # Retried without logprobs, and without loosening the JSON constraint.
    assert completions.calls[1]["response_format"] == {"type": "json_object"}
    assert "logprobs" not in completions.calls[1]
    # And not asked again for the next chunk.
    assert "logprobs" not in completions.calls[-1]
    summary = json.loads((out_dir / "kramer.confidence.json")
                         .read_text(encoding="utf-8"))["summary"]
    assert not summary["available"]
    assert summary["buckets"]["edges"]["unscored"] == 2


def test_refusing_prompt_logprobs_keeps_the_output_side(tmp_path, profile):
    completions = FakeCompletions(REPLIES, refuse="prompt_logprobs")
    _, out_dir, _ = run_annotation(tmp_path, profile, completions, prompt_logprobs=0)
    assert completions.calls[-1]["logprobs"] is True
    assert "extra_body" not in completions.calls[-1]
    summary = json.loads((out_dir / "kramer.confidence.json")
                         .read_text(encoding="utf-8"))["summary"]
    assert summary["available"]


def test_anthropic_reports_why_there_is_nothing(tmp_path, profile, monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=object))
    client = LLMClient(LLMSettings(provider="anthropic", model="claude-opus-5",
                                   api_key="sk-ant-test", structured_output="tool_use",
                                   logprobs=True))

    class Messages:
        def create(self, **kwargs):
            block = SimpleNamespace(type="tool_use", input={"nodes": [], "edges": []})
            return SimpleNamespace(content=[block])

    client._client = SimpleNamespace(messages=Messages())
    client.complete_json("system", "user", {"type": "object"}, schema_in_prompt=False)
    assert client.last_trace is None
    assert "Anthropic" in client.trace_unavailable


# ---------------------------------------------------------------------------
# Following ids through the merge
# ---------------------------------------------------------------------------


def test_consolidation_reports_where_every_id_went():
    graph = {"nodes": [{"id": "n1", **SALINITY}, {"id": "n2", **SALINITY}],
             "edges": [{"id": "e1", "subject": "n1", "predicate": "causes",
                        "object": "n2", "original_sentence": "x"},
                       {"id": "e2", "subject": "n2", "predicate": "causes",
                        "object": "n1", "original_sentence": "x"}]}
    merged, report = Consolidator().consolidate([graph, graph])
    (only_node,) = merged["nodes"]
    assert report.node_id_maps[0] == {"n1": only_node["id"], "n2": only_node["id"]}
    (only_edge,) = merged["edges"]  # both edges are now the same self-loop
    assert report.edge_id_maps[1] == {"e1": only_edge["id"], "e2": only_edge["id"]}


def test_sidecars_are_not_graphs_and_are_rekeyed_after_the_merge(tmp_path, profile):
    from graphingest.confidence import write_merged_index

    _, out_dir, _ = run_annotation(tmp_path, profile, FakeCompletions(REPLIES))
    paths = collect_paths([out_dir])
    assert [path.name for path in paths] == ["kramer.yaml"]

    maps: dict = {}
    merged, _ = merge_graphs([load_graph(paths[0])], profile.version, id_maps=maps)
    (written,) = write_merged_index(paths, maps, False, tmp_path)
    index = json.loads(written.read_text(encoding="utf-8"))
    assert set(index["edges"]) == {edge["id"] for edge in merged["edges"]}
    assert all(entry["sources"][0]["document"] == "kramer"
               for entry in index["nodes"].values())


def test_the_shipped_config_leaves_confidence_off():
    assert LLMSettings.from_config().logprobs is False
    settings = ConfidenceSettings.from_config()
    assert settings.enabled is False
    assert settings.bucket(0.95) == "high" and settings.bucket(0.7) == "medium"
    assert settings.bucket(0.2) == "low" and settings.bucket(None) == "unscored"
