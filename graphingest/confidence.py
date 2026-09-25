"""Token-level confidence for every extracted node and edge.

Opt-in (``--confidence``), because it multiplies what a run writes to disk.
With it on, the client asks the endpoint for ``logprobs`` and ``top_logprobs``
on the generated tokens and, with ``--prompt-logprobs``, for ``prompt_logprobs``
on the input (a vLLM extension). This module turns those token streams into
something a reviewer can act on:

* **Per field.** The generated JSON is re-parsed with character positions, and
  every token is attributed to the field whose value it spells. A categorical
  field (an enum, a boolean, an endpoint reference) is scored by the joint
  probability of its value — "increased" at 0.55 — together with the
  alternatives the model weighed at its least certain token, "decreased" at
  0.41. That second half is the useful part: it names the confusion, not only
  that there was one. A free-text field is scored per token (the geometric
  mean), since a long phrase is not less likely to be right for being long.
* **Per annotation.** A node or edge takes the weakest of its *core* fields
  (``confidence.core_fields`` in ``config/pipeline.yaml``): the fields that
  make the claim what it is. Every other field is still recorded, but a model
  hesitating over ``reversibility`` is not a reason to doubt that A causes B.
  An edge's *claim* score is additionally capped by its two endpoint nodes: a
  confident arrow between uncertain nodes is an uncertain claim.
* **Per document and corpus.** Buckets (high / medium / low), the weakest
  claims with their alternatives, and which fields are most often weakest —
  that last one says which part of the prompt or schema is ambiguous.

Nothing here writes into the graph. The graph stays schema-valid and the
evidence sits beside it in ``<slug>.confidence.json``, keyed by the ids the
graph actually carries after consolidation and reconciliation have rewritten
them.

What these numbers are not: calibrated probabilities that a claim is true.
They are the model's own predictive uncertainty about the next token, useful
for ranking what to review first, and the bucket thresholds are a triage
convention. See the README for the caveats that matter.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import DEFAULT_LLM_CONFIG, load_yaml
from .graph_io import atomic_write, atomic_write_json

LOGGER = logging.getLogger("ingest.confidence")

#: OpenAI's ceiling on ``top_logprobs``, and vLLM's default ``--max-logprobs``.
MAX_TOP_LOGPROBS = 20

#: What a logprob of ``-inf`` (a token the grammar masked) is recorded as, so
#: every number in the report is finite JSON.
FLOOR_LOGPROB = -100.0

BUCKETS = ("high", "medium", "low", "unscored")

DEFAULT_CORE_FIELDS = {
    "node": ["entity_type", "entity_term", "measured_attribute",
             "state_or_change_qualifier"],
    "edge": ["subject", "predicate", "object", "negated", "claim_strength"],
}

#: Fields that are bookkeeping, not claims: the pipeline fills or overwrites
#: them, so the model's certainty about them says nothing about the extraction.
#: ``annotation_confidence`` is the model's *stated* confidence; it is reported
#: beside the token-derived score, not folded into it.
_EXCLUDED = {
    "id", "annotator", "annotation_confidence", "annotation_timestamp",
    "annotation_notes", "embedding_text", "embedding_vector", "variable_key",
    "start_char", "end_char", "section", "source_document",
}
#: Fields whose value is a choice among the nodes already written.
_REFERENCES = {"subject", "object", "mediator_node_ids", "moderator_node_ids",
               "comparator_node_id"}
#: Fields that should be copied verbatim from the article.
_QUOTES = {"original_sentence"}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceSettings:
    """The ``confidence:`` block of config/pipeline.yaml."""

    enabled: bool = False
    top_logprobs: int = 5
    #: ``None`` is off; an int asks vLLM for that many alternatives per prompt
    #: token (0 = the prompt token's own logprob only, which is all we use).
    prompt_logprobs: Optional[int] = None
    #: Keep every token, with its alternatives, per field. This is what makes
    #: the sidecar large; the scores and summaries do not need it.
    store_tokens: bool = True
    high: float = 0.9
    medium: float = 0.6
    core_fields: dict = field(default_factory=lambda: {
        kind: list(names) for kind, names in DEFAULT_CORE_FIELDS.items()
    })
    #: How many of the weakest claims each summary lists.
    review_count: int = 25

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "ConfidenceSettings":
        try:
            block = load_yaml(path or DEFAULT_LLM_CONFIG).get("confidence") or {}
        except FileNotFoundError:
            block = {}
        buckets = block.get("buckets") or {}
        core = block.get("core_fields") or {}
        settings = cls(
            enabled=bool(block.get("enabled", False)),
            top_logprobs=int(block.get("top_logprobs", cls.top_logprobs)),
            prompt_logprobs=block.get("prompt_logprobs"),
            store_tokens=bool(block.get("store_tokens", True)),
            high=float(buckets.get("high", cls.high)),
            medium=float(buckets.get("medium", cls.medium)),
            review_count=int(block.get("review_count", cls.review_count)),
        )
        for kind in ("node", "edge"):
            if core.get(kind):
                settings.core_fields[kind] = list(core[kind])
        settings.top_logprobs = max(0, min(settings.top_logprobs, MAX_TOP_LOGPROBS))
        return settings

    def bucket(self, score: Optional[float]) -> str:
        if score is None:
            return "unscored"
        if score >= self.high:
            return "high"
        if score >= self.medium:
            return "medium"
        return "low"

    def describe(self) -> dict:
        return {
            "top_logprobs": self.top_logprobs,
            "prompt_logprobs": self.prompt_logprobs,
            "thresholds": {"high": self.high, "medium": self.medium},
            "core_fields": self.core_fields,
        }


# ---------------------------------------------------------------------------
# Token streams
# ---------------------------------------------------------------------------


@dataclass
class TokenLogprob:
    token: str
    logprob: float
    #: The alternatives the server returned, most likely first.
    top: list[tuple[str, float]] = field(default_factory=list)
    #: Exact bytes, when the server sends them. A character split across two
    #: tokens decodes to U+FFFD in each; the bytes still add up.
    raw: Optional[bytes] = None


@dataclass
class GenerationTrace:
    """One completion: the text the caller parsed, and the tokens behind it."""

    text: str
    tokens: list[TokenLogprob]
    prompt_tokens: list[TokenLogprob] = field(default_factory=list)
    finish_reason: Optional[str] = None


def _get(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    value = getattr(item, name, None)
    if value is None:
        extra = getattr(item, "model_extra", None) or {}
        value = extra.get(name)
    return value


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return FLOOR_LOGPROB
    return number if math.isfinite(number) else FLOOR_LOGPROB


def tokens_from_openai(logprobs: Any) -> list[TokenLogprob]:
    """Tokens from an OpenAI-shaped ``choice.logprobs`` (vLLM, OpenAI, Ollama)."""
    content = _get(logprobs, "content") if logprobs is not None else None
    tokens: list[TokenLogprob] = []
    for item in content or []:
        raw = _get(item, "bytes")
        tokens.append(
            TokenLogprob(
                token=_get(item, "token") or "",
                logprob=_finite(_get(item, "logprob")),
                top=[
                    (_get(alt, "token") or "", _finite(_get(alt, "logprob")))
                    for alt in (_get(item, "top_logprobs") or [])
                ],
                raw=bytes(raw) if raw is not None else None,
            )
        )
    return tokens


def tokens_from_prompt_logprobs(response: Any) -> list[TokenLogprob]:
    """Tokens from vLLM's ``prompt_logprobs``: one entry per prompt token.

    Each entry maps token ids to ``{logprob, rank, decoded_token}`` and holds
    the prompt token itself plus the top-k. Which key is the prompt token is
    read from ``prompt_token_ids`` when the server sends it; otherwise it is
    the first key, which is where vLLM puts it. The first entry is ``None``:
    nothing precedes the first token, so it has no probability.
    """
    entries = _get(response, "prompt_logprobs")
    if not entries:
        return []
    ids = _get(response, "prompt_token_ids") or []
    tokens: list[TokenLogprob] = []
    for index, entry in enumerate(entries):
        if not entry:
            tokens.append(TokenLogprob("", 0.0))
            continue
        items = list(entry.items()) if isinstance(entry, dict) else []
        if not items:
            tokens.append(TokenLogprob("", 0.0))
            continue
        chosen = None
        if index < len(ids):
            chosen = entry.get(str(ids[index])) or entry.get(ids[index])
        if chosen is None:
            chosen = items[0][1]
        ranked = sorted(
            (value for _, value in items if _get(value, "rank") is not None),
            key=lambda value: _get(value, "rank"),
        )
        tokens.append(
            TokenLogprob(
                token=_get(chosen, "decoded_token") or "",
                logprob=_finite(_get(chosen, "logprob")),
                top=[(_get(value, "decoded_token") or "", _finite(_get(value, "logprob")))
                     for value in ranked],
            )
        )
    return tokens


# ---------------------------------------------------------------------------
# Aligning tokens to text
# ---------------------------------------------------------------------------


Span = Optional[tuple[int, int]]


def _joined(tokens: list[TokenLogprob]) -> tuple[str, list[tuple[int, int]]]:
    """The tokens' concatenated text, and each token's character span in it.

    With bytes for every token the spans are exact, multi-byte characters
    split across tokens included. Without them, the token strings are simply
    concatenated and a split character costs a character of drift, which the
    greedy aligner below resynchronises.
    """
    if tokens and all(token.raw is not None for token in tokens):
        data = b"".join(token.raw for token in tokens)
        chars: list[str] = []
        char_of_byte = [0] * (len(data) + 1)
        position = 0
        while position < len(data):
            lead = data[position]
            width = 1 if lead < 0x80 else 2 if lead >> 5 == 0b110 else \
                3 if lead >> 4 == 0b1110 else 4 if lead >> 3 == 0b11110 else 1
            piece = data[position:position + width]
            try:
                chars.append(piece.decode("utf-8"))
            except UnicodeDecodeError:
                width, piece = 1, data[position:position + 1]
                chars.append("\ufffd")
            for offset in range(width):
                char_of_byte[position + offset] = len(chars) - 1
            position += width
        char_of_byte[len(data)] = len(chars)
        spans, cursor = [], 0
        for token in tokens:
            start, end = cursor, cursor + len(token.raw)
            first = char_of_byte[start]
            # A token that ends inside a character still "spells" it.
            last = char_of_byte[end] if end == len(data) or char_of_byte[end] != char_of_byte[end - 1] \
                else char_of_byte[end] + 1
            spans.append((first, max(first, last) if end > start else first))
            cursor = end
        return "".join(chars), spans
    spans, cursor = [], 0
    for token in tokens:
        spans.append((cursor, cursor + len(token.token)))
        cursor += len(token.token)
    return "".join(token.token for token in tokens), spans


def align_tokens(
    tokens: list[TokenLogprob], text: str, from_end: bool = False
) -> list[Span]:
    """Each token's character span within ``text``, or ``None`` outside it.

    ``text`` may be a slice of what the tokens spell: the JSON after a
    reasoning preamble (search ``from_end``, since a draft of the answer can
    appear in the reasoning), or the article inside a prompt.
    """
    if not tokens or not text:
        return [None] * len(tokens)
    joined, spans = _joined(tokens)
    found = joined.rfind(text) if from_end else joined.find(text)
    if found >= 0:
        return [_clip(start - found, end - found, len(text)) for start, end in spans]
    return _greedy_align(tokens, spans, joined, text, from_end)


def _clip(start: int, end: int, length: int) -> Span:
    if end <= 0 or start >= length or end <= start:
        return None
    return max(0, start), min(length, end)


def _greedy_align(
    tokens: list[TokenLogprob],
    spans: list[tuple[int, int]],
    joined: str,
    text: str,
    from_end: bool,
) -> list[Span]:
    """Align token by token from an anchor, resynchronising after mismatches."""
    result: list[Span] = [None] * len(tokens)
    anchor_at, text_at = -1, 0
    for length in (64, 32, 16):
        probe = text[:length]
        if len(probe) < 8:
            break
        anchor_at = joined.rfind(probe) if from_end else joined.find(probe)
        if anchor_at >= 0:
            break
    if anchor_at < 0:
        # The start may be where a split character sits; try the first long
        # ASCII run instead and work out where the text starts from there.
        run = re.search(r"[ -~]{24,}", text)
        if run:
            probe = run.group(0)[:48]
            found = joined.rfind(probe) if from_end else joined.find(probe)
            if found >= 0:
                anchor_at, text_at = found, run.start()
    if anchor_at < 0:
        return result

    first = next((i for i, (s, e) in enumerate(spans) if e > anchor_at), None)
    if first is None:
        return result
    position = text_at + (spans[first][0] - anchor_at)
    misses = 0
    for index in range(first, len(tokens)):
        if position >= len(text):
            break
        piece = tokens[index].token
        if not piece:
            continue
        if position >= 0 and text.startswith(piece, position):
            result[index] = (position, position + len(piece))
            position += len(piece)
            misses = 0
            continue
        if position < 0:  # the anchor token straddles the start of the text
            end = position + len(piece)
            result[index] = _clip(position, end, len(text))
            position = end
            continue
        clean = piece.replace("\ufffd", "")
        if not clean:
            # Part of a character split across tokens; it spells at most one.
            result[index] = (position, min(position + 1, len(text)))
            continue
        window = text.find(clean, position, position + len(clean) + 16)
        if window >= 0:
            start = max(position, window - (len(piece) - len(clean)))
            result[index] = (start, window + len(clean))
            position = window + len(clean)
            misses = 0
            continue
        misses += 1
        if misses > 32:
            LOGGER.debug("Token alignment lost at token %d; stopping", index)
            break
    return result


# ---------------------------------------------------------------------------
# JSON with positions
# ---------------------------------------------------------------------------


_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?")
_THINK_END = re.compile(r"</think>", re.IGNORECASE)
_scanstring = json.decoder.scanstring  # type: ignore[attr-defined]

Path_ = tuple  # a JSON path: keys and list indices


class _Positions:
    """A strict JSON parser that records where every value sits."""

    def __init__(self, text: str):
        self.text = text
        self.spans: dict[Path_, tuple[int, int]] = {}

    def _skip(self, at: int) -> int:
        while at < len(self.text) and self.text[at] in " \t\r\n":
            at += 1
        return at

    def value(self, at: int, path: Path_) -> tuple[Any, int]:
        at = self._skip(at)
        if at >= len(self.text):
            raise ValueError("unexpected end")
        char = self.text[at]
        if char == "{":
            result, end = self._object(at, path)
        elif char == "[":
            result, end = self._array(at, path)
        elif char == '"':
            result, end = _scanstring(self.text, at + 1)
        else:
            for literal, parsed in (("true", True), ("false", False), ("null", None)):
                if self.text.startswith(literal, at):
                    result, end = parsed, at + len(literal)
                    break
            else:
                match = _NUMBER.match(self.text, at)
                if not match:
                    raise ValueError(f"unexpected {char!r} at {at}")
                token = match.group(0)
                result = float(token) if any(c in token for c in ".eE") else int(token)
                end = match.end()
        self.spans[path] = (at, end)
        return result, end

    def _object(self, at: int, path: Path_) -> tuple[dict, int]:
        result: dict = {}
        at = self._skip(at + 1)
        if self.text[at:at + 1] == "}":
            return result, at + 1
        while True:
            at = self._skip(at)
            if self.text[at:at + 1] != '"':
                raise ValueError(f"expected a key at {at}")
            key, at = _scanstring(self.text, at + 1)
            at = self._skip(at)
            if self.text[at:at + 1] != ":":
                raise ValueError(f"expected ':' at {at}")
            result[key], at = self.value(at + 1, path + (key,))
            at = self._skip(at)
            if self.text[at:at + 1] == ",":
                at += 1
                continue
            if self.text[at:at + 1] == "}":
                return result, at + 1
            raise ValueError(f"expected ',' or '}}' at {at}")

    def _array(self, at: int, path: Path_) -> tuple[list, int]:
        result: list = []
        at = self._skip(at + 1)
        if self.text[at:at + 1] == "]":
            return result, at + 1
        while True:
            item, at = self.value(at, path + (len(result),))
            result.append(item)
            at = self._skip(at)
            if self.text[at:at + 1] == ",":
                at += 1
                continue
            if self.text[at:at + 1] == "]":
                return result, at + 1
            raise ValueError(f"expected ',' or ']' at {at}")


def locate_json_object(text: str) -> Optional[tuple[dict, dict[Path_, tuple[int, int]]]]:
    """The first JSON object in ``text`` that parses, with each value's span.

    Mirrors ``llm_client.extract_json_object``: a reasoning block is skipped,
    and so is a fence or any prose before the object.
    """
    ends = [match.end() for match in _THINK_END.finditer(text)]
    search_from = ends[-1] if ends else 0
    attempts = 0
    at = text.find("{", search_from)
    while at >= 0 and attempts < 50:
        parser = _Positions(text)
        try:
            value, _ = parser.value(at, ())
            if isinstance(value, dict):
                return value, parser.spans
        except (ValueError, IndexError, json.JSONDecodeError, RecursionError):
            pass
        attempts += 1
        at = text.find("{", at + 1)
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def enum_slot_names(profile: Any) -> set[str]:
    """Every slot, in any class the extraction reaches, whose range is an enum."""
    names: set[str] = set()
    for class_profile in getattr(profile, "classes", {}).values():
        for slot in class_profile.slots:
            if slot.is_enum:
                names.add(slot.name)
    return names


def _field_label(path: Path_) -> str:
    label = ""
    for part in path:
        label += f"[{part}]" if isinstance(part, int) else (f".{part}" if label else part)
    return label


def _kind(path: Path_, value: Any, enums: set[str]) -> Optional[str]:
    """decision, content or quote — or None for a field not worth scoring."""
    keys = [part for part in path if isinstance(part, str)]
    if not keys or any(key in _EXCLUDED for key in keys):
        return None
    name = keys[-1]
    if name in _QUOTES or (name == "text" and "source_spans" in keys):
        return "quote"
    if name in _REFERENCES or name in enums or isinstance(value, bool):
        return "decision"
    if isinstance(value, (dict, list)):
        return None
    return "content"


def _probability(logprob: float) -> float:
    return math.exp(max(logprob, FLOOR_LOGPROB))


def _alternatives(token: TokenLogprob, limit: int = 5) -> list[dict]:
    return [
        {"token": alt, "p": round(_probability(lp), 4)}
        for alt, lp in token.top[:limit]
    ]


def _token_dump(token: TokenLogprob) -> dict:
    return {
        "t": token.token,
        "lp": round(token.logprob, 4),
        "alt": [[alt, round(lp, 4)] for alt, lp in token.top],
    }


def score_field(
    kind: str, tokens: list[TokenLogprob], store_tokens: bool
) -> Optional[dict]:
    """Score one field from the tokens that spell its value."""
    if not tokens:
        return None
    logprobs = [token.logprob for token in tokens]
    joint = sum(logprobs)
    mean = joint / len(logprobs)
    pivot = min(tokens, key=lambda token: token.logprob)
    # A category is one choice and is scored as one: P(the whole value).
    # A phrase is scored per token, so length is not mistaken for doubt.
    probability = _probability(joint) if kind == "decision" else _probability(mean)
    record = {
        "kind": kind,
        "p": round(probability, 4),
        "mean_logprob": round(mean, 4),
        "min_logprob": round(pivot.logprob, 4),
        "tokens": len(tokens),
        "pivot": {
            "token": pivot.token,
            "p": round(_probability(pivot.logprob), 4),
            "alternatives": _alternatives(pivot),
        },
    }
    if store_tokens:
        record["token_logprobs"] = [_token_dump(token) for token in tokens]
    return record


def _entropy(tokens: list[TokenLogprob]) -> Optional[float]:
    """Mean entropy (nats) over the top-k, with the unseen mass as one outcome."""
    values = []
    for token in tokens:
        if not token.top:
            continue
        probabilities = [_probability(lp) for _, lp in token.top]
        rest = max(0.0, 1.0 - sum(probabilities))
        values.append(-sum(p * math.log(p) for p in probabilities + [rest] if p > 0))
    return round(sum(values) / len(values), 4) if values else None


def _perplexity(logprobs: list[float]) -> Optional[float]:
    if not logprobs:
        return None
    return round(math.exp(-sum(logprobs) / len(logprobs)), 4)


class _TokenIndex:
    """Aligned tokens in text order, for finding the ones inside a span fast."""

    def __init__(self, tokens: list[TokenLogprob], spans: list[Span]):
        pairs = [(span, token) for span, token in zip(spans, tokens) if span is not None]
        self.starts = [span[0] for span, _ in pairs]
        self.ends = [span[1] for span, _ in pairs]
        self.tokens = [token for _, token in pairs]

    def within(self, start: int, end: int) -> list[TokenLogprob]:
        low = bisect.bisect_right(self.ends, start)
        high = bisect.bisect_left(self.starts, end)
        return self.tokens[low:high]

    def at(self, position: int) -> Optional[TokenLogprob]:
        found = self.within(position, position + 1)
        return found[0] if found else None


def score_item(
    kind: str,
    item: dict,
    item_span: Optional[tuple[int, int]],
    field_spans: list[tuple[Path_, tuple[int, int]]],
    index: _TokenIndex,
    enums: set[str],
    settings: ConfidenceSettings,
) -> dict:
    """Score one node or edge. ``field_spans`` are paths relative to the item."""
    fields: dict[str, dict] = {}
    for path, (start, end) in field_spans:
        value = _value_at(item, path)
        field_kind = _kind(path, value, enums)
        if field_kind is None:
            continue
        if isinstance(value, str):  # the quotes are syntax, not the value
            start, end = start + 1, end - 1
        scored = score_field(field_kind, index.within(start, end), settings.store_tokens)
        if scored is not None:
            scored["value"] = value
            fields[_field_label(path)] = scored

    # Did the model mean to emit this item at all? The token that opens it is
    # the moment it chose "another one" over closing the list.
    existence = None
    opening = index.at(item_span[0]) if item_span else None
    if opening is not None:
        existence = {
            "p": round(_probability(opening.logprob), 4),
            "token": opening.token,
            "alternatives": _alternatives(opening),
        }

    core = set(settings.core_fields.get(kind) or [])
    core_fields = {label: f for label, f in fields.items() if label in core}
    scored_fields = core_fields or {
        label: f for label, f in fields.items() if f["kind"] != "quote"
    }
    weakest = min(scored_fields, key=lambda label: scored_fields[label]["p"])         if scored_fields else None

    def lowest(which: str, pool: dict) -> Optional[float]:
        values = [f["p"] for f in pool.values() if f["kind"] == which]
        return round(min(values), 4) if values else None

    secondary = {label: f["p"] for label, f in fields.items()
                 if label not in core and f["kind"] == "decision"}
    return {
        "raw_id": item.get("id"),
        "score": scored_fields[weakest]["p"] if weakest else None,
        "weakest_field": weakest,
        "decision": lowest("decision", core_fields),
        "content": lowest("content", core_fields),
        "quote_fidelity": lowest("quote", fields),
        "secondary_mean": _mean(secondary.values()),
        "secondary_below_medium": sorted(
            label for label, p in secondary.items() if p < settings.medium
        ),
        "existence": existence,
        "entropy": _entropy(index.within(*item_span)) if item_span else None,
        "stated_confidence": item.get("annotation_confidence"),
        "fields": fields,
    }


def _value_at(item: Any, path: Path_) -> Any:
    for part in path:
        try:
            item = item[part]
        except (KeyError, IndexError, TypeError):
            return None
    return item


def score_extraction(
    trace: GenerationTrace,
    parsed: dict,
    enums: set[str],
    settings: ConfidenceSettings,
) -> dict:
    """Score every node and edge in one completion.

    Returns ``{"available", "reason", "output", "nodes": [...], "edges": [...]}``
    with one record per raw item, by position, or ``available: False`` and a
    reason when the tokens cannot be tied to the JSON.
    """
    output = {
        "tokens": len(trace.tokens),
        "perplexity": _perplexity([t.logprob for t in trace.tokens]),
        "finish_reason": trace.finish_reason,
    }
    if not trace.tokens:
        return {"available": False, "reason": "the endpoint returned no logprobs",
                "output": output, "nodes": [], "edges": []}
    located = locate_json_object(trace.text)
    if located is None or located[0] != parsed:
        return {"available": False,
                "reason": "the parsed JSON could not be located in the reply text",
                "output": output, "nodes": [], "edges": []}
    value, spans = located
    token_spans = align_tokens(trace.tokens, trace.text, from_end=True)
    aligned = [t for t, span in zip(trace.tokens, token_spans) if span is not None]
    if not aligned:
        return {"available": False,
                "reason": "the tokens could not be aligned to the reply text",
                "output": output, "nodes": [], "edges": []}
    output.update({
        "answer_tokens": len(aligned),
        "reasoning_tokens": len(trace.tokens) - len(aligned),
        "answer_perplexity": _perplexity([t.logprob for t in aligned]),
    })
    index = _TokenIndex(trace.tokens, token_spans)
    by_item: dict[tuple, list] = {}
    for path, span in spans.items():
        if len(path) > 2 and path[0] in ("nodes", "edges"):
            by_item.setdefault(path[:2], []).append((path[2:], span))
    records: dict[str, list[Optional[dict]]] = {"nodes": [], "edges": []}
    for kind, key in (("node", "nodes"), ("edge", "edges")):
        for position, item in enumerate(value.get(key) or []):
            records[key].append(
                score_item(kind, item, spans.get((key, position)),
                           by_item.get((key, position), []), index, enums, settings)
                if isinstance(item, dict) else None
            )
    return {"available": True, "reason": None, "output": output, **records}


# ---------------------------------------------------------------------------
# The prompt side: how surprising the article was
# ---------------------------------------------------------------------------


@dataclass
class PromptSurprisal:
    """Per-character access to prompt-token logprobs over one chunk."""

    offsets: list[tuple[int, int, float]]  # chunk-relative start, end, logprob

    def over(self, start: int, end: int) -> Optional[dict]:
        values = [lp for s, e, lp in self.offsets if s < end and e > start]
        if not values:
            return None
        return {"mean_logprob": round(sum(values) / len(values), 4),
                "perplexity": _perplexity(values), "tokens": len(values)}


def score_prompt(
    trace: GenerationTrace, chunk_text: str, window: int = 48, top: int = 5
) -> tuple[dict, Optional[PromptSurprisal]]:
    """Perplexity of the article under the model, and its strangest passages.

    This measures the *input*, not the extraction: garbled conversion (broken
    ligatures, a table read as a column of numbers, OCR noise) reads as a run
    of very surprising tokens, which is what the passages list surfaces.
    """
    if not trace.prompt_tokens:
        return {"available": False, "reason": "the endpoint returned no prompt_logprobs"}, None
    spans = align_tokens(trace.prompt_tokens, chunk_text, from_end=True)
    offsets = [(span[0], span[1], token.logprob)
               for token, span in zip(trace.prompt_tokens, spans) if span is not None]
    if not offsets:
        return {"available": False,
                "reason": "the article could not be found in the prompt tokens"}, None
    values = [lp for _, _, lp in offsets]
    passages = []
    step = max(1, window // 2)
    for start in range(0, max(1, len(offsets) - window + 1), step):
        piece = offsets[start:start + window]
        mean = sum(lp for _, _, lp in piece) / len(piece)
        passages.append((mean, piece[0][0], piece[-1][1]))
    passages.sort()
    chosen: list[tuple[float, int, int]] = []
    for mean, begin, finish in passages:
        if all(finish <= b or begin >= f for _, b, f in chosen):
            chosen.append((mean, begin, finish))
        if len(chosen) >= top:
            break
    return {
        "available": True,
        "article_tokens": len(offsets),
        "coverage": round(sum(e - s for s, e, _ in offsets) / max(1, len(chunk_text)), 4),
        "perplexity": _perplexity(values),
        "most_surprising": [
            {"start_char": begin, "end_char": finish,
             "perplexity": round(math.exp(-mean), 2),
             "text": chunk_text[begin:finish][:300]}
            for mean, begin, finish in chosen
        ],
    }, PromptSurprisal(offsets)


# ---------------------------------------------------------------------------
# Following annotations through the pipeline
# ---------------------------------------------------------------------------


class ConfidenceTracker:
    """Carries per-item scores from the model's reply to the ids in the graph.

    Ids change three times between the two. Normalization mints them in each
    chunk; consolidation re-derives them from content when the chunks are
    merged; reconciliation swaps a node's id for the existing node it matched.
    Each step reports how, and the tracker follows along, so the sidecar is
    keyed by the ids the saved graph actually has.
    """

    def __init__(self, settings: ConfidenceSettings, profile: Any):
        self.settings = settings
        self.enums = enum_slot_names(profile)
        self.chunks: list[dict] = []
        self.dropped: list[dict] = []
        #: (kind, chunk, id) -> observations, until consolidation re-keys them.
        self._pending: dict[tuple[str, int, str], list[dict]] = {}
        self.nodes: dict[str, list[dict]] = {}
        self.edges: dict[str, list[dict]] = {}
        self._surprisal: Optional[PromptSurprisal] = None

    # -- per chunk ----------------------------------------------------------

    def score_chunk(
        self,
        chunk: int,
        trace: Optional[GenerationTrace],
        raw: dict,
        chunk_text: str,
        unavailable_reason: str = "the endpoint returned no logprobs",
    ) -> dict[int, dict]:
        """Score one reply. Returns records keyed by ``id()`` of the raw items.

        Keyed by object identity because normalization mutates those same
        dicts in place, so the key survives it where a position or an id
        would not.
        """
        self._surprisal = None
        if trace is None:
            self.chunks.append({"chunk": chunk, "available": False,
                                "reason": unavailable_reason})
            return {}
        scored = score_extraction(trace, raw, self.enums, self.settings)
        summary = {"chunk": chunk, "available": scored["available"],
                   "reason": scored["reason"], "output": scored["output"]}
        surprisal = None
        if self.settings.prompt_logprobs is not None:
            summary["prompt"], surprisal = score_prompt(trace, chunk_text)
        self.chunks.append(summary)
        if not scored["available"]:
            LOGGER.warning("Chunk %d: no per-item confidence (%s)", chunk, scored["reason"])
            return {}

        by_object: dict[int, dict] = {}
        for key, kind in (("nodes", "node"), ("edges", "edge")):
            for item, record in zip(raw.get(key) or [], scored[key]):
                if isinstance(item, dict) and record is not None:
                    record["kind"] = kind
                    record["chunk"] = chunk
                    by_object[id(item)] = record
        self._surprisal = surprisal
        return by_object

    def bind_chunk(
        self, chunk: int, graph: dict, by_object: dict[int, dict],
        raw: dict, chunk_start: int = 0,
    ) -> None:
        """After normalization: key each record by the id the item now has."""
        if not by_object:
            return
        surprisal = self._surprisal
        kept: set[int] = set()
        for key, kind in (("nodes", "node"), ("edges", "edge")):
            for item in graph.get(key) or []:
                record = by_object.get(id(item))
                if record is None:
                    continue
                kept.add(id(item))
                spans = item.get("source_spans") or []
                record["quote_found"] = bool(spans) and all(
                    "start_char" in span for span in spans if isinstance(span, dict)
                )
                if surprisal is not None:
                    located = [span for span in spans
                               if isinstance(span, dict) and "start_char" in span]
                    if located:
                        record["source_surprisal"] = surprisal.over(
                            located[0]["start_char"] - chunk_start,
                            located[0]["end_char"] - chunk_start,
                        )
                self._pending.setdefault((kind, chunk, str(item.get("id"))), []).append(record)
        for key in ("nodes", "edges"):
            for item in raw.get(key) or []:
                if isinstance(item, dict) and id(item) in by_object and id(item) not in kept:
                    record = by_object[id(item)]
                    self.dropped.append({
                        "kind": record["kind"], "chunk": chunk,
                        "raw_id": record["raw_id"], "score": record["score"],
                        "reason": "dropped by normalization (unresolvable endpoint)",
                    })

    # -- after the chunks are merged ----------------------------------------

    def after_consolidation(
        self, node_id_maps: list[dict[str, str]], edge_id_maps: list[dict[str, str]]
    ) -> None:
        """Re-key by consolidated id. Maps are per chunk graph, in order."""
        for (kind, chunk, local), records in self._pending.items():
            maps = node_id_maps if kind == "node" else edge_id_maps
            index = chunk - 1
            final = maps[index].get(local, local) if index < len(maps) else local
            target = self.nodes if kind == "node" else self.edges
            target.setdefault(final, []).extend(records)
        self._pending.clear()

    def after_reconciliation(self, matches: Iterable[dict]) -> None:
        """Follow nodes that adopted an existing node's id."""
        for match in matches:
            new, existing = match.get("new_id"), match.get("existing_id")
            if new and existing and new != existing and new in self.nodes:
                self.nodes.setdefault(existing, []).extend(self.nodes.pop(new))

    # -- the result ---------------------------------------------------------

    def finalize(self, graph: dict, document: dict) -> dict:
        """The sidecar for this document, keyed by the graph's own ids."""
        settings = self.settings
        names = {node.get("id"): node.get("name") or node.get("entity_term")
                 for node in graph.get("nodes") or []}
        nodes: dict[str, dict] = {}
        for node in graph.get("nodes") or []:
            observations = self.nodes.get(node.get("id"), [])
            score = _best(observations)
            nodes[node["id"]] = {
                "name": names.get(node["id"]),
                "score": score,
                "bucket": settings.bucket(score),
                "observations": observations,
            }
        edges: dict[str, dict] = {}
        for edge in graph.get("edges") or []:
            observations = self.edges.get(edge.get("id"), [])
            own = _best(observations)
            endpoints = [nodes.get(edge.get(role), {}).get("score")
                         for role in ("subject", "object")]
            claim = None if own is None else min(
                [own] + [score for score in endpoints if score is not None]
            )
            edges[edge["id"]] = {
                "label": _edge_label(edge, names),
                "subject": edge.get("subject"),
                "object": edge.get("object"),
                "score": claim,
                "edge_score": own,
                "bucket": settings.bucket(claim),
                "limited_by": _limited_by(own, endpoints),
                "observations": observations,
            }
        return {
            "document": document,
            "settings": settings.describe(),
            "summary": summarize_document(nodes, edges, self.chunks, self.dropped, settings),
            "chunks": self.chunks,
            "nodes": nodes,
            "edges": edges,
            "dropped": self.dropped,
        }


def _best(observations: list[dict]) -> Optional[float]:
    """An item extracted more than once is as good as its best extraction."""
    scores = [record["score"] for record in observations if record.get("score") is not None]
    return max(scores) if scores else None


def _limited_by(own: Optional[float], endpoints: list[Optional[float]]) -> Optional[str]:
    if own is None:
        return None
    lowest = min([(own, "edge"), *[(score, role) for score, role
                                    in zip(endpoints, ("subject", "object"))
                                    if score is not None]])
    return lowest[1]


def _edge_label(edge: dict, names: dict) -> str:
    subject = names.get(edge.get("subject")) or edge.get("subject")
    obj = names.get(edge.get("object")) or edge.get("object")
    negated = "NOT " if edge.get("negated") else ""
    return f"{subject} --{negated}{edge.get('predicate')}--> {obj}"


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _best_observation(entry: dict) -> Optional[dict]:
    return max(entry["observations"], key=lambda r: r.get("score") or -1, default=None)


def _review_entry(
    identifier: str, entry: dict, kind: str, nodes: Optional[dict] = None
) -> dict:
    """What a reviewer needs about one weak item, without the token dump.

    An edge held down by one of its endpoints is explained by that endpoint:
    the doubt worth showing is the node's, not the arrow's.
    """
    best = _best_observation(entry)
    result = {
        "id": identifier,
        "kind": kind,
        "label": entry.get("label") or entry.get("name"),
        "score": entry["score"],
        "bucket": entry["bucket"],
    }
    source, prefix = best, ""
    if kind == "edge":
        limited_by = entry.get("limited_by")
        result["limited_by"] = limited_by
        endpoint = (nodes or {}).get(entry.get(limited_by) or "")
        if limited_by in ("subject", "object") and endpoint:
            source, prefix = _best_observation(endpoint), f"{limited_by}."
    if source:
        weakest = source.get("weakest_field")
        field_record = source["fields"].get(weakest) if weakest else None
        if field_record:
            result["weakest_field"] = prefix + weakest
            result["value"] = field_record["value"]
            result["value_p"] = field_record["p"]
            result["alternatives"] = field_record["pivot"]["alternatives"]
    if best:
        for extra in ("quote_found", "quote_fidelity", "stated_confidence",
                      "secondary_below_medium"):
            if best.get(extra) not in (None, [], ""):
                result[extra] = best[extra]
    return result


def summarize_document(
    nodes: dict, edges: dict, chunks: list[dict], dropped: list[dict],
    settings: ConfidenceSettings,
) -> dict:
    available = any(chunk.get("available") for chunk in chunks)
    buckets = {
        kind: {bucket: sum(1 for e in entries.values() if e["bucket"] == bucket)
               for bucket in BUCKETS}
        for kind, entries in (("nodes", nodes), ("edges", edges))
    }
    weakest_fields: dict[str, int] = {}
    for entries in (nodes, edges):
        for entry in entries.values():
            for record in entry["observations"]:
                if record.get("weakest_field") and record.get("score") is not None \
                        and record["score"] < settings.high:
                    label = f"{record['kind']}.{record['weakest_field']}"
                    weakest_fields[label] = weakest_fields.get(label, 0) + 1
    review = sorted(
        [_review_entry(i, e, "edge", nodes) for i, e in edges.items()
         if e["score"] is not None]
        + [_review_entry(i, e, "node") for i, e in nodes.items() if e["score"] is not None],
        key=lambda entry: entry["score"],
    )
    summary: dict[str, Any] = {
        "available": available,
        "reasons": sorted({c["reason"] for c in chunks if c.get("reason")}),
        "buckets": buckets,
        "thresholds": {"high": settings.high, "medium": settings.medium},
        "mean_score": {
            kind: _mean([e["score"] for e in entries.values()])
            for kind, entries in (("nodes", nodes), ("edges", edges))
        },
        "weakest_fields": dict(sorted(weakest_fields.items(), key=lambda kv: -kv[1])),
        "quotes_not_found": sum(
            1 for e in edges.values() for r in e["observations"][:1]
            if r.get("quote_found") is False
        ),
        "dropped_by_normalization": len(dropped),
        "review_first": review[: settings.review_count],
    }
    outputs = [c["output"] for c in chunks if c.get("output")]
    if outputs:
        summary["output_perplexity"] = _mean([o.get("answer_perplexity") or o.get("perplexity")
                                              for o in outputs])
        summary["reasoning_tokens"] = sum(o.get("reasoning_tokens") or 0 for o in outputs)
    prompts = [c["prompt"] for c in chunks if (c.get("prompt") or {}).get("available")]
    if prompts:
        summary["article_perplexity"] = _mean([p["perplexity"] for p in prompts])
        summary["most_surprising_passages"] = sorted(
            (passage for p in prompts for passage in p["most_surprising"]),
            key=lambda passage: -passage["perplexity"],
        )[:5]
    return summary


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return round(sum(present) / len(present), 4) if present else None


def sidecar_path(graph_path: Path) -> Path:
    return graph_path.with_name(f"{graph_path.stem}.confidence.json")


def write_sidecar(graph_path: Path, sidecar: dict) -> Path:
    return atomic_write_json(sidecar_path(graph_path), sidecar)


# ---------------------------------------------------------------------------
# Corpus report
# ---------------------------------------------------------------------------


def corpus_report(graphs_dir: Path, settings: ConfidenceSettings) -> Optional[dict]:
    """Roll every ``*.confidence.json`` under ``graphs_dir`` into one report."""
    documents, review, weakest, stated = [], [], {}, {}
    totals = {kind: {bucket: 0 for bucket in BUCKETS} for kind in ("nodes", "edges")}
    for path in sorted(graphs_dir.glob("*.confidence.json")):
        try:
            sidecar = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            LOGGER.warning("Unreadable confidence sidecar %s: %s", path.name, error)
            continue
        summary = sidecar.get("summary") or {}
        slug = path.name[: -len(".confidence.json")]
        for kind in ("nodes", "edges"):
            for bucket, count in (summary.get("buckets") or {}).get(kind, {}).items():
                totals[kind][bucket] = totals[kind].get(bucket, 0) + count
        for label, count in (summary.get("weakest_fields") or {}).items():
            weakest[label] = weakest.get(label, 0) + count
        for entry in summary.get("review_first") or []:
            review.append({"document": slug, **entry})
        for edge in (sidecar.get("edges") or {}).values():
            for record in edge.get("observations")[:1]:
                value = record.get("stated_confidence")
                if isinstance(value, (int, float)) and edge.get("score") is not None:
                    stated.setdefault(round(float(value), 1), []).append(edge["score"])
        documents.append({
            "document": slug,
            "available": summary.get("available"),
            "reasons": summary.get("reasons"),
            "buckets": summary.get("buckets"),
            "mean_score": summary.get("mean_score"),
            "output_perplexity": summary.get("output_perplexity"),
            "article_perplexity": summary.get("article_perplexity"),
            "reasoning_tokens": summary.get("reasoning_tokens"),
            "quotes_not_found": summary.get("quotes_not_found"),
            "dropped_by_normalization": summary.get("dropped_by_normalization"),
        })
    if not documents:
        return None
    review.sort(key=lambda entry: entry["score"])
    edges_first = [entry for entry in review if entry["kind"] == "edge"]
    nodes_first = [entry for entry in review if entry["kind"] == "node"]
    return {
        "documents": len(documents),
        "thresholds": {"high": settings.high, "medium": settings.medium},
        "core_fields": settings.core_fields,
        "buckets": totals,
        "weakest_fields": dict(sorted(weakest.items(), key=lambda kv: -kv[1])),
        "stated_vs_token": {
            str(level): {"edges": len(scores), "mean_token_score": _mean(scores)}
            for level, scores in sorted(stated.items())
        },
        "review_first": edges_first[: settings.review_count * 2],
        "review_nodes_first": nodes_first[: settings.review_count],
        "by_document": documents,
    }


def _bar(counts: dict) -> str:
    total = sum(counts.get(bucket, 0) for bucket in BUCKETS) or 1
    return " · ".join(
        f"{bucket} {counts.get(bucket, 0)} ({100 * counts.get(bucket, 0) / total:.0f}%)"
        for bucket in BUCKETS if counts.get(bucket) or bucket != "unscored"
    )


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict) -> str:
    """The reviewer's page: where the extraction is sure, and where it is not."""
    thresholds = report["thresholds"]
    lines = [
        "# Extraction confidence",
        "",
        f"{report['documents']} document(s). Scores are the model's token "
        f"probabilities, not calibrated odds of being right: use them to decide "
        f"what to read first. **high** ≥ {thresholds['high']}, **medium** ≥ "
        f"{thresholds['medium']}, **low** below that. A node takes its weakest "
        f"core field; an edge's claim takes the weakest of itself and its two "
        f"endpoint nodes.",
        "",
        f"- **Edges (claims):** {_bar(report['buckets']['edges'])}",
        f"- **Nodes:** {_bar(report['buckets']['nodes'])}",
        "",
    ]
    if report["weakest_fields"]:
        lines += ["## Where the model hesitates", "",
                  "How often each field was the weakest part of a non-high item. "
                  "A field that dominates here is usually a prompt or schema "
                  "ambiguity rather than a hard article.", "",
                  "| field | items |", "|---|---:|"]
        lines += [f"| `{label}` | {count} |"
                  for label, count in list(report["weakest_fields"].items())[:12]]
        lines.append("")
    if report["review_first"]:
        lines += ["## Review these claims first", "",
                  "| doc | claim | score | weakest | chose | alternatives |",
                  "|---|---|---:|---|---|---|"]
        for entry in report["review_first"][:40]:
            alternatives = ", ".join(
                f"`{alt['token'].strip() or repr(alt['token'])}` {alt['p']:.2f}"
                for alt in entry.get("alternatives") or []
            )
            weakest = entry.get("weakest_field") or ""
            flags = "" if entry.get("quote_found", True) else " ⚠ quote not in article"
            lines.append(
                f"| {_cell(entry['document'])} | {_cell(entry.get('label'))}{flags} "
                f"| {entry['score']:.2f} | `{_cell(weakest)}` "
                f"| {_cell(entry.get('value'))} ({entry.get('value_p', 0):.2f}) "
                f"| {_cell(alternatives)} |"
            )
        lines.append("")
    if report["stated_vs_token"]:
        lines += ["## Stated versus token confidence", "",
                  "Edges where the model filled `annotation_confidence` itself, "
                  "against what its tokens say. If these do not rise together, "
                  "the stated number is decoration.", "",
                  "| stated | edges | mean token score |", "|---:|---:|---:|"]
        lines += [f"| {level} | {row['edges']} | {row['mean_token_score']} |"
                  for level, row in report["stated_vs_token"].items()]
        lines.append("")
    lines += ["## By document", "",
              "| doc | edges H/M/L | nodes H/M/L | answer ppl | article ppl "
              "| quotes not found | note |",
              "|---|---|---|---:|---:|---:|---|"]
    for doc in report["by_document"]:
        buckets = doc.get("buckets") or {}

        def hml(kind: str) -> str:
            counts = buckets.get(kind) or {}
            return "/".join(str(counts.get(b, 0)) for b in ("high", "medium", "low"))

        note = "" if doc.get("available") else "; ".join(doc.get("reasons") or ["unavailable"])
        lines.append(
            f"| {_cell(doc['document'])} | {hml('edges')} | {hml('nodes')} "
            f"| {_cell(doc.get('output_perplexity'))} | {_cell(doc.get('article_perplexity'))} "
            f"| {_cell(doc.get('quotes_not_found'))} | {_cell(note)} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_corpus_report(
    graphs_dir: Path, out_dir: Path, settings: ConfidenceSettings
) -> Optional[Path]:
    """``confidence_report.json`` and ``confidence_summary.md`` in ``out_dir``."""
    report = corpus_report(graphs_dir, settings)
    if report is None:
        return None
    atomic_write_json(out_dir / "confidence_report.json", report)
    return atomic_write(out_dir / "confidence_summary.md", render_markdown(report))


def print_buckets(report_path: Optional[Path]) -> None:
    if report_path is None:
        return
    report = json.loads((report_path.parent / "confidence_report.json")
                        .read_text(encoding="utf-8"))
    print(f"  confidence, edges: {_bar(report['buckets']['edges'])}")
    print(f"  confidence, nodes: {_bar(report['buckets']['nodes'])}")
    print(f"  wrote {report_path}")


# ---------------------------------------------------------------------------
# After the merge
# ---------------------------------------------------------------------------


#: Per-document sidecars, and what of each entry the merged index keeps.
MERGED_INDEXES = {
    "confidence_index.json": (".confidence.json", ("score", "bucket")),
    "agreement_index.json": (".agreement.json", ("status", "flagged", "support")),
}


def index_merged(
    graph_paths: list[Path],
    node_id_maps: list[dict[str, str]],
    edge_id_maps: list[dict[str, str]],
    offset: int = 0,
    suffix: str = ".confidence.json",
    keep: tuple[str, ...] = ("score", "bucket"),
) -> Optional[dict]:
    """Where each merged node and edge came from, and what its sidecar said.

    The merge re-derives ids, so a per-document sidecar's keys are not the
    merged graph's. ``offset`` is 1 when an existing graph was merged first.
    A merged node seen in several documents lists each; that it was extracted
    confidently, or agreed on, in five papers is itself worth knowing.
    """
    merged: dict[str, dict[str, list]] = {"nodes": {}, "edges": {}}
    found = False
    for index, graph_path in enumerate(graph_paths):
        path = graph_path.with_name(f"{graph_path.stem}{suffix}")
        if not path.exists():
            continue
        found = True
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        slug = graph_path.stem
        for key, maps in (("nodes", node_id_maps), ("edges", edge_id_maps)):
            mapping = maps[index + offset] if index + offset < len(maps) else {}
            for local, entry in (sidecar.get(key) or {}).items():
                merged[key].setdefault(mapping.get(local, local), []).append({
                    "document": slug, "document_item_id": local,
                    **{name: entry.get(name) for name in keep},
                })
    if not found:
        return None
    result: dict = {}
    for key, entries in merged.items():
        result[key] = {}
        for identifier, sources in entries.items():
            entry: dict = {"sources": sources}
            if "score" in keep:
                entry["best"] = max((s["score"] for s in sources
                                     if s.get("score") is not None), default=None)
            if "flagged" in keep:
                entry["flagged"] = any(s.get("flagged") for s in sources)
            result[key][identifier] = entry
    return result


def write_merged_index(
    graph_paths: list[Path], id_maps: dict, has_existing: bool, out_dir: Path
) -> list[Path]:
    """One index per kind of sidecar the merged graphs have."""
    written = []
    for name, (suffix, keep) in MERGED_INDEXES.items():
        index = index_merged(graph_paths, id_maps.get("nodes") or [],
                             id_maps.get("edges") or [], 1 if has_existing else 0,
                             suffix, keep)
        if index is not None:
            written.append(atomic_write_json(out_dir / name, index))
    return written
