"""Provider-agnostic LLM client driven entirely by ``tools/config/llm.yaml``.

This is the *only* module in the toolchain that decides where inference runs.
Scripts ask for structured JSON and get a parsed dict back; how that JSON was
coerced out of the model (native JSON-Schema support, forced tool use, or
prompt-and-repair) is a config concern, not a caller concern.

Ollama is the default and the supported path. ``openai_compatible`` and
``anthropic`` are working code paths for repointing at other infrastructure.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .confidence import (
    MAX_TOP_LOGPROBS,
    GenerationTrace,
    tokens_from_openai,
    tokens_from_prompt_logprobs,
)
from .config import DEFAULT_LLM_CONFIG, load_yaml

LOGGER = logging.getLogger("camo.llm")

SUPPORTED_PROVIDERS = {"ollama", "openai_compatible", "anthropic"}
SUPPORTED_STRUCTURED = {"json_schema", "json_object", "tool_use", "prompt_only"}

#: Name of the synthetic tool used when structured_output is ``tool_use``.
_TOOL_NAME = "emit_result"


class LLMError(RuntimeError):
    """Raised when the model could not be reached or produced unusable output."""


@dataclass
class LLMSettings:
    """Resolved inference settings. Mirrors the ``llm:`` block of llm.yaml."""

    provider: str = "ollama"
    endpoint: str = "http://localhost:11434/v1"
    model: str = "qwen3:32b"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 8192
    timeout: int = 600
    structured_output: str = "json_schema"
    max_repair_attempts: int = 2
    marker_llm: dict = field(default_factory=dict)
    #: Passed through verbatim on OpenAI-compatible requests: server-specific
    #: switches such as vLLM's ``chat_template_kwargs``.
    extra_body: dict = field(default_factory=dict)
    #: Ask for token logprobs (the ``confidence:`` block, or --confidence).
    logprobs: bool = False
    top_logprobs: int = 5
    #: vLLM only. ``None`` is off; an int is the number of alternatives.
    prompt_logprobs: Optional[int] = None

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "LLMSettings":
        data = load_yaml(path or DEFAULT_LLM_CONFIG)
        block = data.get("llm") or {}
        confidence = data.get("confidence") or {}
        settings = cls(
            provider=block.get("provider", cls.provider),
            endpoint=block.get("endpoint", cls.endpoint),
            model=block.get("model", cls.model),
            api_key=block.get("api_key") or "",
            temperature=float(block.get("temperature", cls.temperature)),
            max_tokens=int(block.get("max_tokens", cls.max_tokens)),
            timeout=int(block.get("timeout", cls.timeout)),
            structured_output=block.get("structured_output", cls.structured_output),
            max_repair_attempts=int(
                block.get("max_repair_attempts", cls.max_repair_attempts)
            ),
            marker_llm=data.get("marker_llm") or {},
            extra_body=dict(block.get("extra_body") or {}),
            logprobs=bool(confidence.get("enabled", False)),
            top_logprobs=int(confidence.get("top_logprobs", cls.top_logprobs)),
            prompt_logprobs=confidence.get("prompt_logprobs"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported llm.provider {self.provider!r}; "
                f"expected one of {sorted(SUPPORTED_PROVIDERS)}"
            )
        if self.structured_output not in SUPPORTED_STRUCTURED:
            raise ValueError(
                f"Unsupported llm.structured_output {self.structured_output!r}; "
                f"expected one of {sorted(SUPPORTED_STRUCTURED)}"
            )
        if self.provider == "anthropic" and not self.api_key:
            raise ValueError(
                "provider 'anthropic' requires an api_key; set LLM_API_KEY in the "
                "environment (llm.yaml reads it as ${LLM_API_KEY:-})"
            )

    def resolved_marker_llm(self) -> Optional[dict]:
        """Marker's LLM-assist settings, inheriting endpoint/model when unset."""
        if not self.marker_llm.get("enabled"):
            return None
        return {
            "endpoint": self.marker_llm.get("endpoint") or self.endpoint,
            "model": self.marker_llm.get("model") or self.model,
            "api_key": self.api_key or "not-needed",
        }


class LLMClient:
    """Thin structured-output client over whichever provider config selects."""

    def __init__(self, settings: Optional[LLMSettings] = None, **overrides: Any):
        self.settings = settings or LLMSettings.from_config()
        for key, value in overrides.items():
            if value is not None and hasattr(self.settings, key):
                setattr(self.settings, key, value)
        self.settings.validate()
        self._client: Any = None
        #: Set once a provider refuses the sampling parameters, so the whole
        #: run stops sending them after the first refusal.
        self._sampling_refused = False
        #: Likewise for logprobs: an endpoint that refuses them once will
        #: refuse them for every document.
        self._logprobs_refused = False
        #: Tokens behind the most recent reply, when logprobs were asked for
        #: and returned; ``None`` otherwise. Read it right after a call.
        self.last_trace: Optional[GenerationTrace] = None
        #: Why ``last_trace`` is ``None`` when logprobs were asked for.
        self.trace_unavailable: Optional[str] = None

    # -- public API ---------------------------------------------------------

    def complete_json(
        self,
        system: str,
        user: str,
        json_schema: Optional[dict] = None,
        schema_name: str = "result",
        schema_in_prompt: bool = True,
    ) -> dict:
        """Return a parsed JSON object from the model.

        ``json_schema`` is honoured natively when the configured
        ``structured_output`` mode supports it, and is otherwise appended to the
        prompt. Malformed replies are re-prompted with the parse error up to
        ``max_repair_attempts`` times.

        Set ``schema_in_prompt=False`` when the caller's prompt already
        documents the schema in prose; appending tens of kilobytes of JSON
        Schema on top of that just gives a reasoning model more to chew on.
        """
        mode = self.settings.structured_output
        prompt = user
        if schema_in_prompt and mode in {"prompt_only", "json_object"} and json_schema:
            prompt = (
                f"{user}\n\n"
                "Return a single JSON object conforming to this JSON Schema. "
                "Emit no prose, no markdown fence, and no commentary.\n"
                f"{json.dumps(json_schema, ensure_ascii=False)}"
            )

        last_error: Optional[Exception] = None
        for attempt in range(self.settings.max_repair_attempts + 1):
            if attempt:
                prompt = (
                    f"{prompt}\n\n"
                    f"Your previous reply could not be parsed as JSON: {last_error}. "
                    "Reply with the corrected JSON object only."
                )
                LOGGER.warning(
                    "Retrying structured output (attempt %d/%d)",
                    attempt + 1,
                    self.settings.max_repair_attempts + 1,
                )
            raw = self._dispatch(system, prompt, json_schema, schema_name)
            try:
                return extract_json_object(raw)
            except ValueError as error:
                last_error = error
        raise LLMError(
            f"Model did not return parseable JSON after "
            f"{self.settings.max_repair_attempts + 1} attempt(s): {last_error}"
        )

    def complete_text(self, system: str, user: str) -> str:
        """Return the model's raw text reply."""
        return self._dispatch(system, user, None, "result")

    # -- provider dispatch --------------------------------------------------

    def _dispatch(
        self,
        system: str,
        user: str,
        json_schema: Optional[dict],
        schema_name: str,
    ) -> str:
        self.last_trace = None
        self.trace_unavailable = None
        if self.settings.provider == "anthropic":
            if self.settings.logprobs:
                self.trace_unavailable = "the Anthropic API does not expose logprobs"
            return self._call_anthropic(system, user, json_schema, schema_name)
        return self._call_openai_compatible(system, user, json_schema, schema_name)

    def _openai(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as error:  # pragma: no cover - dependency guard
                raise LLMError(
                    "The 'openai' package is required for the ollama and "
                    "openai_compatible providers: pip install openai"
                ) from error
            self._client = OpenAI(
                base_url=self.settings.endpoint,
                api_key=self.settings.api_key or "not-needed",
                timeout=self.settings.timeout,
            )
        return self._client

    def _call_openai_compatible(
        self,
        system: str,
        user: str,
        json_schema: Optional[dict],
        schema_name: str,
    ) -> str:
        client = self._openai()
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        extra_body = dict(self.settings.extra_body)
        if self.settings.logprobs and not self._logprobs_refused:
            kwargs["logprobs"] = True
            top = max(0, min(self.settings.top_logprobs, MAX_TOP_LOGPROBS))
            if top:
                kwargs["top_logprobs"] = top
            if self.settings.prompt_logprobs is not None:
                extra_body["prompt_logprobs"] = int(self.settings.prompt_logprobs)
        if extra_body:
            kwargs["extra_body"] = extra_body
        mode = self.settings.structured_output
        if mode == "json_object":
            # No grammar: the server only guarantees syntactically valid JSON.
            # The shape comes from the prompt, which already carries the schema.
            kwargs["response_format"] = {"type": "json_object"}
        elif json_schema and mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": False,
                },
            }
        elif json_schema and mode == "tool_use":
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": _TOOL_NAME,
                        "description": "Emit the extraction result.",
                        "parameters": json_schema,
                    },
                }
            ]
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": _TOOL_NAME},
            }

        response = self._create_with_fallback(client, kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return self._traced(response, choice, tool_calls[0].function.arguments)

        content = message.content
        if content:
            return self._traced(response, choice, content)

        # Reasoning models put their chain of thought in a separate field and
        # can spend the whole generation budget there, returning empty content.
        # Saying so beats "empty response", which sends you looking in the
        # wrong place entirely.
        reasoning = (
            getattr(message, "reasoning", None)
            or getattr(message, "reasoning_content", None)
            or ""
        )
        finish_reason = getattr(choice, "finish_reason", None)
        if reasoning and finish_reason == "length":
            raise LLMError(
                f"Model returned no content: it emitted {len(reasoning)} characters "
                f"of reasoning and hit the {self.settings.max_tokens}-token "
                f"generation limit before answering. Raise llm.max_tokens, or use "
                f"a model that does not reason before answering."
            )
        if reasoning:
            # The answer is sometimes inside the reasoning text; let the caller's
            # parser try rather than discarding a usable reply.
            LOGGER.warning(
                "Empty content but %d characters of reasoning; attempting to "
                "parse the reasoning text", len(reasoning),
            )
            return self._traced(response, choice, reasoning)
        raise LLMError(
            f"Model returned an empty response (finish_reason={finish_reason!r})"
        )

    def _traced(self, response: Any, choice: Any, text: str) -> str:
        """Keep the tokens behind ``text`` when logprobs were asked for."""
        if not self.settings.logprobs:
            return text
        if self._logprobs_refused:
            self.trace_unavailable = "the endpoint refused the logprobs parameter"
            return text
        tokens = tokens_from_openai(getattr(choice, "logprobs", None))
        if not tokens:
            self.trace_unavailable = "the endpoint accepted logprobs but returned none"
            return text
        prompt_tokens = (
            tokens_from_prompt_logprobs(response)
            if self.settings.prompt_logprobs is not None else []
        )
        self.last_trace = GenerationTrace(
            text=text, tokens=tokens, prompt_tokens=prompt_tokens,
            finish_reason=getattr(choice, "finish_reason", None),
        )
        return text

    def _create_with_fallback(self, client: Any, kwargs: dict) -> Any:
        """Call the endpoint, degrading structured-output support if refused.

        Ollama's OpenAI-compatible layer has accepted different response_format
        shapes across versions, and some servers reject json_schema outright.
        Rather than pinning a server version we degrade: json_schema ->
        json_object -> plain, so a stricter mode never becomes a hard failure.
        """
        attempts: list[dict] = [kwargs]
        if "response_format" in kwargs:
            relaxed = dict(kwargs)
            relaxed["response_format"] = {"type": "json_object"}
            attempts.append(relaxed)
        if "response_format" in kwargs or "tools" in kwargs:
            plain = {
                key: value
                for key, value in kwargs.items()
                if key not in {"response_format", "tools", "tool_choice"}
            }
            attempts.append(plain)

        last_error: Optional[Exception] = None
        index = 0
        while index < len(attempts):
            attempt = attempts[index]
            try:
                return client.chat.completions.create(**attempt)
            except Exception as error:  # noqa: BLE001 - provider errors vary widely
                last_error = error
                # A server that refuses logprobs has not refused JSON: drop the
                # logprobs and retry the same format, rather than loosening the
                # output constraint for nothing.
                if ("prompt_logprobs" in str(error)
                        and "prompt_logprobs" in (attempt.get("extra_body") or {})):
                    LOGGER.warning(
                        "Endpoint refused prompt_logprobs (vLLM only); keeping "
                        "output logprobs: %s", str(error)[:160],
                    )
                    self.settings.prompt_logprobs = None
                    attempts = [_without_prompt_logprobs(c) for c in attempts]
                    continue
                if _is_logprobs_rejection(error) and _has_logprobs(attempt):
                    LOGGER.warning(
                        "Endpoint refused logprobs (%s); continuing without "
                        "confidence tracking", str(error)[:160],
                    )
                    self._logprobs_refused = True
                    attempts = [_without_logprobs(candidate) for candidate in attempts]
                    continue
                index += 1
                if index < len(attempts):
                    LOGGER.warning(
                        "Endpoint rejected structured-output request (%s); "
                        "retrying with a more permissive format",
                        type(error).__name__,
                    )
        raise LLMError(f"LLM request failed: {last_error}") from last_error

    def _call_anthropic(
        self,
        system: str,
        user: str,
        json_schema: Optional[dict],
        schema_name: str,
    ) -> str:
        try:
            import anthropic
        except ImportError as error:  # pragma: no cover - dependency guard
            raise LLMError(
                "The 'anthropic' package is required for provider 'anthropic': "
                "pip install anthropic"
            ) from error
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self.settings.api_key, timeout=self.settings.timeout
            )
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if not self._sampling_refused:
            kwargs["temperature"] = self.settings.temperature
        if json_schema and self.settings.structured_output != "prompt_only":
            kwargs["tools"] = [
                {
                    "name": _TOOL_NAME,
                    "description": "Emit the extraction result.",
                    "input_schema": json_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": _TOOL_NAME}
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as error:  # noqa: BLE001 - provider errors vary widely
            # Claude 4.6 and later removed the sampling parameters: sending
            # temperature to claude-opus-5 is a 400, not a warning. Rather than
            # keeping a model list that goes stale, drop it on refusal and
            # remember, so it costs one request per run at most.
            if not self._sampling_refused and _is_sampling_rejection(error):
                LOGGER.info(
                    "%s does not accept temperature; retrying without it",
                    self.settings.model,
                )
                self._sampling_refused = True
                kwargs.pop("temperature", None)
                try:
                    response = self._client.messages.create(**kwargs)
                except Exception as retry_error:  # noqa: BLE001
                    raise LLMError(f"LLM request failed: {retry_error}") from retry_error
            else:
                raise LLMError(f"LLM request failed: {error}") from error

        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return json.dumps(block.input, ensure_ascii=False)
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        raise LLMError("Model returned no usable content block")


_LOGPROB_KEYS = {"logprobs", "top_logprobs"}


def _has_logprobs(kwargs: dict) -> bool:
    return bool(_LOGPROB_KEYS & kwargs.keys()) or "prompt_logprobs" in (
        kwargs.get("extra_body") or {}
    )


def _without_logprobs(kwargs: dict) -> dict:
    return _without_prompt_logprobs(
        {key: value for key, value in kwargs.items() if key not in _LOGPROB_KEYS}
    )


def _without_prompt_logprobs(kwargs: dict) -> dict:
    cleaned = dict(kwargs)
    extra = {key: value for key, value in (kwargs.get("extra_body") or {}).items()
             if key != "prompt_logprobs"}
    if extra:
        cleaned["extra_body"] = extra
    else:
        cleaned.pop("extra_body", None)
    return cleaned


def _is_logprobs_rejection(error: Exception) -> bool:
    return "logprob" in str(error).lower()


def _is_sampling_rejection(error: Exception) -> bool:
    """Does this error say the model will not accept temperature/top_p/top_k?

    Matched on the message rather than the exception class, because every
    provider spells this differently and only the text names the parameter.
    """
    text = str(error).lower()
    named = any(word in text for word in ("temperature", "top_p", "top_k", "sampling"))
    refused = any(
        word in text
        for word in ("not supported", "unsupported", "removed", "not permitted",
                     "invalid_request", "unexpected", "not allowed")
    )
    return named and refused


# -- parsing helpers --------------------------------------------------------

_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)
#: Some local models emit a visible reasoning block before the answer.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json_object(raw: str) -> dict:
    """Parse a JSON object out of a model reply.

    Tolerates markdown fences, ``<think>`` preambles, and trailing prose, since
    local models produce all three. Raises ``ValueError`` if no object is found.
    """
    text = _THINK.sub("", raw or "").strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as error:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object found in response") from error
        try:
            parsed = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, RecursionError) as inner:
            raise ValueError(f"malformed JSON object: {inner}") from inner
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed
