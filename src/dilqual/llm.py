"""Thin wrapper around the Claude Messages API used by every pipeline step."""

from __future__ import annotations

from typing import Any

import anthropic
from pydantic import BaseModel

# Models that support server-side refusal fallbacks: if a safety classifier
# declines a request, the API re-runs it on a recommended fallback model.
_FALLBACK_PREFIXES = ("claude-opus-5", "claude-fable-5-1", "claude-fable-5-2")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeRefusal(RuntimeError):
    """Claude declined the request."""


def json_format(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Structured-output format for a Pydantic model."""
    return {"type": "json_schema", "schema": anthropic.transform_schema(model_cls)}


def build_params(
    *,
    model: str,
    system: str | list[dict[str, Any]],
    user: str,
    effort: str | None,
    max_tokens: int,
    output_model: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Request parameters shared by live calls and the Batch API."""
    output_config: dict[str, Any] = {}
    if effort:
        output_config["effort"] = effort
    if output_model is not None:
        output_config["format"] = json_format(output_model)
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if output_config:
        params["output_config"] = output_config
    return params


def cached_system(*blocks: str) -> list[dict[str, Any]]:
    """System prompt blocks with a cache breakpoint on the last block.

    The system prompt (instructions + codebook) is identical for every segment,
    so caching it makes each later request much cheaper.
    """
    out: list[dict[str, Any]] = [{"type": "text", "text": b} for b in blocks if b]
    out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


def text_of(message: Any) -> str:
    return "".join(b.text for b in message.content if b.type == "text")


def check_refusal(message: Any) -> None:
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise ClaudeRefusal(f"Claude declined this request (category: {category}).")
    if message.stop_reason == "max_tokens":
        raise RuntimeError("Response hit max_tokens before finishing.")


def call(client: anthropic.Anthropic, params: dict[str, Any]) -> Any:
    """Make one streamed request and return the final message.

    Streaming avoids HTTP timeouts on long requests (whole-transcript drafting).
    """
    params = dict(params)
    if params["model"].startswith(_FALLBACK_PREFIXES):
        params["betas"] = [_FALLBACK_BETA]
        params["fallbacks"] = "default"
        stream_ctx = client.beta.messages.stream(**params)
    else:
        stream_ctx = client.messages.stream(**params)
    with stream_ctx as stream:
        message = stream.get_final_message()
    check_refusal(message)
    return message


def call_json[T: BaseModel](
    client: anthropic.Anthropic, params: dict[str, Any], output_model: type[T]
) -> tuple[T, Any]:
    """Make a structured-output request and return (parsed model, raw message)."""
    message = call(client, params)
    return output_model.model_validate_json(text_of(message)), message
