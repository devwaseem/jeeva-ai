from __future__ import annotations

import secrets
from collections.abc import Sequence
from typing import Any

SYSTEM_PROMPT = "You are a precise, calibrated decision engine."
_BRANCH_MARKER_PREFIX = "<|JEEVA_BRANCH_POINT:"
MAX_CONTEXT_CHARACTERS = 100_000
MAX_CONTEXT_UTF8_BYTES = 400_000


def validate_context_budget(text: str) -> None:
    """Reject contexts before tokenization when they exceed process-wide limits."""
    if len(text) > MAX_CONTEXT_CHARACTERS:
        raise ValueError(f"Context exceeds the {MAX_CONTEXT_CHARACTERS}-character limit.")
    encoded_size = len(text.encode("utf-8"))
    if encoded_size > MAX_CONTEXT_UTF8_BYTES:
        raise ValueError(f"Context exceeds the {MAX_CONTEXT_UTF8_BYTES}-byte UTF-8 limit.")


def build_contextual_metric_prompts(
    field_prompt: str,
    candidates: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    """Build one query branch and isolated candidate branches for metric scoring.

    Callers append each returned prompt independently to the same document
    prefill. The first prompt represents the decision to make; every remaining
    prompt represents exactly one candidate in stable schema order.
    """
    query_prompt = f"{field_prompt}\n\nRepresent the decision to make."
    candidate_prompts = tuple(
        f"{field_prompt}\n\nCandidate:\n{candidate}\n\nRepresent this candidate."
        for candidate in candidates
    )
    return query_prompt, candidate_prompts


def build_chat_prefill(
    renderer: Any,
    user_content: str | Sequence[dict[str, Any]],
) -> tuple[str, str]:
    """Render a chat template and split it where decision branches are inserted.

    The returned prefix is safe to prefill once. Each semantic decision prompt is
    appended to it, followed by the returned suffix, which closes the user turn
    and opens the assistant generation turn using the model's own template.
    """
    if isinstance(user_content, str):
        validate_context_budget(user_content)
    else:
        validate_context_budget(
            "\n".join(
                str(part.get("text", "")) for part in user_content if part.get("type") == "text"
            )
        )
    marker = _new_branch_marker(user_content)
    marked_content = _append_marker(user_content, marker)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": marked_content},
    ]

    try:
        rendered = renderer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError, ValueError):
        return _plain_fallback(user_content)

    if not isinstance(rendered, str) or marker not in rendered:
        return _plain_fallback(user_content)

    return tuple(rendered.rsplit(marker, maxsplit=1))  # type: ignore[return-value]


def _new_branch_marker(content: str | Sequence[dict[str, Any]]) -> str:
    if isinstance(content, str):
        text = content
    else:
        text = "\n".join(str(part.get("text", "")) for part in content)
    while True:
        marker = f"{_BRANCH_MARKER_PREFIX}{secrets.token_hex(24)}|>"
        if marker not in text:
            return marker


def _append_marker(
    content: str | Sequence[dict[str, Any]],
    marker: str,
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return f"{content}\n{marker}"

    parts = [dict(part) for part in content]
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].get("type") == "text":
            parts[index]["text"] = f"{parts[index].get('text', '')}\n{marker}"
            break
    else:
        parts.append({"type": "text", "text": marker})
    return parts


def _plain_fallback(
    content: str | Sequence[dict[str, Any]],
) -> tuple[str, str]:
    if isinstance(content, str):
        text = content
    else:
        text = "\n".join(
            str(part.get("text", "")) for part in content if part.get("type") == "text"
        )
    return f"System: {SYSTEM_PROMPT}\n\nUser:\n{text}\n", "\n\nAssistant:\n"
