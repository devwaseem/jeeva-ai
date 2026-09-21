"""Bounded JSON input helpers for untrusted local artifacts."""

import json
from pathlib import Path
from typing import Any, BinaryIO

MAX_CONFIG_JSON_BYTES = 1_048_576
MAX_JSON_BYTES = 1_048_576
MAX_JSONL_RECORD_BYTES = 1_048_576


def read_json(path: str | Path, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
    """Read one JSON file without materializing more than ``max_bytes``."""
    source = Path(path)
    with source.open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"JSON file {source} exceeds the {max_bytes}-byte limit.")
    try:
        return json.loads(payload)
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON file {source} is not valid UTF-8.") from exc


def read_jsonl_record(
    source: BinaryIO,
    *,
    max_bytes: int = MAX_JSONL_RECORD_BYTES,
) -> bytes:
    """Read one JSONL record and reject it before an oversized line is materialized."""
    payload = source.readline(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"JSONL record exceeds the {max_bytes}-byte limit.")
    return payload
