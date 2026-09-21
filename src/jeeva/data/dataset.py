import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from jeeva._json import read_jsonl_record
from jeeva.schema.multimodal import MultimodalContext
from jeeva.schema.primitives import BasePrimitive, normalize_schema
from jeeva.schema.targets import TargetDistribution


class JeevaDataset(Dataset):
    """
    Dataset loader for fine-tuning and evaluating Jeeva multi-lane decision models.
    Supports JSONL files and in-memory dictionaries with hybrid hard/soft targets,
    dynamic lane loss masking for partial observations, and multimodal (image+text) inputs.
    """

    def __init__(
        self,
        data: str | Path | Sequence[dict[str, Any]],
        schema: Any,
    ):
        self.primitives = normalize_schema(schema)
        self.records: Sequence[dict[str, Any]]

        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found at: {path}")
            records: list[dict[str, Any]] = []
            with path.open("rb") as source:
                line_num = 0
                while True:
                    try:
                        line = read_jsonl_record(source)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid JSON on line {line_num + 1} in {path}: {exc}"
                        ) from exc
                    if not line:
                        break
                    line_num += 1
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise ValueError(
                            f"Invalid JSON on line {line_num} in {path}: {exc}"
                        ) from exc
            self.records = records
        elif isinstance(data, Sequence):
            self.records = data
        else:
            raise TypeError(
                f"Unsupported data source type: {type(data)}. Expected filepath or sequence of dicts."
            )

    def __len__(self) -> int:
        return len(self.records)

    def _extract_document(self, record: dict[str, Any]) -> str | MultimodalContext:
        text_content: str | None = None
        for candidate_key in ("text", "document", "ticket", "context", "input"):
            if candidate_key in record:
                val = record[candidate_key]
                if isinstance(val, MultimodalContext):
                    return val
                if isinstance(val, str):
                    text_content = val
                    break

        images: list[Any] = []
        if "images" in record:
            raw_images = record["images"]
            if isinstance(raw_images, list):
                images.extend(raw_images)
            else:
                images.append(raw_images)
        elif "image" in record:
            raw_img = record["image"]
            if isinstance(raw_img, list):
                images.extend(raw_img)
            else:
                images.append(raw_img)

        if images:
            prompt_text = (
                text_content if text_content is not None else "Analyze the provided image."
            )
            if not prompt_text.strip():
                prompt_text = "Analyze the provided image."
            return MultimodalContext(text=prompt_text, images=images)

        if text_content is not None:
            return text_content

        raise KeyError(f"Record missing text/document field. Found keys: {list(record.keys())}")

    def _parse_target(
        self,
        primitive: BasePrimitive,
        raw_val: Any,
    ) -> torch.Tensor | None:
        """Convert a hard label or soft distribution into candidate probabilities."""
        if raw_val is None:
            return None
        if isinstance(raw_val, TargetDistribution):
            raw_val = raw_val.probabilities
        return primitive.target_distribution(raw_val)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        document = self._extract_document(record)

        targets: list[torch.Tensor | None] = []
        lane_mask = torch.zeros(len(self.primitives), dtype=torch.float32)

        for lane_idx, prim in enumerate(self.primitives):
            raw_val = record.get(prim.key or f"field_{lane_idx}")
            target_tensor = self._parse_target(prim, raw_val)

            if target_tensor is not None:
                targets.append(target_tensor)
                lane_mask[lane_idx] = 1.0
            else:
                # Missing label for this lane
                targets.append(torch.zeros(len(prim.options), dtype=torch.float32))

        return {
            "document": document,
            "targets": targets,
            "lane_mask": lane_mask,
            "raw_record": record,
        }
