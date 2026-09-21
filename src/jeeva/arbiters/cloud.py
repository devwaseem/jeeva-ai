import json
import math
import os
import re
from typing import Any

import torch

from jeeva.arbiters.base import BaseArbiter
from jeeva.core.prompts import validate_context_budget
from jeeva.schema.primitives import BasePrimitive, DecisionResult


class CloudArbiter(BaseArbiter):
    """Resolve ambiguous fields through an OpenAI-compatible endpoint.

    Providers return one probability for every allowed candidate. Jeeva validates
    the complete distribution and derives the selected value from its unique
    maximum. Provider and response failures are raised instead of becoming
    fabricated decisions.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        timeout: float = 30.0,
        client: Any | None = None,
        mock_responses: dict[str, dict[str, Any]] | None = None,
    ):
        self.model = model
        self.api_key = (
            api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("JEEVA_CLOUD_API_KEY")
        )
        self.base_url = base_url
        self.system_prompt = system_prompt or (
            "You are an expert decision arbiter for structured enterprise extraction. "
            "Analyze the document context and resolve the ambiguous field strictly adhering to the schema. "
            "Return valid JSON with keys: 'selected', 'confidence' (float 0.0-1.0), and 'reasoning'."
        )
        self.timeout = timeout
        self.client = client
        self.mock_responses = mock_responses or {}

    @property
    def name(self) -> str:
        return f"CloudArbiter:{self.model}"

    def _call_remote(
        self,
        document: str,
        instruction: str,
        schema_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch isolated document data and trusted instructions to a provider."""
        # 1. Injected Mock / Offline Override
        field_key = schema_info.get("field_key", "")
        if field_key in self.mock_responses:
            return self.mock_responses[field_key]
        if "default" in self.mock_responses:
            return self.mock_responses["default"]
        system_message = f"{self.system_prompt}\n\n{instruction}"
        document_message = json.dumps({"document": document}, ensure_ascii=False)

        # 2. Custom Client Provided (e.g., OpenAI, MockClient)
        if self.client is not None:
            if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": document_message},
                    ],
                    response_format={"type": "json_object"}
                    if "json" in self.model.lower() or "gpt" in self.model.lower()
                    else None,
                    timeout=self.timeout,
                )
                raw_text = response.choices[0].message.content
                return self._parse_json_response(raw_text)
            elif callable(self.client):
                res = self.client(
                    json.dumps(
                        {
                            "system": self.system_prompt,
                            "instruction": instruction,
                            "document": document,
                        },
                        ensure_ascii=False,
                    )
                )
                if isinstance(res, dict):
                    return res
                return self._parse_json_response(str(res))

        # 3. Standard OpenAI SDK if installed and API key present
        if self.api_key:
            try:
                import openai  # pyright: ignore[reportMissingImports]
            except ImportError as exc:
                raise ImportError(
                    "CloudArbiter API-key clients require the cloud extra. "
                    "Install: pip install 'jeeva-ai[cloud]'"
                ) from exc

            sdk_client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = sdk_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": document_message},
                ],
                timeout=self.timeout,
            )
            raw_text = response.choices[0].message.content
            return self._parse_json_response(raw_text)

        raise RuntimeError(
            "CloudArbiter requires an API key, an injected client, or an explicit "
            f"mock response for field {field_key!r}."
        )

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Extract and parse one size-bounded JSON object from a provider response."""
        if len(text.encode("utf-8")) > 65_536:
            raise ValueError("Cloud arbiter response exceeds the 65536-byte limit.")
        cleaned = re.sub(r"^```json\s*", "", text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"^```\s*$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match is None:
                raise ValueError("Cloud arbiter response did not contain a JSON object.") from None
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("Cloud arbiter response must be a JSON object.")
        return parsed

    def resolve(
        self,
        context: str,
        ambiguous_fields: dict[str, BasePrimitive],
        current_decisions: dict[str, DecisionResult],
        engine: Any,
        base_cache: Any,
        past_len: int,
        threshold: float,
    ) -> dict[str, DecisionResult]:
        validate_context_budget(context)
        resolved: dict[str, DecisionResult] = {}

        for field_key, primitive in ambiguous_fields.items():
            options = primitive.options
            labels = primitive.distribution_labels
            options_fmt = "\n".join(f"- {option}" for option in options)
            probability_template = ", ".join(f"{json.dumps(label)}: null" for label in labels)
            instruction = (
                f"{primitive.prompt}\n\n"
                f"Allowed Candidate Options:\n{options_fmt}\n\n"
                f"Prior Preliminary Assessment: {current_decisions.get(field_key, None)}\n\n"
                "Replace every null below with a finite, nonnegative probability. "
                "Probabilities must sum to 1 and have one unique highest value.\n"
                "Respond with JSON format:\n"
                f'{{"probabilities": {{{probability_template}}}, '
                f'"reasoning": "<concise explanation>"}}'
            )
            schema_info = {
                "field_key": field_key,
                "options": options,
                "probability_labels": labels,
                "primitive": primitive.kind,
            }
            parsed = self._call_remote(context, instruction, schema_info)
            probability_map = parsed.get("probabilities")
            if not isinstance(probability_map, dict):
                raise ValueError(
                    f"Cloud arbiter response for {field_key!r} must contain a probabilities object."
                )
            if any(not isinstance(label, str) for label in probability_map):
                raise ValueError(
                    f"Cloud arbiter probability labels for {field_key!r} must be strings."
                )
            received_labels = set(probability_map)
            expected_labels = set(labels)
            if received_labels != expected_labels:
                missing = sorted(expected_labels - received_labels)
                unknown = sorted(received_labels - expected_labels)
                raise ValueError(
                    f"Cloud arbiter probabilities for {field_key!r} must match the "
                    f"candidate labels exactly; missing={missing!r}, unknown={unknown!r}."
                )

            values: list[float] = []
            for label in labels:
                raw_probability = probability_map[label]
                if isinstance(raw_probability, bool) or not isinstance(
                    raw_probability, (int, float)
                ):
                    raise ValueError(
                        f"Cloud arbiter probability for {field_key!r}/{label!r} must be a number."
                    )
                probability = float(raw_probability)
                if not math.isfinite(probability) or probability < 0.0:
                    raise ValueError(
                        f"Cloud arbiter probability for {field_key!r}/{label!r} "
                        "must be finite and nonnegative."
                    )
                values.append(probability)

            total = sum(values)
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"Cloud arbiter probabilities for {field_key!r} must sum to 1; "
                    f"received {total:.12g}."
                )
            probability_tensor = torch.tensor(values, dtype=torch.float32)
            maximum = torch.max(probability_tensor)
            winners = torch.nonzero(probability_tensor == maximum).flatten()
            if winners.numel() != 1:
                raise ValueError(
                    f"Cloud arbiter probabilities for {field_key!r} must have one "
                    "unique highest value."
                )

            reasoning = str(parsed.get("reasoning", "Resolved by CloudArbiter."))
            result = primitive.result_from_probabilities(
                probability_tensor,
                cascade_threshold=threshold,
                resolved_by=self.name,
                reasoning=reasoning,
            )
            resolved[field_key] = result

        return resolved
