import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

from jeeva import (  # type: ignore[attr-defined]
    Choice,
    DecisionModel,
    DecisionSchema,
    Noul,
    Score,
    SequenceArbiter,
    TargetDistribution,
)
from jeeva.core.engine import EvaluationResult
from jeeva.model import ArtifactError, Example, FitConfig, SchemaError
from jeeva.runtimes.encoder import (
    CandidateMarkerHead,
    EncoderRuntime,
    load_candidate_marker_head,
    save_candidate_marker_head,
)
from jeeva.schema import compile_decision_schema
from jeeva.training.encoder import EncoderTrainer, EncoderTrainingConfig


class MixedDecision(DecisionSchema):
    category = Choice(
        "Choose a category.",
        {
            "alpha": "The state belongs to alpha.",
            "beta": "The state belongs to beta.",
            "gamma": "The state belongs to gamma.",
        },
    )
    allowed = Noul(
        "Is the action allowed?",
        {
            True: "The action is allowed.",
            False: "The action is not allowed.",
        },
    )
    severity = Score(
        "Rate severity.",
        {0: "none", 2: "moderate", 4: "critical"},
        tolerance=1.0,
    )


class CategoryDecision(DecisionSchema):
    category = Choice(
        "Choose a category.",
        {
            "alpha": "The state belongs to alpha.",
            "beta": "The state belongs to beta.",
            "gamma": "The state belongs to gamma.",
        },
    )


class _Encoding(dict):
    def __init__(self, values, sequence_ids):
        super().__init__(values)
        self._sequence_ids = sequence_ids

    def sequence_ids(self, batch_index):
        return self._sequence_ids[batch_index]


class _Tokenizer:
    mask_token = "[MASK]"
    mask_token_id = 99
    unk_token = "[UNK]"
    is_fast = True
    padding_side = "right"

    def _suffix_ids(self, suffix):
        return [self.mask_token_id if token == self.mask_token else 20 for token in suffix.split()]

    def __call__(
        self,
        first,
        second=None,
        *,
        add_special_tokens=True,
        padding=False,
        truncation=None,
        max_length=None,
        return_tensors=None,
    ):
        del add_special_tokens, padding, truncation, max_length, return_tensors
        if isinstance(first, str):
            return {"input_ids": [101, 10, 102, *self._suffix_ids(str(second)), 102]}

        assert second is not None
        rows = []
        sequence_rows = []
        for suffix in second:
            suffix_ids = self._suffix_ids(suffix)
            rows.append([101, 10, 102, *suffix_ids, 102])
            sequence_rows.append([None, 0, None, *([1] * len(suffix_ids)), None])
        width = max(map(len, rows))
        attention = []
        token_types = []
        for index, row in enumerate(rows):
            real_length = len(row)
            pad = width - real_length
            rows[index] = row + [0] * pad
            attention.append([1] * real_length + [0] * pad)
            token_types.append([0 if item != 1 else 1 for item in sequence_rows[index]] + [0] * pad)
            sequence_rows[index] += [None] * pad
        return _Encoding(
            {
                "input_ids": torch.tensor(rows),
                "attention_mask": torch.tensor(attention),
                "token_type_ids": torch.tensor(token_types),
            },
            sequence_rows,
        )

    def save_pretrained(self, destination):
        (Path(destination) / "tokenizer.json").write_text("{}", encoding="utf-8")


class _PositionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(hidden_size=4, sliding_window=16)
        self.calls = 0
        self.last_attention_mask = None
        self.last_position_ids = None

    def forward(self, input_ids, attention_mask, position_ids=None):
        self.calls += 1
        self.last_attention_mask = attention_mask
        self.last_position_ids = position_ids
        if position_ids is None:
            positions = torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.float32,
            ).repeat(input_ids.shape[0], 1)
        else:
            positions = position_ids.to(dtype=torch.float32)
        hidden = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
        hidden[..., 0] = positions + self.anchor
        return SimpleNamespace(last_hidden_state=hidden)

    def save_pretrained(self, destination, **kwargs):
        assert kwargs["safe_serialization"] is True
        (Path(destination) / "config.json").write_text("{}", encoding="utf-8")
        (Path(destination) / "model.safetensors").write_bytes(b"")


class _PositionHead(torch.nn.Module):
    def forward(self, marker_hidden):
        return marker_hidden[..., 0]


def _runtime(max_length=256):
    return EncoderRuntime.from_components(
        model=_PositionModel(),
        tokenizer=_Tokenizer(),
        head=cast(Any, _PositionHead()),
        device="cpu",
        max_length=max_length,
    )


def test_encoder_scores_choice_noul_and_score_in_one_batch():
    runtime = _runtime()
    primitives = list(compile_decision_schema(MixedDecision).primitives.values())

    results = runtime.evaluate_batch(["first", "contains [MASK] text"], primitives)

    assert runtime.model.calls == 1
    assert len(results) == 2
    assert set(results[0].decisions) == {"category", "allowed", "severity"}
    assert len(results[0].decisions["category"].distribution) == 3
    assert len(results[0].decisions["allowed"].distribution) == 2
    assert len(results[0].decisions["severity"].distribution) == 3
    assert results[0].telemetry["architecture"] == "encoder"
    assert results[0].telemetry["batch_size"] == 2
    assert results[0].values == results[1].values


def test_candidate_marker_head_has_nonlinear_capacity():
    head = CandidateMarkerHead(768)

    assert sum(parameter.numel() for parameter in head.parameters()) == 297_985


def test_candidate_marker_head_rejects_unsafe_saved_dimensions(tmp_path: Path):
    save_candidate_marker_head(CandidateMarkerHead(8, 4), tmp_path)
    config_path = tmp_path / "candidate_marker_head.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["scorer_size"] = 100_000_000
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="parameters; maximum"):
        load_candidate_marker_head(
            tmp_path,
            hidden_size=8,
            device="cpu",
            required=True,
        )


def test_candidate_marker_scores_are_invariant_to_candidate_order():
    runtime = _runtime()
    primitives = list(compile_decision_schema(MixedDecision).primitives.values())
    original, _ = runtime.score_batch(["document"], primitives)
    reversed_orders = [list(reversed(range(len(primitive.candidates)))) for primitive in primitives]
    reordered, _ = runtime.score_batch(["document"], primitives, reversed_orders)

    for expected, actual in zip(original, reordered, strict=True):
        torch.testing.assert_close(actual, expected)


def test_candidate_markers_are_isolated_and_share_position_origin():
    runtime = _runtime()
    primitive = next(iter(compile_decision_schema(CategoryDecision).primitives.values()))
    batch, _, _, marker_positions, _ = runtime._encode(["document"], [primitive])
    markers = marker_positions[0]
    full_attention = batch["attention_mask"]["full_attention"][0, 0]
    position_ids = batch["position_ids"][0]

    assert len(markers) == 3
    assert len({int(position_ids[position]) for position in markers}) == 1
    assert bool(full_attention[markers[0], 1])
    assert bool(full_attention[markers[0], markers[0]])
    assert not bool(full_attention[markers[0], markers[1]])
    assert not bool(full_attention[1, markers[0]])


def test_encoder_renders_field_and_candidate_semantics_in_packed_rows():
    class DescribedDecision(DecisionSchema):
        category = Choice(
            "Choose [MASK] one.",
            {
                "world": "Diplomacy [MASK] and elections.",
                "sports": "Teams and matches.",
            },
            examples={
                "world": "International diplomacy.",
                "sports": "A championship match.",
            },
        )

    primitive = next(iter(compile_decision_schema(DescribedDecision).primitives.values()))
    runtime = _runtime()
    field_prompt = runtime._field_prompt(primitive, "Current [MASK] context.")
    world_block = runtime._candidate_block(primitive, 0)
    sports_block = runtime._candidate_block(primitive, 1)

    assert "Instructions:\nChoose [UNK] one." in field_prompt
    assert "Context:\nCurrent [UNK] context." in field_prompt
    assert "Candidate value:\nworld" in world_block
    assert "Candidate criterion:\nDiplomacy [UNK] and elections." in world_block
    assert "Examples satisfying this candidate:\n- International diplomacy." in world_block
    assert "Candidate value:\nsports" in sports_block
    assert world_block.count("[MASK]") == 1
    assert sports_block.count("[MASK]") == 1


def test_encoder_has_no_twenty_six_option_ceiling():
    choices = {f"option-{index}": f"Semantic option {index}." for index in range(30)}

    class LargeTaxonomy(DecisionSchema):
        category = Choice("Choose one.", choices)

    result = _runtime(max_length=512).evaluate(
        "document",
        list(compile_decision_schema(LargeTaxonomy).primitives.values()),
    )
    assert result.decisions["category"].value in choices
    assert len(result.decisions["category"].distribution) == 30


def test_encoder_rejects_candidate_suffix_that_cannot_fit():
    primitives = list(compile_decision_schema(MixedDecision).primitives.values())
    with pytest.raises(ValueError, match="candidate set does not fit"):
        _runtime(max_length=8).evaluate("document", primitives)


def test_encoder_artifact_requires_a_valid_context_budget(tmp_path):
    manifest = {
        "format_version": 11,
        "kind": "trained",
        "base_model": "encoder",
        "runtime": {
            "architecture": "encoder",
            "scorer": "candidate_marker",
            "weights": "full",
        },
        "schema_protocol": "decision-schema-v4",
        "source": {
            "model": {
                "requested_id": "encoder",
                "requested_revision": None,
                "resolved_id": "encoder",
                "resolved_revision": None,
            },
            "tokenizer": {
                "requested_id": "encoder",
                "requested_revision": None,
                "resolved_id": "encoder",
                "resolved_revision": None,
            },
            "local_files_only": False,
        },
        "provenance": {
            "created_at": "2026-01-01T00:00:00+00:00",
            "versions": {
                "jeeva": "0.1.0",
                "python": "3.14.0",
                "torch": "2.9.0",
                "transformers": "5.0.0",
            },
        },
    }
    (tmp_path / "jeeva.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactError, match="runtime.max_length"):
        DecisionModel(tmp_path, device="cpu")


def test_raw_encoder_requires_fit_and_rejects_primary_model_arbiter():
    compiled = compile_decision_schema(MixedDecision)
    with patch("jeeva.model.EncoderRuntime") as runtime_class:
        model = DecisionModel("encoder", architecture="encoder", device="cpu")
        with pytest.raises(ArtifactError, match="must be fitted"):
            model.decide("document", MixedDecision)

        model._trained = True
        model._manifest["schemas"] = {compiled.fingerprint: dict(compiled.descriptor)}
        with pytest.raises(ValueError, match="cannot reuse an encoder"):
            model.decide("document", MixedDecision, arbiter=SequenceArbiter())
        model._validate_arbiter(SequenceArbiter(model_or_id="dedicated-causal-model"))
    runtime_class.return_value.evaluate.assert_not_called()


def test_trained_encoder_rejects_schema_not_seen_during_fit():
    compiled = compile_decision_schema(MixedDecision)
    with patch("jeeva.model.EncoderRuntime") as runtime_class:
        model = DecisionModel("encoder", architecture="encoder", device="cpu")
        model._trained = True
        model._manifest["schemas"] = {compiled.fingerprint: dict(compiled.descriptor)}

        with pytest.raises(SchemaError, match="CategoryDecision.*was not used"):
            model.decide("document", CategoryDecision)

    runtime_class.return_value.evaluate.assert_not_called()


class _FakeEncoderRuntime:
    processor = None
    is_multimodal = False
    scorer = "candidate_marker"
    architecture = "encoder"

    def __init__(
        self,
        source,
        *,
        device,
        dtype,
        max_length,
        trained,
        trainable=False,
        revision=None,
        tokenizer_revision=None,
        local_files_only=False,
    ):
        del device, dtype, max_length, trainable, local_files_only
        self.trained = trained
        self.model = _PositionModel()
        self.tokenizer = _Tokenizer()
        self.head = CandidateMarkerHead(4)
        self.resolved_model_id = source
        self.resolved_model_revision = revision
        self.resolved_tokenizer_id = source
        self.resolved_tokenizer_revision = tokenizer_revision or revision

    @classmethod
    def from_components(cls, *, model, tokenizer, head, device, max_length, **kwargs):
        del kwargs
        runtime = cls.__new__(cls)
        runtime.trained = True
        runtime.model = model
        runtime.tokenizer = tokenizer
        runtime.head = head
        runtime.device = device
        runtime.max_length = max_length
        runtime.resolved_model_id = "encoder"
        runtime.resolved_model_revision = None
        runtime.resolved_tokenizer_id = "encoder"
        runtime.resolved_tokenizer_revision = None
        return runtime


def test_head_adaptation_freezes_encoder_and_optimizes_only_marker_head(tmp_path):
    config = EncoderTrainingConfig(
        epochs=1,
        learning_rate=0.0,
        head_learning_rate=1e-3,
        weight_decay=0.01,
        maximum_gradient_norm=1.0,
        scheduler="constant",
        minimum_learning_rate_ratio=0.05,
        adaptation="head",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        batch_size=2,
        max_length=128,
        brier_weight=0.5,
        ordinal_weight=0.5,
        save_best=True,
        early_stopping_patience=None,
        seed=17,
        progress_updates_per_epoch=0,
        output_dir=str(tmp_path),
        device="cpu",
        dtype="float32",
    )
    primitives = list(compile_decision_schema(MixedDecision).primitives.values())

    with patch("jeeva.training.encoder.EncoderRuntime", _FakeEncoderRuntime):
        trainer = EncoderTrainer("encoder", primitives, config, trained_source=False)

    assert trainer.actual_adaptation == "head"
    assert trainer.backbone_learning_rate is None
    assert all(not parameter.requires_grad for parameter in trainer.runtime.model.parameters())
    assert all(parameter.requires_grad for parameter in trainer.runtime.head.parameters())
    assert {id(parameter) for parameter in trainer._optimized_parameters} == {
        id(parameter) for parameter in trainer.runtime.head.parameters()
    }


def test_preloaded_encoder_is_cast_to_training_dtype(tmp_path):
    config = EncoderTrainingConfig(
        epochs=1,
        learning_rate=0.0,
        head_learning_rate=1e-3,
        weight_decay=0.01,
        maximum_gradient_norm=1.0,
        scheduler="constant",
        minimum_learning_rate_ratio=0.05,
        adaptation="head",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        batch_size=2,
        max_length=128,
        brier_weight=0.5,
        ordinal_weight=0.5,
        save_best=True,
        early_stopping_patience=None,
        seed=17,
        progress_updates_per_epoch=0,
        output_dir=str(tmp_path),
        device="cpu",
        dtype="float32",
    )
    primitives = list(compile_decision_schema(MixedDecision).primitives.values())
    runtime = _FakeEncoderRuntime(
        "encoder",
        device="cpu",
        dtype="float16",
        max_length=128,
        trained=False,
    )
    runtime.model.half()
    runtime.head.half()

    with patch("jeeva.training.encoder.EncoderRuntime", _FakeEncoderRuntime):
        trainer = EncoderTrainer(runtime, primitives, config, trained_source=False)

    assert {parameter.dtype for parameter in trainer.runtime.model.parameters()} == {torch.float32}
    assert {parameter.dtype for parameter in trainer.runtime.head.parameters()} == {torch.float32}


class _FakeEncoderTrainer:
    instances = []

    def __init__(self, source, primitives, config, *, trained_source):
        self.source = source
        self.primitives = primitives
        self.config = config
        self.trained_source = trained_source
        self.runtime = source
        self.instances.append(self)

    def train(self, train_dataset, validation_dataset):
        del train_dataset, validation_dataset
        destination = Path(self.config.output_dir)
        config_name = "adapter_config.json" if self.config.adaptation == "lora" else "config.json"
        (destination / config_name).write_text("{}", encoding="utf-8")
        return {
            "history": [{"train_loss": 0.1}],
            "adaptation": self.config.adaptation,
            "architecture": "encoder",
            "scorer": "candidate_marker",
        }


def test_encoder_artifact_round_trip_and_continued_fit(tmp_path):
    _FakeEncoderTrainer.instances.clear()
    examples = [
        Example(
            "document",
            {"category": "alpha", "allowed": True, "severity": 2.0},
        )
    ]
    config = FitConfig(epochs=1, validation_fraction=0.0, adaptation="full")
    with (
        patch("jeeva.model.EncoderRuntime", _FakeEncoderRuntime),
        patch("jeeva.training.encoder.EncoderTrainer", _FakeEncoderTrainer),
    ):
        base = DecisionModel("encoder", architecture="encoder", device="cpu")
        trained = base.fit(examples, schema=MixedDecision, config=config)
        artifact = trained.save(tmp_path / "encoder-artifact")
        loaded = DecisionModel(artifact, device="cpu")
        continued = loaded.fit(examples, schema=MixedDecision, config=config)

    manifest = json.loads((artifact / "jeeva.json").read_text())
    assert manifest["format_version"] == 11
    assert manifest["runtime"] == {
        "architecture": "encoder",
        "max_length": 512,
        "scorer": "candidate_marker",
        "weights": "full",
    }
    head_config = json.loads((artifact / "candidate_marker_head.json").read_text())
    assert head_config == {
        "format_version": 1,
        "architecture": "candidate_marker_mlp",
        "hidden_size": 4,
        "scorer_size": 64,
    }
    assert loaded.architecture == "encoder"
    assert continued.architecture == "encoder"
    assert _FakeEncoderTrainer.instances[-1].trained_source is True


def test_encoder_fit_calibrates_indexed_jsonl_validation_examples(tmp_path):
    class CalibrationEncoderRuntime(_FakeEncoderRuntime):
        def evaluate_batch(self, contexts, primitives):
            del primitives
            return [
                EvaluationResult(
                    decisions={},
                    candidate_scores={"category": (3.0, 1.0, 0.0)},
                    telemetry={},
                )
                for _ in contexts
            ]

    dataset = tmp_path / "examples.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"context": f"document {index}", "category": "alpha"}) for index in range(6)
        ),
        encoding="utf-8",
    )
    config = FitConfig(
        epochs=1,
        batch_size=2,
        validation_fraction=0.5,
        minimum_calibration_examples=1,
        adaptation="full",
    )

    with (
        patch("jeeva.model.EncoderRuntime", CalibrationEncoderRuntime),
        patch("jeeva.training.encoder.EncoderTrainer", _FakeEncoderTrainer),
    ):
        trained = DecisionModel("encoder", architecture="encoder", device="cpu").fit(
            dataset,
            schema=CategoryDecision,
            config=config,
        )

    field = trained._calibration[compile_decision_schema(CategoryDecision).fingerprint]["category"]
    assert field["samples"] >= 1


def test_encoder_auto_adaptation_uses_lora():
    _FakeEncoderTrainer.instances.clear()
    examples = [
        Example(
            "document",
            {"category": "alpha", "allowed": True, "severity": 2.0},
        )
    ]
    with (
        patch("jeeva.model.EncoderRuntime", _FakeEncoderRuntime),
        patch("jeeva.training.encoder.EncoderTrainer", _FakeEncoderTrainer),
    ):
        base = DecisionModel("encoder", architecture="encoder", device="cpu")
        trained = base.fit(
            examples,
            schema=MixedDecision,
            config=FitConfig(
                epochs=1,
                validation_fraction=0.0,
            ),
        )
        trained.fit(
            examples,
            schema=MixedDecision,
            config=FitConfig(
                epochs=1,
                validation_fraction=0.0,
                adaptation="auto",
            ),
        )

    assert [instance.config.adaptation for instance in _FakeEncoderTrainer.instances] == [
        "lora",
        "lora",
    ]


def test_encoder_calibration_is_batched_and_requires_enough_evidence():
    class CalibrationRuntime:
        batch_sizes: list[int] = []

        def evaluate_batch(self, contexts, primitives):
            del primitives
            self.batch_sizes.append(len(contexts))
            return [
                EvaluationResult(
                    decisions={},
                    candidate_scores={"category": (3.0, 1.0, 0.0)},
                    telemetry={},
                )
                for _ in contexts
            ]

    runtime = CalibrationRuntime()
    model = DecisionModel.__new__(DecisionModel)
    model._runtime = cast(Any, runtime)
    compiled = compile_decision_schema(CategoryDecision)
    examples = [Example(f"document {index}", {"category": "alpha"}) for index in range(5)]

    calibration = model._calibrate_encoder(
        examples,
        compiled,
        FitConfig(batch_size=2, minimum_calibration_examples=5),
    )
    insufficient = model._calibrate_encoder(
        examples[:4],
        compiled,
        FitConfig(batch_size=2, minimum_calibration_examples=5),
    )

    field = calibration[compiled.fingerprint]["category"]
    assert runtime.batch_sizes == [2, 2, 1, 2, 2]
    assert field["brier_after"] <= field["brier_before"]
    assert field["samples"] == 5
    assert insufficient == {}


def test_encoder_temperature_calibration_accepts_soft_reference_targets():
    class CalibrationRuntime:
        def evaluate_batch(self, contexts, primitives):
            del primitives
            return [
                EvaluationResult(
                    decisions={},
                    candidate_scores={"category": (2.0, 1.0, 0.0)},
                    telemetry={},
                )
                for _ in contexts
            ]

    model = DecisionModel.__new__(DecisionModel)
    model._runtime = cast(Any, CalibrationRuntime())
    compiled = compile_decision_schema(CategoryDecision)
    examples = [
        Example(
            f"document {index}",
            {"category": TargetDistribution({"alpha": 0.6, "beta": 0.3, "gamma": 0.1})},
        )
        for index in range(3)
    ]

    calibration = model._calibrate_encoder(
        examples,
        compiled,
        FitConfig(batch_size=2, minimum_calibration_examples=3),
    )

    field = calibration[compiled.fingerprint]["category"]
    assert field["samples"] == 3
    assert "temperature" in field
    assert "threshold" not in field
