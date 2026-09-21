import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from jeeva import (
    BaseArbiter,
    CalibratedDecisionSchema,
    Choice,
    CloudArbiter,
    DecisionModel,
    DecisionSchema,
    Noul,
    Score,
    SequenceArbiter,
    SequenceThinkingArbiter,
    TargetDistribution,
)
from jeeva.core.engine import EvaluationResult
from jeeva.heads.metric import DECODER_HEAD_DIR, BilinearMetricHead
from jeeva.model import ArtifactError, Example, FitConfig, SchemaError, TrainingDataError
from jeeva.schema import compile_decision_schema
from jeeva.schema.primitives import DecisionResult


class Routing(DecisionSchema):
    department = Choice(
        "Choose the department that should handle the request.",
        {
            "billing": "Billing requests.",
            "support": "Technical support requests.",
        },
    )
    urgent = Noul(
        "Does the request require an immediate response?",
        {
            True: "An immediate response is required.",
            False: "A normal response time is acceptable.",
        },
    )


class MountainTopic(DecisionSchema):
    about_mountains = Noul(
        "Is the text primarily about mountains?",
        {
            True: "Mountains are the primary topic.",
            False: "Mountains are not the primary topic.",
        },
    )


class SeverityDecision(DecisionSchema):
    severity = Score(
        "Rate severity.",
        {0: "none", 2: "moderate", 4: "critical"},
        tolerance=1.0,
    )


def test_example_rejects_oversized_context_before_training():
    with pytest.raises(ValueError, match="character limit"):
        Example("x" * 100_001, {"severity": 2})


def _routing_result() -> EvaluationResult:
    return EvaluationResult(
        decisions={
            "department": DecisionResult(
                field="department",
                primitive="choice",
                value="billing",
                is_certain=True,
                margin=0.8,
                raw_score=0.9,
                distribution={"billing": 0.9, "support": 0.1},
            ),
            "urgent": DecisionResult(
                field="urgent",
                primitive="noul",
                value=True,
                is_certain=True,
                margin=0.7,
                raw_score=0.85,
                distribution={"true": 0.85, "false": 0.15},
            ),
        },
        telemetry={"device": "cpu"},
    )


def _mountain_result() -> EvaluationResult:
    primitive = compile_decision_schema(MountainTopic).primitives["about_mountains"]
    return EvaluationResult(
        decisions={
            "about_mountains": primitive.result_from_probabilities(torch.tensor([0.9, 0.1]))
        },
        telemetry={"device": "cpu"},
    )


def _routing_score_result(
    department: tuple[float, float],
    urgent: tuple[float, float],
) -> EvaluationResult:
    result = _routing_result()
    result.candidate_scores = {
        "department": department,
        "urgent": urgent,
    }
    return result


def test_decision_model_returns_dictionary_and_field_evidence():
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        engine_class.return_value.evaluate.return_value = _routing_result()
        decision = DecisionModel("test-model", device="cpu").decide("I was charged twice.", Routing)

    assert decision.value == {"department": "billing", "urgent": True}
    assert decision.calibration == "unvalidated"
    assert decision.is_certain is True
    assert decision.fields["department"].distribution["billing"] == pytest.approx(0.9)
    assert decision.fields["urgent"].tau == pytest.approx(0.3)
    assert decision.fields["urgent"].threshold_source == "default"
    assert decision.telemetry["scorer"] == "letter"


def test_candidate_position_balancing_is_default_for_causal_models():
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        default_model = DecisionModel("test-model", device="cpu")

    assert default_model.candidate_position_balancing is True
    assert engine_class.call_args.kwargs["candidate_position_balancing"] is True

    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        single_order_model = DecisionModel(
            "test-model",
            device="cpu",
            candidate_position_balancing=False,
        )

    assert single_order_model.candidate_position_balancing is False
    assert engine_class.call_args.kwargs["candidate_position_balancing"] is False


def test_candidate_position_balancing_is_inactive_for_encoder_defaults():
    with patch("jeeva.model.EncoderRuntime"):
        model = DecisionModel("encoder-model", architecture="encoder", device="cpu")

    assert model.candidate_position_balancing is False


def test_candidate_position_balancing_rejects_non_letter_runtimes():
    with pytest.raises(ValueError, match="requires decoder_head='letter'"):
        DecisionModel(
            "encoder-model",
            architecture="encoder",
            device="cpu",
            candidate_position_balancing=True,
        )


def test_decision_model_returns_score_evidence():
    primitive = compile_decision_schema(SeverityDecision).primitives["severity"]
    result = EvaluationResult(
        decisions={"severity": primitive.result_from_probabilities(torch.tensor([0.1, 0.7, 0.2]))},
        telemetry={"device": "cpu"},
    )
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        engine_class.return_value.evaluate.return_value = result
        decision = DecisionModel("test-model", device="cpu").decide(
            "Degraded service.",
            SeverityDecision,
        )

    assert decision.value["severity"] == pytest.approx(2.2)
    assert decision.fields["severity"].winning_level == 2.0
    assert decision.fields["severity"].standard_deviation is not None
    assert decision.fields["severity"].certainty_score == pytest.approx(
        result.decisions["severity"].certainty_score
    )


def test_arbiters_are_part_of_the_public_package():
    assert issubclass(SequenceArbiter, BaseArbiter)
    assert issubclass(SequenceThinkingArbiter, BaseArbiter)
    assert issubclass(CloudArbiter, BaseArbiter)


def test_decision_model_warmup_delegates_to_loaded_engine():
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        model = DecisionModel("test-model", device="cpu")
        model.warmup()

    engine_class.return_value.warmup.assert_called_once_with()


def test_decide_forwards_arbiter_and_preserves_resolution_evidence():
    result = _routing_result()
    result.decisions["urgent"].resolved_by = "SequenceArbiter"
    result.decisions["urgent"].reasoning = "The outage language makes urgency explicit."
    arbiter = SequenceArbiter()
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        engine_class.return_value.evaluate.return_value = result
        decision = DecisionModel("test-model", device="cpu").decide(
            "Checkout is down.", Routing, arbiter=arbiter
        )
    assert engine_class.return_value.evaluate.call_args.kwargs["arbiter"] is arbiter
    assert decision.fields["urgent"].resolved_by == "SequenceArbiter"
    assert decision.fields["urgent"].reasoning == "The outage language makes urgency explicit."


def test_decide_many_forwards_arbiter():
    arbiter = SequenceArbiter()
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        engine_class.return_value.evaluate_batch.return_value = [
            _routing_result(),
            _routing_result(),
        ]
        decisions = DecisionModel("test-model", device="cpu").decide_many(
            ["First", "Second"], Routing, arbiter=arbiter
        )
    assert len(decisions) == 2
    assert engine_class.return_value.evaluate_batch.call_args.kwargs["arbiter"] is arbiter


def test_uncalibrated_decision_preserves_zero_shot_ambiguity():
    result = _routing_result()
    result.decisions["urgent"] = DecisionResult(
        field="urgent",
        primitive="noul",
        value=True,
        is_certain=False,
        margin=0.05,
        raw_score=0.525,
        distribution={"true": 0.525, "false": 0.475},
    )
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        engine_class.return_value.evaluate.return_value = result
        decision = DecisionModel("test-model", device="cpu").decide("Maybe urgent.", Routing)
    assert decision.fields["urgent"].is_certain is False
    assert decision.is_ambiguous is True


def test_schema_calibration_averages_internal_null_scores_and_forwards_tare():
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        engine = engine_class.return_value
        engine.evaluate.side_effect = [
            _routing_score_result((2.0, 1.0), (0.6, 0.4)),
            _routing_score_result((4.0, 1.0), (0.9, 0.1)),
            _routing_score_result((3.0, 1.0), (0.3, 0.7)),
            _routing_result(),
        ]
        model = DecisionModel("test-model", device="cpu")

        calibrated = Routing.calibrate(model)
        decision = model.decide("A duplicate charge needs attention.", calibrated)
        engine.evaluate_batch.return_value = [_routing_result()]
        batch = model.decide_many(["A second charge."], calibrated)

    assert isinstance(calibrated, CalibratedDecisionSchema)
    assert calibrated.schema is Routing
    assert calibrated.score_offsets["department"] == pytest.approx((3.0, 1.0))
    assert calibrated.score_offsets["urgent"] == pytest.approx((0.6, 0.4))
    assert decision.value == {"department": "billing", "urgent": True}
    assert batch[0].value["department"] == "billing"

    calibration_calls = engine.evaluate.call_args_list[:3]
    assert [item.kwargs["context"] for item in calibration_calls] == ["", "N/A", "[MASK]"]
    assert all(item.kwargs["_allow_empty_context"] is True for item in calibration_calls)
    assert engine.evaluate.call_args.kwargs["_score_offsets"] == calibrated.score_offsets

    assert engine.evaluate_batch.call_args.kwargs["_score_offsets"] == calibrated.score_offsets


def test_calibrated_schema_is_bound_to_the_model_that_measured_it():
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        engine_class.return_value.evaluate.side_effect = [
            _routing_score_result((2.0, 1.0), (0.6, 0.4)),
            _routing_score_result((2.0, 1.0), (0.6, 0.4)),
            _routing_score_result((2.0, 1.0), (0.6, 0.4)),
        ]
        source_model = DecisionModel("test-model", device="cpu")
        other_model = DecisionModel("test-model", device="cpu")
        calibrated = Routing.calibrate(source_model)
        engine_class.return_value.evaluate.reset_mock()

        with pytest.raises(SchemaError, match="different DecisionModel"):
            other_model.decide("text", calibrated)

    engine_class.return_value.evaluate.assert_not_called()


def test_trained_model_rejects_context_free_calibration():
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        engine_class.return_value.evaluate.side_effect = [
            _routing_score_result((2.0, 1.0), (0.6, 0.4)),
            _routing_score_result((2.0, 1.0), (0.6, 0.4)),
            _routing_score_result((2.0, 1.0), (0.6, 0.4)),
        ]
        model = DecisionModel("test-model", device="cpu")
        calibrated = Routing.calibrate(model)
        model._trained = True
        engine_class.return_value.evaluate.reset_mock()

        with pytest.raises(SchemaError, match="cannot be used with trained models"):
            model.decide("text", calibrated)
        with pytest.raises(SchemaError, match="only available for raw causal models"):
            Routing.calibrate(model)

    engine_class.return_value.evaluate.assert_not_called()


def test_fit_rejects_context_free_calibrated_schema():
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        engine_class.return_value.evaluate.side_effect = [
            _routing_score_result((2.0, 1.0), (0.6, 0.4)),
            _routing_score_result((2.0, 1.0), (0.6, 0.4)),
            _routing_score_result((2.0, 1.0), (0.6, 0.4)),
        ]
        model = DecisionModel("test-model", device="cpu")
        calibrated = Routing.calibrate(model)

        with pytest.raises(SchemaError, match=r"fit\(\) requires a raw DecisionSchema"):
            model.fit([], schema=calibrated)  # type: ignore[arg-type]


def test_schema_fingerprint_is_structural_and_excludes_tau():
    class SameContract(DecisionSchema):
        department = Choice(
            "Choose a department.",
            {"billing": "Billing.", "support": "Support."},
        )
        urgent = Noul(
            "Is this urgent?",
            {True: "Urgent.", False: "Not urgent."},
            tau=0.9,
        )

    class SameContractDifferentTau(DecisionSchema):
        department = Choice(
            "Choose a department.",
            {"billing": "Billing.", "support": "Support."},
        )
        urgent = Noul(
            "Is this urgent?",
            {True: "Urgent.", False: "Not urgent."},
            tau=0.1,
        )

    first = compile_decision_schema(SameContract)
    second = compile_decision_schema(SameContractDifferentTau)
    assert first.fingerprint == second.fingerprint
    assert first.primitives["urgent"].key == "urgent"
    assert SameContract.urgent.key is None


def test_schema_fingerprint_includes_examples():
    class First(DecisionSchema):
        answer = Noul(
            "It is true.",
            {True: "The condition is satisfied.", False: "The condition is not satisfied."},
            examples={True: "One."},
        )

    class Second(DecisionSchema):
        answer = Noul(
            "It is true.",
            {True: "The condition is satisfied.", False: "The condition is not satisfied."},
            examples={True: "Two."},
        )

    assert compile_decision_schema(First).fingerprint != compile_decision_schema(Second).fingerprint


def test_choice_fingerprint_includes_instructions_criteria_order_descriptions_and_examples():
    class Base(DecisionSchema):
        answer = Choice(
            "Choose an answer.",
            {"yes": "Affirmative.", "no": "Negative."},
        )

    class DifferentInstructions(DecisionSchema):
        answer = Choice(
            "Choose the correct answer.",
            {"yes": "Affirmative.", "no": "Negative."},
        )

    class Reordered(DecisionSchema):
        answer = Choice(
            "Choose an answer.",
            {"no": "Negative.", "yes": "Affirmative."},
        )

    class DifferentDescription(DecisionSchema):
        answer = Choice(
            "Choose an answer.",
            {"yes": "The condition holds.", "no": "Negative."},
        )

    class WithExamples(DecisionSchema):
        answer = Choice(
            "Choose an answer.",
            {"yes": "Affirmative.", "no": "Negative."},
            examples={"yes": "The answer is affirmative."},
        )

    base = compile_decision_schema(Base).fingerprint
    assert base != compile_decision_schema(DifferentInstructions).fingerprint
    assert base != compile_decision_schema(Reordered).fingerprint
    assert base != compile_decision_schema(DifferentDescription).fingerprint
    assert base != compile_decision_schema(WithExamples).fingerprint


def test_empty_decision_schema_is_rejected():
    with pytest.raises(TypeError, match="at least one"):

        class Empty(DecisionSchema):
            pass


def test_fit_config_is_the_single_advanced_training_configuration():
    config = FitConfig(
        epochs=7,
        learning_rate=1e-4,
        scheduler="linear",
        lora_rank=16,
        progress_updates_per_epoch=4,
    )
    assert config.effective_lora_alpha == 32
    assert config.progress_updates_per_epoch == 4
    assert config.decoder_head == "letter"
    assert FitConfig(decoder_head="bilinear").decoder_head == "bilinear"
    assert FitConfig(decoder_head="contextual").decoder_head == "contextual"
    assert FitConfig(adaptation="head").adaptation == "head"
    assert FitConfig(adaptation="qlora").adaptation == "qlora"
    with pytest.raises(ValueError, match="scheduler"):
        FitConfig(scheduler="quadratic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decoder_head"):
        FitConfig(decoder_head="dense")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decoder_head"):
        FitConfig(decoder_head="model")  # type: ignore[arg-type]


def test_fit_config_generates_a_fresh_seed_unless_one_is_configured():
    with patch("jeeva.model.secrets.randbelow", side_effect=[7, 11]):
        first = FitConfig()
        second = FitConfig()

    assert first.seed == 7
    assert second.seed == 11
    assert FitConfig(seed=42).seed == 42


def test_causal_fit_rejects_head_only_adaptation():
    with patch("jeeva.runtimes.causal.DecisionEngine"):
        model = DecisionModel("test-model", device="cpu")
        with pytest.raises(ValueError, match="supported only by encoder"):
            model.fit(
                [Example("Please help.", {"department": "support", "urgent": False})],
                schema=Routing,
                config=FitConfig(adaptation="head", validation_fraction=0.0),
            )


def test_encoder_rejects_decoder_head_selection():
    with patch("jeeva.model.EncoderRuntime"):
        model = DecisionModel("encoder-model", architecture="encoder", device="cpu")
    with pytest.raises(ValueError, match="encoders use candidate_marker"):
        model.fit(
            [Example("Please help.", {"department": "support", "urgent": False})],
            schema=Routing,
            config=FitConfig(decoder_head="bilinear", validation_fraction=0.0),
        )


def test_encoder_rejects_qlora_adaptation():
    with patch("jeeva.model.EncoderRuntime"):
        model = DecisionModel("encoder-model", architecture="encoder", device="cpu")
    with pytest.raises(ValueError, match="causal text models"):
        model.fit(
            [Example("Please help.", {"department": "support", "urgent": False})],
            schema=Routing,
            config=FitConfig(adaptation="qlora", validation_fraction=0.0),
        )


def test_causal_qlora_rejects_cpu_before_loading_trainer():
    with patch("jeeva.runtimes.causal.DecisionEngine"):
        model = DecisionModel("test-model", device="cpu")
    with pytest.raises(ValueError, match="MPS or CUDA"):
        model.fit(
            [Example("Please help.", {"department": "support", "urgent": False})],
            schema=Routing,
            config=FitConfig(adaptation="qlora", validation_fraction=0.0),
        )


def test_causal_fit_reuses_loaded_model_without_second_runtime_load():
    runtime_instances = []
    trainer_instances = []

    class MemoryRuntime:
        architecture = "causal"

        def __init__(self, source, **kwargs):
            del kwargs
            self.source = source
            self.model = torch.nn.Linear(2, 2)
            self.tokenizer = SimpleNamespace()
            self.processor = None
            self.engine = SimpleNamespace(head=None)
            self.resolved_model_id = source
            self.resolved_model_revision = None
            self.resolved_tokenizer_id = source
            self.resolved_tokenizer_revision = None
            runtime_instances.append(self)

        @classmethod
        def from_components(cls, *, model, tokenizer, processor, source, **kwargs):
            del kwargs
            runtime = runtime_instances[0]
            runtime.model = model
            runtime.tokenizer = tokenizer
            runtime.processor = processor
            runtime.source = source
            return runtime

    class MemoryTrainer:
        def __init__(
            self,
            model_or_id,
            schema,
            config,
            tokenizer=None,
            processor=None,
            head=None,
        ):
            del schema, processor, head
            self.model_or_id = model_or_id
            self.model = model_or_id
            self.tokenizer = tokenizer
            self.processor = None
            self.head = None
            self.config = config
            self.torch_dtype = torch.float32
            trainer_instances.append(self)

        def train(self, train_dataset, eval_dataset=None):
            del train_dataset, eval_dataset
            Path(self.config.output_dir, "adapter_config.json").write_text(
                '{"base_model_name_or_path":"test-model"}',
                encoding="utf-8",
            )
            Path(self.config.output_dir, "adapter_model.safetensors").write_bytes(b"")
            return {
                "history": [{"train_loss": 0.1}],
                "output_dir": self.config.output_dir,
                "adaptation": "lora",
            }

    with (
        patch("jeeva.model.CausalRuntime", MemoryRuntime),
        patch("jeeva.training.trainer.Trainer", MemoryTrainer),
    ):
        base = DecisionModel("test-model", device="cpu")
        trained = base.fit(
            [Example("Please help.", {"department": "support", "urgent": False})],
            schema=Routing,
            config=FitConfig(validation_fraction=0.0),
        )

    assert len(runtime_instances) == 1
    assert trained is base
    assert trainer_instances[0].model_or_id is runtime_instances[0].model


class _FakeTrainer:
    instances = []

    def __init__(
        self,
        model_or_id,
        schema,
        config,
        tokenizer=None,
        processor=None,
        head=None,
    ):
        self.model_or_id = model_or_id
        self.schema = schema
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.head = head
        self.model = model_or_id
        self.torch_dtype = torch.float32
        self.instances.append(self)

    def train(self, train_dataset, eval_dataset=None):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        with open(f"{self.config.output_dir}/adapter_config.json", "w", encoding="utf-8") as file:
            json.dump({"base_model_name_or_path": str(self.model_or_id)}, file)
        Path(self.config.output_dir, "adapter_model.safetensors").write_bytes(b"")
        if self.config.decoder_head in ("bilinear", "contextual"):
            if self.head is None:
                self.head = BilinearMetricHead(
                    hidden_dim=4,
                    input_strategy=(
                        "contextual" if self.config.decoder_head == "contextual" else "static"
                    ),
                )
            self.head.save_pretrained(f"{self.config.output_dir}/{DECODER_HEAD_DIR}")
        report = {
            "history": [{"train_loss": 0.1}],
            "output_dir": self.config.output_dir,
            "adaptation": self.config.adaptation,
        }
        if self.config.adaptation == "qlora":
            report["quantization"] = {
                "backend": "bitsandbytes",
                "backend_version": "0.50.2",
                "bits": 4,
                "quant_type": "nf4",
                "double_quantization": True,
                "compute_dtype": "float16",
            }
        return report


def test_qlora_fit_persists_quantization_and_auto_continuation():
    _FakeTrainer.instances.clear()
    examples = [Example("Please help.", {"department": "support", "urgent": False})]

    with (
        patch("jeeva.runtimes.causal.DecisionEngine") as engine_class,
        patch("jeeva.training.trainer.Trainer", _FakeTrainer),
    ):
        engine_class.return_value.device = "mps"
        engine_class.return_value.torch_dtype = torch.float16
        base = DecisionModel("test-model", device="mps")
        trained = base.fit(
            examples,
            schema=Routing,
            config=FitConfig(adaptation="qlora", validation_fraction=0.0),
        )
        continued = trained.fit(
            examples,
            schema=Routing,
            config=FitConfig(adaptation="auto", validation_fraction=0.0),
        )

    first, second = _FakeTrainer.instances
    assert first.config.adaptation == "qlora"
    assert first.config.dtype == "float16"
    assert second.config.adaptation == "qlora"
    assert trained is base
    assert continued is base
    assert base._runtime is not None
    assert continued._manifest["runtime"]["weights"] == "adapter"
    assert continued._manifest["fit_config"]["adaptation"] == "qlora"
    report = continued.training_report
    assert report is not None
    assert report["adaptation"] == "qlora"
    assert report["quantization"]["quant_type"] == "nf4"
    assert report["resolved_training_config"]["quantization"]["bits"] == 4
    assert continued._manifest["provenance"]["versions"]["bitsandbytes"] == "0.50.2"


def test_fit_streams_large_jsonl_training_and_validation_files(tmp_path):
    _FakeTrainer.instances.clear()
    training_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    nested = json.dumps(
        {"context": "I was charged twice.", "target": {"department": "billing", "urgent": True}}
    )
    flat = json.dumps({"text": "Please send my invoice.", "department": "billing", "urgent": False})
    training_path.write_text(
        "\n".join(nested if index % 2 == 0 else flat for index in range(10_000)) + "\n",
        encoding="utf-8",
    )
    validation_path.write_text(nested + "\n", encoding="utf-8")

    with (
        patch("jeeva.runtimes.causal.DecisionEngine") as engine_class,
        patch("jeeva.training.trainer.Trainer", _FakeTrainer),
    ):
        engine_class.return_value.evaluate.return_value = _routing_result()
        trained = DecisionModel("test-model", device="cpu").fit(
            training_path,
            schema=Routing,
            validation=validation_path,
            config=FitConfig(minimum_calibration_examples=1),
        )

    trainer = _FakeTrainer.instances[0]
    assert len(trainer.train_dataset) == 10_000
    assert len(trainer.eval_dataset) == 1
    assert not isinstance(trainer.train_dataset.records, list)
    assert trainer.train_dataset[9_999]["document"] == "Please send my invoice."
    assert trainer.train_dataset[9_999]["lane_mask"].tolist() == [1.0, 1.0]
    assert trained._manifest["runtime"]["scorer"] == "letter"


def test_fit_automatic_validation_split_keeps_groups_intact():
    _FakeTrainer.instances.clear()
    examples = [
        Example(
            f"{group}-{variant}",
            {"department": "billing", "urgent": False},
            group=group,
        )
        for group in ("template-a", "template-b", "template-c")
        for variant in ("one", "two")
    ]

    with (
        patch("jeeva.runtimes.causal.DecisionEngine"),
        patch("jeeva.training.trainer.Trainer", _FakeTrainer),
    ):
        DecisionModel("test-model", device="cpu").fit(
            examples,
            schema=Routing,
            config=FitConfig(validation_fraction=0.34, seed=7),
        )

    trainer = _FakeTrainer.instances[0]
    training_documents = {
        trainer.train_dataset[index]["document"] for index in range(len(trainer.train_dataset))
    }
    validation_documents = {
        trainer.eval_dataset[index]["document"] for index in range(len(trainer.eval_dataset))
    }
    assert training_documents
    assert validation_documents
    for group in ("template-a", "template-b", "template-c"):
        group_documents = {f"{group}-one", f"{group}-two"}
        assert not (group_documents & training_documents and group_documents & validation_documents)


def test_fit_jsonl_reports_bad_lines_and_invalid_labels(tmp_path):
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"context":"valid","urgent":true}\n{"context":"broken",', encoding="utf-8")
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"context":"bad","urgent":"yes"}\n', encoding="utf-8")
    with patch("jeeva.runtimes.causal.DecisionEngine"):
        model = DecisionModel("test-model", device="cpu")
        with pytest.raises(TrainingDataError, match=r"broken\.jsonl:2: invalid JSON"):
            model.fit(broken, schema=Routing)
        with pytest.raises(TrainingDataError, match="must be a boolean"):
            model.fit(invalid, schema=Routing)


def test_fit_one_schema_maps_config_and_partial_labels(capsys):
    _FakeTrainer.instances.clear()
    examples = [
        Example("I was charged twice.", {"department": "billing", "urgent": True}),
        Example("Please help.", {"department": "support"}),
    ]
    config = FitConfig(
        epochs=2,
        learning_rate=3e-4,
        scheduler="constant",
        validation_fraction=0.0,
        progress_updates_per_epoch=4,
        seed=17,
    )
    with (
        patch("jeeva.runtimes.causal.DecisionEngine"),
        patch("jeeva.training.trainer.Trainer", _FakeTrainer),
    ):
        trained = DecisionModel("test-model", device="cpu").fit(
            examples, schema=Routing, config=config
        )
    output = capsys.readouterr().out
    assert "JEEVA DECISION MODEL TRAINING" in output
    assert "Architecture          Causal decoder" in output
    assert "Backbone              test-model" in output
    assert "Decision schema       Routing" in output
    assert "Training examples     2" in output
    assert "Epochs                2" in output
    assert "TRAINING COMPLETE" in output
    trainer = _FakeTrainer.instances[0]
    assert trainer.config.epochs == 2
    assert trainer.config.lr == pytest.approx(3e-4)
    assert len(trainer.schema) == 2
    assert trainer.train_dataset[0]["lane_mask"].sum().item() == 2
    assert trainer.train_dataset[1]["lane_mask"].sum().item() == 1
    assert len(trained._manifest["schemas"]) == 1
    report = trained.training_report
    assert len(report["dataset_digest"]) == 64
    assert report["schema_fingerprint"] == compile_decision_schema(Routing).fingerprint
    assert report["run_id"]
    assert report["created_at"].endswith("+00:00")
    assert report["resolved_training_config"]["adaptation"] == "lora"
    assert report["resolved_training_config"]["training_examples"] == 2
    assert report["resolved_training_config"]["seed"] == 17


@pytest.mark.parametrize(
    ("decoder_head", "input_strategy"),
    [("bilinear", "static"), ("contextual", "contextual")],
)
def test_causal_fit_selects_saves_and_reloads_metric_head(
    tmp_path,
    decoder_head,
    input_strategy,
):
    _FakeTrainer.instances.clear()
    artifact = tmp_path / f"{decoder_head}-artifact"
    with (
        patch("jeeva.runtimes.causal.DecisionEngine") as engine_class,
        patch("jeeva.training.trainer.Trainer", _FakeTrainer),
    ):
        engine_class.from_components.return_value = engine_class.return_value
        engine_class.return_value.device = "cpu"
        engine_class.return_value.torch_dtype = torch.float32
        engine_class.return_value.model.config.hidden_size = 4
        engine_class.return_value.model.config.get_text_config.return_value = SimpleNamespace(
            hidden_size=4
        )
        trained = DecisionModel("test-model", device="cpu").fit(
            [Example("Please help.", {"department": "support", "urgent": False})],
            schema=Routing,
            config=FitConfig(decoder_head=decoder_head, validation_fraction=0.0),
        )

        def save_adapter(destination, **kwargs):
            assert kwargs["safe_serialization"] is True
            Path(destination, "adapter_config.json").write_text("{}", encoding="utf-8")
            Path(destination, "adapter_model.safetensors").write_bytes(b"")

        trained._runtime.model.save_pretrained.side_effect = save_adapter
        trained.save(artifact)
        loaded = DecisionModel(artifact, device="cpu")

    trainer = _FakeTrainer.instances[0]
    assert trainer.config.decoder_head == decoder_head
    assert trainer.config.head_lr == pytest.approx(1e-3)
    assert trainer.config.candidate_position_balancing is False
    assert trained._manifest["runtime"]["scorer"] == decoder_head
    assert (artifact / DECODER_HEAD_DIR / "model.safetensors").is_file()
    assert isinstance(loaded._runtime.engine.head, BilinearMetricHead)
    assert loaded._runtime.engine.head.input_strategy == input_strategy


def test_fit_accepts_mixed_hard_and_soft_python_targets():
    _FakeTrainer.instances.clear()
    examples = [
        Example(
            "I was charged twice.",
            {
                "department": TargetDistribution({"billing": 0.8, "support": 0.2}),
                "urgent": True,
            },
        )
    ]
    with (
        patch("jeeva.runtimes.causal.DecisionEngine"),
        patch("jeeva.training.trainer.Trainer", _FakeTrainer),
    ):
        trained = DecisionModel("test-model", device="cpu").fit(
            examples,
            schema=Routing,
            config=FitConfig(validation_fraction=0.0),
        )

    row = _FakeTrainer.instances[0].train_dataset[0]
    assert row["targets"][0].tolist() == pytest.approx([0.8, 0.2])
    assert row["targets"][1].tolist() == pytest.approx([1.0, 0.0])
    assert trained.training_report["soft_target_fields"] == 1
    assert trained.training_report["hard_target_fields"] == 1


def test_fit_decodes_tagged_soft_targets_from_jsonl(tmp_path):
    _FakeTrainer.instances.clear()
    dataset = tmp_path / "soft.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "context": "I was charged twice.",
                "target": {
                    "department": {"$distribution": {"billing": 0.75, "support": 0.25}},
                    "urgent": {"$distribution": {"true": 0.6, "false": 0.4}},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with (
        patch("jeeva.runtimes.causal.DecisionEngine"),
        patch("jeeva.training.trainer.Trainer", _FakeTrainer),
    ):
        trained = DecisionModel("test-model", device="cpu").fit(
            dataset,
            schema=Routing,
            config=FitConfig(validation_fraction=0.0),
        )

    row = _FakeTrainer.instances[0].train_dataset[0]
    assert row["targets"][0].tolist() == pytest.approx([0.75, 0.25])
    assert row["targets"][1].tolist() == pytest.approx([0.6, 0.4])
    assert trained.training_report["soft_target_fields"] == 2
    assert trained.training_report["hard_target_fields"] == 0


def test_manual_tau_wins_calibration_and_is_reported():
    class ManualRouting(DecisionSchema):
        department = Choice(
            "Choose a department.",
            {"billing": "Billing.", "support": "Support."},
        )
        urgent = Noul(
            "Is this urgent?",
            {True: "Urgent.", False: "Not urgent."},
            tau=0.8,
        )

    compiled = compile_decision_schema(ManualRouting)
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        engine_class.return_value.evaluate.return_value = _routing_result()
        model = DecisionModel("test-model", device="cpu")
        model._calibration = {
            compiled.fingerprint: {
                "urgent": {"threshold": 0.2},
                "department": {"threshold": 0.4},
            }
        }
        decision = model.decide("text", ManualRouting)
    primitives = engine_class.return_value.evaluate.call_args.kwargs["schema"]
    urgent = next(primitive for primitive in primitives if primitive.key == "urgent")
    assert urgent.resolve_tau(0.3) == (0.8, "manual")
    result = _routing_result()
    result.decisions["urgent"].tau = 0.8
    result.decisions["urgent"].threshold_source = "manual"
    assert decision.fields["urgent"].calibration == "validated"


def test_causal_threshold_calibration_ignores_soft_reference_targets():
    model = DecisionModel.__new__(DecisionModel)
    model._runtime = SimpleNamespace(evaluate=lambda context, primitives: _routing_result())
    compiled = compile_decision_schema(Routing)
    examples = [
        Example(
            "Ambiguous request.",
            {
                "department": TargetDistribution({"billing": 0.55, "support": 0.45}),
                "urgent": TargetDistribution({True: 0.6, False: 0.4}),
            },
        )
    ]

    calibration = model._calibrate(
        examples,
        compiled,
        FitConfig(minimum_calibration_examples=1),
    )

    assert calibration == {}


def test_unattainable_precision_is_recorded_without_a_calibrated_threshold():
    results = [_routing_result(), _routing_result()]
    for result in results:
        for decision in result.decisions.values():
            decision.margin = 0.5
    result_iterator = iter(results)
    model = DecisionModel.__new__(DecisionModel)
    model._runtime = SimpleNamespace(evaluate=lambda context, primitives: next(result_iterator))
    compiled = compile_decision_schema(Routing)
    calibration = model._calibrate(
        [
            Example(
                "Incorrect prediction.",
                {"department": "support", "urgent": False},
            ),
            Example(
                "Correct prediction.",
                {"department": "billing", "urgent": True},
            ),
        ],
        compiled,
        FitConfig(target_precision=1.0, minimum_calibration_examples=1),
    )

    fields = calibration[compiled.fingerprint]
    assert fields["department"] == {
        "status": "unattainable",
        "coverage": 1.0,
        "precision": 0.5,
        "samples": 2,
        "target_precision": 1.0,
    }
    assert fields["urgent"] == fields["department"]

    model._calibration = calibration
    model._scorer = "dense"
    model._architecture = "causal"
    decision = model._to_decision(compiled, _routing_result())
    assert decision.calibration == "unvalidated"
    assert all(field.calibration == "unvalidated" for field in decision.fields.values())
    assert all(field.threshold_source == "default" for field in decision.fields.values())


def test_trained_causal_model_accepts_schema_not_seen_during_fit():
    _FakeTrainer.instances.clear()
    with (
        patch("jeeva.runtimes.causal.DecisionEngine") as engine_class,
        patch("jeeva.training.trainer.Trainer", _FakeTrainer),
    ):
        engine_class.from_components.return_value = engine_class.return_value
        engine_class.return_value.evaluate.return_value = _mountain_result()
        trained = DecisionModel("test-model", device="cpu").fit(
            [Example("Invoice", {"department": "billing", "urgent": False})],
            schema=Routing,
            config=FitConfig(validation_fraction=0.0),
        )
        decision = trained.decide("Everest is the highest mountain.", MountainTopic)

    assert decision.value == {"about_mountains": True}
    engine_class.return_value.evaluate.assert_called_once()


def test_incompatible_legacy_artifact_is_rejected(tmp_path):
    artifact = tmp_path / "legacy-adapter"
    artifact.mkdir()
    (artifact / "adapter_config.json").write_text("{}", encoding="utf-8")
    (artifact / "jeeva.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "kind": "trained",
                "base_model": "test-model",
                "scorer": "dense",
                "schema_protocol": "finite-pydantic-v1",
                "calibration": {},
            }
        ),
        encoding="utf-8",
    )
    with patch("jeeva.runtimes.causal.DecisionEngine"):
        with pytest.raises(ArtifactError, match="Unsupported Jeeva artifact format"):
            DecisionModel(artifact, device="cpu")


def test_artifact_manifest_is_size_bounded_before_parsing(tmp_path):
    manifest_path = tmp_path / "jeeva.json"
    manifest_path.write_bytes(b" " * 1_048_577)
    model = object.__new__(DecisionModel)

    with pytest.raises(ArtifactError, match="Could not read"):
        model._load_manifest(manifest_path)


def test_format_eleven_golden_manifest_uses_descriptive_dependency_versions():
    artifact = Path(__file__).parent / "fixtures" / "artifact-v11-zero-shot"
    with patch("jeeva.runtimes.causal.DecisionEngine"):
        loaded = DecisionModel(artifact, device="cpu")

    assert loaded.architecture == "causal"
    assert loaded._base_model == "fixture/model"
    assert loaded._manifest["provenance"]["versions"]["torch"] == "2.6.0"


def test_format_ten_artifact_is_rejected(tmp_path):
    artifact = tmp_path / "format-ten"
    artifact.mkdir()
    manifest = json.loads(
        (Path(__file__).parent / "fixtures" / "artifact-v11-zero-shot" / "jeeva.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["format_version"] = 10
    (artifact / "jeeva.json").write_text(json.dumps(manifest), encoding="utf-8")

    with (
        patch("jeeva.runtimes.causal.DecisionEngine"),
        pytest.raises(ArtifactError, match="format 10"),
    ):
        DecisionModel(artifact, device="cpu")


def test_model_revisions_are_resolved_recorded_and_reused(tmp_path):
    artifact = tmp_path / "revision-pinned"
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        engine = engine_class.return_value
        engine.resolved_model_id = "publisher/model"
        engine.resolved_model_revision = "model-commit"
        engine.resolved_tokenizer_id = "publisher/tokenizer"
        engine.resolved_tokenizer_revision = "tokenizer-commit"

        model = DecisionModel(
            "publisher/model",
            device="cpu",
            revision="release",
            tokenizer_revision="tokenizer-release",
            local_files_only=True,
        )
        model.save(artifact)
        DecisionModel(artifact, device="cpu")

    initial_call, reload_call = engine_class.call_args_list
    assert initial_call.kwargs["revision"] == "release"
    assert initial_call.kwargs["tokenizer_revision"] == "tokenizer-release"
    assert initial_call.kwargs["local_files_only"] is True
    assert reload_call.kwargs["revision"] == "model-commit"
    assert reload_call.kwargs["tokenizer_revision"] == "tokenizer-commit"
    assert reload_call.kwargs["local_files_only"] is True
    manifest = json.loads((artifact / "jeeva.json").read_text(encoding="utf-8"))
    assert manifest["source"] == {
        "model": {
            "requested_id": "publisher/model",
            "requested_revision": "release",
            "resolved_id": "publisher/model",
            "resolved_revision": "model-commit",
        },
        "tokenizer": {
            "requested_id": "publisher/model",
            "requested_revision": "tokenizer-release",
            "resolved_id": "publisher/tokenizer",
            "resolved_revision": "tokenizer-commit",
        },
        "local_files_only": True,
    }


def test_artifact_rejects_revision_overrides():
    artifact = Path(__file__).parent / "fixtures" / "artifact-v11-zero-shot"
    with (
        patch("jeeva.runtimes.causal.DecisionEngine"),
        pytest.raises(ArtifactError, match="own their resolved"),
    ):
        DecisionModel(artifact, revision="other")


def test_current_manifest_requires_provenance(tmp_path):
    artifact = tmp_path / "missing-provenance"
    artifact.mkdir()
    manifest = json.loads(
        (Path(__file__).parent / "fixtures" / "artifact-v11-zero-shot" / "jeeva.json").read_text(
            encoding="utf-8"
        )
    )
    del manifest["provenance"]
    (artifact / "jeeva.json").write_text(json.dumps(manifest), encoding="utf-8")

    with (
        patch("jeeva.runtimes.causal.DecisionEngine"),
        pytest.raises(ArtifactError, match="missing required key 'provenance'"),
    ):
        DecisionModel(artifact, device="cpu")


def test_public_zero_shot_decide_save_reload_lifecycle(tmp_path):
    artifact = tmp_path / "artifact"
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        engine_class.return_value.evaluate.return_value = _routing_result()
        model = DecisionModel("test-model", device="cpu")
        before = model.decide("I was charged twice.", Routing)
        model.save(artifact)
        loaded = DecisionModel(artifact, device="cpu")
        after = loaded.decide("I was charged twice.", Routing)

    manifest = json.loads((artifact / "jeeva.json").read_text())
    assert before.value == after.value == {"department": "billing", "urgent": True}
    assert before.fields["department"].distribution == after.fields["department"].distribution
    assert manifest["kind"] == "zero_shot"
    assert manifest["schema_protocol"] == "decision-schema-v4"
    provenance = manifest["provenance"]
    assert provenance["created_at"].endswith("+00:00")
    assert set(provenance["versions"]) == {"jeeva", "python", "torch", "transformers"}
    assert loaded.architecture == "causal"
    assert loaded._base_model == "test-model"
    assert engine_class.call_args.kwargs["mode"] == "letter"


def test_public_causal_fit_save_reload_lifecycle(tmp_path):
    artifact = tmp_path / "trained-artifact"
    examples = [
        Example("I was charged twice.", {"department": "billing", "urgent": True}),
        Example("Please help with setup.", {"department": "support", "urgent": False}),
    ]

    with (
        patch("jeeva.runtimes.causal.DecisionEngine") as engine_class,
        patch("jeeva.training.trainer.Trainer", _FakeTrainer),
    ):
        engine = engine_class.return_value
        engine_class.from_components.return_value = engine
        engine.device = "cpu"
        engine.torch_dtype = torch.float32
        engine.evaluate.return_value = _routing_result()
        base = DecisionModel("test-model", device="cpu")
        trained = base.fit(
            examples,
            schema=Routing,
            config=FitConfig(epochs=1, validation_fraction=0.0),
        )
        before = trained.decide("I was charged twice.", Routing)

        def save_adapter(destination, **kwargs):
            assert kwargs["safe_serialization"] is True
            Path(destination, "adapter_config.json").write_text("{}", encoding="utf-8")
            Path(destination, "adapter_model.safetensors").write_bytes(b"")

        trained._runtime.model.save_pretrained.side_effect = save_adapter
        trained.save(artifact)
        loaded = DecisionModel(artifact, device="cpu")
        after = loaded.decide("I was charged twice.", Routing)

    manifest = json.loads((artifact / "jeeva.json").read_text())
    assert before.value == after.value == {"department": "billing", "urgent": True}
    assert before.fields["urgent"].distribution == after.fields["urgent"].distribution
    assert manifest["kind"] == "trained"
    assert manifest["runtime"]["architecture"] == "causal"
    assert manifest["runtime"]["weights"] == "adapter"
    assert manifest["schemas"]
    assert manifest["training_report"]["resolved_training_config"]["epochs"] == 1
    assert manifest["provenance"]["versions"]["peft"]
    assert loaded.architecture == "causal"


def _trained_manifest() -> dict[str, object]:
    return {
        "format_version": 11,
        "kind": "trained",
        "base_model": "test-model",
        "runtime": {
            "architecture": "causal",
            "scorer": "letter",
            "weights": "adapter",
        },
        "schema_protocol": "decision-schema-v4",
        "schemas": {compile_decision_schema(Routing).fingerprint: {}},
        "calibration": {},
        "source": {
            "model": {
                "requested_id": "test-model",
                "requested_revision": None,
                "resolved_id": "test-model",
                "resolved_revision": None,
            },
            "tokenizer": {
                "requested_id": "test-model",
                "requested_revision": None,
                "resolved_id": "test-model",
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
                "peft": "0.18.0",
            },
        },
    }


def test_save_rejects_an_existing_empty_destination(tmp_path):
    destination = tmp_path / "artifact"
    destination.mkdir()
    with patch("jeeva.runtimes.causal.DecisionEngine"):
        model = DecisionModel("test-model", device="cpu")
        with pytest.raises(ArtifactError, match="already exists"):
            model.save(destination)
    assert destination.is_dir()
    assert not any(destination.iterdir())


def test_save_removes_staging_directory_after_failure(tmp_path):
    destination = tmp_path / "nested" / "artifact"
    with patch("jeeva.runtimes.causal.DecisionEngine"):
        model = DecisionModel("test-model", device="cpu")
        with (
            patch.object(model, "_write_manifest", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            model.save(destination)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


def test_save_merge_writes_a_self_contained_artifact(tmp_path):
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter_config.json").write_text("{}", encoding="utf-8")
    (source / "adapter_model.safetensors").write_bytes(b"")
    (source / "jeeva.json").write_text(json.dumps(_trained_manifest()), encoding="utf-8")
    destination = tmp_path / "merged"
    with patch("jeeva.runtimes.causal.DecisionEngine") as engine_class:
        model = DecisionModel(source, device="cpu")
        with patch.object(
            model,
            "_save_merged_weights",
            side_effect=lambda path: (
                (path / "config.json").write_text("{}", encoding="utf-8"),
                (path / "model.safetensors").write_bytes(b""),
            ),
        ) as save_merged:
            model.save(destination, merge=True)
        loaded = DecisionModel(destination, device="cpu")
    assert loaded._artifact_kind == "merged"
    staging_path = save_merged.call_args.args[0]
    assert staging_path.parent == destination.parent
    assert staging_path.name.startswith(f".{destination.name}.staging-")
    assert engine_class.call_args.kwargs["model_id"] == str(destination)


def test_save_merge_rejects_unfitted_models_without_creating_destination(tmp_path):
    destination = tmp_path / "merged"
    with patch("jeeva.runtimes.causal.DecisionEngine"):
        model = DecisionModel("test-model", device="cpu")
        with pytest.raises(ArtifactError, match="requires a fitted LoRA adapter"):
            model.save(destination, merge=True)
    assert not destination.exists()


def test_merged_export_unmerges_live_adapter_after_save_failure(tmp_path):
    events = []

    class FakeBaseModel:
        def save_pretrained(self, destination, *, state_dict, safe_serialization):
            assert safe_serialization is True
            events.append(("save", destination, state_dict))
            raise OSError("disk full")

    class FakePeftModel:
        def merge_adapter(self, *, safe_merge):
            events.append(("merge", safe_merge))

        def get_base_model(self):
            return FakeBaseModel()

        def unmerge_adapter(self):
            events.append(("unmerge",))

    model = DecisionModel.__new__(DecisionModel)
    model.__dict__["_runtime"] = SimpleNamespace(model=FakePeftModel())
    with (
        patch("peft.PeftModel", FakePeftModel),
        patch("peft.get_base_model_state_dict", return_value={"weight": "merged"}) as state,
        pytest.raises(ArtifactError, match="disk full"),
    ):
        model._save_merged_weights(tmp_path)
    state.assert_called_once_with(model._runtime.model)
    assert events == [
        ("merge", True),
        ("save", str(tmp_path), {"weight": "merged"}),
        ("unmerge",),
    ]
