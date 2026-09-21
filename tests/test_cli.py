import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jeeva.cli import (
    _fit_config_from_args,
    build_parser,
    load_schema_from_file,
    main,
)
from jeeva.schema import Choice, DecisionSchema, Noul, Score


def _write_schema(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "category": {
                    "type": "choice",
                    "instructions": "Choose the department.",
                    "criteria": {
                        "billing": "Billing requests.",
                        "support": "Technical support requests.",
                    },
                    "tau": 0.4,
                },
                "is_urgent": {
                    "type": "noul",
                    "instructions": "Does this need immediate attention?",
                    "criteria": {
                        "true": "Immediate attention is required.",
                        "false": "Normal handling is sufficient.",
                    },
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate severity.",
                    "criteria": [
                        {"value": 0, "description": "No impact."},
                        {"value": 2, "description": "Material degradation."},
                        {"value": 4, "description": "Critical outage."},
                    ],
                    "tolerance": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_schema_from_json(tmp_path):
    loaded = load_schema_from_file(str(_write_schema(tmp_path / "schema.json")))
    fields = loaded.fields()

    assert issubclass(loaded, DecisionSchema)
    assert isinstance(fields["category"], Choice)
    assert isinstance(fields["is_urgent"], Noul)
    assert isinstance(fields["severity"], Score)
    assert fields["severity"].level_values == (0.0, 2.0, 4.0)
    assert fields["category"].tau == pytest.approx(0.4)


def test_schema_loader_rejects_executable_python(tmp_path):
    schema_file = tmp_path / "schema.py"
    schema_file.write_text("raise RuntimeError('must not execute')", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        load_schema_from_file(str(schema_file))


def test_schema_loader_rejects_oversized_json_before_parsing(tmp_path):
    schema_file = tmp_path / "oversized.json"
    schema_file.write_bytes(b" " * 1_048_577)

    with pytest.raises(ValueError, match="byte limit"):
        load_schema_from_file(str(schema_file))


@pytest.mark.parametrize("command", ["decide", "fit", "evaluate", "serve"])
def test_cli_exposes_canonical_commands(command, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["jeeva", command, "--help"])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 0
    assert f"usage: jeeva {command}" in capsys.readouterr().out


def test_fit_flags_cover_public_configuration():
    parser = build_parser()
    args = parser.parse_args(
        [
            "fit",
            "--model",
            "model",
            "--architecture",
            "encoder",
            "--schema",
            "schema.json",
            "--data",
            "train.jsonl",
            "--output",
            "artifact",
            "--epochs",
            "4",
            "--progress-updates-per-epoch",
            "0",
            "--learning-rate",
            "0.0001",
            "--head-learning-rate",
            "0.001",
            "--decoder-head",
            "letter",
            "--scheduler",
            "linear",
            "--minimum-learning-rate-ratio",
            "0.1",
            "--weight-decay",
            "0.02",
            "--maximum-gradient-norm",
            "0.5",
            "--adaptation",
            "lora",
            "--lora-rank",
            "16",
            "--lora-alpha",
            "32",
            "--lora-dropout",
            "0.1",
            "--batch-size",
            "2",
            "--max-length",
            "256",
            "--brier-weight",
            "0.2",
            "--ordinal-weight",
            "0.3",
            "--no-save-best",
            "--no-early-stopping",
            "--validation-fraction",
            "0.25",
            "--seed",
            "7",
            "--target-precision",
            "0.9",
            "--minimum-calibration-examples",
            "10",
        ]
    )

    config = _fit_config_from_args(args)

    assert config.epochs == 4
    assert config.progress_updates_per_epoch == 0
    assert config.learning_rate == pytest.approx(0.0001)
    assert config.head_learning_rate == pytest.approx(0.001)
    assert config.decoder_head == "letter"
    assert config.scheduler == "linear"
    assert config.minimum_learning_rate_ratio == pytest.approx(0.1)
    assert config.weight_decay == pytest.approx(0.02)
    assert config.maximum_gradient_norm == pytest.approx(0.5)
    assert config.adaptation == "lora"
    assert config.lora_rank == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == pytest.approx(0.1)
    assert config.batch_size == 2
    assert config.max_length == 256
    assert config.brier_weight == pytest.approx(0.2)
    assert config.ordinal_weight == pytest.approx(0.3)
    assert config.save_best is False
    assert config.early_stopping_patience is None
    assert config.validation_fraction == pytest.approx(0.25)
    assert config.seed == 7
    assert config.target_precision == pytest.approx(0.9)
    assert config.minimum_calibration_examples == 10


def test_fit_accepts_qlora_adaptation():
    args = build_parser().parse_args(
        [
            "fit",
            "--model",
            "model",
            "--schema",
            "schema.json",
            "--data",
            "train.jsonl",
            "--output",
            "artifact",
            "--adaptation",
            "qlora",
        ]
    )

    assert _fit_config_from_args(args).adaptation == "qlora"


def test_decide_uses_decision_model_and_json_schema(tmp_path, monkeypatch, capsys):
    schema_path = _write_schema(tmp_path / "schema.json")
    calls = []

    class FakeModel:
        def __init__(self, source, **kwargs):
            calls.append(("init", source, kwargs))

        def decide(self, context, schema, arbiter=None):
            calls.append(("decide", context, schema, arbiter))
            field = SimpleNamespace(
                value="billing",
                distribution={"billing": 0.8, "support": 0.2},
                margin=0.6,
                certainty_score=0.6,
                score=0.8,
                entropy=None,
                normalized_entropy=None,
                winning_level=None,
                standard_deviation=None,
                is_certain=True,
                calibration="unvalidated",
                tau=0.3,
                threshold_source="default",
                resolved_by="gatekeeper",
                reasoning=None,
            )
            return SimpleNamespace(
                value={"category": "billing"},
                fields={"category": field},
                is_certain=True,
                is_ambiguous=False,
                calibration="unvalidated",
                telemetry={"total_ms": 1.2},
            )

    monkeypatch.setattr("jeeva.cli.DecisionModel", FakeModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jeeva",
            "decide",
            "--model",
            "model",
            "--revision",
            "model-commit",
            "--tokenizer-revision",
            "tokenizer-commit",
            "--local-files-only",
            "--schema",
            str(schema_path),
            "--context",
            "A duplicate charge.",
            "--json",
        ],
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output["value"] == {"category": "billing"}
    assert output["fields"]["category"]["distribution"]["billing"] == pytest.approx(0.8)
    assert calls[0][0:2] == ("init", "model")
    assert calls[1][0:2] == ("decide", "A duplicate charge.")
    assert calls[0][2]["revision"] == "model-commit"
    assert calls[0][2]["tokenizer_revision"] == "tokenizer-commit"
    assert calls[0][2]["local_files_only"] is True


def test_fit_saves_public_artifact(tmp_path, monkeypatch, capsys):
    schema_path = _write_schema(tmp_path / "schema.json")
    training_path = tmp_path / "train.jsonl"
    training_path.write_text("", encoding="utf-8")
    calls = []

    class TrainedModel:
        architecture = "encoder"
        training_report = {"epochs": 1}

        def save(self, destination, *, merge=False):
            calls.append(("save", Path(destination), merge))
            return Path(destination)

    class FakeModel:
        def __init__(self, source, **kwargs):
            calls.append(("init", source, kwargs))

        def fit(self, examples, *, schema, validation=None, config=None):
            calls.append(("fit", Path(examples), schema, validation, config))
            return TrainedModel()

    monkeypatch.setattr("jeeva.cli.DecisionModel", FakeModel)
    output_path = tmp_path / "artifact"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jeeva",
            "fit",
            "--model",
            "encoder",
            "--architecture",
            "encoder",
            "--schema",
            str(schema_path),
            "--data",
            str(training_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    assert calls[0] == (
        "init",
        "encoder",
        {
            "architecture": "encoder",
            "device": "auto",
            "dtype": "auto",
            "revision": None,
            "tokenizer_revision": None,
            "local_files_only": None,
        },
    )
    assert calls[1][0:2] == ("fit", training_path)
    assert calls[2] == ("save", output_path, False)
    assert f"Saved encoder artifact to {output_path}" in capsys.readouterr().out
