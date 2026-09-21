import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from jeeva._json import read_json
from jeeva._terminal import terminal_text
from jeeva.arbiters import BaseArbiter
from jeeva.data.dataset import JeevaDataset
from jeeva.eval.evaluator import Evaluator
from jeeva.model import Decision, DecisionModel, FitConfig
from jeeva.schema import DecisionSchema, decision_schema_from_dict


def _schema_from_mapping(data: Any) -> type[DecisionSchema]:
    return decision_schema_from_dict(data, name="CliDecisionSchema")


def load_schema_from_file(schema_path: str) -> type[DecisionSchema]:
    """Load one data-only JSON schema declaration."""
    path = Path(schema_path)
    if not path.is_file():
        raise FileNotFoundError(f"Schema file not found at: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError("Production CLI schemas must be JSON files with a .json suffix.")
    return _schema_from_mapping(read_json(path))


def _model_from_args(
    args: argparse.Namespace,
    *,
    include_architecture: bool = False,
) -> DecisionModel:
    kwargs: dict[str, Any] = {
        "device": args.device,
        "dtype": args.dtype,
        "revision": args.revision,
        "tokenizer_revision": args.tokenizer_revision,
        "local_files_only": args.local_files_only,
    }
    if include_architecture and args.architecture is not None:
        kwargs["architecture"] = args.architecture
    balancing = getattr(args, "candidate_position_balancing", None)
    if balancing is not None:
        kwargs["candidate_position_balancing"] = balancing
    return DecisionModel(args.model, **kwargs)


def _arbiter_from_args(args: argparse.Namespace) -> BaseArbiter | None:
    strategy = getattr(args, "arbiter", "none")
    if strategy == "none":
        return None
    if strategy == "sequence":
        from jeeva.arbiters import SequenceArbiter

        return SequenceArbiter(
            model_or_id=args.arbiter_model,
            ambiguity_criterion=args.ambiguity_criterion,
        )
    if strategy == "thinking":
        from jeeva.arbiters import SequenceThinkingArbiter

        return SequenceThinkingArbiter(
            model_or_id=args.arbiter_model,
            ambiguity_criterion=args.ambiguity_criterion,
        )
    if strategy == "cloud":
        from jeeva.arbiters import CloudArbiter

        return CloudArbiter(model=args.cloud_model)
    raise ValueError(f"Unsupported arbiter strategy {strategy!r}.")


def _field_to_dict(field: Any) -> dict[str, Any]:
    return {
        "value": field.value,
        "distribution": dict(field.distribution),
        "margin": field.margin,
        "certainty_score": field.certainty_score,
        "score": field.score,
        "entropy": field.entropy,
        "normalized_entropy": field.normalized_entropy,
        "winning_level": field.winning_level,
        "standard_deviation": field.standard_deviation,
        "is_certain": field.is_certain,
        "calibration": field.calibration,
        "tau": field.tau,
        "threshold_source": field.threshold_source,
        "resolved_by": field.resolved_by,
        "reasoning": field.reasoning,
    }


def _decision_to_dict(decision: Decision | Any) -> dict[str, Any]:
    return {
        "value": dict(decision.value),
        "fields": {path: _field_to_dict(field) for path, field in decision.fields.items()},
        "is_certain": decision.is_certain,
        "is_ambiguous": decision.is_ambiguous,
        "calibration": decision.calibration,
        "telemetry": dict(decision.telemetry),
    }


def _print_decision(decision: Decision | Any, *, as_json: bool) -> None:
    payload = _decision_to_dict(decision)
    if as_json:
        print(json.dumps(payload, sort_keys=True))
        return
    print("Decision:")
    for path, field in decision.fields.items():
        status = "certain" if field.is_certain else "ambiguous"
        print(
            f"  {terminal_text(path)}: {terminal_text(repr(field.value))} "
            f"({status}, certainty={field.certainty_score:.3f})"
        )
    total_ms = decision.telemetry.get("total_ms")
    if total_ms is not None:
        print(f"Latency: {float(total_ms):.2f} ms")


def handle_decide(args: argparse.Namespace) -> None:
    context = args.context
    if not context:
        if sys.stdin.isatty():
            raise ValueError("Provide --context or pipe text into stdin.")
        context = sys.stdin.read().strip()
    if not context:
        raise ValueError("Decision context must not be empty.")

    schema = load_schema_from_file(args.schema)
    model = _model_from_args(args)
    decision = model.decide(context, schema, arbiter=_arbiter_from_args(args))
    _print_decision(decision, as_json=args.json)


def _fit_config_from_args(args: argparse.Namespace) -> FitConfig:
    return FitConfig(
        epochs=args.epochs,
        progress_updates_per_epoch=args.progress_updates_per_epoch,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        decoder_head=args.decoder_head,
        scheduler=args.scheduler,
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        weight_decay=args.weight_decay,
        maximum_gradient_norm=args.maximum_gradient_norm,
        adaptation=args.adaptation,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        max_length=args.max_length,
        brier_weight=args.brier_weight,
        ordinal_weight=args.ordinal_weight,
        save_best=args.save_best,
        early_stopping_patience=(None if args.no_early_stopping else args.early_stopping_patience),
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        target_precision=args.target_precision,
        minimum_calibration_examples=args.minimum_calibration_examples,
    )


def handle_fit(args: argparse.Namespace) -> None:
    schema = load_schema_from_file(args.schema)
    model = _model_from_args(args, include_architecture=True)
    trained = model.fit(
        args.data,
        schema=schema,
        validation=args.validation,
        config=_fit_config_from_args(args),
    )
    artifact = trained.save(args.output, merge=args.merge)
    payload = {
        "artifact": str(artifact),
        "architecture": trained.architecture,
        "training_report": (
            dict(trained.training_report) if trained.training_report is not None else None
        ),
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"Saved {terminal_text(trained.architecture)} artifact to {terminal_text(artifact)}")


def handle_evaluate(args: argparse.Namespace) -> None:
    schema = load_schema_from_file(args.schema)
    model = _model_from_args(args)
    dataset = JeevaDataset(data=args.data, schema=schema)
    scorecard = Evaluator(model=model, schema=schema).run(
        dataset=dataset,
        target_precisions=[args.target_precision],
        arbiter=_arbiter_from_args(args),
    )
    if args.json:
        print(json.dumps(scorecard.to_dict(), sort_keys=True))
    else:
        print(scorecard.summary_table())
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(scorecard.to_dict(), indent=2) + "\n", encoding="utf-8")


def handle_serve(args: argparse.Namespace) -> None:
    from jeeva.serve import run_server

    api_key = None if args.allow_unauthenticated else os.environ.get(args.api_key_env)
    if not args.allow_unauthenticated and not api_key:
        raise ValueError(
            f"Set {args.api_key_env} before serving, or explicitly pass "
            "--allow-unauthenticated for local-only use."
        )
    schema = load_schema_from_file(args.schema)
    model = _model_from_args(args)
    run_server(
        model=model,
        schema=schema,
        host=args.host,
        port=args.port,
        arbiter=_arbiter_from_args(args),
        api_key=api_key,
        allow_unauthenticated=args.allow_unauthenticated,
        max_batch_size=args.max_batch_size,
        max_context_chars=args.max_context_chars,
        max_pending_requests=args.max_pending_requests,
        max_body_bytes=args.max_body_bytes,
        enable_docs=args.enable_docs,
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", "-m", required=True, help="Model ID or Jeeva artifact path.")
    parser.add_argument("--device", default="auto", help="Runtime device (default: auto).")
    parser.add_argument("--dtype", default="auto", help="Runtime dtype (default: auto).")
    parser.add_argument(
        "--revision",
        default=None,
        help="Model repository revision (branch, tag, or immutable commit).",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Tokenizer revision; defaults to --revision.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use only locally cached model files.",
    )


def _add_arbiter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--arbiter",
        choices=("none", "sequence", "thinking", "cloud"),
        default="none",
        help="Optional uncertain-field resolver (default: none).",
    )
    parser.add_argument("--arbiter-model", default=None, help="Optional local arbiter model ID.")
    parser.add_argument("--cloud-model", default="gpt-4o-mini", help="Cloud arbiter model ID.")
    parser.add_argument(
        "--ambiguity-criterion",
        choices=("margin", "entropy", "both"),
        default="margin",
        help="Criterion used by local arbiters (default: margin).",
    )


def _add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--progress-updates-per-epoch", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument(
        "--decoder-head",
        choices=("letter", "bilinear", "contextual"),
        default="letter",
    )
    parser.add_argument("--scheduler", choices=("cosine", "linear", "constant"), default="cosine")
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    parser.add_argument(
        "--adaptation",
        choices=("auto", "head", "full", "lora", "qlora"),
        default="auto",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--ordinal-weight", type=float, default=0.25)
    parser.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the best validation epoch (default: true).",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=FitConfig().seed)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--minimum-calibration-examples", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jeeva",
        description="Typed finite decisions, fitting, evaluation, and serving.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    decide = subparsers.add_parser("decide", help="Run one typed decision.")
    _add_model_arguments(decide)
    decide.add_argument("--schema", required=True, help="Path to a JSON schema file.")
    decide.add_argument("--context", "-c", default=None, help="Context text, or read stdin.")
    decide.add_argument("--json", action="store_true", help="Emit JSON instead of human output.")
    decide.add_argument(
        "--candidate-position-balancing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override causal letter candidate-position balancing.",
    )
    _add_arbiter_arguments(decide)
    decide.set_defaults(handler=handle_decide)

    fit = subparsers.add_parser("fit", help="Fit one DecisionSchema and save an artifact.")
    _add_model_arguments(fit)
    fit.add_argument("--architecture", choices=("causal", "encoder"), default=None)
    fit.add_argument("--schema", required=True, help="Path to a JSON schema file.")
    fit.add_argument("--data", required=True, help="Training JSONL path.")
    fit.add_argument("--validation", default=None, help="Optional validation JSONL path.")
    fit.add_argument("--output", required=True, help="Artifact destination directory.")
    fit.add_argument("--merge", action="store_true", help="Save a merged standalone artifact.")
    fit.add_argument("--json", action="store_true", help="Emit JSON instead of human output.")
    _add_fit_arguments(fit)
    fit.set_defaults(handler=handle_fit)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate public decision behavior.")
    _add_model_arguments(evaluate)
    evaluate.add_argument("--schema", required=True, help="Path to a JSON schema file.")
    evaluate.add_argument("--data", required=True, help="Evaluation JSONL path.")
    evaluate.add_argument("--target-precision", type=float, default=0.98)
    evaluate.add_argument("--output-json", default=None, help="Optional scorecard JSON path.")
    evaluate.add_argument("--json", action="store_true", help="Emit JSON instead of human output.")
    evaluate.add_argument(
        "--candidate-position-balancing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override causal letter candidate-position balancing.",
    )
    _add_arbiter_arguments(evaluate)
    evaluate.set_defaults(handler=handle_evaluate)

    serve = subparsers.add_parser("serve", help="Serve one registered schema and model.")
    _add_model_arguments(serve)
    serve.add_argument("--schema", required=True, help="Path to a JSON schema file.")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    serve.add_argument(
        "--api-key-env",
        default="JEEVA_API_KEY",
        help="Environment variable containing the required bearer token.",
    )
    serve.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="Explicitly disable authentication for local-only use.",
    )
    serve.add_argument("--max-batch-size", type=int, default=32)
    serve.add_argument("--max-context-chars", type=int, default=100_000)
    serve.add_argument("--max-pending-requests", type=int, default=64)
    serve.add_argument("--max-body-bytes", type=int, default=1_000_000)
    serve.add_argument(
        "--enable-docs",
        action="store_true",
        help="Expose OpenAPI and Swagger UI endpoints.",
    )
    serve.add_argument(
        "--candidate-position-balancing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override causal letter candidate-position balancing.",
    )
    _add_arbiter_arguments(serve)
    serve.set_defaults(handler=handle_serve)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
