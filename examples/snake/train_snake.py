"""
Fine-tunes Qwen2.5-0.5B on synthetic Snake gameplay using Jeeva's
typed DecisionModel API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.snake.play_snake import DirectionDecision
from jeeva import DecisionModel, FitConfig


def main():
    parser = argparse.ArgumentParser(description="Fine-tune 0.5B model on Snake gameplay.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base HF model (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("examples/snake/data/snake_train.jsonl"),
        help="Path to training JSONL",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=Path("examples/snake/data/snake_eval.jsonl"),
        help="Path to evaluation JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/snake/checkpoints/snake_adapter"),
        help="Checkpoint destination",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device, such as auto, mps, cuda, or cpu.",
    )
    args = parser.parse_args()

    train_path = args.train_data
    eval_path = args.eval_data
    config = FitConfig(epochs=args.epochs, learning_rate=args.lr)

    if not train_path.exists():
        print(f"\nPreparing data: {train_path} was not found; generating synthetic examples.")

        from examples.snake.generate_data import generate_synthetic_dataset

        train_path.parent.mkdir(parents=True, exist_ok=True)
        train_recs = generate_synthetic_dataset(num_samples=250, seed=42)
        eval_recs = generate_synthetic_dataset(num_samples=50, seed=1337)
        with open(train_path, "w") as f:
            for r in train_recs:
                f.write(json.dumps(r) + "\n")
        with open(eval_path, "w") as f:
            for r in eval_recs:
                f.write(json.dumps(r) + "\n")

    model = DecisionModel(args.model, architecture="causal", device=args.device)
    trained = model.fit(
        train_path,
        schema=DirectionDecision,
        validation=eval_path if eval_path.exists() else None,
        config=config,
    )
    artifact = trained.save(args.output_dir)

    print("\nVerifying the trained artifact with one sample decision.")
    sample_state = (
        "Game: Snake (Grid 10x10)\n"
        "Head: (5, 5)\n"
        "Food: (5, 2) | Target relative: dx=0 (ALIGNED), dy=-3 (NORTH)\n"
        "Current Direction: RIGHT\n"
        "Snake Length: 3 | Score: 0\n"
        "Hazards: UP=SAFE, DOWN=SAFE, LEFT=DANGER, RIGHT=SAFE\n"
        "Board Map (H=Head, o=Body, F=Food, .=Empty):\n"
        ". . . . . . . . . .\n"
        ". . . . . . . . . .\n"
        ". . . . . F . . . .\n"
        ". . . . . . . . . .\n"
        ". . . . . . . . . .\n"
        ". . . o o H . . . .\n"
        ". . . . . . . . . .\n"
        ". . . . . . . . . .\n"
        ". . . . . . . . . .\n"
        ". . . . . . . . . .\n"
    )

    result = trained.decide(sample_state, DirectionDecision)
    print(f"Decided Move: {result.value['direction']}")
    print(f"Certainty:    {result.is_certain}")
    print(f"Telemetry:    {result.telemetry}")
    print(f"\nArtifact saved: {artifact}")
    print(
        "Run: uv run python -m examples.snake.play_snake "
        f"--model {artifact} --device {args.device} --steps 200"
    )


if __name__ == "__main__":
    main()
