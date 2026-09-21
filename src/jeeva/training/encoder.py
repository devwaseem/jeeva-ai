from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from jeeva._terminal import terminal_text
from jeeva.data.dataset import JeevaDataset
from jeeva.runtimes.encoder import EncoderRuntime, save_candidate_marker_head
from jeeva.schema.multimodal import MultimodalContext
from jeeva.schema.primitives import BasePrimitive, Score
from jeeva.training.loss import (  # type: ignore[attr-defined]
    CompositeDecisionLoss,
    backward_finite_loss,
    clip_finite_gradients,
    require_finite_value,
)
from jeeva.types import resolve_dtype


@dataclass(frozen=True)
class EncoderTrainingConfig:
    epochs: int
    learning_rate: float
    head_learning_rate: float | None
    weight_decay: float
    maximum_gradient_norm: float
    scheduler: Literal["cosine", "linear", "constant"]
    minimum_learning_rate_ratio: float
    adaptation: Literal["head", "full", "lora"]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    max_length: int
    brier_weight: float
    ordinal_weight: float
    save_best: bool
    early_stopping_patience: int | None
    seed: int
    progress_updates_per_epoch: int
    output_dir: str
    device: str
    dtype: str
    revision: str | None = None
    tokenizer_revision: str | None = None
    local_files_only: bool = False


class EncoderTrainer:
    """Supervised candidate-marker training over a bidirectional encoder."""

    def __init__(
        self,
        model_or_id: str | EncoderRuntime,
        primitives: list[BasePrimitive],
        config: EncoderTrainingConfig,
        *,
        trained_source: bool,
    ) -> None:
        self.config = config
        self.primitives = primitives
        torch.manual_seed(config.seed)
        if isinstance(model_or_id, EncoderRuntime):
            self.runtime = model_or_id
            self.runtime.max_length = config.max_length
            training_dtype = resolve_dtype(config.dtype, device=config.device)
            self.runtime.model.to(device=config.device, dtype=training_dtype)
            self.runtime.head.to(device=config.device, dtype=training_dtype)
        else:
            self.runtime = EncoderRuntime(
                model_or_id,
                device=config.device,
                dtype=config.dtype,
                max_length=config.max_length,
                trained=trained_source,
                trainable=True,
                revision=config.revision,
                tokenizer_revision=config.tokenizer_revision,
                local_files_only=config.local_files_only,
            )
        self.actual_adaptation = self._configure_adaptation(config.adaptation)
        self.loss = CompositeDecisionLoss(
            brier_weight=config.brier_weight,
            ordinal_weight=config.ordinal_weight,
        )
        backbone_parameters = [
            parameter for parameter in self.runtime.model.parameters() if parameter.requires_grad
        ]
        head_parameters = [
            parameter for parameter in self.runtime.head.parameters() if parameter.requires_grad
        ]
        self.head_learning_rate = config.head_learning_rate or 1e-3
        parameter_groups: list[dict[str, Any]] = []
        if backbone_parameters:
            parameter_groups.append(
                {
                    "name": "backbone",
                    "params": backbone_parameters,
                    "lr": config.learning_rate,
                }
            )
        parameter_groups.append(
            {
                "name": "head",
                "params": head_parameters,
                "lr": self.head_learning_rate,
            }
        )
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=config.weight_decay,
        )
        self._optimized_parameters = backbone_parameters + head_parameters
        self.backbone_learning_rate = config.learning_rate if backbone_parameters else None

    def _configure_adaptation(self, requested: str) -> str:
        if isinstance(self.runtime.model, PeftModel):
            if requested == "full":
                raise ValueError(
                    "Cannot continue a saved encoder adapter with adaptation='full'; "
                    "load a merged/full artifact or keep adaptation='lora'."
                )
            if requested == "head":
                raise ValueError(
                    "Cannot convert a saved LoRA encoder artifact to head-only training; "
                    "keep adaptation='lora'."
                )
            active_adapter = self.runtime.model.active_adapter
            if not isinstance(active_adapter, str):
                raise ValueError("Continuing training requires exactly one active encoder adapter.")
            self.runtime.model.set_adapter(active_adapter)
            return "lora"
        if requested == "head":
            self.runtime.model.requires_grad_(False)
            return "head"
        if requested == "lora":
            lora = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules="all-linear",
                bias="none",
                task_type="FEATURE_EXTRACTION",
                revision=self.runtime.resolved_model_revision,
            )
            self.runtime.model = get_peft_model(self.runtime.model, lora).to(self.config.device)
            return "lora"
        self.runtime.model.requires_grad_(True)
        return "full"

    def _rows(self, dataset: JeevaDataset, indices: list[int]) -> list[dict[str, Any]]:
        rows = [dataset[index] for index in indices]
        if any(isinstance(row["document"], MultimodalContext) for row in rows):
            raise ValueError("Encoder training currently supports text examples only.")
        return rows

    def _batch_loss(
        self,
        rows: list[dict[str, Any]],
        primitives: list[BasePrimitive],
        option_orders: list[list[int]],
    ) -> torch.Tensor:
        contexts = [str(row["document"]) for row in rows]
        field_logits, _ = self.runtime.score_batch(contexts, primitives, option_orders)
        original_indices = {primitive.key: index for index, primitive in enumerate(self.primitives)}
        total_loss = torch.zeros((), device=self.config.device)
        active_fields = 0
        for primitive, logits in zip(primitives, field_logits, strict=True):
            field_index = original_indices[primitive.key]
            active = torch.tensor(
                [bool(row["lane_mask"][field_index]) for row in rows],
                device=self.config.device,
            )
            if not bool(active.any()):
                continue
            targets = torch.stack([row["targets"][field_index] for row in rows]).to(
                self.config.device
            )
            levels = (
                torch.tensor(primitive.level_values, device=self.config.device)
                if isinstance(primitive, Score)
                else None
            )
            total_loss = total_loss + self.loss(
                logits[active],
                targets[active],
                level_values=levels,
            )
            active_fields += 1
        if active_fields == 0:
            raise ValueError("Training batch contains no labeled decision fields.")
        return total_loss / active_fields

    def _validation_metrics(self, dataset: JeevaDataset) -> dict[str, Any]:
        self.runtime.model.eval()
        self.runtime.head.eval()
        field_stats: dict[str, dict[str, Any]] = {
            primitive.key or primitive.kind: {
                "brier": [],
                "targets": [],
                "predictions": [],
                "swap_matches": [],
            }
            for primitive in self.primitives
        }
        reversed_orders = [
            list(reversed(range(len(primitive.candidates)))) for primitive in self.primitives
        ]
        with torch.inference_mode():
            for start in range(0, len(dataset), self.config.batch_size):
                indices = list(range(start, min(len(dataset), start + self.config.batch_size)))
                rows = self._rows(dataset, indices)
                contexts = [str(row["document"]) for row in rows]
                logits_by_field, _ = self.runtime.score_batch(contexts, self.primitives)
                swapped_by_field, _ = self.runtime.score_batch(
                    contexts,
                    self.primitives,
                    reversed_orders,
                )
                for field_index, (primitive, logits, swapped_logits) in enumerate(
                    zip(
                        self.primitives,
                        logits_by_field,
                        swapped_by_field,
                        strict=True,
                    )
                ):
                    stats = field_stats[primitive.key or primitive.kind]
                    for row_index, row in enumerate(rows):
                        if not bool(row["lane_mask"][field_index]):
                            continue
                        target = row["targets"][field_index].to(self.config.device)
                        probability = torch.softmax(logits[row_index], dim=-1)
                        target_index = int(torch.argmax(target).item())
                        prediction = int(torch.argmax(probability).item())
                        swapped_prediction = int(torch.argmax(swapped_logits[row_index]).item())
                        stats["brier"].append(float(torch.mean((probability - target) ** 2).cpu()))
                        stats["targets"].append(target_index)
                        stats["predictions"].append(prediction)
                        stats["swap_matches"].append(prediction == swapped_prediction)

        per_field: dict[str, Any] = {}
        all_brier: list[float] = []
        all_correct: list[bool] = []
        all_swap_matches: list[bool] = []
        field_precisions: list[float] = []
        field_recalls: list[float] = []
        field_f1s: list[float] = []
        for primitive in self.primitives:
            field = primitive.key or primitive.kind
            stats = field_stats[field]
            targets: list[int] = stats["targets"]
            predictions: list[int] = stats["predictions"]
            size = len(primitive.candidates)
            confusion = [[0 for _ in range(size)] for _ in range(size)]
            for target, prediction in zip(targets, predictions, strict=True):
                confusion[target][prediction] += 1
            precisions: list[float] = []
            recalls: list[float] = []
            f1s: list[float] = []
            for candidate_index in range(size):
                true_positive = confusion[candidate_index][candidate_index]
                false_positive = sum(
                    confusion[row][candidate_index] for row in range(size) if row != candidate_index
                )
                false_negative = sum(
                    confusion[candidate_index][column]
                    for column in range(size)
                    if column != candidate_index
                )
                precision = true_positive / max(1, true_positive + false_positive)
                recall = true_positive / max(1, true_positive + false_negative)
                f1 = (
                    0.0
                    if precision + recall == 0.0
                    else 2.0 * precision * recall / (precision + recall)
                )
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
            brier = sum(stats["brier"]) / max(1, len(stats["brier"]))
            accuracy = sum(
                target == prediction
                for target, prediction in zip(targets, predictions, strict=True)
            ) / max(1, len(targets))
            swap_consistency = sum(stats["swap_matches"]) / max(1, len(stats["swap_matches"]))
            macro_precision = sum(precisions) / size
            macro_recall = sum(recalls) / size
            macro_f1 = sum(f1s) / size
            per_field[field] = {
                "samples": len(targets),
                "brier_score": brier,
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "option_swap_consistency": swap_consistency,
                "confusion_matrix": confusion,
            }
            all_brier.extend(stats["brier"])
            all_correct.extend(
                target == prediction
                for target, prediction in zip(targets, predictions, strict=True)
            )
            all_swap_matches.extend(stats["swap_matches"])
            field_precisions.append(macro_precision)
            field_recalls.append(macro_recall)
            field_f1s.append(macro_f1)

        field_count = max(1, len(self.primitives))
        return {
            "brier_score": sum(all_brier) / max(1, len(all_brier)),
            "accuracy": sum(all_correct) / max(1, len(all_correct)),
            "macro_precision": sum(field_precisions) / field_count,
            "macro_recall": sum(field_recalls) / field_count,
            "macro_f1": sum(field_f1s) / field_count,
            "option_swap_consistency": sum(all_swap_matches) / max(1, len(all_swap_matches)),
            "per_field": per_field,
        }

    def _scheduler(self, total_steps: int) -> Any:
        if self.config.scheduler == "constant":
            return None

        def decay(step: int) -> float:
            progress = min(1.0, step / max(1, total_steps))
            if self.config.scheduler == "cosine":
                annealed = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                annealed = 1.0 - progress
            minimum = self.config.minimum_learning_rate_ratio
            return minimum + (1.0 - minimum) * annealed

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=decay)

    def train(
        self,
        train_dataset: JeevaDataset,
        validation_dataset: JeevaDataset | None,
    ) -> dict[str, Any]:
        if len(train_dataset) == 0:
            raise ValueError("Encoder training requires at least one example.")
        steps_per_epoch = math.ceil(len(train_dataset) / self.config.batch_size)
        scheduler = self._scheduler(steps_per_epoch * self.config.epochs)
        history: list[dict[str, Any]] = []
        best_metric = float("inf")
        stale_epochs = 0
        saved = False
        validation_samples = len(validation_dataset) if validation_dataset is not None else 0
        print(
            f"\nTraining started | samples={len(train_dataset)} | "
            f"validation={validation_samples} | epochs={self.config.epochs}",
            flush=True,
        )

        for epoch in range(1, self.config.epochs + 1):
            started = time.perf_counter()
            if self.actual_adaptation == "head" or self.config.device == "mps":
                # Frozen encoders must be deterministic. MPS SDPA also does not
                # support attention dropout; eval mode keeps gradients enabled.
                self.runtime.model.eval()
            else:
                self.runtime.model.train()
            self.runtime.head.train()
            indices = list(range(len(train_dataset)))
            random.Random(self.config.seed + epoch - 1).shuffle(indices)
            total_loss = 0.0
            steps = 0
            progress_interval = (
                max(
                    1,
                    math.ceil(
                        steps_per_epoch
                        / min(self.config.progress_updates_per_epoch, steps_per_epoch)
                    ),
                )
                if self.config.progress_updates_per_epoch
                else None
            )
            print(
                f"Epoch {epoch:02d}/{self.config.epochs:02d} started | samples={len(indices)}",
                flush=True,
            )
            for batch_start in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[batch_start : batch_start + self.config.batch_size]
                rows = self._rows(train_dataset, batch_indices)
                batch_random = random.Random(self.config.seed + epoch * len(indices) + batch_start)
                field_order = list(range(len(self.primitives)))
                batch_random.shuffle(field_order)
                batch_primitives = [self.primitives[index] for index in field_order]
                option_orders: list[list[int]] = []
                for primitive in batch_primitives:
                    candidate_order = list(range(len(primitive.candidates)))
                    batch_random.shuffle(candidate_order)
                    option_orders.append(candidate_order)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._batch_loss(rows, batch_primitives, option_orders)
                step_context = (
                    f"epoch {epoch}, batch {steps + 1}/{steps_per_epoch}, "
                    f"architecture encoder, adaptation {self.actual_adaptation}, "
                    f"device {self.config.device}, dtype {self.config.dtype}"
                )
                backward_finite_loss(loss, context=step_context)
                clip_finite_gradients(
                    self._optimized_parameters,
                    max_norm=self.config.maximum_gradient_norm,
                    context=step_context,
                )
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += float(loss.detach().cpu())
                steps += 1
                if progress_interval is not None and (
                    steps % progress_interval == 0 or steps == steps_per_epoch
                ):
                    completed = min(batch_start + len(batch_indices), len(indices))
                    elapsed = time.perf_counter() - started
                    eta = elapsed * (steps_per_epoch - steps) / max(1, steps)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  Progress {completed}/{len(indices)} "
                        f"({completed / len(indices) * 100:.1f}%) | "
                        f"loss={total_loss / steps:.6f} | "
                        f"lr={current_lr:.2e} | "
                        f"elapsed={elapsed:.1f}s | eta={eta:.1f}s",
                        flush=True,
                    )

            epoch_result: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": total_loss / max(1, steps),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "duration_seconds": time.perf_counter() - started,
            }
            print(
                f"Epoch {epoch:02d}/{self.config.epochs:02d} complete | "
                f"loss={epoch_result['train_loss']:.6f} | "
                f"lr={epoch_result['lr']:.2e} | "
                f"duration={epoch_result['duration_seconds']:.1f}s",
                flush=True,
            )
            metric = epoch_result["train_loss"]
            validation_metrics: dict[str, Any] | None = None
            if validation_dataset is not None and len(validation_dataset):
                print(
                    f"  Validation started | samples={len(validation_dataset)}",
                    flush=True,
                )
                validation_metrics = self._validation_metrics(validation_dataset)
                metric = float(validation_metrics["brier_score"])
                epoch_result.update(
                    {
                        f"eval_{name}": value
                        for name, value in validation_metrics.items()
                        if name != "per_field"
                    }
                )
                epoch_result["eval_fields"] = validation_metrics["per_field"]
                print(
                    "  Validation complete | "
                    f"Brier: {metric:.4f} | "
                    f"accuracy: {validation_metrics['accuracy']:.4f} | "
                    f"swap consistency: "
                    f"{validation_metrics['option_swap_consistency']:.4f}"
                )
            history.append(epoch_result)
            require_finite_value(
                float(metric),
                name="checkpoint metric",
                context=f"epoch {epoch}, architecture encoder",
            )
            improved = metric < best_metric
            if improved:
                best_metric = metric
                stale_epochs = 0
                if self.config.save_best:
                    self.save_checkpoint(self.config.output_dir)
                    saved = True
                    print(
                        f"  Checkpoint updated | path={self.config.output_dir} | "
                        f"metric={best_metric:.6f}"
                    )
            else:
                stale_epochs += 1
            if (
                validation_metrics is not None
                and self.config.early_stopping_patience is not None
                and stale_epochs >= self.config.early_stopping_patience
            ):
                print(
                    f"  Early stopping | no improvement for {stale_epochs} epoch(s)",
                    flush=True,
                )
                break
        if not saved:
            self.save_checkpoint(self.config.output_dir)
            print(f"Checkpoint saved | path={terminal_text(self.config.output_dir)}")
        return {
            "history": history,
            "output_dir": self.config.output_dir,
            "architecture": "encoder",
            "scorer": "candidate_marker",
            "adaptation": self.actual_adaptation,
            "backbone_learning_rate": self.backbone_learning_rate,
            "head_learning_rate": self.head_learning_rate,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self._optimized_parameters
            ),
            "epochs_completed": len(history),
        }

    def save_checkpoint(self, destination: str | Path) -> None:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        self.runtime.model.save_pretrained(path, safe_serialization=True)
        self.runtime.tokenizer.save_pretrained(path)
        save_candidate_marker_head(self.runtime.head, path)
