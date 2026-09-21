"""
Interactive and autonomous terminal Snake game powered by Jeeva DecisionModel.
Supports selectable base models and fine-tuned Jeeva artifacts, single-step
inspection, and live decision telemetry.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Any

from examples.snake.game import Direction, Point, SnakeGame
from jeeva import Choice, DecisionModel, DecisionSchema

# List of presets that are quick to select in CLI
MODEL_PRESETS: dict[str, str] = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama": "unsloth/Llama-3.2-1B-Instruct",
    "llama1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama-base": "unsloth/Llama-3.2-1B",
    "llama-instruct": "unsloth/Llama-3.2-1B-Instruct",
    "ft-0.5b": "examples/snake/checkpoints/snake_adapter",
}


class DirectionDecision(DecisionSchema):
    direction = Choice(
        "Choose the next movement direction.",
        {
            "UP": "Move one cell upward.",
            "DOWN": "Move one cell downward.",
            "LEFT": "Move one cell to the left.",
            "RIGHT": "Move one cell to the right.",
        },
    )


# ANSI 256-color palette for fancy demo rendering
C_RST = "\033[0m"
C_CYAN = "\033[1;38;5;45m"
C_GRN = "\033[1;38;5;82m"
C_YEL = "\033[1;38;5;220m"
C_RED = "\033[1;38;5;196m"
C_MAG = "\033[1;38;5;213m"
C_BLU = "\033[1;38;5;75m"
C_DIM = "\033[38;5;244m"
C_WHT = "\033[1;38;5;255m"
C_CLEAR_LINE = "\033[2K"

HEAD_CHARS = {
    Direction.UP: "▲▲",
    Direction.DOWN: "▼▼",
    Direction.LEFT: "◄◄",
    Direction.RIGHT: "►►",
}

ANSI_REGEX = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_REGEX.sub("", text)


def pad_colored(text: str, width: int) -> str:
    vis_len = len(strip_ansi(text))
    return text + " " * max(0, width - vis_len)


def make_bar(p: float, width: int = 12, color: str = C_GRN) -> str:
    filled = int(round(max(0.0, min(1.0, p)) * width))
    return color + "█" * filled + C_DIM + "░" * (width - filled) + C_RST


def render_board(
    game: SnakeGame,
    chosen_direction: Direction | None,
    is_safe_move: bool,
    jeeva_meta: Any | None,
    fps: float,
    step_num: int,
    model_name: str,
) -> str:
    """Renders ANSI game board side-by-side with live neural decision telemetry."""
    # -------------------------------------------------------------
    # Left Panel: Game Arena (Unicode Box Grid)
    # -------------------------------------------------------------
    head_dir = chosen_direction or game.direction
    head_icon = HEAD_CHARS.get(head_dir, "██")

    board_lines = []
    board_lines.append(f"{C_DIM}╭" + "──" * game.width + f"╮{C_RST}")
    for y in range(game.height):
        row_chars = []
        for x in range(game.width):
            pt = Point(x, y)
            if pt == game.head:
                row_chars.append(f"{C_WHT}{head_icon}{C_RST}")
            elif pt == game.food:
                row_chars.append(f"{C_RED}★ {C_RST}")
            elif len(game.snake) > 1 and pt == game.snake[1]:
                row_chars.append(f"{C_GRN}▓▓{C_RST}")  # Neck segment
            elif pt == game.snake[-1] and len(game.snake) > 2:
                row_chars.append(f"{C_DIM}▒▒{C_RST}")  # Tail segment
            elif pt in game.body_set:
                row_chars.append(f"{C_GRN}██{C_RST}")
            else:
                row_chars.append("  ")
        board_lines.append(f"{C_DIM}│{C_RST}" + "".join(row_chars) + f"{C_DIM}│{C_RST}")
    board_lines.append(f"{C_DIM}╰" + "──" * game.width + f"╯{C_RST}")

    # -------------------------------------------------------------
    # Right Panel: Jeeva Neural Telemetry HUD
    # -------------------------------------------------------------
    telemetry = getattr(jeeva_meta, "telemetry", {}) if jeeva_meta else {}
    prefill_ms = telemetry.get("prefill_ms", 0.0)
    dispatch_ms = telemetry.get("dispatch_ms", 0.0)
    total_ms = telemetry.get("total_ms", 0.0)
    device_str = telemetry.get("device", "mps").upper()

    dir_decision = jeeva_meta.fields.get("direction") if jeeva_meta else None
    dist = dir_decision.distribution if dir_decision else {}
    margin = dir_decision.margin if dir_decision else 0.0
    is_certain = jeeva_meta.is_certain if jeeva_meta else False
    calibration = getattr(jeeva_meta, "calibration", "unvalidated")

    chosen_val = chosen_direction.value if chosen_direction else "NONE"
    latency_line = (
        f"{C_DIM}Latency:{C_RST} Prefill: {C_CYAN}{prefill_ms:5.1f} ms{C_RST} │ "
        f"Dispatch: {C_CYAN}{dispatch_ms:5.1f} ms{C_RST} │ "
        f"Total: {C_GRN}{total_ms:5.1f} ms{C_RST}"
    )

    hud_lines = [
        f"{C_CYAN}⚡ JEEVA ZERO-TOKEN NEURAL HUD ⚡{C_RST}",
        f"{C_DIM}Model:{C_RST} {C_WHT}{model_name}{C_RST}  [{C_MAG}{device_str}{C_RST}]",
        f"{C_DIM}Score:{C_RST} {C_YEL}★ {game.score:<2}{C_RST} │ {C_DIM}Len:{C_RST} {C_GRN}{len(game.snake):<2}{C_RST} │ {C_DIM}Step:{C_RST} {C_WHT}{step_num:<3}{C_RST} │ {C_CYAN}{fps:4.1f} FPS{C_RST}",
        f"{C_DIM}───────────────────────────────────────────────{C_RST}",
        latency_line,
        f"{C_DIM}───────────────────────────────────────────────{C_RST}",
        f"{C_WHT}Action Distribution (Zero-Token Parallel PCD):{C_RST}",
    ]

    for d_name, icon in [("UP", "▲"), ("DOWN", "▼"), ("LEFT", "◄"), ("RIGHT", "►")]:
        p = dist.get(d_name, 0.0)
        bar = make_bar(p, width=12, color=C_GRN if d_name == chosen_val else C_BLU)
        tag = f" {C_GRN}◄ CHOSEN (WINNER){C_RST}" if d_name == chosen_val else ""
        hud_lines.append(f"  {icon} {d_name:<5} [{bar}] {p * 100:5.1f}%{tag}")

    if is_certain:
        cert_badge = f"{C_GRN}● HIGH (Certain){C_RST}"
    else:
        cert_badge = f"{C_YEL}▲ AMBIGUOUS (Close Contest){C_RST}"
    if calibration != "validated":
        cert_badge += f" {C_DIM}[UNVALIDATED THRESHOLD]{C_RST}"
    safe_badge = (
        f"{C_GRN}● SAFE (exact collision check){C_RST}"
        if is_safe_move
        else f"{C_RED}▲ COLLISION WARNING{C_RST}"
    )

    hud_lines.extend(
        [
            f"{C_DIM}───────────────────────────────────────────────{C_RST}",
            f"{C_DIM}Certainty:{C_RST}    {cert_badge} │ {C_DIM}Margin:{C_RST} {C_CYAN}{margin:+5.3f}{C_RST}",
            f"{C_DIM}Hazard Guard:{C_RST} {safe_badge}",
        ]
    )

    # -------------------------------------------------------------
    # Side-by-Side Merge
    # -------------------------------------------------------------
    merged_lines = []
    board_width = game.width * 2 + 2
    max_rows = max(len(board_lines), len(hud_lines))
    for r in range(max_rows):
        left = board_lines[r] if r < len(board_lines) else " " * board_width
        right = hud_lines[r] if r < len(hud_lines) else ""
        merged_lines.append(f"{pad_colored(left, board_width + 2)}  {right}")

    merged_lines.append(
        f"{C_DIM}───────────────────────────────────────────────────────────────────────{C_RST}"
    )
    merged_lines.append(
        f"{C_DIM}Controls: Ctrl+C Stop  │  Enter/R Restart after collision  │  Q Quit{C_RST}"
    )

    return "\n".join(f"{C_CLEAR_LINE}{line}" for line in merged_lines)


def select_safe_direction(game: SnakeGame, decision: Any) -> Direction:
    """Keep the model's ranking while excluding moves known to collide."""
    model_direction = Direction(decision.value["direction"])
    safe_directions = [direction for direction in Direction if game.is_safe_move(direction)]
    if not safe_directions:
        return model_direction

    field = decision.fields.get("direction")
    distribution = field.distribution if field is not None else {}
    return max(
        safe_directions,
        key=lambda direction: (
            distribution.get(direction.value, 0.0),
            direction == model_direction,
        ),
    )


def play(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str | None = None,
    width: int = 10,
    height: int = 10,
    max_steps: int = 200,
    delay: float = 0.15,
    headless: bool = False,
    interactive_step: bool = False,
    seed: int | None = None,
) -> None:
    # Resolve preset if alias given
    target_model = MODEL_PRESETS.get(model.lower(), model)
    print(
        f"\n{C_CYAN}⚡ Loading model:{C_RST} {C_WHT}{target_model}{C_RST} (device: {device or 'auto'})..."
    )
    decision_model = DecisionModel(target_model, device=device or "auto")

    game = SnakeGame(width=width, height=height, seed=seed)

    if not headless:
        # Hide terminal cursor and clear screen once.
        sys.stdout.write("\033[2J\033[?25l\033[H")
        sys.stdout.flush()

    try:
        while True:
            step_num = 0
            total_latency_ms: list[float] = []
            measured_fps = 0.0
            chosen_direction: Direction | None = None
            is_safe_move = True
            jeeva_meta: Any | None = None

            while not game.is_game_over and step_num < max_steps:
                frame_started = time.perf_counter()
                state_text = game.to_state_prompt()

                # Only direction requires model inference. Collision safety is exact game logic.
                jeeva_meta = decision_model.decide(state_text, DirectionDecision)
                telemetry = jeeva_meta.telemetry
                total_latency_ms.append(telemetry.get("total_ms", 0.0))

                chosen_direction = select_safe_direction(game, jeeva_meta)
                is_safe_move = game.is_safe_move(chosen_direction)
                game.step(chosen_direction)
                step_num += 1

                if not headless:
                    sys.stdout.write("\033[H")
                    board_display = render_board(
                        game=game,
                        chosen_direction=chosen_direction,
                        is_safe_move=is_safe_move,
                        jeeva_meta=jeeva_meta,
                        fps=measured_fps,
                        step_num=step_num,
                        model_name=target_model.split("/")[-1],
                    )
                    sys.stdout.write(board_display + "\n")
                    sys.stdout.flush()

                    if interactive_step:
                        input(f"{C_YEL}[Press Enter for next step... (or Ctrl+C to stop)]{C_RST}")
                    else:
                        elapsed = time.perf_counter() - frame_started
                        time.sleep(max(0.0, delay - elapsed))

                frame_elapsed = time.perf_counter() - frame_started
                measured_fps = 0.0 if interactive_step else 1.0 / max(frame_elapsed, 0.001)

            if not headless:
                sys.stdout.write("\033[H")
                print(
                    render_board(
                        game=game,
                        chosen_direction=chosen_direction,
                        is_safe_move=is_safe_move,
                        jeeva_meta=jeeva_meta,
                        fps=measured_fps,
                        step_num=step_num,
                        model_name=target_model.split("/")[-1],
                    )
                )

            avg_lat = (sum(total_latency_ms) / len(total_latency_ms)) if total_latency_ms else 0.0
            p50_lat = (
                sorted(total_latency_ms)[len(total_latency_ms) // 2] if total_latency_ms else 0.0
            )
            outcome = "💀 Game Over (Collision)" if game.is_game_over else "🏁 Completed Max Steps"

            print(
                f"\n{C_DIM}╭─────────────────────────────────────────────────────────────╮{C_RST}"
            )
            print(
                f"{C_DIM}│{C_RST}                     {C_CYAN}🎮 GAME FINISHED!{C_RST}                       {C_DIM}│{C_RST}"
            )
            print(f"{C_DIM}│{C_RST}  Outcome:     {C_WHT}{outcome:<46}{C_RST}{C_DIM}│{C_RST}")
            print(f"{C_DIM}│{C_RST}  Final Score: {C_YEL}★ {game.score:<44}{C_RST}{C_DIM}│{C_RST}")
            print(
                f"{C_DIM}│{C_RST}  Snake Size:  {C_GRN}{len(game.snake)} segments{C_RST}{' ' * max(0, 35 - len(str(len(game.snake))))}{C_DIM}│{C_RST}"
            )
            print(
                f"{C_DIM}│{C_RST}  Total Steps: {C_WHT}{step_num} moves{C_RST}{' ' * max(0, 38 - len(str(step_num)))}{C_DIM}│{C_RST}"
            )
            lat_summary = f"{avg_lat:.2f} ms  │  P50: {p50_lat:.2f} ms"
            print(f"{C_DIM}│{C_RST}  Avg Latency: {C_CYAN}{lat_summary:<44}{C_RST}{C_DIM}│{C_RST}")
            print(
                f"{C_DIM}╰─────────────────────────────────────────────────────────────╯{C_RST}\n"
            )

            if headless or not game.is_game_over:
                break

            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            try:
                restart_choice = input(f"{C_YEL}Press Enter or R to restart; Q to quit:{C_RST} ")
            except EOFError:
                break
            if restart_choice.strip().lower() not in {"", "r", "restart"}:
                break

            game.reset()
            sys.stdout.write("\033[2J\033[?25l\033[H")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n{C_YEL}Game interrupted by user.{C_RST}")
    finally:
        # Restore terminal cursor
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run autonomous Snake with Jeeva DecisionModel across different models."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Full Hugging Face model identifier (e.g. 'unsloth/Llama-3.2-1B', 'Qwen/Qwen2.5-1.5B-Instruct') or path to checkpoint directory",
    )
    parser.add_argument(
        "--step",
        "-s",
        action="store_true",
        help="Single-step mode: press Enter to advance one step and inspect decisions",
    )
    parser.add_argument("--steps", type=int, default=150, help="Maximum gameplay steps")
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.12,
        help="Minimum frame period in seconds; inference time counts toward this period.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible game setups; random by default.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target execution device ('mps', 'cpu', 'cuda'). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without UI screen refresh (for fast testing/benchmarking)",
    )
    args = parser.parse_args()

    play(
        model=args.model,
        device=args.device,
        max_steps=args.steps,
        delay=args.delay,
        headless=args.headless,
        interactive_step=args.step,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
