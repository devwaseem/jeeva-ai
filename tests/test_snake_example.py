"""
Tests for the Snake Game example module.
"""

from types import SimpleNamespace

import pytest

from examples.snake import compare_models, play_snake
from examples.snake.game import Direction, Point, SnakeGame
from examples.snake.generate_data import generate_synthetic_dataset, select_oracle_move
from jeeva import Decision


def test_snake_game_basic_movement():
    game = SnakeGame(width=10, height=10, seed=42)
    initial_head = game.head

    # Default is RIGHT
    over, ate = game.step(Direction.RIGHT)
    assert not over
    assert game.head == Point(initial_head.x + 1, initial_head.y)
    assert len(game.snake) == 3


def test_snake_wall_collision():
    game = SnakeGame(width=5, height=5, seed=42)
    # Move RIGHT until crashing into wall
    for _ in range(10):
        over, _ = game.step(Direction.RIGHT)
        if over:
            break
    assert game.is_game_over


def test_synthetic_data_generation():
    records = generate_synthetic_dataset(num_samples=10, width=8, height=8, seed=42)
    assert len(records) == 10
    for record in records:
        assert set(record) == {"text", "direction"}
        assert record["direction"] in ["UP", "DOWN", "LEFT", "RIGHT"]
        assert "Game: Snake" in record["text"]


def test_snake_comparison_uses_identical_cases_and_reports_local_metrics(tmp_path, monkeypatch):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        '{"text":"north","direction":"UP"}\n{"text":"west","direction":"LEFT"}\n',
        encoding="utf-8",
    )
    cases, digest = compare_models.load_cases(dataset)

    class FakeModel:
        @property
        def architecture(self) -> str:
            return "encoder"

        def decide(self, context, schema):
            assert schema is play_snake.DirectionDecision
            value = "UP" if context == "north" else "RIGHT"
            return Decision(
                value={"direction": value},
                fields={},
                is_certain=True,
                calibration="unvalidated",
                telemetry={},
            )

    clock = iter([1.0, 1.01, 2.0, 2.03])
    monkeypatch.setattr(compare_models.time, "perf_counter", lambda: next(clock))

    result = compare_models.evaluate_model(
        FakeModel(),
        cases,
        label="test encoder",
        source="artifact",
    )

    assert len(digest) == 64
    assert result.correct == 1
    assert result.score == pytest.approx(0.5)
    assert result.throughput_per_second == pytest.approx(50.0)
    assert result.latency_p50_ms == pytest.approx(20.0)
    assert result.latency_p95_ms == pytest.approx(29.0)


def test_oracle_avoids_immediate_danger():
    game = SnakeGame(width=6, height=6, seed=42)
    move = select_oracle_move(game)
    assert game.is_safe_move(move)


def test_snake_move_selection_excludes_known_collisions():
    game = SnakeGame(width=6, height=6, seed=42)
    decision = SimpleNamespace(
        value={"direction": Direction.LEFT.value},
        fields={
            "direction": SimpleNamespace(
                distribution={
                    Direction.UP.value: 0.1,
                    Direction.DOWN.value: 0.7,
                    Direction.LEFT.value: 0.15,
                    Direction.RIGHT.value: 0.05,
                }
            )
        },
    )

    selected = play_snake.select_safe_direction(game, decision)

    assert selected is Direction.DOWN
    assert game.is_safe_move(selected)


def test_snake_reset_advances_a_reproducible_food_sequence():
    first = SnakeGame(width=10, height=10, seed=42)
    second = SnakeGame(width=10, height=10, seed=42)

    first_foods = [first.food]
    second_foods = [second.food]
    for _ in range(3):
        first.reset()
        second.reset()
        first_foods.append(first.food)
        second_foods.append(second.food)

    assert first_foods == second_foods
    assert all(previous != current for previous, current in zip(first_foods, first_foods[1:]))


def test_snake_play_uses_one_lane_and_full_navigation_context(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.contexts = []
            self.schema = None

        def decide(self, context, schema):
            self.contexts.append(context)
            self.schema = schema
            direction = SimpleNamespace(
                value=Direction.RIGHT,
                distribution={"RIGHT": 1.0},
                margin=1.0,
            )
            return SimpleNamespace(
                value={"direction": Direction.RIGHT.value},
                fields={"direction": direction},
                telemetry={"total_ms": 1.0, "device": "cpu"},
                is_certain=True,
            )

    model = FakeModel()
    monkeypatch.setattr(play_snake, "DecisionModel", lambda *args, **kwargs: model)

    play_snake.play(headless=True, seed=42, max_steps=3)

    assert model.schema is play_snake.DirectionDecision
    assert list(model.schema.fields()) == ["direction"]
    assert len(model.contexts) == 3
    for context in model.contexts:
        assert context.startswith("Game: Snake Arena")
        assert "Target relative:" in context
        assert "Wall Distances:" in context
        assert "Move Evaluation (Survival & Food Distance):" in context
        assert "Valid Survival Choices:" in context


def test_snake_play_can_restart_without_reloading_model(monkeypatch):
    session_foods = []

    class TrackingGame(SnakeGame):
        def reset(self):
            super().reset()
            session_foods.append(self.food)

        def is_safe_move(self, direction):
            del direction
            return True

    class FakeModel:
        def decide(self, context, schema):
            direction = SimpleNamespace(
                value=Direction.RIGHT,
                distribution={"RIGHT": 1.0},
                margin=1.0,
            )
            return SimpleNamespace(
                value={"direction": Direction.RIGHT.value},
                fields={"direction": direction},
                telemetry={"total_ms": 1.0, "device": "cpu"},
                is_certain=True,
            )

    model = FakeModel()
    model_loads = []
    responses = iter(["r", "q"])
    monkeypatch.setattr(play_snake, "SnakeGame", TrackingGame)
    monkeypatch.setattr(
        play_snake,
        "DecisionModel",
        lambda *args, **kwargs: model_loads.append((args, kwargs)) or model,
    )
    monkeypatch.setattr(play_snake, "render_board", lambda **kwargs: "")
    monkeypatch.setattr(play_snake.time, "sleep", lambda _: None)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    play_snake.play(
        width=4,
        height=4,
        delay=0.0,
        headless=False,
        seed=42,
    )

    assert len(model_loads) == 1
    assert len(session_foods) == 2
    assert session_foods[0] != session_foods[1]


def test_snake_delay_is_a_minimum_frame_period(monkeypatch):
    class FakeModel:
        def decide(self, context, schema):
            direction = SimpleNamespace(
                value=Direction.RIGHT,
                distribution={"RIGHT": 1.0},
                margin=1.0,
            )
            return SimpleNamespace(
                value={"direction": Direction.RIGHT.value},
                fields={"direction": direction},
                telemetry={"total_ms": 80.0, "device": "cpu"},
                is_certain=True,
            )

    clock = iter([10.0, 10.08, 10.12])
    sleeps = []
    monkeypatch.setattr(play_snake, "DecisionModel", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(play_snake, "render_board", lambda **kwargs: "")
    monkeypatch.setattr(play_snake.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(play_snake.time, "sleep", sleeps.append)
    monkeypatch.setattr("builtins.input", lambda _: "q")

    play_snake.play(width=2, height=4, max_steps=1, headless=False, delay=0.12)

    assert sleeps == [pytest.approx(0.04)]


def test_snake_hud_reports_exact_collision_check():
    game = SnakeGame(width=6, height=6, seed=42)
    direction = SimpleNamespace(
        distribution={"LEFT": 1.0},
        margin=1.0,
    )
    result = SimpleNamespace(
        fields={"direction": direction},
        telemetry={
            "prefill_ms": 12.3,
            "dispatch_ms": 45.6,
            "total_ms": 78.9,
            "device": "mps",
        },
        is_certain=True,
    )

    rendered = play_snake.render_board(
        game=game,
        chosen_direction=Direction.LEFT,
        is_safe_move=False,
        jeeva_meta=result,
        fps=5.0,
        step_num=1,
        model_name="test",
    )

    plain = play_snake.strip_ansi(rendered)
    assert "COLLISION WARNING" in plain
    assert "Prefill:  12.3 ms" in plain
    assert "Dispatch:  45.6 ms" in plain
    assert "Total:  78.9 ms" in plain
    assert plain.count("CHOSEN (WINNER)") == 1
    assert all(line.startswith(play_snake.C_CLEAR_LINE) for line in rendered.splitlines())


def test_snake_cli_uses_step_limit_and_random_seed_by_default(monkeypatch):
    captured = {}
    monkeypatch.setattr(play_snake, "play", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(play_snake.sys, "argv", ["play_snake", "--headless"])

    play_snake.main()

    assert captured["max_steps"] == 150
    assert captured["seed"] is None
