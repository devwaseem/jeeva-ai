"""
Algorithmic Oracle and Synthetic Data Generator for Snake Decision Fine-Tuning.
Uses Breadth-First Search (BFS) for shortest path to food, with flood-fill space
maximization fallback to avoid dead ends.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

from examples.snake.game import DIR_VECTORS, Direction, Point, SnakeGame


def get_bfs_path(game: SnakeGame, target: Point) -> list[Direction] | None:
    """Finds shortest path from head to target avoiding walls and snake body."""
    start = game.head
    if start == target:
        return []

    visited = {start}
    # queue contains (point, path_of_directions)
    queue: deque[tuple[Point, list[Direction]]] = deque([(start, [])])

    while queue:
        current, path = queue.popleft()
        if current == target:
            return path

        for direction, (dx, dy) in DIR_VECTORS.items():
            nx, ny = current.x + dx, current.y + dy
            next_pt = Point(nx, ny)

            # Check boundaries
            if not (0 <= nx < game.width and 0 <= ny < game.height):
                continue

            # Check body collision (for head step, check safe_move)
            if len(path) == 0:
                if not game.is_safe_move(direction):
                    continue
            else:
                if next_pt in game.body_set and next_pt != game.snake[-1]:
                    continue

            if next_pt not in visited:
                visited.add(next_pt)
                queue.append((next_pt, path + [direction]))

    return None


def count_reachable_space(game: SnakeGame, start_pt: Point) -> int:
    """Flood-fills from start_pt to count accessible free tiles."""
    visited = {start_pt}
    queue = deque([start_pt])
    count = 0

    while queue:
        curr = queue.popleft()
        count += 1
        for dx, dy in DIR_VECTORS.values():
            nx, ny = curr.x + dx, curr.y + dy
            pt = Point(nx, ny)
            if 0 <= nx < game.width and 0 <= ny < game.height:
                if pt not in game.body_set and pt not in visited:
                    visited.add(pt)
                    queue.append(pt)
    return count


def select_oracle_move(game: SnakeGame) -> Direction:
    """
    Expert Snake Oracle:
    1. Try BFS shortest path toward food. Verify that following the path doesn't trap snake.
    2. If no path or path is risky, pick the safe move with maximum reachable flood-fill area.
    """
    safe_moves = [d for d in Direction if game.is_safe_move(d)]
    if not safe_moves:
        # No safe move available; return current direction
        return game.direction

    bfs_path = get_bfs_path(game, game.food)
    if bfs_path and len(bfs_path) > 0:
        first_move = bfs_path[0]
        # Check if taking this move leaves reasonable open space
        dx, dy = DIR_VECTORS[first_move]
        next_head = Point(game.head.x + dx, game.head.y + dy)
        space = count_reachable_space(game, next_head)
        if space >= len(game.snake) // 2:
            return first_move

    # Fallback: pick safe direction maximizing reachable space
    best_move = safe_moves[0]
    best_space = -1
    for m in safe_moves:
        dx, dy = DIR_VECTORS[m]
        next_head = Point(game.head.x + dx, game.head.y + dy)
        space = count_reachable_space(game, next_head)
        if space > best_space:
            best_space = space
            best_move = m

    return best_move


def generate_synthetic_dataset(
    num_samples: int = 350,
    width: int = 10,
    height: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Generates synthetic snake gameplay decision records with balanced states."""
    rng = random.Random(seed)
    records: list[dict] = []
    game_count = 0

    while len(records) < num_samples:
        game_count += 1
        game = SnakeGame(width=width, height=height, seed=rng.randint(1, 100000))

        # Play game with oracle until death or maximum steps reached
        max_game_steps = 150
        for _ in range(max_game_steps):
            if game.is_game_over:
                break

            state_text = game.to_state_prompt()
            oracle_dir = select_oracle_move(game)

            record = {
                "text": state_text,
                "direction": oracle_dir.value,
            }
            records.append(record)

            if len(records) >= num_samples:
                break

            # Occasionally introduce a slight random safe wander to diversify board states
            if rng.random() < 0.15:
                safe_dirs = [d for d in Direction if game.is_safe_move(d)]
                move_to_take = rng.choice(safe_dirs) if safe_dirs else oracle_dir
            else:
                move_to_take = oracle_dir

            is_over, _ = game.step(move_to_take)
            if is_over:
                break

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Snake dataset for Jeeva fine-tuning."
    )
    parser.add_argument("--train-samples", type=int, default=300, help="Number of training samples")
    parser.add_argument("--eval-samples", type=int, default=60, help="Number of evaluation samples")
    parser.add_argument(
        "--output-dir", type=str, default="examples/snake/data", help="Output directory"
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {args.train_samples} training samples and {args.eval_samples} evaluation samples..."
    )
    train_records = generate_synthetic_dataset(num_samples=args.train_samples, seed=42)
    eval_records = generate_synthetic_dataset(num_samples=args.eval_samples, seed=1337)

    train_file = out_path / "snake_train.jsonl"
    eval_file = out_path / "snake_eval.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(eval_file, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r) + "\n")

    print(f"Successfully wrote {len(train_records)} records to {train_file}")
    print(f"Successfully wrote {len(eval_records)} records to {eval_file}")


if __name__ == "__main__":
    main()
