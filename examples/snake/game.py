"""
Snake game engine core logic, state representation, and board simulation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


# Coordinate delta mapping
DIR_VECTORS: dict[Direction, tuple[int, int]] = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}

OPPOSITE_DIR: dict[Direction, Direction] = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


@dataclass(frozen=True)
class Point:
    x: int
    y: int


class SnakeGame:
    """
    Grid-based Snake game environment suitable for both interactive play
    and synthetic dataset generation.
    """

    def __init__(self, width: int = 10, height: int = 10, seed: int | None = None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        """Resets the game state while advancing the session's random sequence."""
        previous_food = getattr(self, "food", None)
        mid_x, mid_y = self.width // 2, self.height // 2
        # Head at mid, body trailing to left
        self.snake: list[Point] = [
            Point(mid_x, mid_y),
            Point(mid_x - 1, mid_y),
            Point(mid_x - 2, mid_y),
        ]
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.is_game_over = False
        self.food = self._spawn_food(exclude=previous_food)

    @property
    def head(self) -> Point:
        return self.snake[0]

    @property
    def body_set(self) -> set[Point]:
        return set(self.snake)

    def _spawn_food(self, exclude: Point | None = None) -> Point:
        occupied = self.body_set
        available = [
            Point(x, y)
            for y in range(self.height)
            for x in range(self.width)
            if Point(x, y) not in occupied
        ]
        if not available:
            return self.head  # Win state (board full)
        if exclude is not None and len(available) > 1:
            available = [point for point in available if point != exclude]
        return self.rng.choice(available)

    def is_safe_move(self, direction: Direction | str) -> bool:
        """Checks whether moving in the specified direction is collision-free."""
        if isinstance(direction, str):
            direction = Direction(direction)

        # 180-degree reversal is not permitted and collides with neck if length > 1
        if len(self.snake) > 1 and direction == OPPOSITE_DIR.get(self.direction):
            return False

        dx, dy = DIR_VECTORS[direction]
        next_pos = Point(self.head.x + dx, self.head.y + dy)

        # Check boundary collision
        if not (0 <= next_pos.x < self.width and 0 <= next_pos.y < self.height):
            return False

        # Check body collision (tail will move away unless eating food)
        snake_body_to_check = self.snake[:-1] if next_pos != self.food else self.snake
        if next_pos in snake_body_to_check:
            return False

        return True

    def step(self, direction: Direction | str) -> tuple[bool, bool]:
        """
        Executes one step in the given direction.
        Returns: (is_game_over, ate_food)
        """
        if self.is_game_over:
            return True, False

        if isinstance(direction, str):
            direction = Direction(direction)

        # Cannot directly reverse into own neck
        if len(self.snake) > 1 and direction == OPPOSITE_DIR.get(self.direction):
            direction = self.direction

        self.direction = direction
        dx, dy = DIR_VECTORS[direction]
        new_head = Point(self.head.x + dx, self.head.y + dy)

        # Check boundary collision
        if not (0 <= new_head.x < self.width and 0 <= new_head.y < self.height):
            self.is_game_over = True
            return True, False

        # Check body collision
        if new_head in self.snake[:-1]:
            self.is_game_over = True
            return True, False

        self.snake.insert(0, new_head)
        self.steps += 1

        ate_food = new_head == self.food
        if ate_food:
            self.score += 1
            if len(self.snake) < self.width * self.height:
                self.food = self._spawn_food()
        else:
            self.snake.pop()

        return False, ate_food

    def to_state_prompt(self) -> str:
        """
        Serializes current game observation into a structured affordance representation
        with explicit wall clearances and move-by-move survival/distance analysis.
        """
        head = self.head
        food = self.food
        dx = food.x - head.x
        dy = food.y - head.y

        # Determine relative food direction
        horiz = "EAST (RIGHT)" if dx > 0 else ("WEST (LEFT)" if dx < 0 else "ALIGNED")
        vert = "SOUTH (DOWN)" if dy > 0 else ("NORTH (UP)" if dy < 0 else "ALIGNED")

        dist_up = head.y
        dist_down = (self.height - 1) - head.y
        dist_left = head.x
        dist_right = (self.width - 1) - head.x

        curr_dist = abs(dx) + abs(dy)

        move_lines = []
        safe_moves = []
        best_move = None
        best_dist = float("inf")

        for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            mdx, mdy = DIR_VECTORS[d]
            next_pos = Point(head.x + mdx, head.y + mdy)

            if len(self.snake) > 1 and d == OPPOSITE_DIR.get(self.direction):
                status = "BLOCKED (Neck reversal - CRASH)"
            elif not (0 <= next_pos.x < self.width and 0 <= next_pos.y < self.height):
                status = "BLOCKED (Wall collision - CRASH)"
            elif next_pos in (self.snake[:-1] if next_pos != food else self.snake):
                status = "BLOCKED (Body collision - CRASH)"
            else:
                new_dist = abs(food.x - next_pos.x) + abs(food.y - next_pos.y)
                safe_moves.append(d.value)
                if new_dist < curr_dist:
                    status = f"SAFE (Distance to food: {new_dist} - GETS CLOSER)"
                else:
                    status = f"SAFE (Distance to food: {new_dist})"
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_move = d.value

            move_lines.append(f"- {d.value}: {status}")

        analysis = "\n".join(move_lines)
        safe_str = ", ".join(safe_moves) if safe_moves else "NONE"
        best_str = best_move if best_move else (safe_moves[0] if safe_moves else "NONE")

        return (
            f"Game: Snake Arena ({self.width}x{self.height} Grid)\n"
            f"Snake Head: ({head.x}, {head.y}) | Score: {self.score} | Length: {len(self.snake)}\n"
            f"Food Location: ({food.x}, {food.y}) | Target relative: {horiz}, {vert}\n"
            f"Wall Distances: UP: {dist_up} tiles | DOWN: {dist_down} tiles | LEFT: {dist_left} tiles | RIGHT: {dist_right} tiles\n\n"
            f"Move Evaluation (Survival & Food Distance):\n"
            f"{analysis}\n\n"
            f"Valid Survival Choices: [{safe_str}]\n"
            f"Best Safe Navigation Move: {best_str}\n"
        )
