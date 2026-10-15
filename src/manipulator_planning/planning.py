"""RRT* с проверкой столкновений в ограниченном пространстве суставных углов.

Евклидово расстояние вычисляется после нормирования каждого угла на диапазон
сустава. Пределы конечны; переход через границу диапазона не допускается.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class _Robot(Protocol):
    joint_lower: NDArray[np.float64]
    joint_upper: NDArray[np.float64]


class _Checker(Protocol):
    def is_valid(self, q: ArrayLike) -> bool: ...

    def is_motion_valid(self, a: ArrayLike, b: ArrayLike, max_depth: int = 12) -> bool: ...


@dataclass(slots=True)
class PlanningResult:
    success: bool
    path: NDArray[np.float64] | None
    iterations: int
    node_count: int
    cost: float
    reason: str
    cost_history: list[tuple[int, float]] = field(default_factory=list)


@dataclass(slots=True)
class _Node:
    q: NDArray[np.float64]
    parent: int | None
    cost: float
    children: set[int] = field(default_factory=set)


class RRTStar:
    """RRT* с проверенными рёбрами, пересчётом стоимости и точным достижением цели.

    ``step_size`` и ``neighbor_radius`` используют нормированную суставную метрику.
    Последнее соединение с целью также проходит проверку столкновений.
    ``cost_history`` хранит улучшения в формате ``(iteration, best_cost)``.
    """

    def __init__(
        self,
        robot: _Robot,
        checker: _Checker,
        seed: int = 42,
        step_size: float = 0.22,
        goal_bias: float = 0.15,
        neighbor_radius: float = 1.0,
    ) -> None:
        self.lower = np.asarray(robot.joint_lower, dtype=float).copy()
        self.upper = np.asarray(robot.joint_upper, dtype=float).copy()
        self.ranges = self.upper - self.lower
        if (
            self.lower.ndim != 1
            or self.lower.size == 0
            or self.upper.shape != self.lower.shape
            or not np.all(np.isfinite(self.lower))
            or not np.all(np.isfinite(self.upper))
            or np.any(self.ranges <= 0)
        ):
            raise ValueError("The robot must have finite, increasing joint limits.")
        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be positive and finite.")
        if not np.isfinite(neighbor_radius) or neighbor_radius <= 0:
            raise ValueError("neighbor_radius must be positive and finite.")
        if not 0 <= goal_bias <= 1:
            raise ValueError("goal_bias must be between zero and one.")
        self.checker = checker
        self.seed = seed
        self.step_size = float(step_size)
        self.goal_bias = float(goal_bias)
        self.neighbor_radius = float(neighbor_radius)
        self.nodes: list[_Node] = []

    def distance(self, a: ArrayLike, b: ArrayLike) -> float:
        return float(np.linalg.norm((np.asarray(a) - np.asarray(b)) / self.ranges))

    def _configuration(self, value: ArrayLike, name: str) -> NDArray[np.float64]:
        q = np.asarray(value, dtype=float)
        if q.shape != self.lower.shape or not np.all(np.isfinite(q)):
            raise ValueError(f"{name} must be a finite vector of {self.lower.size} joints.")
        return q.copy()

    def _in_limits(self, q: NDArray[np.float64]) -> bool:
        return bool(np.all(q >= self.lower) and np.all(q <= self.upper))

    def _is_ancestor(self, possible_ancestor: int, node: int) -> bool:
        while node is not None:
            if node == possible_ancestor:
                return True
            node = self.nodes[node].parent
        return False

    def _reparent(self, node: int, parent: int) -> None:
        """Переподключить поддерево и пересчитать стоимость всех его потомков."""
        if node == 0 or self._is_ancestor(node, parent):
            raise ValueError("Reparenting would introduce a cycle or move the root.")
        previous = self.nodes[node].parent
        if previous is not None:
            self.nodes[previous].children.remove(node)
        self.nodes[node].parent = parent
        self.nodes[parent].children.add(node)
        pending = [node]
        while pending:
            current = pending.pop()
            ancestor = self.nodes[current].parent
            assert ancestor is not None
            self.nodes[current].cost = self.nodes[ancestor].cost + self.distance(
                self.nodes[ancestor].q, self.nodes[current].q
            )
            pending.extend(self.nodes[current].children)

    def _path(self, parent: int, goal: NDArray[np.float64]) -> NDArray[np.float64]:
        points = [goal.copy()]
        node: int | None = parent
        while node is not None:
            points.append(self.nodes[node].q.copy())
            node = self.nodes[node].parent
        points.reverse()
        # Выборка с направлением на цель могла уже добавить точную цель в дерево.
        if len(points) > 1 and np.array_equal(points[-2], points[-1]):
            points.pop()
        return np.asarray(points)

    def plan(
        self,
        start: ArrayLike,
        goal: ArrayLike,
        max_iterations: int = 1000,
        stop_on_first: bool = False,
    ) -> PlanningResult:
        start_q = self._configuration(start, "start")
        goal_q = self._configuration(goal, "goal")
        if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0:
            raise ValueError("max_iterations must be a non-negative integer.")
        self.nodes = []
        if not self._in_limits(start_q) or not self.checker.is_valid(start_q):
            return PlanningResult(False, None, 0, 0, np.inf, "Invalid start configuration.")
        self.nodes = [_Node(start_q, None, 0.0)]
        if not self._in_limits(goal_q) or not self.checker.is_valid(goal_q):
            return PlanningResult(False, None, 0, 1, np.inf, "Invalid goal configuration.")
        if np.array_equal(start_q, goal_q):
            return PlanningResult(
                True, start_q[None, :], 0, 1, 0.0, "Start equals goal.", [(0, 0.0)]
            )
        direct_cost = self.distance(start_q, goal_q)
        if self.checker.is_motion_valid(start_q, goal_q):
            return PlanningResult(
                True,
                np.array([start_q, goal_q]),
                0,
                1,
                direct_cost,
                "Direct collision-free connection.",
                [(0, direct_cost)],
            )

        rng = np.random.default_rng(self.seed)
        candidates: list[tuple[int, float]] = []
        best_cost = np.inf
        best_parent: int | None = None
        history: list[tuple[int, float]] = []
        dimensions = self.lower.size

        for iteration in range(1, max_iterations + 1):
            sample = (
                goal_q if rng.random() < self.goal_bias else rng.uniform(self.lower, self.upper)
            )
            positions = np.asarray([node.q for node in self.nodes])
            distances = np.linalg.norm((positions - sample) / self.ranges, axis=1)
            nearest = int(np.argmin(distances))
            delta = sample - self.nodes[nearest].q
            norm = float(distances[nearest])
            if norm <= 1e-12:
                continue
            new_q = self.nodes[nearest].q + min(1.0, self.step_size / norm) * delta
            if not self.checker.is_valid(new_q):
                continue
            if not self.checker.is_motion_valid(self.nodes[nearest].q, new_q):
                continue

            n = len(self.nodes) + 1
            radius = self.neighbor_radius * (np.log(n) / n) ** (1.0 / dimensions)
            near_distances = np.linalg.norm((positions - new_q) / self.ranges, axis=1)
            neighbors = np.flatnonzero(near_distances <= radius).tolist()
            if nearest not in neighbors:
                neighbors.append(nearest)
            parent = nearest
            cost = self.nodes[nearest].cost + self.distance(self.nodes[nearest].q, new_q)
            for neighbor in neighbors:
                candidate_cost = self.nodes[neighbor].cost + float(near_distances[neighbor])
                if candidate_cost + 1e-12 < cost and self.checker.is_motion_valid(
                    self.nodes[neighbor].q, new_q
                ):
                    parent, cost = neighbor, candidate_cost
            new_id = len(self.nodes)
            self.nodes.append(_Node(new_q, parent, cost))
            self.nodes[parent].children.add(new_id)

            for neighbor in neighbors:
                if neighbor == 0 or self._is_ancestor(neighbor, new_id):
                    continue
                candidate_cost = self.nodes[new_id].cost + float(near_distances[neighbor])
                if candidate_cost + 1e-12 < self.nodes[
                    neighbor
                ].cost and self.checker.is_motion_valid(new_q, self.nodes[neighbor].q):
                    self._reparent(neighbor, new_id)

            goal_distance = self.distance(new_q, goal_q)
            if goal_distance <= self.step_size and self.checker.is_motion_valid(new_q, goal_q):
                candidates.append((new_id, goal_distance))
            # Переподключение может улучшить и ранее найденное соединение с целью.
            if candidates:
                parent_id, edge_cost = min(
                    candidates, key=lambda item: self.nodes[item[0]].cost + item[1]
                )
                current_cost = self.nodes[parent_id].cost + edge_cost
                if current_cost + 1e-12 < best_cost:
                    best_parent, best_cost = parent_id, current_cost
                    history.append((iteration, best_cost))
                if stop_on_first:
                    return PlanningResult(
                        True,
                        self._path(best_parent, goal_q),
                        iteration,
                        len(self.nodes),
                        best_cost,
                        "First verified goal connection.",
                        history,
                    )

        if best_parent is None:
            return PlanningResult(
                False,
                None,
                max_iterations,
                len(self.nodes),
                np.inf,
                "Iteration budget exhausted without a collision-free goal connection.",
                history,
            )
        return PlanningResult(
            True,
            self._path(best_parent, goal_q),
            max_iterations,
            len(self.nodes),
            best_cost,
            "Iteration budget completed; returning the best verified path.",
            history,
        )


def simplify_path(path: ArrayLike, checker: _Checker) -> NDArray[np.float64]:
    """Жадно удалять промежуточные точки только после проверки нового ребра."""
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or len(points) == 0 or not np.all(np.isfinite(points)):
        raise ValueError("path must be a non-empty finite array of configurations.")
    if not checker.is_valid(points[0]):
        raise ValueError("The input path has an invalid start configuration.")
    result = [points[0].copy()]
    current = 0
    while current < len(points) - 1:
        following = len(points) - 1
        while following > current + 1 and not checker.is_motion_valid(
            points[current], points[following]
        ):
            following -= 1
        if not checker.is_motion_valid(points[current], points[following]):
            raise ValueError("The input path contains an invalid edge.")
        result.append(points[following].copy())
        current = following
    return np.asarray(result)
