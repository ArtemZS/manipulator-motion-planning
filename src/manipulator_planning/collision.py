"""Знаковый геометрический зазор и консервативная проверка суставных отрезков."""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from manipulator_planning.robot import RobotModel


class CollisionChecker:
    """Проверять каждую капсулу относительно сцены и несоседних капсул.

    Соседние звенья конструктивно пересекаются в общем шарнире, поэтому
    исключены пять соседних пар. Пара link1/base также исключена как интерфейс
    вращающегося крепления. Все остальные пары с основанием, полом,
    препятствиями и несоседними звеньями проверяются.
    """

    def __init__(self, robot: RobotModel, safety_margin: float = 0.015) -> None:
        if not np.isfinite(safety_margin) or safety_margin < 0:
            raise ValueError("safety_margin must be finite and nonnegative")
        self.robot = robot
        self.safety_margin = float(safety_margin)
        self.check_count = 0
        links = robot.link_geom_ids
        base = robot.model.geom("base").id
        self.ignored_pairs = {tuple(sorted((links[i], links[i + 1]))) for i in range(5)} | {
            tuple(sorted((links[0], base)))
        }
        self.pairs = tuple(
            (link, world)
            for link in links
            for world in robot.world_geom_ids
            if tuple(sorted((link, world))) not in self.ignored_pairs
        ) + tuple((links[i], links[j]) for i in range(6) for j in range(i + 2, 6))

    def _within_limits(self, q: np.ndarray) -> bool:
        return bool(
            q.shape == (self.robot.nq,)
            and np.all(np.isfinite(q))
            and np.all(q >= self.robot.joint_lower)
            and np.all(q <= self.robot.joint_upper)
        )

    def minimum_clearance(self, q: ArrayLike) -> float:
        """Наименьшее знаковое расстояние поверхностей; отрицательное означает пересечение."""
        self.robot.set_configuration(q)
        self.check_count += 1
        return min(
            mujoco.mj_geomDistance(self.robot.model, self.robot.data, a, b, 100.0, None)
            for a, b in self.pairs
        )

    def is_valid(self, q: ArrayLike) -> bool:
        q = np.asarray(q, dtype=float)
        return self._within_limits(q) and self.minimum_clearance(q) >= self.safety_margin

    def is_motion_valid(self, a: ArrayLike, b: ArrayLike, max_depth: int = 12) -> bool:
        """Подтвердить безопасность отрезка, отклоняя неразрешённые интервалы.

        От середины интервала каждая движущаяся точка смещается не более чем
        на B. Расстояние пары звеньев изменяется не более чем на 2B. Поэтому
        зазор в середине выше margin+2B подтверждает весь интервал.
        Деление адаптивно. Суставные ограничения выпуклы: допустимость
        обоих концов обеспечивает их соблюдение на всём отрезке.
        """
        if not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        start, end = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if not self.is_valid(start) or not self.is_valid(end):
            return False
        if np.array_equal(start, end):
            return True
        stack = [(start, end, 0)]
        while stack:
            left, right, depth = stack.pop()
            middle = (left + right) * 0.5
            clearance = self.minimum_clearance(middle)
            if clearance < self.safety_margin:
                return False
            displacement = 2.0 * self.robot.motion_bound((right - left) * 0.5)
            if clearance - self.safety_margin > displacement:
                continue
            if depth >= max_depth:
                return False
            stack.append((middle, right, depth + 1))
            stack.append((left, middle, depth + 1))
        return True
