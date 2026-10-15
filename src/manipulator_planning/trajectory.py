"""Зажатые суставные B-сплайны с аналитическими производными по времени."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline


@dataclass(frozen=True, slots=True)
class JointSplineTrajectory:
    """Суставная траектория ``q(t) = spline(t / duration)`` на ``[0, duration]``.

    ``evaluate`` принимает нормированный параметр ``u``, но возвращает производные
    по физическому времени. ``derivative_bounds`` задаёт оценки по параметру ``u``,
    позволяя отдельно вычислить допустимую длительность движения.
    """

    control_points: NDArray[np.float64]
    knots: NDArray[np.float64]
    degree: int
    duration: float
    _spline: BSpline = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        points = np.asarray(self.control_points, dtype=float).copy()
        knots = np.asarray(self.knots, dtype=float).copy()
        if points.ndim != 2 or points.shape[1] == 0 or not np.all(np.isfinite(points)):
            raise ValueError("control_points must be a finite K-by-joints array.")
        if not isinstance(self.degree, (int, np.integer)) or self.degree < 1:
            raise ValueError("degree must be a positive integer.")
        if len(points) < self.degree + 1:
            raise ValueError("At least degree + 1 control points are required.")
        if (
            knots.ndim != 1
            or len(knots) != len(points) + self.degree + 1
            or not np.all(np.isfinite(knots))
            or np.any(np.diff(knots) < 0)
            or not np.isclose(knots[self.degree], 0)
            or not np.isclose(knots[-self.degree - 1], 1)
        ):
            raise ValueError("knots must be nondecreasing and define the domain [0, 1].")
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be positive and finite.")
        points.setflags(write=False)
        knots.setflags(write=False)
        object.__setattr__(self, "control_points", points)
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "degree", int(self.degree))
        object.__setattr__(self, "duration", float(self.duration))
        object.__setattr__(self, "_spline", BSpline(knots, points, self.degree, extrapolate=False))

    def evaluate(self, u: ArrayLike, derivative: int = 0) -> NDArray[np.float64]:
        parameter = np.asarray(u, dtype=float)
        if not np.all(np.isfinite(parameter)) or np.any(parameter < 0) or np.any(parameter > 1):
            raise ValueError("u must lie in the normalized interval [0, 1].")
        if not isinstance(derivative, (int, np.integer)) or derivative < 0:
            raise ValueError("derivative must be a non-negative integer.")
        if derivative > self.degree:
            return np.zeros(parameter.shape + (self.control_points.shape[1],))
        return np.asarray(self._spline(parameter, nu=derivative)) / self.duration**derivative

    def sample(self, n: int = 201) -> dict[str, NDArray[np.float64]]:
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("At least two samples are required.")
        u = np.linspace(0, 1, n)
        return {
            "time": u * self.duration,
            "u": u,
            "q": self.evaluate(u),
            "qd": self.evaluate(u, 1),
            "qdd": self.evaluate(u, 2),
            "qddd": self.evaluate(u, 3),
        }

    def with_duration(self, duration: float) -> JointSplineTrajectory:
        return JointSplineTrajectory(self.control_points, self.knots, self.degree, duration)

    def derivative_bounds(self, order: int = 1) -> NDArray[np.float64]:
        """Оценки ``abs(d**order q / du**order)`` по выпуклой оболочке коэффициентов."""
        if not isinstance(order, (int, np.integer)) or order < 0:
            raise ValueError("order must be a non-negative integer.")
        if order > self.degree:
            return np.zeros(self.control_points.shape[1])
        derivative = self._spline.derivative(order)
        active = len(derivative.t) - derivative.k - 1
        return np.max(np.abs(derivative.c[:active]), axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_points": self.control_points.tolist(),
            "knots": self.knots.tolist(),
            "degree": self.degree,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JointSplineTrajectory:
        return cls(
            np.asarray(data["control_points"], dtype=float),
            np.asarray(data["knots"], dtype=float),
            int(data["degree"]),
            float(data["duration"]),
        )


def fit_path(
    path: ArrayLike,
    n_control_points: int = 12,
    degree: int = 5,
    duration: float = 4.0,
) -> JointSplineTrajectory:
    """Приблизить ломаную методом наименьших квадратов с остановками на концах.

    Первые и последние три настоящих коэффициента сплайна закреплены на концах,
    обеспечивая нулевые граничные скорости и ускорения. Целевые точки следуют
    вдоль ломаной по закону пятой степени с плавным разгоном и торможением.
    После сглаживания необходима проверка столкновений: сплайн может срезать угол.
    """
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or len(points) < 2 or points.shape[1] == 0:
        raise ValueError("path must contain at least two joint configurations.")
    if not np.all(np.isfinite(points)):
        raise ValueError("path must contain only finite numbers.")
    if not isinstance(degree, (int, np.integer)) or degree < 3:
        raise ValueError("Rest-to-rest fitting requires a spline degree of at least three.")
    if not isinstance(n_control_points, (int, np.integer)) or n_control_points < max(6, degree + 1):
        raise ValueError("At least max(6, degree + 1) control points are required.")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, distances > 1e-12]]
    interior_count = n_control_points - degree - 1
    internal = np.linspace(0, 1, interior_count + 2)[1:-1]
    knots = np.r_[np.zeros(degree + 1), internal, np.ones(degree + 1)]
    coefficients = np.empty((n_control_points, points.shape[1]))
    coefficients[:3] = points[0]
    coefficients[-3:] = points[-1]

    if len(points) == 1:
        coefficients[:] = points[0]
        return JointSplineTrajectory(coefficients, knots, degree, duration)

    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    lengths /= lengths[-1]
    u = np.linspace(0, 1, max(201, 20 * n_control_points, 10 * len(points)))
    progress = 10 * u**3 - 15 * u**4 + 6 * u**5
    targets = np.column_stack(
        [np.interp(progress, lengths, points[:, joint]) for joint in range(points.shape[1])]
    )
    basis = BSpline.design_matrix(u, knots, degree).toarray()
    fixed = np.r_[np.arange(3), np.arange(n_control_points - 3, n_control_points)]
    free = np.arange(3, n_control_points - 3)
    if free.size:
        residual = targets - basis[:, fixed] @ coefficients[fixed]
        coefficients[free] = np.linalg.lstsq(basis[:, free], residual, rcond=None)[0]
    return JointSplineTrajectory(coefficients, knots, degree, duration)
