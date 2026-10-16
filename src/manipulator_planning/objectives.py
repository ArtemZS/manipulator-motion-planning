"""Три критерия дипломной работы с единой шкалой физического времени."""

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial.legendre import leggauss

if TYPE_CHECKING:
    from .robot import RobotModel
    from .trajectory import JointSplineTrajectory


@lru_cache(maxsize=32)
def quadrature_nodes(knots: tuple[float, ...], order: int) -> tuple[np.ndarray, np.ndarray]:
    """Квадратура Гаусса — Лежандра на ненулевых интервалах сплайна в [0, 1]."""
    x, weights = leggauss(order)
    unique = np.unique(np.asarray(knots))
    nodes, scaled_weights = [], []
    for a, b in zip(unique[:-1], unique[1:], strict=True):
        if b > a and a >= 0 and b <= 1:
            nodes.extend((a + b) / 2 + x * (b - a) / 2)
            scaled_weights.extend(weights * (b - a) / 2)
    return np.asarray(nodes), np.asarray(scaled_weights)


def evaluate_objectives(
    trajectory: JointSplineTrajectory, robot: RobotModel, quadrature_order: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Вернуть [integral(tau²) dt, integral(jerk²) dt, T] и значения моментов.

    J_tau характеризует управляющие усилия и измеряется в (N m)^2 s.
    Для расчёта электрической энергии требуется другая модель привода.
    """
    u, weights = quadrature_nodes(tuple(trajectory.knots), quadrature_order)
    q, qd, qdd, jerk = (trajectory.evaluate(u, derivative=d) for d in range(4))
    tau = np.asarray([robot.inverse_dynamics(a, b, c) for a, b, c in zip(q, qd, qdd, strict=True)])
    result = np.array(
        [
            trajectory.duration * np.dot(weights, np.sum(tau**2, axis=1)),
            trajectory.duration * np.dot(weights, np.sum(jerk**2, axis=1)),
            trajectory.duration,
        ]
    )
    if not np.isfinite(result).all():
        raise ValueError("Non-finite objective values")
    return result, tau
