"""Физические ограничения и консервативная интервальная проверка сплайна."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .collision import CollisionChecker
    from .robot import RobotModel
    from .trajectory import JointSplineTrajectory


@dataclass
class ValidationReport:
    feasible: bool
    collision_certified: bool
    minimum_sampled_clearance: float
    collision_clearance_lower_bound: float
    position_violation: float
    velocity_ratio: float
    acceleration_ratio: float
    torque_ratio: float
    endpoint_position_error: float
    endpoint_velocity_error: float
    endpoint_acceleration_error: float
    checked_intervals: int
    dynamic_samples: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def certify_collision_free(
    trajectory: JointSplineTrajectory,
    checker: CollisionChecker,
    max_depth: int = 13,
) -> tuple[bool, float, float, int]:
    """Оценить смещение геометрии, включая оба звена при самостолкновениях.

    Оболочка коэффициентов производной сплайна ограничивает изменение углов
    на каждом интервале. Расстояние между движущимися множествами липшицево
    по максимальному смещению их точек. Неразрешённые интервалы отклоняются.
    """
    derivative_bound = trajectory.derivative_bounds(order=1)
    smallest_sample = float("inf")
    smallest_bound = float("inf")
    checked = 0
    for u in (0.0, 1.0):
        q = trajectory.evaluate(u)
        if not checker.is_valid(q):
            return False, checker.minimum_clearance(q), float("-inf"), checked
    stack = [(0.0, 1.0, 0)]
    while stack:
        a, b, depth = stack.pop()
        checked += 1
        clearance = checker.minimum_clearance(trajectory.evaluate((a + b) / 2))
        smallest_sample = min(smallest_sample, clearance)
        if clearance < checker.safety_margin:
            return False, smallest_sample, float("-inf"), checked
        displacement = checker.robot.motion_bound(derivative_bound * (b - a) / 2)
        lower_bound = clearance - 2 * displacement
        if lower_bound >= checker.safety_margin:
            smallest_bound = min(smallest_bound, lower_bound)
            continue
        if depth >= max_depth:
            return False, smallest_sample, lower_bound, checked
        midpoint = (a + b) / 2
        stack.extend(((a, midpoint, depth + 1), (midpoint, b, depth + 1)))
    return True, smallest_sample, smallest_bound, checked


def validate_trajectory(
    trajectory: JointSplineTrajectory,
    robot: RobotModel,
    checker: CollisionChecker,
    samples: int = 301,
    max_depth: int = 13,
) -> ValidationReport:
    """Подтвердить зазор и проверить динамику на заданной плотной сетке.

    Моменты проверяются численно; непрерывная гарантия для них не заявляется.
    Границы углов, скоростей и ускорений используют оболочки коэффициентов сплайна.
    """
    values = trajectory.sample(samples)
    arrays = (values["q"], values["qd"], values["qdd"], values["qddd"])
    if trajectory.duration <= 0 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Trajectory must have finite states and positive duration")
    cp = trajectory.control_points
    position_violation = max(
        0.0,
        float(np.max(robot.joint_lower - cp)),
        float(np.max(cp - robot.joint_upper)),
    )
    velocity_ratio = float(
        np.max(trajectory.derivative_bounds(1) / trajectory.duration / robot.velocity_limits)
    )
    acceleration_ratio = float(
        np.max(trajectory.derivative_bounds(2) / trajectory.duration**2 / robot.acceleration_limits)
    )
    torques = np.array(
        [robot.inverse_dynamics(q, qd, qdd) for q, qd, qdd in zip(*arrays[:3], strict=True)]
    )
    torque_ratio = float(np.max(np.abs(torques) / robot.torque_limits))
    endpoint_position_error = 0.0
    for key, index in (("start", 0), ("goal", -1)):
        target = robot.scene.get(key)
        if target is not None:
            endpoint_position_error = max(
                endpoint_position_error,
                float(np.max(np.abs(values["q"][index] - np.asarray(target)))),
            )
    endpoint_velocity_error = float(np.max(np.abs(values["qd"][[0, -1]])))
    endpoint_acceleration_error = float(np.max(np.abs(values["qdd"][[0, -1]])))
    sampled_clearance = min(checker.minimum_clearance(q) for q in values["q"])
    certified, midpoint_clearance, lower, checked = certify_collision_free(
        trajectory, checker, max_depth
    )
    sampled_clearance = min(sampled_clearance, midpoint_clearance)
    violations = []
    if position_violation > 1e-9:
        violations.append("joint position bounds")
    if velocity_ratio > 1 + 1e-9:
        violations.append("joint velocity bounds")
    if acceleration_ratio > 1 + 1e-9:
        violations.append("joint acceleration bounds")
    if not np.isfinite(torque_ratio) or torque_ratio > 1 + 1e-9:
        violations.append("sampled torque bounds")
    if not certified:
        violations.append("collision or unresolved geometric interval")
    if max(endpoint_position_error, endpoint_velocity_error, endpoint_acceleration_error) > 1e-8:
        violations.append("rest-to-rest endpoint conditions")
    return ValidationReport(
        not violations,
        certified,
        float(sampled_clearance),
        float(lower),
        position_violation,
        velocity_ratio,
        acceleration_ratio,
        torque_ratio,
        endpoint_position_error,
        endpoint_velocity_error,
        endpoint_acceleration_error,
        checked,
        samples,
        "; ".join(violations) or "accepted",
    )
