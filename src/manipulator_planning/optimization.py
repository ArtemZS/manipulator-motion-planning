"""NSGA-II с ограничениями по коэффициентам сплайна и длительности движения."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .objectives import evaluate_objectives
from .validation import ValidationReport, validate_trajectory

if TYPE_CHECKING:
    from .collision import CollisionChecker
    from .robot import RobotModel
    from .trajectory import JointSplineTrajectory


@dataclass
class OptimizationResult:
    trajectories: list[JointSplineTrajectory]
    objectives: np.ndarray
    validations: list[ValidationReport]
    history: list[dict]
    evaluations: int
    sampled_feasible: int
    final_rejections: int


class TrajectoryProblem(ElementwiseProblem):
    """Раздельные F и G сохраняют физический смысл критериев усилий и рывка."""

    def __init__(
        self,
        baseline: JointSplineTrajectory,
        robot: RobotModel,
        checker: CollisionChecker,
        duration_bounds: tuple[float, float],
        shape_radius: float = 0.12,
        collision_samples: int = 51,
    ):
        self.baseline = baseline
        self.robot = robot
        self.checker = checker
        self.collision_samples = collision_samples
        self.free_shape = baseline.control_points[3:-3].shape
        center = baseline.control_points[3:-3]
        lower = np.maximum(center - shape_radius, robot.joint_lower)
        upper = np.minimum(center + shape_radius, robot.joint_upper)
        self.observed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        super().__init__(
            n_var=center.size + 1,
            n_obj=3,
            n_ieq_constr=5,
            xl=np.r_[lower.ravel(), duration_bounds[0]],
            xu=np.r_[upper.ravel(), duration_bounds[1]],
        )

    def encode(self, trajectory: JointSplineTrajectory) -> np.ndarray:
        return np.r_[trajectory.control_points[3:-3].ravel(), trajectory.duration]

    def decode(self, x: np.ndarray) -> JointSplineTrajectory:
        from .trajectory import JointSplineTrajectory

        cp = self.baseline.control_points.copy()
        cp[3:-3] = np.asarray(x[:-1]).reshape(self.free_shape)
        return JointSplineTrajectory(
            control_points=cp,
            knots=self.baseline.knots.copy(),
            degree=self.baseline.degree,
            duration=float(x[-1]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        trajectory = self.decode(x)
        f, tau = evaluate_objectives(trajectory, self.robot)
        q = trajectory.evaluate(np.linspace(0, 1, self.collision_samples))
        clearance = min(self.checker.minimum_clearance(p) for p in q)
        # Оболочки коэффициентов дают консервативные непрерывные границы кинематики.
        cp = trajectory.control_points
        position = max(
            float(
                np.max(
                    (self.robot.joint_lower - cp)
                    / (self.robot.joint_upper - self.robot.joint_lower)
                )
            ),
            float(
                np.max(
                    (cp - self.robot.joint_upper)
                    / (self.robot.joint_upper - self.robot.joint_lower)
                )
            ),
        )
        velocity = np.max(
            trajectory.derivative_bounds(1) / trajectory.duration / self.robot.velocity_limits
        )
        acceleration = np.max(
            trajectory.derivative_bounds(2)
            / trajectory.duration**2
            / self.robot.acceleration_limits
        )
        g = np.array(
            [
                position,
                velocity - 1,
                acceleration - 1,
                np.max(np.abs(tau) / self.robot.torque_limits) - 1,
                (self.checker.safety_margin - clearance) / max(self.checker.safety_margin, 0.01),
            ]
        )
        out["F"], out["G"] = f, g
        self.observed.append((np.array(x).copy(), f.copy(), g.copy()))


class ProgressHistory(Callback):
    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []

    def notify(self, algorithm):
        f = algorithm.pop.get("F")
        valid = np.asarray(algorithm.pop.get("CV")).ravel() <= 0
        front = (
            NonDominatedSorting().do(f[valid], only_non_dominated_front=True) if valid.any() else []
        )
        row = {
            "generation": int(algorithm.n_gen),
            "evaluations": int(algorithm.evaluator.n_eval),
            "feasible_population": int(valid.sum()),
            "nondominated_population": len(front),
        }
        self.rows.append(row)
        if algorithm.n_gen == 1 or algorithm.n_gen % 5 == 0:
            print(
                f"NSGA-II: generation {row['generation']}, "
                f"feasible {row['feasible_population']}/{len(f)}, "
                f"evaluations {row['evaluations']}",
                flush=True,
            )


def optimize_trajectory(
    baseline: JointSplineTrajectory,
    robot: RobotModel,
    checker: CollisionChecker,
    *,
    seed: int = 42,
    population: int = 32,
    generations: int = 24,
    duration_bounds: tuple[float, float] = (1.0, 12.0),
    shape_radius: float = 0.12,
    max_final_candidates: int = 48,
) -> OptimizationResult:
    if population < 4 or generations < 1:
        raise ValueError("population must be >= 4 and generations >= 1")
    if not 0 < duration_bounds[0] < duration_bounds[1]:
        raise ValueError("duration bounds must be positive and increasing")
    if not duration_bounds[0] <= baseline.duration <= duration_bounds[1]:
        raise ValueError("baseline duration is outside optimization bounds")
    if shape_radius <= 0:
        raise ValueError("shape_radius must be positive")
    if max_final_candidates < 4:
        raise ValueError("max_final_candidates must be >= 4")
    problem = TrajectoryProblem(baseline, robot, checker, duration_bounds, shape_radius)
    rng = np.random.default_rng(seed)
    center = problem.encode(baseline)
    initial = np.tile(center, (population, 1))
    # Исходная популяция включает несколько длительностей проверенной формы сплайна.
    for i in range(1, population):
        if i >= min(8, population // 2):
            initial[i, :-1] += rng.normal(0, shape_radius / 4, len(center) - 1)
        initial[i, -1] = rng.uniform(*duration_bounds)
    initial = np.clip(initial, problem.xl, problem.xu)
    callback = ProgressHistory()
    algorithm = NSGA2(
        pop_size=population,
        sampling=initial,
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(prob=1.0, prob_var=1 / problem.n_var, eta=20),
        eliminate_duplicates=True,
    )
    minimize(
        problem,
        algorithm,
        termination=("n_gen", generations),
        seed=seed,
        callback=callback,
        copy_algorithm=False,
        verbose=False,
    )
    feasible = [(x, f) for x, f, g in problem.observed if np.all(g <= 0)]
    # Сохраняем допустимый исходный сплайн при неудачном стохастическом поиске.
    seed_f, _ = evaluate_objectives(baseline, robot, quadrature_order=10)
    feasible.append((center, seed_f))
    unique = {}
    for x, f in feasible:
        unique[tuple(np.round(x, 10))] = (x, f)
    candidates = list(unique.values())
    f = np.array([item[1] for item in candidates])
    layers = NonDominatedSorting().do(f)
    selected_indices = []
    for layer in layers:
        remaining = max_final_candidates - len(selected_indices)
        if remaining <= 0:
            break
        if len(layer) <= remaining:
            selected_indices.extend(layer.tolist())
        else:
            ordered = layer[np.argsort(f[layer, 2])]
            extremes = list(dict.fromkeys(layer[np.argmin(f[layer], axis=0)].tolist()))
            chosen = extremes[:remaining]
            for index in ordered[np.linspace(0, len(ordered) - 1, remaining).astype(int)]:
                if int(index) not in chosen and len(chosen) < remaining:
                    chosen.append(int(index))
            if len(chosen) < remaining:
                chosen.extend(int(i) for i in ordered if int(i) not in chosen)
            selected_indices.extend(chosen[:remaining])
    selected_x = [candidates[i][0] for i in selected_indices]
    if not any(np.array_equal(x, center) for x in selected_x):
        selected_x.append(center)
    accepted, values, reports = [], [], []
    rejected = 0
    print(f"Final validation: {len(selected_x)} candidate trajectories", flush=True)
    for x in selected_x:
        trajectory = problem.decode(x)
        report = validate_trajectory(trajectory, robot, checker)
        if report.feasible:
            accepted.append(trajectory)
            refined, _ = evaluate_objectives(trajectory, robot, quadrature_order=10)
            values.append(refined)
            reports.append(report)
        else:
            rejected += 1
    if not accepted:
        raise RuntimeError("No trajectory passed final validation, including the seed")
    values = np.array(values)
    front = NonDominatedSorting().do(values, only_non_dominated_front=True)
    front = front[np.argsort(values[front, 2])]
    return OptimizationResult(
        [accepted[i] for i in front],
        values[front],
        [reports[i] for i in front],
        callback.rows,
        len(problem.observed),
        len(unique),
        rejected,
    )


def balanced_index(objectives: np.ndarray) -> int:
    """Минимальное нормированное расстояние до идеальной точки с равными весами."""
    span = np.ptp(objectives, axis=0)
    span[span < 1e-12] = 1
    scaled = (objectives - objectives.min(axis=0)) / span
    return int(np.argmin(np.sum(scaled**2, axis=1)))
