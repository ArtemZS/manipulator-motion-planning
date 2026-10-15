"""Полный каскад планирования и воспроизводимое сохранение результатов."""

import json
import platform
from dataclasses import asdict, dataclass
from importlib import metadata
from importlib.resources import files
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from .collision import CollisionChecker
from .objectives import evaluate_objectives
from .optimization import OptimizationResult, balanced_index, optimize_trajectory
from .planning import RRTStar, simplify_path
from .robot import RobotModel
from .trajectory import JointSplineTrajectory, fit_path
from .validation import certify_collision_free, validate_trajectory


@dataclass(frozen=True)
class PipelineOptions:
    seed: int = 42
    rrt_iterations: int = 500
    population: int = 32
    generations: int = 24
    control_points: int = 12
    shape_radius: float = 0.12
    minimum_duration: float = 1.0
    maximum_duration: float = 12.0
    plots: bool = True
    method: str = "full"
    rrt_step_size: float = 0.10
    rrt_goal_bias: float = 0.25
    rrt_local_sample_rate: float = 0.45
    rrt_local_sample_scale: float = 0.12
    views: tuple[str, ...] = ("isometric",)
    frame_count: int = 55

    def validate(self) -> None:
        if self.rrt_iterations < 1:
            raise ValueError("Число итераций RRT* должно быть положительным")
        if self.population < 4 or self.generations < 1:
            raise ValueError("Нужны популяция не меньше 4 и хотя бы одно поколение")
        if self.control_points < 7:
            raise ValueError("Нужно не менее 7 коэффициентов для оптимизации формы")
        if not 0 < self.minimum_duration < self.maximum_duration:
            raise ValueError("Границы длительности должны быть положительными и возрастающими")
        if not np.isfinite(self.shape_radius) or self.shape_radius <= 0:
            raise ValueError("Радиус изменения коэффициентов должен быть положительным")
        if not np.isfinite([self.minimum_duration, self.maximum_duration]).all():
            raise ValueError("Границы длительности должны быть конечными")
        if self.method not in ("full", "baseline"):
            raise ValueError("Метод должен быть full или baseline")
        if not np.isfinite(self.rrt_step_size) or self.rrt_step_size <= 0:
            raise ValueError("Шаг RRT* должен быть конечным и положительным")
        if not 0 <= self.rrt_goal_bias <= 1 or not 0 <= self.rrt_local_sample_rate < 1:
            raise ValueError("Вероятности выборки RRT* должны лежать в [0, 1]")
        if self.rrt_goal_bias + self.rrt_local_sample_rate >= 1:
            raise ValueError("Необходимо сохранить положительную долю равномерной выборки")
        if not np.isfinite(self.rrt_local_sample_scale) or self.rrt_local_sample_scale <= 0:
            raise ValueError("Масштаб локальной выборки должен быть положительным")
        if not self.views or any(
            v not in ("isometric", "front", "side", "top") for v in self.views
        ):
            raise ValueError("Ракурсы: isometric, front, side, top")
        if not isinstance(self.frame_count, int) or not 2 <= self.frame_count <= 240:
            raise ValueError("Число кадров должно быть целым от 2 до 240")


def load_scene(path: Path | str | None = None) -> dict:
    """Прочитать пользовательскую сцену или поставляемый пример из пакета."""
    if isinstance(path, str) and not Path(path).is_file() and not Path(path).suffix:
        from .scenes import load_named_scene

        return load_named_scene(path)
    source = (
        Path(path)
        if path is not None
        else files("manipulator_planning").joinpath("assets/simple.yaml")
    )
    scene = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(scene, dict):
        raise ValueError("Описание сцены должно быть словарём YAML")
    for key in ("start", "goal"):
        value = np.asarray(scene.get(key), dtype=float)
        if value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError(f"Поле {key} должно содержать шесть конечных углов в радианах")
    if not isinstance(scene.get("obstacles", []), list):
        raise ValueError("Поле obstacles должно быть списком")
    return scene


def write_json(path: Path, value: dict) -> None:
    """Записать UTF-8 JSON без нечисловых значений и неполного файла отчёта."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _build_baseline(path, robot, checker, options):
    """Аппроксимировать найденный путь и подобрать допустимую длительность."""
    simplified = simplify_path(path, checker)
    attempts = []
    counts = list(dict.fromkeys((options.control_points, 18, 24, 32, 48)))
    for points in (simplified, path):
        for count in counts:
            fitted = fit_path(points, n_control_points=count)
            # Оболочка коэффициентов обеспечивает соблюдение пределов углов.
            fitted = JointSplineTrajectory(
                np.clip(fitted.control_points, robot.joint_lower, robot.joint_upper),
                fitted.knots,
                fitted.degree,
                fitted.duration,
            )
            collision_free, _, _, _ = certify_collision_free(fitted, checker)
            if not collision_free:
                attempts.append(f"K={count}: сглаживание не прошло проверку геометрии")
                continue
            velocity_time = np.max(fitted.derivative_bounds(1) / robot.velocity_limits)
            acceleration_time = np.sqrt(
                np.max(fitted.derivative_bounds(2) / robot.acceleration_limits)
            )
            initial_time = max(
                options.minimum_duration,
                min(4.0, options.maximum_duration),
                float(velocity_time) * 1.15,
                float(acceleration_time) * 1.15,
            )
            if initial_time > options.maximum_duration:
                attempts.append(f"K={count}: кинематике требуется T>{options.maximum_duration}")
                continue
            for duration in np.linspace(initial_time, options.maximum_duration, 5):
                candidate = fitted.with_duration(float(duration))
                report = validate_trajectory(candidate, robot, checker)
                if report.feasible:
                    return candidate, report, attempts
                attempts.append(f"K={count}, T={duration:.3f}: {report.reason}")
    raise RuntimeError("Не удалось построить допустимый B-сплайн: " + "; ".join(attempts))


def run_pipeline(
    scene: dict,
    output_dir: Path,
    options: PipelineOptions | None = None,
) -> dict:
    """Найти путь, построить сплайн, получить фронт и сохранить тот же результат."""
    options = options or PipelineOptions()
    options.validate()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "schema_version": 1,
        "status": "running",
        "scene": scene.get("name", "custom"),
        "options": asdict(options),
        "seeds": {"rrt": options.seed, "nsga2": options.seed + 1},
        "failure_stage": "model",
        "timings_seconds": {},
    }
    write_json(output_dir / "report.json", progress)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(scene, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    started = perf_counter()
    try:
        return _run(scene, output_dir, options, started, progress)
    except Exception as exc:
        progress.update(status="failed", reason=str(exc))
        elapsed = perf_counter() - started
        timing_key = {
            "planning": "planning",
            "smoothing": "smoothing",
            "optimization": "optimization_and_validation",
        }.get(progress["failure_stage"])
        if timing_key is not None and timing_key not in progress["timings_seconds"]:
            progress["timings_seconds"][timing_key] = max(
                0.0, elapsed - sum(progress["timings_seconds"].values())
            )
        progress["timings_seconds"]["total"] = elapsed
        write_json(output_dir / "report.json", progress)
        raise


def _run(scene, output_dir, options, started, progress):
    robot = RobotModel(scene)
    checker = CollisionChecker(robot, float(scene.get("safety_margin", 0.015)))
    start, goal = np.asarray(scene["start"]), np.asarray(scene["goal"])
    print("1/4: RRT* в пространстве шести суставов", flush=True)
    planning_started = perf_counter()
    progress["failure_stage"] = "planning"
    direct_valid = checker.is_motion_valid(start, goal)
    planner = RRTStar(
        robot,
        checker,
        seed=options.seed,
        step_size=options.rrt_step_size,
        goal_bias=options.rrt_goal_bias,
        local_sample_rate=options.rrt_local_sample_rate,
        local_sample_scale=options.rrt_local_sample_scale,
    )
    planning = planner.plan(start, goal, max_iterations=options.rrt_iterations)
    planning_time = perf_counter() - planning_started
    progress["timings_seconds"]["planning"] = planning_time
    progress["planning"] = {
        "direct_motion_valid": bool(direct_valid),
        "iterations": planning.iterations,
        "nodes": planning.node_count,
        "path_points": len(planning.path) if planning.path is not None else 0,
        "cost_history": planning.cost_history,
        "reason": planning.reason,
    }
    if not planning.success:
        raise RuntimeError(f"RRT*: {planning.reason}")
    print(f"RRT*: {len(planning.path)} точек, узлов {planning.node_count}", flush=True)
    print("2/4: суставный B-сплайн и проверка ограничений", flush=True)
    smoothing_started = perf_counter()
    progress["failure_stage"] = "smoothing"
    write_json(output_dir / "report.json", progress)
    baseline, baseline_validation, fit_attempts = _build_baseline(
        planning.path, robot, checker, options
    )
    baseline_objectives, _ = evaluate_objectives(baseline, robot, quadrature_order=10)
    smoothing_time = perf_counter() - smoothing_started
    progress["timings_seconds"]["smoothing"] = smoothing_time
    progress["baseline"] = {
        "objectives": baseline_objectives.tolist(),
        "control_points": len(baseline.control_points),
        "validation": baseline_validation.to_dict(),
        "fit_attempts": fit_attempts,
    }
    write_json(output_dir / "baseline.json", baseline.to_dict())
    print(
        f"B-сплайн: K={len(baseline.control_points)}, T={baseline.duration:.3f} с, "
        f"зазор >= {baseline_validation.collision_clearance_lower_bound:.4f} м",
        flush=True,
    )
    print("3/4: NSGA-II по коэффициентам и длительности", flush=True)
    optimization_started = perf_counter()
    progress["failure_stage"] = "optimization"
    write_json(output_dir / "report.json", progress)
    if options.method == "full":
        optimized = optimize_trajectory(
            baseline,
            robot,
            checker,
            seed=options.seed + 1,
            population=options.population,
            generations=options.generations,
            duration_bounds=(options.minimum_duration, options.maximum_duration),
            shape_radius=options.shape_radius,
        )
        optimization_time = perf_counter() - optimization_started
    else:
        print("NSGA-II пропущен: сохраняется исходная допустимая траектория", flush=True)
        optimized = OptimizationResult(
            [baseline], np.array([baseline_objectives]), [baseline_validation], [], 0, 1, 0
        )
        optimization_time = 0.0
    progress["timings_seconds"]["optimization_and_validation"] = optimization_time
    progress["failure_stage"] = "artifacts"
    selected = balanced_index(optimized.objectives)
    print(f"4/4: сохранение {len(optimized.trajectories)} допустимых решений", flush=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(scene, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    write_json(
        output_dir / "trajectories.json",
        {
            "schema_version": 1,
            "baseline": baseline.to_dict(),
            "solutions": [t.to_dict() for t in optimized.trajectories],
            "selected_index": selected,
        },
    )
    arrays = {"rrt_path": planning.path, "objectives": optimized.objectives}
    for index, trajectory in enumerate(optimized.trajectories):
        samples = trajectory.sample(201)
        for key, value in samples.items():
            arrays[f"solution_{index}_{key}"] = value
        arrays[f"solution_{index}_tau"] = np.array(
            [
                robot.inverse_dynamics(q, qd, qdd)
                for q, qd, qdd in zip(samples["q"], samples["qd"], samples["qdd"], strict=True)
            ]
        )
    np.savez_compressed(output_dir / "trajectories.npz", **arrays)
    np.savetxt(
        output_dir / "pareto.csv",
        optimized.objectives,
        delimiter=",",
        header="effort_N2m2s,jerk_rad2_s5,duration_s",
        comments="",
    )
    assets = {}
    rendering_started = perf_counter()
    if options.plots:
        from .visualization import create_report

        assets = create_report(
            output_dir,
            robot,
            scene,
            optimized.trajectories,
            optimized.objectives,
            selected,
            baseline=baseline,
            views=options.views,
            frame_count=options.frame_count,
        )
    report = {
        "schema_version": 1,
        "status": "success",
        "method": options.method,
        "failure_stage": None,
        "scene": scene.get("name", "custom"),
        "options": asdict(options),
        "environment": {
            "python": platform.python_version(),
            "packages": {
                name: metadata.version(name)
                for name in ("manipulator-planning", "mujoco", "numpy", "scipy", "pymoo")
            },
        },
        "seeds": {"rrt": options.seed, "nsga2": options.seed + 1},
        "objectives": {
            "names": ["J_tau", "J_jerk", "T"],
            "units": ["(N m)^2 s", "rad^2 / s^5", "s"],
            "effort_is_not_electrical_energy": True,
        },
        "planning": {
            "direct_motion_valid": bool(direct_valid),
            "iterations": planning.iterations,
            "nodes": planning.node_count,
            "path_points": len(planning.path),
            "normalized_path_length": planning.cost,
            "cost_history": planning.cost_history,
            "start_error": float(np.linalg.norm(planning.path[0] - start)),
            "goal_error": float(np.linalg.norm(planning.path[-1] - goal)),
        },
        "baseline": {
            "objectives": baseline_objectives.tolist(),
            "control_points": len(baseline.control_points),
            "validation": baseline_validation.to_dict(),
            "fit_attempts": fit_attempts,
        },
        "optimization": {
            "evaluations": optimized.evaluations,
            "sampled_feasible_candidates": optimized.sampled_feasible,
            "rejected_at_final_validation": optimized.final_rejections,
            "history": optimized.history,
        },
        "selected_index": selected,
        "solutions": [
            {"index": i, "objectives": f.tolist(), "validation": v.to_dict()}
            for i, (f, v) in enumerate(
                zip(optimized.objectives, optimized.validations, strict=True)
            )
        ],
        "checks": {
            "geometric_interval_certification": True,
            "kinematic_bounds": "spline coefficient hulls",
            "torque_validation": "301 time samples, not a continuous-time certificate",
            "pareto_front": "finite approximation among validated candidates",
            "animation": "kinematic playback, not closed-loop dynamic tracking",
        },
        "collision_queries": checker.check_count,
        "timings_seconds": {
            "planning": planning_time,
            "smoothing": smoothing_time,
            "optimization_and_validation": optimization_time,
            "artifacts": perf_counter() - rendering_started,
            "total": perf_counter() - started,
        },
        "artifacts": assets,
    }
    write_json(output_dir / "report.json", report)
    print(f"Готово: {output_dir / 'report.json'}", flush=True)
    return report


def replay_results(
    directory: Path,
    *,
    regenerate: bool = True,
    views: tuple[str, ...] | None = None,
    frame_count: int = 55,
) -> dict:
    """Восстановить сохранённые сплайны и повторить проверку без оптимизации."""
    directory = directory.resolve()
    scene = load_scene(directory / "config.yaml")
    saved = json.loads((directory / "trajectories.json").read_text(encoding="utf-8"))
    if saved.get("schema_version") != 1:
        raise ValueError("Неизвестная версия сохранённого результата")
    trajectories = [JointSplineTrajectory.from_dict(t) for t in saved["solutions"]]
    baseline = JointSplineTrajectory.from_dict(saved["baseline"])
    selected = int(saved["selected_index"])
    if not trajectories or not 0 <= selected < len(trajectories):
        raise ValueError("Некорректный индекс выбранного решения")
    robot = RobotModel(scene)
    checker = CollisionChecker(robot, scene.get("safety_margin", 0.015))
    reports = [validate_trajectory(t, robot, checker).to_dict() for t in trajectories]
    if not all(item["feasible"] for item in reports):
        raise RuntimeError("Сохранённый результат не прошёл повторную проверку")
    objectives = np.array([evaluate_objectives(t, robot, 10)[0] for t in trajectories])
    artifacts = {}
    if regenerate:
        from .visualization import create_report

        artifacts = create_report(
            directory,
            robot,
            scene,
            trajectories,
            objectives,
            selected,
            baseline=baseline,
            views=views or ("isometric",),
            frame_count=frame_count,
        )
    result = {
        "status": "success",
        "solutions": reports,
        "objectives": objectives.tolist(),
        "artifacts": artifacts,
    }
    write_json(directory / "replay_report.json", result)
    return result
