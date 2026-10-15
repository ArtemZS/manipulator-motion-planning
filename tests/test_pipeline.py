"""Сквозные проверки каскада, сохранения и независимого воспроизведения."""

import json

import numpy as np
import pytest
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from manipulator_planning.cli import main
from manipulator_planning.optimization import balanced_index
from manipulator_planning.pipeline import (
    PipelineOptions,
    load_scene,
    replay_results,
    run_pipeline,
)
from manipulator_planning.trajectory import JointSplineTrajectory


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pipeline")
    scene = load_scene()
    # Готовый обход служит только геометрическим тестам и недоступен пайплайну.
    scene.pop("diagnostic_detour", None)
    report = run_pipeline(
        scene,
        directory,
        PipelineOptions(rrt_iterations=180, population=8, generations=3, plots=False),
    )
    return directory, report


def test_full_cascade_returns_physical_nondominated_solutions(completed_run):
    directory, report = completed_run
    assert report["status"] == "success"
    assert not report["planning"]["direct_motion_valid"]
    assert report["planning"]["goal_error"] == 0
    assert report["optimization"]["evaluations"] == 24
    assert len(report["solutions"]) >= 2
    objectives = np.array([item["objectives"] for item in report["solutions"]])
    assert np.ptp(objectives[:, 2]) > 0.1
    front = NonDominatedSorting().do(objectives, only_non_dominated_front=True)
    assert len(front) == len(objectives)
    for item in report["solutions"]:
        validation = item["validation"]
        assert validation["feasible"] and validation["collision_certified"]
        assert validation["collision_clearance_lower_bound"] >= 0.015
        assert validation["velocity_ratio"] <= 1
        assert validation["acceleration_ratio"] <= 1
        assert validation["torque_ratio"] <= 1
        assert validation["endpoint_position_error"] < 1e-10
    stored = json.loads((directory / "trajectories.json").read_text(encoding="utf-8"))
    trajectories = [JointSplineTrajectory.from_dict(item) for item in stored["solutions"]]
    baseline = JointSplineTrajectory.from_dict(stored["baseline"])
    assert any(
        np.max(np.abs(t.control_points - baseline.control_points)) > 1e-5 for t in trajectories
    )
    with np.load(directory / "trajectories.npz", allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["objectives"], objectives)
        for i, trajectory in enumerate(trajectories):
            np.testing.assert_allclose(arrays[f"solution_{i}_q"], trajectory.sample()["q"])


def test_saved_coefficients_replay_to_the_same_objectives(completed_run):
    directory, report = completed_run
    replay = replay_results(directory, regenerate=False)
    np.testing.assert_allclose(
        replay["objectives"],
        [item["objectives"] for item in report["solutions"]],
        rtol=1e-12,
        atol=1e-12,
    )
    assert all(item["feasible"] for item in replay["solutions"])


def test_replay_passes_original_rrt_and_saved_spline_to_renderer(completed_run, monkeypatch):
    directory, report = completed_run
    received = {}

    def capture(*args, **kwargs):
        received["trajectory"] = args[3][args[5]]
        received["rrt_path"] = kwargs["rrt_path"]
        return {}

    monkeypatch.setattr("manipulator_planning.visualization.create_report", capture)
    replay = replay_results(directory)
    with np.load(directory / "trajectories.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(received["rrt_path"], saved["rrt_path"])
        np.testing.assert_allclose(
            received["trajectory"].sample()["q"],
            saved[f"solution_{report['selected_index']}_q"],
        )
    variants = replay["representative_variants"]
    assert len(variants) == 4
    assert variants[-1]["solution_index"] == report["selected_index"]


def test_invalid_start_is_a_failed_run_not_a_successful_one(tmp_path):
    scene = load_scene()
    scene["start"][0] = 100
    with pytest.raises(RuntimeError, match="RRT"):
        run_pipeline(scene, tmp_path, PipelineOptions(plots=False))
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"


def test_options_and_cli_reject_invalid_requests(tmp_path):
    with pytest.raises(ValueError):
        PipelineOptions(population=3).validate()
    assert main(["demo", "--population", "3", "--output", str(tmp_path)]) == 1
    assert main(["replay", str(tmp_path)]) == 1
    assert balanced_index(np.array([[1.0, 4.0, 5.0], [4.0, 1.0, 5.0]])) == 0
