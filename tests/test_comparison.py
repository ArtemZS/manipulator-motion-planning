"""Общий исходный сплайн сравниваемых методов и сохранность частичного результата."""

import json
from dataclasses import replace

import numpy as np
import pytest

from manipulator_planning.pipeline import PipelineOptions, load_scene, run_pipeline


def test_baseline_only_has_same_seed_path_and_no_optimizer(tmp_path):
    scene = load_scene("simple")
    options = PipelineOptions(rrt_iterations=180, population=6, generations=2, plots=False)
    full = run_pipeline(scene, tmp_path / "full", options)
    baseline = run_pipeline(scene, tmp_path / "baseline", replace(options, method="baseline"))
    assert baseline["method"] == "baseline"
    assert baseline["optimization"]["evaluations"] == 0
    assert baseline["timings_seconds"]["optimization_and_validation"] == 0
    assert len(baseline["solutions"]) == 1
    np.testing.assert_array_equal(
        full["baseline"]["objectives"], baseline["baseline"]["objectives"]
    )
    with np.load(tmp_path / "full" / "trajectories.npz") as a:
        with np.load(tmp_path / "baseline" / "trajectories.npz") as b:
            np.testing.assert_array_equal(a["rrt_path"], b["rrt_path"])


def test_optimizer_failure_preserves_baseline_and_failure_stage(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Тестовый отказ оптимизатора")

    monkeypatch.setattr("manipulator_planning.pipeline.optimize_trajectory", fail)
    with pytest.raises(RuntimeError, match="Тестовый отказ"):
        run_pipeline(load_scene(), tmp_path, PipelineOptions(rrt_iterations=180, plots=False))
    partial = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert partial["status"] == "failed"
    assert partial["failure_stage"] == "optimization"
    assert partial["baseline"]["validation"]["feasible"]
    assert partial["timings_seconds"]["smoothing"] > 0
    assert partial["timings_seconds"]["optimization_and_validation"] >= 0
    assert (tmp_path / "baseline.json").exists()
