"""Парность экспериментов, сохранение отказов и корректность статистики."""

import copy
import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from manipulator_planning import experiments
from manipulator_planning.pipeline import PipelineOptions, write_json


def _solution(values):
    """Собрать минимальную проверенную траекторию для тестирования статистики."""
    return {
        "objectives": list(values),
        "validation": {"feasible": True, "collision_clearance_lower_bound": 0.02},
    }


def _report():
    """Сохранить компромисс с увеличением усилий и уменьшением рывка и времени."""
    return {
        "status": "success",
        "baseline": _solution([6.0, 6.0, 6.0]),
        "solutions": [_solution([9.0, 2.0, 4.0]), _solution([2.0, 9.0, 4.0])],
        "selected_index": 0,
        "timings_seconds": {
            "planning": 1.0,
            "smoothing": 2.0,
            "optimization_and_validation": 7.0,
            "artifacts": 100.0,
            "total": 110.0,
        },
    }


@pytest.fixture
def isolated_batch(monkeypatch):
    """Исключить тяжёлую физику и рендеринг из проверки протокола экспериментов."""
    from manipulator_planning import scenes

    monkeypatch.setattr(scenes, "load_named_scene", lambda name: {"name": name, "obstacles": []})
    monkeypatch.setattr(experiments, "_source_hash", lambda: "fixed-source")
    monkeypatch.setattr(experiments, "_create_report", lambda *args: None)


def test_pairs_share_one_preprocessing_and_failures_are_retained(
    tmp_path, monkeypatch, isolated_batch
):
    calls = []

    def runner(scene, directory, options):
        calls.append(options)
        assert not options.plots and options.method == "full"
        if options.seed == 2:
            raise RuntimeError("RRT*: исчерпан бюджет")
        if options.seed == 3:
            partial = _report()
            partial.update(
                status="failed", failure_stage="optimization", reason="Ошибка оптимизации"
            )
            partial.pop("solutions")
            write_json(directory / "report.json", partial)
            raise RuntimeError("Ошибка оптимизации")
        return _report()

    monkeypatch.setattr(experiments, "run_pipeline", runner)
    summary = experiments.run_experiments(["test"], [1, 2, 3], tmp_path, PipelineOptions())
    assert len(calls) == 3
    baseline, full = summary["methods"]
    assert (baseline["attempted"], baseline["succeeded"], baseline["failed"]) == (3, 2, 1)
    assert (full["attempted"], full["succeeded"], full["failed"]) == (3, 1, 2)
    assert baseline["metrics_successful_runs"]["runtime_numerical_s"]["median"] == 3.0
    assert full["metrics_successful_runs"]["runtime_numerical_s"]["median"] == 10.0
    balanced = summary["paired"][0]
    assert balanced["paired_successes"] == 1
    assert balanced["changes_full_minus_baseline"]["J_tau"]["percent"]["median"] == 50.0
    with (tmp_path / "runs.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert rows[2]["success"] == "False" and rows[2]["failure_stage"] == "planning"
    assert rows[-1]["failure_stage"] == "optimization"
    assert rows[-2]["success"] == "True"
    assert len(list(tmp_path.glob("runs/test/seed_*/pair.json"))) == 3


def test_resume_reuses_only_exact_scene_options_code_and_environment(
    tmp_path,
    monkeypatch,
    isolated_batch,
):
    calls = []

    def runner(scene, directory, options):
        calls.append(options.seed)
        return _report()

    monkeypatch.setattr(experiments, "run_pipeline", runner)
    options = PipelineOptions(population=8, generations=2)
    experiments.run_experiments(["test"], [1], tmp_path, options)
    experiments.run_experiments(["test"], [1], tmp_path, options)
    assert calls == [1]
    experiments.run_experiments(["test"], [1], tmp_path, replace(options, population=12))
    assert calls == [1, 1]
    monkeypatch.setattr(experiments, "_source_hash", lambda: "changed-source")
    experiments.run_experiments(["test"], [1], tmp_path, replace(options, population=12))
    assert calls == [1, 1, 1]
    environment = experiments._environment()
    environment["python"] = "3.14.changed"
    monkeypatch.setattr(experiments, "_environment", lambda: environment)
    experiments.run_experiments(["test"], [1], tmp_path, replace(options, population=12))
    assert calls == [1, 1, 1, 1]
    experiments.run_experiments(
        ["test"], [1], tmp_path, replace(options, population=12), resume=False
    )
    assert calls == [1, 1, 1, 1, 1]


def test_hypervolume_uses_one_reference_for_all_seeds_and_retains_baseline():
    first, second = _report(), _report()
    second["solutions"] = [_solution([20.0, 1.0, 12.0])]
    pairs = [
        {"scene": "test", "seed": 1, "report": first},
        {"scene": "test", "seed": 2, "report": second},
    ]
    rows = [
        row
        for pair in pairs
        for row in experiments._paired_rows(pair["scene"], pair["seed"], pair["report"], ".")
    ]
    summary = experiments.summarize_experiments(pairs, rows)
    normalization = summary["normalization_by_scene"]["test"]
    np.testing.assert_allclose(normalization["ideal"], [2, 1, 4])
    np.testing.assert_allclose(normalization["nadir"], [20, 9, 12])
    assert normalization["reference_normalized"] == [1.1, 1.1, 1.1]
    assert rows[0]["hypervolume"] == rows[2]["hypervolume"]
    assert rows[1]["hypervolume"] >= rows[0]["hypervolume"]
    assert rows[3]["hypervolume"] >= rows[2]["hypervolume"]
    # Компромисс и крайние решения рассматриваются отдельно.
    selections = {item["selection"]: item for item in summary["paired"]}
    assert set(selections) == {
        "balanced",
        "min_effort",
        "min_jerk",
        "min_time",
        "front_and_runtime",
    }
    assert selections["balanced"]["changes_full_minus_baseline"]["J_tau"]["percent"]["median"] > 0
    assert (
        selections["min_effort"]["changes_full_minus_baseline"]["J_tau"]["percent"]["median"] <= 0
    )


def test_bootstrap_handles_zero_baseline_empty_samples_and_single_seed():
    result = experiments._paired_change([0, 2, 4], [1, 1, 2])
    assert result["percent"]["n"] == 2
    assert result["percent"]["undefined_zero_baseline"] == 1
    assert result["percent"]["median"] == -50
    assert result["percent"]["median_bootstrap_95_ci"] == [-50, -50]
    assert result["absolute"]["n"] == 3
    assert experiments._paired_change([], [])["percent"]["median"] is None
    assert experiments._paired_change([1], [2])["absolute"]["median_bootstrap_95_ci"] is None


def test_failed_attempts_keep_runtime_cost_in_summary():
    report = {
        "status": "failed",
        "failure_stage": "planning",
        "reason": "Исчерпан бюджет",
        "timings_seconds": {"planning": 12.0},
    }
    pairs = [{"scene": "test", "seed": 1, "report": report}]
    rows = experiments._paired_rows("test", 1, report, ".")
    summary = experiments.summarize_experiments(pairs, rows)
    assert all(row["runtime_numerical_s"] == 12 for row in rows)
    for item in summary["methods"]:
        assert item["runtime_all_attempts"]["median"] == 12
        assert item["metrics_successful_runs"]["runtime_numerical_s"]["n"] == 0
    assert experiments._flatten_summary(summary)[0]["runtime_all_attempts_median"] == 12


def test_statistics_artifacts_render_even_without_successful_runs(tmp_path):
    report = {"status": "failed", "failure_stage": "planning", "reason": "Исчерпан бюджет"}
    rows = experiments._paired_rows("test", 7, report, ".")
    summary = experiments.summarize_experiments(
        [{"scene": "test", "seed": 7, "report": report}], rows
    )
    experiments._create_report(summary, rows, tmp_path)
    assert (tmp_path / "statistics.png").stat().st_size > 10_000
    assert "0/1" in (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Парные эксперименты" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert all(item["success_rate"] == 0 for item in summary["methods"])
    assert summary["normalization_by_scene"]["test"] is None


def test_corrupt_cache_is_recomputed_and_failed_cache_is_retained(
    tmp_path,
    monkeypatch,
    isolated_batch,
):
    calls = []

    def runner(scene, directory, options):
        calls.append(options.seed)
        raise RuntimeError("RRT*: исчерпан бюджет")

    monkeypatch.setattr(experiments, "run_pipeline", runner)
    experiments.run_experiments(["test"], [1], tmp_path, PipelineOptions())
    experiments.run_experiments(["test"], [1], tmp_path, PipelineOptions())
    assert calls == [1]
    cache = tmp_path / "runs/test/seed_1/pair.json"
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert saved["report"]["status"] == "failed"
    for corrupt in ("{broken", "null", "[]", '{"completed":true,"report":null}'):
        cache.write_text(corrupt, encoding="utf-8")
        experiments.run_experiments(["test"], [1], tmp_path, PipelineOptions())
    assert calls == [1] * 5


def test_fingerprint_covers_scene_and_input_validation(tmp_path, isolated_batch):
    first = experiments._fingerprint({"obstacles": []}, PipelineOptions(), "source", {})
    changed = experiments._fingerprint({"obstacles": ["new"]}, PipelineOptions(), "source", {})
    assert first != changed
    for names, seeds in (
        ([], [1]),
        (["test"], []),
        (["test", "test"], [1]),
        (["test"], [1, 1]),
        (["test"], [-1]),
    ):
        with pytest.raises(ValueError):
            experiments.run_experiments(names, seeds, tmp_path, PipelineOptions())


def test_invalid_solutions_are_never_counted_as_success():
    report = copy.deepcopy(_report())
    report["solutions"][0]["validation"]["feasible"] = False
    rows = experiments._paired_rows("test", 1, report, ".")
    assert rows[0]["success"] and not rows[1]["success"]
    report["solutions"][0] = _solution([float("nan"), 2, 3])
    assert not experiments._paired_rows("test", 1, report, ".")[1]["success"]


def test_spawn_workers_read_separate_caches_and_parent_builds_one_summary(
    tmp_path,
    monkeypatch,
    isolated_batch,
):
    monkeypatch.setattr(experiments, "run_pipeline", lambda *args: _report())
    options = PipelineOptions(population=4, generations=1)
    expected = experiments.run_experiments(["test"], [1, 2], tmp_path, options)
    for seed in (1, 2):
        cache = tmp_path / f"runs/test/seed_{seed}/pair.json"
        pair = json.loads(cache.read_text(encoding="utf-8"))
        pair["fingerprint"] = experiments._fingerprint(
            {"name": "test", "obstacles": []},
            replace(options, seed=seed, plots=False),
            "fixed-source",
            experiments._environment(),
            2,
        )
        write_json(cache, pair)
    actual = experiments.run_experiments(["test"], [1, 2], tmp_path, options, workers=2)
    assert actual["methods"] == expected["methods"]
    assert actual["protocol"]["workers"] == 2
    assert actual["protocol"]["process_start_method"] == "spawn"
    with pytest.raises(ValueError):
        experiments.run_experiments(["test"], [1], tmp_path, options, workers=0)
