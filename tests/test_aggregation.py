"""Объединение серий не должно подменять fingerprints или скрывать вычислительные отказы."""

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from manipulator_planning import aggregation
from manipulator_planning.experiments import pipeline_calculation_hash
from manipulator_planning.io import write_json
from manipulator_planning.pipeline import PipelineOptions


@pytest.fixture
def series(tmp_path, monkeypatch):
    """Создать две сцены, обновлённую сцену и повтор ошибки записи без запуска физики."""
    hashes = {"planning.py": "unchanged"}
    monkeypatch.setattr(aggregation, "calculation_hashes", lambda: hashes)
    monkeypatch.setattr(
        aggregation, "load_named_scene", lambda name: {"name": name, "obstacles": []}
    )
    monkeypatch.setattr(aggregation, "_create_report", lambda *args: None)

    def create(name, scene_names, seeds, workers=2):
        root = tmp_path / name
        root.mkdir()
        options = PipelineOptions(population=4, generations=1, plots=False)
        summary = {
            "protocol_version": 1,
            "scenes": scene_names,
            "seeds": seeds,
            "options": asdict(options),
            "environment": {"python": "3.14", "packages": {}},
            "source_hash": f"source-{name}",
            "calculation_sources": hashes,
            "protocol": {"workers": workers},
        }
        write_json(root / "summary.json", summary)
        for scene_name in scene_names:
            for seed in seeds:
                directory = root / "runs" / scene_name / f"seed_{seed}"
                directory.mkdir(parents=True)
                solution = {"objectives": [4.0, 2.0, 3.0], "validation": {"feasible": True}}
                report = {
                    "status": "success",
                    "options": asdict(replace(options, seed=seed)),
                    "baseline": solution,
                    "solutions": [solution],
                    "selected_index": 0,
                    "timings_seconds": {
                        "planning": 1.0,
                        "smoothing": 1.0,
                        "optimization_and_validation": 2.0,
                    },
                }
                pair = {
                    "protocol_version": 1,
                    "completed": True,
                    "fingerprint": f"fingerprint-{name}-{scene_name}-{seed}",
                    "scene": scene_name,
                    "seed": seed,
                    "report": report,
                }
                write_json(directory / "pair.json", pair)
                write_json(directory / "report.json", report)
                (directory / "config.yaml").write_text(
                    yaml.safe_dump({"name": scene_name, "obstacles": []}), encoding="utf-8"
                )
                (directory / "trajectories.npz").write_bytes(b"unchanged numeric artifact")
        return root

    old = create("old", ["a", "b"], [0, 1])
    new = create("new", ["a"], [0, 1])
    recovery = create("recovery", ["b"], [1], workers=1)
    path = old / "runs/b/seed_1/pair.json"
    pair = json.loads(path.read_text(encoding="utf-8"))
    pair["report"].update(status="failed", failure_stage="artifacts", reason="WinError 5")
    write_json(path, pair)
    return old, new, recovery, tmp_path / "combined"


def test_combination_preserves_original_pairs_and_records_replaced_operational_failure(series):
    old, new, recovery, target = series
    original_failure = (old / "runs/b/seed_1/pair.json").read_bytes()
    result = aggregation.combine_experiments(
        {"a": new, "b": old},
        target,
        replacements={("b", 1): recovery},
        seeds=[0, 1],
    )
    assert all(item["succeeded"] == 2 for item in result["methods"])
    assert result["source_hash"] is None
    assert result["aggregation"]["physical_attempts"] == 5
    assert result["protocol"]["worker_counts"] == [1, 2]
    assert (target / "runs/a/seed_0/pair.json").read_bytes() == (
        new / "runs/a/seed_0/pair.json"
    ).read_bytes()
    assert (target / "runs/b/seed_1/pair.json").read_bytes() == (
        recovery / "runs/b/seed_1/pair.json"
    ).read_bytes()
    assert (old / "runs/b/seed_1/pair.json").read_bytes() == original_failure
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    entry = next(item for item in provenance["pairs"] if item["scene"] == "b" and item["seed"] == 1)
    assert entry["original_fingerprint"] == "fingerprint-recovery-b-1"
    assert entry["superseded_attempt"]["failure_stage"] == "artifacts"
    with (target / "runs.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8 and rows[-1]["execution_workers"] == "1"
    # Повторное явное объединение идемпотентно и не меняет скопированный кеш.
    aggregation.combine_experiments({"a": new, "b": old}, target, replacements={("b", 1): recovery})


@pytest.mark.parametrize(
    "field", ["options", "environment", "calculation_sources", "seed", "scene"]
)
def test_incompatible_sources_are_rejected_before_copying(series, field):
    old, new, recovery, target = series
    if field in {"options", "environment", "calculation_sources"}:
        path = new / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if field == "options":
            summary[field]["generations"] = 999
        elif field == "environment":
            summary[field]["python"] = "other"
        else:
            summary[field] = {"planning.py": "changed"}
        write_json(path, summary)
    elif field == "seed":
        path = new / "runs/a/seed_1/pair.json"
        pair = json.loads(path.read_text(encoding="utf-8"))
        pair["report"]["options"]["seed"] = 99
        write_json(path, pair)
    else:
        (new / "runs/a/seed_1/config.yaml").write_text("name: obsolete\n", encoding="utf-8")
    with pytest.raises(ValueError):
        aggregation.combine_experiments({"a": new, "b": old}, target)
    assert not target.exists()


@pytest.mark.parametrize("stage", [None, "planning", "smoothing", "optimization"])
def test_operational_recovery_cannot_replace_numerical_failure_or_success(series, stage):
    old, new, recovery, target = series
    path = old / "runs/b/seed_1/pair.json"
    pair = json.loads(path.read_text(encoding="utf-8"))
    pair["report"].update(status="failed" if stage else "success", failure_stage=stage)
    write_json(path, pair)
    with pytest.raises(ValueError, match="Операционная замена"):
        aggregation.combine_experiments(
            {"a": new, "b": old}, target, replacements={("b", 1): recovery}
        )
    assert not target.exists()


def test_old_series_require_explicit_recorded_compatibility_assumption(series, monkeypatch):
    old, new, recovery, target = series
    path = old / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    del summary["calculation_sources"]
    write_json(path, summary)
    with pytest.raises(ValueError, match="контрольная ревизия"):
        aggregation.combine_experiments({"b": old}, target)
    monkeypatch.setattr(
        aggregation, "_reference_hashes", lambda revision: {"planning.py": "unchanged"}
    )
    result = aggregation.combine_experiments({"b": old}, target, reference_revision="123abcd")
    assert result["aggregation"]["original_fingerprints_preserved"]
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["reference_revision_assumption"] == "123abcd"
    assert "предпосылка" in provenance["origins"][str(old)]["numerical_compatibility_basis"]


def test_existing_different_pair_and_overlapping_output_are_rejected(series):
    old, new, recovery, target = series
    with pytest.raises(ValueError, match="пересекаться"):
        aggregation.combine_experiments({"a": old}, old / "merged")
    conflicting = target / "runs/a/seed_0"
    conflicting.mkdir(parents=True)
    (conflicting / "pair.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="другую пару"):
        aggregation.combine_experiments({"a": old}, target)


def test_pipeline_compatibility_hash_ignores_rendering_but_detects_changed_planning():
    source = Path(aggregation.__file__).with_name("pipeline.py").read_bytes()
    original = pipeline_calculation_hash(source)
    rendered = source.replace(b"assets = create_report(", b"assets = different_report(")
    assert rendered != source and pipeline_calculation_hash(rendered) == original
    changed = source.replace(b"step_size=options.rrt_step_size", b"step_size=0.123")
    assert changed != source and pipeline_calculation_hash(changed) != original
