"""Галерея выбирает успешные запуски и сохраняет происхождение их чисел."""

import json

from manipulator_planning.showcase import _RESULT_FILES, create_showcase


def test_gallery_uses_successful_seed_and_preserves_saved_spline(tmp_path, monkeypatch):
    series = tmp_path / "series"
    series.mkdir()
    (series / "summary.json").write_text(
        json.dumps({"scenes": ["gate", "simple"], "seeds": [0, 1]})
    )
    for name, seed, success in (
        ("gate", 0, False),
        ("gate", 1, True),
        ("gate", 99, True),
        ("simple", 0, True),
    ):
        run = series / "runs" / name / f"seed_{seed}"
        run.mkdir(parents=True)
        pair = {
            "completed": True,
            "scene": name,
            "seed": seed,
            "fingerprint": f"{name}-{seed}",
            "report": {"status": "success" if success else "failed"},
        }
        (run / "pair.json").write_text(json.dumps(pair))
        for filename in _RESULT_FILES:
            (run / filename).write_text(f"{name}-{seed}-{filename}")
    rendered = []

    def replay(directory, *, views, frame_count):
        rendered.append((directory.name, views, frame_count))
        (directory / "variants.csv").write_text("variant,duration_s\nbalanced,4\n")

    monkeypatch.setattr("manipulator_planning.showcase.replay_results", replay)
    output = tmp_path / "gallery"
    # Предпочтительный seed 99 существует на диске, но отсутствует в текущей серии.
    result = create_showcase(series, output, frame_count=17, preferred_seed=99)
    assert [(item["scene"], item["seed"]) for item in result["scenes"]] == [
        ("gate", 1),
        ("simple", 0),
    ]
    assert rendered == [("gate", ("isometric", "top"), 17), ("simple", ("isometric", "top"), 17)]
    assert (output / "gate" / "trajectories.json").read_text() == "gate-1-trajectories.json"
    assert (output / "gate" / "origin.json").exists()
    assert "gate/view.html" in (output / "index.html").read_text(encoding="utf-8")
    assert "../series/index.html" in (output / "index.html").read_text(encoding="utf-8")
    assert len((output / "variants.csv").read_text(encoding="utf-8-sig").splitlines()) == 3
