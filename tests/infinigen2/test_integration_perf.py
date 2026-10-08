# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "integration_v2"
sys.path.insert(0, str(_SCRIPTS))

import baseline_diff  # noqa: E402
from display import (  # noqa: E402
    build_comparison_data,
    build_version_totals,
    collect_images_structured,
)


def _write_event(
    root: Path,
    name: str,
    variant: str,
    base_tris: int,
    subdiv_tris: int,
    cpu: float,
    gpu: float,
    legacy: bool = False,
    generator: str = "chair_rand",
) -> None:
    events = root / name / "render_index" / "events"
    events.mkdir(parents=True, exist_ok=True)
    img = f"{name}/object-{generator}-obj-cycles-{variant}/camera-0/0001.png"
    (root / name / img).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(root / name / img)
    event = {
        "generator": generator,
        "asset_type": "object",
        "variant_key": f"obj-cycles-{variant}",
        "status": "success",
        "cmd": ["infinigen", "chair_rand"],
        "images": [img],
        "duration_sec": cpu + gpu,
        "cpu_time_sec": cpu,
        "gpu_time_sec": gpu,
    }
    if legacy:
        event["tris"] = subdiv_tris
    else:
        event["base_tris"] = base_tris
        event["subdiv_tris"] = subdiv_tris
    (events / f"{variant}.json").write_text(json.dumps(event))


def _write_run(
    root: Path,
    name: str,
    base_tris: int,
    subdiv_tris: int,
    cpu: float,
    gpu: float,
    legacy: bool = False,
    generator: str = "chair_rand",
) -> Path:
    for variant in ("0", "1"):
        _write_event(
            root, name, variant, base_tris, subdiv_tris, cpu, gpu, legacy, generator
        )
    return root / name


def _write_alias_manifest(run: Path) -> None:
    manifest = [
        {
            "name": "pkg.table_circle_rand",
            "old_names": ["circle_table_rand"],
        }
    ]
    (run / "manifest.json").write_text(json.dumps(manifest))


def test_perf_gate_flags_subdiv_tris_regression(tmp_path):
    base = _write_run(tmp_path, "base", 3000, 48000, cpu=12.0, gpu=4.0)
    pr = _write_run(tmp_path, "pr", 3000, 57600, cpu=12.0, gpu=4.0)  # +20% subdiv tris
    report = baseline_diff.compare(pr, base, threshold=0.05)
    assert report["fail_count"] == 1
    assert report["results"][0]["status"] == "fail"
    assert "subdiv_tris" in report["results"][0]["regressions"]
    assert baseline_diff.asset_verdicts(report)["chair_rand"] == "changed"


# A coarser cage at a higher subdiv level leaves the evaluated count flat.
def test_perf_gate_flags_base_tris_regression_alone(tmp_path):
    base = _write_run(tmp_path, "base", 3000, 48000, cpu=12.0, gpu=4.0)
    pr = _write_run(tmp_path, "pr", 3600, 48000, cpu=12.0, gpu=4.0)
    report = baseline_diff.compare(pr, base, threshold=0.05)
    assert report["fail_count"] == 1
    assert list(report["results"][0]["regressions"]) == ["base_tris"]


def test_perf_gate_no_regression_when_within_threshold(tmp_path):
    base = _write_run(tmp_path, "base", 3000, 48000, cpu=12.0, gpu=4.0)
    pr = _write_run(tmp_path, "pr", 3050, 49000, cpu=12.4, gpu=4.1)  # <5% everywhere
    report = baseline_diff.compare(pr, base, threshold=0.05)
    assert report["fail_count"] == 0


@pytest.mark.parametrize(
    ("pr_subdiv_tris", "expected_status"),
    [(48000, "ok"), (57600, "fail")],
)
def test_perf_gate_compares_renamed_generator(
    tmp_path: Path, pr_subdiv_tris: int, expected_status: str
) -> None:
    base = _write_run(
        tmp_path, "base", 3000, 48000, 12.0, 4.0, generator="circle_table_rand"
    )
    pr = _write_run(
        tmp_path,
        "pr",
        3000,
        pr_subdiv_tris,
        12.0,
        4.0,
        generator="table_circle_rand",
    )
    _write_alias_manifest(pr)

    report = baseline_diff.compare(pr, base, threshold=0.05)

    assert report["missing_count"] == 0
    assert report["results"][0]["asset"] == "table_circle_rand"
    assert report["results"][0]["status"] == expected_status


def test_perf_annotation_uses_canonical_row_name(tmp_path: Path) -> None:
    base = _write_run(
        tmp_path, "base", 3000, 48000, 12.0, 4.0, generator="circle_table_rand"
    )
    pr = _write_run(
        tmp_path, "pr", 3000, 57600, 12.0, 4.0, generator="table_circle_rand"
    )
    _write_alias_manifest(pr)
    rows = [{"asset": "table_circle_rand"}]

    baseline_diff.annotate_rows(rows, pr, base, threshold=0.05)

    assert rows[0]["perf_regressed"] is True
    assert rows[0]["perf_summary"] == "subdiv +20%"


def test_perf_gate_prefers_baseline_current_name_over_alias(tmp_path: Path) -> None:
    base = _write_run(
        tmp_path, "base", 3000, 99000, 12.0, 4.0, generator="circle_table_rand"
    )
    _write_event(
        tmp_path, "base", "2", 3000, 48000, 12.0, 4.0, generator="table_circle_rand"
    )
    pr = _write_run(
        tmp_path, "pr", 3000, 48000, 12.0, 4.0, generator="table_circle_rand"
    )
    _write_alias_manifest(pr)

    report = baseline_diff.compare(pr, base, threshold=0.05)

    assert [r["asset"] for r in report["results"]] == ["table_circle_rand"]
    assert report["results"][0]["status"] == "ok"


# A lone noisy sample (+900% cpu) shouldn't dominate the asset's delta.
def test_perf_gate_ignores_single_noisy_sample(tmp_path):
    for variant, cpu in (("0", 12.5), ("1", 12.4), ("2", 120.0)):
        _write_event(tmp_path, "base", variant, 3000, 48000, cpu=12.0, gpu=4.0)
        _write_event(tmp_path, "pr", variant, 3000, 48000, cpu=cpu, gpu=4.0)
    report = baseline_diff.compare(tmp_path / "pr", tmp_path / "base", threshold=0.05)
    assert report["fail_count"] == 0
    assert report["results"][0]["status"] == "ok"


def test_display_marks_after_worse_and_better(tmp_path):
    _write_run(tmp_path, "base", 3000, 48000, cpu=12.0, gpu=4.0)
    # subdiv tris worse, base cage better, cpu better
    _write_run(tmp_path, "pr", 2400, 57600, cpu=10.0, gpu=4.0)
    names = ["base", "pr"]
    results = [
        collect_images_structured(tmp_path / "base", "base"),
        collect_images_structured(tmp_path / "pr", "pr"),
    ]
    rows = build_comparison_data(results, names)
    after = next(o for o in rows[0]["objects"] if o["version"] == "pr")
    before = next(o for o in rows[0]["objects"] if o["version"] == "base")
    assert after["metrics"]["subdiv_tris_cls"] == "worse"
    assert after["metrics"]["base_tris_cls"] == "better"
    assert after["metrics"]["cpu_cls"] == "better"
    assert before["metrics"]["subdiv_tris_cls"] == "same"

    totals = build_version_totals(rows, names)
    assert totals["pr"]["subdiv_tris_cls"] == "worse"
    assert totals["pr"]["base_tris_cls"] == "better"
    assert totals["base"]["subdiv_tris_cls"] == "same"


# Archives written before the split only carry "tris"; it is the subdivided count.
def test_legacy_tris_events_still_gate_and_display(tmp_path):
    base = _write_run(tmp_path, "base", 0, 48000, cpu=12.0, gpu=4.0, legacy=True)
    pr = _write_run(tmp_path, "pr", 0, 57600, cpu=12.0, gpu=4.0, legacy=True)
    report = baseline_diff.compare(pr, base, threshold=0.05)
    assert list(report["results"][0]["regressions"]) == ["subdiv_tris"]

    names = ["base", "pr"]
    results = [
        collect_images_structured(tmp_path / "base", "base"),
        collect_images_structured(tmp_path / "pr", "pr"),
    ]
    rows = build_comparison_data(results, names)
    after = next(o for o in rows[0]["objects"] if o["version"] == "pr")
    assert after["metrics"]["subdiv_tris"] == "57.6k"
    assert after["metrics"]["base_tris"] is None
