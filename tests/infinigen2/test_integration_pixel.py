# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "integration_v2"
sys.path.insert(0, str(_SCRIPTS))

import baseline_diff  # noqa: E402
import render_trajectory_video  # noqa: E402


def _write_run(root: Path, name: str, color: tuple[int, int, int]) -> Path:
    events = root / name / "render_index" / "events"
    events.mkdir(parents=True, exist_ok=True)

    still = "camera-traj0/Camera/0000.png"
    video = "camera-traj0/image_Camera.mp4"
    (root / name / still).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(root / name / still)
    (root / name / video).write_bytes(b"not an image")

    event = {
        "generator": "camera_monocular_in_bbox_rand",
        "variant_key": "workbench-traj0",
        "status": "success",
        "images": [still, video],
    }
    (events / "traj0.json").write_text(json.dumps(event))
    return root / name


def _write_renamed_run(
    root: Path,
    name: str,
    generator: str,
    color: tuple[int, int, int],
    include_asset_dir: bool = True,
) -> Path:
    run = root / name
    events = run / "render_index" / "events"
    events.mkdir(parents=True)
    asset_dir = f"object-{generator}-obj-cycles-0"
    image = f"{asset_dir}/Image.png"
    (run / asset_dir).mkdir()
    Image.new("RGB", (4, 4), color).save(run / image)
    event = {"generator": generator, "variant_key": "obj-cycles-0", "images": [image]}
    if include_asset_dir:
        event["asset_dir"] = asset_dir
    (events / "render.json").write_text(json.dumps(event))
    return run


def _write_alias_manifest(run: Path) -> None:
    manifest = [
        {
            "name": "pkg.table_circle_rand",
            "old_names": ["circle_table_rand"],
        }
    ]
    (run / "manifest.json").write_text(json.dumps(manifest))


def test_pixel_diff_skips_videos(tmp_path):
    base = _write_run(tmp_path, "base", (0, 0, 0))
    pr = _write_run(tmp_path, "pr", (0, 0, 0))
    report = baseline_diff.compare_pixel(pr, base)
    assert report["total"] == 1
    assert report["fail_count"] == 0
    assert report["results"][0]["image"].endswith(".png")


def test_pixel_diff_flags_changed_still(tmp_path):
    base = _write_run(tmp_path, "base", (0, 0, 0))
    pr = _write_run(tmp_path, "pr", (255, 255, 255))
    report = baseline_diff.compare_pixel(pr, base)
    assert report["fail_count"] == 1


def test_pixel_diff_pairs_identical_renamed_generator(tmp_path: Path) -> None:
    base = _write_renamed_run(tmp_path, "base", "circle_table_rand", (24, 48, 72))
    pr = _write_renamed_run(tmp_path, "pr", "table_circle_rand", (24, 48, 72))
    _write_alias_manifest(pr)

    report = baseline_diff.compare_pixel(pr, base)

    assert report["fail_count"] == 0
    assert report["missing_count"] == 0
    assert report["results"][0]["asset"] == "table_circle_rand"
    assert "object-table_circle_rand" in report["results"][0]["image"]
    assert "object-circle_table_rand" in report["results"][0]["baseline_image"]


def test_pixel_diff_flags_changed_renamed_generator(tmp_path: Path) -> None:
    base = _write_renamed_run(tmp_path, "base", "circle_table_rand", (0, 0, 0))
    pr = _write_renamed_run(tmp_path, "pr", "table_circle_rand", (255, 255, 255))
    _write_alias_manifest(pr)

    report = baseline_diff.compare_pixel(pr, base)

    assert report["fail_count"] == 1
    assert report["missing_count"] == 0
    assert report["results"][0]["asset"] == "table_circle_rand"


def _write_camera_run(root: Path, name: str, scene: str) -> Path:
    run = root / name
    events = run / "render_index" / "events"
    events.mkdir(parents=True)
    asset_dir = f"camera-{scene}-workbench-traj0"
    image = f"{asset_dir}/0000.png"
    (run / asset_dir).mkdir()
    Image.new("RGB", (4, 4), (24, 48, 72)).save(run / image)
    event = {
        "generator": "camera_monocular_in_bbox_rand",
        "variant_key": f"{scene}-workbench-traj0",
        "asset_dir": asset_dir,
        "images": [image],
    }
    (events / "render.json").write_text(json.dumps(event))
    return run


def test_pixel_diff_pairs_renamed_generator_inside_variant(tmp_path: Path) -> None:
    base = _write_camera_run(tmp_path, "base", "circle_table_rand")
    pr = _write_camera_run(tmp_path, "pr", "table_circle_rand")
    _write_alias_manifest(pr)

    report = baseline_diff.compare_pixel(pr, base)

    assert report["missing_count"] == 0
    assert report["fail_count"] == 0


def test_pixel_diff_without_alias_keeps_literal_paths(tmp_path: Path) -> None:
    base = _write_renamed_run(tmp_path, "base", "circle_table_rand", (24, 48, 72))
    pr = _write_renamed_run(tmp_path, "pr", "table_circle_rand", (24, 48, 72))

    report = baseline_diff.compare_pixel(pr, base)

    assert report["missing_count"] == 1


def test_pixel_diff_pairs_single_image_legacy_events(tmp_path: Path) -> None:
    base = _write_renamed_run(
        tmp_path, "base", "circle_table_rand", (24, 48, 72), False
    )
    pr = _write_renamed_run(tmp_path, "pr", "table_circle_rand", (24, 48, 72), False)
    _write_alias_manifest(pr)

    report = baseline_diff.compare_pixel(pr, base)

    assert report["missing_count"] == 0


def test_camera_render_uses_cli_script_under_coverage(tmp_path, monkeypatch):
    args = render_trajectory_video.argparse.Namespace(
        output=tmp_path,
        scene="room_livingroom_rand",
        camera="orbit_rand",
        seed=0,
        frames=[0, 47],
        resolution=[640, 360],
    )
    captured = {}

    def fake_run(cmd):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setenv("INFINIGEN_COVERAGE", "1")
    monkeypatch.setattr(render_trajectory_video.subprocess, "run", fake_run)

    assert render_trajectory_video.render_frames(args) == 0
    assert captured["cmd"][:6] == [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--parallel-mode",
        "--rcfile=pyproject.toml",
    ]
    assert Path(captured["cmd"][6]).name == "infinigen2"
