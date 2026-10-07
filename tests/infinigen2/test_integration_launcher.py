import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "integration_v2"
sys.path.insert(0, str(_SCRIPTS))

import launch_andromeda  # noqa: E402


def _write_fake_uv(path: Path) -> None:
    path.write_text(
        '#!/bin/sh\nprintf "%s\\0" "$@" > "$UV_CALLS.$$.txt"\n'
        'case "$*" in *preset_parents*) printf "{}" ;; esac\n'
    )
    path.chmod(0o755)


def test_launch_shell_disables_uv_sync(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_uv(fake_bin / "uv")
    calls = tmp_path / "uv-calls.txt"
    categories = {
        name: ""
        for name in (
            "OBJECTS",
            "MASKS",
            "DISPLACEMENTS",
            "PRESETS",
            "ENVIRONMENTS",
            "CAMERAS",
            "SCENES",
        )
    }
    env = os.environ | categories
    env |= {
        "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
        "UV_CALLS": str(calls),
        "MATERIALS": "example_material",
        "INTEGRATION_SLOT_INDEX": "1",
    }

    result = subprocess.run(
        ["bash", "scripts/integration_v2/launch.sh", str(tmp_path / "out"), "1"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    uv_calls = [
        path.read_bytes().split(b"\0")[:2] for path in tmp_path.glob("uv-calls.*.txt")
    ]
    assert uv_calls
    assert all(call == [b"run", b"--no-sync"] for call in uv_calls)


def test_launch_shell_plans_full_house_tour(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_uv(fake_bin / "uv")
    calls = tmp_path / "uv-calls.txt"
    categories = {
        name: ""
        for name in (
            "MATERIALS",
            "OBJECTS",
            "MASKS",
            "DISPLACEMENTS",
            "PRESETS",
            "ENVIRONMENTS",
            "CAMERAS",
            "SCENES",
        )
    }
    env = os.environ | categories
    env |= {
        "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
        "UV_CALLS": str(calls),
        "INTEGRATION_SLOT_INDEX": "0",
    }

    result = subprocess.run(
        ["bash", "scripts/integration_v2/launch.sh", str(tmp_path / "out"), "1"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    uv_calls = [
        path.read_bytes().split(b"\0") for path in tmp_path.glob("uv-calls.*.txt")
    ]
    call = next(call for call in uv_calls if b"examples/house_tour/render.py" in call)
    frame_index = call.index(b"--frames")
    render_index = call.index(b"--render_frames")
    assert call[frame_index + 1 : frame_index + 3] == [b"0", b"239"]
    assert call[render_index + 1 : render_index + 3] == [b"0", b"3"]


def _scene_outputs_for_slot(tmp_path: Path, slot_idx: int, slot_count: int) -> set:
    repo = Path(__file__).resolve().parents[2]
    slot_dir = tmp_path / f"slot{slot_idx}"
    fake_bin = slot_dir / "bin"
    fake_bin.mkdir(parents=True)
    _write_fake_uv(fake_bin / "uv")
    categories = {
        name: ""
        for name in (
            "MATERIALS",
            "OBJECTS",
            "MASKS",
            "DISPLACEMENTS",
            "PRESETS",
            "ENVIRONMENTS",
            "CAMERAS",
        )
    }
    env = os.environ | categories
    env |= {
        "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
        "UV_CALLS": str(slot_dir / "uv-calls.txt"),
        "SCENES": "slow_rand\nfast_rand",
        "SCENE_CMDS": "slow_rand\x1f\x1f3\nfast_rand\x1f\x1f2\nskipped_rand\x1f\x1f2",
        "INTEGRATION_SLOT_INDEX": str(slot_idx),
        "INTEGRATION_SLOT_COUNT": str(slot_count),
    }

    result = subprocess.run(
        ["bash", "scripts/integration_v2/launch.sh", str(slot_dir / "out"), "1"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = [path.read_bytes().split(b"\0") for path in slot_dir.glob("uv-calls.*")]
    outputs = [
        call[call.index(b"--output") + 1] for call in calls if b"--output" in call
    ]
    return {Path(o.decode()).name for o in outputs if b"/scene-" in o}


def test_launch_shell_spreads_scene_seeds_across_slots(tmp_path: Path) -> None:
    per_slot = [_scene_outputs_for_slot(tmp_path, idx, 3) for idx in range(3)]

    assert per_slot[0] == {
        "scene-slow_rand-demo-cycles-0",
        "scene-fast_rand-demo-cycles-0",
    }
    assert per_slot[1] == {
        "scene-slow_rand-demo-cycles-1",
        "scene-fast_rand-demo-cycles-1",
    }
    assert per_slot[2] == {"scene-slow_rand-demo-cycles-2"}


def test_launcher_marks_each_render_slot(monkeypatch, tmp_path):
    launched = []

    class Popen:
        def __init__(self, cmd, env, text):
            launched.append(env["INTEGRATION_SLOT_INDEX"])

        def wait(self):
            return 0

    monkeypatch.setattr(
        sys, "argv", ["launch_andromeda.py", "--output_path", str(tmp_path)]
    )
    monkeypatch.setattr(launch_andromeda, "list_items", lambda *args: [])
    monkeypatch.setattr(launch_andromeda, "resolve_gpu_ids", lambda gpus: ["0", "1"])
    monkeypatch.setattr(launch_andromeda, "render_runner", lambda output_path: "runner")
    monkeypatch.setattr(launch_andromeda.subprocess, "Popen", Popen)
    monkeypatch.setattr(
        launch_andromeda, "failed_render_names", lambda output_path: ([], [])
    )

    assert launch_andromeda.main() == 0
    assert launched == ["0", "1"]
