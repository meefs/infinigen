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
