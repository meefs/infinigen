"""Derive the CLI command that reproduces a given example render.

Mirrors the per-category invocations in scripts/integration_v2/launch.sh,
dropping the run-specific --output flag.
"""

import json
from pathlib import Path

_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "infinigen2" / "manifest.json"
)

_DEFAULT_PIPELINES = {
    "Material": "{short} material_cube render_cycles",
    "Mask": "{short} material_plane_uv render_cycles",
    "Displacement": "{short} material_cube render_cycles",
    "Object": "{short} object_demo render_cycles",
    "Scene": "{short} render_cycles",
    "Environment": "{short} material_monkey render_cycles",
}

_OPTIONS = {
    "Material": "--passes rgb --displacement_mode DISPLACEMENT -r 192 192 -s 128",
    "Mask": "--passes rgb -r 384 384 -s 128",
    "Displacement": "--passes rgb --displacement_mode DISPLACEMENT -r 384 384 -s 128",
    "Object": "--passes rgb -r 512 512 -s 128",
    "Scene": "--passes rgb -r 480 480 -s 256",
    "Environment": "--passes rgb -r 512 512 -s 128",
}


def _manifest_commands() -> dict[str, str]:
    if not _MANIFEST_PATH.exists():
        return {}
    entries = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    commands = {}
    for entry in entries:
        name = entry.get("name")
        command = entry.get("integration_test_string")
        if not name or not command:
            continue
        commands[name] = command
        command_parts = command.split(maxsplit=1)
        if len(command_parts) != 2:
            continue
        module = name.rsplit(".", 1)[0]
        for preset in entry.get("presets", []):
            commands[f"{module}.{preset}"] = f"{preset} {command_parts[1]}"
    return commands


_MANIFEST_COMMANDS = _manifest_commands()


def pipeline(category: str, name: str) -> str | None:
    pipeline_template = _DEFAULT_PIPELINES.get(category)
    if pipeline_template is None:
        return None
    short = name.rsplit(".", 1)[-1]
    return _MANIFEST_COMMANDS.get(name, pipeline_template.format(short=short))


def archive_variant(category: str, name: str) -> str | None:
    command = pipeline(category, name)
    if command is None:
        return None
    parts = command.split()
    if len(parts) < 3:
        return None
    demo = parts[1].removeprefix("material_").replace("_", "")
    renderer = parts[2].removeprefix("render_")
    return f"{demo}-{renderer}"


def replicate_command(category: str, name: str, seed: int) -> str | None:
    command = pipeline(category, name)
    options = _OPTIONS.get(category)
    if command is None or options is None:
        return None
    return f"infinigen2 {command} --seed {seed} {options}"


if __name__ == "__main__":
    print(
        replicate_command(
            "Material", "infinigen2.shaders.composites.bricks.bricks_rand", 0
        )
    )
    print(replicate_command("Mask", "infinigen2.shaders.masks.cracks.cracks_rand", 3))
    print(
        replicate_command(
            "Object",
            "infinigen2.objects.random_primitives.primitive_with_effect_rand",
            0,
        )
    )
    print(
        replicate_command("Scene", "infinigen2.scenes.demo_material.material_sphere", 5)
    )
    print(
        replicate_command(
            "Exporter", "infinigen2.exporters.render_cycles.render_cycles", 0
        )
    )
