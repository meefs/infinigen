# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json
from collections.abc import Iterable
from pathlib import Path


def parse_ci_compare_aliases(manifest: object) -> dict[str, str]:
    if not isinstance(manifest, list):
        raise ValueError("manifest must be a list")

    current_names: set[str] = set()
    entries: list[tuple[str, object]] = []
    for entry in manifest:
        if not isinstance(entry, dict):
            raise ValueError("manifest entries must be objects")
        qualified_name = entry.get("name")
        if not isinstance(qualified_name, str) or not qualified_name:
            raise ValueError("manifest entries must have a non-empty string name")
        current_name = qualified_name.rsplit(".", 1)[-1]
        if current_name in current_names:
            raise ValueError(f"duplicate manifest shortname: {current_name}")
        current_names.add(current_name)
        if "old_names" in entry:
            entries.append((current_name, entry["old_names"]))

    aliases: dict[str, str] = {}
    for current_name, old_names in entries:
        if not isinstance(old_names, list):
            raise ValueError(f"old_names for {current_name} must be a list")
        for old_name in old_names:
            if not isinstance(old_name, str) or not old_name:
                raise ValueError(
                    f"old_names for {current_name} must contain non-empty strings"
                )
            if old_name == current_name:
                raise ValueError(f"{current_name} cannot alias itself")
            if old_name in current_names:
                raise ValueError(
                    f"old name {old_name} conflicts with a current manifest shortname"
                )
            previous = aliases.get(old_name)
            if previous is not None:
                raise ValueError(
                    f"old name {old_name} maps to both {previous} and {current_name}"
                )
            aliases[old_name] = current_name
    return aliases


def load_ci_compare_aliases(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read comparison aliases from {manifest_path}"
        ) from exc
    return parse_ci_compare_aliases(manifest)


def archive_aliases(aliases: dict[str, str], names: Iterable[str]) -> dict[str, str]:
    # An archive that rendered both an old name and its successor keeps both rows.
    present = set(names)
    applied = {
        old: new
        for old, new in aliases.items()
        if old in present and new not in present
    }
    sources: dict[str, str] = {}
    for old, new in applied.items():
        if new in sources:
            raise ValueError(f"{sources[new]} and {old} both alias {new}")
        sources[new] = old
    return applied


def canonical_variant(variant: str, aliases: dict[str, str]) -> str:
    return "-".join(aliases.get(part, part) for part in variant.split("-"))
