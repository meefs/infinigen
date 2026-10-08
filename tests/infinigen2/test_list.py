import sys

import pandas as pd

from infinigen2 import list as generator_list


def test_manifest_presets_are_plain_names() -> None:
    for record in generator_list.GENERATORS_MANIFEST.to_dict("records"):
        presets = record.get("presets")
        if isinstance(presets, list):
            assert all(isinstance(preset, str) for preset in presets)


def test_list_sorts_by_category_then_shortname(monkeypatch, capsys) -> None:
    manifest = pd.DataFrame(
        [
            {"category": "Object", "name": "pkg.zebra_rand"},
            {"category": "Material", "name": "pkg.blue_rand"},
            {"category": "Object", "name": "pkg.apple_rand"},
        ]
    )
    monkeypatch.setattr(generator_list, "GENERATORS_MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", ["infinigen2.list", "--columns", "shortname"])

    generator_list._main()

    assert capsys.readouterr().out.splitlines() == [
        "blue_rand",
        "apple_rand",
        "zebra_rand",
    ]
