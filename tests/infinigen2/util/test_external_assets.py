# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from pathlib import Path

from infinigen2.util import external_assets


def test_resolve_asset_paths_with_glob(tmp_path: Path):
    (tmp_path / "forks").mkdir()
    (tmp_path / "spoons").mkdir()
    (tmp_path / "forks" / "a.obj").write_text("o test\n")
    (tmp_path / "spoons" / "b.fbx").write_text("fake")
    (tmp_path / "spoons" / "ignore.txt").write_text("ignore")

    pattern = Path(str(tmp_path / "*" / "*"))
    paths = external_assets._resolve_asset_paths(pattern)
    assert [p.name for p in paths] == ["a.obj", "b.fbx"]
