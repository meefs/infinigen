# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import procfunc as pf
import pytest

from infinigen2.scenes import indoor_space
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.room import bathroom_setup


def test_sink_wall_row_passes_room_box(
    monkeypatch: pytest.MonkeyPatch, rng: pf.RNG
) -> None:
    captured_box: list[tuple[pf.Vector | None, pf.Vector | None]] = []

    def capture_setup(
        _rng: pf.RNG,
        bbox_min: pf.Vector | None = None,
        bbox_max: pf.Vector | None = None,
    ) -> bathroom_setup.BathroomSinkSetupResult:
        captured_box.append((bbox_min, bbox_max))
        return bathroom_setup.BathroomSinkSetupResult([], [], [], [], [], [], [])

    def capture_row(
        _rng: pf.RNG,
        _unit: list[pf.MeshObject],
        _room_dimensions: pf.Vector,
        _colliders: ccol.CollisionSet,
        _wall_colliders: ccol.CollisionSet,
        yaw: float,
    ) -> indoor_space.WallRowResult:
        return indoor_space.WallRowResult([], yaw)

    monkeypatch.setattr(indoor_space, "bathroom_sink_setup_rand", capture_setup)
    monkeypatch.setattr(indoor_space, "_wall_row_of_unit_rand", capture_row)
    dimensions = pf.Vector((6.0, 5.0, 3.0))
    colliders = ccol.collision_set([])

    indoor_space.sink_wall_row_rand(rng, dimensions, colliders, colliders, 0.0)

    assert captured_box == [(pf.Vector((0.0, 0.0, 0.0)), dimensions)]
