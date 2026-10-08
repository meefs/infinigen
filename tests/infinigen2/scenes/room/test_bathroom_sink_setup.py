# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bpy
import numpy as np
import procfunc as pf
import pytest

from infinigen2.scenes import indoor_space
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.room import bathroom_setup


@pytest.mark.parametrize("seed", [0, 7])
def test_bathroom_sink_setup_owns_generated_objects(seed: int) -> None:
    original_objects = set(bpy.context.scene.objects)
    result = bathroom_setup.bathroom_sink_setup_rand(np.random.default_rng(seed))

    generated_objects = set(bpy.context.scene.objects) - original_objects
    owned = {obj.item() for obj in result.all_objects}
    assert result.bathroom_sinks
    assert len(owned) == len(result.all_objects)
    assert owned == generated_objects


def test_sink_wall_row_generates_and_cleans_templates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(indoor_space, "back_face_grounded", lambda *_: True)
    original_objects = set(bpy.context.scene.objects)
    row = indoor_space.sink_wall_row_rand(
        np.random.default_rng(3),
        room_dimensions=pf.Vector((5.0, 5.0, 3.0)),
        colliders=ccol.collision_set([]),
        wall_colliders=ccol.collision_set([]),
        yaw=0.0,
    )

    assert row.grid.all_objects
    row_objects = {obj.item() for obj in row.grid.all_objects}
    leaked = set(bpy.context.scene.objects) - original_objects - row_objects
    assert all(len(getattr(obj.data, "polygons", [])) == 0 for obj in leaked), [
        (obj.name, obj.type) for obj in leaked
    ]
