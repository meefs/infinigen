# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np
import procfunc as pf
import pytest

from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.room import room


def test_ceiling_light_choice_uses_separate_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ceiling_rngs: set[int] = set()
    original_ceiling = room.ceiling_feature_rand
    original_choice = pf.control.choice

    def capture_ceiling(rng: pf.RNG, shape: room.RoomShapeResult) -> object:
        ceiling_rngs.add(id(rng))
        return original_ceiling(rng, shape)

    def assert_separate_choice(
        rng: pf.RNG, options: list[tuple[object, float]]
    ) -> object:
        assert id(rng) not in ceiling_rngs
        return original_choice(rng, options)

    monkeypatch.setattr(room, "ceiling_feature_rand", capture_ceiling)
    monkeypatch.setattr(pf.control, "choice", assert_separate_choice)

    room.room_unfurnished_rand(
        np.random.default_rng(7), dimensions=pf.Vector((4.0, 5.0, 2.7))
    )


def test_room_rand_returns_a_default_camera_facing_inward() -> None:
    dimensions = pf.Vector((4.5, 5.5, 2.7))
    result = room.room_rand(np.random.default_rng(19), dimensions=dimensions)
    camera = result.cameras[0].item()

    assert result.dimensions == dimensions
    assert 0 < camera.location.x < dimensions.x
    assert 0 < camera.location.y < dimensions.y
    direction = camera.matrix_world.to_quaternion() @ pf.Vector((0, 0, -1))
    center = pf.Vector((dimensions.x / 2, dimensions.y / 2, camera.location.z))
    assert direction.dot(center - camera.location) > 0
    affordances = [
        obj
        for obj in result.colliders.objs
        if obj.item().name.startswith("door_affordance")
    ]
    assert affordances
    assert all(obj not in result.all_objects for obj in affordances)
    assert all(obj.item().hide_render for obj in affordances)


def test_room_uses_door_affordance_only_for_placement() -> None:
    dimensions = pf.Vector((4.5, 5.5, 2.7))
    result = room.room_unfurnished_rand(
        np.random.default_rng(19), dimensions=dimensions
    )

    affordances = [
        obj
        for obj in result.colliders.objs
        if obj.item().name.startswith("door_affordance")
    ]
    assert affordances
    assert all(obj not in result.all_objects for obj in affordances)

    collider = affordances[0]
    collider_item = collider.item()
    center = collider_item.matrix_world @ pf.Vector((0.0, 0.0, 0.0))
    normal = collider_item.matrix_world.to_3x3() @ pf.Vector((1.0, 0.0, 0.0))
    normal.normalize()
    collider_min, collider_max = pf.ops.attr.bbox_min_max(collider, global_coords=False)
    extent = collider_max - collider_min
    probe = pf.ops.primitives.mesh_cube(size=0.1)
    probe_location = center + normal * float(extent[0] * 0.375)
    pf.ops.object.set_transform(
        probe,
        location=probe_location,
    )

    assert ccol.intersection_test(result.colliders, probe)
    physical_colliders = ccol.collision_set(result.all_objects)
    assert not ccol.intersection_test(physical_colliders, probe)
