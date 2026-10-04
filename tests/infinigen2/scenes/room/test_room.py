# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np
import procfunc as pf
import pytest

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
