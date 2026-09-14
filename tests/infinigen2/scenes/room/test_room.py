# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np
import procfunc as pf

from infinigen2.scenes.room import room


def test_room_rand_returns_a_default_camera_facing_inward() -> None:
    dimensions = pf.Vector((4.5, 5.5, 2.7))
    result = room.room_rand(np.random.default_rng(19), dimensions=dimensions)
    camera = result.cameras[0].item()

    assert 0 < camera.location.x < dimensions.x
    assert 0 < camera.location.y < dimensions.y
    direction = camera.matrix_world.to_quaternion() @ pf.Vector((0, 0, -1))
    center = pf.Vector((dimensions.x / 2, dimensions.y / 2, camera.location.z))
    assert direction.dot(center - camera.location) > 0
