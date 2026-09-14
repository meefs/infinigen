# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np
import procfunc as pf

from infinigen2 import generate


def test_dummy_camera_ignores_scene_specific_metadata() -> None:
    obj = pf.ops.primitives.mesh_cube(size=2.0)
    first = generate._dummy_camera({"objects": [obj]})
    second = generate._dummy_camera({"objects": [obj], "floors": [obj]})

    np.testing.assert_allclose(first.item().location, second.item().location)
    np.testing.assert_allclose(
        first.item().rotation_euler, second.item().rotation_euler
    )
