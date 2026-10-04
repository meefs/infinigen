# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np

from infinigen2.objects import plant_pot


def test_plant_pot_active_uv_covers_joined_parts() -> None:
    obj = plant_pot.plant_pot_small_rand(np.random.default_rng(0)).mesh.item()
    mesh = obj.data
    layer = next(layer for layer in mesh.uv_layers if layer.active_render)
    part = mesh.attributes["plant_pot_part"].data

    for part_id in (0, 1):
        values = np.array(
            [
                layer.data[loop.index].uv[:]
                for loop in mesh.loops
                if part[loop.vertex_index].value == part_id
            ]
        )
        assert np.ptp(values, axis=0).min() > 0
