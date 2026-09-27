# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lance Phan: original wrinkle nodegroups
# - Alexander Raistrick: transpile and refactor to procfunc/v2

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.base_materials.fabric import fabric_rand
from infinigen2.shaders.displacements.wrinkles import (
    wrinkles_fabric_rand,
    wrinkles_rug_rand,
)

__all__ = ["fabric_bedding_wrinkled_rand", "fabric_wrinkled_rand"]


def _combine(
    fabric: pf.Material,
    wrinkles: pf.ProcNode[pf.Vector],
) -> pf.Material:
    return pf.Material(
        surface=fabric.surface,
        displacement=fabric.displacement + wrinkles,
    )


def fabric_wrinkled_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    base_color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    rng_fabric, rng_wrinkles = rng.spawn(2)
    material = fabric_rand(rng_fabric, vector, base_color=base_color)
    wrinkles = wrinkles_fabric_rand(rng_wrinkles, vector)
    return _combine(material, wrinkles)


def fabric_bedding_wrinkled_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    base_color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    rng_fabric, rng_choice, rng_wrinkles = rng.spawn(3)
    material = fabric_rand(rng_fabric, vector, base_color=base_color)
    wrinkles_func = pf.control.choice(
        rng_choice,
        [
            (wrinkles_fabric_rand, 3.0),
            (wrinkles_rug_rand, 2.0),
        ],
    )
    wrinkles = wrinkles_func(rng_wrinkles, vector)
    return _combine(material, wrinkles)
