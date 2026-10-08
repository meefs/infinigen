# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.base_materials.fabric import fabric_rand
from infinigen2.shaders.displacements.wrinkles import (
    wrinkles_fabric_rand,
    wrinkles_rug_rand,
)

__all__ = [
    "fabric_wrinkled_rand",
    "wrinkles_big_overlay_rand",
    "wrinkles_overlay_rand",
    "wrinkles_small_overlay_rand",
]


def wrinkles_small_overlay_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    material: pf.Material,
) -> pf.Material:
    wrinkles = wrinkles_fabric_rand(rng, vector)
    return pf.Material(
        surface=material.surface,
        displacement=material.displacement + wrinkles,
    )


def wrinkles_big_overlay_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    material: pf.Material,
) -> pf.Material:
    wrinkles = wrinkles_rug_rand(rng, vector)
    return pf.Material(
        surface=material.surface,
        displacement=material.displacement + wrinkles,
    )


def wrinkles_overlay_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    material: pf.Material,
) -> pf.Material:
    rng_choice, rng_wrinkles = rng.spawn(2)
    wrinkles_func = pf.control.choice(
        rng_choice,
        [
            (wrinkles_small_overlay_rand, 3.0),
            (wrinkles_big_overlay_rand, 2.0),
        ],
    )
    return wrinkles_func(rng_wrinkles, vector, material)


def fabric_wrinkled_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    base_color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    rng_fabric, rng_wrinkles = rng.spawn(2)
    material = fabric_rand(rng_fabric, vector, base_color=base_color)
    return wrinkles_overlay_rand(rng_wrinkles, vector, material)
