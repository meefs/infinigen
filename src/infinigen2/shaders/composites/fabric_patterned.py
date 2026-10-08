# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Michael Cai, Alexander Raistrick: refactor for Infinigen2

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.base_materials.fabric import (
    fabric_color_rand,
    fabric_rand,
)
from infinigen2.shaders.masks.tile_shapes import (
    TileShapeResult,
    tile_coord_transform_rand,
    tile_mask_rand,
)

__all__ = [
    "fabric_patterned_rand",
    "patterned_color_rand",
]


def patterned_color_rand(
    rng,
    color1: t.SocketOrVal[pf.Color],
    color2: t.SocketOrVal[pf.Color],
    color3: t.SocketOrVal[pf.Color],
    tile_mask_result: TileShapeResult,
):
    mask = tile_mask_result.mask.astype(dtype=float) > 0.1
    mask1 = tile_mask_result.tile_type_1.astype(dtype=float) > 0.1
    mask2 = tile_mask_result.tile_type_2.astype(dtype=float) > 0.1

    color_mixed1 = pf.control.choice(
        rng,
        [
            (color1, 1),
            (pf.nodes.color.mix_rgb(factor=mask1, a=color1, b=color2), 1),
            (pf.nodes.color.mix_rgb(factor=mask, a=color1, b=color3), 1),
        ],
    )
    color = pf.control.choice(
        rng,
        [
            (color_mixed1, 1),
            (pf.nodes.color.mix_rgb(factor=mask2, a=color_mixed1, b=color3), 1),
        ],
    )
    return color


def fabric_patterned_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    color1: t.SocketOrVal[pf.Color] | None = None,
    color2: t.SocketOrVal[pf.Color] | None = None,
    color3: t.SocketOrVal[pf.Color] | None = None,
    tile_mask: TileShapeResult | None = None,
    scale: float | None = None,
    translucency: float = 0.0,
) -> pf.Material:
    rngs = rng.spawn(9)

    if scale is None:
        scale = pf.random.uniform(rngs[0], 10.0, 50.0)
    if tile_mask is None:
        tile_mask = tile_mask_rand(
            rngs[0],
            tile_coord_transform_rand(rngs[1], vector, scale=scale),
        )
    mask = tile_mask.mask.astype(dtype=float) > 0.1

    if color1 is None:
        rng_c1, rng_c2, rng_c3 = rngs[2].spawn(3)
        color1 = fabric_color_rand(rng_c1)
        color2 = fabric_color_rand(rng_c2)
        color3 = fabric_color_rand(rng_c3)

    color = patterned_color_rand(rngs[5], color1, color2, color3, tile_mask)

    small_indent = pf.random.uniform(rngs[6], 0.0, 0.0025)

    displacement_offset = pf.control.choice(
        rngs[7],
        [
            (0, 2),
            (mask * small_indent, 1),
            (mask * -small_indent, 1),
        ],
    )

    res = fabric_rand(
        rngs[8], vector=vector, base_color=color, translucency=translucency
    )

    displacement = res.displacement + pf.nodes.shader.displacement(
        height=displacement_offset, midlevel=0.0
    )
    return pf.Material(surface=res.surface, displacement=displacement)
