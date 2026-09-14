# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
import math
from typing import NamedTuple

import numpy as np
import procfunc as pf

from infinigen2.objects import rug, table
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding, keep_unobstructed
from infinigen2.scenes.placement.snap import snap_to_plane
from infinigen2.scenes.setup_utils import (
    MeshResult,
    jitter_object_rotation_rand,
    retry_place,
    sofa_object_rand,
)

__all__ = ["CenteredSofaSetupResult", "centered_sofa_setup_rand"]

logger = logging.getLogger(__name__)


class CenteredSofaSetupResult(NamedTuple):
    sofas: list[MeshResult]
    coffee_tables: list[MeshResult]
    rugs: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]


def _rug_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
) -> list[pf.MeshObject]:
    wall_clearance = 1.2
    avail_x = room_dimensions.x - 2 * wall_clearance
    avail_y = room_dimensions.y - 2 * wall_clearance
    length = pf.random.uniform(rng, min(1.0, avail_x), avail_x)
    width = pf.random.uniform(rng, min(1.0, avail_y), avail_y)
    thickness = pf.random.uniform(rng, 0.01, 0.02)
    result = rug.rug_rand(rng, dimensions=pf.Vector((length, width, thickness)))
    result.mesh.item().name = rug.rug_rand.__name__
    cx = pf.random.uniform(
        rng,
        wall_clearance + length / 2,
        room_dimensions.x - wall_clearance - length / 2,
    )
    cy = pf.random.uniform(
        rng, wall_clearance + width / 2, room_dimensions.y - wall_clearance - width / 2
    )
    pf.ops.object.set_transform(result.mesh, location=(cx, cy, 0.001))
    return [result.mesh]


def _snap_facing_rug(
    rng: pf.RNG,
    child: MeshResult,
    rug_obj: pf.MeshObject,
    parent_side: str,
) -> None:
    snap_to_plane(
        child=child.mesh,
        parent=rug_obj,
        parent_side=parent_side,
        child_side="front",
        margin=pf.random.uniform(rng, -0.1, 0.5),
        placement=pf.random.uniform(rng, 0.35, 0.65),
        constraint_axis=pf.Vector((0, 0, 1)),
    )


def centered_sofa_setup_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
    wall_planes: list[pf.MeshObject] | None = None,
) -> CenteredSofaSetupResult:
    """Sofas ringing a central rug facing inward, with an optional center coffee
    table. Works without walls, so it is registered as a standalone Scene;
    `wall_planes` is accepted only for signature parity with the wall variant."""
    del wall_planes
    if room_dimensions is None:
        room_dimensions = pf.Vector((5.0, 5.0, 3.0))
    if colliders is None:
        colliders = ccol.collision_set([])
    (
        rng_rug,
        rng_arrange,
        rng_sofas,
        rng_jitter_amount,
        rng_jitter_objects,
        rng_coffee,
        rng_output,
    ) = rng.spawn(7)
    rug_objs = _rug_rand(rng_rug, room_dimensions=room_dimensions)
    rug_obj = rug_objs[0]
    cmin, cmax = (
        np.array(bound)
        for bound in pf.ops.attr.bbox_min_max(rug_obj, global_coords=True)
    )
    center = (cmin + cmax) / 2

    side_names = ["right", "left", "front", "back"]
    n_sides = pf.random.randint(rng_arrange, 2, 4)
    sides = [
        side_names[int(i)]
        for i in rng_arrange.choice(len(side_names), size=n_sides, replace=False)
    ]
    sofa_rngs = rng_sofas.spawn(n_sides)
    sofas = []
    for sofa_rng in sofa_rngs:
        sofas.append(sofa_object_rand(sofa_rng))
    placed_sofas = []
    for side, sofa_rng, sofa_obj in zip(sides, sofa_rngs, sofas, strict=True):
        placed_sofas.append(
            retry_place(
                sofa_rng,
                sofa_obj,
                colliders,
                _snap_facing_rug,
                rug_obj=rug_obj,
                parent_side=side,
            )
        )
    placed_sofas = keep_unobstructed(placed_sofas, center, colliders)
    sofa_objs, colliders = keep_non_colliding(placed_sofas, colliders)
    logger.info(f"Placed {len(sofa_objs)} carpet sofas out of {n_sides} attempts")

    max_angle = math.radians(5) * pf.random.uniform(rng_jitter_amount, 0.0, 1.0) ** 3
    sofa_rngs = rng_jitter_objects.spawn(len(sofa_objs))
    for sofa_obj, sofa_rng in zip(sofa_objs, sofa_rngs, strict=True):
        jitter_object_rotation_rand(sofa_rng, sofa_obj.mesh, max_angle, colliders)

    def _place_coffee(rng: pf.RNG) -> list[MeshResult]:
        child = table.coffee_table_rand(rng)
        child.mesh.item().location = (
            center[0] + pf.random.uniform(rng, -0.2, 0.2),
            center[1] + pf.random.uniform(rng, -0.2, 0.2),
            0.01,
        )
        return [child]

    center_coffee = _place_coffee(rng_coffee)
    center_coffee, colliders = keep_non_colliding(center_coffee, colliders)

    # sofas already ringed the carpet above; drop the rug ~1/3 of the time
    out_rugs = pf.control.choice(rng_output, [(rug_objs, 2.0), ([], 1.0)])

    all_objects = [r.mesh for r in sofa_objs + center_coffee] + out_rugs
    return CenteredSofaSetupResult(sofa_objs, center_coffee, out_rugs, all_objects)
