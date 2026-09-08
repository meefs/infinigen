# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import NamedTuple

import numpy as np
import procfunc as pf

from infinigen2.objects import rug, table
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding, keep_unobstructed
from infinigen2.scenes.placement.snap import snap_to_plane
from infinigen2.scenes.setup_utils import MeshResult, retry_place, sofa_object_rand

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
    rug_objs = _rug_rand(rng, room_dimensions=room_dimensions)
    rug_obj = rug_objs[0]
    cmin, cmax = (
        np.array(bound)
        for bound in pf.ops.attr.bbox_min_max(rug_obj, global_coords=True)
    )
    center = (cmin + cmax) / 2

    # one sofa per chosen rug side, snapped to that side facing inward
    side_names = ["right", "left", "front", "back"]
    n_sides = pf.random.randint(rng, 2, 5)
    sides = [
        side_names[int(i)]
        for i in rng.choice(len(side_names), size=n_sides, replace=False)
    ]
    n = len(sides)
    rngs = rng.spawn(n)
    sofas = []
    for i in range(n):
        sofas.append(sofa_object_rand(rngs[i]))
    placed_sofas = []
    for i in range(n):
        sofa_obj = retry_place(
            rngs[i],
            sofas[i],
            colliders,
            _snap_facing_rug,
            rug_obj=rug_obj,
            parent_side=sides[i],
        )
        placed_sofas.append(sofa_obj)
    sofas = placed_sofas
    sofas = keep_unobstructed(sofas, center, colliders)
    sofa_objs, colliders = keep_non_colliding(sofas, colliders)
    logger.info(f"Placed {len(sofa_objs)} carpet sofas out of {n} attempts")

    def _place_coffee(rng: pf.RNG) -> list[MeshResult]:
        child = table.coffee_table_rand(rng)
        child.mesh.item().location = (
            center[0] + pf.random.uniform(rng, -0.2, 0.2),
            center[1] + pf.random.uniform(rng, -0.2, 0.2),
            0.01,
        )
        return [child]

    center_coffee = _place_coffee(rng)
    center_coffee, colliders = keep_non_colliding(center_coffee, colliders)

    # sofas already ringed the carpet above; drop the rug ~1/3 of the time
    out_rugs = pf.control.choice(rng, [(rug_objs, 2.0), ([], 1.0)])

    all_objects = [r.mesh for r in sofa_objs + center_coffee] + out_rugs
    return CenteredSofaSetupResult(sofa_objs, center_coffee, out_rugs, all_objects)
