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
    side_table_object_rand,
    snap_back_front,
    snap_side_by_side,
    sofa_object_rand,
    standalone_wall_planes,
)

__all__ = [
    "SofaSetupResult",
    "centered_sofa_setup_rand",
    "sofa_setup_rand",
    "wall_sofa_setup_rand",
]

logger = logging.getLogger(__name__)


class SofaSetupResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    sofas: list[MeshResult]
    rugs: list[pf.MeshObject]
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]


def _rug_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    wall_clearance: float,
) -> list[pf.MeshObject]:
    avail_x = room_dimensions.x - 2 * wall_clearance
    avail_y = room_dimensions.y - 2 * wall_clearance
    length = pf.random.uniform(rng, min(1.0, avail_x), avail_x)
    width = pf.random.uniform(rng, min(1.0, avail_y), avail_y)
    thickness = pf.random.uniform(rng, 0.01, 0.02)
    result = rug.rug_rand(rng, dimensions=pf.Vector((length, width, thickness)))
    result.mesh.item().name = rug.rug_rand.__name__
    cx_min = wall_clearance + length / 2
    cx_max = room_dimensions.x - wall_clearance - length / 2
    cx = room_dimensions.x / 2
    if cx_min < cx_max:
        cx = pf.random.uniform(rng, cx_min, cx_max)
    cy_min = wall_clearance + width / 2
    cy_max = room_dimensions.y - wall_clearance - width / 2
    cy = room_dimensions.y / 2
    if cy_min < cy_max:
        cy = pf.random.uniform(rng, cy_min, cy_max)
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


def _center_coffee_table_rand(
    rng: pf.RNG,
    center: np.ndarray,
) -> list[MeshResult]:
    child = table.coffee_table_rand(rng)
    child.mesh.item().location = (
        center[0] + pf.random.uniform(rng, -0.2, 0.2),
        center[1] + pf.random.uniform(rng, -0.2, 0.2),
        0.01,
    )
    return [child]


def centered_sofa_setup_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
    wall_planes: list[pf.MeshObject] | None = None,
) -> SofaSetupResult:
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
    rug_objs = _rug_rand(rng_rug, room_dimensions, wall_clearance=1.2)
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
    sofas = [sofa_object_rand(sofa_rng) for sofa_rng in sofa_rngs]
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

    center_coffee = _center_coffee_table_rand(rng_coffee, center)
    center_coffee, _ = keep_non_colliding(center_coffee, colliders)
    out_rugs = pf.control.choice(rng_output, [(rug_objs, 2.0), ([], 1.0)])
    all_objects = [r.mesh for r in sofa_objs + center_coffee] + out_rugs
    return SofaSetupResult(
        all_objects=all_objects,
        sofas=sofa_objs,
        rugs=out_rugs,
        storage_containers=[r.mesh for r in sofa_objs],
        supports=[r.mesh for r in center_coffee] + out_rugs,
        storages=[r.mesh for r in center_coffee],
    )


def wall_sofa_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject] | None = None,
    room_dimensions: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> SofaSetupResult:
    standalone_walls: list[pf.MeshObject] = []
    if wall_planes is None:
        wall_planes = standalone_wall_planes()
        standalone_walls = wall_planes
    if room_dimensions is None:
        room_dimensions = pf.Vector((5.0, 15.0, 3.0))
    if colliders is None:
        colliders = ccol.collision_set(wall_planes)
    elif standalone_walls:
        colliders = ccol.collision_set(
            colliders.objs + standalone_walls,
            cache=colliders,
        )

    n = pf.random.randint(rng, 0, 8)
    rngs = rng.spawn(n)
    sofas = [sofa_object_rand(rngs[i]) for i in range(n)]
    placed_sofas = []
    for i in range(n):
        placed_sofas.append(
            retry_place(
                rngs[i], sofas[i], colliders, snap_back_front, parents=wall_planes
            )
        )
    sofa_objs, _ = keep_non_colliding(placed_sofas, colliders)
    logger.info(f"Placed {len(sofa_objs)} wall sofas out of {n} attempts")
    rug_func = pf.control.choice(
        rng,
        [
            (_rug_rand, 1.0),
            (lambda *_, **__: [], 1.0),
        ],
    )
    rug_objs = rug_func(rng, room_dimensions, wall_clearance=0.3)
    return SofaSetupResult(
        all_objects=standalone_walls + [r.mesh for r in sofa_objs] + rug_objs,
        sofas=sofa_objs,
        rugs=rug_objs,
        storage_containers=[r.mesh for r in sofa_objs],
        supports=rug_objs,
        storages=[],
    )


def sofa_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject] | None = None,
    room_dimensions: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> SofaSetupResult:
    if colliders is None:
        colliders = ccol.collision_set(wall_planes or [])
    rng_arr, rng_side = rng.spawn(2)

    arrangement_func = pf.control.choice(
        rng_arr,
        [
            (wall_sofa_setup_rand, 3.0),
            (centered_sofa_setup_rand, 7.0),
        ],
    )
    arrangement = arrangement_func(
        rng_arr,
        wall_planes=wall_planes,
        room_dimensions=room_dimensions,
        colliders=colliders,
    )
    sofa_meshes = [r.mesh for r in arrangement.sofas]
    colliders = ccol.collision_set(
        colliders.objs + arrangement.all_objects,
        cache=colliders,
    )

    n = min(pf.random.randint(rng_side, 1, 4), len(arrangement.sofas))
    rngs = rng_side.spawn(n)
    side_tables = [side_table_object_rand(rngs[i]) for i in range(n)]
    placed_side_tables = []
    for i in range(n):
        placed_side_tables.append(
            retry_place(
                rngs[i],
                side_tables[i],
                colliders,
                snap_side_by_side,
                parents=sofa_meshes,
                attempts=12,
            )
        )
    side_tables, _ = keep_non_colliding(placed_side_tables, colliders)
    logger.info(f"Placed {len(side_tables)} side tables out of {n} attempts")

    return SofaSetupResult(
        all_objects=arrangement.all_objects + [r.mesh for r in side_tables],
        sofas=arrangement.sofas,
        rugs=arrangement.rugs,
        storage_containers=list(arrangement.storage_containers),
        supports=arrangement.supports + [r.mesh for r in side_tables],
        storages=arrangement.storages + [r.mesh for r in side_tables],
    )
