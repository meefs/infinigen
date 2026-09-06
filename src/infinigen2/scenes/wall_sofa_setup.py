# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import NamedTuple

import procfunc as pf

from infinigen2.objects import rug, sofa
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.setup_utils import (
    MeshResult,
    retry_place,
    snap_back_front,
    standalone_wall_planes,
)

__all__ = ["WallSofaSetupResult", "wall_sofa_setup_rand"]

logger = logging.getLogger(__name__)


class WallSofaSetupResult(NamedTuple):
    sofas: list[MeshResult]
    coffee_tables: list[MeshResult]
    rugs: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]


def _rug_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
) -> list[pf.MeshObject]:
    wall_clearance = 0.3
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


def wall_sofa_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject] | None = None,
    room_dimensions: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> WallSofaSetupResult:
    """Sofas snapped against walls, plus an optional rug."""
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
    sofas = []
    for i in range(n):
        sofas.append(sofa.sofa_rand(rngs[i]))
    placed_sofas = []
    for i in range(n):
        sofa_obj = retry_place(
            rngs[i], sofas[i], colliders, snap_back_front, parents=wall_planes
        )
        placed_sofas.append(sofa_obj)
    sofas = placed_sofas
    sofa_objs, colliders = keep_non_colliding(sofas, colliders)
    logger.info(f"Placed {len(sofa_objs)} wall sofas out of {n} attempts")
    rug_func = pf.control.choice(
        rng,
        [
            (_rug_rand, 1.0),
            (lambda *_, **__: [], 1.0),
        ],
    )
    rug_objs = rug_func(rng, room_dimensions=room_dimensions)
    all_objects = standalone_walls + [r.mesh for r in sofa_objs] + rug_objs
    return WallSofaSetupResult(sofa_objs, [], rug_objs, all_objects)
