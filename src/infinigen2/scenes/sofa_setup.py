# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import NamedTuple

import procfunc as pf

from infinigen2.scenes.centered_sofa_setup import centered_sofa_setup_rand
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.setup_utils import (
    MeshResult,
    retry_place,
    side_table_object_rand,
    snap_side_by_side,
)
from infinigen2.scenes.wall_sofa_setup import wall_sofa_setup_rand

__all__ = ["SofaSetupResult", "sofa_setup_rand"]

logger = logging.getLogger(__name__)


class SofaSetupResult(NamedTuple):
    sofas: list[MeshResult]
    coffee_tables: list[MeshResult]
    side_tables: list[MeshResult]
    rugs: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]


def sofa_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject] | None = None,
    room_dimensions: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> SofaSetupResult:
    """A sofa grouping (choice of wall-snapped or rug-centered) plus side tables beside
    the sofas. `wall_planes` passes through to the wall-snapped variant. Lamps are
    placed by the caller from the returned sofas and side tables (see sofa_lamps_rand)."""
    if colliders is None:
        colliders = ccol.collision_set(wall_planes or [])
    rng_arr, rng_side = rng.spawn(2)

    arrangement_func = pf.control.choice(
        rng_arr,
        [
            (wall_sofa_setup_rand, 2.0),
            (centered_sofa_setup_rand, 3.0),
        ],
    )
    arrangement = arrangement_func(
        rng_arr,
        wall_planes=wall_planes,
        room_dimensions=room_dimensions,
        colliders=colliders,
    )
    sofa_meshes = [r.mesh for r in arrangement.sofas]
    coffee_tables = list(arrangement.coffee_tables)
    solid = sofa_meshes + [r.mesh for r in coffee_tables]
    colliders = ccol.collision_set(colliders.objs + solid, cache=colliders)

    n = min(pf.random.randint(rng_side, 1, 4), len(arrangement.sofas))
    rngs = rng_side.spawn(n)
    side_tables = [side_table_object_rand(rngs[i]) for i in range(n)]
    placed_side_tables = []
    for i in range(n):
        side_table = retry_place(
            rngs[i],
            side_tables[i],
            colliders,
            snap_side_by_side,
            parents=sofa_meshes,
            attempts=12,
        )
        placed_side_tables.append(side_table)
    side_tables = placed_side_tables
    side_tables, colliders = keep_non_colliding(side_tables, colliders)
    logger.info(f"Placed {len(side_tables)} side tables out of {n} attempts")

    all_objects = arrangement.all_objects + [r.mesh for r in side_tables]
    return SofaSetupResult(
        sofas=arrangement.sofas,
        coffee_tables=coffee_tables,
        side_tables=side_tables,
        rugs=list(arrangement.rugs),
        all_objects=all_objects,
    )
