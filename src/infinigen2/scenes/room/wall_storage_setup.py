# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import functools
import logging
from typing import NamedTuple

import procfunc as pf

from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.setup_utils import (
    MeshResult,
    back_face_grounded,
    retry_place,
    snap_back_front,
    standalone_wall_planes,
    storage_object_rand,
)

__all__ = ["WallStorageSetupResult", "wall_storage_setup_rand"]

logger = logging.getLogger(__name__)


class WallStorageSetupResult(NamedTuple):
    storage: list[MeshResult]
    all_objects: list[pf.MeshObject]
    colliders: ccol.CollisionSet
    storage_containers: list[pf.MeshObject]
    storage_supports: list[pf.MeshObject]


@pf.tracer.grammar
def wall_storage_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject] | None = None,
    room_dimensions: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> WallStorageSetupResult:
    del room_dimensions
    standalone_walls: list[pf.MeshObject] = []
    if wall_planes is None:
        wall_planes = standalone_wall_planes()
        standalone_walls = wall_planes
    if colliders is None:
        colliders = ccol.collision_set(wall_planes)
    elif standalone_walls:
        colliders = ccol.collision_set(
            colliders.objs + standalone_walls,
            cache=colliders,
        )

    n = pf.random.randint(rng, 4, 10)
    rngs = rng.spawn(n)
    storage = [storage_object_rand(rngs[i]) for i in range(n)]
    wall_margins = [pf.random.uniform(rngs[i], 0.03, 0.10) for i in range(n)]
    wall_colliders = ccol.collision_set(wall_planes)
    placed_storage = []
    for i in range(n):
        grounded = functools.partial(
            back_face_grounded,
            colliders=wall_colliders,
            margin=wall_margins[i],
        )
        storage_obj = retry_place(
            rngs[i],
            storage[i],
            colliders,
            snap_back_front,
            attempts=12,
            parents=wall_planes,
            margin=wall_margins[i],
            accept_fn=grounded,
        )
        placed_storage.append(storage_obj)
    storage_objects, colliders = keep_non_colliding(placed_storage, colliders)
    logger.info(f"Placed {len(storage_objects)} storage objects out of {n} attempts")
    all_objects = standalone_walls + [r.mesh for r in storage_objects]
    storage_meshes = [r.mesh for r in storage_objects]
    return WallStorageSetupResult(
        storage=storage_objects,
        all_objects=all_objects,
        colliders=colliders,
        storage_containers=storage_meshes,
        storage_supports=storage_meshes,
    )
