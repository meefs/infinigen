# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import functools
from typing import NamedTuple, cast

import numpy as np
import procfunc as pf

from infinigen2.objects import chair
from infinigen2.objects import desk as desk_object
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.snap import snap_to_plane
from infinigen2.scenes.setup_utils import (
    back_face_grounded,
    retry_place,
    snap_back_front,
)

__all__ = [
    "DeskSetupResult",
    "desk_setup_rand",
]


class DeskSetupResult(NamedTuple):
    desk: pf.MeshObject
    chair: pf.MeshObject
    all_objects: list[pf.MeshObject]
    storage_containers: list[pf.MeshObject]
    storage_supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]


def _chair_pose_rand(rng: pf.RNG) -> tuple[float, float, float]:
    rng_placement, rng_margin, rng_yaw_active, rng_yaw, rng_yaw_sign = rng.spawn(5)
    placement = pf.random.clip_gaussian(rng_placement, 0.5, 0.15, 0.3, 0.7)
    margin = pf.random.clip_gaussian(rng_margin, 0.12, 0.16, -0.12, 0.45)
    yaw_active = pf.control.choice(rng_yaw_active, [(False, 2.0), (True, 1.0)])
    yaw = 0.0
    if yaw_active:
        yaw_magnitude = pf.random.uniform(rng_yaw, np.deg2rad(5), np.deg2rad(22))
        yaw_sign = pf.control.choice(rng_yaw_sign, [(-1.0, 1.0), (1.0, 1.0)])
        yaw = yaw_sign * yaw_magnitude
    return placement, margin, yaw


def _place_chair_rand(
    rng: pf.RNG, chair_result: chair.ChairResult, desk: pf.MeshObject
) -> None:
    placement, margin, yaw = _chair_pose_rand(rng)
    snap_to_plane(
        child=chair_result.mesh,
        parent=desk,
        placement=placement,
        child_side="front",
        parent_side="front",
        margin=margin,
    )
    chair_result.mesh.item().rotation_euler.z += yaw


def _place_desk_against_wall_rand(
    rng: pf.RNG,
    desk_result: desk_object.DeskResult,
    wall_planes: list[pf.MeshObject],
    margin: float,
) -> None:
    snap_back_front(
        rng,
        desk_result,
        wall_planes,
        placement=pf.random.uniform(rng, 0.10, 0.90),
        margin=margin,
    )


def desk_setup_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    wall_planes: list[pf.MeshObject] | None = None,
    colliders: ccol.CollisionSet | None = None,
    attempts: int = 12,
) -> DeskSetupResult | None:
    """A desk and chair, snapped to a wall when wall planes are supplied."""
    if colliders is None:
        colliders = ccol.collision_set([])

    rng_desk, rng_chair, rng_desk_pose, rng_chair_pose = rng.spawn(4)
    desk_result = desk_object.desk_rand(rng_desk, dimensions=dimensions)
    rng_chair_choice, rng_chair_gen = rng_chair.spawn(2)
    chair_func = pf.control.choice(
        rng_chair_choice,
        [(chair.office_chair_rand, 2.0), (chair.chair_rand, 1.0)],
    )
    chair_result = chair_func(rng_chair_gen)
    chair_result.mesh.item().name = "desk_chair"

    if wall_planes:
        wall_margin = pf.random.uniform(rng_desk_pose, 0.03, 0.10)
        grounded = functools.partial(
            back_face_grounded,
            colliders=ccol.collision_set(cast(list[pf.Object], wall_planes)),
            margin=wall_margin,
        )
        desk_result = retry_place(
            rng_desk_pose,
            desk_result,
            colliders,
            _place_desk_against_wall_rand,
            attempts=attempts,
            accept_fn=grounded,
            wall_planes=wall_planes,
            margin=wall_margin,
        )
        if desk_result is None:
            return None

    chair_colliders = ccol.collision_set(
        [*colliders.objs, desk_result.mesh],
        cache=colliders,
    )
    chair_result = retry_place(
        rng_chair_pose,
        chair_result,
        chair_colliders,
        _place_chair_rand,
        attempts=attempts,
        desk=desk_result.mesh,
    )
    if chair_result is None:
        return None
    desk = desk_result.mesh
    chair_obj = chair_result.mesh
    return DeskSetupResult(
        desk=desk,
        chair=chair_obj,
        all_objects=[desk, chair_obj],
        storage_containers=[],
        storage_supports=[desk],
        storages=[desk],
    )
