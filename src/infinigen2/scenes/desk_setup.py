# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import NamedTuple

import numpy as np
import procfunc as pf

from infinigen2.objects import chair, lamp
from infinigen2.objects import desk as desk_object
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.snap import snap_to_plane

__all__ = [
    "DeskSetupResult",
    "desk_setup_in_room_rand",
    "desk_setup_rand",
]


class DeskSetupResult(NamedTuple):
    desk: pf.MeshObject
    chair: pf.MeshObject
    lamps: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]
    lights: list[pf.LightObject]


def _chair_pose_rand(rng: pf.RNG) -> tuple[float, float, float]:
    rng_placement, rng_margin, rng_yaw_active, rng_yaw, rng_yaw_sign = rng.spawn(5)
    placement = pf.random.clip_gaussian(rng_placement, 0.5, 0.25, 0.05, 0.95)
    margin = pf.random.clip_gaussian(rng_margin, 0.12, 0.16, -0.12, 0.45)
    yaw_active = pf.control.choice(rng_yaw_active, [(False, 2.0), (True, 1.0)])
    yaw = 0.0
    if yaw_active:
        yaw_magnitude = pf.random.uniform(rng_yaw, np.deg2rad(5), np.deg2rad(22))
        yaw_sign = pf.control.choice(rng_yaw_sign, [(-1.0, 1.0), (1.0, 1.0)])
        yaw = yaw_sign * yaw_magnitude
    return placement, margin, yaw


def _place_chair_rand(
    rng: pf.RNG, chair_obj: pf.MeshObject, desk: pf.MeshObject
) -> None:
    placement, margin, yaw = _chair_pose_rand(rng)
    snap_to_plane(
        child=chair_obj,
        parent=desk,
        placement=placement,
        child_side="front",
        parent_side="front",
        margin=margin,
    )
    chair_obj.item().rotation_euler.z += yaw


def _lamp_xy_frac_rand(rng: pf.RNG) -> tuple[float, float]:
    rng_depth, rng_side, rng_width = rng.spawn(3)
    depth_frac = pf.random.uniform(rng_depth, 0.10, 0.35)
    side = pf.control.choice(rng_side, [(-1.0, 1.0), (1.0, 1.0)])
    width_from_edge = pf.random.uniform(rng_width, 0.05, 0.20)
    width_frac = width_from_edge if side < 0 else 1.0 - width_from_edge
    return depth_frac, width_frac


def _place_lamp_rand(rng: pf.RNG, lamp_obj: pf.MeshObject, desk: pf.MeshObject) -> None:
    desk_min, desk_max = (
        np.array(v) for v in pf.ops.attr.bbox_min_max(desk, global_coords=False)
    )
    lamp_min, lamp_max = (
        np.array(v) for v in pf.ops.attr.bbox_min_max(lamp_obj, global_coords=False)
    )
    desk_size = desk_max - desk_min
    lamp_size = lamp_max - lamp_min
    fit_scale = min(
        1.0,
        0.65 * desk_size[0] / lamp_size[0],
        0.35 * desk_size[1] / lamp_size[1],
    )
    lamp_obj.item().scale.x *= fit_scale
    lamp_obj.item().scale.y *= fit_scale
    lamp_min[:2] *= fit_scale
    lamp_max[:2] *= fit_scale
    pf.ops.mesh.transform_apply(lamp_obj, location=False, rotation=False, scale=True)

    location_min = desk_min[:2] - lamp_min[:2]
    location_max = desk_max[:2] - lamp_max[:2]
    x_frac, y_frac = _lamp_xy_frac_rand(rng)
    local_location = pf.Vector(
        (
            location_min[0] + (location_max[0] - location_min[0]) * x_frac,
            location_min[1] + (location_max[1] - location_min[1]) * y_frac,
            desk_max[2] + 0.002 - lamp_min[2],
        )
    )
    lamp_obj.item().location = desk.item().matrix_world @ local_location


def desk_setup_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    include_lamp: bool | None = None,
) -> DeskSetupResult:
    """A compositional desk, one loosely centered chair, and optional lamp."""
    (
        rng_desk,
        rng_chair,
        rng_chair_pose,
        rng_lamp_active,
        rng_lamp,
        rng_lamp_pose,
    ) = rng.spawn(6)

    desk = desk_object.desk_rand(rng_desk, dimensions=dimensions).mesh

    rng_chair_choice, rng_chair_gen = rng_chair.spawn(2)
    chair_func = pf.control.choice(
        rng_chair_choice,
        [(chair.office_chair_rand, 2.0), (chair.chair_rand, 1.0)],
    )
    chair_obj = chair_func(rng_chair_gen).mesh
    chair_obj.item().name = "desk_chair"
    _place_chair_rand(rng_chair_pose, chair_obj, desk)

    if include_lamp is None:
        include_lamp = pf.control.choice(rng_lamp_active, [(False, 1.0), (True, 1.0)])

    lamps: list[pf.MeshObject] = []
    lights: list[pf.LightObject] = []
    if include_lamp:
        lamp_result = lamp.desk_lamp_rand(rng_lamp)
        lamp_result.mesh.item().name = "desk_lamp"
        _place_lamp_rand(rng_lamp_pose, lamp_result.mesh, desk)
        lamps.append(lamp_result.mesh)
        lights.append(lamp_result.light)

    all_objects = [desk, chair_obj, *lamps]
    return DeskSetupResult(desk, chair_obj, lamps, all_objects, lights)


def _place_desk_against_wall_rand(
    rng: pf.RNG,
    desk: pf.MeshObject,
    wall_planes: list[pf.MeshObject],
    room_dimensions: pf.Vector,
) -> None:
    del room_dimensions
    wall = wall_planes[int(rng.integers(len(wall_planes)))]
    snap_to_plane(
        child=desk,
        parent=wall,
        placement=pf.random.uniform(rng, 0.10, 0.90),
        child_side="back",
        parent_side="front",
        margin=pf.random.uniform(rng, 0.03, 0.10),
    )


def _place_desk_freestanding_rand(
    rng: pf.RNG,
    desk: pf.MeshObject,
    wall_planes: list[pf.MeshObject],
    room_dimensions: pf.Vector,
) -> None:
    del wall_planes
    desk_min, _ = pf.ops.attr.bbox_min_max(desk, global_coords=False)
    location = (
        pf.random.uniform(rng, 0.15, 0.85) * room_dimensions.x,
        pf.random.uniform(rng, 0.15, 0.85) * room_dimensions.y,
        0.001 - desk_min[2],
    )
    yaw = pf.control.choice(
        rng,
        [
            (0.0, 1.0),
            (np.pi / 2, 1.0),
            (np.pi, 1.0),
            (3 * np.pi / 2, 1.0),
        ],
    )
    rotation = (0.0, 0.0, yaw)
    pf.ops.object.set_transform(desk, location=location, rotation_euler=rotation)


def _desk_placement_mode_rand(rng: pf.RNG) -> str:
    return pf.control.choice(
        rng,
        [("against_wall", 1.0), ("freestanding", 1.0)],
    )


def _setup_intersects(setup: DeskSetupResult, colliders: ccol.CollisionSet) -> bool:
    return any(ccol.intersection_test(colliders, obj) for obj in setup.all_objects)


def desk_setup_in_room_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    placement_mode: str | None = None,
    include_lamp: bool | None = None,
    attempts: int = 12,
) -> DeskSetupResult | None:
    rng_setup, rng_mode, rng_attempts = rng.spawn(3)
    setup = desk_setup_rand(rng_setup, include_lamp=include_lamp)
    for obj in [setup.chair, *setup.lamps]:
        obj.item().parent = setup.desk.item()

    if placement_mode is None:
        placement_mode = _desk_placement_mode_rand(rng_mode)
    placement_funcs = {
        "against_wall": _place_desk_against_wall_rand,
        "freestanding": _place_desk_freestanding_rand,
    }
    place_func = placement_funcs[placement_mode]

    for attempt_rng in rng_attempts.spawn(attempts):
        place_func(attempt_rng, setup.desk, wall_planes, room_dimensions)
        if not _setup_intersects(setup, colliders):
            return setup

    for obj in setup.all_objects:
        obj.item().name = obj.item().name + "_FAILED_PLACEMENT"
    return None
