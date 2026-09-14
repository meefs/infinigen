# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import Callable, Protocol, TypeVar, runtime_checkable

import numpy as np
import procfunc as pf

from infinigen2.objects import (
    lamp,
    sofa,
    storage,
    table,
    vase,
)
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.placement.snap import snap_to_plane

__all__ = [
    "MeshResult",
    "back_face_grounded",
    "jitter_object_rotation_rand",
    "random_bbox_poses_animation_rand",
    "retry_place",
    "side_table_object_rand",
    "snap_back_front",
    "snap_on_top",
    "snap_side_by_side",
    "snap_to_wall",
    "sofa_lamps_rand",
    "sofa_object_rand",
    "standalone_wall_planes",
    "storage_object_rand",
    "table_decoration_object_rand",
]

logger = logging.getLogger(__name__)

MR = TypeVar("MR", bound="MeshResult")


@runtime_checkable
class MeshResult(Protocol):
    @property
    def mesh(self) -> pf.MeshObject: ...


def _yaw_about_point(
    location: tuple[float, float, float],
    center: np.ndarray,
    yaw: float,
) -> tuple[float, float, float]:
    """Rotate `location` by `yaw` about the vertical axis through `center`."""
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    x, y = rot @ (np.asarray(location[:2]) - center[:2]) + center[:2]
    return (x, y, location[2])


def jitter_object_rotation_rand(
    rng: pf.RNG,
    obj: pf.MeshObject,
    max_angle: float,
    colliders: ccol.CollisionSet,
    attempts: int = 12,
) -> None:
    """Retry yaw against every other object in `colliders`."""
    item = obj.item()
    location = tuple(item.location)
    rotation = tuple(item.rotation_euler)
    bbox_min, bbox_max = pf.ops.attr.bbox_min_max(obj, global_coords=True)
    center = (np.asarray(bbox_min) + np.asarray(bbox_max)) / 2
    other_objs = [other for other in colliders.objs if other.item() is not item]
    other_colliders = ccol.collision_set(other_objs, cache=colliders)
    for attempt_rng in rng.spawn(attempts):
        yaw = pf.random.uniform(attempt_rng, -max_angle, max_angle)
        pf.ops.object.set_transform(
            obj,
            location=_yaw_about_point(location, center, yaw),
            rotation_euler=(rotation[0], rotation[1], rotation[2] + yaw),
        )
        if not ccol.intersection_test(other_colliders, obj):
            return
    pf.ops.object.set_transform(obj, location=location, rotation_euler=rotation)


@pf.tracer.grammar
def random_bbox_poses_animation_rand(
    camera: pf.CameraObject,
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    frame_start: int = 1,
    frame_end: int = 1,
    wall_margin: float = 0.5,
    ceil_margin: float = 0.3,
    floor_margin: float = 0.3,
) -> None:
    cam = camera.item()
    n_frames = max(frame_end - frame_start + 1, 1)
    frame_rngs = rng.spawn(n_frames)
    for i in range(n_frames):
        frame = frame_start + i
        r = frame_rngs[i]
        x = pf.random.uniform(r, wall_margin, room_dimensions.x - wall_margin)
        y = pf.random.uniform(r, wall_margin, room_dimensions.y - wall_margin)
        z = pf.random.clip_gaussian(
            r, 1.5, 0.4, floor_margin, room_dimensions.z - ceil_margin
        )
        yaw = pf.random.uniform(r, -np.pi, np.pi)
        pitch = pf.random.clip_gaussian(r, np.pi / 2, 0.3, np.pi / 4, 3 * np.pi / 4)
        roll = pf.random.clip_gaussian(r, 0.0, 0.05, -0.2, 0.2)
        cam.location = (x, y, z)
        cam.rotation_euler = (pitch, roll, yaw)
        cam.keyframe_insert("location", frame=frame)
        cam.keyframe_insert("rotation_euler", frame=frame)


@pf.tracer.grammar
def table_decoration_object_rand(
    rng: pf.RNG,
) -> MeshResult:
    func = pf.control.choice(
        rng,
        [
            (vase.vase_rand, 1.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


@pf.tracer.grammar
def side_table_object_rand(rng: pf.RNG) -> MeshResult:
    func = pf.control.choice(
        rng,
        [
            (table.side_table_rand, 1.0),
            (storage.storage_side_table_rand, 1.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


def sofa_object_rand(rng: pf.RNG) -> MeshResult:
    func = pf.control.choice(
        rng,
        [
            (sofa.sofa_rand, 1.0),
            (sofa.sofa_with_base_rand, 1.0),
        ],
    )
    return func(rng)


@pf.tracer.grammar
def storage_object_rand(rng: pf.RNG) -> MeshResult:
    func = pf.control.choice(
        rng,
        [
            (storage.storage_rand, 1.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


def standalone_wall_planes(
    length: float = 15.0,
    height: float = 3.0,
) -> list[pf.MeshObject]:
    grid = pf.nodes.geo.mesh_grid(vertices_x=2, vertices_y=2)
    position = pf.nodes.geo.input_position()
    u = (position.x + 0.5) * length
    v = (position.y + 0.5) * height
    geometry = pf.nodes.geo.set_position(
        grid.mesh,
        position=pf.nodes.math.combine_xyz(y=u, z=v),
    )
    wall = pf.nodes.to_mesh_object(geometry)
    wall.item().name = "standalone_wall"
    return [wall]


def retry_place(
    rng: pf.RNG,
    child: MR,
    colliders: ccol.CollisionSet,
    place_fn: Callable[..., None],
    attempts: int = 7,
    accept_fn: Callable[[pf.MeshObject], bool] | None = None,
    **kwargs,
) -> MR | None:
    """Re-pose `child` with `place_fn` until it clears the existing `colliders`
    (and satisfies `accept_fn`, if given), or give up and return None. Reads
    `colliders` but never extends it; keep_non_colliding owns updates and
    intra-batch collisions.
    """
    for r in rng.spawn(attempts):
        place_fn(r, child, **kwargs)
        if ccol.intersection_test(colliders, child.mesh):
            continue
        if accept_fn is not None and not accept_fn(child.mesh):
            continue
        return child
    child.mesh.item().name = child.mesh.item().name + "_FAILED_PLACEMENT"
    return None


def snap_to_wall(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    placement: float | None = None,
    margin: float | None = None,
    child_side: str | None = None,
) -> None:
    if placement is None:
        placement = pf.random.uniform(rng, 0.1, 0.9)
    if margin is None:
        margin = pf.random.clip_gaussian(rng, 0.15, 0.1, 0.1, 0.4)
    if child_side is None:
        child_side = pf.control.choice(rng, [("back", 1.0), ("left", 1.0)])
    snap_to_plane(
        child=child.mesh,
        parent=rng.choice(parents),
        placement=placement,
        margin=margin,
        child_side=child_side,
        parent_side="front",
    )


def snap_back_front(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    placement: float | None = None,
    margin: float | None = None,
) -> None:
    snap_to_wall(
        rng, child, parents, placement=placement, margin=margin, child_side="back"
    )


def back_face_grounded(
    obj: pf.MeshObject,
    colliders: ccol.CollisionSet,
    margin: float,
    eps: float = 0.5,
) -> bool:
    """True iff the corners and center of `obj`'s back (-X local) face have a collider
    directly behind it within (1 + eps) * margin. Rejects placements where part
    of the back overhangs a window/door alcove (no nearby wall behind a sample).
    """
    bmin, bmax = pf.ops.attr.bbox_min_max(obj, global_coords=False)
    samples_local = np.array(
        [
            [bmin[0], bmin[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmax[1], bmax[2]],
            [bmin[0], (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2],
        ]
    )
    mw = np.array(obj.item().matrix_world)
    samples_world = samples_local @ mw[:3, :3].T + mw[:3, 3]
    back_normal = mw[:3, :3] @ np.array([-1.0, 0.0, 0.0])
    back_normal = back_normal / np.linalg.norm(back_normal)

    hits, ray_idx, _ = ccol.raycast(
        colliders, samples_world, np.tile(back_normal, (len(samples_world), 1))
    )
    threshold = (1.0 + eps) * margin
    grounded = np.zeros(len(samples_world), dtype=bool)
    for loc, ri in zip(hits, ray_idx, strict=False):
        if np.linalg.norm(loc - samples_world[ri]) <= threshold:
            grounded[ri] = True
    return bool(grounded.all())


def snap_side_by_side(rng: pf.RNG, child: MR, parents: list[pf.MeshObject]) -> None:
    sides = pf.control.choice(rng, [(("left", "right"), 0.5), (("right", "left"), 0.5)])
    snap_to_plane(
        child.mesh,
        parent=rng.choice(parents),
        placement=rng.uniform(0.05, 0.95),
        margin=pf.random.clip_gaussian(rng, 0.07, 0.06, 0.02, 0.25),
        child_side=sides[0],
        parent_side=sides[1],
    )


def snap_on_top(
    rng: pf.RNG,
    child: MR,
    parents: list[pf.MeshObject],
    xy_frac: tuple[float, float] = (0.5, 0.5),
) -> None:
    parent = rng.choice(list(parents))
    bbox_min, bbox_max = pf.ops.attr.bbox_min_max(parent, global_coords=True)
    child.mesh.item().location = (
        bbox_min[0] + (bbox_max[0] - bbox_min[0]) * xy_frac[0],
        bbox_min[1] + (bbox_max[1] - bbox_min[1]) * xy_frac[1],
        bbox_max[2] + 0.002,
    )


def sofa_lamps_rand(
    rng: pf.RNG,
    sofa_meshes: list[pf.MeshObject],
    side_table_meshes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[MeshResult], list[MeshResult], list[pf.LightObject], ccol.CollisionSet]:
    """Place floor lamps beside `sofa_meshes` and table lamps atop `side_table_meshes`,
    culling against `colliders`. Returns (floor_lamps, table_lamps, lights, colliders);
    each lamp's light is kept in `lights` ~2/3 of the time."""
    rng_lamp, rng_table_lamp = rng.spawn(2)

    n = min(pf.random.randint(rng_lamp, 0, 3), len(sofa_meshes))
    rngs = rng_lamp.spawn(n)
    floor_lamps = [lamp.floor_lamp_rand(rngs[i]) for i in range(n)]
    placed_floor_lamps = []
    for i in range(n):
        floor_lamp = retry_place(
            rngs[i],
            floor_lamps[i],
            colliders,
            snap_side_by_side,
            parents=sofa_meshes,
        )
        placed_floor_lamps.append(floor_lamp)
    floor_lamps, colliders = keep_non_colliding(placed_floor_lamps, colliders)
    logger.info(f"Placed {len(floor_lamps)} floor lamps out of {n} attempts")
    floor_lamp_lights = [r.light for r in floor_lamps if r.light is not None]
    lights: list[pf.LightObject] = []
    lights += pf.control.choice(rng_lamp, [(floor_lamp_lights, 2), ([], 1)])

    n = min(pf.random.randint(rng_table_lamp, 1, 3), len(side_table_meshes))
    rngs = rng_table_lamp.spawn(n)
    table_lamps = [lamp.desk_lamp_rand(rngs[i]) for i in range(n)]
    placed_table_lamps = []
    for i in range(n):
        table_lamp = retry_place(
            rngs[i],
            table_lamps[i],
            colliders,
            snap_on_top,
            parents=side_table_meshes,
        )
        placed_table_lamps.append(table_lamp)
    table_lamps, colliders = keep_non_colliding(placed_table_lamps, colliders)
    logger.info(f"Placed {len(table_lamps)} table lamps out of {n} attempts")
    table_lamp_lights = [r.light for r in table_lamps if r.light is not None]
    lights += pf.control.choice(rng_table_lamp, [(table_lamp_lights, 2), ([], 1)])

    return floor_lamps, table_lamps, lights, colliders
