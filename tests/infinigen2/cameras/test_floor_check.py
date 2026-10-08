# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import functools

import bpy
import numpy as np
import procfunc as pf
import pytest

import infinigen2.scenes.placement.collision as ccol
from infinigen2.cameras import camera_cube_free_space_check, monocular
from infinigen2.util.errors import RejectedScene

DIMENSIONS = pf.Vector((6.0, 6.0, 3.0))


def _rect(size_x: float, size_y: float, center_x: float, center_y: float):
    grid = pf.nodes.geo.mesh_grid(
        vertices_x=2, vertices_y=2, size_x=size_x, size_y=size_y
    )
    return pf.nodes.geo.set_position(
        geometry=grid.mesh, offset=(center_x, center_y, 0.0)
    )


def _notched_floor() -> pf.MeshObject:
    """An L outline over the 6x6 bounding box, leaving the far quadrant a void
    that a camera must not be placed over."""
    pf.ops.object.clear_scene()
    joined = pf.nodes.geo.join_geometry(
        [_rect(6.0, 3.0, 3.0, 1.5), _rect(3.0, 3.0, 1.5, 4.5)]
    )
    return pf.nodes.to_mesh_object(joined)


def _camera_accept_pred(
    camera: pf.CameraObject,
    colliders: ccol.CollisionSet,
    floor_colliders: ccol.CollisionSet,
) -> bool:
    if not camera_cube_free_space_check(camera, colliders, forward_clearance=0.75):
        return False
    origin = np.array([camera.item().matrix_world.translation])
    _hits, ray_indices, _tri_indices = ccol.raycast(
        floor_colliders, origin, np.array([[0.0, 0.0, -1.0]])
    )
    return len(ray_indices) > 0


def _assert_trajectory_on_floor(
    camera: pf.CameraObject,
    floor_colliders: ccol.CollisionSet,
    frames: int,
) -> None:
    empty_colliders = ccol.collision_set([])
    for frame in range(frames + 1):
        bpy.context.scene.frame_set(frame)
        assert _camera_accept_pred(camera, empty_colliders, floor_colliders), (
            f"pan pose at frame {frame} is over the void"
        )


def test_linear_pan_keeps_every_pose_over_the_floor():
    floor = _notched_floor()
    colliders = ccol.collision_set([floor])
    floor_colliders = ccol.collision_set([floor])
    accept_pred = functools.partial(
        _camera_accept_pred, floor_colliders=floor_colliders
    )
    frames = 12

    for seed in range(3):
        cameras = monocular.camera_linear_pan_rand(
            rng=np.random.default_rng(seed),
            objects=[floor],
            colliders=colliders,
            bbox=(np.zeros(3), np.array(DIMENSIONS)),
            frame_start=0,
            frame_end=frames,
            accept_pred=accept_pred,
        )
        _assert_trajectory_on_floor(cameras[0], floor_colliders, frames)


def test_linear_pan_consults_the_caller_predicate():
    floor = _notched_floor()
    colliders = ccol.collision_set([floor])
    calls = []

    def reject(camera: pf.CameraObject, colliders: ccol.CollisionSet) -> bool:
        calls.append(camera)
        return False

    with pytest.raises(RejectedScene):
        monocular.camera_linear_pan_rand(
            rng=np.random.default_rng(0),
            objects=[floor],
            colliders=colliders,
            bbox=(np.zeros(3), np.array(DIMENSIONS)),
            frame_start=0,
            frame_end=4,
            max_tries=3,
            accept_pred=reject,
        )

    assert calls, "accept_pred was never consulted"
