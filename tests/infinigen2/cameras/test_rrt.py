# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np
import procfunc as pf

from infinigen2.cameras import rrt
from infinigen2.scenes.placement import collision as ccol


def test_rrt_trajectory_starts_at_first_goal() -> None:
    bbox = (np.full(3, -2.0), np.full(3, 2.0))
    start = np.asarray((0.5, -0.25, 0.75))

    camera = rrt.camera_rrt_trajectory(
        np.random.default_rng(0),
        ccol.collision_set([]),
        bbox,
        frame_start=1,
        frame_end=1,
        goals=[start],
    )

    np.testing.assert_allclose(camera.item().location, start)


def test_rrt_path_routes_around_wall() -> None:
    wall = pf.nodes.to_mesh_object(pf.nodes.geo.mesh_cube(size=(0.2, 3.0, 4.0)).mesh)
    colliders = ccol.collision_set([wall])
    bbox = (np.array((-3.0, -3.0, -1.0)), np.array((3.0, 3.0, 1.0)))
    start = np.array((-1.5, 0.0, 0.0))
    goal = np.array((1.5, 0.0, 0.0))

    path = rrt.rrt_path(
        np.random.default_rng(0),
        colliders,
        bbox,
        start,
        goal,
        clearance=0.2,
        step_range=(0.5, 1.0),
    )

    np.testing.assert_allclose(path[-1], goal)
    assert len(path) > 1
    for first, second in zip([start, *path[:-1]], path, strict=True):
        assert rrt._edge_free(first, second, colliders, 0.2)


def test_camera_follow_path_spans_frame_range() -> None:
    points = [np.zeros(3), np.array((1.0, 0.0, 0.0)), np.array((1.0, 3.0, 0.0))]

    camera = rrt.camera_follow_path(np.random.default_rng(0), points, 1, 24)

    curves = camera.item().animation_data.action.fcurves
    location_frames = sorted(
        {
            point.co.x
            for curve in curves
            if curve.data_path == "location"
            for point in curve.keyframe_points
        }
    )
    np.testing.assert_allclose(location_frames, [1.0, 6.75, 24.0])
