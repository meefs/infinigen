# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np
import procfunc as pf

from infinigen2.cameras import rrt
from infinigen2.scenes.placement import collision as ccol


def test_rrt_camera_accepts_explicit_start_location() -> None:
    bounds = pf.ops.primitives.mesh_cube(size=4.0)
    start = np.asarray((0.5, -0.25, 0.75))

    camera = rrt.camera_rrt(
        np.random.default_rng(0),
        ccol.collision_set([]),
        [bounds],
        start_location=start,
    )

    np.testing.assert_allclose(camera.item().location, start)


def test_rrt_camera_retries_goal_location_sampler() -> None:
    bounds = pf.ops.primitives.mesh_cube(size=4.0)
    start = np.asarray((0.0, 0.0, 0.0))
    goal = np.asarray((1.0, 0.0, 0.0))
    candidates = iter((np.asarray((3.0, 0.0, 0.0)), goal))

    def sample_goal(_rng: pf.RNG) -> np.ndarray:
        return next(candidates)

    camera = rrt.camera_rrt(
        np.random.default_rng(0),
        ccol.collision_set([]),
        [bounds],
        start_location=start,
        goal_location_samplers=[sample_goal],
        frame_start=1,
        frame_end=2,
        speed_mps_range=(24.0, 24.0),
    )

    np.testing.assert_allclose(camera.item().location, goal)


def test_rrt_camera_stretches_required_goals_to_frame_end() -> None:
    bounds = pf.ops.primitives.mesh_cube(size=4.0)
    goal = np.asarray((1.0, 0.0, 0.0))

    camera = rrt.camera_rrt(
        np.random.default_rng(0),
        ccol.collision_set([]),
        [bounds],
        start_location=np.zeros(3),
        goal_location_samplers=[lambda _rng: goal],
        wander_after_goals=False,
        frame_start=1,
        frame_end=24,
        speed_mps_range=(24.0, 24.0),
    )

    curves = camera.item().animation_data.action.fcurves
    location_frames = {
        point.co.x
        for curve in curves
        if curve.data_path == "location"
        for point in curve.keyframe_points
    }
    assert location_frames == {1.0, 24.0}
