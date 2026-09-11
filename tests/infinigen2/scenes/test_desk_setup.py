# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import cast

import numpy as np
import procfunc as pf
import pytest

from infinigen2.scenes import desk_setup
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.room import room, room_shape


def test_desk_setup_parameter_ranges() -> None:
    rngs = np.random.default_rng(0).spawn(128)
    poses = [desk_setup._chair_pose_rand(rng) for rng in rngs]
    lamp_fracs = [desk_setup._lamp_xy_frac_rand(rng) for rng in rngs]

    assert all(0.05 <= placement <= 0.95 for placement, _, _ in poses)
    assert all(-0.12 <= margin <= 0.45 for _, margin, _ in poses)

    yaws = [yaw for _, _, yaw in poses]
    assert 0.0 in yaws
    assert any(np.deg2rad(5) <= abs(yaw) <= np.deg2rad(22) for yaw in yaws)

    assert all(0.10 <= x <= 0.35 for x, _ in lamp_fracs)
    assert all(0.05 <= min(y, 1.0 - y) <= 0.20 for _, y in lamp_fracs)
    left_rate = np.mean([y < 0.5 for _, y in lamp_fracs])
    assert left_rate == pytest.approx(0.50, abs=0.15)


def test_desk_setup_places_chair_and_lamp(rng: pf.RNG) -> None:
    dimensions = pf.Vector((0.65, 1.40, 0.74))
    result = desk_setup.desk_setup_rand(rng, dimensions, include_lamp=True)
    desk_min, desk_max = pf.ops.attr.bbox_min_max(result.desk, global_coords=True)
    chair_min, chair_max = pf.ops.attr.bbox_min_max(result.chair, global_coords=True)
    lamp_min, lamp_max = pf.ops.attr.bbox_min_max(result.lamps[0], global_coords=True)

    desk_size = np.array(desk_max) - np.array(desk_min)
    chair_center = (np.array(chair_min) + np.array(chair_max)) / 2
    lamp_center = (np.array(lamp_min) + np.array(lamp_max)) / 2

    np.testing.assert_allclose(desk_size, dimensions, atol=0.01)
    assert abs(chair_min[2]) < 0.01
    assert chair_center[0] > desk_max[0]
    assert lamp_min[2] - desk_max[2] == pytest.approx(0.002, abs=1e-4)
    assert lamp_min[0] >= desk_min[0]
    assert lamp_max[0] <= desk_max[0]
    assert lamp_min[1] >= desk_min[1]
    assert lamp_max[1] <= desk_max[1]
    assert abs(lamp_center[1]) > 0.25
    assert result.all_objects == [result.desk, result.chair, *result.lamps]
    assert len(result.lamps) == len(result.lights) == 1
    assert result.lights[0].item().parent is result.lamps[0].item()


def test_desk_room_selection_odds() -> None:
    rngs = np.random.default_rng(1).spawn(4096)
    desk_rate = np.mean([room._desk_active_rand(rng) for rng in rngs])
    modes = [desk_setup._desk_placement_mode_rand(rng) for rng in rngs]
    wall_rate = np.mean([mode == "against_wall" for mode in modes])

    assert desk_rate == pytest.approx(0.30, abs=0.03)
    assert wall_rate == pytest.approx(0.50, abs=0.03)


def test_freestanding_desk_setup_moves_as_group(rng: pf.RNG) -> None:
    result = desk_setup.desk_setup_in_room_rand(
        rng,
        wall_planes=[],
        room_dimensions=pf.Vector((5.0, 4.0, 3.0)),
        colliders=ccol.collision_set([]),
        placement_mode="freestanding",
        include_lamp=True,
    )

    assert result is not None
    assert result.chair.item().parent is result.desk.item()
    assert all(lamp.item().parent is result.desk.item() for lamp in result.lamps)
    assert len(result.lamps) == 1
    assert 0.75 <= result.desk.item().location.x <= 4.25
    assert 0.60 <= result.desk.item().location.y <= 3.40
    quarter_turns = result.desk.item().rotation_euler.z / (np.pi / 2)
    assert quarter_turns == pytest.approx(round(quarter_turns), abs=1e-6)
    _, desk_max = pf.ops.attr.bbox_min_max(result.desk, global_coords=True)
    lamp_min, _ = pf.ops.attr.bbox_min_max(result.lamps[0], global_coords=True)
    assert lamp_min[2] >= desk_max[2]
    objects = cast(list[pf.Object], result.all_objects)
    assert ccol.n_colliders(ccol.collision_set(objects)) == 3


def test_against_wall_desk_setup_clears_room_shell() -> None:
    dimensions = pf.Vector((5.0, 4.0, 3.0))
    shape = room_shape.room_shape_rand(np.random.default_rng(3), dimensions)
    shell = cast(list[pf.Object], [shape.walls, shape.floor])
    colliders = ccol.collision_set(shell)
    result = desk_setup.desk_setup_in_room_rand(
        np.random.default_rng(7),
        wall_planes=shape.flat_walls,
        room_dimensions=shape.dimensions,
        colliders=colliders,
        placement_mode="against_wall",
        include_lamp=True,
    )

    assert result is not None
    assert all(not ccol.intersection_test(colliders, obj) for obj in result.all_objects)
