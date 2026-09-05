# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import collections.abc

import numpy as np
import procfunc as pf
import pytest

from infinigen2 import generate
from infinigen2.scenes.room import room_surface_features
from infinigen2.shaders import functionality_lists
from infinigen2.util import mesh as mesh_util

WALL_GENERATORS = [
    room_surface_features.wall_plain_rand,
    room_surface_features.wall_windows_rand,
    room_surface_features.wall_painting_grid_rand,
    room_surface_features.wall_board_shelf_rand,
    room_surface_features.wall_storage_shelf_rand,
    room_surface_features.wall_cubby_rand,
    room_surface_features.wall_storage_flush_rand,
    room_surface_features.wall_doors_rand,
    room_surface_features.wall_full_window_rand,
    room_surface_features.wall_arrangement_rand,
]


def _evaluated_world_extent(obj: pf.MeshObject) -> np.ndarray:
    item = obj.item()
    for modifier in item.modifiers:
        modifier.show_viewport = True
        if modifier.type == "SUBSURF":
            modifier.levels = modifier.render_levels
    lo, hi = generate._tight_world_bbox(obj)
    return hi - lo


@pytest.mark.parametrize("func", WALL_GENERATORS, ids=lambda func: func.__name__)
def test_wall_generators_run_standalone(
    func: collections.abc.Callable[[pf.RNG], room_surface_features.WallResult],
    rng: pf.RNG,
) -> None:
    result = func(rng)

    assert result.wall_planes
    assert result.all_objects
    assert len(result.all_objects) == len({id(obj) for obj in result.all_objects})


def test_standalone_wall_has_vertical_metric_uvs(rng: pf.RNG) -> None:
    wall = room_surface_features._standalone_wall_rand(rng, width=1.75, height=2.25)
    uvs = pf.ops.attr.uv_coords(wall)
    bbox_min, bbox_max = pf.ops.attr.bbox_min_max(wall)
    extent = np.array(bbox_max) - np.array(bbox_min)

    assert np.allclose(uvs.max(axis=0) - uvs.min(axis=0), (1.75, 2.25))
    assert np.allclose(extent, (0.0, 1.75, 2.25))
    assert np.allclose(pf.ops.attr.polygon_normals(wall), ((1.0, 0.0, 0.0),))
    assert mesh_util.uv_winding_sign(wall) == -1.0


def test_wall_generator_accepts_precreased_wall(rng: pf.RNG) -> None:
    wall = room_surface_features._standalone_wall_rand(rng, width=3.0, height=2.5)
    pf.ops.attr.write_attribute(wall, 0.0, "crease_edge", domain="EDGE")
    material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    result = room_surface_features.wall_plain_rand(
        rng, wall=wall, wall_material=material
    )
    creases = pf.ops.attr.read_attribute(
        result.wall_planes[0], "crease_edge", domain="EDGE"
    )

    assert np.all(creases == 1.0)


@pytest.mark.parametrize("func", WALL_GENERATORS, ids=lambda func: func.__name__)
def test_wall_generators_preserve_outer_extent_after_subdivision(
    func: collections.abc.Callable[..., room_surface_features.WallResult],
) -> None:
    rng = np.random.default_rng(42)
    wall = room_surface_features._standalone_wall_rand(rng, width=3.0, height=2.5)
    material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    result = func(rng, wall=wall, wall_material=material)
    extent = _evaluated_world_extent(result.wall_planes[0])

    np.testing.assert_allclose(extent[1:], (3.0, 2.5), atol=0.003)


@pytest.mark.parametrize(
    "func",
    [
        room_surface_features.wall_doors_rand,
        room_surface_features.wall_full_window_rand,
    ],
    ids=lambda func: func.__name__,
)
def test_wall_cutouts_fall_back_on_narrow_wall(
    func: collections.abc.Callable[..., room_surface_features.WallResult],
    rng: pf.RNG,
) -> None:
    wall = room_surface_features._standalone_wall_rand(rng, width=0.02, height=2.5)
    material = functionality_lists.wall_material_rand(
        np.random.default_rng(100), pf.nodes.shader.coord().uv
    )

    result = func(rng, wall=wall, wall_material=material)

    assert len(result.wall_planes) == 1
    assert result.decorations == {}
