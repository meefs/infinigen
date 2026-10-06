# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np
import procfunc as pf
from shapely.geometry import Point, Polygon

from infinigen2.scenes.house import floor_plan
from infinigen2.scenes.house import unfurnished as house
from infinigen2.scenes.house.floor_mesh import house_floor_profile_to_planes


def _rectangle_profile(
    rng, dimensions: tuple[float, float], room_count: int, wall_thickness=None
) -> floor_plan.HouseFloorProfileResult:
    width, depth = dimensions
    ring = ((0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth))
    return floor_plan.room_solve(
        rng, (ring,), room_count, wall_thickness=wall_thickness
    )


def _world_vertices(obj: pf.MeshObject) -> np.ndarray:
    item = obj.item()
    return np.asarray([item.matrix_world @ vertex.co for vertex in item.data.vertices])


def test_wall_planes_overlap_floor_and_ceiling_without_xy_expansion(rng) -> None:
    profile = _rectangle_profile(rng, (12, 10), 4, wall_thickness=0.3)
    shape = house_floor_profile_to_planes(
        profile.room_boundaries, profile.walls, height=2.75
    )
    bounds = [
        pf.ops.attr.bbox_min_max(plane.obj, global_coords=True)
        for plane in shape.planes_interior + shape.planes_exterior
    ]
    lows = np.min([np.asarray(low) for low, _high in bounds], axis=0)
    highs = np.max([np.asarray(high) for _low, high in bounds], axis=0)

    np.testing.assert_allclose(lows[:2], (0.15, 0.15), atol=1e-5)
    np.testing.assert_allclose(
        highs[:2], np.asarray(shape.dimensions[:2]) - 0.15, atol=1e-5
    )
    np.testing.assert_allclose(lows[2], -0.01, atol=1e-5)
    np.testing.assert_allclose(highs[2], 2.76, atol=1e-5)
    floor_low = min(pf.ops.attr.bbox_min_max(obj)[0][2] for obj in shape.floors)
    ceiling_high = max(pf.ops.attr.bbox_min_max(obj)[1][2] for obj in shape.ceilings)
    assert lows[2] < floor_low
    assert highs[2] > ceiling_high


def test_house_unfurnished_rand_returns_rooms_with_shell_and_door_graph() -> None:
    result = house.house_unfurnished_rand(
        np.random.default_rng(0),
        dimensions=(12.0, 10.0),
        room_count=4,
        height=2.75,
        wall_thickness=0.3,
    )
    walls = [obj for room in result.rooms for obj in room.walls]
    bounds = [pf.ops.attr.bbox_min_max(wall, global_coords=True) for wall in walls]
    lows = np.min([np.asarray(low) for low, _high in bounds], axis=0)
    highs = np.max([np.asarray(high) for _low, high in bounds], axis=0)

    assert np.all(lows > -0.05) and np.all(highs < (12.05, 10.05, 2.8))
    assert len(result.rooms) == 4
    assert result.environment is not None
    door_clearances = [
        obj
        for obj in result.colliders.objs
        if obj.item().name.startswith("house_door_affordance")
    ]
    floors_and_ceilings = [
        obj for room in result.rooms for obj in (room.floor, room.ceiling)
    ]
    assert all(obj in result.colliders.objs for obj in floors_and_ceilings)
    assert door_clearances
    assert all(obj not in result.all_objects for obj in door_clearances)
    assert all(obj.item().hide_render for obj in door_clearances)
    room_lights = [light for room in result.rooms for light in room.lights]
    assert room_lights
    assert all(light in result.lights for light in room_lights)

    graph_keys = set()
    for room in result.rooms:
        assert room.boundary_rings
        assert room.doors.keys() <= room.neighbors.keys()
        assert room.doorway_centers.keys() == room.neighbors.keys()
        required = [room.floor, room.ceiling, *room.walls, *room.doors.values()]
        assert all(obj in room.all_objects for obj in required)
        assert all(obj in result.all_objects for obj in room.all_objects)
        for wall_index, neighbor_index in room.neighbors.items():
            neighbor = result.rooms[neighbor_index]
            assert neighbor.neighbors[wall_index] == room.room_index
            assert (
                neighbor.doorway_centers[wall_index] == room.doorway_centers[wall_index]
            )
            if wall_index in room.doors:
                assert neighbor.doors[wall_index] is room.doors[wall_index]
            graph_keys.add(wall_index)
    assert len(graph_keys) == len(result.rooms) - 1


def test_house_default_camera_stays_in_a_room_and_faces_inward() -> None:
    result = house.house_unfurnished_rand(
        np.random.default_rng(18), dimensions=(7.0, 6.0), room_count=4
    )
    camera = result.cameras[0]
    point = Point(camera.item().location.x, camera.item().location.y)
    polygons = []
    floor = result.rooms[0].floor
    vertices = _world_vertices(floor)
    polygons = [
        Polygon(vertices[list(face.vertices), :2])
        for face in floor.item().data.polygons
    ]

    assert any(polygon.covers(point) for polygon in polygons)
    direction = camera.item().matrix_world.to_quaternion() @ pf.Vector((0, 0, -1))
    room_center = np.mean([polygon.centroid.coords[0] for polygon in polygons], axis=0)
    inward = (
        pf.Vector((*room_center, camera.item().location.z)) - camera.item().location
    )
    assert direction.dot(inward) > 0
