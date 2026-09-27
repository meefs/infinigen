# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import procfunc as pf

from infinigen2.objects import (
    bathroom_hardware,
    bathtub,
    sink,
    storage,
    tap,
    toilet,
    wall_art,
)
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.placement.retry import repeat_attempts
from infinigen2.scenes.placement.snap import snap_to_plane
from infinigen2.scenes.setup_utils import (
    BareMeshResult,
    MeshResult,
    back_face_grounded,
    retry_place,
    snap_back_front,
    standalone_wall_planes,
)
from infinigen2.util.errors import RejectedScene
from infinigen2.util.scene_cleanup import delete_object

if TYPE_CHECKING:
    from infinigen2.scenes.room.wall_base import WallResult

__all__ = [
    "BathroomSetupResult",
    "bathroom_setup_accept_pred",
    "bathroom_setup_rand",
    "bathroom_wall_arrangement_rand",
]

logger = logging.getLogger(__name__)

_WALL_MARGIN = 0.003
_MIN_SINK_TOP_HEIGHT = 0.8
_MAX_SINK_TOP_HEIGHT = 0.9
_MAX_TAP_HEIGHT = 0.4
_MIN_TAP_BACK_MARGIN = bathtub.MIN_BACK_MARGIN * 0.1
_MAX_TAP_BACK_MARGIN = bathtub.MIN_BACK_MARGIN
_TAP_BACK_OFFSET = 0.05
_TAP_SURFACE_INSET = 0.01


class _BathtubSetupResult(NamedTuple):
    bathtubs: list[MeshResult]
    taps: list[MeshResult]
    hardware: list[MeshResult]


class _BathroomFixtures(NamedTuple):
    sink_setup: BathroomSinkSetupResult
    toilets: list[MeshResult]
    bathtub_setup: _BathtubSetupResult


class BathroomSetupResult(NamedTuple):
    named_objects: dict[str, list[pf.MeshObject]]
    all_objects: list[pf.MeshObject]
    colliders: ccol.CollisionSet
    temporary_objects: list[pf.MeshObject]
    storage_containers: list[pf.MeshObject]
    storage_supports: list[pf.MeshObject]


def _snap_fixture_against_wall(
    rng: pf.RNG,
    child: MeshResult,
    parents: list[pf.MeshObject],
    margin: float,
) -> None:
    snap_back_front(
        rng,
        child,
        parents,
        placement=pf.random.uniform(rng, 0.0, 1.0),
        margin=margin,
    )


def _place_tap_at_back(
    tap_obj: pf.MeshObject,
    fixture: pf.MeshObject,
    back_offset: float = _TAP_BACK_OFFSET,
) -> None:
    minimum, maximum = pf.ops.attr.bbox_min_max(fixture, global_coords=False)
    location = pf.Vector(
        (
            minimum[0] + back_offset,
            (minimum[1] + maximum[1]) / 2.0,
            maximum[2] - _TAP_SURFACE_INSET,
        )
    )
    tap_obj.item().matrix_world = fixture.item().matrix_world @ pf.Matrix.Translation(
        location
    )


@pf.tracer.grammar
def bathroom_wall_arrangement_rand(
    rng: pf.RNG,
    wall: pf.MeshObject | None = None,
    window_obj: pf.MeshObject | None = None,
    window_portal: pf.LightObject | None = None,
    wall_material: pf.Material | None = None,
    window_spacing: float | None = None,
    window_bottom: float | None = None,
    wall_thickness: float = 0.05,
) -> WallResult:
    from infinigen2.scenes.room.wall_base import wall_plain_rand  # keep-local
    from infinigen2.scenes.room.wall_cutouts import (  # keep-local
        wall_cubby_rand,
        wall_full_window_rand,
        wall_painting_grid_rand,
        wall_windows_rand,
    )
    from infinigen2.scenes.room.wall_mounts import (  # keep-local
        wall_board_shelf_rand,
        wall_storage_flush_rand,
    )

    rng_choice, rng_feature = rng.spawn(2)
    option = pf.control.choice(
        rng_choice,
        [
            (
                functools.partial(
                    wall_plain_rand,
                    wall=wall,
                    wall_material=wall_material,
                    wall_thickness=wall_thickness,
                ),
                3.0,
            ),
            (
                functools.partial(
                    wall_windows_rand,
                    wall=wall,
                    wall_material=wall_material,
                    window_obj=window_obj,
                    window_portal=window_portal,
                    window_spacing=window_spacing,
                    window_bottom=window_bottom,
                    wall_thickness=wall_thickness,
                ),
                3.0,
            ),
            (
                functools.partial(
                    wall_painting_grid_rand,
                    wall=wall,
                    wall_material=wall_material,
                    wall_thickness=wall_thickness,
                ),
                1.0,
            ),
            (
                functools.partial(
                    wall_board_shelf_rand,
                    wall=wall,
                    wall_material=wall_material,
                    wall_thickness=wall_thickness,
                ),
                0.75,
            ),
            (
                functools.partial(
                    wall_storage_flush_rand,
                    wall=wall,
                    wall_material=wall_material,
                    wall_thickness=wall_thickness,
                ),
                0.5,
            ),
            (
                functools.partial(
                    wall_cubby_rand,
                    wall=wall,
                    wall_material=wall_material,
                    wall_thickness=wall_thickness,
                ),
                0.5,
            ),
            (
                functools.partial(
                    wall_full_window_rand,
                    wall=wall,
                    wall_material=wall_material,
                    wall_thickness=wall_thickness,
                ),
                0.6,
            ),
        ],
    )
    return option(rng_feature)


def _toilet_against_wall_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    clearance_fixtures: tuple[pf.MeshObject, ...],
) -> list[MeshResult]:
    rng_toilet, rng_place = rng.spawn(2)
    result = toilet.toilet_rand(rng_toilet)
    minimum, _ = pf.ops.attr.bbox_min_max(result.mesh, global_coords=False)
    pf.ops.object.set_transform(result.mesh, location=(0.0, 0.0, 0.001 - minimum[2]))
    valid = functools.partial(
        bathroom_fixture_placement_valid,
        wall_colliders=ccol.collision_set(wall_planes),
        colliders=colliders,
        wall_margin=_WALL_MARGIN,
        clearance_fixtures=clearance_fixtures,
    )
    placed = retry_place(
        rng_place,
        result,
        colliders,
        _snap_fixture_against_wall,
        attempts=64,
        accept_fn=valid,
        parents=wall_planes,
        margin=_WALL_MARGIN,
    )
    if placed is None:
        delete_object(result.mesh.item())
        return []
    return [placed]


def _no_bathtub_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> _BathtubSetupResult:
    del rng, wall_planes, colliders
    return _BathtubSetupResult([], [], [])


def _bathtub_wall_hardware_rand(rng: pf.RNG) -> list[MeshResult]:
    return [bathroom_hardware.bathroom_hardware_rand(rng)]


def _place_bathtub_composite(
    rng: pf.RNG,
    child: MeshResult,
    parents: list[pf.MeshObject],
    child_side: str,
    margin: float,
    child_matrix: pf.Matrix,
    tap_result: MeshResult,
    tap_matrix: pf.Matrix,
    hardware: list[MeshResult],
) -> None:
    rng_wall, rng_place, rng_hardware_height = rng.spawn(3)
    wall = rng_wall.choice(parents)
    placement = pf.random.uniform(rng_place, 0.0, 1.0)
    snap_to_plane(
        child.mesh,
        wall,
        placement=placement,
        margin=margin,
        child_side=child_side,
        parent_side="front",
    )
    delta = child.mesh.item().matrix_world @ child_matrix.inverted()
    tap_result.mesh.item().matrix_world = delta @ tap_matrix
    bathtub_top = pf.ops.attr.bbox_min_max(child.mesh, global_coords=True)[1][2]
    for result in hardware:
        hardware_minimum, _ = pf.ops.attr.bbox_min_max(result.mesh, global_coords=False)
        location = pf.Vector(result.mesh.item().location)
        location.z = bathtub_top + pf.random.uniform(rng_hardware_height, 0.15, 0.4)
        location.z -= hardware_minimum[2]
        pf.ops.object.set_transform(result.mesh, location=location)
        snap_to_plane(
            result.mesh,
            wall,
            placement=placement,
            margin=margin,
            child_side="back",
            parent_side="front",
        )


def _bathtub_components_valid(
    bathtub_mesh: pf.MeshObject,
    tap_result: MeshResult,
    hardware: list[MeshResult],
    colliders: ccol.CollisionSet,
    wall_colliders: ccol.CollisionSet,
    margin: float,
    child_side: str,
) -> bool:
    if not back_face_grounded(
        bathtub_mesh,
        wall_colliders,
        margin,
        side=child_side,
    ):
        return False
    if ccol.intersection_test(colliders, tap_result.mesh):
        return False
    component_colliders = ccol.collision_set(
        colliders.objs + [bathtub_mesh, tap_result.mesh], cache=colliders
    )
    for result in hardware:
        if not back_face_grounded(result.mesh, wall_colliders, margin):
            return False
        if ccol.intersection_test(component_colliders, result.mesh):
            return False
    return True


def _bathtub_against_wall_side_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    child_side: str,
) -> _BathtubSetupResult | None:
    (
        rng_bathtub,
        rng_tap,
        rng_hardware_choice,
        rng_hardware,
        rng_place,
        rng_tap_placement,
    ) = rng.spawn(6)
    result = bathtub.bathtub_rand(rng_bathtub)
    result.mesh.item().rotation_mode = "XYZ"
    pf.ops.object.set_transform(result.mesh, rotation_euler=(0.0, 0.0, 0.0))
    tap_result = tap.tap_rand(rng_tap)
    tap_minimum, _ = pf.ops.attr.bbox_min_max(tap_result.mesh, global_coords=False)
    tap_back_margin = pf.random.uniform(
        rng_tap_placement, _MIN_TAP_BACK_MARGIN, _MAX_TAP_BACK_MARGIN
    )
    tap_back_offset = tap_back_margin - tap_minimum[0]
    _place_tap_at_back(tap_result.mesh, result.mesh, tap_back_offset)
    include_hardware = pf.control.choice(
        rng_hardware_choice, [(True, 1.0), (False, 1.0)]
    )
    hardware = []
    if include_hardware:
        hardware = _bathtub_wall_hardware_rand(rng_hardware)
    result_matrix = result.mesh.item().matrix_world.copy()
    tap_matrix = tap_result.mesh.item().matrix_world.copy()
    placed = retry_place(
        rng_place,
        result,
        colliders,
        _place_bathtub_composite,
        attempts=64,
        accept_fn=functools.partial(
            _bathtub_components_valid,
            tap_result=tap_result,
            hardware=hardware,
            colliders=colliders,
            wall_colliders=ccol.collision_set(wall_planes),
            margin=_WALL_MARGIN,
            child_side=child_side,
        ),
        parents=wall_planes,
        child_side=child_side,
        margin=_WALL_MARGIN,
        child_matrix=result_matrix,
        tap_result=tap_result,
        tap_matrix=tap_matrix,
        hardware=hardware,
    )
    if placed is None:
        delete_object(result.mesh.item())
        delete_object(tap_result.mesh.item())
        for item in hardware:
            delete_object(item.mesh.item())
        return None
    return _BathtubSetupResult([result], [tap_result], hardware)


def _bathtub_against_wall_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> _BathtubSetupResult | None:
    rng_side, rng_bathtub = rng.spawn(2)
    child_side = pf.control.choice(rng_side, [("left", 1.0), ("right", 1.0)])
    return _bathtub_against_wall_side_rand(
        rng_bathtub, wall_planes, colliders, child_side
    )


def _match_hardware_height(
    rng: pf.RNG,
    hardware: MeshResult,
    target: pf.MeshObject,
) -> None:
    target_minimum, target_maximum = pf.ops.attr.bbox_min_max(
        target, global_coords=False
    )
    hardware_minimum, hardware_maximum = pf.ops.attr.bbox_min_max(
        hardware.mesh, global_coords=False
    )
    target_center = (target_minimum[2] + target_maximum[2]) / 2.0
    hardware_center = (hardware_minimum[2] + hardware_maximum[2]) / 2.0
    location = pf.Vector(target.item().location)
    location.z += target_center - hardware_center
    location.z += pf.random.uniform(rng, -0.2, 0.2)
    pf.ops.object.set_transform(
        hardware.mesh,
        location=location,
        rotation_euler=target.item().rotation_euler,
    )


def _snap_hardware_side_rand(
    rng: pf.RNG,
    hardware: MeshResult,
    parents: list[pf.MeshObject],
    child_side: str,
    parent_side: str,
) -> None:
    rng_target, rng_height, rng_gap = rng.spawn(3)
    target = rng_target.choice(parents)
    _match_hardware_height(rng_height, hardware, target)
    snap_to_plane(
        hardware.mesh,
        target,
        placement=0.0,
        margin=pf.random.uniform(rng_gap, 0.04, 0.18),
        child_side=child_side,
        parent_side=parent_side,
    )


def _snap_hardware_above_rand(
    rng: pf.RNG,
    hardware: MeshResult,
    parents: list[pf.MeshObject],
) -> None:
    rng_target, rng_offset, rng_gap = rng.spawn(3)
    target = rng_target.choice(parents)
    target_minimum, target_maximum = pf.ops.attr.bbox_min_max(
        target, global_coords=False
    )
    target_width = target_maximum[1] - target_minimum[1]
    lateral_offset = target_width * pf.random.uniform(rng_offset, -0.35, 0.35)
    target_matrix = target.item().matrix_world
    location = pf.Vector(target.item().location)
    location += target_matrix.to_3x3() @ pf.Vector((0.0, lateral_offset, 0.0))
    pf.ops.object.set_transform(
        hardware.mesh,
        location=location,
        rotation_euler=target.item().rotation_euler,
    )
    constraint_axis = target_matrix.to_3x3() @ pf.Vector((0.0, 1.0, 0.0))
    snap_to_plane(
        hardware.mesh,
        target,
        placement=0.0,
        margin=pf.random.uniform(rng_gap, 0.04, 0.18),
        child_side="bottom",
        parent_side="top",
        constraint_axis=constraint_axis,
    )


def _snap_hardware_adjacent_rand(
    rng: pf.RNG,
    hardware: MeshResult,
    parents: list[pf.MeshObject],
) -> None:
    rng_choice, rng_left, rng_right, rng_above = rng.spawn(4)
    place_func = pf.control.choice(
        rng_choice,
        [
            (
                functools.partial(
                    _snap_hardware_side_rand,
                    rng_left,
                    child_side="right",
                    parent_side="left",
                ),
                1.0,
            ),
            (
                functools.partial(
                    _snap_hardware_side_rand,
                    rng_right,
                    child_side="left",
                    parent_side="right",
                ),
                1.0,
            ),
            (functools.partial(_snap_hardware_above_rand, rng_above), 1.0),
        ],
    )
    place_func(hardware, parents)


def _bathroom_hardware_objects_rand(
    rng: pf.RNG,
    parents: list[pf.MeshObject],
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[MeshResult], ccol.CollisionSet]:
    if not parents:
        return [], colliders
    rng_count, rng_hardware = rng.spawn(2)
    n = pf.random.randint(rng_count, 0, 4)
    placed_hardware = []
    wall_colliders = ccol.collision_set(wall_planes)
    for i, rng_item in enumerate(rng_hardware.spawn(n)):
        rng_generate, rng_place = rng_item.spawn(2)
        hardware = bathroom_hardware.bathroom_hardware_rand(rng_generate)
        grounded = functools.partial(
            back_face_grounded,
            colliders=wall_colliders,
            margin=_WALL_MARGIN,
        )
        placed = retry_place(
            rng_place,
            hardware,
            colliders,
            _snap_hardware_adjacent_rand,
            attempts=16,
            accept_fn=grounded,
            parents=parents,
        )
        if placed is None:
            delete_object(hardware.mesh.item())
            continue
        hardware.mesh.item().name = f"bathroom_hardware_{i}"
        placed_hardware.append(placed)
        colliders = ccol.collision_set(colliders.objs + [placed.mesh], cache=colliders)
    logger.info(
        "Placed %d bathroom hardware out of %d attempts", len(placed_hardware), n
    )
    return placed_hardware, colliders


def _fixture_result(
    fixtures: _BathroomFixtures,
    colliders: ccol.CollisionSet,
) -> BathroomSetupResult:
    sink_setup = fixtures.sink_setup
    sinks = [result.mesh for result in sink_setup.bathroom_sinks]
    sink_supports = [result.mesh for result in sink_setup.sink_supports]
    sink_cabinets = [
        support for support in sink_supports if support.item().name == "sink_cabinet"
    ]
    wall_storage = [result.mesh for result in sink_setup.wall_storage]
    bathtubs = [result.mesh for result in fixtures.bathtub_setup.bathtubs]
    named_objects = {
        "bathroom_sink": _name_objects(sinks, "bathroom_sink"),
        "sink_tap": _name_objects(
            [result.mesh for result in sink_setup.sink_taps], "sink_tap"
        ),
        "sink_support": _name_objects(sink_supports, "sink_support"),
        "mirror": _name_objects(
            [result.mesh for result in sink_setup.mirrors], "mirror"
        ),
        "bathroom_wall_storage": _name_objects(
            wall_storage,
            "bathroom_wall_storage",
        ),
        "toilet": _name_objects([result.mesh for result in fixtures.toilets], "toilet"),
        "bathtub": _name_objects(bathtubs, "bathtub"),
        "bathtub_tap": _name_objects(
            [result.mesh for result in fixtures.bathtub_setup.taps], "bathtub_tap"
        ),
        "bathtub_hardware": _name_objects(
            [result.mesh for result in fixtures.bathtub_setup.hardware],
            "bathtub_hardware",
        ),
        "bathroom_hardware": [],
    }
    all_objects = [obj for objects in named_objects.values() for obj in objects]
    storage_containers = sinks + bathtubs + sink_cabinets + wall_storage
    storage_supports = sink_cabinets + wall_storage
    return BathroomSetupResult(
        named_objects=named_objects,
        all_objects=all_objects,
        colliders=colliders,
        temporary_objects=[],
        storage_containers=storage_containers,
        storage_supports=storage_supports,
    )


def bathroom_setup_accept_pred(
    setup: BathroomSetupResult,
    colliders: ccol.CollisionSet,
    probe_size: float = 0.5,
    probe_height: float = 1.8,
    probe_gap: float = 0.05,
    probe_floor_clearance: float = 0.01,
) -> bool:
    sinks = setup.named_objects["bathroom_sink"]
    toilets = setup.named_objects["toilet"]
    if not sinks or not toilets:
        return False
    fixtures = sinks + toilets
    all_obstacles = colliders.objs + setup.all_objects
    for fixture in fixtures:
        obstacles = ccol.collision_set(
            [obj for obj in all_obstacles if obj.item() is not fixture.item()],
            cache=colliders,
        )
        if not bathroom_fixture_front_clear(
            fixture,
            obstacles,
            probe_size,
            probe_height,
            probe_gap,
            probe_floor_clearance,
        ):
            return False
    return True


def _bathroom_fixtures_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    sink_obj: pf.MeshObject | None,
    include_bathtub: bool,
) -> _BathroomFixtures | None:
    rng_sink, rng_bathtub, rng_toilet = rng.spawn(3)
    sink_setup = _bathroom_sink_components_rand(
        rng_sink,
        sink_obj=sink_obj,
        wall_planes=wall_planes,
        room_dimensions=room_dimensions,
        colliders=colliders,
    )
    if not sink_setup.bathroom_sinks:
        _delete_objects(sink_setup.all_objects, sink_obj)
        return None
    colliders = ccol.collision_set(
        colliders.objs + sink_setup.all_objects, cache=colliders
    )
    if include_bathtub:
        bathtub_setup = _bathtub_against_wall_rand(rng_bathtub, wall_planes, colliders)
    else:
        bathtub_setup = _no_bathtub_setup_rand(rng_bathtub, wall_planes, colliders)
    if bathtub_setup is None:
        _delete_objects(sink_setup.all_objects, sink_obj)
        return None
    bathtub_objects = (
        [result.mesh for result in bathtub_setup.bathtubs]
        + [result.mesh for result in bathtub_setup.taps]
        + [result.mesh for result in bathtub_setup.hardware]
    )
    colliders = ccol.collision_set(colliders.objs + bathtub_objects, cache=colliders)
    toilets = _toilet_against_wall_rand(
        rng_toilet,
        wall_planes,
        colliders,
        tuple(result.mesh for result in sink_setup.bathroom_sinks),
    )
    return _BathroomFixtures(sink_setup, toilets, bathtub_setup)


def _delete_objects(
    objects: list[pf.MeshObject],
    preserve: pf.MeshObject | None,
) -> None:
    for obj in objects:
        if preserve is not None and obj.item() is preserve.item():
            continue
        delete_object(obj.item())


def _name_materials(obj: pf.MeshObject, base: str) -> None:
    for i, slot in enumerate(obj.item().material_slots):
        if slot.material is not None:
            slot.material.name = f"{base}_{i}"


def _name_objects(
    objects: list[pf.MeshObject],
    category: str,
) -> list[pf.MeshObject]:
    for i, obj in enumerate(objects):
        name = f"{category}.{i:02d}"
        obj.item().name = name
        _name_materials(obj, name)
    return objects


def _finalize_bathroom_setup_rand(
    rng: pf.RNG,
    setup: BathroomSetupResult,
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> BathroomSetupResult:
    clearances = [
        bathroom_fixture_clearance(obj)
        for obj in setup.named_objects["bathroom_sink"] + setup.named_objects["toilet"]
    ]
    setup_colliders = ccol.collision_set(
        colliders.objs + setup.all_objects + clearances, cache=colliders
    )
    hardware_targets = [
        obj
        for category in (
            "bathroom_sink",
            "mirror",
            "bathroom_wall_storage",
            "toilet",
        )
        for obj in setup.named_objects[category]
    ]
    hardware, setup_colliders = _bathroom_hardware_objects_rand(
        rng,
        hardware_targets,
        wall_planes,
        setup_colliders,
    )
    named_objects = dict(setup.named_objects)
    named_objects["bathroom_hardware"] = _name_objects(
        [result.mesh for result in hardware], "bathroom_hardware"
    )
    all_objects = [obj for objects in named_objects.values() for obj in objects]
    return BathroomSetupResult(
        named_objects=named_objects,
        all_objects=all_objects,
        colliders=setup_colliders,
        temporary_objects=clearances,
        storage_containers=setup.storage_containers,
        storage_supports=setup.storage_supports,
    )


def _bathroom_setup_attempt_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    sink_obj: pf.MeshObject | None,
    include_bathtub: bool,
) -> BathroomSetupResult | None:
    rng_fixtures, rng_hardware = rng.spawn(2)
    fixtures = _bathroom_fixtures_rand(
        rng_fixtures,
        wall_planes,
        room_dimensions,
        colliders,
        sink_obj,
        include_bathtub,
    )
    if fixtures is None:
        return None
    setup = _fixture_result(fixtures, colliders)
    if bathroom_setup_accept_pred(setup, colliders):
        return _finalize_bathroom_setup_rand(
            rng_hardware, setup, wall_planes, colliders
        )
    _delete_objects(setup.all_objects, sink_obj)
    return None


def _repeat_bathroom_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    sink_obj: pf.MeshObject | None,
) -> BathroomSetupResult:
    rng_bathtub_choice, rng_attempts, rng_without_bathtub = rng.spawn(3)
    include_bathtub = pf.control.choice(rng_bathtub_choice, [(True, 1.0), (False, 1.0)])
    setup = repeat_attempts(
        _bathroom_setup_attempt_rand,
        rng_attempts,
        attempts=12,
        wall_planes=wall_planes,
        room_dimensions=room_dimensions,
        colliders=colliders,
        sink_obj=sink_obj,
        include_bathtub=include_bathtub,
    )
    if setup is None and include_bathtub:
        setup = repeat_attempts(
            _bathroom_setup_attempt_rand,
            rng_without_bathtub,
            attempts=12,
            wall_planes=wall_planes,
            room_dimensions=room_dimensions,
            colliders=colliders,
            sink_obj=sink_obj,
            include_bathtub=False,
        )
    if setup is None:
        raise RejectedScene("Could not place a valid bathroom setup")
    return setup


def _bathroom_setup_demo_rand(
    rng: pf.RNG,
    sink: pf.MeshObject | None,
) -> BathroomSetupResult:
    room_dimensions = pf.Vector((5.0, 5.0, 3.0))
    wall = standalone_wall_planes(
        length=room_dimensions.y,
        height=room_dimensions.z,
    )[0]
    colliders = ccol.collision_set([wall])
    setup = _repeat_bathroom_setup_rand(
        rng,
        [wall],
        room_dimensions,
        colliders,
        sink,
    )
    delete_object(wall.item())
    ground = pf.ops.primitives.mesh_single_vertex()
    ground.item().name = "bathroom_setup_ground"
    named_objects = dict(setup.named_objects)
    named_objects["bathroom_setup_ground"] = [ground]
    all_objects = [obj for objects in named_objects.values() for obj in objects]
    colliders = ccol.collision_set(all_objects + setup.temporary_objects)
    return setup._replace(
        named_objects=named_objects,
        all_objects=all_objects,
        colliders=colliders,
    )


@pf.tracer.grammar
def bathroom_setup_rand(
    rng: pf.RNG,
    sink: pf.MeshObject | None = None,
    wall_planes: list[pf.MeshObject] | None = None,
    room_dimensions: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> BathroomSetupResult:
    if wall_planes is None:
        return _bathroom_setup_demo_rand(rng, sink)
    if room_dimensions is None:
        room_dimensions = pf.Vector((5.0, 5.0, 3.0))
    if colliders is None:
        colliders = ccol.collision_set(wall_planes)
    return _repeat_bathroom_setup_rand(
        rng,
        wall_planes,
        room_dimensions,
        colliders,
        sink,
    )


class _BathroomSinkSetupParts(NamedTuple):
    bathroom_sinks: list[MeshResult]
    sink_taps: list[MeshResult]
    sink_supports: list[MeshResult]
    mirrors: list[MeshResult]
    wall_storage: list[MeshResult]


class BathroomSinkSetupResult(NamedTuple):
    bathroom_sinks: list[MeshResult]
    sink_taps: list[MeshResult]
    sink_supports: list[MeshResult]
    mirrors: list[MeshResult]
    wall_storage: list[MeshResult]
    all_objects: list[pf.MeshObject]


def _translate_objects(objects: list[pf.MeshObject], translation: pf.Vector) -> None:
    for obj in objects:
        location = pf.Vector(obj.item().location) + translation
        pf.ops.object.set_transform(obj, location=location)


def _place_sink_over_support(
    sink_parts: list[MeshResult], support_height: float
) -> None:
    minimum, maximum = pf.ops.attr.bbox_min_max(sink_parts[0].mesh, global_coords=False)
    translation = pf.Vector(
        (
            -minimum[0],
            -(minimum[1] + maximum[1]) / 2.0,
            support_height + 0.001 - minimum[2],
        )
    )
    _translate_objects([part.mesh for part in sink_parts], translation)


def _bathroom_result(
    sink_parts: list[MeshResult], supports: list[MeshResult]
) -> _BathroomSinkSetupParts:
    return _BathroomSinkSetupParts(
        bathroom_sinks=sink_parts[:1],
        sink_taps=sink_parts[1:],
        sink_supports=supports,
        mirrors=[],
        wall_storage=[],
    )


def _sink_bathroom_for_setup_rand(
    rng: pf.RNG,
    width: float,
    size: float,
    depth: float,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
) -> MeshResult:
    return bathtub.sink_bathroom_rand(
        rng,
        width=width,
        size=size,
        depth=depth,
        surface_material=surface_material,
        metal_material=metal_material,
    )


def _sink_kitchen_for_setup_rand(
    rng: pf.RNG,
    width: float,
    size: float,
    depth: float,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
) -> MeshResult:
    del metal_material
    rim_margin = 0.03
    tap_margin = 0.11
    result = sink.sink_rand(
        rng,
        material=surface_material,
        width=width - rim_margin,
        depth=size - rim_margin - tap_margin,
        upper_height=depth,
        margin=rim_margin,
        water_tap_margin=tap_margin,
    )
    delete_object(result.cutter.item())
    return result


def _bathroom_sink_parts_rand(
    rng: pf.RNG,
    width: float | None = None,
    size: float | None = None,
    depth: float | None = None,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
    tap_material: pf.Material | None = None,
) -> list[MeshResult]:
    (
        rng_choice,
        rng_width,
        rng_size,
        rng_depth,
        rng_bathroom,
        rng_kitchen,
        rng_tap,
    ) = rng.spawn(7)
    if width is None:
        width = pf.random.uniform(rng_width, 0.5, 0.75)
    if size is None:
        size = pf.random.uniform(rng_size, 0.4, 0.55)
    if depth is None:
        depth = pf.random.uniform(rng_depth, 0.14, 0.22)
    sink_kind = pf.control.choice(rng_choice, [("bathroom", 3.0), ("kitchen", 1.0)])
    if sink_kind == "kitchen":
        sink_result = _sink_kitchen_for_setup_rand(
            rng_kitchen,
            width,
            size,
            depth,
            surface_material,
            metal_material,
        )
    else:
        sink_result = _sink_bathroom_for_setup_rand(
            rng_bathroom,
            width,
            size,
            depth,
            surface_material,
            metal_material,
        )
    tap_result = tap.tap_rand(rng_tap, material=tap_material)
    tap_result.mesh.item().name = "sink_tap"
    _place_tap_at_back(tap_result.mesh, sink_result.mesh)
    return [sink_result, tap_result]


def _floating_sink_setup_rand(
    rng: pf.RNG,
    height: float | None = None,
    width: float | None = None,
    size: float | None = None,
    depth: float | None = None,
    base_inset: float | None = None,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
    tap_material: pf.Material | None = None,
) -> _BathroomSinkSetupParts:
    del base_inset
    rng_height, rng_sink = rng.spawn(2)
    if height is None:
        height = pf.random.uniform(
            rng_height, _MIN_SINK_TOP_HEIGHT, _MAX_SINK_TOP_HEIGHT
        )
    sink_parts = _bathroom_sink_parts_rand(
        rng_sink,
        width=width,
        size=size,
        depth=depth,
        surface_material=surface_material,
        metal_material=metal_material,
        tap_material=tap_material,
    )
    minimum, maximum = pf.ops.attr.bbox_min_max(sink_parts[0].mesh, global_coords=False)
    _place_sink_over_support(sink_parts, height - (maximum[2] - minimum[2]))
    return _bathroom_result(sink_parts, [])


def _pedestal_sink_setup_rand(
    rng: pf.RNG,
    height: float | None = None,
    width: float | None = None,
    size: float | None = None,
    depth: float | None = None,
    base_inset: float | None = None,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
    tap_material: pf.Material | None = None,
) -> _BathroomSinkSetupParts:
    del base_inset
    rng_height, rng_sink, rng_pedestal = rng.spawn(3)
    if height is None:
        height = pf.random.uniform(
            rng_height, _MIN_SINK_TOP_HEIGHT, _MAX_SINK_TOP_HEIGHT
        )
    sink_parts = _bathroom_sink_parts_rand(
        rng_sink,
        width=width,
        size=size,
        depth=depth,
        surface_material=surface_material,
        metal_material=metal_material,
        tap_material=tap_material,
    )
    minimum, maximum = pf.ops.attr.bbox_min_max(sink_parts[0].mesh, global_coords=False)
    sink_width = maximum[1] - minimum[1]
    sink_depth = maximum[0] - minimum[0]
    support_height = height - (maximum[2] - minimum[2])
    rng_dimensions, rng_shape = rng_pedestal.spawn(2)
    pedestal = bathtub.sink_pedestal(
        height=support_height,
        top_radius=sink_width * pf.random.uniform(rng_dimensions, 0.075, 0.1),
        bottom_radius=sink_width * pf.random.uniform(rng_dimensions, 0.1, 0.3),
        is_circular=pf.control.choice(rng_shape, [(True, 1.0), (False, 1.0)]),
        material=surface_material,
    )
    pf.ops.object.set_transform(
        pedestal.mesh,
        location=(sink_depth / 2.0, 0.0, 0.001),
    )
    _place_sink_over_support(sink_parts, support_height)
    return _bathroom_result(sink_parts, [pedestal])


def _bathroom_storage_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
) -> storage.StorageResult:
    rng_choice, rng_storage = rng.spawn(2)
    storage_kind = pf.control.choice(rng_choice, [("composite", 1.0), ("cabinet", 1.0)])
    if storage_kind == "cabinet":
        return storage.cabinet_with_door_rand(rng_storage, dimensions=dimensions)
    return storage.storage_composite_rand(rng_storage, dimensions=dimensions)


def _place_support_under_sink(
    support: pf.MeshObject,
    sink_obj: pf.MeshObject,
) -> None:
    sink_minimum, sink_maximum = pf.ops.attr.bbox_min_max(sink_obj, global_coords=False)
    support_minimum, support_maximum = pf.ops.attr.bbox_min_max(
        support, global_coords=False
    )
    sink_center = (sink_minimum + sink_maximum) / 2.0
    support_center = (support_minimum + support_maximum) / 2.0
    origin = sink_obj.item().matrix_world.translation
    floor_world = pf.Vector((origin.x, origin.y, 0.001))
    floor_local = sink_obj.item().matrix_world.inverted() @ floor_world
    translation = pf.Vector(
        (
            sink_center[0] - support_center[0],
            sink_center[1] - support_center[1],
            floor_local.z - support_minimum[2],
        )
    )
    support.item().matrix_world = sink_obj.item().matrix_world @ pf.Matrix.Translation(
        translation
    )


def _existing_sink_setup_rand(
    rng: pf.RNG,
    sink_obj: pf.MeshObject,
    base_inset: float | None,
    surface_material: pf.Material | None,
    tap_material: pf.Material | None,
) -> _BathroomSinkSetupParts:
    rng_support, rng_tap, rng_inset, rng_pedestal, rng_cabinet = rng.spawn(5)
    tap_result = tap.tap_rand(rng_tap, material=tap_material)
    tap_result.mesh.item().name = "sink_tap"
    _place_tap_at_back(tap_result.mesh, sink_obj)

    minimum, maximum = pf.ops.attr.bbox_min_max(sink_obj, global_coords=False)
    origin = sink_obj.item().matrix_world.translation
    floor_world = pf.Vector((origin.x, origin.y, 0.001))
    floor_local = sink_obj.item().matrix_world.inverted() @ floor_world
    support_height = minimum[2] - floor_local.z
    support_height = max(support_height, 0.05)
    sink_depth = maximum[0] - minimum[0]
    sink_width = maximum[1] - minimum[1]
    support_style = pf.control.choice(
        rng_support,
        [("floating", 1.0), ("pedestal", 1.0), ("cabinet", 1.0)],
    )
    supports: list[MeshResult] = []
    if support_style == "pedestal":
        rng_dimensions, rng_shape = rng_pedestal.spawn(2)
        pedestal = bathtub.sink_pedestal(
            height=support_height,
            top_radius=sink_width * pf.random.uniform(rng_dimensions, 0.075, 0.1),
            bottom_radius=sink_width * pf.random.uniform(rng_dimensions, 0.1, 0.3),
            is_circular=pf.control.choice(rng_shape, [(True, 1.0), (False, 1.0)]),
            material=surface_material,
        )
        _place_support_under_sink(pedestal.mesh, sink_obj)
        supports = [pedestal]
    elif support_style == "cabinet":
        if base_inset is None:
            base_inset = pf.random.uniform(rng_inset, 0.0, 0.1)
        support_depth = max(sink_depth - base_inset, 0.1)
        support_width = max(sink_width - base_inset, 0.1)
        cabinet_result = _bathroom_storage_rand(
            rng_cabinet,
            pf.Vector((support_depth, support_width, support_height)),
        )
        cabinet = cabinet_result.mesh
        cabinet_minimum, cabinet_maximum = pf.ops.attr.bbox_min_max(
            cabinet, global_coords=False
        )
        cabinet_dimensions = cabinet_maximum - cabinet_minimum
        pf.ops.mesh.transform(
            cabinet,
            scale=(
                support_depth / cabinet_dimensions[0],
                support_width / cabinet_dimensions[1],
                support_height / cabinet_dimensions[2],
            ),
        )
        cabinet.item().name = "sink_cabinet"
        _place_support_under_sink(cabinet, sink_obj)
        supports = [cabinet_result]
    return _BathroomSinkSetupParts(
        bathroom_sinks=[BareMeshResult(sink_obj)],
        sink_taps=[tap_result],
        sink_supports=supports,
        mirrors=[],
        wall_storage=[],
    )


def _cabinet_sink_setup_rand(
    rng: pf.RNG,
    height: float | None = None,
    width: float | None = None,
    size: float | None = None,
    depth: float | None = None,
    base_inset: float | None = None,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
    tap_material: pf.Material | None = None,
) -> _BathroomSinkSetupParts:
    rng_height, rng_sink, rng_storage, rng_inset = rng.spawn(4)
    if height is None:
        height = pf.random.uniform(
            rng_height, _MIN_SINK_TOP_HEIGHT, _MAX_SINK_TOP_HEIGHT
        )
    sink_parts = _bathroom_sink_parts_rand(
        rng_sink,
        width=width,
        size=size,
        depth=depth,
        surface_material=surface_material,
        metal_material=metal_material,
        tap_material=tap_material,
    )
    minimum, maximum = pf.ops.attr.bbox_min_max(sink_parts[0].mesh, global_coords=False)
    sink_width = maximum[1] - minimum[1]
    sink_depth = maximum[0] - minimum[0]
    sink_height = maximum[2] - minimum[2]
    support_height = height - sink_height
    if base_inset is None:
        base_inset = pf.random.uniform(rng_inset, 0.0, 0.1)
    support_depth = sink_depth - base_inset
    support_width = sink_width - base_inset
    cabinet_result = _bathroom_storage_rand(
        rng_storage,
        dimensions=pf.Vector((support_depth, support_width, support_height)),
    )
    cabinet = cabinet_result.mesh
    cabinet_minimum, cabinet_maximum = pf.ops.attr.bbox_min_max(
        cabinet,
        global_coords=False,
    )
    cabinet_dimensions = cabinet_maximum - cabinet_minimum
    pf.ops.mesh.transform(
        cabinet,
        scale=(
            support_depth / cabinet_dimensions[0],
            support_width / cabinet_dimensions[1],
            1.0,
        ),
    )
    cabinet.item().name = "sink_cabinet"
    pf.ops.object.set_transform(
        cabinet,
        location=(0.0, -support_width / 2.0, 0.001),
    )
    _place_sink_over_support(sink_parts, support_height)
    return _bathroom_result(sink_parts, [cabinet_result])


def _bathroom_with_mirror_rand(
    rng: pf.RNG,
    setup: _BathroomSinkSetupParts,
    dimensions: pf.Vector | None = None,
    room_height: float | None = None,
) -> _BathroomSinkSetupParts:
    rng_width, rng_height, rng_depth, rng_gap, rng_mirror = rng.spawn(5)
    sink_result = setup.bathroom_sinks[0]
    sink_minimum, sink_maximum = pf.ops.attr.bbox_min_max(
        sink_result.mesh, global_coords=False
    )
    _, sink_world_maximum = pf.ops.attr.bbox_min_max(
        sink_result.mesh, global_coords=True
    )
    gap = pf.random.uniform(rng_gap, 0.04, 0.2)
    if dimensions is None:
        sink_width = sink_maximum[1] - sink_minimum[1]
        maximum_height = 1.8
        if room_height is not None:
            available_height = room_height - sink_world_maximum[2] - gap - 0.02
            maximum_height = max(0.55, min(maximum_height, available_height))
        dimensions = pf.Vector(
            (
                pf.random.uniform(rng_depth, 0.02, 0.04),
                sink_width * pf.random.uniform(rng_width, 0.75, 1.5),
                pf.random.uniform(rng_height, 0.55, maximum_height),
            )
        )
    mirror = wall_art.mirror_rand(rng_mirror, dimensions=dimensions)
    mirror.mesh.item().name = "mirror"
    mirror_minimum, mirror_maximum = pf.ops.attr.bbox_min_max(
        mirror.mesh, global_coords=False
    )
    translation = pf.Vector(
        (
            sink_minimum[0] - mirror_minimum[0],
            (sink_minimum[1] + sink_maximum[1]) / 2.0
            - (mirror_minimum[1] + mirror_maximum[1]) / 2.0,
            sink_maximum[2] + gap - mirror_minimum[2],
        )
    )
    pf.ops.object.set_transform(mirror.mesh, location=translation)
    mirror.mesh.item().matrix_world = (
        sink_result.mesh.item().matrix_world @ mirror.mesh.item().matrix_world
    )
    return _BathroomSinkSetupParts(
        bathroom_sinks=setup.bathroom_sinks,
        sink_taps=setup.sink_taps,
        sink_supports=setup.sink_supports,
        mirrors=[mirror],
        wall_storage=setup.wall_storage,
    )


def _bathroom_with_wall_storage_rand(
    rng: pf.RNG,
    setup: _BathroomSinkSetupParts,
    depth: float | None = None,
    height: float | None = None,
    room_height: float | None = None,
) -> _BathroomSinkSetupParts:
    rng_depth, rng_height, rng_gap, rng_storage = rng.spawn(4)
    sink_result = setup.bathroom_sinks[0]
    sink_minimum, sink_maximum = pf.ops.attr.bbox_min_max(
        sink_result.mesh, global_coords=False
    )
    _, sink_world_maximum = pf.ops.attr.bbox_min_max(
        sink_result.mesh, global_coords=True
    )
    sink_width = sink_maximum[1] - sink_minimum[1]
    gap = pf.random.uniform(rng_gap, _MAX_TAP_HEIGHT, _MAX_TAP_HEIGHT + 0.1)
    if depth is None:
        depth = pf.random.uniform(rng_depth, 0.07, 0.2)
    if height is None:
        maximum_height = 0.96
        if room_height is not None:
            available_height = room_height - sink_world_maximum[2] - gap - 0.02
            maximum_height = max(0.36, min(maximum_height, available_height))
        height = pf.random.uniform(rng_height, 0.36, maximum_height)
    cabinet_result = _bathroom_storage_rand(
        rng_storage,
        dimensions=pf.Vector((depth, sink_width, height)),
    )
    cabinet = cabinet_result.mesh
    cabinet_minimum, cabinet_maximum = pf.ops.attr.bbox_min_max(
        cabinet, global_coords=False
    )
    cabinet_dimensions = cabinet_maximum - cabinet_minimum
    pf.ops.mesh.transform(
        cabinet,
        scale=(
            depth / cabinet_dimensions[0],
            sink_width / cabinet_dimensions[1],
            1.0,
        ),
    )
    cabinet.item().name = "bathroom_wall_storage"
    cabinet_minimum, cabinet_maximum = pf.ops.attr.bbox_min_max(
        cabinet, global_coords=False
    )
    translation = pf.Vector(
        (
            sink_minimum[0] - cabinet_minimum[0],
            (sink_minimum[1] + sink_maximum[1]) / 2.0
            - (cabinet_minimum[1] + cabinet_maximum[1]) / 2.0,
            sink_maximum[2] + gap - cabinet_minimum[2],
        )
    )
    pf.ops.object.set_transform(cabinet, location=translation)
    cabinet.item().matrix_world = (
        sink_result.mesh.item().matrix_world @ cabinet.item().matrix_world
    )
    return _BathroomSinkSetupParts(
        bathroom_sinks=setup.bathroom_sinks,
        sink_taps=setup.sink_taps,
        sink_supports=setup.sink_supports,
        mirrors=setup.mirrors,
        wall_storage=[cabinet_result],
    )


def _no_bathroom_wall_feature_rand(
    rng: pf.RNG,
    setup: _BathroomSinkSetupParts,
    room_height: float | None = None,
) -> _BathroomSinkSetupParts:
    del rng, room_height
    return setup


def _bathroom_wall_feature_rand(
    rng: pf.RNG,
    setup: _BathroomSinkSetupParts,
    kind: str,
    room_height: float | None,
    mirror_dimensions: pf.Vector | None,
    storage_depth: float | None,
    storage_height: float | None,
) -> _BathroomSinkSetupParts:
    if kind == "mirror":
        return _bathroom_with_mirror_rand(
            rng, setup, dimensions=mirror_dimensions, room_height=room_height
        )
    if kind == "storage":
        return _bathroom_with_wall_storage_rand(
            rng,
            setup,
            depth=storage_depth,
            height=storage_height,
            room_height=room_height,
        )
    return _no_bathroom_wall_feature_rand(rng, setup, room_height=room_height)


def _keep_wall_grounded(
    objects: list[MeshResult],
    wall_colliders: ccol.CollisionSet,
    wall_margin: float,
) -> list[MeshResult]:
    grounded = []
    for result in objects:
        if back_face_grounded(result.mesh, wall_colliders, wall_margin):
            grounded.append(result)
        else:
            result.mesh.item().name += "_UNGROUNDED"
    return grounded


def _keep_valid_bathroom_sink_components(
    setup: BathroomSinkSetupResult,
    colliders: ccol.CollisionSet,
    wall_planes: list[pf.MeshObject],
    wall_margin: float,
) -> BathroomSinkSetupResult:
    wall_colliders = ccol.collision_set(wall_planes)
    mirrors = _keep_wall_grounded(setup.mirrors, wall_colliders, wall_margin)
    wall_storage = _keep_wall_grounded(
        setup.wall_storage,
        wall_colliders,
        wall_margin,
    )
    sink_supports, colliders = keep_non_colliding(setup.sink_supports, colliders)
    mirrors, colliders = keep_non_colliding(mirrors, colliders)
    wall_storage, colliders = keep_non_colliding(wall_storage, colliders)
    all_objects = (
        [result.mesh for result in setup.bathroom_sinks]
        + [result.mesh for result in setup.sink_taps]
        + [result.mesh for result in sink_supports]
        + [result.mesh for result in mirrors]
        + [result.mesh for result in wall_storage]
    )
    return BathroomSinkSetupResult(
        bathroom_sinks=setup.bathroom_sinks,
        sink_taps=setup.sink_taps,
        sink_supports=sink_supports,
        mirrors=mirrors,
        wall_storage=wall_storage,
        all_objects=all_objects,
    )


def _bathroom_sink_result(
    setup: _BathroomSinkSetupParts,
) -> BathroomSinkSetupResult:
    all_objects = (
        [result.mesh for result in setup.bathroom_sinks]
        + [result.mesh for result in setup.sink_taps]
        + [result.mesh for result in setup.sink_supports]
        + [result.mesh for result in setup.mirrors]
        + [result.mesh for result in setup.wall_storage]
    )
    return BathroomSinkSetupResult(
        bathroom_sinks=setup.bathroom_sinks,
        sink_taps=setup.sink_taps,
        sink_supports=setup.sink_supports,
        mirrors=setup.mirrors,
        wall_storage=setup.wall_storage,
        all_objects=all_objects,
    )


def _bathroom_fixture_clearance_transform(
    fixture: pf.MeshObject,
    probe_size: float = 0.5,
    probe_height: float = 1.8,
    probe_gap: float = 0.05,
    probe_floor_clearance: float = 0.01,
) -> np.ndarray:
    minimum, maximum = pf.ops.attr.bbox_min_max(fixture, global_coords=False)
    center = (minimum + maximum) / 2.0
    center[0] = maximum[0] + probe_gap + probe_size / 2.0
    transform = np.array(fixture.item().matrix_world, dtype=np.float64)
    center_world = transform @ np.append(center, 1.0)
    center_world[2] = probe_floor_clearance + probe_height / 2.0
    transform[:3, 3] = center_world[:3]
    return transform


def bathroom_fixture_clearance(
    fixture: pf.MeshObject,
    probe_size: float = 0.5,
    probe_height: float = 1.8,
    probe_gap: float = 0.05,
    probe_floor_clearance: float = 0.01,
) -> pf.MeshObject:
    transform = _bathroom_fixture_clearance_transform(
        fixture,
        probe_size,
        probe_height,
        probe_gap,
        probe_floor_clearance,
    )
    clearance = pf.ops.primitives.mesh_cube(size=1.0)
    pf.ops.mesh.transform(
        clearance,
        scale=(probe_size, probe_size, probe_height),
    )
    clearance.item().matrix_world = pf.Matrix(transform)
    clearance.item().name = "bathroom_fixture_clearance"
    clearance.item().hide_render = True
    clearance.item().display_type = "WIRE"
    return clearance


def bathroom_fixture_front_clear(
    fixture: pf.MeshObject,
    colliders: ccol.CollisionSet,
    probe_size: float = 0.5,
    probe_height: float = 1.8,
    probe_gap: float = 0.05,
    probe_floor_clearance: float = 0.01,
) -> bool:
    transform = _bathroom_fixture_clearance_transform(
        fixture,
        probe_size,
        probe_height,
        probe_gap,
        probe_floor_clearance,
    )
    return not ccol.box_intersection_test(
        colliders,
        transform,
        size=(probe_size, probe_size, probe_height),
    )


def bathroom_fixture_placement_valid(
    fixture: pf.MeshObject,
    wall_colliders: ccol.CollisionSet,
    colliders: ccol.CollisionSet,
    wall_margin: float,
    clearance_fixtures: tuple[pf.MeshObject, ...] = (),
) -> bool:
    if not back_face_grounded(fixture, wall_colliders, wall_margin):
        return False
    if not bathroom_fixture_front_clear(fixture, colliders):
        return False
    fixture_collider = ccol.collision_set([fixture])
    return all(
        bathroom_fixture_front_clear(existing, fixture_collider)
        for existing in clearance_fixtures
    )


def _place_sink_composite(
    rng: pf.RNG,
    child: MeshResult,
    parents: list[pf.MeshObject],
    margin: float,
    child_matrix: pf.Matrix,
    components: list[MeshResult],
    component_matrices: list[pf.Matrix],
) -> None:
    _snap_fixture_against_wall(rng, child, parents, margin)
    delta = child.mesh.item().matrix_world @ child_matrix.inverted()
    for component, matrix in zip(components, component_matrices, strict=True):
        component.mesh.item().matrix_world = delta @ matrix


def _place_bathroom_sink_setup_rand(
    rng: pf.RNG,
    setup: _BathroomSinkSetupParts,
    wall_planes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    wall_margin: float,
    wall_feature_kind: str,
    wall_feature_rng: pf.RNG,
    room_height: float | None,
    mirror_dimensions: pf.Vector | None,
    wall_storage_depth: float | None,
    wall_storage_height: float | None,
) -> BathroomSinkSetupResult:
    sink_result = setup.bathroom_sinks[0]
    sink_result.mesh.item().rotation_mode = "XYZ"
    components = setup.sink_taps + setup.sink_supports
    sink_matrix = sink_result.mesh.item().matrix_world.copy()
    component_matrices = [
        result.mesh.item().matrix_world.copy() for result in components
    ]
    wall_colliders = ccol.collision_set(wall_planes)
    valid = functools.partial(
        bathroom_fixture_placement_valid,
        wall_colliders=wall_colliders,
        colliders=colliders,
        wall_margin=wall_margin,
    )
    placed_sink = retry_place(
        rng,
        sink_result,
        colliders,
        _place_sink_composite,
        attempts=20,
        accept_fn=valid,
        parents=wall_planes,
        margin=wall_margin,
        child_matrix=sink_matrix,
        components=components,
        component_matrices=component_matrices,
    )
    if placed_sink is None:
        return BathroomSinkSetupResult([], [], [], [], [], [])
    feature_colliders = ccol.collision_set(
        colliders.objs + [result.mesh for result in setup.sink_supports],
        cache=colliders,
    )
    result = repeat_attempts(
        _bathroom_wall_feature_attempt_rand,
        wall_feature_rng,
        attempts=5,
        setup=setup,
        wall_feature_kind=wall_feature_kind,
        room_height=room_height,
        mirror_dimensions=mirror_dimensions,
        wall_storage_depth=wall_storage_depth,
        wall_storage_height=wall_storage_height,
        colliders=feature_colliders,
        wall_colliders=wall_colliders,
        wall_margin=wall_margin,
    )
    if result is not None:
        setup = result
    result_setup = _bathroom_sink_result(setup)
    return _keep_valid_bathroom_sink_components(
        result_setup,
        colliders,
        wall_planes,
        wall_margin,
    )


def _bathroom_wall_feature_attempt_rand(
    rng: pf.RNG,
    setup: _BathroomSinkSetupParts,
    wall_feature_kind: str,
    room_height: float | None,
    mirror_dimensions: pf.Vector | None,
    wall_storage_depth: float | None,
    wall_storage_height: float | None,
    colliders: ccol.CollisionSet,
    wall_colliders: ccol.CollisionSet,
    wall_margin: float,
) -> _BathroomSinkSetupParts | None:
    result = _bathroom_wall_feature_rand(
        rng,
        setup,
        wall_feature_kind,
        room_height,
        mirror_dimensions,
        wall_storage_depth,
        wall_storage_height,
    )
    features = result.mirrors + result.wall_storage
    grounded = _keep_wall_grounded(features, wall_colliders, wall_margin)
    kept, _ = keep_non_colliding(features, colliders)
    if len(grounded) == len(features) == len(kept):
        return result
    return None


def _add_bathroom_demo_ground(
    setup: BathroomSinkSetupResult,
) -> BathroomSinkSetupResult:
    ground = pf.ops.primitives.mesh_single_vertex()
    ground.item().name = "bathroom_setup_ground"
    return BathroomSinkSetupResult(
        bathroom_sinks=setup.bathroom_sinks,
        sink_taps=setup.sink_taps,
        sink_supports=setup.sink_supports,
        mirrors=setup.mirrors,
        wall_storage=setup.wall_storage,
        all_objects=setup.all_objects + [ground],
    )


def _bathroom_sink_components_rand(
    rng: pf.RNG,
    sink_obj: pf.MeshObject | None = None,
    wall_planes: list[pf.MeshObject] | None = None,
    room_dimensions: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
    width: float | None = None,
    size: float | None = None,
    depth: float | None = None,
    height: float | None = None,
    base_inset: float | None = None,
    mirror_dimensions: pf.Vector | None = None,
    wall_storage_depth: float | None = None,
    wall_storage_height: float | None = None,
    wall_margin: float = _WALL_MARGIN,
    surface_material: pf.Material | None = None,
    metal_material: pf.Material | None = None,
    tap_material: pf.Material | None = None,
) -> BathroomSinkSetupResult:
    (
        rng_support_choice,
        rng_floating,
        rng_pedestal,
        rng_cabinet,
        rng_wall_feature_choice,
        _,
        _,
        _,
        rng_place,
        rng_wall_feature,
    ) = rng.spawn(10)
    if sink_obj is None:
        support_kind = pf.control.choice(
            rng_support_choice,
            [
                ("floating", 1.0),
                ("pedestal", 1.0),
                ("cabinet", 1.0),
            ],
        )
        setup_kwargs = {
            "height": height,
            "width": width,
            "size": size,
            "depth": depth,
            "base_inset": base_inset,
            "surface_material": surface_material,
            "metal_material": metal_material,
            "tap_material": tap_material,
        }
        if support_kind == "pedestal":
            setup = _pedestal_sink_setup_rand(rng_pedestal, **setup_kwargs)
        elif support_kind == "cabinet":
            setup = _cabinet_sink_setup_rand(rng_cabinet, **setup_kwargs)
        else:
            setup = _floating_sink_setup_rand(rng_floating, **setup_kwargs)
    else:
        setup = _existing_sink_setup_rand(
            rng_support_choice,
            sink_obj,
            base_inset,
            surface_material,
            tap_material,
        )
    wall_feature_kind = pf.control.choice(
        rng_wall_feature_choice,
        [
            ("none", 0.5),
            ("mirror", 1.0),
            ("storage", 1.0),
        ],
    )
    room_height = None
    if room_dimensions is not None:
        room_height = room_dimensions[2]
    if wall_planes is None:
        result = _bathroom_wall_feature_rand(
            rng_wall_feature,
            setup,
            wall_feature_kind,
            room_height,
            mirror_dimensions,
            wall_storage_depth,
            wall_storage_height,
        )
        return _add_bathroom_demo_ground(_bathroom_sink_result(result))
    if colliders is None:
        colliders = ccol.collision_set(wall_planes)
    if sink_obj is not None:
        wall_colliders = ccol.collision_set(wall_planes)
        feature_colliders = ccol.collision_set(
            colliders.objs + [result.mesh for result in setup.sink_supports],
            cache=colliders,
        )
        result = repeat_attempts(
            _bathroom_wall_feature_attempt_rand,
            rng_wall_feature,
            attempts=5,
            setup=setup,
            wall_feature_kind=wall_feature_kind,
            room_height=room_height,
            mirror_dimensions=mirror_dimensions,
            wall_storage_depth=wall_storage_depth,
            wall_storage_height=wall_storage_height,
            colliders=feature_colliders,
            wall_colliders=wall_colliders,
            wall_margin=wall_margin,
        )
        if result is not None:
            setup = result
        return _keep_valid_bathroom_sink_components(
            _bathroom_sink_result(setup),
            colliders,
            wall_planes,
            wall_margin,
        )
    return _place_bathroom_sink_setup_rand(
        rng_place,
        setup,
        wall_planes,
        colliders,
        wall_margin,
        wall_feature_kind,
        rng_wall_feature,
        room_height,
        mirror_dimensions,
        wall_storage_depth,
        wall_storage_height,
    )
