# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import re
from typing import NamedTuple, cast

import bpy
import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import chair
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.distribute import propagate_modifiers_to_instances
from infinigen2.scenes.room.cocktail_table_setup import cocktail_table_setup_rand
from infinigen2.scenes.room.desk_setup import desk_setup_rand
from infinigen2.scenes.room.dining_table_setup import dining_setup_rand
from infinigen2.scenes.room.room import (
    RoomResult,
    room_unfurnished_rand,
)
from infinigen2.scenes.room.sofa_setup import centered_sofa_setup_rand
from infinigen2.scenes.setup_utils import sofa_object_rand, storage_object_rand

__all__ = [
    "SetupGridInstancesResult",
    "SetupGridResult",
    "centered_sofa_grid_rand",
    "chair_grid_rand",
    "classroom_grid_rand",
    "cocktail_grid_rand",
    "desk_grid_rand",
    "dining_grid_rand",
    "indoor_space_rand",
    "setup_grid_instances",
    "setup_grid_rand",
    "sofa_grid_rand",
    "storage_grid_rand",
]


class SetupGridResult(NamedTuple):
    all_objects: list[pf.MeshObject]


class SetupGridInstancesResult(NamedTuple):
    instances: pf.ProcNode[t.Instances]
    points: pf.ProcNode[t.Geometry]


@pf.nodes.node_function
def setup_grid_instances(
    fill_dimensions: t.SocketOrVal[pf.Vector],
    setup_collection: t.SocketOrVal[pf.Collection],
    spacing_meters: t.SocketOrVal[pf.Vector],
) -> SetupGridInstancesResult:
    source = pf.nodes.geo.collection_info(setup_collection)
    bounds = pf.nodes.geo.bound_box(pf.nodes.geo.realize_instances(source))
    size = pf.nodes.math.separate_xyz(bounds.max - bounds.min)
    fill = pf.nodes.math.separate_xyz(fill_dimensions)
    spacing = pf.nodes.math.separate_xyz(spacing_meters)

    pitch_x = pf.nodes.math.maximum(size.x + spacing.x, 0.001)
    pitch_y = pf.nodes.math.maximum(size.y + spacing.y, 0.001)
    count_x = pf.nodes.math.maximum(
        pf.nodes.math.floor((fill.x + spacing.x) / pitch_x), 1.0
    ).astype(dtype=int)
    count_y = pf.nodes.math.maximum(
        pf.nodes.math.floor((fill.y + spacing.y) / pitch_y), 1.0
    ).astype(dtype=int)

    count_x_f = count_x.astype(dtype=float)
    count_y_f = count_y.astype(dtype=float)
    occupied_x = size.x + (count_x_f - 1.0) * pitch_x
    occupied_y = size.y + (count_y_f - 1.0) * pitch_y
    start_x = (fill.x - occupied_x + size.x) * 0.5
    start_y = (fill.y - occupied_y + size.y) * 0.5

    x_line = pf.nodes.geo.mesh_line(
        count=count_x,
        start_location=pf.nodes.math.combine_xyz(x=start_x, y=start_y),
        offset=pf.nodes.math.combine_xyz(x=pitch_x),
    )
    y_line = pf.nodes.geo.mesh_line(
        count=count_y,
        start_location=(0.0, 0.0, 0.0),
        offset=pf.nodes.math.combine_xyz(y=pitch_y),
    )
    rows = pf.nodes.geo.instance_on_points(
        points=cast("pf.ProcNode[t.Geometry]", y_line),
        instance=cast("pf.ProcNode[t.Geometry]", x_line),
    )
    points = pf.nodes.geo.realize_instances(rows)

    center = pf.nodes.math.separate_xyz((bounds.min + bounds.max) * 0.5)
    normalized = pf.nodes.geo.translate_instances(
        source,
        translation=pf.nodes.math.combine_xyz(
            x=-center.x,
            y=-center.y,
            z=-bounds.min.z,
        ),
    )
    instances = pf.nodes.geo.instance_on_points(
        points=points,
        instance=cast("pf.ProcNode[t.Geometry]", normalized),
    )
    return SetupGridInstancesResult(instances, points)


def _data_stem(name: str) -> str:
    return re.sub(r"\.\d+$", "", name)


def _grid_aliases(
    setup_collection: pf.Collection,
    setup_objects: list[pf.MeshObject],
    fill_dimensions: pf.Vector,
    spacing: pf.Vector,
) -> list[list[pf.MeshObject]]:
    templates = []
    labels = {}
    seen = set()
    for obj in setup_objects:
        data = obj.item().data
        key = data.as_pointer()
        if key in seen:
            continue
        seen.add(key)
        label = _data_stem(obj.item().name)
        data.name = f"indoor_grid_source_{len(templates):03d}_{label}"
        labels[_data_stem(data.name)] = label
        templates.append(obj)

    grid = setup_grid_instances(fill_dimensions, setup_collection, spacing)
    aliases = pf.nodes.to_aliases(grid.instances)
    propagate_modifiers_to_instances(templates, aliases)
    for i, alias in enumerate(aliases):
        label = labels[_data_stem(alias.item().data.name)]
        alias.item().name = f"{label}.{i:03d}"
    point_obj = pf.nodes.to_mesh_object(cast("pf.ProcNode[pf.MeshObject]", grid.points))
    points = pf.ops.attr.vertex_positions(point_obj, global_coords=True)
    setup_groups = [[] for _ in points]
    for alias in aliases:
        bbox_min, bbox_max = pf.ops.attr.bbox_min_max(alias, global_coords=True)
        center_x = (bbox_min[0] + bbox_max[0]) * 0.5
        center_y = (bbox_min[1] + bbox_max[1]) * 0.5
        index = min(
            range(len(points)),
            key=lambda i: (center_x - points[i][0]) ** 2
            + (center_y - points[i][1]) ** 2,
        )
        setup_groups[index].append(alias)
    bpy.data.objects.remove(point_obj.item(), do_unlink=True)
    return setup_groups


def _keep_non_colliding_setups(
    setup_groups: list[list[pf.MeshObject]],
    colliders: ccol.CollisionSet,
) -> list[pf.MeshObject]:
    kept = []
    for setup in setup_groups:
        if any(ccol.intersection_test(colliders, obj) for obj in setup):
            for obj in setup:
                obj.item().name = obj.item().name + "_COLLIDE"
            continue
        kept.extend(setup)
        colliders = ccol.collision_set(colliders.objs + setup, cache=colliders)
    return kept


def _setup_dimensions(setup_objects: list[pf.MeshObject]) -> pf.Vector:
    bounds = [
        pf.ops.attr.bbox_min_max(obj, global_coords=True) for obj in setup_objects
    ]
    bbox_min = pf.Vector(tuple(min(lo[i] for lo, _ in bounds) for i in range(3)))
    bbox_max = pf.Vector(tuple(max(hi[i] for _, hi in bounds) for i in range(3)))
    return bbox_max - bbox_min


def _grid_region_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    setup_dimensions: pf.Vector,
    margin_fraction_max: float,
) -> tuple[pf.Vector, pf.Vector]:
    rng_margin_x, rng_margin_y, rng_centered, rng_offset_x, rng_offset_y = rng.spawn(5)
    margin_x = room_dimensions.x * pf.random.uniform(
        rng_margin_x, 0.05, margin_fraction_max
    )
    margin_y = room_dimensions.y * pf.random.uniform(
        rng_margin_y, 0.05, margin_fraction_max
    )
    margin_x = min(margin_x, max(room_dimensions.x - setup_dimensions.x, 0.0))
    margin_y = min(margin_y, max(room_dimensions.y - setup_dimensions.y, 0.0))
    fill_dimensions = pf.Vector(
        (room_dimensions.x - margin_x, room_dimensions.y - margin_y, room_dimensions.z)
    )
    centered = pf.control.choice(rng_centered, [(False, 1.0), (True, 1.0)])
    if centered:
        return fill_dimensions, pf.Vector((margin_x * 0.5, margin_y * 0.5, 0.0))
    offset = pf.Vector(
        (
            margin_x * pf.random.uniform(rng_offset_x, 0.0, 1.0),
            margin_y * pf.random.uniform(rng_offset_y, 0.0, 1.0),
            0.0,
        )
    )
    return fill_dimensions, offset


@pf.tracer.grammar
def setup_grid_rand(
    rng: pf.RNG,
    setup: list[pf.MeshObject] | pf.Collection,
    spacing: pf.Vector,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    margin_fraction_max: float = 0.6,
) -> SetupGridResult:
    if isinstance(setup, pf.Collection):
        setup_collection = setup
        setup_objects = cast("list[pf.MeshObject]", list(setup))
    else:
        setup_objects = list(setup)
        setup_collection = pf.Collection(setup_objects, name="indoor_grid_source")
    if not setup_objects:
        raise ValueError("setup must contain at least one mesh object")
    setup_dimensions = _setup_dimensions(setup_objects)
    fill_dimensions, offset = _grid_region_rand(
        rng,
        room_dimensions,
        setup_dimensions,
        margin_fraction_max,
    )
    setup_groups = _grid_aliases(
        setup_collection,
        setup_objects,
        fill_dimensions,
        spacing,
    )
    for setup in setup_groups:
        for obj in setup:
            obj.item().location += offset
    kept = _keep_non_colliding_setups(setup_groups, colliders)
    return SetupGridResult(kept)


@pf.tracer.grammar
def storage_grid_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    spacing: pf.Vector | None = None,
) -> SetupGridResult:
    rng_setup, rng_spacing, rng_grid = rng.spawn(3)
    storage = storage_object_rand(rng_setup).mesh
    storage.item().name = "storage"
    if spacing is None:
        spacing = pf.Vector((pf.random.uniform(rng_spacing, 0.8, 1.8), 0.0, 0.0))
    return setup_grid_rand(
        rng_grid,
        [storage],
        spacing,
        room_dimensions,
        colliders,
    )


@pf.tracer.grammar
def desk_grid_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    spacing: pf.Vector | None = None,
) -> SetupGridResult:
    rng_setup, rng_spacing, rng_grid = rng.spawn(3)
    rng_rows, rng_seats = rng_spacing.spawn(2)
    setup = desk_setup_rand(rng_setup)
    setup.desk.item().name = "desk"
    setup.chair.item().name = "desk_chair"
    if spacing is None:
        spacing = pf.Vector(
            (
                pf.random.uniform(rng_rows, 0.6, 1.5),
                pf.random.uniform(rng_seats, 0.6, 1.5),
                0.0,
            )
        )
    return setup_grid_rand(
        rng_grid,
        setup.all_objects,
        spacing,
        room_dimensions,
        colliders,
        margin_fraction_max=0.2,
    )


@pf.tracer.grammar
def classroom_grid_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    spacing: pf.Vector | None = None,
) -> SetupGridResult:
    rng_setup, rng_spacing, rng_grid = rng.spawn(3)
    setup = desk_setup_rand(rng_setup)
    setup.desk.item().name = "desk"
    setup.chair.item().name = "desk_chair"
    if spacing is None:
        spacing = pf.Vector((pf.random.uniform(rng_spacing, 0.6, 1.5), 0.0, 0.0))
    return setup_grid_rand(
        rng_grid,
        setup.all_objects,
        spacing,
        room_dimensions,
        colliders,
        margin_fraction_max=0.2,
    )


@pf.tracer.grammar
def chair_grid_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    spacing: pf.Vector | None = None,
) -> SetupGridResult:
    rng_setup, rng_spacing, rng_grid = rng.spawn(3)
    rng_rows, rng_seats = rng_spacing.spawn(2)
    chair_obj = chair.chair_rand(rng_setup).mesh
    chair_obj.item().name = "chair"
    if spacing is None:
        spacing = pf.Vector(
            (
                pf.random.uniform(rng_rows, 0.45, 0.8),
                pf.random.uniform(rng_seats, 0.02, 0.12),
                0.0,
            )
        )
    return setup_grid_rand(
        rng_grid,
        [chair_obj],
        spacing,
        room_dimensions,
        colliders,
    )


@pf.tracer.grammar
def sofa_grid_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    spacing: pf.Vector | None = None,
) -> SetupGridResult:
    rng_setup, rng_spacing, rng_grid = rng.spawn(3)
    rng_rows, rng_seats = rng_spacing.spawn(2)
    sofa = sofa_object_rand(rng_setup).mesh
    sofa.item().name = "sofa"
    if spacing is None:
        spacing = pf.Vector(
            (
                pf.random.uniform(rng_rows, 0.8, 1.5),
                pf.random.uniform(rng_seats, 0.05, 0.3),
                0.0,
            )
        )
    return setup_grid_rand(
        rng_grid,
        [sofa],
        spacing,
        room_dimensions,
        colliders,
    )


def _group_spacing_rand(rng: pf.RNG) -> pf.Vector:
    rng_x, rng_y = rng.spawn(2)
    return pf.Vector(
        (
            pf.random.uniform(rng_x, 1.0, 2.0),
            pf.random.uniform(rng_y, 1.0, 2.0),
            0.0,
        )
    )


@pf.tracer.grammar
def dining_grid_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    spacing: pf.Vector | None = None,
) -> SetupGridResult:
    rng_setup, rng_spacing, rng_grid = rng.spawn(3)
    setup = dining_setup_rand(rng_setup)
    setup.dining_table.item().name = "dining_table"
    for chair_obj in setup.chairs:
        chair_obj.item().name = "dining_chair"
    if spacing is None:
        spacing = _group_spacing_rand(rng_spacing)
    return setup_grid_rand(
        rng_grid,
        setup.all_objects,
        spacing,
        room_dimensions,
        colliders,
    )


@pf.tracer.grammar
def cocktail_grid_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    spacing: pf.Vector | None = None,
) -> SetupGridResult:
    rng_setup, rng_spacing, rng_grid = rng.spawn(3)
    setup = cocktail_table_setup_rand(rng_setup)
    setup.dining_table.item().name = "cocktail_table"
    for chair_obj in setup.chairs:
        chair_obj.item().name = "cocktail_chair"
    if spacing is None:
        spacing = _group_spacing_rand(rng_spacing)
    return setup_grid_rand(
        rng_grid,
        setup.all_objects,
        spacing,
        room_dimensions,
        colliders,
    )


@pf.tracer.grammar
def centered_sofa_grid_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    spacing: pf.Vector | None = None,
) -> SetupGridResult:
    rng_setup, rng_spacing, rng_grid = rng.spawn(3)
    setup = centered_sofa_setup_rand(rng_setup)
    for sofa in setup.sofas:
        sofa.mesh.item().name = "sofa"
    for table in setup.storages:
        table.item().name = "coffee_table"
    if spacing is None:
        spacing = _group_spacing_rand(rng_spacing)
    return setup_grid_rand(
        rng_grid,
        setup.all_objects,
        spacing,
        room_dimensions,
        colliders,
    )


def _indoor_space_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    rng_short, rng_long, rng_height, rng_orientation = rng.spawn(4)
    short_side = pf.random.uniform(rng_short, 4.55, 8.0)
    long_side = pf.random.uniform(rng_long, max(short_side, 5.85), 16.0)
    height = pf.random.clip_gaussian(rng_height, 3.2, 1.2, 2.2, 6.0)
    swap_axes = pf.control.choice(rng_orientation, [(False, 1.0), (True, 1.0)])
    if swap_axes:
        return pf.Vector((long_side, short_side, height))
    return pf.Vector((short_side, long_side, height))


@pf.tracer.grammar
def indoor_space_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    """A broad indoor shell filled by a repeated furniture-setup grid."""
    rng_dimensions, rng_room, rng_grid = rng.spawn(3)
    if dimensions is None:
        dimensions = _indoor_space_dimensions_rand(rng_dimensions)

    room = room_unfurnished_rand(
        rng_room,
        dimensions,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    rng_grid_choice, rng_grid_generate = rng_grid.spawn(2)
    grid_func = pf.control.choice(
        rng_grid_choice,
        [
            (storage_grid_rand, 1.0),
            (desk_grid_rand, 2.0),
            (classroom_grid_rand, 1.0),
            (chair_grid_rand, 1.0),
            (sofa_grid_rand, 1.0),
            (dining_grid_rand, 2.0),
            (cocktail_grid_rand, 1.0),
            (centered_sofa_grid_rand, 1.0),
        ],
    )
    grid = grid_func(
        rng_grid_generate,
        dimensions,
        room.colliders,
    )
    all_objects = room.all_objects + grid.all_objects
    return RoomResult(
        all_objects=all_objects,
        cameras=room.cameras,
        lights=room.lights,
        colliders=ccol.collision_set(
            room.colliders.objs + grid.all_objects,
            cache=room.colliders,
        ),
        floor=room.floor,
        dimensions=room.dimensions,
        storage_containers=room.storage_containers,
        supports=room.supports,
        storages=room.storages,
        wall_planes=room.wall_planes,
    )
