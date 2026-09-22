# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
import math
from typing import NamedTuple, cast

import procfunc as pf

from infinigen2.cameras import framing
from infinigen2.lighting import sky_lighting
from infinigen2.objects import window
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.room.bathroom_setup import bathroom_setup_rand
from infinigen2.scenes.room.bed_setup import bed_setup_rand
from infinigen2.scenes.room.ceiling_features import ceiling_feature_rand
from infinigen2.scenes.room.decoration_objects import (
    decorate_floor_objects_rand,
    decorate_small_objects_rand,
    decorate_surface_objects_rand,
)
from infinigen2.scenes.room.desk_setup import desk_setup_rand
from infinigen2.scenes.room.dining_table_setup import dining_table_setup_rand
from infinigen2.scenes.room.room_shape import (
    RoomShapeResult,
    room_shape_rand,
)
from infinigen2.scenes.room.skirting import skirting_rand
from infinigen2.scenes.room.sofa_setup import sofa_setup_rand, wall_sofa_setup_rand
from infinigen2.scenes.room.wall_base import (
    ROOM_SUBSURF_LEVELS,
    WallResult,
    extrude_for_thickness,
    overlap_wall_plane_edges,
    resolve_wall_inputs,
    wall_plain_rand,
)
from infinigen2.scenes.room.wall_cutouts import (
    wall_cubby_rand,
    wall_doors_rand,
    wall_full_window_rand,
    wall_painting_grid_rand,
    wall_windows_rand,
)
from infinigen2.scenes.room.wall_mounts import (
    wall_board_shelf_rand,
    wall_storage_flush_rand,
)
from infinigen2.scenes.room.wall_storage_setup import wall_storage_setup_rand
from infinigen2.shaders.functionality_lists import wall_material_rand
from infinigen2.util.scene_cleanup import delete_object

__all__ = [
    "RoomResult",
    "livingroom_rand",
    "room_bathroom_rand",
    "room_bedroom_rand",
    "room_diningroom_rand",
    "room_livingroom_rand",
    "room_unfurnished_rand",
    "room_walls_rand",
    "room_rand",
    "wall_arrangement_rand",
]

logger = logging.getLogger(__name__)


class RoomResult(NamedTuple):
    """Containers expose internal upward surfaces; supports accept objects on top.
    One object may provide both capabilities."""

    all_objects: list[pf.MeshObject]
    cameras: list[pf.CameraObject]
    lights: list[pf.LightObject]
    colliders: ccol.CollisionSet
    floor: pf.MeshObject
    storage_containers: list[pf.MeshObject]
    storage_supports: list[pf.MeshObject]
    wall_planes: list[pf.MeshObject]


def _name_materials(obj: pf.MeshObject, base: str) -> None:
    for j, slot in enumerate(obj.item().material_slots):
        if slot.material is not None:
            slot.material.name = f"{base}_{j}"


def _rename(
    objs: list[pf.MeshObject], name: str, keep_source_name: bool = False
) -> list[pf.MeshObject]:
    """Name each object (and its materials) `{name}.NN` and return them as a list, so
    a scene's objects can be gathered by concatenating single-category _rename calls."""
    named = list(objs)
    for i, obj in enumerate(named):
        object_name = name
        if keep_source_name:
            object_name = f"{name}_{obj.item().name}"
        object_name = f"{object_name}.{i:02d}"
        obj.item().name = object_name
        _name_materials(obj, object_name)
    return named


@pf.tracer.grammar
def wall_arrangement_rand(
    rng: pf.RNG,
    wall: pf.MeshObject | None = None,
    window_obj: pf.MeshObject | None = None,
    window_portal: pf.LightObject | None = None,
    wall_material: pf.Material | None = None,
    window_spacing: float | None = None,
    window_bottom: float | None = None,
    wall_thickness: float = 0.05,
) -> WallResult:
    rng, wall, wall_material = resolve_wall_inputs(rng, wall, wall_material)

    def plain(rng, wall, wall_material):
        return wall_plain_rand(rng, wall, wall_material, wall_thickness=wall_thickness)

    def windows(rng, wall, wall_material):
        return wall_windows_rand(
            rng,
            wall,
            wall_material,
            window_obj=window_obj,
            window_portal=window_portal,
            window_spacing=window_spacing,
            window_bottom=window_bottom,
            wall_thickness=wall_thickness,
        )

    def painting_grid(rng, wall, wall_material):
        return wall_painting_grid_rand(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    def board_shelf(rng, wall, wall_material):
        return wall_board_shelf_rand(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    def storage_flush(rng, wall, wall_material):
        return wall_storage_flush_rand(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    def cubby(rng, wall, wall_material):
        return wall_cubby_rand(rng, wall, wall_material, wall_thickness=wall_thickness)

    def full_wall_window(rng, wall, wall_material):
        return wall_full_window_rand(
            rng, wall, wall_material, wall_thickness=wall_thickness
        )

    rng_choice, rng_feature = rng.spawn(2)
    option = pf.control.choice(
        rng_choice,
        [
            (plain, 0.5),
            (windows, 3.0),
            (painting_grid, 1.0),
            (board_shelf, 1.5),
            (storage_flush, 1.0),
            (cubby, 0.5),
            (full_wall_window, 0.6),
        ],
    )
    return option(rng=rng_feature, wall=wall, wall_material=wall_material)


def _cull_wall_decorations(
    result: WallResult,
    colliders: ccol.CollisionSet,
) -> tuple[WallResult, ccol.CollisionSet]:
    flat = [obj for objects in result.decorations.values() for obj in objects]
    kept, colliders = keep_non_colliding(flat, colliders, key=lambda obj: obj)
    kept_objects = {obj.item() for obj in kept}
    dropped = {obj.item() for obj in flat if obj.item() not in kept_objects}

    def without_dropped(objects: list[pf.MeshObject]) -> list[pf.MeshObject]:
        return [obj for obj in objects if obj.item() not in dropped]

    return (
        result._replace(
            all_objects=without_dropped(result.all_objects),
            storage_containers=without_dropped(result.storage_containers),
            storage_supports=without_dropped(result.storage_supports),
            decorations={
                kind: without_dropped(objects)
                for kind, objects in result.decorations.items()
            },
        ),
        colliders,
    )


@pf.tracer.grammar
def room_walls_rand(
    rng: pf.RNG,
    shape: RoomShapeResult,
    wall_thickness: float = 0.1,
) -> WallResult:
    vec_wall = pf.nodes.shader.coord().uv

    (
        rng_materials,
        rng_window,
        rng_wall_order,
        rng_door,
        rng_walls,
    ) = rng.spawn(5)
    rng_mat_1, rng_mat_2 = rng_materials.spawn(2)
    wall_material_1 = wall_material_rand(rng_mat_1, vec_wall)
    wall_material_2 = wall_material_rand(rng_mat_2, vec_wall)

    wall_back = extrude_for_thickness(shape.walls, wall_thickness)
    wall_back.item().name = "room_wall_back"

    pf.ops.object.set_material(
        shape.walls,
        surface=wall_material_1.surface,
        displacement=wall_material_1.displacement,
    )
    pf.ops.modifier.subdivide_surface(
        shape.walls, levels=ROOM_SUBSURF_LEVELS, _skip_apply=True
    )

    edge_gap_pct = 0.10
    edge_gap = edge_gap_pct * shape.dimensions.z
    usable_height = shape.dimensions.z - 2.0 * edge_gap
    window_height_pct = pf.random.clip_gaussian(
        rng_window, 0.75, 0.2, 0.6, 1.0 - 2.0 * edge_gap_pct
    )
    window_height = shape.dimensions.z * window_height_pct
    # window size is independent of walls; too-wide windows fall back to plain wall
    max_window_width = max(2.0, max(shape.dimensions.x, shape.dimensions.y) - 0.5)
    window_width = (
        1.0 + (max_window_width - 1.0) * pf.random.uniform(rng_window, 0.0, 1.0) ** 2
    )
    window_dimensions = window.window_dimensions_rand(
        rng_window, width=window_width, height=window_height
    )
    window_result = window.window_composite_rand(
        rng_window, dimensions=window_dimensions
    )
    window_obj = window_result.mesh
    window_portal = window_result.light
    wall_offset = pf.Vector((0.0, window_dimensions.y * -0.5, 0.0))
    pf.ops.object.set_transform(window_obj, location=wall_offset)
    pf.ops.mesh.transform_apply(window_obj)
    if window_portal is not None:
        pf.ops.object.set_transform(
            window_portal,
            location=window_portal.item().location + wall_offset,
        )

    _depth, _width, _height = window_obj.item().dimensions
    wmin, _wmax = pf.ops.attr.bbox_min_max(window_obj)
    free_height = usable_height - _height
    window_bottom_pct = pf.random.clip_gaussian(rng_window, 0.7, 0.15, 0.35, 0.85)
    window_bottom = edge_gap + free_height * window_bottom_pct - wmin[2]
    window_spacing = pf.random.uniform(rng_window, 0.1, 0.25) * _width

    wall_planes = []
    backs = [wall_back]
    sills = []
    storage_containers = []
    storage_supports = []
    lights = []
    decorations: dict[str, list[pf.MeshObject]] = {}
    all_objects = [wall_back, shape.walls]

    def material_and_rng(rng_wall: pf.RNG) -> tuple[pf.Material, pf.RNG]:
        rng_wall_mat, rng_wall_dec = rng_wall.spawn(2)
        mat = pf.control.choice(
            rng_wall_mat,
            [(wall_material_1, 3.0), (wall_material_2, 1.0)],
        )
        return mat, rng_wall_dec

    walls = list(shape.flat_walls)
    rng_wall_order.shuffle(walls)

    door_wall = walls.pop()
    door_material, rng_door_setup = material_and_rng(rng_door)
    door_result = wall_doors_rand(
        rng_door_setup,
        door_wall,
        door_material,
        wall_thickness=wall_thickness,
    )

    arrangement_results = []
    for wall, rng_wall in zip(walls, rng_walls.spawn(len(walls)), strict=True):
        material, rng_wall_setup = material_and_rng(rng_wall)
        arrangement_results.append(
            wall_arrangement_rand(
                rng_wall_setup,
                wall,
                window_obj,
                window_portal,
                wall_material=material,
                window_spacing=window_spacing,
                window_bottom=window_bottom,
                wall_thickness=wall_thickness,
            )
        )

    colliders = ccol.collision_set([])
    results = [door_result, *arrangement_results]
    for unculled_result in results:
        result, colliders = _cull_wall_decorations(unculled_result, colliders)
        wall_planes.extend(result.wall_planes)
        backs.extend(result.backs)
        sills.extend(result.sills)
        storage_containers.extend(result.storage_containers)
        storage_supports.extend(result.storage_supports)
        lights.extend(result.lights)
        all_objects.extend(result.all_objects)
        for kind, objs in result.decorations.items():
            decorations.setdefault(kind, []).extend(objs)

    for kind, objs in sorted(decorations.items()):
        logger.info(f"Created {len(objs)} wall {kind} objects")
    logger.info(
        "Created %d wall storage containers and %d supports",
        len(storage_containers),
        len(storage_supports),
    )

    return WallResult(
        all_objects=all_objects,
        wall_planes=wall_planes,
        corner_walls=[shape.walls],
        backs=backs,
        sills=sills,
        storage_containers=storage_containers,
        storage_supports=storage_supports,
        lights=lights,
        decorations=decorations,
    )


def _livingroom_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    rng_area, rng_aspect, rng_height = rng.spawn(3)
    area = pf.random.clip_gaussian(rng_area, 22.0, 5.0, 15.0, 30.0)
    aspect = pf.random.clip_gaussian(rng_aspect, 1.3, 0.25, 1.0, 1.8)
    width = math.sqrt(area / aspect)
    depth = area / width
    height = pf.random.clip_gaussian(rng_height, 2.7, 0.2, 2.5, 3.2)
    return pf.Vector((width, depth, height))


def _diningroom_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    rng_area, rng_aspect, rng_height = rng.spawn(3)
    area = pf.random.clip_gaussian(rng_area, 16.0, 4.0, 11.5, 25.0)
    aspect = pf.random.clip_gaussian(rng_aspect, 1.2, 0.2, 1.0, 1.35)
    width = math.sqrt(area / aspect)
    depth = area / width
    height = pf.random.clip_gaussian(rng_height, 2.65, 0.2, 2.5, 3.2)
    return pf.Vector((width, depth, height))


def _bedroom_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    rng_area, rng_aspect, rng_height = rng.spawn(3)
    area = pf.random.clip_gaussian(rng_area, 15.5, 3.0, 11.5, 22.0)
    aspect = pf.random.clip_gaussian(rng_aspect, 1.18, 0.15, 1.0, 1.45)
    width = math.sqrt(area / aspect)
    depth = area / width
    height = pf.random.clip_gaussian(rng_height, 2.6, 0.2, 2.5, 3.2)
    return pf.Vector((width, depth, height))


def _bathroom_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    rng_area, rng_aspect, rng_height = rng.spawn(3)
    area = pf.random.clip_gaussian(rng_area, 5.2, 1.3, 3.7, 9.0)
    maximum_aspect = min(2.1, area / 1.5**2)
    aspect = pf.random.clip_gaussian(rng_aspect, 1.55, 0.25, 1.2, maximum_aspect)
    width = math.sqrt(area / aspect)
    depth = area / width
    height = pf.random.clip_gaussian(rng_height, 2.45, 0.15, 2.2, 2.8)
    return pf.Vector((width, depth, height))


@pf.tracer.grammar
def room_unfurnished_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    del frame_start, frame_end
    rng_shape, rng_walls, rng_ceiling, rng_skirting, rng_sky = rng.spawn(5)
    shape = room_shape_rand(rng_shape, dimensions=dimensions)
    logger.info(f"Created room shape with {len(shape.flat_walls)} flat walls")
    wall_thickness = 0.1
    wall_result = room_walls_rand(
        rng_walls,
        shape,
        wall_thickness=wall_thickness,
    )
    ceiling_result = ceiling_feature_rand(rng_ceiling, shape)
    skirt_objs = skirting_rand(
        rng_skirting, walls=wall_result.wall_planes + [shape.walls]
    )
    all_objects = (
        _rename([shape.floor], "room_floor")
        + _rename([ceiling_result.ceiling], "room_ceiling")
        + _rename(wall_result.wall_planes, "room_wall")
        + _rename([shape.walls], "room_wall_corners")
        + _rename(skirt_objs, "room_skirting")
        + _rename(wall_result.backs + ceiling_result.backs, "room_wall_back")
        + _rename(wall_result.sills + ceiling_result.sills, "room_wall_sill")
        + _rename(ceiling_result.light_meshes, "ceiling_light")
    )
    for name, objects in wall_result.decorations.items():
        all_objects += _rename(objects, name)
    container_objects = {obj.item() for obj in wall_result.storage_containers}
    storage_objects = wall_result.storage_containers + [
        obj
        for obj in wall_result.storage_supports
        if obj.item() not in container_objects
    ]
    for wall_plane in wall_result.wall_planes:
        overlap_wall_plane_edges(wall_plane, wall_thickness)
    windows = wall_result.decorations.get("window", [])
    doors = wall_result.decorations.get("door", [])
    colliders = ccol.collision_set(
        wall_result.wall_planes
        + [shape.floor, shape.walls, ceiling_result.ceiling]
        + ceiling_result.light_meshes
        + storage_objects
        + windows
        + doors
    )
    sky = sky_lighting.hosek_wilkie_sky_with_sun_lamp_rand(rng_sky)
    return RoomResult(
        all_objects=all_objects,
        cameras=[framing.camera_in_room_corner(shape.floor, float(shape.dimensions.z))],
        lights=ceiling_result.lights + wall_result.lights + sky.lights,
        colliders=colliders,
        floor=shape.floor,
        storage_containers=wall_result.storage_containers,
        storage_supports=wall_result.storage_supports,
        wall_planes=wall_result.wall_planes,
    )


def _with_objects(
    colliders: ccol.CollisionSet,
    objects: list[pf.MeshObject],
) -> ccol.CollisionSet:
    return ccol.collision_set(
        colliders.objs + objects,
        cache=colliders,
    )


@pf.tracer.grammar
def room_livingroom_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    rng_dimensions, rng_room, rng_sofa, rng_desk, rng_storage, rng_decor = rng.spawn(6)
    if dimensions is None:
        dimensions = _livingroom_dimensions_rand(rng_dimensions)
    room = room_unfurnished_rand(
        rng_room, dimensions, frame_start=frame_start, frame_end=frame_end
    )
    sofa_setup = sofa_setup_rand(
        rng_sofa,
        wall_planes=room.wall_planes,
        room_dimensions=dimensions,
        colliders=room.colliders,
    )
    _rename([result.mesh for result in sofa_setup.sofas], "sofa")
    colliders = _with_objects(room.colliders, sofa_setup.all_objects)

    desk_objects = []
    desk_containers = []
    desk_supports = []
    rng_desk_active, rng_desk_setup = rng_desk.spawn(2)
    if pf.control.choice(rng_desk_active, [(False, 2.0), (True, 1.0)]):
        desk_setup = desk_setup_rand(
            rng_desk_setup,
            wall_planes=room.wall_planes,
            colliders=colliders,
        )
        if desk_setup is not None:
            _rename([desk_setup.desk], "desk")
            _rename([desk_setup.chair], "desk_chair")
            desk_objects = desk_setup.all_objects
            desk_containers = desk_setup.storage_containers
            desk_supports = desk_setup.storage_supports
            colliders = _with_objects(colliders, desk_objects)

    storage_setup = wall_storage_setup_rand(
        rng_storage,
        wall_planes=room.wall_planes,
        room_dimensions=dimensions,
        colliders=colliders,
    )
    all_objects = (
        room.all_objects
        + sofa_setup.all_objects
        + desk_objects
        + storage_setup.all_objects
    )
    storage_containers = (
        room.storage_containers
        + sofa_setup.storage_containers
        + desk_containers
        + storage_setup.storage_containers
    )
    storage_supports = (
        room.storage_supports
        + sofa_setup.storage_supports
        + desk_supports
        + storage_setup.storage_supports
    )
    rng_floor, rng_surface, rng_small = rng_decor.spawn(3)
    floor_result = decorate_floor_objects_rand(
        rng_floor,
        objects=all_objects,
        colliders=storage_setup.colliders,
        floor=room.floor,
        wall_planes=room.wall_planes,
        storage=storage_supports,
    )
    surface_result = decorate_surface_objects_rand(
        rng_surface,
        objects=floor_result.all_objects,
        colliders=floor_result.colliders,
        support_tops=storage_supports,
    )
    small_result = decorate_small_objects_rand(
        rng_small,
        objects=surface_result.all_objects,
        colliders=surface_result.colliders,
        containers=storage_containers,
        support_tops=storage_supports,
    )
    all_objects = small_result.all_objects
    decoration_lights = floor_result.lights + surface_result.lights
    colliders = ccol.collision_set(cast(list[pf.Object], all_objects))
    return RoomResult(
        all_objects=all_objects,
        cameras=room.cameras,
        lights=room.lights + decoration_lights,
        colliders=colliders,
        floor=room.floor,
        storage_containers=storage_containers,
        storage_supports=storage_supports,
        wall_planes=room.wall_planes,
    )


@pf.tracer.grammar
def room_diningroom_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    rng_dimensions, rng_room, rng_dining, rng_storage, rng_decor = rng.spawn(5)
    if dimensions is None:
        dimensions = _diningroom_dimensions_rand(rng_dimensions)
    room = room_unfurnished_rand(
        rng_room, dimensions, frame_start=frame_start, frame_end=frame_end
    )
    dining_setup = dining_table_setup_rand(
        rng_dining,
        wall_planes=room.wall_planes,
        room_dimensions=dimensions,
        colliders=room.colliders,
    )
    _rename([result.mesh for result in dining_setup.dining_tables], "dining_table")
    _rename([result.mesh for result in dining_setup.dining_chairs], "dining_chair")
    colliders = _with_objects(room.colliders, dining_setup.all_objects)
    storage_setup = wall_storage_setup_rand(
        rng_storage,
        wall_planes=room.wall_planes,
        room_dimensions=dimensions,
        colliders=colliders,
    )
    all_objects = (
        room.all_objects + dining_setup.all_objects + storage_setup.all_objects
    )
    storage_containers = (
        room.storage_containers
        + dining_setup.storage_containers
        + storage_setup.storage_containers
    )
    storage_supports = (
        room.storage_supports
        + dining_setup.storage_supports
        + storage_setup.storage_supports
    )
    rng_floor, rng_surface, rng_small = rng_decor.spawn(3)
    floor_result = decorate_floor_objects_rand(
        rng_floor,
        objects=all_objects,
        colliders=storage_setup.colliders,
        floor=room.floor,
        wall_planes=room.wall_planes,
        storage=storage_supports,
    )
    surface_result = decorate_surface_objects_rand(
        rng_surface,
        objects=floor_result.all_objects,
        colliders=floor_result.colliders,
        support_tops=storage_supports,
    )
    small_result = decorate_small_objects_rand(
        rng_small,
        objects=surface_result.all_objects,
        colliders=surface_result.colliders,
        containers=storage_containers,
        support_tops=storage_supports,
    )
    all_objects = small_result.all_objects
    decoration_lights = floor_result.lights + surface_result.lights
    colliders = ccol.collision_set(cast(list[pf.Object], all_objects))
    return RoomResult(
        all_objects=all_objects,
        cameras=room.cameras,
        lights=room.lights + decoration_lights,
        colliders=colliders,
        floor=room.floor,
        storage_containers=storage_containers,
        storage_supports=storage_supports,
        wall_planes=room.wall_planes,
    )


@pf.tracer.grammar
def room_bedroom_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    (
        rng_dimensions,
        rng_room,
        rng_bed,
        rng_sofa,
        rng_desk,
        rng_storage,
        rng_decor,
    ) = rng.spawn(7)
    if dimensions is None:
        dimensions = _bedroom_dimensions_rand(rng_dimensions)
    room = room_unfurnished_rand(
        rng_room, dimensions, frame_start=frame_start, frame_end=frame_end
    )
    bed_setup = bed_setup_rand(
        rng_bed,
        wall_planes=room.wall_planes,
        room_dimensions=dimensions,
        colliders=room.colliders,
    )
    colliders = bed_setup.colliders

    def no_sofa():
        return None

    def wall_sofa():
        return wall_sofa_setup_rand(
            rng_sofa_setup,
            wall_planes=room.wall_planes,
            room_dimensions=dimensions,
            colliders=colliders,
        )

    rng_sofa_active, rng_sofa_setup = rng_sofa.spawn(2)
    sofa_func = pf.control.choice(
        rng_sofa_active,
        [(no_sofa, 2.0), (wall_sofa, 1.0)],
    )
    sofa_setup = sofa_func()
    sofa_objects = []
    sofa_containers = []
    sofa_supports = []
    if sofa_setup is not None:
        _rename([result.mesh for result in sofa_setup.sofas], "sofa")
        sofa_objects = sofa_setup.all_objects
        sofa_containers = sofa_setup.storage_containers
        sofa_supports = sofa_setup.storage_supports
        colliders = _with_objects(colliders, sofa_objects)

    desk_objects = []
    desk_containers = []
    desk_supports = []
    rng_desk_active, rng_desk_setup = rng_desk.spawn(2)
    if pf.control.choice(rng_desk_active, [(False, 1.0), (True, 1.0)]):
        desk_setup = desk_setup_rand(
            rng_desk_setup,
            wall_planes=room.wall_planes,
            colliders=colliders,
        )
        if desk_setup is not None:
            _rename([desk_setup.desk], "desk")
            _rename([desk_setup.chair], "desk_chair")
            desk_objects = desk_setup.all_objects
            desk_containers = desk_setup.storage_containers
            desk_supports = desk_setup.storage_supports
            colliders = _with_objects(colliders, desk_objects)

    storage_setup = wall_storage_setup_rand(
        rng_storage,
        wall_planes=room.wall_planes,
        room_dimensions=dimensions,
        colliders=colliders,
    )
    all_objects = (
        room.all_objects
        + bed_setup.all_objects
        + sofa_objects
        + desk_objects
        + storage_setup.all_objects
    )
    storage_containers = (
        room.storage_containers
        + bed_setup.storage_containers
        + sofa_containers
        + desk_containers
        + storage_setup.storage_containers
    )
    storage_supports = (
        room.storage_supports
        + bed_setup.storage_supports
        + sofa_supports
        + desk_supports
        + storage_setup.storage_supports
    )
    rng_floor, rng_surface, rng_small = rng_decor.spawn(3)
    floor_result = decorate_floor_objects_rand(
        rng_floor,
        objects=all_objects,
        colliders=storage_setup.colliders,
        floor=room.floor,
        wall_planes=room.wall_planes,
        storage=storage_supports,
    )
    surface_result = decorate_surface_objects_rand(
        rng_surface,
        objects=floor_result.all_objects,
        colliders=floor_result.colliders,
        support_tops=storage_supports,
    )
    small_result = decorate_small_objects_rand(
        rng_small,
        objects=surface_result.all_objects,
        colliders=surface_result.colliders,
        containers=storage_containers,
        support_tops=storage_supports,
    )
    all_objects = small_result.all_objects
    decoration_lights = floor_result.lights + surface_result.lights
    colliders = ccol.collision_set(cast(list[pf.Object], all_objects))
    return RoomResult(
        all_objects=all_objects,
        cameras=room.cameras,
        lights=room.lights + bed_setup.lights + decoration_lights,
        colliders=colliders,
        floor=room.floor,
        storage_containers=storage_containers,
        storage_supports=storage_supports,
        wall_planes=room.wall_planes,
    )


@pf.tracer.grammar
def room_bathroom_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    rng_dimensions, rng_room, rng_setup, rng_storage, rng_decor = rng.spawn(5)
    if dimensions is None:
        dimensions = _bathroom_dimensions_rand(rng_dimensions)
    room = room_unfurnished_rand(
        rng_room,
        dimensions,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    bathroom_setup = bathroom_setup_rand(
        rng_setup,
        wall_planes=room.wall_planes,
        room_dimensions=dimensions,
        colliders=room.colliders,
    )
    colliders = bathroom_setup.colliders
    storage_objects = []
    storage_containers = []
    storage_supports = []
    rng_storage_active, rng_storage_setup = rng_storage.spawn(2)
    if pf.control.choice(rng_storage_active, [(False, 1.0), (True, 1.0)]):
        storage_setup = wall_storage_setup_rand(
            rng_storage_setup,
            wall_planes=room.wall_planes,
            room_dimensions=dimensions,
            colliders=colliders,
        )
        storage_objects = storage_setup.all_objects
        storage_containers = storage_setup.storage_containers
        storage_supports = storage_setup.storage_supports
        colliders = storage_setup.colliders
    all_objects = room.all_objects + bathroom_setup.all_objects + storage_objects
    containers = (
        room.storage_containers + bathroom_setup.storage_containers + storage_containers
    )
    supports = (
        room.storage_supports + bathroom_setup.storage_supports + storage_supports
    )
    rng_surface, rng_small = rng_decor.spawn(2)
    surface_result = decorate_surface_objects_rand(
        rng_surface,
        objects=all_objects,
        colliders=colliders,
        support_tops=supports,
    )
    small_result = decorate_small_objects_rand(
        rng_small,
        objects=surface_result.all_objects,
        colliders=surface_result.colliders,
        containers=containers,
        support_tops=supports,
    )
    all_objects = small_result.all_objects
    decoration_lights = surface_result.lights
    colliders = ccol.collision_set(cast(list[pf.Object], all_objects))
    for temporary in bathroom_setup.temporary_objects:
        delete_object(temporary.item())
    return RoomResult(
        all_objects=all_objects,
        cameras=room.cameras,
        lights=room.lights + decoration_lights,
        colliders=colliders,
        floor=room.floor,
        storage_containers=containers,
        storage_supports=supports,
        wall_planes=room.wall_planes,
    )


livingroom_rand = room_livingroom_rand


@pf.tracer.grammar
def room_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> RoomResult:
    room_func = pf.control.choice(
        rng,
        [
            (room_livingroom_rand, 1.0),
            (room_bathroom_rand, 1.0),
            (room_diningroom_rand, 1.0),
            (room_bedroom_rand, 1.0),
        ],
    )
    return room_func(
        rng,
        dimensions=dimensions,
        frame_start=frame_start,
        frame_end=frame_end,
    )
