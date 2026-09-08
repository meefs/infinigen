# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import NamedTuple

import procfunc as pf

from infinigen2.lighting import sky_lighting
from infinigen2.objects import window
from infinigen2.scenes.desk_setup import desk_setup_in_room_rand
from infinigen2.scenes.dining_table_setup import dining_table_setup_rand
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.room.ceiling_features import ceiling_feature_rand
from infinigen2.scenes.room.room_shape import (
    RoomShapeResult,
    room_shape_rand,
)
from infinigen2.scenes.room.room_small_objects import (
    objects_scatter_rand,
    objects_scattered_on_surface,
    small_objects_collection_rand,
)
from infinigen2.scenes.room.skirting import skirting_rand
from infinigen2.scenes.room.wall_base import (
    ROOM_SUBSURF_LEVELS,
    WallResult,
    _extrude_for_thickness,
    _resolve_wall_inputs,
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
from infinigen2.scenes.setup_utils import (
    MeshResult,
    retry_place,
    snap_on_top,
    sofa_lamps_rand,
    table_decoration_object_rand,
)
from infinigen2.scenes.sofa_setup import sofa_setup_rand
from infinigen2.scenes.wall_storage_setup import wall_storage_setup_rand
from infinigen2.shaders.functionality_lists import wall_material_rand

__all__ = [
    "LivingroomResult",
    "livingroom_rand",
    "room_walls_rand",
    "room_rand",
    "wall_arrangement_rand",
]

logger = logging.getLogger(__name__)


class LivingroomResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    lights: list[pf.LightObject]
    colliders: ccol.CollisionSet
    floor: pf.MeshObject
    dimensions: pf.Vector | None = None


def _name_materials(obj: pf.MeshObject, base: str) -> None:
    for j, slot in enumerate(obj.item().material_slots):
        if slot.material is not None:
            slot.material.name = f"{base}_{j}"


def _rename(objs: list[pf.MeshObject], name: str) -> list[pf.MeshObject]:
    """Name each object (and its materials) `{name}.NN` and return them as a list, so
    a scene's objects can be gathered by concatenating single-category _rename calls."""
    named = list(objs)
    for i, obj in enumerate(named):
        obj.item().name = f"{name}.{i:02d}"
        _name_materials(obj, f"{name}.{i:02d}")
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
    rng, wall, wall_material = _resolve_wall_inputs(rng, wall, wall_material)

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

    def doors(rng, wall, wall_material):
        return wall_doors_rand(rng, wall, wall_material, wall_thickness=wall_thickness)

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
            (doors, 0.2),
            (full_wall_window, 0.6),
        ],
    )
    return option(rng=rng_feature, wall=wall, wall_material=wall_material)


@pf.tracer.grammar
def room_walls_rand(
    rng: pf.RNG,
    shape: RoomShapeResult,
    wall_thickness: float = 0.1,
) -> WallResult:
    vec_wall = pf.nodes.shader.coord().uv

    rng_materials, rng_window, rng_walls = rng.spawn(3)
    rng_mat_1, rng_mat_2 = rng_materials.spawn(2)
    wall_material_1 = wall_material_rand(rng_mat_1, vec_wall)
    wall_material_2 = wall_material_rand(rng_mat_2, vec_wall)

    wall_back = _extrude_for_thickness(shape.walls, wall_thickness)
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
    window_result = window.window_rand(rng_window, dimensions=window_dimensions)
    window_obj = window_result.mesh
    window_portal = window_result.light

    pf.ops.object.set_transform(
        window_obj, scale=(1.0, 1.0, window_height / window_obj.item().dimensions.z)
    )
    pf.ops.mesh.transform_apply(window_obj)

    _depth, _width, _height = window_obj.item().dimensions
    wmin, _wmax = pf.ops.attr.bbox_min_max(window_obj)
    free_height = usable_height - _height
    window_bottom_pct = pf.random.clip_gaussian(rng_window, 0.7, 0.15, 0.35, 0.85)
    window_bottom = edge_gap + free_height * window_bottom_pct - wmin[2]
    window_spacing = pf.random.uniform(rng_window, 0.1, 0.25) * _width

    wall_planes = []
    backs = [wall_back]
    sills = []
    storage = []
    lights = []
    decorations: dict[str, list[pf.MeshObject]] = {}

    # cull decorations against those on other walls (seed empty; walls are coincident)
    colliders = ccol.collision_set([])
    for wall, rng_wall in zip(
        shape.flat_walls, rng_walls.spawn(len(shape.flat_walls)), strict=True
    ):
        rng_wall_mat, rng_wall_dec = rng_wall.spawn(2)
        mat = pf.control.choice(
            rng_wall_mat,
            [(wall_material_1, 3), (wall_material_2, 1)],
        )
        result = wall_arrangement_rand(
            rng_wall_dec,
            wall,
            window_obj,
            window_portal,
            wall_material=mat,
            window_spacing=window_spacing,
            window_bottom=window_bottom,
            wall_thickness=wall_thickness,
        )
        flat_decorations = [o for objs in result.decorations.values() for o in objs]
        kept, colliders = keep_non_colliding(
            flat_decorations, colliders, key=lambda o: o
        )
        dropped = set(id(o) for o in flat_decorations) - set(id(o) for o in kept)
        wall_planes.extend(result.wall_planes)
        backs.extend(result.backs)
        sills.extend(result.sills)
        storage.extend(o for o in result.storage if id(o) not in dropped)
        lights.extend(result.lights)
        for kind, objs in result.decorations.items():
            decorations.setdefault(kind, []).extend(
                o for o in objs if id(o) not in dropped
            )

    for kind, objs in sorted(decorations.items()):
        logger.info(f"Created {len(objs)} wall {kind} objects")
    logger.info(f"Created {len(storage)} wall storage surfaces")

    # storage aliases also appear under decorations, so dedup by identity
    objects = wall_planes + [shape.walls] + backs + sills + storage
    objects += [o for objs in decorations.values() for o in objs]
    return WallResult(
        all_objects=list({id(o): o for o in objects}.values()),
        wall_planes=wall_planes,
        corner_walls=[shape.walls],
        backs=backs,
        sills=sills,
        storage=storage,
        lights=lights,
        decorations=decorations,
    )


def _surface_decorations(
    rng: pf.RNG,
    surface_meshes: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
) -> tuple[list[MeshResult], ccol.CollisionSet]:
    n = min(pf.random.randint(rng, 1, 4), 2 * len(surface_meshes))
    rngs = rng.spawn(n)
    decorations = [table_decoration_object_rand(rngs[i]) for i in range(n)]
    placed_decorations = []
    for i in range(n):
        decoration = retry_place(
            rngs[i],
            decorations[i],
            colliders,
            snap_on_top,
            parents=surface_meshes,
        )
        placed_decorations.append(decoration)
    decorations, colliders = keep_non_colliding(placed_decorations, colliders)
    logger.info(f"Placed {len(decorations)} decoration objects out of {n} attempts")
    return decorations, colliders


def _desk_active_rand(rng: pf.RNG) -> bool:
    return pf.control.choice(rng, [(False, 7.0), (True, 3.0)])


def _furnished_room_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None,
    setups: list[tuple[object, float]],
) -> LivingroomResult:
    """Build the room shell, pick one furniture centerpiece from `setups`, add wall
    storage, lamps, surface decorations and scattered small objects. After the
    centerpiece choice, a desk setup is independently added 30% of the time.
    `setups` is a weighted list of setup entrypoints (e.g. sofa_setup_rand,
    dining_table_setup_rand), each called with (rng, wall_planes, room_dimensions,
    colliders)."""
    # rng lanes: 0 room shell, 1 furniture, 2 small objects
    rng_room, rng_furniture, rng_small = rng.spawn(3)

    rng_shape, rng_walls, rng_ceiling, rng_skirting = rng_room.spawn(4)
    shape = room_shape_rand(rng_shape, dimensions=dimensions)
    logger.info(f"Created room shape with {len(shape.flat_walls)} flat walls")
    wall_result = room_walls_rand(rng_walls, shape)
    logger.info(
        f"Created wall features with {len(wall_result.wall_planes)} wall planes"
    )
    ceiling_result = ceiling_feature_rand(rng_ceiling, shape)
    logger.info("Created ceiling and floor features")
    skirt_objs = skirting_rand(
        rng_skirting, walls=wall_result.wall_planes + [shape.walls]
    )
    logger.info(f"Created {len(skirt_objs)} skirting objects")

    all_room = (
        _rename([shape.floor], "room_floor")
        + _rename([ceiling_result.ceiling], "room_ceiling")
        + _rename(wall_result.wall_planes, "room_wall")
        + _rename([shape.walls], "room_wall_corners")
        + _rename(skirt_objs, "room_skirting")
        + _rename(wall_result.backs + ceiling_result.backs, "room_wall_back")
        + _rename(wall_result.sills + ceiling_result.sills, "room_wall_sill")
        + _rename(ceiling_result.light_meshes, "ceiling_light")
    )
    for name, objs in wall_result.decorations.items():
        all_room += _rename(objs, name)

    lights = ceiling_result.lights + wall_result.lights
    windows = wall_result.decorations.get("window", [])
    storage_surfaces = wall_result.storage
    windowsills = wall_result.sills + ceiling_result.sills
    room_dimensions = shape.dimensions

    (
        rng_sky,
        rng_setup,
        rng_desk,
        rng_lamp,
        rng_storage_place,
        rng_decor,
    ) = rng_furniture.spawn(6)
    sky = sky_lighting.hosek_wilkie_sky_with_sun_lamp_rand(rng_sky)
    colliders = ccol.collision_set(
        wall_result.wall_planes
        + [shape.floor, shape.walls]
        + storage_surfaces
        + windows
    )

    setup_func = pf.control.choice(rng_setup, setups)
    setup = setup_func(
        rng_setup,
        wall_planes=wall_result.wall_planes,
        room_dimensions=room_dimensions,
        colliders=colliders,
    )
    sofas = list(getattr(setup, "sofas", []))
    coffee_tables = list(getattr(setup, "coffee_tables", []))
    side_tables = list(getattr(setup, "side_tables", []))
    dining_tables = list(getattr(setup, "dining_tables", []))
    dining_chairs = list(getattr(setup, "dining_chairs", []))
    solid = [
        r.mesh
        for r in sofas + coffee_tables + side_tables + dining_tables + dining_chairs
    ]
    colliders = ccol.collision_set(colliders.objs + solid, cache=colliders)

    rng_desk_active, rng_desk_setup = rng_desk.spawn(2)
    desk_result = None
    if _desk_active_rand(rng_desk_active):
        desk_result = desk_setup_in_room_rand(
            rng_desk_setup,
            wall_planes=wall_result.wall_planes,
            room_dimensions=room_dimensions,
            colliders=colliders,
        )
    desks = [desk_result.desk] if desk_result is not None else []
    desk_chairs = [desk_result.chair] if desk_result is not None else []
    desk_lamps = list(desk_result.lamps) if desk_result is not None else []
    desk_lights = list(desk_result.lights) if desk_result is not None else []
    desk_objects = desk_result.all_objects if desk_result is not None else []
    colliders = ccol.collision_set(colliders.objs + desk_objects, cache=colliders)

    floor_lamps, table_lamps, lamp_lights, colliders = sofa_lamps_rand(
        rng_lamp,
        [r.mesh for r in sofas],
        [r.mesh for r in side_tables],
        colliders,
    )

    storage_setup = wall_storage_setup_rand(
        rng_storage_place,
        wall_planes=wall_result.wall_planes,
        room_dimensions=room_dimensions,
        colliders=colliders,
    )
    storage_objects = storage_setup.storage
    colliders = storage_setup.colliders

    surface_results = dining_tables + coffee_tables + side_tables + storage_objects
    surface_meshes = [r.mesh for r in surface_results] + desks
    decorations, colliders = _surface_decorations(rng_decor, surface_meshes, colliders)

    lights = lights + sky.lights + lamp_lights + desk_lights

    all_furniture = (
        _rename(list(getattr(setup, "rugs", [])), "rug")
        + _rename([r.mesh for r in sofas], "sofa")
        + _rename([r.mesh for r in storage_objects], "storage")
        + _rename([r.mesh for r in coffee_tables], "coffee_table")
        + _rename([r.mesh for r in side_tables], "side_table")
        + _rename([r.mesh for r in floor_lamps], "floor_lamp")
        + _rename([r.mesh for r in table_lamps], "table_lamp")
        + _rename([r.mesh for r in decorations], "decoration")
        + _rename([r.mesh for r in dining_tables], "dining_table")
        + _rename([r.mesh for r in dining_chairs], "dining_chair")
        + _rename(desks, "desk")
        + _rename(desk_chairs, "desk_chair")
        + _rename(desk_lamps, "desk_lamp")
    )

    furnished_objects = all_room + all_furniture
    base_colliders = ccol.collision_set(furnished_objects, cache=colliders)
    wall_shelves = storage_surfaces

    rng_pool, rng_place = rng_small.spawn(2)
    pool = small_objects_collection_rand(rng_pool)
    (
        _rng_dining,
        _rng_coffee,
        rng_side,
        rng_storage,
        _rng_sofa,
        _rng_rug,
        rng_shelf,
        rng_sill,
    ) = rng_place.spawn(8)

    colliders = base_colliders
    if wall_shelves:
        colliders = ccol.collision_set(colliders.objs + wall_shelves, cache=colliders)
    small_objects: list = []

    placed, colliders = objects_scattered_on_surface(
        rng_side, [r.mesh for r in side_tables], pool, colliders
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on side tables")

    placed, colliders = objects_scattered_on_surface(
        rng_storage,
        [r.mesh for r in storage_objects],
        pool,
        colliders,
        skip_prob=1 / 3,
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on floor storage")

    placed, colliders = objects_scatter_rand(
        rng_shelf, wall_shelves, pool, colliders, skip_prob=0.0
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on wall shelves")

    placed, colliders = objects_scattered_on_surface(
        rng_sill, windowsills, pool, colliders, skip_prob=2 / 3
    )
    small_objects += placed
    logger.info(f"Placed {len(placed)} small objects on windowsills")

    all_objects = furnished_objects + small_objects

    return LivingroomResult(
        all_objects=all_objects,
        lights=lights,
        colliders=ccol.collision_set(all_objects, cache=base_colliders),
        floor=shape.floor,
        dimensions=room_dimensions,
    )


@pf.tracer.grammar
def livingroom_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> LivingroomResult:
    """A furnished living room whose centerpiece is a sofa grouping."""
    return _furnished_room_rand(rng, dimensions, [(sofa_setup_rand, 1.0)])


@pf.tracer.grammar
def room_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    frame_start: int = 1,
    frame_end: int = 1,
) -> LivingroomResult:
    """Everything livingroom_rand does, but the furniture centerpiece is a choice
    between a sofa grouping and a dining setup (table with chairs around it)."""
    return _furnished_room_rand(
        rng,
        dimensions,
        [(sofa_setup_rand, 1.0), (dining_table_setup_rand, 1.0)],
    )
