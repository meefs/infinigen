# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

"""Room contents layered onto an existing unfurnished house."""

import logging
import time
from collections.abc import Sequence
from typing import NamedTuple

import bpy
import procfunc as pf
import shapely

from infinigen2.objects import lamp
from infinigen2.scenes.house.floor_plan import PolygonRings
from infinigen2.scenes.house.unfurnished import (
    HouseResult,
    HouseRoomResult,
    house_unfurnished_rand,
)
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.room.bathroom_setup import bathroom_setup_rand
from infinigen2.scenes.room.bed_setup import (
    BedSetupResult,
    bed_dimensions_rand,
    bed_setup_rand,
)
from infinigen2.scenes.room.decoration_objects import (
    decorate_floor_objects_rand,
    decorate_surface_objects_rand,
    decoration_collection_primitives_and_real_rand,
    decoration_collection_primitives_rand,
)
from infinigen2.scenes.room.desk_setup import DeskSetupResult, desk_setup_rand
from infinigen2.scenes.room.dining_table_setup import (
    DiningTableSetupResult,
    dining_table_setup_rand,
)
from infinigen2.scenes.room.kitchen_setup import kitchen_setup_rand
from infinigen2.scenes.room.room import decorate_room_small_objects_rand
from infinigen2.scenes.room.sofa_setup import (
    SofaSetupResult,
    centered_sofa_setup_rand,
    side_tables_rand,
    wall_anchored_sofa_setup_rand,
    wall_sofa_setup_rand,
)
from infinigen2.scenes.room.wall_storage_setup import (
    WallStorageSetupResult,
    wall_storage_setup_rand,
)
from infinigen2.util.errors import RejectedScene
from infinigen2.util.scene_cleanup import delete_objects

__all__ = [
    "HouseRoomFurnishingResult",
    "HouseRoomFurnitureResult",
    "SetupResult",
    "furnish_house_room_furniture_rand",
    "furnish_house_room_rand",
    "house_furnished_rand",
    "link_house_room_collection",
    "setup_from_space_rand",
]

logger = logging.getLogger(__name__)


class SetupResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    colliders: ccol.CollisionSet
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]
    lamps: list[lamp.LampResult]
    temporary_objects: list[pf.MeshObject]


class HouseRoomFurnitureResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    lights: list[pf.LightObject]
    colliders: ccol.CollisionSet
    temporary_objects: list[pf.MeshObject]
    storage_containers: list[pf.MeshObject]
    supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]


class HouseRoomFurnishingResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    small_objects: list[pf.MeshObject]
    lights: list[pf.LightObject]


def _bbox_min(region: shapely.Polygon) -> pf.Vector:
    min_x, min_y, _, _ = region.bounds
    return pf.Vector((min_x, min_y, 0.0))


def _bbox_max(region: shapely.Polygon, height: float) -> pf.Vector:
    _, _, max_x, max_y = region.bounds
    return pf.Vector((max_x, max_y, height))


def _empty_setup(colliders: ccol.CollisionSet) -> SetupResult:
    return SetupResult(
        all_objects=[],
        colliders=colliders,
        storage_containers=[],
        supports=[],
        storages=[],
        lamps=[],
        temporary_objects=[],
    )


def _no_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    return _empty_setup(colliders)


def _furniture_setup(
    result: SofaSetupResult
    | DiningTableSetupResult
    | BedSetupResult
    | DeskSetupResult
    | WallStorageSetupResult,
    colliders: ccol.CollisionSet,
    lamps: list[lamp.LampResult],
) -> SetupResult:
    return SetupResult(
        all_objects=result.all_objects,
        colliders=ccol.collision_set(
            [*colliders.objs, *result.all_objects], cache=colliders
        ),
        storage_containers=result.storage_containers,
        supports=result.supports,
        storages=result.storages,
        lamps=lamps,
        temporary_objects=[],
    )


def _bathroom_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    try:
        result = bathroom_setup_rand(
            rng,
            wall_planes=wall_planes,
            bbox_min=_bbox_min(region),
            bbox_max=_bbox_max(region, height),
            colliders=colliders,
        )
    except RejectedScene:
        return _empty_setup(colliders)
    return SetupResult(
        all_objects=result.all_objects,
        colliders=result.colliders,
        storage_containers=result.storage_containers,
        supports=result.supports,
        storages=result.storages,
        lamps=[],
        temporary_objects=result.temporary_objects,
    )


def _wall_sofa_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    result = wall_sofa_setup_rand(rng, wall_planes=wall_planes, colliders=colliders)
    return _furniture_setup(result, colliders, [])


def _wall_anchored_sofa_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    result = wall_anchored_sofa_setup_rand(
        rng, wall_planes=wall_planes, colliders=colliders
    )
    return _furniture_setup(result, colliders, [])


def _centered_sofa_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    rng_setup, rng_side = rng.spawn(2)
    result = centered_sofa_setup_rand(
        rng_setup,
        bbox_min=_bbox_min(region),
        bbox_max=_bbox_max(region, height),
        colliders=colliders,
        region=region,
    )
    colliders = ccol.collision_set(
        [*colliders.objs, *result.all_objects], cache=colliders
    )
    side_tables = side_tables_rand(rng_side, result.sofas, colliders, max_tables=3)
    side_meshes = [r.mesh for r in side_tables]
    return SetupResult(
        all_objects=result.all_objects + side_meshes,
        colliders=ccol.collision_set([*colliders.objs, *side_meshes], cache=colliders),
        storage_containers=result.storage_containers,
        supports=result.supports + side_meshes,
        storages=result.storages + side_meshes,
        lamps=[],
        temporary_objects=[],
    )


def _dining_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    rng_length, rng_setup = rng.spawn(2)
    area = region.area * pf.random.uniform(rng_length, 0.85, 1.15)
    table_length = min(max(area / 12.0, 1.2), 2.7)
    result = dining_table_setup_rand(
        rng_setup,
        wall_planes=wall_planes,
        bbox_min=_bbox_min(region),
        bbox_max=_bbox_max(region, height),
        colliders=colliders,
        region=region,
        table_length=table_length,
    )
    return _furniture_setup(result, colliders, [])


def _bed_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
    bed_dimensions: pf.Vector | None = None,
    furniture: Sequence[pf.MeshObject] = (),
) -> SetupResult:
    try:
        result = bed_setup_rand(
            rng,
            wall_planes=wall_planes,
            bbox_min=_bbox_min(region),
            bbox_max=_bbox_max(region, height),
            colliders=colliders,
            bed_dimensions=bed_dimensions,
            furniture=furniture,
        )
    except RejectedScene:
        return _empty_setup(colliders)
    pairs = zip(result.bedside_lamps, result.lights, strict=True)
    lamps = [lamp.LampResult(mesh, light) for mesh, light in pairs]
    return _furniture_setup(result, colliders, lamps)


def _multi_bed_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    rng_count, rng_beds = rng.spawn(2)
    result = _empty_setup(colliders)
    for bed_rng in rng_beds.spawn(pf.random.randint(rng_count, 2, 4)):
        rng_area, rng_dimensions, rng_bed, rng_aspect = bed_rng.spawn(4)
        area = pf.random.uniform(rng_area, 1.8, 2.2)
        aspect = pf.random.uniform(rng_aspect, 1.85, 2.05)
        dimensions = bed_dimensions_rand(
            rng_dimensions,
            _bbox_min(region),
            _bbox_max(region, height),
            area=area,
            aspect=aspect,
        )
        bed = _bed_setup(
            rng_bed,
            region,
            wall_planes,
            height,
            result.colliders,
            bed_dimensions=dimensions,
            furniture=result.all_objects,
        )
        result = _combined(result, bed)
    return result


def _kitchen_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    result = kitchen_setup_rand(
        rng,
        wall_planes=wall_planes,
        bbox_min=_bbox_min(region),
        bbox_max=_bbox_max(region, height),
        colliders=colliders,
        region=region,
    )
    return SetupResult(
        all_objects=result.all_objects,
        colliders=result.colliders,
        storage_containers=result.storage_containers,
        supports=result.supports,
        storages=result.storages,
        lamps=[],
        temporary_objects=[],
    )


def _desk_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    result = desk_setup_rand(rng, wall_planes=wall_planes, colliders=colliders)
    if result is None:
        return _empty_setup(colliders)
    return _furniture_setup(result, colliders, [])


def _wall_storage_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    result = wall_storage_setup_rand(rng, wall_planes=wall_planes, colliders=colliders)
    return _furniture_setup(result, colliders, [])


def _combined(first: SetupResult, second: SetupResult) -> SetupResult:
    return SetupResult(
        all_objects=[*first.all_objects, *second.all_objects],
        colliders=second.colliders,
        storage_containers=[*first.storage_containers, *second.storage_containers],
        supports=[*first.supports, *second.supports],
        storages=[*first.storages, *second.storages],
        lamps=[*first.lamps, *second.lamps],
        temporary_objects=[*first.temporary_objects, *second.temporary_objects],
    )


def _sofa_and_dining_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    rng_sofa, rng_dining = rng.spawn(2)
    sofa = _centered_sofa_setup(rng_sofa, region, wall_planes, height, colliders)
    dining = _dining_setup(rng_dining, region, wall_planes, height, sofa.colliders)
    return _combined(sofa, dining)


def _kitchen_and_dining_setup(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    rng_kitchen, rng_dining = rng.spawn(2)
    kitchen = _kitchen_setup(rng_kitchen, region, wall_planes, height, colliders)
    dining = _dining_setup(rng_dining, region, wall_planes, height, kitchen.colliders)
    return _combined(kitchen, dining)


def setup_from_space_rand(
    rng: pf.RNG,
    region: shapely.Polygon,
    wall_planes: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> SetupResult:
    rng_main_choice, rng_main, rng_desk_choice, rng_desk, rng_storage = rng.spawn(5)
    main_options = [
        (_wall_sofa_setup, 1.0),
        (_wall_anchored_sofa_setup, 1.0),
        (_dining_setup, 1.0),
        (_bed_setup, 1.0),
        (_multi_bed_setup, 0.5),
        (_kitchen_setup, 1.0),
    ]
    desk_options = [(_no_setup, 1.0)]
    if region.area < 10.0:
        main_options = [
            (_no_setup, 1.0),
            (_bathroom_setup, 1.0),
        ]
    if region.area >= 24.0:
        main_options.append((_sofa_and_dining_setup, 3.0))
        main_options.append((_kitchen_and_dining_setup, 1.0))
        desk_options.append((_desk_setup, 1.0))
    main_func = pf.control.choice(rng_main_choice, main_options)
    main = main_func(rng_main, region, wall_planes, height, colliders)
    logger.info(
        f"Placed {len(main.all_objects)} {main_func.__name__} objects "
        f"in a {region.area:.1f}m2 room"
    )
    desk_func = pf.control.choice(rng_desk_choice, desk_options)
    desk = desk_func(rng_desk, region, wall_planes, height, main.colliders)
    logger.info(f"Placed {len(desk.all_objects)} {desk_func.__name__} objects")
    storage = _wall_storage_setup(
        rng_storage, region, wall_planes, height, desk.colliders
    )
    return _combined(_combined(main, desk), storage)


@pf.tracer.grammar
def furnish_house_room_furniture_rand(
    rng: pf.RNG,
    boundary_rings: PolygonRings,
    floor: pf.MeshObject,
    flat_walls: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
) -> HouseRoomFurnitureResult:
    rng_setup, rng_floor, rng_surface = rng.spawn(3)
    exterior, *interiors = boundary_rings
    region = shapely.Polygon(exterior, interiors)
    setup = setup_from_space_rand(rng_setup, region, flat_walls, height, colliders)
    floor_result = decorate_floor_objects_rand(
        rng_floor,
        objects=setup.all_objects,
        colliders=ccol.collision_set(
            [*setup.colliders.objs, *setup.temporary_objects], cache=setup.colliders
        ),
        floor=floor,
        wall_planes=flat_walls,
        storage=setup.supports,
    )
    surface_result = decorate_surface_objects_rand(
        rng_surface,
        objects=floor_result.all_objects,
        colliders=floor_result.colliders,
        support_tops=setup.supports,
        storages=setup.storages,
    )
    lamp_lights = [result.light for result in setup.lamps]
    return HouseRoomFurnitureResult(
        all_objects=surface_result.all_objects,
        lights=[*lamp_lights, *floor_result.lights, *surface_result.lights],
        colliders=surface_result.colliders,
        temporary_objects=setup.temporary_objects,
        storage_containers=setup.storage_containers,
        supports=setup.supports,
        storages=setup.storages,
    )


@pf.tracer.grammar
def furnish_house_room_rand(
    rng: pf.RNG,
    boundary_rings: PolygonRings,
    floor: pf.MeshObject,
    flat_walls: list[pf.MeshObject],
    height: float,
    colliders: ccol.CollisionSet,
    nonstorage_collection: pf.Collection | None = None,
    storage_collection: pf.Collection | None = None,
) -> HouseRoomFurnishingResult:
    rng_furniture, rng_small = rng.spawn(2)
    furniture = furnish_house_room_furniture_rand(
        rng_furniture, boundary_rings, floor, flat_walls, height, colliders
    )
    small_result = decorate_room_small_objects_rand(
        rng_small,
        objects=furniture.all_objects,
        colliders=furniture.colliders,
        containers=furniture.storage_containers,
        support_tops=furniture.supports,
        storages=furniture.storages,
        nonstorage_collection=nonstorage_collection,
        storage_collection=storage_collection,
    )
    n_small = len(small_result.all_objects) - len(furniture.all_objects)
    logger.info(
        f"Placed {len(furniture.all_objects)} furniture and {n_small} small objects"
    )
    delete_objects([obj.item() for obj in furniture.temporary_objects])
    furniture_objects = set(furniture.all_objects)
    return HouseRoomFurnishingResult(
        all_objects=small_result.all_objects,
        small_objects=[
            o for o in small_result.all_objects if o not in furniture_objects
        ],
        lights=furniture.lights,
    )


def link_house_room_collection(
    room_index: int, objects: list[pf.MeshObject], lights: list[pf.LightObject]
) -> bpy.types.Collection:
    collection = bpy.data.collections.new(f"house_room_{room_index:02d}")
    bpy.context.scene.collection.children.link(collection)
    items = [obj.item() for obj in [*objects, *lights]]
    links = [(parent, item) for item in items for parent in item.users_collection]
    for parent, item in links:
        parent.objects.unlink(item)
    for item in items:
        collection.objects.link(item)
    collection.hide_viewport = True
    return collection


@pf.tracer.grammar
def house_furnished_rand(
    rng: pf.RNG,
    dimensions: tuple[float, float] | None = None,
    room_count: int | None = None,
    height: float | None = None,
    wall_thickness: float | None = None,
    door_open_angle_deg: float | None = None,
) -> HouseResult:
    """Build an unfurnished house, then furnish each solved room."""
    rng_house, rng_rooms, rng_nonstorage, rng_storage = rng.spawn(4)
    house = house_unfurnished_rand(
        rng_house,
        dimensions=dimensions,
        room_count=room_count,
        height=height,
        wall_thickness=wall_thickness,
        door_open_angle_deg=door_open_angle_deg,
    )
    nonstorage_collection = decoration_collection_primitives_rand(rng_nonstorage)
    storage_collection = decoration_collection_primitives_and_real_rand(rng_storage)
    colliders = house.colliders
    rooms = []
    objects = []
    small_objects = []
    lights = []
    collections = []
    room_rngs = rng_rooms.spawn(len(house.rooms))
    for room, room_rng in zip(house.rooms, room_rngs, strict=True):
        logger.info(
            f"Furnishing house room {room.room_index} "
            f"({len(rooms) + 1} of {len(house.rooms)})"
        )
        start_time = time.perf_counter()  # validate-ignore: test_determinism
        furnished = furnish_house_room_rand(
            room_rng,
            room.boundary_rings,
            room.floor,
            room.flat_walls,
            house.dimensions.z,
            colliders,
            nonstorage_collection=nonstorage_collection,
            storage_collection=storage_collection,
        )
        end_time = time.perf_counter()  # validate-ignore: test_determinism
        logger.info(
            f"Finished house room {room.room_index} in {end_time - start_time:.3f}s"
        )
        colliders = ccol.collision_set(
            [*colliders.objs, *furnished.all_objects], cache=colliders
        )
        room_objects = [*room.all_objects, *furnished.all_objects]
        room_lights = [*room.lights, *furnished.lights]
        collection = link_house_room_collection(
            room.room_index, room_objects, room_lights
        )
        collections.append(collection)
        rooms.append(
            HouseRoomResult(
                room_index=room.room_index,
                boundary_rings=room.boundary_rings,
                all_objects=room_objects,
                lights=room_lights,
                floor=room.floor,
                ceiling=room.ceiling,
                walls=room.walls,
                flat_walls=room.flat_walls,
                neighbors=room.neighbors,
                doors=room.doors,
                doorway_centers=room.doorway_centers,
            )
        )
        objects += furnished.all_objects
        small_objects += furnished.small_objects
        lights += furnished.lights
    for collection in collections:
        collection.hide_viewport = False
    return HouseResult(
        all_objects=[*house.all_objects, *objects],
        small_objects=small_objects,
        cameras=house.cameras,
        lights=[*house.lights, *lights],
        rooms=tuple(rooms),
        dimensions=house.dimensions,
        colliders=colliders,
        environment=house.environment,
    )
