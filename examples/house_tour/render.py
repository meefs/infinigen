#!/usr/bin/env python
"""Render an RRT camera tour constrained to connected rooms of a house."""

# ruff: noqa: I001, E402

import argparse
import logging
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path

import bpy
import numpy as np

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

import procfunc as pf
from procfunc.util.teardown import skip_teardown_on_exit

from infinigen2.cameras import camera_cube_free_space_check, rrt_camera
from infinigen2.cameras.rrt import RRTPolicyError
from infinigen2.exporters.render_cycles import render_cycles
from infinigen2.exporters.util.format import ExportType, RenderPass
from infinigen2.scenes.house.furnishings import (
    HouseRoomFurnitureResult,
    furnish_house_room_furniture_rand,
    link_house_room_collection,
)
from infinigen2.scenes.house.unfurnished import (
    HouseResult,
    HouseRoomResult,
    house_unfurnished_rand,
)
import infinigen2.scenes.placement.collision as ccol
from infinigen2.scenes.room.decoration_objects import (
    decoration_collection_primitives_and_real_rand,
    decoration_collection_primitives_rand,
)
from infinigen2.scenes.room.room import decorate_room_small_objects_rand
from infinigen2.scenes.placement.retry import repeat_attempts
from infinigen2.util.errors import RejectedScene
from infinigen2.util.polycount import estimated_eval_tricount
from infinigen2.util.render_metadata import time_step, write_render_metadata
from infinigen2.util.scene_cleanup import cleanup_except, delete_object, delete_objects

logger = logging.getLogger(__name__)


def _parse_seed(value: str) -> int:
    return int(value, 0)


def _connected_rooms_rand(
    rng: pf.RNG,
    rooms: tuple[HouseRoomResult, ...],
    count_range: tuple[int, int] = (3, 4),
) -> tuple[HouseRoomResult, ...]:
    room_count = len(rooms)
    low = max(1, count_range[0])
    high = min(room_count, count_range[1])
    if low > high:
        raise RejectedScene("House does not contain the requested number of rooms")

    adjacency = {room.room_index: set(room.neighbors.values()) for room in rooms}

    target = int(rng.integers(low, high + 1))
    for _ in range(room_count * 2):
        selected = [int(rng.integers(0, room_count))]
        while len(selected) < target:
            frontier = sorted(adjacency[selected[-1]] - set(selected))
            if not frontier:
                break
            selected.append(frontier[int(rng.integers(0, len(frontier)))])
        if len(selected) == target:
            return tuple(rooms[index] for index in selected)
    raise RejectedScene("Could not select a connected room path")


def remove_subdivision_outside_rooms(
    objects: list[pf.MeshObject], rooms: tuple[HouseRoomResult, ...]
) -> int:
    selected = {obj.item().as_pointer() for room in rooms for obj in room.all_objects}
    removed = 0
    for obj in objects:
        if obj.item().as_pointer() in selected:
            continue
        for modifier in list(obj.item().modifiers):
            if modifier.type != "SUBSURF":
                continue
            obj.item().modifiers.remove(modifier)
            removed += 1
    return removed


def _point_over_floors(
    point: np.ndarray,
    floor_colliders: ccol.CollisionSet,
    height_range: tuple[float, float],
) -> bool:
    if not _point_at_camera_height(point, height_range):
        return False
    origins = np.asarray(point, dtype=np.float64).reshape(1, 3)
    directions = np.asarray(((0.0, 0.0, -1.0),))
    _hits, ray_indices, _tri_indices = ccol.raycast(
        floor_colliders, origins, directions
    )
    return len(ray_indices) > 0


def _point_at_camera_height(
    point: np.ndarray, height_range: tuple[float, float]
) -> bool:
    return bool(height_range[0] <= point[2] <= height_range[1])


def _sample_start_location(
    rng: pf.RNG,
    floor: pf.MeshObject,
    floor_colliders: ccol.CollisionSet,
    obstacles: ccol.CollisionSet,
    height_range: tuple[float, float],
    camera_clearance: float,
    center_xy: np.ndarray | None = None,
    radius: float = 1.25,
    attempts: int = 1000,
) -> np.ndarray:
    low, high = pf.ops.attr.bbox_min_max(floor, global_coords=True)
    if center_xy is not None:
        low[:2] = np.maximum(low[:2], center_xy - radius)
        high[:2] = np.minimum(high[:2], center_xy + radius)
    for _ in range(attempts):
        point = np.asarray(
            (
                rng.uniform(low[0], high[0]),
                rng.uniform(low[1], high[1]),
                rng.uniform(*height_range),
            ),
            dtype=np.float64,
        )
        if not _point_over_floors(point, floor_colliders, height_range):
            continue
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = point
        if not ccol.box_intersection_test(obstacles, transform, size=camera_clearance):
            return point
    raise RejectedScene("Could not place a camera location in the room")


def _shared_doorway_center(
    first: HouseRoomResult, second: HouseRoomResult
) -> tuple[float, float]:
    for wall_index, neighbor in first.neighbors.items():
        if neighbor == second.room_index:
            return first.doorway_centers[wall_index]
    raise RejectedScene("Consecutive tour rooms do not share a door")


def _camera_accept_pred(
    camera: pf.CameraObject,
    colliders: ccol.CollisionSet,
    height_range: tuple[float, float],
    camera_clearance: float,
) -> bool:
    z = float(camera.item().matrix_world.translation.z)
    if not height_range[0] <= z <= height_range[1]:
        return False
    return camera_cube_free_space_check(
        camera,
        colliders,
        probe_size=camera_clearance,
    )


def _tour_location_samplers(
    colliders: ccol.CollisionSet,
    rooms: tuple[HouseRoomResult, ...],
    height_range: tuple[float, float],
    camera_clearance: float,
) -> list[Callable[[pf.RNG], np.ndarray]]:
    specs = [(rooms[0].floor, None)]
    for first, second in zip(rooms[:-1], rooms[1:], strict=True):
        center_xy = np.asarray(_shared_doorway_center(first, second))
        specs.extend(
            (
                (first.floor, center_xy),
                (second.floor, center_xy),
                (second.floor, None),
            )
        )
    return [
        partial(
            _sample_start_location,
            floor=floor,
            floor_colliders=ccol.collision_set([floor]),
            obstacles=colliders,
            height_range=height_range,
            camera_clearance=camera_clearance,
            center_xy=center_xy,
        )
        for floor, center_xy in specs
    ]


def _trajectory_accepted(
    camera: pf.CameraObject,
    colliders: ccol.CollisionSet,
    frame_start: int,
    frame_end: int,
    height_range: tuple[float, float],
    camera_clearance: float,
) -> bool:
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        if _camera_accept_pred(camera, colliders, height_range, camera_clearance):
            continue
        logger.info("Rejected house tour camera at frame %d", frame)
        return False
    return True


def _delete_new_cameras(existing: set[int]) -> None:
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA" and obj.as_pointer() not in existing:
            delete_object(obj)


def _house_tour_camera_attempt(
    rng: pf.RNG,
    house_objects: list[pf.MeshObject],
    colliders: ccol.CollisionSet,
    rooms: tuple[HouseRoomResult, ...],
    frame_start: int,
    frame_end: int,
    height_range: tuple[float, float],
    camera_clearance: float,
) -> pf.CameraObject | None:
    start_rng, rrt_rng = rng.spawn(2)
    location_samplers = _tour_location_samplers(
        colliders, rooms, height_range, camera_clearance
    )
    existing_cameras = {
        obj.as_pointer() for obj in bpy.data.objects if obj.type == "CAMERA"
    }
    point_accept_pred = partial(_point_at_camera_height, height_range=height_range)
    try:
        camera = rrt_camera(
            rrt_rng,
            colliders,
            house_objects,
            start_location=location_samplers[0](start_rng),
            goal_location_samplers=location_samplers[1:],
            wander_after_goals=False,
            frame_start=frame_start,
            frame_end=frame_end,
            focal_length_mm=8.0,
            rot_std_deg=(0.0, 0.0, 0.0),
            max_abs_roll_deg=0.0,
            max_abs_pitch_offset_deg=0.0,
            step_range=(0.25, 0.75),
            stride_range=(1, 4),
            min_node_dist_to_obstacle=camera_clearance,
            max_rrt_iter=300,
            max_path_retries=10,
            camera_clearance=camera_clearance,
            step_predicate=point_accept_pred,
        )
    except (RejectedScene, RRTPolicyError) as error:
        logger.info("Rejected house tour plan: %s", error)
        camera = None
    if camera is not None and _trajectory_accepted(
        camera,
        colliders,
        frame_start,
        frame_end,
        height_range,
        camera_clearance,
    ):
        return camera
    _delete_new_cameras(existing_cameras)
    return None


def _furnish_tour_rooms_rand(
    rng: pf.RNG, house: HouseResult, selected: tuple[HouseRoomResult, ...]
) -> list[HouseRoomFurnitureResult]:
    colliders = house.colliders
    furniture = []
    for room, room_rng in zip(selected, rng.spawn(len(selected)), strict=True):
        result = furnish_house_room_furniture_rand(
            room_rng,
            room.boundary_rings,
            room.floor,
            room.flat_walls,
            house.dimensions.z,
            colliders,
        )
        colliders = ccol.collision_set(
            [*colliders.objs, *result.all_objects], cache=colliders
        )
        furniture.append(result)
    return furniture


def _decorate_tour_rooms_rand(
    rng: pf.RNG,
    house: HouseResult,
    selected: tuple[HouseRoomResult, ...],
    furniture: list[HouseRoomFurnitureResult],
) -> HouseResult:
    decorated_rooms = {}
    objects = []
    small_objects = []
    lights = []
    collections = []
    rng_rooms, rng_nonstorage, rng_storage = rng.spawn(3)
    nonstorage_collection = decoration_collection_primitives_rand(rng_nonstorage)
    storage_collection = decoration_collection_primitives_and_real_rand(rng_storage)
    room_rngs = rng_rooms.spawn(len(selected))
    for room, result, room_rng in zip(selected, furniture, room_rngs, strict=True):
        small = decorate_room_small_objects_rand(
            room_rng,
            objects=result.all_objects,
            colliders=result.colliders,
            containers=result.storage_containers,
            support_tops=result.supports,
            storages=result.storages,
            nonstorage_collection=nonstorage_collection,
            storage_collection=storage_collection,
        )
        delete_objects([obj.item() for obj in result.temporary_objects])
        furniture_objects = set(result.all_objects)
        small_objects += [o for o in small.all_objects if o not in furniture_objects]
        room_objects = [*room.all_objects, *small.all_objects]
        room_lights = [*room.lights, *result.lights]
        collections.append(
            link_house_room_collection(room.room_index, room_objects, room_lights)
        )
        decorated_rooms[room.room_index] = HouseRoomResult(
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
        objects += small.all_objects
        lights += result.lights
    for collection in collections:
        collection.hide_viewport = False
    rooms = tuple(decorated_rooms.get(room.room_index, room) for room in house.rooms)
    return HouseResult(
        all_objects=[*house.all_objects, *objects],
        small_objects=[*house.small_objects, *small_objects],
        cameras=house.cameras,
        lights=[*house.lights, *lights],
        rooms=rooms,
        dimensions=house.dimensions,
        colliders=ccol.collision_set(
            [*house.colliders.objs, *objects], cache=house.colliders
        ),
        environment=house.environment,
    )


def build_scene(
    seed: int,
    trajectory_seed: int | None = None,
    frame_start: int = 1,
    frame_end: int = 240,
    resolution: tuple[int, int] = (640, 360),
    room_count: int = 5,
) -> tuple[
    list[pf.Object],
    list[pf.LightObject],
    pf.CameraObject,
    dict[str, float],
]:
    if trajectory_seed is None:
        trajectory_seed = seed
    pf.ops.object.clear_scene()
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.fps = 24
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end

    scene_rng = np.random.default_rng(seed)
    rng_house, rng_furnish, rng_small = scene_rng.spawn(3)
    trajectory_rng = np.random.default_rng(trajectory_seed)
    rng_rooms, rng_camera = trajectory_rng.spawn(2)
    logger.info("Building house %s trajectory %s", hex(seed), hex(trajectory_seed))
    times = {}
    with time_step(times, "house"):
        house = house_unfurnished_rand(rng_house, room_count=room_count)
        allowed_rooms = _connected_rooms_rand(rng_rooms, house.rooms)
        furniture = _furnish_tour_rooms_rand(rng_furnish, house, allowed_rooms)
        furniture_objects = [obj for result in furniture for obj in result.all_objects]
        house_objects = [*house.all_objects, *furniture_objects]
        tour_colliders = ccol.collision_set(house_objects, cache=house.colliders)

    height_range = (1.35, 1.75)
    camera_clearance = 0.25
    with time_step(times, "camera"):
        place = partial(
            _house_tour_camera_attempt,
            house_objects=house_objects,
            colliders=tour_colliders,
            rooms=allowed_rooms,
            frame_start=frame_start,
            frame_end=frame_end,
            height_range=height_range,
            camera_clearance=camera_clearance,
        )
        camera = repeat_attempts(place, rng_camera, attempts=10)
        if camera is None:
            raise RejectedScene("Could not plan an accepted house tour")
    with time_step(times, "small_objects"):
        house = _decorate_tour_rooms_rand(rng_small, house, allowed_rooms, furniture)
    allowed_rooms = tuple(house.rooms[room.room_index] for room in allowed_rooms)
    allowed_room_indices = [room.room_index for room in allowed_rooms]
    camera.item()["allowed_rooms"] = allowed_room_indices
    logger.info("Tour is constrained to rooms %s", allowed_room_indices)
    removed = remove_subdivision_outside_rooms(house.all_objects, allowed_rooms)
    logger.info("Removed %d subdivision modifiers outside the tour", removed)

    objects: list[pf.Object] = list(house.all_objects)
    lights = list(house.lights)
    cleanup_except([*objects, *lights, camera])
    camera.item().name = "Camera"
    return objects, lights, camera, times


def load_scene(
    blend: Path,
) -> tuple[list[pf.Object], list[pf.LightObject], pf.CameraObject, dict[str, float]]:
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene_objects = list(bpy.context.scene.objects)
    objects = [pf.Object(o) for o in scene_objects if o.type == "MESH"]
    lights = [pf.LightObject(o) for o in scene_objects if o.type == "LIGHT"]
    cameras = [o for o in scene_objects if o.type == "CAMERA"]
    if len(cameras) != 1:
        raise ValueError(f"Expected one camera in {blend}, found {len(cameras)}")
    return objects, lights, pf.CameraObject(cameras[0]), {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an RRT house tour")
    parser.add_argument("--seed", type=_parse_seed, default=None)
    parser.add_argument(
        "--trajectory_seed",
        type=_parse_seed,
        default=None,
        help="Seed for room selection and camera motion; defaults to --seed.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/house_tour"))
    parser.add_argument("--frames", type=int, nargs=2, default=(1, 240))
    parser.add_argument(
        "--render_frames",
        type=int,
        nargs=2,
        default=None,
        help="Sub-range of --frames to render. The tour is always planned over "
        "--frames, so shards render consecutive slices of one identical trajectory.",
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=(640, 360))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--room_count", type=int, default=5)
    parser.add_argument("--save_blend", type=Path, default=None)
    parser.add_argument(
        "--load_blend",
        type=Path,
        default=None,
        help="Render a scene saved by --save_blend instead of building one.",
    )
    parser.add_argument("--max_render_tris", type=int, default=32_000_000)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "big")
    trajectory_seed = args.trajectory_seed if args.trajectory_seed is not None else seed
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    frame_start, frame_end = args.frames
    render_start, render_end = args.render_frames or args.frames
    resolution = tuple(args.resolution)

    if args.load_blend is not None:
        objects, lights, camera, times = load_scene(args.load_blend)
    else:
        objects, lights, camera, times = build_scene(
            seed,
            trajectory_seed,
            frame_start,
            frame_end,
            resolution,
            args.room_count,
        )
    if args.save_blend is not None:
        pf.ops.file.save_blend(output_path=args.save_blend)
        return

    render_tris, object_tris = estimated_eval_tricount(objects)
    logger.info("Render-level triangles: %d", render_tris)
    if args.max_render_tris and render_tris > args.max_render_tris:
        raise RejectedScene(
            f"Scene exceeded triangle limit: {render_tris=} "
            f"limit={args.max_render_tris} highest_objects={object_tris[-10:]}"
        )

    render_passes = [
        RenderPass(ExportType.IMAGE, Path("%c/%f.png"), np.dtype(np.uint8)),
        RenderPass(ExportType.CAMERA, Path("%c/camera.npz"), np.dtype(np.float32)),
    ]
    with time_step(times, "render"):
        exports = render_cycles(
            objects=objects,
            lights=lights,
            camera=camera,
            output_folder=output,
            render_passes=render_passes,
            frame_start=render_start,
            frame_end=render_end,
            resolution=resolution,
            min_samples=min(32, args.samples),
            max_samples=args.samples,
            film_exposure=2.0,
        )
    write_render_metadata(
        output=output,
        seed=seed,
        trajectory_seed=trajectory_seed,
        times=times,
        exports=exports,
        build_keys={"house", "camera", "small_objects"},
        render_keys={"render"},
        n_frames=render_end - render_start + 1,
    )
    for paths in exports.values():
        for path in paths:
            print(path)


if __name__ == "__main__":
    with skip_teardown_on_exit():
        main()
