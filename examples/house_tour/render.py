#!/usr/bin/env python
"""Render an RRT camera tour constrained to connected rooms of a house."""

# ruff: noqa: I001, E402

import argparse
import logging
import os
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

from infinigen2.cameras import (
    camera_cube_free_space_check,
    camera_rrt_trajectory,
    total_bbox,
)
from infinigen2.cameras.rrt import RRTPolicyError
from infinigen2.exporters.realize_mesh import bake_shared_modifier_prefixes
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


def _random_point_in_room(
    rng: pf.RNG,
    room: HouseRoomResult,
    colliders: ccol.CollisionSet,
    height_range: tuple[float, float],
    camera_clearance: float,
    attempts: int = 1000,
) -> np.ndarray:
    low, high = pf.ops.attr.bbox_min_max(room.floor, global_coords=True)
    floor_colliders = ccol.collision_set([room.floor])
    transform = np.eye(4)
    for _ in range(attempts):
        x, y = rng.uniform(low[:2], high[:2])
        point = np.array((x, y, rng.uniform(*height_range)))
        if not _point_over_floors(point, floor_colliders, height_range):
            continue
        transform[:3, 3] = point
        if not ccol.box_intersection_test(colliders, transform, size=camera_clearance):
            return point
    raise RejectedScene("Could not place a camera location in the room")


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


def _doorway_point(
    room: HouseRoomResult,
    next_room: HouseRoomResult,
    height_range: tuple[float, float],
) -> np.ndarray:
    wall = next(
        wall
        for wall, neighbor in room.neighbors.items()
        if neighbor == next_room.room_index
    )
    x, y = room.doorway_centers[wall]
    return np.array((x, y, np.mean(height_range)))


def _next_tour_room(
    rng: pf.RNG,
    room: HouseRoomResult,
    prev_room: HouseRoomResult | None,
    by_index: dict[int, HouseRoomResult],
) -> HouseRoomResult:
    neighbors = sorted(set(room.neighbors.values()) & set(by_index))
    forward = [i for i in neighbors if prev_room is None or i != prev_room.room_index]
    options = forward or neighbors
    return by_index[options[int(rng.integers(len(options)))]]


def _tour_goals(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    rooms: tuple[HouseRoomResult, ...],
    height_range: tuple[float, float],
    camera_clearance: float,
    distance: float,
    extra_point_prob: float = 0.25,
) -> list[np.ndarray]:
    room_point = partial(
        _random_point_in_room,
        rng,
        colliders=colliders,
        height_range=height_range,
        camera_clearance=camera_clearance,
    )
    by_index = {room.room_index: room for room in rooms}
    prev_room = None
    room = rooms[0]
    goals = [room_point(room)]
    while np.linalg.norm(np.diff(goals, axis=0), axis=1).sum() < distance:
        next_room = _next_tour_room(rng, room, prev_room, by_index)
        goals.append(_doorway_point(room, next_room, height_range))
        prev_room, room = room, next_room
        goals.append(room_point(room))
        if rng.random() < extra_point_prob:
            goals.append(room_point(room))
    return goals


def _house_tour_camera_attempt(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    rooms: tuple[HouseRoomResult, ...],
    frame_start: int,
    frame_end: int,
    height_range: tuple[float, float],
    camera_clearance: float,
    speed_mps_range: tuple[float, float] = (0.67, 1.0),
) -> pf.CameraObject | None:
    goals_rng, camera_rng = rng.spawn(2)
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    speed = goals_rng.uniform(*speed_mps_range)
    distance = speed * (frame_end - frame_start) / fps
    plan_clearance = camera_clearance * np.sqrt(3)
    try:
        goals = _tour_goals(
            goals_rng, colliders, rooms, height_range, plan_clearance, distance
        )
        logger.info("Tour has %d goals for %.1fm of walking", len(goals), distance)
        camera = camera_rrt_trajectory(
            camera_rng,
            colliders,
            total_bbox([room.floor for room in rooms]),
            frame_start,
            frame_end,
            goals=goals,
            height_range=height_range,
            clearance=plan_clearance,
            step_range=(0.25, 0.75),
            max_rrt_iter=1000,
            focal_length_mm=8.0,
            rot_std_deg=(10.0, 0.0, 0.0),
            max_abs_roll_deg=0.0,
            max_abs_pitch_offset_deg=15.0,
        )
    except (RejectedScene, RRTPolicyError) as error:
        logger.info("Rejected house tour plan: %s", error)
        return None
    if _trajectory_accepted(
        camera, colliders, frame_start, frame_end, height_range, camera_clearance
    ):
        return camera
    delete_object(camera.item())
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

    height_range = (1.1, 1.9)
    camera_clearance = 0.25
    with time_step(times, "camera"):
        place = partial(
            _house_tour_camera_attempt,
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

    bake_shared_modifier_prefixes(objects)
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
