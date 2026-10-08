# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Dylan Li: original Infinigen RRT planner (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/core/util/rrt.py)
# - Karhan Kayan, Alexander Raistrick: refactor for Infinigen2

from collections.abc import Callable

import bpy
import numpy as np
import procfunc as pf

import infinigen2.scenes.placement.collision as ccol
from infinigen2.scenes.placement.retry import repeat_attempts
from infinigen2.util.errors import RejectedScene

__all__ = [
    "RRTPolicyError",
    "camera_follow_path",
    "camera_rrt_trajectory",
    "rrt_path",
]


class RRTPolicyError(ValueError):
    pass


def _point_valid(
    point: np.ndarray,
    colliders: ccol.CollisionSet,
    bbox: tuple[np.ndarray, np.ndarray],
    clearance: float,
    point_pred: Callable[[np.ndarray], bool] | None,
) -> bool:
    if np.any(point < bbox[0]) or np.any(point > bbox[1]):
        return False
    transform = np.eye(4)
    transform[:3, 3] = point
    if ccol.box_intersection_test(colliders, transform, size=clearance):
        return False
    return point_pred is None or point_pred(point)


def _edge_free(
    p1: np.ndarray, p2: np.ndarray, colliders: ccol.CollisionSet, clearance: float
) -> bool:
    delta = p2 - p1
    length = float(np.linalg.norm(delta))
    if length < 1e-8:
        return True
    axis = delta / length
    reference = (0.0, 1.0, 0.0) if abs(axis[2]) > 0.9 else (0.0, 0.0, 1.0)
    second = np.cross(axis, reference)
    second /= np.linalg.norm(second)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((axis, second, np.cross(axis, second)))
    transform[:3, 3] = (p1 + p2) / 2
    size = (length, clearance, clearance)
    return not ccol.box_intersection_test(colliders, transform, size=size)


def _steer(nodes: list[np.ndarray], target: np.ndarray, step: float) -> np.ndarray:
    dists = np.linalg.norm(np.asarray(nodes) - target, axis=1)
    nearest = nodes[int(np.argmin(dists))]
    dist = float(dists.min())
    if dist <= step:
        return target
    return nearest + (target - nearest) * (step / dist)


def _rewire(
    parents: list[int],
    costs: list[float],
    neighbors: list[int],
    dists: np.ndarray,
) -> None:
    new_index = len(costs) - 1
    for i, dist in zip(neighbors, dists, strict=True):
        if costs[new_index] + dist < costs[i]:
            parents[i] = new_index
            costs[i] = costs[new_index] + dist


def _backtrack(nodes: list[np.ndarray], parents: list[int]) -> list[np.ndarray]:
    path = []
    i = len(nodes) - 1
    while i != 0 and len(path) < len(nodes):
        path.append(nodes[i])
        i = parents[i]
    if i != 0:
        raise RRTPolicyError("RRT tree contains a cycle")
    return path[::-1]


def rrt_path(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    bbox: tuple[np.ndarray, np.ndarray],
    start: np.ndarray,
    goal: np.ndarray,
    clearance: float = 0.1,
    point_pred: Callable[[np.ndarray], bool] | None = None,
    step_range: tuple[float, float] = (1.0, 2.0),
    goal_bias: float = 0.1,
    max_iter: int = 2000,
) -> list[np.ndarray]:
    """RRT* waypoints from `start` (excluded) to `goal` (included) through points
    with a free `clearance` cube that satisfy `point_pred`, joined by edges a
    `clearance`-wide box can sweep without hitting `colliders`."""
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    for point in (start, goal):
        if not _point_valid(point, colliders, bbox, clearance, point_pred):
            raise RRTPolicyError(f"RRT endpoint {tuple(point)} is invalid")
    if _edge_free(start, goal, colliders, clearance):
        return [goal]

    nodes = [start]
    parents = [-1]
    costs = [0.0]
    for _ in range(max_iter):
        step = float(rng.uniform(*step_range))
        sample_goal = rng.random() < goal_bias
        target = goal if sample_goal else rng.uniform(bbox[0], bbox[1])
        new = _steer(nodes, target, step)
        if not _point_valid(new, colliders, bbox, clearance, point_pred):
            continue
        dists = np.linalg.norm(np.asarray(nodes) - new, axis=1)
        near = np.flatnonzero(dists <= step_range[1])
        free = [int(i) for i in near if _edge_free(nodes[i], new, colliders, clearance)]
        if not free:
            continue
        free_dists = dists[free]
        parent_slot = int(np.argmin(np.asarray(costs)[free] + free_dists))
        nodes.append(new)
        parents.append(free[parent_slot])
        costs.append(costs[free[parent_slot]] + float(free_dists[parent_slot]))
        _rewire(parents, costs, free, free_dists)
        goal_dist = float(np.linalg.norm(goal - new))
        if goal_dist > step_range[1] or not _edge_free(new, goal, colliders, clearance):
            continue
        if goal_dist > 1e-8:
            nodes.append(goal)
            parents.append(len(nodes) - 2)
            costs.append(costs[-1] + goal_dist)
        return _backtrack(nodes, parents)
    raise RRTPolicyError(f"RRT found no path from {tuple(start)} to {tuple(goal)}")


def _unwrap_angle_near(angle: float, reference: float) -> float:
    return reference + (angle - reference + np.pi) % (2 * np.pi) - np.pi


def _location_at_frame(
    keyframes: list[tuple[float, np.ndarray]], frame: float
) -> np.ndarray:
    frames = np.asarray([keyframe for keyframe, _ in keyframes])
    upper = int(np.searchsorted(frames, frame, side="right"))
    if upper == 0:
        return keyframes[0][1].copy()
    if upper == len(keyframes):
        return keyframes[-1][1].copy()
    first_frame, first_location = keyframes[upper - 1]
    second_frame, second_location = keyframes[upper]
    fraction = (frame - first_frame) / (second_frame - first_frame)
    return first_location + fraction * (second_location - first_location)


def _motion_direction_at_frame(
    keyframes: list[tuple[float, np.ndarray]],
    frame: float,
    lookahead_distance: float,
) -> np.ndarray:
    origin = _location_at_frame(keyframes, frame)
    future = [location for keyframe, location in keyframes if keyframe > frame]
    for location in future:
        if np.linalg.norm((location - origin)[:2]) >= lookahead_distance:
            return location - origin
    if future:
        return future[-1] - origin
    past = [location for keyframe, location in keyframes if keyframe < frame]
    for location in reversed(past):
        if np.linalg.norm((origin - location)[:2]) >= lookahead_distance:
            return origin - location
    if past:
        return origin - past[0]
    return np.zeros(3)


def _sample_motion_biased_rotation(
    rng: pf.RNG,
    current_rotation: np.ndarray,
    motion_direction: np.ndarray,
    rot_std_deg: tuple[float, float, float],
    yaw_motion_std_deg: float,
    max_abs_roll_rad: float,
    max_abs_pitch_offset_rad: float,
) -> np.ndarray:
    target = current_rotation.copy()
    jitter = np.deg2rad(rng.normal(0.0, np.asarray(rot_std_deg)))
    target[:2] = np.array((np.pi / 2, 0.0)) + jitter[:2]
    target[0] = np.clip(
        target[0],
        np.pi / 2 - max_abs_pitch_offset_rad,
        np.pi / 2 + max_abs_pitch_offset_rad,
    )
    target[1] = np.clip(target[1], -max_abs_roll_rad, max_abs_roll_rad)
    if np.linalg.norm(motion_direction[:2]) < 1e-8:
        return target
    motion_yaw = np.arctan2(-motion_direction[0], motion_direction[1])
    yaw_offset_deg = pf.random.clip_gaussian(
        rng, 0.0, yaw_motion_std_deg, -180.0, 180.0
    )
    target_yaw = motion_yaw + np.deg2rad(yaw_offset_deg)
    target[2] = _unwrap_angle_near(target_yaw, current_rotation[2])
    return target


def _animate_path_rotation(
    rng: pf.RNG,
    camera: pf.CameraObject,
    location_keyframes: list[tuple[float, np.ndarray]],
    frame_start: float,
    frame_end: float,
    fps: float,
    rot_std_deg: tuple[float, float, float],
    yaw_motion_std_deg: float,
    rotation_interval_sec_range: tuple[float, float],
    rotation_lookahead_m: float,
    max_abs_roll_rad: float,
    max_abs_pitch_offset_rad: float,
) -> None:
    frame = frame_start
    rotation = np.asarray(camera.item().rotation_euler, dtype=np.float64)
    while True:
        direction = _motion_direction_at_frame(
            location_keyframes, frame, rotation_lookahead_m
        )
        rotation = _sample_motion_biased_rotation(
            rng,
            rotation,
            direction,
            rot_std_deg,
            yaw_motion_std_deg,
            max_abs_roll_rad,
            max_abs_pitch_offset_rad,
        )
        camera.item().rotation_euler = rotation
        camera.item().keyframe_insert("rotation_euler", frame=frame)
        if frame >= frame_end:
            return
        interval_seconds = float(rng.uniform(*rotation_interval_sec_range))
        frame = min(frame_end, frame + interval_seconds * fps)


def camera_follow_path(
    rng: pf.RNG,
    points: list[np.ndarray],
    frame_start: int,
    frame_end: int,
    focal_length_mm: float = 15,
    rot_std_deg: tuple[float, float, float] = (20.0, 20.0, 20.0),
    yaw_motion_std_deg: float = 80.0,
    rotation_interval_sec_range: tuple[float, float] = (1.5, 4.0),
    rotation_lookahead_m: float = 1.0,
    max_abs_roll_deg: float = 25.0,
    max_abs_pitch_offset_deg: float = 25.0,
) -> pf.CameraObject:
    camera = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)
    points = [np.asarray(point, dtype=np.float64) for point in points]
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arclength = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if arclength[-1] <= 1e-8:
        location_keyframes = [(float(frame_start), points[0])]
    else:
        frames = frame_start + (frame_end - frame_start) * arclength / arclength[-1]
        location_keyframes = list(zip(frames.tolist(), points, strict=True))
    for frame, point in location_keyframes:
        camera.item().location = point
        camera.item().keyframe_insert("location", frame=frame)

    init_yaw = float(rng.uniform(-np.pi, np.pi))
    camera.item().rotation_euler = (np.pi / 2, 0.0, init_yaw)
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    _animate_path_rotation(
        rng,
        camera,
        location_keyframes,
        float(frame_start),
        float(frame_end),
        fps,
        rot_std_deg,
        yaw_motion_std_deg,
        rotation_interval_sec_range,
        rotation_lookahead_m,
        np.deg2rad(max_abs_roll_deg),
        np.deg2rad(max_abs_pitch_offset_deg),
    )
    for fcurve in camera.item().animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"
    return camera


def _random_valid_point(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    bbox: tuple[np.ndarray, np.ndarray],
    clearance: float,
    point_pred: Callable[[np.ndarray], bool] | None,
    attempts: int = 1000,
) -> np.ndarray:
    for _ in range(attempts):
        point = rng.uniform(bbox[0], bbox[1])
        if _point_valid(point, colliders, bbox, clearance, point_pred):
            return point
    raise RejectedScene("Could not find a valid RRT camera goal")


def _random_goal_leg(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    bbox: tuple[np.ndarray, np.ndarray],
    start: np.ndarray,
    clearance: float,
    point_pred: Callable[[np.ndarray], bool] | None,
    step_range: tuple[float, float],
    max_rrt_iter: int,
) -> list[np.ndarray] | None:
    goal_rng, path_rng = rng.spawn(2)
    goal = _random_valid_point(goal_rng, colliders, bbox, clearance, point_pred)
    try:
        return rrt_path(
            path_rng,
            colliders,
            bbox,
            start,
            goal,
            clearance=clearance,
            point_pred=point_pred,
            step_range=step_range,
            max_iter=max_rrt_iter,
        )
    except RRTPolicyError:
        return None


def _random_goals_path(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    bbox: tuple[np.ndarray, np.ndarray],
    n_goals: int,
    clearance: float,
    point_pred: Callable[[np.ndarray], bool] | None,
    step_range: tuple[float, float],
    max_rrt_iter: int,
    attempts_per_goal: int = 20,
) -> list[np.ndarray]:
    start_rng, legs_rng = rng.spawn(2)
    points = [_random_valid_point(start_rng, colliders, bbox, clearance, point_pred)]
    for _ in range(n_goals):
        leg = repeat_attempts(
            _random_goal_leg,
            legs_rng,
            attempts_per_goal,
            colliders=colliders,
            bbox=bbox,
            start=points[-1],
            clearance=clearance,
            point_pred=point_pred,
            step_range=step_range,
            max_rrt_iter=max_rrt_iter,
        )
        if leg is None:
            raise RejectedScene(f"RRT could not reach any goal from {points[-1]}")
        points += leg
    return points


def _goals_path(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    bbox: tuple[np.ndarray, np.ndarray],
    goals: list[np.ndarray],
    clearance: float,
    point_pred: Callable[[np.ndarray], bool] | None,
    step_range: tuple[float, float],
    max_rrt_iter: int,
) -> list[np.ndarray]:
    points = [np.asarray(goals[0], dtype=np.float64)]
    for goal in goals[1:]:
        points += rrt_path(
            rng,
            colliders,
            bbox,
            points[-1],
            goal,
            clearance=clearance,
            point_pred=point_pred,
            step_range=step_range,
            max_iter=max_rrt_iter,
        )
    return points


def camera_rrt_trajectory(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    bbox: tuple[np.ndarray, np.ndarray],
    frame_start: int,
    frame_end: int,
    goals: list[np.ndarray] | None = None,
    n_random_goals: int = 4,
    height_range: tuple[float, float] = (1.0, 2.0),
    clearance: float = 0.6,
    point_pred: Callable[[np.ndarray], bool] | None = None,
    step_range: tuple[float, float] = (1.0, 2.0),
    max_rrt_iter: int = 2000,
    focal_length_mm: float = 15,
    rot_std_deg: tuple[float, float, float] = (20.0, 20.0, 20.0),
    max_abs_roll_deg: float = 25.0,
    max_abs_pitch_offset_deg: float = 25.0,
) -> pf.CameraObject:
    path_rng, follow_rng = rng.spawn(2)
    low = np.array(bbox[0], dtype=np.float64)
    high = np.array(bbox[1], dtype=np.float64)
    low[2], high[2] = height_range
    bbox = (low, high)
    if goals is None:
        points = _random_goals_path(
            path_rng,
            colliders,
            bbox,
            n_random_goals,
            clearance,
            point_pred,
            step_range,
            max_rrt_iter,
        )
    else:
        points = _goals_path(
            path_rng,
            colliders,
            bbox,
            goals,
            clearance,
            point_pred,
            step_range,
            max_rrt_iter,
        )
    return camera_follow_path(
        follow_rng,
        points,
        frame_start,
        frame_end,
        focal_length_mm=focal_length_mm,
        rot_std_deg=rot_std_deg,
        max_abs_roll_deg=max_abs_roll_deg,
        max_abs_pitch_offset_deg=max_abs_pitch_offset_deg,
    )
