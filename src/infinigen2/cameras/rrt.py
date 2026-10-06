# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan: original RRT camera
# - Alexander Raistrick: collider-scoped traversal

from collections.abc import Sequence
from typing import Callable

import bpy
import numpy as np
import procfunc as pf

import infinigen2.scenes.placement.collision as ccol
from infinigen2.cameras.util import total_bbox
from infinigen2.util.errors import RejectedScene

__all__ = [
    "RRTPolicyError",
    "rrt_camera",
]


class RRTPolicyError(ValueError):
    pass


class _RRTPlanner:
    def __init__(
        self,
        rng: pf.RNG,
        colliders: ccol.CollisionSet,
        bbox: tuple[np.ndarray, np.ndarray],
        validate_node: Callable[[np.ndarray], bool],
        step_range: tuple[float, float] = (1.0, 1.0),
        stride_range: tuple[int, int] = (16, 32),
        min_node_dist_to_obstacle: float = 0.2,
        segment_probe_size: float = 0.1,
        segment_check_spacing: float = 0.08,
        segment_min_checks: int = 4,
        segment_predicate: Callable[[np.ndarray], bool] | None = None,
        max_iter: int = 2000,
    ):
        self.rng = rng
        self.colliders = colliders
        self.validate_node = validate_node
        self.bbox_min = np.asarray(bbox[0], dtype=np.float64)
        self.bbox_max = np.asarray(bbox[1], dtype=np.float64)
        self.step_range = step_range
        self.stride_range = stride_range
        self.segment_probe_size = segment_probe_size
        self.segment_check_spacing = segment_check_spacing
        self.segment_min_checks = segment_min_checks
        self.segment_predicate = segment_predicate
        self.max_iter = max_iter
        self.step = float(self.rng.uniform(*self.step_range))
        self.vertices: dict[
            tuple[float, float, float], tuple[tuple[float, float, float] | None, float]
        ] = {}

        self.collision_check_dirs: list[np.ndarray] = []
        if min_node_dist_to_obstacle > 0:
            thetas = [2 * np.pi * i / 8 for i in range(8)]
            phis = [np.pi * i / 4 for i in range(5)]
            for theta in thetas:
                for phi in phis:
                    self.collision_check_dirs.append(
                        min_node_dist_to_obstacle
                        * np.array(
                            [
                                np.cos(theta) * np.sin(phi),
                                np.sin(theta) * np.sin(phi),
                                np.cos(phi),
                            ],
                            dtype=np.float64,
                        )
                    )

    def _is_in_bbox(self, coord: np.ndarray) -> bool:
        return not (np.any(coord < self.bbox_min) or np.any(coord > self.bbox_max))

    def _line_not_valid(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> bool:
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        delta = p2 - p1
        length = float(np.linalg.norm(delta))
        if length < 1e-8:
            return not self._is_valid(p1)
        if not self._is_valid(p1) or not self._is_valid(p2):
            return True
        axis = delta / length
        reference = np.asarray((0.0, 0.0, 1.0))
        if abs(float(np.dot(axis, reference))) > 0.9:
            reference = np.asarray((0.0, 1.0, 0.0))
        second = np.cross(axis, reference)
        second /= np.linalg.norm(second)
        third = np.cross(axis, second)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.column_stack((axis, second, third))
        transform[:3, 3] = (p1 + p2) / 2
        size = (length, self.segment_probe_size, self.segment_probe_size)
        if ccol.box_intersection_test(self.colliders, transform, size=size):
            return True
        if self.segment_predicate is None:
            return False
        n_checks = max(
            self.segment_min_checks,
            int(np.ceil(length / self.segment_check_spacing)),
        )
        return not _segment_valid(
            p1,
            p2,
            self.segment_predicate,
            n_checks,
        )

    def _prox_check(self, x: np.ndarray) -> bool:
        for direction in self.collision_check_dirs:
            if self._line_not_valid(x, x + direction):
                return False
        return True

    def _is_valid(self, node: np.ndarray) -> bool:
        if not self._is_in_bbox(node):
            return False
        if not self.validate_node(node):
            return False
        return True

    def _rand_node(self) -> np.ndarray:
        return self.rng.uniform(self.bbox_min, self.bbox_max)

    def _rand_valid_node(self, max_iter: int = 500) -> np.ndarray:
        for _ in range(max_iter):
            node = self._rand_node()
            if self._is_valid(node):
                return node
        raise RRTPolicyError("RRT could not find a valid random node")

    def _parent(
        self, x: tuple[float, float, float]
    ) -> tuple[float, float, float] | None:
        return self.vertices[x][0] if x in self.vertices else None

    def _cost(self, x: tuple[float, float, float]) -> float | None:
        return self.vertices[x][1] if x in self.vertices else None

    def _get_vertices(self) -> np.ndarray:
        return np.asarray(list(self.vertices.keys()), dtype=np.float64)

    def _wireup(
        self,
        x: tuple[float, float, float],
        y: tuple[float, float, float],
    ) -> None:
        y_cost = self._cost(y)
        assert y_cost is not None
        self.vertices[x] = (y, y_cost + self._dist(np.asarray(x), np.asarray(y)))

    @staticmethod
    def _dist(pos1: np.ndarray, pos2: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(pos1) - np.asarray(pos2)))

    def _sample_free(self, target: np.ndarray, bias: float = 0.1) -> np.ndarray:
        if self.rng.random() < bias:
            return np.asarray(target)
        return self._rand_node()

    def _nearest(self, x: np.ndarray) -> tuple[float, float, float]:
        vertices = self._get_vertices()
        dists = np.linalg.norm(vertices - x[None, :], axis=1)
        return tuple(vertices[np.argmin(dists)])

    def _neighborhood(
        self,
        x: np.ndarray,
        radius: float | None = None,
        max_iter: int = 10,
    ) -> np.ndarray:
        vertices = self._get_vertices()
        num_verts = len(vertices)
        if num_verts == 0:
            return np.empty((0, 3), dtype=np.float64)

        gamma = 5.0
        eta = self.step
        nearpoints = np.empty((0, 3), dtype=np.float64)
        if radius is None:
            safe_n = max(num_verts, 2)
            r = min(gamma * ((np.log(safe_n) / safe_n) ** (1 / 3)), eta)
        else:
            r = radius

        i = 0
        while len(nearpoints) == 0:
            if i > max_iter:
                return np.empty((0, 3), dtype=np.float64)
            inside = np.linalg.norm(vertices - x[None, :], axis=1) < r
            nearpoints = vertices[inside]
            i += 1
            r += eta
        return nearpoints

    def _steer(
        self, x: np.ndarray, direction: np.ndarray
    ) -> tuple[float, float, float]:
        if np.allclose(x, direction):
            return tuple(x)
        dist = self._dist(x, direction)
        step = min(dist, self.step)
        increment = (direction - x) / dist * step
        return tuple(x + increment)

    def _choose_parent(
        self,
        xnew: tuple[float, float, float],
        xnear_arr: np.ndarray,
    ) -> tuple[tuple[float, float, float] | None, list[bool]]:
        xmin = None
        cmin = None
        collisions: list[bool] = []
        for xnear_np in xnear_arr:
            xnear = tuple(xnear_np)
            xnear_cost = self._cost(xnear)
            if xnear_cost is None:
                collisions.append(True)
                continue
            c1 = xnear_cost + self._dist(np.asarray(xnew), np.asarray(xnear))
            collide = self._line_not_valid(np.asarray(xnew), np.asarray(xnear))
            collisions.append(collide)
            if not collide and (cmin is None or c1 < cmin):
                xmin, cmin = xnear, c1
        return xmin, collisions

    def _rewire_neighborhood(
        self,
        xnew: tuple[float, float, float],
        xnear_arr: np.ndarray,
        collisions: list[bool],
    ) -> None:
        xnew_cost = self._cost(xnew)
        if xnew_cost is None:
            return
        for i, xnear_np in enumerate(xnear_arr):
            xnear = tuple(xnear_np)
            xnear_cost = self._cost(xnear)
            if xnear_cost is None:
                continue
            c2 = xnew_cost + self._dist(np.asarray(xnew), np.asarray(xnear))
            if not collisions[i] and c2 < xnear_cost:
                self._wireup(xnear, xnew)

    def generate_path(
        self,
        start: np.ndarray | tuple[float, float, float] | None = None,
        goal: np.ndarray | tuple[float, float, float] | None = None,
    ) -> list[tuple[float, float, float]]:
        x0 = (
            tuple(self._rand_valid_node())
            if start is None
            else tuple(np.asarray(start))
        )
        xt = tuple(self._rand_valid_node()) if goal is None else tuple(np.asarray(goal))
        if not self._is_valid(np.asarray(x0)):
            raise RRTPolicyError(f"RRT started with invalid node {x0}")
        if not self._is_valid(np.asarray(xt)):
            raise RRTPolicyError(f"RRT goal is invalid node {xt}")
        if not self._line_not_valid(np.asarray(x0), np.asarray(xt)):
            return [xt]

        self.vertices = {x0: (None, 0.0)}
        n_iter = 0
        while n_iter < self.max_iter:
            xrand = self._sample_free(np.asarray(xt))
            xnearest = self._nearest(xrand)
            xnew = self._steer(np.asarray(xnearest), np.asarray(xrand))
            xnew_np = np.asarray(xnew)

            if self._prox_check(xnew_np):
                xnear_arr = self._neighborhood(xnew_np)
                xmin, collisions = self._choose_parent(xnew, xnear_arr)
                if xmin is not None:
                    self._wireup(xnew, xmin)
                    self._rewire_neighborhood(xnew, xnear_arr, collisions)

                    if self._dist(
                        np.asarray(xnew), np.asarray(xt)
                    ) < self.step and not (
                        self._line_not_valid(np.asarray(xnew), np.asarray(xt))
                    ):
                        if xnew != xt:
                            self._wireup(xt, xnew)
                        break

            self.step = float(self.rng.uniform(*self.step_range))
            n_iter += 1

        if xt not in self.vertices:
            near_goal = self._neighborhood(np.asarray(xt), self.step, max_iter=1000)
            candidates = [
                tuple(node)
                for node in near_goal
                if not self._line_not_valid(np.asarray(node), np.asarray(xt))
            ]
            if candidates:
                costs = [_candidate_path_cost(self, node, xt) for node in candidates]
                self._wireup(xt, candidates[int(np.argmin(costs))])
        if xt not in self.vertices:
            raise RRTPolicyError(f"RRT could not find path from {x0} to {xt}")
        x = xt

        path: list[tuple[float, float, float]] = []
        while x != x0:
            path.append(x)
            parent = self._parent(x)
            if parent is None:
                raise RRTPolicyError(
                    "RRT path construction failed due to missing parent"
                )
            x = parent
        path.reverse()
        return path

    def next_goal(
        self,
        start: np.ndarray | tuple[float, float, float],
        max_iter: int = 100,
    ) -> np.ndarray:
        start = np.asarray(start, dtype=np.float64)
        r_range = (
            self.step_range[0] * self.stride_range[0],
            self.step_range[1] * self.stride_range[1],
        )
        theta_range = (0.0, 2 * np.pi)

        for _ in range(max_iter):
            r = self.rng.uniform(*r_range)
            z = self.rng.uniform(self.bbox_min[2], self.bbox_max[2])
            theta = self.rng.uniform(*theta_range)

            translation = (
                np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64) * r
            )
            translation[2] = z - start[2]
            nxt = start + translation
            if self._is_valid(nxt):
                return nxt
        raise RRTPolicyError(
            f"RRT could not find next goal node from start {tuple(start)}"
        )


def _candidate_path_cost(
    planner: _RRTPlanner,
    node: tuple[float, float, float],
    goal: tuple[float, float, float],
) -> float:
    cost = planner._cost(node)
    if cost is None:
        return float("inf")
    return cost + planner._dist(np.asarray(node), np.asarray(goal))


def _validate_rrt_node(
    colliders: ccol.CollisionSet,
    node: np.ndarray,
    probe_size: float = 0.1,
) -> bool:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = node
    return not ccol.box_intersection_test(
        colliders, transform=transform, size=probe_size
    )


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
    target[:2] += jitter[:2]
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


def _animate_rrt_rotation(
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


def _segment_valid(
    start: np.ndarray,
    end: np.ndarray,
    predicate: Callable[[np.ndarray], bool],
    n_checks: int,
) -> bool:
    for t in np.linspace(0, 1, n_checks + 2)[1:-1]:
        if not predicate(start + t * (end - start)):
            return False
    return True


def _sample_enclosed_start(
    planner: _RRTPlanner,
    colliders: ccol.CollisionSet,
    max_iter: int = 1000,
) -> np.ndarray:
    directions = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        )
    )
    for _ in range(max_iter):
        candidate = planner._rand_valid_node()
        origins = np.broadcast_to(candidate, directions.shape)
        locations, _index_ray, _index_tri = ccol.raycast(colliders, origins, directions)
        if len(locations) == len(directions):
            return candidate
    raise RejectedScene("Could not find an indoor valid camera start")


def _path_or_fallback(
    rng: pf.RNG,
    planner: _RRTPlanner,
    start_loc: np.ndarray,
    max_goal_attempts: int,
    max_path_retries: int,
    step_range: tuple[float, float],
) -> list[tuple[float, float, float]]:
    for _ in range(max_path_retries):
        try:
            goal = planner.next_goal(start=start_loc, max_iter=max_goal_attempts)
            path = planner.generate_path(start=start_loc, goal=goal)
        except RRTPolicyError:
            continue
        if len(path) > 0:
            return path

    for _ in range(max_goal_attempts):
        r = float(rng.uniform(step_range[0] * 0.25, step_range[1] * 1.0))
        theta = float(rng.uniform(0.0, 2 * np.pi))
        zoff = float(rng.uniform(-0.1, 0.1))
        candidate = start_loc + np.array(
            [r * np.cos(theta), r * np.sin(theta), zoff], dtype=np.float64
        )
        if not planner._is_valid(candidate):
            continue
        if planner._line_not_valid(start_loc, candidate):
            continue
        return [tuple(candidate)]

    return [tuple(start_loc)]


def _path_to_goal(
    rng: pf.RNG,
    planner: _RRTPlanner,
    start_loc: np.ndarray,
    goal_location_sampler: Callable[[pf.RNG], np.ndarray],
    max_path_retries: int,
) -> list[tuple[float, float, float]]:
    for goal_rng in rng.spawn(max_path_retries):
        goal = np.asarray(goal_location_sampler(goal_rng), dtype=np.float64)
        try:
            path = planner.generate_path(start=start_loc, goal=goal)
        except RRTPolicyError:
            continue
        if path:
            return path
    raise RRTPolicyError(f"RRT could not reach a required goal from {tuple(start_loc)}")


def _stretch_required_goal_path(
    camera: pf.CameraObject,
    location_keyframes: list[tuple[float, np.ndarray]],
    frame_start: float,
    frame_end: float,
    frame_curr: float,
    goal_ind: int,
    goal_count: int,
    path_ind: int,
    path_count: int,
) -> list[tuple[float, np.ndarray]]:
    if goal_ind < goal_count or path_ind < path_count:
        raise RRTPolicyError("RRT did not reach all required goals")
    duration = frame_curr - frame_start
    if duration <= 0:
        raise RRTPolicyError("RRT required-goal path has no duration")
    scale = (frame_end - frame_start) / duration
    action = camera.item().animation_data.action
    for fcurve in action.fcurves:
        if fcurve.data_path != "location":
            continue
        for keyframe in fcurve.keyframe_points:
            keyframe.co.x = frame_start + (keyframe.co.x - frame_start) * scale
    return [
        (frame_start + (frame - frame_start) * scale, location)
        for frame, location in location_keyframes
    ]


def rrt_camera(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    objects: list[pf.MeshObject],
    start_location: np.ndarray | tuple[float, float, float] | None = None,
    goal_location_samplers: Sequence[Callable[[pf.RNG], np.ndarray]] = (),
    wander_after_goals: bool = True,
    frame_start: int = 1,
    frame_end: int = 1,
    focal_length_mm: float = 15,
    margin: float = 0.05,
    step_range: tuple[float, float] = (1, 2),
    stride_range: tuple[int, int] = (64, 128),
    min_node_dist_to_obstacle: float = 0.4,
    max_rrt_iter: int = 2000,
    max_goal_attempts: int = 200,
    max_path_retries: int = 80,
    speed_mps_range: tuple[float, float] = (1.0, 1.5),
    rot_std_deg: tuple[float, float, float] = (20.0, 20.0, 20.0),
    yaw_motion_std_deg: float = 80.0,
    rotation_interval_sec_range: tuple[float, float] = (1.5, 4.0),
    rotation_lookahead_m: float = 1.0,
    max_abs_roll_deg: float = 25.0,
    max_abs_pitch_offset_deg: float = 25.0,
    camera_clearance: float = 0.1,
    segment_check_spacing: float = 0.08,
    step_predicate: Callable[[np.ndarray], bool] | None = None,
    n_intermediate_checks: int = 4,
) -> pf.CameraObject:
    camera = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)
    bbox = total_bbox(objects)

    bbox_min, bbox_max = np.asarray(bbox[0]).copy(), np.asarray(bbox[1]).copy()
    bbox_min += margin
    bbox_max -= margin
    if np.any(bbox_min >= bbox_max):
        raise RejectedScene("RRT camera bbox is invalid after applying margins")

    planner = _RRTPlanner(
        rng=rng,
        colliders=colliders,
        bbox=(bbox_min, bbox_max),
        validate_node=lambda node: _validate_rrt_node(
            colliders=colliders,
            node=node,
            probe_size=camera_clearance,
        )
        and (step_predicate is None or step_predicate(node)),
        step_range=step_range,
        stride_range=stride_range,
        min_node_dist_to_obstacle=min_node_dist_to_obstacle,
        segment_probe_size=camera_clearance,
        segment_check_spacing=segment_check_spacing,
        segment_min_checks=n_intermediate_checks,
        segment_predicate=step_predicate,
        max_iter=max_rrt_iter,
    )

    frame_start_f = float(frame_start)
    frame_end_f = float(frame_end)
    frame_curr = frame_start_f
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base

    start_loc = (
        _sample_enclosed_start(planner=planner, colliders=colliders)
        if start_location is None
        else np.asarray(start_location, dtype=np.float64)
    )
    if not planner._is_valid(start_loc):
        raise RRTPolicyError(f"RRT started with invalid node {tuple(start_loc)}")

    init_rot = (np.pi / 2, 0.0, float(rng.uniform(-np.pi, np.pi)))
    pf.ops.object.set_transform(camera, location=start_loc, rotation_euler=init_rot)

    camera.item().keyframe_insert("location", frame=frame_start)
    location_keyframes = [(frame_start_f, np.asarray(start_loc).copy())]

    path: list[tuple[float, float, float]] = []
    path_ind = 0
    goal_ind = 0
    max_abs_roll_rad = np.deg2rad(max_abs_roll_deg)
    max_abs_pitch_offset_rad = np.deg2rad(max_abs_pitch_offset_deg)
    while frame_curr < frame_end_f - 1e-6:
        if path_ind >= len(path):
            if goal_ind < len(goal_location_samplers):
                path = _path_to_goal(
                    rng,
                    planner,
                    start_loc,
                    goal_location_samplers[goal_ind],
                    max_path_retries,
                )
                goal_ind += 1
            elif not wander_after_goals and goal_location_samplers:
                break
            else:
                path = _path_or_fallback(
                    rng=rng,
                    planner=planner,
                    start_loc=start_loc,
                    max_goal_attempts=max_goal_attempts,
                    max_path_retries=max_path_retries,
                    step_range=step_range,
                )
            path_ind = 0

        waypoint = np.asarray(path[path_ind], dtype=np.float64)
        segment = waypoint - start_loc
        segment_dist = float(np.linalg.norm(segment))
        if segment_dist < 1e-6:
            path_ind += 1
            continue

        speed = float(rng.uniform(*speed_mps_range))
        duration_frames = max(1.0, segment_dist / max(speed, 1e-4) * fps)
        frame_next = min(frame_end_f, frame_curr + duration_frames)
        frac = (frame_next - frame_curr) / duration_frames

        next_loc = start_loc + segment * frac
        if planner._line_not_valid(start_loc, next_loc):
            path_ind += 1
            continue

        pf.ops.object.set_transform(camera, location=next_loc)

        keyframe = int(round(frame_next))
        keyframe = max(keyframe, int(np.floor(frame_curr)) + 1)
        camera.item().keyframe_insert("location", frame=keyframe)
        location_keyframes.append((float(keyframe), next_loc.copy()))

        start_loc = next_loc
        frame_curr = frame_next
        if frac >= 1.0 - 1e-6:
            path_ind += 1

    if not wander_after_goals and goal_location_samplers:
        location_keyframes = _stretch_required_goal_path(
            camera,
            location_keyframes,
            frame_start_f,
            frame_end_f,
            frame_curr,
            goal_ind,
            len(goal_location_samplers),
            path_ind,
            len(path),
        )

    _animate_rrt_rotation(
        rng,
        camera,
        location_keyframes,
        frame_start_f,
        frame_end_f,
        fps,
        rot_std_deg,
        yaw_motion_std_deg,
        rotation_interval_sec_range,
        rotation_lookahead_m,
        max_abs_roll_rad,
        max_abs_pitch_offset_rad,
    )

    action = camera.item().animation_data.action
    for fcurve in action.fcurves:
        if fcurve.data_path not in {"location", "rotation_euler"}:
            continue
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"

    return camera
