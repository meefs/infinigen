# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""ProcFunc/Blender adapters for dependency-free house floor plans."""

from itertools import pairwise
from typing import NamedTuple

import numpy as np
import procfunc as pf
import shapely
from mathutils.geometry import tessellate_polygon
from shapely.geometry.base import BaseGeometry

from infinigen2.scenes.house.floor_plan import (
    PolygonRings,
    WallSegment,
    house_room_polygon,
    wall_is_exterior,
    wall_length,
)

__all__ = [
    "HouseCeilingCutoutResult",
    "HouseFloorMeshPlanesResult",
    "WallPlane",
    "house_ceiling_cutout",
    "house_floor_profile_to_planes",
]


class WallPlane(NamedTuple):
    """One room-facing wall quad, with the layout data its setup needs."""

    obj: pf.MeshObject
    wall_index: int
    room_index: int
    thickness: float
    length: float
    start: tuple[float, float]
    end: tuple[float, float]


class HouseFloorMeshPlanesResult(NamedTuple):
    planes_interior: list[WallPlane]
    planes_exterior: list[WallPlane]
    floors: list[pf.MeshObject]
    ceilings: list[pf.MeshObject]
    dimensions: pf.Vector


class HouseCeilingCutoutResult(NamedTuple):
    ceiling: pf.MeshObject
    sill: pf.MeshObject
    back: pf.MeshObject


def _mesh_from_faces(faces, name: str, corner_uvs: np.ndarray) -> pf.MeshObject:
    vertices = np.asarray([point for face in faces for point in face], dtype=float)
    stops = np.cumsum([len(face) for face in faces])
    starts = np.concatenate(([0], stops[:-1]))
    indices = [range(start, stop) for start, stop in zip(starts, stops, strict=True)]
    mesh = pf.ops.primitives.mesh_from_numpy(vertices=vertices, faces=indices)
    mesh.item().name = name
    pf.ops.attr.write_attribute(mesh, 1.0, "crease_edge", domain="EDGE")
    pf.ops.attr.write_attribute(
        mesh,
        np.asarray(corner_uvs, dtype=float).reshape(-1, 2),
        "UVMap",
        domain="CORNER",
    )
    return mesh


def _face_signed_area(face: tuple[tuple[float, float, float], ...]) -> float:
    return 0.5 * sum(
        x * face[(index + 1) % len(face)][1] - face[(index + 1) % len(face)][0] * y
        for index, (x, y, _) in enumerate(face)
    )


def _polygon_faces(
    polygon: BaseGeometry, z: float, ceiling: bool
) -> list[tuple[tuple[float, float, float], ...]]:
    rings = [polygon.exterior, *polygon.interiors]
    points = [list(ring.coords[:-1]) for ring in rings]
    loops = [[pf.Vector((x, y, 0.0)) for x, y in ring] for ring in points]
    vertices = [point for ring in points for point in ring]
    faces = []
    for triangle in tessellate_polygon(loops):
        face = tuple((*vertices[index], z) for index in triangle)
        signed_area = _face_signed_area(face)
        if abs(signed_area) <= 1e-12:
            continue
        points_up = signed_area > 0
        if points_up == ceiling:
            face = tuple(reversed(face))
        faces.append(face)
    return faces


def _horizontal_surface(
    polygon: BaseGeometry, z: float, name: str, ceiling: bool
) -> pf.MeshObject:
    faces = _polygon_faces(polygon, z, ceiling)
    uvs = [[(x, y) for x, y, _ in face] for face in faces]
    return _mesh_from_faces(faces, name, np.asarray(uvs))


def _hole_sides(
    corners: np.ndarray, z: float, depth: float
) -> list[tuple[tuple[float, float, float], ...]]:
    sides = []
    for (x0, y0), (x1, y1) in zip(corners, np.roll(corners, -1, axis=0), strict=True):
        sides.append(
            ((x1, y1, z), (x0, y0, z), (x0, y0, z + depth), (x1, y1, z + depth))
        )
    return sides


def _side_uvs(
    sides: list[tuple[tuple[float, float, float], ...]], depth: float
) -> np.ndarray:
    lengths = [float(np.linalg.norm(np.subtract(side[0], side[1]))) for side in sides]
    return np.asarray(
        [((0, 0), (length, 0), (length, depth), (0, depth)) for length in lengths]
    )


def house_ceiling_cutout(
    polygon: BaseGeometry,
    centers: np.ndarray,
    footprint: tuple[float, float],
    height: float,
    depth: float,
) -> HouseCeilingCutoutResult:
    """Pierce a room ceiling polygon with footprint boxes lying fully inside it."""
    z = height + 0.005
    half = np.asarray(footprint) / 2
    unit = np.asarray(((-1, -1), (1, -1), (1, 1), (-1, 1)))
    holes = [center + unit * half for center in centers]
    pierced = shapely.Polygon(polygon.exterior, [*polygon.interiors, *holes])
    sides = [side for hole in holes for side in _hole_sides(hole, z, depth)]
    rings = [np.asarray(r.coords[:-1]) for r in (polygon.exterior, *polygon.interiors)]
    rim = [side for ring in rings for side in _hole_sides(ring, z, depth)]
    top = _polygon_faces(pierced, z + depth, False)
    top_uvs = np.asarray([(x, y) for face in top for x, y, _ in face])
    back_uvs = np.concatenate([top_uvs, _side_uvs(rim, depth).reshape(-1, 2)])
    return HouseCeilingCutoutResult(
        ceiling=_horizontal_surface(pierced, z, "house_ceiling_cut", True),
        sill=_mesh_from_faces(sides, "house_ceiling_sill", _side_uvs(sides, depth)),
        back=_mesh_from_faces(top + rim, "house_ceiling_back", back_uvs),
    )


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _edge_wall_spans(
    walls: tuple[WallSegment, ...],
    room_index: int,
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[tuple[int, tuple[float, float], tuple[float, float]]]:
    edge_start = np.asarray(start, dtype=float)
    edge_end = np.asarray(end, dtype=float)
    edge_length = float(np.linalg.norm(edge_end - edge_start))
    direction = (edge_end - edge_start) / edge_length
    candidates = []
    for wall_index, wall in enumerate(walls):
        if room_index not in wall.room_indices:
            continue
        wall_start = np.asarray(wall.start, dtype=float)
        wall_end = np.asarray(wall.end, dtype=float)
        wall_direction = (wall_end - wall_start) / wall_length(wall)
        if abs(_cross_2d(direction, wall_direction)) > 1e-6:
            continue
        distance = abs(_cross_2d(wall_start - edge_start, direction))
        if abs(distance - wall.thickness / 2) > 1e-5:
            continue
        projections = (
            float(np.dot(wall_start - edge_start, direction)),
            float(np.dot(wall_end - edge_start, direction)),
        )
        low, high = sorted(projections)
        if min(high, edge_length) - max(low, 0.0) <= 1e-6:
            continue
        candidates.append((low, high, wall_index))
    candidates.sort()
    if not candidates:
        raise ValueError(f"Finished room edge {start}->{end} has no source wall")
    cuts = [0.0]
    cuts.extend(
        (first[1] + second[0]) / 2
        for first, second in zip(candidates, candidates[1:], strict=False)
    )
    cuts.append(edge_length)
    return [
        (
            candidate[2],
            tuple(edge_start + direction * low),
            tuple(edge_start + direction * high),
        )
        for candidate, low, high in zip(candidates, cuts, cuts[1:], strict=False)
    ]


def _polygon_edges(
    polygon: BaseGeometry,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    rings = [polygon.exterior, *polygon.interiors]
    return [edge for ring in rings for edge in pairwise(ring.coords)]


def _wall_plane(
    wall_index: int,
    room_index: int,
    start: tuple[float, float],
    end: tuple[float, float],
    height: float,
    thickness: float,
) -> WallPlane:
    low = -0.01
    high = height + 0.01
    x0, y0 = start
    x1, y1 = end
    # Square-room convention: first edge is vertical (V/up), second is U/along-wall.
    quad = (x0, y0, low), (x0, y0, high), (x1, y1, high), (x1, y1, low)
    length = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
    uvs = ((0.0, low), (0.0, high), (length, high), (length, low))
    name = f"house_wall.{wall_index:02d}.{room_index:02d}"
    return WallPlane(
        obj=_mesh_from_faces([quad], name, np.asarray(uvs)),
        wall_index=wall_index,
        room_index=room_index,
        thickness=thickness,
        length=length,
        start=start,
        end=end,
    )


def house_floor_profile_to_planes(
    room_boundaries: tuple[PolygonRings, ...],
    walls: tuple[WallSegment, ...],
    height: float = 2.8,
) -> HouseFloorMeshPlanesResult:
    """Build one room-facing quad per wall side, plus per-room floors and ceilings."""
    planes = []
    floors = []
    ceilings = []
    for room_index, rings in enumerate(room_boundaries):
        polygon = house_room_polygon(rings)
        for start, end in _polygon_edges(polygon):
            spans = _edge_wall_spans(walls, room_index, start, end)
            planes += [
                _wall_plane(
                    wall_index,
                    room_index,
                    low,
                    high,
                    height,
                    walls[wall_index].thickness,
                )
                for wall_index, low, high in spans
            ]
        floor_name = f"room_floor.{room_index:02d}"
        floors.append(_horizontal_surface(polygon, -0.005, floor_name, False))
        ceiling_name = f"room_ceiling.{room_index:02d}"
        ceilings.append(
            _horizontal_surface(polygon, height + 0.005, ceiling_name, True)
        )
    corners = np.asarray([(*wall.start, *wall.end) for wall in walls])
    extent = corners.reshape(-1, 2).max(axis=0)
    exterior = [p for p in planes if wall_is_exterior(walls[p.wall_index])]
    interior = [p for p in planes if not wall_is_exterior(walls[p.wall_index])]
    return HouseFloorMeshPlanesResult(
        planes_interior=interior,
        planes_exterior=exterior,
        floors=floors,
        ceilings=ceilings,
        dimensions=pf.Vector((*extent, height)),
    )
