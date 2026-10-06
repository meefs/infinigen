# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen v1 contour decoration algorithm
# - Alexander Raistrick: functional Infinigen2 port

"""Pure sampling of the exterior polygon used by the room solver.

Ported from Infinigen v1 ``GraphMaker.suggest_dimensions`` and ``ContourFactory``.
"""

from __future__ import annotations

from itertools import pairwise
from math import exp, log, sqrt
from typing import NamedTuple

import numpy as np
import procfunc as pf

from infinigen2.scenes.house.floor_plan import Point2D, PolygonRings

__all__ = ["HouseOutlineResult", "house_outline_rand"]


class HouseOutlineResult(NamedTuple):
    boundary_rings: PolygonRings
    fillet_segments: tuple[tuple[Point2D, Point2D], ...]


def _arc(
    corner: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    radius: float,
    n: int,
) -> list[Point2D]:
    center = corner + (outgoing - incoming) * radius
    angles = np.linspace(0.0, np.pi / 2, n + 1)
    offsets = np.outer(np.sin(angles), incoming) - np.outer(np.cos(angles), outgoing)
    return [(float(x), float(y)) for x, y in center + offsets * radius]


def _square_corner(
    rng: pf.RNG,
    corner: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    max_radius: float,
) -> list[Point2D]:
    return [(float(corner[0]), float(corner[1]))]


def _chamfer_corner(
    rng: pf.RNG,
    corner: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    max_radius: float,
) -> list[Point2D]:
    return _arc(corner, incoming, outgoing, max_radius, 1)


def _round_corner(
    rng: pf.RNG,
    corner: np.ndarray,
    incoming: np.ndarray,
    outgoing: np.ndarray,
    max_radius: float,
) -> list[Point2D]:
    return _arc(corner, incoming, outgoing, max_radius, pf.random.randint(rng, 4, 7))


def house_outline_rand(
    rng: pf.RNG,
    *,
    room_count: int,
    dimensions: tuple[float, float] | None = None,
) -> HouseOutlineResult:
    """Sample a box house outline in metres, with some corners chamfered or rounded."""
    rng_dimensions, rng_corners = rng.spawn(2)
    if dimensions is None:
        slackness = exp(pf.random.uniform(rng_dimensions, log(1.1), log(1.3)))
        aspect_ratio = pf.random.uniform(rng_dimensions, 0.7, 1.0)
        area = room_count * 12.0 * slackness
        dimensions = (sqrt(area * aspect_ratio), sqrt(area / aspect_ratio))
    width, depth = dimensions
    corners = np.asarray(((0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)))
    edge_directions = np.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)))
    max_radius = min(4.0, min(width, depth) / 2) - 0.5
    ring = []
    fillet_segments = []
    for index, corner_rng in enumerate(rng_corners.spawn(4)):
        rng_choice, rng_corner = corner_rng.spawn(2)
        corner_fn = pf.control.choice(
            rng_choice,
            [(_square_corner, 0.7), (_chamfer_corner, 0.2), (_round_corner, 0.1)],
        )
        incoming = edge_directions[index - 1]
        outgoing = edge_directions[index]
        points = corner_fn(rng_corner, corners[index], incoming, outgoing, max_radius)
        ring += points
        fillet_segments += pairwise(points)
    return HouseOutlineResult(
        boundary_rings=(tuple(ring),), fillet_segments=tuple(fillet_segments)
    )
