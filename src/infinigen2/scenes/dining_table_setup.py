# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
import math
from typing import NamedTuple, TypeVar

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import chair, rug, table
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.placement.culling import place_surrounding
from infinigen2.scenes.placement.distribute import propagate_modifiers_to_instances
from infinigen2.scenes.placement.retry import repeat_attempts
from infinigen2.scenes.setup_utils import (
    MeshResult,
    jitter_object_rotation_rand,
    retry_place,
)

__all__ = [
    "DiningSetupResult",
    "DiningTableSetupResult",
    "arrange_dining_chairs",
    "dining_setup_rand",
    "dining_table_setup_rand",
]

logger = logging.getLogger(__name__)
MR = TypeVar("MR", bound=MeshResult)
_CHAIR_POSITION_JITTER = 0.30


class _BareMeshResult(NamedTuple):
    mesh: pf.MeshObject


class DiningSetupResult(NamedTuple):
    dining_table: pf.MeshObject
    chairs: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]


class DiningTableSetupResult(NamedTuple):
    dining_tables: list[MeshResult]
    dining_chairs: list[MeshResult]
    rugs: list[pf.MeshObject]
    all_objects: list[pf.MeshObject]


def _place_on_floor(rng: pf.RNG, child: MR, room_dimensions: pf.Vector) -> None:
    bmin, _ = pf.ops.attr.bbox_min_max(child.mesh, global_coords=False)
    child.mesh.item().location = (
        pf.random.uniform(rng, 0.2, 0.8) * room_dimensions.x,
        pf.random.uniform(rng, 0.4, 0.6) * room_dimensions.y,
        0.001 - bmin[2],
    )


def _rug_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector,
) -> list[pf.MeshObject]:
    wall_clearance = 0.3
    avail_x = room_dimensions.x - 2 * wall_clearance
    avail_y = room_dimensions.y - 2 * wall_clearance
    length = pf.random.uniform(rng, min(1.0, avail_x), avail_x)
    width = pf.random.uniform(rng, min(1.0, avail_y), avail_y)
    thickness = pf.random.uniform(rng, 0.01, 0.02)
    result = rug.rug_rand(rng, dimensions=pf.Vector((length, width, thickness)))
    result.mesh.item().name = rug.rug_rand.__name__
    cx = pf.random.uniform(
        rng,
        wall_clearance + length / 2,
        room_dimensions.x - wall_clearance - length / 2,
    )
    cy = pf.random.uniform(
        rng, wall_clearance + width / 2, room_dimensions.y - wall_clearance - width / 2
    )
    pf.ops.object.set_transform(result.mesh, location=(cx, cy, 0.001))
    return [result.mesh]


def _place_in_free_floorspace(
    rng: pf.RNG,
    child: MR,
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
    clearance: float = 2.0,
    attempts: int = 7,
) -> MR | None:
    """Place `child` at a random floor location whose `clearance`x-footprint box
    (at the child's own height) clears all existing colliders. Returns the placed
    child, or None if no clear spot is found.
    """

    def footprint_clears(mesh: pf.MeshObject) -> bool:
        lo, hi = (
            np.array(b) for b in pf.ops.attr.bbox_min_max(mesh, global_coords=True)
        )
        ext = hi - lo
        transform = np.eye(4)
        transform[:3, 3] = (lo + hi) / 2
        return not ccol.box_intersection_test(
            colliders, transform, [clearance * ext[0], clearance * ext[1], ext[2]]
        )

    return retry_place(
        rng,
        child,
        colliders,
        _place_on_floor,
        attempts=attempts,
        accept_fn=footprint_clears,
        room_dimensions=room_dimensions,
    )


@pf.nodes.node_function
def _chairs_on_edge(
    chair_geo: pf.ProcNode[pf.MeshObject],
    edge: pf.ProcNode[pf.CurveObject],
    inward: t.SocketOrVal[pf.Vector],
    spacing: t.SocketOrVal[float],
    edge_offset: t.SocketOrVal[float],
    disorder: t.SocketOrVal[float],
    seed: t.SocketOrVal[int],
) -> pf.ProcNode[t.Instances]:
    points = pf.nodes.geo.curve_to_points_length(curve=edge, length=spacing)
    position_jitter = disorder * _CHAIR_POSITION_JITTER
    backward = pf.nodes.func.random_value(
        min=0.0,
        max=position_jitter,
        seed=pf.nodes.math.add(seed, 1).astype(dtype=int),
    )
    lateral = pf.nodes.func.random_value(
        min=position_jitter * -1.0,
        max=position_jitter,
        seed=pf.nodes.math.add(seed, 2).astype(dtype=int),
    )
    outward = inward * -1.0
    chair_centers = pf.nodes.geo.set_position(
        geometry=points.points,
        offset=outward * (edge_offset + backward) + points.tangent * lateral,
    )
    rotation = pf.nodes.func.axes_to_rotation(
        primary_axis_vector=inward,
        secondary_axis_vector=(0.0, 0.0, 1.0),
        primary_axis="X",
        secondary_axis="Z",
    )
    return pf.nodes.geo.instance_on_points(
        points=chair_centers,
        instance=chair_geo,
        rotation=rotation,
    )


def arrange_dining_chairs(
    chair_obj: pf.MeshObject,
    dining_table: pf.MeshObject,
    chair_spacing: float,
    tuck: float,
    edge_margin: float,
    include_ends: bool = True,
    disorder: float = 1.0,
    offset_seed: int = 0,
) -> list[pf.MeshObject]:
    """Instance one chair design along the table's bounding-box edge curves."""
    collection = pf.types.Collection([chair_obj], name="dining_chair")
    chair_geo = pf.nodes.geo.collection_info(collection, separate_children=True)
    table_info = pf.nodes.geo.object_info(dining_table)
    tbox = pf.nodes.geo.bound_box(table_info.geometry)
    tmin = pf.nodes.math.separate_xyz(tbox.min)
    tmax = pf.nodes.math.separate_xyz(tbox.max)
    cbox = pf.nodes.geo.bound_box(pf.nodes.geo.realize_instances(chair_geo))
    csize = pf.nodes.math.separate_xyz(cbox.max - cbox.min)
    chair_depth, chair_width = csize.x, csize.y
    trim = edge_margin + chair_width * 0.5
    pitch = chair_width + chair_spacing
    edge_offset = chair_depth * 0.5 - tuck

    right_edge = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(x=tmax.x, y=tmin.y + trim),
        end=pf.nodes.math.combine_xyz(x=tmax.x, y=tmax.y - trim),
    )
    right = _chairs_on_edge(
        chair_geo,
        right_edge,
        (-1.0, 0.0, 0.0),
        pitch,
        edge_offset,
        disorder,
        offset_seed + 1,
    )
    left_edge = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(x=tmin.x, y=tmin.y + trim),
        end=pf.nodes.math.combine_xyz(x=tmin.x, y=tmax.y - trim),
    )
    left = _chairs_on_edge(
        chair_geo,
        left_edge,
        (1.0, 0.0, 0.0),
        pitch,
        edge_offset,
        disorder,
        offset_seed + 2,
    )
    edge_instances = [right, left]
    if include_ends:
        top_edge = pf.nodes.geo.curve_line(
            start=pf.nodes.math.combine_xyz(x=tmin.x + trim, y=tmax.y),
            end=pf.nodes.math.combine_xyz(x=tmax.x - trim, y=tmax.y),
        )
        top = _chairs_on_edge(
            chair_geo,
            top_edge,
            (0.0, -1.0, 0.0),
            pitch,
            edge_offset,
            disorder,
            offset_seed + 3,
        )
        bottom_edge = pf.nodes.geo.curve_line(
            start=pf.nodes.math.combine_xyz(x=tmin.x + trim, y=tmin.y),
            end=pf.nodes.math.combine_xyz(x=tmax.x - trim, y=tmin.y),
        )
        bottom = _chairs_on_edge(
            chair_geo,
            bottom_edge,
            (0.0, 1.0, 0.0),
            pitch,
            edge_offset,
            disorder,
            offset_seed + 4,
        )
        edge_instances.extend([top, bottom])
    table_loc = pf.nodes.math.separate_xyz(table_info.location)
    posed = pf.nodes.geo.transform(
        pf.nodes.geo.join_geometry(edge_instances),
        translation=pf.nodes.math.combine_xyz(
            x=table_loc.x, y=table_loc.y, z=0.001 - cbox.min.z
        ),
        rotation=table_info.rotation,
    )
    chairs = pf.nodes.to_aliases(posed)
    propagate_modifiers_to_instances([chair_obj], chairs)
    return chairs


def dining_setup_rand(
    rng: pf.RNG,
    chair_spacing: float | None = None,
    tuck: float | None = None,
    edge_margin: float | None = None,
    include_ends: bool | None = None,
    dining_table: pf.MeshObject | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> DiningSetupResult:
    """A dining table with a single dining chair design instanced around all four
    sides facing inward. Builds the table, one chair, then arranges copies of the
    chair around it; chairs follow the table's pose. Pass `dining_table` to arrange
    chairs around an existing (already posed) table instead of sampling one."""
    if colliders is None:
        colliders = ccol.collision_set([])
    rng_table_choice, rng_table, rng_attempts, rng_jitter_objects = rng.spawn(4)

    if dining_table is None:
        table_fn = pf.control.choice(
            rng_table_choice,
            [
                (table.dining_table_rand, 1.0),
                (table.circular_dining_table_rand, 1.0),
            ],
        )
        dining_table = table_fn(rng_table).mesh
    table_min, table_max = pf.ops.attr.bbox_min_max(
        dining_table,
        global_coords=False,
    )
    table_dimensions = table_max - table_min

    def attempt(rng: pf.RNG) -> tuple[list[pf.MeshObject], float] | None:
        rng_params, rng_chair_dims, rng_chair, rng_disorder = rng.spawn(4)
        spacing = chair_spacing
        if spacing is None:
            spacing = pf.random.uniform(rng_params, 0.0625, 0.1875)
        chair_tuck = tuck
        if chair_tuck is None:
            chair_tuck = pf.random.clip_gaussian(rng_params, 0.1, 0.12, -0.08, 0.3)
        margin = edge_margin
        if margin is None:
            margin = pf.random.uniform(rng_params, 0.08, 0.20)
        ends = include_ends
        if ends is None:
            ends = pf.control.choice(rng_params, [(True, 1.0), (False, 1.0)])

        seat_clearance = pf.random.uniform(rng_chair_dims, 0.27, 0.30)
        chair_dims = chair.dining_chair_dimensions_rand(
            rng_chair_dims,
            seat_elevation=table_dimensions[2] - seat_clearance,
        )
        chair_res = chair.chair_rand(rng_chair, dimensions=chair_dims)
        disorder = pf.random.uniform(rng_disorder, 0.0, 1.0) ** 2
        chairs = arrange_dining_chairs(
            chair_res.mesh,
            dining_table,
            spacing,
            chair_tuck,
            margin,
            ends,
            disorder=disorder,
            offset_seed=pf.random.randint(rng_params, 0, 2**20),
        )
        max_angle = (
            pf.random.uniform(rng_disorder, math.radians(30), math.radians(52.5))
            * disorder
        )
        arrangement_colliders = ccol.collision_set([dining_table, *chairs])
        if ccol.any_self_collision(arrangement_colliders):
            return None
        if any(ccol.intersection_test(colliders, chair_obj) for chair_obj in chairs):
            return None
        return chairs, max_angle

    arrangement = repeat_attempts(attempt, rng_attempts, attempts=12)
    if arrangement is None:
        raise RuntimeError("Could not generate a collision-free dining arrangement")
    chairs, max_angle = arrangement

    rotation_colliders = ccol.collision_set(
        [*colliders.objs, dining_table, *chairs], cache=colliders
    )
    chair_rngs = rng_jitter_objects.spawn(len(chairs))
    for chair_obj, chair_rng in zip(chairs, chair_rngs, strict=True):
        jitter_object_rotation_rand(chair_rng, chair_obj, max_angle, rotation_colliders)

    all_objects = [dining_table, *chairs]
    return DiningSetupResult(
        dining_table=dining_table, chairs=chairs, all_objects=all_objects
    )


def dining_table_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject] | None = None,
    room_dimensions: pf.Vector | None = None,
    colliders: ccol.CollisionSet | None = None,
) -> DiningTableSetupResult:
    """Place a dining table in clear floor space, arrange chairs around it, and
    optionally add a rug. Chairs are culled against `colliders`, retrying the
    arrangement with a fresh chair design and margins when too few survive."""
    del wall_planes
    if room_dimensions is None:
        room_dimensions = pf.Vector((5.0, 15.0, 3.0))
    if colliders is None:
        colliders = ccol.collision_set([])
    rng_table_choice, rng_table, rng_place, rng_setup, rng_rug = rng.spawn(5)

    table_fn = pf.control.choice(
        rng_table_choice,
        [
            (table.dining_table_rand, 1.0),
            (table.circular_dining_table_rand, 1.0),
        ],
    )
    table_res = table_fn(rng_table)
    dining_table = _BareMeshResult(mesh=table_res.mesh)

    placed = _place_in_free_floorspace(
        rng_place,
        dining_table,
        room_dimensions,
        colliders,
    )
    diningtable_objs = [placed] if placed is not None else []
    logger.info(f"Placed {len(diningtable_objs)} dining tables")

    chair_objs: list[MeshResult] = []
    if diningtable_objs:

        def arrange_chairs(rng: pf.RNG, parent: pf.MeshObject) -> list[pf.MeshObject]:
            return dining_setup_rand(
                rng, dining_table=parent, colliders=colliders
            ).chairs

        # colliders here excludes the table so the obstruction ray hits the wall behind it
        chair_meshes, colliders = place_surrounding(
            rng_setup, diningtable_objs[0].mesh, arrange_chairs, colliders
        )
        chair_objs = [_BareMeshResult(mesh=c) for c in chair_meshes]
        logger.info(f"Kept {len(chair_objs)} dining chairs after obstruction/collision")

    colliders = ccol.collision_set(
        colliders.objs + [r.mesh for r in diningtable_objs], cache=colliders
    )
    rug_func = pf.control.choice(
        rng_rug,
        [
            (_rug_rand, 1.0),
            (lambda *_, **__: [], 1.0),
        ],
    )
    rug_objs = rug_func(rng_rug, room_dimensions=room_dimensions)
    all_objects = [r.mesh for r in diningtable_objs + chair_objs] + rug_objs
    return DiningTableSetupResult(diningtable_objs, chair_objs, rug_objs, all_objects)
