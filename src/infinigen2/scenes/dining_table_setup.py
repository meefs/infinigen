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
from infinigen2.scenes.setup_utils import (
    MeshResult,
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
def _chair_row(
    chair_geo: pf.ProcNode[pf.MeshObject],
    count: t.SocketOrVal[int],
    dist: t.SocketOrVal[float],
    span: t.SocketOrVal[float],
    theta: t.SocketOrVal[float] = 0.0,
    seed: t.SocketOrVal[int] = 0,
    rot_jitter: t.SocketOrVal[float] = 0.0,
    facing_rotation: t.SocketOrVal[float] = math.pi,
) -> pf.ProcNode[t.Instances]:
    """One row of `count` chairs centered over `span` along +Y at local +X=dist,
    each facing -X (toward the origin) with a small random z jitter, then the whole
    row rotated by `theta` about Z to reach any of the table's four sides."""
    count_f = count.astype(dtype=float)
    is_multi = pf.nodes.func.greater_than(a=count_f, b=1.0).astype(dtype=float)
    start = span * -0.5 * is_multi
    step = span / pf.nodes.math.maximum(count_f - 1.0, 1.0) * is_multi
    line = pf.nodes.geo.mesh_line(
        count=count,
        start_location=pf.nodes.math.combine_xyz(x=dist, y=start),
        offset=pf.nodes.math.combine_xyz(y=step),
    )
    jitter = pf.nodes.func.random_value(min=-1.0, max=1.0, seed=seed) * rot_jitter
    row = pf.nodes.geo.instance_on_points(
        points=line,
        instance=chair_geo,
        rotation=pf.nodes.math.combine_xyz(z=facing_rotation + jitter).astype(
            dtype=pf.Euler
        ),
    )
    return pf.nodes.geo.transform(
        row, rotation=pf.nodes.math.combine_xyz(z=theta).astype(dtype=pf.Euler)
    )


def arrange_dining_chairs(
    chair_obj: pf.MeshObject,
    dining_table: pf.MeshObject,
    chair_spacing: float,
    tuck: float,
    edge_margin: float,
    rot_jitter: float,
    include_ends: bool = True,
    long_chair_objs: tuple[pf.MeshObject, pf.MeshObject] | None = None,
    long_chair_tucks: tuple[float, float] | None = None,
    long_chair_face_outward: tuple[bool, bool] | None = None,
    long_chair_rot_jitters: tuple[float, float] | None = None,
) -> list[pf.MeshObject]:
    """Instance `chair_obj` around the sides of `dining_table`, facing inward, posed
    by the table's current transform, and realize the instances into one object per
    chair. Table and chair footprints are measured from bounding boxes inside the
    nodegraph. `chair_spacing` is the gap between adjacent chairs, `tuck` the meters
    the chair front is pulled in past the table edge (negative leaves a gap),
    `edge_margin` the clearance kept at each table corner so chairs never overhang
    the ends, `rot_jitter` the random per-chair z rotation (radians). `include_ends`
    toggles the chairs on the +/-Y ends."""
    # TODO replace the four per-side rows with one geonode pass: split the table bbox
    # into disconnected edge curves, resample each by length for spacing/count, then
    # instance chairs facing inward -- drops most of these args. Needs visual iteration.
    collection = pf.types.Collection([chair_obj], name="dining_chair")
    chair_geo = pf.nodes.geo.collection_info(collection, separate_children=True)
    if long_chair_objs is None:
        long_chair_objs = (chair_obj, chair_obj)
    if long_chair_tucks is None:
        long_chair_tucks = (tuck, tuck)
    if long_chair_face_outward is None:
        long_chair_face_outward = (False, False)
    if long_chair_rot_jitters is None:
        long_chair_rot_jitters = (rot_jitter, rot_jitter)
    long_chair_geos = []
    for i, long_chair_obj in enumerate(long_chair_objs):
        collection = pf.types.Collection([long_chair_obj], name=f"dining_long_side_{i}")
        long_chair_geos.append(
            pf.nodes.geo.collection_info(collection, separate_children=True)
        )

    table_info = pf.nodes.geo.object_info(dining_table)
    tbox = pf.nodes.geo.bound_box(table_info.geometry)
    tsize = pf.nodes.math.separate_xyz(tbox.max - tbox.min)
    # the bound_box min/max sockets ignore instances, so realize the chair first
    cbox = pf.nodes.geo.bound_box(pf.nodes.geo.realize_instances(chair_geo))
    csize = pf.nodes.math.separate_xyz(cbox.max - cbox.min)
    chair_depth, chair_width = csize.x, csize.y
    dist_y = tsize.y * 0.5 + chair_depth * 0.5 - tuck
    # subtract end margins and a half-chair per end so chair edges, not centers, fit
    avail_x = tsize.x - 2.0 * edge_margin
    avail_y = tsize.y - 2.0 * edge_margin
    short_pitch = chair_width + chair_spacing
    n_x = pf.nodes.math.maximum(
        pf.nodes.math.floor((avail_x + chair_spacing) / short_pitch), 1.0
    ).astype(dtype=int)
    span_x = pf.nodes.math.maximum(avail_x - chair_width, 0.0)

    def make_row(geo, count, dist, span, theta, seed, row_jitter, face_outward=False):
        return _chair_row(
            chair_geo=geo,
            count=count,
            dist=dist,
            span=span,
            theta=theta,
            seed=seed,
            rot_jitter=row_jitter,
            facing_rotation=0.0 if face_outward else math.pi,
        )

    rows = []
    long_bottom = cbox.min.z
    for i, long_chair_geo in enumerate(long_chair_geos):
        long_box = pf.nodes.geo.bound_box(
            pf.nodes.geo.realize_instances(long_chair_geo)
        )
        long_size = pf.nodes.math.separate_xyz(long_box.max - long_box.min)
        long_depth, long_width = long_size.x, long_size.y
        dist_x = tsize.x * 0.5 + long_depth * 0.5 - long_chair_tucks[i]
        long_pitch = long_width + chair_spacing
        n_y = pf.nodes.math.maximum(
            pf.nodes.math.floor((avail_y + chair_spacing) / long_pitch), 1.0
        ).astype(dtype=int)
        span_y = pf.nodes.math.maximum(avail_y - long_width, 0.0)
        rows.append(
            make_row(
                long_chair_geo,
                n_y,
                dist_x,
                span_y,
                math.pi * i,
                i + 1,
                long_chair_rot_jitters[i],
                long_chair_face_outward[i],
            )
        )
        long_bottom = pf.nodes.math.minimum(long_bottom, long_box.min.z)
    if include_ends:
        rows.append(
            make_row(chair_geo, n_x, dist_y, span_x, math.pi * 0.5, 3, rot_jitter)
        )
        rows.append(
            make_row(chair_geo, n_x, dist_y, span_x, math.pi * 1.5, 4, rot_jitter)
        )
    # chairs stand on the floor (z=0.001 like _place_on_floor), not at the table's z
    table_loc = pf.nodes.math.separate_xyz(table_info.location)
    posed = pf.nodes.geo.transform(
        pf.nodes.geo.join_geometry(rows),
        translation=pf.nodes.math.combine_xyz(
            x=table_loc.x, y=table_loc.y, z=0.001 - long_bottom
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
    rot_jitter: float | None = None,
    include_ends: bool | None = None,
    dining_table: pf.MeshObject | None = None,
) -> DiningSetupResult:
    """A dining table with a single dining chair design instanced around all four
    sides facing inward. Builds the table, one chair, then arranges copies of the
    chair around it; chairs follow the table's pose. Pass `dining_table` to arrange
    chairs around an existing (already posed) table instead of sampling one."""
    rng, rng_dims, rng_table, rng_chair_dims, rng_chair = rng.spawn(5)
    if chair_spacing is None:
        chair_spacing = pf.random.uniform(rng, 0.0625, 0.1875)
    if tuck is None:
        tuck = pf.random.clip_gaussian(rng, 0.1, 0.12, -0.08, 0.3)
    if edge_margin is None:
        edge_margin = pf.random.uniform(rng, 0.08, 0.20)
    if rot_jitter is None:
        rot_jitter = pf.random.uniform(rng, 0.05, 0.10)
    if include_ends is None:
        include_ends = pf.control.choice(rng, [(True, 1.0), (False, 1.0)])

    if dining_table is None:
        table_dimensions = table.table_dimensions_rand(rng_dims)
        dining_table = table.dining_table_rand(
            rng_table,
            dimensions=table_dimensions,
        ).mesh
    else:
        table_min, table_max = pf.ops.attr.bbox_min_max(
            dining_table,
            global_coords=False,
        )
        table_dimensions = table_max - table_min

    seat_clearance = pf.random.uniform(rng_chair_dims, 0.27, 0.30)
    chair_dims = chair.dining_chair_dimensions_rand(
        rng_chair_dims,
        seat_elevation=table_dimensions[2] - seat_clearance,
    )
    chair_res = chair.chair_rand(rng_chair, dimensions=chair_dims)

    chairs = arrange_dining_chairs(
        chair_res.mesh,
        dining_table,
        chair_spacing,
        tuck,
        edge_margin,
        rot_jitter,
        include_ends,
    )

    all_objects = [dining_table, *chairs]
    return DiningSetupResult(
        dining_table=dining_table, chairs=chairs, all_objects=all_objects
    )


def _arrange_chairs_around(rng: pf.RNG, dining_table: pf.MeshObject) -> list:
    return dining_setup_rand(rng, dining_table=dining_table).chairs


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
    rng_table, rng_place, rng_setup, rng_rug = rng.spawn(4)

    dims = table.table_dimensions_rand(rng_table)
    table_res = table.dining_table_rand(rng_table, dimensions=dims)
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
        # colliders here excludes the table so the obstruction ray hits the wall behind it
        chair_meshes, colliders = place_surrounding(
            rng_setup, diningtable_objs[0].mesh, _arrange_chairs_around, colliders
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
