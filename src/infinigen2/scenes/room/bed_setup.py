# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import functools
from typing import NamedTuple, cast

import procfunc as pf

from infinigen2.objects import bedside_table, lamp
from infinigen2.objects.bed import BedResult, bed_rand
from infinigen2.scenes.placement import collision
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.placement.snap import snap_to_plane
from infinigen2.scenes.setup_utils import (
    BareMeshResult,
    MeshResult,
    back_face_grounded,
    retry_place,
    snap_back_front,
)
from infinigen2.util.errors import RejectedScene
from infinigen2.util.scene_cleanup import delete_object

__all__ = ["BedSetupResult", "bed_setup_rand"]


class BedSetupResult(NamedTuple):
    bed: pf.MeshObject
    mattress: pf.MeshObject | None
    bedside_tables: list[pf.MeshObject]
    bedside_lamps: list[pf.MeshObject]
    lights: list[pf.LightObject]
    all_objects: list[pf.MeshObject]
    colliders: collision.CollisionSet
    storage_containers: list[pf.MeshObject]
    storage_supports: list[pf.MeshObject]
    storages: list[pf.MeshObject]


_WALL_MARGIN_MIN = 0.0254
_WALL_MARGIN_MAX = 0.127
_BED_GENERATION_ATTEMPTS = 8


def _bed_dimensions_rand(
    rng: pf.RNG,
    room_dimensions: pf.Vector | None,
) -> pf.Vector | None:
    if room_dimensions is None:
        return None
    rng_length, rng_width, rng_thickness = rng.spawn(3)
    minimum_span = min(room_dimensions.x, room_dimensions.y)
    maximum_length = max(1.2, min(2.2, minimum_span - 0.6))
    minimum_length = min(1.85, maximum_length)
    length_mean = min(2.0, maximum_length)
    length = pf.random.clip_gaussian(
        rng_length, length_mean, 0.08, minimum_length, maximum_length
    )
    maximum_width = min(1.8, minimum_span - 0.9)
    width_options = [
        (width, 1.0) for width in (0.9, 1.2, 1.4, 1.6, 1.8) if width <= maximum_width
    ]
    if not width_options:
        width_options = [(max(0.6, maximum_width), 1.0)]
    width = pf.control.choice(rng_width, width_options)
    thickness = pf.random.uniform(rng_thickness, 0.20, 0.24)
    return pf.Vector((length, width, thickness))


def _place_bed_against_wall(
    rng: pf.RNG,
    bed_result: MeshResult,
    wall_planes: list[pf.MeshObject],
    colliders: collision.CollisionSet,
) -> MeshResult | None:
    margin = pf.random.uniform(rng, _WALL_MARGIN_MIN, _WALL_MARGIN_MAX)
    grounded = functools.partial(
        back_face_grounded,
        colliders=collision.collision_set(cast(list[pf.Object], wall_planes)),
        margin=margin,
    )
    return retry_place(
        rng,
        bed_result,
        colliders,
        _snap_bed_against_wall,
        attempts=32,
        accept_fn=grounded,
        parents=wall_planes,
        margin=margin,
    )


def _snap_bed_against_wall(
    rng: pf.RNG,
    child: MeshResult,
    parents: list[pf.MeshObject],
    margin: float,
) -> None:
    snap_back_front(
        rng,
        child,
        parents,
        placement=pf.random.uniform(rng, 0.0, 1.0),
        margin=margin,
    )


def _delete_bed_result(result: BedResult) -> None:
    delete_object(result.mattress_child.item())
    delete_object(result.mesh.item())


def _generated_bed_rand(
    rng_bed: pf.RNG,
    rng_place: pf.RNG,
    rng_retry: pf.RNG,
    dimensions: pf.Vector | None,
    wall_planes: list[pf.MeshObject] | None,
    colliders: collision.CollisionSet,
) -> BedResult:
    result = bed_rand(rng_bed, dimensions=dimensions)
    if not wall_planes:
        return result
    placed = _place_bed_against_wall(rng_place, result, wall_planes, colliders)
    if placed is not None:
        return result
    _delete_bed_result(result)
    for retry_rng in rng_retry.spawn(_BED_GENERATION_ATTEMPTS - 1):
        rng_asset, rng_placement = retry_rng.spawn(2)
        result = bed_rand(rng_asset, dimensions=dimensions)
        placed = _place_bed_against_wall(
            rng_placement,
            result,
            wall_planes,
            colliders,
        )
        if placed is not None:
            return result
        _delete_bed_result(result)
    raise RejectedScene("Could not place bed against a wall")


def bed_setup_rand(
    rng: pf.RNG,
    bed: pf.MeshObject | None = None,
    wall_planes: list[pf.MeshObject] | None = None,
    room_dimensions: pf.Vector | None = None,
    colliders: collision.CollisionSet | None = None,
) -> BedSetupResult:
    (
        r_dimensions,
        r_bed,
        r_bed_place,
        r_bed_retry,
        r_table,
        r_table_placement,
        r_table_choice,
        r_lamp,
        r_lamp_choices,
    ) = rng.spawn(9)
    dimensions = _bed_dimensions_rand(r_dimensions, room_dimensions)
    if colliders is None:
        wall_objects = cast(list[pf.Object], wall_planes or [])
        colliders = collision.collision_set(wall_objects)
    if bed is None:
        bed_result = _generated_bed_rand(
            r_bed,
            r_bed_place,
            r_bed_retry,
            dimensions,
            wall_planes,
            colliders,
        )
        bed_mesh = bed_result.mesh
        mattress = bed_result.mattress_child
    else:
        bed_mesh = bed
        mattress = next(
            (
                pf.MeshObject(child)
                for child in bed_mesh.item().children
                if child.type == "MESH" and child.name.startswith("mattress")
            ),
            None,
        )
        external = [
            obj
            for obj in colliders.objs
            if obj.item() is not bed_mesh.item()
            and (mattress is None or obj.item() is not mattress.item())
        ]
        colliders = collision.collision_set(external, cache=colliders)
        original_matrix = bed_mesh.item().matrix_world.copy()
        original_name = bed_mesh.item().name
        placement_result = BareMeshResult(bed_mesh)
        placed: MeshResult | None = placement_result
        if wall_planes:
            placed = _place_bed_against_wall(
                r_bed_place,
                placement_result,
                wall_planes,
                colliders,
            )
        if placed is None:
            bed_mesh.item().matrix_world = original_matrix
            bed_mesh.item().name = original_name
            raise RejectedScene("Could not place bed against a wall")
        bed_mesh = placed.mesh
    objects = []
    objects.append(bed_mesh)
    if mattress is not None:
        objects.append(mattress)
    anchor_objects = list(objects)
    if any(collision.intersection_test(colliders, obj) for obj in anchor_objects):
        raise RejectedScene("Bed collides at its selected placement")
    bed_mesh.item().name = "bed"

    side_table = bedside_table.bedside_table_composite_rand(r_table).mesh
    table_candidates = []
    for (parent_side, child_side), r_gap in zip(
        [("left", "right"), ("right", "left")],
        r_table_placement.spawn(2),
        strict=True,
    ):
        table_candidates.append(
            snap_to_plane(
                child=pf.ops.object.alias(side_table),
                parent=bed_mesh,
                placement=0.0,
                child_side=child_side,
                parent_side=parent_side,
                margin=pf.random.uniform(r_gap, 0.05, 0.10),
            )
        )
    left, right = table_candidates
    bedside_tables = pf.control.choice(
        r_table_choice,
        [
            ([], 0.09),
            ([left], 0.21),
            ([right], 0.21),
            ([left, right], 0.49),
        ],
    )
    objects += bedside_tables
    for table_mesh in bedside_tables:
        table_mesh.item().name = "bedside_table"

    tmin, tmax = pf.ops.attr.bbox_min_max(side_table, global_coords=False)
    table_size = pf.Vector(tmax) - pf.Vector(tmin)
    lamp_template = lamp.desk_lamp_rand(
        r_lamp,
        base_radius=0.125 * min(table_size[0], table_size[1]),
    )
    setup_colliders = collision.collision_set(
        colliders.objs + anchor_objects,
        cache=colliders,
    )
    kept_tables, setup_colliders = keep_non_colliding(
        bedside_tables,
        setup_colliders,
        key=lambda obj: obj,
    )
    kept = {table_mesh.item() for table_mesh in kept_tables}
    lamps: list[lamp.LampResult] = []
    for table_mesh, lamp_rng in zip(
        bedside_tables,
        r_lamp_choices.spawn(len(bedside_tables)),
        strict=True,
    ):
        if not pf.control.choice(lamp_rng, [(True, 0.5), (False, 0.5)]):
            continue
        if table_mesh.item() not in kept:
            continue
        lamp_mesh = pf.ops.object.alias(lamp_template.mesh)
        light_obj = pf.ops.object.duplicate(lamp_template.light, linked=True)
        assert light_obj is not None
        light = cast(pf.LightObject, light_obj)
        light.item().parent = lamp_mesh.item()
        light.item().matrix_parent_inverse.identity()
        light.item().location = lamp_template.light.item().location
        snap_to_plane(
            child=lamp_mesh,
            parent=table_mesh,
            child_side="bottom",
            parent_side="top",
            margin=0.003,
            constraint_axis=None,
        )
        light_matrix = light.item().matrix_world.copy()
        light.item().parent = None
        light.item().matrix_world = light_matrix
        lamp_mesh.item().name = "bedside_lamp"
        lamps.append(lamp.LampResult(mesh=lamp_mesh, light=light))

    lamps, setup_colliders = keep_non_colliding(lamps, setup_colliders)
    bedside_tables = kept_tables
    bedside_lamps = [result.mesh for result in lamps]
    lights = [result.light for result in lamps]
    objects = [*anchor_objects, *bedside_tables, *bedside_lamps]
    setup_colliders = collision.collision_set(
        colliders.objs + objects,
        cache=setup_colliders,
    )
    return BedSetupResult(
        bed=bed_mesh,
        mattress=mattress,
        bedside_tables=bedside_tables,
        bedside_lamps=bedside_lamps,
        lights=lights,
        all_objects=objects,
        colliders=setup_colliders,
        storage_containers=[],
        storage_supports=bedside_tables + ([mattress] if mattress is not None else []),
        storages=bedside_tables,
    )
