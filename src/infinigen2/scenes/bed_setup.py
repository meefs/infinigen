# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from typing import NamedTuple, cast

import procfunc as pf

from infinigen2.objects import bedside_table, lamp
from infinigen2.objects.bed import bed_rand
from infinigen2.scenes.placement import collision
from infinigen2.scenes.placement.culling import keep_non_colliding
from infinigen2.scenes.placement.snap import snap_to_plane
from infinigen2.util.errors import RejectedScene

__all__ = ["BedSetupResult", "bed_setup_rand"]


class BedSetupResult(NamedTuple):
    bed: pf.MeshObject
    mattress: pf.MeshObject | None
    bedside_tables: list[pf.MeshObject]
    bedside_lamps: list[pf.MeshObject]
    lights: list[pf.LightObject]
    all_objects: list[pf.MeshObject]
    colliders: collision.CollisionSet


def bed_setup_rand(
    rng: pf.RNG,
    bed: pf.MeshObject | None = None,
    colliders: collision.CollisionSet | None = None,
) -> BedSetupResult:
    (
        r_bed,
        r_table,
        r_table_placement,
        r_table_choice,
        r_lamp,
        r_lamp_choices,
    ) = rng.spawn(6)
    if bed is None:
        bed_result = bed_rand(r_bed)
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
    if colliders is None:
        colliders = collision.collision_set([])
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
    lamp_candidates: list[tuple[pf.MeshObject, pf.MeshObject, pf.LightObject]] = []
    for table_mesh, lamp_rng in zip(
        bedside_tables,
        r_lamp_choices.spawn(len(bedside_tables)),
        strict=True,
    ):
        if not pf.control.choice(lamp_rng, [(True, 0.5), (False, 0.5)]):
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
        lamp_candidates.append((table_mesh, lamp_mesh, light))
        objects.append(lamp_mesh)

    setup_colliders = collision.collision_set(
        colliders.objs + anchor_objects,
        cache=colliders,
    )
    candidates: list[pf.MeshObject | None] = [
        *bedside_tables,
        *(lamp_mesh for _, lamp_mesh, _ in lamp_candidates),
    ]
    kept_objects, setup_colliders = keep_non_colliding(
        candidates,
        setup_colliders,
        key=lambda obj: obj,
    )
    kept = {obj.item() for obj in kept_objects}
    bedside_tables = [table for table in bedside_tables if table.item() in kept]
    lamp_candidates = [
        candidate
        for candidate in lamp_candidates
        if candidate[0].item() in kept and candidate[1].item() in kept
    ]
    bedside_lamps = [lamp_mesh for _, lamp_mesh, _ in lamp_candidates]
    lights = [light for _, _, light in lamp_candidates]
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
    )
