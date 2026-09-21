# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import NamedTuple

import procfunc as pf

from infinigen2.objects import chair, storage, table
from infinigen2.shaders.base_materials import fabric, leather
from infinigen2.shaders.composites import fabric_patterned
from infinigen2.shaders.functionality_lists import (
    fabric_art_rand,
    furniture_material_rand,
)
from infinigen2.util import mesh

__all__ = ["BedResult", "bed", "bed_rand"]


class BedResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.generator
def bed(
    base: pf.MeshObject,
    headboard: pf.MeshObject,
    footboard: pf.MeshObject | None,
    dimensions: pf.Vector,
    frame_height: float,
    base_material: pf.Material,
    frame_material: pf.Material,
    mattress_material: pf.Material,
) -> BedResult:
    platform = mesh.box_with_support_loops(
        size=(dimensions.x + 0.08, dimensions.y + 0.08, 0.10),
        vertices_x=4,
        vertices_y=4,
        vertices_z=4,
        support_loop_offset=(0.01, 0.01, 0.01),
    )
    platform = pf.nodes.to_mesh_object(platform)
    pf.ops.object.set_transform(platform, location=(0, 0, frame_height - 0.05))
    pf.ops.object.set_material(
        platform,
        surface=frame_material.surface,
        displacement=frame_material.displacement,
    )
    # one level here plus the assembled bed's two keeps the soft parts at three
    pf.ops.modifier.subdivide_surface(platform, levels=1)
    mattress = mesh.box_with_support_loops(
        size=dimensions,
        vertices_x=4,
        vertices_y=4,
        vertices_z=4,
        support_loop_offset=(0.12, 0.12, 0.07),
    )
    mattress = pf.nodes.to_mesh_object(mattress)
    pf.ops.object.set_transform(
        mattress, location=(0, 0, frame_height + dimensions.z * 0.5)
    )
    pf.ops.object.set_material(
        mattress,
        surface=mattress_material.surface,
        displacement=mattress_material.displacement,
    )
    pf.ops.modifier.subdivide_surface(mattress, levels=1)
    pf.ops.object.set_transform(
        headboard, location=(-dimensions.x * 0.5 - 0.04, 0, frame_height - 0.10)
    )
    pf.ops.object.set_material(
        base, surface=base_material.surface, displacement=base_material.displacement
    )
    pf.ops.object.join(platform, base)
    pf.ops.object.join(platform, headboard)
    if footboard is not None:
        pf.ops.object.set_transform(
            footboard,
            location=(dimensions.x * 0.5 + 0.04, 0, frame_height - 0.10),
            rotation_euler=(0, 0, math.pi),
        )
        pf.ops.object.join(platform, footboard)
    pf.ops.object.join(platform, mattress)
    pf.ops.uv.cube_project(platform, uv_name="UVMap")
    # a join drops each part's own stack, so the assembled bed carries the subdivision
    pf.ops.modifier.subdivide_surface(platform, levels=2, _skip_apply=True)
    return BedResult(platform)


def _bed_base_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    material: pf.Material,
) -> table.TableResult:
    rng_choice, rng_body = rng.spawn(2)

    def storage_base(
        r: pf.RNG, dims: pf.Vector, base_material: pf.Material
    ) -> table.TableResult:
        depth = dims.y * 0.5 - 0.01
        length = dims.x - 0.08
        first = storage.storage_composite_rand(
            r,
            dimensions=pf.Vector((depth, length, dims.z - 0.05)),
            n_spaces_z=1,
            frame_thickness=0.02,
            frame_material=base_material,
            row_divider_width=0.025,
            col_divider_width=0.02,
            back_width=0.02,
        ).mesh
        pf.ops.object.set_transform(first, location=(-depth * 0.5, -length * 0.5, 0))
        pf.ops.mesh.transform_apply(first, location=True, rotation=False, scale=False)
        second = first.clone()
        pf.ops.object.set_transform(
            first,
            location=(0, (depth + 0.02) * 0.5, 0),
            rotation_euler=(0, 0, math.pi * 0.5),
        )
        pf.ops.object.set_transform(
            second,
            location=(0, -(depth + 0.02) * 0.5, 0),
            rotation_euler=(0, 0, -math.pi * 0.5),
        )
        pf.ops.object.join(first, second)
        pf.ops.mesh.transform_apply(first, location=True, rotation=True, scale=False)
        return table.TableResult(first)

    def leg_base(
        r: pf.RNG, dims: pf.Vector, _material: pf.Material
    ) -> table.TableResult:
        r_choice, r_body = r.spawn(2)
        base_fn = pf.control.choice(
            r_choice,
            [
                (table.base_straight_rand, 1.0),
                (table.base_square_rand, 1.0),
            ],
        )
        return base_fn(
            r_body,
            dimensions=dims,
            leg_diameter=0.06,
            leg_placement_bottom_scale=1.0,
        )

    base_fn = pf.control.choice(
        rng_choice,
        [
            (leg_base, 2.0),
            (storage_base, 1.0),
        ],
    )
    return base_fn(rng_body, dimensions, material)


def _frame_material_rand(rng: pf.RNG, hard_material: pf.Material) -> pf.Material:
    r_choice, r_material = rng.spawn(2)
    vector = pf.nodes.shader.coord().uv
    fn = pf.control.choice(
        r_choice,
        [
            (lambda _rng, _vector: hard_material, 2.0),
            (fabric.fabric_rand, 1.0),
            (leather.leather_rand, 1.0),
        ],
    )
    return fn(r_material, vector)


def _mattress_material_rand(rng: pf.RNG) -> pf.Material:
    r_hue, r_saturation, r_value, r_choice, r_material = rng.spawn(5)
    color = pf.color.hsv_color(
        hue=pf.random.uniform(r_hue, 0.0, 1.0),
        saturation=pf.random.uniform(r_saturation, 0.0, 0.15),
        value=pf.random.uniform(r_value, 0.60, 0.90),
    )
    material_fn = pf.control.choice(
        r_choice,
        [
            (partial(fabric.fabric_rand, base_color=color), 2.0),
            (fabric_patterned.fabric_patterned_rand, 3.0),
            (fabric_art_rand, 1.0),
        ],
    )
    return material_fn(r_material, pf.nodes.shader.coord().uv)


def bed_rand(rng: pf.RNG, dimensions: pf.Vector | None = None) -> BedResult:
    r_size, r_thick, r_frame, r_head, r_base, r_style, r_material, r_fabric = rng.spawn(
        8
    )
    if dimensions is None:
        width = pf.control.choice(
            r_size, [(0.90, 1.0), (1.20, 1.0), (1.40, 1.0), (1.60, 1.0), (1.80, 1.0)]
        )
        dimensions = pf.Vector((2.0, width, pf.random.uniform(r_thick, 0.20, 0.24)))
    frame_height = pf.random.uniform(r_frame, 0.32, 0.40)
    head_height = pf.random.uniform(r_head, 0.85, 1.00)
    r_hard, r_upholstery = r_material.spawn(2)
    r_back_choice, r_head_board, r_foot_choice, r_foot_board = r_style.spawn(4)
    material = furniture_material_rand(r_hard, pf.nodes.shader.coord().uv)
    frame_material = _frame_material_rand(r_upholstery, material)
    mattress_material = _mattress_material_rand(r_fabric)
    base = _bed_base_rand(
        r_base,
        pf.Vector((dimensions.x, dimensions.y, frame_height - 0.05)),
        material,
    ).mesh

    def solid_board(r: pf.RNG, board_dimensions: pf.Vector) -> pf.MeshObject:
        top_rise = pf.random.clip_gaussian(r, 0.05, 0.035, 0.0, 0.12)
        profile = pf.Vector(
            (
                board_dimensions.x,
                board_dimensions.y,
                board_dimensions.z - top_rise * 0.75,
            )
        )
        obj = pf.nodes.to_mesh_object(
            chair.chair_back_solid(profile, top_rise=top_rise)
        )
        pf.ops.object.set_material(
            obj,
            surface=frame_material.surface,
            displacement=frame_material.displacement,
        )
        return obj

    def slatted_board(r: pf.RNG, board_dimensions: pf.Vector) -> pf.MeshObject:
        def slat_count_rand(
            r_count: pf.RNG, board_width: float, slat_width: float
        ) -> int:
            max_slats = max(2, math.floor(board_width / (1.2 * slat_width)))
            return pf.random.randint(r_count, 2, max_slats + 1)

        return chair.chair_back_rand(
            r,
            dimensions=board_dimensions,
            material=material,
            slat_count_rand=slat_count_rand,
        ).mesh

    # one style for the pair, so a footboard always matches its headboard
    board_fn = pf.control.choice(
        r_back_choice, [(solid_board, 1.0), (slatted_board, 1.0)]
    )

    def matching_board(r: pf.RNG, board_dimensions: pf.Vector) -> pf.MeshObject | None:
        return board_fn(r, board_dimensions)

    def no_board(_r: pf.RNG, _board_dimensions: pf.Vector) -> pf.MeshObject | None:
        return None

    r_head_thickness, r_head_shape = r_head_board.spawn(2)
    head = board_fn(
        r_head_shape,
        pf.Vector(
            (
                pf.random.uniform(r_head_thickness, 0.035, 0.07),
                dimensions.y + 0.08,
                head_height - frame_height + 0.10,
            )
        ),
    )
    foot_fn = pf.control.choice(r_foot_choice, [(no_board, 2.0), (matching_board, 1.0)])
    foot = foot_fn(
        r_foot_board,
        pf.Vector((0.05, dimensions.y + 0.08, 0.10 + dimensions.z * 0.75)),
    )
    return bed(
        base,
        head,
        foot,
        dimensions,
        frame_height,
        material,
        frame_material,
        mattress_material,
    )
