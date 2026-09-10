# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging

import procfunc as pf

from infinigen2.objects import storage
from infinigen2.scenes.placement.distribute import propagate_modifiers_to_instances
from infinigen2.scenes.room.wall_base import (
    WallResult,
    _extrude_for_thickness,
    _fit_grid_margins,
    _plain_wall,
    _plane_to_posed_canonical_mesh,
    _resolve_wall_inputs,
    _seat_upright_cabinet,
    _subdivide_wall_plane,
    _wall_storage_width_rand,
    _wall_uv_dimensions,
    upright_cabinet_footprint,
)
from infinigen2.shaders.functionality_lists import furniture_material_rand
from infinigen2.util import mesh as mesh_util
from infinigen2.uv_surface import grid_placement

__all__ = [
    "wall_board_shelf_rand",
    "wall_storage_flush_rand",
]

logger = logging.getLogger(__name__)


@pf.tracer.grammar
def wall_board_shelf_rand(
    rng: pf.RNG,
    wall: pf.MeshObject | None = None,
    wall_material: pf.Material | None = None,
    wall_thickness: float = 0.05,
) -> WallResult:
    rng, wall, wall_material = _resolve_wall_inputs(rng, wall, wall_material)
    wall_width, wall_height = _wall_uv_dimensions(wall)

    shelf_depth = pf.random.uniform(rng, 0.27, 0.675)
    shelf_thickness = pf.random.uniform(rng, 0.02, 0.05)
    spacing_x = pf.random.uniform(rng, 0.03, 0.25)
    spacing_y = pf.random.uniform(rng, 0.5, 0.65)

    # shelf width spans 0.7m up to the full available wall, skewed toward narrow
    # (squared) so multi-column runs are common while a single full-width shelf
    # stays possible; the column count emerges from the width+spacing+margin fit
    # (like the window grid), never chosen up front
    side_margin = wall_width * pf.random.uniform(rng, 0.04, 0.25) * 2.0
    available_x = wall_width - side_margin
    shelf_width = (
        0.7 + (max(0.7, available_x) - 0.7) * pf.random.uniform(rng, 0.0, 1.0) ** 2
    )

    # canonical wall frame: X=depth out of wall, Y=width, Z=thickness up
    cube = pf.nodes.geo.mesh_cube(size=(shelf_depth, shelf_width, shelf_thickness))
    slab_geo = pf.nodes.geo.transform(
        cube.mesh, translation=(shelf_depth * 0.5, 0.0, 0.0)
    )
    slab_geo = mesh_util.metric_box_uv(slab_geo)
    slab = pf.nodes.to_mesh_object(slab_geo)

    footprint = pf.nodes.to_mesh_object(
        pf.nodes.geo.mesh_cube(size=(0.0, shelf_width, shelf_thickness)).mesh
    )
    vec = pf.nodes.shader.coord().uv
    shelf_material = furniture_material_rand(rng, vec)
    pf.ops.object.set_material(
        slab,
        surface=shelf_material.surface,
        displacement=shelf_material.displacement,
    )

    top_frac = pf.random.uniform(rng, 0.12, 0.25)
    bot_frac = pf.random.uniform(rng, top_frac, 0.50)
    margin_top = wall_height * top_frac
    margin_bottom = wall_height * bot_frac
    margin_split = pf.random.uniform(rng, 0.375, 0.625)

    uv_meters = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute

    grid_res = grid_placement.grid_from_spacing(
        uv_surface=wall,
        target_uv=uv_meters,
        instance=footprint,
        spacing=pf.Vector((spacing_x, spacing_y, 0)),
        margin_low=pf.Vector((side_margin * margin_split, margin_bottom, 0)),
        margin_high=pf.Vector((side_margin * (1 - margin_split), margin_top, 0)),
        y_instances_max=pf.random.randint(rng, 4, 12),
    )
    instances = grid_placement.place_instances_on_uv_grid(
        surface=wall,
        uv_field=uv_meters,
        grid_mesh=grid_res.grid_mesh,
        query_uv=grid_res.query_uv,
        instance=slab,
    )
    shelf_aliases = pf.nodes.to_aliases(instances)
    if not shelf_aliases:
        logger.warning(
            "wall_board_shelf: no shelves fit on %.2fx%.2fm wall (width %.2f, "
            "spacing_y %.2f, margins t/b %.2f/%.2f)",
            wall_width,
            wall_height,
            shelf_width,
            spacing_y,
            margin_top,
            margin_bottom,
        )

    wall_thick = _extrude_for_thickness(wall, wall_thickness)
    wall_thick.item().name = "room_wall_back"
    pf.ops.object.set_material(
        wall,
        surface=wall_material.surface,
        displacement=wall_material.displacement,
    )
    _subdivide_wall_plane(wall)
    wall = _plane_to_posed_canonical_mesh(wall)

    return WallResult(
        all_objects=[wall, wall_thick, *shelf_aliases],
        wall_planes=[wall],
        backs=[wall_thick],
        sills=[],
        storage=shelf_aliases,
        lights=[],
        decorations={"wall_board_shelf": shelf_aliases},
    )


@pf.tracer.grammar
def wall_storage_flush_rand(
    rng: pf.RNG,
    wall: pf.MeshObject | None = None,
    wall_material: pf.Material | None = None,
    wall_thickness: float = 0.05,
) -> WallResult:
    rng, wall, wall_material = _resolve_wall_inputs(rng, wall, wall_material)
    wall_width, wall_height = _wall_uv_dimensions(wall)

    depth = pf.random.uniform(rng, 0.3, 0.61)
    min_margin = depth * 0.5

    def _short_band() -> float:
        return wall_height * pf.random.uniform(rng, 0.20, 0.50)

    def _tall_band() -> float:
        return wall_height * pf.random.uniform(rng, 0.60, 0.98)

    height = pf.control.choice(rng, [(_short_band, 1.0), (_tall_band, 1.0)])()

    width = _wall_storage_width_rand(rng, wall_width, min_margin)
    spacing_x = pf.random.uniform(rng, 0.1, 0.5)

    margin_split = pf.random.uniform(rng, 0.375, 0.625)
    margin_low_x, margin_high_x, _ = _fit_grid_margins(
        wall_width, width, spacing_x, min_margin, margin_split
    )

    cab = storage.storage_cell_shelf_rand(
        rng,
        dimensions=pf.Vector((depth, width, height)),
    ).mesh
    cab = _seat_upright_cabinet(cab, width, height, back_depth=0.0)
    footprint = upright_cabinet_footprint(width, height)

    uv_meters = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute
    grid_res = grid_placement.grid_from_spacing(
        uv_surface=wall,
        target_uv=uv_meters,
        instance=footprint,
        spacing=pf.Vector((spacing_x, 0, 0)),
        margin_low=pf.Vector((margin_low_x, 0.0, 0)),
        margin_high=pf.Vector((margin_high_x, 0.0, 0)),
        y_instances_max=1,
    )
    instances = grid_placement.place_instances_on_uv_grid(
        surface=wall,
        uv_field=uv_meters,
        grid_mesh=grid_res.grid_mesh,
        query_uv=grid_res.query_uv,
        instance=cab,
        normal_offset=pf.random.uniform(rng, 0.02, 0.05),
    )
    aliases = pf.nodes.to_aliases(instances)
    propagate_modifiers_to_instances([cab], aliases)
    if not aliases:
        logger.warning(
            "wall_storage_flush: no cabinets fit on %.2fm-wide wall "
            "(cabinet width %.2f, spacing_x %.2f, margins %.2f/%.2f)",
            wall_width,
            width,
            spacing_x,
            margin_low_x,
            margin_high_x,
        )

    wall, wall_thick = _plain_wall(wall, wall_material, wall_thickness)
    return WallResult(
        all_objects=[wall, wall_thick, *aliases],
        wall_planes=[wall],
        backs=[wall_thick],
        sills=[],
        storage=aliases,
        lights=[],
        decorations={"wall_storage": aliases},
    )
