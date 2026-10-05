# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import NamedTuple

import numpy as np
import procfunc as pf

from infinigen2.objects import lamp, window
from infinigen2.objects.ceiling_light import ceiling_light_rand
from infinigen2.scenes.placement.distribute import (
    duplicates,
    propagate_modifiers_to_instances,
)
from infinigen2.scenes.room.room_shape import RoomShapeResult
from infinigen2.scenes.room.wall_base import (
    ROOM_SUBSURF_LEVELS,
    extrude_for_thickness,
    fit_grid_margins,
    name_objects,
)
from infinigen2.scenes.room.wall_cutouts import (
    arrange_window_portals,
    cutout_spaced_instances,
)
from infinigen2.shaders.functionality_lists import (
    ceiling_material_rand,
    floor_material_rand,
)
from infinigen2.util import mesh as mesh_util
from infinigen2.uv_surface import grid_placement

logger = logging.getLogger(__name__)

__all__ = [
    "CeilingFeaturesResult",
    "ceiling_feature_rand",
    "ceiling_light_bars_rand",
    "ceiling_light_placement_rand",
    "ceiling_skylights_rand",
]


class CeilingFeaturesResult(NamedTuple):
    floor: pf.MeshObject
    ceiling: pf.MeshObject
    backs: list[pf.MeshObject]  # ceiling-back lightblocker meshes
    sills: list[pf.MeshObject]  # skylight/bar reveal meshes
    light_meshes: list[pf.MeshObject]  # lamp/skylight/bar housing meshes
    lights: list[pf.LightObject]


@pf.tracer.grammar
def ceiling_light_placement_rand(
    rng: pf.RNG,
    ceiling: pf.MeshObject,
    dimensions: pf.Vector,
) -> tuple[list[pf.MeshObject], list[pf.LightObject]]:
    lumens = dimensions.x * dimensions.y * pf.random.uniform(rng, 300, 700)
    total_energy = lumens / 177

    spacing_x = pf.random.uniform(rng, 1.5, 2.5)
    spacing_y = pf.random.uniform(rng, 1.5, 2.5)
    margin_x = min(pf.random.uniform(rng, 0.4, 1.5), 0.3 * dimensions.x)
    margin_y = min(pf.random.uniform(rng, 0.4, 1.5), 0.3 * dimensions.y)

    template_fn = pf.control.choice(
        rng,
        [
            (ceiling_light_rand, 2.5),
            (lamp.ceiling_shade_lamp_rand, 1.0),
        ],
    )
    lamp_template = template_fn(rng, energy=total_energy)
    lamp_template.mesh.item().name = template_fn.__name__
    mesh_template = lamp_template.mesh

    # light offset relative to template origin
    light_offset = None
    if lamp_template.light is not None:
        light_offset = np.array(
            lamp_template.light.item().location - mesh_template.item().location
        )

    # bake the template's authored orientation, then centre it in the ceiling plane
    pf.ops.mesh.transform_apply(mesh_template)
    bmin, bmax = pf.ops.attr.bbox_min_max(mesh_template)
    center = (np.array(bmin) + np.array(bmax)) / 2
    pf.ops.object.set_transform(mesh_template, location=(-center[0], -center[1], 0.0))
    pf.ops.mesh.transform_apply(mesh_template)

    # local +X -> ceiling normal (down); this offset re-hangs the Z-up lamp so it keeps
    # its authored orientation (net-identity, so light_offset stays valid)
    lamp_hang = (np.pi / 2, 0.0, -np.pi / 2)

    lamp_w = np.array(bmax) - np.array(bmin)
    gap_x = max(0.1, spacing_x - lamp_w[0])
    gap_y = max(0.1, spacing_y - lamp_w[1])

    # 2d grid placement, no cutting
    pf.ops.uv.cube_project(ceiling, uv_name="UVMap")
    ceiling_uvs = pf.ops.attr.uv_coords(ceiling)
    extent_x = ceiling_uvs[:, 0].max() - ceiling_uvs[:, 0].min()
    extent_y = ceiling_uvs[:, 1].max() - ceiling_uvs[:, 1].min()
    margin_x_low, margin_x_high, n_x = fit_grid_margins(
        extent_x, lamp_w[0], gap_x, margin_x, 0.5
    )
    margin_y_low, margin_y_high, n_y = fit_grid_margins(
        extent_y, lamp_w[1], gap_y, margin_y, 0.5
    )
    uv_meters = pf.nodes.geo.input_named_attribute(
        name="UVMap", data_type=pf.NodeDataType.FLOAT_VECTOR
    ).attribute
    grid_res = grid_placement.grid_from_spacing(
        uv_surface=ceiling,
        target_uv=uv_meters,
        instance=mesh_template,
        spacing=pf.Vector((gap_x, gap_y, 0)),
        margin_low=pf.Vector((margin_x_low, margin_y_low, 0)),
        margin_high=pf.Vector((margin_x_high, margin_y_high, 0)),
        x_instances_max=n_x,
        y_instances_max=n_y,
        rotation_offset=lamp_hang,
    )
    instances = grid_placement.place_instances_on_uv_grid(
        surface=ceiling,
        uv_field=uv_meters,
        grid_mesh=grid_res.grid_mesh,
        query_uv=grid_res.query_uv,
        instance=mesh_template,
        secondary_axis_vector=(0, 1, 0),
        rotation_offset=lamp_hang,
    )
    meshes = pf.nodes.to_aliases(instances)
    propagate_modifiers_to_instances([mesh_template], meshes)
    logger.info(
        "Placed %d ceiling lamps in %.1fx%.1fm room",
        len(meshes),
        dimensions.x,
        dimensions.y,
    )
    if not meshes:
        logger.warning("Ceiling lamp grid produced no lamps")

    if lamp_template.light is None or not meshes or light_offset is None:
        return meshes, []

    light_locations = np.array([m.item().location for m in meshes]) + light_offset
    lights = duplicates(lamp_template.light, light_locations)
    # rescale to the actual placed count to hit total_energy
    for light in lights:
        light.item().data.energy = total_energy / len(lights)
    return meshes, lights


@pf.tracer.grammar
def ceiling_skylights_rand(
    rng: pf.RNG,
    ceiling: pf.MeshObject,
    ceiling_material: pf.Material,
) -> tuple[
    pf.MeshObject,
    list[pf.MeshObject],
    list[pf.MeshObject],
    list[pf.MeshObject],
    list[pf.LightObject],
]:
    # metric planar UVs for the cutout grid
    pf.ops.uv.cube_project(ceiling, uv_name="UVMap")

    ceiling_uvs = pf.ops.attr.uv_coords(ceiling)
    extent_x = ceiling_uvs[:, 0].max() - ceiling_uvs[:, 0].min()
    extent_y = ceiling_uvs[:, 1].max() - ceiling_uvs[:, 1].min()

    # each side must fit the smaller extent (either orientation), plus headroom
    floor_margin = 0.1
    max_side = max(0.45, min(extent_x, extent_y) - 2 * floor_margin - 0.1)

    # short side + aspect, then random long axis
    skylight_short = min(pf.random.uniform(rng, 0.45, 1.1), max_side)
    skylight_aspect = pf.random.uniform(rng, 1.0, 10.0)
    skylight_long = min(skylight_short * skylight_aspect, 3.5, max_side)
    skylight_width, skylight_length = pf.control.choice(
        rng,
        [
            ((skylight_short, skylight_long), 1.0),
            ((skylight_long, skylight_short), 1.0),
        ],
    )
    window_dimensions = window.window_dimensions_rand(
        rng, width=skylight_width, height=skylight_length
    )
    # skylights never get curtains
    window_result = window.window_from_profile_rand(
        rng, window.rectangular_profile(window_dimensions)
    )
    win = window_result.mesh

    # centre at origin; unified placement lays it into the ceiling (local +X
    # follows the down-facing normal). footprint on the ceiling is Y x Z.
    bmin, bmax = pf.ops.attr.bbox_min_max(win)
    center = (np.array(bmin) + np.array(bmax)) / 2
    pf.ops.object.set_transform(win, location=-center)
    pf.ops.mesh.transform_apply(win)
    if window_result.light is not None:
        pf.ops.object.set_transform(
            window_result.light,
            location=np.array(window_result.light.item().location) - center,
        )
    bmin, bmax = pf.ops.attr.bbox_min_max(win)
    win_dims = (np.array(bmax) - np.array(bmin))[[1, 2]]

    spacing = pf.random.uniform(rng, 0.4, 1.8)
    margin_frac = pf.random.uniform(rng, 0.0, 1.0)

    # fit per-axis counts; min_margin within [floor, (extent-win)/2]
    extents = (extent_x, extent_y)
    margins = []
    for axis, n_cap in ((0, 4), (1, 3)):
        extent = extents[axis]
        max_margin = max(floor_margin, (extent - win_dims[axis]) / 2)
        min_margin = floor_margin + (max_margin - floor_margin) * margin_frac
        split = pf.random.uniform(rng, 0.4, 0.6)
        low, high, _ = fit_grid_margins(
            extent, win_dims[axis], spacing, min_margin, split, n_cap
        )
        margins.append((low, high))

    reveal_depth = pf.random.uniform(rng, 0.1, 1.0)
    recess_pct = pf.random.uniform(rng, 0.0, 1.0)

    geom, sill, lightblocker, skylight_aliases, _trim_edges = cutout_spaced_instances(
        surface=ceiling,
        instance=win,
        surface_material=ceiling_material,
        spacing=pf.Vector((spacing, spacing, 0)),
        margin_low=pf.Vector((margins[0][0], margins[1][0], 0)),
        margin_high=pf.Vector((margins[0][1], margins[1][1], 0)),
        x_instances_max=4,
        y_instances_max=3,
        canonical_up_axis="Y",
        instance_secondary_axis=(0, 1, 0),
        wall_thickness=reveal_depth,
        recess_pct=recess_pct,
        chamfer=pf.random.clip_gaussian(rng, 0.006, 0.002, 0.004, 0.010),
    )

    portals: list[pf.LightObject] = []
    if window_result.light is not None and skylight_aliases:
        portals = arrange_window_portals(skylight_aliases, win, window_result.light)

    backs = [lightblocker] if lightblocker is not None else []
    sills = [sill] if sill is not None else []
    return geom, backs, sills, skylight_aliases, portals


@pf.tracer.grammar
def ceiling_light_bars_rand(
    rng: pf.RNG,
    ceiling: pf.MeshObject,
    ceiling_material: pf.Material,
    dimensions: pf.Vector,
) -> tuple[
    pf.MeshObject,
    list[pf.MeshObject],
    list[pf.MeshObject],
    list[pf.MeshObject],
    list[pf.LightObject],
]:
    # metric planar UVs for the cutout grid
    pf.ops.uv.cube_project(ceiling, uv_name="UVMap")

    ceiling_uvs = pf.ops.attr.uv_coords(ceiling)
    extent_x = ceiling_uvs[:, 0].max() - ceiling_uvs[:, 0].min()
    extent_y = ceiling_uvs[:, 1].max() - ceiling_uvs[:, 1].min()

    # bars run long along x, thin along y, rows stacked across y
    bar_length = min(pf.random.uniform(rng, 1.5, 4.0), extent_x - 0.4)
    bar_width = pf.random.uniform(rng, 0.02, 1.0)
    bar_depth = pf.random.uniform(rng, 0.03, 0.06)

    # thin box housing, x/y-centered (symmetric -> flip-invariant)
    cube = mesh_util.box(size=(bar_length, bar_width, bar_depth))
    housing = pf.nodes.to_mesh_object(cube)
    pf.ops.uv.cube_project(housing, uv_name="UVMap")
    pf.ops.object.set_material(
        housing,
        surface=ceiling_material.surface,
        displacement=ceiling_material.displacement,
    )

    spacing = pf.random.uniform(rng, 0.4, 2.4)
    min_margin_x = pf.random.uniform(rng, 0.1, 0.3)
    min_margin_y = pf.random.uniform(rng, 0.3, 1.2)

    # rows across y, single centered span along x
    split_y = pf.random.uniform(rng, 0.4, 0.6)
    margin_y_low, margin_y_high, _ = fit_grid_margins(
        extent_y, bar_width, spacing, min_margin_y, split_y, n_cap=4
    )
    split_x = pf.random.uniform(rng, 0.4, 0.6)
    margin_x_low, margin_x_high, _ = fit_grid_margins(
        extent_x, bar_length, spacing, min_margin_x, split_x, n_cap=1
    )

    reveal_depth = pf.random.uniform(rng, 0.1, 1.0)
    recess_pct = pf.random.uniform(rng, 0.0, 1.0)

    geom, sill, lightblocker, bar_aliases, _trim_edges = cutout_spaced_instances(
        surface=ceiling,
        instance=housing,
        surface_material=ceiling_material,
        spacing=pf.Vector((spacing, spacing, 0)),
        margin_low=pf.Vector((margin_x_low, margin_y_low, 0)),
        margin_high=pf.Vector((margin_x_high, margin_y_high, 0)),
        x_instances_max=1,
        y_instances_max=4,
        canonical_up_axis="Y",
        instance_secondary_axis=(0, 1, 0),
        wall_thickness=reveal_depth,
        recess_pct=recess_pct,
        chamfer=pf.random.clip_gaussian(rng, 0.006, 0.002, 0.004, 0.010),
        rotation_offset=(np.pi / 2, 0.0, np.pi / 2),
    )

    bar_ceiling_locs = [np.array(alias.item().location) for alias in bar_aliases]

    lumens = dimensions.x * dimensions.y * pf.random.uniform(rng, 300, 700)
    per_energy = lumens / 177 / max(1, len(bar_aliases))

    # shared blackbody temperature, indoor range
    temperature = pf.random.clip_gaussian(rng, 4500, 1000, 2000, 8000)

    # one area lamp per bar, just below the ceiling, facing down
    lights: list[pf.LightObject] = []
    for ceiling_loc in bar_ceiling_locs:
        lamp_energy = per_energy * pf.random.uniform(rng, 0.75, 1.25)
        light = pf.ops.primitives.light.area_lamp(
            shape="RECTANGLE",
            size_x=bar_length,
            size_y=bar_width,
            energy=lamp_energy,
        )
        blackbody = pf.nodes.color.blackbody(temperature=temperature)
        emission = pf.nodes.shader.emission(color=blackbody, strength=1.0)
        pf.nodes.to_light(light, surface=emission)
        light.item().location = (
            ceiling_loc[0],
            ceiling_loc[1],
            ceiling_loc[2] - 0.03,
        )
        lights.append(light)

    backs = [lightblocker] if lightblocker is not None else []
    sills = [sill] if sill is not None else []
    return geom, backs, sills, bar_aliases, lights


@pf.tracer.grammar
def ceiling_feature_rand(
    rng: pf.RNG,
    shape: RoomShapeResult,
    wall_thickness: float = 0.1,
) -> CeilingFeaturesResult:
    vec_pos = pf.nodes.shader.geometry().position

    rng_floor_mat, rng_ceiling_mat, rng_choice, rng_feature = rng.spawn(4)

    floor_mat = floor_material_rand(rng_floor_mat, vec_pos)
    pf.ops.object.set_material(
        shape.floor,
        surface=floor_mat.surface,
        displacement=floor_mat.displacement,
    )
    pf.ops.modifier.subdivide_surface(
        shape.floor, levels=ROOM_SUBSURF_LEVELS, _skip_apply=True
    )

    ceiling_mat = ceiling_material_rand(rng_ceiling_mat, vec_pos)
    pf.ops.object.set_material(
        shape.ceiling,
        surface=ceiling_mat.surface,
        displacement=ceiling_mat.displacement,
    )
    pf.ops.modifier.subdivide_surface(
        shape.ceiling, levels=ROOM_SUBSURF_LEVELS, _skip_apply=True
    )

    def ceiling_plain_with_lights(rng: pf.RNG):
        ceiling_back = extrude_for_thickness(shape.ceiling, wall_thickness)
        ceiling_back.item().name = "room_wall_back"
        lamp_meshes, lamp_lights = ceiling_light_placement_rand(
            rng, shape.ceiling, dimensions=shape.dimensions
        )
        return shape.ceiling, [ceiling_back], [], lamp_meshes, lamp_lights

    def ceiling_skylights(rng: pf.RNG):
        return ceiling_skylights_rand(rng, shape.ceiling, ceiling_mat)

    def ceiling_light_bars(rng: pf.RNG):
        return ceiling_light_bars_rand(
            rng,
            shape.ceiling,
            ceiling_mat,
            dimensions=shape.dimensions,
        )

    option = pf.control.choice(
        rng_choice,
        [
            (ceiling_plain_with_lights, 4.0),
            (ceiling_skylights, 1.0),
            (ceiling_light_bars, 1.0),
        ],
    )
    ceiling_geom, backs, sills, light_meshes, ceiling_lights = option(rng_feature)

    name_objects([shape.floor], "room_floor")
    name_objects([ceiling_geom], "room_ceiling")
    name_objects(backs, "room_ceiling_back")
    name_objects(sills, "room_ceiling_sill")
    name_objects(light_meshes, "ceiling_light")
    return CeilingFeaturesResult(
        floor=shape.floor,
        ceiling=ceiling_geom,
        backs=backs,
        sills=sills,
        light_meshes=light_meshes,
        lights=ceiling_lights,
    )
