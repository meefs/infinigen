# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/tables/dining_table.py)
# - Alexander Raistrick: transpile to procfunc/v2

import math
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import storage
from infinigen2.shaders.functionality_lists import (
    furniture_material_rand,
    table_top_material_rand,
)
from infinigen2.util import curve, mesh

__all__ = [
    "TableResult",
    "base_square",
    "base_square_rand",
    "base_straight",
    "base_straight_rand",
    "cocktail_table_rand",
    "coffee_table_storage_rand",
    "coffee_table_dimensions_rand",
    "coffee_table_rand",
    "dining_table_rand",
    "pedestal_base",
    "pedestal_base_rand",
    "side_table_dimensions_rand",
    "side_table_rand",
    "table_dimensions_rand",
    "table_top",
]


class TableResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _n_gon_profile(
    profile_n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    profile_aspect_ratio: t.SocketOrVal[float],
    profile_fillet_ratio: t.SocketOrVal[float] = 0.0,
) -> t.ProcNode[pf.CurveObject]:
    curve_circle_radius = pf.nodes.math.constant(0.5)
    curve_circle = pf.nodes.geo.curve_circle(
        resolution=profile_n_gon, radius=curve_circle_radius
    )

    transform_rotation = pf.nodes.math.combine_xyz(
        z=3.1416 / profile_n_gon.astype(dtype=float)
    )
    transform = pf.nodes.geo.transform(
        geometry=curve_circle,
        rotation=transform_rotation.astype(dtype=pf.Euler),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=transform,
        rotation=(0.0, 0.0, -1.5708),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_2_scale = pf.nodes.math.combine_xyz(
        x=profile_width,
        y=profile_aspect_ratio * profile_width,
        z=1.0,
    )
    transform_2 = pf.nodes.geo.transform(
        geometry=transform_1,
        scale=transform_2_scale,
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    fillet_curve = pf.nodes.geo.fillet_curve_poly(
        curve=transform_2,
        radius=profile_width * profile_fillet_ratio,
        limit_radius=True,
        count=4,
    )
    return fillet_curve


@pf.nodes.node_function
def _merge_curve(
    curve: t.ProcNode[pf.CurveObject],
) -> t.ProcNode[pf.CurveObject]:
    curve_to = pf.nodes.geo.curve_to_mesh(curve)

    merge_by_distance = pf.nodes.geo.merge_by_distance(curve_to)

    to_curve = pf.nodes.geo.mesh_to_curve(merge_by_distance)
    return to_curve


class _NGonCylinderResult(NamedTuple):
    mesh: t.ProcNode[pf.MeshObject]
    profile_curve: t.ProcNode[pf.CurveObject]
    caps: t.ProcNode[pf.MeshObject]


@pf.nodes.node_function
def _n_gon_cylinder(
    radius_curve: t.ProcNode[pf.CurveObject],
    n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    height: t.SocketOrVal[float] = 1.0,
    aspect_ratio: t.SocketOrVal[float] = 1.0,
    fillet_ratio: t.SocketOrVal[float] = 0.2,
    profile_resolution: t.SocketOrVal[int] = 16,
    resolution: t.SocketOrVal[int] = 64,
) -> _NGonCylinderResult:
    mesh_position_z_to_min = height * -1.0

    curve_line_end = pf.nodes.math.combine_xyz(z=mesh_position_z_to_min)
    curve_line = pf.nodes.geo.curve_line(end=curve_line_end, start=(0, 0, 0))

    set_curve_tilt = pf.nodes.geo.set_curve_tilt(curve=curve_line, tilt=3.1416)

    resample_curve_count_1 = pf.nodes.geo.resample_curve_count(
        curve=set_curve_tilt, count=resolution
    )

    spline_parameter = pf.nodes.geo.spline_parameter()

    capture_attribute = pf.nodes.geo.capture_attribute(
        geometry=resample_curve_count_1,
        attribute=spline_parameter.factor,
    )

    n_gon_profile_result = _n_gon_profile(
        profile_n_gon=n_gon,
        profile_width=profile_width,
        profile_aspect_ratio=aspect_ratio,
        profile_fillet_ratio=fillet_ratio,
    )

    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=n_gon_profile_result,
        count=profile_resolution,
    )

    curve_to = curve.curve_to_mesh_with_uv(
        curve=capture_attribute.geometry,
        profile=resample_curve_count,
        fill_caps=True,
    ).mesh

    input_position = pf.nodes.geo.input_position()

    sample_curve = pf.nodes.geo.sample_curve(
        curves=radius_curve,
        factor=capture_attribute.attribute,
        value=0.0,
        use_all_curves=True,
    )

    mesh_position_x_vector = pf.nodes.math.combine_xyz(
        x=sample_curve.position.x, y=sample_curve.position.y
    )
    mesh_position_x = pf.nodes.math.vector_length(mesh_position_x_vector)

    input_position_1 = pf.nodes.geo.input_position()

    attribute_statistic = pf.nodes.geo.attribute_statistic(
        geometry=radius_curve,
        attribute=input_position_1.z,
    )

    mesh_position_z = pf.nodes.math.map_range(
        value=sample_curve.position.z,
        from_max=attribute_statistic.max,
        from_min=attribute_statistic.min,
        to_max=0.0,
        to_min=mesh_position_z_to_min,
    )
    mesh_position = pf.nodes.math.combine_xyz(
        x=input_position.x * mesh_position_x,
        y=input_position.y * mesh_position_x,
        z=mesh_position_z,
    )

    set_position = pf.nodes.geo.set_position(geometry=curve_to, position=mesh_position)

    input_index = pf.nodes.geo.input_index()

    attribute_domain_size = pf.nodes.geo.attribute_domain_size(curve_to)

    caps_selection_b = attribute_domain_size.face_count.astype(dtype=float) - 2.0
    caps_selection = pf.nodes.func.less_than(
        a=input_index, b=caps_selection_b.astype(dtype=int)
    )

    delete = pf.nodes.geo.delete_geometry(
        geometry=curve_to,
        selection=caps_selection,
        domain="FACE",
    )
    return _NGonCylinderResult(
        mesh=set_position,
        profile_curve=resample_curve_count,
        caps=delete,
    )


@pf.nodes.node_function
def _strecher(
    n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    curve_line = pf.nodes.geo.curve_line(start=(1.0, 0.0, 1.0), end=(1.0, 0.0, -1.0))

    n_gon_cylinder_result = _n_gon_cylinder(
        radius_curve=curve_line,
        height=1.0,
        n_gon=n_gon,
        profile_width=profile_width,
        aspect_ratio=1.0,
        fillet_ratio=0.2,
        profile_resolution=8,
        resolution=2,
    )
    return n_gon_cylinder_result.mesh


@pf.nodes.node_function
def _create_anchors(
    profile_n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    profile_aspect_ratio: t.SocketOrVal[float],
    profile_rotation: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    set_switch = pf.nodes.func.equal(a=profile_n_gon, b=1)
    set_a_switch = pf.nodes.func.equal(a=profile_n_gon, b=2)

    n_gon_profile_result = _n_gon_profile(
        profile_n_gon=profile_n_gon,
        profile_width=profile_width,
        profile_aspect_ratio=profile_aspect_ratio,
        profile_fillet_ratio=0.0,
    )

    curve_to_points = pf.nodes.geo.curve_to_points_evaluated(curve=n_gon_profile_result)
    curve_line_start = pf.nodes.math.combine_xyz(profile_width * 0.3535)
    curve_line_end = pf.nodes.math.combine_xyz(profile_width * -0.3535)
    curve_line = pf.nodes.geo.curve_line(start=curve_line_start, end=curve_line_end)
    curve_to_points_1 = pf.nodes.geo.curve_to_points_evaluated(curve=curve_line)

    set_a = pf.nodes.func.switch(
        switch=set_a_switch,
        a=curve_to_points.points,
        b=curve_to_points_1.points,
    )

    points = pf.nodes.geo.points(position=(0, 0, 0))

    set_point_radius_points = pf.nodes.func.switch(switch=set_switch, a=set_a, b=points)
    set_point_radius = pf.nodes.geo.set_point_radius(set_point_radius_points)

    rotation = pf.nodes.math.combine_xyz(z=profile_rotation)

    transform = pf.nodes.geo.transform(
        geometry=set_point_radius,
        rotation=rotation.astype(dtype=pf.Euler),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    return transform


@pf.nodes.node_function
def _create_legs_and_strechers(
    anchors: t.ProcNode[pf.MeshObject],
    keep_legs: t.SocketOrVal[bool],
    leg_instance: t.ProcNode[pf.MeshObject],
    table_height: t.SocketOrVal[float],
    leg_bottom_relative_scale: t.SocketOrVal[float],
    leg_bottom_relative_rotation: t.SocketOrVal[float],
    keep_odd_strechers: t.SocketOrVal[bool],
    keep_even_strechers: t.SocketOrVal[bool],
    strecher_instance: t.ProcNode[pf.MeshObject],
    strecher_index_increment: t.SocketOrVal[int],
    strecher_relative_position: t.SocketOrVal[float],
    leg_bottom_offset: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    transform_translation = pf.nodes.math.combine_xyz(z=table_height)
    transform = pf.nodes.geo.transform(
        geometry=anchors,
        translation=transform_translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    input_position = pf.nodes.geo.input_position()

    set_b_vector_b = pf.nodes.math.combine_xyz(z=leg_bottom_offset)
    set_b_vector = transform_translation - set_b_vector_b
    set_b_rotation = pf.nodes.math.combine_xyz(0, 0, leg_bottom_relative_rotation)
    set_b_1 = pf.nodes.math.vector_rotate_euler(
        vector=input_position - set_b_vector,
        rotation=set_b_rotation,
        center=(0, 0, 0),
    )
    splay_height_factor = pf.nodes.math.minimum(table_height / 0.4, 1.0)
    splay_scale = 1.0 + (leg_bottom_relative_scale - 1.0) * splay_height_factor
    set_b_0 = pf.nodes.math.combine_xyz(
        x=splay_scale,
        y=splay_scale,
        z=1.0,
    )
    set_position_position_vector = input_position - (set_b_1 * set_b_0)
    set_position_position = pf.nodes.math.vector_scale(
        vector=set_position_position_vector,
        scale=strecher_relative_position * -1.0,
    )

    input_position_1 = pf.nodes.geo.input_position()

    set_position = pf.nodes.geo.set_position(
        geometry=transform,
        position=set_position_position + input_position_1,
    )

    input_index = pf.nodes.geo.input_index()

    instance_2 = input_index.astype(dtype=float) % 2.0
    instance_a_a = pf.nodes.func.boolean_and(
        a=instance_2.astype(dtype=bool), b=keep_odd_strechers
    )
    instance_a_b_b = pf.nodes.func.boolean_not(instance_2.astype(dtype=bool))
    instance_a_b = pf.nodes.func.boolean_and(a=keep_even_strechers, b=instance_a_b_b)
    instance_a = pf.nodes.func.boolean_or(a=instance_a_a, b=instance_a_b)

    attribute_domain_size = pf.nodes.geo.attribute_domain_size(
        geometry=transform, component="POINTCLOUD"
    )

    instance_b_switch = pf.nodes.func.equal(
        a=attribute_domain_size.point_count.astype(dtype=float)
        / strecher_index_increment.astype(dtype=float),
        b=2.0,
        epsilon=0.001,
    )
    input_index_1 = pf.nodes.geo.input_index()

    instance_1 = attribute_domain_size.point_count.astype(dtype=float) / 2.0
    instance_b_b = pf.nodes.func.less_than(
        a=input_index_1, b=instance_1.astype(dtype=int)
    )
    instance_b = pf.nodes.func.switch(switch=instance_b_switch, a=True, b=instance_b_b)
    instance_on_points_selection = pf.nodes.func.boolean_and(a=instance_a, b=instance_b)

    input_position_2 = pf.nodes.geo.input_position()

    field = (
        input_index.astype(dtype=float) + strecher_index_increment.astype(dtype=float)
    ) % attribute_domain_size.point_count.astype(dtype=float)
    field_at_index = pf.nodes.geo.field_at_index(
        value=input_position_2, index=field.astype(dtype=int)
    )

    instance_z_vector = input_position_2 - field_at_index
    instance_0_rotation = pf.nodes.func.align_euler_to_vector(
        vector=instance_z_vector,
        axis="Z",
        rotation=(0, 0, 0),
        factor=1.0,
    )
    instance = pf.nodes.func.align_euler_to_vector(
        rotation=instance_0_rotation,
        pivot_axis="Z",
        factor=1.0,
        vector=(0, 0, 1),
    )
    instance_z = pf.nodes.math.vector_length(instance_z_vector)
    instance_on_points_scale = pf.nodes.math.combine_xyz(x=1.0, y=1.0, z=instance_z)
    # increment 0 aims every anchor at itself; a zero-length stretcher is a flat shell
    spans_a_gap = pf.nodes.func.greater_than(a=instance_z, b=1e-6)
    instance_on_points = pf.nodes.geo.instance_on_points(
        points=set_position,
        instance=strecher_instance,
        selection=pf.nodes.func.boolean_and(
            a=instance_on_points_selection, b=spans_a_gap
        ),
        rotation=instance.astype(dtype=pf.Euler),
        scale=instance_on_points_scale,
    )

    realize_instances = pf.nodes.geo.realize_instances(instance_on_points)

    leg_rotation = pf.nodes.func.align_euler_to_vector(
        vector=set_position_position_vector,
        axis="Z",
        rotation=(0, 0, 0),
        factor=1.0,
    )
    instance_scale_z = pf.nodes.math.vector_length(set_position_position_vector)
    instance_scale = pf.nodes.math.combine_xyz(x=1.0, y=1.0, z=instance_scale_z)
    instance_on_points_1 = pf.nodes.geo.instance_on_points(
        points=transform,
        instance=leg_instance,
        rotation=leg_rotation.astype(dtype=pf.Euler),
        scale=instance_scale,
    )

    realize_instances_1 = pf.nodes.geo.realize_instances(instance_on_points_1)

    geometries = pf.nodes.func.switch(switch=keep_legs, b=realize_instances_1)

    join = pf.nodes.geo.join_geometry([realize_instances, geometries])
    return join


@pf.nodes.node_function
def _pedestal_profile(
    height: t.SocketOrVal[float],
    top_radius: t.SocketOrVal[float],
    bottom_radius: t.SocketOrVal[float],
    flare: t.SocketOrVal[float],
    concavity: t.SocketOrVal[float],
    neck_scale: t.SocketOrVal[float] = 1.0,
    resolution: t.SocketOrVal[int] = 12,
) -> t.ProcNode[pf.CurveObject]:
    """Closed silhouette curve (x=radius, y=z) from the axis at the top, out to
    (top_radius, height), down to (bottom_radius, 0), and back to the axis, so
    the lathed surface is capped. flare (0-1) sets how far down the column stays
    skinny before flaring out; concavity (0-1) bends the flare from a convex
    bulge to a sharp concave sweep; neck_scale bulges (>1) or waists (<1) the
    column."""
    knee_z = height * (1.0 - flare)
    end_handle_x = bottom_radius + concavity * (top_radius - bottom_radius)
    end_handle_z = knee_z * (1.0 - concavity)
    body = pf.nodes.geo.curve_bezier_segment(
        start=pf.nodes.math.combine_xyz(x=top_radius, y=height),
        start_handle=pf.nodes.math.combine_xyz(x=top_radius * neck_scale, y=knee_z),
        end_handle=pf.nodes.math.combine_xyz(x=end_handle_x, y=end_handle_z),
        end=pf.nodes.math.combine_xyz(x=bottom_radius),
        resolution=resolution,
    )
    top_cap = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(y=height),
        end=pf.nodes.math.combine_xyz(x=top_radius, y=height),
    )
    bottom_cap = pf.nodes.geo.curve_line(
        start=pf.nodes.math.combine_xyz(x=bottom_radius), end=(0.0, 0.0, 0.0)
    )
    return pf.nodes.geo.join_geometry([top_cap, body, bottom_cap])


@pf.nodes.node_function
def _pedestal_sweep(
    radius_curve: pf.ProcNode[pf.CurveObject],
    resolution: t.SocketOrVal[int] = 16,
) -> t.ProcNode[pf.MeshObject]:
    """Lathe a silhouette curve (x=radius, y=z) around the z axis by sweeping it
    around a circle. Profile x offsets add to the unit circle's radius, so shift
    x by -1; mirroring y maps profile y to +Z and keeps faces outward."""
    # subdivision pulls an n-gon ring in to (2 + cos(2pi/n))/3 of its radius
    ring_angle = math.tau / resolution.astype(dtype=float)
    ring_shrink = (2.0 + pf.nodes.math.cos(ring_angle)) / 3.0
    profile_scale = pf.nodes.math.combine_xyz(x=1.0 / ring_shrink, y=-1.0, z=1.0)
    profile = pf.nodes.geo.transform(
        geometry=radius_curve, translation=(-1.0, 0.0, 0.0), scale=profile_scale
    )
    circle = pf.nodes.geo.curve_circle(resolution=resolution, radius=1.0)
    swept = curve.curve_to_mesh_with_uv(curve=circle, profile=profile).mesh
    return pf.nodes.geo.merge_by_distance(swept, distance=1e-5)


@pf.nodes.node_function
def _leg_square(
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    fillet_radius: t.SocketOrVal[float],
    has_bottom_connector: t.SocketOrVal[bool],
    profile_n_gon: t.SocketOrVal[int],
    profile_width: t.SocketOrVal[float],
    profile_aspect_ratio: t.SocketOrVal[float],
    profile_fillet_ratio: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    curve_circle = pf.nodes.geo.curve_circle(resolution=4, radius=0.7071)
    curve_circle = pf.nodes.geo.set_spline_cyclic(
        curve_circle, cyclic=has_bottom_connector
    )

    merge_curve_result = _merge_curve(curve=curve_circle)

    set_curve_tilt_tilt = pf.nodes.math.map_range(
        value=has_bottom_connector.astype(dtype=float),
        to_max=3.1416,
        to_min=1.5708,
    )
    set_curve_tilt = pf.nodes.geo.set_curve_tilt(
        curve=merge_curve_result, tilt=set_curve_tilt_tilt
    )

    transform = pf.nodes.geo.transform(
        geometry=set_curve_tilt,
        rotation=(0.0, 0.0, 0.7854),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    transform_1 = pf.nodes.geo.transform(
        geometry=transform,
        translation=(0.0, 0.0, -0.5),
        rotation=(0.0, 1.5708, 0.0),
        scale=(1, 1, 1),
    )
    n_gon_profile_result = _n_gon_profile(
        profile_n_gon=profile_n_gon,
        profile_width=profile_width,
        profile_aspect_ratio=profile_aspect_ratio,
        profile_fillet_ratio=profile_fillet_ratio,
    )
    profile_mesh = pf.nodes.geo.curve_to_mesh(n_gon_profile_result)
    profile_z_radius = pf.nodes.geo.bound_box(profile_mesh).max.y
    bottom_profile_radius = pf.nodes.func.switch(
        switch=has_bottom_connector, a=0.0, b=profile_z_radius
    )
    centerline_height = height - profile_z_radius - bottom_profile_radius
    transform_2_scale = pf.nodes.math.combine_xyz(x=1.0, y=width, z=centerline_height)
    transform_2 = pf.nodes.geo.transform(
        geometry=transform_1,
        scale=transform_2_scale,
        translation=pf.nodes.math.combine_xyz(z=profile_z_radius * -1.0),
        rotation=(0, 0, 0),
    )

    set_curve_radius = pf.nodes.geo.set_curve_radius(curve=transform_2, radius=1.0)

    fillet_curve = pf.nodes.geo.fillet_curve_poly(
        curve=set_curve_radius,
        radius=fillet_radius,
        limit_radius=True,
        count=4,
    )

    curve_to = curve.curve_to_mesh_with_uv(
        curve=fillet_curve,
        profile=n_gon_profile_result,
        fill_caps=True,
    ).mesh

    transform_3 = pf.nodes.geo.transform(
        geometry=curve_to,
        rotation=(0.0, 0.0, 0.0),
        translation=(0, 0, 0),
        scale=(1, 1, 1),
    )

    set_shade_smooth = pf.nodes.geo.set_shade_smooth(
        geometry=transform_3, shade_smooth=False
    )
    return set_shade_smooth


@pf.nodes.node_function
def _leg_straight(
    leg_height: t.SocketOrVal[float],
    leg_diameter: t.SocketOrVal[float],
    resolution: t.SocketOrVal[int],
    n_gon: t.SocketOrVal[int],
    fillet_ratio: t.SocketOrVal[float],
) -> t.ProcNode[pf.CurveObject]:
    radius_curve_result = pf.nodes.geo.curve_bezier(
        resolution=resolution,
        start=(1.0, 0.0, 1.0),
        middle=(0.95, 0.0, 0.0),
        end=(0.5, 0.0, -1.0),
    )

    n_gon_cylinder_result = _n_gon_cylinder(
        radius_curve=radius_curve_result,
        height=leg_height,
        n_gon=n_gon,
        profile_width=leg_diameter,
        aspect_ratio=1.0,
        fillet_ratio=fillet_ratio,
        profile_resolution=8,
        resolution=resolution,
    )
    return n_gon_cylinder_result.mesh


@pf.nodes.node_function
def table_top(
    size: t.SocketOrVal[pf.Vector] = (1.4, 0.8, 0.05),
    support_loop_offset: t.SocketOrVal[pf.Vector] = (0.02, 0.02, 0.01),
) -> t.ProcNode[pf.MeshObject]:
    """Slab spanning z in [0, size.z]; support_loop_offset sets edge roundness.
    Subdivision is left to an unapplied modifier on the final object."""
    box = mesh.corner_box(size=size, support_loop_offset=support_loop_offset)
    smooth = pf.nodes.geo.set_shade_smooth(geometry=box.mesh, shade_smooth=True)

    translation = pf.nodes.math.combine_xyz(z=size.z * 0.5)
    return pf.nodes.geo.transform(
        geometry=smooth,
        translation=translation,
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )


@pf.nodes.node_function
def _base_straight_geometry(
    dimensions: t.SocketOrVal[pf.Vector],
    leg_diameter: t.SocketOrVal[float],
    leg_inset: t.SocketOrVal[float],
    leg_placement_bottom_scale: t.SocketOrVal[float],
    stretcher_increment: t.SocketOrVal[int],
    stretcher_relative_pos: t.SocketOrVal[float],
    leg_placement_top_scale: t.SocketOrVal[float] = 1.0,
) -> t.ProcNode[pf.MeshObject]:
    """4-leg base with optional stretchers."""
    x, y, z = dimensions.x, dimensions.y, dimensions.z
    leg_span_x = (x - leg_diameter - 2.0 * leg_inset) * leg_placement_top_scale
    leg_span_y = (y - leg_diameter - 2.0 * leg_inset) * leg_placement_top_scale
    anchors = _create_anchors(
        profile_n_gon=4,
        profile_width=1.414 * leg_span_x,
        profile_aspect_ratio=leg_span_y / leg_span_x,
        profile_rotation=0.0,
    )

    leg = _leg_straight(
        leg_height=1.0,
        leg_diameter=leg_diameter,
        resolution=8,
        n_gon=4,
        fillet_ratio=0.1,
    )

    stretcher_geo = _strecher(n_gon=4, profile_width=leg_diameter * 0.5)

    return _create_legs_and_strechers(
        anchors=anchors,
        keep_legs=True,
        leg_instance=leg,
        table_height=z,
        leg_bottom_relative_scale=leg_placement_bottom_scale,
        leg_bottom_relative_rotation=0.0,
        keep_odd_strechers=True,
        keep_even_strechers=True,
        strecher_instance=stretcher_geo,
        strecher_index_increment=stretcher_increment,
        strecher_relative_position=stretcher_relative_pos,
        leg_bottom_offset=0.0,
    )


@pf.nodes.node_function
def _base_square_geometry(
    dimensions: t.SocketOrVal[pf.Vector],
    leg_diameter: t.SocketOrVal[float],
    leg_placement_top_scale: t.SocketOrVal[float],
    leg_placement_bottom_scale: t.SocketOrVal[float],
    has_bottom_connector: t.SocketOrVal[bool],
) -> t.ProcNode[pf.MeshObject]:
    """2 box-frame legs."""
    x, y, z = dimensions.x, dimensions.y, dimensions.z
    anchors = _create_anchors(
        profile_n_gon=2,
        profile_width=1.414 * x * leg_placement_top_scale,
        profile_aspect_ratio=y / x,
        profile_rotation=0.0,
    )

    leg = _leg_square(
        height=1.0,
        width=y * leg_placement_top_scale,
        fillet_radius=0.03,
        has_bottom_connector=has_bottom_connector,
        profile_n_gon=4,
        profile_width=leg_diameter,
        profile_aspect_ratio=1.0,
        profile_fillet_ratio=0.1,
    )

    empty_stretcher = pf.nodes.geo.points(position=(0, 0, 0))

    return _create_legs_and_strechers(
        anchors=anchors,
        keep_legs=True,
        leg_instance=leg,
        table_height=z,
        leg_bottom_relative_scale=leg_placement_bottom_scale,
        leg_bottom_relative_rotation=0.0,
        keep_odd_strechers=False,
        keep_even_strechers=False,
        strecher_instance=empty_stretcher,
        strecher_index_increment=1,
        strecher_relative_position=0.0,
        leg_bottom_offset=0.0,
    )


def table_dimensions_rand(
    rng: pf.RNG,
    width: float | None = None,
    height: float | None = None,
) -> pf.Vector:
    """Default dining table dimensions."""
    aspect = pf.random.clip_gaussian(rng, 0.6, 0.2, 0.4, 1)

    if width is None:
        width = pf.random.clip_gaussian(rng, 0.975, 0.3, 0.675, 1.5)
    if height is None:
        height = pf.random.uniform(rng, 0.72, 0.76)
    depth = width / aspect
    width = min(width, 1.875)
    return (width, depth, height)


def base_straight(
    dimensions: pf.Vector | None = None,
    leg_diameter: float = 0.06,
    leg_inset: float = 0.1,
    leg_placement_bottom_scale: float = 1.0,
    stretcher_increment: int = 1,
    stretcher_relative_pos: float = 0.4,
) -> TableResult:
    """4-leg base with optional stretchers."""
    if dimensions is None:
        dimensions = pf.Vector((1.4, 0.8, 0.75))
    geo = _base_straight_geometry(
        dimensions=dimensions,
        leg_diameter=leg_diameter,
        leg_inset=leg_inset,
        leg_placement_bottom_scale=leg_placement_bottom_scale,
        stretcher_increment=stretcher_increment,
        stretcher_relative_pos=stretcher_relative_pos,
    )
    geo = mesh.crease_sharp(geo, threshold_degrees=40.0)
    obj = pf.nodes.to_mesh_object(geo)
    pf.ops.modifier.subdivide_surface(obj, levels=1, _skip_apply=True)
    return TableResult(mesh=obj)


def base_straight_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    leg_diameter: float | None = None,
    leg_placement_bottom_scale: float | None = None,
    leg_diameter_range: tuple[float, float] = (0.02, 0.10),
    leg_inset_range: tuple[float, float] | None = None,
    leg_placement_top_scale: float = 0.8,
    close_edges: bool = False,
) -> TableResult:
    """4-leg base with optional stretchers. leg_placement_bottom_scale > 1 splays
    the legs outward; pass 1.0 for an upright base to sit under a carcass."""
    rng, rng_dims, rng_mat, rng_inset = rng.spawn(4)
    if dimensions is None:
        dimensions = table_dimensions_rand(rng_dims)
    # drawn unconditionally so explicit overrides do not shift the stream
    sampled_diameter = pf.random.uniform(rng, *leg_diameter_range)
    if leg_diameter is None:
        leg_diameter = sampled_diameter
    sampled_bottom_scale = pf.random.uniform(rng, 0.95, 1.25)
    if leg_placement_bottom_scale is None:
        leg_placement_bottom_scale = 1.0 if close_edges else sampled_bottom_scale
    leg_inset = 0.0
    if leg_inset_range is not None and not close_edges:
        leg_inset = pf.random.uniform(rng_inset, *leg_inset_range)
    geo = _base_straight_geometry(
        dimensions=dimensions,
        leg_diameter=leg_diameter,
        leg_inset=leg_inset,
        leg_placement_top_scale=1.0 if close_edges else leg_placement_top_scale,
        leg_placement_bottom_scale=leg_placement_bottom_scale,
        stretcher_increment=pf.control.choice(rng, [(0, 1.0), (1, 1.0), (2, 1.0)]),
        stretcher_relative_pos=pf.random.uniform(rng, 0.2, 0.6),
    )
    geo = mesh.crease_sharp(geo, threshold_degrees=40.0)
    obj = pf.nodes.to_mesh_object(geo)
    if material is None:
        material = furniture_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    pf.ops.object.set_material(
        obj, surface=material.surface, displacement=material.displacement
    )
    pf.ops.modifier.subdivide_surface(obj, levels=1, _skip_apply=True)
    return TableResult(mesh=obj)


def pedestal_base(
    height: float,
    top_radius: float = 0.03,
    bottom_radius: float = 0.2,
    flare: float = 0.75,
    concavity: float = 0.0,
    neck_scale: float = 1.0,
    resolution: int = 16,
) -> TableResult:
    """Rotationally symmetric pedestal: stays skinny then flares out low
    (flare/concavity); neck_scale bulges or waists the upper column."""
    radius_curve = _pedestal_profile(
        height=height,
        top_radius=top_radius,
        bottom_radius=bottom_radius,
        flare=flare,
        concavity=concavity,
        neck_scale=neck_scale,
        resolution=resolution,
    )
    geo = _pedestal_sweep(radius_curve, resolution=resolution)
    geo = mesh.crease_sharp(geo, threshold_degrees=40.0)
    obj = pf.nodes.to_mesh_object(geo)
    pf.ops.modifier.subdivide_surface(obj, levels=3, _skip_apply=True)
    return TableResult(mesh=obj)


def _pedestal_profile_rand(
    rng: pf.RNG, height: float, top_radius: float, bottom_radius: float
) -> pf.ProcNode[pf.CurveObject]:
    rng_flare, rng_concavity, rng_neck = rng.spawn(3)
    return _pedestal_profile(
        height=height,
        top_radius=top_radius,
        bottom_radius=bottom_radius,
        flare=pf.random.uniform(rng_flare, 0.5, 0.85),
        concavity=pf.random.uniform(rng_concavity, 0.0, 1.0),
        neck_scale=pf.random.uniform(rng_neck, 0.5, 2.0),
    )


def pedestal_base_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    top_radius_range: tuple[float, float] | None = None,
    bottom_radius_range: tuple[float, float] | None = None,
) -> TableResult:
    """Rotationally symmetric pedestal spanning z in [0, dimensions.z]: a
    low-flare silhouette with random concavity and a bulged/waisted upper
    column. Radius ranges default from the footprint; callers override them."""
    rng, rng_dims, rng_mat, rng_profile = rng.spawn(4)
    if dimensions is None:
        dimensions = table_dimensions_rand(rng_dims)
    x = dimensions[0]
    y = dimensions[1]
    z = dimensions[2]
    footprint = min(x, y)
    if top_radius_range is None:
        top_radius_range = (0.015 * x, 0.06 * x)
    if bottom_radius_range is None:
        bottom_radius_range = (0.30 * footprint, 0.55 * footprint)
    top_radius = pf.random.uniform(rng, *top_radius_range)
    bottom_radius = pf.random.uniform(rng, *bottom_radius_range)
    radius_curve = _pedestal_profile_rand(rng_profile, z, top_radius, bottom_radius)
    geo = _pedestal_sweep(radius_curve)
    geo = mesh.crease_sharp(geo, threshold_degrees=40.0)
    obj = pf.nodes.to_mesh_object(geo)
    if material is None:
        material = furniture_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    pf.ops.object.set_material(
        obj, surface=material.surface, displacement=material.displacement
    )
    pf.ops.modifier.subdivide_surface(obj, levels=3, _skip_apply=True)
    return TableResult(mesh=obj)


def base_square(
    dimensions: pf.Vector | None = None,
    leg_diameter: float = 0.085,
    leg_placement_top_scale: float = 0.8,
    leg_placement_bottom_scale: float = 1.0,
    has_bottom_connector: bool = True,
) -> TableResult:
    """2 box-frame legs."""
    if dimensions is None:
        dimensions = pf.Vector((1.4, 0.8, 0.75))
    geo = _base_square_geometry(
        dimensions=dimensions,
        leg_diameter=leg_diameter,
        leg_placement_top_scale=leg_placement_top_scale,
        leg_placement_bottom_scale=leg_placement_bottom_scale,
        has_bottom_connector=has_bottom_connector,
    )
    geo = mesh.crease_sharp(geo, threshold_degrees=40.0)
    obj = pf.nodes.to_mesh_object(geo)
    pf.ops.modifier.subdivide_surface(obj, levels=1, _skip_apply=True)
    return TableResult(mesh=obj)


def base_square_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    leg_diameter: float | None = None,
    leg_placement_bottom_scale: float | None = None,
    leg_diameter_range: tuple[float, float] = (0.03, 0.14),
    leg_placement_top_scale: float = 0.8,
    close_edges: bool = False,
) -> TableResult:
    """2 box-frame legs. leg_placement_bottom_scale > 1 splays the frames outward."""
    rng, rng_dims, rng_mat = rng.spawn(3)
    if dimensions is None:
        dimensions = table_dimensions_rand(rng_dims)
    # drawn unconditionally so an explicit leg_diameter does not shift the stream
    sampled_diameter = pf.random.uniform(rng, *leg_diameter_range)
    if leg_diameter is None:
        leg_diameter = sampled_diameter
    if leg_placement_bottom_scale is None:
        leg_placement_bottom_scale = 0.98 if close_edges else 1.0
    geo = _base_square_geometry(
        dimensions=dimensions,
        leg_diameter=leg_diameter,
        leg_placement_top_scale=0.94 if close_edges else leg_placement_top_scale,
        leg_placement_bottom_scale=leg_placement_bottom_scale,
        has_bottom_connector=pf.control.choice(rng, [(True, 2.0), (False, 1.0)]),
    )
    geo = mesh.crease_sharp(geo, threshold_degrees=40.0)
    obj = pf.nodes.to_mesh_object(geo)
    if material is None:
        material = furniture_material_rand(rng_mat, pf.nodes.shader.coord().uv)
    pf.ops.object.set_material(
        obj, surface=material.surface, displacement=material.displacement
    )
    pf.ops.modifier.subdivide_surface(obj, levels=1, _skip_apply=True)
    return TableResult(mesh=obj)


def _table_straight_base_rand(rng: pf.RNG, dimensions: pf.Vector) -> TableResult:
    footprint = min(dimensions[0], dimensions[1])
    return base_straight_rand(
        rng,
        dimensions,
        leg_diameter_range=(0.02, 0.02 + 0.16 * footprint),
        leg_inset_range=(0.05 * footprint, 0.15 * footprint),
        leg_placement_top_scale=1.0,
    )


def _table_square_base_rand(rng: pf.RNG, dimensions: pf.Vector) -> TableResult:
    footprint = min(dimensions[0], dimensions[1])
    return base_square_rand(
        rng,
        dimensions,
        leg_diameter_range=(0.02, 0.02 + 0.12 * footprint),
    )


def _table_pedestal_rand(rng: pf.RNG, dimensions: pf.Vector) -> TableResult:
    x = dimensions[0]
    return pedestal_base_rand(
        rng,
        dimensions,
        top_radius_range=(0.012 * x, 0.05 * x),
        bottom_radius_range=(0.14 * x, 0.29 * x),
    )


def dining_table_rand(
    rng: pf.RNG,
    dimensions: tuple[float, float, float] | None = None,
    base: pf.MeshObject | None = None,
    top_thickness: float | None = None,
    top_support_loop_offset: pf.Vector | None = None,
    top_material: pf.Material | None = None,
    leg_material: pf.Material | None = None,
) -> TableResult:
    rng, rng_dims, rng_top_mat, rng_leg_mat, rng_base_choice, rng_base = rng.spawn(6)
    if dimensions is None:
        dimensions = table_dimensions_rand(rng_dims)

    x = dimensions[0]
    y = dimensions[1]
    z = dimensions[2]

    if top_thickness is None:
        top_thickness = pf.random.uniform(rng, 0.03, 0.08)

    top_height = z - top_thickness

    vec = pf.nodes.shader.coord().uv
    if top_material is None:
        top_material = table_top_material_rand(rng_top_mat, vec)
    if leg_material is None:
        leg_material = furniture_material_rand(rng_leg_mat, vec)

    if top_support_loop_offset is None:
        corner_frac_x = pf.random.uniform(rng, 0.1, 0.5)
        corner_frac_y = pf.random.uniform(rng, 0.1, 0.5)
        edge_frac = pf.random.uniform(rng, 0.1, 0.5)
        corner_shrink = pf.random.uniform(rng, 0.0, 1.0) ** 2
        top_support_loop_offset = pf.Vector(
            (
                corner_frac_x * corner_shrink * x,
                corner_frac_y * corner_shrink * y,
                edge_frac * top_thickness * (1.0 - corner_shrink),
            )
        )
    top = table_top(
        size=(x, y, top_thickness),
        support_loop_offset=top_support_loop_offset,
    )
    top = pf.nodes.geo.transform(
        top,
        translation=(0, 0, top_height),
        rotation=(0, 0, 0),
        scale=(1, 1, 1),
    )
    top = pf.nodes.to_mesh_object(top)
    pf.ops.object.set_material(
        top, surface=top_material.surface, displacement=top_material.displacement
    )

    if base is None:
        base_options = [
            (_table_straight_base_rand, 2.0),
            (_table_pedestal_rand, 1.0),
            (_table_square_base_rand, 0.6),
        ]
        base_fn = pf.control.choice(rng_base_choice, base_options)
        res = base_fn(rng=rng_base, dimensions=(x, y, top_height))
        base = res.mesh

    pf.ops.object.set_material(
        base, surface=leg_material.surface, displacement=leg_material.displacement
    )

    pf.ops.object.join(top, base)
    pf.ops.modifier.subdivide_surface(top, levels=3, _skip_apply=True)
    return TableResult(mesh=top)


def side_table_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    """Side table dimensions."""
    return (
        pf.random.uniform(rng, 0.4, 0.6),
        pf.random.uniform(rng, 0.4, 0.6),
        pf.random.uniform(rng, 0.3, 0.5),
    )


def side_table_rand(rng: pf.RNG) -> TableResult:
    """Side table."""
    rng, rng_dims, rng_table = rng.spawn(3)
    dimensions = side_table_dimensions_rand(rng_dims)
    top_thickness = pf.random.uniform(rng, 0.01, 0.04)
    return dining_table_rand(rng_table, dimensions, top_thickness=top_thickness)


def coffee_table_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    """Coffee table dimensions."""
    return (
        pf.random.uniform(rng, 0.6, 0.9),
        pf.random.uniform(rng, 1.0, 1.5),
        pf.random.uniform(rng, 0.3, 0.5),
    )


def _coffee_table_legged_rand(rng: pf.RNG) -> TableResult:
    """Low rectangular coffee table on legs."""
    rng, rng_dims, rng_table = rng.spawn(3)
    dimensions = coffee_table_dimensions_rand(rng_dims)
    top_thickness = pf.random.uniform(rng, 0.02, 0.04)
    return dining_table_rand(rng_table, dimensions, top_thickness=top_thickness)


def coffee_table_storage_rand(rng: pf.RNG) -> TableResult:
    result = storage.storage_coffee_table_rand(rng)
    mesh.center_footprint(result.mesh)
    return TableResult(mesh=result.mesh)


def coffee_table_rand(rng: pf.RNG) -> TableResult:
    """Coffee table with a 25% storage-with-legs chance."""
    func = pf.control.choice(
        rng,
        [
            (_coffee_table_legged_rand, 3.0),
            (coffee_table_storage_rand, 1.0),
        ],
    )
    result = func(rng)
    result.mesh.item().name = func.__name__
    return result


def _cocktail_table_pedestal_rand(rng: pf.RNG, dimensions: pf.Vector) -> TableResult:
    x = dimensions[0]
    return pedestal_base_rand(
        rng,
        dimensions,
        top_radius_range=(0.012 * x, 0.05 * x),
        bottom_radius_range=(0.325 * x, 0.52 * x),
    )


def cocktail_table_rand(rng: pf.RNG) -> TableResult:
    """Square cocktail/bar table, usually with a wide pedestal base."""
    rng_dims, rng_thickness, rng_base_choice, rng_base, rng_table = rng.spawn(5)
    x = pf.random.uniform(rng_dims, 0.5, 0.8)
    height = pf.random.uniform(rng_dims, 1.0, 1.1)
    top_thickness = pf.random.uniform(rng_thickness, 0.03, 0.08)
    base_fn = pf.control.choice(
        rng_base_choice,
        [
            (_cocktail_table_pedestal_rand, 2.0),
            (_table_straight_base_rand, 1.0),
            (_table_square_base_rand, 1.0),
        ],
    )
    base = base_fn(rng_base, (x, x, height - top_thickness))
    return dining_table_rand(
        rng_table,
        (x, x, height),
        base=base.mesh,
        top_thickness=top_thickness,
    )


if __name__ == "__main__":
    table_top_result = table_top()
    leg_straight_result = _leg_straight()
    leg_square_result = _leg_square()

    create_legs_and_strechers_result = _create_legs_and_strechers()
    create_anchors_result = _create_anchors()

    strecher_result = _strecher()
