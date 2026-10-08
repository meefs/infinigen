# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen toilet (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/bathroom/toilet.py)
# - Alexander Raistrick: refactor for Infinigen2

from typing import Literal, NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.base_materials import ceramic, metal_brushed, plastic
from infinigen2.util import mesh

__all__ = ["ToiletHardwareType", "ToiletResult", "toilet", "toilet_rand"]
ToiletHardwareType = Literal["button", "handle"]


class ToiletResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _toilet_bowl(
    material: t.SocketOrVal[pf.Material],
    seat_material: t.SocketOrVal[pf.Material],
    size: t.SocketOrVal[float],
    width: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    size_mid: t.SocketOrVal[float],
    depth: t.SocketOrVal[float],
    tube_scale: t.SocketOrVal[float],
    thickness: t.SocketOrVal[float],
    extrude_height: t.SocketOrVal[float],
    stand_depth: t.SocketOrVal[float],
    stand_scale: t.SocketOrVal[float],
    bottom_offset: t.SocketOrVal[float],
    back_thickness: t.SocketOrVal[float],
    back_size: t.SocketOrVal[float],
    back_scale: t.SocketOrVal[float],
    seat_thickness: t.SocketOrVal[float],
    seat_size: t.SocketOrVal[float],
    has_seat_cut: t.SocketOrVal[bool],
    tank_width: t.SocketOrVal[float],
    cover_rotation: t.SocketOrVal[float],
    tube_bridge_profile_factor: t.SocketOrVal[float],
    stand_bridge_profile_factor: t.SocketOrVal[float],
    curve_front: t.SocketOrVal[float],
    curve_front_side: t.SocketOrVal[float],
    curve_back_side: t.SocketOrVal[float],
    curve_back: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    loft_index = pf.nodes.geo.input_index().astype(dtype=float)
    loft_position = pf.nodes.geo.input_position()
    loft_u = loft_index / 8.0
    loft_front = pf.nodes.func.boolean_or(a=loft_u < 1.0, b=loft_u > 3.0)
    loft_s = 1.0 - pf.nodes.math.absolute(pf.nodes.math.modulo(loft_u, 2.0) - 1.0)
    loft_r = 1.0 - loft_s
    loft_length = pf.nodes.func.switch(
        loft_front, a=size * (1.0 - size_mid), b=size * size_mid
    )
    loft_half_width = width * 0.5
    loft_handle = (
        pf.nodes.math.sqrt(
            loft_length * loft_length + loft_half_width * loft_half_width
        )
        / 2.5614
    )
    loft_handle_axial = (
        pf.nodes.func.switch(loft_front, a=curve_back_side, b=curve_front_side)
        * loft_handle
    )
    loft_handle_side = (
        pf.nodes.func.switch(loft_front, a=curve_back, b=curve_front) * loft_handle
    )
    loft_axial = (
        loft_length * loft_r * loft_r * (1.0 + 2.0 * loft_s)
        + 3.0 * loft_handle_axial * loft_r * loft_s * loft_s
    )
    loft_side = (
        loft_half_width * loft_s * loft_s * (3.0 - 2.0 * loft_s)
        + 3.0 * loft_handle_side * loft_r * loft_r * loft_s
    )
    loft_axial_sign = pf.nodes.func.switch(loft_front, a=-1.0, b=1.0)
    loft_side_sign = pf.nodes.func.switch(loft_u < 2.0, a=-1.0, b=1.0)
    loft_profile_position = pf.nodes.math.combine_xyz(
        x=loft_axial * loft_axial_sign, y=loft_side * loft_side_sign
    )
    loft_circle = pf.nodes.geo.mesh_circle(vertices=32)
    loft_profile = pf.nodes.geo.set_position(
        loft_circle, position=loft_profile_position
    )
    loft_neighbor = pf.nodes.geo.sample_index(
        loft_profile,
        index=pf.nodes.math.modulo(loft_index + 1.0, 32.0).astype(dtype=int),
        value=loft_position,
    )
    loft_edge_length = pf.nodes.math.vector_length(loft_neighbor - loft_position)
    loft_weighted_center = (loft_position + loft_neighbor) * (loft_edge_length * 0.5)
    loft_center = pf.nodes.geo.attribute_statistic(
        loft_profile, attribute=loft_weighted_center
    ).mean
    loft_center_2 = loft_center * (
        1.0
        / pf.nodes.geo.attribute_statistic(
            loft_profile, attribute=loft_edge_length
        ).mean
    )
    loft_grid = pf.nodes.geo.mesh_grid(vertices_x=6, vertices_y=33).mesh
    loft_ring_index = pf.nodes.math.modulo(loft_index, 33.0)
    loft_along = pf.nodes.math.floor(loft_index / 33.0) / 5.0
    loft_p = pf.nodes.geo.sample_index(
        loft_profile,
        index=pf.nodes.math.modulo(loft_ring_index, 32.0).astype(dtype=int),
        value=loft_position,
    )
    loft_prev = pf.nodes.geo.sample_index(
        loft_profile,
        index=pf.nodes.math.modulo(loft_ring_index + 31.0, 32.0).astype(dtype=int),
        value=loft_position,
    )
    loft_next = pf.nodes.geo.sample_index(
        loft_profile,
        index=pf.nodes.math.modulo(loft_ring_index + 1.0, 32.0).astype(dtype=int),
        value=loft_position,
    )
    loft_shift = pf.nodes.math.combine_xyz(
        x=-tube_scale * size * (1.0 - size_mid) * 0.5, z=-depth
    )
    loft_q = loft_p * tube_scale + loft_shift
    loft_e1 = loft_p - loft_prev
    loft_e2 = loft_next - loft_p
    loft_d1 = (loft_p + loft_prev) * ((tube_scale - 1.0) * 0.5) + loft_shift
    loft_d2 = (loft_p + loft_next) * ((tube_scale - 1.0) * 0.5) + loft_shift
    loft_tangent1 = loft_d1 - loft_e1 * (
        pf.nodes.math.vector_dot_product(loft_d1, loft_e1)
        / pf.nodes.math.vector_dot_product(loft_e1, loft_e1)
    )
    loft_tangent2 = loft_d2 - loft_e2 * (
        pf.nodes.math.vector_dot_product(loft_d2, loft_e2)
        / pf.nodes.math.vector_dot_product(loft_e2, loft_e2)
    )
    loft_tangent = pf.nodes.math.vector_normalize(
        pf.nodes.math.vector_normalize(loft_tangent1)
        + pf.nodes.math.vector_normalize(loft_tangent2)
    )
    loft_handle_length = pf.nodes.math.vector_length(loft_q - loft_p) * 0.375
    loft_blend = loft_along * loft_along * (3.0 - 2.0 * loft_along)
    loft_handle_blend = 3.0 * loft_along * (1.0 - loft_along) * (1.0 - 2.0 * loft_along)
    loft_loft = (
        loft_p * (1.0 - loft_blend)
        + loft_q * loft_blend
        + loft_tangent * (loft_handle_length * loft_handle_blend)
    )
    loft_center_bottom = loft_center_2 * tube_scale + loft_shift
    loft_center_handle = (
        pf.nodes.math.vector_length(loft_center_bottom - loft_center_2) * 0.375
    )
    loft_center_curve = (
        loft_center_2 * (1.0 - loft_blend) + loft_center_bottom * loft_blend
    )
    loft_center_curve_2 = loft_center_curve + pf.nodes.math.combine_xyz(
        z=-loft_center_handle * loft_handle_blend
    )
    loft_bell = 1.0 - pf.nodes.math.absolute(2.0 * loft_along - 1.0)
    loft_expansion = 1.0 + tube_bridge_profile_factor * loft_bell * loft_bell * (
        3.0 - 2.0 * loft_bell
    )
    loft_loft_2 = (
        loft_center_curve_2 + (loft_loft - loft_center_curve_2) * loft_expansion
    )
    loft_mesh = pf.nodes.geo.set_position(loft_grid, position=loft_loft_2)
    loft_mesh_2 = pf.nodes.geo.merge_by_distance(loft_mesh, distance=1e-06)
    shell_index = pf.nodes.geo.input_index()
    shell_corner0 = pf.nodes.geo.corners_of_vertex(
        vertex_index=shell_index, sort_index=0
    )
    shell_corner1 = pf.nodes.geo.corners_of_vertex(
        vertex_index=shell_index, sort_index=1
    )
    shell_corner2 = pf.nodes.geo.corners_of_vertex(
        vertex_index=shell_index, sort_index=2
    )
    shell_corner3 = pf.nodes.geo.corners_of_vertex(
        vertex_index=shell_index, sort_index=3
    )
    shell_face0 = pf.nodes.geo.face_of_corner(shell_corner0.corner_index).face_index
    shell_face1 = pf.nodes.geo.face_of_corner(shell_corner1.corner_index).face_index
    shell_face2 = pf.nodes.geo.face_of_corner(shell_corner2.corner_index).face_index
    shell_face3 = pf.nodes.geo.face_of_corner(shell_corner3.corner_index).face_index
    shell_normal = pf.nodes.geo.input_normal()
    shell_n0 = pf.nodes.geo.field_at_index(
        shell_normal, index=shell_face0, domain="FACE"
    )
    shell_n1 = pf.nodes.geo.field_at_index(
        shell_normal, index=shell_face1, domain="FACE"
    )
    shell_n2 = pf.nodes.geo.field_at_index(
        shell_normal, index=shell_face2, domain="FACE"
    )
    shell_n3 = pf.nodes.geo.field_at_index(
        shell_normal, index=shell_face3, domain="FACE"
    )
    shell_mean = (shell_n0 + shell_n1 + shell_n2 + shell_n3) * 0.25
    shell_offset = shell_mean * (
        thickness / pf.nodes.math.vector_dot_product(shell_mean, shell_mean)
    )
    shell_n0_xyz = pf.nodes.math.separate_xyz(shell_n0)
    shell_n1_xyz = pf.nodes.math.separate_xyz(shell_n1)
    shell_determinant = (
        shell_n0_xyz.x * shell_n1_xyz.y - shell_n0_xyz.y * shell_n1_xyz.x
    )
    shell_boundary_x = thickness * (shell_n1_xyz.y - shell_n0_xyz.y) / shell_determinant
    shell_boundary_y = thickness * (shell_n0_xyz.x - shell_n1_xyz.x) / shell_determinant
    shell_boundary = pf.nodes.math.combine_xyz(x=shell_boundary_x, y=shell_boundary_y)
    shell_offset_2 = pf.nodes.func.switch(
        shell_corner0.total == 2, a=shell_offset, b=shell_boundary
    )
    shell_captured = pf.nodes.geo.capture_attribute(
        loft_mesh_2, domain="POINT", wall_offset=shell_offset_2
    )
    shell_wall_offset = shell_captured.attributes["wall_offset"]
    shell_outer = pf.nodes.geo.set_position(
        shell_captured.geometry, offset=shell_wall_offset
    )
    shell_inner = pf.nodes.geo.set_position(
        shell_captured.geometry, offset=shell_wall_offset * (-1e-05 / thickness)
    )
    shell_extruded = pf.nodes.geo.extrude_mesh(
        shell_inner, offset=pf.nodes.math.combine_xyz(), individual=False
    )
    shell_extruded_outer = pf.nodes.geo.set_position(
        shell_extruded.mesh,
        offset=shell_wall_offset * (1.0 + 1e-05 / thickness),
        selection=shell_extruded.top,
    )
    shell_solid = pf.nodes.geo.join_geometry(
        [pf.nodes.geo.flip_faces(shell_inner), shell_extruded_outer]
    )
    shell_solid_2 = pf.nodes.geo.merge_by_distance(shell_solid, distance=1e-06)
    shell_normal_z = pf.nodes.math.separate_xyz(pf.nodes.geo.input_normal()).z
    shell_rim_offset = pf.nodes.math.combine_xyz(z=thickness + extrude_height)
    shell_rim = pf.nodes.geo.extrude_mesh(
        shell_solid_2,
        selection=shell_normal_z > 0.9,
        offset=shell_rim_offset,
        individual=False,
    ).mesh
    shell_position_z = pf.nodes.math.separate_xyz(loft_position).z
    shell_clamp_offset = pf.nodes.math.combine_xyz(
        z=pf.nodes.math.minimum(extrude_height - shell_position_z, 0.0)
    )
    shell_rim_2 = pf.nodes.geo.set_position(shell_rim, offset=shell_clamp_offset)
    profile = pf.nodes.geo.separate_geometry(
        loft_mesh_2, selection=shell_index < 32
    ).selection
    edge = pf.nodes.geo.input_mesh_edge_vertices()
    direction = pf.nodes.math.vector_normalize(edge.position_2 - edge.position_1)
    horizontal = pf.nodes.math.absolute(direction.z) < 0.1
    under_depth = loft_position.z < -stand_depth
    score = (
        -loft_position.x
        - horizontal.astype(dtype=float)
        - under_depth.astype(dtype=float)
    )
    minimum = pf.nodes.geo.attribute_statistic(
        shell_outer, attribute=score, domain="EDGE"
    ).min
    best_edge = pf.nodes.geo.attribute_statistic(
        shell_outer,
        attribute=shell_index.astype(dtype=float),
        selection=score <= minimum,
        domain="EDGE",
    ).min.astype(dtype=int)
    row_field = pf.nodes.geo.field_on_domain(
        pf.nodes.math.floor(shell_index.astype(dtype=float) / 32.0), domain="POINT"
    )
    row = pf.nodes.geo.sample_index(
        shell_outer, index=best_edge, value=row_field, domain="EDGE"
    )
    top_position = pf.nodes.geo.sample_index(
        shell_outer,
        index=(shell_index.astype(dtype=float) + 32.0 * row).astype(dtype=int),
        value=loft_position,
    )
    top = pf.nodes.geo.set_position(profile, position=top_position)
    bottom_shift = pf.nodes.math.combine_xyz(
        x=-tube_scale * size * (1.0 - size_mid) * bottom_offset * 0.5, z=-height
    )
    bottom_scale = pf.nodes.math.combine_xyz(
        x=stand_scale, y=stand_scale, z=stand_scale
    )
    bottom = pf.nodes.geo.transform(
        profile, scale=bottom_scale, translation=bottom_shift
    )
    stand_next_index = pf.nodes.math.modulo(
        shell_index.astype(dtype=float) + 1.0, 32.0
    ).astype(dtype=int)
    stand_top_next = pf.nodes.geo.sample_index(
        top, index=stand_next_index, value=loft_position
    )
    stand_bottom_next = pf.nodes.geo.sample_index(
        bottom, index=stand_next_index, value=loft_position
    )
    stand_top_edge = pf.nodes.math.vector_length(stand_top_next - loft_position)
    stand_bottom_edge = pf.nodes.math.vector_length(stand_bottom_next - loft_position)
    stand_top_center = pf.nodes.geo.attribute_statistic(
        top, attribute=(loft_position + stand_top_next) * (0.5 * stand_top_edge)
    ).mean
    stand_top_center_2 = stand_top_center * (
        1.0 / pf.nodes.geo.attribute_statistic(top, attribute=stand_top_edge).mean
    )
    stand_bottom_center = pf.nodes.geo.attribute_statistic(
        bottom,
        attribute=(loft_position + stand_bottom_next) * (0.5 * stand_bottom_edge),
    ).mean
    stand_bottom_center_2 = stand_bottom_center * (
        1.0 / pf.nodes.geo.attribute_statistic(bottom, attribute=stand_bottom_edge).mean
    )
    stand_top_cross = pf.nodes.math.vector_cross_product(loft_position, stand_top_next)
    stand_top_normal = pf.nodes.math.vector_normalize(
        pf.nodes.geo.attribute_statistic(top, attribute=stand_top_cross).mean
    )
    stand_top_normal_2 = pf.nodes.func.switch(
        stand_top_normal.z > 0.0, a=stand_top_normal, b=-stand_top_normal
    )
    stand_normal_dot = stand_top_normal_2.z
    stand_handle_factor = 1.333333 * (1.0 + stand_normal_dot) - 0.75 * stand_normal_dot
    stand_handle = pf.nodes.math.vector_length(
        stand_bottom_center_2 - stand_top_center_2
    ) * (0.5 * stand_handle_factor)
    stand_along = shell_index.astype(dtype=float) / 5.0
    stand_reverse = 1.0 - stand_along
    stand_handle_a = stand_top_center_2 + stand_top_normal_2 * stand_handle
    stand_handle_b = stand_bottom_center_2 + pf.nodes.math.combine_xyz(z=stand_handle)
    stand_center = stand_top_center_2 * (stand_reverse * stand_reverse * stand_reverse)
    stand_center_2 = stand_center + stand_handle_a * (
        3.0 * stand_reverse * stand_reverse * stand_along
    )
    stand_center_3 = stand_center_2 + stand_handle_b * (
        3.0 * stand_reverse * stand_along * stand_along
    )
    stand_center_4 = stand_center_3 + stand_bottom_center_2 * (
        stand_along * stand_along * stand_along
    )
    stand_path = pf.nodes.geo.mesh_line(
        start_location=(0.0, 0.0, 0.0), offset=(0.0, 0.0, 1.0), count=6
    )
    stand_path_2 = pf.nodes.geo.set_position(stand_path, position=stand_center_4)
    stand_previous = pf.nodes.geo.sample_index(
        stand_path_2,
        index=(shell_index.astype(dtype=float) - 1.0).astype(dtype=int),
        value=loft_position,
        clamp=True,
    )
    stand_following = pf.nodes.geo.sample_index(
        stand_path_2,
        index=(shell_index.astype(dtype=float) + 1.0).astype(dtype=int),
        value=loft_position,
        clamp=True,
    )
    stand_direction_a = pf.nodes.math.vector_normalize(loft_position - stand_previous)
    stand_direction_b = pf.nodes.math.vector_normalize(stand_following - loft_position)
    stand_direction = pf.nodes.math.vector_normalize(
        stand_direction_a + stand_direction_b
    )
    stand_direction_2 = pf.nodes.func.switch(
        shell_index == 0, a=stand_direction, b=stand_top_normal_2
    )
    stand_direction_3 = pf.nodes.func.switch(
        shell_index == 5, a=stand_direction_2, b=(0.0, 0.0, -1.0)
    )
    stand_angle_field = -pf.nodes.math.atan2(stand_direction_3.x, -stand_direction_3.z)
    stand_path_capture = pf.nodes.geo.capture_attribute(
        stand_path_2, value=stand_angle_field, domain="POINT"
    )
    stand_ring_index = pf.nodes.math.modulo(shell_index.astype(dtype=float), 33.0)
    stand_row = pf.nodes.math.floor(shell_index.astype(dtype=float) / 33.0).astype(
        dtype=int
    )
    stand_along_2 = stand_row.astype(dtype=float) / 5.0
    stand_p = pf.nodes.geo.sample_index(
        top,
        index=pf.nodes.math.modulo(stand_ring_index, 32.0).astype(dtype=int),
        value=loft_position,
    )
    stand_q = pf.nodes.geo.sample_index(
        bottom,
        index=pf.nodes.math.modulo(stand_ring_index, 32.0).astype(dtype=int),
        value=loft_position,
    )
    stand_angle = pf.nodes.geo.sample_index(
        stand_path_capture.geometry, index=stand_row, value=stand_path_capture.value
    )
    stand_top_angle = -pf.nodes.math.atan2(stand_top_normal_2.x, -stand_top_normal_2.z)
    stand_unrotate = pf.nodes.math.combine_xyz(y=-stand_top_angle).astype(
        dtype=pf.Euler
    )
    stand_p_2 = pf.nodes.func.rotate_vector(
        stand_p - stand_top_center_2, rotation=stand_unrotate
    )
    stand_shape = (
        stand_p_2 * (1.0 - stand_along_2)
        + (stand_q - stand_bottom_center_2) * stand_along_2
    )
    stand_rotation = pf.nodes.math.combine_xyz(y=stand_angle).astype(dtype=pf.Euler)
    stand_shape_2 = pf.nodes.func.rotate_vector(stand_shape, rotation=stand_rotation)
    stand_bell = 1.0 - pf.nodes.math.absolute(2.0 * stand_along_2 - 1.0)
    stand_scale = 1.0 + stand_bridge_profile_factor * stand_bell * stand_bell * (
        3.0 - 2.0 * stand_bell
    )
    stand_origin = pf.nodes.geo.sample_index(
        stand_path_2, index=stand_row, value=loft_position
    )
    stand_mesh = pf.nodes.geo.set_position(
        loft_grid, position=stand_origin + stand_shape_2 * stand_scale
    )
    stand_result = pf.nodes.geo.merge_by_distance(stand_mesh, distance=1e-06)
    back_back_length = size * (1.0 - size_mid)
    back_rear = -back_back_length - back_size
    back_bottom = -depth + thickness * 0.1
    back_top = extrude_height * 0.25
    back_front = -back_back_length + back_thickness
    back_width = tank_width * back_scale * 0.5
    back_dimensions = pf.nodes.math.combine_xyz(
        x=back_front - back_rear,
        y=back_width,
        z=back_top - back_bottom,
    )
    back_location = pf.nodes.math.combine_xyz(
        x=back_rear,
        y=back_width * -0.5,
        z=back_bottom,
    )
    back_result = mesh.box(
        size=back_dimensions,
        location=back_location,
        anchor=(0.0, 0.0, 0.0),
    )
    seat_xyz = pf.nodes.math.separate_xyz(loft_position)
    seat_top = pf.nodes.geo.separate_geometry(
        shell_rim_2,
        selection=seat_xyz.z > extrude_height * (2.0 / 3.0),
        domain="FACE",
    ).selection
    seat_x1 = pf.nodes.math.separate_xyz(edge.position_1).x
    seat_x2 = pf.nodes.math.separate_xyz(edge.position_2).x
    seat_back_edges = pf.nodes.func.boolean_and(
        a=seat_x1 < -back_back_length - seat_thickness,
        b=seat_x2 < -back_back_length - seat_thickness,
    )
    seat_back_offset = pf.nodes.math.combine_xyz(x=-seat_size - thickness * 2.0)
    seat_plane = pf.nodes.geo.extrude_mesh(
        seat_top, selection=seat_back_edges, offset=seat_back_offset, mode="EDGES"
    ).mesh
    seat_clamp = pf.nodes.math.combine_xyz(
        x=pf.nodes.math.maximum(-back_back_length - seat_size - seat_xyz.x, 0.0)
    )
    seat_plane_2 = pf.nodes.geo.set_position(seat_plane, offset=seat_clamp)
    seat_ring_position = pf.nodes.geo.sample_index(
        shell_rim_2, index=loft_index.astype(dtype=int), value=loft_position
    )
    seat_ring = pf.nodes.geo.set_position(loft_circle, position=seat_ring_position)
    seat_previous = pf.nodes.geo.sample_index(
        shell_rim_2,
        index=pf.nodes.math.modulo(loft_index + 31.0, 32.0).astype(dtype=int),
        value=loft_position,
    )
    seat_following = pf.nodes.geo.sample_index(
        shell_rim_2,
        index=pf.nodes.math.modulo(loft_index + 1.0, 32.0).astype(dtype=int),
        value=loft_position,
    )
    seat_incoming = pf.nodes.math.vector_normalize(seat_ring_position - seat_previous)
    seat_outgoing = pf.nodes.math.vector_normalize(seat_following - seat_ring_position)
    seat_cosine = pf.nodes.math.vector_dot_product(seat_incoming, seat_outgoing)
    seat_minimum = pf.nodes.geo.attribute_statistic(
        seat_ring, selection=loft_index < 16.5, attribute=seat_cosine
    ).min
    seat_sharpest = pf.nodes.func.boolean_and(
        a=loft_index < 16.5, b=seat_cosine < seat_minimum + 1e-07
    )
    seat_start = pf.nodes.geo.attribute_statistic(
        seat_ring, selection=seat_sharpest, attribute=loft_index
    ).min
    seat_row = pf.nodes.math.floor(loft_index * 0.5)
    seat_column = pf.nodes.math.modulo(loft_index, 2.0)
    seat_profile_index = pf.nodes.func.switch(
        seat_column > 0.5,
        a=47.0 - seat_start - seat_row,
        b=48.0 - seat_start + seat_row,
    )
    seat_profile_index_2 = pf.nodes.math.modulo(seat_profile_index, 32.0)
    seat_fill_position = pf.nodes.geo.sample_index(
        shell_rim_2,
        index=seat_profile_index_2.astype(dtype=int),
        value=loft_position,
    )
    seat_fill_position_2 = seat_fill_position + pf.nodes.math.combine_xyz(
        z=extrude_height
    )
    seat_fill_grid = pf.nodes.geo.mesh_grid(vertices_x=16, vertices_y=2).mesh
    seat_fill_grid_2 = pf.nodes.geo.set_position(
        seat_fill_grid, position=seat_fill_position_2
    )
    seat_fill_grid_3 = pf.nodes.geo.flip_faces(seat_fill_grid_2)
    seat_filled = pf.nodes.geo.join_geometry([seat_plane_2, seat_fill_grid_3])
    seat_filled_2 = pf.nodes.geo.merge_by_distance(seat_filled, distance=1e-06)
    seat_vertical = pf.nodes.math.combine_xyz(z=extrude_height)
    seat_front = size_mid * size - thickness * 0.5
    seat_opening = pf.nodes.func.boolean_and(
        has_seat_cut, pf.nodes.geo.input_position().x > seat_front
    )
    seat_topology = pf.nodes.geo.delete_geometry(
        seat_plane_2, selection=seat_opening, domain="FACE"
    )
    seat_seat = pf.nodes.geo.extrude_mesh(
        seat_topology, offset=seat_vertical, individual=False
    ).mesh
    seat_seat_2 = pf.nodes.geo.join_geometry(
        [pf.nodes.geo.flip_faces(seat_topology), seat_seat]
    )
    seat_seat_4 = pf.nodes.geo.merge_by_distance(seat_seat_2, distance=1e-06)
    seat_lid = pf.nodes.geo.extrude_mesh(
        seat_filled_2, offset=seat_vertical, individual=False
    ).mesh
    seat_lid_2 = pf.nodes.geo.join_geometry(
        [pf.nodes.geo.flip_faces(seat_filled_2), seat_lid]
    )
    seat_lid_3 = pf.nodes.geo.merge_by_distance(seat_lid_2, distance=1e-06)
    seat_hinge_x = back_back_length + seat_size - extrude_height * 0.5
    seat_lid_4 = pf.nodes.geo.transform(
        seat_lid_3,
        translation=pf.nodes.math.combine_xyz(x=seat_hinge_x, z=-extrude_height * 0.5),
    )
    seat_lid_5 = pf.nodes.geo.transform(
        seat_lid_4,
        rotation=pf.nodes.math.combine_xyz(y=cover_rotation),
        translation=pf.nodes.math.combine_xyz(x=-seat_hinge_x, z=extrude_height * 1.5),
    )
    shell_material = pf.nodes.geo.set_material(shell_rim_2, material)
    shell_data = pf.nodes.geo.capture_attribute(
        shell_material, domain="FACE", marker=1.0
    )
    seat_seat_5 = pf.nodes.geo.set_material(seat_seat_4, seat_material)
    seat_data = pf.nodes.geo.capture_attribute(seat_seat_5, domain="FACE", marker=1.0)
    seat_lid_6 = pf.nodes.geo.set_material(seat_lid_5, seat_material)
    lid_data = pf.nodes.geo.capture_attribute(seat_lid_6, domain="FACE", marker=1.0)
    joined_parts = pf.nodes.geo.join_geometry(
        [shell_data.geometry, seat_data.geometry, lid_data.geometry]
    )
    bevel_data = pf.nodes.geo.capture_attribute(
        joined_parts,
        domain="FACE",
        part=seat_data.attributes["marker"] + lid_data.attributes["marker"] * 2.0,
        segments=(
            shell_data.attributes["marker"] * 2.0
            + seat_data.attributes["marker"]
            + lid_data.attributes["marker"]
        ),
    )
    bevel_input = bevel_data.geometry
    bevel_part = bevel_data.attributes["part"]
    bevel_segments = bevel_data.attributes["segments"]
    bowl_bevel_width = extrude_height * 0.5
    bowl_bevel_selected = (
        pf.nodes.geo.input_mesh_edge_angle().unsigned_angle > 0.5235987756
    )
    bowl_bevel_edge = pf.nodes.geo.edges_of_corner(shell_index)
    bowl_bevel_selected_next = pf.nodes.geo.field_at_index(
        value=bowl_bevel_selected,
        index=bowl_bevel_edge.next_edge_index,
        domain="EDGE",
    )
    bowl_bevel_selected_prev = pf.nodes.geo.field_at_index(
        value=bowl_bevel_selected,
        index=bowl_bevel_edge.previous_edge_index,
        domain="EDGE",
    )
    bowl_bevel_chosen = pf.nodes.func.boolean_or(
        a=bowl_bevel_selected_next, b=bowl_bevel_selected_prev
    )
    bowl_bevel_next_corner = pf.nodes.geo.offset_corner_in_face(shell_index, offset=1)
    bowl_bevel_prev_corner = pf.nodes.geo.offset_corner_in_face(shell_index, offset=-1)
    bowl_bevel_pnext = pf.nodes.geo.field_at_index(
        value=loft_position, index=bowl_bevel_next_corner, domain="CORNER"
    )
    bowl_bevel_pprev = pf.nodes.geo.field_at_index(
        value=loft_position, index=bowl_bevel_prev_corner, domain="CORNER"
    )
    bowl_bevel_unselected_direction = pf.nodes.func.switch(
        bowl_bevel_selected_next,
        a=bowl_bevel_pnext - loft_position,
        b=bowl_bevel_pprev - loft_position,
    )
    bowl_bevel_direction = pf.nodes.math.vector_normalize(
        bowl_bevel_unselected_direction
    )
    bowl_bevel_dn = pf.nodes.math.vector_normalize(bowl_bevel_pnext - loft_position)
    bowl_bevel_dp = pf.nodes.math.vector_normalize(bowl_bevel_pprev - loft_position)
    bowl_bevel_sine = pf.nodes.math.vector_length(
        pf.nodes.math.vector_cross_product(bowl_bevel_dn, bowl_bevel_dp)
    )
    bowl_bevel_cosine = pf.nodes.math.vector_dot_product(bowl_bevel_dn, bowl_bevel_dp)
    bowl_bevel_next_sine = pf.nodes.geo.field_at_index(
        value=bowl_bevel_sine, index=bowl_bevel_next_corner, domain="CORNER"
    )
    bowl_bevel_next_cosine = pf.nodes.geo.field_at_index(
        value=bowl_bevel_cosine, index=bowl_bevel_next_corner, domain="CORNER"
    )
    bowl_bevel_next_selected = pf.nodes.geo.field_at_index(
        value=bowl_bevel_selected_next, index=bowl_bevel_next_corner, domain="CORNER"
    )
    bowl_bevel_ka = bowl_bevel_selected_prev.astype(dtype=float)
    bowl_bevel_kb = bowl_bevel_selected_next.astype(dtype=float)
    bowl_bevel_kc = bowl_bevel_next_selected.astype(dtype=float)
    bowl_bevel_denominator = (
        bowl_bevel_ka + bowl_bevel_kb * bowl_bevel_cosine
    ) / bowl_bevel_sine + (
        bowl_bevel_kc + bowl_bevel_kb * bowl_bevel_next_cosine
    ) / bowl_bevel_next_sine
    bowl_bevel_edge_length = pf.nodes.math.vector_length(
        bowl_bevel_pnext - loft_position
    )
    bowl_bevel_collapse = bowl_bevel_edge_length / bowl_bevel_denominator
    bowl_bevel_valid_collapse = pf.nodes.func.boolean_and(
        a=bowl_bevel_denominator > 1e-06,
        b=bowl_bevel_ka + bowl_bevel_kb + bowl_bevel_kc > 1.5,
    )
    bowl_bevel_collapse_2 = pf.nodes.func.switch(
        bowl_bevel_valid_collapse, a=1000000.0, b=bowl_bevel_collapse
    )
    bowl_bevel_previous_length = pf.nodes.math.vector_length(
        bowl_bevel_pprev - loft_position
    )
    bowl_bevel_next_length = pf.nodes.geo.field_at_index(
        value=bowl_bevel_edge_length, index=bowl_bevel_next_corner, domain="CORNER"
    )
    bowl_bevel_slide_a = pf.nodes.func.switch(
        pf.nodes.func.boolean_and(
            a=bowl_bevel_selected_next,
            b=pf.nodes.func.boolean_not(bowl_bevel_selected_prev),
        ),
        a=1000000.0,
        b=bowl_bevel_previous_length * bowl_bevel_sine,
    )
    bowl_bevel_slide_c = pf.nodes.func.switch(
        pf.nodes.func.boolean_and(
            a=bowl_bevel_selected_next,
            b=pf.nodes.func.boolean_not(bowl_bevel_next_selected),
        ),
        a=1000000.0,
        b=bowl_bevel_next_length * bowl_bevel_next_sine,
    )
    bowl_bevel_collision_width = pf.nodes.math.minimum(
        bowl_bevel_collapse_2,
        pf.nodes.math.minimum(bowl_bevel_slide_a, bowl_bevel_slide_c),
    )
    bowl_bevel_limit0 = pf.nodes.geo.attribute_statistic(
        bevel_input,
        selection=bevel_part == 0.0,
        attribute=bowl_bevel_collision_width,
        domain="CORNER",
    ).min
    bowl_bevel_limit1 = pf.nodes.geo.attribute_statistic(
        bevel_input,
        selection=bevel_part == 1.0,
        attribute=bowl_bevel_collision_width,
        domain="CORNER",
    ).min
    bowl_bevel_limit2 = pf.nodes.geo.attribute_statistic(
        bevel_input,
        selection=bevel_part == 2.0,
        attribute=bowl_bevel_collision_width,
        domain="CORNER",
    ).min
    bowl_bevel_limit = pf.nodes.func.switch(
        bevel_part == 1.0, a=bowl_bevel_limit0, b=bowl_bevel_limit1
    )
    bowl_bevel_limit_2 = pf.nodes.func.switch(
        bevel_part == 2.0, a=bowl_bevel_limit, b=bowl_bevel_limit2
    )
    bowl_bevel_width_2 = (
        pf.nodes.math.minimum(bowl_bevel_width, bowl_bevel_limit_2) * 0.9998
    )
    bowl_bevel_vertex = pf.nodes.geo.vertex_of_corner(shell_index)
    bowl_bevel_selected_count = pf.nodes.geo.accumulate_field(
        value=bowl_bevel_selected_next.astype(dtype=int),
        group_id=bowl_bevel_vertex,
        domain="CORNER",
    ).total
    bowl_bevel_slide_edge = pf.nodes.func.switch(
        bowl_bevel_selected_next,
        a=bowl_bevel_edge.next_edge_index,
        b=bowl_bevel_edge.previous_edge_index,
    )
    bowl_bevel_edge_count = pf.nodes.geo.attribute_domain_size(
        bevel_input
    ).edge_count.astype(dtype=float)
    bowl_bevel_group = (
        bowl_bevel_vertex.astype(dtype=float) * bowl_bevel_edge_count
        + bowl_bevel_slide_edge.astype(dtype=float)
    ).astype(dtype=int)
    bowl_bevel_offset_sum = pf.nodes.geo.accumulate_field(
        value=bowl_bevel_chosen.astype(dtype=float) / bowl_bevel_sine,
        group_id=bowl_bevel_group,
        domain="CORNER",
    ).total
    bowl_bevel_offset_count = pf.nodes.geo.accumulate_field(
        value=bowl_bevel_chosen.astype(dtype=float),
        group_id=bowl_bevel_group,
        domain="CORNER",
    ).total
    bowl_bevel_offset = bowl_bevel_offset_sum / pf.nodes.math.maximum(
        bowl_bevel_offset_count, 1.0
    )
    bowl_bevel_endpoint = loft_position + bowl_bevel_direction * (
        bowl_bevel_offset * bowl_bevel_width_2
    )
    bowl_bevel_face_index = pf.nodes.geo.face_of_corner(shell_index).face_index
    bowl_bevel_face_normal = pf.nodes.geo.field_at_index(
        value=pf.nodes.geo.input_normal(), index=bowl_bevel_face_index, domain="FACE"
    )
    bowl_bevel_corner_cross = pf.nodes.math.vector_cross_product(
        bowl_bevel_dn, bowl_bevel_dp
    )
    bowl_bevel_corner_dot = pf.nodes.math.vector_dot_product(
        bowl_bevel_corner_cross, bowl_bevel_face_normal
    )
    bowl_bevel_corner_sign = pf.nodes.func.switch(
        bowl_bevel_corner_dot < 0.0,
        a=1.0,
        b=-1.0,
    )
    bowl_bevel_corner_offset = (bowl_bevel_dn + bowl_bevel_dp) * (
        bowl_bevel_corner_sign * bowl_bevel_width_2 / bowl_bevel_sine
    )
    bowl_bevel_face_center = pf.nodes.geo.field_at_index(
        value=loft_position, index=bowl_bevel_face_index, domain="FACE"
    )
    bowl_bevel_straight_cross = pf.nodes.math.vector_cross_product(
        bowl_bevel_face_normal, bowl_bevel_dn
    )
    bowl_bevel_straight_direction = pf.nodes.math.vector_normalize(
        bowl_bevel_straight_cross
    )
    bowl_bevel_center_offset = bowl_bevel_face_center - loft_position
    bowl_bevel_straight_dot = pf.nodes.math.vector_dot_product(
        bowl_bevel_straight_direction, bowl_bevel_center_offset
    )
    bowl_bevel_straight_sign = pf.nodes.func.switch(
        bowl_bevel_straight_dot < 0.0,
        a=1.0,
        b=-1.0,
    )
    bowl_bevel_corner_offset_2 = pf.nodes.func.switch(
        bowl_bevel_sine < 0.0001,
        a=bowl_bevel_corner_offset,
        b=bowl_bevel_straight_direction
        * (bowl_bevel_straight_sign * bowl_bevel_width_2),
    )
    bowl_bevel_both_selected = pf.nodes.func.boolean_and(
        a=bowl_bevel_selected_next, b=bowl_bevel_selected_prev
    )
    bowl_bevel_endpoint_2 = pf.nodes.func.switch(
        bowl_bevel_both_selected,
        a=bowl_bevel_endpoint,
        b=loft_position + bowl_bevel_corner_offset_2,
    )
    bowl_bevel_edge_sum = edge.position_1 + edge.position_2
    bowl_bevel_chosen_weight = -bowl_bevel_selected.astype(dtype=float)
    bowl_bevel_chosen_edge0 = pf.nodes.geo.edges_of_vertex(
        bowl_bevel_vertex, weights=bowl_bevel_chosen_weight, sort_index=0
    ).edge_index
    bowl_bevel_chosen_edge1 = pf.nodes.geo.edges_of_vertex(
        bowl_bevel_vertex, weights=bowl_bevel_chosen_weight, sort_index=1
    ).edge_index
    bowl_bevel_chosen_edge2 = pf.nodes.geo.edges_of_vertex(
        bowl_bevel_vertex, weights=bowl_bevel_chosen_weight, sort_index=2
    ).edge_index
    bowl_bevel_d0 = pf.nodes.math.vector_normalize(
        pf.nodes.geo.field_at_index(
            value=bowl_bevel_edge_sum, index=bowl_bevel_chosen_edge0, domain="EDGE"
        )
        - loft_position * 2.0
    )
    bowl_bevel_d1 = pf.nodes.math.vector_normalize(
        pf.nodes.geo.field_at_index(
            value=bowl_bevel_edge_sum, index=bowl_bevel_chosen_edge1, domain="EDGE"
        )
        - loft_position * 2.0
    )
    bowl_bevel_d2 = pf.nodes.math.vector_normalize(
        pf.nodes.geo.field_at_index(
            value=bowl_bevel_edge_sum, index=bowl_bevel_chosen_edge2, domain="EDGE"
        )
        - loft_position * 2.0
    )
    bowl_bevel_alignment0 = pf.nodes.math.absolute(
        pf.nodes.math.vector_dot_product(bowl_bevel_d0, bowl_bevel_face_normal)
    )
    bowl_bevel_alignment1 = pf.nodes.math.absolute(
        pf.nodes.math.vector_dot_product(bowl_bevel_d1, bowl_bevel_face_normal)
    )
    bowl_bevel_alignment2 = pf.nodes.math.absolute(
        pf.nodes.math.vector_dot_product(bowl_bevel_d2, bowl_bevel_face_normal)
    )
    bowl_bevel_omit0 = pf.nodes.func.boolean_and(
        a=bowl_bevel_alignment0 > bowl_bevel_alignment1,
        b=bowl_bevel_alignment0 > bowl_bevel_alignment2,
    )
    bowl_bevel_omit1 = pf.nodes.func.boolean_and(
        a=bowl_bevel_alignment1 > bowl_bevel_alignment0,
        b=bowl_bevel_alignment1 > bowl_bevel_alignment2,
    )
    bowl_bevel_pair_a = pf.nodes.func.switch(
        bowl_bevel_omit0, a=bowl_bevel_d0, b=bowl_bevel_d1
    )
    bowl_bevel_pair_b = pf.nodes.func.switch(
        pf.nodes.func.boolean_or(a=bowl_bevel_omit0, b=bowl_bevel_omit1),
        a=bowl_bevel_d1,
        b=bowl_bevel_d2,
    )
    bowl_bevel_pair_sum = bowl_bevel_pair_a + bowl_bevel_pair_b
    bowl_bevel_pair_basis = bowl_bevel_dn + bowl_bevel_dp
    bowl_bevel_pair_dot = pf.nodes.math.vector_dot_product(
        bowl_bevel_pair_sum, bowl_bevel_pair_basis
    )
    bowl_bevel_pair_sign = pf.nodes.func.switch(
        bowl_bevel_pair_dot < 0.0,
        a=1.0,
        b=-1.0,
    )
    bowl_bevel_pair_cross = pf.nodes.math.vector_cross_product(
        bowl_bevel_pair_a, bowl_bevel_pair_b
    )
    bowl_bevel_pair_sine = pf.nodes.math.vector_length(bowl_bevel_pair_cross)
    bowl_bevel_triple_endpoint = loft_position + bowl_bevel_pair_sum * (
        bowl_bevel_pair_sign * bowl_bevel_width_2 / bowl_bevel_pair_sine
    )
    bowl_bevel_all_directions = pf.nodes.geo.accumulate_field(
        value=bowl_bevel_dn, group_id=bowl_bevel_vertex, domain="CORNER"
    ).total
    bowl_bevel_slide_directions = pf.nodes.geo.accumulate_field(
        value=bowl_bevel_direction * bowl_bevel_chosen.astype(dtype=float),
        group_id=bowl_bevel_vertex,
        domain="CORNER",
    ).total
    bowl_bevel_chosen_directions = pf.nodes.geo.accumulate_field(
        value=bowl_bevel_dn * bowl_bevel_selected_next.astype(dtype=float),
        group_id=bowl_bevel_vertex,
        domain="CORNER",
    ).total
    bowl_bevel_cap_direction = pf.nodes.math.vector_normalize(
        bowl_bevel_all_directions
        - bowl_bevel_slide_directions
        - bowl_bevel_chosen_directions
    )
    bowl_bevel_cap_position = (
        loft_position + bowl_bevel_cap_direction * bowl_bevel_width_2
    )
    bowl_bevel_fallback = pf.nodes.func.switch(
        bowl_bevel_selected_count == 1, a=loft_position, b=bowl_bevel_cap_position
    )
    bowl_bevel_endpoint_3 = pf.nodes.func.switch(
        bowl_bevel_chosen, a=bowl_bevel_fallback, b=bowl_bevel_endpoint_2
    )
    bowl_bevel_endpoint_4 = pf.nodes.func.switch(
        bowl_bevel_selected_count == 3,
        a=bowl_bevel_endpoint_3,
        b=bowl_bevel_triple_endpoint,
    )
    bowl_bevel_cap_a = pf.nodes.geo.accumulate_field(
        value=bowl_bevel_endpoint_4 * bowl_bevel_selected_next.astype(dtype=float),
        group_id=bowl_bevel_vertex,
        domain="CORNER",
    ).total
    bowl_bevel_cap_b = pf.nodes.geo.accumulate_field(
        value=bowl_bevel_endpoint_4 * bowl_bevel_selected_prev.astype(dtype=float),
        group_id=bowl_bevel_vertex,
        domain="CORNER",
    ).total
    bowl_bevel_next_cap_dot = pf.nodes.math.vector_dot_product(
        bowl_bevel_cap_direction, bowl_bevel_dn
    )
    bowl_bevel_previous_cap_dot = pf.nodes.math.vector_dot_product(
        bowl_bevel_cap_direction, bowl_bevel_dp
    )
    bowl_bevel_next_cap = bowl_bevel_next_cap_dot > bowl_bevel_previous_cap_dot
    bowl_bevel_extra = pf.nodes.func.switch(
        bowl_bevel_next_cap, a=bowl_bevel_cap_a, b=bowl_bevel_cap_b
    )
    bowl_bevel_unchosen = pf.nodes.func.boolean_and(
        a=bowl_bevel_selected_count == 1,
        b=pf.nodes.func.boolean_not(bowl_bevel_chosen),
    )
    bowl_bevel_first = pf.nodes.func.switch(
        pf.nodes.func.boolean_and(a=bowl_bevel_unchosen, b=bowl_bevel_next_cap),
        a=bowl_bevel_endpoint_4,
        b=bowl_bevel_extra,
    )
    bowl_bevel_second = pf.nodes.func.switch(
        pf.nodes.func.boolean_and(
            a=bowl_bevel_unchosen, b=pf.nodes.func.boolean_not(bowl_bevel_next_cap)
        ),
        a=bowl_bevel_endpoint_4,
        b=bowl_bevel_extra,
    )
    bowl_bevel_face_source = pf.nodes.geo.capture_attribute(
        bevel_input, domain="CORNER", first=bowl_bevel_first, second=bowl_bevel_second
    )
    bowl_bevel_face_corners = pf.nodes.geo.corners_of_face(shell_index, sort_index=0)
    bowl_bevel_face_ids = pf.nodes.geo.capture_attribute(
        bowl_bevel_face_source.geometry,
        domain="FACE",
        corner_start=bowl_bevel_face_corners.corner_index,
        corner_count=bowl_bevel_face_corners.total,
    )
    bowl_bevel_face_points = pf.nodes.geo.mesh_to_points(
        bowl_bevel_face_ids.geometry, mode="FACES"
    )
    bowl_bevel_face_points_2 = pf.nodes.geo.set_position(
        bowl_bevel_face_points, position=pf.nodes.math.combine_xyz()
    )
    bowl_bevel_max_corners = pf.nodes.geo.attribute_statistic(
        bevel_input,
        attribute=bowl_bevel_face_corners.total.astype(dtype=float),
        domain="FACE",
    ).max
    bowl_bevel_face_template = pf.nodes.geo.mesh_circle(
        vertices=(bowl_bevel_max_corners * 2.0).astype(dtype=int), fill_type="NGON"
    )
    bowl_bevel_face_template_2 = pf.nodes.geo.capture_attribute(
        bowl_bevel_face_template, domain="POINT", local=shell_index
    )
    bowl_bevel_separate = pf.nodes.geo.realize_instances(
        pf.nodes.geo.instance_on_points(
            bowl_bevel_face_points_2, instance=bowl_bevel_face_template_2.geometry
        )
    )
    bowl_bevel_face_start = bowl_bevel_face_ids.attributes["corner_start"]
    bowl_bevel_local_index = bowl_bevel_face_template_2.attributes["local"].astype(
        dtype=float
    )
    bowl_bevel_corner_offset_3 = pf.nodes.math.minimum(
        pf.nodes.math.floor(bowl_bevel_local_index / 2.0),
        bowl_bevel_face_ids.attributes["corner_count"].astype(dtype=float) - 1.0,
    )
    bowl_bevel_face_corner = (
        bowl_bevel_face_start.astype(dtype=float) + bowl_bevel_corner_offset_3
    ).astype(dtype=int)
    bowl_bevel_sample_first = pf.nodes.geo.sample_index(
        bowl_bevel_face_source.geometry,
        index=bowl_bevel_face_corner,
        value=bowl_bevel_face_source.attributes["first"],
        domain="CORNER",
    )
    bowl_bevel_sample_second = pf.nodes.geo.sample_index(
        bowl_bevel_face_source.geometry,
        index=bowl_bevel_face_corner,
        value=bowl_bevel_face_source.attributes["second"],
        domain="CORNER",
    )
    bowl_bevel_face_position = pf.nodes.func.switch(
        pf.nodes.math.modulo(bowl_bevel_local_index, 2.0) > 0.5,
        a=bowl_bevel_sample_first,
        b=bowl_bevel_sample_second,
    )
    bowl_bevel_separate_2 = pf.nodes.geo.set_position(
        bowl_bevel_separate, position=bowl_bevel_face_position
    )
    bowl_bevel_cap_data = pf.nodes.geo.capture_attribute(
        bevel_input,
        domain="CORNER",
        a=bowl_bevel_cap_a,
        b=bowl_bevel_cap_b,
        cap=bowl_bevel_cap_position,
        count=bowl_bevel_selected_count,
    )
    bowl_bevel_cap_data_2 = pf.nodes.geo.capture_attribute(
        bowl_bevel_cap_data.geometry,
        domain="POINT",
        origin=loft_position,
        a=bowl_bevel_cap_data.attributes["a"],
        b=bowl_bevel_cap_data.attributes["b"],
        cap=bowl_bevel_cap_data.attributes["cap"],
        count=bowl_bevel_cap_data.attributes["count"],
    )
    bowl_bevel_cap_points = pf.nodes.geo.mesh_to_points(
        bowl_bevel_cap_data_2.geometry,
        selection=bowl_bevel_cap_data_2.attributes["count"] == 1,
        mode="VERTICES",
    )
    bowl_bevel_cap_points_2 = pf.nodes.geo.set_position(
        bowl_bevel_cap_points, position=pf.nodes.math.combine_xyz()
    )
    bowl_bevel_cap_grid = pf.nodes.geo.mesh_grid(vertices_x=3, vertices_y=2).mesh
    bowl_bevel_caps = pf.nodes.geo.realize_instances(
        pf.nodes.geo.instance_on_points(
            bowl_bevel_cap_points_2, instance=bowl_bevel_cap_grid
        )
    )
    bowl_bevel_cap_mid = (
        bowl_bevel_cap_data_2.attributes["origin"]
        + (
            bowl_bevel_cap_data_2.attributes["a"]
            + bowl_bevel_cap_data_2.attributes["b"]
            - bowl_bevel_cap_data_2.attributes["origin"] * 2.0
        )
        * 0.2928932188134524
    )
    bowl_bevel_cap_mid_2 = pf.nodes.func.switch(
        bevel_segments < 2.0,
        a=bowl_bevel_cap_mid,
        b=bowl_bevel_cap_data_2.attributes["a"],
    )
    bowl_bevel_cap_target = pf.nodes.func.switch(
        loft_position.x < -0.1,
        a=bowl_bevel_cap_mid_2,
        b=bowl_bevel_cap_data_2.attributes["a"],
    )
    bowl_bevel_cap_target_2 = pf.nodes.func.switch(
        loft_position.x > 0.1,
        a=bowl_bevel_cap_target,
        b=bowl_bevel_cap_data_2.attributes["b"],
    )
    bowl_bevel_cap_target_3 = pf.nodes.func.switch(
        loft_position.y > 0.0,
        a=bowl_bevel_cap_target_2,
        b=bowl_bevel_cap_data_2.attributes["cap"],
    )
    bowl_bevel_caps_2 = pf.nodes.geo.set_position(
        bowl_bevel_caps, position=bowl_bevel_cap_target_3
    )
    bowl_bevel_caps_3 = pf.nodes.geo.flip_faces(bowl_bevel_caps_2)
    bowl_bevel_captured = pf.nodes.geo.capture_attribute(
        bevel_input,
        domain="CORNER",
        endpoint=bowl_bevel_endpoint_4,
        count=bowl_bevel_selected_count,
    )
    bowl_bevel_original = bowl_bevel_captured.geometry
    bowl_bevel_endpoint_5 = bowl_bevel_captured.attributes["endpoint"]
    bowl_bevel_triple_edge = pf.nodes.geo.edges_of_vertex(
        shell_index, weights=bowl_bevel_chosen_weight, sort_index=0
    ).edge_index
    bowl_bevel_triple_ca = pf.nodes.geo.corners_of_edge(
        bowl_bevel_triple_edge, sort_index=0
    ).corner_index
    bowl_bevel_triple_cb = pf.nodes.geo.corners_of_edge(
        bowl_bevel_triple_edge, sort_index=1
    ).corner_index
    bowl_bevel_triple_ca_2 = pf.nodes.geo.offset_corner_in_face(
        bowl_bevel_triple_ca,
        offset=(
            pf.nodes.geo.vertex_of_corner(bowl_bevel_triple_ca) != shell_index
        ).astype(dtype=int),
    )
    bowl_bevel_triple_cb_2 = pf.nodes.geo.offset_corner_in_face(
        bowl_bevel_triple_cb,
        offset=(
            pf.nodes.geo.vertex_of_corner(bowl_bevel_triple_cb) != shell_index
        ).astype(dtype=int),
    )
    bowl_bevel_triple_a = pf.nodes.geo.field_at_index(
        value=bowl_bevel_endpoint_5, index=bowl_bevel_triple_ca_2, domain="CORNER"
    )
    bowl_bevel_triple_b = pf.nodes.geo.field_at_index(
        value=bowl_bevel_endpoint_5, index=bowl_bevel_triple_cb_2, domain="CORNER"
    )
    bowl_bevel_triple_ab = pf.nodes.geo.capture_attribute(
        bowl_bevel_original,
        domain="POINT",
        a=bowl_bevel_triple_a,
        b=bowl_bevel_triple_b,
    )
    bowl_bevel_third_distance = pf.nodes.math.minimum(
        pf.nodes.math.vector_length(
            bowl_bevel_endpoint_5 - bowl_bevel_triple_ab.attributes["a"]
        ),
        pf.nodes.math.vector_length(
            bowl_bevel_endpoint_5 - bowl_bevel_triple_ab.attributes["b"]
        ),
    )
    bowl_bevel_triple_c = pf.nodes.geo.field_at_index(
        value=bowl_bevel_endpoint_5,
        index=pf.nodes.geo.corners_of_vertex(
            shell_index, weights=-bowl_bevel_third_distance, sort_index=0
        ).corner_index,
        domain="CORNER",
    )
    bowl_bevel_triples_data = pf.nodes.geo.capture_attribute(
        bowl_bevel_triple_ab.geometry,
        domain="POINT",
        a=bowl_bevel_triple_ab.attributes["a"],
        b=bowl_bevel_triple_ab.attributes["b"],
        c=bowl_bevel_triple_c,
        normal=pf.nodes.geo.input_normal(),
        count=bowl_bevel_captured.attributes["count"],
    )
    bowl_bevel_triple_points = pf.nodes.geo.mesh_to_points(
        bowl_bevel_triples_data.geometry,
        selection=bowl_bevel_triples_data.attributes["count"] == 3,
        mode="VERTICES",
    )
    bowl_bevel_triple_points_2 = pf.nodes.geo.set_position(
        bowl_bevel_triple_points, position=pf.nodes.math.combine_xyz()
    )
    bowl_bevel_triangle = pf.nodes.geo.capture_attribute(
        pf.nodes.geo.mesh_circle(vertices=3, fill_type="NGON"),
        domain="POINT",
        local=shell_index,
    )
    bowl_bevel_triples = pf.nodes.geo.realize_instances(
        pf.nodes.geo.instance_on_points(
            bowl_bevel_triple_points_2, instance=bowl_bevel_triangle.geometry
        )
    )
    bowl_bevel_triple_target = pf.nodes.func.switch(
        bowl_bevel_triangle.attributes["local"] > 0,
        a=bowl_bevel_triples_data.attributes["a"],
        b=bowl_bevel_triples_data.attributes["b"],
    )
    bowl_bevel_triple_target_2 = pf.nodes.func.switch(
        bowl_bevel_triangle.attributes["local"] > 1,
        a=bowl_bevel_triple_target,
        b=bowl_bevel_triples_data.attributes["c"],
    )
    bowl_bevel_triples_2 = pf.nodes.geo.set_position(
        bowl_bevel_triples, position=bowl_bevel_triple_target_2
    )
    bowl_bevel_triple_normal = pf.nodes.math.vector_cross_product(
        bowl_bevel_triples_data.attributes["b"]
        - bowl_bevel_triples_data.attributes["a"],
        bowl_bevel_triples_data.attributes["c"]
        - bowl_bevel_triples_data.attributes["a"],
    )
    bowl_bevel_triple_dot = pf.nodes.math.vector_dot_product(
        bowl_bevel_triple_normal, bowl_bevel_triples_data.attributes["normal"]
    )
    bowl_bevel_triples_3 = pf.nodes.geo.flip_faces(
        bowl_bevel_triples_2,
        selection=bowl_bevel_triple_dot < 0.0,
    )
    bowl_bevel_c0 = pf.nodes.geo.corners_of_edge(shell_index, sort_index=0).corner_index
    bowl_bevel_c1 = pf.nodes.geo.corners_of_edge(shell_index, sort_index=1).corner_index
    bowl_bevel_c0next = pf.nodes.geo.offset_corner_in_face(bowl_bevel_c0, offset=1)
    bowl_bevel_c1next = pf.nodes.geo.offset_corner_in_face(bowl_bevel_c1, offset=1)
    bowl_bevel_a0 = pf.nodes.geo.field_at_index(
        value=bowl_bevel_endpoint_5, index=bowl_bevel_c0, domain="CORNER"
    )
    bowl_bevel_a1 = pf.nodes.geo.field_at_index(
        value=bowl_bevel_endpoint_5, index=bowl_bevel_c0next, domain="CORNER"
    )
    bowl_bevel_b0 = pf.nodes.geo.field_at_index(
        value=bowl_bevel_endpoint_5, index=bowl_bevel_c1next, domain="CORNER"
    )
    bowl_bevel_b1 = pf.nodes.geo.field_at_index(
        value=bowl_bevel_endpoint_5, index=bowl_bevel_c1, domain="CORNER"
    )
    bowl_bevel_p0 = pf.nodes.geo.field_at_index(
        value=loft_position, index=bowl_bevel_c0, domain="CORNER"
    )
    bowl_bevel_p1 = pf.nodes.geo.field_at_index(
        value=loft_position, index=bowl_bevel_c0next, domain="CORNER"
    )
    bowl_bevel_captured_2 = pf.nodes.geo.capture_attribute(
        bowl_bevel_original,
        domain="EDGE",
        a0=bowl_bevel_a0,
        a1=bowl_bevel_a1,
        b0=bowl_bevel_b0,
        b1=bowl_bevel_b1,
        p0=bowl_bevel_p0,
        p1=bowl_bevel_p1,
    )
    bowl_bevel_points = pf.nodes.geo.mesh_to_points(
        bowl_bevel_captured_2.geometry,
        position=(0.0, 0.0, 0.0),
        selection=bowl_bevel_selected,
        mode="EDGES",
    )
    bowl_bevel_points_2 = pf.nodes.geo.set_position(
        bowl_bevel_points, position=pf.nodes.math.combine_xyz(x=0.0, y=0.0, z=0.0)
    )
    bowl_bevel_instance = pf.nodes.geo.instance_on_points(
        bowl_bevel_points_2, instance=bowl_bevel_cap_grid
    )
    bowl_bevel_patches = pf.nodes.geo.realize_instances(bowl_bevel_instance)
    bowl_bevel_along = loft_position.y > 0.0
    bowl_bevel_a = pf.nodes.func.switch(
        bowl_bevel_along,
        a=bowl_bevel_captured_2.attributes["a0"],
        b=bowl_bevel_captured_2.attributes["a1"],
    )
    bowl_bevel_b = pf.nodes.func.switch(
        bowl_bevel_along,
        a=bowl_bevel_captured_2.attributes["b0"],
        b=bowl_bevel_captured_2.attributes["b1"],
    )
    bowl_bevel_p = pf.nodes.func.switch(
        bowl_bevel_along,
        a=bowl_bevel_captured_2.attributes["p0"],
        b=bowl_bevel_captured_2.attributes["p1"],
    )
    bowl_bevel_mid = (
        bowl_bevel_p
        + (bowl_bevel_a + bowl_bevel_b - bowl_bevel_p * 2.0) * 0.2928932188134524
    )
    bowl_bevel_mid_2 = pf.nodes.func.switch(
        bevel_segments < 2.0, a=bowl_bevel_mid, b=bowl_bevel_a
    )
    bowl_bevel_target = pf.nodes.func.switch(
        loft_position.x < -0.1, a=bowl_bevel_mid_2, b=bowl_bevel_a
    )
    bowl_bevel_target_2 = pf.nodes.func.switch(
        loft_position.x > 0.1, a=bowl_bevel_target, b=bowl_bevel_b
    )
    bowl_bevel_patches_2 = pf.nodes.geo.set_position(
        bowl_bevel_patches, position=bowl_bevel_target_2
    )
    bowl_bevel_result = pf.nodes.geo.join_geometry(
        [
            bowl_bevel_separate_2,
            bowl_bevel_patches_2,
            bowl_bevel_caps_3,
            bowl_bevel_triples_3,
        ]
    )
    bowl_bevel_result_2 = pf.nodes.geo.merge_by_distance(
        bowl_bevel_result, selection=bevel_part == 0.0, distance=1e-06
    )
    bowl_bevel_result_3 = pf.nodes.geo.merge_by_distance(
        bowl_bevel_result_2, selection=bevel_part == 1.0, distance=1e-06
    )
    bowl_bevel_result_4 = pf.nodes.geo.merge_by_distance(
        bowl_bevel_result_3, selection=bevel_part == 2.0, distance=1e-06
    )
    bowl_material = pf.nodes.geo.set_material(bowl_bevel_result_4, material)
    is_seat = pf.nodes.func.boolean_or(a=bevel_part == 1.0, b=bevel_part == 2.0)
    bowl_result = pf.nodes.geo.set_material(
        bowl_material, seat_material, selection=is_seat
    )
    stand_material = pf.nodes.geo.set_material(stand_result, material)
    back_material = pf.nodes.geo.set_material(back_result, material)
    geometry = pf.nodes.geo.join_geometry([bowl_result, stand_material])
    geometry_2 = pf.nodes.geo.subdivision_surface(geometry, level=1)
    limit_f = pf.nodes.geo.capture_attribute(
        geometry_2, domain="FACE", center=loft_position
    )
    limit_boundary = (pf.nodes.geo.input_mesh_edge_neighbors() < 2).astype(float)
    limit_e = pf.nodes.geo.capture_attribute(
        limit_f.geometry,
        domain="EDGE",
        midpoint=loft_position,
        boundary=limit_boundary,
        boundary_midpoint=loft_position * limit_boundary,
    )
    limit_n = pf.nodes.geo.input_mesh_vertex_neighbors().vertex_count.astype(float)
    limit_fa = pf.nodes.geo.field_on_domain(
        limit_f.attributes["center"], domain="POINT"
    )
    limit_ra = pf.nodes.geo.field_on_domain(
        limit_e.attributes["midpoint"], domain="POINT"
    )
    limit_vp = (limit_fa * 4.0 + limit_ra * 4.0 + loft_position * (limit_n - 3.0)) * (
        1.0 / (limit_n + 5.0)
    )
    limit_bw = pf.nodes.geo.field_on_domain(
        limit_e.attributes["boundary"], domain="POINT"
    )
    limit_bm = pf.nodes.geo.field_on_domain(
        limit_e.attributes["boundary_midpoint"], domain="POINT"
    )
    limit_boundary_position = (loft_position + limit_bm * (2.0 / limit_bw)) * (
        1.0 / 3.0
    )
    limit_vp_2 = pf.nodes.func.switch(limit_bw > 0.0, limit_vp, limit_boundary_position)
    limit_result = pf.nodes.geo.set_position(limit_e.geometry, position=limit_vp_2)
    bowl_with_back = pf.nodes.geo.join_geometry([limit_result, back_material])
    return pf.nodes.geo.set_shade_smooth(bowl_with_back)


@pf.nodes.node_function
def _toilet_button(
    hardware_material: t.SocketOrVal[pf.Material],
    tank_height: t.SocketOrVal[float],
    tank_cap_height: t.SocketOrVal[float],
    hardware_radius: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    button_result = pf.nodes.geo.mesh_cylinder(
        vertices=32,
        radius=hardware_radius,
        depth=tank_cap_height * 0.5 + 0.001,
        fill_type="NGON",
    )
    button_height = tank_height + tank_cap_height * 0.25 + 0.0005
    button_2 = pf.nodes.geo.transform(
        button_result.mesh, translation=pf.nodes.math.combine_xyz(z=button_height)
    )
    return pf.nodes.geo.set_material(button_2, material=hardware_material)


@pf.nodes.node_function
def _toilet_handle(
    hardware_material: t.SocketOrVal[pf.Material],
    tank_width: t.SocketOrVal[float],
    tank_size: t.SocketOrVal[float],
    tank_height: t.SocketOrVal[float],
    hardware_on_side: t.SocketOrVal[bool],
    hardware_radius: t.SocketOrVal[float],
    hardware_cap: t.SocketOrVal[float],
    hardware_length: t.SocketOrVal[float],
    handle_bevel_width: t.SocketOrVal[float],
    handle_lever_x_factor: t.SocketOrVal[float],
    handle_lever_z_factor: t.SocketOrVal[float],
    handle_mount_offset: t.SocketOrVal[float],
    handle_height_offset: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    p = pf.nodes.geo.input_position()
    az = pf.nodes.math.absolute(p.z)
    handles = pf.nodes.geo.mesh_line(
        start_location=(0, 0, 0), offset=(0, 0, 0), count=2
    )
    handle_id = pf.nodes.geo.capture_attribute(
        handles, domain="POINT", lever=pf.nodes.geo.input_index() > 0
    )
    cylinder_result = pf.nodes.geo.mesh_cylinder(
        vertices=16, side_segments=5, radius=1.0, depth=2.0, fill_type="NGON"
    )
    hardware_instances = pf.nodes.geo.instance_on_points(
        handle_id.geometry, cylinder_result.mesh
    )
    hardware = pf.nodes.geo.realize_instances(hardware_instances)
    lever = handle_id.attributes["lever"]
    radius = pf.nodes.func.switch(lever, hardware_radius, hardware_radius * 0.5)
    length = pf.nodes.func.switch(lever, hardware_cap, hardware_length)
    hb = pf.nodes.math.minimum(
        handle_bevel_width,
        pf.nodes.math.minimum(hardware_cap * 0.5, hardware_radius * 0.5 * 0.9951847267),
    )
    inset = pf.nodes.func.switch(az > 0.4, 0.0, 0.2928932188)
    inset_2 = pf.nodes.func.switch(az > 0.8, inset, 1.0)
    radial = radius - hb * inset_2 / 0.9951847267
    height_inset = pf.nodes.func.switch(az > 0.4, 1.0, 0.2928932188)
    height_inset_2 = pf.nodes.func.switch(az > 0.8, height_inset, 0.0)
    hz = (length * 0.5 - hb * height_inset_2) * pf.nodes.math.sign(p.z) + length * 0.5
    hx = p.x * radial
    hy = p.y * radial
    mount_pos = pf.nodes.math.combine_xyz(hx, -hz, hy)
    lever_pos = pf.nodes.math.combine_xyz(
        hz - hardware_radius * handle_lever_x_factor,
        hy - hardware_cap,
        -hx - hardware_radius * handle_lever_z_factor,
    )
    handle_pos = pf.nodes.func.switch(lever, mount_pos, lever_pos)
    front_pos = pf.nodes.math.combine_xyz(handle_pos.y, -handle_pos.x, handle_pos.z)
    handle_pos_2 = pf.nodes.func.switch(hardware_on_side, front_pos, handle_pos)
    handle_x = pf.nodes.func.switch(
        hardware_on_side,
        -tank_width * 0.5,
        -tank_width * 0.5 + hardware_radius + handle_mount_offset,
    )
    handle_y = pf.nodes.func.switch(
        hardware_on_side,
        -tank_size * 0.5 + hardware_radius + handle_mount_offset,
        -tank_size * 0.5,
    )
    handle_origin = pf.nodes.math.combine_xyz(
        handle_x, handle_y, tank_height - hardware_radius - handle_height_offset
    )
    hardware_2 = pf.nodes.geo.set_position(
        hardware, position=handle_pos_2 + handle_origin
    )
    return pf.nodes.geo.set_material(hardware_2, material=hardware_material)


@pf.nodes.node_function
def _toilet_tank(
    material: t.SocketOrVal[pf.Material],
    hardware: t.SocketOrVal[pf.MeshObject],
    back_length: t.SocketOrVal[float],
    back_size: t.SocketOrVal[float],
    tank_width: t.SocketOrVal[float],
    tank_size: t.SocketOrVal[float],
    tank_height: t.SocketOrVal[float],
    tank_cap_height: t.SocketOrVal[float],
    tank_cap_extrude: t.SocketOrVal[float],
    tank_cap_bevel_width: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    cube = pf.nodes.geo.mesh_cube(
        size=(6, 6, 6), vertices_x=7, vertices_y=7, vertices_z=7
    ).mesh
    p = pf.nodes.geo.input_position()
    x, y, z = (p.x, p.y, p.z)
    ax = pf.nodes.math.absolute(x)
    ay = pf.nodes.math.absolute(y)
    az = pf.nodes.math.absolute(z)
    ex = ax > 2.5
    ey = ay > 2.5
    ez = az > 2.5
    count = ex.astype(float) + ey.astype(float) + ez.astype(float)
    bevel = pf.nodes.math.minimum(
        0.1,
        pf.nodes.math.minimum(tank_width, pf.nodes.math.minimum(tank_size, tank_height))
        / 4.0,
    )
    offset = bevel / pf.nodes.math.sqrt(count)
    px = pf.nodes.func.switch(ax > 1.5, ax * tank_width / 4.0, tank_width / 2.0 - bevel)
    py = pf.nodes.func.switch(ay > 1.5, ay * tank_size / 4.0, tank_size / 2.0 - bevel)
    pz = pf.nodes.func.switch(
        az > 1.5, az * tank_height / 4.0, tank_height / 2.0 - bevel
    )
    px_2 = (px + ex.astype(float) * offset) * pf.nodes.math.sign(x)
    py_2 = (py + ey.astype(float) * offset) * pf.nodes.math.sign(y)
    pz_2 = (pz + ez.astype(float) * offset) * pf.nodes.math.sign(z)
    mesh = pf.nodes.geo.set_position(
        cube, position=pf.nodes.math.combine_xyz(px_2, py_2, pz_2)
    )
    tank = pf.nodes.geo.transform(
        mesh, translation=pf.nodes.math.combine_xyz(z=tank_height * 0.5)
    )
    cap_cube = pf.nodes.geo.mesh_cube(
        size=(5, 5, 5), vertices_x=6, vertices_y=6, vertices_z=6
    ).mesh
    cx = (ax > 2.0).astype(float)
    cy = (ay > 2.0).astype(float)
    cz = (az > 2.0).astype(float)
    mx = (ax > 1.0).astype(float) * (ax < 2.0).astype(float)
    my = (ay > 1.0).astype(float) * (ay < 2.0).astype(float)
    mz = (az > 1.0).astype(float) * (az < 2.0).astype(float)
    nc = cx + cy + cz
    nm = mx + my + mz
    cc = pf.nodes.func.switch(nm > 0.5, 1.0, 0.9238795325)
    cc_2 = pf.nodes.func.switch(nm > 1.5, cc, 0.832011)
    c2 = pf.nodes.func.switch(nm > 0.5, 0.7071067812, 0.656254)
    cc_3 = pf.nodes.func.switch(nc > 1.5, cc_2, c2)
    cc_4 = pf.nodes.func.switch(nc > 2.5, cc_3, 0.5773502692)
    mc = pf.nodes.func.switch(nm > 1.5, 0.3826834324, 0.392274)
    mc_2 = pf.nodes.func.switch(nc > 1.5, mc, 0.37237)
    cb = pf.nodes.math.minimum(tank_cap_bevel_width, tank_cap_height * 0.5)
    cpx = (
        tank_width * 0.5 + tank_cap_extrude - cb + cb * (cx * cc_4 + mx * mc_2)
    ) * pf.nodes.math.sign(x)
    cpy = (
        tank_size * 0.5 + tank_cap_extrude - cb + cb * (cy * cc_4 + my * mc_2)
    ) * pf.nodes.math.sign(y)
    cpz = (
        tank_cap_height * 0.5 - cb + cb * (cz * cc_4 + mz * mc_2)
    ) * pf.nodes.math.sign(z)
    cap_mesh = pf.nodes.geo.set_position(
        cap_cube, position=pf.nodes.math.combine_xyz(cpx, cpy, cpz)
    )
    cap = pf.nodes.geo.transform(
        cap_mesh, translation=pf.nodes.math.combine_xyz(z=tank_height)
    )
    tank_3 = pf.nodes.geo.set_material(tank, material=material)
    cap_3 = pf.nodes.geo.set_material(cap, material=material)
    mesh_2 = pf.nodes.geo.join_geometry([tank_3, cap_3, hardware])
    mesh_3 = pf.nodes.geo.transform(
        mesh_2,
        translation=pf.nodes.math.combine_xyz(
            y=back_length + back_size - tank_size * 0.5
        ),
    )
    mesh_4 = pf.nodes.geo.transform(mesh_3, rotation=(0.0, 0.0, 1.5707963267948966))
    mesh_5 = pf.nodes.geo.subdivision_surface(mesh_4, level=1)
    f = pf.nodes.geo.capture_attribute(mesh_5, domain="FACE", center=p)
    e = pf.nodes.geo.capture_attribute(f.geometry, domain="EDGE", midpoint=p)
    n = pf.nodes.geo.input_mesh_vertex_neighbors().vertex_count.astype(float)
    fa = pf.nodes.geo.field_on_domain(f.attributes["center"], domain="POINT")
    ra = pf.nodes.geo.field_on_domain(e.attributes["midpoint"], domain="POINT")
    vp = (fa * 4.0 + ra * 4.0 + p * (n - 3.0)) * (1.0 / (n + 5.0))
    mesh_6 = pf.nodes.geo.set_position(e.geometry, position=vp)
    return pf.nodes.geo.set_shade_smooth(mesh_6)


def _validate(
    hardware_type: ToiletHardwareType, curve_scale: tuple[float, float, float, float]
) -> None:
    if hardware_type not in {"button", "handle"}:
        raise ValueError(f"Unknown toilet hardware type {hardware_type!r}")
    if len(curve_scale) != 4 or min(curve_scale) <= 0.0:
        raise ValueError("curve_scale must contain four positive values")


@pf.tracer.generator
def toilet(
    material: pf.Material,
    seat_material: pf.Material,
    hardware_material: pf.Material,
    size: float,
    width: float,
    height: float,
    size_mid: float,
    curve_scale: tuple[float, float, float, float],
    depth: float,
    tube_scale: float,
    thickness: float,
    extrude_height: float,
    stand_depth: float,
    stand_scale: float,
    bottom_offset: float,
    back_thickness: float,
    back_size: float,
    back_scale: float,
    seat_thickness: float,
    seat_size: float,
    has_seat_cut: bool,
    tank_width: float,
    tank_height: float,
    tank_size: float,
    tank_cap_height: float,
    tank_cap_extrude: float,
    cover_rotation: float,
    hardware_type: ToiletHardwareType,
    hardware_cap: float,
    hardware_radius: float,
    hardware_length: float,
    hardware_on_side: bool,
    tube_bridge_profile_factor: float,
    stand_bridge_profile_factor: float,
    tank_cap_bevel_width: float,
    handle_lever_x_factor: float,
    handle_lever_z_factor: float,
    handle_mount_offset: float,
    handle_height_offset: float,
    handle_bevel_width: float,
) -> ToiletResult:
    _validate(hardware_type, curve_scale)
    bowl = _toilet_bowl(
        material=material,
        seat_material=seat_material,
        size=size,
        width=width,
        tank_width=tank_width,
        height=height,
        size_mid=size_mid,
        depth=depth,
        tube_scale=tube_scale,
        thickness=thickness,
        extrude_height=extrude_height,
        stand_depth=stand_depth,
        stand_scale=stand_scale,
        bottom_offset=bottom_offset,
        back_thickness=back_thickness,
        back_size=back_size,
        back_scale=back_scale,
        seat_thickness=seat_thickness,
        seat_size=seat_size,
        has_seat_cut=has_seat_cut,
        cover_rotation=cover_rotation,
        tube_bridge_profile_factor=tube_bridge_profile_factor,
        stand_bridge_profile_factor=stand_bridge_profile_factor,
        curve_front=curve_scale[0],
        curve_front_side=curve_scale[1],
        curve_back_side=curve_scale[2],
        curve_back=curve_scale[3],
    )
    if hardware_type == "handle":
        hardware = _toilet_handle(
            hardware_material=hardware_material,
            tank_width=tank_width,
            tank_size=tank_size,
            tank_height=tank_height,
            hardware_on_side=hardware_on_side,
            hardware_radius=hardware_radius,
            hardware_cap=hardware_cap,
            hardware_length=hardware_length,
            handle_bevel_width=handle_bevel_width,
            handle_lever_x_factor=handle_lever_x_factor,
            handle_lever_z_factor=handle_lever_z_factor,
            handle_mount_offset=handle_mount_offset,
            handle_height_offset=handle_height_offset,
        )
    else:
        hardware = _toilet_button(
            hardware_material=hardware_material,
            tank_height=tank_height,
            tank_cap_height=tank_cap_height,
            hardware_radius=hardware_radius,
        )
    tank = _toilet_tank(
        material=material,
        hardware=hardware,
        back_length=size * (1.0 - size_mid),
        back_size=back_size,
        tank_width=tank_width,
        tank_size=tank_size,
        tank_height=tank_height,
        tank_cap_height=tank_cap_height,
        tank_cap_extrude=tank_cap_extrude,
        tank_cap_bevel_width=tank_cap_bevel_width,
    )
    geometry = pf.nodes.geo.join_geometry([bowl, tank])
    geometry = pf.nodes.geo.transform(
        geometry, translation=pf.nodes.math.combine_xyz(z=height)
    )
    obj = pf.nodes.to_mesh_object(geometry)
    obj.item().name = toilet.__name__
    return ToiletResult(mesh=obj)


def toilet_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
    seat_material: pf.Material | None = None,
    hardware_material: pf.Material | None = None,
    size: float | None = None,
    width: float | None = None,
    height: float | None = None,
    hardware_type: ToiletHardwareType | None = None,
    has_seat_cut: bool | None = None,
    hardware_on_side: bool | None = None,
    tank_cap_extrude: float | None = None,
    cover_rotation: float | None = None,
) -> ToiletResult:
    (
        rng_size,
        rng_shape,
        rng_hardware_type,
        rng_hardware_side,
        rng_seat_cut,
        rng_cap,
        rng_body,
        rng_seat,
        rng_hardware,
    ) = rng.spawn(9)
    if size is None:
        size = pf.random.uniform(rng_size, 0.4, 0.5)
    if width is None:
        width = size * pf.random.uniform(rng_shape, 0.7, 0.8)
    if height is None:
        height = size * pf.random.uniform(rng_shape, 0.8, 0.9)
    if cover_rotation is None:
        cover_rotation = 0.0
    if hardware_type is None:
        hardware_type = pf.control.choice(
            rng_hardware_type, [("button", 1.0), ("handle", 1.0)]
        )
    if hardware_on_side is None:
        hardware_on_side = pf.control.choice(
            rng_hardware_side, [(True, 1.0), (False, 1.0)]
        )
    if has_seat_cut is None:
        has_seat_cut = pf.control.choice(rng_seat_cut, [(True, 1.0), (False, 9.0)])
    if tank_cap_extrude is None:
        tank_cap_extrude = pf.control.choice(rng_cap, [(0.0, 1.0), (0.0075, 1.0)])
    size_mid = pf.random.uniform(rng_shape, 0.55, 0.7)
    curve = pf.random.log_uniform(rng_shape, 0.75, 1.3, size=(4,))
    depth = size * pf.random.uniform(rng_shape, 0.5, 0.6)
    tube_scale = pf.random.uniform(rng_shape, 0.22, 0.38)
    thickness = pf.random.uniform(rng_shape, 0.05, 0.06)
    extrude_height = pf.random.uniform(rng_shape, 0.015, 0.02)
    stand_scale = pf.random.uniform(rng_shape, 0.6, 0.92)
    bottom_offset = pf.random.uniform(rng_shape, 0.5, 1.5)
    back_size = size * pf.random.uniform(rng_shape, 0.5, 0.7)
    seat_size = thickness * 1.4
    tank_width = width * pf.random.uniform(rng_shape, 0.95, 1.25)
    tank_height = height * pf.random.uniform(rng_shape, 0.55, 1.05)
    tank_cap_height = pf.random.uniform(rng_shape, 0.025, 0.05)
    back_scale = pf.random.uniform(rng_shape, 0.8, 1.0)
    vector = pf.nodes.shader.coord().object
    if material is None:
        material = ceramic.ceramic_rand(rng_body, vector)
    if seat_material is None:
        seat_material = plastic.plastic_grayscale_rand(rng_seat, vector)
    if hardware_material is None:
        hardware_material = metal_brushed.metal_brushed_linear_rand(
            rng_hardware, vector
        )
    return toilet(
        material=material,
        seat_material=seat_material,
        hardware_material=hardware_material,
        size=size,
        width=width,
        height=height,
        size_mid=size_mid,
        curve_scale=(curve[0], curve[1], curve[2], curve[3]),
        depth=depth,
        tube_scale=tube_scale,
        thickness=thickness,
        extrude_height=extrude_height,
        stand_depth=depth * 0.9,
        stand_scale=stand_scale,
        bottom_offset=bottom_offset,
        back_thickness=thickness * 0.4,
        back_size=back_size,
        back_scale=back_scale,
        seat_thickness=thickness * 0.2,
        seat_size=seat_size,
        has_seat_cut=has_seat_cut,
        tank_width=tank_width,
        tank_height=tank_height,
        tank_size=back_size - seat_size - 0.025,
        tank_cap_height=tank_cap_height,
        tank_cap_extrude=tank_cap_extrude,
        cover_rotation=cover_rotation,
        hardware_type=hardware_type,
        hardware_cap=0.0125,
        hardware_radius=0.0175,
        hardware_length=0.045,
        hardware_on_side=hardware_on_side,
        tube_bridge_profile_factor=0.15,
        stand_bridge_profile_factor=0.075,
        tank_cap_bevel_width=0.00875,
        handle_lever_x_factor=0.25,
        handle_lever_z_factor=0.25,
        handle_mount_offset=0.015,
        handle_height_offset=0.025,
        handle_bevel_width=0.0075,
    )
