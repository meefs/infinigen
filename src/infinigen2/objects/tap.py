# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: transpiled from Infinigen v1 (Stamatis Alexandropoulos), ported to v2

from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t
from procfunc.nodes.util.bpy_node_info import NodeDataType

from infinigen2.shaders.functionality_lists import decorative_material_rand
from infinigen2.util.curve import curve_to_mesh_with_uv
from infinigen2.util.mesh import metric_box_uv

__all__ = ["TapResult", "tap", "tap_rand"]


class TapResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _lever_handle() -> pf.ProcNode[pf.MeshObject]:
    curve = pf.nodes.geo.curve_bezier_segment(
        start=(0.0, 0.0, 0.0),
        start_handle=(0.0, 0.0, 0.28),
        end_handle=(0.08, 0.0, 0.28),
        end=(0.4, 0.0, 0.36),
    )
    spline_parameter = pf.nodes.geo.spline_parameter()
    radius = pf.nodes.math.float_curve(
        factor=1.0,
        value=spline_parameter.factor,
        curve=np.array([[0.0, 0.975], [1.0, 0.1625]], dtype=np.float64),
    )
    curve = pf.nodes.geo.set_curve_radius(curve=curve, radius=radius * 1.3)
    profile = pf.nodes.geo.curve_circle(radius=0.08)
    handle = curve_to_mesh_with_uv(
        curve=curve,
        profile=profile,
        fill_caps=True,
    ).mesh
    position = pf.nodes.geo.input_position()
    y_scale = pf.nodes.math.map_range(
        value=position.x,
        from_min=0.08,
        from_max=0.4,
        to_max=2.5,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )
    position = pf.nodes.math.combine_xyz(
        x=position.x,
        y=position.y * y_scale,
        z=position.z,
    )
    handle = pf.nodes.geo.set_position(
        geometry=handle,
        position=position,
        offset=(0.0, 0.0, 0.0),
    )
    handle = pf.nodes.geo.subdivision_surface(mesh=handle, level=2)
    return pf.nodes.geo.set_shade_smooth(handle)


@pf.nodes.node_function
def _tap_geometry(
    material: t.SocketOrVal[pf.Material],
    base_width: t.SocketOrVal[float] = 0.1,
    tap_head: t.SocketOrVal[float] = 0.9,
    rotation_z: t.SocketOrVal[float] = 6.25,
    tap_height: t.SocketOrVal[float] = 0.75,
    base_radius: t.SocketOrVal[float] = 0.02,
    switch: t.SocketOrVal[bool] = False,
    curl: t.SocketOrVal[float] = -0.112,
    hand_type: t.SocketOrVal[bool] = True,
    hands_length_x: t.SocketOrVal[float] = 1.0,
    hands_length_y: t.SocketOrVal[float] = 1.25,
    one_side: t.SocketOrVal[bool] = False,
    different_type: t.SocketOrVal[bool] = False,
    length_one_side: t.SocketOrVal[bool] = False,
) -> pf.ProcNode[pf.MeshObject]:
    base_curve = pf.nodes.geo.curve_quadrilateral(width=base_width, height=0.28)
    base_curve = pf.nodes.geo.fillet_curve_poly(
        curve=base_curve,
        radius=base_radius,
        count=19,
    )
    base = pf.nodes.geo.fill_curve(base_curve)
    base = pf.nodes.geo.extrude_mesh(mesh=base, offset_scale=0.02)
    base_mesh = metric_box_uv(base.mesh)

    stem_cap_curve = pf.nodes.geo.curve_circle(radius=0.02)
    stem_cap = pf.nodes.geo.fill_curve(stem_cap_curve)
    stem_cap = pf.nodes.geo.extrude_mesh(mesh=stem_cap, offset_scale=0.06)
    stem_cap_mesh = metric_box_uv(stem_cap.mesh)

    lever = _lever_handle()
    lever_left = pf.nodes.geo.transform(
        geometry=lever,
        translation=(0.0, 0.08, 0.0),
        rotation=(0.0, 0.0, 2.618),
        scale=(0.3, 0.3, 0.3),
    )
    lever_right = pf.nodes.geo.transform(
        geometry=lever,
        translation=(0.0, -0.08, 0.0),
        rotation=(0.0, 0.0, 3.6652),
        scale=(0.3, 0.3, 0.3),
    )
    lever_handles = pf.nodes.geo.join_geometry([lever_left, lever_right])

    thin_handle = pf.nodes.geo.mesh_cylinder(
        vertices=41,
        side_segments=39,
        radius=0.002,
        depth=0.04,
    )
    thin_handle_mesh = metric_box_uv(thin_handle.mesh)
    thin_right_short = pf.nodes.geo.transform(
        geometry=thin_handle_mesh,
        translation=(0.0, -0.032, 0.06),
        rotation=(0.0, 0.0, 0.0855),
        scale=(1.0, 1.0, 1.1),
    )
    thin_right_long = pf.nodes.geo.transform(
        geometry=thin_right_short,
        translation=(0.0, -0.004, -0.002),
        scale=(4.1, 1.0, 1.0),
    )
    thin_right_length = pf.nodes.func.switch(
        switch=length_one_side,
        a=thin_right_short,
        b=thin_right_long,
        data_type=NodeDataType.GEOMETRY,
    )
    thin_right = pf.nodes.func.switch(
        switch=one_side,
        a=thin_right_short,
        b=thin_right_length,
        data_type=NodeDataType.GEOMETRY,
    )
    thin_left = pf.nodes.geo.transform(
        geometry=thin_handle_mesh,
        translation=(0.0, 0.032, 0.06),
        scale=(1.0, 1.0, 1.1),
    )
    thin_left = pf.nodes.func.switch(
        switch=one_side,
        a=thin_left,
        data_type=NodeDataType.GEOMETRY,
    )
    thin_handles = pf.nodes.geo.join_geometry([thin_right, thin_left])

    thick_handle = pf.nodes.geo.mesh_cylinder(
        vertices=41,
        side_segments=39,
        radius=0.012,
        depth=0.04,
    )
    thick_handle_mesh = metric_box_uv(thick_handle.mesh)
    thick_right = pf.nodes.geo.transform(
        geometry=thick_handle_mesh,
        translation=(0.0, -0.02, 0.04),
        rotation=(1.5708, 0.0, 0.0),
    )
    thick_left = pf.nodes.geo.transform(
        geometry=thick_handle_mesh,
        translation=(0.0, 0.02, 0.04),
        rotation=(1.5708, 0.0, 0.0),
    )
    thick_left = pf.nodes.func.switch(
        switch=one_side,
        a=thick_left,
        data_type=NodeDataType.GEOMETRY,
    )
    bar_handles = pf.nodes.geo.join_geometry([thin_handles, thick_right, thick_left])
    bar_handles = pf.nodes.geo.transform(
        geometry=bar_handles,
        scale=pf.nodes.math.combine_xyz(
            x=hands_length_x,
            y=hands_length_y,
            z=1.0,
        ),
    )
    handles = pf.nodes.func.switch(
        switch=hand_type,
        a=lever_handles,
        b=bar_handles,
        data_type=NodeDataType.GEOMETRY,
    )

    arc = pf.nodes.geo.curve_circle(radius=0.08)
    arc = pf.nodes.geo.transform(
        geometry=arc,
        translation=(0.0, 0.08, 0.0),
    )
    arc = pf.nodes.geo.transform(
        geometry=arc,
        rotation=(-1.5708, 1.5708, 0.0),
        scale=(1.0, 0.7, 1.0),
    )
    curve = pf.nodes.geo.curve_bezier_segment(
        start=(0.0, 0.0, 0.0),
        start_handle=(0.0, 0.48, 0.0),
        end_handle=pf.nodes.math.combine_xyz(x=0.08, y=curl),
        end=(-0.02, 0.04, 0.0),
        resolution=177,
    )
    curve = pf.nodes.geo.trim_curve(curve=curve, end=0.6625)
    curve = pf.nodes.geo.transform(
        geometry=curve,
        rotation=(1.5708, 0.0, 2.522),
        scale=(5.2, 0.5, 7.8),
    )
    profile = pf.nodes.geo.curve_circle(radius=0.012)
    arc_spout = curve_to_mesh_with_uv(curve=arc, profile=profile).mesh
    curved_spout = curve_to_mesh_with_uv(
        curve=curve,
        profile=profile,
    ).mesh
    spout_curve = pf.nodes.func.switch(
        switch=switch,
        a=arc_spout,
        b=curved_spout,
        data_type=NodeDataType.GEOMETRY,
    )
    position = pf.nodes.geo.input_position()
    selection = pf.nodes.func.switch(
        switch=switch,
        a=position.z > -0.004,
        b=1.0,
        data_type=NodeDataType.FLOAT,
    )
    spout_curve = pf.nodes.geo.separate_geometry(
        geometry=spout_curve,
        selection=selection.astype(dtype=bool),
    )
    spout_scale = pf.nodes.math.combine_xyz(x=1.0, y=1.0, z=tap_head)
    spout_scale = pf.nodes.func.switch(
        switch=switch,
        a=spout_scale,
        b=(1.0, 1.0, 1.0),
        data_type=NodeDataType.FLOAT_VECTOR,
    )
    spout_curve = pf.nodes.geo.transform(
        geometry=spout_curve.selection,
        translation=(0.0, 0.0, 0.24),
        scale=spout_scale,
    )
    stem = pf.nodes.geo.curve_line(
        start=(0.0, 0.0, 0.0),
        end=(0.0, 0.0, 0.24),
    )
    stem = curve_to_mesh_with_uv(curve=stem, profile=profile).mesh
    spout = pf.nodes.geo.join_geometry([spout_curve, stem])
    spout = pf.nodes.geo.transform(
        geometry=spout,
        rotation=pf.nodes.math.combine_xyz(z=rotation_z).astype(dtype=pf.Euler),
        scale=pf.nodes.math.combine_xyz(x=1.0, y=1.0, z=tap_height),
    )
    standard = pf.nodes.geo.join_geometry([stem_cap_mesh, handles, spout])

    vessel_tip = pf.nodes.geo.mesh_cylinder(vertices=318, radius=0.008, depth=0.012)
    vessel_tip = pf.nodes.geo.transform(
        geometry=vessel_tip.mesh,
        translation=(0.238, 0.0, 0.152),
    )
    vessel_tip = metric_box_uv(vessel_tip)
    vessel_tube = pf.nodes.geo.mesh_cylinder(vertices=100, radius=0.004, depth=0.28)
    vessel_tube = pf.nodes.geo.set_position(
        geometry=vessel_tube.mesh,
        offset=(0.0, 0.0, 0.0),
    )
    vessel_tube = pf.nodes.geo.transform(
        geometry=vessel_tube,
        translation=(0.12, 0.0, 0.1),
        rotation=(0.0, -2.042, 0.0),
        scale=(1.7, 3.1, 1.0),
    )
    vessel_tube = metric_box_uv(vessel_tube)
    vessel_spout = pf.nodes.geo.join_geometry([vessel_tip, vessel_tube])
    vessel_spout = pf.nodes.geo.transform(
        geometry=vessel_spout,
        scale=(0.9, 1.0, 1.0),
    )

    vessel_base_curve = pf.nodes.geo.curve_circle(resolution=307, radius=0.022)
    vessel_base = pf.nodes.geo.fill_curve(vessel_base_curve)
    vessel_base = pf.nodes.geo.extrude_mesh(mesh=vessel_base, offset_scale=0.06)
    vessel_base_mesh = metric_box_uv(vessel_base.mesh)
    vessel_handle_curve = pf.nodes.geo.curve_bezier_segment(
        start=(0.0, 0.0, 0.0),
        start_handle=(0.0, 0.0, 0.28),
        end_handle=(0.08, 0.0, 0.28),
        end=(0.4, 0.0, 0.36),
        resolution=54,
    )
    spline_parameter = pf.nodes.geo.spline_parameter()
    vessel_radius = pf.nodes.math.float_curve(
        factor=1.0,
        value=spline_parameter.factor,
        curve=np.array(
            [[0.0, 0.975], [0.6295, 0.4125], [1.0, 0.1625]],
            dtype=np.float64,
        ),
    )
    vessel_handle_curve = pf.nodes.geo.set_curve_radius(
        curve=vessel_handle_curve,
        radius=vessel_radius * 1.3,
    )
    vessel_profile = pf.nodes.geo.curve_circle(radius=0.04)
    vessel_handle = curve_to_mesh_with_uv(
        curve=vessel_handle_curve,
        profile=vessel_profile,
        fill_caps=True,
    ).mesh
    position = pf.nodes.geo.input_position()
    vessel_y_scale = pf.nodes.math.map_range(
        value=position.x,
        from_min=0.08,
        from_max=0.4,
        to_max=2.5,
        to_min=1.0,
        data_type=NodeDataType.FLOAT,
    )
    vessel_position = pf.nodes.math.combine_xyz(
        x=position.x,
        y=position.y * vessel_y_scale,
        z=position.z,
    )
    vessel_handle = pf.nodes.geo.set_position(
        geometry=vessel_handle,
        position=vessel_position,
        offset=(0.0, 0.0, 0.0),
    )
    vessel_handle = pf.nodes.geo.subdivision_surface(vessel_handle)
    vessel_handle = pf.nodes.geo.set_shade_smooth(vessel_handle)
    vessel_handle = pf.nodes.geo.transform(
        geometry=vessel_handle,
        translation=(0.0, 0.0, 0.04),
        rotation=(0.0, 0.0, 0.6807),
        scale=(0.4, 0.4, 0.3),
    )
    vessel = pf.nodes.geo.join_geometry([vessel_spout, vessel_base_mesh, vessel_handle])
    vessel = pf.nodes.geo.transform(
        geometry=vessel,
        rotation=(0.0, 0.0, 3.1416),
    )

    body = pf.nodes.func.switch(
        switch=different_type,
        a=standard,
        b=vessel,
        data_type=NodeDataType.GEOMETRY,
    )
    body = pf.nodes.geo.join_geometry([base_mesh, body])
    return pf.nodes.geo.set_material(geometry=body, material=material)


def _finish(geometry: pf.ProcNode) -> TapResult:
    geometry = pf.nodes.geo.transform(
        geometry=geometry,
        rotation=(0.0, 0.0, 3.14159265),
    )
    bounds = pf.nodes.geo.bound_box(geometry)
    geometry = pf.nodes.geo.transform(
        geometry=geometry,
        translation=pf.nodes.math.combine_xyz(z=bounds.min.z * -1.0),
    )
    geometry = metric_box_uv(geometry)
    obj = pf.nodes.to_mesh_object(pf.nodes.geo.realize_instances(geometry))
    return TapResult(mesh=obj)


def tap(
    material: pf.Material | None = None,
    base_width: float = 0.1,
    tap_head: float = 0.9,
    rotation_z: float = 6.25,
    tap_height: float = 0.75,
    base_radius: float = 0.02,
    switch: bool = False,
    curl: float = -0.112,
    hand_type: bool = True,
    hands_length_x: float = 1.0,
    hands_length_y: float = 1.25,
    one_side: bool = False,
    different_type: bool = False,
    length_one_side: bool = False,
) -> TapResult:
    if material is None:
        material = pf.Material(surface=pf.nodes.shader.principled_bsdf(metallic=1.0))
    geometry = _tap_geometry(
        material=material,
        base_width=base_width,
        tap_head=tap_head,
        rotation_z=rotation_z,
        tap_height=tap_height,
        base_radius=base_radius,
        switch=switch,
        curl=curl,
        hand_type=hand_type,
        hands_length_x=hands_length_x,
        hands_length_y=hands_length_y,
        one_side=one_side,
        different_type=different_type,
        length_one_side=length_one_side,
    )
    return _finish(geometry)


def tap_rand(
    rng: pf.RNG,
    material: pf.Material | None = None,
    base_width: float | None = None,
    tap_head: float | None = None,
    rotation_z: float | None = None,
    tap_height: float | None = None,
    base_radius: float | None = None,
    switch: bool | None = None,
    curl: float | None = None,
    hand_type: bool | None = None,
    hands_length_x: float | None = None,
    hands_length_y: float | None = None,
    one_side: bool | None = None,
    different_type: bool | None = None,
    length_one_side: bool | None = None,
) -> TapResult:
    (
        rng_material,
        rng_base_width,
        rng_tap_head,
        rng_rotation,
        rng_height,
        rng_base_radius,
        rng_switch,
        rng_curl,
        rng_hand_type,
        rng_hand_x,
        rng_hand_y,
        rng_one_side,
        rng_different_type,
        rng_length_one_side,
    ) = rng.spawn(14)
    vector = pf.nodes.shader.coord().uv
    if material is None:
        material = decorative_material_rand(rng_material, vector)
    if base_width is None:
        base_width = pf.random.uniform(rng_base_width, 0.08, 0.12)
    if tap_head is None:
        tap_head = pf.random.uniform(rng_tap_head, 0.7, 1.1)
    if rotation_z is None:
        rotation_z = pf.random.uniform(rng_rotation, 5.5, 7.0)
    if tap_height is None:
        tap_height = pf.random.uniform(rng_height, 0.5, 1.0)
    if base_radius is None:
        base_radius = pf.random.uniform(rng_base_radius, 0.0, 0.04)
    if switch is None:
        switch = pf.control.choice(rng_switch, [(True, 1.0), (False, 1.0)])
    if curl is None:
        curl = pf.random.uniform(rng_curl, -0.2, -0.024)
    if hand_type is None:
        hand_type = pf.control.choice(rng_hand_type, [(True, 4.0), (False, 1.0)])
    if hands_length_x is None:
        hands_length_x = pf.random.uniform(rng_hand_x, 0.75, 1.25)
    if hands_length_y is None:
        hands_length_y = pf.random.uniform(rng_hand_y, 0.95, 1.55)
    if one_side is None:
        one_side = pf.control.choice(rng_one_side, [(True, 1.0), (False, 1.0)])
    if different_type is None:
        different_type = pf.control.choice(
            rng_different_type,
            [(True, 1.0), (False, 4.0)],
        )
    if length_one_side is None:
        length_one_side = pf.control.choice(
            rng_length_one_side,
            [(True, 1.0), (False, 4.0)],
        )
    return tap(
        material=material,
        base_width=base_width,
        tap_head=tap_head,
        rotation_z=rotation_z,
        tap_height=tap_height,
        base_radius=base_radius,
        switch=switch,
        curl=curl,
        hand_type=hand_type,
        hands_length_x=hands_length_x,
        hands_length_y=hands_length_y,
        one_side=one_side,
        different_type=different_type,
        length_one_side=length_one_side,
    )
