# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import math
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import handles
from infinigen2.shaders import functionality_lists
from infinigen2.util import mesh

__all__ = [
    "DrawerResult",
    "drawer",
    "drawer_handle_rand",
    "drawer_rand",
]


class DrawerResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _panel(
    dimensions: t.SocketOrVal[pf.Vector], origin: t.SocketOrVal[pf.Vector]
) -> pf.ProcNode:
    box = pf.nodes.geo.mesh_cube(size=dimensions)
    geo = pf.nodes.geo.transform(box.mesh, translation=origin + dimensions * 0.5)
    return mesh.metric_box_uv(geo)


@pf.nodes.node_function
def _front_coordinate(
    coordinate: t.SocketOrVal[float],
    size: t.SocketOrVal[float],
    ring_width: t.SocketOrVal[float],
) -> pf.ProcNode:
    inner = size * 0.5 - ring_width
    edge = pf.nodes.func.greater_than(pf.nodes.math.absolute(coordinate), 0.99).astype(
        float
    )
    return pf.nodes.math.sign(coordinate) * (inner + ring_width * edge)


@pf.nodes.node_function
def _front_panel(
    dimensions: t.SocketOrVal[pf.Vector],
    origin: t.SocketOrVal[pf.Vector],
    ring_widths: t.SocketOrVal[pf.Vector],
    bevel: t.SocketOrVal[float],
) -> pf.ProcNode:
    box = pf.nodes.geo.mesh_cube(size=(2, 2, 2), vertices_y=4, vertices_z=4)
    position = pf.nodes.geo.input_position()
    y = _front_coordinate(position.y, dimensions.y, ring_widths.y)
    z = _front_coordinate(position.z, dimensions.z, ring_widths.z)
    edge_distance = pf.nodes.math.minimum(
        dimensions.y * 0.5 - pf.nodes.math.absolute(y),
        dimensions.z * 0.5 - pf.nodes.math.absolute(z),
    )
    chamfer = pf.nodes.math.maximum(bevel - edge_distance, 0.0)
    front = pf.nodes.func.greater_than(position.x, 0.0).astype(float)
    x = front * (dimensions.x - chamfer)
    coords = pf.nodes.math.combine_xyz(
        x, y + dimensions.y * 0.5, z + dimensions.z * 0.5
    )
    geo = pf.nodes.geo.set_position(box.mesh, position=origin + coords)
    return mesh.metric_box_uv(geo)


@pf.tracer.generator
def drawer(
    dimensions: pf.Vector,
    material: pf.Material,
    front_origin: pf.Vector,
    front_dimensions: pf.Vector,
    clearance: float = 0.002,
    thickness: float = 0.012,
    front_bevel: float = 0,
    opening_fraction: float = 0,
) -> DrawerResult:
    """Tray fitting a min-corner slot of `dimensions`, fronted at `front_origin`.

    Opening translates the whole drawer by a fraction of its tray depth along +X.
    """
    _, width, height = dimensions
    board, gap = thickness, clearance
    inner_width, inner_height = width - 2 * gap, height - 2 * gap
    tray_depth = front_origin.x - gap
    bottom = _panel((tray_depth, inner_width, board), (gap, gap, gap))
    left = _panel((tray_depth, board, inner_height - board), (gap, gap, gap + board))
    right = _panel(
        (tray_depth, board, inner_height - board),
        (gap, gap + inner_width - board, gap + board),
    )
    back = _panel(
        (board, inner_width - 2 * board, inner_height - board),
        (gap, gap + board, gap + board),
    )
    ring_y = front_bevel if front_bevel > 0 else front_dimensions.y * 0.25
    ring_z = front_bevel if front_bevel > 0 else front_dimensions.z * 0.25
    front = _front_panel(
        front_dimensions, front_origin, (0, ring_y, ring_z), front_bevel
    )
    geo = pf.nodes.geo.join_geometry([bottom, left, right, back, front])
    geo = mesh.crease_sharp(geo, threshold_degrees=30.0)
    geo = pf.nodes.geo.set_material(geo, material=material)
    obj = pf.nodes.to_mesh_object(geo)
    pf.ops.modifier.bevel(obj, width=min(0.001, board * 0.1), segments=2)
    pf.ops.object.set_transform(obj, location=(tray_depth * opening_fraction, 0, 0))
    pf.ops.mesh.transform_apply(obj, location=True, rotation=False, scale=False)
    obj.item().name = "drawer"
    # left unapplied so a handle joined in afterwards renders at the same density
    pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    return DrawerResult(obj)


def drawer_handle_rand(rng: pf.RNG, dimensions: pf.Vector) -> pf.MeshObject:
    """Bar pull grip-sized to the drawer front, or a knob."""
    rng_style, rng_handle = rng.spawn(2)

    def bar_pull(r: pf.RNG) -> handles.HandleResult:
        return handles.bar_pull_handle_rand(
            r, grip_length=min(0.16, dimensions.y * 0.6)
        )

    handle_fn = pf.control.choice(
        rng_style, [(bar_pull, 1.0), (handles.knob_handle_rand, 1.0)]
    )
    return handle_fn(rng_handle).mesh


def drawer_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    frame_widths: tuple[float, float, float, float] = (0.02, 0.02, 0.02, 0.02),
    front_thickness: float = 0.018,
    opening_fraction: float | None = None,
) -> DrawerResult:
    """Sample a drawer; frame widths follow left/right/bottom/top slot edges."""
    r_size, r_material, r_handle, r_front, r_bevel, r_open = rng.spawn(6)
    r_front_choice, r_front_body = r_front.spawn(2)
    r_bevel_choice, r_bevel_body = r_bevel.spawn(2)
    if dimensions is None:
        dimensions = pf.Vector(
            (
                pf.random.uniform(r_size, 0.25, 0.55),
                pf.random.uniform(r_size, 0.25, 0.70),
                pf.random.uniform(r_size, 0.10, 0.30),
            )
        )
    if material is None:
        material = functionality_lists.furniture_material_rand(
            r_material, pf.nodes.shader.coord().uv
        )
    if opening_fraction is None:
        r_choice, r_distance = r_open.spawn(2)
        opening_fraction = pf.control.choice(
            r_choice, [(0.0, 2.0), (pf.random.uniform(r_distance, 0.05, 0.30), 1.0)]
        )
    depth, width, height = dimensions.x, dimensions.y, dimensions.z
    smallest = min(depth, width, height)
    clearance = min(0.002, smallest * 0.02)

    def inset_front(_r: pf.RNG) -> tuple[pf.Vector, pf.Vector]:
        origin = pf.Vector((depth - clearance - front_thickness, clearance, clearance))
        size = (front_thickness, width - 2 * clearance, height - 2 * clearance)
        return origin, pf.Vector(size)

    def overlay_front(r: pf.RNG) -> tuple[pf.Vector, pf.Vector]:
        fraction = pf.random.uniform(r, 0.2, 1 / 3)
        left, right, bottom, top = (fraction * edge for edge in frame_widths)
        origin = pf.Vector((depth, -left, -bottom))
        size = (front_thickness, width + left + right, height + bottom + top)
        return origin, pf.Vector(size)

    front_fn = pf.control.choice(
        r_front_choice, [(inset_front, 1.0), (overlay_front, 1.0)]
    )
    front = front_fn(r_front_body)
    front_origin, front_dimensions = front[0], front[1]
    front_bevel = pf.control.choice(
        r_bevel_choice,
        [(0.0, 1.0), (pf.random.uniform(r_bevel_body, 0.001, 0.004), 1.0)],
    )
    result = drawer(
        dimensions,
        material,
        front_origin,
        front_dimensions,
        clearance=clearance,
        thickness=min(0.012, smallest * 0.08),
        front_bevel=front_bevel,
        opening_fraction=opening_fraction,
    )
    handle = drawer_handle_rand(r_handle, front_dimensions)
    pf.ops.object.set_transform(
        handle,
        location=(
            front_origin.x + front_thickness,
            front_origin.y + front_dimensions.y * 0.5,
            front_origin.z + front_dimensions.z * 0.5,
        ),
        rotation_euler=(math.pi / 2, 0, 0),
    )
    pf.ops.object.join(result.mesh, handle)
    return result
