# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import math
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.functionality_lists import (
    fabric_general_rand,
    fabric_sturdy_rand,
)
from infinigen2.util import mesh as mesh_util
from infinigen2.util.curve import curve_to_mesh_with_uv

__all__ = [
    "CushionResult",
    "cushion_bed_pillow_rand",
    "cushion_box_geometry",
    "cushion_knife_edge_geometry",
    "cushion_throw_pillow_rand",
    "optional_piping_radius_rand",
    "support_loop_offset",
]


class CushionResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def support_loop_offset(
    radius: t.SocketOrVal[float],
    size: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.Vector]:
    return pf.nodes.math.vector_minimum(
        pf.nodes.math.combine_xyz(x=radius, y=radius, z=radius), size * 0.4
    )


@pf.nodes.node_function
def _anchored(
    geometry: t.SocketOrVal[pf.MeshObject],
    size: t.SocketOrVal[pf.Vector],
    location: t.SocketOrVal[pf.Vector],
    anchor: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[pf.MeshObject]:
    center_factor = pf.nodes.math.map_range(
        value=anchor,
        from_min=(0.0, 0.0, 0.0),
        from_max=(1.0, 1.0, 1.0),
        to_min=(0.5, 0.5, 0.5),
        to_max=(-0.5, -0.5, -0.5),
    )
    center = pf.nodes.math.vector_multiply_add(a=center_factor, b=size, addend=location)
    return pf.nodes.geo.transform(
        geometry=geometry, translation=center, rotation=(0, 0, 0), scale=(1, 1, 1)
    )


@pf.nodes.node_function
def piping_along_seam(
    mesh: t.SocketOrVal[pf.MeshObject],
    seam: t.SocketOrVal[bool],
    piping_radius: t.SocketOrVal[float],
) -> pf.ProcNode[pf.MeshObject]:
    """Welt cord tube along the fully-creased ``seam`` edge loops of ``mesh``.

    A fully creased edge chain and a tube swept along the same poly chain both
    subdivide to the cubic B-spline of its vertices, so the cord stays on the
    seam. A hexagon ring subdivides to a circle of 5/6 its circumradius."""
    loops = pf.nodes.geo.mesh_to_curve(mesh, selection=seam)
    loops = pf.nodes.geo.remove_attribute(loops, name="crease_edge")
    profile = pf.nodes.geo.curve_circle(resolution=6, radius=piping_radius * 1.2)
    tube = curve_to_mesh_with_uv(curve=loops, profile=profile).mesh
    has_piping = pf.nodes.func.greater_than(a=piping_radius, b=1e-5)
    return pf.nodes.func.switch(switch=has_piping, b=tube)


@pf.nodes.node_function
def _seam_edges(
    half_size: t.SocketOrVal[pf.Vector],
) -> pf.ProcNode[bool]:
    edge = pf.nodes.geo.input_mesh_edge_vertices()
    p1 = pf.nodes.math.vector_absolute(edge.position_1)
    p2 = pf.nodes.math.vector_absolute(edge.position_2)
    limit = half_size - (1e-5, 1e-5, 1e-5)
    on_cap = pf.nodes.func.boolean_and(a=p1.z > limit.z, b=p2.z > limit.z)
    along_x_side = pf.nodes.func.boolean_and(a=p1.x > limit.x, b=p2.x > limit.x)
    along_y_side = pf.nodes.func.boolean_and(a=p1.y > limit.y, b=p2.y > limit.y)
    on_rim = pf.nodes.func.boolean_or(a=along_x_side, b=along_y_side)
    return pf.nodes.func.boolean_and(a=on_cap, b=on_rim)


@pf.nodes.node_function
def cushion_box_geometry(
    size: t.SocketOrVal[pf.Vector],
    material: t.SocketOrVal[pf.Material],
    piping_material: t.SocketOrVal[pf.Material],
    location: t.SocketOrVal[pf.Vector] = (0, 0, 0),
    anchor: t.SocketOrVal[pf.Vector] = (0.5, 0.5, 0.5),
    edge_radius: t.SocketOrVal[float] = 0.03,
    crown: t.SocketOrVal[float] = 0.0,
    piping_radius: t.SocketOrVal[float] = 0.0,
    seam_crease: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[pf.MeshObject]:
    """Boxed cushion within ``size``, placed like ``box_with_support_loops``.
    Local z is thickness: boxing ``size.z - 2 * crown`` plus a domed panel on
    each side. Piped seams are fully creased, unpiped ones get ``seam_crease``."""
    boxing = size.z - crown * 2.0
    body_size = pf.nodes.math.combine_xyz(x=size.x, y=size.y, z=boxing)
    loop_offset = support_loop_offset(edge_radius, body_size)
    body = mesh_util.box_with_support_loops(
        size=body_size,
        location=(0, 0, 0),
        anchor=(0.5, 0.5, 0.5),
        vertices_x=7,
        vertices_y=7,
        vertices_z=4,
        support_loop_offset=loop_offset,
    )
    seam = _seam_edges(body_size * 0.5)
    has_piping = pf.nodes.func.greater_than(a=piping_radius, b=1e-5)
    crease = pf.nodes.func.switch(switch=has_piping, a=seam_crease, b=1.0)
    body = pf.nodes.geo.store_named_attribute(
        geometry=body, domain="EDGE", name="crease_edge", selection=seam, value=crease
    )

    position = pf.nodes.geo.input_position()
    dome_x = pf.nodes.math.cos(position.x / size.x * math.pi)
    dome_y = pf.nodes.math.cos(position.y / size.y * math.pi)
    on_panel = pf.nodes.math.absolute(position.z) > boxing * 0.5 - 1e-5
    lift = dome_x * dome_y * crown * on_panel.astype(dtype=float)
    body = pf.nodes.geo.set_position(
        geometry=body,
        offset=pf.nodes.math.combine_xyz(z=pf.nodes.math.sign(position.z) * lift),
    )

    piping = piping_along_seam(body, seam, piping_radius)
    body = pf.nodes.geo.set_material(body, material)
    piping = pf.nodes.geo.set_material(piping, piping_material)
    cushion = pf.nodes.geo.join_geometry([body, piping])
    return _anchored(cushion, size, location, anchor)


def knife_edge_fullness_rand(rng: pf.RNG) -> float:
    return pf.random.uniform(rng, 1.4, 3.0)


@pf.nodes.node_function
def _superellipse(
    u: t.SocketOrVal[float], fullness: t.SocketOrVal[float]
) -> pf.ProcNode[float]:
    inside = 1.0 - pf.nodes.math.power(pf.nodes.math.absolute(u), fullness)
    return pf.nodes.math.power(pf.nodes.math.maximum(inside, 0.0), 1.0 / fullness)


@pf.nodes.node_function
def _bunch_to_edges(s: t.SocketOrVal[float]) -> pf.ProcNode[float]:
    toward_edge = pf.nodes.math.maximum(1.0 - pf.nodes.math.absolute(s), 0.0)
    return pf.nodes.math.sign(s) * (1.0 - toward_edge * toward_edge * toward_edge)


@pf.nodes.node_function
def cushion_knife_edge_geometry(
    size: t.SocketOrVal[pf.Vector],
    material: t.SocketOrVal[pf.Material],
    piping_material: t.SocketOrVal[pf.Material],
    location: t.SocketOrVal[pf.Vector] = (0, 0, 0),
    anchor: t.SocketOrVal[pf.Vector] = (0.5, 0.5, 0.5),
    fullness: t.SocketOrVal[float] = 2.0,
    corner_pinch: t.SocketOrVal[float] = 0.05,
    piping_radius: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[pf.MeshObject]:
    """Throw pillow: two panels sewn along one creased seam, ``size.z`` thick at
    the centre. Each panel follows the superellipse ``_superellipse`` so it
    meets the seam convex; higher ``fullness`` is squarer. ``corner_pinch`` is
    the fraction each side bows inward at its middle."""
    grid = pf.nodes.geo.mesh_grid(size_x=2.0, size_y=2.0, vertices_x=7, vertices_y=7)
    position = pf.nodes.geo.input_position()
    u = _bunch_to_edges(position.x)
    v = _bunch_to_edges(position.y)
    falloff_u = 1.0 - u * u
    falloff_v = 1.0 - v * v
    bulge = _superellipse(u, fullness) * _superellipse(v, fullness)
    panel_position = pf.nodes.math.combine_xyz(
        x=u * size.x * 0.5 * (1.0 - corner_pinch * falloff_v),
        y=v * size.y * 0.5 * (1.0 - corner_pinch * falloff_u),
        z=bulge * size.z * 0.5,
    )
    top = pf.nodes.geo.set_position(geometry=grid.mesh, position=panel_position)
    panel_uv = pf.nodes.geo.input_position()
    top = pf.nodes.geo.store_named_attribute(
        geometry=top,
        name="UVMap",
        value=pf.nodes.math.combine_xyz(x=panel_uv.x, y=panel_uv.y),
        domain="CORNER",
        data_type="FLOAT2",
    )
    bottom = pf.nodes.geo.transform(
        geometry=top, translation=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, -1)
    )
    bottom = pf.nodes.geo.flip_faces(bottom)
    body = pf.nodes.geo.join_geometry([top, bottom])
    body = pf.nodes.geo.merge_by_distance(body, distance=1e-5)

    edge = pf.nodes.geo.input_mesh_edge_vertices()
    seam = pf.nodes.func.boolean_and(
        a=pf.nodes.math.absolute(edge.position_1.z) < 1e-6,
        b=pf.nodes.math.absolute(edge.position_2.z) < 1e-6,
    )
    body = pf.nodes.geo.store_named_attribute(
        geometry=body, domain="EDGE", name="crease_edge", selection=seam, value=1.0
    )
    piping = piping_along_seam(body, seam, piping_radius)
    body = pf.nodes.geo.set_material(body, material)
    piping = pf.nodes.geo.set_material(piping, piping_material)
    cushion = pf.nodes.geo.join_geometry([body, piping])
    return _anchored(cushion, size, location, anchor)


def _cushion_object(geometry: pf.ProcNode[pf.MeshObject]) -> CushionResult:
    obj = pf.nodes.to_mesh_object(geometry)
    pf.ops.modifier.subdivide_surface(obj, levels=6, _skip_apply=True)
    return CushionResult(mesh=obj)


def _no_piping_rand(rng: pf.RNG) -> float:
    return 0.0


def piping_radius_rand(rng: pf.RNG) -> float:
    """Finished welt radius: 4/32-6/32 in cord (3.2-4.8 mm) wrapped in fabric."""
    return pf.random.uniform(rng, 0.0025, 0.0055)


def optional_piping_radius_rand(rng: pf.RNG, piped_weight: float) -> float:
    rng_choice, rng_radius = rng.spawn(2)
    piping_fn = pf.control.choice(
        rng_choice,
        [(piping_radius_rand, piped_weight), (_no_piping_rand, 1.0 - piped_weight)],
    )
    return piping_fn(rng_radius)


def _knife_edge_pillow(
    rng: pf.RNG,
    size: pf.Vector,
    piping_radius: float | None,
    material: pf.Material | None,
    fullness: float | None = None,
) -> CushionResult:
    rng, rng_piping, rng_fabric = rng.spawn(3)
    if fullness is None:
        fullness = knife_edge_fullness_rand(rng)
    if piping_radius is None:
        piping_radius = optional_piping_radius_rand(rng_piping, 0.4)
    if material is None:
        material = fabric_sturdy_rand(rng_fabric, pf.nodes.shader.coord().uv)
    corner_pinch = pf.random.uniform(rng, 0.02, 0.07)
    geometry = cushion_knife_edge_geometry(
        size=size,
        material=material,
        piping_material=material,
        anchor=(0.5, 0.5, 0.0),
        fullness=fullness,
        corner_pinch=corner_pinch,
        piping_radius=piping_radius,
    )
    return _cushion_object(geometry)


def cushion_throw_pillow_rand(
    rng: pf.RNG,
    size: pf.Vector | None = None,
    piping_radius: float | None = None,
    material: pf.Material | None = None,
    fullness: float | None = None,
) -> CushionResult:
    """Square throw pillow, 16-24 in (0.40-0.60 m) across."""
    rng, rng_pillow = rng.spawn(2)
    if size is None:
        side = pf.random.uniform(rng, 0.4, 0.6)
        thickness = side * pf.random.uniform(rng, 0.25, 0.33)
        depth = side * pf.random.uniform(rng, 0.97, 1.03)
        size = pf.Vector((side, depth, thickness))
    return _knife_edge_pillow(rng_pillow, size, piping_radius, material, fullness)


def cushion_bed_pillow_rand(
    rng: pf.RNG,
    size: pf.Vector | None = None,
    fullness: float | None = None,
    material: pf.Material | None = None,
) -> CushionResult:
    """Sleeping pillow, 19-21 in deep, 26-36 in long (standard to king), 6-9 in loft."""
    rng, rng_fabric, rng_pillow = rng.spawn(3)
    if size is None:
        short_side = pf.random.uniform(rng, 0.48, 0.53)
        long_side = pf.random.uniform(rng, 0.66, 0.92)
        loft = pf.random.uniform(rng, 0.15, 0.22)
        size = pf.Vector((short_side, long_side, loft))
    if material is None:
        material = fabric_general_rand(rng_fabric, pf.nodes.shader.coord().uv)
    return _knife_edge_pillow(rng_pillow, size, 0.0, material, fullness)
