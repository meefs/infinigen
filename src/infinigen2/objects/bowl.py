# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import vase

__all__ = ["BowlResult", "bowl", "bowl_rand"]


class BowlResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def _bowl_geometry(
    diameter: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
    base_scale: t.SocketOrVal[float],
    profile_fullness: t.SocketOrVal[float],
    profile_slope: t.SocketOrVal[float],
    rib_count: t.SocketOrVal[int],
    rib_depth: t.SocketOrVal[float],
    rib_sharpness: t.SocketOrVal[float],
    twist: t.SocketOrVal[float],
    u_resolution: t.SocketOrVal[int],
    v_resolution: t.SocketOrVal[int],
) -> t.ProcNode[pf.MeshObject]:
    radius = diameter * 0.5
    base_radius = radius * base_scale
    start = pf.nodes.math.combine_xyz(x=base_radius)
    middle_radius = base_radius + (radius - base_radius) * profile_fullness
    middle = pf.nodes.math.combine_xyz(x=middle_radius, z=height * profile_slope)
    end = pf.nodes.math.combine_xyz(x=radius, z=height)
    section_resolution = v_resolution.astype(dtype=float) * 4.0
    section = pf.nodes.geo.curve_bezier(
        start=start,
        middle=middle,
        end=end,
        resolution=section_resolution.astype(dtype=int),
    )
    rib_segments = rib_count.astype(dtype=float) * 2.0
    samples_per_segment = u_resolution.astype(dtype=float) / rib_segments
    angular_resolution = pf.nodes.math.ceil(samples_per_segment) * rib_segments
    side_segments = v_resolution.astype(dtype=float) - 1.0
    cylinder = pf.nodes.geo.mesh_cylinder(
        vertices=angular_resolution.astype(dtype=int),
        side_segments=side_segments.astype(dtype=int),
        radius=1.0,
        depth=2.0,
    )
    geo = pf.nodes.geo.store_named_attribute(
        geometry=cylinder.mesh,
        name="UVMap",
        value=cylinder.uv_map,
        domain="CORNER",
        data_type="FLOAT2",
    )
    geo = pf.nodes.geo.delete_geometry(geo, selection=cylinder.top, domain="FACE")
    position = pf.nodes.geo.input_position()
    section_position = pf.nodes.geo.sample_curve(
        curves=section, factor=(position.z + 1.0) * 0.5, data_type="FLOAT"
    ).position
    elevation = section_position.z / height
    angle = pf.nodes.math.atan2(position.y, position.x)
    phase = angle * rib_count.astype(dtype=float)
    phase_cosine = pf.nodes.math.cos(phase)
    smooth_rib = (1.0 + phase_cosine) * 0.5
    sharp_rib = 1.0 - pf.nodes.math.acos(phase_cosine) / 3.141592653589793
    rib = smooth_rib + rib_sharpness * (sharp_rib - smooth_rib)
    fade = elevation * (2.0 - elevation)
    r = section_position.x * (1.0 - rib_depth * fade * rib)
    angle = angle + twist * elevation
    x = r * pf.nodes.math.cos(angle)
    y = r * pf.nodes.math.sin(angle)
    surface_position = pf.nodes.math.combine_xyz(x=x, y=y, z=section_position.z)
    geo = pf.nodes.geo.set_position(geo, position=surface_position)
    return geo


def bowl(
    diameter: float = 0.24,
    height: float = 0.09,
    base_scale: float = 0.4,
    profile_fullness: float = 0.85,
    profile_slope: float = 0.0,
    rib_count: int = 12,
    rib_depth: float = 0.0,
    rib_sharpness: float = 0.0,
    twist: float = 0.0,
    thickness: float = 0.003,
    u_resolution: int = 192,
    v_resolution: int = 32,
    material: pf.Material | None = None,
) -> BowlResult:
    if material is None:
        surface = pf.nodes.shader.principled_bsdf()
        material = pf.Material(surface=surface)
    geo = _bowl_geometry(
        diameter=diameter,
        height=height,
        base_scale=base_scale,
        profile_fullness=profile_fullness,
        profile_slope=profile_slope,
        rib_count=rib_count,
        rib_depth=rib_depth,
        rib_sharpness=rib_sharpness,
        twist=twist,
        u_resolution=u_resolution,
        v_resolution=v_resolution,
    )
    geo = pf.nodes.geo.set_material(geo, material)
    obj = pf.nodes.to_mesh_object(geo)
    pf.ops.modifier.solidify(obj, thickness=thickness, offset=-1.0)
    pf.ops.object.shade_smooth(obj)
    return BowlResult(mesh=obj)


def bowl_rand(rng: pf.RNG) -> BowlResult:
    rng, rng_ribs, rng_count, rng_sharp, rng_twist, rng_mat = rng.spawn(6)
    diameter = pf.random.uniform(rng, 0.14, 0.34)
    decorated_depth = pf.random.uniform(rng, 0.025, 0.1)
    rib_depth = pf.control.choice(rng_ribs, [(0.0, 0.5), (decorated_depth, 0.5)])
    broad_lobes = pf.random.randint(rng, 3, 7)
    fine_ribs = pf.random.randint(rng, 8, 25)
    rib_count = pf.control.choice(rng_count, [(broad_lobes, 0.4), (fine_ribs, 0.6)])
    rib_sharpness = pf.control.choice(rng_sharp, [(0.0, 0.6), (1.0, 0.4)])
    twist_angle = pf.random.uniform(rng, -0.8, 0.8)
    twist = pf.control.choice(rng_twist, [(0.0, 0.65), (twist_angle, 0.35)])
    uv = pf.nodes.shader.coord().uv
    material = vase.vase_material_rand(rng_mat, uv)
    height = diameter * pf.random.uniform(rng, 0.22, 0.48)
    base_scale = pf.random.uniform(rng, 0.3, 0.55)
    profile_fullness = pf.random.uniform(rng, 0.5, 1.0)
    profile_slope = pf.random.uniform(rng, 0.0, 0.16)
    thickness = diameter * pf.random.uniform(rng, 0.01, 0.018)
    return bowl(
        diameter=diameter,
        height=height,
        base_scale=base_scale,
        profile_fullness=profile_fullness,
        profile_slope=profile_slope,
        rib_count=rib_count,
        rib_depth=rib_depth,
        rib_sharpness=rib_sharpness,
        twist=twist,
        thickness=thickness,
        material=material,
    )
