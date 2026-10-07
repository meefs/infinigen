# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Ankit Goyal, Zeyu Ma: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/main/infinigen/assets/materials/terrain/cracked_ground.py)
# - Alexander Raistrick: port to procfunc/v2
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=PIZ_wi3yFUM&list=PLsGl9GczcgBs6TtApKKK-L_0Nm6fovNPk&index=98 by Ryan King Art

import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "cracked_ground",
    "ground_cracked_rand",
]


# Summed noise octaves peak well below their analytic bound; this is the measured
# fraction they actually reach, so the anchor sits on the real surface.
_CEILING_FILL = 0.72


@pf.nodes.node_function
def cracked_ground(
    vector: t.SocketOrVal[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.3005, 0.1119, 0.0284)),
    crack_color: t.SocketOrVal[pf.Color] = pf.Color((0.2016, 0.1070, 0.0685)),
    grain_color: t.SocketOrVal[pf.Color] = pf.Color((0.6038, 0.4397, 0.2159)),
    roughness: t.SocketOrVal[float] = 0.9,
    w: t.SocketOrVal[float] = 0.0,
    crack_scale: t.SocketOrVal[float] = 2.0,
    mask_scale: t.SocketOrVal[float] = 2.0,
    relief_scale: t.SocketOrVal[float] = 3.0,
    crack_density: t.SocketOrVal[float] = 0.475,
    crack_width: t.SocketOrVal[float] = 0.025,
    grain_scale: t.SocketOrVal[float] = 60.0,
    crack_depth: t.SocketOrVal[float] = -0.025,
    displacement_height: t.SocketOrVal[float] = 0.5,
) -> pf.Material:
    relief_noise = pf.nodes.texture.noise(
        vector=vector,
        w=w,
        scale=relief_scale,
        detail=15.0,
        roughness=0.5375,
        noise_dimensions="4D",
    )

    crack_warp = pf.nodes.texture.noise(
        vector=vector,
        w=w,
        scale=crack_scale,
        detail=15.0,
        noise_dimensions="4D",
    )
    crack_cells = pf.nodes.texture.voronoi_distance(
        vector=crack_warp.color.astype(dtype=pf.Vector),
        w=w,
        scale=2.3,
        voronoi_dimensions="4D",
    )
    crack_lines = pf.nodes.math.map_range(
        value=crack_cells,
        from_min=0.0,
        from_max=crack_width,
        to_min=1.0,
        to_max=0.0,
    )

    mask_noise = pf.nodes.texture.noise(
        vector=vector,
        w=w,
        scale=mask_scale,
        detail=15.0,
        noise_dimensions="4D",
    )
    mask_center = 1.0 - crack_density
    crack_mask = pf.nodes.math.map_range(
        value=mask_noise.fac,
        from_min=mask_center - 0.02,
        from_max=mask_center + 0.02,
    )

    crack = crack_lines * crack_mask

    grain_cells = pf.nodes.texture.voronoi(
        vector=vector,
        w=w,
        scale=grain_scale,
        voronoi_dimensions="4D",
    )
    grain = pf.nodes.math.map_range(value=grain_cells.distance, from_min=0.9)

    height = displacement_height * (
        relief_noise.fac * 0.3 + crack * crack_depth + grain * 0.02
    )
    ceiling = (
        _CEILING_FILL
        * displacement_height
        * (0.3 + pf.nodes.math.maximum(crack_depth, 0.0) + 0.02)
    )
    displacement = pf.nodes.shader.displacement(height=height, midlevel=ceiling)

    tint_noise = pf.nodes.texture.noise(vector=vector, scale=15.0, detail=10.0)
    tint = pf.nodes.color.separate_rgb(color=tint_noise.color)
    hue_shift = pf.nodes.math.map_range(
        value=tint.red,
        from_min=0.4,
        from_max=0.7,
        to_min=0.49,
        to_max=0.51,
    )
    value_shift = pf.nodes.math.map_range(
        value=tint.green,
        from_min=0.4,
        from_max=0.72,
        to_min=0.4,
        to_max=1.1,
    )
    base = pf.nodes.color.hue_saturation(
        color=color,
        fac=1.0,
        hue=hue_shift,
        value=value_shift,
    )

    cracked = pf.nodes.color.mix_rgb(factor=crack, a=base, b=crack_color)
    grained = pf.nodes.color.mix_rgb(factor=grain, a=cracked, b=grain_color)

    surface = pf.nodes.shader.principled_bsdf(
        base_color=grained,
        specular_ior_level=0.2,
        roughness=roughness,
    )

    return pf.Material(
        surface=surface,
        displacement=displacement,
        volume=None,
    )


def ground_cracked_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    r_color, r_crack_color, r_grain_color, r_shape, r_depth = rng.spawn(5)

    displacement_height = pf.random.uniform(r_depth, 0.3, 0.7)
    crack_meters = pf.random.uniform(r_depth, 0.005, 0.02)

    if color is None:
        color = pf.color.hsv_color(
            hue=pf.random.uniform(r_color, 0.0, 0.15115),
            saturation=pf.random.uniform(r_color, 0.80549, 1.0),
            value=pf.random.uniform(r_color, 0.2005, 0.4005),
        )

    crack_color = pf.color.hsv_color(
        hue=pf.random.uniform(r_crack_color, 0.0, 0.14821),
        saturation=pf.random.uniform(r_crack_color, 0.56022, 0.76022),
        value=pf.random.uniform(r_crack_color, 0.1016, 0.3016),
    )
    grain_color = pf.color.hsv_color(
        hue=pf.random.uniform(r_grain_color, 0.0, 0.19616),
        saturation=pf.random.uniform(r_grain_color, 0.54243, 0.74243),
        value=pf.random.uniform(r_grain_color, 0.5038, 0.7038),
    )

    return cracked_ground(
        vector=vector,
        color=color,
        crack_color=crack_color,
        grain_color=grain_color,
        w=pf.random.uniform(r_shape, -10000.0, 10000.0),
        crack_scale=pf.random.uniform(r_shape, 1.0, 3.0),
        mask_scale=pf.random.uniform(r_shape, 1.0, 3.0),
        relief_scale=pf.random.uniform(r_shape, 2.0, 4.0),
        crack_density=pf.random.uniform(r_shape, 0.4, 0.55),
        crack_width=pf.random.uniform(r_shape, 0.01, 0.04),
        grain_scale=pf.random.uniform(r_shape, 20.0, 100.0),
        crack_depth=-crack_meters / displacement_height,
        displacement_height=displacement_height,
    )
