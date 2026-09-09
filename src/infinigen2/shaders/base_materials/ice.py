# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Ankit Goyal, Zeyu Ma, Hongyu Wen: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/main/infinigen/assets/materials/terrain/ice.py)
# - Alexander Raistrick: port to procfunc/v2

import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "ice",
    "ice_color_rand",
    "ice_rand",
]


# Summed noise octaves peak well below their analytic bound; this is the measured
# fraction they actually reach, so the anchor sits on the real surface.
_CEILING_FILL = 0.52


@pf.nodes.node_function
def ice(
    vector: t.SocketOrVal[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.6469, 0.6947, 0.9522)),
    roughness: t.SocketOrVal[float] = 1.0,
    ridge_w: t.SocketOrVal[float] = 5.0,
    ridge_scale: t.SocketOrVal[float] = 8.0,
    ridge_height: t.SocketOrVal[float] = 0.03,
    swell_w: t.SocketOrVal[float] = 5.0,
    swell_scale: t.SocketOrVal[float] = 1.5,
    swell_height: t.SocketOrVal[float] = 0.08,
) -> pf.Material:
    surface_noise = pf.nodes.texture.noise(
        vector=vector,
        w=6.5,
        scale=4.0,
        detail=15.0,
        noise_dimensions="4D",
    )
    surface_roughness = pf.nodes.color.color_ramp(
        fac=surface_noise.fac,
        interpolation="LINEAR",
        points=[
            (0.5, (0.0844, 0.0844, 0.0844, 1.0)),
            (0.75, (1.0, 1.0, 1.0, 1.0)),
        ],
    )
    surface = pf.nodes.shader.principled_bsdf(
        base_color=color,
        roughness=surface_roughness.color.astype(dtype=float) * roughness,
        ior=1.31,
        subsurface_weight=1.0,
        subsurface_radius=(0.001, 0.001, 0.002),
    )

    ridge_noise = pf.nodes.texture.noise(
        vector=vector,
        w=ridge_w,
        scale=ridge_scale,
        detail=20.0,
        roughness=1.0,
        noise_dimensions="4D",
    )
    ridge = pf.nodes.color.color_ramp(
        fac=ridge_noise.fac,
        interpolation="LINEAR",
        points=[
            (0.5, (0.0, 0.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 1.0, 1.0)),
        ],
    )
    swell_noise = pf.nodes.texture.noise(
        vector=vector,
        w=swell_w,
        scale=swell_scale,
        detail=15.0,
        roughness=0.7,
        distortion=1.5,
        noise_dimensions="4D",
    )

    height = (
        ridge.color.astype(dtype=float) * ridge_height + swell_noise.fac * swell_height
    )
    displacement = pf.nodes.shader.displacement(
        height=height, midlevel=_CEILING_FILL * (ridge_height + swell_height)
    )

    return pf.Material(surface=surface, displacement=displacement, volume=None)


def ice_color_rand(rng: pf.RNG) -> pf.Color:
    hue = pf.random.uniform(rng, 0.5906, 0.6906)
    saturation = pf.random.uniform(rng, 0.2206, 0.4206)
    value = pf.random.uniform(rng, 0.8522, 1.0)
    return pf.color.hsv_color(hue=hue, saturation=saturation, value=value)


def ice_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    lanes = rng.spawn(5)
    if color is None:
        color = ice_color_rand(lanes[0])
    ridge_w = pf.random.uniform(lanes[1], 0, 10)
    ridge_scale = pf.random.uniform(lanes[2], 7, 9)
    swell_w = pf.random.uniform(lanes[3], 0, 10)
    swell_scale = pf.random.uniform(lanes[4], 1.3, 1.7)

    return ice(
        vector,
        color=color,
        ridge_w=ridge_w,
        ridge_scale=ridge_scale,
        swell_w=swell_w,
        swell_scale=swell_scale,
    )
