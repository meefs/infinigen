# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma, Mingzhe Wang, and Ankit Goyal: original Infinigen dirt material (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/materials/terrain/dirt.py)
# - Alexander Raistrick: refactor for Infinigen2
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=pEXHCsrTsco

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "dirt",
    "dirt_rand",
]


def _mountain_octave(
    vector: t.SocketOrVal[pf.Vector],
    w: t.SocketOrVal[float],
    scale: t.SocketOrVal[float],
    height: t.SocketOrVal[float],
) -> pf.ProcNode[float]:
    noise = pf.nodes.texture.noise(
        vector=vector,
        w=w,
        scale=scale,
        detail=7.0,
        roughness=0.7,
        noise_dimensions="4D",
    )
    return (noise.fac - 0.5) * height


# Summed noise octaves peak well below their analytic bound; this is the measured
# fraction they actually reach, so the anchor sits on the real surface.
_CEILING_FILL = 0.4


@pf.nodes.node_function
def dirt(
    vector: t.SocketOrVal[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.18, 0.075, 0.05)),
    color_dark: t.SocketOrVal[pf.Color] = pf.Color((0.19, 0.03, 0.02)),
    roughness: t.SocketOrVal[float] = 0.0,
    roughness_high: t.SocketOrVal[float] = 1.0,
    crack_density: t.SocketOrVal[float] = 0.05,
    crack_scale: t.SocketOrVal[float] = 5.0,
    crack_width: t.SocketOrVal[float] = 0.03,
    mid_noise_scale: t.SocketOrVal[float] = 5.0,
    large_noise_scale: t.SocketOrVal[float] = 20.0,
    mountain_scale_0: t.SocketOrVal[float] = 3.0,
    mountain_scale_1: t.SocketOrVal[float] = 3.0,
    mountain_scale_2: t.SocketOrVal[float] = 3.0,
    mountain_height_0: t.SocketOrVal[float] = 0.18,
    mountain_height_1: t.SocketOrVal[float] = 0.18,
    mountain_height_2: t.SocketOrVal[float] = 0.18,
    displacement_scale: t.SocketOrVal[float] = 1.0,
    w: t.SocketOrVal[float] = 0.0,
) -> pf.Material:
    detail_noise = pf.nodes.texture.noise(
        vector=vector,
        w=w,
        scale=8.0 * 0.5,
        detail=16.0,
        noise_dimensions="4D",
    )

    crack_mask_noise = pf.nodes.texture.noise(
        vector=vector,
        w=w + 11.0,
        scale=5.0 * 0.5,
        noise_dimensions="4D",
    )
    crack_mask = pf.nodes.math.map_range(
        value=crack_mask_noise.fac,
        from_min=0.445 + 2.0 * crack_density - 0.1,
        from_max=0.505 + 2.0 * crack_density - 0.1,
    )

    warp_noise = pf.nodes.texture.noise(
        vector=vector,
        w=w + 23.0,
        scale=1.0 * 0.5,
        detail=16.0,
        noise_dimensions="4D",
    )
    crack_distance = pf.nodes.texture.voronoi_distance(
        vector=warp_noise.color.astype(dtype=pf.Vector),
        scale=crack_scale,
    )
    crack_lines = pf.nodes.math.map_range(
        value=crack_distance,
        from_min=0.0,
        from_max=crack_width,
    )
    cracks = pf.nodes.math.mix(a=crack_lines, b=0.5, factor=crack_mask)

    mid_noise = pf.nodes.texture.noise(
        vector=vector,
        w=w + 37.0,
        scale=mid_noise_scale,
        noise_dimensions="4D",
    )
    large_noise = pf.nodes.texture.noise(
        vector=vector,
        w=w + 53.0,
        scale=large_noise_scale,
        noise_dimensions="4D",
    )
    large_ramp = pf.nodes.color.color_ramp(
        fac=large_noise.fac,
        points=[
            (0.0, pf.Color((0.0, 0.0, 0.0))),
            (0.3, pf.Color((0.5, 0.5, 0.5))),
            (0.7, pf.Color((0.5, 0.5, 0.5))),
            (1.0, pf.Color((1.0, 1.0, 1.0))),
        ],
    )
    large_height = pf.nodes.color.rgb_to_bw(large_ramp.color)

    height_cracks = 0.08 * (0.5 * cracks)
    height_mid = 0.05 * (mid_noise.fac - 0.5)
    height_large = 0.1 * (large_height - 0.5)

    mountain_0 = _mountain_octave(vector, w + 67.0, mountain_scale_0, mountain_height_0)
    mountain_1 = _mountain_octave(vector, w + 79.0, mountain_scale_1, mountain_height_1)
    mountain_2 = _mountain_octave(vector, w + 97.0, mountain_scale_2, mountain_height_2)
    height_mountain = pf.nodes.math.maximum(
        pf.nodes.math.maximum(mountain_0, mountain_1), mountain_2
    )

    mountain_ceiling = 0.5 * pf.nodes.math.maximum(
        pf.nodes.math.maximum(mountain_height_0, mountain_height_1), mountain_height_2
    )
    displacement = pf.nodes.shader.displacement(
        height=height_cracks + height_mid + height_large + height_mountain,
        midlevel=_CEILING_FILL * (0.04 + 0.025 + 0.05 + mountain_ceiling),
        scale=displacement_scale,
    )

    color_mid_blend = pf.nodes.math.map_range(
        value=detail_noise.fac,
        from_min=0.223,
        from_max=0.71,
    )
    color_dark_blend = pf.nodes.math.map_range(
        value=detail_noise.fac,
        from_min=0.71,
        from_max=1.0,
    )
    dirt_color = pf.nodes.color.mix_rgb(
        factor=color_dark_blend,
        a=pf.nodes.color.mix_rgb(
            factor=color_mid_blend,
            a=pf.Color((0.0, 0.0, 0.0)),
            b=color,
        ),
        b=color_dark,
    )
    base_color = pf.nodes.color.mix_rgb(
        factor=cracks,
        a=pf.Color((0.0, 0.0, 0.0)),
        b=dirt_color,
    )

    surface_roughness = pf.nodes.math.map_range(
        value=detail_noise.fac,
        from_min=0.08,
        from_max=0.768,
        to_min=roughness,
        to_max=roughness_high,
    )

    surface = pf.nodes.shader.principled_bsdf(
        base_color=base_color,
        roughness=surface_roughness,
    )

    return pf.Material(surface=surface, displacement=displacement, volume=None)


def _jitter_color(rng: pf.RNG, base: np.ndarray, offset: float) -> pf.Color:
    jitter = np.array(
        [
            pf.random.uniform(rng, -offset, offset),
            pf.random.uniform(rng, -offset, offset),
            pf.random.uniform(rng, -offset, offset),
        ]
    )
    return pf.color.rgb_color(rgb=np.clip(base + jitter, 0.0, 1.0))


def dirt_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    r_color, r_color_dark, r_crack, r_noise, r_mountain, r_w = rng.spawn(6)

    if color is None:
        color = _jitter_color(r_color, np.array([0.18, 0.075, 0.05]), 0.05)

    return dirt(
        vector=vector,
        color=color,
        color_dark=_jitter_color(r_color_dark, np.array([0.19, 0.03, 0.02]), 0.05),
        crack_density=pf.random.uniform(r_crack, 0.0, 0.1),
        crack_scale=pf.random.uniform(r_crack, 5.0, 15.0) * 0.5,
        crack_width=pf.random.uniform(r_crack, 0.01, 0.05),
        mid_noise_scale=pf.random.log_uniform(r_noise, 5.0 * 3 / 4, 5.0 * 4 / 3),
        large_noise_scale=pf.random.log_uniform(r_noise, 20.0 * 3 / 4, 20.0 * 4 / 3),
        mountain_scale_0=pf.random.uniform(r_mountain, 1.0, 5.0),
        mountain_scale_1=pf.random.uniform(r_mountain, 1.0, 5.0),
        mountain_scale_2=pf.random.uniform(r_mountain, 1.0, 5.0),
        mountain_height_0=10 ** pf.random.uniform(r_mountain, -1.0, -0.5),
        mountain_height_1=10 ** pf.random.uniform(r_mountain, -1.0, -0.5),
        mountain_height_2=10 ** pf.random.uniform(r_mountain, -1.0, -0.5),
        w=pf.random.uniform(r_w, 0.0, 10.0),
    )
