# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Ankit Goyal, Mingzhe Wang, Zeyu Ma: original Infinigen stone material (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/materials/terrain/stone.py)
# - Alexander Raistrick: refactor for Infinigen2
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=YKRK82JeBo8 by Ryan King Art

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "stone",
    "stone_rand",
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
        detail=9.0,
        roughness=0.6,
        noise_dimensions="4D",
    )
    return (noise.fac - 0.5) * height


@pf.nodes.node_function
def stone(
    vector: t.SocketOrVal[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.075, 0.075, 0.075)),
    color_low: t.SocketOrVal[pf.Color] = pf.Color((0.025, 0.025, 0.025)),
    color_high: t.SocketOrVal[pf.Color] = pf.Color((0.025, 0.025, 0.025)),
    roughness: t.SocketOrVal[float] = 0.65,
    roughness_high: t.SocketOrVal[float] = 0.75,
    bumps_scale: t.SocketOrVal[float] = 7.5,
    bumps_height: t.SocketOrVal[float] = 0.115,
    crack_density: t.SocketOrVal[float] = 0.05,
    crack_width: t.SocketOrVal[float] = 0.1,
    wave_scale: t.SocketOrVal[float] = 2.0,
    wave_distortion: t.SocketOrVal[float] = 6.0,
    wave_detail: t.SocketOrVal[float] = 15.0,
    mid_noise_scale: t.SocketOrVal[float] = 5.0,
    large_noise_scale: t.SocketOrVal[float] = 20.0,
    mountain_scale_0: t.SocketOrVal[float] = 12.0,
    mountain_scale_1: t.SocketOrVal[float] = 15.0,
    mountain_scale_2: t.SocketOrVal[float] = 18.0,
    mountain_height_0: t.SocketOrVal[float] = 0.01,
    mountain_height_1: t.SocketOrVal[float] = 0.01,
    mountain_height_2: t.SocketOrVal[float] = 0.01,
    displacement_scale: t.SocketOrVal[float] = 1.0,
    w: t.SocketOrVal[float] = 0.0,
) -> pf.Material:
    # v1 Musgrave 4D FBM, dimension 2 and lacunarity 2, so roughness = lacunarity ** -dimension
    bumps = pf.nodes.texture.noise(
        vector=vector,
        w=w,
        scale=bumps_scale,
        detail=2.0,
        roughness=0.25,
        lacunarity=2.0,
        noise_type="FBM",
        normalize=False,
        noise_dimensions="4D",
    ).fac

    detail_noise = pf.nodes.texture.noise(
        vector=vector,
        w=w + 11.0,
        scale=12.0 * 0.5,
        detail=16.0,
        noise_dimensions="4D",
    )

    crack_mask_noise = pf.nodes.texture.noise(
        vector=vector,
        w=w + 23.0,
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
        w=w + 37.0,
        scale=1.0 * 0.5,
        detail=16.0,
        noise_dimensions="4D",
    )
    wave = pf.nodes.texture.wave(
        vector=warp_noise.color.astype(dtype=pf.Vector),
        scale=wave_scale,
        distortion=wave_distortion,
        detail=wave_detail,
    )
    crack_lines = pf.nodes.math.map_range(
        value=wave.fac,
        from_min=0.0,
        from_max=crack_width,
    )
    cracks = pf.nodes.math.mix(a=crack_lines, b=0.5, factor=crack_mask)

    mid_noise = pf.nodes.texture.noise(
        vector=vector,
        w=w + 53.0,
        scale=mid_noise_scale,
        noise_dimensions="4D",
    )
    large_noise = pf.nodes.texture.noise(
        vector=vector,
        w=w + 67.0,
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

    height_fine = 0.08 * (bumps * bumps_height + 0.1 * detail_noise.fac + 0.05 * cracks)
    height_mid = 0.05 * (mid_noise.fac - 0.5)
    height_large = 0.1 * (large_height - 0.5)

    mountain_0 = _mountain_octave(vector, w + 79.0, mountain_scale_0, mountain_height_0)
    mountain_1 = _mountain_octave(vector, w + 97.0, mountain_scale_1, mountain_height_1)
    mountain_2 = _mountain_octave(
        vector, w + 113.0, mountain_scale_2, mountain_height_2
    )
    height_mountain = pf.nodes.math.maximum(
        pf.nodes.math.maximum(mountain_0, mountain_1), mountain_2
    )

    displacement = pf.nodes.shader.displacement(
        height=height_fine + height_mid + height_large + height_mountain,
        midlevel=0.0,
        scale=displacement_scale,
    )

    color_mid_blend = pf.nodes.math.map_range(
        value=detail_noise.fac,
        from_min=0.223,
        from_max=0.509,
    )
    color_high_blend = pf.nodes.math.map_range(
        value=detail_noise.fac,
        from_min=0.509,
        from_max=1.0,
    )
    stone_color = pf.nodes.color.mix_rgb(
        factor=color_high_blend,
        a=pf.nodes.color.mix_rgb(factor=color_mid_blend, a=color_low, b=color),
        b=color_high,
    )
    base_color = pf.nodes.color.mix_rgb(
        factor=cracks,
        a=pf.Color((0.0, 0.0, 0.0)),
        b=stone_color,
    )

    surface_roughness = pf.nodes.math.map_range(
        value=detail_noise.fac,
        from_min=0.082,
        from_max=0.768,
        to_min=roughness,
        to_max=roughness_high,
    )

    surface = pf.nodes.shader.principled_bsdf(
        base_color=base_color,
        roughness=surface_roughness,
    )

    return pf.Material(surface=surface, displacement=displacement, volume=None)


def _stone_gray_rand(rng: pf.RNG, low: float, high: float, jitter: float) -> pf.Color:
    gray = pf.random.uniform(rng, low, high)
    channels = np.clip(
        [
            gray + pf.random.uniform(rng, -jitter, jitter),
            gray + pf.random.uniform(rng, -jitter, jitter),
            gray + pf.random.uniform(rng, -jitter, jitter),
        ],
        0.0,
        1.0,
    )
    return pf.color.rgb_color(rgb=channels)


def stone_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    (
        r_color,
        r_color_low,
        r_color_high,
        r_roughness,
        r_bumps,
        r_crack,
        r_wave,
        r_noise,
        r_mountain,
        r_w,
    ) = rng.spawn(10)

    if color is None:
        color = _stone_gray_rand(r_color, 0.05, 0.1, 0.01)

    return stone(
        vector=vector,
        color=color,
        color_low=_stone_gray_rand(r_color_low, 0.0, 0.05, 0.0),
        color_high=_stone_gray_rand(r_color_high, 0.0, 0.05, 0.01),
        roughness=pf.random.uniform(r_roughness, 0.6, 0.7),
        roughness_high=pf.random.uniform(r_roughness, 0.7, 0.8),
        bumps_scale=pf.random.uniform(r_bumps, 0.0, 30.0) * 0.5,
        bumps_height=pf.random.uniform(r_bumps, 0.08, 0.15),
        crack_density=pf.random.uniform(r_crack, 0.0, 0.1),
        crack_width=pf.random.uniform(r_crack, 0.08, 0.12),
        wave_scale=pf.random.clip_gaussian(r_wave, 2.0, 0.5, 0.5, 3.5),
        wave_distortion=pf.random.clip_gaussian(r_wave, 6.0, 2.0, 0.0, 12.0),
        wave_detail=pf.random.clip_gaussian(r_wave, 15.0, 5.0, 0.0, 15.0),
        mid_noise_scale=pf.random.log_uniform(r_noise, 5.0 * 3 / 4, 5.0 * 4 / 3),
        large_noise_scale=pf.random.log_uniform(r_noise, 20.0 * 3 / 4, 20.0 * 4 / 3),
        mountain_scale_0=pf.random.uniform(r_mountain, 10.0, 20.0),
        mountain_scale_1=pf.random.uniform(r_mountain, 10.0, 20.0),
        mountain_scale_2=pf.random.uniform(r_mountain, 10.0, 20.0),
        mountain_height_0=pf.random.log_uniform(r_mountain, 0.007, 0.013),
        mountain_height_1=pf.random.log_uniform(r_mountain, 0.007, 0.013),
        mountain_height_2=pf.random.log_uniform(r_mountain, 0.007, 0.013),
        w=pf.random.uniform(r_w, 0.0, 10.0),
    )
