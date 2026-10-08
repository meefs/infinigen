# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Ankit Goyal, Zeyu Ma, Lingjie Mei: original Infinigen soil material (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/materials/terrain/soil.py)
# - Alexander Raistrick: refactor for Infinigen2

import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "soil",
    "soil_rand",
]


def _pebbles(
    vector: t.SocketOrVal[pf.Vector],
    scale: t.SocketOrVal[float],
    noise_mag: t.SocketOrVal[float],
    noise_w: t.SocketOrVal[float],
    voronoi_w: t.SocketOrVal[float],
) -> pf.ProcNode[float]:
    warp = pf.nodes.texture.noise(
        vector=vector,
        w=noise_w,
        scale=scale,
        noise_dimensions="4D",
    )
    warped = warp.color.astype(dtype=pf.Vector) * noise_mag + vector
    cells = pf.nodes.texture.voronoi(
        vector=warped,
        w=voronoi_w,
        scale=scale,
        voronoi_dimensions="4D",
    )
    return cells.distance


def _pebble_profile(
    distance: pf.ProcNode[float],
    roundness: t.SocketOrVal[float],
    amount: t.SocketOrVal[float],
) -> pf.ProcNode[float]:
    # v1 ramp (0, r) -> (amount/8, r/2) -> (amount, 0) as two clamped segments, stops are sampled
    shoulder = pf.nodes.math.map_range(
        value=distance,
        from_min=0.0,
        from_max=amount / 8,
        to_min=roundness,
        to_max=roundness / 2,
    )
    falloff = pf.nodes.math.map_range(
        value=distance,
        from_min=amount / 8,
        from_max=amount,
        to_min=0.0,
        to_max=-roundness / 2,
    )
    return shoulder + falloff


@pf.nodes.node_function
def soil(
    vector: t.SocketOrVal[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.1867, 0.0733, 0.028)),
    color_secondary: t.SocketOrVal[pf.Color] = pf.Color((0.1467, 0.0604, 0.0233)),
    pebble_color: t.SocketOrVal[pf.Color] = pf.Color((0.3813, 0.1714, 0.0782)),
    pebble_color_secondary: t.SocketOrVal[pf.Color] = pf.Color((0.314, 0.1274, 0.0578)),
    roughness: t.SocketOrVal[float] = 1.0,
    shade_w: t.SocketOrVal[float] = 5.0,
    pebble1_scale: t.SocketOrVal[float] = 3.5,
    pebble1_noise_factor: t.SocketOrVal[float] = 1.75,
    pebble1_noise_w: t.SocketOrVal[float] = 5.0,
    pebble1_voronoi_w: t.SocketOrVal[float] = 5.0,
    pebble1_roundness: t.SocketOrVal[float] = 0.75,
    pebble1_amount: t.SocketOrVal[float] = 0.35,
    pebble2_scale: t.SocketOrVal[float] = 7.0,
    pebble2_noise_factor: t.SocketOrVal[float] = 1.75,
    pebble2_noise_w: t.SocketOrVal[float] = 5.0,
    pebble2_voronoi_w: t.SocketOrVal[float] = 5.0,
    pebble2_roundness: t.SocketOrVal[float] = 0.55,
    pebble2_amount: t.SocketOrVal[float] = 0.35,
    pebble3_scale: t.SocketOrVal[float] = 15.0,
    pebble3_noise_mag: t.SocketOrVal[float] = 0.2,
    pebble3_noise_w: t.SocketOrVal[float] = 5.0,
    pebble3_voronoi_w: t.SocketOrVal[float] = 5.0,
    displacement_height: t.SocketOrVal[float] = 0.1,
) -> pf.Material:
    big_stone = _pebble_profile(
        _pebbles(
            vector,
            scale=pebble1_scale,
            noise_mag=pebble1_noise_factor / pebble1_scale,
            noise_w=pebble1_noise_w,
            voronoi_w=pebble1_voronoi_w,
        ),
        roundness=pebble1_roundness,
        amount=pebble1_amount,
    )
    small_stone = _pebble_profile(
        _pebbles(
            vector,
            scale=pebble2_scale,
            noise_mag=pebble2_noise_factor / pebble2_scale,
            noise_w=pebble2_noise_w,
            voronoi_w=pebble2_voronoi_w,
        ),
        roundness=pebble2_roundness,
        amount=pebble2_amount,
    )
    grit = pf.nodes.math.map_range(
        value=_pebbles(
            vector,
            scale=pebble3_scale,
            noise_mag=pebble3_noise_mag,
            noise_w=pebble3_noise_w,
            voronoi_w=pebble3_voronoi_w,
        ),
        from_min=0.0,
        from_max=0.9,
        to_min=0.15,
        to_max=0.0,
    )

    height = displacement_height * (big_stone + small_stone + grit)
    displacement = pf.nodes.shader.displacement(height=height, midlevel=0.0)

    occlusion = pf.nodes.shader.ambient_occlusion()
    occlusion_ramp = pf.nodes.math.map_range(
        value=occlusion.ao,
        from_min=0.8,
        from_max=1.0,
        to_min=0.0,
        to_max=1.0,
    )
    shade_noise = pf.nodes.texture.noise(
        vector=vector,
        w=shade_w,
        scale=10.0,
        noise_dimensions="4D",
    )
    shade = occlusion_ramp * 0.5 + shade_noise.fac * 0.5

    soil_color = pf.nodes.color.mix_rgb(factor=shade, a=color, b=color_secondary)
    stone_color = pf.nodes.color.mix_rgb(
        factor=big_stone, a=pebble_color, b=pebble_color_secondary
    )
    base_color = pf.nodes.color.mix_rgb(factor=big_stone, a=soil_color, b=stone_color)

    roughness_ramp = pf.nodes.color.color_ramp(
        fac=1.0 - big_stone,
        interpolation="LINEAR",
        points=[
            (0.0, (0.5, 0.5, 0.5, 1.0)),
            (0.8636, (0.7, 0.7, 0.7, 1.0)),
            (0.9427, (0.95, 0.95, 0.95, 1.0)),
            (1.0, (0.98, 0.98, 0.98, 1.0)),
        ],
    )

    surface = pf.nodes.shader.principled_bsdf(
        base_color=base_color,
        specular_ior_level=0.2,
        roughness=roughness_ramp.color.astype(dtype=float) * roughness,
    )

    return pf.Material(surface=surface, displacement=displacement, volume=None)


def soil_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    (
        r_color,
        r_color_secondary,
        r_pebble_color,
        r_pebble_color_secondary,
        r_shade,
        r_pebble1,
        r_pebble2,
        r_pebble3,
    ) = rng.spawn(8)

    if color is None:
        color = pf.color.hsv_color(
            hue=pf.random.uniform(r_color, 0.0, 0.09762),
            saturation=pf.random.uniform(r_color, 0.75, 0.95),
            value=pf.random.uniform(r_color, 0.08667, 0.28667),
        )
    color_secondary = pf.color.hsv_color(
        hue=pf.random.uniform(r_color_secondary, 0.00009, 0.10009),
        saturation=pf.random.uniform(r_color_secondary, 0.74091, 0.94091),
        value=pf.random.uniform(r_color_secondary, 0.04667, 0.24667),
    )
    pebble_color = pf.color.hsv_color(
        hue=pf.random.uniform(r_pebble_color, 0.0, 0.15125),
        saturation=pf.random.uniform(r_pebble_color, 0.69491, 0.89491),
        value=pf.random.uniform(r_pebble_color, 0.2813, 0.4813),
    )
    pebble_color_secondary = pf.color.hsv_color(
        hue=pf.random.uniform(r_pebble_color_secondary, 0.0, 0.14528),
        saturation=pf.random.uniform(r_pebble_color_secondary, 0.71592, 0.91592),
        value=pf.random.uniform(r_pebble_color_secondary, 0.214, 0.414),
    )

    return soil(
        vector,
        color=color,
        color_secondary=color_secondary,
        pebble_color=pebble_color,
        pebble_color_secondary=pebble_color_secondary,
        shade_w=pf.random.uniform(r_shade, 0.0, 10.0),
        pebble1_scale=pf.random.uniform(r_pebble1, 2.0, 5.0),
        pebble1_noise_factor=pf.random.uniform(r_pebble1, 1.5, 2.0),
        pebble1_noise_w=pf.random.uniform(r_pebble1, 0.0, 10.0),
        pebble1_voronoi_w=pf.random.uniform(r_pebble1, 0.0, 10.0),
        pebble1_roundness=pf.random.uniform(r_pebble1, 0.5, 1.0),
        pebble1_amount=pf.random.uniform(r_pebble1, 0.2, 0.5),
        pebble2_scale=pf.random.uniform(r_pebble2, 5.0, 9.0),
        pebble2_noise_factor=pf.random.uniform(r_pebble2, 1.5, 2.0),
        pebble2_noise_w=pf.random.uniform(r_pebble2, 0.0, 10.0),
        pebble2_voronoi_w=pf.random.uniform(r_pebble2, 0.0, 10.0),
        pebble2_roundness=pf.random.uniform(r_pebble2, 0.3, 0.8),
        pebble2_amount=pf.random.uniform(r_pebble2, 0.2, 0.5),
        pebble3_scale=pf.random.uniform(r_pebble3, 12.0, 18.0),
        pebble3_noise_mag=pf.random.uniform(r_pebble3, 0.05, 0.35),
        pebble3_noise_w=pf.random.uniform(r_pebble3, 0.0, 10.0),
        pebble3_voronoi_w=pf.random.uniform(r_pebble3, 0.0, 10.0),
    )
