# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Ankit Goyal, Mingzhe Wang, Zeyu Ma: original Infinigen cobble stone material (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/materials/terrain/cobble_stone.py)
# - Alexander Raistrick: refactor for Infinigen2
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=9Tq-6HReNEk by Ryan King Art

import colorsys

import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "cobble_stone",
    "cobble_stone_rand",
]


@pf.nodes.node_function
def cobble_stone(
    vector: t.SocketOrVal[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.047, 0.068, 0.069)),
    color_dark: t.SocketOrVal[pf.Color] = pf.Color((0.014, 0.013, 0.014)),
    roughness: t.SocketOrVal[float] = 0.25,
    roughness_high: t.SocketOrVal[float] = 0.75,
    stone_scale: t.SocketOrVal[float] = 6.0,
    stone_randomness: t.SocketOrVal[float] = 0.7,
    stone_depth: t.SocketOrVal[float] = 0.03,
    gap_width: t.SocketOrVal[float] = 0.275,
    warp_scale: t.SocketOrVal[float] = 6.0,
    cell_noise_scale: t.SocketOrVal[float] = 20.0,
    bump_scale: t.SocketOrVal[float] = 20.0,
    bump_depth: t.SocketOrVal[float] = 0.015,
    color_noise_scale: t.SocketOrVal[float] = 0.4,
    displacement_scale: t.SocketOrVal[float] = 1.0,
    w: t.SocketOrVal[float] = 0.0,
) -> pf.Material:
    warp = pf.nodes.texture.noise(
        vector=vector,
        w=w,
        scale=warp_scale,
        noise_dimensions="4D",
    ).fac

    cells = pf.nodes.texture.voronoi(
        vector=vector,
        w=w + 11.0,
        scale=stone_scale,
        randomness=stone_randomness,
        voronoi_dimensions="4D",
    )
    cell_noise = pf.nodes.texture.noise(
        vector=cells.position,
        scale=cell_noise_scale,
    ).fac
    cell_split = pf.nodes.color.rgb_to_bw(
        pf.nodes.color.color_ramp(
            fac=cell_noise,
            points=[
                (0.1159, pf.Color((0.0, 0.0, 0.0))),
                (0.475, pf.Color((1.0, 1.0, 1.0))),
            ],
            interpolation="CONSTANT",
        ).color
    )

    edge_coarse = pf.nodes.texture.voronoi_distance(
        vector=vector,
        w=w + 23.0,
        scale=stone_scale,
        randomness=stone_randomness,
        voronoi_dimensions="4D",
    )
    edge_fine = pf.nodes.texture.voronoi_distance(
        vector=vector,
        w=w + 37.0,
        scale=1.5 * stone_scale,
        randomness=stone_randomness,
        voronoi_dimensions="4D",
    )
    edge = pf.nodes.math.mix(a=edge_coarse, b=edge_fine, factor=cell_split)
    edge_warped = pf.nodes.math.mix(a=edge, b=0.5, factor=warp)

    # v1 colorramp black at gap_width, white at 0.377, linear
    stones = pf.nodes.math.map_range(
        value=edge_warped,
        from_min=gap_width,
        from_max=0.377,
    )

    bump = pf.nodes.texture.noise(
        vector=vector,
        scale=bump_scale,
        detail=10.0,
        distortion=2.0,
    ).fac
    displacement = pf.nodes.shader.displacement(
        height=stones * stone_depth + (bump - 0.5) * bump_depth,
        midlevel=0.0,
        scale=displacement_scale,
    )

    tint_noise = pf.nodes.texture.noise(
        vector=vector,
        w=w + 53.0,
        scale=color_noise_scale,
        noise_dimensions="4D",
    ).fac
    tint = pf.nodes.color.mix_rgb(factor=tint_noise, a=color_dark, b=color)
    base_color = pf.nodes.color.mix_rgb(
        factor=stones,
        a=pf.Color((0.0, 0.0, 0.0)),
        b=tint,
    )
    surface_roughness = pf.nodes.math.map_range(
        value=stones,
        to_min=roughness_high,
        to_max=roughness,
    )
    surface = pf.nodes.shader.principled_bsdf(
        base_color=base_color,
        roughness=surface_roughness,
    )

    return pf.Material(surface=surface, displacement=displacement, volume=None)


def _neighbour_color(
    rng: pf.RNG,
    rgb: tuple[float, float, float],
    hue_diff: float,
    sat_diff: float,
    val_diff: float,
) -> pf.Color:
    hue, sat, val = colorsys.rgb_to_hsv(*rgb)
    return pf.color.hsv_color(
        hue=pf.random.uniform(rng, max(0.0, hue - hue_diff), min(1.0, hue + hue_diff)),
        saturation=pf.random.uniform(
            rng, max(0.0, sat - sat_diff), min(1.0, sat + sat_diff)
        ),
        value=pf.random.uniform(
            rng, max(0.0, val - val_diff), min(1.0, val + val_diff)
        ),
    )


def cobble_stone_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    r_color, r_dark, r_roughness, r_stone, r_bump, r_noise, r_w = rng.spawn(7)

    if color is None:
        color = _neighbour_color(r_color, (0.047, 0.068, 0.069), 0.2, 0.1, 0.1)

    return cobble_stone(
        vector=vector,
        color=color,
        color_dark=_neighbour_color(r_dark, (0.014, 0.013, 0.014), 0.2, 0.1, 0.1),
        roughness=pf.random.clip_gaussian(r_roughness, 0.25, 0.05, 0.1, 0.4),
        roughness_high=pf.random.clip_gaussian(r_roughness, 0.75, 0.05, 0.6, 0.9),
        stone_scale=pf.random.uniform(r_stone, 9.0, 15.0) / 2,
        stone_randomness=pf.random.uniform(r_stone, 0.5, 0.9),
        stone_depth=pf.random.uniform(r_stone, 0.02, 0.04),
        gap_width=pf.random.uniform(r_stone, 0.26, 0.29),
        warp_scale=pf.random.clip_gaussian(r_noise, 6.0, 0.5, 4.5, 7.5),
        cell_noise_scale=pf.random.clip_gaussian(r_noise, 20.0, 2.0, 14.0, 26.0),
        bump_scale=pf.random.clip_gaussian(r_bump, 20.0, 2.0, 14.0, 26.0),
        bump_depth=pf.random.uniform(r_bump, 0.01, 0.02),
        color_noise_scale=pf.random.clip_gaussian(r_noise, 10.0, 1.5, 5.5, 14.5) / 25,
        w=pf.random.uniform(r_w, -5.0, 5.0),
    )
