# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Mingzhe Wang: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/main/infinigen/assets/materials/terrain/mud.py)
# - Alexander Raistrick: port to procfunc/v2

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "mud",
    "mud_rand",
]

_CRACK_CURVE = np.array([[0.0, 0.0], [0.3386, 0.0844], [0.8114, 0.6312], [1.0, 0.7656]])


# Summed noise octaves peak well below their analytic bound; this is the measured
# fraction they actually reach, so the anchor sits on the real surface.
_CEILING_FILL = 0.88


@pf.nodes.node_function
def mud(
    vector: t.SocketOrVal[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.0216, 0.0145, 0.0113)),
    color_light: t.SocketOrVal[pf.Color] = pf.Color((0.0424, 0.0308, 0.0142)),
    color_noise_scale: t.SocketOrVal[float] = 5.0,
    color_noise_w: t.SocketOrVal[float] = 9.6366,
    roughness: t.SocketOrVal[float] = 0.5,
    roughness_min: t.SocketOrVal[float] = 0.1,
    wet_scale: t.SocketOrVal[float] = 0.2,
    wet_w: t.SocketOrVal[float] = 0.0,
    wet_threshold: t.SocketOrVal[float] = 0.1045,
    specular_wet: t.SocketOrVal[float] = 0.9,
    specular_dry: t.SocketOrVal[float] = 0.7,
    warp: t.SocketOrVal[float] = 0.6,
    bump_scale: t.SocketOrVal[float] = 50.0,
    crack_scale: t.SocketOrVal[float] = 3.0,
    crack_height: t.SocketOrVal[float] = 2.0,
    displacement_scale: t.SocketOrVal[float] = 0.04,
) -> pf.Material:
    color_noise = pf.nodes.texture.noise(
        vector=vector,
        w=color_noise_w,
        scale=color_noise_scale,
        noise_dimensions="4D",
    )
    base_color = pf.nodes.color.mix_rgb(
        factor=color_noise.fac,
        a=color,
        b=color_light,
    )

    # v1 Musgrave dimension 2, lacunarity 2, so roughness = lacunarity ** -dimension
    ridges = pf.nodes.texture.noise(
        vector=vector,
        w=wet_w,
        scale=wet_scale,
        detail=2.0,
        roughness=0.25,
        lacunarity=2.0,
        noise_type="RIDGED_MULTIFRACTAL",
        normalize=False,
        noise_dimensions="4D",
    ).fac
    dry_mask = pf.nodes.math.map_range(
        value=ridges,
        from_min=0.0,
        from_max=wet_threshold,
        to_min=1.0,
        to_max=0.0,
    )

    specular = pf.nodes.math.map_range(
        value=dry_mask,
        to_min=specular_wet,
        to_max=specular_dry,
    )
    dry_roughness = pf.nodes.math.map_range(
        value=color_noise.fac,
        to_min=roughness_min,
        to_max=roughness,
    )

    surface = pf.nodes.shader.principled_bsdf(
        base_color=base_color,
        specular_ior_level=specular,
        roughness=dry_mask * dry_roughness,
    )

    warp_noise = pf.nodes.texture.noise(vector=vector)
    warped = pf.nodes.color.mix_rgb(
        factor=warp,
        a=warp_noise.color,
        b=vector.astype(dtype=pf.Color),
    ).astype(dtype=pf.Vector)

    bumps = pf.nodes.texture.noise(vector=warped, scale=bump_scale).fac
    cracks = pf.nodes.color.color_ramp(
        fac=pf.nodes.texture.voronoi(vector=warped, scale=crack_scale).distance,
        points=[(0.0, pf.Color((1.0, 1.0, 1.0))), (1.0, pf.Color((0.0, 0.0, 0.0)))],
    ).color.astype(dtype=float)
    crack_profile = pf.nodes.math.float_curve(
        factor=1.0,
        value=cracks,
        curve=_CRACK_CURVE,
    )
    height = (bumps + crack_profile * crack_height) * displacement_scale

    ceiling = (
        _CEILING_FILL
        * (1.0 + _CRACK_CURVE[:, 1].max() * crack_height)
        * displacement_scale
    )
    displacement = pf.nodes.shader.displacement(height=height, midlevel=ceiling)

    return pf.Material(
        surface=surface,
        displacement=displacement,
        volume=None,
    )


def _jitter_channel(rng: pf.RNG, value: float) -> float:
    return pf.random.clip_gaussian(
        rng, value, 0.005, max(0.0, value - 0.015), value + 0.015
    )


def _mud_color_rand(rng: pf.RNG, r: float, g: float, b: float) -> pf.Color:
    return pf.Color(
        (
            _jitter_channel(rng, r),
            _jitter_channel(rng, g),
            _jitter_channel(rng, b),
        )
    )


def mud_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    r_color, r_shader, r_disp = rng.spawn(3)

    if color is None:
        color = _mud_color_rand(r_color, 0.0216, 0.0145, 0.0113)

    return mud(
        vector=vector,
        color=color,
        color_light=_mud_color_rand(r_color, 0.0424, 0.0308, 0.0142),
        color_noise_scale=pf.random.clip_gaussian(r_shader, 5.0, 0.5, 3.5, 6.5),
        roughness=pf.random.uniform(r_shader, 0.45, 0.55),
        roughness_min=pf.random.uniform(r_shader, 0.05, 0.15),
        wet_w=pf.random.uniform(r_shader, -10.0, 10.0),
        wet_threshold=pf.random.clip_gaussian(r_shader, 0.1045, 0.01, 0.0745, 0.1345),
        specular_wet=pf.random.uniform(r_shader, 0.85, 0.95),
        specular_dry=pf.random.uniform(r_shader, 0.65, 0.75),
        warp=pf.random.clip_gaussian(r_disp, 0.6, 0.1, 0.3, 0.9),
        bump_scale=pf.random.clip_gaussian(r_disp, 50.0, 5.0, 35.0, 65.0),
        crack_scale=pf.random.clip_gaussian(r_disp, 3.0, 0.5, 1.5, 4.5),
        crack_height=pf.random.clip_gaussian(r_disp, 2.0, 0.2, 1.4, 2.6),
        displacement_scale=pf.random.clip_gaussian(r_disp, 0.04, 0.005, 0.025, 0.055),
    )
