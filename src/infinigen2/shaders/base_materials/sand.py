# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/main/infinigen/assets/materials/terrain/sand.py)
# - Alexander Raistrick: port to procfunc/v2

import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "sand",
    "sand_color_rand",
    "sand_rand",
]


@pf.nodes.node_function
def _sand_wave(
    vector: t.SocketOrVal[pf.Vector],
    shift: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    wave_scale: t.SocketOrVal[float] = 1.0,
    wave_distortion: t.SocketOrVal[float] = 4.0,
    warp_scale: t.SocketOrVal[float] = 1.0,
    warp_detail: t.SocketOrVal[float] = 9.0,
    noise_scale: t.SocketOrVal[float] = 125.0,
    noise_detail: t.SocketOrVal[float] = 9.0,
    noise_roughness: t.SocketOrVal[float] = 0.9,
) -> pf.ProcNode[float]:
    warp = pf.nodes.texture.noise(vector=vector, scale=warp_scale, detail=warp_detail)
    ripple = pf.nodes.texture.wave(
        vector=vector + shift + warp.color.astype(dtype=pf.Vector),
        scale=wave_scale,
        distortion=wave_distortion,
    )
    ripple_beat = pf.nodes.texture.wave(
        vector=vector + shift + (313.0, 571.0, 149.0),
        scale=wave_scale * 0.98,
        distortion=wave_distortion,
    )
    grain = pf.nodes.texture.noise(
        vector=vector + shift + (677.0, 41.0, 823.0),
        scale=noise_scale,
        detail=noise_detail,
        roughness=noise_roughness,
    )
    dune = pf.nodes.texture.noise(
        vector=vector + shift + (89.0, 907.0, 449.0), scale=0.1
    )
    magnitude = pf.nodes.math.clamp(
        pf.nodes.math.power(1e5, dune.fac - 0.6), min=0.0, max=1.0
    )
    return (ripple.fac + ripple_beat.fac + grain.fac) * magnitude * 0.01


@pf.nodes.node_function
def sand(
    vector: t.SocketOrVal[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.446, 0.189, 0.074)),
    roughness: t.SocketOrVal[float] = 1.0,
    wave_scale_1: t.SocketOrVal[float] = 1.0,
    wave_scale_2: t.SocketOrVal[float] = 1.0,
    wave_scale_3: t.SocketOrVal[float] = 1.0,
    wave_distortion: t.SocketOrVal[float] = 4.0,
    warp_scale: t.SocketOrVal[float] = 1.0,
    warp_detail: t.SocketOrVal[float] = 9.0,
    noise_scale: t.SocketOrVal[float] = 125.0,
    noise_detail: t.SocketOrVal[float] = 9.0,
    noise_roughness: t.SocketOrVal[float] = 0.9,
    shift_1: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    shift_2: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    shift_3: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
) -> pf.Material:
    surface = pf.nodes.shader.principled_bsdf(base_color=color, roughness=roughness)

    height = (
        _sand_wave(
            vector=vector,
            shift=shift_1,
            wave_scale=wave_scale_1,
            wave_distortion=wave_distortion,
            warp_scale=warp_scale,
            warp_detail=warp_detail,
            noise_scale=noise_scale,
            noise_detail=noise_detail,
            noise_roughness=noise_roughness,
        )
        + _sand_wave(
            vector=vector,
            shift=shift_2,
            wave_scale=wave_scale_2,
            wave_distortion=wave_distortion,
            warp_scale=warp_scale,
            warp_detail=warp_detail,
            noise_scale=noise_scale,
            noise_detail=noise_detail,
            noise_roughness=noise_roughness,
        )
        + _sand_wave(
            vector=vector,
            shift=shift_3,
            wave_scale=wave_scale_3,
            wave_distortion=wave_distortion,
            warp_scale=warp_scale,
            warp_detail=warp_detail,
            noise_scale=noise_scale,
            noise_detail=noise_detail,
            noise_roughness=noise_roughness,
        )
    )
    displacement = pf.nodes.shader.displacement(height=height, midlevel=0.0)
    return pf.Material(surface=surface, displacement=displacement, volume=None)


def _sand_shift(rng: pf.RNG) -> pf.Vector:
    return pf.Vector(
        (
            pf.random.uniform(rng, 0.0, 999.0),
            pf.random.uniform(rng, 0.0, 999.0),
            pf.random.uniform(rng, 0.0, 999.0),
        )
    )


def sand_color_rand(rng: pf.RNG) -> pf.Color:
    return pf.color.hsv_color(
        hue=pf.random.uniform(rng, 0.03, 0.13),
        saturation=pf.random.uniform(rng, 0.34, 0.97),
        value=pf.random.uniform(rng, 0.05, 0.79),
    )


def sand_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    r_color, r_scale, r_shift_1, r_shift_2, r_shift_3 = rng.spawn(5)

    if color is None:
        color = sand_color_rand(r_color)

    return sand(
        vector=vector,
        color=color,
        wave_scale_1=pf.random.log_uniform(r_scale, 0.2, 4.0),
        wave_scale_2=pf.random.log_uniform(r_scale, 0.2, 4.0),
        wave_scale_3=pf.random.log_uniform(r_scale, 0.2, 4.0),
        shift_1=_sand_shift(r_shift_1),
        shift_2=_sand_shift(r_shift_2),
        shift_3=_sand_shift(r_shift_3),
    )
