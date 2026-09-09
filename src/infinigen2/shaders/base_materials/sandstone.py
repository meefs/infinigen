# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Ankit Goyal, Mingzhe Wang, Zeyu Ma: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/main/infinigen/assets/materials/terrain/sandstone.py)
# - Alexander Raistrick: port to procfunc/v2

import procfunc as pf
from procfunc.nodes import types as t

__all__ = ["sandstone", "sandstone_rand"]


@pf.nodes.node_function
def _sandstone_crack(
    vector: t.SocketOrVal[pf.Vector],
    noise_scale: t.SocketOrVal[float] = 2.0,
    voronoi_scale: t.SocketOrVal[float] = 5.0,
    magnitude: t.SocketOrVal[float] = 0.005,
    noise_w: t.SocketOrVal[float] = 0.0,
    voronoi_w: t.SocketOrVal[float] = 0.0,
) -> pf.ProcNode[float]:
    warp = pf.nodes.texture.noise(
        vector=vector,
        scale=noise_scale,
        distortion=1.0,
        w=noise_w,
        noise_dimensions="4D",
    )
    distance = pf.nodes.texture.voronoi_distance(
        vector=warp.color.astype(dtype=pf.Vector),
        scale=voronoi_scale,
        w=voronoi_w,
        voronoi_dimensions="4D",
    )
    edge = pf.nodes.math.map_range(
        value=distance, from_min=0.0, from_max=0.06, to_min=0.0, to_max=1.0
    )
    return (edge - 1.0) * magnitude


@pf.nodes.node_function
def _sandstone_mountain(
    vector: t.SocketOrVal[pf.Vector],
    shift: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    scale: t.SocketOrVal[float] = 15.0,
    zscale: t.SocketOrVal[float] = 0.07,
) -> pf.ProcNode[float]:
    noise = pf.nodes.texture.noise(
        vector=vector + shift, scale=scale, detail=9.0, roughness=0.6
    )
    return (noise.fac - 0.5) * zscale


@pf.nodes.node_function
def sandstone(
    vector: t.SocketOrVal[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.262, 0.057, 0.035)),
    color_bright: t.SocketOrVal[pf.Color] = pf.Color((0.8, 0.225, 0.135)),
    roughness: t.SocketOrVal[float] = 0.9,
    specular_ior_level: t.SocketOrVal[float] = 0.1,
    shade_dark_stop: t.SocketOrVal[float] = 0.28,
    shade_mid_stop: t.SocketOrVal[float] = 0.42,
    shade_light_stop: t.SocketOrVal[float] = 0.58,
    shade_w_1: t.SocketOrVal[float] = 0.0,
    shade_w_2: t.SocketOrVal[float] = 0.0,
    grain_magnitude: t.SocketOrVal[float] = 0.4,
    grain_w_1: t.SocketOrVal[float] = 0.0,
    grain_w_2: t.SocketOrVal[float] = 0.0,
    side_step_magnitude: t.SocketOrVal[float] = 1.0,
    side_step_alpha_x: t.SocketOrVal[float] = 1.0,
    side_step_alpha_y: t.SocketOrVal[float] = 1.0,
    side_step_w: t.SocketOrVal[float] = 0.0,
    side_step_warp_w_x: t.SocketOrVal[float] = 0.0,
    side_step_warp_w_y: t.SocketOrVal[float] = 0.0,
    side_step_mask_w: t.SocketOrVal[float] = 0.0,
    crack_magnitude_1: t.SocketOrVal[float] = 0.006,
    crack_magnitude_2: t.SocketOrVal[float] = 0.006,
    crack_w_1: t.SocketOrVal[float] = 0.0,
    crack_voronoi_w_1: t.SocketOrVal[float] = 0.0,
    crack_w_2: t.SocketOrVal[float] = 0.0,
    crack_voronoi_w_2: t.SocketOrVal[float] = 0.0,
    crack_mask_w: t.SocketOrVal[float] = 0.0,
    stripe_magnitude: t.SocketOrVal[float] = 0.005,
    stripe_scale: t.SocketOrVal[float] = 20.0,
    mountain_scale_1: t.SocketOrVal[float] = 15.0,
    mountain_scale_2: t.SocketOrVal[float] = 15.0,
    mountain_scale_3: t.SocketOrVal[float] = 15.0,
    mountain_zscale_1: t.SocketOrVal[float] = 0.07,
    mountain_zscale_2: t.SocketOrVal[float] = 0.07,
    mountain_zscale_3: t.SocketOrVal[float] = 0.07,
    mountain_shift_1: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    mountain_shift_2: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    mountain_shift_3: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
) -> pf.Material:
    occlusion = pf.nodes.shader.ambient_occlusion()
    occlusion_ramp = pf.nodes.math.map_range(
        value=occlusion.ao, from_min=0.8, from_max=1.0, to_min=0.0, to_max=1.0
    )
    shade_noise_1 = pf.nodes.texture.noise(
        vector=vector, scale=2.0, w=shade_w_1, noise_dimensions="4D"
    )
    shade_noise_2 = pf.nodes.texture.noise(
        vector=vector, scale=10.0, w=shade_w_2, noise_dimensions="4D"
    )
    shade = occlusion_ramp * ((shade_noise_1.fac * shade_noise_2.fac) / 0.5)

    # three-stop grey ramp (dark -> 0.296 -> white) written as two clamped segments
    shade_dark = pf.nodes.math.map_range(
        value=shade,
        from_min=shade_dark_stop,
        from_max=shade_mid_stop,
        to_min=0.0,
        to_max=0.296,
    )
    shade_light = pf.nodes.math.map_range(
        value=shade,
        from_min=shade_mid_stop,
        from_max=shade_light_stop,
        to_min=0.0,
        to_max=0.704,
    )
    base_color = pf.nodes.color.mix_rgb(
        factor=shade_dark + shade_light, a=color, b=color_bright
    )
    surface = pf.nodes.shader.principled_bsdf(
        base_color=base_color,
        roughness=roughness,
        specular_ior_level=specular_ior_level,
    )

    mountain_1 = _sandstone_mountain(
        vector=vector,
        shift=mountain_shift_1,
        scale=mountain_scale_1,
        zscale=mountain_zscale_1,
    )
    mountain_2 = _sandstone_mountain(
        vector=vector,
        shift=mountain_shift_2,
        scale=mountain_scale_2,
        zscale=mountain_zscale_2,
    )
    mountain_3 = _sandstone_mountain(
        vector=vector,
        shift=mountain_shift_3,
        scale=mountain_scale_3,
        zscale=mountain_zscale_3,
    )
    mountain = pf.nodes.math.maximum(
        pf.nodes.math.maximum(mountain_1, mountain_2), mountain_3
    )

    grain_noise_1 = pf.nodes.texture.noise(
        vector=vector, scale=200.0, roughness=0.5, w=grain_w_1, noise_dimensions="4D"
    )
    grain_noise_2 = pf.nodes.texture.noise(
        vector=vector, scale=8.0, detail=0.0, w=grain_w_2, noise_dimensions="4D"
    )
    grain = (
        (grain_noise_1.fac * 0.5 + grain_noise_2.fac * 0.15) * 0.05
    ) * grain_magnitude

    position = pf.nodes.math.separate_xyz(vector)
    side_warp_x = pf.nodes.texture.noise(
        vector=vector, scale=2.0, w=side_step_warp_w_x, noise_dimensions="4D"
    )
    side_warp_y = pf.nodes.texture.noise(
        vector=vector, scale=2.0, w=side_step_warp_w_y, noise_dimensions="4D"
    )
    side_poly = side_step_alpha_x * (
        position.x + side_warp_x.fac * 0.5
    ) + side_step_alpha_y * (position.y + side_warp_y.fac * 0.5)
    side_noise = pf.nodes.texture.noise(
        vector=pf.nodes.math.combine_xyz(x=side_poly, y=side_poly, z=side_poly),
        scale=10.0,
        w=side_step_w,
        noise_dimensions="4D",
    )
    side_mask_noise = pf.nodes.texture.noise(
        vector=vector, scale=2.0, w=side_step_mask_w, noise_dimensions="4D"
    )
    side_mask = pf.nodes.math.map_range(
        value=side_mask_noise.fac, from_min=0.4, from_max=0.6, to_min=0.0, to_max=1.0
    )
    side_step = ((side_noise.fac * side_mask) * 0.02) * side_step_magnitude

    crack_mask_noise = pf.nodes.texture.noise(
        vector=vector, scale=5.0, w=crack_mask_w, noise_dimensions="4D"
    )
    crack_mask = pf.nodes.math.map_range(
        value=crack_mask_noise.fac, from_min=0.4, from_max=0.6, to_min=0.0, to_max=1.0
    )
    cracks = crack_mask * (
        _sandstone_crack(
            vector=vector,
            noise_scale=2.0,
            voronoi_scale=2.0,
            magnitude=crack_magnitude_1,
            noise_w=crack_w_1,
            voronoi_w=crack_voronoi_w_1,
        )
        + _sandstone_crack(
            vector=vector,
            noise_scale=3.0,
            voronoi_scale=3.0,
            magnitude=crack_magnitude_2,
            noise_w=crack_w_2,
            voronoi_w=crack_voronoi_w_2,
        )
    )

    stripe_warp = pf.nodes.texture.noise(vector=vector, scale=1.0).fac * 0.2
    stripe = (
        pf.nodes.texture.wave(
            vector=vector
            + pf.nodes.math.combine_xyz(x=stripe_warp, y=stripe_warp, z=stripe_warp),
            scale=stripe_scale,
            bands_direction="Z",
            wave_profile="SAW",
        ).fac
        * stripe_magnitude
    )

    height = mountain + grain + side_step + cracks + stripe
    displacement = pf.nodes.shader.displacement(height=height, midlevel=0.0)
    return pf.Material(surface=surface, displacement=displacement, volume=None)


def _random_shift(rng: pf.RNG) -> pf.Vector:
    return pf.Vector(
        (
            pf.random.uniform(rng, 0.0, 999.0),
            pf.random.uniform(rng, 0.0, 999.0),
            pf.random.uniform(rng, 0.0, 999.0),
        )
    )


def sandstone_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    (
        r_color,
        r_shade,
        r_grain,
        r_side,
        r_crack,
        r_mountain,
        r_w,
        r_shift_1,
        r_shift_2,
        r_shift_3,
    ) = rng.spawn(10)

    color_bright = color
    if color is None:
        color = pf.color.hsv_color(
            hue=pf.random.uniform(r_color, 0.0, 0.1162),
            saturation=pf.random.uniform(r_color, 0.7664, 0.9664),
            value=pf.random.uniform(r_color, 0.162, 0.362),
        )
        color_bright = pf.color.hsv_color(
            hue=pf.random.uniform(r_color, 0.0, 0.1226),
            saturation=pf.random.uniform(r_color, 0.7313, 0.9313),
            value=pf.random.uniform(r_color, 0.7, 0.9),
        )

    return sandstone(
        vector=vector,
        color=color,
        color_bright=color_bright,
        shade_dark_stop=0.28 + pf.random.uniform(r_shade, -0.1, 0.1),
        shade_mid_stop=0.42 + pf.random.uniform(r_shade, -0.1, 0.1),
        shade_light_stop=0.58 + pf.random.uniform(r_shade, -0.1, 0.1),
        shade_w_1=pf.random.uniform(r_w, 0.0, 10.0),
        shade_w_2=pf.random.uniform(r_w, 0.0, 10.0),
        grain_magnitude=pf.random.uniform(r_grain, 0.1, 0.5),
        grain_w_1=pf.random.uniform(r_w, 0.0, 10.0),
        grain_w_2=pf.random.uniform(r_w, 0.0, 10.0),
        side_step_magnitude=pf.random.uniform(r_side, 0.0, 1.5),
        side_step_alpha_x=pf.random.uniform(r_side, 0.0, 2.0),
        side_step_alpha_y=pf.random.uniform(r_side, 0.0, 2.0),
        side_step_w=pf.random.uniform(r_w, 0.0, 10.0),
        side_step_warp_w_x=pf.random.uniform(r_w, 0.0, 10.0),
        side_step_warp_w_y=pf.random.uniform(r_w, 0.0, 10.0),
        side_step_mask_w=pf.random.uniform(r_w, 0.0, 10.0),
        crack_magnitude_1=pf.random.uniform(r_crack, 0.0, 0.012),
        crack_magnitude_2=pf.random.uniform(r_crack, 0.0, 0.012),
        crack_w_1=pf.random.uniform(r_w, 0.0, 10.0),
        crack_voronoi_w_1=pf.random.uniform(r_w, 0.0, 10.0),
        crack_w_2=pf.random.uniform(r_w, 0.0, 10.0),
        crack_voronoi_w_2=pf.random.uniform(r_w, 0.0, 10.0),
        crack_mask_w=pf.random.uniform(r_w, 0.0, 10.0),
        mountain_scale_1=pf.random.uniform(r_mountain, 10.0, 20.0),
        mountain_scale_2=pf.random.uniform(r_mountain, 10.0, 20.0),
        mountain_scale_3=pf.random.uniform(r_mountain, 10.0, 20.0),
        mountain_zscale_1=pf.random.log_uniform(r_mountain, 0.05, 0.1),
        mountain_zscale_2=pf.random.log_uniform(r_mountain, 0.05, 0.1),
        mountain_zscale_3=pf.random.log_uniform(r_mountain, 0.05, 0.1),
        mountain_shift_1=_random_shift(r_shift_1),
        mountain_shift_2=_random_shift(r_shift_2),
        mountain_shift_3=_random_shift(r_shift_3),
    )
