# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Ankit Goyal, Zeyu Ma: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/main/infinigen/assets/materials/fluid/lava.py)
# - Alexander Raistrick: port to procfunc/v2

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.util.coord import coord_warp

__all__ = [
    "lava",
    "lava_rand",
]


@pf.nodes.node_function
def lava(
    vector: t.SocketOrVal[pf.Vector],
    color: t.SocketOrVal[pf.Color] = pf.Color((0.02, 0.02, 0.02)),
    roughness: t.SocketOrVal[float] = 0.75,
    flow_scale: t.SocketOrVal[float] = 0.15,
    flow_stretch_x: t.SocketOrVal[float] = 1.0,
    flow_stretch_y: t.SocketOrVal[float] = 0.35,
    flow_w: t.SocketOrVal[float] = 0.0,
    wave_scale: t.SocketOrVal[float] = 2.0,
    warp_size: t.SocketOrVal[float] = 2.5,
    warp_strength: t.SocketOrVal[float] = 1.5,
    warp_phase: t.SocketOrVal[float] = 0.0,
    warp_strength_2: t.SocketOrVal[float] = 0.8,
    warp_phase_2: t.SocketOrVal[float] = 0.0,
    band_bend: t.SocketOrVal[float] = 6.0,
    wave_height: t.SocketOrVal[float] = 0.05,
    bump_scale: t.SocketOrVal[float] = 4.0,
    bump_w: t.SocketOrVal[float] = 0.0,
    bump_height: t.SocketOrVal[float] = 0.05,
    crack_scale: t.SocketOrVal[float] = 1.0,
    crack_w_1: t.SocketOrVal[float] = 0.0,
    crack_voronoi_w_1: t.SocketOrVal[float] = 0.0,
    crack_low_1: t.SocketOrVal[float] = 0.02,
    crack_high_1: t.SocketOrVal[float] = 0.125,
    crack_w_2: t.SocketOrVal[float] = 0.0,
    crack_voronoi_w_2: t.SocketOrVal[float] = 0.0,
    crack_low_2: t.SocketOrVal[float] = 0.005,
    crack_high_2: t.SocketOrVal[float] = 0.35,
    ridge_influence: t.SocketOrVal[float] = 0.18,
    turbulence: t.SocketOrVal[float] = 0.4,
    ao_gain: t.SocketOrVal[float] = 3.0,
    ao_max: t.SocketOrVal[float] = 0.7,
    crust_low: t.SocketOrVal[float] = 0.3,
    crust_high: t.SocketOrVal[float] = 0.8,
    rock_noise_w: t.SocketOrVal[float] = 0.0,
    emission_strength: t.SocketOrVal[float] = 40.0,
    emission_w: t.SocketOrVal[float] = 0.0,
    temperature_min: t.SocketOrVal[float] = 1250.0,
    temperature_max: t.SocketOrVal[float] = 1750.0,
) -> pf.Material:
    coords = pf.nodes.math.separate_xyz(vector)
    flow_vector = pf.nodes.math.combine_xyz(
        x=coords.x * flow_scale * flow_stretch_x,
        y=coords.y * flow_scale * flow_stretch_y,
        z=coords.z * flow_scale,
    )
    flow = pf.nodes.texture.noise(
        vector=flow_vector,
        w=flow_w,
        detail=6.0,
        roughness=0.55,
        noise_dimensions="4D",
    ).fac
    lava_dir = pf.nodes.math.float_curve(
        factor=1.0,
        value=flow,
        curve=np.array([[0.0, 0.0], [0.25, 0.4937], [0.5818, 0.8625], [1.0, 1.0]]),
    )

    warp_1 = coord_warp(
        vector=vector * wave_scale,
        size=warp_size,
        strength=warp_strength,
        detail=4.0,
        phase=warp_phase,
    )
    warp_2 = coord_warp(
        vector=warp_1.vector,
        size=warp_size * 0.3,
        strength=warp_strength_2,
        detail=4.0,
        phase=warp_phase_2,
    )
    warped = pf.nodes.math.separate_xyz(warp_2.vector)
    band_vector = pf.nodes.math.combine_xyz(
        x=warped.x + lava_dir * band_bend,
        y=warped.y,
        z=warped.z,
    )
    wave = pf.nodes.texture.wave(
        vector=band_vector, scale=1.0, distortion=1.0, detail=0.0
    ).fac
    bump = pf.nodes.texture.voronoi_smooth_f1(
        vector=vector, w=bump_w, scale=bump_scale, voronoi_dimensions="4D"
    ).distance
    relief_amplitude = wave_height + bump_height
    relief = (wave * wave_height + bump * bump_height) / relief_amplitude
    displacement = pf.nodes.shader.displacement(
        height=relief * relief_amplitude, midlevel=0.0
    )

    ao_strength = pf.nodes.math.map_range(
        value=relief_amplitude * wave_scale * ao_gain, to_max=1.0, to_min=0.0
    )
    ao_occlusion = ao_strength * ao_max
    ao_proxy = pf.nodes.math.map_range(
        value=relief, to_min=1.0 - ao_occlusion, to_max=1.0
    )

    crack_vector = vector * crack_scale
    crack_noise_1 = pf.nodes.texture.noise(
        vector=crack_vector,
        w=crack_w_1,
        detail=16.0,
        distortion=2.0,
        noise_dimensions="4D",
    ).fac
    crack_1 = pf.nodes.texture.voronoi_distance(
        vector=crack_noise_1.astype(dtype=pf.Vector),
        w=crack_voronoi_w_1,
        scale=10.0,
        voronoi_dimensions="4D",
    )
    crack_mask_1 = pf.nodes.math.map_range(
        value=crack_1, from_min=crack_low_1, from_max=crack_high_1
    )

    crack_noise_2 = pf.nodes.texture.noise(
        vector=crack_vector, w=crack_w_2, distortion=2.0, noise_dimensions="4D"
    ).fac
    crack_2 = pf.nodes.texture.voronoi_distance(
        vector=crack_noise_2.astype(dtype=pf.Vector),
        w=crack_voronoi_w_2,
        scale=10.0,
        voronoi_dimensions="4D",
    )
    crack_mask_2 = pf.nodes.math.map_range(
        value=crack_2, from_min=crack_low_2, from_max=crack_high_2
    )

    ridges = pf.nodes.math.map_range(
        value=wave, to_min=ridge_influence * -1.0, to_max=ridge_influence
    )
    crust = (crack_mask_1 + crack_mask_2) * 0.5 + ridges
    crevice = pf.nodes.math.map_range(
        value=crust, to_min=ao_proxy - turbulence, to_max=ao_proxy
    )
    rock_mask = pf.nodes.math.map_range(
        value=crevice + ao_occlusion * 0.5,
        from_min=crust_low,
        from_max=crust_high,
    )

    temperature = temperature_min + (1.0 - lava_dir) * (
        temperature_max - temperature_min
    )
    emission_noise = pf.nodes.texture.noise(
        vector=vector, w=emission_w, scale=0.5, noise_dimensions="4D"
    ).fac
    emission = pf.nodes.shader.emission(
        color=pf.nodes.color.blackbody(temperature=temperature),
        strength=emission_noise + emission_strength,
    )

    rock_noise = pf.nodes.texture.noise(
        vector=vector, w=rock_noise_w, scale=0.5, detail=10.0, noise_dimensions="4D"
    ).fac
    rock = pf.nodes.shader.principled_bsdf(
        base_color=pf.nodes.color.mix_rgb(
            factor=rock_noise, a=pf.Color((0.0, 0.0, 0.0)), b=color
        ),
        roughness=roughness,
    )

    surface = pf.nodes.shader.mix_shader(factor=rock_mask, a=emission, b=rock)

    return pf.Material(
        surface=surface,
        displacement=displacement,
        volume=None,
    )


def lava_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
    color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    rng_rock, rng_flow, rng_disp, rng_crack, rng_emit = rng.spawn(5)

    if color is None:
        color = pf.color.hsv_color(
            hue=pf.random.uniform(rng_rock, 0.0, 1.0),
            saturation=pf.random.uniform(rng_rock, 0.0, 0.02),
            value=pf.random.uniform(rng_rock, 0.0, 0.05),
        )

    temperature_min = pf.random.uniform(rng_emit, 1000.0, 1500.0)
    crust_low = pf.random.uniform(rng_crack, 0.44, 0.58)

    return lava(
        vector=vector,
        color=color,
        roughness=pf.random.uniform(rng_rock, 0.6, 0.9),
        flow_scale=pf.random.uniform(rng_flow, 0.08, 0.25),
        flow_stretch_x=pf.random.uniform(rng_flow, 0.6, 1.4),
        flow_stretch_y=pf.random.uniform(rng_flow, 0.2, 0.6),
        flow_w=pf.random.uniform(rng_flow, 0.0, 10.0),
        wave_scale=pf.random.uniform(rng_disp, 1.5, 3.5),
        warp_size=pf.random.uniform(rng_disp, 1.5, 4.0),
        warp_strength=pf.random.uniform(rng_disp, 1.1, 1.9),
        warp_phase=pf.random.uniform(rng_disp, 0.0, 10.0),
        warp_strength_2=pf.random.uniform(rng_disp, 0.5, 1.0),
        warp_phase_2=pf.random.uniform(rng_disp, 0.0, 10.0),
        band_bend=pf.random.uniform(rng_disp, 4.0, 9.0),
        wave_height=pf.random.uniform(rng_disp, 0.03, 0.08),
        bump_scale=pf.random.uniform(rng_disp, 2.0, 6.0),
        bump_w=pf.random.uniform(rng_disp, 0.0, 10.0),
        bump_height=pf.random.uniform(rng_disp, 0.02, 0.06),
        crack_scale=pf.random.uniform(rng_crack, 0.7, 1.5),
        crack_w_1=pf.random.uniform(rng_crack, 0.0, 10.0),
        crack_voronoi_w_1=pf.random.uniform(rng_crack, 0.0, 10.0),
        crack_low_1=pf.random.uniform(rng_crack, 0.01, 0.03),
        crack_high_1=pf.random.uniform(rng_crack, 0.1, 0.15),
        crack_w_2=pf.random.uniform(rng_crack, 0.0, 10.0),
        crack_voronoi_w_2=pf.random.uniform(rng_crack, 0.0, 10.0),
        crack_low_2=pf.random.uniform(rng_crack, 0.0, 0.01),
        crack_high_2=pf.random.uniform(rng_crack, 0.25, 0.45),
        ridge_influence=pf.random.uniform(rng_crack, 0.1, 0.25),
        turbulence=pf.random.uniform(rng_crack, 0.25, 0.6),
        ao_gain=pf.random.uniform(rng_disp, 2.0, 4.0),
        ao_max=pf.random.uniform(rng_disp, 0.4, 0.75),
        crust_low=crust_low,
        crust_high=crust_low + pf.random.uniform(rng_crack, 0.28, 0.40),
        rock_noise_w=pf.random.uniform(rng_rock, 0.0, 10.0),
        emission_strength=pf.random.uniform(rng_emit, 20.0, 60.0),
        emission_w=pf.random.uniform(rng_emit, 0.0, 10.0),
        temperature_min=temperature_min,
        temperature_max=temperature_min + pf.random.uniform(rng_emit, 0.0, 1000.0),
    )
