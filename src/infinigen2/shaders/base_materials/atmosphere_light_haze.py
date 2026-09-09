# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick, Zeyu Ma: original Infinigen v1 nodegroup (https://github.com/princeton-vl/infinigen/blob/main/infinigen/assets/materials/fluid/atmosphere_light_haze.py)
# - Alexander Raistrick: port to procfunc/v2

import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "atmosphere_light_haze",
    "atmosphere_light_haze_color_rand",
    "atmosphere_light_haze_rand",
]


@pf.nodes.node_function
def atmosphere_light_haze(
    color: t.SocketOrVal[pf.Color],
    density: t.SocketOrVal[float] = 0.003,
    anisotropy: t.SocketOrVal[float] = 0.5,
) -> pf.Material:
    volume = pf.nodes.shader.volume_principled(
        color=color,
        density=density,
        anisotropy=anisotropy,
    )
    return pf.Material(surface=None, displacement=None, volume=volume)


def atmosphere_light_haze_color_rand(rng: pf.RNG) -> pf.Color:
    hue = pf.random.uniform(rng, 0.0, 1.0)
    sat = pf.random.uniform(rng, 0.0, 0.2)
    val = pf.random.uniform(rng, 0.8, 1.0)
    return pf.color.hsv_color(hue=hue, saturation=sat, value=val)


def atmosphere_light_haze_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector] | None = None,
    color: t.SocketOrVal[pf.Color] | None = None,
) -> pf.Material:
    del vector
    rng_color, rng_density = rng.spawn(2)
    if color is None:
        color = atmosphere_light_haze_color_rand(rng_color)

    return atmosphere_light_haze(
        color=color,
        density=pf.random.uniform(rng_density, 0.0, 0.006),
        anisotropy=0.5,
    )
