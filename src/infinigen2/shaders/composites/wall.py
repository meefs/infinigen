# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf

from infinigen2.shaders.base_materials import (
    ceramic,
    concrete,
    gravel_concrete,
    paint,
    stone_smooth,
)
from infinigen2.shaders.composites import tiles
from infinigen2.shaders.masks import cracks
from infinigen2.shaders.masks.tile_shapes import (
    tile_coord_transform_rand,
    tile_mask_rand,
)

__all__ = ["wall_flaked_tile_rand"]


def _substrate_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    rng_choice, rng_material = rng.spawn(2)
    material_func = pf.control.choice(
        rng_choice,
        [
            (paint.paint_rand, 3.0),
            (concrete.concrete_rand, 1.0),
            (stone_smooth.stone_smooth_rand, 0.5),
            (gravel_concrete.gravel_concrete_rand, 0.5),
        ],
    )
    return material_func(rng_material, vector)


def _ceramic_tile_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    rng_choice, rng_material = rng.spawn(2)
    material_func = pf.control.choice(
        rng_choice,
        [
            (ceramic.ceramic_rand, 1.0),
            (tiles.ceramic_colored_rand, 2.0),
        ],
    )
    return material_func(rng_material, vector)


def _tile_layer_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> tuple[pf.Material, pf.ProcNode[pf.Vector]]:
    rng_scale, rng_transform, rng_mask, rng_shift, rng_tile, rng_material = rng.spawn(6)
    scale = 1.0 / pf.random.clip_gaussian(rng_scale, 0.2, 0.4, 0.04, 1.0)
    mask_vector = tile_coord_transform_rand(rng_transform, vector, scale=scale)
    tile_mask = tile_mask_rand(rng_mask, mask_vector)
    tile_vector = tiles.shifted_vector_rand(rng_shift, vector, tile_mask)
    tile = _ceramic_tile_rand(rng_tile, tile_vector)
    material = tiles.tile_indoor_wall_rand(
        rng_material,
        vector,
        tile=tile,
        tile_mask=tile_mask,
        scale=scale,
    )
    scale_vector = pf.nodes.math.combine_xyz(x=scale, y=scale, z=scale)
    return material, tile_mask.cell_center_pos / scale_vector


def _displacement_overwrite_layer_material(
    substrate: pf.Material,
    layer: pf.Material,
    mask: pf.ProcNode[float],
    substrate_recess: float,
) -> pf.Material:
    surface = pf.nodes.shader.mix_shader(
        factor=mask,
        a=substrate.surface,
        b=layer.surface,
    )
    recess = pf.nodes.shader.displacement(height=-substrate_recess, midlevel=0.0)
    recessed_substrate = substrate.displacement + recess
    displacement = pf.nodes.math.mix(
        factor=mask,
        a=recessed_substrate,
        b=layer.displacement,
    )
    return pf.Material(surface=surface, displacement=displacement)


def wall_flaked_tile_rand(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector],
) -> pf.Material:
    rng_substrate, rng_tile, rng_mask, rng_recess, rng_flake_size = rng.spawn(5)
    substrate = _substrate_rand(rng_substrate, vector)
    tile, cell_center = _tile_layer_rand(rng_tile, vector)
    flake_size = pf.random.uniform(rng_flake_size, 0.225, 7.5)
    mask = cracks.flake_mask_rand(
        rng_mask,
        cell_center,
        flake_size=flake_size,
    ).mask
    substrate_recess = pf.random.uniform(rng_recess, 0.001, 0.02)
    return _displacement_overwrite_layer_material(
        substrate,
        tile,
        mask,
        substrate_recess,
    )
