# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from typing import NamedTuple

import numpy as np
import procfunc as pf

from infinigen2.objects import bowl, phyllo_plant, vase
from infinigen2.shaders.base_materials import (
    dirt,
    granite,
    gravel_concrete,
    mud,
    sand,
    soil,
)

__all__ = [
    "PlantPotResult",
    "plant_pot",
    "plant_pot_large_rand",
    "plant_pot_small_rand",
]


class PlantPotResult(NamedTuple):
    mesh: pf.MeshObject


@pf.tracer.primitive
def _assemble(
    pot: pf.MeshObject,
    fill: pf.MeshObject,
    plant: pf.MeshObject,
    plant_z: float,
) -> pf.MeshObject:
    plant.item().location.z = plant_z
    for obj, part in ((pot, 0), (plant, 1), (fill, 2)):
        values = np.full(len(obj.item().data.vertices), part, dtype=np.int32)
        pf.ops.attr.write_attribute(
            obj, values, "plant_pot_part", "POINT", overwrite=True
        )
    pf.ops.object.join(plant, pot)
    pf.ops.object.join(plant, fill)
    pf.ops.mesh.transform_apply(plant)
    plant.item().name = "plant_pot"
    return plant


def plant_pot(
    plant: pf.MeshObject,
    base_radius: float = 0.12,
    rim_ratio: float = 1.2,
    height: float = 0.1,
    profile_fullness: float = 0.6,
    profile_slope: float = 0.45,
    rib_count: int = 12,
    rib_depth: float = 0.0,
    twist: float = 0.0,
    material: pf.Material | None = None,
    soil_material: pf.Material | None = None,
) -> PlantPotResult:
    thickness = base_radius * 0.035
    fill_height = height * 0.85
    pot = bowl.pot(
        diameter=2.0 * base_radius * rim_ratio,
        height=height,
        base_scale=1.0 / rim_ratio,
        profile_fullness=profile_fullness,
        profile_slope=profile_slope,
        rib_count=rib_count,
        rib_depth=rib_depth,
        twist=twist,
        thickness=thickness,
        inside_height=fill_height,
        u_resolution=48,
        v_resolution=8,
        material=material,
    )
    if soil_material is None:
        soil_material = granite.granite_no_displacement_preset(
            pf.nodes.shader.coord().object
        )
    pf.ops.object.set_material(pot.inside_volume, soil_material)
    plant_z = fill_height - thickness
    return PlantPotResult(_assemble(pot.mesh, pot.inside_volume, plant, plant_z))


def _phyllo_basal_pot_rand(
    rng: pf.RNG, foliage_radius: float, foliage_height: float
) -> phyllo_plant.PhylloPlantResult:
    rng_size, rng_height, rng_radius, rng_length, rng_width, rng_plant = rng.spawn(6)
    leaf_size = pf.random.uniform(rng_size, 0.65, 1.0)
    stem_height = foliage_height * pf.random.uniform(rng_height, 0.04, 0.16)
    stem_radius = foliage_radius * pf.random.uniform(rng_radius, 0.035, 0.07)
    leaf_reach = foliage_radius - stem_radius
    leaf_length = leaf_reach * pf.random.uniform(rng_length, 0.75, 0.95)
    leaf_length = leaf_length / (1.45 * 1.25 * leaf_size)
    return phyllo_plant.plant_phyllo_basal_rand(
        rng_plant,
        leaf_size=leaf_size,
        leaf_width=pf.random.uniform(rng_width, 0.65, 1.0),
        plant_height=foliage_height,
        stem_height=stem_height,
        stem_radius=stem_radius,
        leaf_length=leaf_length,
        stem_curve_x_degrees=0.0,
        stem_curve_y_degrees=0.0,
        stem_warble=0.0,
    )


def _phyllo_ascending_pot_rand(
    rng: pf.RNG, foliage_radius: float, foliage_height: float
) -> phyllo_plant.PhylloPlantResult:
    rng_size, rng_height, rng_radius, rng_length, rng_width, rng_plant = rng.spawn(6)
    leaf_size = pf.random.uniform(rng_size, 0.65, 1.0)
    stem_height = foliage_height * pf.random.uniform(rng_height, 0.45, 0.7)
    stem_radius = foliage_radius * pf.random.uniform(rng_radius, 0.025, 0.05)
    leaf_reach = foliage_radius - stem_radius
    leaf_length = leaf_reach * pf.random.uniform(rng_length, 0.75, 0.95)
    leaf_length = leaf_length / (1.45 * 1.25 * leaf_size)
    return phyllo_plant.plant_phyllo_ascending_rand(
        rng_plant,
        sparse=False,
        leaf_size=leaf_size,
        leaf_width=pf.random.uniform(rng_width, 0.65, 1.0),
        plant_height=foliage_height,
        stem_height=stem_height,
        stem_radius=stem_radius,
        leaf_length=leaf_length,
        stem_curve_x_degrees=0.0,
        stem_curve_y_degrees=0.0,
        stem_warble=0.0,
    )


def _plant_pot_rand(
    rng: pf.RNG,
    base_radius: float,
    rim_ratio: float,
    height: float,
    foliage_radius: float,
    foliage_height: float,
) -> PlantPotResult:
    (
        rng_choice,
        rng_plant,
        rng_material,
        rng_rib_choice,
        rng_rib_depth,
        rng_twist_choice,
        rng_twist_amount,
        rng_profile_fullness,
        rng_profile_slope,
        rng_rib_count,
        rng_soil_choice,
        rng_soil_material,
    ) = rng.spawn(12)
    plant_fn = pf.control.choice(
        rng_choice,
        [
            (_phyllo_basal_pot_rand, 1.0),
            (_phyllo_ascending_pot_rand, 1.0),
        ],
    )
    plant = plant_fn(rng_plant, foliage_radius, foliage_height).mesh
    material = vase.vase_material_rand(rng_material, pf.nodes.shader.coord().uv)
    decorated_depth = pf.random.uniform(rng_rib_depth, 0.025, 0.07)
    rib_depth = pf.control.choice(rng_rib_choice, [(0.0, 1.0), (decorated_depth, 1.0)])
    twist_amount = pf.random.uniform(rng_twist_amount, -0.6, 0.6)
    twist = pf.control.choice(rng_twist_choice, [(0.0, 2.0), (twist_amount, 1.0)])
    soil_material_fn = pf.control.choice(
        rng_soil_choice,
        [
            (dirt.dirt_rand, 1.0),
            (mud.mud_rand, 1.0),
            (soil.soil_rand, 1.0),
            (sand.sand_rand, 1.0),
            (granite.granite_rand, 0.5),
            (gravel_concrete.gravel_concrete_rand, 0.5),
        ],
    )
    soil_material = soil_material_fn(rng_soil_material, pf.nodes.shader.coord().object)
    soil_material = pf.Material(
        surface=soil_material.surface,
        volume=soil_material.volume,
    )
    return plant_pot(
        plant=plant,
        base_radius=base_radius,
        rim_ratio=rim_ratio,
        height=height,
        profile_fullness=pf.random.uniform(rng_profile_fullness, 0.4, 0.8),
        profile_slope=pf.random.uniform(rng_profile_slope, 0.3, 0.65),
        rib_count=pf.random.randint(rng_rib_count, 6, 13),
        rib_depth=rib_depth,
        twist=twist,
        material=material,
        soil_material=soil_material,
    )


def plant_pot_large_rand(rng: pf.RNG) -> PlantPotResult:
    rng_radius, rng_rim, rng_height, rng_width, rng_foliage, rng_build = rng.spawn(6)
    base_radius = pf.random.log_uniform(rng_radius, 0.085, 0.2)
    rim_ratio = pf.random.uniform(rng_rim, 1.05, 1.25)
    diameter = 2.0 * base_radius * rim_ratio
    return _plant_pot_rand(
        rng_build,
        base_radius=base_radius,
        rim_ratio=rim_ratio,
        height=diameter * pf.random.log_uniform(rng_height, 0.65, 1.05),
        foliage_radius=base_radius * rim_ratio * pf.random.uniform(rng_width, 1.5, 4.0),
        foliage_height=base_radius * pf.random.log_uniform(rng_foliage, 6.0, 16.0),
    )


def plant_pot_small_rand(rng: pf.RNG) -> PlantPotResult:
    rng_radius, rng_rim, rng_height, rng_width, rng_foliage, rng_build = rng.spawn(6)
    base_radius = pf.random.log_uniform(rng_radius, 0.0225, 0.1125)
    rim_ratio = pf.random.uniform(rng_rim, 1.05, 1.3)
    diameter = 2.0 * base_radius * rim_ratio
    return _plant_pot_rand(
        rng_build,
        base_radius=base_radius,
        rim_ratio=rim_ratio,
        height=diameter * pf.random.log_uniform(rng_height, 0.45, 0.9),
        foliage_radius=base_radius * rim_ratio * pf.random.uniform(rng_width, 1.0, 2.5),
        foliage_height=base_radius * pf.random.log_uniform(rng_foliage, 1.0, 6.0),
    )
