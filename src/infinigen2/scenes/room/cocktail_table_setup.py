# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf

from infinigen2.objects import table
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.room.dining_table_setup import (
    DiningSetupResult,
    DiningTableSetupResult,
    dining_setup_rand,
    table_wall_setup_rand,
)

__all__ = ["cocktail_table_setup_rand", "cocktail_table_wall_setup_rand"]


def cocktail_table_setup_rand(rng: pf.RNG) -> DiningSetupResult:
    """A cocktail table with height-matched tall chairs arranged around it."""
    rng_table, rng_setup = rng.spawn(2)
    cocktail_table = table.cocktail_table_rand(rng_table).mesh
    return dining_setup_rand(rng_setup, dining_table=cocktail_table)


def cocktail_table_wall_setup_rand(
    rng: pf.RNG,
    wall_planes: list[pf.MeshObject],
    room_dimensions: pf.Vector,
    colliders: ccol.CollisionSet,
) -> DiningTableSetupResult:
    """A cocktail table with any side against a wall and tall chairs around it."""
    del room_dimensions
    rng_table, rng_setup = rng.spawn(2)
    cocktail_table = table.cocktail_table_rand(rng_table)
    return table_wall_setup_rand(rng_setup, cocktail_table, wall_planes, colliders)
