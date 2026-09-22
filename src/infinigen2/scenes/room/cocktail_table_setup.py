# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf

from infinigen2.objects import table
from infinigen2.scenes.room.dining_table_setup import (
    DiningSetupResult,
    dining_setup_rand,
)

__all__ = ["cocktail_table_setup_rand"]


def cocktail_table_setup_rand(rng: pf.RNG) -> DiningSetupResult:
    """A cocktail table with height-matched tall chairs arranged around it."""
    rng_table, rng_setup = rng.spawn(2)
    cocktail_table = table.cocktail_table_rand(rng_table).mesh
    return dining_setup_rand(rng_setup, dining_table=cocktail_table)
