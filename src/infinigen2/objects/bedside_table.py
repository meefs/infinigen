# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import procfunc as pf

from infinigen2.objects import desk, storage, table
from infinigen2.util.mesh import center_footprint

__all__ = ["bedside_table_composite_rand"]


def bedside_table_composite_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
) -> table.TableResult:
    """Bedside-sized storage, regular table, or desk with under-top storage."""
    rng_depth, rng_width, rng_height, rng_style, rng_body = rng.spawn(5)
    if dimensions is None:
        dimensions = pf.Vector(
            (
                pf.random.uniform(rng_depth, 0.35, 0.48),
                pf.random.uniform(rng_width, 0.40, 0.47),
                pf.random.uniform(rng_height, 0.55, 0.70),
            )
        )
    body = pf.control.choice(
        rng_style,
        [
            (storage.storage_composite_rand, 1.0),
            (table.dining_table_rand, 1.0),
            (desk.desk_with_top_storage_rand, 1.0),
        ],
    )
    result = body(rng_body, dimensions=dimensions)
    center_footprint(result.mesh)
    return table.TableResult(mesh=result.mesh)
