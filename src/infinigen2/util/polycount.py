# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf

__all__ = ["estimated_eval_tricount"]


def _estimated_object_tricount(obj: pf.Object) -> int:
    """Estimate one object's render-level triangle count without evaluating it.

    Subdivision turns each n-gon into n quads at the first level and quadruples
    thereafter, so the render-level count follows from the corner count alone. Exact for
    a base mesh plus deferred subsurf; deformation-only modifiers do not affect it.
    """
    item = obj.item()
    if item.type != "MESH":
        return 0
    mesh = item.data
    levels = sum(
        modifier.render_levels
        for modifier in item.modifiers
        if modifier.type == "SUBSURF"
    )
    if levels:
        return 2 * len(mesh.loops) * 4 ** (levels - 1)
    return len(mesh.loops) - 2 * len(mesh.polygons)


def _geometry_key(obj: pf.Object) -> int:
    item = obj.item()
    if item.type == "MESH" and not item.modifiers:
        return item.data.as_pointer()
    return item.as_pointer()


def estimated_eval_tricount(
    objects: list[pf.Object],
) -> tuple[int, list[tuple[pf.Object, int]]]:
    """Estimate total and ascending per-object render-level triangle counts.

    Modifier-free objects sharing mesh data are instanced by Cycles, so the total counts
    their geometry once; objects with modifiers evaluate separately and each count.
    """
    object_tris = [(obj, _estimated_object_tricount(obj)) for obj in objects]
    object_tris.sort(key=lambda entry: entry[1])
    unique = {_geometry_key(obj): count for obj, count in object_tris}
    return sum(unique.values()), object_tris
