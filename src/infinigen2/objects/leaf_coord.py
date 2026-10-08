# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import procfunc as pf

LEAF_COORD_UV_LAYER = "UVMap"


def normalize_leaf_coord(obj: pf.MeshObject) -> None:
    coordinate = pf.ops.attr.read_attribute(obj, "coordinate", domain="POINT")
    planar = coordinate[:, :2]
    span = planar.max(axis=0) - planar.min(axis=0)
    span = span + (span == 0.0)
    normalized = (planar - planar.min(axis=0)) / span
    loop_vertices = pf.ops.attr.loop_vertex_indices(obj)
    pf.ops.attr.uv_coords_new(obj, LEAF_COORD_UV_LAYER, do_init=False)
    pf.ops.attr.write_uv_coords(obj, normalized[loop_vertices])


def leaf_coord() -> pf.ProcNode[pf.Vector]:
    """Read the deliberate 0-1 leaf coordinate from the active UV layer."""
    return pf.nodes.shader.coord().uv
