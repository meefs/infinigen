# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.uv_surface import grid_placement

__all__ = ["UvCutoutResult", "uv_cutout_instances"]


class UvCutoutResult(NamedTuple):
    mesh: pf.ProcNode[pf.MeshObject]
    is_cutout_face: pf.ProcNode[bool]
    holed: pf.ProcNode[pf.MeshObject]
    instances: pf.ProcNode[t.Instances]


def uv_cutout_instances(
    surface: pf.ProcNode[pf.MeshObject],
    uv_field: t.SocketOrVal[pf.Vector],
    grid: grid_placement.GridFromSpacingResult,
    instance: pf.ProcNode[pf.MeshObject] | None,
    footprint: pf.ProcNode[pf.MeshObject] | None,
    face_expand_margin: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    rotation_offset: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    secondary_axis_vector: t.SocketOrVal[pf.Vector] = (0, 0, 1),
    normal_offset: t.SocketOrVal[float] = 0.0,
) -> UvCutoutResult:
    """Remesh a metres-UV surface around footprint holes at each grid point, and
    place `instance` over each hole. Instances are authored with +X along the
    surface normal, +Y along U and +Z along V before `rotation_offset`."""
    faces = grid_placement.faces_for_instance_grid_bboxes(
        target_surface=surface,
        target_uv=uv_field,
        instance=footprint,
        query_grid=grid.grid_mesh,
        instance_uvs=grid.query_uv,
        grid_index_x=grid.index_x,
        grid_index_y=grid.index_y,
        verts_per_instance_x=2,
        verts_per_instance_y=2,
        margin_verts_x=1,
        margin_verts_y=1,
        face_expand_margin=face_expand_margin,
        rotation_offset=rotation_offset,
    )
    holed = pf.nodes.geo.separate_geometry(
        faces.mesh, selection=faces.is_instance_face, domain="FACE"
    ).inverted
    instances = grid_placement.place_instances_on_uv_grid(
        surface=surface,
        uv_field=uv_field,
        grid_mesh=grid.grid_mesh,
        query_uv=grid.query_uv,
        instance=instance,
        secondary_axis_vector=secondary_axis_vector,
        rotation_offset=rotation_offset,
        normal_offset=normal_offset,
    )
    return UvCutoutResult(faces.mesh, faces.is_instance_face, holed, instances)
