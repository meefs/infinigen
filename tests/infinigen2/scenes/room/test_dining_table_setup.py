# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np
import procfunc as pf
from mathutils import Euler, Matrix

from infinigen2.scenes.room import dining_table_setup


def _rectangular_cube(dimensions: tuple[float, float, float]) -> pf.MeshObject:
    obj = pf.ops.primitives.mesh_cube(size=1.0)
    pf.ops.mesh.transform(obj, scale=dimensions)
    return obj


def test_dining_chairs_follow_full_rigid_table_pose() -> None:
    table = _rectangular_cube((2.4, 1.2, 0.8))
    chair = _rectangular_cube((0.4, 0.4, 0.8))
    base = dining_table_setup.arrange_dining_chairs(
        chair,
        table,
        chair_spacing=0.2,
        tuck=0.05,
        edge_margin=0.1,
        disorder=0.0,
    )
    base_vertices = [
        pf.ops.attr.vertex_positions(obj, global_coords=True) for obj in base
    ]
    pose = Matrix.Translation((2.0, -1.0, 1.5))
    pose @= Euler((0.2, -0.15, 0.6)).to_matrix().to_4x4()
    table.item().matrix_world = pose

    transformed = dining_table_setup.arrange_dining_chairs(
        chair,
        table,
        chair_spacing=0.2,
        tuck=0.05,
        edge_margin=0.1,
        disorder=0.0,
    )

    assert len(transformed) == len(base)
    for vertices, transformed_obj in zip(base_vertices, transformed, strict=True):
        homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
        expected = homogeneous @ np.asarray(pose).T
        actual = pf.ops.attr.vertex_positions(transformed_obj, global_coords=True)
        np.testing.assert_allclose(actual, expected[:, :3], atol=1e-5)
