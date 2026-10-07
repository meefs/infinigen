# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bpy
import numpy as np
import procfunc as pf

from infinigen2.scenes import floating_objects


def _cube_without_uvs() -> pf.MeshObject:
    obj = pf.ops.primitives.mesh_cube()
    mesh = obj.item().data
    for layer in list(mesh.uv_layers):
        mesh.uv_layers.remove(layer)
    return obj


def test_override_material_preserves_existing_uv_map() -> None:
    bpy.ops.wm.read_homefile(use_empty=True)
    obj = _cube_without_uvs()
    mesh = obj.item().data
    layer = mesh.uv_layers.new(name="AuthoredUV")
    authored_uvs = np.arange(len(layer.data) * 2, dtype=float).reshape(-1, 2)
    layer.data.foreach_set("uv", authored_uvs.ravel())

    floating_objects._override_material(np.random.default_rng(0), obj)

    assert [layer.name for layer in mesh.uv_layers] == ["AuthoredUV"]
    actual_uvs = np.array([loop.uv[:] for loop in layer.data])
    np.testing.assert_allclose(actual_uvs, authored_uvs)


def test_override_material_creates_uv_map_when_missing() -> None:
    bpy.ops.wm.read_homefile(use_empty=True)
    obj = _cube_without_uvs()

    floating_objects._override_material(np.random.default_rng(0), obj)

    mesh = obj.item().data
    assert [layer.name for layer in mesh.uv_layers] == ["UVMap"]
    assert (
        np.ptp([loop.uv[:] for loop in mesh.uv_layers["UVMap"].data], axis=0).min() > 0
    )
