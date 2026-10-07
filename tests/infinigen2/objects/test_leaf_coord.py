# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import numpy as np
import procfunc as pf
import pytest

from infinigen2.objects import leaf, leaf_broadleaf, leaf_ginko, leaf_maple
from infinigen2.objects.leaf_coord import LEAF_COORD_UV_LAYER

LeafResult = (
    leaf.LeafResult
    | leaf_broadleaf.LeafBroadleafResult
    | leaf_ginko.LeafGinkoResult
    | leaf_maple.LeafMapleResult
)


@pytest.mark.parametrize(
    "generator",
    [
        leaf.leaf_simple_rand,
        leaf_broadleaf.leaf_broadleaf_rand,
        leaf_ginko.leaf_ginkgo_rand,
        leaf_maple.leaf_maple_rand,
    ],
)
def test_leaf_material_uses_explicit_normalized_coordinates(
    generator: Callable[[pf.RNG], LeafResult],
) -> None:
    result = generator(np.random.default_rng(0))
    values = pf.ops.attr.uv_coords(result.mesh)
    texture_coordinate_nodes = [
        node
        for node in result.mesh.item().data.materials[0].node_tree.nodes
        if node.bl_idname == "ShaderNodeTexCoord"
    ]

    assert np.allclose(values.min(axis=0), 0.0)
    assert np.allclose(values.max(axis=0), 1.0)
    assert result.mesh.item().data.uv_layers.active.name == LEAF_COORD_UV_LAYER
    assert result.mesh.item().data.uv_layers[LEAF_COORD_UV_LAYER].active_render
    assert len(texture_coordinate_nodes) == 1
