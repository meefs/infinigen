# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from pathlib import Path
from typing import Any, Callable, NamedTuple, cast

import numpy as np
import procfunc as pf

from infinigen2 import generate


class ObjectResult(NamedTuple):
    mesh: pf.MeshObject


def cube_rand(rng: np.random.Generator) -> ObjectResult:
    mesh = pf.ops.primitives.mesh_cube(size=1.0)
    pf.ops.modifier.subdivide_surface(mesh, levels=1, _skip_apply=True)
    return ObjectResult(mesh=mesh)


def test_execute_generators_reports_subdivision_triangles(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    generators: list[tuple[str, str, Callable[..., Any]]] = [
        ("cube_rand", "Object", cube_rand)
    ]

    result = cast(
        dict[str, Any], generate.execute_generators(tmp_path, generators, rng, {})
    )

    assert result["base_tris"] == 12
    assert result["subdiv_tris"] == 48
