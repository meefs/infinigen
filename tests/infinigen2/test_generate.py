# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from pathlib import Path
from typing import Any, Callable, NamedTuple, cast

import bpy
import numpy as np
import procfunc as pf

from infinigen2 import generate
from infinigen2.scenes.placement import collision as ccol


class ObjectResult(NamedTuple):
    mesh: pf.MeshObject


class SceneResult(NamedTuple):
    all_objects: list[pf.MeshObject]
    cameras: list[pf.CameraObject]
    lights: list[pf.LightObject]
    colliders: ccol.CollisionSet


def cube_rand(rng: np.random.Generator) -> ObjectResult:
    mesh = pf.ops.primitives.mesh_cube(size=1.0)
    pf.ops.modifier.subdivide_surface(mesh, levels=1, _skip_apply=True)
    return ObjectResult(mesh=mesh)


def scene_with_collider_rand(rng: np.random.Generator) -> SceneResult:
    del rng
    visible = pf.ops.primitives.mesh_cube(size=1.0)
    visible.item().name = "visible"
    collider = pf.ops.primitives.mesh_cube(size=1.0)
    collider.item().name = "collider_only"
    stray = pf.ops.primitives.mesh_cube(size=1.0)
    stray.item().name = "stray"
    colliders = ccol.collision_set([visible, collider])
    return SceneResult([visible], [], [], colliders)


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


def test_execute_generators_preserves_hidden_collider_only_objects(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    generators: list[tuple[str, str, Callable[..., Any]]] = [
        ("scene_with_collider_rand", "Scene", scene_with_collider_rand)
    ]

    result = cast(
        dict[str, Any], generate.execute_generators(tmp_path, generators, rng, {})
    )

    collider = result["colliders"].objs[1]
    assert collider.item().name == "collider_only"
    assert collider.item().hide_render
    assert not result["objects"][0].item().hide_render
    assert "stray" not in bpy.data.objects
