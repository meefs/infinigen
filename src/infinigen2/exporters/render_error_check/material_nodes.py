# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging

import bpy
import procfunc as pf

from infinigen2 import context
from infinigen2.exporters.render_error_check.util import (
    context_materials,
    iter_all_nodes,
)

logger = logging.getLogger(__name__)


class MaterialNodeError(ValueError):
    pass


# nodes whose linked Normal/Coat Normal input encodes bump lost under displacement
_NORMAL_INPUT_NODE_TYPES = frozenset(
    {
        "ShaderNodeAmbientOcclusion",
        "ShaderNodeBevel",
        "ShaderNodeBsdfAnisotropic",
        "ShaderNodeBsdfDiffuse",
        "ShaderNodeBsdfGlass",
        "ShaderNodeBsdfPrincipled",
        "ShaderNodeBsdfRefraction",
        "ShaderNodeBsdfSheen",
        "ShaderNodeBsdfToon",
        "ShaderNodeBsdfTranslucent",
        "ShaderNodeFresnel",
        "ShaderNodeLayerWeight",
        "ShaderNodeSubsurfaceScattering",
    }
)

_NORMAL_INPUT_SOCKETS = ("Normal", "Coat Normal")


def _material_nodes(material: bpy.types.Material) -> list[bpy.types.Node]:
    if not material.use_nodes or material.node_tree is None:
        return []
    return list(iter_all_nodes(material.node_tree))


def _node_normal_inputs(node: bpy.types.Node) -> list[str]:
    name = node.bl_idname
    if name == "ShaderNodeNormalMap":
        return [f"{name}: use the displacement output instead of normals"]
    if name not in _NORMAL_INPUT_NODE_TYPES:
        return []
    return [
        f"{name}: {sock!r} input set; use displacement instead"
        for sock in _NORMAL_INPUT_SOCKETS
        if node.inputs.get(sock) is not None and node.inputs[sock].is_linked
    ]


def normal_input_used(material: bpy.types.Material) -> list[str]:
    return [
        f"{material.name}: {msg}"
        for node in _material_nodes(material)
        for msg in _node_normal_inputs(node)
    ]


def _node_vector_unlinked(node: bpy.types.Node) -> bool:
    if not node.bl_idname.startswith("ShaderNodeTex"):
        return False
    vec = node.inputs.get("Vector")
    return vec is not None and vec.enabled and not vec.is_linked


def unlinked_texture_vector(material: bpy.types.Material) -> list[str]:
    return [
        f"{material.name}: {node.bl_idname}: Vector input unlinked, so Cycles samples "
        "Generated coords instead of the intended sample vector; pass an explicit vector"
        for node in _material_nodes(material)
        if _node_vector_unlinked(node)
    ]


def _raise_or_warn_issues(check: str, issues: list[str]):
    if not issues:
        return
    error = MaterialNodeError(
        f"materials contain invalid shader nodes [{check}]: {issues}"
    )
    mode = getattr(context.globals, "error_mode_" + check)
    context.raise_or_warn(mode, error, logger)


def assert_material_nodes_valid(objects: list[pf.MeshObject] | None = None):
    materials = context_materials(objects)
    normal = [i for m in materials for i in normal_input_used(m)]
    vector = [i for m in materials for i in unlinked_texture_vector(m)]
    _raise_or_warn_issues("material_normal_input", normal)
    _raise_or_warn_issues("material_texture_vector", vector)
