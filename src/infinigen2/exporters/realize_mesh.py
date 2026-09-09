# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: David Yan

import logging
from collections.abc import Iterable

import bpy
import procfunc as pf

from infinigen2.exporters.render_cycles import configure_cycles_devices

__all__ = [
    "convert_shader_displacement",
    "evaluate_shared_subdivision_to_shared_data",
    "realize_scene",
]

logger = logging.getLogger(__name__)


def _geometry_node_group_empty_new():
    group = bpy.data.node_groups.new("Geometry Nodes", "GeometryNodeTree")
    group.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    input_node = group.nodes.new("NodeGroupInput")
    output_node = group.nodes.new("NodeGroupOutput")
    output_node.is_active_output = True

    input_node.select = False
    output_node.select = False

    input_node.location.x = -200 - input_node.width
    output_node.location.x = 200

    group.links.new(output_node.inputs[0], input_node.outputs[0])

    return group


def _bake_displacement_to_vcols(obj, vcol_name="Displacement"):
    # Cycles fast bake
    scn = bpy.context.scene
    old_settings = {
        "engine": scn.render.engine,
        "device": scn.cycles.device,
        "samples": scn.cycles.samples,
        "view": scn.view_settings.view_transform,
    }
    scn.render.engine = "CYCLES"
    configure_cycles_devices("BEST_AVAILABLE")
    scn.cycles.samples = 1
    scn.view_settings.view_transform = "Standard"

    # Ensure vertex color attribute
    if vcol_name not in obj.data.color_attributes:
        obj.data.color_attributes.new(
            name=vcol_name, domain="POINT", type="FLOAT_COLOR"
        )
    obj.data.attributes.active_color = obj.data.color_attributes[vcol_name]

    # Patch materials: Surface <- Emission(displacement)
    patched = []
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.node_tree:
            continue
        nt = mat.node_tree
        nodes, links = nt.nodes, nt.links

        if "Material Output" not in nodes:
            continue
        out = nodes["Material Output"]

        disp_input = out.inputs["Displacement"]
        if not disp_input.is_linked:
            continue

        disp_link = disp_input.links[0]
        disp_source_socket = disp_link.from_socket

        # Save original Surface link
        surf_link = out.inputs["Surface"].links[0]
        orig_from = surf_link.from_socket
        links.remove(surf_link)

        # Temp emission: displacement -> add(0.5) -> emission -> surface
        add = nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        add.inputs[1].default_value = (0.5, 0.5, 0.5)
        add.name = "__TMP_ADD__"

        emit = nodes.new("ShaderNodeEmission")
        emit.name = "__TMP_EMIT__"

        links.new(disp_source_socket, add.inputs[0])
        links.new(add.outputs["Vector"], emit.inputs["Color"])
        links.new(emit.outputs["Emission"], out.inputs["Surface"])

        patched.append((mat, orig_from, emit, add))

    # Bake
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT", target="VERTEX_COLORS")
    obj.select_set(False)

    # Restore materials
    for mat, orig_from, emit, add in patched:
        nt = mat.node_tree
        nodes, links = nt.nodes, nt.links
        out = nodes["Material Output"]

        links.remove(out.inputs["Surface"].links[0])  # tmp emit link
        links.remove(out.inputs["Displacement"].links[0])
        nodes.remove(emit)
        nodes.remove(add)
        links.new(orig_from, out.inputs["Surface"])

    scn.render.engine = old_settings["engine"]
    scn.cycles.device = old_settings["device"]
    scn.cycles.samples = old_settings["samples"]
    scn.view_settings.view_transform = old_settings["view"]

    return


def _add_geo_displacement(obj, scale_val=1.0, vcol_name="Displacement", apply=True):
    mod = obj.modifiers.new("Displacement", "NODES")
    mod.node_group = _geometry_node_group_empty_new()
    ng = mod.node_group
    nodes, links = ng.nodes, ng.links

    group_in = nodes["Group Input"]
    group_out = nodes["Group Output"]

    nodes.new("GeometryNodeInputNormal")

    attr = nodes.new("GeometryNodeInputNamedAttribute")
    attr.data_type = "FLOAT_COLOR"
    attr.inputs[0].default_value = vcol_name

    sub = nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (0.5, 0.5, 0.5)  # SUBTRACT OFFSET

    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs["Scale"].default_value = scale_val

    set_pos = nodes.new("GeometryNodeSetPosition")

    links.new(group_in.outputs["Geometry"], set_pos.inputs["Geometry"])

    # Logic: Attribute -> Subtract(0.5) -> Scale -> Offset
    links.new(attr.outputs[0], sub.inputs[0])
    links.new(sub.outputs[0], scale.inputs["Vector"])
    links.new(scale.outputs["Vector"], set_pos.inputs["Offset"])

    links.new(set_pos.outputs["Geometry"], group_out.inputs["Geometry"])

    if apply:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)
        obj.select_set(False)


def convert_shader_displacement(obj, scale_val=1.0, apply_geo_modifier=False):
    _bake_displacement_to_vcols(obj, vcol_name="Displacement")
    _add_geo_displacement(
        obj, scale_val=scale_val, vcol_name="Displacement", apply=apply_geo_modifier
    )


def _subsurf_settings(mod: bpy.types.Modifier) -> tuple:
    skip = ("rna_type", "type", "name", "is_active", "show_expanded")
    return tuple(
        (p.identifier, repr(getattr(mod, p.identifier)))
        for p in mod.bl_rna.properties
        if not p.is_readonly and p.identifier not in skip
    )


def _bake_shared_stack(objs: list[bpy.types.Object]) -> None:
    lead = objs[0]
    was_disabled = lead.hide_viewport
    was_hidden = lead.hide_get()
    lead.hide_viewport = False
    lead.hide_set(False)
    for mod in lead.modifiers:
        if mod.type == "SUBSURF":
            mod.levels = mod.render_levels
        mod.show_viewport = mod.show_render
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    baked = bpy.data.meshes.new_from_object(
        lead.evaluated_get(depsgraph), depsgraph=depsgraph
    )
    lead.hide_viewport = was_disabled
    lead.hide_set(was_hidden)
    old = lead.data
    name = old.name
    for obj in objs:
        obj.modifiers.clear()
        obj.data = baked
    if old.users == 0:
        bpy.data.meshes.remove(old)
    baked.name = name


def evaluate_shared_subdivision_to_shared_data(
    objects: Iterable[pf.MeshObject],
) -> None:
    """Blender evaluates a modifier stack once per object with no reuse across objects
    sharing a datablock, so N aliases each deferring an identical subsurf pay for N
    copies of the same result. Bake each such group's stack into one shared mesh;
    singletons keep their deferred stack."""
    bpy.context.view_layer.update()
    groups = {}
    for wrapped in objects:
        obj = wrapped.item()
        if obj is None or obj.type != "MESH" or obj.data is None or not obj.modifiers:
            continue
        if any(mod.type != "SUBSURF" for mod in obj.modifiers):
            continue
        key = (
            obj.data.as_pointer(),
            tuple(_subsurf_settings(mod) for mod in obj.modifiers),
        )
        groups.setdefault(key, []).append(obj)
    for group in groups.values():
        if len(group) < 2:
            continue
        _bake_shared_stack(group)
        logger.info(
            f"Baked one shared modifier stack for {len(group)} objects "
            f"sharing {group[0].data.name!r}"
        )


def _apply_subsurf(obj: bpy.types.Object, subdivided: dict) -> None:
    # aliases share one datablock, which modifier_apply refuses; subdivide it once, reuse
    mods = [mod for mod in obj.modifiers if mod.type == "SUBSURF"]
    if not mods:
        return
    key = (obj.data.as_pointer(), tuple(_subsurf_settings(mod) for mod in mods))
    shared = subdivided.get(key)
    if shared is not None:
        for mod in mods:
            obj.modifiers.remove(mod)
        obj.data = shared
        return
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    for mod in mods:
        bpy.ops.object.modifier_apply(modifier=mod.name, single_user=True)
    obj.select_set(False)
    subdivided[key] = obj.data


def realize_scene():
    """
    Realizes entire scene, potentially expensive ; could be modified to on realize only in view.
    """
    subdivided = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.data:
            continue

        has_displacement = False
        for slot in obj.material_slots:
            if not slot.material or not slot.material.node_tree:
                continue
            nodes = slot.material.node_tree.nodes
            if "Material Output" not in nodes:
                continue
            out = nodes["Material Output"]
            if out.inputs["Displacement"].is_linked:
                has_displacement = True
                break

        _apply_subsurf(obj, subdivided)

        if has_displacement:
            convert_shader_displacement(obj)
