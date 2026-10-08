# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import math
from typing import NamedTuple

import bpy
import numpy as np
import procfunc as pf
from procfunc.ops._util import execute_object_op
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from infinigen2.shaders.base_materials import plastic
from infinigen2.shaders.functionality_lists import fabric_sturdy_rand

from .human import HumanResult, human_rand

__all__ = [
    "HumanClothedResult",
    "human_clothed_rand",
]


class HumanClothedResult(NamedTuple):
    mesh: pf.MeshObject
    armature: pf.ArmatureObject
    all_objects: list[pf.MeshObject]


class _Topology(NamedTuple):
    edges: np.ndarray
    face_edges: np.ndarray
    edge_faces: np.ndarray


def _clothing_material_rand(rng: pf.RNG) -> pf.Material:
    rng_choice, rng_material = rng.spawn(2)
    material_function = pf.control.choice(
        rng_choice,
        [
            (fabric_sturdy_rand, 4.0),
            (plastic.plastic_rand, 1.0),
        ],
    )
    return material_function(rng_material, pf.nodes.shader.coord().uv)


# MPFB base-mesh edge loops, each given by seed edges (vertex pairs); option k cuts at loop k
def _loop_options() -> dict[str, np.ndarray]:
    sleeve = [
        ((10234, 10237), (3566, 3569)),
        ((10166, 10172), (3498, 3504)),
        ((10115, 10121), (3447, 3453)),
        ((10064, 10070), (3396, 3402)),
        ((9979, 9985), (3311, 3317)),
        ((8408, 8409), (1736, 1737)),
        ((8366, 8372), (1694, 1700)),
        ((8102, 8144), (1414, 1456)),
    ]
    collar = [(776, 928), (800, 808), (1032, 1040), (760, 822), (1464, 1512)]
    collar += [(1488, 1489)]
    hem = [(4216, 4224), (4199, 4200), (4198, 4206), (4189, 4190)]
    leg = [
        ((12976, 12984), (6288, 6376)),
        ((12960, 12963), (4800, 6361)),
        ((11416, 11447), (4824, 4827)),
        ((11432, 11437), (4808, 6415)),
        ((11392, 11396), (4776, 4784)),
        ((11376, 11377), (4760, 4763)),
        ((11368, 13009), (4736, 4740)),
        ((11335, 11336), (4717, 4718)),
        ((11297, 11298), (4679, 4680)),
        ((11259, 11260), (4641, 4642)),
        ((11221, 11222), (4603, 4604)),
        ((11183, 11184), (4565, 4566)),
        ((11145, 11146), (4527, 4528)),
        ((11120, 11121), (4504, 6399)),
        ((11088, 11089), (4470, 4471)),
        ((11050, 11051), (4432, 4433)),
    ]
    waist = [(4189, 4190), (4198, 4206), (4199, 4200), (4216, 4224), (4336, 4339)]
    return {
        "sleeve": np.array(sleeve),
        "collar": np.array(collar)[:, None],
        "hem": np.array(hem)[:, None],
        "leg": np.array(leg),
        "waist": np.array(waist)[:, None],
    }


def _topology(mesh: pf.MeshObject) -> _Topology:
    data = mesh.item().data
    counts = (len(data.vertices), len(data.edges), len(data.polygons))
    if counts != (13_380, 26_756, 13_378):
        raise RuntimeError(f"Clothing expects MPFB's base mesh, got {counts}")
    corner_edges = pf.ops.attr.loop_edge_indices(mesh)
    face_edges = corner_edges[pf.ops.attr.loop_starts(mesh)[:, None] + np.arange(4)]
    corner_faces = np.repeat(np.arange(len(face_edges)), 4)
    order = np.argsort(face_edges.ravel(), kind="stable")
    edge_faces = corner_faces[order].reshape(-1, 2)
    return _Topology(pf.ops.attr.edge_indices(mesh), face_edges, edge_faces)


def _graph(count: int, links: np.ndarray) -> coo_matrix:
    ones = np.ones(len(links))
    return coo_matrix((ones, (links[:, 0], links[:, 1])), shape=(count, count))


def _connected_labels(count: int, links: np.ndarray) -> np.ndarray:
    return connected_components(_graph(count, links), directed=False)[1]


def _unmark(mesh: pf.MeshObject, names: list[str]) -> None:
    attributes = mesh.item().data.attributes
    for name in names:
        attributes.remove(attributes[name])


def _named(name: str) -> pf.ProcNode:
    return pf.nodes.geo.input_named_attribute(
        name, data_type=pf.nodes.NodeDataType.BOOLEAN
    ).attribute


def _garment_object(
    geometry: pf.ProcNode[pf.MeshObject], human_result: HumanResult
) -> pf.MeshObject:
    mesh = pf.nodes.to_mesh_object(geometry)
    body = human_result.mesh.item()
    obj = mesh.item()
    obj.parent = body.parent
    obj.matrix_parent_inverse = body.matrix_parent_inverse.copy()
    obj.matrix_basis = body.matrix_basis.copy()
    return mesh


def _fitted_cylinder(
    body: pf.ProcNode[pf.MeshObject],
    part: str,
    z_range: tuple[float, float],
    ease: float,
    flare: float,
    top_fade: float,
    hang: tuple[float, float],
    hang_below: float,
    resolution: tuple[int, int] = (64, 40),
) -> pf.ProcNode[pf.MeshObject]:
    z_min, z_max = z_range
    around, rings = resolution
    position = pf.nodes.geo.input_position()
    piece = pf.nodes.geo.separate_geometry(
        body, selection=_named(part), domain="FACE"
    ).selection

    hanging = pf.nodes.geo.delete_geometry(
        piece, selection=position.z > z_max - hang_below
    )
    steps = pf.nodes.geo.mesh_line(
        start_location=(0.0, 0.0, hang[0]),
        offset=(0.0, 0.0, (hang[1] - hang[0]) / 9.0),
        count=10,
    )
    hanging = pf.nodes.geo.instance_on_points(steps, hanging)
    piece = pf.nodes.geo.join_geometry([piece, pf.nodes.geo.realize_instances(hanging)])

    bounds = pf.nodes.geo.bound_box(piece)
    center = (bounds.min + bounds.max) * 0.5
    primitive = pf.nodes.geo.mesh_cylinder(
        vertices=around, side_segments=rings, depth=z_max - z_min
    )
    middle = pf.nodes.math.combine_xyz(x=center.x, y=center.y, z=(z_min + z_max) / 2)
    cylinder = pf.nodes.geo.transform(primitive.mesh, translation=middle)

    axis = pf.nodes.math.combine_xyz(x=center.x, y=center.y, z=position.z)
    inward = pf.nodes.math.vector_normalize(axis - position)
    hit = pf.nodes.geo.raycast(piece, source_position=position, ray_direction=inward)
    depth = pf.nodes.func.switch(hit.is_hit, 0.0, 1.0 - hit.hit_distance)
    radius = pf.nodes.math.maximum(depth, pf.nodes.geo.blur_attribute(depth, 16))
    radius = pf.nodes.geo.blur_attribute(radius, 8)

    cuff = pf.nodes.math.map_range(position.z, from_min=z_min + 0.18, from_max=z_min)
    grow = ease + flare * 0.06 * cuff * cuff
    shrink = pf.nodes.math.map_range(
        position.z, from_min=z_max + 0.001, from_max=z_max + 0.001 - top_fade
    )
    size = (radius + grow) * shrink

    girth = pf.nodes.geo.attribute_statistic(cylinder, size).mean * 2.0 * math.pi
    uv = pf.nodes.math.combine_xyz(x=primitive.uv_map.x * girth, y=position.z, z=0.0)
    shaped = pf.nodes.geo.set_position(
        cylinder, axis - pf.nodes.math.vector_scale(inward, size)
    )
    return pf.nodes.geo.store_named_attribute(
        shaped, name="UVMap", value=uv, domain="CORNER", data_type="FLOAT2"
    )


def _border_field() -> pf.ProcNode:
    faces = pf.nodes.geo.input_mesh_edge_neighbors()
    open_edge = pf.nodes.func.boolean_and(faces > 0, faces < 2)
    open_count = pf.nodes.func.switch(open_edge, 0.0, 1.0)
    return pf.nodes.geo.field_on_domain(open_count, domain="EDGE") > 0.0


def _drop_hem(
    garment: pf.ProcNode[pf.MeshObject],
    hem_band: tuple[float, float],
    hem_extension: float,
    drape: float,
) -> pf.ProcNode[pf.MeshObject]:
    hem_top, hem_half_width = hem_band
    position = pf.nodes.geo.input_position()
    within = pf.nodes.math.absolute(position.x) < hem_half_width
    band = pf.nodes.func.boolean_and(within, position.z < hem_top)
    hem_edge = pf.nodes.func.boolean_and(_border_field(), band)

    ends = pf.nodes.geo.input_mesh_edge_vertices()
    length = pf.nodes.math.vector_distance(ends.position_1, ends.position_2)
    distance = pf.nodes.geo.input_shortest_edge_paths(
        end_vertex=hem_edge, edge_cost=length
    ).total_cost
    captured = pf.nodes.geo.capture_attribute(
        garment,
        domain="POINT",
        hem=pf.nodes.math.map_range(
            distance, from_min=0.18 + hem_extension, from_max=0.0
        ),
    )
    hem = captured.hem

    lowered = pf.nodes.math.combine_xyz(x=0.0, y=0.0, z=-hem_extension * hem)
    garment = pf.nodes.geo.set_position(captured.geometry, offset=lowered)
    hem_z = pf.nodes.geo.attribute_statistic(
        garment, position.z, selection=hem > 0.999
    ).mean
    level = pf.nodes.math.combine_xyz(
        x=0.0, y=0.0, z=drape * hem * (hem_z - position.z)
    )
    return pf.nodes.geo.set_position(garment, offset=level)


def _garment(
    body: pf.ProcNode[pf.MeshObject],
    solids: list[pf.ProcNode[pf.MeshObject]],
    clearance: float,
    hem_band: tuple[float, float],
    hem_extension: float,
    drape: float,
    opening: float,
) -> pf.ProcNode[pf.MeshObject]:
    distance = pf.nodes.geo.mesh_to_sdf_grid(
        mesh=pf.nodes.geo.join_geometry([body, *solids]), voxel_size=0.01, band_width=3
    )
    target = pf.nodes.geo.grid_to_mesh(
        grid=distance, threshold=clearance, adaptivity=0.0
    )

    garment = pf.nodes.geo.separate_geometry(
        body, selection=_named("garment_kept"), domain="FACE"
    ).selection
    garment = _drop_hem(garment, hem_band, hem_extension, drape)

    position = pf.nodes.geo.input_position()
    interior = pf.nodes.func.boolean_not(_border_field())
    nearest = pf.nodes.geo.proximity(target, sample_position=position).position
    garment = pf.nodes.geo.set_position(garment, nearest)
    blurred = pf.nodes.geo.blur_attribute(position)
    garment = pf.nodes.geo.set_position(garment, blurred, selection=interior)
    garment = pf.nodes.geo.set_position(garment, nearest, selection=interior)

    opened = pf.nodes.math.combine_xyz(
        x=pf.nodes.math.sign(position.x) * max(opening * 0.5, 0.015),
        y=position.y - max(opening * 0.5, 0.02),
        z=position.z,
    )
    return pf.nodes.geo.set_position(garment, opened, selection=_named("garment_seam"))


def _loop_labels(topology: _Topology) -> np.ndarray:
    edges, _, edge_faces = topology
    valence = np.bincount(edges.ravel())
    incident = np.argsort(edges.ravel(), kind="stable") // 2
    regular = np.flatnonzero(valence == 4)
    around = incident[(np.cumsum(valence) - valence)[regular, None] + np.arange(4)]
    first, second = np.triu_indices(4, 1)
    left, right = around[:, first], around[:, second]
    shared = edge_faces[left][..., :, None] == edge_faces[right][..., None, :]
    straight = ~shared.any(axis=(-2, -1))
    links = np.stack([left[straight], right[straight]], axis=1)
    return _connected_labels(len(edges), links)


def _loop_seeds(edges: np.ndarray, cuts: dict[str, float]) -> np.ndarray:
    options = _loop_options()
    chosen = [
        options[part][min(math.ceil(count), len(options[part]) - 1)]
        for part, count in cuts.items()
    ]
    base = edges.max() + 1
    edge_keys = np.sort(edges, axis=1) @ (base, 1)
    pair_keys = np.sort(np.concatenate(chosen), axis=1) @ (base, 1)
    order = np.argsort(edge_keys)
    return order[np.searchsorted(edge_keys[order], pair_keys)]


def _region(topology: _Topology, cuts: dict[str, float], start_face: int) -> np.ndarray:
    edges, face_edges, edge_faces = topology
    labels = _loop_labels(topology)
    cut = np.isin(labels, labels[_loop_seeds(edges, cuts)])
    pieces = _connected_labels(len(face_edges), edge_faces[~cut])
    region = pieces == pieces[start_face]
    border = region[edge_faces].sum(axis=1) == 1
    if not np.array_equal(border, cut):
        raise RuntimeError("Clothing region is not bounded exactly by its cut loops")
    return region


def _region_vertices(mesh: pf.MeshObject, region: np.ndarray) -> np.ndarray:
    faces = pf.ops.attr.polygon_vertex_indices(mesh, 4)
    return np.bincount(faces[region].ravel(), minlength=faces.max() + 1) > 0


def _copy_visible_body(body: pf.MeshObject) -> pf.MeshObject:
    obj = body.item().copy()
    obj.data = body.item().data.copy()
    bpy.context.collection.objects.link(obj)
    for modifier in list(obj.modifiers):
        if modifier.type != "MASK":
            obj.modifiers.remove(modifier)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    baked = bpy.data.meshes.new_from_object(
        obj.evaluated_get(depsgraph),
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    original = obj.data
    obj.data = baked
    bpy.data.meshes.remove(original)
    obj.modifiers.clear()
    return pf.MeshObject(obj)


def _transfer_vertex_groups(source: pf.MeshObject, target: pf.MeshObject) -> None:
    modifier = target.item().modifiers.new("Body weights", "DATA_TRANSFER")
    modifier.object = source.item()
    modifier.use_vert_data = True
    modifier.data_types_verts = {"VGROUP_WEIGHTS"}
    modifier.vert_mapping = "POLYINTERP_NEAREST"
    execute_object_op(
        bpy.ops.object.datalayout_transfer, active=target, modifier=modifier.name
    )
    execute_object_op(
        bpy.ops.object.modifier_apply, active=target, modifier=modifier.name
    )


def _finish_garment(
    mesh: pf.MeshObject,
    human_result: HumanResult,
    material: pf.Material,
    label: str,
) -> pf.MeshObject:
    obj = mesh.item()
    obj.data.materials.clear()
    obj.data.materials.append(material.item())
    pf.ops.attr.write_material_index(mesh, 0)
    obj.data.polygons.foreach_set("use_smooth", np.ones(len(obj.data.polygons), bool))
    obj.modifiers.new("Armature", "ARMATURE").object = human_result.armature.item()
    obj.name = f"{human_result.mesh.item().name}_{label}"
    pf.ops.modifier.subdivide_surface(mesh, levels=2, _skip_apply=True)
    return mesh


def _top(
    human_result: HumanResult,
    material: pf.Material,
    clearance: float,
    cuts: dict[str, float],
    fit: float,
    hem_extension: float,
    drape: float,
    opening: float,
    layer_ease: float,
    label: str,
) -> pf.MeshObject:
    source = _copy_visible_body(human_result.mesh)
    topology = _topology(source)
    chest_face = 3773
    region = _region(topology, cuts, chest_face)
    torso_cuts = {"sleeve": 7.0, "collar": 0.0, "leg": 15.0}
    torso_faces = _region(topology, torso_cuts, chest_face)
    body_coordinates = pf.ops.attr.vertex_positions(source)
    centers = pf.ops.attr.polygon_centers(source)
    front = (np.abs(centers[:, 0]) < opening * 0.5) & (centers[:, 1] < 0.0)
    opening_faces = region & front
    kept = region & ~opening_faces
    edges, _, edge_faces = topology
    seam_edges = opening_faces[edge_faces].any(axis=1) & kept[edge_faces].any(axis=1)
    seam = np.zeros(len(body_coordinates), dtype=bool)
    seam[edges[seam_edges]] = True
    labels = _loop_labels(topology)
    hem_seeds = _loop_seeds(edges, {"hem": cuts["hem"]})
    hem_loop = body_coordinates[edges[np.isin(labels, labels[hem_seeds])]]
    hem_band = (
        float(hem_loop[..., 2].max()) + clearance,
        float(np.abs(hem_loop[..., 0]).max()) + clearance + 0.02,
    )
    z = body_coordinates[_region_vertices(source, kept), 2]
    z_range = (float(z.min()) - hem_extension, float(z.max()))
    pf.ops.attr.write_attribute(source, torso_faces, "garment_torso", "FACE")
    pf.ops.attr.write_attribute(source, kept, "garment_kept", "FACE")
    pf.ops.attr.write_attribute(source, seam, "garment_seam", "POINT")

    ease = 0.03 * (0.5 + fit) + layer_ease - clearance
    flare = max(drape, float(hem_extension > 0.0)) * 0.4
    hang_length = drape * 0.4 + hem_extension
    hang = (0.3 * hang_length, -hang_length)
    body = pf.nodes.geo.object_info(source).geometry
    torso = _fitted_cylinder(
        body, "garment_torso", z_range, ease, flare, 0.06, hang, 0.15
    )
    garment = _garment(
        body, [torso], clearance, hem_band, hem_extension, drape, opening
    )
    mesh = _garment_object(garment, human_result)
    bpy.data.objects.remove(source.item(), do_unlink=True)

    _unmark(mesh, ["garment_torso", "garment_kept", "garment_seam"])
    pf.ops.mesh.unsubdivide(mesh, iterations=2)
    return _finish_garment(mesh, human_result, material, label)


def _shirt_rand(
    rng: pf.RNG,
    human_result: HumanResult,
    material: pf.Material | None = None,
    clearance: float | None = None,
    sleeve_num_loops: float | None = None,
    collar_num_loops: float | None = None,
    hem_num_loops: float | None = None,
    fit: float | None = None,
    hem_extension: float | None = None,
    drape: float | None = None,
) -> pf.MeshObject:
    lanes = rng.spawn(9)
    if material is None:
        material = _clothing_material_rand(lanes[0])
    if clearance is None:
        clearance = pf.random.clip_gaussian(lanes[1], 0.0175, 0.002, 0.015, 0.02)
    if sleeve_num_loops is None:
        sleeve_num_loops = pf.random.uniform(lanes[2], 0.0, 7.0)
    if collar_num_loops is None:
        collar_num_loops = pf.random.clip_gaussian(lanes[3], 2.5, 1.5, 0.0, 5.0)
    if hem_num_loops is None:
        hem_num_loops = pf.random.uniform(lanes[4], 0.0, 3.0)
    if fit is None:
        fit = pf.random.clip_gaussian(lanes[5], 0.55, 0.20, 0.0, 1.0)
    if hem_extension is None:
        hem_extension = pf.random.clip_gaussian(lanes[7], 0.08, 0.06, 0.0, 0.20)
    if drape is None:
        drape = pf.random.clip_gaussian(lanes[8], 0.55, 0.25, 0.0, 1.0)
    return _top(
        human_result,
        material,
        clearance,
        {"sleeve": sleeve_num_loops, "collar": collar_num_loops, "hem": hem_num_loops},
        fit,
        hem_extension,
        drape,
        0.0,
        0.0,
        "shirt",
    )


def _full_length_rand(rng: pf.RNG) -> float:
    return pf.random.uniform(rng, 0.0, 2.0)


def _any_length_rand(rng: pf.RNG) -> float:
    return pf.random.uniform(rng, 0.0, 15.0)


def _long_length_rand(rng: pf.RNG) -> float:
    rng_choice, rng_length = rng.spawn(2)
    length_function = pf.control.choice(
        rng_choice, [(_full_length_rand, 1.0), (_any_length_rand, 1.0)]
    )
    return length_function(rng_length)


def _lower_body(mesh: pf.MeshObject) -> tuple[_Topology, np.ndarray, np.ndarray]:
    topology = _topology(mesh)
    hip_face = 10589
    lower_body = _region(topology, {"waist": 0.0, "leg": 0.0}, hip_face)
    pelvis = _region(topology, {"waist": 0.0, "leg": 15.0}, hip_face)
    return topology, lower_body, pelvis


def _legwear_rand(
    rng: pf.RNG,
    human_result: HumanResult,
    label: str,
    length_num_loops: float,
) -> pf.MeshObject:
    lanes = rng.spawn(5)
    material = _clothing_material_rand(lanes[0])
    waist_num_loops = pf.random.uniform(lanes[1], 0.0, 2.5)
    fit = pf.random.clip_gaussian(lanes[2], 0.45, 0.20, 0.0, 1.0)
    cuff_flare = pf.random.clip_gaussian(lanes[4], 0.15, 0.25, 0.0, 1.0)
    source = _copy_visible_body(human_result.mesh)
    topology, lower_body, pelvis = _lower_body(source)
    body_coordinates = pf.ops.attr.vertex_positions(source)
    cuts = {"leg": length_num_loops, "waist": waist_num_loops}
    region = _region(topology, cuts, 10589)
    z = body_coordinates[_region_vertices(source, region), 2]
    z_min = float(z.min())
    z_max = float(z.max())
    span = z_max - z_min
    pelvis_start = z_max - min(0.28, span * 0.60)
    pelvis_end = z_max - min(0.10, span * 0.25)
    face_x = pf.ops.attr.polygon_centers(source)[:, 0]
    on_left = face_x >= np.median(face_x[pelvis])
    pf.ops.attr.write_attribute(source, lower_body & on_left, "garment_left", "FACE")
    pf.ops.attr.write_attribute(source, lower_body & ~on_left, "garment_right", "FACE")
    pf.ops.attr.write_attribute(source, pelvis, "garment_pelvis", "FACE")
    pf.ops.attr.write_attribute(source, region, "garment_kept", "FACE")

    ease = 0.015 * (0.5 + fit) - 0.011
    legs = (z_min, pelvis_end)
    hips = (pelvis_start, z_max)
    body = pf.nodes.geo.object_info(source).geometry
    still = (0.0, 0.0)
    left = _fitted_cylinder(
        body, "garment_left", legs, ease, cuff_flare, 0.02, still, 0.0
    )
    right = _fitted_cylinder(
        body, "garment_right", legs, ease, cuff_flare, 0.02, still, 0.0
    )
    hip = _fitted_cylinder(body, "garment_pelvis", hips, ease, 0.0, 0.10, still, 0.0)
    garment = _garment(body, [left, right, hip], 0.011, (0.0, 0.0), 0.0, 0.0, 0.0)
    mesh = _garment_object(garment, human_result)
    bpy.data.objects.remove(source.item(), do_unlink=True)

    _unmark(mesh, ["garment_left", "garment_right", "garment_pelvis", "garment_kept"])
    pf.ops.mesh.unsubdivide(mesh, iterations=2)
    return _finish_garment(mesh, human_result, material, label)


def _pants_rand(rng: pf.RNG, human_result: HumanResult) -> pf.MeshObject:
    rng_length, rng_pants = rng.spawn(2)
    length_num_loops = _long_length_rand(rng_length)
    return _legwear_rand(rng_pants, human_result, "pants", length_num_loops)


def _shorts_rand(rng: pf.RNG, human_result: HumanResult) -> pf.MeshObject:
    rng_length, rng_shorts = rng.spawn(2)
    length_num_loops = pf.random.clip_gaussian(rng_length, 12.0, 3.5, 0.0, 15.0)
    return _legwear_rand(rng_shorts, human_result, "shorts", length_num_loops)


def _skirt_rand(
    rng: pf.RNG,
    human_result: HumanResult,
    material: pf.Material | None = None,
) -> pf.MeshObject:
    lanes = rng.spawn(6)
    if material is None:
        material = _clothing_material_rand(lanes[0])
    length_num_loops = _long_length_rand(lanes[1])
    waist_num_loops = pf.random.uniform(lanes[2], 0.0, 2.5)
    fit = pf.random.clip_gaussian(lanes[3], 0.45, 0.20, 0.0, 1.0)
    flare = pf.random.clip_gaussian(lanes[5], 0.20, 0.15, 0.0, 1.0)
    source = _copy_visible_body(human_result.mesh)
    topology, lower_body, _ = _lower_body(source)
    coordinates = pf.ops.attr.vertex_positions(source)
    labels = _loop_labels(topology)
    waist_seeds = _loop_seeds(topology.edges, {"waist": waist_num_loops})
    hem_seeds = _loop_seeds(topology.edges, {"leg": length_num_loops})
    waist_edges = topology.edges[np.isin(labels, labels[waist_seeds])]
    hem_edges = topology.edges[np.isin(labels, labels[hem_seeds])]
    top_z = float(coordinates[np.unique(waist_edges), 2].mean())
    hem_z = min(float(coordinates[np.unique(hem_edges), 2].mean()), top_z - 0.12)
    pf.ops.attr.write_attribute(source, lower_body, "garment_lower_body", "FACE")

    body = pf.nodes.geo.object_info(source).geometry
    ease = 0.012 + 0.006 * (0.5 + fit)
    tube = _fitted_cylinder(
        body,
        "garment_lower_body",
        (hem_z, top_z),
        ease,
        flare,
        0.001,
        (0.0, hem_z - top_z),
        0.0,
        (48, 16),
    )
    caps = pf.nodes.geo.input_mesh_face_neighbors().vertex_count > 4
    tube = pf.nodes.geo.delete_geometry(
        tube, selection=caps, domain="FACE", mode="ONLY_FACE"
    )
    mesh = _garment_object(tube, human_result)
    _transfer_vertex_groups(source, mesh)
    bpy.data.objects.remove(source.item(), do_unlink=True)
    return _finish_garment(mesh, human_result, material, "skirt")


def _jacket_over_shirt_rand(
    rng: pf.RNG, human_result: HumanResult
) -> list[pf.MeshObject]:
    lanes = rng.spawn(11)
    clearance = pf.random.clip_gaussian(lanes[0], 0.028, 0.002, 0.024, 0.03)
    sleeve_num_loops = pf.random.uniform(lanes[1], 0.0, 7.0)
    collar_num_loops = pf.random.clip_gaussian(lanes[2], 3.5, 1.2, 1.0, 5.0)
    hem_num_loops = pf.random.uniform(lanes[3], 0.0, 3.0)
    fit = pf.random.clip_gaussian(lanes[4], 0.80, 0.15, 0.25, 1.0)
    hem_extension = pf.random.clip_gaussian(lanes[6], 0.14, 0.12, 0.0, 0.45)
    drape = pf.random.clip_gaussian(lanes[7], 0.68, 0.16, 0.0, 0.85)
    opening = pf.random.clip_gaussian(lanes[8], 0.07, 0.025, 0.03, 0.14)
    jacket_material = _clothing_material_rand(lanes[9])
    shirt = _shirt_rand(
        lanes[10],
        human_result,
        clearance=min(clearance * 0.6, 0.02),
        sleeve_num_loops=sleeve_num_loops,
        collar_num_loops=max(collar_num_loops - 1.0, 0.0),
        hem_num_loops=hem_num_loops,
        fit=max(fit - 0.25, 0.0),
        hem_extension=min(hem_extension * 0.7, 0.12),
        drape=max(drape - 0.15, 0.0),
    )
    jacket = _top(
        human_result,
        jacket_material,
        clearance,
        {"sleeve": sleeve_num_loops, "collar": collar_num_loops, "hem": hem_num_loops},
        fit,
        hem_extension,
        drape,
        opening,
        0.005,
        "jacket",
    )
    return [shirt, jacket]


def _shirt_pants_rand(rng: pf.RNG, human_result: HumanResult) -> list[pf.MeshObject]:
    rng_top, rng_bottom = rng.spawn(2)
    return [_shirt_rand(rng_top, human_result), _pants_rand(rng_bottom, human_result)]


def _shirt_skirt_rand(rng: pf.RNG, human_result: HumanResult) -> list[pf.MeshObject]:
    rng_top, rng_bottom = rng.spawn(2)
    shirt = _shirt_rand(rng_top, human_result, hem_num_loops=2.0)
    return [shirt, _skirt_rand(rng_bottom, human_result)]


def _jacket_pants_rand(rng: pf.RNG, human_result: HumanResult) -> list[pf.MeshObject]:
    rng_top, rng_bottom = rng.spawn(2)
    tops = _jacket_over_shirt_rand(rng_top, human_result)
    return [*tops, _pants_rand(rng_bottom, human_result)]


def _jacket_skirt_rand(rng: pf.RNG, human_result: HumanResult) -> list[pf.MeshObject]:
    rng_top, rng_bottom = rng.spawn(2)
    tops = _jacket_over_shirt_rand(rng_top, human_result)
    return [*tops, _skirt_rand(rng_bottom, human_result)]


def _dress_rand(rng: pf.RNG, human_result: HumanResult) -> list[pf.MeshObject]:
    lanes = rng.spawn(5)
    material = _clothing_material_rand(lanes[0])
    collar_num_loops = pf.random.clip_gaussian(lanes[1], 2.5, 1.2, 0.0, 4.0)
    fit = pf.random.clip_gaussian(lanes[2], 0.55, 0.20, 0.0, 1.0)
    bodice = _shirt_rand(
        lanes[3],
        human_result,
        material=material,
        collar_num_loops=collar_num_loops,
        fit=fit,
        hem_extension=0.0,
        drape=0.0,
    )
    skirt = _skirt_rand(lanes[4], human_result, material=material)
    mesh = pf.ops.object.joined(bodice=bodice, skirt=skirt)
    bpy.data.objects.remove(bodice.item(), do_unlink=True)
    bpy.data.objects.remove(skirt.item(), do_unlink=True)
    mesh.item().name = f"{human_result.mesh.item().name}_dress"
    return [mesh]


def human_clothed_rand(
    rng: pf.RNG, skin_material: pf.Material | None = None
) -> HumanClothedResult:
    rng_human, rng_choice, rng_outfit = rng.spawn(3)
    human_result = human_rand(rng_human, skin_material)
    outfit_function = pf.control.choice(
        rng_choice,
        [
            (_shirt_pants_rand, 9.0),
            (_shirt_skirt_rand, 2.0),
            (_jacket_pants_rand, 3.0),
            (_jacket_skirt_rand, 1.0),
            (_dress_rand, 3.0),
        ],
    )
    garments = outfit_function(rng_outfit, human_result)
    return HumanClothedResult(
        mesh=human_result.mesh,
        armature=human_result.armature,
        all_objects=[human_result.mesh, *garments],
    )
