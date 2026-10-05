from functools import partial

import bpy
import numpy as np
import procfunc as pf
import pytest

from infinigen2.objects import wall_art
from infinigen2.shaders import functionality_lists
from infinigen2.shaders.masks import graphicdesign


def _build(seed: int) -> tuple[np.ndarray, bpy.types.Material]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vector = pf.nodes.shader.coord().uv
    color = graphicdesign.art_rand(rng=np.random.default_rng(seed), vector=vector)
    assert isinstance(color, pf.ProcNode)
    assert len(bpy.data.materials) == 0
    material_result = pf.Material(surface=pf.nodes.shader.emission(color=color))
    material = material_result.item()
    assert len(bpy.data.images) == 1
    image = bpy.data.images[0]
    assert (image.size[0], image.size[1]) == (400, 400)
    assert image.packed_file is not None
    pixels = np.empty(400 * 400 * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(400, 400, 4), material


def test_art_image_is_seeded_and_nonflat():
    first, _ = _build(7)
    repeated, _ = _build(7)
    different, _ = _build(8)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert np.isfinite(first).all()
    assert first[..., :3].std() > 0.05
    assert first[..., :3].mean() < 0.8
    assert (first[..., 3] == 1.0).mean() > 0.99


def test_art_color_preserves_image_warping_without_bsdf():
    _, material = _build(11)
    nodes = material.node_tree.nodes
    images = [node for node in nodes if node.bl_idname == "ShaderNodeTexImage"]
    voronoi = [node for node in nodes if node.bl_idname == "ShaderNodeTexVoronoi"]
    noise = [node for node in nodes if node.bl_idname == "ShaderNodeTexNoise"]
    principled = [
        node for node in nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"
    ]
    assert len(images) == 1
    assert images[0].image == bpy.data.images[0]
    assert len(voronoi) == 1
    assert voronoi[0].inputs["Scale"].default_value == 60.0
    assert len(noise) == 2
    assert not principled


@pytest.mark.parametrize(
    ("material_rand", "kwargs"),
    [
        (functionality_lists.fabric_art_rand, {"translucency": 0.0}),
        (functionality_lists._art_patterned_paint_rand, {}),
    ],
)
def test_art_pattern_preserves_physical_base_material(material_rand, kwargs):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vector = pf.nodes.shader.coord().uv
    material = material_rand(
        np.random.default_rng(13),
        vector,
        **kwargs,
    ).item()
    nodes = material.node_tree.nodes
    output = next(
        node for node in nodes if node.bl_idname == "ShaderNodeOutputMaterial"
    )
    assert any(node.bl_idname == "ShaderNodeTexImage" for node in nodes)
    assert any(node.bl_idname == "ShaderNodeGroup" for node in nodes)
    assert output.inputs["Displacement"].is_linked


def test_wall_art_uses_normalized_uv_coordinates():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vector = pf.nodes.shader.coord().uv
    material = wall_art._art_panel_material_rand(
        np.random.default_rng(23),
        vector,
        dimensions=pf.Vector((0.04, 0.8, 1.2)),
        frame_width=0.05,
    ).item()
    texcoord = next(
        node
        for node in material.node_tree.nodes
        if node.bl_idname == "ShaderNodeTexCoord"
    )
    assert texcoord.outputs["UV"].is_linked
    assert not texcoord.outputs["Generated"].is_linked
    assert any(
        node.bl_idname == "ShaderNodeVectorMath" and node.operation == "DIVIDE"
        for node in material.node_tree.nodes
    )


def _choice_options(monkeypatch, call, target):
    original_choice = pf.control.choice

    def capture_choice(rng, options):
        funcs = [
            option.func if isinstance(option, partial) else option
            for option, _ in options
        ]
        if target in funcs:
            raise RuntimeError(options)
        return original_choice(rng, options)

    monkeypatch.setattr(pf.control, "choice", capture_choice)
    with pytest.raises(RuntimeError) as exc_info:
        call()
    return exc_info.value.args[0]


@pytest.mark.parametrize(
    ("material_rand", "target"),
    [
        (
            functionality_lists.fabric_general_rand,
            functionality_lists.fabric_art_rand,
        ),
        (
            functionality_lists.wall_material_rand,
            functionality_lists._art_patterned_paint_rand,
        ),
    ],
)
def test_art_is_half_weight_indoor_material_option(monkeypatch, material_rand, target):
    vector = pf.nodes.shader.coord().uv
    options = _choice_options(
        monkeypatch,
        lambda: material_rand(np.random.default_rng(17), vector),
        target,
    )
    weights = {
        option.func if isinstance(option, partial) else option: weight
        for option, weight in options
    }
    assert weights[target] == 0.5


def test_art_is_half_of_wall_art_panel_choices(monkeypatch):
    vector = pf.nodes.shader.coord().uv
    frame_material = pf.Material(surface=pf.nodes.shader.principled_bsdf())
    options = _choice_options(
        monkeypatch,
        lambda: wall_art.wall_art_rand(
            np.random.default_rng(19),
            dimensions=pf.Vector((0.04, 0.8, 1.2)),
            frame_material=frame_material,
        ),
        wall_art._art_panel_material_rand,
    )
    weights = {
        option.func if isinstance(option, partial) else option: weight
        for option, weight in options
    }
    assert weights[wall_art._art_panel_material_rand] == 0.5
    assert [weight for _, weight in options] == [0.5, 0.5]
