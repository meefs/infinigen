import bpy
import numpy as np
import procfunc as pf

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
