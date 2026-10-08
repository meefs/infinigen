import bpy
import pytest

from infinigen2.lighting import sky_lighting


def _sky_node(world: bpy.types.World) -> bpy.types.Node:
    group_nodes = [
        node for node in world.node_tree.nodes if node.bl_idname == "ShaderNodeGroup"
    ]
    assert len(group_nodes) == 1
    sky_nodes = [
        node
        for node in group_nodes[0].node_tree.nodes
        if node.bl_idname == "ShaderNodeTexSky"
    ]
    assert len(sky_nodes) == 1
    return sky_nodes[0]


@pytest.mark.parametrize(
    ("function_name", "sky_type"),
    [
        ("_nishita_sky", "NISHITA"),
        ("_hosek_wilkie_sky", "HOSEK_WILKIE"),
    ],
)
def test_sky_model_builds(function_name: str, sky_type: str) -> None:
    world = getattr(sky_lighting, function_name)().item()
    assert _sky_node(world).sky_type == sky_type


def test_nishita_sky_builds_without_sun_disc() -> None:
    world = sky_lighting._nishita_sky(sun_disc=False).item()
    sky = _sky_node(world)
    assert sky.sky_type == "NISHITA"
    assert sky.sun_disc is False
