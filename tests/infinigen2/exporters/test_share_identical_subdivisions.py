# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf

from infinigen2.exporters.realize_mesh import (
    evaluate_shared_subdivision_to_shared_data,
)
from infinigen2.scenes.placement.distribute import propagate_modifiers_to_instances


def _template(name, levels):
    obj = pf.ops.primitives.mesh_cube(size=0.2)
    obj.item().name = obj.item().data.name = name
    pf.ops.modifier.subdivide_surface(obj, levels=levels, _skip_apply=True)
    return obj


def _aliases_of(templates):
    points = pf.ops.primitives.mesh_grid(size=2, x_subdivisions=3, y_subdivisions=3)
    instances = pf.nodes.geo.instance_on_points(
        points=pf.nodes.geo.object_info(points, transform_space="RELATIVE").geometry,
        instance=pf.nodes.geo.collection_info(
            pf.Collection(templates), separate_children=True
        ),
        pick_instance=True,
    )
    aliases = pf.nodes.to_aliases(instances)
    propagate_modifiers_to_instances(templates, aliases)
    return aliases


def test_grouped_aliases_share_one_baked_datablock():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 2)]
    aliases = _aliases_of(templates)
    evaluate_shared_subdivision_to_shared_data(aliases)

    assert len(aliases) > 1
    assert len({alias.item().data.as_pointer() for alias in aliases}) == 1
    for alias in aliases:
        assert not list(alias.item().modifiers)
        assert len(alias.item().data.vertices) > 8
    assert [mod.type for mod in templates[0].item().modifiers] == ["SUBSURF"]
    assert len(templates[0].item().data.vertices) == 8


def test_groups_stay_separate_per_template():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 1), _template("tpl_b", 2)]
    aliases = _aliases_of(templates)
    evaluate_shared_subdivision_to_shared_data(aliases)

    counts = {len(alias.item().data.vertices) for alias in aliases}
    assert len(counts) == 2
    assert len({alias.item().data.as_pointer() for alias in aliases}) == 2


def test_singleton_keeps_deferred_stack():
    pf.ops.object.clear_scene()
    lone = _template("lone", 2)
    evaluate_shared_subdivision_to_shared_data([lone])

    assert [mod.type for mod in lone.item().modifiers] == ["SUBSURF"]
    assert len(lone.item().data.vertices) == 8


def test_differing_stacks_on_shared_data_stay_deferred():
    pf.ops.object.clear_scene()
    a = _template("stack_a", 1)
    b = pf.ops.primitives.mesh_cube(size=0.2)
    b.item().data = a.item().data
    pf.ops.modifier.subdivide_surface(b, levels=3, _skip_apply=True)
    evaluate_shared_subdivision_to_shared_data([a, b])

    assert [mod.levels for mod in a.item().modifiers] == [1]
    assert [mod.levels for mod in b.item().modifiers] == [3]
    assert a.item().data.as_pointer() == b.item().data.as_pointer()


def test_second_run_is_a_noop():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 2)]
    aliases = _aliases_of(templates)
    objects = aliases
    evaluate_shared_subdivision_to_shared_data(objects)
    once = len(aliases[0].item().data.vertices)
    evaluate_shared_subdivision_to_shared_data(objects)

    assert len(aliases[0].item().data.vertices) == once
    assert len({alias.item().data.as_pointer() for alias in aliases}) == 1


def test_bake_respects_render_visibility():
    pf.ops.object.clear_scene()
    templates = [_template("tpl_a", 2)]
    aliases = _aliases_of(templates)
    for alias in aliases:
        assert not alias.item().modifiers[0].show_viewport
    evaluate_shared_subdivision_to_shared_data(aliases)

    for alias in aliases:
        assert len(alias.item().data.vertices) > 8
