# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf

from infinigen2.objects import random_primitives
from infinigen2.objects.lamp import LampResult
from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.room import decoration_objects


def _resolve_label(label: str) -> tuple[object, object]:
    primitive, _, effect = label.partition("_effect_")
    return (
        getattr(random_primitives, primitive, None),
        getattr(random_primitives, f"effect_{effect}", None),
    )


def test_small_objects_pool_has_unique_stems_and_sampler_labels(rng: pf.RNG) -> None:
    pool = decoration_objects.small_objects_collection_rand(rng)
    data_names = [obj.item().data.name for obj in pool]
    labels = [decoration_objects._smallobj_label(name) for name in data_names]

    assert len(data_names) == len(set(data_names))

    for label in labels:
        assert "." not in label, label
        primitive, effect = _resolve_label(label)
        assert callable(primitive), label
        assert callable(effect), label


def test_small_objects_are_never_labelled_after_the_wrapper(rng: pf.RNG) -> None:
    pool = decoration_objects.small_objects_collection_rand(rng)
    labels = {decoration_objects._smallobj_label(obj.item().data.name) for obj in pool}

    assert random_primitives.primitive_with_effect_rand.__name__ not in labels
    assert random_primitives.primitive_rand.__name__ not in labels
    assert len(labels) > 1


def test_small_objects_label_survives_alias_copy_suffix() -> None:
    label = "cube_rand_effect_twist"
    assert decoration_objects._smallobj_label(f"{label}_004.003") == label


def test_rejected_surface_lamp_does_not_return_light(monkeypatch, rng: pf.RNG) -> None:
    lamp_result = LampResult(
        mesh=pf.ops.primitives.mesh_cube(size=0.2),
        light=pf.ops.primitives.point_lamp(energy=1000),
    )
    support = pf.ops.primitives.mesh_cube(size=1.0)
    monkeypatch.setattr(
        decoration_objects,
        "_sample_surface_collection",
        lambda *_: [lamp_result],
    )
    monkeypatch.setattr(decoration_objects, "retry_place", lambda *_, **__: None)

    result = decoration_objects.decorate_surface_objects_rand(
        rng,
        objects=[support],
        colliders=ccol.collision_set([]),
        support_tops=[support],
    )

    assert lamp_result.mesh not in result.all_objects
    assert lamp_result.light not in result.lights


def test_kept_surface_lamp_preserves_original_result(monkeypatch, rng: pf.RNG) -> None:
    lamp_result = LampResult(
        mesh=pf.ops.primitives.mesh_cube(size=0.2),
        light=pf.ops.primitives.point_lamp(energy=1000),
    )
    support = pf.ops.primitives.mesh_cube(size=1.0)
    seen = []
    monkeypatch.setattr(
        decoration_objects,
        "_sample_surface_collection",
        lambda *_: [lamp_result],
    )

    def keep_original(_rng, child, *_args, **_kwargs):
        seen.append(child)
        return child

    monkeypatch.setattr(decoration_objects, "retry_place", keep_original)
    result = decoration_objects.decorate_surface_objects_rand(
        rng,
        objects=[support],
        colliders=ccol.collision_set([]),
        support_tops=[support],
    )

    assert seen == [lamp_result]
    assert lamp_result.mesh in result.all_objects
    assert result.lights == [lamp_result.light]
