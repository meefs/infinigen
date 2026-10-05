# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import procfunc as pf

from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.room import bathroom_setup
from infinigen2.scenes.setup_utils import BareMeshResult


def _fixture_result(
    sink_setup: bathroom_setup.BathroomSinkSetupResult,
) -> bathroom_setup.BathroomSetupResult:
    fixtures = bathroom_setup._BathroomFixtures(
        sink_setup=sink_setup,
        toilets=[],
        bathtub_setup=bathroom_setup._BathtubSetupResult([], [], []),
    )
    return bathroom_setup._fixture_result(fixtures, ccol.collision_set([]))


def test_sink_storage_capabilities_do_not_depend_on_names() -> None:
    sink = pf.ops.primitives.mesh_cube(size=0.4)
    cabinet = pf.ops.primitives.mesh_cube(size=0.5)
    parts = bathroom_setup._bathroom_result(
        [BareMeshResult(sink)], [], [BareMeshResult(cabinet)]
    )
    cabinet.item().name = "renamed_after_creation"

    result = _fixture_result(bathroom_setup._bathroom_sink_result(parts))

    assert result.storage_containers == [sink, cabinet]
    assert result.supports == [sink, cabinet]
    assert result.storages == [cabinet]


def test_culled_sink_support_loses_storage_capabilities(monkeypatch) -> None:
    sink = pf.ops.primitives.mesh_cube(size=0.4)
    cabinet = pf.ops.primitives.mesh_cube(size=0.5)
    setup = bathroom_setup._bathroom_sink_result(
        bathroom_setup._bathroom_result(
            [BareMeshResult(sink)], [], [BareMeshResult(cabinet)]
        )
    )

    def cull_supports(items, colliders):
        if items == setup.sink_supports:
            return [], colliders
        return items, colliders

    monkeypatch.setattr(bathroom_setup, "keep_non_colliding", cull_supports)
    culled = bathroom_setup._keep_valid_bathroom_sink_components(
        setup,
        ccol.collision_set([]),
        wall_planes=[],
        wall_margin=0.05,
    )
    result = _fixture_result(culled)

    assert result.storage_containers == [sink]
    assert result.supports == [sink]
    assert result.storages == []
