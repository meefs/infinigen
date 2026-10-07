# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bpy
import procfunc as pf

from infinigen2.scenes.placement import collision as ccol
from infinigen2.scenes.room import bathroom_setup
from infinigen2.scenes.setup_utils import BareMeshResult, standalone_wall_planes


def _cabinet_unit() -> bathroom_setup._SinkUnit:
    sink = pf.ops.primitives.mesh_cube(size=0.4)
    tap = pf.ops.primitives.mesh_cube(size=0.05)
    cabinet = BareMeshResult(pf.ops.primitives.mesh_cube(size=0.5))
    return bathroom_setup._SinkUnit(
        BareMeshResult(sink), BareMeshResult(tap), [cabinet], [cabinet]
    )


def _fixture_result(
    sink_setup: bathroom_setup.BathroomSinkSetupResult,
) -> bathroom_setup.BathroomSetupResult:
    return bathroom_setup._fixture_result(
        sink_setup,
        [],
        bathroom_setup._BathtubSetupResult([], [], []),
        [],
        ccol.collision_set([]),
        [],
    )


def test_sink_storage_capabilities_do_not_depend_on_names() -> None:
    unit = _cabinet_unit()
    sink_setup = bathroom_setup._sink_setup_result(
        unit, unit.supports, unit.storages, bathroom_setup._WallFeatureResult([], [])
    )
    unit.supports[0].mesh.item().name = "renamed_after_creation"

    result = _fixture_result(sink_setup)

    sink = unit.sink.mesh
    cabinet = unit.supports[0].mesh
    assert result.storage_containers == [sink, cabinet]
    assert result.supports == [sink, cabinet]
    assert result.storages == [cabinet]


def test_culled_sink_support_loses_storage_capabilities(rng: pf.RNG) -> None:
    unit = _cabinet_unit()
    blocker = pf.ops.primitives.mesh_cube(size=0.5)

    sink_setup = bathroom_setup._sink_extras_rand(
        rng, unit, [], 3.0, ccol.collision_set([blocker])
    )
    result = _fixture_result(sink_setup)

    assert sink_setup.sink_supports == []
    assert result.storage_containers == [unit.sink.mesh]
    assert result.supports == [unit.sink.mesh]
    assert result.storages == []


def test_existing_sink_setup_keeps_caller_sink(rng: pf.RNG) -> None:
    wall = standalone_wall_planes(length=5.0, height=3.0)[0]
    sink = pf.ops.primitives.mesh_cube(size=0.5)
    pf.ops.object.set_transform(sink, location=(0.255, 2.5, 0.65))

    result = bathroom_setup.bathroom_existing_sink_setup_rand(rng, sink, [wall])

    assert sink.item().name in bpy.data.objects
    assert result.named_objects["bathroom_sink"] == [sink]
    assert sink.item() not in {obj.item() for obj in result.temporary_objects}
