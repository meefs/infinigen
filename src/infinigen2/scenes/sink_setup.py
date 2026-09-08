# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: ported from Infinigen v1 SinkFactory (Hongyu Wen, Meenal Parakh, Stamatis Alexandropoulos, Alexander Raistrick)

from typing import NamedTuple

import procfunc as pf

from infinigen2.objects.sink import sink_rand
from infinigen2.objects.tap import tap_rand

__all__ = ["SinkSetupResult", "sink_setup_rand"]


class SinkSetupResult(NamedTuple):
    all_objects: list[pf.MeshObject]


@pf.tracer.grammar
def sink_setup_rand(
    rng: pf.RNG,
    sink_material: pf.Material | None = None,
    tap_material: pf.Material | None = None,
) -> SinkSetupResult:
    rng_sink, rng_tap = rng.spawn(2)
    sink_result = sink_rand(rng_sink, material=sink_material)
    tap_object = tap_rand(rng_tap, material=tap_material).mesh
    pf.ops.object.set_transform(tap_object, location=sink_result.tap_mount)
    return SinkSetupResult(all_objects=[sink_result.mesh, tap_object])
