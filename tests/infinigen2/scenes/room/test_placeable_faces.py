# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from typing import Callable

import numpy as np
import procfunc as pf
import pytest

from infinigen2.objects import bathtub, bed, desk, sink, table, toilet
from infinigen2.scenes import setup_utils
from infinigen2.scenes.room import bed_setup
from infinigen2.scenes.room import decoration_objects as deco

SEEDS = [0, 1]

HOSTS: dict[str, Callable[[pf.RNG], pf.MeshObject]] = {
    "sofa": lambda rng: setup_utils.sofa_object_rand(rng).mesh,
    "mattress": lambda rng: bed.bed_rand(rng).mattress_child,
    "storage": lambda rng: setup_utils.storage_object_rand(rng).mesh,
    "side_table": lambda rng: setup_utils.side_table_object_rand(rng).mesh,
    "coffee_table": lambda rng: table.table_coffee_rand(rng).mesh,
    "coffee_table_storage": lambda rng: table.table_coffee_storage_rand(rng).mesh,
    "bedside_table": lambda rng: bed_setup.table_bedside_composite_rand(rng).mesh,
    "desk": lambda rng: desk.desk_rand(rng).mesh,
    "toilet": lambda rng: toilet.toilet_rand(rng).mesh,
    "sink": lambda rng: sink.sink_rand(rng).mesh,
    "bathroom_sink": lambda rng: bathtub.sink_bathroom_rand(rng).mesh,
    "bathtub": lambda rng: bathtub.bathtub_rand(rng).mesh,
}

MIN_AREA = {
    ("sofa", "container"): 0.3,
    ("mattress", "support"): 1.0,
    ("storage", "container"): 0.1,
    ("storage", "support"): 0.05,
    ("side_table", "support"): 0.05,
    ("coffee_table", "support"): 0.2,
    ("coffee_table_storage", "container"): 0.1,
    ("bedside_table", "support"): 0.05,
    ("desk", "support"): 0.3,
    ("toilet", "support"): 0.02,
    ("sink", "support"): 0.05,
    ("sink", "container"): 0.1,
    ("bathroom_sink", "support"): 0.02,
    ("bathroom_sink", "container"): 0.05,
    ("bathtub", "support"): 0.1,
    ("bathtub", "container"): 0.1,
}

ROLES = {"container": deco.container_faces, "support": deco.support_top_faces}


@pf.nodes.node_function
def _eligible_faces(
    parent: pf.ProcNode[pf.MeshObject],
    selection: pf.ProcNode[bool],
) -> pf.ProcNode[pf.MeshObject]:
    return pf.nodes.geo.separate_geometry(
        geometry=parent, selection=selection, domain="FACE"
    ).selection


def _area(geometry: pf.ProcNode) -> float:
    return float(pf.ops.attr.polygon_areas(pf.nodes.to_mesh_object(geometry)).sum())


def _raw_and_filtered(host: pf.MeshObject, role: str) -> tuple[float, float]:
    raw = _area(_eligible_faces(host, ROLES[role](host)))
    filtered = _area(deco.placeable_faces(host, ROLES[role](host)))
    return raw, filtered


def test_small_islands_are_dropped() -> None:
    big = pf.nodes.geo.mesh_grid(vertices_x=2, vertices_y=2, size_x=0.2, size_y=0.2)
    small = pf.nodes.geo.transform(
        pf.nodes.geo.mesh_grid(
            vertices_x=2, vertices_y=2, size_x=0.03, size_y=0.03
        ).mesh,
        translation=(1.0, 0.0, 0.0),
    )
    plates = pf.nodes.to_mesh_object(pf.nodes.geo.join_geometry([big.mesh, small]))

    assert _area(deco.placeable_faces(plates, True)) == pytest.approx(0.04, rel=1e-3)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(("host_name", "role"), sorted(MIN_AREA))
def test_filter_keeps_most_of_each_host(host_name: str, role: str, seed: int) -> None:
    host = HOSTS[host_name](np.random.default_rng(seed))
    raw, filtered = _raw_and_filtered(host, role)
    if raw == 0:
        pytest.skip(f"{host_name} has no {role} faces")
    assert filtered >= 0.5 * raw, (
        f"{host_name} {role}: kept {filtered:.4f} of {raw:.4f}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(("host_name", "role"), sorted(MIN_AREA))
def test_hosts_keep_placeable_area(host_name: str, role: str, seed: int) -> None:
    host = HOSTS[host_name](np.random.default_rng(seed))
    _, filtered = _raw_and_filtered(host, role)
    assert filtered >= MIN_AREA[(host_name, role)], (
        f"{host_name} {role}: {filtered:.4f}"
    )
