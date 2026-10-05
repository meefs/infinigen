# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json
import random  # validate-ignore: test_determinism
from functools import cache
from pathlib import Path
from typing import NamedTuple

import procfunc as pf

__all__ = [
    "HumanResult",
    "human",
    "human_rand",
]


class HumanResult(NamedTuple):
    mesh: pf.MeshObject
    armature: pf.ArmatureObject


def _mpfb_services() -> dict:
    try:
        import mpfb  # keep-local
    except ImportError as e:
        raise ImportError("human() requires the mpfb package") from e
    return mpfb.initialize()["SERVICES"]


@cache
def _detail_catalog() -> dict[str, list[dict]]:
    services = _mpfb_services()
    targets_directory = Path(services["LocationService"].get_mpfb_data("targets"))
    catalog = json.loads((targets_directory / "target.json").read_text())
    sections = (
        "head",
        "forehead",
        "eyebrows",
        "eyes",
        "ears",
        "nose",
        "cheek",
        "mouth",
        "chin",
        "neck",
        "torso",
        "breast",
        "stomach",
        "hip",
        "pelvis",
        "buttocks",
        "arms",
        "hands",
        "legs",
        "feet",
    )
    return {name: catalog[name]["categories"] for name in sections}


def _human_body(
    macro: dict, detail_stack: list[dict], skin_material: pf.Material | None
) -> HumanResult:
    services = _mpfb_services()
    body = services["HumanService"].create_human(
        feet_on_ground=True,
        scale=0.1,
        macro_detail_dict=macro,
    )
    if detail_stack:
        services["TargetService"].bulk_load_targets(body, detail_stack)
    rig = services["HumanService"].add_builtin_rig(body, "default", import_weights=True)
    rig.location.z -= services["ObjectService"].get_lowest_point(body)

    mesh = pf.MeshObject(body)
    if skin_material is not None:
        pf.ops.object.set_material(mesh, skin_material)
    pf.ops.modifier.subdivide_surface(mesh, levels=2, _skip_apply=True)
    return HumanResult(mesh=mesh, armature=pf.ArmatureObject(rig))


def _mpfb_age(age: float) -> float:
    return 0.34375 + 0.65625 * age


def human(
    gender: float = 0.5,
    age: float = 0.25,
    muscle: float = 0.5,
    weight: float = 0.5,
    proportions: float = 0.5,
    height: float = 0.5,
    skin_material: pf.Material | None = None,
) -> HumanResult:
    macro = _mpfb_services()["TargetService"].get_default_macro_info_dict()
    macro["gender"] = gender
    macro["age"] = _mpfb_age(age)
    macro["muscle"] = muscle
    macro["weight"] = weight
    macro["proportions"] = proportions
    macro["height"] = height
    return _human_body(macro, [], skin_material)


def human_rand(rng: pf.RNG, skin_material: pf.Material | None = None) -> HumanResult:
    randomizer = _mpfb_services()["RandomizationService"]
    mpfb_rng = random.Random(int(rng.integers(2**32)))
    spec = randomizer.get_default_phenotype_spec()
    spec["phenotype"]["discrete_age"] = False
    age = spec["phenotype"]["attributes"]["age"]
    age["neutral"] = _mpfb_age(0.5)
    age["deviation"] = _mpfb_age(1.0) - _mpfb_age(0.5)
    macro = randomizer.randomize_macro_info_dict(spec, mpfb_rng)
    catalog = _detail_catalog()
    detail_spec = randomizer.get_default_detail_spec(list(catalog))
    stack = randomizer.pick_random_details(detail_spec, catalog, mpfb_rng)
    return _human_body(macro, stack, skin_material)
