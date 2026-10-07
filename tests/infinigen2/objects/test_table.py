# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import numpy as np
import procfunc as pf
import pytest

from infinigen2.objects import furniture_bases, table


@pytest.mark.parametrize(
    "generator", [furniture_bases.base_straight_rand, furniture_bases.base_square_rand]
)
def test_base_choice_uses_unsampled_rng(
    monkeypatch: pytest.MonkeyPatch,
    generator: Callable[..., table.TableResult],
) -> None:
    sampled_rngs: set[int] = set()
    original_uniform = pf.random.uniform
    original_choice = pf.control.choice

    def capture_uniform(rng: pf.RNG, *args: float) -> float:
        sampled_rngs.add(id(rng))
        return original_uniform(rng, *args)

    def assert_clean_choice(rng: pf.RNG, options: list[tuple[object, float]]) -> object:
        assert id(rng) not in sampled_rngs
        return original_choice(rng, options)

    monkeypatch.setattr(pf.random, "uniform", capture_uniform)
    monkeypatch.setattr(pf.control, "choice", assert_clean_choice)
    material = pf.Material(surface=pf.nodes.shader.principled_bsdf())

    generator(
        np.random.default_rng(3),
        dimensions=pf.Vector((1.2, 0.7, 0.7)),
        material=material,
    )
