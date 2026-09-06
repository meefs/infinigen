# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import numpy as np
import pytest

from infinigen2.shaders.base_materials import plastic


def test_plastic_grayscale_color_rand_is_bimodal() -> None:
    rngs = np.random.default_rng(0).spawn(4096)
    colors = [plastic.plastic_grayscale_color_rand(rng) for rng in rngs]
    values = [color.v for color in colors]
    perceptual_values = np.asarray(values) ** (1 / 2.2)
    dark_rate = np.mean(perceptual_values <= 0.15)
    middle_rate = np.mean((perceptual_values > 0.15) & (perceptual_values < 0.35))

    assert dark_rate == pytest.approx(0.6, abs=0.03)
    assert middle_rate == 0.0
