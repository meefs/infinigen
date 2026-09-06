import colorsys

import numpy as np
import procfunc as pf
import pytest

from infinigen2.shaders import functionality_lists


def test_wall_paint_color_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    colors: list[pf.Color] = []

    def capture_material(
        _rng: np.random.Generator,
        _vector: object,
        displacement_pct: float,
        base_color: pf.Color | None,
    ) -> None:
        assert 0.0 <= displacement_pct <= 0.8
        assert base_color is not None
        colors.append(base_color)

    monkeypatch.setattr(functionality_lists.paint, "paint_rand", capture_material)

    for seed in range(4096):
        functionality_lists.paint_wall_rand(np.random.default_rng(seed), None)

    hsv = np.asarray([colorsys.rgb_to_hsv(*color) for color in colors])
    saturations = hsv[:, 1]
    values = hsv[:, 2]
    assert np.median(saturations) < 0.06
    assert 0.14 < np.mean(saturations > 0.7) < 0.17
    assert np.median(values) < 0.53
    assert np.mean(values < 0.1) > 0.04
    assert np.mean(values < 0.25) > 0.18
    assert np.mean(values > 0.9) > 0.025
