# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np

from infinigen2 import GENERATORS_MANIFEST
from infinigen2.scenes.house import furnishings
from infinigen2.scenes.house import unfurnished as house


def test_house_furnished_rand_tracks_room_contents() -> None:
    result = furnishings.house_furnished_rand(
        np.random.default_rng(5),
        dimensions=(8.0, 7.0),
        room_count=1,
        height=2.8,
    )

    assert len(result.rooms) == 1
    room_objects = [obj for room in result.rooms for obj in room.all_objects]
    assert all(obj in result.all_objects for obj in room_objects)
    assert all(obj in result.all_objects for obj in result.small_objects)


def test_furnish_house_room_rand_returns_only_new_objects() -> None:
    result = house.house_unfurnished_rand(
        np.random.default_rng(5),
        dimensions=(8.0, 7.0),
        room_count=1,
        height=2.8,
    )
    selected = result.rooms[0]

    furnished = furnishings.furnish_house_room_rand(
        np.random.default_rng(1),
        selected.boundary_rings,
        selected.floor,
        selected.flat_walls,
        result.dimensions.z,
        result.colliders,
    )

    assert furnished.all_objects
    assert all(obj not in result.all_objects for obj in furnished.all_objects)
    assert all(obj in furnished.all_objects for obj in furnished.small_objects)


def test_house_manifest_exposes_unfurnished_and_furnished_levels() -> None:
    names = GENERATORS_MANIFEST["name"].str.rsplit(".", n=1).str[-1]
    entries = GENERATORS_MANIFEST[
        names.isin(
            ["house_shape_rand", "house_unfurnished_rand", "house_furnished_rand"]
        )
    ]

    assert set(entries["name"].str.rsplit(".", n=1).str[-1]) == {
        "house_unfurnished_rand",
        "house_furnished_rand",
    }
