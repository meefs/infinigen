# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf
import pytest
from procfunc.util.manifest import import_item

from infinigen2 import list as list_command

_MASK_PRESETS = [
    pathspec
    for pathspec in list_command.preset_dotted_names()
    if list_command.preset_categories()[pathspec.rsplit(".", 1)[-1]] == "Mask"
]


@pytest.mark.parametrize("pathspec", _MASK_PRESETS)
def test_mask_preset_exposes_mask(pathspec: str) -> None:
    vector = pf.nodes.shader.coord().object
    result = import_item(pathspec)(vector=vector)

    assert isinstance(result.mask, pf.ProcNode)
