# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from typing import Iterable

import bpy
import procfunc as pf

__all__ = [
    "cleanup_except",
    "delete_object",
    "delete_objects",
]

logger = logging.getLogger(__name__)


def delete_objects(objs: Iterable[bpy.types.Object]) -> None:
    unique = list({obj.as_pointer(): obj for obj in objs}.values())
    if not unique:
        return
    owned = {
        obj.data.as_pointer(): obj.data
        for obj in unique
        if obj.type in ("MESH", "LIGHT") and obj.data is not None
    }
    logger.debug(f"Deleting {len(unique)} objects: {[obj.name for obj in unique]}")
    bpy.data.batch_remove(unique)
    orphans = [data for data in owned.values() if data.users == 0]
    if orphans:
        bpy.data.batch_remove(orphans)


def delete_object(obj: bpy.types.Object) -> None:
    delete_objects([obj])


def cleanup_except(keep: Iterable[pf.Object]) -> list[str]:
    """Delete every bpy.data.objects entry not in `keep`.

    Placement helpers (e.g. repeat_attempts) leave failed-placement objects
    in the scene. They are excluded from the returned `all_objects` lists but
    still live in the blend and would otherwise be rendered. Pass the union of
    objects/cameras/lights you want kept and this removes the rest.
    """
    valid = {o.item() for o in keep}
    doomed = [asset for asset in bpy.data.objects if asset not in valid]
    cleaned = []
    for asset in doomed:
        cleaned.append(asset.name)
        asset.name = asset.name + "_CLEANED"
    delete_objects(doomed)
    logger.info(f"Cleaned {len(cleaned)} stray objects from scene")
    return cleaned
