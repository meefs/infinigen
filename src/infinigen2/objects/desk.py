# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from collections.abc import Callable
from typing import NamedTuple

import procfunc as pf

from infinigen2.objects import storage, table
from infinigen2.shaders.functionality_lists import table_top_material_rand

__all__ = [
    "DeskResult",
    "desk_dimensions_rand",
    "desk_rand",
]


class DeskResult(NamedTuple):
    mesh: pf.MeshObject


_DeskBase = Callable[[pf.RNG, pf.Vector, float], pf.MeshObject]
_DeskSurface = Callable[[pf.RNG, pf.Vector, _DeskBase], pf.MeshObject]


def desk_dimensions_rand(rng: pf.RNG) -> pf.Vector:
    """Desk footprint and height in meters, ordered as depth, width, height."""
    rng_depth, rng_width, rng_height = rng.spawn(3)
    return pf.Vector(
        (
            pf.random.uniform(rng_depth, 0.55, 0.90),
            pf.random.uniform(rng_width, 0.90, 2.40),
            pf.random.uniform(rng_height, 0.70, 0.78),
        )
    )


def _side_cabinet_width_frac_rand(rng: pf.RNG) -> float:
    return pf.random.uniform(rng, 0.15, 0.30)


def _desktop_support_loop_offset_rand(rng: pf.RNG) -> pf.Vector:
    rng_corner, rng_edge = rng.spawn(2)
    corner_offset = pf.random.uniform(rng_corner, 0.008, 0.025)
    edge_offset = pf.random.uniform(rng_edge, 0.002, 0.006)
    return pf.Vector((corner_offset, corner_offset, edge_offset))


def _integrated_storage_height_rand(rng: pf.RNG) -> float:
    return pf.random.uniform(rng, 0.10, 0.18)


def _under_top_storage_height_rand(rng: pf.RNG) -> float:
    return pf.random.uniform(rng, 0.08, 0.14)


def _side_cabinet_back_width_rand(rng: pf.RNG) -> float:
    rng_choice, rng_width = rng.spawn(2)
    return pf.control.choice(
        rng_choice,
        [
            (0.0, 1.0),
            (pf.random.uniform(rng_width, 0.012, 0.018), 1.0),
        ],
    )


def _side_cell_shelf_base_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    top_height: float,
    width_frac: float,
    back_width: float | None = None,
) -> pf.MeshObject:
    rng_frame, rng_row, rng_col, rng_back = rng.spawn(4)
    if back_width is None:
        back_width = _side_cabinet_back_width_rand(rng_back)
    cabinet_width = dimensions.y * width_frac
    frame_thickness = pf.random.uniform(rng_frame, 0.015, 0.025)
    row_divider_width = pf.random.uniform(rng_row, 0.012, 0.022)
    col_divider_width = pf.random.uniform(rng_col, 0.012, 0.022)
    cabinet_dimensions = pf.Vector((dimensions.x, cabinet_width, top_height))
    left = storage.storage_cell_shelf(
        dimensions=cabinet_dimensions,
        n_spaces_y=1,
        n_spaces_z=2,
        frame_thickness=frame_thickness,
        row_divider_width=row_divider_width,
        col_divider_width=col_divider_width,
        back_width=back_width,
    ).mesh
    right = storage.storage_cell_shelf(
        dimensions=cabinet_dimensions,
        n_spaces_y=1,
        n_spaces_z=2,
        frame_thickness=frame_thickness,
        row_divider_width=row_divider_width,
        col_divider_width=col_divider_width,
        back_width=back_width,
    ).mesh
    pf.ops.object.set_transform(
        left,
        location=(-dimensions.x / 2, -dimensions.y / 2, 0.0),
    )
    pf.ops.object.set_transform(
        right,
        location=(-dimensions.x / 2, dimensions.y / 2 - cabinet_width, 0.0),
    )
    pf.ops.object.join(left, right)
    return left


def _thin_cell_shelf_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    cabinet_height: float,
    bottom_height: float,
    frame_material: pf.Material | None = None,
) -> pf.MeshObject:
    rng_slots, rng_frame, rng_row, rng_col, rng_back = rng.spawn(5)
    n_spaces_y = pf.control.choice(rng_slots, [(2, 1.0), (3, 1.0), (4, 1.0)])
    result = storage.storage_cell_shelf(
        dimensions=pf.Vector((dimensions.x, dimensions.y, cabinet_height)),
        n_spaces_y=n_spaces_y,
        n_spaces_z=1,
        frame_thickness=pf.random.uniform(rng_frame, 0.012, 0.018),
        row_divider_width=pf.random.uniform(rng_row, 0.010, 0.016),
        col_divider_width=pf.random.uniform(rng_col, 0.010, 0.016),
        back_width=pf.random.uniform(rng_back, 0.012, 0.018),
        frame_material=frame_material,
    ).mesh
    pf.ops.object.set_transform(
        result,
        location=(-dimensions.x / 2, -dimensions.y / 2, bottom_height),
    )
    return result


def _ordinary_base_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    top_height: float,
) -> pf.MeshObject:
    return table.base_straight_rand(
        rng,
        dimensions=pf.Vector((dimensions.x, dimensions.y, top_height)),
        close_edges=True,
    ).mesh


def _cabinet_base_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    top_height: float,
) -> pf.MeshObject:
    rng_width, rng_base = rng.spawn(2)
    width_frac = _side_cabinet_width_frac_rand(rng_width)
    return _side_cell_shelf_base_rand(
        rng_base,
        dimensions,
        top_height,
        width_frac,
    )


def _desk_base_rand(rng: pf.RNG) -> _DeskBase:
    return pf.control.choice(
        rng,
        [
            (_ordinary_base_rand, 1.0),
            (_cabinet_base_rand, 1.0),
        ],
    )


def _desk_base_for_style_rand(
    rng: pf.RNG,
    base_style: str | None,
) -> _DeskBase:
    if base_style is None:
        return _desk_base_rand(rng)
    base_styles = {
        "ordinary": _ordinary_base_rand,
        "cabinet": _cabinet_base_rand,
    }
    return base_styles[base_style]


def _tabletop_surface_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    base_fn: _DeskBase,
) -> pf.MeshObject:
    rng_top, rng_top_shape, rng_base, rng_table = rng.spawn(4)
    top_thickness = pf.random.uniform(rng_top, 0.025, 0.055)
    top_height = dimensions.z - top_thickness
    base = base_fn(rng_base, dimensions, top_height)
    result = table.dining_table_rand(
        rng_table,
        dimensions=dimensions,
        base=base,
        top_thickness=top_thickness,
        top_support_loop_offset=_desktop_support_loop_offset_rand(rng_top_shape),
    )
    return result.mesh


def _integrated_storage_surface_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    base_fn: _DeskBase,
) -> pf.MeshObject:
    rng_height, rng_base, rng_cabinet, rng_material = rng.spawn(4)
    cabinet_height = _integrated_storage_height_rand(rng_height)
    base_height = dimensions.z - cabinet_height
    base = base_fn(rng_base, dimensions, base_height)
    material = table_top_material_rand(rng_material, pf.nodes.shader.coord().uv)
    cabinet = _thin_cell_shelf_rand(
        rng_cabinet,
        dimensions,
        cabinet_height=cabinet_height,
        bottom_height=base_height,
        frame_material=material,
    )
    pf.ops.object.join(cabinet, base)
    return cabinet


def _tabletop_storage_surface_rand(
    rng: pf.RNG,
    dimensions: pf.Vector,
    base_fn: _DeskBase,
) -> pf.MeshObject:
    (
        rng_top,
        rng_top_shape,
        rng_height,
        rng_base,
        rng_cabinet,
        rng_table,
    ) = rng.spawn(6)
    top_thickness = pf.random.uniform(rng_top, 0.025, 0.055)
    top_height = dimensions.z - top_thickness
    cabinet_height = _under_top_storage_height_rand(rng_height)
    base_height = top_height - cabinet_height
    base = base_fn(rng_base, dimensions, base_height)
    cabinet = _thin_cell_shelf_rand(
        rng_cabinet,
        dimensions,
        cabinet_height=cabinet_height,
        bottom_height=base_height,
    )
    pf.ops.object.join(base, cabinet)
    result = table.dining_table_rand(
        rng_table,
        dimensions=dimensions,
        base=base,
        top_thickness=top_thickness,
        top_support_loop_offset=_desktop_support_loop_offset_rand(rng_top_shape),
    )
    return result.mesh


def _desk_surface_rand(rng: pf.RNG) -> _DeskSurface:
    return pf.control.choice(
        rng,
        [
            (_tabletop_surface_rand, 1.0),
            (_integrated_storage_surface_rand, 1.0),
            (_tabletop_storage_surface_rand, 1.0),
        ],
    )


def _desk_surface_for_style_rand(
    rng: pf.RNG,
    surface_style: str | None,
) -> _DeskSurface:
    if surface_style is None:
        return _desk_surface_rand(rng)
    surface_styles = {
        "tabletop": _tabletop_surface_rand,
        "integrated_storage": _integrated_storage_surface_rand,
        "tabletop_storage": _tabletop_storage_surface_rand,
    }
    return surface_styles[surface_style]


def _desk_composition_rand(
    rng_base: pf.RNG,
    rng_surface: pf.RNG,
) -> tuple[_DeskBase, _DeskSurface]:
    base_fn = pf.control.choice(
        rng_base,
        [
            (_ordinary_base_rand, 7.0),
            (_cabinet_base_rand, 3.0),
        ],
    )
    if base_fn is _ordinary_base_rand:
        surface_fn = pf.control.choice(
            rng_surface,
            [
                (_tabletop_surface_rand, 5.0),
                (_integrated_storage_surface_rand, 1.0),
                (_tabletop_storage_surface_rand, 1.0),
            ],
        )
        return base_fn, surface_fn
    return base_fn, _desk_surface_rand(rng_surface)


def desk_rand(
    rng: pf.RNG,
    dimensions: pf.Vector | None = None,
    base_style: str | None = None,
    surface_style: str | None = None,
) -> DeskResult:
    """Desk with composable base and work-surface storage styles."""
    (
        rng_dims,
        rng_base_choice,
        rng_surface_choice,
        rng_surface,
    ) = rng.spawn(4)
    if dimensions is None:
        dimensions = desk_dimensions_rand(rng_dims)
    if base_style is None and surface_style is None:
        base_fn, surface_fn = _desk_composition_rand(
            rng_base_choice,
            rng_surface_choice,
        )
    else:
        base_fn = _desk_base_for_style_rand(rng_base_choice, base_style)
        surface_fn = _desk_surface_for_style_rand(rng_surface_choice, surface_style)
    mesh = surface_fn(
        rng_surface,
        dimensions,
        base_fn,
    )
    mesh.item().name = "desk"
    return DeskResult(mesh=mesh)
