# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""Processing operations for two-dimensional height maps."""

from collections.abc import Mapping
from operator import index

import numpy as np

from .io import _order_surface, _real_array

__all__ = [
    "fill_missing_heights",
    "subtract_minimum_height",
    "transpose_surface",
    "flip_surface",
    "rotate_surface",
    "slice_surface",
]


def fill_missing_heights(
    surface: Mapping[str, np.ndarray], radius: int
) -> dict[str, np.ndarray]:
    """Fill NaN heights with the median of finite neighbors within radius pixels.

    The radius is a positive integer. The circular neighborhood is clipped
    to the map boundaries and sampled from the original height array. A NaN
    with no finite neighbors within the radius remains NaN.

    Return a new surface dictionary with a copy of z and the original axes.
    """
    if isinstance(radius, (bool, np.bool_)):
        raise TypeError("radius must be an integer number of pixels")
    radius = index(radius)
    if radius < 1:
        raise ValueError("radius must be at least one pixel")

    x, y = np.asarray(surface["x"]), np.asarray(surface["y"])
    z = np.asarray(surface["z"])
    if x.ndim != 1 or y.ndim != 1 or z.shape != (y.size, x.size):
        raise ValueError("expected 1D x and y and a height array of shape (len(y), len(x))")
    if z.dtype.kind not in "fiu":
        raise ValueError("expected real numeric heights")

    filled = z.copy()
    ny, nx = z.shape
    radius_squared = radius**2
    for row, col in np.argwhere(np.isnan(z)):
        row, col = int(row), int(col)
        top, bottom = max(0, row - radius), min(ny, row + radius + 1)
        left, right = max(0, col - radius), min(nx, col + radius + 1)
        dy, dx = np.ogrid[top - row:bottom - row, left - col:right - col]
        local = z[top:bottom, left:right]
        neighbors = local[(dx**2 + dy**2 <= radius_squared) & np.isfinite(local)]
        if neighbors.size:
            filled[row, col] = np.median(neighbors)

    return {**surface, "z": filled}


def subtract_minimum_height(surface: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Subtract the smallest finite height in metres, preserving NaNs."""
    x, y, z = _surface_arrays(surface)
    shifted = z.copy()
    finite = np.isfinite(z)
    if finite.any():
        shifted[finite] -= z[finite].min()
    return {**surface, "x": x, "y": y, "z": shifted}


def transpose_surface(surface: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Swap the physical x and y coordinates and transpose the heights."""
    x, y, z = _surface_arrays(surface)
    return {**surface, "x": y, "y": x, "z": z.T.copy()}


def flip_surface(
    surface: Mapping[str, np.ndarray], axis: str
) -> dict[str, np.ndarray]:
    """Reflect the x or y coordinate about the midpoint of its extent.

    The reflected axis is returned in increasing order, with its samples
    reordered accordingly. Pass "x" for left/right or "y" for up/down.
    """
    x, y, z = _surface_arrays(surface)
    if axis == "x":
        x, z = _reflect_axis(x), z[:, ::-1]
    elif axis == "y":
        y, z = _reflect_axis(y), z[::-1, :]
    else:
        raise ValueError("axis must be 'x' or 'y'")
    return {**surface, "x": x, "y": y, "z": z.copy()}


def rotate_surface(
    surface: Mapping[str, np.ndarray], angle_deg: float, *, fill_value: float = 0.0
) -> dict[str, np.ndarray]:
    """Rotate a surface about the midpoint of its x and y extents.

    Angles are in degrees, positive counterclockwise in the physical x-y
    plane. Quarter turns preserve heights and NaNs exactly. Other angles
    require regular axes with at least two samples each. Their two pitches
    are retained independently on an expanded grid, with bilinear sampling
    in physical coordinates. Points outside the input grid or requiring a
    nonfinite sample are set to fill_value, in metres.
    """
    angle_deg, fill_value = float(angle_deg), float(fill_value)
    if not np.isfinite(angle_deg):
        raise ValueError("angle_deg must be finite")
    if not np.isfinite(fill_value):
        raise ValueError("fill_value must be finite")

    x, y, z = _surface_arrays(surface)
    angle_deg %= 360.0
    cx = (x[0] + x[-1]) / 2
    cy = (y[0] + y[-1]) / 2

    if angle_deg == 0:
        return {**surface, "x": x, "y": y, "z": z.copy()}
    if angle_deg == 90:
        return {**surface, "x": cx - (y[::-1] - cy), "y": cy + (x - cx),
                "z": np.rot90(z, -1).copy()}
    if angle_deg == 180:
        return {**surface, "x": _reflect_axis(x), "y": _reflect_axis(y),
                "z": z[::-1, ::-1].copy()}
    if angle_deg == 270:
        return {**surface, "x": cx + (y - cy), "y": cy - (x[::-1] - cx),
                "z": np.rot90(z).copy()}

    dx, dy = _regular_pitch(x, "x"), _regular_pitch(y, "y")
    radians = np.deg2rad(angle_deg)
    cosine, sine = np.cos(radians), np.sin(radians)
    width, height = x[-1] - x[0], y[-1] - y[0]
    rotated_width = abs(cosine) * width + abs(sine) * height
    rotated_height = abs(sine) * width + abs(cosine) * height
    rx = _centered_grid(rotated_width, dx)
    ry = _centered_grid(rotated_height, dy)

    # Inverse rotation maps each output sample back into the original grid.
    source_x = cosine * rx[None, :] + sine * ry[:, None]
    source_y = -sine * rx[None, :] + cosine * ry[:, None]
    rotated = _sample_bilinear(x - cx, y - cy, z, source_x, source_y, fill_value)
    return {**surface, "x": rx + cx, "y": ry + cy, "z": rotated}


def slice_surface(
    surface: Mapping[str, np.ndarray], direction: str, position: float
) -> dict[str, np.ndarray | float | str]:
    """Extract a linearly interpolated height profile at position, in metres.

    Use "h" for a cut parallel to x at the given y, or "v" for a cut
    parallel to y at the given x. Return the full parallel axis, the
    corresponding heights, the requested perpendicular position, and direction.

    The position must be inside the inclusive perpendicular axis range.
    At a measured coordinate the original cut is returned. Between two
    coordinates, a missing height on either bracketing cut produces NaN.
    """
    if direction not in ("h", "v"):
        raise ValueError("direction must be 'h' or 'v'")
    position = float(position)
    x, y, z = _surface_arrays(surface)
    if direction == "h":
        axis, perpendicular, cuts = x, y, z
    else:
        axis, perpendicular, cuts = y, x, z.T

    if not np.isfinite(position) or not perpendicular[0] <= position <= perpendicular[-1]:
        raise ValueError(
            f"position must be within [{perpendicular[0]}, {perpendicular[-1]}] m"
        )

    upper = np.searchsorted(perpendicular, position)
    if perpendicular[upper] == position:
        profile = cuts[upper].copy()
    else:
        lower = upper - 1
        weight = (position - perpendicular[lower]) / (perpendicular[upper] - perpendicular[lower])
        profile = (1 - weight) * cuts[lower] + weight * cuts[upper]

    return {"axis": axis.copy(), "profile": profile, "position": position, "direction": direction}


def _surface_arrays(surface):
    return _order_surface(*(_real_array(surface[key]) for key in ("x", "y", "z")))


def _reflect_axis(axis):
    return axis[0] + (axis[-1] - axis[::-1])


def _regular_pitch(axis, name):
    if axis.size < 2:
        raise ValueError(f"arbitrary rotation requires at least two {name} samples")
    pitch = (axis[-1] - axis[0]) / (axis.size - 1)
    # Native HDF5 profiles can contain regularly sampled axes rounded to float32.
    if not np.allclose(np.diff(axis), pitch, rtol=1e-4, atol=0):
        raise ValueError(f"arbitrary rotation requires a regularly spaced {name} axis")
    return pitch


def _centered_grid(span, pitch):
    intervals = int(np.ceil(span / pitch))
    return (np.arange(intervals + 1) - intervals / 2) * pitch


def _sample_bilinear(x, y, z, source_x, source_y, fill_value):
    # The tolerance covers rounding at the footprint edge or at exact samples.
    x_tol = 8 * np.finfo(float).eps * (x[-1] - x[0])
    y_tol = 8 * np.finfo(float).eps * (y[-1] - y[0])
    inside = ((source_x >= x[0] - x_tol) & (source_x <= x[-1] + x_tol)
              & (source_y >= y[0] - y_tol) & (source_y <= y[-1] + y_tol))
    px = np.clip(source_x[inside], x[0], x[-1])
    py = np.clip(source_y[inside], y[0], y[-1])
    ix = np.clip(np.searchsorted(x, px, side="right") - 1, 0, x.size - 2)
    iy = np.clip(np.searchsorted(y, py, side="right") - 1, 0, y.size - 2)

    tx = (px - x[ix]) / (x[ix + 1] - x[ix])
    ty = (py - y[iy]) / (y[iy + 1] - y[iy])
    for fraction, offset, grid, indices, tolerance in (
        (tx, px, x, ix, x_tol), (ty, py, y, iy, y_tol)
    ):
        fraction[np.abs(offset - grid[indices]) <= tolerance] = 0.0
        fraction[np.abs(offset - grid[indices + 1]) <= tolerance] = 1.0

    values = np.zeros(px.size)
    supported = np.ones(px.size, dtype=bool)
    for row, col, weight in (
        (iy, ix, (1 - ty) * (1 - tx)),
        (iy, ix + 1, (1 - ty) * tx),
        (iy + 1, ix, ty * (1 - tx)),
        (iy + 1, ix + 1, ty * tx),
    ):
        sample = z[row, col]
        finite = np.isfinite(sample)
        supported &= finite | (weight == 0)
        values += np.where(finite, sample, 0.0) * weight

    rotated = np.full(source_x.shape, fill_value)
    rotated[inside] = np.where(supported, values, fill_value)
    return rotated
