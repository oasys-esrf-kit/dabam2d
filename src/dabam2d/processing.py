# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""Processing operations for two-dimensional height maps."""

from collections.abc import Mapping
from operator import index

import numpy as np

__all__ = ["fill_missing_heights"]


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
