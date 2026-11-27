# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""Legacy azimuthal averaging from SpeckleMetroPy, in pixel coordinates."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _pixel_axes(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    x = np.arange(nx, dtype=float) - (nx // 2)
    y = np.arange(ny, dtype=float) - (ny // 2)
    return x, y



def radial_mean_interpolated(
    signal_2d: np.ndarray,
    *,
    r_max: float | None = None,
    nr: int | None = None,
    ntheta: int | None = None,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Radial mean of a 2D signal via polar sampling + interpolation (high-definition).

    Parameters:
        signal_2d (np.ndarray):
            2D signal to radially average.
        r_max (float | None):
            Maximum radius in pixels. If None, uses the inscribed circle radius in pixels.
        nr (int | None):
            Number of radial samples. If None, uses floor(r_max) + 1.
        ntheta (int | None):
            Number of angular samples. If None, uses 0.5 degree sampling.
        fill_value (float):
            Value used outside interpolation bounds (default: 0.0).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - radial (np.ndarray): radially averaged values.
            - r (np.ndarray): radial distances (pixels).

    Raises:
        ValueError
    """
    z = np.asarray(signal_2d, dtype=float)
    if z.ndim != 2:
        raise ValueError("signal_2d must be a 2D array.")
    if not np.isfinite(z).all():
        raise ValueError("signal_2d contains non-finite values.")

    ny, nx = z.shape
    x, y = _pixel_axes((ny, nx))

    if r_max is None:
        r_max = min(float(-x[0]), float(x[-1]), float(-y[0]), float(y[-1]))
    if r_max <= 0:
        raise ValueError("r_max must be > 0 (or leave it as None with valid shape).")

    if nr is None:
        nr = int(np.floor(r_max)) + 1
    if ntheta is None:
        ntheta = 2 * 360

    if nr <= 1:
        raise ValueError("nr must be > 1.")
    if ntheta <= 3:
        raise ValueError("ntheta must be > 3.")

    r = np.linspace(0.0, r_max, nr)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)

    R_grid, THETA_grid = np.meshgrid(r, theta, indexing="ij")
    X_grid = R_grid * np.cos(THETA_grid)
    Y_grid = R_grid * np.sin(THETA_grid)

    interp = RegularGridInterpolator((y, x), z, bounds_error=False, fill_value=fill_value)

    points = np.column_stack([Y_grid.ravel(), X_grid.ravel()])
    values = interp(points).reshape(R_grid.shape)

    radial = np.mean(values, axis=1)
    return radial, r
