# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""Plot two-dimensional metrology height profiles."""

from __future__ import annotations

from collections.abc import Mapping
from math import floor, log10
from typing import TYPE_CHECKING

import numpy as np

from .io import _order_surface, _real_array

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

__all__ = ["plot_height_profile", "show"]

_AXES_UNITS = {"m": (1.0, "m"), "mm": (1e3, "mm"), "um": (1e6, "µm")}
_HEIGHT_UNITS = {"um": (1e6, "µm"), "nm": (1e9, "nm"), "A": (1e10, "Å")}


def plot_height_profile(
    surface: Mapping[str, np.ndarray],
    *,
    axes_unit: str = "mm",
    height_unit: str = "nm",
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Colormap = "viridis",
    scale_bar: bool = False,
    aspect_ratio: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a surface dictionary whose x, y and z arrays are in metres.

    Use m, mm or um for both lateral axes, and um, nm or A for height.
    Supply x and y limits in axes_unit and colour limits in height_unit.
    None uses the corresponding automatic limit. Set aspect_ratio=False
    to fill the plot area for long, narrow surfaces. By default the plot
    preserves the physical aspect ratio. NaN heights are left blank.
    Pass a Matplotlib colormap name or Colormap instance as cmap.

    Return the Matplotlib figure and map axes without showing the figure.
    Pass an existing ax to embed the plot in another figure.
    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    if axes_unit not in _AXES_UNITS:
        raise ValueError("axes_unit must be 'm', 'mm', or 'um'")
    if height_unit not in _HEIGHT_UNITS:
        raise ValueError("height_unit must be 'um', 'nm', or 'A'")
    axis_factor, axis_label = _AXES_UNITS[axes_unit]
    height_factor, height_label = _HEIGHT_UNITS[height_unit]
    _validate_limits(xmin, xmax, "x")
    _validate_limits(ymin, ymax, "y")
    _validate_limits(vmin, vmax, "height")

    x, y, z = _order_surface(*(_real_array(surface[key]) for key in ("x", "y", "z")))
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.get_figure()

    mesh = ax.pcolormesh(
        x * axis_factor,
        y * axis_factor,
        z * height_factor,
        cmap=cmap,
        shading="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set(
        xlabel=f"x ({axis_label})",
        ylabel=f"y ({axis_label})",
        aspect="equal" if aspect_ratio else "auto",
    )
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    fig.colorbar(mesh, ax=ax, label=f"Height ({height_label})")

    if scale_bar:
        left, right = ax.get_xlim()
        target = abs(right - left) / 4
        decade = 10.0 ** floor(log10(target))
        length = max(step for step in (1, 2, 5) if step <= target / decade) * decade
        bar = AnchoredSizeBar(
            ax.transData, length, f"{length:g} {axis_label}",
            loc="lower right", pad=0.4, sep=3, frameon=True,
        )
        ax.add_artist(bar)

    return fig, ax


def show(*, block: bool | None = None) -> None:
    """Display open Matplotlib figures using the active backend.

    With block=None, Matplotlib chooses whether to wait for the windows to close.
    """
    from matplotlib import pyplot as plt

    plt.show(block=block)


def _validate_limits(lower, upper, label):
    for limit in (lower, upper):
        if limit is not None and not np.isfinite(limit):
            raise ValueError(f"{label} limits must be finite")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(f"{label} minimum must be less than its maximum")
