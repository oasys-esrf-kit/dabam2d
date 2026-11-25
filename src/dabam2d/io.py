# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""Read and write two-dimensional metrology height profiles."""

from collections.abc import Mapping
from pathlib import Path
from struct import unpack_from
from tempfile import NamedTemporaryFile

import h5py
import numpy as np

__all__ = ["read_height_profile", "write_height_profile"]

# Binary MetroPro formats 1, 2 and 3.
_METROPRO_HEADERS = {
    b"\x88\x1b\x03\x6f": (1, 834),
    b"\x88\x1b\x03\x70": (2, 834),
    b"\x88\x1b\x03\x71": (3, 4096),
}
_PHASE_RESOLUTIONS = {0: 4096, 1: 32768, 2: 131072}
_INVALID_PHASE = 2147483640
_LENGTH_UNITS = {
    "meters": 1.0,
    "centimeters": 1e-2,
    "millimeters": 1e-3,
    "micrometers": 1e-6,
    "nanometers": 1e-9,
    "angstroms": 1e-10,
    "feet": 0.3048,
    "inches": 0.0254,
    "mils": 2.54e-5,
    "microinches": 2.54e-8,
    "nanoinches": 2.54e-11,
}


def read_height_profile(
    path: str | Path, *, axes_scale: float = 1.0, height_scale: float = 1.0
) -> dict[str, np.ndarray]:
    """Load a height map with one-dimensional x and y and z indexed as z[y, x].

    Supports DABAM2D HDF5, Zygo MetroPro DAT, Zygo Mx DATX, Zygo XYZ
    Format 1, SRW tabulated 2D surfaces, and comma-separated x,y,z triplets.
    Headerless triplets use the supplied scales to convert to metres. For
    calibrated formats the reader decodes to metres before applying the scales.
    Known invalid heights are represented by NaN. Axes are returned in ascending
    order. The scales must be finite, and axes_scale must be nonzero.

    Raises OSError for file access failures and ValueError for invalid or
    unsupported profile data.
    """
    path = Path(path)
    axes_scale, height_scale = float(axes_scale), float(height_scale)
    if not np.isfinite([axes_scale, height_scale]).all() or axes_scale == 0:
        raise ValueError("Scales must be finite and axes_scale must be nonzero")

    with path.open("rb") as stream:
        prefix = stream.read(64)

    try:
        if prefix[:4] in _METROPRO_HEADERS:
            x, y, z = _read_metropro(path)
        elif prefix.startswith(b"Zygo XYZ Data File"):
            x, y, z = _read_zygo_xyz(path)
        elif h5py.is_hdf5(path):
            with h5py.File(path, "r") as file:
                if "surface_file" in file:
                    x, y, z = (file[f"surface_file/{key}"][()] for key in "XYZ")
                elif "Data/Surface" in file:
                    x, y, z = _read_datx(file)
                else:
                    raise ValueError("unrecognized HDF5 height map")
        else:
            x, y, z = _read_text(path)

        z = _real_array(z)
        if np.isinf(z).any():
            raise ValueError("height contains infinite values")
        with np.errstate(over="ignore"):
            x = _real_array(x) * axes_scale
            y = _real_array(y) * axes_scale
            z = z * height_scale
        x, y, z = _order_surface(x, y, z)
        y -= np.nanmean(y)
        x -= np.nanmean(x)
        if np.isinf(z).any():
            raise ValueError("scaled height exceeds the floating-point range")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error

    return {"x": x, "y": y, "z": z}


def write_height_profile(
    path: str | Path, surface: Mapping[str, np.ndarray], *, overwrite: bool = False
) -> None:
    """Save a DABAM2D height map as float64 datasets under /surface_file.

    The surface has one-dimensional x and y coordinates in metres and a
    height array z in metres with shape (len(y), len(x)). Descending axes
    are reordered with their corresponding samples. NaN heights are
    preserved. Z uses lossless gzip compression with byte shuffling.
    Existing files raise FileExistsError unless overwrite is true.
    """
    path = Path(path)
    try:
        x, y, z = (_real_array(surface[key]) for key in ("x", "y", "z"))
        x, y, z = _order_surface(x, y, z)
        if np.isinf(z).any():
            raise ValueError("height contains infinite values")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error

    # Publish only after HDF5 has finished writing. A hard link creates the
    # destination exclusively, including when two writers race for that name.
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"File already exists: {path}")
    with NamedTemporaryFile(dir=path.parent, suffix=".h5", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        with h5py.File(pending, "w") as file:
            group = file.create_group("surface_file")
            group.create_dataset("X", data=x)
            group.create_dataset("Y", data=y)
            group.create_dataset("Z", data=z, compression="gzip", shuffle=True)
        if overwrite:
            pending.replace(path)
        else:
            path.hardlink_to(pending)
    finally:
        pending.unlink(missing_ok=True)


def _order_surface(x, y, z):
    if x.ndim != 1 or y.ndim != 1 or z.shape != (y.size, x.size):
        raise ValueError("expected 1D x and y and a height array of shape (len(y), len(x))")
    x, reverse_x = _ordered_axis(x)
    y, reverse_y = _ordered_axis(y)
    if reverse_x:
        z = z[:, ::-1]
    if reverse_y:
        z = z[::-1, :]
    return x, y, z


def _real_array(data):
    data = np.asarray(data)
    if data.dtype.kind not in "fiu":
        raise ValueError("expected real numeric coordinates and heights")
    return data.astype(float, copy=False)


def _ordered_axis(axis):
    if axis.size == 0 or not np.isfinite(axis).all():
        raise ValueError("coordinate axes must be nonempty and finite")
    if np.all(axis[1:] > axis[:-1]):
        return axis, False
    if np.all(axis[1:] < axis[:-1]):
        return axis[::-1], True
    raise ValueError("coordinate axes must be strictly monotonic")


def _read_text(path):
    with path.open(encoding="utf-8-sig") as stream:
        first_row = next(
            (line.split("#", 1)[0].strip() for line in stream
             if line.split("#", 1)[0].strip()),
            "",
        )
        if not first_row:
            raise ValueError("empty height map")
        stream.seek(0)
        if "," in first_row:
            data = np.loadtxt(stream, delimiter=",", ndmin=2)
            if data.shape[1] != 3:
                raise ValueError("expected three columns: x, y, height")
            x, ix = np.unique(data[:, 0], return_inverse=True)
            y, iy = np.unique(data[:, 1], return_inverse=True)
            return x, y, _grid(ix, iy, data[:, 2], x.size, y.size)

        # SRW's column helper transposes the disk table. Its surface consumer
        # then takes longitudinal coordinates across the first row.
        data = np.loadtxt(stream, ndmin=2)
    if min(data.shape) < 2:
        raise ValueError("expected an SRW surface table with two coordinate axes")
    return data[1:, 0], data[0, 1:], data[1:, 1:].T


def _grid(ix, iy, heights, nx, ny):
    if nx <= 0 or ny <= 0 or nx * ny != heights.size:
        raise ValueError(f"expected {nx} × {ny} samples, found {heights.size}")
    if np.any((ix < 0) | (ix >= nx)) or np.any((iy < 0) | (iy >= ny)):
        raise ValueError("sample index is outside the declared grid")
    indices = iy * nx + ix
    if np.unique(indices).size != indices.size:
        raise ValueError("duplicate coordinate pair in height map")
    z = np.empty((ny, nx))
    z.flat[indices] = heights
    return z


def _read_zygo_xyz(path):
    with path.open(encoding="latin-1") as stream:
        header = []
        for line in stream:
            if line.strip() == "#":
                break
            header.append(line)
        else:
            raise ValueError("missing Zygo XYZ data marker")
        if len(header) < 8 or header[0].strip() != "Zygo XYZ Data File - Format 1":
            raise ValueError("expected a Zygo XYZ Format 1 header")
        aperture = header[3].split()
        calibration = header[7].split()
        if len(aperture) < 4 or len(calibration) < 7:
            raise ValueError("incomplete Zygo XYZ calibration")
        x0, y0, nx, ny = map(int, aperture[:4])
        scale, wavelength, _, obliquity, _, pitch = map(float, calibration[1:7])
        rows = (line.replace("No Data", "nan").replace("NoData", "nan") for line in stream)
        data = np.loadtxt(rows, ndmin=2)
    if data.shape[1] != 3:
        raise ValueError("expected three Zygo XYZ columns")
    coordinates = data[:, :2]
    if not np.isfinite(coordinates).all() or np.any(coordinates != np.trunc(coordinates)):
        raise ValueError("Zygo XYZ pixel indices must be finite integers")
    # Mx can export the invalid count as a rounded numeric height. Match the
    # sentinel for each MetroPro phase resolution, since XYZ lacks the binary
    # resolution field used to distinguish them.
    heights = data[:, 2] * 1e-6
    phase_scale = scale * wavelength * obliquity
    if phase_scale > 0:
        for resolution in _PHASE_RESOLUTIONS.values():
            invalid_height = _INVALID_PHASE * phase_scale / resolution
            heights[np.isclose(heights, invalid_height, rtol=5e-7, atol=0)] = np.nan
    z = _grid(coordinates[:, 0].astype(np.intp) - x0,
              coordinates[:, 1].astype(np.intp) - y0, heights, nx, ny)
    return _pixel_axis(x0, nx, pitch), _pixel_axis(y0, ny, pitch), z


def _pixel_axis(start, size, pitch):
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("missing or invalid lateral calibration")
    return (start + np.arange(size, dtype=float)) * pitch


def _read_metropro(path):
    # Offsets for the fields shared by all three MetroPro header revisions.
    with path.open("rb") as stream:
        prefix = stream.read(10)
        if len(prefix) != 10:
            raise ValueError("truncated MetroPro header")
        version, header_size = unpack_from(">HI", prefix, 4)
        if (version, header_size) != _METROPRO_HEADERS[prefix[:4]]:
            raise ValueError("invalid MetroPro header version or size")
        header = prefix + stream.read(header_size - len(prefix))
        if len(header) != header_size:
            raise ValueError("truncated MetroPro header")

        intensity_width, intensity_height, buckets = unpack_from(">HHH", header, 52)
        x0, y0, nx, ny = unpack_from(">hhHH", header, 64)
        scale, wavelength, _, obliquity, _, pitch = unpack_from(">6f", header, 164)
        resolution, = unpack_from(">H", header, 218)
        pixel_width, pixel_height = unpack_from(">2f", header, 486)
        if resolution not in _PHASE_RESOLUTIONS or nx == 0 or ny == 0:
            raise ValueError("invalid MetroPro phase resolution or surface dimensions")
        if not np.isfinite([scale, wavelength, obliquity]).all() or wavelength <= 0:
            raise ValueError("invalid MetroPro height calibration")

        stream.seek(2 * intensity_width * intensity_height * buckets, 1)
        data_start = stream.tell()
        if stream.seek(0, 2) - data_start < 4 * nx * ny:
            raise ValueError("truncated MetroPro phase data")
        stream.seek(data_start)
        counts = np.fromfile(stream, dtype=">i4", count=nx * ny)
    if counts.size != nx * ny:
        raise ValueError("truncated MetroPro phase data")
    counts = counts.reshape(ny, nx)
    z = counts * (scale * wavelength * obliquity / _PHASE_RESOLUTIONS[resolution])
    z[counts >= _INVALID_PHASE] = np.nan
    x = _pixel_axis(x0, nx, pixel_width if pixel_width > 0 else pitch)
    y = _pixel_axis(y0, ny, pixel_height if pixel_height > 0 else pitch)
    return x, y, z


def _attribute(attrs, key):
    return np.asarray(attrs[key]).item()


def _unit(value):
    return (value.decode("ascii") if isinstance(value, bytes) else str(value)).lower()


def _datx_resolution(file, surface):
    # Follow the Mx measurement links when calibration lives outside the surface.
    if "MetaData" in file:
        links = {(source, link): target for source, link, target in file["MetaData"][:]}
        for (measurement, link), target in links.items():
            if link == b"Surface" and links.get((target, b"Path")) == surface.name.encode():
                attrs = file[links[measurement, b"Attributes"]].attrs
                for context in ("Surface Data Context", "Data Context"):
                    key = f"{context}.Lateral Resolution"
                    if f"{key}:Value" in attrs and f"{key}:Unit" in attrs:
                        unit = _unit(_attribute(attrs, f"{key}:Unit"))
                        return _attribute(attrs, f"{key}:Value") * _LENGTH_UNITS[unit]
    raise ValueError("missing Zygo DATX lateral resolution")


def _read_datx(file):
    group = file["Data/Surface"]
    if isinstance(group, h5py.Dataset):
        surface = group
    else:
        surfaces = [item for item in group.values() if isinstance(item, h5py.Dataset)]
        if len(surfaces) != 1:
            raise ValueError("expected exactly one Zygo DATX surface")
        surface = surfaces[0]
    if surface.ndim != 2:
        raise ValueError("expected a two-dimensional Zygo DATX surface")

    attrs = surface.attrs
    ny, nx = surface.shape
    x0 = y0 = 0
    if "Coordinates" in attrs:
        coords = np.asarray(attrs["Coordinates"]).reshape(-1)[0]
        if (coords["Width"], coords["Height"]) != (nx, ny):
            raise ValueError("Zygo DATX window and surface dimensions disagree")
        x0, y0 = int(coords["ColumnStart"]), int(coords["RowStart"])

    axes = []
    for key, start, size in (("X Converter", x0, nx), ("Y Converter", y0, ny)):
        if key in attrs:
            converter = np.asarray(attrs[key]).reshape(-1)[0]
            if _unit(converter["Category"]) != "lateralcat" or _unit(converter["BaseUnit"]) != "pixels":
                raise ValueError(f"unsupported Zygo DATX {key}")
            offset, pitch, *_ = converter["Parameters"]
        else:
            offset, pitch = 0.0, _datx_resolution(file, surface)
        axes.append(offset + _pixel_axis(start, size, pitch))

    z = _real_array(surface[()])
    if "No Data" in attrs:
        z[z >= _attribute(attrs, "No Data")] = np.nan
    unit = _unit(_attribute(attrs, "Unit"))
    if unit == "fringes":
        factor = (
            _attribute(attrs, "Wavelength")
            * _attribute(attrs, "Obliquity Factor")
            * _attribute(attrs, "Interferometric Scale Factor")
        )
    else:
        factor = _LENGTH_UNITS[unit]
    return axes[0], axes[1], z * factor
