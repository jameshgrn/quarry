"""Shared SWOT metadata extraction utilities.

Shared by SLCConnector, PIXCConnector, and any future SWOT product connectors.
NOT used by FOFStackConnector (different attribute schema).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def decode_hdf5_attr(val: Any) -> Any:
    """Decode a single HDF5 attribute value to a Python native type."""
    if isinstance(val, bytes):
        return val.decode()
    if isinstance(val, np.ndarray):
        if val.size == 1:
            item = val.item()
            return item.decode() if isinstance(item, bytes) else item
        return val.tolist()
    if isinstance(val, np.generic):
        item = val.item()
        return item.decode() if isinstance(item, bytes) else item
    return val


def extract_swot_metadata(attrs: dict[str, Any]) -> dict[str, Any]:
    """Extract common SWOT root attributes shared across L1B/L2 products.

    Args:
        attrs: dict(h5py.File.attrs) — raw HDF5 root attributes.

    Returns:
        Dict with normalized SWOT metadata keys.
    """
    d = decode_hdf5_attr

    return {
        "tile_name": d(attrs.get("tile_name", b"")),
        "cycle": int(d(attrs.get("cycle_number", [0]))),
        "pass_number": int(d(attrs.get("pass_number", [0]))),
        "swath_side": d(attrs.get("swath_side", b"")),
        "time_start": d(attrs.get("time_coverage_start", b"")),
        "time_end": d(attrs.get("time_coverage_end", b"")),
        "lat_bounds": (
            float(d(attrs.get("geospatial_lat_min", [0]))),
            float(d(attrs.get("geospatial_lat_max", [0]))),
        ),
        "lon_bounds": (
            float(d(attrs.get("geospatial_lon_min", [0]))),
            float(d(attrs.get("geospatial_lon_max", [0]))),
        ),
    }


def extract_slc_metadata(attrs: dict[str, Any]) -> dict[str, Any]:
    """Extract SLC-specific attributes on top of common SWOT metadata.

    Args:
        attrs: dict(h5py.File.attrs) — raw HDF5 root attributes.

    Returns:
        Dict with common SWOT + SLC-specific metadata.
    """
    d = decode_hdf5_attr
    meta = extract_swot_metadata(attrs)
    meta.update(
        {
            "transmit_antenna": d(attrs.get("transmit_antenna", b"")),
            "wavelength_m": float(d(attrs.get("wavelength", [0]))),
            "near_range_m": float(d(attrs.get("near_range", [0]))),
            "range_spacing_m": float(d(attrs.get("nominal_slant_range_spacing", [0]))),
            "azimuth_resolution_m": float(d(attrs.get("slc_along_track_resolution", [0]))),
        }
    )
    return meta
