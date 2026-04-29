"""Tests for SLCConnector (structural mapper, no processing)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from quarry_connectors.slc import SLCConnector

# ---------------------------------------------------------------------------
# Fixtures — synthetic SWOT SLC HDF5 files
# ---------------------------------------------------------------------------


@pytest.fixture
def slc_h5(tmp_path: Path) -> Path:
    """Minimal SWOT SLC HDF5 file with expected structure."""
    path = tmp_path / "SWOT_L1B_HR_SLC_001_100_200L.h5"
    with h5py.File(str(path), "w") as f:
        # Root attributes (SWOT metadata)
        f.attrs["tile_name"] = b"200L"
        f.attrs["cycle_number"] = np.array([1])
        f.attrs["pass_number"] = np.array([100])
        f.attrs["swath_side"] = b"L"
        f.attrs["transmit_antenna"] = b"plus_y"
        f.attrs["wavelength"] = np.array([0.00836])
        f.attrs["near_range"] = np.array([850000.0])
        f.attrs["nominal_slant_range_spacing"] = np.array([0.75])
        f.attrs["slc_along_track_resolution"] = np.array([6.0])
        f.attrs["geospatial_lat_min"] = np.array([30.0])
        f.attrs["geospatial_lat_max"] = np.array([31.0])
        f.attrs["geospatial_lon_min"] = np.array([-90.0])
        f.attrs["geospatial_lon_max"] = np.array([-89.0])
        f.attrs["time_coverage_start"] = b"2024-01-01T00:00:00Z"
        f.attrs["time_coverage_end"] = b"2024-01-01T00:01:00Z"

        # SLC datasets — complex stored as (H, W, 2) [real, imag]
        slc_grp = f.create_group("slc")
        for name in ["slc_plus_y", "slc_minus_y"]:
            data = np.stack(
                [
                    np.random.rand(20, 30).astype(np.float32),
                    np.random.rand(20, 30).astype(np.float32),
                ],
                axis=-1,
            )
            slc_grp.create_dataset(name, data=data)

        # Calibration datasets
        xf_grp = f.create_group("xfactor")
        xf_grp.create_dataset("xfactor_plus_y", data=np.ones((20, 30), dtype=np.float32))
        xf_grp.create_dataset("xfactor_minus_y", data=np.ones((20, 30), dtype=np.float32))

        # Noise datasets
        noise_grp = f.create_group("noise")
        noise_grp.create_dataset("noise_plus_y", data=np.full(20, 0.1, dtype=np.float32))
        noise_grp.create_dataset("noise_minus_y", data=np.full(20, 0.1, dtype=np.float32))

    return path


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_name():
    assert SLCConnector().name == "slc"


def test_capabilities():
    from quarry_core.connector import ConnectorCapability

    caps = SLCConnector().capabilities
    assert ConnectorCapability.MATERIALIZE in caps
    assert ConnectorCapability.DISCOVER in caps
    assert ConnectorCapability.METADATA_ONLY in caps
    assert ConnectorCapability.MATERIALIZE_LAZY in caps


def test_satisfies_connector_protocol():
    from quarry_core.connector import Connector

    assert isinstance(SLCConnector(), Connector)


# ---------------------------------------------------------------------------
# Discover — file-level (list datasets)
# ---------------------------------------------------------------------------


def test_discover_file_lists_slc_datasets(slc_h5: Path):
    conn = SLCConnector()
    entries = conn.discover(str(slc_h5))

    names = {e.name for e in entries}
    assert "slc_plus_y" in names
    assert "slc_minus_y" in names
    assert "xfactor_plus_y" in names
    assert "noise_plus_y" in names
    assert len(entries) == 6


def test_discover_file_has_roles(slc_h5: Path):
    conn = SLCConnector()
    entries = conn.discover(str(slc_h5))

    roles = {e.name: e.metadata["role"] for e in entries}
    assert roles["slc_plus_y"] == "data"
    assert roles["xfactor_plus_y"] == "calibration"
    assert roles["noise_plus_y"] == "noise"


def test_discover_file_has_swot_metadata(slc_h5: Path):
    conn = SLCConnector()
    entries = conn.discover(str(slc_h5))

    entry = [e for e in entries if e.name == "slc_plus_y"][0]
    assert entry.metadata["cycle"] == 1
    assert entry.metadata["pass"] == 100
    assert entry.metadata["swath"] == "L"
    assert entry.metadata["tile"] == "200L"


def test_discover_file_has_spatial_hint(slc_h5: Path):
    conn = SLCConnector()
    entries = conn.discover(str(slc_h5))

    entry = entries[0]
    assert entry.spatial_hint["crs"] == "EPSG:4326"
    extent = entry.spatial_hint["extent"]
    assert extent[0] == pytest.approx(-90.0)


# ---------------------------------------------------------------------------
# Discover — directory-level
# ---------------------------------------------------------------------------


def test_discover_directory(slc_h5: Path):
    conn = SLCConnector()
    entries = conn.discover(str(slc_h5.parent))

    assert len(entries) == 1
    assert entries[0].metadata["cycle"] == 1


def test_discover_empty_directory(tmp_path: Path):
    conn = SLCConnector()
    assert conn.discover(str(tmp_path)) == []


def test_discover_skips_non_hdf5(tmp_path: Path):
    (tmp_path / "test.txt").touch()
    (tmp_path / "test.tif").touch()
    conn = SLCConnector()
    assert conn.discover(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# Materialize — lazy
# ---------------------------------------------------------------------------


def test_materialize_lazy_by_logical_name(slc_h5: Path, tmp_path: Path):
    conn = SLCConnector()
    result = conn.materialize(f"{slc_h5}::slc_plus_y", tmp_path, lazy=True)

    assert result.strategy == "lazy_handle"
    assert result.artifact.metadata["source"] == "slc"
    assert result.artifact.metadata["role"] == "data"
    assert "swot" in result.artifact.metadata


def test_materialize_lazy_by_hdf5_path(slc_h5: Path, tmp_path: Path):
    conn = SLCConnector()
    result = conn.materialize(f"{slc_h5}::/slc/slc_plus_y", tmp_path, lazy=True)

    assert result.strategy == "lazy_handle"
    assert result.artifact.metadata["role"] == "data"


def test_materialize_lazy_auto_select(slc_h5: Path, tmp_path: Path):
    conn = SLCConnector()
    result = conn.materialize(str(slc_h5), tmp_path, lazy=True)

    assert result.strategy == "lazy_handle"
    assert result.artifact.metadata["source"] == "slc"


# ---------------------------------------------------------------------------
# Materialize — eager
# ---------------------------------------------------------------------------


def test_materialize_eager_writes_geotiff(slc_h5: Path, tmp_path: Path):
    import rasterio

    conn = SLCConnector()
    result = conn.materialize(f"{slc_h5}::slc_plus_y", tmp_path)

    assert result.strategy == "normalized"
    output = Path(result.artifact.backing.uri)
    assert output.exists()

    with rasterio.open(str(output)) as src:
        # SLC complex pair → 2 bands (real, imag)
        assert src.count == 2


def test_materialize_eager_xfactor(slc_h5: Path, tmp_path: Path):
    import rasterio

    conn = SLCConnector()
    result = conn.materialize(f"{slc_h5}::xfactor_plus_y", tmp_path)

    assert result.artifact.metadata["role"] == "calibration"
    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.count == 1


def test_materialize_eager_has_swot_metadata(slc_h5: Path, tmp_path: Path):
    conn = SLCConnector()
    result = conn.materialize(f"{slc_h5}::slc_plus_y", tmp_path)

    swot = result.artifact.metadata["swot"]
    assert swot["cycle"] == 1
    assert swot["pass_number"] == 100
    assert swot["wavelength_m"] == pytest.approx(0.00836)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_returns_swot_fields(slc_h5: Path):
    conn = SLCConnector()
    meta = conn.metadata(str(slc_h5))

    assert meta["tile_name"] == "200L"
    assert meta["cycle"] == 1
    assert meta["pass_number"] == 100
    assert "datasets" in meta
    assert "slc_plus_y" in meta["datasets"]


def test_metadata_includes_group_structure(slc_h5: Path):
    conn = SLCConnector()
    meta = conn.metadata(str(slc_h5))

    tree = meta["group_structure"]
    assert "/slc" in tree
    assert "/xfactor" in tree
    assert "/noise" in tree


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_materialize_missing_file(tmp_path: Path):
    from quarry_core.connector import MaterializeError

    conn = SLCConnector()
    with pytest.raises(MaterializeError, match="File not found"):
        conn.materialize("/nonexistent/file.h5", tmp_path)


def test_materialize_wrong_extension(tmp_path: Path):
    from quarry_core.connector import MaterializeError

    conn = SLCConnector()
    with pytest.raises(MaterializeError, match="Not an HDF5 file"):
        conn.materialize("/some/path/file.txt", tmp_path)


def test_metadata_missing_file():
    from quarry_core.connector import MaterializeError

    conn = SLCConnector()
    with pytest.raises(MaterializeError, match="File not found"):
        conn.metadata("/nonexistent/file.h5")
