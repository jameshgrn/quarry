"""PIXCConnector pressure test.

Lane: connector

Stress points:
1. Protocol conformance (name, capabilities, Connector isinstance)
2. Discover — file-level and directory-level listing
3. Materialize lazy — returns lazy_handle with correct spatial descriptor
4. Materialize eager — rasterizes sparse pixel cloud to 4-band GeoTIFF
5. target_grid parameter — rasterize onto a specific grid
6. Band descriptions in output (sig0, height, water_frac, classification)
7. Metadata extraction (tile, cycle, pass, swath, pixel count, bounds)
8. Error handling (missing file, wrong extension, missing pixel_cloud group)
9. Lineage records source path and resolution
10. PIXCMetadata dataclass populated correctly
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from quarry_connectors.pixc import PIXCConnector
from quarry_core.connector import ConnectorCapability, MaterializeError

# ---------------------------------------------------------------------------
# Fixtures — synthetic SWOT PIXC HDF5 files
# ---------------------------------------------------------------------------

NUM_PIXELS = 500


@pytest.fixture
def pixc_h5(tmp_path: Path) -> Path:
    """Minimal SWOT PIXC HDF5 file with pixel_cloud group."""
    path = tmp_path / "SWOT_L2_HR_PIXC_016_089_133L_20240601.nc"
    rng = np.random.default_rng(42)

    lat_min, lat_max = 30.0, 31.0
    lon_min, lon_max = -90.0, -89.0

    with h5py.File(str(path), "w") as f:
        # Root attributes
        f.attrs["tile_name"] = b"133L"
        f.attrs["cycle_number"] = np.array([16])
        f.attrs["pass_number"] = np.array([89])
        f.attrs["swath_side"] = b"L"
        f.attrs["time_coverage_start"] = b"2024-06-01T08:39:25Z"
        f.attrs["time_coverage_end"] = b"2024-06-01T08:39:36Z"
        f.attrs["geospatial_lat_min"] = np.array([lat_min])
        f.attrs["geospatial_lat_max"] = np.array([lat_max])
        f.attrs["geospatial_lon_min"] = np.array([lon_min])
        f.attrs["geospatial_lon_max"] = np.array([lon_max])

        # pixel_cloud group
        pc = f.create_group("pixel_cloud")
        pc.create_dataset(
            "latitude",
            data=rng.uniform(lat_min, lat_max, NUM_PIXELS).astype(np.float64),
        )
        pc.create_dataset(
            "longitude",
            data=rng.uniform(lon_min, lon_max, NUM_PIXELS).astype(np.float64),
        )
        pc.create_dataset(
            "height",
            data=rng.uniform(100.0, 110.0, NUM_PIXELS).astype(np.float32),
        )
        pc.create_dataset(
            "sig0",
            data=rng.uniform(-20.0, 0.0, NUM_PIXELS).astype(np.float32),
        )
        pc.create_dataset(
            "classification",
            data=rng.integers(1, 8, NUM_PIXELS).astype(np.uint8),
        )
        pc.create_dataset(
            "water_frac",
            data=rng.uniform(0.0, 1.0, NUM_PIXELS).astype(np.float32),
        )

    return path


@pytest.fixture
def pixc_h5_no_pixel_cloud(tmp_path: Path) -> Path:
    """HDF5 file missing the pixel_cloud group."""
    path = tmp_path / "no_pixel_cloud.nc"
    with h5py.File(str(path), "w") as f:
        f.attrs["tile_name"] = b"bad"
    return path


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_name():
    assert PIXCConnector().name == "pixc"


def test_capabilities():
    caps = PIXCConnector().capabilities
    assert ConnectorCapability.MATERIALIZE in caps
    assert ConnectorCapability.DISCOVER in caps
    assert ConnectorCapability.METADATA_ONLY in caps
    assert ConnectorCapability.MATERIALIZE_LAZY in caps


def test_satisfies_connector_protocol():
    from quarry_core.connector import Connector

    assert isinstance(PIXCConnector(), Connector)


# ---------------------------------------------------------------------------
# Discover — file-level
# ---------------------------------------------------------------------------


def test_discover_directory_lists_pixc_files(pixc_h5: Path):
    conn = PIXCConnector()
    entries = conn.discover(str(pixc_h5.parent))
    assert len(entries) == 1
    assert entries[0].metadata["cycle"] == 16
    assert entries[0].metadata["pass"] == 89
    assert entries[0].metadata["swath"] == "L"


def test_discover_has_spatial_hint(pixc_h5: Path):
    conn = PIXCConnector()
    entries = conn.discover(str(pixc_h5.parent))
    entry = entries[0]
    assert entry.spatial_hint["crs"] == "EPSG:4326"
    extent = entry.spatial_hint["extent"]
    assert extent[0] == pytest.approx(-90.0)
    assert extent[1] == pytest.approx(30.0)


def test_discover_has_pixel_count(pixc_h5: Path):
    conn = PIXCConnector()
    entries = conn.discover(str(pixc_h5.parent))
    assert entries[0].metadata["pixels"] == NUM_PIXELS


def test_discover_empty_directory(tmp_path: Path):
    conn = PIXCConnector()
    assert conn.discover(str(tmp_path)) == []


def test_discover_skips_non_hdf5(tmp_path: Path):
    (tmp_path / "test.txt").touch()
    (tmp_path / "test.tif").touch()
    conn = PIXCConnector()
    assert conn.discover(str(tmp_path)) == []


def test_discover_skips_non_pixc_hdf5(pixc_h5_no_pixel_cloud: Path):
    """Files without pixel_cloud group are silently skipped."""
    conn = PIXCConnector()
    entries = conn.discover(str(pixc_h5_no_pixel_cloud.parent))
    assert entries == []


def test_discover_none_query_uses_cwd():
    conn = PIXCConnector()
    # Should not raise — uses "." as default
    entries = conn.discover(None)
    assert isinstance(entries, list)


def test_discover_dict_query(pixc_h5: Path):
    conn = PIXCConnector()
    entries = conn.discover({"path": str(pixc_h5.parent), "recursive": False})
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# Materialize — lazy
# ---------------------------------------------------------------------------


def test_materialize_lazy_returns_lazy_handle(pixc_h5: Path, tmp_path: Path):
    conn = PIXCConnector()
    result = conn.materialize(str(pixc_h5), tmp_path, lazy=True)
    assert result.strategy == "lazy_handle"


def test_materialize_lazy_artifact_type(pixc_h5: Path, tmp_path: Path):
    from quarry_core.artifact import ArtifactType

    conn = PIXCConnector()
    result = conn.materialize(str(pixc_h5), tmp_path, lazy=True)
    assert result.artifact.type == ArtifactType.RASTER


def test_materialize_lazy_spatial_descriptor(pixc_h5: Path, tmp_path: Path):
    conn = PIXCConnector()
    result = conn.materialize(str(pixc_h5), tmp_path, lazy=True)
    spatial = result.artifact.spatial
    assert spatial.crs == "EPSG:4326"
    assert spatial.band_count == 4
    assert spatial.extent[0] == pytest.approx(-90.0)
    assert spatial.extent[2] == pytest.approx(-89.0)


def test_materialize_lazy_metadata(pixc_h5: Path, tmp_path: Path):
    conn = PIXCConnector()
    result = conn.materialize(str(pixc_h5), tmp_path, lazy=True)
    meta = result.artifact.metadata
    assert meta["source"] == "pixc"
    assert meta["cycle"] == 16
    assert meta["pass"] == 89
    assert meta["swath"] == "L"
    assert meta["pixels"] == NUM_PIXELS


# ---------------------------------------------------------------------------
# Materialize — eager
# ---------------------------------------------------------------------------


def test_materialize_eager_writes_geotiff(pixc_h5: Path, tmp_path: Path):
    import rasterio

    conn = PIXCConnector()
    result = conn.materialize(str(pixc_h5), tmp_path)

    assert result.strategy == "normalized"
    output = Path(result.artifact.backing.uri)
    assert output.exists()

    with rasterio.open(str(output)) as src:
        assert src.count == 4
        assert str(src.crs) == "EPSG:4326"


def test_materialize_eager_band_descriptions(pixc_h5: Path, tmp_path: Path):
    import rasterio

    conn = PIXCConnector()
    result = conn.materialize(str(pixc_h5), tmp_path)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.descriptions[0] == "sig0"
        assert src.descriptions[1] == "height"
        assert src.descriptions[2] == "water_frac"
        assert src.descriptions[3] == "classification"


def test_materialize_eager_spatial_descriptor(pixc_h5: Path, tmp_path: Path):
    conn = PIXCConnector()
    result = conn.materialize(str(pixc_h5), tmp_path)

    spatial = result.artifact.spatial
    assert spatial.crs == "EPSG:4326"
    assert spatial.band_count == 4
    assert spatial.resolution is not None


def test_materialize_eager_has_content_hash(pixc_h5: Path, tmp_path: Path):
    conn = PIXCConnector()
    result = conn.materialize(str(pixc_h5), tmp_path)
    assert result.artifact.backing.content_hash is not None
    assert result.artifact.backing.size_bytes > 0


def test_materialize_eager_lineage(pixc_h5: Path, tmp_path: Path):
    conn = PIXCConnector()
    result = conn.materialize(str(pixc_h5), tmp_path)

    lineage = result.artifact.lineage
    assert lineage.operation == "pixc_rasterize"
    assert lineage.params["pixels"] == NUM_PIXELS
    assert lineage.params["resolution_m"] == 30.0


def test_materialize_eager_metadata_bands(pixc_h5: Path, tmp_path: Path):
    conn = PIXCConnector()
    result = conn.materialize(str(pixc_h5), tmp_path)

    assert list(result.artifact.metadata["bands"]) == [
        "sig0",
        "height",
        "water_frac",
        "classification",
    ]
    assert result.artifact.metadata["source"] == "pixc"
    assert result.artifact.metadata["format"] == "geotiff"


def test_materialize_eager_custom_resolution(pixc_h5: Path, tmp_path: Path):
    import rasterio

    conn = PIXCConnector()
    result_30 = conn.materialize(str(pixc_h5), tmp_path / "out30", resolution_m=30.0)
    result_100 = conn.materialize(str(pixc_h5), tmp_path / "out100", resolution_m=100.0)

    with (
        rasterio.open(result_30.artifact.backing.uri) as src30,
        rasterio.open(result_100.artifact.backing.uri) as src100,
    ):
        # Coarser resolution → fewer pixels
        assert src100.width <= src30.width
        assert src100.height <= src30.height


def test_materialize_eager_raster_has_data(pixc_h5: Path, tmp_path: Path):
    """Rasterized output should have some non-NaN values."""
    import rasterio

    conn = PIXCConnector()
    result = conn.materialize(str(pixc_h5), tmp_path)

    with rasterio.open(result.artifact.backing.uri) as src:
        sig0 = src.read(1)
        assert np.any(np.isfinite(sig0)), "Expected some valid sig0 values"


# ---------------------------------------------------------------------------
# target_grid parameter
# ---------------------------------------------------------------------------


def test_materialize_target_grid(pixc_h5: Path, tmp_path: Path):
    """target_grid overrides extent and resolution."""
    import rasterio

    conn = PIXCConnector()
    # Target grid: tight bbox at 0.001 degree resolution
    target = (-90.0, 30.0, -89.0, 31.0, 0.001)
    result = conn.materialize(str(pixc_h5), tmp_path, target_grid=target)

    with rasterio.open(result.artifact.backing.uri) as src:
        assert src.width == 1000
        assert src.height == 1000
        bounds = src.bounds
        assert bounds.left == pytest.approx(-90.0)
        assert bounds.bottom == pytest.approx(30.0)


def test_materialize_target_grid_alignment(pixc_h5: Path, tmp_path: Path):
    """Two rasters with same target_grid should have identical dimensions."""
    import rasterio

    conn = PIXCConnector()
    target = (-90.0, 30.0, -89.5, 30.5, 0.005)
    result = conn.materialize(str(pixc_h5), tmp_path, target_grid=target)

    with rasterio.open(result.artifact.backing.uri) as src:
        assert src.width == 100
        assert src.height == 100


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_returns_fields(pixc_h5: Path):
    conn = PIXCConnector()
    meta = conn.metadata(str(pixc_h5))

    assert meta["tile_name"] == "133L"
    assert meta["cycle"] == 16
    assert meta["pass"] == 89
    assert meta["swath_side"] == "L"
    assert meta["num_pixels"] == NUM_PIXELS
    assert meta["crs"] == "EPSG:4326"
    assert meta["lat_bounds"] == (30.0, 31.0)
    assert meta["lon_bounds"] == (-90.0, -89.0)


def test_metadata_has_time_coverage(pixc_h5: Path):
    conn = PIXCConnector()
    meta = conn.metadata(str(pixc_h5))
    assert meta["time_start"] == "2024-06-01T08:39:25Z"
    assert meta["time_end"] == "2024-06-01T08:39:36Z"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_materialize_missing_file(tmp_path: Path):
    conn = PIXCConnector()
    with pytest.raises(MaterializeError, match="File not found"):
        conn.materialize("/nonexistent/file.nc", tmp_path)


def test_materialize_wrong_extension(tmp_path: Path):
    conn = PIXCConnector()
    with pytest.raises(MaterializeError, match="Not a NetCDF/HDF5 file"):
        conn.materialize("/some/path/file.txt", tmp_path)


def test_materialize_no_pixel_cloud_group(pixc_h5_no_pixel_cloud: Path, tmp_path: Path):
    conn = PIXCConnector()
    with pytest.raises(MaterializeError, match="No pixel_cloud group"):
        conn.materialize(str(pixc_h5_no_pixel_cloud), tmp_path)


def test_metadata_missing_file():
    conn = PIXCConnector()
    with pytest.raises(MaterializeError, match="File not found"):
        conn.metadata("/nonexistent/file.nc")
