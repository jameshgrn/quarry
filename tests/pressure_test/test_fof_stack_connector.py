"""Pressure tests for FOFStackConnector (structural mapper, no processing).

Lane: connector

Tests fof-compiler stack.nc file reading with gridded bands at 30m EPSG:4326.
Key band: water_frequency (float32, 0-1).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from quarry_connectors.fof_stack import FOFStackConnector
from quarry_core.connector import Connector, ConnectorCapability, MaterializeError

# ---------------------------------------------------------------------------
# Fixtures — synthetic fof-compiler stack.nc files
# ---------------------------------------------------------------------------


@pytest.fixture
def fof_stack_nc(tmp_path: Path) -> Path:
    """Minimal fof-compiler stack.nc file with expected structure."""
    path = tmp_path / "stack.nc"
    with h5py.File(str(path), "w") as f:
        # Root attributes
        f.attrs["crs"] = "EPSG:4326"

        # 1D coordinate arrays (lon/lat range)
        # 60 x values (longitude), 50 y values (latitude)
        x_vals = np.linspace(-90.0, -89.0, 60, dtype=np.float64)
        y_vals = np.linspace(30.0, 31.0, 50, dtype=np.float64)
        f.create_dataset("x", data=x_vals)
        f.create_dataset("y", data=y_vals)

        # 2D data bands
        water_freq = np.random.rand(50, 60).astype(np.float32)
        elevation = np.random.rand(50, 60).astype(np.float32)
        f.create_dataset("water_frequency", data=water_freq)
        f.create_dataset("elevation", data=elevation)

    return path


@pytest.fixture
def fof_stack_nc_alt_coords(tmp_path: Path) -> Path:
    """FOF stack file using lon/lat instead of x/y coordinate names."""
    path = tmp_path / "stack_alt.nc"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"

        # Alternative coordinate names
        lon_vals = np.linspace(-95.0, -94.0, 40, dtype=np.float64)
        lat_vals = np.linspace(35.0, 36.0, 30, dtype=np.float64)
        f.create_dataset("lon", data=lon_vals)
        f.create_dataset("lat", data=lat_vals)

        # Data bands
        water_freq = np.random.rand(30, 40).astype(np.float32)
        f.create_dataset("water_frequency", data=water_freq)

    return path


@pytest.fixture
def fof_stack_nc_longitude_latitude(tmp_path: Path) -> Path:
    """FOF stack file using longitude/latitude full names."""
    path = tmp_path / "stack_full.nc"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"

        # Full coordinate names
        lon_vals = np.linspace(-100.0, -99.0, 20, dtype=np.float64)
        lat_vals = np.linspace(40.0, 41.0, 25, dtype=np.float64)
        f.create_dataset("longitude", data=lon_vals)
        f.create_dataset("latitude", data=lat_vals)

        # Data bands
        water_freq = np.random.rand(25, 20).astype(np.float32)
        f.create_dataset("water_frequency", data=water_freq)

    return path


@pytest.fixture
def fof_stack_nc_no_crs_attr(tmp_path: Path) -> Path:
    """FOF stack file without CRS attribute (should default to EPSG:4326)."""
    path = tmp_path / "stack_no_crs.nc"
    with h5py.File(str(path), "w") as f:
        # No CRS attribute
        x_vals = np.linspace(-90.0, -89.0, 60, dtype=np.float64)
        y_vals = np.linspace(30.0, 31.0, 50, dtype=np.float64)
        f.create_dataset("x", data=x_vals)
        f.create_dataset("y", data=y_vals)

        water_freq = np.random.rand(50, 60).astype(np.float32)
        f.create_dataset("water_frequency", data=water_freq)

    return path


@pytest.fixture
def fof_stack_nc_3d_band(tmp_path: Path) -> Path:
    """FOF stack file with 3D band (time, y, x)."""
    path = tmp_path / "stack_3d.nc"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"

        x_vals = np.linspace(-90.0, -89.0, 60, dtype=np.float64)
        y_vals = np.linspace(30.0, 31.0, 50, dtype=np.float64)
        f.create_dataset("x", data=x_vals)
        f.create_dataset("y", data=y_vals)

        # 3D band: (time=3, y=50, x=60)
        water_freq_3d = np.random.rand(3, 50, 60).astype(np.float32)
        f.create_dataset("water_frequency", data=water_freq_3d)

    return path


@pytest.fixture
def fof_stack_nc_many_bands(tmp_path: Path) -> Path:
    """FOF stack file with many bands for subset testing."""
    path = tmp_path / "stack_many.nc"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"

        x_vals = np.linspace(-90.0, -89.0, 60, dtype=np.float64)
        y_vals = np.linspace(30.0, 31.0, 50, dtype=np.float64)
        f.create_dataset("x", data=x_vals)
        f.create_dataset("y", data=y_vals)

        # Multiple data bands
        for band_name in ["water_frequency", "elevation", "slope", "aspect", "hand"]:
            data = np.random.rand(50, 60).astype(np.float32)
            f.create_dataset(band_name, data=data)

    return path


@pytest.fixture
def fof_stack_nc_no_coords(tmp_path: Path) -> Path:
    """FOF stack file without coordinate arrays."""
    path = tmp_path / "stack_no_coords.nc"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"

        # No x/y or lon/lat arrays
        water_freq = np.random.rand(50, 60).astype(np.float32)
        f.create_dataset("water_frequency", data=water_freq)

    return path


@pytest.fixture
def fof_stack_nc_with_time_coord(tmp_path: Path) -> Path:
    """FOF stack file with time coordinate (should be excluded from data bands)."""
    path = tmp_path / "stack_time.nc"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"

        x_vals = np.linspace(-90.0, -89.0, 60, dtype=np.float64)
        y_vals = np.linspace(30.0, 31.0, 50, dtype=np.float64)
        time_vals = np.array([0, 1, 2], dtype=np.int32)
        f.create_dataset("x", data=x_vals)
        f.create_dataset("y", data=y_vals)
        f.create_dataset("time", data=time_vals)

        # 3D data band
        water_freq = np.random.rand(3, 50, 60).astype(np.float32)
        f.create_dataset("water_frequency", data=water_freq)

    return path


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_name():
    """Connector name is 'fof_stack'."""
    assert FOFStackConnector().name == "fof_stack"


def test_capabilities():
    """Connector declares all expected capabilities."""
    caps = FOFStackConnector().capabilities
    assert ConnectorCapability.MATERIALIZE in caps
    assert ConnectorCapability.DISCOVER in caps
    assert ConnectorCapability.METADATA_ONLY in caps
    assert ConnectorCapability.MATERIALIZE_LAZY in caps


def test_satisfies_connector_protocol():
    """Connector satisfies the Connector protocol."""
    assert isinstance(FOFStackConnector(), Connector)


def test_capabilities_is_flag_combination():
    """Capabilities can be combined as flags."""
    caps = FOFStackConnector().capabilities
    assert caps & ConnectorCapability.MATERIALIZE
    assert caps & ConnectorCapability.DISCOVER
    assert caps & ConnectorCapability.METADATA_ONLY
    assert caps & ConnectorCapability.MATERIALIZE_LAZY


# ---------------------------------------------------------------------------
# Discover — file-level
# ---------------------------------------------------------------------------


def test_discover_file_lists_bands(fof_stack_nc: Path):
    """Discover returns catalog entries for stack.nc files."""
    conn = FOFStackConnector()
    entries = conn.discover(str(fof_stack_nc.parent))

    assert len(entries) == 1
    assert entries[0].name == "stack"


def test_discover_file_has_spatial_hint(fof_stack_nc: Path):
    """Catalog entries include spatial hints (CRS and extent)."""
    conn = FOFStackConnector()
    entries = conn.discover(str(fof_stack_nc.parent))

    entry = entries[0]
    assert entry.spatial_hint["crs"] == "EPSG:4326"
    extent = entry.spatial_hint["extent"]
    assert extent is not None
    assert len(extent) == 4


def test_discover_file_has_band_metadata(fof_stack_nc: Path):
    """Catalog entries include band list in metadata."""
    conn = FOFStackConnector()
    entries = conn.discover(str(fof_stack_nc.parent))

    entry = entries[0]
    bands = entry.metadata["bands"]
    assert "water_frequency" in bands
    assert "elevation" in bands


def test_discover_file_has_size_bytes(fof_stack_nc: Path):
    """Catalog entries include file size."""
    conn = FOFStackConnector()
    entries = conn.discover(str(fof_stack_nc.parent))

    entry = entries[0]
    assert entry.metadata["size_bytes"] > 0


def test_discover_file_source_ref_is_path(fof_stack_nc: Path):
    """Catalog entry source_ref is the full file path."""
    conn = FOFStackConnector()
    entries = conn.discover(str(fof_stack_nc.parent))

    entry = entries[0]
    assert entry.source_ref == str(fof_stack_nc)


# ---------------------------------------------------------------------------
# Discover — directory-level
# ---------------------------------------------------------------------------


def test_discover_directory_lists_multiple_files(tmp_path: Path):
    """Discover lists all .nc files in a directory."""
    # Create multiple stack files
    for i in range(3):
        path = tmp_path / f"stack_{i}.nc"
        with h5py.File(str(path), "w") as f:
            f.attrs["crs"] = "EPSG:4326"
            f.create_dataset("x", data=np.linspace(-90, -89, 10))
            f.create_dataset("y", data=np.linspace(30, 31, 10))
            f.create_dataset("water_frequency", data=np.random.rand(10, 10).astype(np.float32))

    conn = FOFStackConnector()
    entries = conn.discover(str(tmp_path))

    assert len(entries) == 3
    names = {e.name for e in entries}
    assert names == {"stack_0", "stack_1", "stack_2"}


def test_discover_empty_directory(tmp_path: Path):
    """Discover returns empty list for empty directory."""
    conn = FOFStackConnector()
    entries = conn.discover(str(tmp_path))
    assert entries == []


def test_discover_skips_non_nc_files(tmp_path: Path):
    """Discover skips non-.nc files."""
    (tmp_path / "test.txt").touch()
    (tmp_path / "test.tif").touch()
    (tmp_path / "test.h5").touch()

    conn = FOFStackConnector()
    entries = conn.discover(str(tmp_path))
    assert entries == []


def test_discover_skips_invalid_nc_files(tmp_path: Path):
    """Discover skips .nc files that cannot be parsed."""
    # Create an invalid .nc file (just text)
    (tmp_path / "invalid.nc").write_text("not a valid hdf5 file")

    conn = FOFStackConnector()
    entries = conn.discover(str(tmp_path))
    assert entries == []


def test_discover_recursive(tmp_path: Path):
    """Discover with recursive=True finds files in subdirectories."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    # Create file in subdirectory
    path = subdir / "nested.nc"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"
        f.create_dataset("x", data=np.linspace(-90, -89, 10))
        f.create_dataset("y", data=np.linspace(30, 31, 10))
        f.create_dataset("water_frequency", data=np.random.rand(10, 10).astype(np.float32))

    conn = FOFStackConnector()
    entries = conn.discover({"path": str(tmp_path), "recursive": True})

    assert len(entries) == 1
    assert entries[0].name == "nested"


def test_discover_non_recursive_skips_subdirs(tmp_path: Path):
    """Discover without recursive=False skips subdirectories."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    path = subdir / "nested.nc"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"
        f.create_dataset("x", data=np.linspace(-90, -89, 10))
        f.create_dataset("y", data=np.linspace(30, 31, 10))
        f.create_dataset("water_frequency", data=np.random.rand(10, 10).astype(np.float32))

    conn = FOFStackConnector()
    entries = conn.discover(str(tmp_path))  # Default is non-recursive

    assert entries == []


def test_discover_with_none_query_defaults_to_cwd(tmp_path: Path, monkeypatch):
    """Discover with None query defaults to current directory."""
    # Change to tmp_path
    monkeypatch.chdir(tmp_path)

    # Create a file in current directory
    path = tmp_path / "current.nc"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"
        f.create_dataset("x", data=np.linspace(-90, -89, 10))
        f.create_dataset("y", data=np.linspace(30, 31, 10))
        f.create_dataset("water_frequency", data=np.random.rand(10, 10).astype(np.float32))

    conn = FOFStackConnector()
    entries = conn.discover(None)

    assert len(entries) == 1


# ---------------------------------------------------------------------------
# Materialize — lazy
# ---------------------------------------------------------------------------


def test_materialize_lazy_returns_lazy_handle(fof_stack_nc: Path, tmp_path: Path):
    """Lazy materialize returns lazy_handle strategy."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path, lazy=True)

    assert result.strategy == "lazy_handle"


def test_materialize_lazy_artifact_type(fof_stack_nc: Path, tmp_path: Path):
    """Lazy materialize returns RASTER artifact type."""
    from quarry_core.artifact import ArtifactType

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path, lazy=True)

    assert result.artifact.type == ArtifactType.RASTER


def test_materialize_lazy_artifact_name(fof_stack_nc: Path, tmp_path: Path):
    """Lazy materialize artifact name is file stem."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path, lazy=True)

    assert result.artifact.name == "stack"


def test_materialize_lazy_backing_is_lazy_handle(fof_stack_nc: Path, tmp_path: Path):
    """Lazy materialize backing store is LAZY_HANDLE kind."""
    from quarry_core.artifact import BackingStoreKind

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path, lazy=True)

    assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE
    assert result.artifact.backing.uri == str(fof_stack_nc)


def test_materialize_lazy_spatial_descriptor(fof_stack_nc: Path, tmp_path: Path):
    """Lazy materialize includes correct spatial descriptor."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path, lazy=True)

    spatial = result.artifact.spatial
    assert spatial.crs == "EPSG:4326"
    assert spatial.extent is not None
    assert spatial.band_count == 2  # water_frequency and elevation


def test_materialize_lazy_metadata(fof_stack_nc: Path, tmp_path: Path):
    """Lazy materialize includes source and bands in metadata."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path, lazy=True)

    assert result.artifact.metadata["source"] == "fof_stack"
    assert "water_frequency" in result.artifact.metadata["bands"]
    assert "elevation" in result.artifact.metadata["bands"]


def test_materialize_lazy_lineage(fof_stack_nc: Path, tmp_path: Path):
    """Lazy materialize includes correct lineage."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path, lazy=True)

    assert result.artifact.lineage.operation == "materialize"
    assert result.artifact.lineage.params["lazy"] is True


def test_materialize_lazy_with_alt_coords(fof_stack_nc_alt_coords: Path, tmp_path: Path):
    """Lazy materialize works with lon/lat coordinate names."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc_alt_coords), tmp_path, lazy=True)

    assert result.strategy == "lazy_handle"
    spatial = result.artifact.spatial
    assert spatial.crs == "EPSG:4326"
    assert spatial.extent is not None


def test_materialize_lazy_with_longitude_latitude(
    fof_stack_nc_longitude_latitude: Path, tmp_path: Path
):
    """Lazy materialize works with longitude/latitude coordinate names."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc_longitude_latitude), tmp_path, lazy=True)

    assert result.strategy == "lazy_handle"
    spatial = result.artifact.spatial
    assert spatial.crs == "EPSG:4326"


def test_materialize_lazy_no_crs_attr_defaults_to_epsg4326(
    fof_stack_nc_no_crs_attr: Path, tmp_path: Path
):
    """Lazy materialize defaults to EPSG:4326 when no CRS attribute."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc_no_crs_attr), tmp_path, lazy=True)

    assert result.artifact.spatial.crs == "EPSG:4326"


# ---------------------------------------------------------------------------
# Materialize — eager
# ---------------------------------------------------------------------------


def test_materialize_eager_writes_geotiff(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize writes a GeoTIFF file."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path, lazy=False)

    assert result.strategy == "normalized"
    output = Path(result.artifact.backing.uri)
    assert output.exists()
    assert output.suffix == ".tif"

    with rasterio.open(str(output)) as src:
        assert src.count == 2  # water_frequency and elevation


def test_materialize_eager_band_count(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize produces correct band count."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.count == 2


def test_materialize_eager_crs(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize produces correct CRS."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.crs.to_string() == "EPSG:4326"


def test_materialize_eager_extent(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize produces correct extent."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        bounds = src.bounds
        assert bounds.left == pytest.approx(-90.0)
        assert bounds.right == pytest.approx(-89.0)
        assert bounds.bottom == pytest.approx(30.0)
        assert bounds.top == pytest.approx(31.0)


def test_materialize_eager_band_descriptions(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize includes band descriptions."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        descriptions = src.descriptions
        assert "water_frequency" in descriptions
        assert "elevation" in descriptions


def test_materialize_eager_dtype(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize produces float32 output."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.dtypes[0] == "float32"


def test_materialize_eager_compression(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize uses deflate compression."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.compression.value.lower() == "deflate"


def test_materialize_eager_nodata(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize sets NaN as nodata."""
    import numpy as np
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert np.isnan(src.nodata)


def test_materialize_eager_artifact_backing(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize artifact has LOCAL_FILE backing."""
    from quarry_core.artifact import BackingStoreKind

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
    assert result.artifact.backing.size_bytes > 0
    assert result.artifact.backing.content_hash is not None


def test_materialize_eager_artifact_spatial(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize artifact has correct spatial descriptor."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    spatial = result.artifact.spatial
    assert spatial.crs == "EPSG:4326"
    assert spatial.band_count == 2
    assert spatial.resolution is not None


def test_materialize_eager_artifact_metadata(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize artifact has correct metadata."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    assert result.artifact.metadata["source"] == "fof_stack"
    bands = result.artifact.metadata["bands"]
    assert "water_frequency" in bands
    assert "elevation" in bands


def test_materialize_eager_artifact_lineage(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize artifact has correct lineage."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    lineage = result.artifact.lineage
    assert lineage.operation == "fof_stack_materialize"
    assert lineage.params["source"] == str(fof_stack_nc)


def test_materialize_eager_output_in_workspace_subdirectory(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize creates output in workspace subdirectory named after file stem."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    output = Path(result.artifact.backing.uri)
    assert output.parent.name == "stack"
    assert output.name == "stack.tif"


def test_materialize_eager_3d_band_first_slice(fof_stack_nc_3d_band: Path, tmp_path: Path):
    """Eager materialize takes first slice of 3D bands."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc_3d_band), tmp_path)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        # 3D band (3, 50, 60) -> first slice (50, 60)
        assert src.count == 1
        assert src.height == 50
        assert src.width == 60


def test_materialize_eager_no_coords_uses_pixel_extent(
    fof_stack_nc_no_coords: Path, tmp_path: Path
):
    """Eager materialize uses pixel extent when no coordinate arrays."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc_no_coords), tmp_path)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        # When no coords, extent is (0, 0, width, height)
        assert src.bounds.left == 0
        assert src.bounds.bottom == 0
        assert src.bounds.right == 60
        assert src.bounds.top == 50


# ---------------------------------------------------------------------------
# Materialize — eager with bands parameter
# ---------------------------------------------------------------------------


def test_materialize_eager_with_bands_subset(fof_stack_nc_many_bands: Path, tmp_path: Path):
    """Eager materialize with bands parameter selects subset."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(
        str(fof_stack_nc_many_bands), tmp_path, bands=["water_frequency", "elevation"]
    )

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.count == 2
        assert "water_frequency" in src.descriptions
        assert "elevation" in src.descriptions


def test_materialize_eager_with_single_band(fof_stack_nc_many_bands: Path, tmp_path: Path):
    """Eager materialize with single band parameter."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc_many_bands), tmp_path, bands=["water_frequency"])

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.count == 1
        assert src.descriptions[0] == "water_frequency"


def test_materialize_eager_with_bands_preserves_order(
    fof_stack_nc_many_bands: Path, tmp_path: Path
):
    """Eager materialize preserves band order from bands parameter."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(
        str(fof_stack_nc_many_bands), tmp_path, bands=["elevation", "water_frequency"]
    )

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.descriptions[0] == "elevation"
        assert src.descriptions[1] == "water_frequency"


def test_materialize_eager_with_bands_updates_artifact_metadata(
    fof_stack_nc_many_bands: Path, tmp_path: Path
):
    """Eager materialize with bands updates artifact metadata band list."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc_many_bands), tmp_path, bands=["water_frequency"])

    assert list(result.artifact.metadata["bands"]) == ["water_frequency"]
    assert result.artifact.spatial.band_count == 1


def test_materialize_eager_with_bands_lineage_includes_bands(
    fof_stack_nc_many_bands: Path, tmp_path: Path
):
    """Eager materialize with bands includes bands in lineage."""
    conn = FOFStackConnector()
    result = conn.materialize(
        str(fof_stack_nc_many_bands), tmp_path, bands=["water_frequency", "slope"]
    )

    assert list(result.artifact.lineage.params["bands"]) == ["water_frequency", "slope"]


def test_materialize_eager_with_invalid_band_raises_error(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize with invalid band name raises MaterializeError."""
    conn = FOFStackConnector()
    with pytest.raises(MaterializeError, match="No valid bands"):
        conn.materialize(str(fof_stack_nc), tmp_path, bands=["nonexistent_band"])


def test_materialize_eager_with_partial_invalid_band(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize filters out invalid bands, keeps valid ones."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path, bands=["water_frequency", "nonexistent"])

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.count == 1
        assert src.descriptions[0] == "water_frequency"


def test_materialize_eager_with_empty_bands_list_uses_all(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize with empty bands list uses all available bands."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path, bands=[])

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.count == 2  # All bands


def test_materialize_eager_with_none_bands_uses_all(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize with None bands uses all available bands."""
    import rasterio

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path, bands=None)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.count == 2  # All bands


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_returns_bands_list(fof_stack_nc: Path):
    """Metadata returns list of data bands."""
    conn = FOFStackConnector()
    meta = conn.metadata(str(fof_stack_nc))

    assert "water_frequency" in meta["bands"]
    assert "elevation" in meta["bands"]


def test_metadata_returns_crs(fof_stack_nc: Path):
    """Metadata returns CRS."""
    conn = FOFStackConnector()
    meta = conn.metadata(str(fof_stack_nc))

    assert meta["crs"] == "EPSG:4326"


def test_metadata_returns_extent(fof_stack_nc: Path):
    """Metadata returns spatial extent."""
    conn = FOFStackConnector()
    meta = conn.metadata(str(fof_stack_nc))

    extent = meta["extent"]
    assert extent[0] == pytest.approx(-90.0)  # xmin
    assert extent[1] == pytest.approx(30.0)  # ymin
    assert extent[2] == pytest.approx(-89.0)  # xmax
    assert extent[3] == pytest.approx(31.0)  # ymax


def test_metadata_returns_resolution(fof_stack_nc: Path):
    """Metadata returns resolution."""
    conn = FOFStackConnector()
    meta = conn.metadata(str(fof_stack_nc))

    resolution = meta["resolution"]
    assert resolution is not None
    assert len(resolution) == 2
    assert resolution[0] > 0  # x resolution
    assert resolution[1] > 0  # y resolution


def test_metadata_returns_shape(fof_stack_nc: Path):
    """Metadata returns data shape."""
    conn = FOFStackConnector()
    meta = conn.metadata(str(fof_stack_nc))

    assert meta["shape"] == [50, 60]  # (y, x) or (height, width)


def test_metadata_excludes_coordinate_arrays(fof_stack_nc: Path):
    """Metadata excludes x, y coordinate arrays from bands list."""
    conn = FOFStackConnector()
    meta = conn.metadata(str(fof_stack_nc))

    assert "x" not in meta["bands"]
    assert "y" not in meta["bands"]


def test_metadata_excludes_time_coordinate(fof_stack_nc_with_time_coord: Path):
    """Metadata excludes time coordinate array from bands list."""
    conn = FOFStackConnector()
    meta = conn.metadata(str(fof_stack_nc_with_time_coord))

    assert "time" not in meta["bands"]
    assert "water_frequency" in meta["bands"]


def test_metadata_with_alt_coords(fof_stack_nc_alt_coords: Path):
    """Metadata works with lon/lat coordinate names."""
    conn = FOFStackConnector()
    meta = conn.metadata(str(fof_stack_nc_alt_coords))

    assert meta["crs"] == "EPSG:4326"
    assert meta["extent"] is not None
    assert "lon" not in meta["bands"]
    assert "lat" not in meta["bands"]


def test_metadata_no_crs_attr_defaults_to_epsg4326(fof_stack_nc_no_crs_attr: Path):
    """Metadata defaults to EPSG:4326 when no CRS attribute."""
    conn = FOFStackConnector()
    meta = conn.metadata(str(fof_stack_nc_no_crs_attr))

    assert meta["crs"] == "EPSG:4326"


def test_metadata_no_coords_has_none_extent(fof_stack_nc_no_coords: Path):
    """Metadata has None extent when no coordinate arrays."""
    conn = FOFStackConnector()
    meta = conn.metadata(str(fof_stack_nc_no_coords))

    assert meta["extent"] is None
    assert meta["resolution"] is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_materialize_missing_file(tmp_path: Path):
    """Materialize raises MaterializeError for missing file."""
    conn = FOFStackConnector()
    with pytest.raises(MaterializeError, match="File not found"):
        conn.materialize("/nonexistent/file.nc", tmp_path)


def test_materialize_wrong_extension(tmp_path: Path):
    """Materialize raises MaterializeError for wrong file extension."""
    conn = FOFStackConnector()
    with pytest.raises(MaterializeError, match="Not a NetCDF/HDF5 file"):
        conn.materialize("/some/path/file.txt", tmp_path)


def test_materialize_wrong_extension_variations(tmp_path: Path):
    """Materialize rejects various non-NetCDF/HDF5 extensions."""
    conn = FOFStackConnector()

    for ext in [".tif", ".geojson", ".shp", ".csv", ".parquet", ".zarr"]:
        with pytest.raises(MaterializeError, match="Not a NetCDF/HDF5 file"):
            conn.materialize(f"/some/path/file{ext}", tmp_path)


def test_materialize_accepts_nc4_extension(tmp_path: Path):
    """Materialize accepts .nc4 extension."""
    # Create a valid .nc4 file
    path = tmp_path / "test.nc4"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"
        f.create_dataset("x", data=np.linspace(-90, -89, 10))
        f.create_dataset("y", data=np.linspace(30, 31, 10))
        f.create_dataset("water_frequency", data=np.random.rand(10, 10).astype(np.float32))

    conn = FOFStackConnector()
    result = conn.materialize(str(path), tmp_path, lazy=True)
    assert result.strategy == "lazy_handle"


def test_materialize_accepts_h5_extension(tmp_path: Path):
    """Materialize accepts .h5 extension."""
    path = tmp_path / "test.h5"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"
        f.create_dataset("x", data=np.linspace(-90, -89, 10))
        f.create_dataset("y", data=np.linspace(30, 31, 10))
        f.create_dataset("water_frequency", data=np.random.rand(10, 10).astype(np.float32))

    conn = FOFStackConnector()
    result = conn.materialize(str(path), tmp_path, lazy=True)
    assert result.strategy == "lazy_handle"


def test_materialize_accepts_hdf5_extension(tmp_path: Path):
    """Materialize accepts .hdf5 extension."""
    path = tmp_path / "test.hdf5"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"
        f.create_dataset("x", data=np.linspace(-90, -89, 10))
        f.create_dataset("y", data=np.linspace(30, 31, 10))
        f.create_dataset("water_frequency", data=np.random.rand(10, 10).astype(np.float32))

    conn = FOFStackConnector()
    result = conn.materialize(str(path), tmp_path, lazy=True)
    assert result.strategy == "lazy_handle"


def test_materialize_case_insensitive_extension(tmp_path: Path):
    """Materialize accepts case-insensitive extensions."""
    path = tmp_path / "test.NC"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"
        f.create_dataset("x", data=np.linspace(-90, -89, 10))
        f.create_dataset("y", data=np.linspace(30, 31, 10))
        f.create_dataset("water_frequency", data=np.random.rand(10, 10).astype(np.float32))

    conn = FOFStackConnector()
    result = conn.materialize(str(path), tmp_path, lazy=True)
    assert result.strategy == "lazy_handle"


def test_metadata_missing_file():
    """Metadata raises MaterializeError for missing file."""
    conn = FOFStackConnector()
    with pytest.raises(MaterializeError, match="File not found"):
        conn.metadata("/nonexistent/file.nc")


def test_materialize_eager_no_valid_bands_raises_error(tmp_path: Path):
    """Eager materialize raises error when no valid bands to extract."""
    # Create file with only coordinate arrays (no data bands)
    path = tmp_path / "no_bands.nc"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"
        f.create_dataset("x", data=np.linspace(-90, -89, 10))
        f.create_dataset("y", data=np.linspace(30, 31, 10))

    conn = FOFStackConnector()
    with pytest.raises(MaterializeError, match="No valid bands"):
        conn.materialize(str(path), tmp_path, bands=["nonexistent"])


def test_materialize_corrupt_file(tmp_path: Path):
    """Materialize handles corrupt HDF5 files gracefully."""
    path = tmp_path / "corrupt.nc"
    path.write_text("not a valid hdf5 file content")

    conn = FOFStackConnector()
    with pytest.raises(Exception):  # h5py will raise an error
        conn.materialize(str(path), tmp_path)


def test_metadata_corrupt_file(tmp_path: Path):
    """Metadata handles corrupt HDF5 files gracefully."""
    path = tmp_path / "corrupt.nc"
    path.write_text("not a valid hdf5 file content")

    conn = FOFStackConnector()
    with pytest.raises(Exception):  # h5py will raise an error
        conn.metadata(str(path))


# ---------------------------------------------------------------------------
# Edge cases and additional tests
# ---------------------------------------------------------------------------


def test_materialize_lazy_vs_eager_same_metadata(fof_stack_nc: Path, tmp_path: Path):
    """Lazy and eager materialize produce equivalent metadata."""
    conn = FOFStackConnector()

    lazy_result = conn.materialize(str(fof_stack_nc), tmp_path, lazy=True)
    eager_result = conn.materialize(str(fof_stack_nc), tmp_path, lazy=False)

    # Spatial descriptors should match
    assert lazy_result.artifact.spatial.crs == eager_result.artifact.spatial.crs
    assert lazy_result.artifact.spatial.band_count == eager_result.artifact.spatial.band_count


def test_materialize_preserves_source_ref(fof_stack_nc: Path, tmp_path: Path):
    """Materialize result preserves original source_ref."""
    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), tmp_path)

    assert result.source_ref == str(fof_stack_nc)


def test_discover_preserves_file_order(tmp_path: Path):
    """Discover returns files in consistent order."""
    # Create files in specific order
    for name in ["c_stack.nc", "a_stack.nc", "b_stack.nc"]:
        path = tmp_path / name
        with h5py.File(str(path), "w") as f:
            f.attrs["crs"] = "EPSG:4326"
            f.create_dataset("x", data=np.linspace(-90, -89, 10))
            f.create_dataset("y", data=np.linspace(30, 31, 10))
            f.create_dataset("water_frequency", data=np.random.rand(10, 10).astype(np.float32))

    conn = FOFStackConnector()
    entries = conn.discover(str(tmp_path))

    names = sorted(e.name for e in entries)
    assert names == ["a_stack", "b_stack", "c_stack"]


def test_materialize_with_path_object(fof_stack_nc: Path, tmp_path: Path):
    """Materialize accepts Path objects as source_ref."""
    conn = FOFStackConnector()
    result = conn.materialize(fof_stack_nc, tmp_path, lazy=True)

    assert result.strategy == "lazy_handle"


def test_metadata_with_path_object(fof_stack_nc: Path):
    """Metadata accepts Path objects as source_ref."""
    conn = FOFStackConnector()
    meta = conn.metadata(fof_stack_nc)

    assert meta["crs"] == "EPSG:4326"


def test_materialize_workspace_created_if_missing(fof_stack_nc: Path, tmp_path: Path):
    """Materialize creates workspace directory if it doesn't exist."""
    workspace = tmp_path / "nested" / "workspace"
    assert not workspace.exists()

    conn = FOFStackConnector()
    result = conn.materialize(str(fof_stack_nc), workspace)

    assert workspace.exists()
    assert Path(result.artifact.backing.uri).exists()


def test_materialize_eager_data_integrity(fof_stack_nc: Path, tmp_path: Path):
    """Eager materialize preserves data values accurately."""
    import rasterio

    # Create file with known values
    path = tmp_path / "known.nc"
    known_data = np.arange(50 * 60, dtype=np.float32).reshape(50, 60)
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"
        f.create_dataset("x", data=np.linspace(-90, -89, 60))
        f.create_dataset("y", data=np.linspace(30, 31, 50))
        f.create_dataset("water_frequency", data=known_data)

    conn = FOFStackConnector()
    result = conn.materialize(str(path), tmp_path)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        read_data = src.read(1)
        np.testing.assert_array_equal(read_data, known_data)
