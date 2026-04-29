"""Pressure tests for HDF5Connector."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from quarry_connectors.hdf5 import HDF5Connector

# ---------------------------------------------------------------------------
# Fixtures — synthetic HDF5 files
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_h5(tmp_path: Path) -> Path:
    """HDF5 with a single 2D dataset + 1D coordinate arrays + CRS attr."""
    path = tmp_path / "simple.h5"
    with h5py.File(str(path), "w") as f:
        f.attrs["crs"] = "EPSG:4326"
        f.create_dataset("latitude", data=np.linspace(30, 31, 100))
        f.create_dataset("longitude", data=np.linspace(-90, -89, 200))
        f.create_dataset("elevation", data=np.random.rand(100, 200).astype(np.float32))
        f["elevation"].attrs["units"] = "meters"
    return path


@pytest.fixture
def nested_h5(tmp_path: Path) -> Path:
    """HDF5 with nested group structure."""
    path = tmp_path / "nested.h5"
    with h5py.File(str(path), "w") as f:
        f.attrs["mission"] = "test"
        grp = f.create_group("science_data")
        grp.create_dataset("water_height", data=np.random.rand(50, 80).astype(np.float32))
        grp.create_dataset("water_area", data=np.random.rand(50, 80).astype(np.float32))
        coord = f.create_group("coordinates")
        coord.create_dataset("lat", data=np.linspace(40, 41, 50))
        coord.create_dataset("lon", data=np.linspace(-100, -99, 80))
    return path


@pytest.fixture
def complex_h5(tmp_path: Path) -> Path:
    """HDF5 with complex pair array (H, W, 2) [real, imag]."""
    path = tmp_path / "complex.h5"
    with h5py.File(str(path), "w") as f:
        real = np.random.rand(30, 40).astype(np.float32)
        imag = np.random.rand(30, 40).astype(np.float32)
        data = np.stack([real, imag], axis=-1)
        f.create_dataset("signal", data=data)
    return path


@pytest.fixture
def no_coords_h5(tmp_path: Path) -> Path:
    """HDF5 with data but no coordinate arrays or CRS."""
    path = tmp_path / "no_coords.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("image", data=np.random.rand(64, 128).astype(np.float32))
    return path


@pytest.fixture
def multiband_h5(tmp_path: Path) -> Path:
    """HDF5 with multiple same-shape datasets in one group."""
    path = tmp_path / "multiband.h5"
    with h5py.File(str(path), "w") as f:
        grp = f.create_group("bands")
        for name in ["red", "green", "blue", "nir"]:
            grp.create_dataset(name, data=np.random.rand(100, 100).astype(np.float32))
        f.create_dataset("x", data=np.linspace(0, 1, 100))
        f.create_dataset("y", data=np.linspace(0, 1, 100))
    return path


@pytest.fixture
def cf_coords_h5(tmp_path: Path) -> Path:
    """HDF5 with CF-convention coordinates attribute."""
    path = tmp_path / "cf.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("lat", data=np.linspace(10, 11, 60))
        f["lat"].attrs["units"] = "degrees_north"
        f.create_dataset("lon", data=np.linspace(20, 21, 90))
        f["lon"].attrs["units"] = "degrees_east"
        ds = f.create_dataset("temperature", data=np.random.rand(60, 90).astype(np.float32))
        ds.attrs["coordinates"] = "lat lon"
    return path


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_name():
    assert HDF5Connector().name == "hdf5"


def test_capabilities():
    from quarry_core.connector import ConnectorCapability

    caps = HDF5Connector().capabilities
    assert ConnectorCapability.MATERIALIZE in caps
    assert ConnectorCapability.DISCOVER in caps
    assert ConnectorCapability.METADATA_ONLY in caps
    assert ConnectorCapability.MATERIALIZE_LAZY in caps


def test_satisfies_connector_protocol():
    from quarry_core.connector import Connector

    assert isinstance(HDF5Connector(), Connector)


# ---------------------------------------------------------------------------
# Source ref parsing
# ---------------------------------------------------------------------------


def test_parse_plain_path():
    conn = HDF5Connector()
    path, ds = conn._parse_source_ref("file.h5")
    assert path == "file.h5"
    assert ds is None


def test_parse_with_dataset():
    conn = HDF5Connector()
    path, ds = conn._parse_source_ref("file.h5::/group/dataset")
    assert path == "file.h5"
    assert ds == "/group/dataset"


def test_parse_with_dataset_no_leading_slash():
    conn = HDF5Connector()
    path, ds = conn._parse_source_ref("file.h5::group/dataset")
    assert path == "file.h5"
    assert ds == "group/dataset"


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def test_discover_simple_file(simple_h5: Path):
    conn = HDF5Connector()
    with h5py.File(str(simple_h5), "r") as f:
        datasets = conn._discover_datasets(f)

    paths = {d.path for d in datasets}
    assert "/elevation" in paths
    assert "/latitude" in paths
    assert "/longitude" in paths

    # latitude and longitude should be classified as coordinates
    coords = {d.path for d in datasets if d.is_coordinate}
    assert "/latitude" in coords
    assert "/longitude" in coords
    assert "/elevation" not in coords


def test_discover_nested_file(nested_h5: Path):
    conn = HDF5Connector()
    with h5py.File(str(nested_h5), "r") as f:
        datasets = conn._discover_datasets(f)

    paths = {d.path for d in datasets}
    assert "/science_data/water_height" in paths
    assert "/science_data/water_area" in paths
    assert "/coordinates/lat" in paths
    assert "/coordinates/lon" in paths


def test_discover_complex_pair(complex_h5: Path):
    conn = HDF5Connector()
    with h5py.File(str(complex_h5), "r") as f:
        datasets = conn._discover_datasets(f)

    signal = [d for d in datasets if d.path == "/signal"][0]
    assert signal.is_complex_pair is True
    assert signal.shape == (30, 40, 2)


def test_discover_directory(tmp_path: Path):
    # Create a couple of HDF5 files
    for name in ["a.h5", "b.hdf5"]:
        with h5py.File(str(tmp_path / name), "w") as f:
            f.create_dataset("data", data=np.zeros((10, 10)))

    # Also create a non-HDF5 file
    (tmp_path / "c.txt").touch()

    conn = HDF5Connector()
    entries = conn.discover(str(tmp_path))

    names = {e.name for e in entries}
    assert "a" in names
    assert "b" in names
    assert len(entries) == 2


def test_discover_empty_directory(tmp_path: Path):
    conn = HDF5Connector()
    assert conn.discover(str(tmp_path)) == []


def test_discover_file_lists_datasets(simple_h5: Path):
    conn = HDF5Connector()
    entries = conn.discover(str(simple_h5))

    # Should list non-coordinate datasets only
    names = {e.name for e in entries}
    assert "/elevation" in names
    assert "/latitude" not in names
    assert "/longitude" not in names


def test_discover_multiband(multiband_h5: Path):
    conn = HDF5Connector()
    entries = conn.discover(str(multiband_h5))

    names = {e.name for e in entries}
    assert "/bands/red" in names
    assert "/bands/green" in names
    assert "/bands/blue" in names
    assert "/bands/nir" in names
    # x and y should be filtered as coordinates
    assert "/x" not in names
    assert "/y" not in names


# ---------------------------------------------------------------------------
# Auto-select
# ---------------------------------------------------------------------------


def test_auto_select_largest_2d(nested_h5: Path):
    conn = HDF5Connector()
    with h5py.File(str(nested_h5), "r") as f:
        datasets = conn._discover_datasets(f)
        selected = conn._auto_select_dataset(datasets)

    assert selected is not None
    assert selected.ndim >= 2
    assert not selected.is_coordinate


def test_auto_select_skips_coordinates(simple_h5: Path):
    conn = HDF5Connector()
    with h5py.File(str(simple_h5), "r") as f:
        datasets = conn._discover_datasets(f)
        selected = conn._auto_select_dataset(datasets)

    assert selected is not None
    assert selected.path == "/elevation"


def test_auto_select_no_2d_datasets(tmp_path: Path):
    """File with only 1D data should return None."""
    path = tmp_path / "only_1d.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("timeseries", data=np.arange(100))

    conn = HDF5Connector()
    with h5py.File(str(path), "r") as f:
        datasets = conn._discover_datasets(f)
        selected = conn._auto_select_dataset(datasets)

    assert selected is None


# ---------------------------------------------------------------------------
# Coordinate inference
# ---------------------------------------------------------------------------


def test_infer_coords_simple(simple_h5: Path):
    conn = HDF5Connector()
    with h5py.File(str(simple_h5), "r") as f:
        datasets = conn._discover_datasets(f)
        info = [d for d in datasets if d.path == "/elevation"][0]
        spatial, coord_map = conn._infer_coordinates(f, info)

    assert spatial.crs == "EPSG:4326"
    assert spatial.extent is not None
    xmin, ymin, xmax, ymax = spatial.extent
    assert xmin == pytest.approx(-90.0)
    assert xmax == pytest.approx(-89.0)
    assert ymin == pytest.approx(30.0)
    assert ymax == pytest.approx(31.0)
    assert spatial.resolution is not None
    assert "x" in coord_map
    assert "y" in coord_map


def test_infer_coords_cf_convention(cf_coords_h5: Path):
    conn = HDF5Connector()
    with h5py.File(str(cf_coords_h5), "r") as f:
        datasets = conn._discover_datasets(f)
        info = [d for d in datasets if d.path == "/temperature"][0]
        spatial, coord_map = conn._infer_coordinates(f, info)

    assert spatial.crs == "EPSG:4326"
    assert "x" in coord_map
    assert "y" in coord_map
    xmin, ymin, xmax, ymax = spatial.extent
    assert xmin == pytest.approx(20.0)
    assert ymin == pytest.approx(10.0)


def test_infer_coords_none(no_coords_h5: Path):
    conn = HDF5Connector()
    with h5py.File(str(no_coords_h5), "r") as f:
        datasets = conn._discover_datasets(f)
        info = [d for d in datasets if d.path == "/image"][0]
        spatial, coord_map = conn._infer_coordinates(f, info)

    assert spatial.crs is None
    assert spatial.extent == (0, 0, 128, 64)
    assert coord_map == {}


# ---------------------------------------------------------------------------
# Lazy materialization
# ---------------------------------------------------------------------------


def test_materialize_lazy_simple(simple_h5: Path, tmp_path: Path):
    conn = HDF5Connector()
    result = conn.materialize(str(simple_h5), tmp_path, lazy=True)

    assert result.strategy == "lazy_handle"
    assert result.artifact.backing.kind.name == "LAZY_HANDLE"
    assert result.artifact.spatial.crs == "EPSG:4326"
    assert result.artifact.metadata["dataset_path"] == "/elevation"

    # No files should be written in workspace
    written = list(tmp_path.glob("*.tif"))
    assert len(written) == 0


def test_materialize_lazy_explicit_dataset(nested_h5: Path, tmp_path: Path):
    conn = HDF5Connector()
    ref = f"{nested_h5}::/science_data/water_area"
    result = conn.materialize(ref, tmp_path, lazy=True)

    assert result.artifact.metadata["dataset_path"] == "/science_data/water_area"


def test_materialize_lazy_auto_select(nested_h5: Path, tmp_path: Path):
    conn = HDF5Connector()
    result = conn.materialize(str(nested_h5), tmp_path, lazy=True)

    # Should auto-select one of the science_data datasets
    assert result.artifact.metadata["dataset_path"].startswith("/science_data/")


# ---------------------------------------------------------------------------
# Eager materialization
# ---------------------------------------------------------------------------


def test_materialize_eager_writes_geotiff(simple_h5: Path, tmp_path: Path):
    import rasterio

    conn = HDF5Connector()
    result = conn.materialize(str(simple_h5), tmp_path)

    assert result.strategy == "normalized"
    assert result.artifact.backing.kind.name == "LOCAL_FILE"

    output = Path(result.artifact.backing.uri)
    assert output.exists()

    # Verify readable by rasterio
    with rasterio.open(str(output)) as src:
        assert src.count == 1
        assert src.height == 100
        assert src.width == 200
        data = src.read(1)
        assert data.shape == (100, 200)


def test_materialize_eager_complex_pair(complex_h5: Path, tmp_path: Path):
    import rasterio

    conn = HDF5Connector()
    result = conn.materialize(str(complex_h5), tmp_path)

    assert result.artifact.spatial.band_count == 2
    assert result.artifact.metadata["is_complex_pair"] is True

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.count == 2
        assert src.descriptions[0] == "real"
        assert src.descriptions[1] == "imag"


def test_materialize_eager_no_coords(no_coords_h5: Path, tmp_path: Path):
    import rasterio

    conn = HDF5Connector()
    result = conn.materialize(str(no_coords_h5), tmp_path)

    assert result.artifact.spatial.crs is None
    assert result.artifact.spatial.extent == (0, 0, 128, 64)

    output = Path(result.artifact.backing.uri)
    with rasterio.open(str(output)) as src:
        assert src.crs is None


def test_materialize_eager_has_content_hash(simple_h5: Path, tmp_path: Path):
    conn = HDF5Connector()
    result = conn.materialize(str(simple_h5), tmp_path)
    assert result.artifact.backing.content_hash is not None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_returns_group_structure(nested_h5: Path):
    conn = HDF5Connector()
    meta = conn.metadata(str(nested_h5))

    assert "group_structure" in meta
    tree = meta["group_structure"]
    assert "/science_data" in tree
    assert "water_height" in tree["/science_data"]
    assert "water_area" in tree["/science_data"]


def test_metadata_global_attrs(simple_h5: Path):
    conn = HDF5Connector()
    meta = conn.metadata(str(simple_h5))

    assert meta["global_attributes"]["crs"] == "EPSG:4326"
    assert meta["dataset"] == "/elevation"


def test_metadata_explicit_dataset(nested_h5: Path):
    conn = HDF5Connector()
    meta = conn.metadata(f"{nested_h5}::/science_data/water_area")

    assert meta["dataset"] == "/science_data/water_area"
    assert meta["shape"] == (50, 80)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_materialize_missing_file(tmp_path: Path):
    from quarry_core.connector import MaterializeError

    conn = HDF5Connector()
    with pytest.raises(MaterializeError, match="File not found"):
        conn.materialize("/nonexistent/file.h5", tmp_path)


def test_materialize_wrong_extension(tmp_path: Path):
    from quarry_core.connector import MaterializeError

    conn = HDF5Connector()
    with pytest.raises(MaterializeError, match="Not an HDF5 file"):
        conn.materialize("/some/path/file.txt", tmp_path)


def test_materialize_nonexistent_dataset(simple_h5: Path, tmp_path: Path):
    from quarry_core.connector import MaterializeError

    conn = HDF5Connector()
    with pytest.raises(MaterializeError, match="not found"):
        conn.materialize(f"{simple_h5}::/bogus/path", tmp_path)


def test_metadata_missing_file():
    from quarry_core.connector import MaterializeError

    conn = HDF5Connector()
    with pytest.raises(MaterializeError, match="File not found"):
        conn.metadata("/nonexistent/file.h5")


def test_materialize_no_suitable_dataset(tmp_path: Path):
    """File with only 1D data should fail auto-select."""
    from quarry_core.connector import MaterializeError

    path = tmp_path / "only_1d.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("timeseries", data=np.arange(100))

    conn = HDF5Connector()
    with pytest.raises(MaterializeError, match="No suitable 2D"):
        conn.materialize(str(path), tmp_path)
