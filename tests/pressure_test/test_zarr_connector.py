"""Pressure test: ZarrConnector.

Lane: connector

Validates Zarr store materialization:
- source_ref parsing (path, path::variable, SourceRef with params)
- Variable auto-selection (first array in group)
- Lazy = metadata-only with LAZY_HANDLE backing
- Eager = wrap in place with LOCAL_FILE backing
- Discover: list arrays in store, list .zarr stores in directory
- Metadata: shape, chunks, dtype, compressor, CRS extraction
- Error handling: nonexistent paths, not a zarr store, nonexistent variable
"""

from __future__ import annotations

import numpy as np
import pytest
from quarry_connectors.zarr_connector import ZarrConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import Connector, ConnectorCapability, MaterializeError
from quarry_core.source_ref import SourceRef, SourceRefKind

# Skip all tests if zarr not installed
zarr = pytest.importorskip("zarr")

# ---------------------------------------------------------------------------
# Fixtures: create real Zarr stores
# ---------------------------------------------------------------------------


def _create_zarr_store(path, arrays_dict, attrs=None):
    """Create a Zarr store with arrays.

    Args:
        path: directory path (will be created as .zarr)
        arrays_dict: {"var_name": {"data": np.array, "chunks": (10,10), "attrs": {}}}
        attrs: root group attributes
    """
    store = zarr.open(str(path), mode="w")
    if attrs:
        store.attrs.update(attrs)
    for name, spec in arrays_dict.items():
        # Use create_array for zarr v3 compatibility
        data = spec["data"]
        chunks = spec.get("chunks")
        # zarr v3: use create_array with data=, not create_dataset
        arr = store.create_array(name, data=data, chunks=chunks)
        if "attrs" in spec:
            arr.attrs.update(spec["attrs"])
    return store


@pytest.fixture
def zarr_elevation(tmp_path):
    """Create a Zarr store with single 2D elevation array."""
    path = tmp_path / "elevation.zarr"
    data = np.random.rand(100, 100).astype("float32")
    _create_zarr_store(
        path,
        {
            "elevation": {
                "data": data,
                "chunks": (50, 50),
                "attrs": {"crs_wkt": "EPSG:4326"},
            }
        },
        attrs={
            "geospatial_lat_min": -90,
            "geospatial_lat_max": 90,
            "geospatial_lon_min": -180,
            "geospatial_lon_max": 180,
        },
    )
    return path


@pytest.fixture
def zarr_multivar(tmp_path):
    """Create a Zarr store with multiple arrays (multi-variable)."""
    path = tmp_path / "climate.zarr"
    temp_data = np.random.rand(10, 50, 50).astype("float32")  # time, lat, lon
    precip_data = np.random.rand(10, 50, 50).astype("float32")
    _create_zarr_store(
        path,
        {
            "temperature": {
                "data": temp_data,
                "chunks": (5, 25, 25),
                "attrs": {"units": "K", "long_name": "Surface temperature"},
            },
            "precipitation": {
                "data": precip_data,
                "chunks": (5, 25, 25),
                "attrs": {"units": "mm", "long_name": "Precipitation"},
            },
        },
    )
    return path


@pytest.fixture
def zarr_with_crs(tmp_path):
    """Create a Zarr store with CRS in various attribute conventions."""
    path = tmp_path / "georeferenced.zarr"
    data = np.random.rand(64, 64).astype("float32")
    _create_zarr_store(
        path,
        {
            "data": {
                "data": data,
                "chunks": (32, 32),
                "attrs": {
                    "crs": "EPSG:32633",  # Direct CRS attribute
                    "geospatial_x_min": 500000,
                    "geospatial_x_max": 501000,
                    "geospatial_y_min": 6000000,
                    "geospatial_y_max": 6001000,
                },
            }
        },
    )
    return path


@pytest.fixture
def directory_with_zarr(tmp_path, zarr_elevation, zarr_multivar):
    """Directory containing multiple Zarr stores."""
    return tmp_path


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_connector_protocol(self):
        conn = ZarrConnector()
        assert isinstance(conn, Connector)

    def test_capabilities(self):
        conn = ZarrConnector()
        caps = conn.capabilities
        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps
        assert ConnectorCapability.DISCOVER in caps

    def test_name(self):
        conn = ZarrConnector()
        assert conn.name == "zarr"


# ---------------------------------------------------------------------------
# Eager materialization
# ---------------------------------------------------------------------------


class TestZarrEagerLocal:
    """Validate eager materialization of local Zarr stores."""

    def test_eager_produces_raster(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path)

        assert result.artifact.type == ArtifactType.RASTER

    def test_eager_produces_local_file_backing(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert ".zarr" in result.artifact.backing.uri

    def test_eager_wrapped_local_strategy(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_shape(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path)

        # Artifact metadata is frozen, lists become tuples
        assert result.artifact.metadata["shape"] == (100, 100)

    def test_eager_band_count_2d(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path)

        assert result.artifact.spatial.band_count == 1

    def test_eager_band_count_3d(self, zarr_multivar, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_multivar), tmp_path)

        # 3D array (10, 50, 50) -> band_count should be 10 (time dimension)
        assert result.artifact.spatial.band_count == 10

    def test_eager_extent_from_attrs(self, zarr_with_crs, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_with_crs), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(500000)
        assert ymin == pytest.approx(6000000)
        assert xmax == pytest.approx(501000)
        assert ymax == pytest.approx(6001000)

    def test_eager_lineage(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path)

        assert result.artifact.lineage.params["source"] == "zarr"
        assert result.artifact.lineage.params["lazy"] is False
        assert "shape" in result.artifact.lineage.params

    def test_eager_metadata_includes_chunks(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path)

        # Artifact metadata is frozen, lists become tuples
        assert result.artifact.metadata["chunks"] == (50, 50)

    def test_eager_metadata_includes_dtype(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path)

        assert "float32" in result.artifact.metadata["dtype"]


# ---------------------------------------------------------------------------
# Lazy materialization
# ---------------------------------------------------------------------------


class TestZarrLazyLocal:
    """Validate lazy (metadata-only) materialization."""

    def test_lazy_backing_kind(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_has_spatial(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path, lazy=True)

        assert result.artifact.spatial.band_count is not None
        # Artifact metadata is frozen, lists become tuples
        assert result.artifact.metadata["shape"] == (100, 100)

    def test_lazy_lineage_records_lazy_flag(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True

    def test_lazy_backing_uri(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path, lazy=True)

        assert ".zarr" in result.artifact.backing.uri


# ---------------------------------------------------------------------------
# Variable selection
# ---------------------------------------------------------------------------


class TestZarrVariableSelection:
    """Validate variable selection via :: separator."""

    def test_default_selects_first_array(self, zarr_multivar, tmp_path):
        """When no variable specified, should select first array."""
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_multivar), tmp_path, lazy=True)

        # First array in group should be "temperature" (alphabetically or by creation order)
        assert result.artifact.name in ["temperature", "precipitation"]

    def test_explicit_variable_via_separator(self, zarr_multivar, tmp_path):
        """Select specific variable via :: separator."""
        conn = ZarrConnector()
        result = conn.materialize(f"{zarr_multivar}::precipitation", tmp_path, lazy=True)

        assert result.artifact.name == "precipitation"
        assert result.artifact.metadata["variable"] == "precipitation"

    def test_explicit_variable_different_shape(self, zarr_multivar, tmp_path):
        """Selected variable should have correct shape."""
        conn = ZarrConnector()
        result = conn.materialize(f"{zarr_multivar}::temperature", tmp_path, lazy=True)

        # Artifact metadata is frozen, lists become tuples
        assert result.artifact.metadata["shape"] == (10, 50, 50)

    def test_available_arrays_in_metadata(self, zarr_multivar, tmp_path):
        """Discover should list available arrays."""
        conn = ZarrConnector()
        entries = conn.discover(str(zarr_multivar))

        names = {e.name for e in entries}
        assert "temperature" in names
        assert "precipitation" in names


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


class TestZarrMetadata:
    """Validate metadata extraction."""

    def test_metadata_shape(self, zarr_elevation):
        conn = ZarrConnector()
        meta = conn.metadata(str(zarr_elevation))

        # metadata() returns lists from _get_array_metadata
        assert meta["shape"] == [100, 100]

    def test_metadata_chunks(self, zarr_elevation):
        conn = ZarrConnector()
        meta = conn.metadata(str(zarr_elevation))

        # metadata() returns lists from _get_array_metadata
        assert meta["chunks"] == [50, 50]

    def test_metadata_dtype(self, zarr_elevation):
        conn = ZarrConnector()
        meta = conn.metadata(str(zarr_elevation))

        assert "float32" in meta["dtype"]

    def test_metadata_compressor(self, zarr_elevation):
        conn = ZarrConnector()
        meta = conn.metadata(str(zarr_elevation))

        # Compressor may be None or a compressor object
        assert "compressor" in meta

    def test_metadata_crs_wkt_extraction(self, zarr_elevation):
        """Extract CRS from crs_wkt attribute."""
        conn = ZarrConnector()
        meta = conn.metadata(str(zarr_elevation))

        assert meta["crs"] is not None
        assert "4326" in meta["crs"]

    def test_metadata_crs_direct_extraction(self, zarr_with_crs):
        """Extract CRS from direct crs attribute."""
        conn = ZarrConnector()
        meta = conn.metadata(str(zarr_with_crs))

        assert meta["crs"] is not None
        assert "32633" in meta["crs"]

    def test_metadata_attrs_included(self, zarr_multivar):
        conn = ZarrConnector()
        meta = conn.metadata(f"{zarr_multivar}::temperature")

        assert "attrs" in meta
        assert meta["attrs"]["units"] == "K"

    def test_metadata_band_count(self, zarr_multivar):
        conn = ZarrConnector()
        meta = conn.metadata(f"{zarr_multivar}::temperature")

        # 3D array: (10, 50, 50) -> band_count = min(10, 50, 50) = 10
        assert meta["band_count"] == 10


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestZarrDiscover:
    """Validate discovery of Zarr arrays and stores."""

    def test_discover_lists_arrays_in_store(self, zarr_multivar):
        """When query is a Zarr store, list arrays within."""
        conn = ZarrConnector()
        entries = conn.discover(str(zarr_multivar))

        names = {e.name for e in entries}
        assert "temperature" in names
        assert "precipitation" in names

    def test_discover_source_refs_have_separator(self, zarr_multivar):
        """Array entries should have :: separator in source_ref."""
        conn = ZarrConnector()
        entries = conn.discover(str(zarr_multivar))

        for entry in entries:
            assert "::" in entry.source_ref

    def test_discover_includes_array_metadata(self, zarr_multivar):
        conn = ZarrConnector()
        entries = conn.discover(str(zarr_multivar))

        temp_entry = next(e for e in entries if e.name == "temperature")
        assert "shape" in temp_entry.metadata
        assert "chunks" in temp_entry.metadata
        assert "dtype" in temp_entry.metadata

    def test_discover_lists_zarr_stores_in_directory(self, directory_with_zarr):
        """When query is a directory, list .zarr stores."""
        conn = ZarrConnector()
        entries = conn.discover(str(directory_with_zarr))

        names = {e.name for e in entries}
        assert "elevation" in names
        assert "climate" in names

    def test_discover_requires_path(self):
        conn = ZarrConnector()
        with pytest.raises(MaterializeError):
            conn.discover()

    def test_discover_string_query(self, zarr_multivar):
        conn = ZarrConnector()
        entries = conn.discover(str(zarr_multivar))

        assert len(entries) >= 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestZarrErrors:
    """Validate error cases."""

    def test_nonexistent_path_raises(self, tmp_path):
        conn = ZarrConnector()
        nonexistent = tmp_path / "does_not_exist.zarr"

        with pytest.raises(MaterializeError, match="not found"):
            conn.materialize(str(nonexistent), tmp_path)

    def test_not_a_zarr_store_raises(self, tmp_path):
        conn = ZarrConnector()
        bad_path = tmp_path / "not_zarr.txt"
        bad_path.write_text("this is not a zarr store")

        with pytest.raises(MaterializeError, match="Not a Zarr"):
            conn.materialize(str(bad_path), tmp_path)

    def test_nonexistent_variable_raises(self, zarr_multivar, tmp_path):
        conn = ZarrConnector()

        with pytest.raises(MaterializeError, match="not found"):
            conn.materialize(f"{zarr_multivar}::nonexistent_var", tmp_path)

    def test_discover_nonexistent_path_raises(self, tmp_path):
        conn = ZarrConnector()
        nonexistent = tmp_path / "nope.zarr"

        with pytest.raises(MaterializeError, match="not found"):
            conn.discover(str(nonexistent))

    def test_metadata_nonexistent_path_raises(self, tmp_path):
        conn = ZarrConnector()
        nonexistent = tmp_path / "nope.zarr"

        with pytest.raises(MaterializeError, match="not found"):
            conn.metadata(str(nonexistent))


# ---------------------------------------------------------------------------
# SourceRef support
# ---------------------------------------------------------------------------


class TestZarrSourceRef:
    """Validate SourceRef handling."""

    def test_sourceref_local(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        ref = SourceRef.local(str(zarr_elevation))
        result = conn.materialize(ref, tmp_path, lazy=True)

        assert result.artifact.type == ArtifactType.RASTER

    def test_raw_string_path(self, zarr_elevation, tmp_path):
        conn = ZarrConnector()
        result = conn.materialize(str(zarr_elevation), tmp_path, lazy=True)

        assert result.artifact.type == ArtifactType.RASTER

    def test_sourceref_with_variable_param(self, zarr_multivar, tmp_path):
        """SourceRef with variable in params."""
        conn = ZarrConnector()
        ref = SourceRef(
            raw=f"{zarr_multivar}::precipitation",
            kind=SourceRefKind.LOCAL_RASTER,
            params={"path": str(zarr_multivar), "variable": "precipitation"},
        )
        result = conn.materialize(ref, tmp_path, lazy=True)

        assert result.artifact.name == "precipitation"


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestZarrCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        conn = ZarrConnector()
        assert conn.name == "zarr"

    def test_capabilities(self):
        from quarry_core.connector import ConnectorCapability

        conn = ZarrConnector()
        caps = conn.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps


# ---------------------------------------------------------------------------
# Zarr not installed handling
# ---------------------------------------------------------------------------


class TestZarrNotInstalled:
    """Validate behavior when zarr is not installed."""

    def test_materialize_raises_without_zarr(self, zarr_elevation, tmp_path, monkeypatch):
        """Should raise MaterializeError with install hint."""
        # Temporarily disable zarr
        monkeypatch.setattr("quarry_connectors.zarr_connector.HAS_ZARR", False)
        monkeypatch.setattr("quarry_connectors.zarr_connector.zarr", None)

        conn = ZarrConnector()
        with pytest.raises(MaterializeError, match="zarr package not installed"):
            conn.materialize(str(zarr_elevation), tmp_path)

    def test_discover_raises_without_zarr(self, zarr_elevation, monkeypatch):
        """Should raise MaterializeError with install hint."""
        monkeypatch.setattr("quarry_connectors.zarr_connector.HAS_ZARR", False)
        monkeypatch.setattr("quarry_connectors.zarr_connector.zarr", None)

        conn = ZarrConnector()
        with pytest.raises(MaterializeError, match="zarr package not installed"):
            conn.discover(str(zarr_elevation))

    def test_metadata_raises_without_zarr(self, zarr_elevation, monkeypatch):
        """Should raise MaterializeError with install hint."""
        monkeypatch.setattr("quarry_connectors.zarr_connector.HAS_ZARR", False)
        monkeypatch.setattr("quarry_connectors.zarr_connector.zarr", None)

        conn = ZarrConnector()
        with pytest.raises(MaterializeError, match="zarr package not installed"):
            conn.metadata(str(zarr_elevation))
