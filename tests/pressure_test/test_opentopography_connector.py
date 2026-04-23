"""
Pressure test: OpenTopographyConnector.

Lane: connector

Validates OpenTopography DEM materialization:
- source_ref parsing (dataset ID, opentopo:// URL, SourceRef with params)
- lazy mode: LAZY_HANDLE artifact with spatial descriptor from bbox
- eager mode: download GeoTIFF, RASTER artifact with full metadata
- discover: returns entries for all known datasets
- metadata: returns dataset info without materialization
- error handling: unknown dataset, missing bbox, API errors

NO real API calls — all HTTP requests are mocked.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
from quarry_connectors.opentopography import (
    _DATASET_INFO,
    KNOWN_DATASETS,
    OpenTopographyConnector,
)
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_dem_bytes(tmp_path):
    """Create a real GeoTIFF as mock API response bytes."""
    path = tmp_path / "mock_dem.tif"
    transform = from_bounds(-120, 35, -119, 36, 100, 100)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.random.rand(1, 100, 100).astype("float32"))
    return path.read_bytes()


@pytest.fixture()
def connector_with_key():
    """Connector with API key."""
    return OpenTopographyConnector(api_key="test_api_key_12345")


@pytest.fixture()
def connector_no_key():
    """Connector without API key."""
    return OpenTopographyConnector()


@pytest.fixture()
def valid_bbox():
    """Valid bounding box for testing."""
    return (-120.0, 35.0, -119.0, 36.0)  # west, south, east, north


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_requests_get(mock_dem_bytes, status_code=200):
    """Create a mock requests.get that returns the mock DEM bytes."""

    def mock_get(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.raise_for_status = MagicMock()
        if status_code >= 400:
            from requests import HTTPError

            mock_resp.raise_for_status.side_effect = HTTPError(
                f"{status_code} Error", response=mock_resp
            )
        mock_resp.raw = io.BytesIO(mock_dem_bytes)
        mock_resp.text = "Error" if status_code >= 400 else "OK"
        return mock_resp

    return mock_get


def _mock_requests_get_error(status_code, error_text="API Error"):
    """Create a mock requests.get that returns an error."""

    def mock_get(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = error_text
        from requests import HTTPError

        mock_resp.raise_for_status.side_effect = HTTPError(
            f"{status_code} Error", response=mock_resp
        )
        return mock_resp

    return mock_get


# ---------------------------------------------------------------------------
# SourceRef parsing tests
# ---------------------------------------------------------------------------


class TestOpenTopoSourceRefParsing:
    """Validate source_ref parsing for various formats."""

    def test_raw_dataset_id_only(self, connector_no_key, valid_bbox):
        """Raw string dataset ID requires bbox in params."""
        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})
        result = connector_no_key.materialize(ref, MagicMock(), lazy=True)

        assert result.artifact.metadata["dataset"] == "SRTMGL1"

    def test_opentopo_url_with_bbox(self, connector_no_key):
        """opentopo://DATASET_ID?west=...&south=...&east=...&north=... format."""
        ref = "opentopo://SRTMGL1?west=-120&south=35&east=-119&north=36"
        result = connector_no_key.materialize(ref, MagicMock(), lazy=True)

        assert result.artifact.metadata["dataset"] == "SRTMGL1"
        assert result.artifact.spatial.extent == (-120.0, 35.0, -119.0, 36.0)

    def test_sourceref_with_bbox_tuple(self, connector_no_key, valid_bbox):
        """SourceRef with bbox as tuple."""
        ref = SourceRef(
            raw="SRTMGL1",
            params={"dataset": "SRTMGL1", "bbox": valid_bbox},
        )
        result = connector_no_key.materialize(ref, MagicMock(), lazy=True)

        assert result.artifact.spatial.extent == valid_bbox

    def test_sourceref_with_individual_coords(self, connector_no_key):
        """SourceRef with west/south/east/north as separate params."""
        ref = SourceRef(
            raw="SRTMGL1",
            params={
                "dataset": "SRTMGL1",
                "west": -120.0,
                "south": 35.0,
                "east": -119.0,
                "north": 36.0,
            },
        )
        result = connector_no_key.materialize(ref, MagicMock(), lazy=True)

        assert result.artifact.spatial.extent == (-120.0, 35.0, -119.0, 36.0)

    def test_sourceref_dataset_in_raw(self, connector_no_key, valid_bbox):
        """SourceRef with dataset ID in raw, bbox in params."""
        ref = SourceRef(raw="COP30", params={"bbox": valid_bbox})
        result = connector_no_key.materialize(ref, MagicMock(), lazy=True)

        assert result.artifact.metadata["dataset"] == "COP30"


# ---------------------------------------------------------------------------
# Eager materialization tests
# ---------------------------------------------------------------------------


class TestOpenTopoEagerMaterialization:
    """Validate eager (download) materialization with mocked HTTP."""

    def test_eager_produces_raster_artifact(
        self, connector_with_key, valid_bbox, mock_dem_bytes, tmp_path, monkeypatch
    ):
        """Eager mode produces RASTER artifact with LOCAL_FILE backing."""
        monkeypatch.setattr(
            "quarry_connectors.opentopography.requests.get",
            _mock_requests_get(mock_dem_bytes),
        )

        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})
        result = connector_with_key.materialize(ref, tmp_path, lazy=False)

        assert result.artifact.type == ArtifactType.RASTER
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.strategy == "fetched_remote"

    def test_eager_downloads_geotiff(
        self, connector_with_key, valid_bbox, mock_dem_bytes, tmp_path, monkeypatch
    ):
        """Downloaded file is a valid GeoTIFF."""
        monkeypatch.setattr(
            "quarry_connectors.opentopography.requests.get",
            _mock_requests_get(mock_dem_bytes),
        )

        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})
        result = connector_with_key.materialize(ref, tmp_path, lazy=False)

        path = result.artifact.backing.uri
        assert path.endswith(".tif")

        # Verify it's a valid GeoTIFF
        with rasterio.open(path) as src:
            assert src.crs.to_string() == "EPSG:4326"
            assert src.count == 1

    def test_eager_spatial_descriptor_from_file(
        self, connector_with_key, valid_bbox, mock_dem_bytes, tmp_path, monkeypatch
    ):
        """Spatial descriptor is read from downloaded file."""
        monkeypatch.setattr(
            "quarry_connectors.opentopography.requests.get",
            _mock_requests_get(mock_dem_bytes),
        )

        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})
        result = connector_with_key.materialize(ref, tmp_path, lazy=False)

        spatial = result.artifact.spatial
        assert spatial.crs == "EPSG:4326"
        assert spatial.extent is not None
        assert spatial.band_count == 1
        assert spatial.resolution is not None

    def test_eager_lineage_records_params(
        self, connector_with_key, valid_bbox, mock_dem_bytes, tmp_path, monkeypatch
    ):
        """Lineage records source, dataset, bbox, lazy flag."""
        monkeypatch.setattr(
            "quarry_connectors.opentopography.requests.get",
            _mock_requests_get(mock_dem_bytes),
        )

        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})
        result = connector_with_key.materialize(ref, tmp_path, lazy=False)

        lineage = result.artifact.lineage
        assert lineage.params["source"] == "opentopography"
        assert lineage.params["dataset"] == "SRTMGL1"
        assert lineage.params["bbox"] == valid_bbox
        assert lineage.params["lazy"] is False

    def test_eager_content_hash_present(
        self, connector_with_key, valid_bbox, mock_dem_bytes, tmp_path, monkeypatch
    ):
        """Downloaded artifact has content hash."""
        monkeypatch.setattr(
            "quarry_connectors.opentopography.requests.get",
            _mock_requests_get(mock_dem_bytes),
        )

        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})
        result = connector_with_key.materialize(ref, tmp_path, lazy=False)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_requires_api_key(self, connector_no_key, valid_bbox, tmp_path):
        """Eager mode without API key raises error."""
        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})

        with pytest.raises(MaterializeError) as exc_info:
            connector_no_key.materialize(ref, tmp_path, lazy=False)

        assert "API key required" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Lazy materialization tests
# ---------------------------------------------------------------------------


class TestOpenTopoLazyMaterialization:
    """Validate lazy (metadata-only) materialization."""

    def test_lazy_produces_lazy_handle(self, connector_no_key, valid_bbox, tmp_path):
        """Lazy mode produces LAZY_HANDLE backing."""
        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})
        result = connector_no_key.materialize(ref, tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE
        assert result.strategy == "lazy_handle"

    def test_lazy_no_http_request(self, connector_no_key, valid_bbox, tmp_path, monkeypatch):
        """Lazy mode makes no HTTP requests."""
        http_called = False

        def mock_get(*args, **kwargs):
            nonlocal http_called
            http_called = True
            raise Exception("Should not be called in lazy mode")

        monkeypatch.setattr("quarry_connectors.opentopography.requests.get", mock_get)

        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})
        result = connector_no_key.materialize(ref, tmp_path, lazy=True)

        assert not http_called
        assert result.strategy == "lazy_handle"

    def test_lazy_spatial_descriptor_from_bbox(self, connector_no_key, valid_bbox, tmp_path):
        """Spatial descriptor uses provided bbox in lazy mode."""
        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})
        result = connector_no_key.materialize(ref, tmp_path, lazy=True)

        spatial = result.artifact.spatial
        assert spatial.crs == "EPSG:4326"
        assert spatial.extent == valid_bbox
        assert spatial.band_count == 1

    def test_lazy_api_url_in_backing_uri(self, connector_with_key, valid_bbox, tmp_path):
        """Lazy handle URI contains the API URL with all parameters."""
        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})
        result = connector_with_key.materialize(ref, tmp_path, lazy=True)

        uri = result.artifact.backing.uri
        assert "portal.opentopography.org/API/globaldem" in uri
        assert "demtype=SRTMGL1" in uri
        assert "west=-120.0" in uri
        assert "south=35.0" in uri
        assert "east=-119.0" in uri
        assert "north=36.0" in uri
        assert "API_Key=test_api_key_12345" in uri

    def test_lazy_lineage_records_lazy_flag(self, connector_no_key, valid_bbox, tmp_path):
        """Lineage records lazy=True."""
        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})
        result = connector_no_key.materialize(ref, tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True

    def test_lazy_estimate_resolution(self, connector_no_key, valid_bbox, tmp_path):
        """Lazy mode estimates resolution from dataset info."""
        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})
        result = connector_no_key.materialize(ref, tmp_path, lazy=True)

        # SRTMGL1 has 30m resolution
        spatial = result.artifact.spatial
        assert spatial.resolution is not None
        # ~30m / 111000m per degree ≈ 0.00027 degrees
        assert spatial.resolution[0] > 0
        assert spatial.resolution[1] > 0


# ---------------------------------------------------------------------------
# Discover tests
# ---------------------------------------------------------------------------


class TestOpenTopoDiscover:
    """Validate dataset discovery."""

    def test_discover_returns_all_known_datasets(self, connector_no_key):
        """Discover returns entries for all known datasets."""
        entries = connector_no_key.discover()

        dataset_names = {e.name for e in entries}
        assert dataset_names == KNOWN_DATASETS

    def test_discover_entries_have_source_refs(self, connector_no_key):
        """Each entry has a valid source_ref."""
        entries = connector_no_key.discover()

        for entry in entries:
            assert entry.source_ref.startswith("opentopo://")
            dataset_id = entry.source_ref.replace("opentopo://", "")
            assert dataset_id in KNOWN_DATASETS

    def test_discover_entries_have_descriptions(self, connector_no_key):
        """Each entry has a description from _DATASET_INFO."""
        entries = connector_no_key.discover()

        for entry in entries:
            assert entry.description
            assert entry.description == _DATASET_INFO[entry.name]["description"]

    def test_discover_spatial_hint_global_extent(self, connector_no_key):
        """Spatial hint includes global extent and CRS."""
        entries = connector_no_key.discover()

        for entry in entries:
            assert entry.spatial_hint["crs"] == "EPSG:4326"
            assert entry.spatial_hint["extent"] == (-180.0, -90.0, 180.0, 90.0)

    def test_discover_metadata_includes_resolution(self, connector_no_key):
        """Entry metadata includes resolution info."""
        entries = connector_no_key.discover()

        for entry in entries:
            assert "resolution" in entry.metadata
            assert entry.metadata["resolution"] == _DATASET_INFO[entry.name]["resolution"]


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestOpenTopoMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_dataset_info(self, connector_no_key):
        """Metadata returns dataset description, CRS, resolution, coverage."""
        meta = connector_no_key.metadata("SRTMGL1")

        assert meta["dataset_id"] == "SRTMGL1"
        assert meta["description"] == "SRTM GL1 30m"
        assert meta["crs"] == "EPSG:4326"
        assert meta["resolution_meters"] == 30
        assert meta["coverage"] == "60S-60N"

    def test_metadata_opentopo_url(self, connector_no_key):
        """Metadata works with opentopo:// URL format."""
        meta = connector_no_key.metadata("opentopo://COP30")

        assert meta["dataset_id"] == "COP30"
        assert meta["description"] == "Copernicus DEM 30m"

    def test_metadata_all_known_datasets(self, connector_no_key):
        """Metadata works for all known datasets."""
        for dataset_id in KNOWN_DATASETS:
            meta = connector_no_key.metadata(dataset_id)
            assert meta["dataset_id"] == dataset_id
            assert meta["crs"] == "EPSG:4326"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestOpenTopoErrors:
    """Validate error cases."""

    def test_unknown_dataset_raises(self, connector_no_key, valid_bbox, tmp_path):
        """Unknown dataset ID raises MaterializeError."""
        ref = SourceRef(raw="UNKNOWN_DATASET", params={"bbox": valid_bbox})

        with pytest.raises(MaterializeError) as exc_info:
            connector_no_key.materialize(ref, tmp_path, lazy=True)

        assert "Unknown dataset" in str(exc_info.value)
        assert "UNKNOWN_DATASET" in str(exc_info.value)

    def test_missing_bbox_raises(self, connector_no_key, tmp_path):
        """Missing bbox raises MaterializeError."""
        ref = "SRTMGL1"  # No bbox

        with pytest.raises(MaterializeError) as exc_info:
            connector_no_key.materialize(ref, tmp_path, lazy=True)

        assert "Bounding box required" in str(exc_info.value)

    def test_api_error_400(self, connector_with_key, valid_bbox, tmp_path, monkeypatch):
        """API 400 error raises MaterializeError."""
        monkeypatch.setattr(
            "quarry_connectors.opentopography.requests.get",
            _mock_requests_get_error(400, "Invalid bounding box"),
        )

        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})

        with pytest.raises(MaterializeError) as exc_info:
            connector_with_key.materialize(ref, tmp_path, lazy=False)

        assert "400" in str(exc_info.value)

    def test_api_error_500(self, connector_with_key, valid_bbox, tmp_path, monkeypatch):
        """API 500 error raises MaterializeError."""
        monkeypatch.setattr(
            "quarry_connectors.opentopography.requests.get",
            _mock_requests_get_error(500, "Internal Server Error"),
        )

        ref = SourceRef(raw="SRTMGL1", params={"bbox": valid_bbox})

        with pytest.raises(MaterializeError) as exc_info:
            connector_with_key.materialize(ref, tmp_path, lazy=False)

        assert "500" in str(exc_info.value)

    def test_authenticate_requires_api_key(self, connector_no_key):
        """authenticate() requires api_key in credentials."""
        with pytest.raises(MaterializeError) as exc_info:
            connector_no_key.authenticate({})

        assert "api_key" in str(exc_info.value)

    def test_authenticate_sets_api_key(self, connector_no_key):
        """authenticate() sets the API key."""
        connector_no_key.authenticate({"api_key": "new_key_123"})

        # Verify by checking lazy materialization includes key in URL
        ref = SourceRef(raw="SRTMGL1", params={"bbox": (-120, 35, -119, 36)})
        result = connector_no_key.materialize(ref, MagicMock(), lazy=True)

        assert "new_key_123" in result.artifact.backing.uri


# ---------------------------------------------------------------------------
# Capabilities tests
# ---------------------------------------------------------------------------


class TestOpenTopoCapabilities:
    """Validate connector capabilities."""

    def test_name_is_opentopography(self, connector_no_key):
        assert connector_no_key.name == "opentopography"

    def test_capabilities_include_materialize(self, connector_no_key):
        from quarry_core.connector import ConnectorCapability

        caps = connector_no_key.capabilities
        assert ConnectorCapability.MATERIALIZE in caps

    def test_capabilities_include_discover(self, connector_no_key):
        from quarry_core.connector import ConnectorCapability

        caps = connector_no_key.capabilities
        assert ConnectorCapability.DISCOVER in caps

    def test_capabilities_include_authenticate(self, connector_no_key):
        from quarry_core.connector import ConnectorCapability

        caps = connector_no_key.capabilities
        assert ConnectorCapability.AUTHENTICATE in caps

    def test_capabilities_include_lazy(self, connector_no_key):
        from quarry_core.connector import ConnectorCapability

        caps = connector_no_key.capabilities
        assert ConnectorCapability.MATERIALIZE_LAZY in caps

    def test_capabilities_include_metadata(self, connector_no_key):
        from quarry_core.connector import ConnectorCapability

        caps = connector_no_key.capabilities
        assert ConnectorCapability.METADATA_ONLY in caps
