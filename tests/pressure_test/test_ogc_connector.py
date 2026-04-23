"""
Pressure test: OGCServicesConnector.

Lane: connector

Validates WMS and WFS OGC service materialization:
- source_ref parsing (wms::url::layer, wfs::url::layer, SourceRef with params)
- WMS materialization: lazy (LAZY_HANDLE) vs eager (LOCAL_FILE raster)
- WFS materialization: lazy (LAZY_HANDLE) vs eager (LOCAL_FILE vector)
- discover: list layers from GetCapabilities
- metadata: layer info without materialization
- error handling: missing owslib, invalid service type, layer not found

Since we can't hit real OGC services, we use monkeypatch to mock owslib responses.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from quarry_connectors.ogc_services import HAS_OWSLIB, OGCServicesConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def mock_wms_capabilities():
    """Create a mock WMS capabilities response."""
    mock_wms = MagicMock()
    mock_wms.version = "1.3.0"
    mock_wms.url = "https://example.com/wms"

    # Mock layer
    mock_layer = MagicMock()
    mock_layer.title = "Test Layer"
    mock_layer.abstract = "A test layer for unit testing"
    mock_layer.boundingBoxWGS84 = (-180, -90, 180, 90)
    mock_layer.boundingBox = (-180, -90, 180, 90)
    mock_layer.crsOptions = ["EPSG:4326", "EPSG:3857"]
    mock_layer.keywords = ["test", "mock"]

    # Set up contents dict and __getitem__ to return the layer
    mock_wms.contents = {"test_layer": mock_layer}
    mock_wms.__getitem__ = MagicMock(return_value=mock_layer)

    # Mock GetMap operation
    mock_operation = MagicMock()
    mock_operation.formatOptions = ["image/tiff", "image/png", "image/jpeg"]
    mock_wms.getOperationByName = MagicMock(return_value=mock_operation)

    return mock_wms


@pytest.fixture()
def mock_wfs_capabilities():
    """Create a mock WFS capabilities response."""
    mock_wfs = MagicMock()
    mock_wfs.version = "2.0.0"
    mock_wfs.url = "https://example.com/wfs"

    # Mock feature type
    mock_feature = MagicMock()
    mock_feature.title = "Test Feature Type"
    mock_feature.abstract = "A test feature type for unit testing"
    mock_feature.boundingBoxWGS84 = (-122.5, 37.5, -122.0, 38.0)
    mock_feature.boundingBox = (-122.5, 37.5, -122.0, 38.0)
    mock_feature.crsOptions = ["EPSG:4326", "EPSG:32610"]
    mock_feature.keywords = ["test", "vector"]

    # Set up contents dict and __getitem__ to return the feature
    mock_wfs.contents = {"test_feature": mock_feature}
    mock_wfs.__getitem__ = MagicMock(return_value=mock_feature)

    # Mock GetFeature operation
    mock_operation = MagicMock()
    mock_operation.formatOptions = ["application/json", "GML3", "GML2"]
    mock_wfs.getOperationByName = MagicMock(return_value=mock_operation)

    return mock_wfs


@pytest.fixture()
def connector():
    """Create an OGCServicesConnector instance."""
    return OGCServicesConnector()


@pytest.fixture()
def workspace(tmp_path):
    """Create a temporary workspace directory."""
    return tmp_path / "workspace"


# -----------------------------------------------------------------------------
# Source Ref Parsing Tests
# -----------------------------------------------------------------------------


class TestOGCSourceRefParsing:
    """Validate source_ref parsing for OGC services."""

    def test_parse_wms_string(self, connector):
        """Parse 'wms::url::layer' format."""
        service_type, url, layer, params = connector._parse_source_ref(
            "wms::https://example.com/wms::test_layer"
        )
        assert service_type == "wms"
        assert url == "https://example.com/wms"
        assert layer == "test_layer"
        assert params == {}

    def test_parse_wfs_string(self, connector):
        """Parse 'wfs::url::layer' format."""
        service_type, url, layer, params = connector._parse_source_ref(
            "wfs::https://example.com/wfs::test_feature"
        )
        assert service_type == "wfs"
        assert url == "https://example.com/wfs"
        assert layer == "test_feature"
        assert params == {}

    def test_parse_sourceref_with_params(self, connector):
        """Parse SourceRef with service, url, layer params."""
        ref = SourceRef(
            raw="wms::https://example.com/wms::test_layer",
            kind=None,  # type: ignore[arg-type]
            params={
                "service": "wms",
                "url": "https://example.com/wms",
                "layer": "test_layer",
                "bbox": "-180,-90,180,90",
                "width": 2048,
            },
        )
        service_type, url, layer, params = connector._parse_source_ref(ref)
        assert service_type == "wms"
        assert url == "https://example.com/wms"
        assert layer == "test_layer"
        assert params["bbox"] == "-180,-90,180,90"
        assert params["width"] == 2048

    def test_parse_invalid_format_raises(self, connector):
        """Invalid format raises MaterializeError."""
        with pytest.raises(MaterializeError):
            connector._parse_source_ref("invalid_format")

    def test_parse_missing_service_type(self, connector):
        """Missing service type in SourceRef raises error."""
        ref = SourceRef(
            raw="https://example.com/wms::test_layer",
            kind=None,  # type: ignore[arg-type]
            params={"url": "https://example.com/wms", "layer": "test_layer"},
        )
        # Should infer from raw string prefix if present
        with pytest.raises(MaterializeError):
            connector._parse_source_ref(ref)


# -----------------------------------------------------------------------------
# WMS Materialization Tests
# -----------------------------------------------------------------------------


class TestOGCWMSMaterialization:
    """Validate WMS layer materialization."""

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wms_lazy_materialization(self, connector, workspace, mock_wms_capabilities):
        """Lazy WMS materialization produces LAZY_HANDLE artifact."""
        with patch(
            "quarry_connectors.ogc_services.WebMapService", return_value=mock_wms_capabilities
        ):
            result = connector.materialize(
                "wms::https://example.com/wms::test_layer",
                workspace,
                lazy=True,
            )

        assert result.artifact.type == ArtifactType.RASTER
        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE
        assert result.strategy == "lazy_handle"
        assert result.artifact.name == "test_layer"
        assert result.artifact.spatial.crs == "EPSG:4326"
        # Extent comes from the layer's boundingBox
        assert result.artifact.spatial.extent == (-180, -90, 180, 90)

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wms_lazy_lineage(self, connector, workspace, mock_wms_capabilities):
        """Lazy WMS materialization records correct lineage."""
        with patch(
            "quarry_connectors.ogc_services.WebMapService", return_value=mock_wms_capabilities
        ):
            result = connector.materialize(
                "wms::https://example.com/wms::test_layer",
                workspace,
                lazy=True,
            )

        lineage = result.artifact.lineage
        assert lineage.params["source"] == "ogc"
        assert lineage.params["service_type"] == "wms"
        assert lineage.params["url"] == "https://example.com/wms"
        assert lineage.params["layer"] == "test_layer"
        assert lineage.params["lazy"] is True

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wms_lazy_metadata(self, connector, workspace, mock_wms_capabilities):
        """Lazy WMS artifact includes layer metadata."""
        with patch(
            "quarry_connectors.ogc_services.WebMapService", return_value=mock_wms_capabilities
        ):
            result = connector.materialize(
                "wms::https://example.com/wms::test_layer",
                workspace,
                lazy=True,
            )

        meta = result.artifact.metadata
        assert meta["service_type"] == "wms"
        # Title is accessed from the layer object
        assert meta["title"] == "Test Layer"
        assert "EPSG:4326" in meta["crs_options"]

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wms_eager_materialization(self, connector, workspace, mock_wms_capabilities):
        """Eager WMS materialization downloads raster image."""
        # Mock the requests.get call for GetMap
        mock_response = MagicMock()
        mock_response.iter_content = MagicMock(return_value=[b"fake image data"])
        mock_response.raise_for_status = MagicMock()

        with patch(
            "quarry_connectors.ogc_services.WebMapService", return_value=mock_wms_capabilities
        ):
            with patch("quarry_connectors.ogc_services.requests.get", return_value=mock_response):
                result = connector.materialize(
                    "wms::https://example.com/wms::test_layer",
                    workspace,
                    lazy=False,
                )

        assert result.artifact.type == ArtifactType.RASTER
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.strategy == "fetched_remote"
        assert result.artifact.backing.size_bytes is not None
        assert result.artifact.backing.content_hash is not None

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wms_layer_not_found(self, connector, workspace, mock_wms_capabilities):
        """Requesting non-existent WMS layer raises MaterializeError."""
        # Set up mock to raise KeyError for missing layer
        mock_wms_capabilities.__getitem__ = MagicMock(side_effect=KeyError("nonexistent_layer"))

        with patch(
            "quarry_connectors.ogc_services.WebMapService", return_value=mock_wms_capabilities
        ):
            with pytest.raises(MaterializeError) as exc_info:
                connector.materialize(
                    "wms::https://example.com/wms::nonexistent_layer",
                    workspace,
                )

        assert "nonexistent_layer" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()


# -----------------------------------------------------------------------------
# WFS Materialization Tests
# -----------------------------------------------------------------------------


class TestOGCWFSMaterialization:
    """Validate WFS layer materialization."""

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wfs_lazy_materialization(self, connector, workspace, mock_wfs_capabilities):
        """Lazy WFS materialization produces LAZY_HANDLE artifact."""
        with patch(
            "quarry_connectors.ogc_services.WebFeatureService", return_value=mock_wfs_capabilities
        ):
            result = connector.materialize(
                "wfs::https://example.com/wfs::test_feature",
                workspace,
                lazy=True,
            )

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE
        assert result.strategy == "lazy_handle"
        assert result.artifact.name == "test_feature"
        assert result.artifact.spatial.crs == "EPSG:4326"

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wfs_lazy_lineage(self, connector, workspace, mock_wfs_capabilities):
        """Lazy WFS materialization records correct lineage."""
        with patch(
            "quarry_connectors.ogc_services.WebFeatureService", return_value=mock_wfs_capabilities
        ):
            result = connector.materialize(
                "wfs::https://example.com/wfs::test_feature",
                workspace,
                lazy=True,
            )

        lineage = result.artifact.lineage
        assert lineage.params["source"] == "ogc"
        assert lineage.params["service_type"] == "wfs"
        assert lineage.params["lazy"] is True

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wfs_eager_materialization(self, connector, workspace, mock_wfs_capabilities):
        """Eager WFS materialization downloads vector data."""
        # Skip if fiona not available
        try:
            import fiona  # noqa: F401
        except ImportError:
            pytest.skip("fiona not installed")

        # Create a minimal valid GeoJSON
        geojson_data = (
            b'{"type": "FeatureCollection", "features": [{"type": "Feature", '
            b'"properties": {"name": "test"}, "geometry": {"type": "Point", '
            b'"coordinates": [0, 0]}}]}'
        )

        # Mock the getfeature response
        mock_response = MagicMock()
        mock_response.read = MagicMock(return_value=geojson_data)
        mock_wfs_capabilities.getfeature = MagicMock(return_value=mock_response)

        with patch(
            "quarry_connectors.ogc_services.WebFeatureService", return_value=mock_wfs_capabilities
        ):
            with patch("fiona.open") as mock_fiona_open:
                # Mock fiona behavior - first call reads temp geojson, second writes gpkg
                mock_src = MagicMock()
                mock_src.schema = {"geometry": "Point", "properties": {"name": "str"}}
                mock_src.crs = {"init": "epsg:4326"}
                mock_src.__iter__ = MagicMock(return_value=iter([]))
                mock_fiona_open.return_value.__enter__ = MagicMock(return_value=mock_src)
                mock_fiona_open.return_value.__exit__ = MagicMock(return_value=False)

                result = connector.materialize(
                    "wfs::https://example.com/wfs::test_feature",
                    workspace,
                    lazy=False,
                )

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.strategy == "fetched_remote"

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wfs_layer_not_found(self, connector, workspace, mock_wfs_capabilities):
        """Requesting non-existent WFS layer raises MaterializeError."""
        # Set up mock to raise KeyError for missing layer
        mock_wfs_capabilities.__getitem__ = MagicMock(side_effect=KeyError("nonexistent_feature"))

        with patch(
            "quarry_connectors.ogc_services.WebFeatureService", return_value=mock_wfs_capabilities
        ):
            with pytest.raises(MaterializeError) as exc_info:
                connector.materialize(
                    "wfs::https://example.com/wfs::nonexistent_feature",
                    workspace,
                )

        assert "nonexistent_feature" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Discover Tests
# -----------------------------------------------------------------------------


class TestOGCDiscover:
    """Validate OGC service discovery."""

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_discover_wms_layers(self, connector, mock_wms_capabilities):
        """Discover WMS layers from GetCapabilities."""
        with patch(
            "quarry_connectors.ogc_services.WebMapService", return_value=mock_wms_capabilities
        ):
            entries = connector.discover({"url": "https://example.com/wms", "service": "wms"})

        assert len(entries) == 1
        assert entries[0].name == "test_layer"
        assert entries[0].source_ref == "wms::https://example.com/wms::test_layer"
        assert entries[0].description == "Test Layer"
        assert "EPSG:4326" in entries[0].spatial_hint["crs"]

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_discover_wfs_layers(self, connector, mock_wfs_capabilities):
        """Discover WFS layers from GetCapabilities."""
        with patch(
            "quarry_connectors.ogc_services.WebFeatureService", return_value=mock_wfs_capabilities
        ):
            entries = connector.discover({"url": "https://example.com/wfs", "service": "wfs"})

        assert len(entries) == 1
        assert entries[0].name == "test_feature"
        assert entries[0].source_ref == "wfs::https://example.com/wfs::test_feature"

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_discover_requires_url(self, connector):
        """Discover requires URL parameter."""
        with pytest.raises(MaterializeError) as exc_info:
            connector.discover({"service": "wms"})

        assert "url" in str(exc_info.value).lower()

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_discover_requires_service_type(self, connector):
        """Discover requires service type parameter."""
        with pytest.raises(MaterializeError) as exc_info:
            connector.discover({"url": "https://example.com/wms"})

        assert "service type" in str(exc_info.value).lower()


# -----------------------------------------------------------------------------
# Metadata Tests
# -----------------------------------------------------------------------------


class TestOGCMetadata:
    """Validate metadata retrieval without materialization."""

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wms_metadata(self, connector, mock_wms_capabilities):
        """Get WMS layer metadata."""
        with patch(
            "quarry_connectors.ogc_services.WebMapService", return_value=mock_wms_capabilities
        ):
            meta = connector.metadata("wms::https://example.com/wms::test_layer")

        # The mock layer's title is a MagicMock that returns "Test Layer"
        assert str(meta["title"]) == "Test Layer"
        assert str(meta["abstract"]) == "A test layer for unit testing"
        assert "EPSG:4326" in meta["crs_options"]
        assert "image/tiff" in meta["formats"]
        assert "test" in meta["keywords"]

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wfs_metadata(self, connector, mock_wfs_capabilities):
        """Get WFS layer metadata."""
        with patch(
            "quarry_connectors.ogc_services.WebFeatureService", return_value=mock_wfs_capabilities
        ):
            meta = connector.metadata("wfs::https://example.com/wfs::test_feature")

        # The mock feature's title is a MagicMock that returns "Test Feature Type"
        assert str(meta["title"]) == "Test Feature Type"
        assert str(meta["abstract"]) == "A test feature type for unit testing"
        assert "EPSG:4326" in meta["crs_options"]
        assert "application/json" in meta["formats"]


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


class TestOGCErrors:
    """Validate error handling."""

    def test_missing_owslib_materialize(self, connector, workspace):
        """Materialize raises error when owslib is not available."""
        with patch("quarry_connectors.ogc_services.HAS_OWSLIB", False):
            with pytest.raises(MaterializeError) as exc_info:
                connector.materialize(
                    "wms::https://example.com/wms::test_layer",
                    workspace,
                )

        assert "owslib" in str(exc_info.value).lower()
        assert "pip install" in str(exc_info.value).lower()

    def test_missing_owslib_discover(self, connector):
        """Discover raises error when owslib is not available."""
        with patch("quarry_connectors.ogc_services.HAS_OWSLIB", False):
            with pytest.raises(MaterializeError) as exc_info:
                connector.discover({"url": "https://example.com/wms", "service": "wms"})

        assert "owslib" in str(exc_info.value).lower()

    def test_missing_owslib_metadata(self, connector):
        """Metadata raises error when owslib is not available."""
        with patch("quarry_connectors.ogc_services.HAS_OWSLIB", False):
            with pytest.raises(MaterializeError) as exc_info:
                connector.metadata("wms::https://example.com/wms::test_layer")

        assert "owslib" in str(exc_info.value).lower()

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_invalid_service_type_materialize(self, connector, workspace):
        """Invalid service type raises MaterializeError."""
        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize(
                "invalid::https://example.com/service::layer",
                workspace,
            )

        assert "unsupported service type" in str(exc_info.value).lower()

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wms_connection_failure(self, connector, workspace):
        """WMS connection failure raises MaterializeError."""
        with patch(
            "quarry_connectors.ogc_services.WebMapService",
            side_effect=Exception("Connection refused"),
        ):
            with pytest.raises(MaterializeError) as exc_info:
                connector.materialize(
                    "wms::https://example.com/wms::test_layer",
                    workspace,
                )

        assert "failed to connect" in str(exc_info.value).lower()

    @pytest.mark.skipif(not HAS_OWSLIB, reason="owslib not installed")
    def test_wfs_connection_failure(self, connector, workspace):
        """WFS connection failure raises MaterializeError."""
        with patch(
            "quarry_connectors.ogc_services.WebFeatureService",
            side_effect=Exception("Connection refused"),
        ):
            with pytest.raises(MaterializeError) as exc_info:
                connector.materialize(
                    "wfs::https://example.com/wfs::test_feature",
                    workspace,
                )

        assert "failed to connect" in str(exc_info.value).lower()


# -----------------------------------------------------------------------------
# Connector Properties Tests
# -----------------------------------------------------------------------------


class TestOGCConnectorProperties:
    """Validate connector properties."""

    def test_connector_name(self, connector):
        """Connector has correct name."""
        assert connector.name == "ogc_services"

    def test_connector_capabilities(self, connector):
        """Connector declares correct capabilities."""
        from quarry_core.connector import ConnectorCapability

        caps = connector.capabilities
        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps

    def test_connector_constructor_defaults(self):
        """Connector accepts default parameters."""
        auth = {"username": "test", "password": "secret"}
        conn = OGCServicesConnector(
            service_url="https://example.com/wms",
            service_type="wms",
            version="1.3.0",
            auth=auth,
        )
        assert conn._default_service_url == "https://example.com/wms"
        assert conn._default_service_type == "wms"
        assert conn._default_version == "1.3.0"
        assert conn._auth == auth


# -----------------------------------------------------------------------------
# Helper Method Tests
# -----------------------------------------------------------------------------


class TestOGCHelpers:
    """Validate helper methods."""

    def test_select_format_exact_match(self, connector):
        """Select format with exact match."""
        available = ["image/png", "image/tiff", "image/jpeg"]
        preferred = ["image/tiff", "image/png"]
        result = connector._select_format(available, preferred)
        assert result == "image/tiff"

    def test_select_format_partial_match(self, connector):
        """Select format with partial match."""
        available = ["application/json", "GML3", "GML2"]
        preferred = ["application/geo+json", "application/json"]
        result = connector._select_format(available, preferred)
        assert result == "application/json"

    def test_select_format_fallback(self, connector):
        """Select format falls back to first available."""
        available = ["image/png"]
        preferred = ["image/tiff"]
        result = connector._select_format(available, preferred)
        assert result == "image/png"

    def test_select_format_empty(self, connector):
        """Select format with empty available returns None."""
        result = connector._select_format([], ["image/tiff"])
        assert result is None
