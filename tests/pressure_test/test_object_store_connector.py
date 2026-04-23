"""
Pressure test: ObjectStoreConnector.

Lane: connector

Validates cloud object storage materialization using GDAL virtual filesystems:
- Source ref parsing (s3://, gs://, az://, https://)
- VSI path mapping (/vsis3/, /vsigs/, /vsiaz/, /vsicurl/)
- File type detection by extension
- Lazy materialization (metadata-only via /vsi* paths)
- Eager materialization (download to workspace)
- Local file fallback for testing
- Error handling for unsupported schemes and missing files

Since we can't hit real S3/GCS in tests, we test:
1. Parsing and VSI mapping logic directly
2. Local file materialization as fallback
3. Error cases
"""

from __future__ import annotations

import pytest
from quarry_connectors.object_store import ObjectStoreConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def connector():
    """Create an ObjectStoreConnector instance."""
    return ObjectStoreConnector()


@pytest.fixture()
def connector_with_creds():
    """Create an ObjectStoreConnector with mock credentials."""
    return ObjectStoreConnector(
        credentials={
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "region": "us-east-1",
        }
    )


@pytest.fixture()
def sample_geotiff(tmp_path):
    """Create a sample GeoTIFF file for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / "sample.tif"
    transform = from_bounds(-122.5, 37.5, -122.0, 38.0, 100, 100)

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
        import numpy as np

        dst.write(np.full((100, 100), 1.0, dtype="float32"), 1)

    return path


@pytest.fixture()
def sample_geopackage(tmp_path):
    """Create a sample GeoPackage file for testing."""
    import fiona
    from fiona.crs import CRS

    path = tmp_path / "sample.gpkg"

    schema = {
        "geometry": "Point",
        "properties": {"id": "int", "name": "str"},
    }

    with fiona.open(
        str(path),
        "w",
        driver="GPKG",
        schema=schema,
        crs=CRS.from_epsg(4326),
    ) as dst:
        dst.write(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-122.4, 37.8]},
                "properties": {"id": 1, "name": "test"},
            }
        )

    return path


# ---------------------------------------------------------------------------
# Source Ref Parsing Tests
# ---------------------------------------------------------------------------


class TestObjectStoreSourceRefParsing:
    """Validate source_ref parsing for various URL schemes."""

    def test_parse_s3_url(self, connector):
        """Parse s3://bucket/key.tif format."""
        parsed = connector._parse_source_ref("s3://my-bucket/data/raster.tif")

        assert parsed.scheme == "s3"
        assert parsed.bucket == "my-bucket"
        assert parsed.path == "data/raster.tif"
        assert parsed.vsi_path == "/vsis3/my-bucket/data/raster.tif"

    def test_parse_gs_url(self, connector):
        """Parse gs://bucket/blob.gpkg format."""
        parsed = connector._parse_source_ref("gs://my-project/vectors/data.gpkg")

        assert parsed.scheme == "gs"
        assert parsed.bucket == "my-project"
        assert parsed.path == "vectors/data.gpkg"
        assert parsed.vsi_path == "/vsigs/my-project/vectors/data.gpkg"

    def test_parse_az_url(self, connector):
        """Parse az://container/blob.tif format."""
        parsed = connector._parse_source_ref("az://my-container/rasters/image.tif")

        assert parsed.scheme == "az"
        assert parsed.bucket == "my-container"
        assert parsed.path == "rasters/image.tif"
        assert parsed.vsi_path == "/vsiaz/my-container/rasters/image.tif"

    def test_parse_https_url(self, connector):
        """Parse https://example.com/data.tif format."""
        parsed = connector._parse_source_ref("https://example.com/geodata/image.tif")

        assert parsed.scheme == "https"
        assert parsed.bucket is None
        assert parsed.path == "https://example.com/geodata/image.tif"
        assert parsed.vsi_path == "/vsicurl/https://example.com/geodata/image.tif"

    def test_parse_http_url(self, connector):
        """Parse http://example.com/data.tif format."""
        parsed = connector._parse_source_ref("http://example.com/geodata/image.tif")

        assert parsed.scheme == "http"
        assert parsed.vsi_path == "/vsicurl/http://example.com/geodata/image.tif"

    def test_parse_file_url(self, connector):
        """Parse file:///path/to/data.tif format."""
        parsed = connector._parse_source_ref("file:///data/raster.tif")

        assert parsed.scheme == "file"
        assert parsed.bucket is None
        assert parsed.path == "/data/raster.tif"

    def test_parse_absolute_path(self, connector):
        """Parse absolute path as fallback."""
        parsed = connector._parse_source_ref("/data/raster.tif")

        assert parsed.scheme == "file"
        assert parsed.path == "/data/raster.tif"

    def test_parse_with_sourceref(self, connector):
        """Parse from SourceRef object."""
        ref = SourceRef.uri("s3://bucket/data.tif")
        parsed = connector._parse_source_ref(ref)

        assert parsed.scheme == "s3"
        assert parsed.bucket == "bucket"


# ---------------------------------------------------------------------------
# VSI Path Mapping Tests
# ---------------------------------------------------------------------------


class TestObjectStoreVsiMapping:
    """Validate GDAL virtual filesystem path mapping."""

    def test_s3_vsi_mapping(self, connector):
        """s3://bucket/key → /vsis3/bucket/key"""
        parsed = connector._parse_source_ref("s3://landsat-pds/L8/001/002/LC08.tif")
        assert parsed.vsi_path == "/vsis3/landsat-pds/L8/001/002/LC08.tif"

    def test_gs_vsi_mapping(self, connector):
        """gs://bucket/blob → /vsigs/bucket/blob"""
        parsed = connector._parse_source_ref("gs://gcp-public-data-landsat/L8/001/002/LC08.tif")
        assert parsed.vsi_path == "/vsigs/gcp-public-data-landsat/L8/001/002/LC08.tif"

    def test_az_vsi_mapping(self, connector):
        """az://container/blob → /vsiaz/container/blob"""
        parsed = connector._parse_source_ref("az://mycontainer/data/image.tif")
        assert parsed.vsi_path == "/vsiaz/mycontainer/data/image.tif"

    def test_https_vsi_mapping(self, connector):
        """https:// → /vsicurl/https://..."""
        parsed = connector._parse_source_ref("https://example.com/data.tif")
        assert parsed.vsi_path.startswith("/vsicurl/")
        assert "https://example.com/data.tif" in parsed.vsi_path

    def test_nested_path_vsi_mapping(self, connector):
        """Deeply nested paths map correctly."""
        parsed = connector._parse_source_ref("s3://bucket/a/b/c/d/e/final.tif")
        assert parsed.vsi_path == "/vsis3/bucket/a/b/c/d/e/final.tif"

    def test_path_with_special_chars(self, connector):
        """Paths with special characters map correctly."""
        parsed = connector._parse_source_ref("s3://bucket/data/file_name-v1.2.tif")
        assert parsed.vsi_path == "/vsis3/bucket/data/file_name-v1.2.tif"


# ---------------------------------------------------------------------------
# File Type Detection Tests
# ---------------------------------------------------------------------------


class TestObjectStoreFileTypeDetection:
    """Validate file type detection by extension."""

    def test_detect_raster_tif(self, connector):
        assert connector._detect_file_type("data.tif") == "raster"
        assert connector._detect_file_type("data.TIF") == "raster"

    def test_detect_raster_tiff(self, connector):
        assert connector._detect_file_type("data.tiff") == "raster"

    def test_detect_raster_geotiff(self, connector):
        assert connector._detect_file_type("data.geotiff") == "raster"

    def test_detect_raster_jp2(self, connector):
        assert connector._detect_file_type("data.jp2") == "raster"

    def test_detect_vector_shp(self, connector):
        assert connector._detect_file_type("data.shp") == "vector"

    def test_detect_vector_geojson(self, connector):
        assert connector._detect_file_type("data.geojson") == "vector"

    def test_detect_vector_gpkg(self, connector):
        assert connector._detect_file_type("data.gpkg") == "vector"

    def test_detect_vector_parquet(self, connector):
        assert connector._detect_file_type("data.parquet") == "vector"

    def test_detect_vector_geoparquet(self, connector):
        assert connector._detect_file_type("data.geoparquet") == "vector"

    def test_detect_table_csv(self, connector):
        assert connector._detect_file_type("data.csv") == "table"

    def test_detect_unknown_extension(self, connector):
        assert connector._detect_file_type("data.unknown") == "unknown"
        assert connector._detect_file_type("data") == "unknown"


# ---------------------------------------------------------------------------
# Local File Fallback Tests
# ---------------------------------------------------------------------------


class TestObjectStoreLocalFallback:
    """Validate local file materialization as fallback for testing."""

    def test_local_geotiff_eager_materialization(self, connector, sample_geotiff, tmp_path):
        """Materialize a local GeoTIFF file eagerly."""
        result = connector.materialize(str(sample_geotiff), tmp_path, lazy=False)

        assert result.artifact.type == ArtifactType.RASTER
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.strategy == "wrapped_local"

    def test_local_geotiff_lazy_materialization(self, connector, sample_geotiff, tmp_path):
        """Materialize a local GeoTIFF file lazily."""
        result = connector.materialize(str(sample_geotiff), tmp_path, lazy=True)

        assert result.artifact.type == ArtifactType.RASTER
        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE
        assert result.strategy == "lazy_handle"

    def test_local_geopackage_eager_materialization(self, connector, sample_geopackage, tmp_path):
        """Materialize a local GeoPackage file eagerly."""
        result = connector.materialize(str(sample_geopackage), tmp_path, lazy=False)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE

    def test_local_geotiff_metadata(self, connector, sample_geotiff):
        """Get metadata for a local GeoTIFF."""
        meta = connector.metadata(str(sample_geotiff))

        assert meta["file_type"] == "raster"
        assert meta["crs"] is not None
        assert meta["extent"] is not None
        assert meta["band_count"] == 1

    def test_local_geopackage_metadata(self, connector, sample_geopackage):
        """Get metadata for a local GeoPackage."""
        meta = connector.metadata(str(sample_geopackage))

        assert meta["file_type"] == "vector"
        assert meta["crs"] is not None
        assert meta["extent"] is not None
        assert meta["feature_count"] == 1

    def test_lazy_artifact_has_vsi_uri(self, connector, sample_geotiff, tmp_path):
        """Lazy artifact backing URI should be a VSI-style path."""
        result = connector.materialize(str(sample_geotiff), tmp_path, lazy=True)

        assert result.artifact.backing.uri == str(sample_geotiff)

    def test_eager_artifact_has_content_hash(self, connector, sample_geotiff, tmp_path):
        """Eager artifact should have content hash."""
        result = connector.materialize(str(sample_geotiff), tmp_path, lazy=False)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_artifact_has_size(self, connector, sample_geotiff, tmp_path):
        """Eager artifact should have size_bytes."""
        result = connector.materialize(str(sample_geotiff), tmp_path, lazy=False)

        assert result.artifact.backing.size_bytes is not None
        assert result.artifact.backing.size_bytes > 0


# ---------------------------------------------------------------------------
# Lineage and Metadata Tests
# ---------------------------------------------------------------------------


class TestObjectStoreLineage:
    """Validate lineage and metadata in artifacts."""

    def test_lineage_records_source(self, connector, sample_geotiff, tmp_path):
        result = connector.materialize(str(sample_geotiff), tmp_path, lazy=True)

        assert result.artifact.lineage.params["source"] == "object_store"

    def test_lineage_records_scheme(self, connector, sample_geotiff, tmp_path):
        result = connector.materialize(str(sample_geotiff), tmp_path, lazy=True)

        assert result.artifact.lineage.params["scheme"] == "file"

    def test_lineage_records_lazy_flag(self, connector, sample_geotiff, tmp_path):
        result_lazy = connector.materialize(str(sample_geotiff), tmp_path, lazy=True)
        result_eager = connector.materialize(str(sample_geotiff), tmp_path, lazy=False)

        assert result_lazy.artifact.lineage.params["lazy"] is True
        assert result_eager.artifact.lineage.params["lazy"] is False

    def test_lineage_records_file_type(self, connector, sample_geotiff, tmp_path):
        result = connector.materialize(str(sample_geotiff), tmp_path, lazy=True)

        assert result.artifact.lineage.params["file_type"] == "raster"

    def test_metadata_includes_driver(self, connector, sample_geotiff):
        meta = connector.metadata(str(sample_geotiff))

        assert meta["driver"] is not None


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestObjectStoreErrors:
    """Validate error cases."""

    def test_unsupported_scheme_raises(self, connector, tmp_path):
        """Unsupported URL scheme should raise MaterializeError."""
        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize("ftp://example.com/data.tif", tmp_path)

        assert "Unsupported scheme" in str(exc_info.value)

    def test_nonexistent_local_file_raises(self, connector, tmp_path):
        """Non-existent local file should raise MaterializeError."""
        with pytest.raises(MaterializeError):
            connector.materialize("/nonexistent/path/file.tif", tmp_path)

    def test_discover_raises_not_implemented(self, connector):
        """Discover should raise explaining it needs boto3."""
        with pytest.raises(MaterializeError) as exc_info:
            connector.discover()

        assert "boto3" in str(exc_info.value).lower() or "Discover" in str(exc_info.value)

    def test_parse_invalid_url(self, connector):
        """Invalid URL format should raise MaterializeError."""
        with pytest.raises(MaterializeError):
            connector._parse_source_ref("not-a-valid-url")


# ---------------------------------------------------------------------------
# Credentials/Auth Tests
# ---------------------------------------------------------------------------


class TestObjectStoreAuth:
    """Validate authentication context manager."""

    def test_auth_context_sets_env_vars(self, connector_with_creds):
        import os

        # Before context
        assert (
            os.environ.get("AWS_ACCESS_KEY_ID") is None
            or os.environ.get("AWS_ACCESS_KEY_ID") != "AKIAIOSFODNN7EXAMPLE"
        )

        with connector_with_creds._auth_context():
            # Inside context
            assert os.environ.get("AWS_ACCESS_KEY_ID") == "AKIAIOSFODNN7EXAMPLE"
            assert (
                os.environ.get("AWS_SECRET_ACCESS_KEY")
                == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
            )
            assert os.environ.get("AWS_REGION") == "us-east-1"

        # After context - should be restored
        # Note: if these were set in the environment before, they would be restored
        # We just verify the context manager exits cleanly

    def test_no_credentials_no_env_changes(self, connector):
        import os

        orig_aws_key = os.environ.get("AWS_ACCESS_KEY_ID")

        with connector._auth_context():
            # No changes expected
            assert os.environ.get("AWS_ACCESS_KEY_ID") == orig_aws_key


# ---------------------------------------------------------------------------
# Connector Properties Tests
# ---------------------------------------------------------------------------


class TestObjectStoreConnectorProperties:
    """Validate connector properties."""

    def test_name(self, connector):
        assert connector.name == "object_store"

    def test_capabilities(self, connector):
        caps = connector.capabilities

        from quarry_core.connector import ConnectorCapability

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps
        assert ConnectorCapability.DISCOVER not in caps


# ---------------------------------------------------------------------------
# Artifact Name Derivation Tests
# ---------------------------------------------------------------------------


class TestObjectStoreNameDerivation:
    """Validate artifact name derivation from paths."""

    def test_derive_name_simple(self, connector):
        assert connector._derive_name("data.tif") == "data"

    def test_derive_name_with_path(self, connector):
        assert connector._derive_name("/path/to/data.tif") == "data"

    def test_derive_name_nested(self, connector):
        assert connector._derive_name("s3://bucket/a/b/c/data.tif") == "data"

    def test_get_extension_simple(self, connector):
        assert connector._get_extension("data.tif") == ".tif"

    def test_get_extension_no_extension(self, connector):
        assert connector._get_extension("data") == ".bin"


# ---------------------------------------------------------------------------
# Integration with SourceRef
# ---------------------------------------------------------------------------


class TestObjectStoreSourceRefIntegration:
    """Validate integration with SourceRef system."""

    def test_s3_sourceref_uri(self, connector):
        ref = SourceRef.uri("s3://bucket/data.tif")
        parsed = connector._parse_source_ref(ref)

        assert parsed.scheme == "s3"
        assert parsed.bucket == "bucket"

    def test_https_sourceref_uri(self, connector):
        ref = SourceRef.uri("https://example.com/data.tif")
        parsed = connector._parse_source_ref(ref)

        assert parsed.scheme == "https"
        assert parsed.vsi_path.startswith("/vsicurl/")
