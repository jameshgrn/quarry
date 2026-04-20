"""COG connector pressure test.

Adversarial stress points:
1. source_ref is a plain URI/path — trivial parsing, but 4th connector shape
2. Remote vs local branching: /vsicurl/ for HTTP, direct for local
3. COG validation: is it actually cloud-optimized? (overviews, tiling)
4. I/O metrics in artifact metadata (bytes transferred, read count)
5. Overlap with LocalFileConnector — when do you use which?
6. Lazy = header-only metadata via GDAL virtual fs, no download
7. Eager = full download for remote, wrap-in-place for local
8. SpatialDescriptor populated from COG internal metadata
9. Lineage captures source type (remote/local), I/O metrics

All tests use real local GeoTIFFs (created in fixtures) — no network mocking.
"""

import numpy as np
import pytest
import rasterio
from quarry_connectors.cog import COGConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import Connector, ConnectorCapability, MaterializeError
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Fixtures: create real COG and non-COG GeoTIFFs
# ---------------------------------------------------------------------------


@pytest.fixture
def cog_path(tmp_path):
    """Create a valid Cloud-Optimized GeoTIFF (tiled, with overviews)."""
    path = tmp_path / "valid_cog.tif"
    h, w = 256, 256
    data = np.random.default_rng(42).integers(0, 1000, (h, w), dtype=np.int16)
    transform = from_bounds(-90.0, 30.0, -89.0, 31.0, w, h)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=2,
        dtype="int16",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
        tiled=True,
        blockxsize=128,
        blockysize=128,
    ) as dst:
        dst.write(data, 1)
        dst.write(data * 2, 2)
        dst.build_overviews([2, 4], rasterio.enums.Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")

    return path


@pytest.fixture
def non_cog_path(tmp_path):
    """Create a regular (non-tiled, no overviews) GeoTIFF."""
    path = tmp_path / "not_a_cog.tif"
    h, w = 64, 64
    data = np.ones((h, w), dtype=np.float32)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs="EPSG:32617",
        transform=from_bounds(500000, 3300000, 501000, 3301000, w, h),
    ) as dst:
        dst.write(data, 1)

    return path


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_connector_protocol(self):
        conn = COGConnector()
        assert isinstance(conn, Connector)

    def test_capabilities(self):
        conn = COGConnector()
        caps = conn.capabilities
        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps
        # COGs are individual files, no catalog to discover
        assert ConnectorCapability.DISCOVER not in caps

    def test_name(self):
        conn = COGConnector()
        assert conn.name == "cog"


# ---------------------------------------------------------------------------
# source_ref parsing
# ---------------------------------------------------------------------------


class TestSourceRef:
    """source_ref for COG is just a URI — trivial, but confirms the 4th shape."""

    def test_local_path(self, cog_path, tmp_path):
        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=True)
        assert result.source_ref == str(cog_path)

    def test_http_url_shape(self):
        """Verify connector accepts URL-shaped source_refs without network call."""
        conn = COGConnector()
        # Just test parsing — actual materialization would need network
        assert conn._classify_source("https://storage.googleapis.com/bucket/dem.tif") == "remote"
        assert conn._classify_source("s3://mybucket/raster.tif") == "remote"
        assert conn._classify_source("/tmp/local.tif") == "local"

    def test_source_ref_observation(self):
        """Document: source_ref for COG is trivially a URI.

        Compare across connectors:
        - LocalFile: path
        - STAC: collection/item::asset (convention separators)
        - PostGIS: schema.table or SELECT query (heuristic prefix-sniffing)
        - COG: plain URI/path

        Four shapes, all str. The strain is not in parsing each one,
        but in *choosing which connector to use for a given source_ref*.
        A local .tif could go to either LocalFile or COG connector.
        """
        conn = COGConnector()
        assert conn._classify_source("/data/dem.tif") == "local"
        # Same path would work in LocalFileConnector too — connector selection is the strain


# ---------------------------------------------------------------------------
# Lazy materialization (header-only)
# ---------------------------------------------------------------------------


class TestLazyMaterialization:
    """Lazy = full metadata from headers, no data copied."""

    def test_lazy_local_cog(self, cog_path, tmp_path):
        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=True)

        assert result.artifact.type == ArtifactType.RASTER
        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE
        assert result.strategy == "lazy_handle"

    def test_lazy_has_full_spatial(self, cog_path, tmp_path):
        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=True)
        spatial = result.artifact.spatial

        assert spatial.crs == "EPSG:4326"
        assert spatial.extent is not None
        assert spatial.extent[0] == pytest.approx(-90.0, abs=0.01)
        assert spatial.band_count == 2
        assert spatial.resolution is not None

    def test_lazy_has_cog_metadata(self, cog_path, tmp_path):
        """Lazy artifacts carry COG-specific metadata."""
        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=True)
        meta = result.artifact.metadata

        assert meta["is_cog"] is True
        assert meta["block_size"] == (128, 128)
        assert meta["overview_levels"] == [2, 4]
        assert "compression" in meta  # may be None if uncompressed
        assert meta["driver"] == "GTiff"


# ---------------------------------------------------------------------------
# Eager materialization
# ---------------------------------------------------------------------------


class TestEagerMaterialization:
    """Eager for local = wrap in place (no copy). Eager for remote = download."""

    def test_eager_local_wraps_in_place(self, cog_path, tmp_path):
        """Local COG: eager wraps the file (doesn't copy)."""
        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=False)

        assert result.artifact.type == ArtifactType.RASTER
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        # For local files, uri points to original path (wrap, not copy)
        assert result.artifact.backing.uri == str(cog_path)
        assert result.strategy == "wrapped_local"

    def test_eager_has_content_hash(self, cog_path, tmp_path):
        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=False)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256

    def test_eager_has_size_bytes(self, cog_path, tmp_path):
        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=False)

        assert result.artifact.backing.size_bytes > 0
        assert result.artifact.backing.size_bytes == cog_path.stat().st_size


# ---------------------------------------------------------------------------
# COG validation
# ---------------------------------------------------------------------------


class TestCOGValidation:
    """The connector can distinguish real COGs from regular GeoTIFFs."""

    def test_valid_cog_detected(self, cog_path, tmp_path):
        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=True)
        assert result.artifact.metadata["is_cog"] is True

    def test_non_cog_detected(self, non_cog_path, tmp_path):
        """Non-tiled GeoTIFF is flagged as not COG-compliant."""
        conn = COGConnector()
        result = conn.materialize(str(non_cog_path), tmp_path, lazy=True)
        assert result.artifact.metadata["is_cog"] is False

    def test_non_cog_still_materializes(self, non_cog_path, tmp_path):
        """Non-COG files still materialize — connector doesn't reject them."""
        conn = COGConnector()
        result = conn.materialize(str(non_cog_path), tmp_path, lazy=False)
        assert result.artifact.type == ArtifactType.RASTER
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE

    def test_validate_mode_rejects_non_cog(self, non_cog_path, tmp_path):
        """Strict mode rejects non-COG files."""
        conn = COGConnector(strict_cog=True)
        with pytest.raises(MaterializeError, match="not a valid COG"):
            conn.materialize(str(non_cog_path), tmp_path)


# ---------------------------------------------------------------------------
# I/O metrics (raided pattern from Hydrops)
# ---------------------------------------------------------------------------


class TestIOMetrics:
    """I/O accounting in artifact metadata — bytes read, efficiency hints."""

    def test_metadata_includes_io_stats(self, cog_path, tmp_path):
        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=False)
        meta = result.artifact.metadata

        assert "size_bytes" in meta
        assert meta["size_bytes"] > 0

    def test_lazy_reports_zero_data_transfer(self, cog_path, tmp_path):
        """Lazy materialization transfers header bytes only."""
        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=True)
        lineage = result.artifact.lineage.params

        assert lineage["lazy"] is True
        assert lineage["data_transferred"] == 0


# ---------------------------------------------------------------------------
# Lineage provenance
# ---------------------------------------------------------------------------


class TestLineageProvenance:
    def test_lineage_captures_source_info(self, cog_path, tmp_path):
        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=True)
        params = result.artifact.lineage.params

        assert params["source"] == "cog"
        assert params["source_type"] == "local"
        assert params["source_ref"] == str(cog_path)

    def test_lineage_captures_cog_structure(self, cog_path, tmp_path):
        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=True)
        params = result.artifact.lineage.params

        assert "overview_levels" in params
        assert "block_size" in params


# ---------------------------------------------------------------------------
# Metadata inspection
# ---------------------------------------------------------------------------


class TestMetadataInspection:
    def test_metadata_returns_cog_details(self, cog_path):
        conn = COGConnector()
        meta = conn.metadata(str(cog_path))

        assert isinstance(meta, dict)
        assert meta["crs"] == "EPSG:4326"
        assert meta["band_count"] == 2
        assert meta["is_cog"] is True
        assert meta["block_size"] == (128, 128)
        assert meta["overview_levels"] == [2, 4]
        assert meta["dtype"] == "int16"
        assert "extent" in meta
        assert "resolution" in meta


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_nonexistent_file_raises(self, tmp_path):
        conn = COGConnector()
        with pytest.raises(MaterializeError, match="not found"):
            conn.materialize("/nonexistent/path.tif", tmp_path)

    def test_non_raster_file_raises(self, tmp_path):
        """A text file with .tif extension should fail."""
        fake = tmp_path / "fake.tif"
        fake.write_text("not a raster")
        conn = COGConnector()
        with pytest.raises(MaterializeError):
            conn.materialize(str(fake), tmp_path)


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------


class TestRegistryRoundTrip:
    def test_cog_artifact_persists_in_registry(self, cog_path, tmp_path):
        from quarry_registry.registry import Registry

        conn = COGConnector()
        result = conn.materialize(str(cog_path), tmp_path, lazy=True)

        registry = Registry(tmp_path / "test.duckdb")
        registry.save_artifact(result.artifact)

        loaded = registry.get_artifact(result.artifact.id)
        assert loaded is not None
        assert loaded.type == ArtifactType.RASTER
        assert loaded.backing.kind == BackingStoreKind.LAZY_HANDLE
        assert loaded.spatial.crs == "EPSG:4326"
        assert loaded.spatial.band_count == 2
