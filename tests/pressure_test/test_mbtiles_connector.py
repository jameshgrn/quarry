"""
Pressure test: MBTilesConnector.

Lane: connector

Validates MBTiles file materialization:
- source_ref parsing (local path)
- local eager: wrap in place, LOCAL_FILE backing
- local lazy: metadata only, LAZY_HANDLE backing
- discover: list .mbtiles files in directory
- metadata: read without materializing
- format detection: PNG raster, JPEG raster, PBF vector
- error handling: nonexistent files, invalid SQLite, missing tables
"""

from __future__ import annotations

import sqlite3

import pytest
from quarry_connectors.mbtiles import MBTilesConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import ConnectorCapability, MaterializeError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_mbtiles(path, metadata_dict, tiles):
    """Create an MBTiles file.

    Args:
        path: output path
        metadata_dict: {"name": "test", "format": "png", "bounds": "-180,-85,180,85", ...}
        tiles: list of (zoom, col, row, tile_bytes) tuples
    """
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE metadata (name TEXT, value TEXT)")
    conn.execute(
        "CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, "
        "tile_row INTEGER, tile_data BLOB)"
    )
    conn.execute("CREATE UNIQUE INDEX tile_index ON tiles (zoom_level, tile_column, tile_row)")

    for key, value in metadata_dict.items():
        conn.execute("INSERT INTO metadata VALUES (?, ?)", (key, value))

    for zoom, col, row, data in tiles:
        conn.execute("INSERT INTO tiles VALUES (?, ?, ?, ?)", (zoom, col, row, data))

    conn.commit()
    conn.close()


# Fake tile data with magic bytes for format detection
FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
FAKE_JPEG = b"\xff\xd8" + b"\x00" * 100
FAKE_PBF = b"\x1f\x8b" + b"\x00" * 100  # gzip compressed = vector tile


@pytest.fixture()
def mbtiles_png(tmp_path):
    """Create an MBTiles file with PNG raster tiles."""
    path = tmp_path / "test_raster.mbtiles"

    metadata = {
        "name": "Test Raster Tiles",
        "description": "Test PNG raster tiles",
        "format": "png",
        "bounds": "-180.0,-85.0,180.0,85.0",
        "center": "0.0,0.0,0",
        "minzoom": "0",
        "maxzoom": "5",
        "type": "baselayer",
        "version": "1.0.0",
        "attribution": "Test Attribution",
    }

    tiles = [
        (0, 0, 0, FAKE_PNG),
        (1, 0, 0, FAKE_PNG),
        (1, 1, 0, FAKE_PNG),
        (1, 0, 1, FAKE_PNG),
        (1, 1, 1, FAKE_PNG),
    ]

    _create_mbtiles(path, metadata, tiles)
    return path


@pytest.fixture()
def mbtiles_pbf(tmp_path):
    """Create an MBTiles file with PBF vector tiles."""
    path = tmp_path / "test_vector.mbtiles"

    metadata = {
        "name": "Test Vector Tiles",
        "description": "Test PBF vector tiles",
        "format": "pbf",
        "bounds": "-180.0,-85.0,180.0,85.0",
        "center": "0.0,0.0,0",
        "minzoom": "0",
        "maxzoom": "14",
        "type": "overlay",
        "version": "2.0.0",
    }

    tiles = [
        (0, 0, 0, FAKE_PBF),
        (1, 0, 0, FAKE_PBF),
        (1, 1, 0, FAKE_PBF),
    ]

    _create_mbtiles(path, metadata, tiles)
    return path


@pytest.fixture()
def mbtiles_jpeg(tmp_path):
    """Create an MBTiles file with JPEG raster tiles."""
    path = tmp_path / "test_jpeg.mbtiles"

    metadata = {
        "name": "Test JPEG Tiles",
        "description": "Test JPEG raster tiles",
        "format": "jpeg",
        "bounds": "-90.0,-45.0,90.0,45.0",
        "minzoom": "5",
        "maxzoom": "10",
        "type": "baselayer",
    }

    tiles = [
        (5, 16, 16, FAKE_JPEG),
        (5, 17, 16, FAKE_JPEG),
    ]

    _create_mbtiles(path, metadata, tiles)
    return path


@pytest.fixture()
def mbtiles_format_from_tiles(tmp_path):
    """Create MBTiles without format in metadata - format detected from tiles."""
    path = tmp_path / "test_detected_format.mbtiles"

    metadata = {
        "name": "Detected Format",
        "bounds": "-180.0,-85.0,180.0,85.0",
        # No format key - will be detected from tile data
    }

    tiles = [
        (0, 0, 0, FAKE_PNG),
    ]

    _create_mbtiles(path, metadata, tiles)
    return path


@pytest.fixture()
def directory_with_mbtiles(tmp_path, mbtiles_png, mbtiles_pbf):
    """Create a directory with multiple .mbtiles files."""
    # Files are already in tmp_path from fixtures
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Local Materialization
# ---------------------------------------------------------------------------


class TestMBTilesEagerLocal:
    """Validate eager materialization of local MBTiles files."""

    def test_eager_png_produces_raster(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path)

        assert result.artifact.type == ArtifactType.RASTER

    def test_eager_jpeg_produces_raster(self, mbtiles_jpeg, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_jpeg), tmp_path)

        assert result.artifact.type == ArtifactType.RASTER

    def test_eager_pbf_produces_vector(self, mbtiles_pbf, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_pbf), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_eager_produces_local_file_backing(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".mbtiles")

    def test_eager_wrapped_local_strategy(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_crs_is_web_mercator(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path)

        assert result.artifact.spatial.crs == "EPSG:3857"

    def test_eager_extent_from_bounds(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(-180.0)
        assert ymin == pytest.approx(-85.0)
        assert xmax == pytest.approx(180.0)
        assert ymax == pytest.approx(85.0)

    def test_eager_content_hash_present(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path)

        assert result.artifact.lineage.params["source"] == "mbtiles"
        assert result.artifact.lineage.params["path"] == str(mbtiles_png)
        assert result.artifact.lineage.params["lazy"] is False
        assert result.artifact.lineage.params["format"] == "png"

    def test_eager_metadata(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path)

        assert result.artifact.metadata["mbtiles_name"] == "Test Raster Tiles"
        assert result.artifact.metadata["mbtiles_description"] == "Test PNG raster tiles"
        assert result.artifact.metadata["format"] == "png"
        assert result.artifact.metadata["tile_count"] == 5
        assert result.artifact.metadata["minzoom"] == 0
        assert result.artifact.metadata["maxzoom"] == 5
        assert result.artifact.metadata["type"] == "baselayer"
        assert result.artifact.metadata["version"] == "1.0.0"
        assert result.artifact.metadata["attribution"] == "Test Attribution"

    def test_eager_name_from_filename(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path)

        assert result.artifact.name == "test_raster"

    def test_format_detected_from_tiles(self, mbtiles_format_from_tiles, tmp_path):
        """When format not in metadata, detect from tile magic bytes."""
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_format_from_tiles), tmp_path)

        assert result.artifact.metadata["format"] == "png"
        assert result.artifact.type == ArtifactType.RASTER


# ---------------------------------------------------------------------------
# Lazy Local Materialization
# ---------------------------------------------------------------------------


class TestMBTilesLazyLocal:
    """Validate lazy (metadata-only) materialization of local MBTiles files."""

    def test_lazy_backing_kind(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_crs(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path, lazy=True)

        assert result.artifact.spatial.crs == "EPSG:3857"

    def test_lazy_extent(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path, lazy=True)

        extent = result.artifact.spatial.extent
        assert extent is not None
        assert extent[0] == pytest.approx(-180.0)

    def test_lazy_backing_uri(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path, lazy=True)

        assert result.artifact.backing.uri == str(mbtiles_png)

    def test_lazy_lineage(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True
        assert result.artifact.lineage.params["format"] == "png"

    def test_lazy_tile_count(self, mbtiles_png, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_png), tmp_path, lazy=True)

        assert result.artifact.metadata["tile_count"] == 5

    def test_lazy_vector_type(self, mbtiles_pbf, tmp_path):
        connector = MBTilesConnector()
        result = connector.materialize(str(mbtiles_pbf), tmp_path, lazy=True)

        assert result.artifact.type == ArtifactType.VECTOR


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMBTilesMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_name(self, mbtiles_png):
        connector = MBTilesConnector()
        meta = connector.metadata(str(mbtiles_png))

        assert meta["name"] == "Test Raster Tiles"

    def test_metadata_returns_description(self, mbtiles_png):
        connector = MBTilesConnector()
        meta = connector.metadata(str(mbtiles_png))

        assert meta["description"] == "Test PNG raster tiles"

    def test_metadata_returns_format(self, mbtiles_png):
        connector = MBTilesConnector()
        meta = connector.metadata(str(mbtiles_png))

        assert meta["format"] == "png"

    def test_metadata_returns_bounds(self, mbtiles_png):
        connector = MBTilesConnector()
        meta = connector.metadata(str(mbtiles_png))

        bounds = meta["bounds"]
        assert bounds is not None
        assert len(bounds) == 4
        assert bounds[0] == pytest.approx(-180.0)

    def test_metadata_returns_center(self, mbtiles_png):
        connector = MBTilesConnector()
        meta = connector.metadata(str(mbtiles_png))

        center = meta["center"]
        assert center is not None
        assert len(center) == 2
        assert center[0] == pytest.approx(0.0)
        assert center[1] == pytest.approx(0.0)

    def test_metadata_returns_zoom_levels(self, mbtiles_png):
        connector = MBTilesConnector()
        meta = connector.metadata(str(mbtiles_png))

        assert meta["minzoom"] == 0
        assert meta["maxzoom"] == 5

    def test_metadata_returns_tile_count(self, mbtiles_png):
        connector = MBTilesConnector()
        meta = connector.metadata(str(mbtiles_png))

        assert meta["tile_count"] == 5

    def test_metadata_returns_type(self, mbtiles_png):
        connector = MBTilesConnector()
        meta = connector.metadata(str(mbtiles_png))

        assert meta["type"] == "baselayer"

    def test_metadata_returns_version(self, mbtiles_png):
        connector = MBTilesConnector()
        meta = connector.metadata(str(mbtiles_png))

        assert meta["version"] == "1.0.0"

    def test_metadata_returns_attribution(self, mbtiles_png):
        connector = MBTilesConnector()
        meta = connector.metadata(str(mbtiles_png))

        assert meta["attribution"] == "Test Attribution"

    def test_metadata_vector_format(self, mbtiles_pbf):
        connector = MBTilesConnector()
        meta = connector.metadata(str(mbtiles_pbf))

        assert meta["format"] == "pbf"
        assert meta["type"] == "overlay"


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestMBTilesDiscover:
    """Validate file discovery."""

    def test_discover_lists_mbtiles_files(self, directory_with_mbtiles):
        connector = MBTilesConnector()
        entries = connector.discover(str(directory_with_mbtiles))

        names = {e.name for e in entries}
        assert "test_raster" in names
        assert "test_vector" in names

    def test_discover_source_refs(self, directory_with_mbtiles):
        connector = MBTilesConnector()
        entries = connector.discover(str(directory_with_mbtiles))

        for entry in entries:
            assert entry.source_ref.endswith(".mbtiles") or entry.source_ref.endswith(".MBTILES")

    def test_discover_with_dict_query(self, directory_with_mbtiles):
        connector = MBTilesConnector()
        entries = connector.discover({"path": str(directory_with_mbtiles)})

        assert len(entries) >= 2

    def test_discover_empty_directory(self, tmp_path):
        connector = MBTilesConnector()
        entries = connector.discover(str(tmp_path))

        assert entries == []

    def test_discover_nonexistent_directory_raises(self, tmp_path):
        connector = MBTilesConnector()
        nonexistent = tmp_path / "no_such_dir"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))

    def test_discover_file_raises(self, mbtiles_png):
        connector = MBTilesConnector()

        with pytest.raises(MaterializeError):
            connector.discover(str(mbtiles_png))


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestMBTilesErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path):
        connector = MBTilesConnector()
        nonexistent = tmp_path / "does_not_exist.mbtiles"

        with pytest.raises(MaterializeError):
            connector.materialize(str(nonexistent), tmp_path)

    def test_not_sqlite_raises(self, tmp_path):
        connector = MBTilesConnector()
        bad_file = tmp_path / "not_mbtiles.mbtiles"
        bad_file.write_text("this is not a valid sqlite file")

        with pytest.raises(MaterializeError):
            connector.materialize(str(bad_file), tmp_path)

    def test_missing_metadata_table_raises(self, tmp_path):
        connector = MBTilesConnector()
        bad_file = tmp_path / "no_metadata.mbtiles"

        # Create SQLite file without metadata table
        conn = sqlite3.connect(str(bad_file))
        conn.execute(
            "CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, "
            "tile_row INTEGER, tile_data BLOB)"
        )
        conn.commit()
        conn.close()

        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize(str(bad_file), tmp_path)
        assert "metadata" in str(exc_info.value).lower()

    def test_missing_tiles_table_raises(self, tmp_path):
        connector = MBTilesConnector()
        bad_file = tmp_path / "no_tiles.mbtiles"

        # Create SQLite file without tiles table
        conn = sqlite3.connect(str(bad_file))
        conn.execute("CREATE TABLE metadata (name TEXT, value TEXT)")
        conn.commit()
        conn.close()

        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize(str(bad_file), tmp_path)
        assert "tiles" in str(exc_info.value).lower()

    def test_metadata_nonexistent_file_raises(self, tmp_path):
        connector = MBTilesConnector()
        nonexistent = tmp_path / "does_not_exist.mbtiles"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))

    def test_metadata_not_sqlite_raises(self, tmp_path):
        connector = MBTilesConnector()
        bad_file = tmp_path / "not_mbtiles.mbtiles"
        bad_file.write_text("this is not a valid sqlite file")

        with pytest.raises(MaterializeError):
            connector.metadata(str(bad_file))

    def test_discover_no_path_raises(self):
        connector = MBTilesConnector()

        with pytest.raises(MaterializeError):
            connector.discover()


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestMBTilesCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        connector = MBTilesConnector()
        assert connector.name == "mbtiles"

    def test_capabilities(self):
        connector = MBTilesConnector()
        caps = connector.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps
