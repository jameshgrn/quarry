"""
Pressure test: GeoParquetConnector.

Lane: connector

Validates GeoParquet file materialization following the DuckDB connector pattern:
- source_ref parsing (path string, SourceRef.local())
- geometry vs non-geometry branching → VECTOR vs TABLE
- lazy = metadata-only with LAZY_HANDLE backing
- eager = dump to GeoPackage (spatial) or CSV (non-spatial)
- discover: list parquet files in directory
- metadata: read parquet metadata without materializing
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
import shapely.io
from quarry_connectors.geoparquet import GeoParquetConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_geoparquet_with_pyarrow(path, geometries, properties, crs="EPSG:4326"):
    """Create a GeoParquet file using pyarrow with geo metadata.

    Args:
        path: Output file path
        geometries: List of shapely geometries
        properties: Dict of column names to lists of values
        crs: CRS string
    """
    # Encode geometries as WKB
    wkb_values = [shapely.io.to_wkb(g) for g in geometries]

    # Build arrow arrays
    arrays = {"geometry": pa.array(wkb_values, type=pa.binary())}
    for col_name, values in properties.items():
        arrays[col_name] = pa.array(values)

    # Create table
    table = pa.table(arrays)

    # Compute bbox
    bounds = shapely.bounds(geometries)
    xmin = min(b[0] for b in bounds)
    ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds)
    ymax = max(b[3] for b in bounds)

    # Build geo metadata (GeoParquet spec 1.0)
    geo_metadata = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "crs": {"type": "name", "properties": {"name": crs}},
                "bbox": [xmin, ymin, xmax, ymax],
            }
        },
    }

    # Add geo metadata to schema
    metadata = {b"geo": json.dumps(geo_metadata).encode()}
    table = table.replace_schema_metadata(metadata)

    # Write parquet
    pq.write_table(table, path)


@pytest.fixture()
def geoparquet_file(tmp_path):
    """Create a GeoParquet file with 3 point geometries."""
    path = tmp_path / "points.geoparquet"

    geometries = [
        shapely.Point(1.0, 2.0),
        shapely.Point(3.0, 4.0),
        shapely.Point(5.0, 6.0),
    ]

    properties = {
        "id": [1, 2, 3],
        "name": ["alpha", "beta", "gamma"],
        "value": [42.5, 99.1, 0.0],
    }

    _create_geoparquet_with_pyarrow(path, geometries, properties)
    return path


@pytest.fixture()
def geoparquet_file_with_bbox(tmp_path):
    """Create a GeoParquet file with polygon geometries."""
    path = tmp_path / "zones.geoparquet"

    geometries = [
        shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        shapely.Polygon([(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]),
    ]

    properties = {
        "zone_id": ["A", "B"],
        "area_km2": [100.0, 250.5],
    }

    _create_geoparquet_with_pyarrow(path, geometries, properties)
    return path


@pytest.fixture()
def plain_parquet_file(tmp_path):
    """Create a non-spatial parquet file (no geo metadata)."""
    path = tmp_path / "readings.parquet"

    table = pa.table(
        {
            "station_id": [1, 2, 3, 4],
            "temperature": [22.5, 23.1, 21.8, 24.0],
            "humidity": [45, 50, 55, 60],
        }
    )

    pq.write_table(table, path)
    return path


@pytest.fixture()
def directory_with_files(tmp_path, geoparquet_file, plain_parquet_file):
    """Create a directory with mixed parquet files."""
    # Files are already in tmp_path from fixtures
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Materialization
# ---------------------------------------------------------------------------


class TestGeoParquetEagerMaterialization:
    """Validate eager materialization of GeoParquet files → GeoPackage."""

    def test_eager_produces_vector(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.name == "points"

    def test_eager_produces_gpkg(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".gpkg")

    def test_eager_feature_count(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_eager_content_hash_present(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path)

        assert result.artifact.lineage.params["source"] == "geoparquet"
        assert result.artifact.lineage.params["path"] == str(geoparquet_file)
        assert result.artifact.lineage.params["lazy"] is False

    def test_eager_strategy(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_spatial_extent(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(1.0)
        assert ymin == pytest.approx(2.0)
        assert xmax == pytest.approx(5.0)
        assert ymax == pytest.approx(6.0)

    def test_eager_crs(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path)

        assert result.artifact.spatial.crs is not None
        assert "4326" in result.artifact.spatial.crs

    def test_eager_gpkg_readable(self, geoparquet_file, tmp_path):
        """Output GeoPackage is readable by fiona."""
        import fiona

        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path)

        with fiona.open(result.artifact.backing.uri) as src:
            features = list(src)
        assert len(features) == 3


# ---------------------------------------------------------------------------
# Lazy Materialization
# ---------------------------------------------------------------------------


class TestGeoParquetLazyMaterialization:
    """Validate lazy (metadata-only) materialization."""

    def test_lazy_backing_kind(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_feature_count(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path, lazy=True)

        assert result.artifact.spatial.feature_count == 3

    def test_lazy_backing_uri(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path, lazy=True)

        assert "geoparquet://" in result.artifact.backing.uri
        assert "points" in result.artifact.backing.uri

    def test_lazy_lineage(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True

    def test_lazy_spatial_from_metadata(self, geoparquet_file_with_bbox, tmp_path):
        """Lazy mode populates spatial descriptor from geo metadata."""
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file_with_bbox), tmp_path, lazy=True)

        assert result.artifact.spatial.extent is not None
        assert result.artifact.spatial.crs is not None


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestGeoParquetDiscover:
    """Validate file discovery."""

    def test_discover_lists_parquet_files(self, directory_with_files):
        connector = GeoParquetConnector()
        entries = connector.discover(str(directory_with_files))

        names = {e.name for e in entries}
        assert "points" in names
        assert "readings" in names

    def test_discover_identifies_geoparquet(self, directory_with_files):
        connector = GeoParquetConnector()
        entries = connector.discover(str(directory_with_files))

        for entry in entries:
            if entry.name == "points":
                assert entry.metadata["is_geoparquet"] is True
            elif entry.name == "readings":
                assert entry.metadata["is_geoparquet"] is False

    def test_discover_source_refs(self, directory_with_files):
        connector = GeoParquetConnector()
        entries = connector.discover(str(directory_with_files))

        for entry in entries:
            assert entry.source_ref.endswith(".parquet") or entry.source_ref.endswith(".geoparquet")

    def test_discover_with_path_override(self, directory_with_files):
        connector = GeoParquetConnector()
        entries = connector.discover({"path": str(directory_with_files)})

        assert len(entries) >= 2


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestGeoParquetMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_columns(self, geoparquet_file):
        connector = GeoParquetConnector()
        meta = connector.metadata(str(geoparquet_file))

        col_names = [c["name"] for c in meta["columns"]]
        assert "id" in col_names
        assert "name" in col_names
        assert "geometry" in col_names

    def test_metadata_feature_count(self, geoparquet_file):
        connector = GeoParquetConnector()
        meta = connector.metadata(str(geoparquet_file))

        assert meta["feature_count"] == 3

    def test_metadata_is_geoparquet(self, geoparquet_file):
        connector = GeoParquetConnector()
        meta = connector.metadata(str(geoparquet_file))

        assert meta["is_geoparquet"] is True

    def test_metadata_geo_metadata_present(self, geoparquet_file):
        connector = GeoParquetConnector()
        meta = connector.metadata(str(geoparquet_file))

        assert meta["geo_metadata"] is not None
        assert "primary_column" in meta["geo_metadata"]
        assert "columns" in meta["geo_metadata"]

    def test_metadata_row_groups(self, geoparquet_file):
        connector = GeoParquetConnector()
        meta = connector.metadata(str(geoparquet_file))

        assert meta["row_groups"] >= 1

    def test_metadata_plain_parquet(self, plain_parquet_file):
        connector = GeoParquetConnector()
        meta = connector.metadata(str(plain_parquet_file))

        assert meta["is_geoparquet"] is False
        assert meta["geo_metadata"] is None


# ---------------------------------------------------------------------------
# Non-spatial Parquet
# ---------------------------------------------------------------------------


class TestGeoParquetNonSpatial:
    """Validate handling of non-spatial parquet files."""

    def test_plain_parquet_produces_table(self, plain_parquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(plain_parquet_file), tmp_path)

        assert result.artifact.type == ArtifactType.TABLE

    def test_plain_parquet_produces_csv(self, plain_parquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(plain_parquet_file), tmp_path)

        assert result.artifact.backing.uri.endswith(".csv")

    def test_plain_parquet_feature_count(self, plain_parquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(plain_parquet_file), tmp_path)

        assert result.artifact.spatial.feature_count == 4


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestGeoParquetErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path):
        connector = GeoParquetConnector()
        nonexistent = tmp_path / "does_not_exist.geoparquet"

        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize(str(nonexistent), tmp_path)

        assert "not found" in str(exc_info.value).lower()

    def test_non_parquet_file_raises(self, tmp_path):
        connector = GeoParquetConnector()
        bad_file = tmp_path / "not_parquet.txt"
        bad_file.write_text("this is not parquet data")

        with pytest.raises(MaterializeError):
            connector.materialize(str(bad_file), tmp_path)

    def test_discover_nonexistent_directory_raises(self, tmp_path):
        connector = GeoParquetConnector()
        nonexistent = tmp_path / "no_such_dir"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))

    def test_metadata_nonexistent_file_raises(self, tmp_path):
        connector = GeoParquetConnector()
        nonexistent = tmp_path / "does_not_exist.geoparquet"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))


# ---------------------------------------------------------------------------
# SourceRef Support
# ---------------------------------------------------------------------------


class TestGeoParquetSourceRef:
    """Validate SourceRef handling."""

    def test_sourceref_local(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        ref = SourceRef.local(str(geoparquet_file))
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_raw_string_path(self, geoparquet_file, tmp_path):
        connector = GeoParquetConnector()
        result = connector.materialize(str(geoparquet_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
