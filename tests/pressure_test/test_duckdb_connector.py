"""
Pressure test: DuckDBConnector.

Lane: connector

Validates DuckDB table/query materialization following the PostGIS connector pattern:
- source_ref parsing (path::table, path::query, SourceRef.duckdb())
- geometry vs non-geometry branching → VECTOR vs TABLE
- lazy = metadata-only with DUCKDB backing
- eager = dump to GeoPackage (spatial) or CSV (non-spatial)
- discover: list tables
- graceful handling when spatial extension is unavailable
"""

from __future__ import annotations

import csv

import duckdb
import pytest
from quarry_connectors.duckdb_connector import DuckDBConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.source_ref import SourceRef, SourceRefKind

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_with_table(tmp_path):
    """Create a DuckDB file with a simple non-spatial table."""
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE readings (station_id INTEGER, value DOUBLE, label VARCHAR)")
    conn.execute(
        "INSERT INTO readings VALUES (1, 42.5, 'alpha'), (2, 99.1, 'beta'), (3, 0.0, 'gamma')"
    )
    conn.close()
    return db_path


@pytest.fixture()
def db_with_spatial(tmp_path):
    """Create a DuckDB file with a spatial table (if spatial extension available)."""
    db_path = str(tmp_path / "spatial.duckdb")
    conn = duckdb.connect(db_path)
    try:
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
    except Exception:
        conn.close()
        pytest.skip("DuckDB spatial extension not available")

    conn.execute("CREATE TABLE points (id INTEGER, name VARCHAR, geom GEOMETRY)")
    conn.execute(
        "INSERT INTO points VALUES "
        "(1, 'a', ST_Point(1.0, 2.0)), "
        "(2, 'b', ST_Point(3.0, 4.0)), "
        "(3, 'c', ST_Point(5.0, 6.0))"
    )
    conn.close()
    return db_path


@pytest.fixture()
def db_with_spatial_srid(tmp_path):
    """Create a DuckDB file with a spatial table that has an explicit SRID."""
    db_path = str(tmp_path / "srid.duckdb")
    conn = duckdb.connect(db_path)
    try:
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
    except Exception:
        conn.close()
        pytest.skip("DuckDB spatial extension not available")

    conn.execute("CREATE TABLE parcels (id INTEGER, geom GEOMETRY)")
    conn.execute(
        "INSERT INTO parcels VALUES "
        "(1, ST_Transform(ST_Point(-122.4, 37.8), 'EPSG:4326', 'EPSG:32610')), "
        "(2, ST_Transform(ST_Point(-122.3, 37.7), 'EPSG:4326', 'EPSG:32610'))"
    )
    conn.close()
    return db_path


@pytest.fixture()
def db_with_multiple_tables(tmp_path):
    """Create a DuckDB file with multiple tables."""
    db_path = str(tmp_path / "multi.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("CREATE TABLE sensors (id INTEGER, type VARCHAR)")
    conn.execute("INSERT INTO sensors VALUES (1, 'temp'), (2, 'pressure')")
    conn.execute("CREATE TABLE logs (ts TIMESTAMP, msg VARCHAR)")
    conn.execute("INSERT INTO logs VALUES (NOW(), 'started'), (NOW(), 'running')")
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# SourceRef construction
# ---------------------------------------------------------------------------


class TestSourceRefDuckDB:
    """Validate SourceRef factory methods and infer() for DuckDB."""

    def test_duckdb_factory_table(self):
        ref = SourceRef.duckdb("/data/my.duckdb", "readings")
        assert ref.kind == SourceRefKind.DUCKDB
        assert ref.raw == "/data/my.duckdb::readings"
        assert ref.params["db_path"] == "/data/my.duckdb"
        assert ref.params["table"] == "readings"

    def test_duckdb_factory_query(self):
        sql = "SELECT * FROM readings WHERE value > 10"
        ref = SourceRef.duckdb_query("/data/my.duckdb", sql)
        assert ref.kind == SourceRefKind.DUCKDB
        assert ref.raw == f"/data/my.duckdb::{sql}"
        assert ref.params["db_path"] == "/data/my.duckdb"
        assert ref.params["query"] == sql

    def test_infer_duckdb_table(self):
        ref = SourceRef.infer("/data/file.duckdb::my_table")
        assert ref.kind == SourceRefKind.DUCKDB
        assert ref.params["db_path"] == "/data/file.duckdb"
        assert ref.params["table"] == "my_table"

    def test_infer_duckdb_query(self):
        ref = SourceRef.infer("/data/file.duckdb::SELECT id FROM t")
        assert ref.kind == SourceRefKind.DUCKDB
        assert ref.params["db_path"] == "/data/file.duckdb"
        assert ref.params["query"] == "SELECT id FROM t"

    def test_infer_db_extension(self):
        """'.db' extension also classified as DUCKDB."""
        ref = SourceRef.infer("/data/analytics.db::metrics")
        assert ref.kind == SourceRefKind.DUCKDB
        assert ref.params["table"] == "metrics"

    def test_roundtrip(self):
        ref = SourceRef.duckdb("/tmp/test.duckdb", "zones")
        assert str(ref) == "/tmp/test.duckdb::zones"

    def test_frozen_hashable(self):
        ref = SourceRef.duckdb("/tmp/test.duckdb", "zones")
        assert hash(ref) is not None
        s = {ref}
        assert ref in s


# ---------------------------------------------------------------------------
# Non-spatial table materialization
# ---------------------------------------------------------------------------


class TestDuckDBTableMaterialization:
    """Validate eager materialization of non-spatial tables → CSV."""

    def test_eager_table_produces_csv(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.TABLE
        assert result.artifact.name == "readings"
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".csv")

    def test_csv_has_correct_rows(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path)

        with open(result.artifact.backing.uri) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["station_id"] == "1"
        assert rows[1]["value"] == "99.1"

    def test_csv_has_correct_schema(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path)

        with open(result.artifact.backing.uri) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert set(row.keys()) == {"station_id", "value", "label"}

    def test_feature_count_matches(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_lineage_records_source(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.lineage.params["source"] == "duckdb"
        assert result.artifact.lineage.params["table"] == "readings"
        assert result.artifact.lineage.params["db_path"] == db_with_table

    def test_content_hash_present(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.backing.content_hash is not None

    def test_strategy_is_fetched(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path)

        assert result.strategy == "fetched_local"


# ---------------------------------------------------------------------------
# Lazy materialization
# ---------------------------------------------------------------------------


class TestDuckDBLazyMaterialization:
    """Validate lazy (metadata-only) materialization."""

    def test_lazy_backing_kind(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.DUCKDB

    def test_lazy_strategy(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_has_row_count(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path, lazy=True)

        assert result.artifact.spatial.feature_count == 3

    def test_lazy_backing_uri(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path, lazy=True)

        assert "duckdb://" in result.artifact.backing.uri
        assert "readings" in result.artifact.backing.uri

    def test_lazy_lineage_records_lazy_flag(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True


# ---------------------------------------------------------------------------
# Query materialization
# ---------------------------------------------------------------------------


class TestDuckDBQueryMaterialization:
    """Validate query-based materialization."""

    def test_query_produces_filtered_csv(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        sql = "SELECT station_id, value FROM readings WHERE value > 10"
        ref = SourceRef.duckdb_query(db_with_table, sql)
        result = connector.materialize(ref, tmp_path)

        with open(result.artifact.backing.uri) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # only station_id 1 (42.5) and 2 (99.1)

    def test_query_artifact_name(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb_query(db_with_table, "SELECT * FROM readings")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.name == "query_result"

    def test_query_lineage_records_sql(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        sql = "SELECT * FROM readings LIMIT 1"
        ref = SourceRef.duckdb_query(db_with_table, sql)
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.lineage.params["query"] == sql


# ---------------------------------------------------------------------------
# Spatial table materialization
# ---------------------------------------------------------------------------


class TestDuckDBSpatialMaterialization:
    """Validate spatial table materialization → GeoPackage."""

    def test_spatial_produces_vector(self, db_with_spatial, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_spatial, "points")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.backing.uri.endswith(".gpkg")

    def test_spatial_feature_count(self, db_with_spatial, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_spatial, "points")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_spatial_extent(self, db_with_spatial, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_spatial, "points")
        result = connector.materialize(ref, tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(1.0)
        assert ymin == pytest.approx(2.0)
        assert xmax == pytest.approx(5.0)
        assert ymax == pytest.approx(6.0)

    def test_spatial_metadata_has_geometry_info(self, db_with_spatial, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_spatial, "points")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.metadata["geometry_column"] == "geom"
        assert result.artifact.metadata["has_spatial"] is True

    def test_spatial_gpkg_readable(self, db_with_spatial, tmp_path):
        """Output GeoPackage is readable by fiona."""
        import fiona

        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_spatial, "points")
        result = connector.materialize(ref, tmp_path)

        with fiona.open(result.artifact.backing.uri) as src:
            features = list(src)
        assert len(features) == 3

    def test_spatial_lazy_has_extent(self, db_with_spatial, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_spatial, "points")
        result = connector.materialize(ref, tmp_path, lazy=True)

        assert result.artifact.spatial.extent is not None
        assert result.artifact.backing.kind == BackingStoreKind.DUCKDB


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestDuckDBDiscover:
    """Validate table discovery."""

    def test_discover_lists_tables(self, db_with_multiple_tables):
        connector = DuckDBConnector(db_path=db_with_multiple_tables)
        entries = connector.discover()

        names = {e.name for e in entries}
        assert "sensors" in names
        assert "logs" in names

    def test_discover_with_db_path_override(self, db_with_multiple_tables):
        connector = DuckDBConnector()
        entries = connector.discover(query={"db_path": db_with_multiple_tables})

        assert len(entries) >= 2

    def test_discover_source_refs_parseable(self, db_with_multiple_tables):
        connector = DuckDBConnector(db_path=db_with_multiple_tables)
        entries = connector.discover()

        for entry in entries:
            assert "::" in entry.source_ref


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestDuckDBErrors:
    """Validate error cases."""

    def test_nonexistent_db_raises(self, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(str(tmp_path / "nope.duckdb"), "t")

        with pytest.raises(Exception):
            connector.materialize(ref, tmp_path)

    def test_nonexistent_table_raises(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "no_such_table")

        with pytest.raises(Exception):
            connector.materialize(ref, tmp_path)

    def test_no_db_path_raises(self, tmp_path):
        connector = DuckDBConnector()

        with pytest.raises(Exception):
            connector.materialize("just_a_table", tmp_path)

    def test_discover_no_db_path_raises(self):
        connector = DuckDBConnector()

        with pytest.raises(Exception):
            connector.discover()


# ---------------------------------------------------------------------------
# Source ref parsing
# ---------------------------------------------------------------------------


class TestDuckDBSourceRefParsing:
    """Validate source_ref parsing edge cases."""

    def test_raw_string_with_separator(self, db_with_table, tmp_path):
        """Raw string with :: separator works."""
        connector = DuckDBConnector()
        result = connector.materialize(f"{db_with_table}::readings", tmp_path)

        assert result.artifact.type == ArtifactType.TABLE
        assert result.artifact.name == "readings"

    def test_raw_string_query(self, db_with_table, tmp_path):
        connector = DuckDBConnector()
        result = connector.materialize(f"{db_with_table}::SELECT * FROM readings LIMIT 1", tmp_path)

        assert result.artifact.name == "query_result"

    def test_connector_default_db_path(self, db_with_table, tmp_path):
        """Connector's default db_path used when raw string has no ::."""
        connector = DuckDBConnector(db_path=db_with_table)
        result = connector.materialize("readings", tmp_path)

        assert result.artifact.type == ArtifactType.TABLE
        assert result.artifact.name == "readings"

    def test_sourceref_overrides_connector_default(self, db_with_table, tmp_path):
        """SourceRef params take precedence over connector's default db_path."""
        connector = DuckDBConnector(db_path="/nonexistent.duckdb")
        ref = SourceRef.duckdb(db_with_table, "readings")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.TABLE


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestDuckDBMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_columns(self, db_with_table):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        meta = connector.metadata(ref)

        col_names = [c["name"] for c in meta["columns"]]
        assert "station_id" in col_names
        assert "value" in col_names
        assert "label" in col_names

    def test_metadata_row_count(self, db_with_table):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        meta = connector.metadata(ref)

        assert meta["row_count"] == 3

    def test_metadata_no_geometry_for_table(self, db_with_table):
        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_with_table, "readings")
        meta = connector.metadata(ref)

        assert meta["geometry_column"] is None


# ---------------------------------------------------------------------------
# SQL injection defense
# ---------------------------------------------------------------------------


class TestDuckDBInjectionDefense:
    """Validate that identifier quoting prevents SQL injection."""

    def test_malicious_table_name_does_not_execute(self, tmp_path):
        """A table name containing SQL injection payload should not execute arbitrary SQL."""
        db_path = str(tmp_path / "injection.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("CREATE TABLE safe_table (id INTEGER)")
        conn.execute("INSERT INTO safe_table VALUES (1)")
        conn.close()

        connector = DuckDBConnector()
        # This should fail with "table not found", NOT execute the DROP
        ref = SourceRef.duckdb(db_path, "safe_table; DROP TABLE safe_table; --")
        with pytest.raises(Exception):
            connector.materialize(ref, tmp_path)

        # Verify original table still exists (injection did NOT execute)
        conn = duckdb.connect(db_path, read_only=True)
        row = conn.execute("SELECT COUNT(*) FROM safe_table").fetchone()
        conn.close()
        assert row[0] == 1

    def test_table_with_special_chars_quoted(self, tmp_path):
        """Table names with special characters are handled via quoting."""
        db_path = str(tmp_path / "special.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute('CREATE TABLE "my-table.v2" (val INTEGER)')
        conn.execute('INSERT INTO "my-table.v2" VALUES (42)')
        conn.close()

        connector = DuckDBConnector()
        ref = SourceRef.duckdb(db_path, "my-table.v2")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.TABLE
        assert result.artifact.spatial.feature_count == 1
