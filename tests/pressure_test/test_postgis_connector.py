"""PostGIS connector pressure test.

Lane: connector

Adversarial stress points:
1. source_ref parsing: schema.table, bare table, SELECT query — where does str feel underfit?
2. Geometry vs non-geometry branching → VECTOR vs TABLE artifact type
3. Lazy vs eager materialization: POSTGIS backing vs dumped GeoPackage/CSV
4. Connection lifecycle: connect, introspect, disconnect
5. Metadata richness: SRID, extent, feature_count, column schema
6. SpatialDescriptor population from actual PostGIS metadata queries
7. Lineage captures connection provenance (host, db, schema, table/query)
8. Discover: list tables in schema with geometry info

All tests mock the database — no live PostGIS needed.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from quarry_connectors.postgis import PostGISConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import Connector, ConnectorCapability, MaterializeError

# ---------------------------------------------------------------------------
# Fixtures: mock database responses
# ---------------------------------------------------------------------------


def _mock_cursor_factory(responses):
    """Create a mock cursor that returns canned results for successive queries.

    responses: list of (description, rows) tuples returned for each execute() call.
    """
    cursor = MagicMock()
    call_index = [0]

    def _execute(query, params=None):
        pass

    def _fetchall():
        idx = call_index[0]
        call_index[0] += 1
        if idx < len(responses):
            desc, rows = responses[idx]
            cursor.description = desc
            return rows
        return []

    def _fetchone():
        idx = call_index[0]
        call_index[0] += 1
        if idx < len(responses):
            desc, rows = responses[idx]
            cursor.description = desc
            return rows[0] if rows else None
        return None

    cursor.execute = MagicMock(side_effect=_execute)
    cursor.fetchall = MagicMock(side_effect=_fetchall)
    cursor.fetchone = MagicMock(side_effect=_fetchone)
    return cursor


def _geometry_table_introspection():
    """Mock responses for introspecting a geometry table (rivers)."""
    # Query 1: Check table exists + get columns
    columns_desc = [("column_name",), ("data_type",), ("udt_name",)]
    columns_rows = [
        ("id", "integer", "int4"),
        ("name", "character varying", "varchar"),
        ("geom", "USER-DEFINED", "geometry"),
        ("length_km", "double precision", "float8"),
    ]

    # Query 2: Get geometry column details (type, SRID)
    geom_desc = [("type",), ("srid",), ("coord_dimension",)]
    geom_rows = [("MULTILINESTRING", 4326, 2)]

    # Query 3: Get extent via ST_Extent
    extent_desc = [("xmin",), ("ymin",), ("xmax",), ("ymax",)]
    extent_rows = [(-95.5, 29.0, -89.5, 35.5)]

    # Query 4: Get row count
    count_desc = [("count",)]
    count_rows = [(1247,)]

    return [
        (columns_desc, columns_rows),
        (geom_desc, geom_rows),
        (extent_desc, extent_rows),
        (count_desc, count_rows),
    ]


def _non_geometry_table_introspection():
    """Mock responses for introspecting a plain table (measurements)."""
    columns_desc = [("column_name",), ("data_type",), ("udt_name",)]
    columns_rows = [
        ("id", "integer", "int4"),
        ("station_id", "character varying", "varchar"),
        ("value", "double precision", "float8"),
        ("timestamp", "timestamp with time zone", "timestamptz"),
    ]

    # Query 2: No geometry columns
    geom_desc = [("type",), ("srid",), ("coord_dimension",)]
    geom_rows = []

    # Query 3: row count
    count_desc = [("count",)]
    count_rows = [(50000,)]

    return [
        (columns_desc, columns_rows),
        (geom_desc, geom_rows),
        (count_desc, count_rows),
    ]


def _make_connector(schema="public"):
    return PostGISConnector(
        host="localhost",
        port=5432,
        dbname="testdb",
        user="testuser",
        password="testpass",
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_connector_protocol(self):
        conn = _make_connector()
        assert isinstance(conn, Connector)

    def test_capabilities_include_required(self):
        conn = _make_connector()
        caps = conn.capabilities
        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps

    def test_name_is_postgis(self):
        conn = _make_connector()
        assert conn.name == "postgis"


# ---------------------------------------------------------------------------
# source_ref parsing
# ---------------------------------------------------------------------------


class TestSourceRefParsing:
    """The core adversarial target: does source_ref: str hold for PostGIS?"""

    def test_schema_dot_table(self):
        conn = _make_connector()
        schema, table, query = conn._parse_source_ref("hydro.rivers")
        assert schema == "hydro"
        assert table == "rivers"
        assert query is None

    def test_bare_table_uses_default_schema(self):
        conn = _make_connector(schema="public")
        schema, table, query = conn._parse_source_ref("rivers")
        assert schema == "public"
        assert table == "rivers"
        assert query is None

    def test_select_query(self):
        conn = _make_connector()
        sql = "SELECT id, geom FROM rivers WHERE length_km > 100"
        schema, table, query = conn._parse_source_ref(sql)
        assert schema is None
        assert table is None
        assert query == sql

    def test_select_case_insensitive(self):
        conn = _make_connector()
        sql = "select * from rivers"
        _, _, query = conn._parse_source_ref(sql)
        assert query == sql

    def test_double_quoted_table_with_schema(self):
        """Schema.table where table has uppercase — valid PostGIS identifier."""
        conn = _make_connector()
        schema, table, query = conn._parse_source_ref('public."RiverData"')
        assert schema == "public"
        assert table == '"RiverData"'
        assert query is None

    def test_source_ref_friction_observation(self):
        """Document: source_ref as str IS starting to feel strained here.

        For STAC: collection/item::asset — convention separator
        For PostGIS: schema.table — convention dot separator
        For query: must sniff 'SELECT' prefix

        A SourceRef type would let connectors parse upfront and validate.
        But it's not broken yet — just awkward.
        """
        conn = _make_connector()
        # All three shapes parse correctly as str
        assert conn._parse_source_ref("hydro.rivers")[1] == "rivers"
        assert conn._parse_source_ref("rivers")[1] == "rivers"
        assert conn._parse_source_ref("SELECT 1")[2] == "SELECT 1"


# ---------------------------------------------------------------------------
# Geometry vs non-geometry branching
# ---------------------------------------------------------------------------


class TestArtifactTypeBranching:
    """Does the connector correctly branch VECTOR vs TABLE based on geometry presence?"""

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_geometry_table_produces_vector(self, mock_connect, tmp_path):
        conn = _make_connector()
        cursor = _mock_cursor_factory(_geometry_table_introspection())
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        result = conn.materialize("hydro.rivers", tmp_path, lazy=True)
        assert result.artifact.type == ArtifactType.VECTOR

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_non_geometry_table_produces_table(self, mock_connect, tmp_path):
        conn = _make_connector()
        cursor = _mock_cursor_factory(_non_geometry_table_introspection())
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        result = conn.materialize("measurements", tmp_path, lazy=True)
        assert result.artifact.type == ArtifactType.TABLE


# ---------------------------------------------------------------------------
# Lazy materialization
# ---------------------------------------------------------------------------


class TestLazyMaterialization:
    """Lazy = full metadata, no data transfer. Backing is POSTGIS kind."""

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_lazy_produces_postgis_backing(self, mock_connect, tmp_path):
        conn = _make_connector()
        cursor = _mock_cursor_factory(_geometry_table_introspection())
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        result = conn.materialize("hydro.rivers", tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.POSTGIS
        assert "hydro.rivers" in result.artifact.backing.uri
        assert result.strategy == "lazy_handle"

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_lazy_has_full_spatial_metadata(self, mock_connect, tmp_path):
        conn = _make_connector()
        cursor = _mock_cursor_factory(_geometry_table_introspection())
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        result = conn.materialize("hydro.rivers", tmp_path, lazy=True)
        spatial = result.artifact.spatial

        assert spatial.crs == "EPSG:4326"
        assert spatial.extent == (-95.5, 29.0, -89.5, 35.5)
        assert spatial.feature_count == 1247

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_lazy_table_has_feature_count(self, mock_connect, tmp_path):
        """Even non-geometry tables report row count as feature_count."""
        conn = _make_connector()
        cursor = _mock_cursor_factory(_non_geometry_table_introspection())
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        result = conn.materialize("measurements", tmp_path, lazy=True)
        assert result.artifact.spatial.feature_count == 50000


# ---------------------------------------------------------------------------
# Eager materialization
# ---------------------------------------------------------------------------


class TestEagerMaterialization:
    """Eager = data transferred to workspace. VECTOR → GeoPackage, TABLE → CSV."""

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_eager_vector_creates_gpkg(self, mock_connect, tmp_path):
        """Eager materialization of geometry table produces a .gpkg file."""
        conn = _make_connector()

        # Build mock with introspection + data fetch responses
        introspection = _geometry_table_introspection()
        # Add data fetch response (WKB hex geometries + attributes)
        data_desc = [("id",), ("name",), ("geom",), ("length_km",)]
        # WKB for MULTILINESTRING(((0 0, 1 1))) in hex — matches schema geometry type
        wkb_hex = (
            "01050000000100000001020000000200000000000000000000000000000000000000"
            "000000000000F03F000000000000F03F"
        )
        data_rows = [
            (1, "Mississippi", wkb_hex, 3730.0),
            (2, "Missouri", wkb_hex, 3768.0),
        ]
        introspection.append((data_desc, data_rows))

        cursor = _mock_cursor_factory(introspection)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        result = conn.materialize("hydro.rivers", tmp_path, lazy=False)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".gpkg")
        assert result.strategy == "fetched_remote"
        assert Path(result.artifact.backing.uri).exists()

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_eager_table_creates_csv(self, mock_connect, tmp_path):
        """Eager materialization of non-geometry table produces a .csv file."""
        conn = _make_connector()

        introspection = _non_geometry_table_introspection()
        # Add data fetch response
        data_desc = [("id",), ("station_id",), ("value",), ("timestamp",)]
        data_rows = [
            (1, "USGS-07289000", 42.5, "2024-01-15T12:00:00+00:00"),
            (2, "USGS-07289000", 43.1, "2024-01-15T13:00:00+00:00"),
        ]
        introspection.append((data_desc, data_rows))

        cursor = _mock_cursor_factory(introspection)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        result = conn.materialize("measurements", tmp_path, lazy=False)

        assert result.artifact.type == ArtifactType.TABLE
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".csv")
        assert Path(result.artifact.backing.uri).exists()


# ---------------------------------------------------------------------------
# Lineage provenance
# ---------------------------------------------------------------------------


class TestLineageProvenance:
    """Lineage must capture PostGIS-specific provenance."""

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_lineage_captures_connection_info(self, mock_connect, tmp_path):
        conn = _make_connector()
        cursor = _mock_cursor_factory(_geometry_table_introspection())
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        result = conn.materialize("hydro.rivers", tmp_path, lazy=True)
        params = result.artifact.lineage.params

        assert params["source"] == "postgis"
        assert params["host"] == "localhost"
        assert params["dbname"] == "testdb"
        assert params["schema"] == "hydro"
        assert params["table"] == "rivers"
        assert params["lazy"] is True

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_lineage_captures_query_source(self, mock_connect, tmp_path):
        """When source_ref is a query, lineage stores the SQL."""
        conn = _make_connector()
        cursor = _mock_cursor_factory(_geometry_table_introspection())
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        sql = "SELECT id, geom FROM rivers WHERE length_km > 100"
        result = conn.materialize(sql, tmp_path, lazy=True)
        params = result.artifact.lineage.params

        assert params["query"] == sql
        assert "table" not in params


# ---------------------------------------------------------------------------
# Metadata inspection
# ---------------------------------------------------------------------------


class TestMetadataInspection:
    """metadata() should return schema info without materializing."""

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_metadata_returns_column_info(self, mock_connect):
        conn = _make_connector()
        cursor = _mock_cursor_factory(_geometry_table_introspection())
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        meta = conn.metadata("hydro.rivers")

        assert "columns" in meta
        assert any(c["name"] == "geom" for c in meta["columns"])
        assert meta["geometry_type"] == "MULTILINESTRING"
        assert meta["srid"] == 4326
        assert meta["row_count"] == 1247

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_metadata_non_geometry_table(self, mock_connect):
        conn = _make_connector()
        cursor = _mock_cursor_factory(_non_geometry_table_introspection())
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        meta = conn.metadata("measurements")

        assert meta["geometry_type"] is None
        assert meta["srid"] is None
        assert meta["row_count"] == 50000


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestDiscover:
    """discover() lists tables in schema with geometry info."""

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_discover_returns_tables(self, mock_connect):
        conn = _make_connector()

        # Mock: list tables query
        tables_desc = [("table_name",), ("geometry_column",), ("type",), ("srid",)]
        tables_rows = [
            ("rivers", "geom", "MULTILINESTRING", 4326),
            ("lakes", "geom", "MULTIPOLYGON", 4326),
            ("stations", None, None, None),
        ]

        cursor = _mock_cursor_factory([(tables_desc, tables_rows)])
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        entries = conn.discover()

        assert len(entries) == 3
        # source_ref should be schema.table
        assert entries[0].source_ref == "public.rivers"
        assert entries[0].metadata["geometry_type"] == "MULTILINESTRING"
        assert entries[2].metadata["geometry_type"] is None

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_discover_with_schema_filter(self, mock_connect):
        conn = _make_connector(schema="hydro")

        tables_desc = [("table_name",), ("geometry_column",), ("type",), ("srid",)]
        tables_rows = [("rivers", "geom", "MULTILINESTRING", 4326)]

        cursor = _mock_cursor_factory([(tables_desc, tables_rows)])
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        entries = conn.discover()
        assert entries[0].source_ref == "hydro.rivers"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_connection_failure_raises_materialize_error(self, mock_connect, tmp_path):
        mock_connect.side_effect = Exception("Connection refused")
        conn = _make_connector()

        with pytest.raises(MaterializeError, match="Connection refused"):
            conn.materialize("rivers", tmp_path, lazy=True)

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_table_not_found_raises_materialize_error(self, mock_connect, tmp_path):
        conn = _make_connector()

        # Return empty columns = table doesn't exist
        cursor = _mock_cursor_factory([([], [])])
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(MaterializeError, match="not found"):
            conn.materialize("nonexistent_table", tmp_path, lazy=True)


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------


class TestRegistryRoundTrip:
    """PostGIS artifacts must survive registry persistence."""

    @patch("quarry_connectors.postgis.psycopg.connect")
    def test_postgis_artifact_persists_in_registry(self, mock_connect, tmp_path):
        from quarry_registry.registry import Registry

        conn = _make_connector()
        cursor = _mock_cursor_factory(_geometry_table_introspection())
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        result = conn.materialize("hydro.rivers", tmp_path, lazy=True)

        # Persist in registry
        registry = Registry(tmp_path / "test.duckdb")
        registry.save_artifact(result.artifact)

        # Retrieve
        loaded = registry.get_artifact(result.artifact.id)
        assert loaded is not None
        assert loaded.type == ArtifactType.VECTOR
        assert loaded.backing.kind == BackingStoreKind.POSTGIS
        assert loaded.spatial.crs == "EPSG:4326"
