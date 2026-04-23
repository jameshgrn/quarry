"""PostGISConnector — materializes PostGIS tables/queries into canonical artifacts.

Lane: connector

Pressures:
- source_ref (schema.table, bare table, SELECT query — str is strained here)
- connection lifecycle (no file handle, must connect/introspect/disconnect)
- geometry vs non-geometry branching → VECTOR vs TABLE
- lazy = metadata-only with POSTGIS backing, eager = dump to GeoPackage/CSV
- metadata richness (SRID, geometry type, extent, column schema, row count)

Design decisions:
- Constructor takes connection params (host, port, dbname, user, password, schema)
- source_ref parsing:
  - starts with SELECT/select → raw query
  - contains '.' → schema.table
  - otherwise → default_schema.table
- Eager vector: dump to GeoPackage via fiona (geometry as WKB → shapely → __geo_interface__)
- Eager table: dump to CSV (stdlib)
- Lazy: POSTGIS BackingStore with connection URI (password stripped)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import psycopg
import shapely.wkb
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    Lineage,
    SpatialDescriptor,
    content_hash,
)
from quarry_core.connector import (
    CatalogEntry,
    ConnectorCapability,
    MaterializeError,
    MaterializeResult,
)

if TYPE_CHECKING:
    from quarry_core.source_ref import SourceRef


@dataclass
class IntrospectionResult:
    """Base introspection result with common fields."""

    columns: list[dict[str, Any]]
    geometry_type: str | None
    srid: int | None
    extent: tuple[float, float, float, float] | None
    row_count: int
    kind: Literal["table", "query"]


@dataclass
class TableIntrospection(IntrospectionResult):
    """Introspection result for a database table.

    Has full geometry metadata and accurate row count from the database.
    """

    kind: Literal["table"] = "table"


@dataclass
class QueryIntrospection(IntrospectionResult):
    """Introspection result for a raw SQL query.

    Geometry metadata is not available without executing the query.
    Row count is always 0 (unknown without full execution).
    """

    kind: Literal["query"] = "query"


class PostGISConnector:
    """Materializes PostGIS tables and queries into canonical Quarry artifacts.

    Configured with connection params. Default schema for bare table references.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "postgres",
        user: str = "postgres",
        password: str = "",
        schema: str = "public",
    ):
        self._host = host
        self._port = port
        self._dbname = dbname
        self._user = user
        self._password = password
        self._schema = schema

    @property
    def name(self) -> str:
        return "postgis"

    @property
    def capabilities(self) -> ConnectorCapability:
        return (
            ConnectorCapability.MATERIALIZE
            | ConnectorCapability.DISCOVER
            | ConnectorCapability.MATERIALIZE_LAZY
            | ConnectorCapability.METADATA_ONLY
        )

    def materialize(
        self,
        source_ref: SourceRef | str,
        workspace: Path,
        *,
        lazy: bool = False,
    ) -> MaterializeResult:
        """Materialize a PostGIS table or query into a canonical artifact.

        source_ref formats:
            "schema.table"       — schema-qualified table
            "table"              — uses connector's default schema
            "SELECT ..."         — raw SQL query

        FRICTION NOTE: source_ref as str is strained here.
        schema.table uses dot separator (conflicts with any table name containing dots).
        Query detection relies on prefix sniffing. A SourceRef type would make
        schema, table, and query first-class and parseable without heuristics.
        """
        schema, table, query = self._parse_source_ref(source_ref)

        try:
            with psycopg.connect(self._conninfo()) as conn:
                with conn.cursor() as cur:
                    introspection = self._introspect(cur, schema, table, query)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, str(e)) from e

        has_geometry = introspection.geometry_type is not None
        artifact_type = ArtifactType.VECTOR if has_geometry else ArtifactType.TABLE

        # Build spatial descriptor
        spatial = SpatialDescriptor(
            crs=f"EPSG:{introspection.srid}" if introspection.srid else None,
            extent=introspection.extent,
            feature_count=introspection.row_count,
        )

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "postgis",
            "host": self._host,
            "dbname": self._dbname,
            "lazy": lazy,
        }
        if query:
            lineage_params["query"] = query
        else:
            lineage_params["schema"] = schema
            lineage_params["table"] = table

        if lazy:
            # Lazy: metadata only, data stays in PostGIS
            backing_uri = self._backing_uri(schema, table, query)
            artifact = Artifact(
                type=artifact_type,
                name=table or "query_result",
                backing=BackingStore(
                    kind=BackingStoreKind.POSTGIS,
                    uri=backing_uri,
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=self._artifact_metadata(introspection),
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"PostGIS {'query' if query else f'{schema}.{table}'} — metadata only",
            )

        # Eager: fetch data and dump to local file
        try:
            with psycopg.connect(self._conninfo()) as conn:
                with conn.cursor() as cur:
                    if has_geometry:
                        output_path = self._dump_vector(
                            cur, schema, table, query, introspection, workspace
                        )
                    else:
                        output_path = self._dump_table(
                            cur, schema, table, query, introspection, workspace
                        )
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Data fetch failed: {e}") from e

        artifact = Artifact(
            type=artifact_type,
            name=table or "query_result",
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(output_path),
                size_bytes=output_path.stat().st_size,
                content_hash=content_hash(output_path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata=self._artifact_metadata(introspection),
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="fetched_remote",
            source_ref=source_ref,
            notes=f"Dumped {output_path.name} ({output_path.stat().st_size} bytes)",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """List tables in the configured schema, with geometry info.

        query as dict supports:
            schema: str (override default schema)
        """
        schema = self._schema
        if isinstance(query, dict):
            schema = query.get("schema", self._schema)
        elif isinstance(query, str):
            schema = query

        sql = """
            SELECT
                t.table_name,
                g.f_geometry_column,
                g.type,
                g.srid
            FROM information_schema.tables t
            LEFT JOIN geometry_columns g
                ON g.f_table_schema = t.table_schema
                AND g.f_table_name = t.table_name
            WHERE t.table_schema = %s
                AND t.table_type = 'BASE TABLE'
            ORDER BY t.table_name
        """

        try:
            with psycopg.connect(self._conninfo()) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (schema,))
                    rows = cur.fetchall()
        except Exception as e:
            raise MaterializeError(f"discover({schema})", str(e)) from e

        entries = []
        for row in rows:
            table_name, geom_col, geom_type, srid = row
            entries.append(
                CatalogEntry(
                    source_ref=f"{schema}.{table_name}",
                    name=table_name,
                    metadata={
                        "geometry_column": geom_col,
                        "geometry_type": geom_type,
                        "srid": srid,
                    },
                )
            )

        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get table/query metadata without materializing data."""
        schema, table, query = self._parse_source_ref(source_ref)

        try:
            with psycopg.connect(self._conninfo()) as conn:
                with conn.cursor() as cur:
                    introspection = self._introspect(cur, schema, table, query)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, str(e)) from e

        return {
            "columns": introspection.columns,
            "geometry_type": introspection.geometry_type,
            "srid": introspection.srid,
            "row_count": introspection.row_count,
            "extent": introspection.extent,
        }

    # -----------------------------------------------------------------------
    # Public helper: source_ref parsing (exposed for testing)
    # -----------------------------------------------------------------------

    def _parse_source_ref(
        self, source_ref: SourceRef | str
    ) -> tuple[str | None, str | None, str | None]:
        """Parse source_ref into (schema, table, query).

        Returns:
            (schema, table, None) for table references
            (None, None, query) for SQL queries
        """
        from quarry_core.source_ref import SourceRef, SourceRefKind

        if isinstance(source_ref, SourceRef):
            if source_ref.kind == SourceRefKind.DATABASE_REF:
                params = source_ref.params or {}
                if "query" in params:
                    return (None, None, params["query"])
                return (params.get("schema"), params.get("table"), None)
            stripped = source_ref.raw.strip()
        else:
            stripped = source_ref.strip()

        # Query detection: starts with SELECT (case-insensitive)
        if stripped.upper().startswith("SELECT "):
            return (None, None, stripped)

        # Schema.table detection: contains dot (not inside quotes)
        if "." in stripped:
            parts = stripped.split(".", 1)
            return (parts[0], parts[1], None)

        # Bare table: use default schema
        return (self._schema, stripped, None)

    # -----------------------------------------------------------------------
    # Private: connection
    # -----------------------------------------------------------------------

    def _conninfo(self) -> str:
        """Build psycopg connection string."""
        return (
            f"host={self._host} port={self._port} dbname={self._dbname} "
            f"user={self._user} password={self._password}"
        )

    def _backing_uri(self, schema: str | None, table: str | None, query: str | None) -> str:
        """Build a backing URI for POSTGIS backing store (password stripped)."""
        base = f"postgresql://{self._user}@{self._host}:{self._port}/{self._dbname}"
        if query:
            return f"{base}?query={query[:100]}"
        return f"{base}/{schema}.{table}"

    # -----------------------------------------------------------------------
    # Private: introspection
    # -----------------------------------------------------------------------

    def _introspect(
        self,
        cur: Any,
        schema: str | None,
        table: str | None,
        query: str | None,
    ) -> TableIntrospection | QueryIntrospection:
        """Introspect a table or query to get column info, geometry details, extent, count."""
        if query:
            return self._introspect_query(cur, query)
        return self._introspect_table(cur, schema, table)

    def _introspect_table(self, cur: Any, schema: str, table: str) -> TableIntrospection:
        """Introspect a schema.table."""
        # Get columns
        cur.execute(
            """
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (schema, table),
        )
        columns_raw = cur.fetchall()

        if not columns_raw:
            raise MaterializeError(
                f"{schema}.{table}",
                f"Table not found or has no columns: {schema}.{table}",
            )

        columns = [{"name": row[0], "data_type": row[1], "udt_name": row[2]} for row in columns_raw]

        # Check for geometry column
        cur.execute(
            """
            SELECT type, srid, coord_dimension
            FROM geometry_columns
            WHERE f_table_schema = %s AND f_table_name = %s
            """,
            (schema, table),
        )
        geom_info = cur.fetchall()

        geometry_type = None
        srid = None
        extent = None

        if geom_info:
            geometry_type = geom_info[0][0]
            srid = geom_info[0][1]

            # Get extent
            geom_col = next((c["name"] for c in columns if c["udt_name"] == "geometry"), "geom")
            cur.execute(
                f"SELECT ST_XMin(e), ST_YMin(e), ST_XMax(e), ST_YMax(e) "
                f"FROM (SELECT ST_Extent({geom_col}) AS e FROM {schema}.{table}) sub"
            )
            extent_row = cur.fetchall()
            if extent_row and extent_row[0][0] is not None:
                extent = (extent_row[0][0], extent_row[0][1], extent_row[0][2], extent_row[0][3])

        # Get row count
        cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")  # noqa: S608
        count_row = cur.fetchall()
        row_count = count_row[0][0] if count_row else 0

        return TableIntrospection(
            columns=columns,
            geometry_type=geometry_type,
            srid=srid,
            extent=extent,
            row_count=row_count,
        )

    def _introspect_query(self, cur: Any, query: str) -> QueryIntrospection:
        """Introspect a query by wrapping in a LIMIT 0 to get column info."""
        # Use the query with LIMIT 0 to get column info without fetching data
        cur.execute(f"SELECT * FROM ({query}) sub LIMIT 0")  # noqa: S608
        cur.fetchall()  # discard rows, we only need description

        columns = []
        if cur.description:
            columns = [{"name": desc[0], "data_type": "unknown"} for desc in cur.description]

        # For queries, we can't easily get geometry info without running it
        # Return minimal introspection
        return QueryIntrospection(
            columns=columns,
            geometry_type=None,
            srid=None,
            extent=None,
            row_count=0,
        )

    # -----------------------------------------------------------------------
    # Private: eager dump (vector → GeoPackage)
    # -----------------------------------------------------------------------

    def _dump_vector(
        self,
        cur: Any,
        schema: str | None,
        table: str | None,
        query: str | None,
        introspection: TableIntrospection | QueryIntrospection,
        workspace: Path,
    ) -> Path:
        """Dump a geometry table/query to GeoPackage."""
        import fiona
        from fiona.crs import CRS

        # Determine the SQL to fetch data
        if query:
            fetch_sql = query
        else:
            fetch_sql = f"SELECT * FROM {schema}.{table}"  # noqa: S608

        cur.execute(fetch_sql)
        rows = cur.fetchall()

        # Find geometry column index
        columns = introspection.columns
        geom_idx = None
        for i, col in enumerate(columns):
            if col.get("udt_name") == "geometry" or col["name"] == "geom":
                geom_idx = i
                break

        if geom_idx is None:
            raise MaterializeError(
                f"{schema}.{table}" if table else "query",
                "No geometry column found for vector dump",
            )

        # Build fiona schema
        properties = {}
        for i, col in enumerate(columns):
            if i == geom_idx:
                continue
            properties[col["name"]] = _pg_type_to_fiona(col.get("data_type", "str"))

        geom_type = _normalize_geometry_type(introspection.geometry_type or "Unknown")
        srid = introspection.srid or 4326

        output_name = table or "query_result"
        output_path = workspace / f"{output_name}.gpkg"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fiona_schema = {"geometry": geom_type, "properties": properties}
        crs = CRS.from_epsg(srid)

        with fiona.open(output_path, "w", driver="GPKG", schema=fiona_schema, crs=crs) as dst:
            for row in rows:
                # Parse geometry from WKB hex
                geom_value = row[geom_idx]
                if geom_value is None:
                    continue
                if isinstance(geom_value, str):
                    geom = shapely.wkb.loads(bytes.fromhex(geom_value))
                else:
                    geom = shapely.wkb.loads(geom_value)

                # Build properties dict
                props = {}
                for i, col in enumerate(columns):
                    if i == geom_idx:
                        continue
                    props[col["name"]] = row[i]

                dst.write(
                    {
                        "geometry": geom.__geo_interface__,
                        "properties": props,
                    }
                )

        return output_path

    # -----------------------------------------------------------------------
    # Private: eager dump (table → CSV)
    # -----------------------------------------------------------------------

    def _dump_table(
        self,
        cur: Any,
        schema: str | None,
        table: str | None,
        query: str | None,
        introspection: TableIntrospection | QueryIntrospection,
        workspace: Path,
    ) -> Path:
        """Dump a non-geometry table/query to CSV."""
        if query:
            fetch_sql = query
        else:
            fetch_sql = f"SELECT * FROM {schema}.{table}"  # noqa: S608

        cur.execute(fetch_sql)
        rows = cur.fetchall()

        columns = introspection.columns
        col_names = [c["name"] for c in columns]

        output_name = table or "query_result"
        output_path = workspace / f"{output_name}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
            for row in rows:
                writer.writerow(row)

        return output_path

    # -----------------------------------------------------------------------
    # Private: metadata helpers
    # -----------------------------------------------------------------------

    def _artifact_metadata(
        self, introspection: TableIntrospection | QueryIntrospection
    ) -> dict[str, Any]:
        """Build artifact metadata bag from introspection results."""
        meta: dict[str, Any] = {
            "source": "postgis",
            "host": self._host,
            "dbname": self._dbname,
        }
        if introspection.geometry_type:
            meta["geometry_type"] = introspection.geometry_type
        if introspection.srid:
            meta["srid"] = introspection.srid
        meta["column_count"] = len(introspection.columns)
        return meta


def _pg_type_to_fiona(pg_type: str) -> str:
    """Map PostgreSQL data types to fiona property types."""
    mapping = {
        "integer": "int",
        "bigint": "int",
        "smallint": "int",
        "double precision": "float",
        "real": "float",
        "numeric": "float",
        "character varying": "str",
        "text": "str",
        "boolean": "bool",
        "date": "date",
        "timestamp with time zone": "datetime",
        "timestamp without time zone": "datetime",
    }
    return mapping.get(pg_type, "str")


def _normalize_geometry_type(pg_type: str) -> str:
    """Normalize PostGIS geometry type names to Fiona-compatible mixed case.

    PostGIS returns UPPERCASE (e.g. MULTILINESTRING), Fiona expects
    mixed case (e.g. MultiLineString).
    """
    mapping = {
        "POINT": "Point",
        "LINESTRING": "LineString",
        "POLYGON": "Polygon",
        "MULTIPOINT": "MultiPoint",
        "MULTILINESTRING": "MultiLineString",
        "MULTIPOLYGON": "MultiPolygon",
        "GEOMETRYCOLLECTION": "GeometryCollection",
    }
    return mapping.get(pg_type.upper(), pg_type)
