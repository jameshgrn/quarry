"""DuckDBConnector — materializes DuckDB tables/queries into canonical artifacts.

Lane: connector

Follows the PostGIS connector pattern:
- source_ref: path.duckdb::table or path.duckdb::SELECT ...
- geometry vs non-geometry branching → VECTOR vs TABLE
- lazy = metadata-only with DUCKDB backing, eager = dump to GeoPackage/CSV
- spatial extension loaded gracefully — non-spatial tables always work

Design decisions:
- Constructor takes optional db_path (overridden by source_ref params)
- :: separator between db_path and table/query (mirrors STAC :: convention)
- Spatial extension: attempted on connect, not required
- Eager vector: dump to GeoPackage via fiona (WKB → shapely → geo_interface)
- Eager table: dump to CSV (stdlib)
- Lazy: DUCKDB BackingStore with db_path + table URI
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import duckdb
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

# DuckDB column types that indicate geometry when spatial extension is loaded.
_GEOMETRY_TYPES = frozenset(
    {
        "GEOMETRY",
        "POINT",
        "LINESTRING",
        "POLYGON",
        "MULTIPOINT",
        "MULTILINESTRING",
        "MULTIPOLYGON",
        "GEOMETRYCOLLECTION",
        "POINT_2D",
        "LINESTRING_2D",
        "POLYGON_2D",
        "BOX_2D",
        "WKB_BLOB",
    }
)


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier to prevent injection.

    Doubles any embedded double-quotes, then wraps in double-quotes.
    Standard SQL identifier quoting.
    """
    return '"' + name.replace('"', '""') + '"'


@dataclass
class IntrospectionResult:
    """Introspection of a DuckDB table or query."""

    columns: list[dict[str, Any]]
    geometry_column: str | None
    geometry_type: str | None
    srid: int | None
    extent: tuple[float, float, float, float] | None
    row_count: int
    kind: Literal["table", "query"]
    has_spatial: bool


class DuckDBConnector:
    """Materializes DuckDB tables and queries into canonical Quarry artifacts.

    Configured with an optional default db_path. source_ref params override it.
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path

    @property
    def name(self) -> str:
        return "duckdb"

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
        """Materialize a DuckDB table or query into a canonical artifact.

        source_ref formats:
            "path/to/file.duckdb::table_name"      — table from a DuckDB file
            "path/to/file.duckdb::SELECT ..."       — query against a DuckDB file
        """
        db_path, table, query = self._parse_source_ref(source_ref)

        try:
            conn = duckdb.connect(db_path, read_only=True)
            try:
                has_spatial = self._try_load_spatial(conn)
                introspection = self._introspect(conn, table, query, has_spatial)
            finally:
                conn.close()
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, str(e)) from e

        has_geometry = introspection.geometry_column is not None
        artifact_type = ArtifactType.VECTOR if has_geometry else ArtifactType.TABLE

        spatial = SpatialDescriptor(
            crs=f"EPSG:{introspection.srid}" if introspection.srid else None,
            extent=introspection.extent,
            feature_count=introspection.row_count,
        )

        lineage_params: dict[str, Any] = {
            "source": "duckdb",
            "db_path": db_path,
            "lazy": lazy,
        }
        if query:
            lineage_params["query"] = query
        else:
            lineage_params["table"] = table

        if lazy:
            backing_uri = self._backing_uri(db_path, table, query)
            artifact = Artifact(
                type=artifact_type,
                name=table or "query_result",
                backing=BackingStore(
                    kind=BackingStoreKind.DUCKDB,
                    uri=backing_uri,
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=self._artifact_metadata(introspection, db_path),
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"DuckDB {'query' if query else table} — metadata only",
            )

        # Eager: fetch data and dump to local file
        try:
            conn = duckdb.connect(db_path, read_only=True)
            try:
                if has_spatial:
                    self._try_load_spatial(conn)
                if has_geometry:
                    output_path = self._dump_vector(conn, table, query, introspection, workspace)
                else:
                    output_path = self._dump_table(conn, table, query, introspection, workspace)
            finally:
                conn.close()
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
            metadata=self._artifact_metadata(introspection, db_path),
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="fetched_local",
            source_ref=source_ref,
            notes=f"Dumped {output_path.name} ({output_path.stat().st_size} bytes)",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """List tables in the DuckDB database.

        query as dict supports:
            db_path: str (override default db_path)
        """
        db_path = self._db_path
        if isinstance(query, dict):
            db_path = query.get("db_path", self._db_path)
        elif isinstance(query, str):
            db_path = query

        if not db_path:
            raise MaterializeError("discover", "No db_path specified")

        conn = duckdb.connect(db_path, read_only=True)
        try:
            has_spatial = self._try_load_spatial(conn)
            rows = conn.execute("SHOW TABLES").fetchall()

            entries = []
            for (table_name,) in rows:
                geom_info = self._detect_geometry_column(conn, table_name, has_spatial)
                entries.append(
                    CatalogEntry(
                        source_ref=f"{db_path}::{table_name}",
                        name=table_name,
                        metadata={
                            "geometry_column": geom_info[0] if geom_info else None,
                            "geometry_type": geom_info[1] if geom_info else None,
                        },
                    )
                )
        finally:
            conn.close()

        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get table/query metadata without materializing data."""
        db_path, table, query = self._parse_source_ref(source_ref)

        conn = duckdb.connect(db_path, read_only=True)
        try:
            has_spatial = self._try_load_spatial(conn)
            introspection = self._introspect(conn, table, query, has_spatial)
        finally:
            conn.close()

        return {
            "columns": introspection.columns,
            "geometry_column": introspection.geometry_column,
            "geometry_type": introspection.geometry_type,
            "srid": introspection.srid,
            "row_count": introspection.row_count,
            "extent": introspection.extent,
            "has_spatial": introspection.has_spatial,
        }

    # -----------------------------------------------------------------------
    # Source ref parsing
    # -----------------------------------------------------------------------

    def _parse_source_ref(self, source_ref: SourceRef | str) -> tuple[str, str | None, str | None]:
        """Parse source_ref into (db_path, table, query).

        Returns:
            (db_path, table, None) for table references
            (db_path, None, query) for SQL queries
        """
        from quarry_core.source_ref import SourceRef, SourceRefKind

        if isinstance(source_ref, SourceRef):
            if source_ref.kind == SourceRefKind.DUCKDB:
                params = source_ref.params or {}
                db_path = params.get("db_path", self._db_path)
                if not db_path:
                    raise MaterializeError(source_ref, "No db_path in SourceRef or connector")
                if "query" in params:
                    return (db_path, None, params["query"])
                return (db_path, params.get("table"), None)
            raw = source_ref.raw.strip()
        else:
            raw = source_ref.strip()

        # Parse :: separator
        if "::" in raw:
            db_part, ref_part = raw.split("::", 1)
            ref_part = ref_part.strip()
            if ref_part.upper().startswith("SELECT "):
                return (db_part, None, ref_part)
            return (db_part, ref_part, None)

        # Fallback: use connector's default db_path, treat raw as table name
        if self._db_path:
            return (self._db_path, raw, None)

        raise MaterializeError(
            source_ref,
            "Cannot parse DuckDB source_ref: expected 'path.duckdb::table'"
            " or 'path.duckdb::SELECT ...'",
        )

    # -----------------------------------------------------------------------
    # Spatial extension
    # -----------------------------------------------------------------------

    @staticmethod
    def _try_load_spatial(conn: duckdb.DuckDBPyConnection) -> bool:
        """Try to load the spatial extension. Returns True if available."""
        try:
            conn.execute("LOAD spatial")
            return True
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def _introspect(
        self,
        conn: duckdb.DuckDBPyConnection,
        table: str | None,
        query: str | None,
        has_spatial: bool,
    ) -> IntrospectionResult:
        if query:
            return self._introspect_query(conn, query, has_spatial)
        if table:
            return self._introspect_table(conn, table, has_spatial)
        raise MaterializeError("introspect", "Neither table nor query specified")

    def _introspect_table(
        self,
        conn: duckdb.DuckDBPyConnection,
        table: str,
        has_spatial: bool,
    ) -> IntrospectionResult:
        # Get columns via DESCRIBE
        tq = _quote_ident(table)
        try:
            desc = conn.execute(f"DESCRIBE {tq}").fetchall()
        except Exception as e:
            raise MaterializeError(table, f"Table not found or cannot describe: {e}") from e

        columns = [{"name": row[0], "data_type": row[1]} for row in desc]

        # Detect geometry
        geom_info = self._detect_geometry_column(conn, table, has_spatial)
        geometry_column = geom_info[0] if geom_info else None
        geometry_type = geom_info[1] if geom_info else None

        # Row count
        count_row = conn.execute(f"SELECT COUNT(*) FROM {tq}").fetchone()
        row_count = count_row[0] if count_row else 0

        # SRID and extent (if spatial)
        srid = None
        extent = None
        if geometry_column and has_spatial:
            srid = self._get_srid(conn, table, geometry_column)
            extent = self._get_extent(conn, table, geometry_column)

        return IntrospectionResult(
            columns=columns,
            geometry_column=geometry_column,
            geometry_type=geometry_type,
            srid=srid,
            extent=extent,
            row_count=row_count,
            kind="table",
            has_spatial=has_spatial,
        )

    def _introspect_query(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        has_spatial: bool,
    ) -> IntrospectionResult:
        # Run with LIMIT 0 to get column info
        try:
            result = conn.execute(f"SELECT * FROM ({query}) sub LIMIT 0")
            columns = [{"name": desc[0], "data_type": desc[1]} for desc in result.description]
        except Exception as e:
            raise MaterializeError(query[:80], f"Query introspection failed: {e}") from e

        return IntrospectionResult(
            columns=columns,
            geometry_column=None,
            geometry_type=None,
            srid=None,
            extent=None,
            row_count=0,
            kind="query",
            has_spatial=has_spatial,
        )

    def _detect_geometry_column(
        self,
        conn: duckdb.DuckDBPyConnection,
        table: str,
        has_spatial: bool,
    ) -> tuple[str, str] | None:
        """Detect geometry column by type. Returns (column_name, geometry_type) or None."""
        if not has_spatial:
            return None

        tq = _quote_ident(table)
        try:
            desc = conn.execute(f"DESCRIBE {tq}").fetchall()
        except Exception:
            return None

        for row in desc:
            col_name, col_type = row[0], row[1].upper()
            if col_type in _GEOMETRY_TYPES:
                return (col_name, col_type)

        return None

    @staticmethod
    def _get_srid(
        conn: duckdb.DuckDBPyConnection,
        table: str,
        geom_col: str,
    ) -> int | None:
        """Extract SRID from the first non-null geometry."""
        tq, gq = _quote_ident(table), _quote_ident(geom_col)
        try:
            row = conn.execute(
                f"SELECT ST_SRID({gq}) FROM {tq} WHERE {gq} IS NOT NULL LIMIT 1"
            ).fetchone()
            return row[0] if row and row[0] else None
        except Exception:
            return None

    @staticmethod
    def _get_extent(
        conn: duckdb.DuckDBPyConnection,
        table: str,
        geom_col: str,
    ) -> tuple[float, float, float, float] | None:
        """Compute extent of all geometries.

        Uses per-row ST_XMin/XMax/YMin/YMax with MIN/MAX aggregation
        because ST_Extent returns BOX_2D which ST_XMin doesn't accept.
        """
        tq, gq = _quote_ident(table), _quote_ident(geom_col)
        try:
            row = conn.execute(
                f"SELECT MIN(ST_XMin({gq})), MIN(ST_YMin({gq})), "
                f"MAX(ST_XMax({gq})), MAX(ST_YMax({gq})) "
                f"FROM {tq} WHERE {gq} IS NOT NULL"
            ).fetchone()
            if row and row[0] is not None:
                return (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        except Exception:
            pass
        return None

    # -----------------------------------------------------------------------
    # Eager dump: vector → GeoPackage
    # -----------------------------------------------------------------------

    def _dump_vector(
        self,
        conn: duckdb.DuckDBPyConnection,
        table: str | None,
        query: str | None,
        introspection: IntrospectionResult,
        workspace: Path,
    ) -> Path:
        import fiona
        import shapely.wkb
        from fiona.crs import CRS

        geom_col = introspection.geometry_column
        gq = _quote_ident(geom_col)
        fetch_sql = query if query else f"SELECT * FROM {_quote_ident(table)}"

        # Fetch with geometry as WKB
        non_geom_cols = [c["name"] for c in introspection.columns if c["name"] != geom_col]
        select_cols = ", ".join(_quote_ident(c) for c in non_geom_cols)
        if select_cols:
            wkb_sql = f"SELECT {select_cols}, ST_AsWKB({gq}) AS _geom_wkb FROM ({fetch_sql}) _src"
        else:
            wkb_sql = f"SELECT ST_AsWKB({gq}) AS _geom_wkb FROM ({fetch_sql}) _src"

        rows = conn.execute(wkb_sql).fetchall()

        # Build fiona schema
        properties = {}
        for col in introspection.columns:
            if col["name"] == geom_col:
                continue
            properties[col["name"]] = _duckdb_type_to_fiona(col.get("data_type", "VARCHAR"))

        geom_type = _normalize_geometry_type(introspection.geometry_type or "Geometry")
        srid = introspection.srid or 4326

        output_name = table or "query_result"
        output_path = workspace / f"{output_name}.gpkg"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fiona_schema = {"geometry": geom_type, "properties": properties}
        crs = CRS.from_epsg(srid)

        with fiona.open(output_path, "w", driver="GPKG", schema=fiona_schema, crs=crs) as dst:
            for row in rows:
                # Last column is _geom_wkb
                geom_wkb = row[-1]
                if geom_wkb is None:
                    continue
                geom = shapely.wkb.loads(
                    bytes(geom_wkb) if not isinstance(geom_wkb, bytes) else geom_wkb
                )

                props = {}
                for i, col in enumerate(introspection.columns):
                    if col["name"] == geom_col:
                        continue
                    # Map column position to row position (geom_col excluded from select)
                    col_idx = non_geom_cols.index(col["name"])
                    props[col["name"]] = row[col_idx]

                dst.write(
                    {
                        "geometry": geom.__geo_interface__,
                        "properties": props,
                    }
                )

        return output_path

    # -----------------------------------------------------------------------
    # Eager dump: table → CSV
    # -----------------------------------------------------------------------

    def _dump_table(
        self,
        conn: duckdb.DuckDBPyConnection,
        table: str | None,
        query: str | None,
        introspection: IntrospectionResult,
        workspace: Path,
    ) -> Path:
        fetch_sql = query if query else f"SELECT * FROM {_quote_ident(table)}"
        rows = conn.execute(fetch_sql).fetchall()

        col_names = [c["name"] for c in introspection.columns]

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
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _backing_uri(db_path: str, table: str | None, query: str | None) -> str:
        if query:
            return f"duckdb://{db_path}?query={query[:100]}"
        return f"duckdb://{db_path}/{table}"

    @staticmethod
    def _artifact_metadata(introspection: IntrospectionResult, db_path: str) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "source": "duckdb",
            "db_path": db_path,
            "has_spatial": introspection.has_spatial,
        }
        if introspection.geometry_type:
            meta["geometry_type"] = introspection.geometry_type
        if introspection.geometry_column:
            meta["geometry_column"] = introspection.geometry_column
        if introspection.srid:
            meta["srid"] = introspection.srid
        meta["column_count"] = len(introspection.columns)
        return meta


def _duckdb_type_to_fiona(duckdb_type: str) -> str:
    """Map DuckDB data types to fiona property types."""
    upper = duckdb_type.upper()
    if upper in ("INTEGER", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT", "INT"):
        return "int"
    if upper in ("DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC"):
        return "float"
    if upper in ("BOOLEAN", "BOOL"):
        return "bool"
    if upper == "DATE":
        return "date"
    if upper in ("TIMESTAMP", "TIMESTAMP WITH TIME ZONE", "TIMESTAMPTZ"):
        return "datetime"
    return "str"


def _normalize_geometry_type(duckdb_type: str) -> str:
    """Normalize DuckDB geometry type names to Fiona-compatible mixed case."""
    mapping = {
        "POINT": "Point",
        "POINT_2D": "Point",
        "LINESTRING": "LineString",
        "LINESTRING_2D": "LineString",
        "POLYGON": "Polygon",
        "POLYGON_2D": "Polygon",
        "MULTIPOINT": "MultiPoint",
        "MULTILINESTRING": "MultiLineString",
        "MULTIPOLYGON": "MultiPolygon",
        "GEOMETRYCOLLECTION": "GeometryCollection",
        "GEOMETRY": "Unknown",
    }
    return mapping.get(duckdb_type.upper(), duckdb_type)
