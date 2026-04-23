"""OvertureConnector — materializes Overture Maps Foundation data via DuckDB.

Lane: connector

Downloads and materializes Overture Maps data (buildings, roads, places, etc.)
from remote Parquet files using DuckDB + httpfs extension.

Design decisions:
- Source ref: overture://theme/type with optional bbox in params
- Lazy mode: validate theme/type, produce LAZY_HANDLE with S3 URL — no DuckDB/HTTP
- Eager mode: DuckDB in-memory + httpfs → read remote Parquet → dump to GeoPackage
- Geometry is WKB bytes in 'geometry' column; bbox struct enables spatial pushdown
- max_rows safety limit prevents accidental full-table scans
- _parquet_source() method enables test injection of local data
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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

# S3 bucket and default release
_S3_BUCKET = "overturemaps-us-west-2"
_DEFAULT_RELEASE = "2024-12-18.0"

# Known Overture themes → list of types
KNOWN_THEMES: dict[str, list[str]] = {
    "buildings": ["building", "building_part"],
    "transportation": ["segment", "connector"],
    "places": ["place"],
    "base": ["land", "water", "infrastructure", "land_cover", "land_use"],
    "divisions": ["division", "division_area", "division_boundary"],
    "addresses": ["address"],
}

# Flat set of all valid (theme, type) pairs for fast lookup
_VALID_PAIRS: set[tuple[str, str]] = {
    (theme, typ) for theme, types in KNOWN_THEMES.items() for typ in types
}


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier to prevent injection."""
    return '"' + name.replace('"', '""') + '"'


class OvertureConnector:
    """Materializes Overture Maps Foundation data into canonical Quarry artifacts.

    Uses DuckDB + httpfs to read remote Parquet files from S3.
    """

    def __init__(
        self,
        release: str = _DEFAULT_RELEASE,
        max_rows: int = 10000,
    ):
        self._release = release
        self._max_rows = max_rows

    @property
    def name(self) -> str:
        return "overture"

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
        """Materialize Overture Maps data into a canonical artifact.

        source_ref formats:
            "overture://buildings/building"     — theme/type
            SourceRef with params: {"theme": ..., "type": ..., "bbox": (w,s,e,n)}
        """
        theme, typ, bbox = self._parse_source_ref(source_ref)
        self._validate_theme_type(source_ref, theme, typ)

        lineage_params: dict[str, Any] = {
            "source": "overture",
            "theme": theme,
            "type": typ,
            "release": self._release,
            "bbox": bbox,
            "lazy": lazy,
            "max_rows": self._max_rows,
        }

        if lazy:
            s3_url = self._s3_url(theme, typ)
            spatial = SpatialDescriptor(
                crs="EPSG:4326",
                extent=bbox,
            )
            artifact = Artifact(
                type=ArtifactType.VECTOR,
                name=f"{theme}_{typ}",
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=s3_url,
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata={
                    "source": "overture",
                    "theme": theme,
                    "type": typ,
                    "release": self._release,
                },
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"Overture {theme}/{typ} — metadata only",
            )

        # Eager: query via DuckDB + httpfs, dump to GeoPackage
        output_path, row_count, extent = self._fetch_and_dump(
            theme, typ, bbox, workspace, source_ref
        )

        spatial = SpatialDescriptor(
            crs="EPSG:4326",
            extent=extent,
            feature_count=row_count,
        )

        artifact = Artifact(
            type=ArtifactType.VECTOR,
            name=f"{theme}_{typ}",
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(output_path),
                size_bytes=output_path.stat().st_size,
                content_hash=content_hash(output_path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata={
                "source": "overture",
                "theme": theme,
                "type": typ,
                "release": self._release,
                "row_count": row_count,
            },
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="fetched_remote",
            source_ref=source_ref,
            notes=f"Dumped {output_path.name} ({row_count} features, "
            f"{output_path.stat().st_size} bytes)",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """List available Overture themes/types.

        query as str: filter by theme name.
        query as dict: {"theme": str} to filter.
        """
        theme_filter = None
        if isinstance(query, str):
            theme_filter = query
        elif isinstance(query, dict):
            theme_filter = query.get("theme")

        entries: list[CatalogEntry] = []
        for theme, types in KNOWN_THEMES.items():
            if theme_filter and theme != theme_filter:
                continue
            for typ in types:
                entries.append(
                    CatalogEntry(
                        source_ref=f"overture://{theme}/{typ}",
                        name=f"{theme}/{typ}",
                        description=f"Overture Maps {theme} — {typ}",
                        spatial_hint={"crs": "EPSG:4326", "coverage": "global"},
                        metadata={
                            "theme": theme,
                            "type": typ,
                            "release": self._release,
                        },
                    )
                )
        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get metadata for an Overture theme/type without materializing."""
        theme, typ, _ = self._parse_source_ref(source_ref)
        self._validate_theme_type(source_ref, theme, typ)

        return {
            "theme": theme,
            "type": typ,
            "release": self._release,
            "s3_url": self._s3_url(theme, typ),
            "crs": "EPSG:4326",
            "coverage": "global",
        }

    # -----------------------------------------------------------------------
    # Source ref parsing
    # -----------------------------------------------------------------------

    def _parse_source_ref(
        self, source_ref: SourceRef | str
    ) -> tuple[str, str, tuple[float, float, float, float] | None]:
        """Parse source_ref into (theme, type, bbox).

        Returns:
            (theme, type, bbox) where bbox is (west, south, east, north) or None.
        """
        from quarry_core.source_ref import SourceRef

        if isinstance(source_ref, SourceRef):
            params = dict(source_ref.params) if source_ref.params else {}

            # Try params first
            theme = params.get("theme")
            typ = params.get("type")
            bbox = self._extract_bbox(params)

            # Fall back to parsing raw
            if not theme or not typ:
                parsed_theme, parsed_typ, parsed_bbox = self._parse_overture_url(source_ref.raw)
                theme = theme or parsed_theme
                typ = typ or parsed_typ
                bbox = bbox or parsed_bbox

            if not theme or not typ:
                raise MaterializeError(
                    source_ref,
                    "Cannot parse Overture source_ref: need theme and type. "
                    "Use 'overture://theme/type' or params={theme, type}.",
                )
            return theme, typ, bbox

        # Raw string
        theme, typ, bbox = self._parse_overture_url(source_ref.strip())
        if not theme or not typ:
            raise MaterializeError(
                source_ref,
                "Cannot parse Overture source_ref: expected 'overture://theme/type'.",
            )
        return theme, typ, bbox

    def _parse_overture_url(
        self, raw: str
    ) -> tuple[str | None, str | None, tuple[float, float, float, float] | None]:
        """Parse 'overture://theme/type' format."""
        if not raw.startswith("overture://"):
            return None, None, None

        path = raw[len("overture://") :]
        if "/" not in path:
            return path, None, None

        parts = path.split("/", 1)
        return parts[0], parts[1], None

    @staticmethod
    def _extract_bbox(
        params: dict[str, Any],
    ) -> tuple[float, float, float, float] | None:
        """Extract bbox from params dict."""
        bbox = params.get("bbox")
        if bbox:
            try:
                w, s, e, n = bbox
                return (float(w), float(s), float(e), float(n))
            except (ValueError, TypeError):
                pass
        return None

    def _validate_theme_type(self, source_ref: SourceRef | str, theme: str, typ: str) -> None:
        """Validate theme/type against known Overture themes."""
        if theme not in KNOWN_THEMES:
            raise MaterializeError(
                source_ref,
                f"Unknown Overture theme '{theme}'. Known: {sorted(KNOWN_THEMES.keys())}",
            )
        if (theme, typ) not in _VALID_PAIRS:
            raise MaterializeError(
                source_ref,
                f"Unknown type '{typ}' for theme '{theme}'. Known types: {KNOWN_THEMES[theme]}",
            )

    # -----------------------------------------------------------------------
    # S3 URL construction
    # -----------------------------------------------------------------------

    def _s3_url(self, theme: str, typ: str) -> str:
        """Build S3 URL for Overture Parquet files."""
        return f"s3://{_S3_BUCKET}/release/{self._release}/theme={theme}/type={typ}/*"

    # -----------------------------------------------------------------------
    # Parquet source (overridable for testing)
    # -----------------------------------------------------------------------

    def _parquet_source(self, theme: str, typ: str) -> str:
        """Return the parquet source expression for DuckDB.

        Override in tests to point at local data instead of S3.
        """
        s3_url = self._s3_url(theme, typ)
        return f"read_parquet('{s3_url}', filename=true, hive_partitioning=true)"

    # -----------------------------------------------------------------------
    # Eager fetch + dump
    # -----------------------------------------------------------------------

    def _fetch_and_dump(
        self,
        theme: str,
        typ: str,
        bbox: tuple[float, float, float, float] | None,
        workspace: Path,
        source_ref: SourceRef | str,
    ) -> tuple[Path, int, tuple[float, float, float, float] | None]:
        """Fetch Overture data via DuckDB and dump to GeoPackage.

        Returns (output_path, row_count, extent).
        """
        import fiona
        import shapely.wkb
        from fiona.crs import CRS

        try:
            conn = duckdb.connect(":memory:")
            try:
                self._setup_extensions(conn)
                parquet_src = self._parquet_source(theme, typ)

                # Build query with bbox filter and row limit
                where_clauses = []
                if bbox:
                    w, s, e, n = bbox
                    where_clauses.append(
                        f"bbox.xmin >= {w} AND bbox.xmax <= {e} "
                        f"AND bbox.ymin >= {s} AND bbox.ymax <= {n}"
                    )

                where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                query = f"SELECT * FROM {parquet_src}{where_sql} LIMIT {self._max_rows}"

                # Introspect columns
                desc = conn.execute(f"DESCRIBE ({query})").fetchall()
                columns = [{"name": row[0], "type": row[1]} for row in desc]
                col_names = [c["name"] for c in columns]

                # Identify geometry column
                geom_col = "geometry" if "geometry" in col_names else None
                if not geom_col:
                    raise MaterializeError(
                        source_ref,
                        "No 'geometry' column found in Overture data.",
                    )

                # Build WKB fetch query
                non_geom_cols = [c for c in col_names if c != geom_col]
                select_parts = [_quote_ident(c) for c in non_geom_cols]
                select_parts.append(
                    f"ST_AsWKB(ST_GeomFromWKB({_quote_ident(geom_col)})) AS _geom_wkb"
                )
                select_sql = ", ".join(select_parts)

                wkb_query = (
                    f"SELECT {select_sql} FROM {parquet_src}{where_sql} LIMIT {self._max_rows}"
                )
                rows = conn.execute(wkb_query).fetchall()

            finally:
                conn.close()
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"DuckDB query failed: {e}") from e

        if not rows:
            raise MaterializeError(
                source_ref,
                f"No features returned for {theme}/{typ}"
                + (f" within bbox {bbox}" if bbox else ""),
            )

        # Build fiona schema from non-geometry columns
        properties: dict[str, str] = {}
        for col in columns:
            if col["name"] == geom_col:
                continue
            properties[col["name"]] = "str"  # safe default for Overture's complex types

        output_path = workspace / f"{theme}_{typ}.gpkg"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        crs = CRS.from_epsg(4326)

        # Detect geometry type from first row
        first_geom_wkb = None
        for row in rows:
            if row[-1] is not None:
                raw = row[-1]
                first_geom_wkb = bytes(raw) if not isinstance(raw, bytes) else raw
                break
        geom_type = "Unknown"
        if first_geom_wkb:
            first_geom = shapely.wkb.loads(first_geom_wkb)
            geom_type = first_geom.geom_type

        fiona_schema = {"geometry": geom_type, "properties": properties}

        # Track extent
        xmin_acc, ymin_acc, xmax_acc, ymax_acc = (
            float("inf"),
            float("inf"),
            float("-inf"),
            float("-inf"),
        )
        written = 0

        with fiona.open(output_path, "w", driver="GPKG", schema=fiona_schema, crs=crs) as dst:
            for row in rows:
                geom_wkb = row[-1]  # _geom_wkb is last
                if geom_wkb is None:
                    continue

                geom = shapely.wkb.loads(
                    bytes(geom_wkb) if not isinstance(geom_wkb, bytes) else geom_wkb
                )

                # Update extent
                bounds = geom.bounds  # (minx, miny, maxx, maxy)
                xmin_acc = min(xmin_acc, bounds[0])
                ymin_acc = min(ymin_acc, bounds[1])
                xmax_acc = max(xmax_acc, bounds[2])
                ymax_acc = max(ymax_acc, bounds[3])

                props = {}
                for i, col in enumerate(columns):
                    if col["name"] == geom_col:
                        continue
                    col_idx = non_geom_cols.index(col["name"])
                    val = row[col_idx]
                    # Stringify complex types (structs, lists)
                    props[col["name"]] = str(val) if val is not None else None

                dst.write(
                    {
                        "geometry": geom.__geo_interface__,
                        "properties": props,
                    }
                )
                written += 1

        extent = None
        if written > 0:
            extent = (xmin_acc, ymin_acc, xmax_acc, ymax_acc)

        return output_path, written, extent

    @staticmethod
    def _setup_extensions(conn: duckdb.DuckDBPyConnection) -> None:
        """Install and load httpfs + spatial extensions."""
        try:
            conn.execute("INSTALL httpfs")
            conn.execute("LOAD httpfs")
            conn.execute("SET s3_region='us-west-2'")
        except Exception:
            pass  # httpfs may already be loaded or unavailable for tests
        try:
            conn.execute("INSTALL spatial")
            conn.execute("LOAD spatial")
        except Exception:
            pass  # spatial may already be loaded
