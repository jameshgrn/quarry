"""GeoParquetConnector — materializes GeoParquet files into canonical artifacts.

Lane: connector

Follows the DuckDB connector pattern:
- source_ref: path to .parquet or .geoparquet file
- geometry vs non-geometry branching → VECTOR vs TABLE
- lazy = metadata-only with LAZY_HANDLE backing, eager = dump to GeoPackage
- Uses pyarrow for metadata reading (geo metadata from Arrow schema)
- Uses pyarrow + shapely for reading GeoParquet data
- Uses fiona for writing to GeoPackage (GPKG driver is always available)

Design decisions:
- Constructor takes optional default_path (rarely used)
- Geo metadata read from Arrow schema metadata key "geo" (JSON)
- Eager vector: read with pyarrow/shapely, dump to GeoPackage via fiona
- Non-geospatial parquet files treated as TABLE artifacts
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fiona
import shapely.io
from fiona.crs import CRS

# pyarrow is an optional dependency (install via quarry-connectors[geoparquet])
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]
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


def _is_geoparquet(path: Path) -> bool:
    """Check if a parquet file has geo metadata (is a GeoParquet file)."""
    if not HAS_PYARROW:
        return False
    try:
        pf = pq.ParquetFile(str(path))
        metadata = pf.schema_arrow.metadata
        if metadata and b"geo" in metadata:
            return True
    except (OSError, pa.ArrowInvalid, pa.ArrowIOError):
        pass
    return False


def _read_geo_metadata(path: Path) -> dict[str, Any] | None:
    """Read the geo metadata from a GeoParquet file.

    Returns the parsed JSON from the "geo" key in Arrow schema metadata,
    or None if not present or invalid.
    """
    if not HAS_PYARROW:
        return None
    try:
        pf = pq.ParquetFile(str(path))
        metadata = pf.schema_arrow.metadata
        if metadata and b"geo" in metadata:
            return json.loads(metadata[b"geo"])
    except (OSError, pa.ArrowInvalid, pa.ArrowIOError, json.JSONDecodeError):
        pass
    return None


def _extract_extent_from_geo_metadata(
    geo_meta: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    """Extract bounding box from geo metadata.

    GeoParquet spec stores bbox per column in columns[primary_column]["bbox"].
    """
    try:
        primary_column = geo_meta.get("primary_column")
        if not primary_column:
            return None
        columns = geo_meta.get("columns", {})
        col_meta = columns.get(primary_column, {})
        bbox = col_meta.get("bbox")
        if bbox and len(bbox) == 4:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _extract_crs_from_geo_metadata(geo_meta: dict[str, Any]) -> str | None:
    """Extract CRS from geo metadata.

    GeoParquet spec stores CRS in columns[primary_column]["crs"] or
    a top-level "crs" field.
    """
    try:
        primary_column = geo_meta.get("primary_column")
        columns = geo_meta.get("columns", {})
        col_meta = columns.get(primary_column, {}) if primary_column else {}
        # Try column-level CRS first
        crs = col_meta.get("crs")
        if crs:
            # CRS can be a dict with PROJJSON structure or a string
            if isinstance(crs, dict):
                # Try to extract EPSG code from PROJJSON
                if "id" in crs and crs["id"].get("authority") == "EPSG":
                    return f"EPSG:{crs['id']['code']}"
            return str(crs)
        # Fall back to top-level CRS
        crs = geo_meta.get("crs")
        if crs:
            return str(crs)
    except (KeyError, TypeError):
        pass
    return None


def _extract_srid_from_crs(crs_str: str | None) -> int | None:
    """Extract SRID integer from CRS string like 'EPSG:4326'."""
    if not crs_str:
        return None
    try:
        if "EPSG:" in crs_str.upper():
            code = crs_str.split(":")[-1]
            return int(code)
    except (ValueError, AttributeError):
        pass
    return None


class GeoParquetConnector:
    """Materializes GeoParquet files into canonical Quarry artifacts."""

    def __init__(self, default_path: str | None = None):
        self._default_path = default_path

    @property
    def name(self) -> str:
        return "geoparquet"

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
        """Materialize a GeoParquet file into a canonical artifact.

        source_ref: path to .parquet or .geoparquet file
        """
        if not HAS_PYARROW:
            raise MaterializeError(
                source_ref,
                "pyarrow is required for GeoParquet. "
                "Install with: uv pip install quarry-connectors[geoparquet]",
            )
        path = self._parse_source_ref(source_ref)

        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        # Check if it's a valid parquet file and has geo metadata
        is_geo = _is_geoparquet(path)
        artifact_type = ArtifactType.VECTOR if is_geo else ArtifactType.TABLE

        # Read parquet metadata
        try:
            pf = pq.ParquetFile(str(path))
            num_rows = pf.metadata.num_rows
            row_groups = pf.metadata.num_row_groups
            schema = pf.schema_arrow
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read parquet metadata: {e}") from e

        # Read geo metadata if present
        geo_meta = _read_geo_metadata(path) if is_geo else None

        lineage_params: dict[str, Any] = {
            "source": "geoparquet",
            "path": str(path),
            "lazy": lazy,
        }

        if lazy:
            # Lazy mode: metadata only, LAZY_HANDLE backing
            extent = _extract_extent_from_geo_metadata(geo_meta) if geo_meta else None
            crs = _extract_crs_from_geo_metadata(geo_meta) if geo_meta else None

            spatial = SpatialDescriptor(
                crs=crs,
                extent=extent,
                feature_count=num_rows,
            )

            artifact = Artifact(
                type=artifact_type,
                name=path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=f"geoparquet://{path}",
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=self._artifact_metadata(path, schema, geo_meta, row_groups, is_geo),
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"GeoParquet metadata only — {num_rows} rows",
            )

        # Eager mode: read data and dump to GeoPackage (vector) or CSV (table)
        try:
            if is_geo:
                output_path, spatial = self._dump_vector(path, workspace, geo_meta)
            else:
                output_path = self._dump_table(path, workspace)
                spatial = SpatialDescriptor(feature_count=num_rows)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Data fetch failed: {e}") from e

        artifact = Artifact(
            type=artifact_type,
            name=path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(output_path),
                size_bytes=output_path.stat().st_size,
                content_hash=content_hash(output_path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata=self._artifact_metadata(path, schema, geo_meta, row_groups, is_geo),
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="wrapped_local",
            source_ref=source_ref,
            notes=f"Dumped {output_path.name} ({output_path.stat().st_size} bytes)",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """List .parquet and .geoparquet files in a directory.

        query: directory path as string or dict with "path" key
        """
        if isinstance(query, dict):
            dir_path = query.get("path", self._default_path)
        elif isinstance(query, str):
            dir_path = query
        else:
            dir_path = self._default_path

        if not dir_path:
            raise MaterializeError("discover", "No path specified")

        path = Path(dir_path)
        if not path.is_dir():
            raise MaterializeError("discover", f"Not a directory: {dir_path}")

        entries = []
        for ext in ("*.parquet", "*.geoparquet"):
            for file_path in path.glob(ext):
                is_geo = _is_geoparquet(file_path)
                entries.append(
                    CatalogEntry(
                        source_ref=str(file_path),
                        name=file_path.stem,
                        metadata={
                            "is_geoparquet": is_geo,
                            "extension": file_path.suffix,
                        },
                    )
                )

        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get file metadata without materializing data."""
        if not HAS_PYARROW:
            raise MaterializeError(
                source_ref,
                "pyarrow is required for GeoParquet. "
                "Install with: uv pip install quarry-connectors[geoparquet]",
            )
        path = self._parse_source_ref(source_ref)

        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        try:
            pf = pq.ParquetFile(str(path))
            schema = pf.schema_arrow
            num_rows = pf.metadata.num_rows
            row_groups = pf.metadata.num_row_groups
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read parquet metadata: {e}") from e

        is_geo = _is_geoparquet(path)
        geo_meta = _read_geo_metadata(path) if is_geo else None

        columns = [{"name": name, "type": str(schema.field(name).type)} for name in schema.names]

        crs = _extract_crs_from_geo_metadata(geo_meta) if geo_meta else None
        extent = _extract_extent_from_geo_metadata(geo_meta) if geo_meta else None

        return {
            "columns": columns,
            "crs": crs,
            "extent": extent,
            "feature_count": num_rows,
            "geo_metadata": geo_meta,
            "row_groups": row_groups,
            "is_geoparquet": is_geo,
        }

    # -----------------------------------------------------------------------
    # Source ref parsing
    # -----------------------------------------------------------------------

    def _parse_source_ref(self, source_ref: SourceRef | str) -> Path:
        """Parse source_ref into a Path."""
        from quarry_core.source_ref import SourceRef

        if isinstance(source_ref, SourceRef):
            params = source_ref.params or {}
            path_str = params.get("path")
            if not path_str:
                # Try to extract from raw
                path_str = source_ref.raw
        else:
            path_str = source_ref.strip()

        return Path(path_str)

    # -----------------------------------------------------------------------
    # Eager dump: vector → GeoPackage
    # -----------------------------------------------------------------------

    def _dump_vector(
        self, path: Path, workspace: Path, geo_meta: dict[str, Any] | None
    ) -> tuple[Path, SpatialDescriptor]:
        """Dump GeoParquet to GeoPackage using pyarrow + shapely + fiona.

        Returns (output_path, spatial_descriptor).
        """
        output_path = workspace / f"{path.stem}.gpkg"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read parquet data
        table = pq.read_table(str(path))

        # Get geometry column name from metadata
        geom_col = "geometry"
        if geo_meta:
            geom_col = geo_meta.get("primary_column", "geometry")

        # Get column names excluding geometry
        col_names = [name for name in table.column_names if name != geom_col]

        # Build fiona schema
        properties = {}
        for col_name in col_names:
            field = table.schema.field(col_name)
            properties[col_name] = _arrow_type_to_fiona(field.type)

        # Determine geometry type from metadata or infer from data
        geom_type = "Unknown"
        if geo_meta:
            columns = geo_meta.get("columns", {})
            col_meta = columns.get(geom_col, {})
            geom_types = col_meta.get("geometry_types")
            if geom_types and len(geom_types) > 0:
                geom_type = _normalize_geometry_type(geom_types[0])

        # Get CRS — never silently default to EPSG:4326
        crs_str = _extract_crs_from_geo_metadata(geo_meta)
        srid = _extract_srid_from_crs(crs_str)
        if srid is not None:
            crs = CRS.from_epsg(srid)
        elif crs_str:
            try:
                crs = CRS.from_user_input(crs_str)
            except (fiona.errors.CRSError, ValueError):
                crs = None  # CRS present but unparseable — omit rather than guess
        else:
            crs = None

        # Prepare fiona schema
        fiona_schema = {"geometry": geom_type, "properties": properties}

        # Compute extent while writing (vectorized per batch via shapely.total_bounds)
        xmin, ymin, xmax, ymax = float("inf"), float("inf"), float("-inf"), float("-inf")
        feature_count = 0

        with fiona.open(
            str(output_path),
            "w",
            driver="GPKG",
            schema=fiona_schema,
            crs=crs,
        ) as dst:
            for batch in table.to_batches():
                geom_array = batch.column(geom_col)
                wkb_values = geom_array.to_pylist()

                # Decode all WKB in batch at once, skip nulls
                valid_wkb = [w for w in wkb_values if w is not None]
                if valid_wkb:
                    geoms = shapely.io.from_wkb(valid_wkb)
                    batch_bounds = shapely.total_bounds(geoms)
                    xmin = min(xmin, batch_bounds[0])
                    ymin = min(ymin, batch_bounds[1])
                    xmax = max(xmax, batch_bounds[2])
                    ymax = max(ymax, batch_bounds[3])

                # Get property columns
                prop_arrays = {name: batch.column(name).to_pylist() for name in col_names}

                # Write features using pre-parsed geometries from the batch
                geom_idx = 0
                for i, wkb in enumerate(wkb_values):
                    if wkb is None:
                        continue

                    geom = geoms[geom_idx]
                    geom_idx += 1

                    props = {name: prop_arrays[name][i] for name in col_names}
                    dst.write({"geometry": geom.__geo_interface__, "properties": props})
                    feature_count += 1

        extent = (xmin, ymin, xmax, ymax) if feature_count > 0 else None
        spatial = SpatialDescriptor(
            crs=crs_str,
            extent=extent,
            feature_count=feature_count,
        )

        return output_path, spatial

    # -----------------------------------------------------------------------
    # Eager dump: table → CSV
    # -----------------------------------------------------------------------

    def _dump_table(self, path: Path, workspace: Path) -> Path:
        """Dump non-spatial parquet to CSV using pyarrow."""
        output_path = workspace / f"{path.stem}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(str(path))

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(table.column_names)
            # Write rows
            for batch in table.to_batches():
                for row in zip(*[batch.column(i).to_pylist() for i in range(batch.num_columns)]):
                    writer.writerow(row)

        return output_path

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _artifact_metadata(
        self,
        path: Path,
        schema: Any,
        geo_meta: dict[str, Any] | None,
        row_groups: int,
        is_geo: bool,
    ) -> dict[str, Any]:
        """Build artifact metadata dict."""
        meta: dict[str, Any] = {
            "source": "geoparquet",
            "path": str(path),
            "is_geoparquet": is_geo,
            "row_groups": row_groups,
        }
        if geo_meta:
            meta["geo_metadata"] = geo_meta
            primary_column = geo_meta.get("primary_column")
            if primary_column:
                meta["geometry_column"] = primary_column
                columns = geo_meta.get("columns", {})
                col_meta = columns.get(primary_column, {})
                geom_types = col_meta.get("geometry_types")
                if geom_types:
                    meta["geometry_types"] = geom_types
        return meta


def _arrow_type_to_fiona(arrow_type: pa.DataType) -> str:
    """Map Arrow data types to fiona property types."""
    if pa.types.is_integer(arrow_type):
        return "int"
    if pa.types.is_floating(arrow_type):
        return "float"
    if pa.types.is_boolean(arrow_type):
        return "bool"
    if pa.types.is_date(arrow_type):
        return "date"
    if pa.types.is_timestamp(arrow_type):
        return "datetime"
    return "str"


def _normalize_geometry_type(geo_type: str) -> str:
    """Normalize GeoParquet geometry type names to Fiona-compatible mixed case."""
    mapping = {
        "Point": "Point",
        "LineString": "LineString",
        "Polygon": "Polygon",
        "MultiPoint": "MultiPoint",
        "MultiLineString": "MultiLineString",
        "MultiPolygon": "MultiPolygon",
        "GeometryCollection": "GeometryCollection",
        "Geometry": "Unknown",
    }
    return mapping.get(geo_type, "Unknown")
