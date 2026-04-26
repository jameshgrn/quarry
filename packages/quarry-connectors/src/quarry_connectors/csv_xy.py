"""CSVXYConnector — materializes CSV/TSV files with coordinate columns into vector artifacts.

Lane: connector

This connector auto-detects lat/lon columns and converts tabular data to vector artifacts.
Supports both eager (GeoPackage output) and lazy (metadata-only) materialization.

Key features:
- Auto-detect coordinate columns (lat/lon, latitude/longitude, x/y, easting/northing)
- Explicit column specification via "path/to/file.csv::lon_col,lat_col" syntax
- Delimiter detection (comma, tab, semicolon)
- CRS inference based on column names
- Skips rows with empty or non-numeric coordinates
- Falls back to TABLE artifact type when no coordinate columns detected
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fiona
from fiona.crs import CRS
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


# Column name patterns for coordinate detection (case-insensitive)
LON_PATTERNS = frozenset({"lon", "longitude", "lng", "long", "x", "easting"})
LAT_PATTERNS = frozenset({"lat", "latitude", "y", "northing"})


def _normalize_column(name: str) -> str:
    """Normalize column name for pattern matching."""
    return name.lower().strip().replace("_", "").replace("-", "")


def _detect_coordinate_columns(
    headers: list[str],
) -> tuple[str | None, str | None, str | None]:
    """Detect coordinate columns from headers.

    Returns:
        Tuple of (lon_col, lat_col, crs_hint) where crs_hint is "EPSG:4326" for lat/lon
        or None for projected coordinates.
    """
    lon_col: str | None = None
    lat_col: str | None = None
    crs_hint: str | None = None

    for header in headers:
        normalized = _normalize_column(header)

        if normalized in LON_PATTERNS:
            lon_col = header
            # Check if this is lat/lon (WGS84) or x/y (projected)
            if normalized in ("lon", "longitude", "lng", "long"):
                crs_hint = "EPSG:4326"
        elif normalized in LAT_PATTERNS:
            lat_col = header
            if normalized in ("lat", "latitude"):
                crs_hint = "EPSG:4326"

    return lon_col, lat_col, crs_hint


def _detect_delimiter(file_path: Path) -> str:
    """Detect CSV delimiter by sniffing the first 8KB."""
    sample = file_path.read_bytes()[:8192].decode("utf-8", errors="replace")

    # Try comma first (most common)
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        pass

    # Fallback: check for tab or semicolon explicitly
    tab_count = sample.count("\t")
    semi_count = sample.count(";")
    comma_count = sample.count(",")

    if tab_count > comma_count and tab_count > semi_count:
        return "\t"
    if semi_count > comma_count:
        return ";"
    return ","


def _parse_source_ref(source_ref: SourceRef | str) -> tuple[Path, str | None, str | None]:
    """Parse source_ref into (file_path, explicit_lon_col, explicit_lat_col).

    Supports:
    - "path/to/file.csv" → auto-detect columns
    - "path/to/file.csv::lon_col,lat_col" → explicit column names
    """
    raw = str(source_ref).strip()

    # Check for explicit column specification
    if "::" in raw:
        path_part, col_part = raw.rsplit("::", 1)
        if "," in col_part:
            lon_col, lat_col = col_part.split(",", 1)
            return Path(path_part), lon_col.strip(), lat_col.strip()
        # Invalid format, treat whole thing as path (will fail later)
        return Path(raw), None, None

    return Path(raw), None, None


def _is_numeric(value: str) -> bool:
    """Check if a string value is numeric."""
    if not value or not value.strip():
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def _read_csv_metadata(
    file_path: Path,
    delimiter: str | None = None,
    explicit_lon_col: str | None = None,
    explicit_lat_col: str | None = None,
) -> dict[str, Any]:
    """Read metadata from a CSV/TSV file.

    Returns dict with:
    - columns: list of column names
    - row_count: number of data rows
    - delimiter: detected or provided delimiter
    - lon_col: detected or explicit longitude column
    - lat_col: detected or explicit latitude column
    - crs_hint: "EPSG:4326" or None
    - has_coordinates: whether coordinate columns were found
    - valid_feature_count: count of rows with valid numeric coordinates
    - extent: (xmin, ymin, xmax, ymax) or None
    """
    if not file_path.exists():
        raise MaterializeError(str(file_path), f"File not found: {file_path}")

    if file_path.stat().st_size == 0:
        raise MaterializeError(str(file_path), "File is empty")

    # Detect delimiter if not provided
    if delimiter is None:
        delimiter = _detect_delimiter(file_path)

    columns: list[str] = []
    row_count = 0
    valid_feature_count = 0
    lon_col: str | None = explicit_lon_col
    lat_col: str | None = explicit_lat_col
    crs_hint: str | None = None
    extent: tuple[float, float, float, float] | None = None

    x_values: list[float] = []
    y_values: list[float] = []

    try:
        with open(file_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)

            # Read header
            try:
                headers = next(reader)
            except StopIteration:
                raise MaterializeError(str(file_path), "File has no header row")

            columns = headers

            # Detect coordinate columns if not explicitly provided
            if lon_col is None or lat_col is None:
                detected_lon, detected_lat, detected_crs = _detect_coordinate_columns(headers)
                if lon_col is None:
                    lon_col = detected_lon
                if lat_col is None:
                    lat_col = detected_lat
                crs_hint = detected_crs
            else:
                # Validate explicit columns exist
                if lon_col not in headers:
                    raise MaterializeError(
                        str(file_path), f"Explicit longitude column not found: {lon_col}"
                    )
                if lat_col not in headers:
                    raise MaterializeError(
                        str(file_path), f"Explicit latitude column not found: {lat_col}"
                    )
                # Determine CRS from explicit column names
                norm_lon = _normalize_column(lon_col)
                norm_lat = _normalize_column(lat_col)
                if norm_lon in ("lon", "longitude", "lng", "long") or norm_lat in (
                    "lat",
                    "latitude",
                ):
                    crs_hint = "EPSG:4326"

            has_coordinates = lon_col is not None and lat_col is not None

            if has_coordinates:
                lon_idx = headers.index(lon_col)
                lat_idx = headers.index(lat_col)

                for row in reader:
                    row_count += 1

                    if len(row) <= max(lon_idx, lat_idx):
                        continue  # Skip malformed rows

                    lon_val = row[lon_idx].strip()
                    lat_val = row[lat_idx].strip()

                    if _is_numeric(lon_val) and _is_numeric(lat_val):
                        try:
                            x = float(lon_val)
                            y = float(lat_val)
                            x_values.append(x)
                            y_values.append(y)
                            valid_feature_count += 1
                        except ValueError:
                            pass  # Skip non-numeric values
            else:
                # Just count rows for table data
                for _ in reader:
                    row_count += 1

    except csv.Error as e:
        raise MaterializeError(str(file_path), f"CSV parsing error: {e}") from e
    except UnicodeDecodeError as e:
        raise MaterializeError(str(file_path), f"File encoding error: {e}") from e

    # Calculate extent from valid coordinates
    if x_values and y_values:
        extent = (min(x_values), min(y_values), max(x_values), max(y_values))

    return {
        "columns": columns,
        "row_count": row_count,
        "delimiter": delimiter,
        "lon_col": lon_col,
        "lat_col": lat_col,
        "crs_hint": crs_hint,
        "has_coordinates": has_coordinates,
        "valid_feature_count": valid_feature_count,
        "extent": extent,
    }


def _write_geopackage(
    input_path: Path,
    output_path: Path,
    metadata: dict[str, Any],
    default_crs: str = "EPSG:4326",
) -> None:
    """Write CSV data to GeoPackage using fiona.

    Builds shapely Points from coordinate columns and writes via fiona.
    Skips rows with empty or non-numeric coordinates.
    """
    headers = metadata["columns"]
    delimiter = metadata["delimiter"]
    lon_col = metadata["lon_col"]
    lat_col = metadata["lat_col"]
    crs_hint = metadata["crs_hint"]

    lon_idx = headers.index(lon_col)
    lat_idx = headers.index(lat_col)

    # Determine CRS
    crs = crs_hint if crs_hint is not None else default_crs

    # Build property schema (all columns except coordinates)
    prop_schema = {}
    for i, header in enumerate(headers):
        if header == lon_col or header == lat_col:
            continue
        # Default to string type for all properties
        prop_schema[header] = "str"

    schema = {"geometry": "Point", "properties": prop_schema}

    # Parse CRS
    if crs.startswith("EPSG:"):
        crs_obj = CRS.from_epsg(int(crs.split(":")[1]))
    else:
        crs_obj = CRS.from_epsg(4326)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with fiona.open(
        str(output_path),
        "w",
        driver="GPKG",
        schema=schema,
        crs=crs_obj,
    ) as dst:
        with open(input_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            next(reader)  # Skip header

            for row in reader:
                if len(row) <= max(lon_idx, lat_idx):
                    continue

                lon_val = row[lon_idx].strip()
                lat_val = row[lat_idx].strip()

                if not _is_numeric(lon_val) or not _is_numeric(lat_val):
                    continue

                try:
                    x = float(lon_val)
                    y = float(lat_val)
                except ValueError:
                    continue

                # Build properties dict
                props = {}
                for i, header in enumerate(headers):
                    if header == lon_col or header == lat_col:
                        continue
                    if i < len(row):
                        props[header] = row[i]
                    else:
                        props[header] = ""

                # Create feature with GeoJSON geometry dict
                feature = {
                    "geometry": {
                        "type": "Point",
                        "coordinates": [x, y],
                    },
                    "properties": props,
                }
                dst.write(feature)


class CSVXYConnector:
    """Materializes CSV/TSV files with coordinate columns into canonical artifacts.

    Auto-detects lat/lon columns and converts tabular data to vector artifacts.
    Supports explicit column specification via "path/to/file.csv::lon_col,lat_col".
    """

    def __init__(self, default_crs: str = "EPSG:4326"):
        """Initialize connector with optional default CRS for projected coordinates.

        Args:
            default_crs: CRS to use for x/y/easting/northing columns (default: EPSG:4326)
        """
        self.default_crs = default_crs

    @property
    def name(self) -> str:
        return "csv_xy"

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
        """Materialize a CSV/TSV file into a canonical artifact.

        Args:
            source_ref: Path to CSV/TSV file, optionally with "::lon_col,lat_col" suffix
            workspace: Where to write output GeoPackage (eager mode)
            lazy: If True, return metadata-only artifact with LAZY_HANDLE backing

        Returns:
            MaterializeResult with the artifact and provenance.

        Raises:
            MaterializeError: If materialization fails.
        """
        file_path, explicit_lon_col, explicit_lat_col = _parse_source_ref(source_ref)

        # Read metadata
        try:
            meta = _read_csv_metadata(
                file_path,
                explicit_lon_col=explicit_lon_col,
                explicit_lat_col=explicit_lat_col,
            )
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read CSV metadata: {e}") from e

        # Determine artifact type and CRS
        has_coords = meta["has_coordinates"]
        crs = meta["crs_hint"] if meta["crs_hint"] is not None else self.default_crs

        # Build spatial descriptor
        if has_coords:
            spatial = SpatialDescriptor(
                crs=crs,
                extent=meta["extent"],
                feature_count=meta["valid_feature_count"],
            )
            artifact_type = ArtifactType.VECTOR
        else:
            spatial = SpatialDescriptor(
                crs=None,
                extent=None,
                feature_count=meta["row_count"],
            )
            artifact_type = ArtifactType.TABLE

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "csv_xy",
            "path": str(file_path),
            "lazy": lazy,
            "delimiter": meta["delimiter"],
            "columns": meta["columns"],
            "row_count": meta["row_count"],
            "detected_lon_col": meta["lon_col"],
            "detected_lat_col": meta["lat_col"],
            "has_coordinates": has_coords,
        }

        # Build artifact metadata
        artifact_meta = {
            "columns": meta["columns"],
            "delimiter": meta["delimiter"],
            "detected_lon_col": meta["lon_col"],
            "detected_lat_col": meta["lat_col"],
            "row_count": meta["row_count"],
            "valid_feature_count": meta["valid_feature_count"],
            "crs": crs if has_coords else None,
            "extent": meta["extent"],
        }

        if lazy:
            # Lazy mode: metadata only, LAZY_HANDLE backing
            artifact = Artifact(
                type=artifact_type,
                name=file_path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=str(file_path),
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"CSV metadata only — {meta['row_count']} rows",
            )

        # Eager mode
        if has_coords:
            # Write to GeoPackage
            output_path = workspace / f"{file_path.stem}.gpkg"
            try:
                _write_geopackage(file_path, output_path, meta, self.default_crs)
            except MaterializeError:
                raise
            except Exception as e:
                raise MaterializeError(source_ref, f"Failed to write GeoPackage: {e}") from e

            artifact = Artifact(
                type=ArtifactType.VECTOR,
                name=file_path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LOCAL_FILE,
                    uri=str(output_path),
                    size_bytes=output_path.stat().st_size,
                    content_hash=content_hash(output_path),
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="wrapped_local",
                source_ref=source_ref,
                notes=f"CSV converted to GeoPackage ({meta['valid_feature_count']} features)",
            )
        else:
            # No coordinates: return as table with local file backing
            artifact = Artifact(
                type=ArtifactType.TABLE,
                name=file_path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LOCAL_FILE,
                    uri=str(file_path),
                    size_bytes=file_path.stat().st_size,
                    content_hash=content_hash(file_path),
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="wrapped_local",
                source_ref=source_ref,
                notes=f"CSV table (no coordinates detected) — {meta['row_count']} rows",
            )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """List .csv/.tsv files in a directory.

        Args:
            query: Directory path as string or dict with "path" key

        Returns:
            List of catalog entries for CSV/TSV files.
        """
        if isinstance(query, dict):
            dir_path = query.get("path")
        elif isinstance(query, str):
            dir_path = query
        else:
            raise MaterializeError("discover", "No path specified")

        if not dir_path:
            raise MaterializeError("discover", "No path specified")

        path = Path(dir_path)
        if not path.is_dir():
            raise MaterializeError("discover", f"Not a directory: {dir_path}")

        seen: set[str] = set()
        entries = []
        for pattern in ("*.csv", "*.CSV", "*.tsv", "*.TSV"):
            for file_path in path.glob(pattern):
                resolved = str(file_path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)

                # Try to detect if this file has coordinates
                has_coords = False
                try:
                    meta = _read_csv_metadata(file_path)
                    has_coords = meta["has_coordinates"]
                except Exception:
                    pass

                entries.append(
                    CatalogEntry(
                        source_ref=str(file_path),
                        name=file_path.stem,
                        metadata={
                            "extension": file_path.suffix.lower(),
                            "has_coordinates": has_coords,
                        },
                    )
                )

        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get CSV/TSV metadata without materializing data.

        Args:
            source_ref: Path to CSV/TSV file, optionally with "::lon_col,lat_col" suffix

        Returns:
            Metadata dict with columns, detected coordinates, row count, etc.
        """
        file_path, explicit_lon_col, explicit_lat_col = _parse_source_ref(source_ref)

        try:
            meta = _read_csv_metadata(
                file_path,
                explicit_lon_col=explicit_lon_col,
                explicit_lat_col=explicit_lat_col,
            )
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read CSV metadata: {e}") from e

        crs = meta["crs_hint"] if meta["crs_hint"] is not None else self.default_crs

        return {
            "columns": meta["columns"],
            "delimiter": meta["delimiter"],
            "detected_lon_col": meta["lon_col"],
            "detected_lat_col": meta["lat_col"],
            "row_count": meta["row_count"],
            "valid_feature_count": meta["valid_feature_count"],
            "has_coordinates": meta["has_coordinates"],
            "crs": crs if meta["has_coordinates"] else None,
            "extent": meta["extent"],
        }
