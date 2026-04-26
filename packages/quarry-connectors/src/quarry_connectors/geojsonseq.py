"""GeoJSONSeqConnector — materializes GeoJSON Sequence (RFC 8142) files.

Lane: connector

GeoJSON Sequence is newline-delimited GeoJSON. Each line is a standalone
GeoJSON Feature. File extensions: .geojsonl, .geojsonseq, .ndjson (when
containing GeoJSON features).

Key features:
- Local file materialization (wrap in place or lazy handle)
- Uses fiona with GeoJSONSeq driver when available (fiona 1.9+ / GDAL 2.4+)
- Falls back to stdlib json + shapely for metadata extraction
- CRS is always WGS84 (EPSG:4326) per GeoJSON spec (RFC 7946)

Design decisions:
- source_ref: local path to .geojsonl/.geojsonseq/.ndjson file
- Lazy = metadata-only with LAZY_HANDLE backing
- Eager local = wrap in place, LOCAL_FILE backing
- Always VECTOR artifact type
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import shapely.geometry
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

# Extensions for GeoJSON Sequence files
GEOJSONSEQ_EXTENSIONS = {".geojsonl", ".geojsonseq", ".ndjson"}


def _is_geojsonseq_file(path: Path) -> bool:
    """Check if file has a GeoJSON Sequence extension (case-insensitive)."""
    return path.suffix.lower() in GEOJSONSEQ_EXTENSIONS


def _read_geojsonseq_fallback(path: Path) -> dict[str, Any]:
    """Read metadata from a GeoJSONSeq file using stdlib json + shapely.

    This is the fallback implementation when fiona's GeoJSONSeq driver
    is not available.

    Args:
        path: Path to the GeoJSONSeq file

    Returns:
        Dict with driver, crs, schema, bounds, feature_count
    """
    features: list[dict[str, Any]] = []
    geometries: list[shapely.geometry.base.BaseGeometry] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                feature = json.loads(line)
                features.append(feature)
                # Extract geometry for extent computation
                if "geometry" in feature and feature["geometry"]:
                    geom = shapely.geometry.shape(feature["geometry"])
                    geometries.append(geom)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

    if not features:
        return {
            "driver": "json_fallback",
            "crs": "EPSG:4326",
            "schema": {"geometry": "Unknown", "properties": {}},
            "bounds": None,
            "feature_count": 0,
        }

    # Extract schema from first feature's properties
    first_feature = features[0]
    properties = first_feature.get("properties", {})
    prop_schema: dict[str, str] = {}
    for key, val in properties.items():
        if isinstance(val, bool):
            prop_schema[key] = "bool"
        elif isinstance(val, int):
            prop_schema[key] = "int"
        elif isinstance(val, float):
            prop_schema[key] = "float"
        else:
            prop_schema[key] = "str"

    # Determine geometry type from first feature
    geom_type = "Unknown"
    if "geometry" in first_feature and first_feature["geometry"]:
        geom_type = first_feature["geometry"].get("type", "Unknown")

    schema = {"geometry": geom_type, "properties": prop_schema}

    # Compute bounds from all geometries
    bounds = None
    if geometries:
        union = shapely.geometry.GeometryCollection(geometries)
        if not union.is_empty:
            minx, miny, maxx, maxy = union.bounds
            bounds = (minx, miny, maxx, maxy)

    return {
        "driver": "json_fallback",
        "crs": "EPSG:4326",  # GeoJSON is always WGS84
        "schema": schema,
        "bounds": bounds,
        "feature_count": len(features),
    }


def _read_geojsonseq_metadata(path: Path) -> dict[str, Any]:
    """Read metadata from a GeoJSONSeq file.

    Tries fiona with GeoJSONSeq driver first, falls back to json/shapely.

    Args:
        path: Path to the GeoJSONSeq file

    Returns:
        Dict with driver, crs, schema, bounds, feature_count
    """
    # Try fiona first
    try:
        import fiona

        # Check if GeoJSONSeq driver is available
        drivers = fiona.supported_drivers
        if "GeoJSONSeq" in drivers:
            with fiona.open(str(path), driver="GeoJSONSeq") as src:
                # Extract CRS
                crs = None
                if src.crs:
                    if isinstance(src.crs, dict):
                        if "init" in src.crs:
                            crs = src.crs["init"]
                        else:
                            crs = str(src.crs)
                    else:
                        crs_str = str(src.crs)
                        if crs_str:
                            crs = crs_str

                # GeoJSON is always EPSG:4326 per spec
                if crs is None:
                    crs = "EPSG:4326"

                # Extract schema
                schema = src.schema

                # Extract bounds
                bounds = src.bounds

                # Feature count
                count = len(src)

                return {
                    "driver": "GeoJSONSeq",
                    "crs": crs,
                    "schema": schema,
                    "bounds": bounds,
                    "feature_count": count,
                }
    except ImportError:
        pass  # fiona not available, use fallback
    except Exception:
        pass  # fiona failed, use fallback

    # Fallback to stdlib json + shapely
    return _read_geojsonseq_fallback(path)


class GeoJSONSeqConnector:
    """Materializes GeoJSON Sequence files into canonical Quarry artifacts.

    Supports local files with .geojsonl, .geojsonseq, or .ndjson extensions.
    Uses fiona with GeoJSONSeq driver when available, falls back to
    stdlib json + shapely for metadata extraction.
    """

    @property
    def name(self) -> str:
        return "geojsonseq"

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
        """Materialize a GeoJSON Sequence file into a canonical artifact.

        source_ref: local path to .geojsonl/.geojsonseq/.ndjson file
        workspace: where to put materialized data if needed (unused for local)
        lazy: if True, return metadata-only artifact with LAZY_HANDLE backing
        """
        # Resolve path from source_ref
        if isinstance(source_ref, str):
            path = Path(source_ref)
        else:
            path = Path(source_ref.raw)

        path = path.resolve()

        # Validate file exists and has correct extension
        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        if not path.is_file():
            raise MaterializeError(source_ref, f"Not a file: {path}")

        if not _is_geojsonseq_file(path):
            raise MaterializeError(
                source_ref,
                f"Not a GeoJSON Sequence file "
                f"(expected .geojsonl, .geojsonseq, or .ndjson): {path}",
            )

        # Read metadata
        try:
            meta = _read_geojsonseq_metadata(path)
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read GeoJSONSeq metadata: {e}") from e

        # Build spatial descriptor
        bounds = meta["bounds"]
        spatial = SpatialDescriptor(
            crs=meta["crs"],
            extent=bounds if bounds else None,
            feature_count=meta["feature_count"],
        )

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "geojsonseq",
            "path": str(path),
            "lazy": lazy,
            "driver_used": meta["driver"],
        }

        # Build artifact metadata
        artifact_meta = {
            "driver": meta["driver"],
            "schema": meta["schema"],
            "crs": meta["crs"],
            "feature_count": meta["feature_count"],
            "bounds": meta["bounds"],
        }

        if lazy:
            # Lazy mode: metadata only, LAZY_HANDLE backing
            artifact = Artifact(
                type=ArtifactType.VECTOR,
                name=path.stem,
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=str(path),
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"GeoJSONSeq metadata only — {meta['feature_count']} features",
            )

        # Eager mode: wrap in place, LOCAL_FILE backing
        artifact = Artifact(
            type=ArtifactType.VECTOR,
            name=path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(path),
                size_bytes=path.stat().st_size,
                content_hash=content_hash(path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata=artifact_meta,
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="wrapped_local",
            source_ref=source_ref,
            notes=f"Local GeoJSONSeq wrapped ({path.stat().st_size} bytes)",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """List GeoJSON Sequence files in a directory.

        query: directory path as string or dict with "path" key
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
        for ext in GEOJSONSEQ_EXTENSIONS:
            for pattern in (f"*{ext}", f"*{ext.upper()}"):
                for file_path in path.glob(pattern):
                    resolved = str(file_path.resolve())
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    entries.append(
                        CatalogEntry(
                            source_ref=str(file_path),
                            name=file_path.stem,
                            metadata={
                                "extension": file_path.suffix,
                            },
                        )
                    )

        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get GeoJSONSeq metadata without materializing data."""
        # Resolve path from source_ref
        if isinstance(source_ref, str):
            path = Path(source_ref)
        else:
            path = Path(source_ref.raw)

        path = path.resolve()

        # Validate file exists and has correct extension
        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        if not path.is_file():
            raise MaterializeError(source_ref, f"Not a file: {path}")

        if not _is_geojsonseq_file(path):
            raise MaterializeError(
                source_ref,
                f"Not a GeoJSON Sequence file "
                f"(expected .geojsonl, .geojsonseq, or .ndjson): {path}",
            )

        try:
            meta = _read_geojsonseq_metadata(path)
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read GeoJSONSeq metadata: {e}") from e

        return {
            "driver": meta["driver"],
            "crs": meta["crs"],
            "schema": meta["schema"],
            "feature_count": meta["feature_count"],
            "extent": meta["bounds"],
        }
