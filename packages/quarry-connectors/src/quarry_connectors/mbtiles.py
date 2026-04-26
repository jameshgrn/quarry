"""MBTilesConnector — materializes MBTiles files into canonical artifacts.

Lane: connector

MBTiles is a SQLite-based format for storing map tiles (raster PNG/JPEG or vector PBF).
This connector reads tile metadata and materializes MBTiles files as artifacts.
Uses stdlib sqlite3 only — no new dependencies.

Key features:
- Local file materialization (wrap in place or lazy handle)
- Reads metadata from SQLite tables (metadata, tiles)
- Format detection: PNG raster, JPEG raster, or PBF vector tiles
- CRS always EPSG:3857 (Web Mercator) per MBTiles spec

Design decisions:
- source_ref: local path to .mbtiles file
- Lazy = metadata-only with LAZY_HANDLE backing
- Eager local = wrap in place, LOCAL_FILE backing
- ArtifactType: RASTER for PNG/JPEG tiles, VECTOR for PBF tiles
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


# Magic bytes for format detection
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"
GZIP_MAGIC = b"\x1f\x8b"


def _is_mbtiles_file(path: Path) -> bool:
    """Check if file has .mbtiles extension (case-insensitive)."""
    return path.suffix.lower() == ".mbtiles"


def _detect_tile_format(tile_data: bytes) -> str:
    """Detect tile format from magic bytes.

    Args:
        tile_data: First bytes of a tile

    Returns:
        "png", "jpeg", or "pbf"
    """
    if tile_data.startswith(PNG_MAGIC):
        return "png"
    if tile_data.startswith(JPEG_MAGIC):
        return "jpeg"
    if tile_data.startswith(GZIP_MAGIC):
        return "pbf"
    return "unknown"


def _read_mbtiles_metadata(path: Path) -> dict[str, Any]:
    """Read metadata from an MBTiles file.

    Args:
        path: Path to .mbtiles file

    Returns:
        Dict with metadata fields and tile info

    Raises:
        MaterializeError: If file is not valid MBTiles
    """
    # Validate SQLite file
    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.DatabaseError as e:
        raise MaterializeError(str(path), f"Not a valid SQLite file: {e}") from e

    try:
        cursor = conn.cursor()

        # Check metadata table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
        if not cursor.fetchone():
            raise MaterializeError(str(path), "Missing required 'metadata' table")

        # Check tiles table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tiles'")
        if not cursor.fetchone():
            raise MaterializeError(str(path), "Missing required 'tiles' table")

        # Read metadata
        cursor.execute("SELECT name, value FROM metadata")
        meta = dict(cursor.fetchall())

        # Get tile count
        cursor.execute("SELECT COUNT(*) FROM tiles")
        tile_count = cursor.fetchone()[0]

        # Detect format from first tile if not in metadata
        format_from_meta = meta.get("format", "").lower()
        if format_from_meta in ("png", "jpeg", "jpg", "pbf"):
            detected_format = format_from_meta
            if detected_format == "jpg":
                detected_format = "jpeg"
        else:
            # Inspect first tile's magic bytes
            cursor.execute("SELECT tile_data FROM tiles LIMIT 1")
            row = cursor.fetchone()
            if row and row[0]:
                detected_format = _detect_tile_format(row[0])
            else:
                detected_format = "unknown"

        conn.close()

        # Parse bounds from metadata
        bounds = None
        bounds_str = meta.get("bounds", "")
        if bounds_str:
            try:
                parts = [float(x) for x in bounds_str.split(",")]
                if len(parts) == 4:
                    bounds = (parts[0], parts[1], parts[2], parts[3])
            except (ValueError, TypeError):
                bounds = None

        # Parse center from metadata
        center = None
        center_str = meta.get("center", "")
        if center_str:
            try:
                parts = [float(x) for x in center_str.split(",")]
                if len(parts) >= 2:
                    center = (parts[0], parts[1])
            except (ValueError, TypeError):
                center = None

        # Parse zoom levels
        minzoom = None
        maxzoom = None
        try:
            if "minzoom" in meta:
                minzoom = int(meta["minzoom"])
            if "maxzoom" in meta:
                maxzoom = int(meta["maxzoom"])
        except (ValueError, TypeError):
            pass

        return {
            "name": meta.get("name", ""),
            "description": meta.get("description", ""),
            "format": detected_format,
            "bounds": bounds,
            "center": center,
            "minzoom": minzoom,
            "maxzoom": maxzoom,
            "tile_count": tile_count,
            "type": meta.get("type", ""),  # overlay or baselayer
            "version": meta.get("version", ""),
            "attribution": meta.get("attribution", ""),
            "raw_metadata": meta,
        }

    except MaterializeError:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise MaterializeError(str(path), f"Failed to read MBTiles metadata: {e}") from e


class MBTilesConnector:
    """Materializes MBTiles files into canonical Quarry artifacts.

    Supports local MBTiles files with raster (PNG/JPEG) or vector (PBF) tiles.
    CRS is always EPSG:3857 (Web Mercator) per MBTiles specification.
    """

    @property
    def name(self) -> str:
        return "mbtiles"

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
        """Materialize an MBTiles file into a canonical artifact.

        source_ref: local path to .mbtiles file
        workspace: where to put materialized data if needed (unused for local)
        lazy: if True, return metadata-only artifact with LAZY_HANDLE backing
        """
        # Parse source_ref to get path
        path = self._parse_source_ref(source_ref)

        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        if not _is_mbtiles_file(path):
            raise MaterializeError(source_ref, f"Not an MBTiles file: {path}")

        # Read metadata from SQLite
        try:
            meta = _read_mbtiles_metadata(path)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read MBTiles: {e}") from e

        # Determine artifact type from format
        detected_format = meta["format"]
        if detected_format == "pbf":
            artifact_type = ArtifactType.VECTOR
        else:
            # PNG, JPEG, or unknown default to RASTER
            artifact_type = ArtifactType.RASTER

        # Build spatial descriptor
        # CRS is always EPSG:3857 per MBTiles spec
        bounds = meta["bounds"]
        spatial = SpatialDescriptor(
            crs="EPSG:3857",
            extent=bounds,
            # No resolution for tiles (varies by zoom)
            # No band_count for tiles
        )

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "mbtiles",
            "path": str(path),
            "lazy": lazy,
            "format": detected_format,
        }

        # Build artifact metadata
        artifact_meta: dict[str, Any] = {
            "mbtiles_name": meta["name"],
            "mbtiles_description": meta["description"],
            "format": detected_format,
            "tile_count": meta["tile_count"],
            "minzoom": meta["minzoom"],
            "maxzoom": meta["maxzoom"],
            "bounds": bounds,
            "center": meta["center"],
            "type": meta["type"],
            "version": meta["version"],
            "attribution": meta["attribution"],
        }

        if lazy:
            # Lazy mode: metadata only, LAZY_HANDLE backing
            artifact = Artifact(
                type=artifact_type,
                name=self._derive_name(path),
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
                notes=(
                    f"MBTiles metadata only — {meta['tile_count']} tiles, format={detected_format}"
                ),
            )

        # Eager mode: wrap local file in place
        local_path = path.resolve()
        strategy = "wrapped_local"
        notes = (
            f"Local MBTiles wrapped ({local_path.stat().st_size} bytes, {meta['tile_count']} tiles)"
        )

        artifact = Artifact(
            type=artifact_type,
            name=self._derive_name(path),
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(local_path),
                size_bytes=local_path.stat().st_size,
                content_hash=content_hash(local_path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata=artifact_meta,
        )

        return MaterializeResult(
            artifact=artifact,
            strategy=strategy,
            source_ref=source_ref,
            notes=notes,
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """List .mbtiles files in a directory.

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
        for pattern in ("*.mbtiles", "*.MBTILES"):
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
        """Get MBTiles metadata without materializing data."""
        path = self._parse_source_ref(source_ref)

        if not path.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        if not _is_mbtiles_file(path):
            raise MaterializeError(source_ref, f"Not an MBTiles file: {path}")

        try:
            meta = _read_mbtiles_metadata(path)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read MBTiles metadata: {e}") from e

        return {
            "name": meta["name"],
            "description": meta["description"],
            "format": meta["format"],
            "bounds": meta["bounds"],
            "center": meta["center"],
            "minzoom": meta["minzoom"],
            "maxzoom": meta["maxzoom"],
            "tile_count": meta["tile_count"],
            "type": meta["type"],
            "version": meta["version"],
            "attribution": meta["attribution"],
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _parse_source_ref(self, source_ref: SourceRef | str) -> Path:
        """Parse source_ref into a Path."""
        if hasattr(source_ref, "raw"):
            # SourceRef object
            return Path(source_ref.raw)
        return Path(str(source_ref))

    def _derive_name(self, path: Path) -> str:
        """Derive artifact name from path."""
        return path.stem
