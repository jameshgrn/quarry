"""LASPointCloudConnector — materializes LAS/LAZ lidar point cloud files into canonical artifacts.

Lane: connector

LAS/LAZ files contain lidar point cloud data with 3D coordinates and attributes.
This connector reads point cloud metadata and materializes them as artifacts.
Uses `laspy` for reading.

Key features:
- Local file materialization (wrap in place or lazy handle)
- Reads point cloud metadata: point count, CRS from VLRs, extent from header
- Supports both .las and .laz (compressed) formats
- CRS extraction from WKT or GeoTIFF VLRs

Design decisions:
- source_ref: local path to .las or .laz file
- Lazy = metadata-only with LAZY_HANDLE backing (header only, no points loaded)
- Eager local = wrap in place, LOCAL_FILE backing (reads all points via laspy.read)
- ArtifactType.VECTOR (point clouds are collections of point geometries)
"""

from __future__ import annotations

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

# Conditional import for optional dependency
try:
    import laspy

    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    laspy = None  # type: ignore


def _is_las_file(path: Path) -> bool:
    """Check if file has .las or .laz extension (case-insensitive)."""
    return path.suffix.lower() in (".las", ".laz")


def _extract_crs(header: Any) -> str | None:
    """Extract CRS string from LAS header VLRs.

    Tries laspy's parse_crs() first (laspy >= 2.0), then falls back to
    manual VLR inspection for WKT (record_id 2112) or GeoTIFF keys.
    """
    # Try laspy's built-in CRS parsing first
    try:
        crs = header.parse_crs()
        if crs is not None:
            return str(crs)
    except Exception:
        pass

    # Fallback: look for WKT in VLRs (OGC WKT record_id 2112)
    for vlr in header.vlrs:
        if vlr.record_id == 2112:  # OGC WKT
            # Try the string attribute first (laspy >= 2.0 WktCoordinateSystemVlr)
            if hasattr(vlr, "string"):
                try:
                    wkt = vlr.string
                    if wkt:
                        return wkt
                except Exception:
                    pass
            # Fallback to record_data decoding
            if hasattr(vlr, "record_data"):
                try:
                    wkt = vlr.record_data.decode("utf-8").rstrip("\x00")
                    if wkt:
                        return wkt
                except Exception:
                    pass

    return None


def _read_las_metadata(path: Path, header_only: bool = False) -> dict[str, Any]:
    """Read metadata from a LAS/LAZ file using laspy.

    Args:
        path: Path to the .las or .laz file
        header_only: If True, only read header (faster, no point data)

    Returns:
        Dict with point_count, crs, extent, point_format, version, etc.
    """
    if not HAS_LASPY:
        raise RuntimeError("laspy is required but not installed")

    try:
        if header_only:
            # Fast path: read header only
            with laspy.open(str(path)) as reader:
                header = reader.header
                point_count = header.point_count
                mins = header.mins
                maxs = header.maxs
                point_format_id = header.point_format.id
                version = f"{header.version.major}.{header.version.minor}"
                scales = (
                    header.scales.tolist()
                    if hasattr(header.scales, "tolist")
                    else list(header.scales)
                )
                offsets = (
                    header.offsets.tolist()
                    if hasattr(header.offsets, "tolist")
                    else list(header.offsets)
                )

                # Check for extra dimensions
                extra_dims = []
                if hasattr(header.point_format, "extra_dims"):
                    extra_dims = [str(dim.name) for dim in header.point_format.extra_dims]

                # Check for color and GPS time support
                has_color = (
                    hasattr(header.point_format, "has_color") and header.point_format.has_color
                )
                has_gps_time = (
                    hasattr(header.point_format, "has_gps_time")
                    and header.point_format.has_gps_time
                )

                crs = _extract_crs(header)
        else:
            # Full read: loads all points
            las = laspy.read(str(path))
            header = las.header
            point_count = header.point_count
            mins = header.mins
            maxs = header.maxs
            point_format_id = header.point_format.id
            version = f"{header.version.major}.{header.version.minor}"
            scales = (
                header.scales.tolist() if hasattr(header.scales, "tolist") else list(header.scales)
            )
            offsets = (
                header.offsets.tolist()
                if hasattr(header.offsets, "tolist")
                else list(header.offsets)
            )

            # Check for extra dimensions
            extra_dims = []
            if hasattr(header.point_format, "extra_dims"):
                extra_dims = [str(dim.name) for dim in header.point_format.extra_dims]

            # Check for color and GPS time support
            has_color = hasattr(header.point_format, "has_color") and header.point_format.has_color
            has_gps_time = (
                hasattr(header.point_format, "has_gps_time") and header.point_format.has_gps_time
            )

            crs = _extract_crs(header)

        # Build extent (2D for spatial descriptor)
        extent = None
        if mins is not None and maxs is not None and len(mins) >= 2 and len(maxs) >= 2:
            extent = (float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1]))

        return {
            "point_count": point_count,
            "crs": crs,
            "extent": extent,
            "extent_3d": (
                float(mins[0]) if mins is not None and len(mins) > 0 else None,
                float(mins[1]) if mins is not None and len(mins) > 1 else None,
                float(mins[2]) if mins is not None and len(mins) > 2 else None,
                float(maxs[0]) if maxs is not None and len(maxs) > 0 else None,
                float(maxs[1]) if maxs is not None and len(maxs) > 1 else None,
                float(maxs[2]) if maxs is not None and len(maxs) > 2 else None,
            ),
            "point_format_id": point_format_id,
            "las_version": version,
            "scales": scales,
            "offsets": offsets,
            "has_color": has_color,
            "has_gps_time": has_gps_time,
            "extra_dims": extra_dims,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to read LAS file: {e}") from e


class LASPointCloudConnector:
    """Materializes LAS/LAZ lidar point cloud files into canonical Quarry artifacts."""

    @property
    def name(self) -> str:
        return "las"

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
        """Materialize a LAS/LAZ file into a canonical artifact.

        source_ref: local path to .las or .laz file
        workspace: where to put materialized data if needed (unused for local files)
        lazy: if True, return metadata-only artifact with LAZY_HANDLE backing
        """
        if not HAS_LASPY:
            raise MaterializeError(
                source_ref,
                "laspy is required but not installed. Install with: pip install laspy[laszip]",
            )

        # Parse source_ref to get path
        path = self._parse_source_ref(source_ref)
        path_obj = Path(path)

        if not path_obj.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        if not _is_las_file(path_obj):
            raise MaterializeError(source_ref, f"Not a LAS/LAZ file: {path}")

        # Read metadata
        try:
            meta = _read_las_metadata(path_obj, header_only=lazy)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read LAS metadata: {e}") from e

        # Build spatial descriptor
        spatial = SpatialDescriptor(
            crs=meta["crs"],
            extent=meta["extent"],
            feature_count=meta["point_count"],
        )

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "las",
            "path": str(path_obj),
            "lazy": lazy,
            "point_count": meta["point_count"],
            "point_format_id": meta["point_format_id"],
            "las_version": meta["las_version"],
        }

        # Build artifact metadata
        artifact_meta = {
            "point_count": meta["point_count"],
            "point_format_id": meta["point_format_id"],
            "las_version": meta["las_version"],
            "crs": meta["crs"],
            "extent": meta["extent"],
            "extent_3d": meta["extent_3d"],
            "scales": meta["scales"],
            "offsets": meta["offsets"],
            "has_color": meta["has_color"],
            "has_gps_time": meta["has_gps_time"],
            "extra_dims": meta["extra_dims"],
        }

        if lazy:
            # Lazy mode: metadata only, LAZY_HANDLE backing
            artifact = Artifact(
                type=ArtifactType.VECTOR,
                name=self._derive_name(path),
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=str(path_obj),
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"LAS metadata only — {meta['point_count']} points",
            )

        # Eager mode: wrap in place with LOCAL_FILE backing
        artifact = Artifact(
            type=ArtifactType.VECTOR,
            name=self._derive_name(path),
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(path_obj),
                size_bytes=path_obj.stat().st_size,
                content_hash=content_hash(path_obj),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata=artifact_meta,
        )

        notes = f"Local LAS wrapped ({path_obj.stat().st_size} bytes, {meta['point_count']} points)"
        return MaterializeResult(
            artifact=artifact,
            strategy="wrapped_local",
            source_ref=source_ref,
            notes=notes,
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """List .las/.laz files in a directory.

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
        for pattern in ("*.las", "*.LAS", "*.laz", "*.LAZ"):
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
        """Get LAS metadata without materializing data."""
        if not HAS_LASPY:
            raise MaterializeError(
                source_ref,
                "laspy is required but not installed. Install with: pip install laspy[laszip]",
            )

        path = self._parse_source_ref(source_ref)
        path_obj = Path(path)

        if not path_obj.exists():
            raise MaterializeError(source_ref, f"File not found: {path}")

        if not _is_las_file(path_obj):
            raise MaterializeError(source_ref, f"Not a LAS/LAZ file: {path}")

        try:
            meta = _read_las_metadata(path_obj, header_only=True)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read LAS metadata: {e}") from e

        return {
            "point_count": meta["point_count"],
            "point_format_id": meta["point_format_id"],
            "las_version": meta["las_version"],
            "crs": meta["crs"],
            "extent": meta["extent"],
            "extent_3d": meta["extent_3d"],
            "scales": meta["scales"],
            "offsets": meta["offsets"],
            "has_color": meta["has_color"],
            "has_gps_time": meta["has_gps_time"],
            "extra_dims": meta["extra_dims"],
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _parse_source_ref(self, source_ref: SourceRef | str) -> str:
        """Parse source_ref into a path string."""
        from quarry_core.source_ref import SourceRef

        if isinstance(source_ref, SourceRef):
            return source_ref.raw
        return source_ref.strip()

    def _derive_name(self, path: str) -> str:
        """Derive artifact name from path."""
        return Path(path).stem
