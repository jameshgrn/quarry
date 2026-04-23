"""FlatGeobufConnector — materializes FlatGeobuf (.fgb) files into canonical artifacts.

Lane: connector

FlatGeobuf is a binary vector format optimized for streaming and random access.
This connector handles both local files and remote HTTP sources with range-request
support via GDAL's /vsicurl/ virtual filesystem.

Key features:
- Local file materialization (wrap in place or lazy handle)
- Remote HTTP materialization with range-request support
- Spatial filtering: bbox parameter for efficient remote reads
- Uses fiona (GDAL) for all I/O — FlatGeobuf driver is built into GDAL

Design decisions:
- source_ref: local path or HTTP/HTTPS URL
- Lazy = metadata-only with LAZY_HANDLE backing (local or remote URI)
- Eager local = wrap in place, LOCAL_FILE backing
- Eager remote = download via requests, LOCAL_FILE backing
- bbox filter: passed to fiona.open() for efficient spatial filtering
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import fiona
import requests
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


def _is_fgb_file(path: Path) -> bool:
    """Check if file has .fgb extension (case-insensitive)."""
    return path.suffix.lower() == ".fgb"


def _read_fgb_metadata(
    path_or_url: str, bbox: tuple[float, float, float, float] | None = None
) -> dict[str, Any]:
    """Read metadata from a FlatGeobuf file using fiona.

    Args:
        path_or_url: Local path or URL to the .fgb file
        bbox: Optional bounding box for spatial filtering (xmin, ymin, xmax, ymax)

    Returns:
        Dict with driver, crs, schema, bounds, feature_count
    """
    open_args: dict[str, Any] = {}
    if bbox is not None:
        open_args["bbox"] = bbox

    with fiona.open(path_or_url, **open_args) as src:
        # Extract CRS
        crs = None
        if src.crs:
            if isinstance(src.crs, dict):
                # Fiona 1.9+ returns CRS as dict
                if "init" in src.crs:
                    crs = src.crs["init"]
                else:
                    crs = str(src.crs)
            else:
                # Older fiona or CRS object
                crs_str = str(src.crs)
                if crs_str:
                    crs = crs_str

        # Extract schema
        schema = src.schema

        # Extract bounds
        bounds = src.bounds

        # Feature count
        count = len(src)

        return {
            "driver": src.driver,
            "crs": crs,
            "schema": schema,
            "bounds": bounds,
            "feature_count": count,
        }


class FlatGeobufConnector:
    """Materializes FlatGeobuf files into canonical Quarry artifacts.

    Supports local files and remote HTTP/HTTPS URLs with range-request
    capability via GDAL's virtual filesystem.
    """

    @property
    def name(self) -> str:
        return "flatgeobuf"

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
        """Materialize a FlatGeobuf file into a canonical artifact.

        source_ref: local path or HTTP/HTTPS URL to .fgb file
        workspace: where to download remote files (eager mode)
        lazy: if True, return metadata-only artifact with LAZY_HANDLE backing

        For remote sources, supports bbox filtering via source_ref params:
            {"bbox": (xmin, ymin, xmax, ymax)}
        """
        path_or_url, is_remote, bbox = self._parse_source_ref(source_ref)

        # Read metadata (with optional bbox filter for remote)
        try:
            if is_remote and bbox is not None:
                # Use GDAL virtual filesystem with bbox for efficient remote read
                vsi_path = self._vsi_path(path_or_url)
                meta = _read_fgb_metadata(vsi_path, bbox=bbox)
            else:
                meta = _read_fgb_metadata(path_or_url)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read FlatGeobuf metadata: {e}") from e

        # Build spatial descriptor
        bounds = meta["bounds"]
        spatial = SpatialDescriptor(
            crs=meta["crs"],
            extent=bounds if bounds else None,
            feature_count=meta["feature_count"],
        )

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "flatgeobuf",
            "path_or_url": path_or_url,
            "lazy": lazy,
            "is_remote": is_remote,
            "bbox_filter": bbox,
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
                name=self._derive_name(path_or_url),
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=path_or_url,
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"FlatGeobuf metadata only — {meta['feature_count']} features",
            )

        # Eager mode
        if is_remote:
            # Download remote file
            download_path = self._download(path_or_url, workspace, bbox)
            local_path = download_path
            lineage_params["data_transferred"] = download_path.stat().st_size
            strategy = "fetched_remote"
            notes = f"Downloaded {download_path.name} ({download_path.stat().st_size} bytes)"
        else:
            # Local file: wrap in place
            local_path = Path(path_or_url).resolve()
            strategy = "wrapped_local"
            notes = f"Local FlatGeobuf wrapped ({local_path.stat().st_size} bytes)"

        artifact = Artifact(
            type=ArtifactType.VECTOR,
            name=self._derive_name(path_or_url),
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
        """List .fgb files in a directory.

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
        for pattern in ("*.fgb", "*.FGB"):
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
        """Get FlatGeobuf metadata without materializing data."""
        path_or_url, is_remote, bbox = self._parse_source_ref(source_ref)

        try:
            if is_remote and bbox is not None:
                vsi_path = self._vsi_path(path_or_url)
                meta = _read_fgb_metadata(vsi_path, bbox=bbox)
            else:
                meta = _read_fgb_metadata(path_or_url)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to read FlatGeobuf metadata: {e}") from e

        return {
            "driver": meta["driver"],
            "crs": meta["crs"],
            "schema": meta["schema"],
            "feature_count": meta["feature_count"],
            "extent": meta["bounds"],
            "is_remote": is_remote,
        }

    # -----------------------------------------------------------------------
    # Source ref parsing
    # -----------------------------------------------------------------------

    def _parse_source_ref(
        self, source_ref: SourceRef | str
    ) -> tuple[str, bool, tuple[float, float, float, float] | None]:
        """Parse source_ref into (path_or_url, is_remote, bbox).

        Returns:
            Tuple of (path_or_url, is_remote_flag, optional_bbox)
        """
        from quarry_core.source_ref import SourceRef, SourceRefKind

        bbox: tuple[float, float, float, float] | None = None

        if isinstance(source_ref, SourceRef):
            # Check for bbox in params
            if source_ref.params:
                bbox_param = source_ref.params.get("bbox")
                if bbox_param is not None:
                    bbox = tuple(bbox_param)  # type: ignore

            if source_ref.kind == SourceRefKind.REMOTE_URI:
                return (source_ref.raw, True, bbox)
            if source_ref.kind in (
                SourceRefKind.LOCAL_PATH,
                SourceRefKind.LOCAL_VECTOR,
            ):
                return (source_ref.raw, False, bbox)
            # For other kinds, fall through to raw string parsing
            raw = source_ref.raw.strip()
        else:
            raw = source_ref.strip()

        # Parse URL scheme
        parsed = urlparse(raw)
        if parsed.scheme in ("http", "https"):
            return (raw, True, bbox)

        # Local path
        return (raw, False, bbox)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _vsi_path(self, url: str) -> str:
        """Convert a remote URL to GDAL virtual filesystem path."""
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            return f"/vsicurl/{url}"
        return url

    def _derive_name(self, path_or_url: str) -> str:
        """Derive artifact name from path or URL."""
        parsed = urlparse(path_or_url)
        if parsed.path:
            return Path(parsed.path).stem
        return Path(path_or_url).stem

    def _download(
        self,
        url: str,
        workspace: Path,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> Path:
        """Download a remote FlatGeobuf to workspace.

        If bbox is provided, uses fiona to read only features within bbox
        and writes them to a new local file.
        """
        filename = self._derive_name(url) + ".fgb"
        download_path = workspace / filename
        download_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if bbox is not None:
                # Use fiona with bbox for efficient spatial download
                vsi_path = self._vsi_path(url)
                self._download_with_bbox(vsi_path, download_path, bbox)
            else:
                # Full download via requests
                resp = requests.get(url, stream=True, timeout=120)
                resp.raise_for_status()
                with open(download_path, "wb") as f:
                    shutil.copyfileobj(resp.raw, f)
        except Exception as e:
            if download_path.exists():
                download_path.unlink()
            raise MaterializeError(url, f"Download failed: {e}") from e

        return download_path

    def _download_with_bbox(
        self,
        vsi_path: str,
        output_path: Path,
        bbox: tuple[float, float, float, float],
    ) -> None:
        """Download features within bbox using fiona.

        Reads features from the remote source with bbox filter and writes
        them to a new local FlatGeobuf file.
        """
        with fiona.open(vsi_path, bbox=bbox) as src:
            # Get schema and CRS from source
            schema = src.schema
            crs = src.crs

            # Create output file
            with fiona.open(
                str(output_path),
                "w",
                driver="FlatGeobuf",
                schema=schema,
                crs=crs,
            ) as dst:
                for feature in src:
                    dst.write(feature)
