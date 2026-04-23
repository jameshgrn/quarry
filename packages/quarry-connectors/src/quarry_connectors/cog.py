"""COGConnector — materializes Cloud-Optimized GeoTIFFs into canonical artifacts.

Pressures:
- source_ref is a plain URI (local path or remote URL) — simplest shape
- Remote vs local branching (GDAL virtual filesystem for HTTP/S3)
- COG validation (tiling, overviews) — distinguishes from generic rasters
- I/O metrics in artifact metadata (raided from Hydrops RasterIOStats pattern)
- Overlap with LocalFileConnector — forces connector selection question

Design decisions:
- source_ref = URI/path (no parsing heuristics needed — just classify scheme)
- Lazy = header-only read via rasterio (no data transfer)
- Eager local = wrap in place (same as LocalFileConnector)
- Eager remote = download to workspace
- COG validation: check tiling + overviews, report in metadata
- strict_cog=True rejects non-COG files; False (default) allows with is_cog=False
- I/O accounting: size_bytes, data_transferred in lineage params
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import rasterio
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
    ConnectorCapability,
    MaterializeError,
    MaterializeResult,
)

if TYPE_CHECKING:
    from quarry_core.source_ref import SourceRef


@dataclass(frozen=True)
class COGMetadata:
    """COG-specific structural metadata."""

    block_size: tuple[int, int] | None
    overview_levels: list[int]
    compression: str | None
    driver: str
    dtype: str
    band_count: int
    size_bytes: int

    @property
    def is_cog(self) -> bool:
        """Return True if the file is a valid COG (tiled with overviews)."""
        return self.block_size is not None and len(self.overview_levels) > 0


class COGConnector:
    """Materializes Cloud-Optimized GeoTIFFs into canonical Quarry artifacts.

    Handles both local paths and remote URLs. Validates COG structure
    (tiling, overviews) and reports I/O metrics.
    """

    def __init__(self, *, strict_cog: bool = False):
        """Initialize COG connector.

        Args:
            strict_cog: If True, reject files that aren't valid COGs.
        """
        self._strict_cog = strict_cog

    @property
    def name(self) -> str:
        return "cog"

    @property
    def capabilities(self) -> ConnectorCapability:
        return (
            ConnectorCapability.MATERIALIZE
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
        """Materialize a COG into a canonical artifact.

        source_ref: local path or remote URL (http://, https://, s3://)

        For local files:
            - lazy: read headers only, LAZY_HANDLE backing
            - eager: wrap in place, LOCAL_FILE backing (no copy)
        For remote files:
            - lazy: read headers via /vsicurl/, LAZY_HANDLE backing
            - eager: download to workspace, LOCAL_FILE backing
        """
        source_type = self._classify_source(source_ref)

        # Validate file accessibility and read metadata
        try:
            cog_meta, spatial = self._inspect(source_ref, source_type)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to open: {e}") from e

        # Strict COG validation
        if self._strict_cog and not cog_meta.is_cog:
            raise MaterializeError(source_ref, "not a valid COG (missing tiling or overviews)")

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "cog",
            "source_type": source_type,
            "source_ref": source_ref,
            "lazy": lazy,
            "is_cog": cog_meta.is_cog,
            "overview_levels": cog_meta.overview_levels,
            "block_size": list(cog_meta.block_size) if cog_meta.block_size else None,
            "data_transferred": 0 if lazy else cog_meta.size_bytes,
        }

        # Build artifact metadata
        artifact_meta = {
            "is_cog": cog_meta.is_cog,
            "block_size": cog_meta.block_size,
            "overview_levels": cog_meta.overview_levels,
            "compression": cog_meta.compression,
            "driver": cog_meta.driver,
            "dtype": cog_meta.dtype,
            "size_bytes": cog_meta.size_bytes,
            "source_type": source_type,
        }

        if lazy:
            artifact = Artifact(
                type=ArtifactType.RASTER,
                name=self._derive_name(source_ref),
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=source_ref,
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata=artifact_meta,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"COG headers read, {'valid' if cog_meta.is_cog else 'not COG-compliant'}",
            )

        # Eager: get the file locally
        if source_type == "local":
            local_path = Path(source_ref).resolve()
            artifact = Artifact(
                type=ArtifactType.RASTER,
                name=self._derive_name(source_ref),
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
                strategy="wrapped_local",
                source_ref=source_ref,
                notes=f"Local COG wrapped ({local_path.stat().st_size} bytes)",
            )

        # Remote: download
        download_path = self._download(source_ref, workspace)
        lineage_params["data_transferred"] = download_path.stat().st_size

        artifact = Artifact(
            type=ArtifactType.RASTER,
            name=self._derive_name(source_ref),
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(download_path),
                size_bytes=download_path.stat().st_size,
                content_hash=content_hash(download_path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata=artifact_meta,
        )
        return MaterializeResult(
            artifact=artifact,
            strategy="fetched_remote",
            source_ref=source_ref,
            notes=f"Downloaded {download_path.name} ({download_path.stat().st_size} bytes)",
        )

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get COG metadata without materializing."""
        source_type = self._classify_source(source_ref)

        try:
            cog_meta, spatial = self._inspect(source_ref, source_type)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to open: {e}") from e

        return {
            "crs": spatial.crs,
            "extent": spatial.extent,
            "resolution": spatial.resolution,
            "band_count": spatial.band_count,
            "is_cog": cog_meta.is_cog,
            "block_size": cog_meta.block_size,
            "overview_levels": cog_meta.overview_levels,
            "compression": cog_meta.compression,
            "driver": cog_meta.driver,
            "dtype": cog_meta.dtype,
            "size_bytes": cog_meta.size_bytes,
        }

    # -----------------------------------------------------------------------
    # Public helper: source classification
    # -----------------------------------------------------------------------

    def _classify_source(self, source_ref: SourceRef | str) -> str:
        """Classify source_ref as 'local' or 'remote'."""
        from quarry_core.source_ref import SourceRef, SourceRefKind

        if isinstance(source_ref, SourceRef):
            if source_ref.kind == SourceRefKind.REMOTE_URI:
                return "remote"
            if source_ref.kind in (
                SourceRefKind.LOCAL_PATH,
                SourceRefKind.LOCAL_RASTER,
                SourceRefKind.LOCAL_VECTOR,
            ):
                return "local"

        parsed = urlparse(str(source_ref))
        if parsed.scheme in ("http", "https", "s3", "gs", "az"):
            return "remote"
        return "local"

    # -----------------------------------------------------------------------
    # Private: inspection
    # -----------------------------------------------------------------------

    def _inspect(
        self, source_ref: SourceRef | str, source_type: str
    ) -> tuple[COGMetadata, SpatialDescriptor]:
        """Open the raster and extract metadata without reading pixel data."""
        if source_type == "local":
            path = Path(source_ref).resolve()
            if not path.exists():
                raise MaterializeError(source_ref, f"File not found: {path}")
            open_path: str | Path = path
            size_bytes = path.stat().st_size
        else:
            # Remote: use GDAL virtual filesystem
            open_path = self._vsi_path(source_ref)
            size_bytes = 0  # Can't stat remote without HEAD request

        with rasterio.open(open_path) as ds:
            bounds = ds.bounds
            spatial = SpatialDescriptor(
                crs=str(ds.crs) if ds.crs else None,
                extent=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                resolution=(ds.res[0], ds.res[1]),
                band_count=ds.count,
            )

            # COG validation: check tiling and overviews
            block_shapes = ds.block_shapes
            block_size = block_shapes[0] if block_shapes else None

            overviews = ds.overviews(1)  # overviews for band 1

            compression = ds.profile.get("compress", ds.compression)

            if size_bytes == 0 and source_type == "local":
                size_bytes = Path(source_ref).stat().st_size

            cog_meta = COGMetadata(
                block_size=block_size,
                overview_levels=overviews,
                compression=compression.value if hasattr(compression, "value") else compression,
                driver=ds.driver,
                dtype=str(ds.dtypes[0]),
                band_count=ds.count,
                size_bytes=size_bytes,
            )

        return cog_meta, spatial

    # -----------------------------------------------------------------------
    # Private: helpers
    # -----------------------------------------------------------------------

    def _vsi_path(self, source_ref: SourceRef | str) -> str:
        """Convert a remote URL to GDAL virtual filesystem path."""
        parsed = urlparse(source_ref)
        if parsed.scheme in ("http", "https"):
            return f"/vsicurl/{source_ref}"
        if parsed.scheme == "s3":
            return f"/vsis3/{parsed.netloc}{parsed.path}"
        if parsed.scheme == "gs":
            return f"/vsigs/{parsed.netloc}{parsed.path}"
        if parsed.scheme == "az":
            return f"/vsiaz/{parsed.netloc}{parsed.path}"
        return source_ref

    def _derive_name(self, source_ref: SourceRef | str) -> str:
        """Derive artifact name from source_ref."""
        parsed = urlparse(source_ref)
        if parsed.path:
            return Path(parsed.path).stem
        return Path(source_ref).stem

    def _download(self, source_ref: SourceRef | str, workspace: Path) -> Path:
        """Download a remote COG to workspace."""
        import requests

        filename = self._derive_name(source_ref) + Path(urlparse(source_ref).path).suffix
        download_path = workspace / filename
        download_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            resp = requests.get(source_ref, stream=True, timeout=120)
            resp.raise_for_status()
            with open(download_path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
        except Exception as e:
            if download_path.exists():
                download_path.unlink()
            raise MaterializeError(source_ref, f"Download failed: {e}") from e

        return download_path
