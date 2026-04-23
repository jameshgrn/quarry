"""ObjectStoreConnector — materializes geospatial files from cloud object storage.

Lane: connector

Handles S3, GCS, and Azure Blob Storage using GDAL's built-in virtual filesystems:
- s3://bucket/key → /vsis3/bucket/key
- gs://bucket/blob → /vsigs/bucket/blob
- az://container/blob → /vsiaz/container/blob
- https:// → /vsicurl/https://...

No boto3/gcsfs required — relies on rasterio/fiona's GDAL bindings for I/O.

Design decisions:
- Source ref parsing: extract scheme and map to GDAL /vsi* paths
- File type detection by extension (.tif → RASTER, .gpkg → VECTOR, etc.)
- Lazy mode: open via /vsi* path, read metadata only, produce LAZY_HANDLE
- Eager mode: download to workspace, produce LOCAL_FILE
- Auth via optional credentials dict → GDAL env vars
- Discover not implemented (requires boto3 for listing)
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
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

# File extension to artifact type mapping
RASTER_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".geotiff",
    ".jp2",
    ".hgt",
    ".nc",
    ".vrt",
    ".h5",
    ".hdf5",
    ".hdf",
}
VECTOR_EXTENSIONS = {
    ".shp",
    ".geojson",
    ".gpkg",
    ".kml",
    ".gml",
    ".fgb",
    ".parquet",
    ".geoparquet",
}
TABLE_EXTENSIONS = {".csv"}


@dataclass(frozen=True)
class ParsedSourceRef:
    """Parsed source reference with scheme and path components."""

    scheme: str
    bucket: str | None  # bucket for s3/gs, container for az
    path: str  # key/blob path within bucket/container
    vsi_path: str  # GDAL virtual filesystem path
    original: str  # original source ref


class ObjectStoreConnector:
    """Materializes geospatial files from cloud object storage.

    Supports S3, GCS, Azure, and HTTP(S) URLs. Uses GDAL's virtual
    filesystems for efficient remote access without full downloads.
    """

    def __init__(self, credentials: dict[str, str] | None = None):
        """Initialize ObjectStore connector.

        Args:
            credentials: Optional dict with keys like:
                - aws_access_key_id, aws_secret_access_key, region
                - gcs_credentials (path to service account JSON)
                - azure_storage_account, azure_storage_key
        """
        self._credentials = credentials or {}

    @property
    def name(self) -> str:
        return "object_store"

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
        """Materialize a cloud object into a canonical artifact.

        source_ref formats:
            s3://bucket/key.tif
            gs://bucket/blob.gpkg
            az://container/blob.tif
            https://example.com/data.tif
        """
        parsed = self._parse_source_ref(source_ref)
        file_type = self._detect_file_type(parsed.path)

        # Set up auth env vars if credentials provided
        with self._auth_context():
            try:
                spatial, driver, size_bytes = self._inspect(parsed, file_type)
            except MaterializeError:
                raise
            except Exception as e:
                raise MaterializeError(source_ref, f"Failed to inspect: {e}") from e

        lineage_params: dict[str, Any] = {
            "source": "object_store",
            "scheme": parsed.scheme,
            "vsi_path": parsed.vsi_path,
            "lazy": lazy,
            "file_type": file_type,
            "original_path": parsed.path,
        }
        if parsed.bucket:
            lineage_params["bucket"] = parsed.bucket

        if lazy:
            artifact = self._build_lazy_artifact(parsed, file_type, spatial, driver, lineage_params)
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"Lazy handle for {parsed.scheme}://{parsed.bucket}/{parsed.path}"
                if parsed.bucket
                else f"Lazy handle for {parsed.scheme}://{parsed.path}",
            )

        # Eager: download to workspace
        try:
            local_path = self._download(parsed, file_type, workspace)
        except MaterializeError:
            raise
        except Exception as e:
            raise MaterializeError(source_ref, f"Download failed: {e}") from e

        lineage_params["data_transferred"] = local_path.stat().st_size

        artifact = Artifact(
            type=self._file_type_to_artifact_type(file_type),
            name=self._derive_name(parsed.path),
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(local_path),
                size_bytes=local_path.stat().st_size,
                content_hash=content_hash(local_path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata={
                "source_scheme": parsed.scheme,
                "source_bucket": parsed.bucket,
                "source_path": parsed.path,
                "driver": driver,
                "file_type": file_type,
            },
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="fetched_remote",
            source_ref=source_ref,
            notes=f"Downloaded {local_path.name} ({local_path.stat().st_size} bytes)",
        )

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get metadata about a cloud object without materializing."""
        parsed = self._parse_source_ref(source_ref)
        file_type = self._detect_file_type(parsed.path)

        with self._auth_context():
            try:
                spatial, driver, size_bytes = self._inspect(parsed, file_type)
            except MaterializeError:
                raise
            except Exception as e:
                raise MaterializeError(source_ref, f"Failed to inspect: {e}") from e

        result: dict[str, Any] = {
            "file_type": file_type,
            "scheme": parsed.scheme,
            "vsi_path": parsed.vsi_path,
            "driver": driver,
        }

        if parsed.bucket:
            result["bucket"] = parsed.bucket
        result["path"] = parsed.path

        if spatial.crs:
            result["crs"] = spatial.crs
        if spatial.extent:
            result["extent"] = spatial.extent
        if spatial.resolution:
            result["resolution"] = spatial.resolution
        if spatial.band_count is not None:
            result["band_count"] = spatial.band_count
        if spatial.feature_count is not None:
            result["feature_count"] = spatial.feature_count
        if size_bytes:
            result["size_bytes"] = size_bytes

        return result

    def discover(self, query: str | dict[str, Any] | None = None) -> list:
        """Discover objects in cloud storage.

        Not implemented — would require boto3/gcsfs for listing.
        """
        raise MaterializeError(
            "discover",
            "Discover requires boto3/gcsfs for cloud storage listing. "
            "Use explicit s3://bucket/key paths instead.",
        )

    # -----------------------------------------------------------------------
    # Source ref parsing
    # -----------------------------------------------------------------------

    def _parse_source_ref(self, source_ref: SourceRef | str) -> ParsedSourceRef:
        """Parse source_ref into components and build VSI path."""
        from quarry_core.source_ref import SourceRef

        if isinstance(source_ref, SourceRef):
            raw = source_ref.raw
        else:
            raw = str(source_ref).strip()

        parsed = urlparse(raw)
        scheme = parsed.scheme.lower()

        if scheme == "s3":
            bucket = parsed.netloc
            path = parsed.path.lstrip("/")
            vsi_path = f"/vsis3/{bucket}/{path}"
        elif scheme == "gs":
            bucket = parsed.netloc
            path = parsed.path.lstrip("/")
            vsi_path = f"/vsigs/{bucket}/{path}"
        elif scheme == "az":
            bucket = parsed.netloc  # container name
            path = parsed.path.lstrip("/")
            vsi_path = f"/vsiaz/{bucket}/{path}"
        elif scheme in ("http", "https"):
            bucket = None
            path = raw  # full URL
            vsi_path = f"/vsicurl/{raw}"
        elif scheme == "file":
            # Local file fallback for testing
            bucket = None
            path = parsed.path
            vsi_path = path
        elif not scheme and raw.startswith("/"):
            # Absolute local path fallback
            bucket = None
            path = raw
            vsi_path = raw
        else:
            raise MaterializeError(
                source_ref,
                f"Unsupported scheme '{scheme}'. Supported: s3://, gs://, az://, https://, file://",
            )

        return ParsedSourceRef(
            scheme=scheme or "file",
            bucket=bucket,
            path=path,
            vsi_path=vsi_path,
            original=raw,
        )

    # -----------------------------------------------------------------------
    # File type detection
    # -----------------------------------------------------------------------

    @staticmethod
    def _detect_file_type(path: str) -> Literal["raster", "vector", "table", "unknown"]:
        """Detect file type from extension."""
        lower = path.lower()
        for ext in RASTER_EXTENSIONS:
            if lower.endswith(ext):
                return "raster"
        for ext in VECTOR_EXTENSIONS:
            if lower.endswith(ext):
                return "vector"
        for ext in TABLE_EXTENSIONS:
            if lower.endswith(ext):
                return "table"
        return "unknown"

    @staticmethod
    def _file_type_to_artifact_type(file_type: str) -> ArtifactType:
        """Map file type to artifact type."""
        if file_type == "raster":
            return ArtifactType.RASTER
        if file_type == "vector":
            return ArtifactType.VECTOR
        return ArtifactType.TABLE

    # -----------------------------------------------------------------------
    # Inspection (metadata extraction)
    # -----------------------------------------------------------------------

    def _inspect(
        self, parsed: ParsedSourceRef, file_type: str
    ) -> tuple[SpatialDescriptor, str | None, int | None]:
        """Inspect the file and return spatial metadata, driver, and size."""
        if file_type == "raster":
            return self._inspect_raster(parsed)
        if file_type == "vector":
            return self._inspect_vector(parsed)
        if file_type == "table":
            return self._inspect_table(parsed)
        # Unknown type: try raster first, then vector
        try:
            return self._inspect_raster(parsed)
        except Exception:
            try:
                return self._inspect_vector(parsed)
            except Exception:
                return (SpatialDescriptor(), None, None)

    def _inspect_raster(
        self, parsed: ParsedSourceRef
    ) -> tuple[SpatialDescriptor, str | None, int | None]:
        """Inspect a raster file via rasterio."""
        with rasterio.open(parsed.vsi_path) as ds:
            bounds = ds.bounds
            spatial = SpatialDescriptor(
                crs=str(ds.crs) if ds.crs else None,
                extent=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                resolution=(ds.res[0], ds.res[1]),
                band_count=ds.count,
            )
            return (spatial, ds.driver, None)

    def _inspect_vector(
        self, parsed: ParsedSourceRef
    ) -> tuple[SpatialDescriptor, str | None, int | None]:
        """Inspect a vector file via fiona."""
        import fiona

        with fiona.open(parsed.vsi_path) as src:
            crs = src.crs.to_string() if src.crs else None
            bounds = src.bounds  # (minx, miny, maxx, maxy)
            spatial = SpatialDescriptor(
                crs=crs,
                extent=bounds,
                feature_count=len(src),
            )
            return (spatial, src.driver, None)

    def _inspect_table(
        self, parsed: ParsedSourceRef
    ) -> tuple[SpatialDescriptor, str | None, int | None]:
        """Inspect a table file (CSV)."""
        # For CSV, we can't easily get row count without reading
        # Return minimal metadata
        return (SpatialDescriptor(), None, None)

    # -----------------------------------------------------------------------
    # Lazy artifact building
    # -----------------------------------------------------------------------

    def _build_lazy_artifact(
        self,
        parsed: ParsedSourceRef,
        file_type: str,
        spatial: SpatialDescriptor,
        driver: str | None,
        lineage_params: dict[str, Any],
    ) -> Artifact:
        """Build a lazy-handle artifact."""
        return Artifact(
            type=self._file_type_to_artifact_type(file_type),
            name=self._derive_name(parsed.path),
            backing=BackingStore(
                kind=BackingStoreKind.LAZY_HANDLE,
                uri=parsed.vsi_path,
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata={
                "source_scheme": parsed.scheme,
                "source_bucket": parsed.bucket,
                "source_path": parsed.path,
                "driver": driver,
                "file_type": file_type,
            },
        )

    # -----------------------------------------------------------------------
    # Download (eager materialization)
    # -----------------------------------------------------------------------

    def _download(self, parsed: ParsedSourceRef, file_type: str, workspace: Path) -> Path:
        """Download file to workspace."""
        output_path = workspace / self._derive_name(parsed.path)
        output_path = output_path.with_suffix(self._get_extension(parsed.path))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if file_type == "raster":
            return self._download_raster(parsed, output_path)
        if file_type == "vector":
            return self._download_vector(parsed, output_path)
        # Unknown or table: use requests
        return self._download_raw(parsed, output_path)

    def _download_raster(self, parsed: ParsedSourceRef, output_path: Path) -> Path:
        """Download raster by reading and writing via rasterio."""
        # Read from VSI path and write to local file
        with rasterio.open(parsed.vsi_path) as src:
            profile = src.profile
            profile.update(driver="GTiff")

            with rasterio.open(str(output_path), "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)

        return output_path

    def _download_vector(self, parsed: ParsedSourceRef, output_path: Path) -> Path:
        """Download vector using fiona to read and write locally."""
        import fiona

        with fiona.open(parsed.vsi_path) as src:
            # Copy to local GeoPackage
            schema = src.schema
            crs = src.crs

            with fiona.open(
                str(output_path),
                "w",
                driver="GPKG",
                schema=schema,
                crs=crs,
            ) as dst:
                for feature in src:
                    dst.write(feature)

        return output_path

    def _download_raw(self, parsed: ParsedSourceRef, output_path: Path) -> Path:
        """Download raw bytes using requests (for unknown types)."""
        import requests

        url = parsed.original
        if parsed.scheme in ("s3", "gs", "az"):
            # For cloud storage without proper SDK, we can't easily download
            # Try to use the VSI path with rasterio copy as fallback
            try:
                rasterio.shutil.copy(parsed.vsi_path, str(output_path))
                return output_path
            except Exception:
                raise MaterializeError(
                    parsed.original,
                    f"Cannot download {parsed.scheme}:// without proper GDAL setup or SDK",
                )

        # HTTP(S) download
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise MaterializeError(parsed.original, f"Download failed: {e}") from e

        return output_path

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _derive_name(path: str) -> str:
        """Derive artifact name from path."""
        return Path(path).stem

    @staticmethod
    def _get_extension(path: str) -> str:
        """Get file extension from path."""
        ext = Path(path).suffix
        if not ext:
            return ".bin"
        return ext

    # -----------------------------------------------------------------------
    # Authentication context manager
    # -----------------------------------------------------------------------

    @contextmanager
    def _auth_context(self):
        """Context manager to set GDAL auth env vars from credentials."""
        if not self._credentials:
            yield
            return

        # Save original env vars
        orig_env = {}
        env_vars = {
            # AWS
            "AWS_ACCESS_KEY_ID": self._credentials.get("aws_access_key_id"),
            "AWS_SECRET_ACCESS_KEY": self._credentials.get("aws_secret_access_key"),
            "AWS_REGION": self._credentials.get("region"),
            "AWS_DEFAULT_REGION": self._credentials.get("region"),
            # GCS
            "GOOGLE_APPLICATION_CREDENTIALS": self._credentials.get("gcs_credentials"),
            # Azure
            "AZURE_STORAGE_ACCOUNT": self._credentials.get("azure_storage_account"),
            "AZURE_STORAGE_KEY": self._credentials.get("azure_storage_key"),
        }

        # Set new values
        for key, value in env_vars.items():
            if value:
                orig_env[key] = os.environ.get(key)
                os.environ[key] = value

        try:
            yield
        finally:
            # Restore original values
            for key, orig_value in orig_env.items():
                if orig_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = orig_value
