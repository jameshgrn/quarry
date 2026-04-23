"""OpenTopographyConnector — materializes DEM raster data from OpenTopography API.

Lane: connector

Downloads Digital Elevation Model (DEM) raster data from the OpenTopography
REST API (https://portal.opentopography.org/API/globaldem).

Design decisions:
- API key required for most datasets (constructor param)
- Source ref formats: "opentopo://DATASET_ID" with bbox params, or raw dataset ID
- Lazy mode: validate dataset + bbox, produce LAZY_HANDLE with spatial descriptor
- Eager mode: download GeoTIFF to workspace, read with rasterio for full metadata
- Discover: returns CatalogEntry for each known global DEM dataset
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs

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

# Known OpenTopography global DEM datasets
KNOWN_DATASETS = {
    "SRTMGL3",
    "SRTMGL1",
    "SRTMGL1_E",
    "AW3D30",
    "AW3D30_E",
    "SRTM15Plus",
    "NASADEM",
    "COP30",
    "COP90",
    "EU_DTM",
    "GEDI_L3",
}

# Dataset metadata for discover/metadata
_DATASET_INFO: dict[str, dict[str, Any]] = {
    "SRTMGL3": {"description": "SRTM GL3 90m", "resolution": 90, "coverage": "60S-60N"},
    "SRTMGL1": {"description": "SRTM GL1 30m", "resolution": 30, "coverage": "60S-60N"},
    "SRTMGL1_E": {
        "description": "SRTM GL1 Ellipsoidal 30m",
        "resolution": 30,
        "coverage": "60S-60N",
    },
    "AW3D30": {"description": "ALOS World 3D 30m", "resolution": 30, "coverage": "82S-82N"},
    "AW3D30_E": {
        "description": "ALOS World 3D Ellipsoidal 30m",
        "resolution": 30,
        "coverage": "82S-82N",
    },
    "SRTM15Plus": {
        "description": "SRTM15+ Global Bathymetry 500m",
        "resolution": 500,
        "coverage": "global",
    },
    "NASADEM": {"description": "NASADEM 30m", "resolution": 30, "coverage": "60S-60N"},
    "COP30": {"description": "Copernicus DEM 30m", "resolution": 30, "coverage": "global"},
    "COP90": {"description": "Copernicus DEM 90m", "resolution": 90, "coverage": "global"},
    "EU_DTM": {"description": "European DTM 30m", "resolution": 30, "coverage": "Europe"},
    "GEDI_L3": {"description": "GEDI L3 1km DTM", "resolution": 1000, "coverage": "52S-52N"},
}

# Global bbox for spatial hints
_GLOBAL_EXTENT = (-180.0, -90.0, 180.0, 90.0)


class OpenTopographyConnector:
    """Materializes DEM raster data from OpenTopography API.

    Configured with an API key (required for most datasets) and optional API URL.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str = "https://portal.opentopography.org/API/globaldem",
    ):
        self._api_key = api_key
        self._api_url = api_url.rstrip("/")

    @property
    def name(self) -> str:
        return "opentopography"

    @property
    def capabilities(self) -> ConnectorCapability:
        return (
            ConnectorCapability.MATERIALIZE
            | ConnectorCapability.DISCOVER
            | ConnectorCapability.AUTHENTICATE
            | ConnectorCapability.MATERIALIZE_LAZY
            | ConnectorCapability.METADATA_ONLY
        )

    def authenticate(self, credentials: dict[str, Any]) -> None:
        """Set API key from credentials.

        Args:
            credentials: Dict with "api_key" key.

        Raises:
            MaterializeError: If api_key not provided.
        """
        api_key = credentials.get("api_key")
        if not api_key:
            raise MaterializeError("authenticate", "credentials must contain 'api_key'")
        self._api_key = api_key

    def materialize(
        self,
        source_ref: SourceRef | str,
        workspace: Path,
        *,
        lazy: bool = False,
    ) -> MaterializeResult:
        """Materialize OpenTopography DEM data into a canonical artifact.

        source_ref formats:
            "SRTMGL1" — dataset ID, requires bbox in SourceRef params
            "opentopo://SRTMGL1?west=-120&south=35&east=-119&north=36" — full URL
            SourceRef with params: {"dataset": "SRTMGL1", "bbox": (w, s, e, n)}
            SourceRef with params: {"dataset": "SRTMGL1", "west": w, "south": s, ...}

        Args:
            source_ref: Dataset reference with bbox.
            workspace: Where to save downloaded GeoTIFF (eager mode).
            lazy: If True, produce LAZY_HANDLE without downloading.

        Returns:
            MaterializeResult with RASTER artifact.

        Raises:
            MaterializeError: If dataset unknown, bbox missing, or API error.
        """
        dataset_id, bbox = self._parse_source_ref(source_ref)

        # Validate dataset
        if dataset_id not in KNOWN_DATASETS:
            raise MaterializeError(
                source_ref,
                f"Unknown dataset '{dataset_id}'. Known: {sorted(KNOWN_DATASETS)}",
            )

        # Validate bbox
        if bbox is None:
            raise MaterializeError(
                source_ref,
                "Bounding box required. Provide via params: "
                "bbox=(w,s,e,n) or west/south/east/north",
            )

        west, south, east, north = bbox

        # Build lineage params
        lineage_params: dict[str, Any] = {
            "source": "opentopography",
            "dataset": dataset_id,
            "bbox": bbox,
            "lazy": lazy,
            "api_url": self._api_url,
        }

        # Get dataset info for spatial descriptor
        ds_info = _DATASET_INFO.get(dataset_id, {})
        resolution = ds_info.get("resolution")

        if lazy:
            # Lazy mode: construct API URL but don't fetch
            api_url = self._build_api_url(dataset_id, west, south, east, north)

            # Estimate resolution in degrees (rough approximation)
            res_degrees = None
            if resolution:
                # ~111km per degree at equator
                res_degrees = (resolution / 111000.0, resolution / 111000.0)

            spatial = SpatialDescriptor(
                crs="EPSG:4326",
                extent=bbox,
                resolution=res_degrees,
                band_count=1,
            )

            artifact = Artifact(
                type=ArtifactType.RASTER,
                name=f"{dataset_id}_{west}_{south}_{east}_{north}",
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=api_url,
                ),
                spatial=spatial,
                lineage=Lineage(operation="materialize", params=lineage_params),
                metadata={
                    "dataset": dataset_id,
                    "description": ds_info.get("description", ""),
                    "coverage": ds_info.get("coverage", ""),
                    "resolution_meters": resolution,
                },
            )

            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"OpenTopography {dataset_id} — metadata only, not downloaded",
            )

        # Eager mode: download the DEM
        if not self._api_key:
            raise MaterializeError(
                source_ref,
                "API key required for eager materialization. "
                "Provide via constructor or authenticate().",
            )

        download_path = self._download_dem(dataset_id, west, south, east, north, workspace)

        # Read spatial metadata from downloaded file
        spatial = self._read_spatial_metadata(download_path)

        artifact = Artifact(
            type=ArtifactType.RASTER,
            name=download_path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(download_path),
                size_bytes=download_path.stat().st_size,
                content_hash=content_hash(download_path),
            ),
            spatial=spatial,
            lineage=Lineage(operation="materialize", params=lineage_params),
            metadata={
                "dataset": dataset_id,
                "description": ds_info.get("description", ""),
                "coverage": ds_info.get("coverage", ""),
                "resolution_meters": resolution,
                "download_path": str(download_path),
            },
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="fetched_remote",
            source_ref=source_ref,
            notes=f"Downloaded {download_path.name} ({download_path.stat().st_size} bytes)",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """List available DEM datasets.

        Returns CatalogEntry for each known OpenTopography dataset.
        """
        entries = []
        for dataset_id, info in _DATASET_INFO.items():
            spatial_hint = {
                "extent": _GLOBAL_EXTENT,
                "crs": "EPSG:4326",
                "coverage": info.get("coverage", ""),
                "resolution_meters": info.get("resolution"),
            }
            entries.append(
                CatalogEntry(
                    source_ref=f"opentopo://{dataset_id}",
                    name=dataset_id,
                    description=info.get("description", ""),
                    spatial_hint=spatial_hint,
                    metadata={
                        "dataset": dataset_id,
                        "resolution": info.get("resolution"),
                        "coverage": info.get("coverage"),
                    },
                )
            )
        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get metadata for a dataset without materializing.

        Args:
            source_ref: Dataset ID or "opentopo://DATASET_ID".

        Returns:
            Dict with dataset_id, description, crs, resolution_info, coverage.
        """
        dataset_id, _ = self._parse_source_ref(source_ref)

        if dataset_id not in KNOWN_DATASETS:
            raise MaterializeError(
                source_ref,
                f"Unknown dataset '{dataset_id}'. Known: {sorted(KNOWN_DATASETS)}",
            )

        ds_info = _DATASET_INFO.get(dataset_id, {})
        return {
            "dataset_id": dataset_id,
            "description": ds_info.get("description", ""),
            "crs": "EPSG:4326",
            "resolution_meters": ds_info.get("resolution"),
            "coverage": ds_info.get("coverage", ""),
            "api_url": self._api_url,
        }

    # -----------------------------------------------------------------------
    # Private: source ref parsing
    # -----------------------------------------------------------------------

    def _parse_source_ref(
        self, source_ref: SourceRef | str
    ) -> tuple[str, tuple[float, float, float, float] | None]:
        """Parse source_ref into (dataset_id, bbox).

        Bbox is (west, south, east, north) in WGS84 decimal degrees.
        """
        from quarry_core.source_ref import SourceRef

        if isinstance(source_ref, SourceRef):
            params = dict(source_ref.params) if source_ref.params else {}

            # Check for opentopo:// URL in raw
            if source_ref.raw.startswith("opentopo://"):
                return self._parse_opentopo_url(source_ref.raw)

            # Dataset from params or raw
            dataset_id = params.get("dataset")
            if not dataset_id:
                # Raw might be just the dataset ID
                dataset_id = source_ref.raw.strip()

            # Bbox from params
            bbox = self._extract_bbox_from_params(params)
            return dataset_id, bbox

        # String source_ref
        raw = source_ref.strip()

        if raw.startswith("opentopo://"):
            return self._parse_opentopo_url(raw)

        # Plain dataset ID
        return raw, None

    def _parse_opentopo_url(self, url: str) -> tuple[str, tuple[float, float, float, float] | None]:
        """Parse opentopo://DATASET_ID?west=...&south=...&east=...&north=... format."""
        # Remove opentopo:// prefix
        path = url[11:]  # len("opentopo://") == 11

        # Split dataset from query string
        if "?" in path:
            dataset_id, query_string = path.split("?", 1)
        else:
            return path, None

        # Parse query params
        parsed = parse_qs(query_string)

        try:
            west = float(parsed["west"][0])
            south = float(parsed["south"][0])
            east = float(parsed["east"][0])
            north = float(parsed["north"][0])
            return dataset_id, (west, south, east, north)
        except (KeyError, IndexError, ValueError):
            return dataset_id, None

    def _extract_bbox_from_params(
        self, params: dict[str, Any]
    ) -> tuple[float, float, float, float] | None:
        """Extract bbox from SourceRef params.

        Supports:
        - bbox: (west, south, east, north) tuple
        - west, south, east, north individual keys
        """
        # Try bbox tuple first
        bbox = params.get("bbox")
        if bbox:
            try:
                w, s, e, n = bbox
                return (float(w), float(s), float(e), float(n))
            except (ValueError, TypeError):
                pass

        # Try individual keys
        try:
            west = params.get("west")
            south = params.get("south")
            east = params.get("east")
            north = params.get("north")

            if all(v is not None for v in (west, south, east, north)):
                return (float(west), float(south), float(east), float(north))
        except (ValueError, TypeError):
            pass

        return None

    # -----------------------------------------------------------------------
    # Private: API interaction
    # -----------------------------------------------------------------------

    def _build_api_url(
        self, dataset_id: str, west: float, south: float, east: float, north: float
    ) -> str:
        """Build OpenTopography API URL for DEM request."""
        params = {
            "demtype": dataset_id,
            "south": south,
            "north": north,
            "west": west,
            "east": east,
            "outputFormat": "GTiff",
        }
        if self._api_key:
            params["API_Key"] = self._api_key

        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self._api_url}?{query}"

    def _download_dem(
        self,
        dataset_id: str,
        west: float,
        south: float,
        east: float,
        north: float,
        workspace: Path,
    ) -> Path:
        """Download DEM from OpenTopography API."""
        api_url = self._build_api_url(dataset_id, west, south, east, north)

        # Create filename
        filename = f"{dataset_id}_{west}_{south}_{east}_{north}.tif"
        download_path = workspace / filename
        download_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            resp = requests.get(api_url, stream=True, timeout=300)
            resp.raise_for_status()

            with open(download_path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
        except requests.HTTPError as e:
            if download_path.exists():
                download_path.unlink()
            raise MaterializeError(
                api_url,
                f"OpenTopography API error: {e.response.status_code} - {e.response.text[:200]}",
            ) from e
        except Exception as e:
            if download_path.exists():
                download_path.unlink()
            raise MaterializeError(
                api_url,
                f"Download failed: {e}",
            ) from e

        return download_path

    def _read_spatial_metadata(self, path: Path) -> SpatialDescriptor:
        """Read spatial metadata from downloaded GeoTIFF."""
        try:
            import rasterio

            with rasterio.open(path) as src:
                crs = src.crs.to_string() if src.crs else "EPSG:4326"
                bounds = src.bounds
                extent = (bounds.left, bounds.bottom, bounds.right, bounds.top)

                # Get resolution
                resolution = None
                if src.transform:
                    resolution = (abs(src.transform.a), abs(src.transform.e))

                return SpatialDescriptor(
                    crs=crs,
                    extent=extent,
                    resolution=resolution,
                    band_count=src.count,
                )
        except Exception as e:
            raise MaterializeError(
                str(path), f"Failed to read spatial metadata from downloaded DEM: {e}"
            ) from e
