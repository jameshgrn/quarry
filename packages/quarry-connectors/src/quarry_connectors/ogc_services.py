"""OGCServicesConnector — materializes WMS and WFS services into canonical artifacts.

Lane: connector

Handles OGC Web Map Service (WMS) and Web Feature Service (WFS) protocols.
- WMS: raster imagery via GetMap → RASTER artifact (GeoTIFF/PNG)
- WFS: vector features via GetFeature → VECTOR artifact (GeoPackage via GeoJSON)

Source ref formats:
    "wms::https://example.com/wms::layer_name"
    "wfs::https://example.com/wfs::layer_name"

Or via SourceRef with params: {"service": "wms"|"wfs", "url": str, "layer": str}

Design decisions:
- Uses owslib for GetCapabilities parsing (optional dependency)
- Uses requests for actual data fetching (GetMap, GetFeature)
- WMS eager: downloads raster image, prefers GeoTIFF, falls back to PNG
- WFS eager: downloads GeoJSON, converts to GeoPackage via fiona
- Lazy mode: GetCapabilities only, produces LAZY_HANDLE artifact with metadata
- Spatial descriptor extracted from layer capabilities (CRS, bbox)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

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

# Optional owslib dependency
try:
    from owslib.wfs import WebFeatureService
    from owslib.wms import WebMapService

    HAS_OWSLIB = True
except ImportError:
    HAS_OWSLIB = False
    WebMapService = None  # type: ignore[misc,assignment]
    WebFeatureService = None  # type: ignore[misc,assignment]

# Default WMS version if not specified
_DEFAULT_WMS_VERSION = "1.3.0"
_DEFAULT_WFS_VERSION = "2.0.0"

# Max image dimension for WMS requests
_MAX_WMS_DIMENSION = 4096

# Preferred WMS formats (in order of preference)
_PREFERRED_WMS_FORMATS = [
    "image/tiff",
    "image/geotiff",
    "image/tiff; application=geotiff",
    "image/png",
    "image/jpeg",
]

# Preferred WFS output formats
_PREFERRED_WFS_FORMATS = [
    "application/json",
    "application/geo+json",
    "text/javascript",  # JSONP fallback
    "GML3",
    "GML2",
]


class OGCServicesConnector:
    """Materializes WMS and WFS OGC services into canonical Quarry artifacts."""

    def __init__(
        self,
        service_url: str | None = None,
        service_type: str | None = None,
        version: str | None = None,
        auth: dict[str, str] | None = None,
    ):
        """Initialize connector with optional defaults.

        Args:
            service_url: Default service endpoint URL
            service_type: Default service type ("wms" or "wfs")
            version: Default service version (e.g., "1.3.0" for WMS, "2.0.0" for WFS)
            auth: Optional authentication dict with "username" and "password"
        """
        self._default_service_url = service_url
        self._default_service_type = service_type
        self._default_version = version
        self._auth = auth

    @property
    def name(self) -> str:
        return "ogc_services"

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
        """Materialize an OGC service layer into a canonical artifact.

        Source ref formats:
            "wms::https://example.com/wms::layer_name"
            "wfs::https://example.com/wfs::layer_name"

        Or via SourceRef with params: {"service": "wms"|"wfs", "url": str, "layer": str}
        """
        if not HAS_OWSLIB:
            raise MaterializeError(
                source_ref,
                "owslib is required for OGC services. Install with: pip install owslib",
            )

        service_type, url, layer_name, params = self._parse_source_ref(source_ref)

        if service_type == "wms":
            return self._materialize_wms(source_ref, url, layer_name, params, workspace, lazy)
        if service_type == "wfs":
            return self._materialize_wfs(source_ref, url, layer_name, params, workspace, lazy)

        raise MaterializeError(
            source_ref, f"Unsupported service type: {service_type}. Use 'wms' or 'wfs'."
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """Discover available layers from an OGC service.

        Query as dict supports:
            url: str (service endpoint URL)
            service: str ("wms" or "wfs")
            version: str (optional)
        """
        if not HAS_OWSLIB:
            raise MaterializeError(
                "discover",
                "owslib is required for OGC services. Install with: pip install owslib",
            )

        if isinstance(query, str):
            query = {"url": query}
        if not isinstance(query, dict):
            query = {}

        url = query.get("url", self._default_service_url)
        service_type = query.get("service", self._default_service_type)
        version = query.get("version", self._default_version)

        if not url:
            raise MaterializeError("discover", "No service URL specified")
        if not service_type:
            raise MaterializeError("discover", "No service type specified (wms or wfs)")

        service_type = service_type.lower()

        if service_type == "wms":
            return self._discover_wms(url, version)
        if service_type == "wfs":
            return self._discover_wfs(url, version)

        raise MaterializeError("discover", f"Unsupported service type: {service_type}")

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get layer metadata without materializing data."""
        if not HAS_OWSLIB:
            raise MaterializeError(
                source_ref,
                "owslib is required for OGC services. Install with: pip install owslib",
            )

        service_type, url, layer_name, _ = self._parse_source_ref(source_ref)

        if service_type == "wms":
            return self._metadata_wms(url, layer_name)
        if service_type == "wfs":
            return self._metadata_wfs(url, layer_name)

        raise MaterializeError(
            source_ref, f"Unsupported service type: {service_type}. Use 'wms' or 'wfs'."
        )

    # -----------------------------------------------------------------------
    # Source ref parsing
    # -----------------------------------------------------------------------

    def _parse_source_ref(
        self, source_ref: SourceRef | str
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Parse source_ref into (service_type, url, layer_name, params).

        Returns:
            Tuple of (service_type, endpoint_url, layer_name, extra_params)
        """
        from quarry_core.source_ref import SourceRef

        if isinstance(source_ref, SourceRef):
            params = dict(source_ref.params) if source_ref.params else {}
            service_type = params.get("service", self._default_service_type)
            url = params.get("url", self._default_service_url)
            layer_name = params.get("layer")

            # Also check for bbox, width, height, srs, format in params
            extra_params = {
                k: v
                for k, v in params.items()
                if k in ("bbox", "width", "height", "srs", "format", "crs")
            }

            if not service_type:
                # Try to infer from raw string
                raw = source_ref.raw
                if raw.startswith("wms::"):
                    service_type = "wms"
                elif raw.startswith("wfs::"):
                    service_type = "wfs"

            if not all([service_type, url, layer_name]):
                raise MaterializeError(
                    source_ref,
                    "SourceRef missing required params: service, url, layer",
                )

            return (service_type.lower(), url, layer_name, extra_params)

        # Parse raw string format: "wms::https://example.com/wms::layer_name"
        raw = source_ref.strip() if isinstance(source_ref, str) else str(source_ref)

        if "::" not in raw:
            raise MaterializeError(
                source_ref,
                "Invalid OGC source ref format. Expected: 'wms::url::layer' or 'wfs::url::layer'",
            )

        parts = raw.split("::")
        if len(parts) < 3:
            raise MaterializeError(
                source_ref,
                "Invalid OGC source ref format. Expected: 'wms::url::layer' or 'wfs::url::layer'",
            )

        service_type = parts[0].lower()
        url = parts[1]
        layer_name = parts[2]

        # Any additional parts are extra params (not expected)
        extra_params: dict[str, Any] = {}

        return (service_type, url, layer_name, extra_params)

    # -----------------------------------------------------------------------
    # WMS materialization
    # -----------------------------------------------------------------------

    def _materialize_wms(
        self,
        source_ref: SourceRef | str,
        url: str,
        layer_name: str,
        params: dict[str, Any],
        workspace: Path,
        lazy: bool,
    ) -> MaterializeResult:
        """Materialize a WMS layer."""
        version = self._default_version or _DEFAULT_WMS_VERSION

        try:
            wms = WebMapService(
                url,
                version=version,
                username=self._auth.get("username") if self._auth else None,
                password=self._auth.get("password") if self._auth else None,
            )
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to connect to WMS: {e}") from e

        # Get layer metadata
        try:
            layer = wms[layer_name]
        except KeyError:
            available = list(wms.contents.keys())
            raise MaterializeError(
                source_ref, f"Layer '{layer_name}' not found. Available: {available}"
            ) from None

        # Extract spatial metadata
        bbox = layer.boundingBoxWGS84 or layer.boundingBox
        crs_options = list(layer.crsOptions) if hasattr(layer, "crsOptions") else []
        default_crs = crs_options[0] if crs_options else "EPSG:4326"

        # Build spatial descriptor
        extent = None
        if bbox and len(bbox) >= 4:
            extent = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

        spatial = SpatialDescriptor(
            crs=default_crs if isinstance(default_crs, str) else str(default_crs),
            extent=extent,
        )

        # Get available formats
        formats = (
            list(wms.getOperationByName("GetMap").formatOptions)
            if hasattr(wms, "getOperationByName")
            else []
        )

        if lazy:
            artifact = Artifact(
                type=ArtifactType.RASTER,
                name=layer_name,
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=f"wms://{url}/{layer_name}",
                ),
                spatial=spatial,
                lineage=Lineage(
                    operation="materialize",
                    params={
                        "source": "ogc",
                        "service_type": "wms",
                        "url": url,
                        "layer": layer_name,
                        "lazy": True,
                        "version": version,
                    },
                ),
                metadata={
                    "service_type": "wms",
                    "title": layer.title,
                    "abstract": getattr(layer, "abstract", None),
                    "crs_options": crs_options,
                    "formats": formats,
                },
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"WMS layer '{layer_name}' — metadata only",
            )

        # Eager: download raster via GetMap
        output_path = self._download_wms_map(
            wms, layer_name, extent, default_crs, formats, params, workspace
        )

        # Update spatial with resolution from image dimensions
        # (would need to parse the image for actual dimensions)

        artifact = Artifact(
            type=ArtifactType.RASTER,
            name=layer_name,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(output_path),
                size_bytes=output_path.stat().st_size,
                content_hash=content_hash(output_path),
            ),
            spatial=spatial,
            lineage=Lineage(
                operation="materialize",
                params={
                    "source": "ogc",
                    "service_type": "wms",
                    "url": url,
                    "layer": layer_name,
                    "lazy": False,
                    "version": version,
                },
            ),
            metadata={
                "service_type": "wms",
                "title": layer.title,
                "format": output_path.suffix.lstrip("."),
            },
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="fetched_remote",
            source_ref=source_ref,
            notes=f"Downloaded WMS layer '{layer_name}' ({output_path.stat().st_size} bytes)",
        )

    def _download_wms_map(
        self,
        wms: Any,
        layer_name: str,
        extent: tuple[float, float, float, float] | None,
        crs: str,
        available_formats: list[str],
        params: dict[str, Any],
        workspace: Path,
    ) -> Path:
        """Download a WMS map via GetMap request."""
        # Determine format
        format_override = params.get("format")
        if format_override:
            img_format = format_override
        else:
            img_format = self._select_format(available_formats, _PREFERRED_WMS_FORMATS)
            if not img_format:
                img_format = "image/png"

        # Determine bbox
        bbox = params.get("bbox")
        if bbox:
            if isinstance(bbox, str):
                bbox = tuple(float(x) for x in bbox.split(","))
        else:
            bbox = extent

        if not bbox:
            raise MaterializeError(layer_name, "No bbox available for WMS request")

        # Determine image dimensions
        width = params.get("width", 1024)
        height = params.get("height", 1024)

        # Clamp to max dimension
        max_dim = max(width, height)
        if max_dim > _MAX_WMS_DIMENSION:
            scale = _MAX_WMS_DIMENSION / max_dim
            width = int(width * scale)
            height = int(height * scale)

        # Determine SRS
        srs = params.get("srs") or params.get("crs") or crs

        # Build GetMap URL
        getmap_url = (
            wms.getOperationByName("GetMap").methods[0]["url"]
            if hasattr(wms, "getOperationByName")
            else wms.url
        )

        request_params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": wms.version,
            "LAYERS": layer_name,
            "STYLES": "",
            "CRS" if wms.version >= "1.3.0" else "SRS": srs,
            "BBOX": ",".join(str(x) for x in bbox),
            "WIDTH": width,
            "HEIGHT": height,
            "FORMAT": img_format,
            "TRANSPARENT": "TRUE",
        }

        url = f"{getmap_url}?{urlencode(request_params)}"

        # Download
        try:
            resp = requests.get(
                url,
                stream=True,
                timeout=120,
                auth=(self._auth["username"], self._auth["password"]) if self._auth else None,
            )
            resp.raise_for_status()
        except Exception as e:
            raise MaterializeError(layer_name, f"WMS GetMap failed: {e}") from e

        # Determine file extension
        ext = ".tif"
        if "png" in img_format.lower():
            ext = ".png"
        elif "jpeg" in img_format.lower() or "jpg" in img_format.lower():
            ext = ".jpg"
        elif "tiff" in img_format.lower():
            ext = ".tif"

        output_path = workspace / f"{layer_name}{ext}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path

    def _discover_wms(self, url: str, version: str | None) -> list[CatalogEntry]:
        """Discover WMS layers."""
        version = version or _DEFAULT_WMS_VERSION

        try:
            wms = WebMapService(url, version=version)
        except Exception as e:
            raise MaterializeError(url, f"Failed to connect to WMS: {e}") from e

        entries = []
        for layer_name, layer in wms.contents.items():
            bbox = layer.boundingBoxWGS84 or layer.boundingBox
            spatial_hint = {}
            if bbox and len(bbox) >= 4:
                spatial_hint = {
                    "extent": (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    "crs": "EPSG:4326",
                }

            crs_options = list(layer.crsOptions) if hasattr(layer, "crsOptions") else []

            entries.append(
                CatalogEntry(
                    source_ref=f"wms::{url}::{layer_name}",
                    name=layer_name,
                    description=layer.title or "",
                    spatial_hint=spatial_hint,
                    metadata={
                        "title": layer.title,
                        "abstract": getattr(layer, "abstract", None),
                        "crs_options": crs_options,
                    },
                )
            )

        return entries

    def _metadata_wms(self, url: str, layer_name: str) -> dict[str, Any]:
        """Get WMS layer metadata."""
        version = self._default_version or _DEFAULT_WMS_VERSION

        try:
            wms = WebMapService(url, version=version)
        except Exception as e:
            raise MaterializeError(url, f"Failed to connect to WMS: {e}") from e

        try:
            layer = wms[layer_name]
        except KeyError:
            available = list(wms.contents.keys())
            raise MaterializeError(layer_name, f"Layer not found. Available: {available}") from None

        bbox = layer.boundingBoxWGS84 or layer.boundingBox
        crs_options = list(layer.crsOptions) if hasattr(layer, "crsOptions") else []
        formats = (
            list(wms.getOperationByName("GetMap").formatOptions)
            if hasattr(wms, "getOperationByName")
            else []
        )

        keywords = []
        if hasattr(layer, "keywords"):
            keywords = list(layer.keywords) if layer.keywords else []

        return {
            "title": layer.title,
            "abstract": getattr(layer, "abstract", None),
            "crs_options": crs_options,
            "bbox": bbox,
            "formats": formats,
            "keywords": keywords,
        }

    # -----------------------------------------------------------------------
    # WFS materialization
    # -----------------------------------------------------------------------

    def _materialize_wfs(
        self,
        source_ref: SourceRef | str,
        url: str,
        layer_name: str,
        params: dict[str, Any],
        workspace: Path,
        lazy: bool,
    ) -> MaterializeResult:
        """Materialize a WFS layer."""
        version = self._default_version or _DEFAULT_WFS_VERSION

        try:
            wfs = WebFeatureService(url, version=version)
        except Exception as e:
            raise MaterializeError(source_ref, f"Failed to connect to WFS: {e}") from e

        # Get layer metadata
        try:
            feature_type = wfs[layer_name]
        except KeyError:
            available = list(wfs.contents.keys())
            raise MaterializeError(
                source_ref, f"Layer '{layer_name}' not found. Available: {available}"
            ) from None

        # Extract spatial metadata
        bbox = feature_type.boundingBoxWGS84 if hasattr(feature_type, "boundingBoxWGS84") else None
        if not bbox and hasattr(feature_type, "boundingBox"):
            bbox = feature_type.boundingBox

        crs_options = []
        if hasattr(feature_type, "crsOptions"):
            crs_options = list(feature_type.crsOptions)
        default_crs = crs_options[0] if crs_options else "EPSG:4326"

        extent = None
        if bbox and len(bbox) >= 4:
            extent = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

        spatial = SpatialDescriptor(
            crs=default_crs if isinstance(default_crs, str) else str(default_crs),
            extent=extent,
        )

        if lazy:
            artifact = Artifact(
                type=ArtifactType.VECTOR,
                name=layer_name,
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=f"wfs://{url}/{layer_name}",
                ),
                spatial=spatial,
                lineage=Lineage(
                    operation="materialize",
                    params={
                        "source": "ogc",
                        "service_type": "wfs",
                        "url": url,
                        "layer": layer_name,
                        "lazy": True,
                        "version": version,
                    },
                ),
                metadata={
                    "service_type": "wfs",
                    "title": feature_type.title,
                    "abstract": getattr(feature_type, "abstract", None),
                    "crs_options": crs_options,
                },
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"WFS layer '{layer_name}' — metadata only",
            )

        # Eager: download features via GetFeature
        output_path = self._download_wfs_features(
            wfs, layer_name, extent, default_crs, params, workspace
        )

        # Count features for spatial descriptor
        feature_count = self._count_features_in_geopackage(output_path)
        spatial = SpatialDescriptor(
            crs=default_crs if isinstance(default_crs, str) else str(default_crs),
            extent=extent,
            feature_count=feature_count,
        )

        artifact = Artifact(
            type=ArtifactType.VECTOR,
            name=layer_name,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(output_path),
                size_bytes=output_path.stat().st_size,
                content_hash=content_hash(output_path),
            ),
            spatial=spatial,
            lineage=Lineage(
                operation="materialize",
                params={
                    "source": "ogc",
                    "service_type": "wfs",
                    "url": url,
                    "layer": layer_name,
                    "lazy": False,
                    "version": version,
                },
            ),
            metadata={
                "service_type": "wfs",
                "title": feature_type.title,
            },
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="fetched_remote",
            source_ref=source_ref,
            notes=f"Downloaded WFS layer '{layer_name}' ({output_path.stat().st_size} bytes)",
        )

    def _download_wfs_features(
        self,
        wfs: Any,
        layer_name: str,
        extent: tuple[float, float, float, float] | None,
        crs: str,
        params: dict[str, Any],
        workspace: Path,
    ) -> Path:
        """Download WFS features via GetFeature and convert to GeoPackage."""
        # Determine output format
        available_formats = []
        if hasattr(wfs, "getOperationByName"):
            try:
                available_formats = list(wfs.getOperationByName("GetFeature").formatOptions)
            except Exception:
                pass

        output_format = self._select_format(available_formats, _PREFERRED_WFS_FORMATS)
        if not output_format:
            output_format = "application/json"

        # Build GetFeature request
        bbox = params.get("bbox")
        if bbox:
            if isinstance(bbox, str):
                bbox = tuple(float(x) for x in bbox.split(","))
        else:
            bbox = extent

        # Use owslib to build the request
        try:
            if bbox:
                resp = wfs.getfeature(
                    typename=[layer_name],
                    bbox=bbox,
                    srsname=crs,
                    outputFormat=output_format,
                )
            else:
                resp = wfs.getfeature(
                    typename=[layer_name],
                    outputFormat=output_format,
                )
        except Exception as e:
            raise MaterializeError(layer_name, f"WFS GetFeature failed: {e}") from e

        # Read response content
        if hasattr(resp, "read"):
            content = resp.read()
        else:
            content = resp

        # Save to temp file first
        suffix = ".geojson" if "json" in output_format.lower() else ".gml"
        temp_path = workspace / f"{layer_name}_temp{suffix}"
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, bytes):
            temp_path.write_bytes(content)
        else:
            temp_path.write_text(str(content), encoding="utf-8")

        # Convert to GeoPackage via fiona
        output_path = workspace / f"{layer_name}.gpkg"

        try:
            import fiona

            with fiona.open(temp_path) as src:
                schema = src.schema
                crs_obj = src.crs

                with fiona.open(output_path, "w", driver="GPKG", schema=schema, crs=crs_obj) as dst:
                    for feature in src:
                        dst.write(feature)

            # Clean up temp file
            temp_path.unlink()

        except Exception as e:
            # If conversion fails, return the raw file
            if temp_path.exists():
                return temp_path
            raise MaterializeError(
                layer_name, f"Failed to convert WFS response to GeoPackage: {e}"
            ) from e

        return output_path

    def _discover_wfs(self, url: str, version: str | None) -> list[CatalogEntry]:
        """Discover WFS layers."""
        version = version or _DEFAULT_WFS_VERSION

        try:
            wfs = WebFeatureService(url, version=version)
        except Exception as e:
            raise MaterializeError(url, f"Failed to connect to WFS: {e}") from e

        entries = []
        for layer_name, feature_type in wfs.contents.items():
            bbox = None
            if hasattr(feature_type, "boundingBoxWGS84"):
                bbox = feature_type.boundingBoxWGS84
            elif hasattr(feature_type, "boundingBox"):
                bbox = feature_type.boundingBox

            spatial_hint = {}
            if bbox and len(bbox) >= 4:
                spatial_hint = {
                    "extent": (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    "crs": "EPSG:4326",
                }

            crs_options = []
            if hasattr(feature_type, "crsOptions"):
                crs_options = list(feature_type.crsOptions)

            entries.append(
                CatalogEntry(
                    source_ref=f"wfs::{url}::{layer_name}",
                    name=layer_name,
                    description=feature_type.title or "",
                    spatial_hint=spatial_hint,
                    metadata={
                        "title": feature_type.title,
                        "abstract": getattr(feature_type, "abstract", None),
                        "crs_options": crs_options,
                    },
                )
            )

        return entries

    def _metadata_wfs(self, url: str, layer_name: str) -> dict[str, Any]:
        """Get WFS layer metadata."""
        version = self._default_version or _DEFAULT_WFS_VERSION

        try:
            wfs = WebFeatureService(url, version=version)
        except Exception as e:
            raise MaterializeError(url, f"Failed to connect to WFS: {e}") from e

        try:
            feature_type = wfs[layer_name]
        except KeyError:
            available = list(wfs.contents.keys())
            raise MaterializeError(layer_name, f"Layer not found. Available: {available}") from None

        bbox = None
        if hasattr(feature_type, "boundingBoxWGS84"):
            bbox = feature_type.boundingBoxWGS84
        elif hasattr(feature_type, "boundingBox"):
            bbox = feature_type.boundingBox

        crs_options = []
        if hasattr(feature_type, "crsOptions"):
            crs_options = list(feature_type.crsOptions)

        available_formats = []
        if hasattr(wfs, "getOperationByName"):
            try:
                available_formats = list(wfs.getOperationByName("GetFeature").formatOptions)
            except Exception:
                pass

        keywords = []
        if hasattr(feature_type, "keywords"):
            keywords = list(feature_type.keywords) if feature_type.keywords else []

        return {
            "title": feature_type.title,
            "abstract": getattr(feature_type, "abstract", None),
            "crs_options": crs_options,
            "bbox": bbox,
            "formats": available_formats,
            "keywords": keywords,
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _select_format(available: list[str], preferred: list[str]) -> str | None:
        """Select the best format from available based on preference order."""
        available_lower = [f.lower() for f in available]
        for pref in preferred:
            if pref.lower() in available_lower:
                idx = available_lower.index(pref.lower())
                return available[idx]
            # Partial match
            for i, avail in enumerate(available_lower):
                if pref.lower() in avail:
                    return available[i]
        return available[0] if available else None

    @staticmethod
    def _count_features_in_geopackage(path: Path) -> int | None:
        """Count features in a GeoPackage file."""
        try:
            import fiona

            with fiona.open(path) as src:
                return len(src)
        except Exception:
            return None
