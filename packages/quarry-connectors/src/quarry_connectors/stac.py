"""STACConnector — materializes STAC items into canonical artifacts.

Pressures:
- source_ref (structured reference needed: item ID + asset key)
- discoverability (catalog search with spatial/temporal filters)
- metadata richness (STAC items are metadata-rich)
- lazy vs eager materialization
- asset selection (items have multiple assets)

Design decisions:
- Connector is configured with an API endpoint and optional defaults
- source_ref is an item self-link URL or "collection/item_id" shorthand
- Asset selection: explicit key > connector default > first geotiff > error on ambiguity
- Lazy materialization creates artifact with full metadata but LAZY_HANDLE backing
- Eager materialization downloads the asset to workspace
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import pystac
import requests
from pystac_client import Client
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

# Map STAC media types to artifact types
_RASTER_MEDIA_TYPES = {
    "image/tiff",
    "image/tiff; application=geotiff",
    "image/tiff; application=geotiff; profile=cloud-optimized",
    "image/vnd.stac.geotiff",
    "application/x-geotiff",
    pystac.MediaType.GEOTIFF,
    pystac.MediaType.COG,
}

_VECTOR_MEDIA_TYPES = {
    "application/geo+json",
    "application/geopackage+sqlite3",
    pystac.MediaType.GEOJSON,
    pystac.MediaType.GEOPACKAGE,
}


def _infer_artifact_type(asset: pystac.Asset) -> ArtifactType:
    """Infer artifact type from STAC asset media type."""
    media_type = asset.media_type or ""
    if media_type in _RASTER_MEDIA_TYPES:
        return ArtifactType.RASTER
    if media_type in _VECTOR_MEDIA_TYPES:
        return ArtifactType.VECTOR
    # Fallback: check href extension
    ext = Path(urlparse(asset.href).path).suffix.lower()
    if ext in {".tif", ".tiff", ".jp2", ".nc", ".vrt"}:
        return ArtifactType.RASTER
    if ext in {".geojson", ".gpkg", ".fgb", ".parquet"}:
        return ArtifactType.VECTOR
    return ArtifactType.RASTER  # default assumption for remote imagery


class STACConnector:
    """Materializes STAC catalog items into canonical Quarry artifacts.

    Configured with an API endpoint. Optional defaults for collection and asset key.
    """

    def __init__(
        self,
        api_url: str,
        collection: str | None = None,
        asset_key: str | None = None,
    ):
        self._api_url = api_url
        self._default_collection = collection
        self._default_asset_key = asset_key
        self._client: Client | None = None

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client.open(self._api_url)
        return self._client

    @property
    def name(self) -> str:
        return "stac"

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
        """Materialize a STAC item asset into a canonical artifact.

        source_ref formats:
            "collection_id/item_id"          — uses connector's default asset key
            "collection_id/item_id::asset_key" — explicit asset selection
            "https://...item self-link"      — full item URL

        FRICTION NOTE: source_ref as str is starting to feel underfit here.
        The :: separator is a convention, not a contract. A SourceRef type
        would make asset_key, collection, and item_id first-class.
        """
        item, asset_key = self._resolve_source_ref(source_ref)
        asset = self._select_asset(item, asset_key)
        artifact_type = _infer_artifact_type(asset)

        # Extract spatial metadata from STAC item (always available, even lazy)
        spatial = self._extract_spatial(item)
        stac_metadata = self._extract_metadata(item, asset_key)

        if lazy:
            artifact = Artifact(
                type=artifact_type,
                name=item.id,
                backing=BackingStore(
                    kind=BackingStoreKind.LAZY_HANDLE,
                    uri=asset.href,
                ),
                spatial=spatial,
                lineage=Lineage(
                    operation="materialize",
                    params={
                        "source": "stac",
                        "api_url": self._api_url,
                        "collection": item.collection_id,
                        "item_id": item.id,
                        "asset_key": asset_key,
                        "lazy": True,
                    },
                ),
                metadata=stac_metadata,
            )
            return MaterializeResult(
                artifact=artifact,
                strategy="lazy_handle",
                source_ref=source_ref,
                notes=f"STAC item {item.id}, asset '{asset_key}' — metadata only, not downloaded",
            )

        # Eager: download the asset
        download_path = self._download_asset(asset, item.id, asset_key, workspace)

        artifact = Artifact(
            type=artifact_type,
            name=item.id,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(download_path),
                size_bytes=download_path.stat().st_size,
                content_hash=content_hash(download_path),
            ),
            spatial=spatial,
            lineage=Lineage(
                operation="materialize",
                params={
                    "source": "stac",
                    "api_url": self._api_url,
                    "collection": item.collection_id,
                    "item_id": item.id,
                    "asset_key": asset_key,
                    "lazy": False,
                },
            ),
            metadata=stac_metadata,
        )

        return MaterializeResult(
            artifact=artifact,
            strategy="fetched_remote",
            source_ref=source_ref,
            notes=f"Downloaded {download_path.name} ({download_path.stat().st_size} bytes)",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """Search STAC catalog for items.

        query as dict supports:
            bbox: [xmin, ymin, xmax, ymax]
            datetime: "2023-01-01/2023-12-31" or single date
            collections: ["collection_id"]
            max_items: int (default 20)
            query: additional STAC query filters
        """
        client = self._get_client()

        if query is None:
            query = {}
        if isinstance(query, str):
            # Treat string as collection name
            query = {"collections": [query]}

        collections = query.get("collections")
        if collections is None and self._default_collection:
            collections = [self._default_collection]

        search = client.search(
            collections=collections,
            bbox=query.get("bbox"),
            datetime=query.get("datetime"),
            max_items=query.get("max_items", 20),
        )

        entries = []
        for item in search.items():
            bbox = item.bbox or []
            spatial_hint = {}
            if len(bbox) >= 4:
                spatial_hint = {
                    "extent": (bbox[0], bbox[1], bbox[2], bbox[3]),
                    "crs": "EPSG:4326",  # STAC bbox is always WGS84
                }
            if item.datetime:
                spatial_hint["datetime"] = item.datetime.isoformat()

            asset_keys = list(item.assets.keys())
            entries.append(
                CatalogEntry(
                    source_ref=f"{item.collection_id}/{item.id}",
                    name=item.id,
                    description=item.properties.get("description", ""),
                    spatial_hint=spatial_hint,
                    metadata={
                        "collection": item.collection_id,
                        "asset_keys": asset_keys,
                        "properties": dict(item.properties),
                    },
                )
            )

        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get full STAC item metadata without downloading any assets."""
        item, asset_key = self._resolve_source_ref(source_ref)
        asset = self._select_asset(item, asset_key)

        return {
            "item_id": item.id,
            "collection": item.collection_id,
            "bbox": item.bbox,
            "datetime": item.datetime.isoformat() if item.datetime else None,
            "properties": dict(item.properties),
            "asset_key": asset_key,
            "asset_href": asset.href,
            "asset_media_type": asset.media_type,
            "asset_roles": asset.roles or [],
            "all_asset_keys": list(item.assets.keys()),
            "stac_extensions": item.stac_extensions or [],
        }

    # -----------------------------------------------------------------------
    # Private: source_ref resolution
    # -----------------------------------------------------------------------

    def _resolve_source_ref(self, source_ref: SourceRef | str) -> tuple[pystac.Item, str]:
        """Parse source_ref and fetch the STAC item.

        Returns (item, asset_key).
        """
        # Parse asset key from :: separator
        asset_key = self._default_asset_key
        ref_part = source_ref
        if "::" in source_ref:
            ref_part, asset_key = source_ref.rsplit("::", 1)

        # Determine if it's a URL or collection/item_id
        if ref_part.startswith("http://") or ref_part.startswith("https://"):
            item = pystac.Item.from_file(ref_part)
        elif "/" in ref_part:
            collection_id, item_id = ref_part.split("/", 1)
            item = self._fetch_item(collection_id, item_id)
        else:
            # Bare item ID — need default collection
            if not self._default_collection:
                raise MaterializeError(
                    source_ref,
                    "Bare item ID requires a default collection on the connector",
                )
            item = self._fetch_item(self._default_collection, ref_part)

        return item, asset_key

    def _fetch_item(self, collection_id: str, item_id: str) -> pystac.Item:
        """Fetch a specific item from the STAC API."""
        client = self._get_client()
        try:
            return client.get_collection(collection_id).get_item(item_id)
        except Exception as e:
            raise MaterializeError(
                f"{collection_id}/{item_id}",
                f"Failed to fetch STAC item: {e}",
            ) from e

    # -----------------------------------------------------------------------
    # Private: asset selection
    # -----------------------------------------------------------------------

    def _select_asset(self, item: pystac.Item, asset_key: str | None) -> pystac.Asset:
        """Select which asset to materialize from a STAC item.

        Strategy:
        1. If asset_key is explicit, use it (error if missing)
        2. If only one asset, use it
        3. Look for common keys: "data", "visual", "B04", etc.
        4. Pick first geotiff asset
        5. Error on ambiguity
        """
        assets = item.assets

        if not assets:
            raise MaterializeError(item.id, "STAC item has no assets")

        if asset_key:
            if asset_key not in assets:
                available = list(assets.keys())
                raise MaterializeError(
                    item.id,
                    f"Asset '{asset_key}' not found. Available: {available}",
                )
            return assets[asset_key]

        # Single asset — no ambiguity
        if len(assets) == 1:
            return next(iter(assets.values()))

        # Try common keys
        for key in ("data", "visual", "image", "default"):
            if key in assets:
                return assets[key]

        # Try first geotiff
        for key, asset in assets.items():
            media_type = asset.media_type or ""
            if media_type in _RASTER_MEDIA_TYPES:
                return asset

        # Ambiguous — error with available keys
        available = list(assets.keys())
        raise MaterializeError(
            item.id,
            f"Multiple assets, cannot auto-select. Available: {available}. "
            f"Specify asset_key explicitly.",
        )

    # -----------------------------------------------------------------------
    # Private: metadata extraction
    # -----------------------------------------------------------------------

    def _extract_spatial(self, item: pystac.Item) -> SpatialDescriptor:
        """Extract spatial descriptor from STAC item metadata."""
        extent = None
        if item.bbox and len(item.bbox) >= 4:
            extent = (item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3])

        # Try to get CRS from proj extension
        crs = None
        proj_epsg = item.properties.get("proj:epsg")
        if proj_epsg:
            crs = f"EPSG:{proj_epsg}"

        # Try to get resolution from proj extension
        resolution = None
        proj_transform = item.properties.get("proj:transform")
        if proj_transform and len(proj_transform) >= 6:
            # Affine transform: [scale_x, shear_x, origin_x, shear_y, scale_y, origin_y]
            resolution = (abs(proj_transform[0]), abs(proj_transform[4]))

        # Band count from eo extension
        band_count = None
        eo_bands = item.properties.get("eo:bands")
        if eo_bands:
            band_count = len(eo_bands)

        return SpatialDescriptor(
            crs=crs,
            extent=extent,
            resolution=resolution,
            band_count=band_count,
        )

    def _extract_metadata(self, item: pystac.Item, asset_key: str | None) -> dict[str, Any]:
        """Extract rich metadata from STAC item for artifact metadata bag."""
        meta: dict[str, Any] = {
            "stac_item_id": item.id,
            "stac_collection": item.collection_id,
            "stac_api_url": self._api_url,
        }

        if asset_key:
            meta["stac_asset_key"] = asset_key

        if item.datetime:
            meta["datetime"] = item.datetime.isoformat()

        # Copy useful properties
        for prop_key in (
            "platform",
            "constellation",
            "instruments",
            "gsd",
            "eo:cloud_cover",
            "sat:orbit_state",
            "view:off_nadir",
        ):
            if prop_key in item.properties:
                meta[prop_key] = item.properties[prop_key]

        return meta

    # -----------------------------------------------------------------------
    # Private: download
    # -----------------------------------------------------------------------

    def _download_asset(
        self,
        asset: pystac.Asset,
        item_id: str,
        asset_key: str | None,
        workspace: Path,
    ) -> Path:
        """Download a STAC asset to workspace."""
        # Determine filename
        url_path = urlparse(asset.href).path
        ext = Path(url_path).suffix or ".tif"
        filename = f"{item_id}"
        if asset_key:
            filename += f"_{asset_key}"
        filename += ext

        download_path = workspace / filename
        download_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            resp = requests.get(asset.href, stream=True, timeout=120)
            resp.raise_for_status()
            with open(download_path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
        except Exception as e:
            if download_path.exists():
                download_path.unlink()
            raise MaterializeError(
                f"{item_id}::{asset_key or 'default'}",
                f"Download failed: {e}",
            ) from e

        return download_path
