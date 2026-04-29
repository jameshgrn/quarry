"""Sentinel2Connector — structural mapper for Sentinel-2 L2A via STAC.

Lane: connector

Maps Sentinel-2 band structure to individual artifacts via STACConnector.
Each band (B02, B03, B04, etc.) is materialized as a separate artifact
with semantic labels (wavelength, resolution, common name).

Default catalog: Element84 Earth Search (sentinel-2-l2a collection).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from quarry_core.artifact import Artifact
from quarry_core.connector import (
    CatalogEntry,
    ConnectorCapability,
    MaterializeError,
    MaterializeResult,
)

from quarry_connectors.stac import STACConnector

if TYPE_CHECKING:
    from quarry_core.source_ref import SourceRef

# Default STAC endpoint
_DEFAULT_API = "https://earth-search.aws.element84.com/v1"
_DEFAULT_COLLECTION = "sentinel-2-l2a"

# Sentinel-2 band catalog: asset_key -> (band_id, common_name, wavelength_nm, gsd_m)
_S2_BANDS: dict[str, tuple[str, str, int, int]] = {
    "coastal": ("B01", "coastal", 443, 60),
    "blue": ("B02", "blue", 490, 10),
    "green": ("B03", "green", 560, 10),
    "red": ("B04", "red", 665, 10),
    "rededge1": ("B05", "rededge", 705, 20),
    "rededge2": ("B06", "rededge", 740, 20),
    "rededge3": ("B07", "rededge", 783, 20),
    "nir": ("B08", "nir", 842, 10),
    "nir08": ("B8A", "nir08", 865, 20),
    "nir09": ("B09", "nir09", 945, 60),
    "cirrus": ("B10", "cirrus", 1375, 60),
    "swir16": ("B11", "swir16", 1610, 20),
    "swir22": ("B12", "swir22", 2190, 20),
    "scl": ("SCL", "scl", 0, 20),
}

# Reverse lookup: band_id -> asset_key
_BAND_TO_ASSET: dict[str, str] = {v[0]: k for k, v in _S2_BANDS.items()}


class Sentinel2Connector:
    """Structural mapper for Sentinel-2 L2A data via STAC.

    Composes STACConnector for catalog access. Adds:
    - Sentinel-2 band structure (B01-B12, B8A, SCL)
    - Semantic discovery with band metadata (wavelength, resolution, name)
    - Cloud cover filtering
    - One artifact per band (canonical)
    """

    def __init__(
        self,
        api_url: str = _DEFAULT_API,
        collection: str = _DEFAULT_COLLECTION,
    ) -> None:
        self._stac = STACConnector(api_url=api_url, collection=collection)
        self._collection = collection

    @property
    def name(self) -> str:
        return "sentinel2"

    @property
    def capabilities(self) -> ConnectorCapability:
        return (
            ConnectorCapability.MATERIALIZE
            | ConnectorCapability.DISCOVER
            | ConnectorCapability.METADATA_ONLY
            | ConnectorCapability.MATERIALIZE_LAZY
        )

    def materialize(
        self,
        source_ref: SourceRef | str,
        workspace: Path,
        *,
        lazy: bool = False,
    ) -> MaterializeResult:
        """Materialize a single Sentinel-2 band as an artifact.

        source_ref formats:
            "item_id::blue"          — by STAC asset key
            "item_id::B04"           — by Sentinel-2 band ID
            "collection/item_id::red" — explicit collection
        """
        stac_ref, asset_key = self._parse_source_ref(source_ref)

        # Resolve band ID to STAC asset key
        resolved_key = self._resolve_band(asset_key) if asset_key else None
        if resolved_key is None and asset_key is not None:
            raise MaterializeError(
                source_ref,
                f"Unknown band '{asset_key}'. "
                f"Available: {list(_S2_BANDS.keys())} or {list(_BAND_TO_ASSET.keys())}",
            )

        # Build full STAC source_ref
        full_ref = f"{stac_ref}::{resolved_key}" if resolved_key else stac_ref

        result = self._stac.materialize(full_ref, workspace, lazy=lazy)

        # Enrich with Sentinel-2 band metadata
        band_info = _S2_BANDS.get(resolved_key or "", None)
        enriched_metadata = {
            **dict(result.artifact.metadata),
            "source": "sentinel2",
        }
        if band_info:
            band_id, common_name, wavelength, gsd = band_info
            enriched_metadata.update(
                {
                    "band_id": band_id,
                    "common_name": common_name,
                    "wavelength_nm": wavelength,
                    "gsd_m": gsd,
                }
            )

        a = result.artifact
        enriched_artifact = Artifact(
            id=a.id,
            type=a.type,
            name=a.name,
            backing=a.backing,
            spatial=a.spatial,
            lineage=a.lineage,
            checks=a.checks,
            metadata=enriched_metadata,
            created_at=a.created_at,
        )

        return MaterializeResult(
            artifact=enriched_artifact,
            strategy=result.strategy,
            source_ref=source_ref,
            notes=f"Sentinel-2 {band_info[0] if band_info else asset_key}",
        )

    def discover(self, query: str | dict[str, Any] | None = None) -> list[CatalogEntry]:
        """Search for Sentinel-2 scenes and list available bands.

        query as dict supports:
            bbox: [xmin, ymin, xmax, ymax]
            datetime: "2024-01-01/2024-01-31"
            max_items: int (default 10)
            max_cloud: float (0-100, default 20)
            bands_only: bool (if True, return one entry per band per scene)

        query as str: treated as datetime range.
        """
        if query is None:
            query = {}
        if isinstance(query, str):
            query = {"datetime": query}

        max_cloud = query.pop("max_cloud", 20)
        bands_only = query.pop("bands_only", False)

        # Set collection
        query.setdefault("collections", [self._collection])
        query.setdefault("max_items", 10)

        # Get scenes from STAC
        stac_entries = self._stac.discover(query)

        if not bands_only:
            # Enrich scene entries with S2 metadata
            entries = []
            for entry in stac_entries:
                cloud_cover = entry.metadata.get("properties", {}).get("eo:cloud_cover")
                if cloud_cover is not None and cloud_cover > max_cloud:
                    continue

                # Identify which S2 bands are available
                asset_keys = entry.metadata.get("asset_keys", [])
                available_bands = [k for k in asset_keys if k in _S2_BANDS]

                entries.append(
                    CatalogEntry(
                        source_ref=entry.source_ref,
                        name=entry.name,
                        description=entry.description,
                        spatial_hint=entry.spatial_hint,
                        metadata={
                            "source": "sentinel2",
                            "cloud_cover": cloud_cover,
                            "available_bands": available_bands,
                            "platform": entry.metadata.get("properties", {}).get("platform"),
                            "datetime": entry.metadata.get("properties", {}).get("datetime"),
                        },
                    )
                )
            return entries

        # bands_only: return one entry per band per scene
        entries = []
        for entry in stac_entries:
            cloud_cover = entry.metadata.get("properties", {}).get("eo:cloud_cover")
            if cloud_cover is not None and cloud_cover > max_cloud:
                continue

            asset_keys = entry.metadata.get("asset_keys", [])
            for asset_key in asset_keys:
                if asset_key not in _S2_BANDS:
                    continue
                band_id, common_name, wavelength, gsd = _S2_BANDS[asset_key]
                entries.append(
                    CatalogEntry(
                        source_ref=f"{entry.source_ref}::{asset_key}",
                        name=f"{entry.name}_{band_id}",
                        spatial_hint=entry.spatial_hint,
                        metadata={
                            "source": "sentinel2",
                            "band_id": band_id,
                            "common_name": common_name,
                            "wavelength_nm": wavelength,
                            "gsd_m": gsd,
                            "cloud_cover": cloud_cover,
                        },
                    )
                )
        return entries

    def metadata(self, source_ref: SourceRef | str) -> dict[str, Any]:
        """Get Sentinel-2 scene + band metadata without materializing."""
        stac_ref, asset_key = self._parse_source_ref(source_ref)
        resolved_key = self._resolve_band(asset_key) if asset_key else None

        full_ref = f"{stac_ref}::{resolved_key}" if resolved_key else stac_ref
        meta = self._stac.metadata(full_ref)

        meta["source"] = "sentinel2"
        if resolved_key and resolved_key in _S2_BANDS:
            band_id, common_name, wavelength, gsd = _S2_BANDS[resolved_key]
            meta.update(
                {
                    "band_id": band_id,
                    "common_name": common_name,
                    "wavelength_nm": wavelength,
                    "gsd_m": gsd,
                }
            )

        # List all available S2 bands
        all_keys = meta.get("all_asset_keys", [])
        meta["available_bands"] = {k: _S2_BANDS[k][0] for k in all_keys if k in _S2_BANDS}

        return meta

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _parse_source_ref(self, source_ref: SourceRef | str) -> tuple[str, str | None]:
        """Parse source_ref into (stac_ref, band_key | None)."""
        raw = str(source_ref).strip()

        if "::" in raw:
            stac_part, band_part = raw.rsplit("::", 1)
            return stac_part.strip(), band_part.strip()

        return raw, None

    def _resolve_band(self, band_key: str) -> str | None:
        """Resolve a band key (asset key or band ID) to STAC asset key."""
        # Already an asset key
        if band_key in _S2_BANDS:
            return band_key

        # Try band ID (e.g., "B04" -> "red")
        upper = band_key.upper()
        if upper in _BAND_TO_ASSET:
            return _BAND_TO_ASSET[upper]

        return None
