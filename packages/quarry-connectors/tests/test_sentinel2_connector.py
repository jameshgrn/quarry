"""Tests for Sentinel2Connector (structural mapper over STAC)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    Lineage,
    SpatialDescriptor,
)
from quarry_core.connector import MaterializeResult
from rasterio.transform import from_origin

from quarry_connectors.sentinel2 import (
    _BAND_TO_ASSET,
    _S2_BANDS,
    Sentinel2Connector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_geotiff(path: Path, height: int = 10, width: int = 10) -> Path:
    """Write a minimal GeoTIFF and return its path."""
    transform = from_origin(0, height, 10, 10)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32610",
        transform=transform,
    ) as dst:
        dst.write(np.ones((height, width), dtype=np.float32), 1)
    return path


def _mock_stac_materialize_result(tmp_path: Path, asset_key: str = "red") -> MaterializeResult:
    """Create a mock MaterializeResult as STACConnector would return."""
    tif_path = _make_geotiff(tmp_path / f"scene_{asset_key}.tif")
    return MaterializeResult(
        artifact=Artifact(
            type=ArtifactType.RASTER,
            name="S2A_T10SEG_20240101",
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(tif_path),
                size_bytes=tif_path.stat().st_size,
            ),
            spatial=SpatialDescriptor(
                crs="EPSG:32610",
                extent=(0, 0, 100, 100),
                resolution=(10.0, 10.0),
                band_count=1,
            ),
            lineage=Lineage(
                operation="materialize",
                params={"source": "stac", "asset_key": asset_key},
            ),
            metadata={
                "stac_item_id": "S2A_T10SEG_20240101",
                "stac_collection": "sentinel-2-l2a",
                "platform": "sentinel-2a",
                "eo:cloud_cover": 5.2,
            },
        ),
        strategy="fetched_remote",
        source_ref=f"sentinel-2-l2a/S2A_T10SEG_20240101::{asset_key}",
    )


def _mock_stac_lazy_result(asset_key: str = "red") -> MaterializeResult:
    """Create a mock lazy MaterializeResult."""
    return MaterializeResult(
        artifact=Artifact(
            type=ArtifactType.RASTER,
            name="S2A_T10SEG_20240101",
            backing=BackingStore(
                kind=BackingStoreKind.LAZY_HANDLE,
                uri="https://example.com/scene_red.tif",
            ),
            spatial=SpatialDescriptor(
                crs="EPSG:32610",
                extent=(0, 0, 100, 100),
            ),
            lineage=Lineage(
                operation="materialize",
                params={"source": "stac", "asset_key": asset_key, "lazy": True},
            ),
            metadata={
                "stac_item_id": "S2A_T10SEG_20240101",
                "stac_collection": "sentinel-2-l2a",
            },
        ),
        strategy="lazy_handle",
        source_ref=f"sentinel-2-l2a/S2A_T10SEG_20240101::{asset_key}",
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_name():
    assert Sentinel2Connector().name == "sentinel2"


def test_capabilities():
    from quarry_core.connector import ConnectorCapability

    caps = Sentinel2Connector().capabilities
    assert ConnectorCapability.MATERIALIZE in caps
    assert ConnectorCapability.DISCOVER in caps
    assert ConnectorCapability.MATERIALIZE_LAZY in caps
    assert ConnectorCapability.METADATA_ONLY in caps


def test_satisfies_connector_protocol():
    from quarry_core.connector import Connector

    assert isinstance(Sentinel2Connector(), Connector)


# ---------------------------------------------------------------------------
# Band catalog
# ---------------------------------------------------------------------------


def test_band_catalog_completeness():
    """All 13 Sentinel-2 bands should be in the catalog."""
    expected_bands = {
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
        "SCL",
    }
    actual_bands = {v[0] for v in _S2_BANDS.values()}
    assert expected_bands == actual_bands


def test_band_to_asset_reverse_lookup():
    assert _BAND_TO_ASSET["B04"] == "red"
    assert _BAND_TO_ASSET["B08"] == "nir"
    assert _BAND_TO_ASSET["SCL"] == "scl"


def test_resolve_band_by_asset_key():
    conn = Sentinel2Connector()
    assert conn._resolve_band("red") == "red"
    assert conn._resolve_band("blue") == "blue"
    assert conn._resolve_band("scl") == "scl"


def test_resolve_band_by_band_id():
    conn = Sentinel2Connector()
    assert conn._resolve_band("B04") == "red"
    assert conn._resolve_band("B8A") == "nir08"
    assert conn._resolve_band("b02") == "blue"  # case insensitive


def test_resolve_band_unknown():
    conn = Sentinel2Connector()
    assert conn._resolve_band("bogus") is None


# ---------------------------------------------------------------------------
# Source ref parsing
# ---------------------------------------------------------------------------


def test_parse_with_band():
    conn = Sentinel2Connector()
    ref, band = conn._parse_source_ref("sentinel-2-l2a/item123::red")
    assert ref == "sentinel-2-l2a/item123"
    assert band == "red"


def test_parse_with_band_id():
    conn = Sentinel2Connector()
    ref, band = conn._parse_source_ref("item123::B04")
    assert ref == "item123"
    assert band == "B04"


def test_parse_no_band():
    conn = Sentinel2Connector()
    ref, band = conn._parse_source_ref("sentinel-2-l2a/item123")
    assert ref == "sentinel-2-l2a/item123"
    assert band is None


# ---------------------------------------------------------------------------
# Materialize (mocked STAC)
# ---------------------------------------------------------------------------


def test_materialize_eager_enriches_metadata(tmp_path: Path):
    conn = Sentinel2Connector()
    mock_result = _mock_stac_materialize_result(tmp_path, "red")

    with patch.object(conn._stac, "materialize", return_value=mock_result):
        result = conn.materialize("item123::red", tmp_path)

    assert result.artifact.metadata["source"] == "sentinel2"
    assert result.artifact.metadata["band_id"] == "B04"
    assert result.artifact.metadata["common_name"] == "red"
    assert result.artifact.metadata["wavelength_nm"] == 665
    assert result.artifact.metadata["gsd_m"] == 10


def test_materialize_eager_by_band_id(tmp_path: Path):
    conn = Sentinel2Connector()
    mock_result = _mock_stac_materialize_result(tmp_path, "nir")

    with patch.object(conn._stac, "materialize", return_value=mock_result):
        result = conn.materialize("item123::B08", tmp_path)

    assert result.artifact.metadata["band_id"] == "B08"
    assert result.artifact.metadata["wavelength_nm"] == 842


def test_materialize_lazy(tmp_path: Path):
    conn = Sentinel2Connector()
    mock_result = _mock_stac_lazy_result("blue")

    with patch.object(conn._stac, "materialize", return_value=mock_result):
        result = conn.materialize("item123::blue", tmp_path, lazy=True)

    assert result.strategy == "lazy_handle"
    assert result.artifact.metadata["source"] == "sentinel2"
    assert result.artifact.metadata["band_id"] == "B02"


def test_materialize_unknown_band(tmp_path: Path):
    from quarry_core.connector import MaterializeError

    conn = Sentinel2Connector()
    with pytest.raises(MaterializeError, match="Unknown band"):
        conn.materialize("item123::bogus", tmp_path)


# ---------------------------------------------------------------------------
# Discover (mocked STAC)
# ---------------------------------------------------------------------------


def _mock_stac_discover():
    """Create mock CatalogEntry list as STACConnector.discover would return."""
    from quarry_core.connector import CatalogEntry

    return [
        CatalogEntry(
            source_ref="sentinel-2-l2a/S2A_T10SEG_20240101",
            name="S2A_T10SEG_20240101",
            spatial_hint={
                "extent": (-122.5, 37.0, -122.0, 37.5),
                "crs": "EPSG:4326",
            },
            metadata={
                "collection": "sentinel-2-l2a",
                "asset_keys": [
                    "blue",
                    "green",
                    "red",
                    "nir",
                    "scl",
                    "rededge1",
                    "thumbnail",
                    "visual",
                ],
                "properties": {
                    "eo:cloud_cover": 5.2,
                    "platform": "sentinel-2a",
                    "datetime": "2024-01-01T10:30:00Z",
                },
            },
        ),
        CatalogEntry(
            source_ref="sentinel-2-l2a/S2A_T10SEG_20240102",
            name="S2A_T10SEG_20240102",
            spatial_hint={
                "extent": (-122.5, 37.0, -122.0, 37.5),
                "crs": "EPSG:4326",
            },
            metadata={
                "collection": "sentinel-2-l2a",
                "asset_keys": ["blue", "green", "red", "nir"],
                "properties": {
                    "eo:cloud_cover": 80.0,
                    "platform": "sentinel-2a",
                    "datetime": "2024-01-02T10:30:00Z",
                },
            },
        ),
    ]


def test_discover_scenes(tmp_path: Path):
    conn = Sentinel2Connector()

    with patch.object(conn._stac, "discover", return_value=_mock_stac_discover()):
        entries = conn.discover(
            {
                "bbox": [-122.5, 37.0, -122.0, 37.5],
                "datetime": "2024-01-01/2024-01-31",
            }
        )

    # Only scene 1 should pass cloud filter (default max_cloud=20)
    assert len(entries) == 1
    assert entries[0].metadata["cloud_cover"] == 5.2
    assert "blue" in entries[0].metadata["available_bands"]
    assert "thumbnail" not in entries[0].metadata["available_bands"]


def test_discover_with_high_cloud_threshold():
    conn = Sentinel2Connector()

    with patch.object(conn._stac, "discover", return_value=_mock_stac_discover()):
        entries = conn.discover({"max_cloud": 100})

    assert len(entries) == 2


def test_discover_bands_only():
    conn = Sentinel2Connector()

    with patch.object(conn._stac, "discover", return_value=_mock_stac_discover()):
        entries = conn.discover({"bands_only": True})

    # Scene 1 has 5 S2 bands (blue, green, red, nir, scl, rededge1)
    # Scene 2 filtered by cloud cover
    s2_bands = [e for e in entries if "band_id" in e.metadata]
    assert len(s2_bands) == 6  # 6 S2 bands from scene 1
    assert any(e.metadata["band_id"] == "B04" for e in s2_bands)


def test_discover_string_query():
    conn = Sentinel2Connector()

    with patch.object(conn._stac, "discover", return_value=[]) as mock:
        conn.discover("2024-01-01/2024-01-31")

    call_args = mock.call_args[0][0]
    assert call_args["datetime"] == "2024-01-01/2024-01-31"
