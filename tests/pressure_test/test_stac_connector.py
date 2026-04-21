"""STAC connector pressure test.

Tests the connector against mocked STAC responses (unit)
and optionally against a real API (integration, marked).

Stress points we're watching:
1. source_ref feeling underfit (str vs structured)
2. Asset selection ambiguity
3. metadata-only vs lazy-handle vs fetched-local boundaries
4. SpatialDescriptor completeness for remote assets
5. Lineage capturing STAC provenance (catalog URL, collection, item, asset)
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from quarry_connectors.stac import STACConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import Connector, ConnectorCapability, MaterializeError

# ---------------------------------------------------------------------------
# Fixtures: mock STAC objects
# ---------------------------------------------------------------------------


def _make_mock_asset(
    href="https://example.com/data/scene.tif",
    media_type="image/tiff; application=geotiff; profile=cloud-optimized",
    roles=None,
):
    asset = MagicMock()
    asset.href = href
    asset.media_type = media_type
    asset.roles = roles or ["data"]
    return asset


def _make_mock_item(
    item_id="S2A_20230615_T10SGD",
    collection_id="sentinel-2-l2a",
    bbox=(-122.5, 37.5, -122.0, 38.0),
    dt=None,
    assets=None,
    properties=None,
):
    item = MagicMock()
    item.id = item_id
    item.collection_id = collection_id
    item.bbox = list(bbox)
    item.datetime = dt or datetime(2023, 6, 15, tzinfo=timezone.utc)
    item.stac_extensions = ["proj", "eo"]

    if properties is None:
        properties = {
            "proj:epsg": 32610,
            "proj:transform": [10.0, 0.0, 399960.0, 0.0, -10.0, 4200000.0],
            "eo:bands": [{"name": "B04"}, {"name": "B03"}, {"name": "B02"}],
            "eo:cloud_cover": 12.5,
            "platform": "sentinel-2a",
            "constellation": "sentinel-2",
            "gsd": 10.0,
        }
    item.properties = properties

    if assets is None:
        assets = {
            "visual": _make_mock_asset(
                href="https://example.com/data/visual.tif",
                roles=["visual"],
            ),
            "B04": _make_mock_asset(
                href="https://example.com/data/B04.tif",
                roles=["data"],
            ),
            "thumbnail": _make_mock_asset(
                href="https://example.com/data/thumb.png",
                media_type="image/png",
                roles=["thumbnail"],
            ),
        }
    item.assets = assets
    return item


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_satisfies_connector_protocol(self):
        conn = STACConnector(api_url="https://example.com/stac")
        assert isinstance(conn, Connector)

    def test_capabilities(self):
        conn = STACConnector(api_url="https://example.com/stac")
        caps = conn.capabilities
        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps


# ---------------------------------------------------------------------------
# Stress point 1: source_ref parsing
# ---------------------------------------------------------------------------


class TestSourceRefParsing:
    """source_ref as str — watching for underfit."""

    @patch.object(STACConnector, "_fetch_item")
    @patch.object(STACConnector, "_download_asset")
    def test_collection_slash_item(self, mock_download, mock_fetch, tmp_path):
        """'collection/item_id' format."""
        item = _make_mock_item()
        mock_fetch.return_value = item
        mock_download.return_value = tmp_path / "scene.tif"
        (tmp_path / "scene.tif").write_bytes(b"fake raster data")

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/S2A_20230615_T10SGD", tmp_path)

        mock_fetch.assert_called_once_with("sentinel-2-l2a", "S2A_20230615_T10SGD")
        assert result.artifact.name == "S2A_20230615_T10SGD"

    @patch.object(STACConnector, "_fetch_item")
    @patch.object(STACConnector, "_download_asset")
    def test_explicit_asset_key_via_double_colon(self, mock_download, mock_fetch, tmp_path):
        """'collection/item_id::asset_key' format."""
        item = _make_mock_item()
        mock_fetch.return_value = item
        mock_download.return_value = tmp_path / "B04.tif"
        (tmp_path / "B04.tif").write_bytes(b"fake band data")

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/S2A_20230615_T10SGD::B04", tmp_path)

        # Should have selected B04 asset specifically
        assert result.artifact.metadata.get("stac_asset_key") == "B04"

    @patch.object(STACConnector, "_fetch_item")
    @patch.object(STACConnector, "_download_asset")
    def test_bare_item_id_with_default_collection(self, mock_download, mock_fetch, tmp_path):
        """Bare item ID uses connector's default collection."""
        item = _make_mock_item()
        mock_fetch.return_value = item
        mock_download.return_value = tmp_path / "scene.tif"
        (tmp_path / "scene.tif").write_bytes(b"fake")

        conn = STACConnector(
            api_url="https://example.com/stac",
            collection="sentinel-2-l2a",
        )
        conn.materialize("S2A_20230615_T10SGD", tmp_path)
        mock_fetch.assert_called_once_with("sentinel-2-l2a", "S2A_20230615_T10SGD")

    def test_bare_item_id_without_default_collection_errors(self, tmp_path):
        """Bare item ID without default collection should fail clearly."""
        conn = STACConnector(api_url="https://example.com/stac")
        with pytest.raises(MaterializeError, match="default collection"):
            conn.materialize("S2A_20230615_T10SGD", tmp_path)


# ---------------------------------------------------------------------------
# Stress point 2: asset selection
# ---------------------------------------------------------------------------


class TestAssetSelection:
    @patch.object(STACConnector, "_fetch_item")
    @patch.object(STACConnector, "_download_asset")
    def test_explicit_key_selects_correct_asset(self, mock_download, mock_fetch, tmp_path):
        item = _make_mock_item()
        mock_fetch.return_value = item
        mock_download.return_value = tmp_path / "B04.tif"
        (tmp_path / "B04.tif").write_bytes(b"band4")

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/item::B04", tmp_path)
        assert result.artifact.metadata["stac_asset_key"] == "B04"

    @patch.object(STACConnector, "_fetch_item")
    def test_missing_asset_key_errors_clearly(self, mock_fetch, tmp_path):
        item = _make_mock_item()
        mock_fetch.return_value = item

        conn = STACConnector(api_url="https://example.com/stac")
        with pytest.raises(MaterializeError, match="not found.*Available"):
            conn.materialize("sentinel-2-l2a/item::nonexistent", tmp_path)

    @patch.object(STACConnector, "_fetch_item")
    @patch.object(STACConnector, "_download_asset")
    def test_single_asset_auto_selects(self, mock_download, mock_fetch, tmp_path):
        """Single asset item needs no explicit key."""
        single_asset = {"data": _make_mock_asset()}
        item = _make_mock_item(assets=single_asset)
        mock_fetch.return_value = item
        mock_download.return_value = tmp_path / "data.tif"
        (tmp_path / "data.tif").write_bytes(b"data")

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/item", tmp_path)
        assert result.artifact.type == ArtifactType.RASTER

    @patch.object(STACConnector, "_fetch_item")
    @patch.object(STACConnector, "_download_asset")
    def test_visual_key_preferred_over_ambiguity(self, mock_download, mock_fetch, tmp_path):
        """'visual' is a preferred key when multiple assets exist."""
        item = _make_mock_item()  # has visual, B04, thumbnail
        mock_fetch.return_value = item
        mock_download.return_value = tmp_path / "visual.tif"
        (tmp_path / "visual.tif").write_bytes(b"rgb")

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/item", tmp_path)
        # Should pick 'visual' from the available keys
        assert result.artifact is not None

    @patch.object(STACConnector, "_fetch_item")
    def test_no_assets_errors(self, mock_fetch, tmp_path):
        item = _make_mock_item(assets={})
        mock_fetch.return_value = item

        conn = STACConnector(api_url="https://example.com/stac")
        with pytest.raises(MaterializeError, match="no assets"):
            conn.materialize("sentinel-2-l2a/item", tmp_path)

    @patch.object(STACConnector, "_fetch_item")
    @patch.object(STACConnector, "_download_asset")
    def test_connector_default_asset_key(self, mock_download, mock_fetch, tmp_path):
        """Connector-level default asset key."""
        item = _make_mock_item()
        mock_fetch.return_value = item
        mock_download.return_value = tmp_path / "B04.tif"
        (tmp_path / "B04.tif").write_bytes(b"band")

        conn = STACConnector(
            api_url="https://example.com/stac",
            asset_key="B04",
        )
        result = conn.materialize("sentinel-2-l2a/item", tmp_path)
        assert result.artifact.metadata["stac_asset_key"] == "B04"


# ---------------------------------------------------------------------------
# Stress point 3: lazy vs eager materialization
# ---------------------------------------------------------------------------


class TestMaterializationModes:
    @patch.object(STACConnector, "_fetch_item")
    def test_lazy_creates_lazy_handle(self, mock_fetch, tmp_path):
        item = _make_mock_item()
        mock_fetch.return_value = item

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/item::visual", tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"
        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE
        assert result.artifact.backing.uri == "https://example.com/data/visual.tif"
        assert not result.artifact.is_materialized

    @patch.object(STACConnector, "_fetch_item")
    def test_lazy_still_has_full_metadata(self, mock_fetch, tmp_path):
        """Lazy materialization extracts all STAC metadata even without download."""
        item = _make_mock_item()
        mock_fetch.return_value = item

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/item::visual", tmp_path, lazy=True)
        art = result.artifact

        # Spatial should be populated from STAC metadata
        assert art.spatial.crs == "EPSG:32610"
        assert art.spatial.extent is not None
        assert art.spatial.resolution is not None
        assert art.spatial.band_count == 3

    @patch.object(STACConnector, "_fetch_item")
    @patch.object(STACConnector, "_download_asset")
    def test_eager_creates_local_file(self, mock_download, mock_fetch, tmp_path):
        item = _make_mock_item()
        mock_fetch.return_value = item
        dl_path = tmp_path / "scene.tif"
        dl_path.write_bytes(b"raster content here")
        mock_download.return_value = dl_path

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/item::visual", tmp_path)

        assert result.strategy == "fetched_remote"
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.is_materialized
        assert result.artifact.backing.size_bytes > 0
        assert result.artifact.backing.content_hash is not None


# ---------------------------------------------------------------------------
# Stress point 4: SpatialDescriptor completeness
# ---------------------------------------------------------------------------


class TestSpatialDescriptorFromSTAC:
    @patch.object(STACConnector, "_fetch_item")
    def test_full_spatial_from_proj_extension(self, mock_fetch, tmp_path):
        item = _make_mock_item()
        mock_fetch.return_value = item

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/item::visual", tmp_path, lazy=True)
        spatial = result.artifact.spatial

        assert spatial.crs == "EPSG:32610"
        assert spatial.extent == (-122.5, 37.5, -122.0, 38.0)
        assert spatial.resolution == (10.0, 10.0)
        assert spatial.band_count == 3

    @patch.object(STACConnector, "_fetch_item")
    def test_minimal_spatial_without_extensions(self, mock_fetch, tmp_path):
        """Item without proj/eo extensions still gets bbox-based spatial."""
        item = _make_mock_item(
            properties={},
            assets={"data": _make_mock_asset()},
        )
        mock_fetch.return_value = item

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/item", tmp_path, lazy=True)
        spatial = result.artifact.spatial

        # Extent from bbox should still work
        assert spatial.extent == (-122.5, 37.5, -122.0, 38.0)
        # These should be None without extensions
        assert spatial.crs is None
        assert spatial.resolution is None
        assert spatial.band_count is None


# ---------------------------------------------------------------------------
# Stress point 5: lineage captures STAC provenance
# ---------------------------------------------------------------------------


class TestLineageProvenance:
    @patch.object(STACConnector, "_fetch_item")
    def test_lineage_captures_stac_context(self, mock_fetch, tmp_path):
        item = _make_mock_item()
        mock_fetch.return_value = item

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/item::B04", tmp_path, lazy=True)
        lineage = result.artifact.lineage

        assert lineage is not None
        assert lineage.operation == "materialize"
        assert lineage.params["source"] == "stac"
        assert lineage.params["api_url"] == "https://example.com/stac"
        assert lineage.params["collection"] == "sentinel-2-l2a"
        assert lineage.params["item_id"] == "S2A_20230615_T10SGD"
        assert lineage.params["asset_key"] == "B04"

    @patch.object(STACConnector, "_fetch_item")
    def test_metadata_captures_platform_info(self, mock_fetch, tmp_path):
        item = _make_mock_item()
        mock_fetch.return_value = item

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/item::visual", tmp_path, lazy=True)
        meta = result.artifact.metadata

        assert meta["stac_item_id"] == "S2A_20230615_T10SGD"
        assert meta["stac_collection"] == "sentinel-2-l2a"
        assert meta["platform"] == "sentinel-2a"
        assert meta["eo:cloud_cover"] == 12.5
        assert meta["gsd"] == 10.0


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestDiscover:
    @patch.object(STACConnector, "_get_client")
    def test_discover_returns_catalog_entries(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        items = [_make_mock_item(item_id=f"item_{i}") for i in range(3)]
        mock_search = MagicMock()
        mock_search.items.return_value = iter(items)
        mock_client.search.return_value = mock_search

        conn = STACConnector(api_url="https://example.com/stac")
        entries = conn.discover({"collections": ["sentinel-2-l2a"], "max_items": 3})

        assert len(entries) == 3
        assert entries[0].source_ref == "sentinel-2-l2a/item_0"
        assert "extent" in entries[0].spatial_hint
        assert "asset_keys" in entries[0].metadata

    @patch.object(STACConnector, "_get_client")
    def test_discover_uses_default_collection(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_search = MagicMock()
        mock_search.items.return_value = iter([])
        mock_client.search.return_value = mock_search

        conn = STACConnector(
            api_url="https://example.com/stac",
            collection="sentinel-2-l2a",
        )
        conn.discover()

        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args
        assert call_kwargs.kwargs.get("collections") == ["sentinel-2-l2a"]


# ---------------------------------------------------------------------------
# Registry round-trip (does STAC artifact persist cleanly?)
# ---------------------------------------------------------------------------


class TestSTACRegistryRoundTrip:
    @patch.object(STACConnector, "_fetch_item")
    def test_lazy_stac_artifact_persists(self, mock_fetch, tmp_path):
        """STAC lazy artifact survives registry round-trip."""
        from quarry_registry.registry import Registry

        item = _make_mock_item()
        mock_fetch.return_value = item

        conn = STACConnector(api_url="https://example.com/stac")
        result = conn.materialize("sentinel-2-l2a/item::visual", tmp_path, lazy=True)

        registry = Registry(tmp_path)
        registry.save_artifact(result.artifact)
        recovered = registry.get_artifact(result.artifact.id)

        assert recovered is not None
        assert recovered.backing.kind == BackingStoreKind.LAZY_HANDLE
        assert recovered.backing.uri == "https://example.com/data/visual.tif"
        assert recovered.spatial.crs == "EPSG:32610"
        assert recovered.spatial.extent == (-122.5, 37.5, -122.0, 38.0)
        assert recovered.metadata["stac_item_id"] == "S2A_20230615_T10SGD"
        assert recovered.metadata["platform"] == "sentinel-2a"
