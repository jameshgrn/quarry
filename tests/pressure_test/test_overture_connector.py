"""
Pressure test: OvertureConnector.

Lane: connector

Validates Overture Maps materialization:
- source_ref parsing (overture://theme/type, SourceRef with params)
- lazy mode: LAZY_HANDLE with S3 URL, no DuckDB/HTTP calls
- eager mode: DuckDB query → GeoPackage dump (tested with local mock data)
- discover: list all known themes/types
- metadata: returns theme/type/release info
- error handling: unknown theme/type, no features
"""

from __future__ import annotations

import duckdb
import pytest
from quarry_connectors.overture import (
    KNOWN_THEMES,
    OvertureConnector,
)
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_overture_db(tmp_path):
    """Create a local Parquet file mimicking Overture schema.

    Exports DuckDB table data to Parquet so the real _fetch_and_dump
    can read it via read_parquet() in an in-memory DuckDB connection.
    """
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
    except duckdb.Error:
        conn.close()
        pytest.skip("DuckDB spatial extension not available")

    conn.execute("""
        CREATE TABLE mock_buildings (
            id VARCHAR,
            names VARCHAR,
            geometry GEOMETRY,
            bbox STRUCT(xmin DOUBLE, ymin DOUBLE, xmax DOUBLE, ymax DOUBLE)
        )
    """)
    conn.execute("""
        INSERT INTO mock_buildings VALUES
        ('b1', 'Building A', ST_Point(1.0, 2.0),
         {xmin: 0.9, ymin: 1.9, xmax: 1.1, ymax: 2.1}),
        ('b2', 'Building B', ST_Point(3.0, 4.0),
         {xmin: 2.9, ymin: 3.9, xmax: 3.1, ymax: 4.1}),
        ('b3', 'Building C', ST_Point(5.0, 6.0),
         {xmin: 4.9, ymin: 5.9, xmax: 5.1, ymax: 6.1})
    """)
    parquet_path = str(tmp_path / "mock_buildings.parquet")
    # Export geometry as WKB BLOB to match real Overture Parquet layout
    conn.execute(
        f"COPY (SELECT id, names, ST_AsWKB(geometry) AS geometry, bbox "
        f"FROM mock_buildings) TO '{parquet_path}' (FORMAT PARQUET)"
    )
    conn.close()
    return parquet_path


class _LocalOvertureConnector(OvertureConnector):
    """Test subclass that reads from a local Parquet file instead of S3.

    Only overrides _parquet_source (for data injection) and _setup_extensions
    (to skip httpfs). The real _fetch_and_dump runs against the local data,
    so the test exercises actual production logic.
    """

    def __init__(self, parquet_path: str, **kwargs):
        super().__init__(**kwargs)
        self._local_parquet_path = parquet_path

    def _parquet_source(self, theme: str, typ: str) -> str:
        """Override to read from local Parquet file."""
        return f"read_parquet('{self._local_parquet_path}')"

    @staticmethod
    def _setup_extensions(conn: duckdb.DuckDBPyConnection) -> None:
        """Only load spatial — skip httpfs since we're reading local data."""
        try:
            conn.execute("INSTALL spatial")
            conn.execute("LOAD spatial")
        except duckdb.Error:
            pass


# ---------------------------------------------------------------------------
# Source ref parsing
# ---------------------------------------------------------------------------


class TestOvertureSourceRefParsing:
    """Validate source_ref parsing for overture:// format."""

    def test_overture_url_format(self):
        connector = OvertureConnector()
        theme, typ, bbox = connector._parse_source_ref("overture://buildings/building")
        assert theme == "buildings"
        assert typ == "building"
        assert bbox is None

    def test_sourceref_with_params(self):
        connector = OvertureConnector()
        ref = SourceRef(
            raw="overture://buildings/building",
            params={"theme": "buildings", "type": "building", "bbox": (1, 2, 3, 4)},
        )
        theme, typ, bbox = connector._parse_source_ref(ref)
        assert theme == "buildings"
        assert typ == "building"
        assert bbox == (1.0, 2.0, 3.0, 4.0)

    def test_sourceref_params_only(self):
        connector = OvertureConnector()
        ref = SourceRef(
            raw="overture://places/place",
            params={"theme": "places", "type": "place"},
        )
        theme, typ, bbox = connector._parse_source_ref(ref)
        assert theme == "places"
        assert typ == "place"

    def test_transportation_segment(self):
        connector = OvertureConnector()
        theme, typ, bbox = connector._parse_source_ref("overture://transportation/segment")
        assert theme == "transportation"
        assert typ == "segment"

    def test_invalid_format_raises(self):
        connector = OvertureConnector()
        with pytest.raises(MaterializeError):
            connector._parse_source_ref("not_valid")

    def test_missing_type_raises(self):
        connector = OvertureConnector()
        with pytest.raises(MaterializeError):
            connector._parse_source_ref("overture://buildings")


# ---------------------------------------------------------------------------
# Lazy materialization
# ---------------------------------------------------------------------------


class TestOvertureLazyMaterialization:
    """Validate lazy (metadata-only) materialization."""

    def test_lazy_produces_lazy_handle(self, tmp_path):
        connector = OvertureConnector()
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=True)
        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE
        assert result.strategy == "lazy_handle"

    def test_lazy_artifact_type_is_vector(self, tmp_path):
        connector = OvertureConnector()
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=True)
        assert result.artifact.type == ArtifactType.VECTOR

    def test_lazy_crs_is_4326(self, tmp_path):
        connector = OvertureConnector()
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=True)
        assert result.artifact.spatial.crs == "EPSG:4326"

    def test_lazy_backing_uri_contains_s3(self, tmp_path):
        connector = OvertureConnector()
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=True)
        uri = result.artifact.backing.uri
        assert "s3://" in uri
        assert "buildings" in uri
        assert "building" in uri

    def test_lazy_lineage_params(self, tmp_path):
        connector = OvertureConnector()
        ref = SourceRef(
            raw="overture://buildings/building",
            params={"theme": "buildings", "type": "building", "bbox": (1, 2, 3, 4)},
        )
        result = connector.materialize(ref, tmp_path, lazy=True)

        params = result.artifact.lineage.params
        assert params["source"] == "overture"
        assert params["theme"] == "buildings"
        assert params["type"] == "building"
        assert params["lazy"] is True
        assert params["bbox"] == (1.0, 2.0, 3.0, 4.0)

    def test_lazy_with_bbox_sets_extent(self, tmp_path):
        connector = OvertureConnector()
        ref = SourceRef(
            raw="overture://buildings/building",
            params={"theme": "buildings", "type": "building", "bbox": (-120, 35, -119, 36)},
        )
        result = connector.materialize(ref, tmp_path, lazy=True)
        assert result.artifact.spatial.extent == (-120.0, 35.0, -119.0, 36.0)

    def test_lazy_without_bbox_has_no_extent(self, tmp_path):
        connector = OvertureConnector()
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=True)
        assert result.artifact.spatial.extent is None

    def test_lazy_metadata_has_theme_type(self, tmp_path):
        connector = OvertureConnector()
        result = connector.materialize("overture://places/place", tmp_path, lazy=True)
        assert result.artifact.metadata["theme"] == "places"
        assert result.artifact.metadata["type"] == "place"


# ---------------------------------------------------------------------------
# Eager materialization (local mock data)
# ---------------------------------------------------------------------------


class TestOvertureEagerMaterialization:
    """Validate eager materialization using local DuckDB mock data."""

    def test_eager_produces_vector_artifact(self, mock_overture_db, tmp_path):
        connector = _LocalOvertureConnector(mock_overture_db)
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=False)
        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE

    def test_eager_strategy_fetched_remote(self, mock_overture_db, tmp_path):
        connector = _LocalOvertureConnector(mock_overture_db)
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=False)
        assert result.strategy == "fetched_remote"

    def test_eager_produces_gpkg(self, mock_overture_db, tmp_path):
        connector = _LocalOvertureConnector(mock_overture_db)
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=False)
        assert result.artifact.backing.uri.endswith(".gpkg")

    def test_eager_feature_count(self, mock_overture_db, tmp_path):
        connector = _LocalOvertureConnector(mock_overture_db)
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=False)
        assert result.artifact.spatial.feature_count == 3

    def test_eager_extent(self, mock_overture_db, tmp_path):
        connector = _LocalOvertureConnector(mock_overture_db)
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=False)
        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(1.0)
        assert ymin == pytest.approx(2.0)
        assert xmax == pytest.approx(5.0)
        assert ymax == pytest.approx(6.0)

    def test_eager_content_hash_present(self, mock_overture_db, tmp_path):
        connector = _LocalOvertureConnector(mock_overture_db)
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=False)
        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64

    def test_eager_gpkg_readable_by_fiona(self, mock_overture_db, tmp_path):
        import fiona

        connector = _LocalOvertureConnector(mock_overture_db)
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=False)
        with fiona.open(result.artifact.backing.uri) as src:
            features = list(src)
        assert len(features) == 3

    def test_eager_lineage_params(self, mock_overture_db, tmp_path):
        connector = _LocalOvertureConnector(mock_overture_db)
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=False)
        params = result.artifact.lineage.params
        assert params["source"] == "overture"
        assert params["theme"] == "buildings"
        assert params["type"] == "building"
        assert params["lazy"] is False

    def test_eager_with_bbox_filter(self, mock_overture_db, tmp_path):
        connector = _LocalOvertureConnector(mock_overture_db)
        ref = SourceRef(
            raw="overture://buildings/building",
            params={
                "theme": "buildings",
                "type": "building",
                "bbox": (0.0, 0.0, 2.0, 3.0),
            },
        )
        result = connector.materialize(ref, tmp_path, lazy=False)
        # Only b1 (1.0, 2.0) should match — bbox struct filter
        assert result.artifact.spatial.feature_count == 1

    def test_eager_max_rows_limit(self, mock_overture_db, tmp_path):
        connector = _LocalOvertureConnector(mock_overture_db, max_rows=2)
        result = connector.materialize("overture://buildings/building", tmp_path, lazy=False)
        assert result.artifact.spatial.feature_count <= 2


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestOvertureDiscover:
    """Validate theme/type discovery."""

    def test_discover_returns_all_types(self):
        connector = OvertureConnector()
        entries = connector.discover()
        total_types = sum(len(types) for types in KNOWN_THEMES.values())
        assert len(entries) == total_types

    def test_discover_source_refs_are_overture_urls(self):
        connector = OvertureConnector()
        entries = connector.discover()
        for entry in entries:
            assert entry.source_ref.startswith("overture://")

    def test_discover_filter_by_theme(self):
        connector = OvertureConnector()
        entries = connector.discover(query="buildings")
        assert len(entries) == len(KNOWN_THEMES["buildings"])
        for entry in entries:
            assert "buildings" in entry.source_ref

    def test_discover_filter_by_theme_dict(self):
        connector = OvertureConnector()
        entries = connector.discover(query={"theme": "transportation"})
        assert len(entries) == len(KNOWN_THEMES["transportation"])

    def test_discover_entries_have_metadata(self):
        connector = OvertureConnector()
        entries = connector.discover()
        for entry in entries:
            assert "theme" in entry.metadata
            assert "type" in entry.metadata
            assert "release" in entry.metadata

    def test_discover_entries_have_spatial_hint(self):
        connector = OvertureConnector()
        entries = connector.discover()
        for entry in entries:
            assert entry.spatial_hint["crs"] == "EPSG:4326"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestOvertureMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_theme_type(self):
        connector = OvertureConnector()
        meta = connector.metadata("overture://buildings/building")
        assert meta["theme"] == "buildings"
        assert meta["type"] == "building"

    def test_metadata_has_release(self):
        connector = OvertureConnector()
        meta = connector.metadata("overture://places/place")
        assert "release" in meta
        assert meta["release"] == "2024-12-18.0"

    def test_metadata_has_s3_url(self):
        connector = OvertureConnector()
        meta = connector.metadata("overture://buildings/building")
        assert "s3://" in meta["s3_url"]
        assert "buildings" in meta["s3_url"]

    def test_metadata_crs_is_4326(self):
        connector = OvertureConnector()
        meta = connector.metadata("overture://base/water")
        assert meta["crs"] == "EPSG:4326"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestOvertureErrors:
    """Validate error cases."""

    def test_unknown_theme_raises(self, tmp_path):
        connector = OvertureConnector()
        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize("overture://fake_theme/fake_type", tmp_path, lazy=True)
        assert "Unknown Overture theme" in str(exc_info.value)

    def test_unknown_type_for_known_theme_raises(self, tmp_path):
        connector = OvertureConnector()
        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize("overture://buildings/fake_type", tmp_path, lazy=True)
        assert "Unknown type" in str(exc_info.value)

    def test_metadata_unknown_theme_raises(self):
        connector = OvertureConnector()
        with pytest.raises(MaterializeError):
            connector.metadata("overture://nope/nope")

    def test_invalid_source_ref_raises(self, tmp_path):
        connector = OvertureConnector()
        with pytest.raises(MaterializeError):
            connector.materialize("just_garbage", tmp_path, lazy=True)


# ---------------------------------------------------------------------------
# Connector properties
# ---------------------------------------------------------------------------


class TestOvertureProperties:
    """Validate connector protocol compliance."""

    def test_name(self):
        assert OvertureConnector().name == "overture"

    def test_capabilities_include_materialize(self):
        from quarry_core.connector import ConnectorCapability

        caps = OvertureConnector().capabilities
        assert ConnectorCapability.MATERIALIZE in caps

    def test_capabilities_include_discover(self):
        from quarry_core.connector import ConnectorCapability

        caps = OvertureConnector().capabilities
        assert ConnectorCapability.DISCOVER in caps

    def test_capabilities_include_lazy(self):
        from quarry_core.connector import ConnectorCapability

        caps = OvertureConnector().capabilities
        assert ConnectorCapability.MATERIALIZE_LAZY in caps

    def test_capabilities_include_metadata(self):
        from quarry_core.connector import ConnectorCapability

        caps = OvertureConnector().capabilities
        assert ConnectorCapability.METADATA_ONLY in caps

    def test_custom_release(self):
        connector = OvertureConnector(release="2025-01-01.0")
        meta = connector.metadata("overture://buildings/building")
        assert meta["release"] == "2025-01-01.0"

    def test_custom_max_rows(self):
        connector = OvertureConnector(max_rows=500)
        assert connector._max_rows == 500
