"""
Pressure test: GeoJSONSeqConnector.

Lane: connector

Validates GeoJSON Sequence (RFC 8142) file materialization:
- source_ref parsing (local path, SourceRef)
- local eager: wrap in place, LOCAL_FILE backing
- local lazy: metadata only, LAZY_HANDLE backing
- discover: list .geojsonl/.geojsonseq/.ndjson files in directory
- metadata: read without materializing
- error handling: nonexistent files, empty files, malformed JSON lines
- multiple extensions: .geojsonl, .geojsonseq, .ndjson
- blank lines between features
"""

from __future__ import annotations

import json

import pytest
from quarry_connectors.geojsonseq import GeoJSONSeqConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_geojsonseq_file(path, features):
    """Create a GeoJSON Sequence file (one JSON object per line).

    Args:
        path: Output file path
        features: List of GeoJSON Feature dicts
    """
    with open(path, "w", encoding="utf-8") as f:
        for feature in features:
            f.write(json.dumps(feature) + "\n")


@pytest.fixture()
def geojsonseq_file(tmp_path):
    """Create a GeoJSONSeq file with 3 point geometries."""
    path = tmp_path / "points.geojsonl"

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
            "properties": {"name": "alpha", "value": 1.0},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [3.0, 4.0]},
            "properties": {"name": "beta", "value": 2.0},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [5.0, 6.0]},
            "properties": {"name": "gamma", "value": 3.0},
        },
    ]

    _create_geojsonseq_file(path, features)
    return path


@pytest.fixture()
def geojsonseq_file_polygons(tmp_path):
    """Create a GeoJSONSeq file with polygon geometries."""
    path = tmp_path / "zones.geojsonl"

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            "properties": {"zone_id": "A", "area_km2": 100.0},
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]],
            },
            "properties": {"zone_id": "B", "area_km2": 250.5},
        },
    ]

    _create_geojsonseq_file(path, features)
    return path


@pytest.fixture()
def ndjson_file(tmp_path):
    """Create a .ndjson file with point geometries."""
    path = tmp_path / "points.ndjson"

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [10.0, 20.0]},
            "properties": {"id": 1, "label": "first"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [30.0, 40.0]},
            "properties": {"id": 2, "label": "second"},
        },
    ]

    _create_geojsonseq_file(path, features)
    return path


@pytest.fixture()
def geojsonseq_file_with_blanks(tmp_path):
    """Create a GeoJSONSeq file with blank lines between features."""
    path = tmp_path / "with_blanks.geojsonl"

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
            "properties": {"name": "first"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [3.0, 4.0]},
            "properties": {"name": "second"},
        },
    ]

    with open(path, "w", encoding="utf-8") as f:
        for feature in features:
            f.write(json.dumps(feature) + "\n")
            f.write("\n")  # Blank line
            f.write("   \n")  # Whitespace-only line

    return path


@pytest.fixture()
def directory_with_geojsonseq(tmp_path, geojsonseq_file, geojsonseq_file_polygons):
    """Create a directory with multiple GeoJSONSeq files."""
    # Files are already in tmp_path from fixtures
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Local Materialization
# ---------------------------------------------------------------------------


class TestGeoJSONSeqEagerLocal:
    """Validate eager materialization of local GeoJSONSeq files."""

    def test_eager_produces_vector(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.name == "points"

    def test_eager_produces_local_file_backing(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".geojsonl")

    def test_eager_wrapped_local_strategy(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_feature_count(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_eager_extent(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(1.0)
        assert ymin == pytest.approx(2.0)
        assert xmax == pytest.approx(5.0)
        assert ymax == pytest.approx(6.0)

    def test_eager_crs(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path)

        assert result.artifact.spatial.crs is not None
        assert "4326" in result.artifact.spatial.crs

    def test_eager_content_hash_present(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path)

        assert result.artifact.lineage.params["source"] == "geojsonseq"
        assert result.artifact.lineage.params["path"] == str(geojsonseq_file.resolve())
        assert result.artifact.lineage.params["lazy"] is False

    def test_eager_metadata_schema(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path)

        assert "schema" in result.artifact.metadata
        assert result.artifact.metadata["schema"]["geometry"] == "Point"
        assert "name" in result.artifact.metadata["schema"]["properties"]
        assert "value" in result.artifact.metadata["schema"]["properties"]

    def test_eager_polygons(self, geojsonseq_file_polygons, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file_polygons), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 2
        assert result.artifact.metadata["schema"]["geometry"] == "Polygon"


# ---------------------------------------------------------------------------
# Lazy Local Materialization
# ---------------------------------------------------------------------------


class TestGeoJSONSeqLazyLocal:
    """Validate lazy (metadata-only) materialization of local GeoJSONSeq files."""

    def test_lazy_backing_kind(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_feature_count(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path, lazy=True)

        assert result.artifact.spatial.feature_count == 3

    def test_lazy_extent(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path, lazy=True)

        extent = result.artifact.spatial.extent
        assert extent is not None
        assert extent[0] == pytest.approx(1.0)

    def test_lazy_backing_uri(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path, lazy=True)

        assert result.artifact.backing.uri == str(geojsonseq_file.resolve())

    def test_lazy_lineage(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------


class TestGeoJSONSeqExtensions:
    """Validate support for different file extensions."""

    def test_geojsonl_extension(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 3

    def test_geojsonseq_extension(self, tmp_path):
        """Test .geojsonseq extension."""
        path = tmp_path / "data.geojsonseq"
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                "properties": {"id": 1},
            },
        ]
        _create_geojsonseq_file(path, features)

        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(path), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 1

    def test_ndjson_extension(self, ndjson_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(ndjson_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 2


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestGeoJSONSeqDiscover:
    """Validate file discovery."""

    def test_discover_lists_geojsonseq_files(self, directory_with_geojsonseq):
        connector = GeoJSONSeqConnector()
        entries = connector.discover(str(directory_with_geojsonseq))

        names = {e.name for e in entries}
        assert "points" in names
        assert "zones" in names

    def test_discover_source_refs(self, directory_with_geojsonseq):
        connector = GeoJSONSeqConnector()
        entries = connector.discover(str(directory_with_geojsonseq))

        for entry in entries:
            assert entry.source_ref.endswith(".geojsonl")

    def test_discover_with_dict_query(self, directory_with_geojsonseq):
        connector = GeoJSONSeqConnector()
        entries = connector.discover({"path": str(directory_with_geojsonseq)})

        assert len(entries) >= 2

    def test_discover_empty_directory(self, tmp_path):
        connector = GeoJSONSeqConnector()
        entries = connector.discover(str(tmp_path))

        assert entries == []

    def test_discover_nonexistent_directory_raises(self, tmp_path):
        connector = GeoJSONSeqConnector()
        nonexistent = tmp_path / "no_such_dir"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestGeoJSONSeqMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_schema(self, geojsonseq_file):
        connector = GeoJSONSeqConnector()
        meta = connector.metadata(str(geojsonseq_file))

        assert "schema" in meta
        assert meta["schema"]["geometry"] == "Point"

    def test_metadata_feature_count(self, geojsonseq_file):
        connector = GeoJSONSeqConnector()
        meta = connector.metadata(str(geojsonseq_file))

        assert meta["feature_count"] == 3

    def test_metadata_crs(self, geojsonseq_file):
        connector = GeoJSONSeqConnector()
        meta = connector.metadata(str(geojsonseq_file))

        assert meta["crs"] is not None
        assert "4326" in meta["crs"]

    def test_metadata_extent(self, geojsonseq_file):
        connector = GeoJSONSeqConnector()
        meta = connector.metadata(str(geojsonseq_file))

        extent = meta["extent"]
        assert extent is not None
        assert len(extent) == 4

    def test_metadata_driver(self, geojsonseq_file):
        connector = GeoJSONSeqConnector()
        meta = connector.metadata(str(geojsonseq_file))

        # Should be either "GeoJSONSeq" (fiona) or "json_fallback"
        assert meta["driver"] in ("GeoJSONSeq", "json_fallback")


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestGeoJSONSeqErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path):
        connector = GeoJSONSeqConnector()
        nonexistent = tmp_path / "does_not_exist.geojsonl"

        with pytest.raises(MaterializeError):
            connector.materialize(str(nonexistent), tmp_path)

    def test_wrong_extension_raises(self, tmp_path):
        connector = GeoJSONSeqConnector()
        bad_file = tmp_path / "not_geojsonseq.txt"
        bad_file.write_text("this is not a geojsonseq file")

        with pytest.raises(MaterializeError):
            connector.materialize(str(bad_file), tmp_path)

    def test_empty_file(self, tmp_path):
        """Empty file should result in 0 features, not an error."""
        empty_file = tmp_path / "empty.geojsonl"
        empty_file.write_text("")

        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(empty_file), tmp_path)

        assert result.artifact.spatial.feature_count == 0

    def test_malformed_json_lines(self, tmp_path):
        """Malformed lines should be skipped, valid lines processed."""
        bad_file = tmp_path / "bad.geojsonl"
        bad_file.write_text(
            '{"type": "Feature", "geometry": {"type": "Point", "coordinates": [1.0, 2.0]}, '
            '"properties": {}}\n'
            "this is not json\n"
            '{"type": "Feature", "geometry": {"type": "Point", "coordinates": [3.0, 4.0]}, '
            '"properties": {}}\n'
        )

        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(bad_file), tmp_path)

        # Should have 2 valid features (malformed line skipped)
        assert result.artifact.spatial.feature_count == 2

    def test_metadata_nonexistent_file_raises(self, tmp_path):
        connector = GeoJSONSeqConnector()
        nonexistent = tmp_path / "does_not_exist.geojsonl"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))

    def test_discover_no_path_raises(self):
        connector = GeoJSONSeqConnector()

        with pytest.raises(MaterializeError):
            connector.discover()

    def test_discover_file_raises(self, geojsonseq_file):
        connector = GeoJSONSeqConnector()

        with pytest.raises(MaterializeError):
            connector.discover(str(geojsonseq_file))


# ---------------------------------------------------------------------------
# SourceRef Support
# ---------------------------------------------------------------------------


class TestGeoJSONSeqSourceRef:
    """Validate SourceRef handling."""

    def test_sourceref_local(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        ref = SourceRef.local(str(geojsonseq_file))
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_raw_string_path(self, geojsonseq_file, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestGeoJSONSeqCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        connector = GeoJSONSeqConnector()
        assert connector.name == "geojsonseq"

    def test_capabilities(self):
        from quarry_core.connector import ConnectorCapability

        connector = GeoJSONSeqConnector()
        caps = connector.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps


# ---------------------------------------------------------------------------
# Blank Lines Handling
# ---------------------------------------------------------------------------


class TestGeoJSONSeqBlankLines:
    """Validate handling of blank/whitespace lines between features."""

    def test_blank_lines_skipped(self, geojsonseq_file_with_blanks, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file_with_blanks), tmp_path)

        # Should have 2 features (blank lines skipped)
        assert result.artifact.spatial.feature_count == 2

    def test_blank_lines_extent_computed(self, geojsonseq_file_with_blanks, tmp_path):
        connector = GeoJSONSeqConnector()
        result = connector.materialize(str(geojsonseq_file_with_blanks), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(1.0)
        assert ymin == pytest.approx(2.0)
        assert xmax == pytest.approx(3.0)
        assert ymax == pytest.approx(4.0)
