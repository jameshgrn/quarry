"""
Pressure test: FlatGeobufConnector.

Lane: connector

Validates FlatGeobuf file materialization:
- source_ref parsing (local path, remote URL, SourceRef)
- local eager: wrap in place, LOCAL_FILE backing
- local lazy: metadata only, LAZY_HANDLE backing
- remote eager: download to workspace
- remote lazy: metadata via /vsicurl/
- discover: list .fgb files in directory
- metadata: read without materializing
- bbox filtering: spatial filter for efficient remote reads
- error handling: nonexistent files, invalid formats
"""

from __future__ import annotations

import pytest
from quarry_connectors.flatgeobuf import FlatGeobufConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_fgb_file(path, geometries, properties, crs="EPSG:4326"):
    """Create a FlatGeobuf file using fiona.

    Args:
        path: Output file path
        geometries: List of GeoJSON geometry dicts
        properties: Dict of column names to lists of values
        crs: CRS string or EPSG code
    """
    import fiona
    from fiona.crs import CRS

    # Determine geometry type from first geometry
    if not geometries:
        raise ValueError("At least one geometry required")

    geom_type = geometries[0]["type"]

    # Build schema
    prop_schema = {}
    for col_name, values in properties.items():
        if values:
            val = values[0]
            if isinstance(val, int):
                prop_schema[col_name] = "int"
            elif isinstance(val, float):
                prop_schema[col_name] = "float"
            else:
                prop_schema[col_name] = "str"

    schema = {"geometry": geom_type, "properties": prop_schema}

    # Parse CRS
    if crs.startswith("EPSG:"):
        crs_obj = CRS.from_epsg(int(crs.split(":")[1]))
    else:
        crs_obj = CRS.from_epsg(4326)

    with fiona.open(str(path), "w", driver="FlatGeobuf", schema=schema, crs=crs_obj) as dst:
        for i, geom in enumerate(geometries):
            props = {k: v[i] for k, v in properties.items()}
            dst.write({"geometry": geom, "properties": props})


@pytest.fixture()
def fgb_file(tmp_path):
    """Create a FlatGeobuf file with 3 point geometries."""
    path = tmp_path / "points.fgb"

    geometries = [
        {"type": "Point", "coordinates": [1.0, 2.0]},
        {"type": "Point", "coordinates": [3.0, 4.0]},
        {"type": "Point", "coordinates": [5.0, 6.0]},
    ]

    properties = {
        "name": ["alpha", "beta", "gamma"],
        "value": [1.0, 2.0, 3.0],
    }

    _create_fgb_file(path, geometries, properties)
    return path


@pytest.fixture()
def fgb_file_polygons(tmp_path):
    """Create a FlatGeobuf file with polygon geometries."""
    path = tmp_path / "zones.fgb"

    geometries = [
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        {"type": "Polygon", "coordinates": [[[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]]},
    ]

    properties = {
        "zone_id": ["A", "B"],
        "area_km2": [100.0, 250.5],
    }

    _create_fgb_file(path, geometries, properties)
    return path


@pytest.fixture()
def directory_with_fgb(tmp_path, fgb_file, fgb_file_polygons):
    """Create a directory with multiple .fgb files."""
    # Files are already in tmp_path from fixtures
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Local Materialization
# ---------------------------------------------------------------------------


class TestFlatGeobufEagerLocal:
    """Validate eager materialization of local FlatGeobuf files."""

    def test_eager_produces_vector(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.name == "points"

    def test_eager_produces_local_file_backing(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".fgb")

    def test_eager_wrapped_local_strategy(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_feature_count(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_eager_extent(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(1.0)
        assert ymin == pytest.approx(2.0)
        assert xmax == pytest.approx(5.0)
        assert ymax == pytest.approx(6.0)

    def test_eager_crs(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path)

        assert result.artifact.spatial.crs is not None
        assert "4326" in result.artifact.spatial.crs

    def test_eager_content_hash_present(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path)

        assert result.artifact.lineage.params["source"] == "flatgeobuf"
        assert result.artifact.lineage.params["path_or_url"] == str(fgb_file)
        assert result.artifact.lineage.params["lazy"] is False
        assert result.artifact.lineage.params["is_remote"] is False

    def test_eager_metadata_schema(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path)

        assert "schema" in result.artifact.metadata
        assert result.artifact.metadata["schema"]["geometry"] == "Point"
        assert "name" in result.artifact.metadata["schema"]["properties"]
        assert "value" in result.artifact.metadata["schema"]["properties"]

    def test_eager_polygons(self, fgb_file_polygons, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file_polygons), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 2
        assert result.artifact.metadata["schema"]["geometry"] == "Polygon"


# ---------------------------------------------------------------------------
# Lazy Local Materialization
# ---------------------------------------------------------------------------


class TestFlatGeobufLazyLocal:
    """Validate lazy (metadata-only) materialization of local FlatGeobuf files."""

    def test_lazy_backing_kind(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_feature_count(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path, lazy=True)

        assert result.artifact.spatial.feature_count == 3

    def test_lazy_extent(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path, lazy=True)

        extent = result.artifact.spatial.extent
        assert extent is not None
        assert extent[0] == pytest.approx(1.0)

    def test_lazy_backing_uri(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path, lazy=True)

        assert result.artifact.backing.uri == str(fgb_file)

    def test_lazy_lineage(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True
        assert result.artifact.lineage.params["is_remote"] is False


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestFlatGeobufDiscover:
    """Validate file discovery."""

    def test_discover_lists_fgb_files(self, directory_with_fgb):
        connector = FlatGeobufConnector()
        entries = connector.discover(str(directory_with_fgb))

        names = {e.name for e in entries}
        assert "points" in names
        assert "zones" in names

    def test_discover_source_refs(self, directory_with_fgb):
        connector = FlatGeobufConnector()
        entries = connector.discover(str(directory_with_fgb))

        for entry in entries:
            assert entry.source_ref.endswith(".fgb") or entry.source_ref.endswith(".FGB")

    def test_discover_with_dict_query(self, directory_with_fgb):
        connector = FlatGeobufConnector()
        entries = connector.discover({"path": str(directory_with_fgb)})

        assert len(entries) >= 2

    def test_discover_empty_directory(self, tmp_path):
        connector = FlatGeobufConnector()
        entries = connector.discover(str(tmp_path))

        assert entries == []

    def test_discover_nonexistent_directory_raises(self, tmp_path):
        connector = FlatGeobufConnector()
        nonexistent = tmp_path / "no_such_dir"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestFlatGeobufMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_schema(self, fgb_file):
        connector = FlatGeobufConnector()
        meta = connector.metadata(str(fgb_file))

        assert "schema" in meta
        assert meta["schema"]["geometry"] == "Point"

    def test_metadata_feature_count(self, fgb_file):
        connector = FlatGeobufConnector()
        meta = connector.metadata(str(fgb_file))

        assert meta["feature_count"] == 3

    def test_metadata_crs(self, fgb_file):
        connector = FlatGeobufConnector()
        meta = connector.metadata(str(fgb_file))

        assert meta["crs"] is not None
        assert "4326" in meta["crs"]

    def test_metadata_extent(self, fgb_file):
        connector = FlatGeobufConnector()
        meta = connector.metadata(str(fgb_file))

        extent = meta["extent"]
        assert extent is not None
        assert len(extent) == 4

    def test_metadata_driver(self, fgb_file):
        connector = FlatGeobufConnector()
        meta = connector.metadata(str(fgb_file))

        assert meta["driver"] == "FlatGeobuf"

    def test_metadata_is_remote_false(self, fgb_file):
        connector = FlatGeobufConnector()
        meta = connector.metadata(str(fgb_file))

        assert meta["is_remote"] is False


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestFlatGeobufErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path):
        connector = FlatGeobufConnector()
        nonexistent = tmp_path / "does_not_exist.fgb"

        with pytest.raises(MaterializeError):
            connector.materialize(str(nonexistent), tmp_path)

    def test_non_fgb_file_raises(self, tmp_path):
        connector = FlatGeobufConnector()
        bad_file = tmp_path / "not_fgb.txt"
        bad_file.write_text("this is not a flatgeobuf file")

        with pytest.raises(MaterializeError):
            connector.materialize(str(bad_file), tmp_path)

    def test_metadata_nonexistent_file_raises(self, tmp_path):
        connector = FlatGeobufConnector()
        nonexistent = tmp_path / "does_not_exist.fgb"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))

    def test_discover_no_path_raises(self):
        connector = FlatGeobufConnector()

        with pytest.raises(MaterializeError):
            connector.discover()

    def test_discover_file_raises(self, fgb_file):
        connector = FlatGeobufConnector()

        with pytest.raises(MaterializeError):
            connector.discover(str(fgb_file))


# ---------------------------------------------------------------------------
# SourceRef Support
# ---------------------------------------------------------------------------


class TestFlatGeobufSourceRef:
    """Validate SourceRef handling."""

    def test_sourceref_local(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        ref = SourceRef.local(str(fgb_file))
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_raw_string_path(self, fgb_file, tmp_path):
        connector = FlatGeobufConnector()
        result = connector.materialize(str(fgb_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_sourceref_with_bbox(self, fgb_file, tmp_path):
        """SourceRef with bbox param is parsed correctly."""
        from quarry_core.source_ref import SourceRef, SourceRefKind

        connector = FlatGeobufConnector()
        # Create SourceRef with bbox in params directly
        ref = SourceRef(
            raw=str(fgb_file),
            kind=SourceRefKind.LOCAL_VECTOR,
            params={"path": str(fgb_file), "bbox": (0, 0, 2, 3)},
        )
        result = connector.materialize(ref, tmp_path, lazy=True)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.lineage.params["bbox_filter"] == (0, 0, 2, 3)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestFlatGeobufCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        connector = FlatGeobufConnector()
        assert connector.name == "flatgeobuf"

    def test_capabilities(self):
        from quarry_core.connector import ConnectorCapability

        connector = FlatGeobufConnector()
        caps = connector.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps
