"""
Pressure test: TopoJSONConnector.

Lane: connector

Validates TopoJSON file materialization:
- source_ref parsing (local path, with ::object_name)
- local eager: wrap in place, LOCAL_FILE backing
- local lazy: metadata only, LAZY_HANDLE backing
- object selection: default (first) object, explicit via ::
- arc decoding: simple arcs, delta-encoded with transform, negative indices
- discover: list objects in file, list files in directory
- metadata: read without materializing
- error handling: nonexistent files, invalid JSON, not Topology type, nonexistent object
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from quarry_connectors.topojson import TopoJSONConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_topojson_file(
    path: Path,
    objects: dict[str, Any],
    arcs: list[list[list[float]]],
    bbox: list[float] | None = None,
    transform: dict[str, Any] | None = None,
) -> None:
    """Create a TopoJSON file.

    Args:
        path: Output file path
        objects: TopoJSON objects dict
        arcs: Arcs array (delta-encoded coordinates)
        bbox: Optional bounding box [xmin, ymin, xmax, ymax]
        transform: Optional transform dict with "scale" and "translate"
    """
    topo: dict[str, Any] = {
        "type": "Topology",
        "objects": objects,
        "arcs": arcs,
    }
    if bbox:
        topo["bbox"] = bbox
    if transform:
        topo["transform"] = transform

    with open(path, "w") as f:
        json.dump(topo, f)


@pytest.fixture()
def topojson_points(tmp_path: Path) -> Path:
    """Create a TopoJSON file with point geometries (no arcs needed)."""
    path = tmp_path / "points.topojson"

    objects = {
        "places": {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [1.0, 2.0], "properties": {"name": "A"}},
                {"type": "Point", "coordinates": [3.0, 4.0], "properties": {"name": "B"}},
                {"type": "Point", "coordinates": [5.0, 6.0], "properties": {"name": "C"}},
            ],
        }
    }

    _create_topojson_file(path, objects, arcs=[], bbox=[1.0, 2.0, 5.0, 6.0])
    return path


@pytest.fixture()
def topojson_polygons(tmp_path: Path) -> Path:
    """Create a TopoJSON file with polygon geometries (with arcs)."""
    path = tmp_path / "zones.topojson"

    # Two adjacent squares sharing an edge
    # Arc 0: square from (0,0) to (1,1)
    # Arc 1: square from (1,0) to (2,1) - shares edge with arc 0
    arcs = [
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],  # Square: (0,0)->(1,0)->(1,1)->(0,1)->(0,0)
        [[1, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],  # Square: (1,0)->(2,0)->(2,1)->(1,1)->(1,0)
    ]

    objects = {
        "zones": {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Polygon", "arcs": [[0]], "properties": {"zone": "A"}},
                {"type": "Polygon", "arcs": [[1]], "properties": {"zone": "B"}},
            ],
        }
    }

    _create_topojson_file(path, objects, arcs, bbox=[0.0, 0.0, 2.0, 1.0])
    return path


@pytest.fixture()
def topojson_multi_object(tmp_path: Path) -> Path:
    """Create a TopoJSON file with multiple named objects."""
    path = tmp_path / "multi.topojson"

    objects = {
        "counties": {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [1.0, 1.0], "properties": {"name": "County1"}},
                {"type": "Point", "coordinates": [2.0, 2.0], "properties": {"name": "County2"}},
            ],
        },
        "states": {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [10.0, 10.0], "properties": {"name": "State1"}},
            ],
        },
        "cities": {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [100.0, 100.0], "properties": {"name": "City1"}},
                {"type": "Point", "coordinates": [200.0, 200.0], "properties": {"name": "City2"}},
                {"type": "Point", "coordinates": [300.0, 300.0], "properties": {"name": "City3"}},
            ],
        },
    }

    _create_topojson_file(path, objects, arcs=[], bbox=[1.0, 1.0, 300.0, 300.0])
    return path


@pytest.fixture()
def topojson_with_transform(tmp_path: Path) -> Path:
    """Create a TopoJSON file with delta-encoded arcs and transform."""
    path = tmp_path / "encoded.topojson"

    # Delta-encoded arcs (small integers)
    # With transform: scale=[0.001, 0.001], translate=[100.0, 200.0]
    # Arc 0: (0,0)->(1000,0)->(1000,1000)->(0,1000)->(0,0)
    # Decoded: (100,200)->(101,200)->(101,201)->(100,201)->(100,200)
    arcs = [
        [[0, 0], [1000, 0], [0, 1000], [-1000, 0], [0, -1000]],
    ]

    objects = {
        "areas": {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Polygon", "arcs": [[0]], "properties": {"id": "area1"}},
            ],
        }
    }

    transform = {
        "scale": [0.001, 0.001],
        "translate": [100.0, 200.0],
    }

    _create_topojson_file(path, objects, arcs, transform=transform)
    return path


@pytest.fixture()
def topojson_negative_arcs(tmp_path: Path) -> Path:
    """Create a TopoJSON file with negative arc indices (reversed arcs)."""
    path = tmp_path / "negative.topojson"

    # Single arc: square from (0,0) to (1,1)
    arcs = [
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
    ]

    # Polygon using positive index (normal)
    # Another polygon using negative index ~0 (reversed arc 0)
    # ~0 = -1 in two's complement (bitwise NOT of 0)
    objects = {
        "shapes": {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Polygon", "arcs": [[0]], "properties": {"id": "normal"}},
                {"type": "Polygon", "arcs": [[~0]], "properties": {"id": "reversed"}},
            ],
        }
    }

    _create_topojson_file(path, objects, arcs)
    return path


@pytest.fixture()
def directory_with_topojson(tmp_path: Path, topojson_points: Path, topojson_polygons: Path) -> Path:
    """Create a directory with multiple .topojson files."""
    # Files are already in tmp_path from fixtures
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Local Materialization
# ---------------------------------------------------------------------------


class TestTopoJSONEagerLocal:
    """Validate eager materialization of local TopoJSON files."""

    def test_eager_produces_vector(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.name == "places"

    def test_eager_produces_local_file_backing(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".topojson")

    def test_eager_wrapped_local_strategy(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_feature_count(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_eager_extent(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(1.0)
        assert ymin == pytest.approx(2.0)
        assert xmax == pytest.approx(5.0)
        assert ymax == pytest.approx(6.0)

    def test_eager_crs(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path)

        assert result.artifact.spatial.crs == "EPSG:4326"

    def test_eager_content_hash_present(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path)

        assert result.artifact.lineage.params["source"] == "topojson"
        assert result.artifact.lineage.params["object_name"] == "places"
        assert result.artifact.lineage.params["lazy"] is False

    def test_eager_metadata(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path)

        assert result.artifact.metadata["object_name"] == "places"
        assert result.artifact.metadata["object_type"] == "GeometryCollection"
        assert result.artifact.metadata["feature_count"] == 3
        assert result.artifact.metadata["assumed_crs"] is True
        assert "Point" in result.artifact.metadata["geometry_types"]

    def test_eager_polygons(self, topojson_polygons: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_polygons), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 2
        assert "Polygon" in result.artifact.metadata["geometry_types"]


# ---------------------------------------------------------------------------
# Lazy Local Materialization
# ---------------------------------------------------------------------------


class TestTopoJSONLazyLocal:
    """Validate lazy (metadata-only) materialization of local TopoJSON files."""

    def test_lazy_backing_kind(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_feature_count(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path, lazy=True)

        assert result.artifact.spatial.feature_count == 3

    def test_lazy_extent(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path, lazy=True)

        extent = result.artifact.spatial.extent
        assert extent is not None
        assert extent[0] == pytest.approx(1.0)

    def test_lazy_backing_uri(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path, lazy=True)

        assert result.artifact.backing.uri == str(topojson_points)

    def test_lazy_lineage(self, topojson_points: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_points), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True


# ---------------------------------------------------------------------------
# Object Selection
# ---------------------------------------------------------------------------


class TestTopoJSONObjectSelection:
    """Validate object selection via :: syntax."""

    def test_default_first_object(self, topojson_multi_object: Path, tmp_path: Path):
        """Without ::object_name, should select first object."""
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_multi_object), tmp_path)

        # First object alphabetically is "cities" (not "counties" - depends on dict order)
        # Actually dict preserves insertion order, so "counties" is first
        assert result.artifact.name == "counties"
        assert result.artifact.spatial.feature_count == 2

    def test_explicit_object_selection(self, topojson_multi_object: Path, tmp_path: Path):
        """With ::object_name, should select specified object."""
        connector = TopoJSONConnector()
        result = connector.materialize(f"{topojson_multi_object}::states", tmp_path)

        assert result.artifact.name == "states"
        assert result.artifact.spatial.feature_count == 1

    def test_explicit_object_selection_cities(self, topojson_multi_object: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(f"{topojson_multi_object}::cities", tmp_path)

        assert result.artifact.name == "cities"
        assert result.artifact.spatial.feature_count == 3

    def test_all_objects_in_metadata(self, topojson_multi_object: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_multi_object), tmp_path)

        all_names = result.artifact.metadata["all_object_names"]
        assert "counties" in all_names
        assert "states" in all_names
        assert "cities" in all_names


# ---------------------------------------------------------------------------
# Arc Decoding
# ---------------------------------------------------------------------------


class TestTopoJSONArcDecoding:
    """Validate arc decoding functionality."""

    def test_simple_arcs(self, topojson_polygons: Path, tmp_path: Path):
        """Polygons with simple arcs should decode correctly."""
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_polygons), tmp_path)

        assert result.artifact.spatial.feature_count == 2
        assert result.artifact.spatial.extent is not None

    def test_delta_encoded_with_transform(self, topojson_with_transform: Path, tmp_path: Path):
        """Delta-encoded arcs with transform should decode correctly."""
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_with_transform), tmp_path)

        assert result.artifact.spatial.feature_count == 1
        # Extent should be computed from decoded coordinates
        extent = result.artifact.spatial.extent
        assert extent is not None
        # Decoded square: (100,200) to (101,201)
        assert extent[0] == pytest.approx(100.0, abs=0.01)
        assert extent[1] == pytest.approx(200.0, abs=0.01)
        assert extent[2] == pytest.approx(101.0, abs=0.01)
        assert extent[3] == pytest.approx(201.0, abs=0.01)

    def test_has_transform_in_metadata(self, topojson_with_transform: Path, tmp_path: Path):
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_with_transform), tmp_path)

        assert result.artifact.metadata["has_transform"] is True
        assert result.artifact.lineage.params["has_transform"] is True

    def test_negative_arc_indices(self, topojson_negative_arcs: Path, tmp_path: Path):
        """Negative arc indices should decode to reversed arcs."""
        connector = TopoJSONConnector()
        result = connector.materialize(str(topojson_negative_arcs), tmp_path)

        # Both polygons should be valid (one with normal arc, one with reversed)
        assert result.artifact.spatial.feature_count == 2


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestTopoJSONDiscover:
    """Validate discovery functionality."""

    def test_discover_lists_objects_in_file(self, topojson_multi_object: Path):
        """When query points to a file, list objects within it."""
        connector = TopoJSONConnector()
        entries = connector.discover(str(topojson_multi_object))

        names = {e.name for e in entries}
        assert "counties" in names
        assert "states" in names
        assert "cities" in names

    def test_discover_object_source_refs(self, topojson_multi_object: Path):
        """Object entries should have ::object_name in source_ref."""
        connector = TopoJSONConnector()
        entries = connector.discover(str(topojson_multi_object))

        for entry in entries:
            assert "::" in entry.source_ref
            assert entry.source_ref.startswith(str(topojson_multi_object))

    def test_discover_object_spatial_hints(self, topojson_multi_object: Path):
        """Object entries should have spatial hints."""
        connector = TopoJSONConnector()
        entries = connector.discover(str(topojson_multi_object))

        for entry in entries:
            assert entry.spatial_hint.get("crs") == "EPSG:4326"
            assert "feature_count" in entry.spatial_hint

    def test_discover_lists_files_in_directory(self, directory_with_topojson: Path):
        """When query points to a directory, list .topojson files."""
        connector = TopoJSONConnector()
        entries = connector.discover(str(directory_with_topojson))

        names = {e.name for e in entries}
        assert "points" in names
        assert "zones" in names

    def test_discover_file_source_refs(self, directory_with_topojson: Path):
        """File entries should have .topojson extension in source_ref."""
        connector = TopoJSONConnector()
        entries = connector.discover(str(directory_with_topojson))

        for entry in entries:
            assert entry.source_ref.endswith(".topojson")

    def test_discover_with_dict_query(self, directory_with_topojson: Path):
        connector = TopoJSONConnector()
        entries = connector.discover({"path": str(directory_with_topojson)})

        assert len(entries) >= 2

    def test_discover_empty_directory(self, tmp_path: Path):
        connector = TopoJSONConnector()
        entries = connector.discover(str(tmp_path))

        assert entries == []

    def test_discover_nonexistent_path_raises(self, tmp_path: Path):
        connector = TopoJSONConnector()
        nonexistent = tmp_path / "no_such_path"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestTopoJSONMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_object_names(self, topojson_multi_object: Path):
        connector = TopoJSONConnector()
        meta = connector.metadata(str(topojson_multi_object))

        assert "counties" in meta["object_names"]
        assert "states" in meta["object_names"]
        assert "cities" in meta["object_names"]

    def test_metadata_returns_object_info(self, topojson_multi_object: Path):
        connector = TopoJSONConnector()
        meta = connector.metadata(str(topojson_multi_object))

        assert meta["objects"]["counties"]["feature_count"] == 2
        assert meta["objects"]["states"]["feature_count"] == 1
        assert meta["objects"]["cities"]["feature_count"] == 3

    def test_metadata_returns_geometry_types(self, topojson_points: Path):
        connector = TopoJSONConnector()
        meta = connector.metadata(str(topojson_points))

        assert "Point" in meta["objects"]["places"]["geometry_types"]

    def test_metadata_returns_arc_count(self, topojson_polygons: Path):
        connector = TopoJSONConnector()
        meta = connector.metadata(str(topojson_polygons))

        assert meta["arc_count"] == 2

    def test_metadata_returns_extent(self, topojson_points: Path):
        connector = TopoJSONConnector()
        meta = connector.metadata(str(topojson_points))

        extent = meta["extent"]
        assert extent is not None
        assert len(extent) == 4

    def test_metadata_returns_crs(self, topojson_points: Path):
        connector = TopoJSONConnector()
        meta = connector.metadata(str(topojson_points))

        assert meta["crs"] == "EPSG:4326"
        assert meta["assumed_crs"] is True

    def test_metadata_has_transform(self, topojson_with_transform: Path):
        connector = TopoJSONConnector()
        meta = connector.metadata(str(topojson_with_transform))

        assert meta["has_transform"] is True


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestTopoJSONErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path: Path):
        connector = TopoJSONConnector()
        nonexistent = tmp_path / "does_not_exist.topojson"

        with pytest.raises(MaterializeError):
            connector.materialize(str(nonexistent), tmp_path)

    def test_not_json_raises(self, tmp_path: Path):
        connector = TopoJSONConnector()
        bad_file = tmp_path / "not_json.topojson"
        bad_file.write_text("this is not json {")

        with pytest.raises(MaterializeError):
            connector.materialize(str(bad_file), tmp_path)

    def test_not_topology_type_raises(self, tmp_path: Path):
        connector = TopoJSONConnector()
        bad_file = tmp_path / "not_topology.topojson"
        bad_file.write_text(json.dumps({"type": "FeatureCollection", "features": []}))

        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize(str(bad_file), tmp_path)

        assert "Topology" in str(exc_info.value)

    def test_missing_objects_raises(self, tmp_path: Path):
        connector = TopoJSONConnector()
        bad_file = tmp_path / "no_objects.topojson"
        bad_file.write_text(json.dumps({"type": "Topology", "arcs": []}))

        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize(str(bad_file), tmp_path)

        assert "objects" in str(exc_info.value)

    def test_nonexistent_object_name_raises(self, topojson_multi_object: Path, tmp_path: Path):
        connector = TopoJSONConnector()

        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize(f"{topojson_multi_object}::nonexistent", tmp_path)

        assert "nonexistent" in str(exc_info.value)
        assert "counties" in str(exc_info.value)  # Should list available objects

    def test_metadata_nonexistent_file_raises(self, tmp_path: Path):
        connector = TopoJSONConnector()
        nonexistent = tmp_path / "does_not_exist.topojson"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))

    def test_discover_no_path_raises(self):
        connector = TopoJSONConnector()

        with pytest.raises(MaterializeError):
            connector.discover()

    def test_discover_non_topojson_file_raises(self, tmp_path: Path):
        connector = TopoJSONConnector()
        txt_file = tmp_path / "not_topojson.txt"
        txt_file.write_text("hello")

        with pytest.raises(MaterializeError):
            connector.discover(str(txt_file))


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestTopoJSONCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        connector = TopoJSONConnector()
        assert connector.name == "topojson"

    def test_capabilities(self):
        from quarry_core.connector import ConnectorCapability

        connector = TopoJSONConnector()
        caps = connector.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps


# ---------------------------------------------------------------------------
# JSON Extension Support
# ---------------------------------------------------------------------------


class TestTopoJSONJsonExtension:
    """Validate support for .json extension with TopoJSON content."""

    def test_json_extension_detected(self, tmp_path: Path):
        """Files with .json extension containing TopoJSON should work."""
        path = tmp_path / "data.json"

        objects = {
            "items": {
                "type": "GeometryCollection",
                "geometries": [
                    {"type": "Point", "coordinates": [1.0, 2.0]},
                ],
            }
        }
        _create_topojson_file(path, objects, arcs=[])

        connector = TopoJSONConnector()
        result = connector.materialize(str(path), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 1

    def test_json_extension_discover(self, tmp_path: Path):
        """Discover should find .json files in directory."""
        path = tmp_path / "data.json"
        objects = {"items": {"type": "GeometryCollection", "geometries": []}}
        _create_topojson_file(path, objects, arcs=[])

        connector = TopoJSONConnector()
        entries = connector.discover(str(tmp_path))

        names = {e.name for e in entries}
        assert "data" in names
