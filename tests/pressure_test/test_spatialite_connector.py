"""
Pressure test: SpatiaLiteConnector.

Lane: connector

Validates SpatiaLite file materialization:
- source_ref parsing (local path, ::layer_name syntax, SourceRef)
- local eager: wrap in place, LOCAL_FILE backing
- local lazy: metadata only, LAZY_HANDLE backing
- multi-layer support: layer selection via ::, default layer behavior
- discover: list layers in file, list files in directory
- metadata: read without materializing
- error handling: nonexistent files, nonexistent layers, bad files
"""

from __future__ import annotations

import pytest
from quarry_connectors.spatialite import SpatiaLiteConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_spatialite_file(path, layer_name, geometries, properties, crs="EPSG:4326"):
    """Create a SpatiaLite file with a single layer using fiona.

    Args:
        path: Output file path
        layer_name: Name of the layer to create
        geometries: List of GeoJSON geometry dicts
        properties: Dict of column names to lists of values
        crs: CRS string or EPSG code
    """
    import fiona
    from fiona.crs import CRS

    # Skip if SQLite driver not available
    if "SQLite" not in fiona.supported_drivers:
        pytest.skip("SQLite driver not available in fiona")

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

    with fiona.open(
        str(path), "w", driver="SQLite", schema=schema, crs=crs_obj, layer=layer_name
    ) as dst:
        for i, geom in enumerate(geometries):
            props = {k: v[i] for k, v in properties.items()}
            dst.write({"geometry": geom, "properties": props})


def _create_multi_layer_spatialite(path, layers_data, crs="EPSG:4326"):
    """Create a SpatiaLite file with multiple layers.

    Args:
        path: Output file path
        layers_data: Dict of layer_name -> {"geometries": [...], "properties": {...}}
        crs: CRS string or EPSG code
    """
    import fiona
    from fiona.crs import CRS

    # Skip if SQLite driver not available
    if "SQLite" not in fiona.supported_drivers:
        pytest.skip("SQLite driver not available in fiona")

    # Parse CRS
    if crs.startswith("EPSG:"):
        crs_obj = CRS.from_epsg(int(crs.split(":")[1]))
    else:
        crs_obj = CRS.from_epsg(4326)

    for layer_name, data in layers_data.items():
        geometries = data["geometries"]
        properties = data["properties"]

        if not geometries:
            continue

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

        with fiona.open(
            str(path), "w", driver="SQLite", schema=schema, crs=crs_obj, layer=layer_name
        ) as dst:
            for i, geom in enumerate(geometries):
                props = {k: v[i] for k, v in properties.items()}
                dst.write({"geometry": geom, "properties": props})


@pytest.fixture()
def spatialite_file(tmp_path):
    """Create a SpatiaLite file with a single layer containing 3 point geometries."""
    path = tmp_path / "points.sqlite"

    geometries = [
        {"type": "Point", "coordinates": [1.0, 2.0]},
        {"type": "Point", "coordinates": [3.0, 4.0]},
        {"type": "Point", "coordinates": [5.0, 6.0]},
    ]

    properties = {
        "name": ["alpha", "beta", "gamma"],
        "value": [1.0, 2.0, 3.0],
    }

    _create_spatialite_file(path, "points_layer", geometries, properties)
    return path


@pytest.fixture()
def spatialite_file_polygons(tmp_path):
    """Create a SpatiaLite file with polygon geometries."""
    path = tmp_path / "zones.sqlite"

    geometries = [
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        {"type": "Polygon", "coordinates": [[[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]]},
    ]

    properties = {
        "zone_id": ["A", "B"],
        "area_km2": [100.0, 250.5],
    }

    _create_spatialite_file(path, "zones_layer", geometries, properties)
    return path


@pytest.fixture()
def multi_layer_spatialite(tmp_path):
    """Create a SpatiaLite file with multiple layers."""
    path = tmp_path / "multi.sqlite"

    layers_data = {
        "cities": {
            "geometries": [
                {"type": "Point", "coordinates": [0.0, 0.0]},
                {"type": "Point", "coordinates": [1.0, 1.0]},
            ],
            "properties": {
                "name": ["Origin", "Destination"],
                "population": [1000000, 2000000],
            },
        },
        "roads": {
            "geometries": [
                {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                {"type": "LineString", "coordinates": [[1, 1], [2, 2]]},
                {"type": "LineString", "coordinates": [[2, 2], [3, 3]]},
            ],
            "properties": {
                "road_name": ["Main St", "Broadway", "Highway 1"],
                "length_km": [1.4, 1.4, 1.4],
            },
        },
    }

    _create_multi_layer_spatialite(path, layers_data)
    return path


@pytest.fixture()
def directory_with_spatialite(tmp_path, spatialite_file, spatialite_file_polygons):
    """Create a directory with multiple .sqlite files."""
    # Files are already in tmp_path from fixtures
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Local Materialization
# ---------------------------------------------------------------------------


class TestSpatiaLiteEagerLocal:
    """Validate eager materialization of local SpatiaLite files."""

    def test_eager_produces_vector(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert "points" in result.artifact.name

    def test_eager_produces_local_file_backing(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".sqlite")

    def test_eager_wrapped_local_strategy(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_feature_count(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_eager_extent(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(1.0)
        assert ymin == pytest.approx(2.0)
        assert xmax == pytest.approx(5.0)
        assert ymax == pytest.approx(6.0)

    def test_eager_crs(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path)

        assert result.artifact.spatial.crs is not None
        assert "4326" in result.artifact.spatial.crs

    def test_eager_content_hash_present(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path)

        assert result.artifact.lineage.params["source"] == "spatialite"
        assert result.artifact.lineage.params["path"] == str(spatialite_file)
        assert result.artifact.lineage.params["lazy"] is False

    def test_eager_metadata_schema(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path)

        assert "schema" in result.artifact.metadata
        assert result.artifact.metadata["schema"]["geometry"] == "Point"
        assert "name" in result.artifact.metadata["schema"]["properties"]
        assert "value" in result.artifact.metadata["schema"]["properties"]

    def test_eager_polygons(self, spatialite_file_polygons, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file_polygons), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 2
        assert result.artifact.metadata["schema"]["geometry"] == "Polygon"


# ---------------------------------------------------------------------------
# Lazy Local Materialization
# ---------------------------------------------------------------------------


class TestSpatiaLiteLazyLocal:
    """Validate lazy (metadata-only) materialization of local SpatiaLite files."""

    def test_lazy_backing_kind(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_feature_count(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path, lazy=True)

        assert result.artifact.spatial.feature_count == 3

    def test_lazy_extent(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path, lazy=True)

        extent = result.artifact.spatial.extent
        assert extent is not None
        assert extent[0] == pytest.approx(1.0)

    def test_lazy_backing_uri(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path, lazy=True)

        assert result.artifact.backing.uri.startswith(str(spatialite_file))
        assert "::" in result.artifact.backing.uri

    def test_lazy_lineage(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True


# ---------------------------------------------------------------------------
# Multi-Layer Support
# ---------------------------------------------------------------------------


class TestSpatiaLiteMultiLayer:
    """Validate multi-layer SpatiaLite handling."""

    def test_default_layer_is_first(self, multi_layer_spatialite, tmp_path):
        """When no ::layer specified, use first layer."""
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(multi_layer_spatialite), tmp_path)

        # First layer is "cities" (alphabetically or creation order)
        assert result.artifact.metadata["layer_name"] in ["cities", "roads"]
        assert result.artifact.spatial.feature_count > 0

    def test_layer_selection_via_separator(self, multi_layer_spatialite, tmp_path):
        """Select specific layer using :: separator."""
        connector = SpatiaLiteConnector()
        result = connector.materialize(f"{multi_layer_spatialite}::roads", tmp_path)

        assert result.artifact.metadata["layer_name"] == "roads"
        assert result.artifact.spatial.feature_count == 3
        assert result.artifact.metadata["schema"]["geometry"] == "LineString"

    def test_layer_selection_cities(self, multi_layer_spatialite, tmp_path):
        """Select cities layer using :: separator."""
        connector = SpatiaLiteConnector()
        result = connector.materialize(f"{multi_layer_spatialite}::cities", tmp_path)

        assert result.artifact.metadata["layer_name"] == "cities"
        assert result.artifact.spatial.feature_count == 2
        assert result.artifact.metadata["schema"]["geometry"] == "Point"

    def test_all_layers_in_metadata(self, multi_layer_spatialite, tmp_path):
        """Metadata contains list of all layers."""
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(multi_layer_spatialite), tmp_path)

        all_layers = result.artifact.metadata["all_layers"]
        assert "cities" in all_layers
        assert "roads" in all_layers

    def test_lazy_with_layer_selection(self, multi_layer_spatialite, tmp_path):
        """Lazy mode works with layer selection."""
        connector = SpatiaLiteConnector()
        result = connector.materialize(f"{multi_layer_spatialite}::cities", tmp_path, lazy=True)

        assert result.artifact.metadata["layer_name"] == "cities"
        assert result.strategy == "lazy_handle"


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestSpatiaLiteDiscover:
    """Validate discovery functionality."""

    def test_discover_lists_layers_in_file(self, multi_layer_spatialite):
        """When query points to a .sqlite file, list layers within it."""
        connector = SpatiaLiteConnector()
        entries = connector.discover(str(multi_layer_spatialite))

        names = {e.name for e in entries}
        assert "cities" in names
        assert "roads" in names

    def test_discover_layer_source_refs(self, multi_layer_spatialite):
        """Layer entries have source_ref with :: separator."""
        connector = SpatiaLiteConnector()
        entries = connector.discover(str(multi_layer_spatialite))

        for entry in entries:
            assert "::" in entry.source_ref
            assert entry.source_ref.endswith(entry.name)

    def test_discover_lists_sqlite_files_in_directory(self, directory_with_spatialite):
        """When query points to a directory, list .sqlite files."""
        connector = SpatiaLiteConnector()
        entries = connector.discover(str(directory_with_spatialite))

        names = {e.name for e in entries}
        assert "points" in names
        assert "zones" in names

    def test_discover_file_source_refs(self, directory_with_spatialite):
        """File entries have source_ref without :: separator."""
        connector = SpatiaLiteConnector()
        entries = connector.discover(str(directory_with_spatialite))

        for entry in entries:
            assert entry.source_ref.endswith(".sqlite") or entry.source_ref.endswith(".SQLITE")

    def test_discover_with_dict_query(self, directory_with_spatialite):
        """Discover accepts dict query with path key."""
        connector = SpatiaLiteConnector()
        entries = connector.discover({"path": str(directory_with_spatialite)})

        assert len(entries) >= 2

    def test_discover_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        connector = SpatiaLiteConnector()
        entries = connector.discover(str(tmp_path))

        assert entries == []

    def test_discover_nonexistent_path_raises(self, tmp_path):
        """Nonexistent path raises MaterializeError."""
        connector = SpatiaLiteConnector()
        nonexistent = tmp_path / "no_such_path"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestSpatiaLiteMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_schema(self, spatialite_file):
        connector = SpatiaLiteConnector()
        meta = connector.metadata(str(spatialite_file))

        assert "schema" in meta
        assert meta["schema"]["geometry"] == "Point"

    def test_metadata_feature_count(self, spatialite_file):
        connector = SpatiaLiteConnector()
        meta = connector.metadata(str(spatialite_file))

        assert meta["feature_count"] == 3

    def test_metadata_crs(self, spatialite_file):
        connector = SpatiaLiteConnector()
        meta = connector.metadata(str(spatialite_file))

        assert meta["crs"] is not None
        assert "4326" in meta["crs"]

    def test_metadata_extent(self, spatialite_file):
        connector = SpatiaLiteConnector()
        meta = connector.metadata(str(spatialite_file))

        extent = meta["extent"]
        assert extent is not None
        assert len(extent) == 4

    def test_metadata_driver(self, spatialite_file):
        connector = SpatiaLiteConnector()
        meta = connector.metadata(str(spatialite_file))

        assert meta["driver"] == "SQLite"

    def test_metadata_layer_name(self, spatialite_file):
        connector = SpatiaLiteConnector()
        meta = connector.metadata(str(spatialite_file))

        assert "layer_name" in meta
        assert meta["layer_name"] is not None

    def test_metadata_all_layers(self, multi_layer_spatialite):
        connector = SpatiaLiteConnector()
        meta = connector.metadata(str(multi_layer_spatialite))

        assert "all_layers" in meta
        assert "cities" in meta["all_layers"]
        assert "roads" in meta["all_layers"]

    def test_metadata_with_layer_selection(self, multi_layer_spatialite):
        connector = SpatiaLiteConnector()
        meta = connector.metadata(f"{multi_layer_spatialite}::roads")

        assert meta["layer_name"] == "roads"
        assert meta["schema"]["geometry"] == "LineString"


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestSpatiaLiteErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path):
        connector = SpatiaLiteConnector()
        nonexistent = tmp_path / "does_not_exist.sqlite"

        with pytest.raises(MaterializeError):
            connector.materialize(str(nonexistent), tmp_path)

    def test_nonexistent_layer_raises(self, multi_layer_spatialite, tmp_path):
        connector = SpatiaLiteConnector()

        with pytest.raises(MaterializeError):
            connector.materialize(f"{multi_layer_spatialite}::nonexistent_layer", tmp_path)

    def test_non_spatialite_file_raises(self, tmp_path):
        connector = SpatiaLiteConnector()
        bad_file = tmp_path / "not_spatialite.txt"
        bad_file.write_text("this is not a spatialite file")

        with pytest.raises(MaterializeError):
            connector.materialize(str(bad_file), tmp_path)

    def test_metadata_nonexistent_file_raises(self, tmp_path):
        connector = SpatiaLiteConnector()
        nonexistent = tmp_path / "does_not_exist.sqlite"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))

    def test_metadata_nonexistent_layer_raises(self, multi_layer_spatialite):
        connector = SpatiaLiteConnector()

        with pytest.raises(MaterializeError):
            connector.metadata(f"{multi_layer_spatialite}::nonexistent_layer")

    def test_discover_no_path_raises(self):
        connector = SpatiaLiteConnector()

        with pytest.raises(MaterializeError):
            connector.discover()


# ---------------------------------------------------------------------------
# SourceRef Support
# ---------------------------------------------------------------------------


class TestSpatiaLiteSourceRef:
    """Validate SourceRef handling."""

    def test_sourceref_local(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        ref = SourceRef.local(str(spatialite_file))
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_raw_string_path(self, spatialite_file, tmp_path):
        connector = SpatiaLiteConnector()
        result = connector.materialize(str(spatialite_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_sourceref_with_layer(self, multi_layer_spatialite, tmp_path):
        """SourceRef with ::layer in raw string."""
        connector = SpatiaLiteConnector()
        ref = SourceRef.local(f"{multi_layer_spatialite}::cities")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.metadata["layer_name"] == "cities"


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestSpatiaLiteCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        connector = SpatiaLiteConnector()
        assert connector.name == "spatialite"

    def test_capabilities(self):
        from quarry_core.connector import ConnectorCapability

        connector = SpatiaLiteConnector()
        caps = connector.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps


# ---------------------------------------------------------------------------
# File Extension Support
# ---------------------------------------------------------------------------


class TestSpatiaLiteExtensions:
    """Validate support for different SpatiaLite file extensions."""

    def test_db_extension(self, tmp_path):
        """Support .db extension."""
        import fiona
        from fiona.crs import CRS

        if "SQLite" not in fiona.supported_drivers:
            pytest.skip("SQLite driver not available in fiona")

        path = tmp_path / "data.db"
        crs = CRS.from_epsg(4326)
        schema = {"geometry": "Point", "properties": {"name": "str"}}

        with fiona.open(
            str(path), "w", driver="SQLite", schema=schema, crs=crs, layer="test"
        ) as dst:
            dst.write(
                {
                    "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                    "properties": {"name": "A"},
                }
            )

        connector = SpatiaLiteConnector()
        result = connector.materialize(str(path), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 1

    def test_spatialite_extension(self, tmp_path):
        """Support .spatialite extension."""
        import fiona
        from fiona.crs import CRS

        if "SQLite" not in fiona.supported_drivers:
            pytest.skip("SQLite driver not available in fiona")

        path = tmp_path / "data.spatialite"
        crs = CRS.from_epsg(4326)
        schema = {"geometry": "Point", "properties": {"name": "str"}}

        with fiona.open(
            str(path), "w", driver="SQLite", schema=schema, crs=crs, layer="test"
        ) as dst:
            dst.write(
                {
                    "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                    "properties": {"name": "A"},
                }
            )

        connector = SpatiaLiteConnector()
        result = connector.materialize(str(path), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 1
