"""
Pressure test: GeoPackageConnector.

Lane: connector

Validates GeoPackage file materialization:
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
from quarry_connectors.geopackage import GeoPackageConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_gpkg_file(path, layer_name, geometries, properties, crs="EPSG:4326"):
    """Create a GeoPackage file with a single layer using fiona.

    Args:
        path: Output file path
        layer_name: Name of the layer to create
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

    with fiona.open(
        str(path), "w", driver="GPKG", schema=schema, crs=crs_obj, layer=layer_name
    ) as dst:
        for i, geom in enumerate(geometries):
            props = {k: v[i] for k, v in properties.items()}
            dst.write({"geometry": geom, "properties": props})


def _create_multi_layer_gpkg(path, layers_data, crs="EPSG:4326"):
    """Create a GeoPackage file with multiple layers.

    Args:
        path: Output file path
        layers_data: Dict of layer_name -> {"geometries": [...], "properties": {...}}
        crs: CRS string or EPSG code
    """
    import fiona
    from fiona.crs import CRS

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
            str(path), "w", driver="GPKG", schema=schema, crs=crs_obj, layer=layer_name
        ) as dst:
            for i, geom in enumerate(geometries):
                props = {k: v[i] for k, v in properties.items()}
                dst.write({"geometry": geom, "properties": props})


@pytest.fixture()
def gpkg_file(tmp_path):
    """Create a GeoPackage file with a single layer containing 3 point geometries."""
    path = tmp_path / "points.gpkg"

    geometries = [
        {"type": "Point", "coordinates": [1.0, 2.0]},
        {"type": "Point", "coordinates": [3.0, 4.0]},
        {"type": "Point", "coordinates": [5.0, 6.0]},
    ]

    properties = {
        "name": ["alpha", "beta", "gamma"],
        "value": [1.0, 2.0, 3.0],
    }

    _create_gpkg_file(path, "points_layer", geometries, properties)
    return path


@pytest.fixture()
def gpkg_file_polygons(tmp_path):
    """Create a GeoPackage file with polygon geometries."""
    path = tmp_path / "zones.gpkg"

    geometries = [
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        {"type": "Polygon", "coordinates": [[[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]]},
    ]

    properties = {
        "zone_id": ["A", "B"],
        "area_km2": [100.0, 250.5],
    }

    _create_gpkg_file(path, "zones_layer", geometries, properties)
    return path


@pytest.fixture()
def multi_layer_gpkg(tmp_path):
    """Create a GeoPackage file with multiple layers."""
    path = tmp_path / "multi.gpkg"

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

    _create_multi_layer_gpkg(path, layers_data)
    return path


@pytest.fixture()
def directory_with_gpkg(tmp_path, gpkg_file, gpkg_file_polygons):
    """Create a directory with multiple .gpkg files."""
    # Files are already in tmp_path from fixtures
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Local Materialization
# ---------------------------------------------------------------------------


class TestGeoPackageEagerLocal:
    """Validate eager materialization of local GeoPackage files."""

    def test_eager_produces_vector(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert "points" in result.artifact.name

    def test_eager_produces_local_file_backing(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".gpkg")

    def test_eager_wrapped_local_strategy(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_feature_count(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_eager_extent(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(1.0)
        assert ymin == pytest.approx(2.0)
        assert xmax == pytest.approx(5.0)
        assert ymax == pytest.approx(6.0)

    def test_eager_crs(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path)

        assert result.artifact.spatial.crs is not None
        assert "4326" in result.artifact.spatial.crs

    def test_eager_content_hash_present(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path)

        assert result.artifact.lineage.params["source"] == "geopackage"
        assert result.artifact.lineage.params["path"] == str(gpkg_file)
        assert result.artifact.lineage.params["lazy"] is False

    def test_eager_metadata_schema(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path)

        assert "schema" in result.artifact.metadata
        assert result.artifact.metadata["schema"]["geometry"] == "Point"
        assert "name" in result.artifact.metadata["schema"]["properties"]
        assert "value" in result.artifact.metadata["schema"]["properties"]

    def test_eager_polygons(self, gpkg_file_polygons, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file_polygons), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 2
        assert result.artifact.metadata["schema"]["geometry"] == "Polygon"


# ---------------------------------------------------------------------------
# Lazy Local Materialization
# ---------------------------------------------------------------------------


class TestGeoPackageLazyLocal:
    """Validate lazy (metadata-only) materialization of local GeoPackage files."""

    def test_lazy_backing_kind(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_feature_count(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path, lazy=True)

        assert result.artifact.spatial.feature_count == 3

    def test_lazy_extent(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path, lazy=True)

        extent = result.artifact.spatial.extent
        assert extent is not None
        assert extent[0] == pytest.approx(1.0)

    def test_lazy_backing_uri(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path, lazy=True)

        assert result.artifact.backing.uri.startswith(str(gpkg_file))
        assert "::" in result.artifact.backing.uri

    def test_lazy_lineage(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True


# ---------------------------------------------------------------------------
# Multi-Layer Support
# ---------------------------------------------------------------------------


class TestGeoPackageMultiLayer:
    """Validate multi-layer GeoPackage handling."""

    def test_default_layer_is_first(self, multi_layer_gpkg, tmp_path):
        """When no ::layer specified, use first layer."""
        connector = GeoPackageConnector()
        result = connector.materialize(str(multi_layer_gpkg), tmp_path)

        # First layer is "cities" (alphabetically or creation order)
        assert result.artifact.metadata["layer_name"] in ["cities", "roads"]
        assert result.artifact.spatial.feature_count > 0

    def test_layer_selection_via_separator(self, multi_layer_gpkg, tmp_path):
        """Select specific layer using :: separator."""
        connector = GeoPackageConnector()
        result = connector.materialize(f"{multi_layer_gpkg}::roads", tmp_path)

        assert result.artifact.metadata["layer_name"] == "roads"
        assert result.artifact.spatial.feature_count == 3
        assert result.artifact.metadata["schema"]["geometry"] == "LineString"

    def test_layer_selection_cities(self, multi_layer_gpkg, tmp_path):
        """Select cities layer using :: separator."""
        connector = GeoPackageConnector()
        result = connector.materialize(f"{multi_layer_gpkg}::cities", tmp_path)

        assert result.artifact.metadata["layer_name"] == "cities"
        assert result.artifact.spatial.feature_count == 2
        assert result.artifact.metadata["schema"]["geometry"] == "Point"

    def test_all_layers_in_metadata(self, multi_layer_gpkg, tmp_path):
        """Metadata contains list of all layers."""
        connector = GeoPackageConnector()
        result = connector.materialize(str(multi_layer_gpkg), tmp_path)

        all_layers = result.artifact.metadata["all_layers"]
        assert "cities" in all_layers
        assert "roads" in all_layers

    def test_lazy_with_layer_selection(self, multi_layer_gpkg, tmp_path):
        """Lazy mode works with layer selection."""
        connector = GeoPackageConnector()
        result = connector.materialize(f"{multi_layer_gpkg}::cities", tmp_path, lazy=True)

        assert result.artifact.metadata["layer_name"] == "cities"
        assert result.strategy == "lazy_handle"


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestGeoPackageDiscover:
    """Validate discovery functionality."""

    def test_discover_lists_layers_in_file(self, multi_layer_gpkg):
        """When query points to a .gpkg file, list layers within it."""
        connector = GeoPackageConnector()
        entries = connector.discover(str(multi_layer_gpkg))

        names = {e.name for e in entries}
        assert "cities" in names
        assert "roads" in names

    def test_discover_layer_source_refs(self, multi_layer_gpkg):
        """Layer entries have source_ref with :: separator."""
        connector = GeoPackageConnector()
        entries = connector.discover(str(multi_layer_gpkg))

        for entry in entries:
            assert "::" in entry.source_ref
            assert entry.source_ref.endswith(entry.name)

    def test_discover_lists_gpkg_files_in_directory(self, directory_with_gpkg):
        """When query points to a directory, list .gpkg files."""
        connector = GeoPackageConnector()
        entries = connector.discover(str(directory_with_gpkg))

        names = {e.name for e in entries}
        assert "points" in names
        assert "zones" in names

    def test_discover_file_source_refs(self, directory_with_gpkg):
        """File entries have source_ref without :: separator."""
        connector = GeoPackageConnector()
        entries = connector.discover(str(directory_with_gpkg))

        for entry in entries:
            assert entry.source_ref.endswith(".gpkg") or entry.source_ref.endswith(".GPKG")

    def test_discover_with_dict_query(self, directory_with_gpkg):
        """Discover accepts dict query with path key."""
        connector = GeoPackageConnector()
        entries = connector.discover({"path": str(directory_with_gpkg)})

        assert len(entries) >= 2

    def test_discover_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        connector = GeoPackageConnector()
        entries = connector.discover(str(tmp_path))

        assert entries == []

    def test_discover_nonexistent_path_raises(self, tmp_path):
        """Nonexistent path raises MaterializeError."""
        connector = GeoPackageConnector()
        nonexistent = tmp_path / "no_such_path"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestGeoPackageMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_schema(self, gpkg_file):
        connector = GeoPackageConnector()
        meta = connector.metadata(str(gpkg_file))

        assert "schema" in meta
        assert meta["schema"]["geometry"] == "Point"

    def test_metadata_feature_count(self, gpkg_file):
        connector = GeoPackageConnector()
        meta = connector.metadata(str(gpkg_file))

        assert meta["feature_count"] == 3

    def test_metadata_crs(self, gpkg_file):
        connector = GeoPackageConnector()
        meta = connector.metadata(str(gpkg_file))

        assert meta["crs"] is not None
        assert "4326" in meta["crs"]

    def test_metadata_extent(self, gpkg_file):
        connector = GeoPackageConnector()
        meta = connector.metadata(str(gpkg_file))

        extent = meta["extent"]
        assert extent is not None
        assert len(extent) == 4

    def test_metadata_driver(self, gpkg_file):
        connector = GeoPackageConnector()
        meta = connector.metadata(str(gpkg_file))

        assert meta["driver"] == "GPKG"

    def test_metadata_layer_name(self, gpkg_file):
        connector = GeoPackageConnector()
        meta = connector.metadata(str(gpkg_file))

        assert "layer_name" in meta
        assert meta["layer_name"] is not None

    def test_metadata_all_layers(self, multi_layer_gpkg):
        connector = GeoPackageConnector()
        meta = connector.metadata(str(multi_layer_gpkg))

        assert "all_layers" in meta
        assert "cities" in meta["all_layers"]
        assert "roads" in meta["all_layers"]

    def test_metadata_with_layer_selection(self, multi_layer_gpkg):
        connector = GeoPackageConnector()
        meta = connector.metadata(f"{multi_layer_gpkg}::roads")

        assert meta["layer_name"] == "roads"
        assert meta["schema"]["geometry"] == "LineString"


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestGeoPackageErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path):
        connector = GeoPackageConnector()
        nonexistent = tmp_path / "does_not_exist.gpkg"

        with pytest.raises(MaterializeError):
            connector.materialize(str(nonexistent), tmp_path)

    def test_nonexistent_layer_raises(self, multi_layer_gpkg, tmp_path):
        connector = GeoPackageConnector()

        with pytest.raises(MaterializeError):
            connector.materialize(f"{multi_layer_gpkg}::nonexistent_layer", tmp_path)

    def test_non_gpkg_file_raises(self, tmp_path):
        connector = GeoPackageConnector()
        bad_file = tmp_path / "not_gpkg.txt"
        bad_file.write_text("this is not a geopackage file")

        with pytest.raises(MaterializeError):
            connector.materialize(str(bad_file), tmp_path)

    def test_metadata_nonexistent_file_raises(self, tmp_path):
        connector = GeoPackageConnector()
        nonexistent = tmp_path / "does_not_exist.gpkg"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))

    def test_metadata_nonexistent_layer_raises(self, multi_layer_gpkg):
        connector = GeoPackageConnector()

        with pytest.raises(MaterializeError):
            connector.metadata(f"{multi_layer_gpkg}::nonexistent_layer")

    def test_discover_no_path_raises(self):
        connector = GeoPackageConnector()

        with pytest.raises(MaterializeError):
            connector.discover()


# ---------------------------------------------------------------------------
# SourceRef Support
# ---------------------------------------------------------------------------


class TestGeoPackageSourceRef:
    """Validate SourceRef handling."""

    def test_sourceref_local(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        ref = SourceRef.local(str(gpkg_file))
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_raw_string_path(self, gpkg_file, tmp_path):
        connector = GeoPackageConnector()
        result = connector.materialize(str(gpkg_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_sourceref_with_layer(self, multi_layer_gpkg, tmp_path):
        """SourceRef with ::layer in raw string."""
        connector = GeoPackageConnector()
        ref = SourceRef.local(f"{multi_layer_gpkg}::cities")
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.metadata["layer_name"] == "cities"


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestGeoPackageCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        connector = GeoPackageConnector()
        assert connector.name == "geopackage"

    def test_capabilities(self):
        from quarry_core.connector import ConnectorCapability

        connector = GeoPackageConnector()
        caps = connector.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps
