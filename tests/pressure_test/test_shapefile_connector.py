"""
Pressure test: ShapefileConnector.

Lane: connector

Validates Shapefile (.shp) materialization:
- source_ref parsing (local path, SourceRef)
- local eager: wrap in place, LOCAL_FILE backing, sidecar validation
- local lazy: metadata only, LAZY_HANDLE backing
- discover: find .shp files with sidecar completeness info
- metadata: read without materializing
- sidecar handling: missing .prj warning, .cpg encoding detection,
  missing .shx/.dbf raises error
- error handling: nonexistent files, non-shp files, corrupt files
"""

from __future__ import annotations

import pytest
from quarry_connectors.shapefile import ShapefileConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_shp_file(path, geometries, properties, crs="EPSG:4326", encoding=None):
    """Create a Shapefile using fiona with all sidecars.

    Args:
        path: Output file path (.shp)
        geometries: List of GeoJSON geometry dicts
        properties: Dict of column names to lists of values
        crs: CRS string or EPSG code
        encoding: Optional encoding to write to .cpg file
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

    with fiona.open(str(path), "w", driver="ESRI Shapefile", schema=schema, crs=crs_obj) as dst:
        for i, geom in enumerate(geometries):
            props = {k: v[i] for k, v in properties.items()}
            dst.write({"geometry": geom, "properties": props})

    # Write .cpg file if encoding specified
    if encoding:
        cpg_path = path.with_suffix(".cpg")
        cpg_path.write_text(encoding, encoding="utf-8")


@pytest.fixture()
def shp_file(tmp_path):
    """Create a Shapefile with 3 point geometries and all sidecars."""
    path = tmp_path / "points.shp"

    geometries = [
        {"type": "Point", "coordinates": [1.0, 2.0]},
        {"type": "Point", "coordinates": [3.0, 4.0]},
        {"type": "Point", "coordinates": [5.0, 6.0]},
    ]

    properties = {
        "name": ["alpha", "beta", "gamma"],
        "value": [1.0, 2.0, 3.0],
    }

    _create_shp_file(path, geometries, properties)
    return path


@pytest.fixture()
def shp_file_polygons(tmp_path):
    """Create a Shapefile with polygon geometries."""
    path = tmp_path / "zones.shp"

    geometries = [
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        {"type": "Polygon", "coordinates": [[[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]]},
    ]

    properties = {
        "zone_id": ["A", "B"],
        "area_km2": [100.0, 250.5],
    }

    _create_shp_file(path, geometries, properties)
    return path


@pytest.fixture()
def shp_file_no_prj(tmp_path):
    """Create a Shapefile without .prj file (missing CRS)."""
    path = tmp_path / "no_crs.shp"

    geometries = [
        {"type": "Point", "coordinates": [1.0, 2.0]},
    ]

    properties = {"name": ["single"]}

    _create_shp_file(path, geometries, properties)

    # Remove the .prj file that fiona created
    prj_path = path.with_suffix(".prj")
    if prj_path.exists():
        prj_path.unlink()

    return path


@pytest.fixture()
def shp_file_with_cpg(tmp_path):
    """Create a Shapefile with .cpg encoding file."""
    path = tmp_path / "encoded.shp"

    geometries = [
        {"type": "Point", "coordinates": [10.0, 20.0]},
    ]

    properties = {"label": ["test"]}

    _create_shp_file(path, geometries, properties, encoding="UTF-8")
    return path


@pytest.fixture()
def directory_with_shp(tmp_path, shp_file, shp_file_polygons):
    """Create a directory with multiple .shp files."""
    # Files are already in tmp_path from fixtures
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Local Materialization
# ---------------------------------------------------------------------------


class TestShapefileEagerLocal:
    """Validate eager materialization of local Shapefiles."""

    def test_eager_produces_vector(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.name == "points"

    def test_eager_produces_local_file_backing(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".shp")

    def test_eager_wrapped_local_strategy(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_feature_count(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_eager_extent(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(1.0)
        assert ymin == pytest.approx(2.0)
        assert xmax == pytest.approx(5.0)
        assert ymax == pytest.approx(6.0)

    def test_eager_crs(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path)

        assert result.artifact.spatial.crs is not None
        assert "4326" in result.artifact.spatial.crs

    def test_eager_content_hash_present(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path)

        assert result.artifact.lineage.params["source"] == "shapefile"
        assert result.artifact.lineage.params["path"] == str(shp_file)
        assert result.artifact.lineage.params["lazy"] is False
        assert "sidecars" in result.artifact.lineage.params

    def test_eager_metadata_schema(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path)

        assert "schema" in result.artifact.metadata
        assert result.artifact.metadata["schema"]["geometry"] == "Point"
        assert "name" in result.artifact.metadata["schema"]["properties"]
        assert "value" in result.artifact.metadata["schema"]["properties"]

    def test_eager_sidecar_inventory(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path)

        sidecars = result.artifact.metadata["sidecars"]
        assert sidecars["shx"] is True
        assert sidecars["dbf"] is True
        assert sidecars["prj"] is True
        # fiona may or may not create .cpg depending on version/settings
        # Just verify the key exists
        assert "cpg" in sidecars

    def test_eager_polygons(self, shp_file_polygons, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file_polygons), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 2
        assert result.artifact.metadata["schema"]["geometry"] == "Polygon"


# ---------------------------------------------------------------------------
# Lazy Local Materialization
# ---------------------------------------------------------------------------


class TestShapefileLazyLocal:
    """Validate lazy (metadata-only) materialization of local Shapefiles."""

    def test_lazy_backing_kind(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_feature_count(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path, lazy=True)

        assert result.artifact.spatial.feature_count == 3

    def test_lazy_extent(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path, lazy=True)

        extent = result.artifact.spatial.extent
        assert extent is not None
        assert extent[0] == pytest.approx(1.0)

    def test_lazy_lineage(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True


# ---------------------------------------------------------------------------
# Sidecar Handling
# ---------------------------------------------------------------------------


class TestShapefileSidecars:
    """Validate sidecar file handling."""

    def test_missing_prj_warning_in_metadata(self, shp_file_no_prj, tmp_path):
        """Missing .prj file sets crs=None and adds missing_prj flag."""
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file_no_prj), tmp_path)

        assert result.artifact.metadata["missing_prj"] is True
        assert result.artifact.metadata["sidecars"]["prj"] is False

    def test_missing_prj_crs_is_none(self, shp_file_no_prj, tmp_path):
        """Missing .prj results in None CRS."""
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file_no_prj), tmp_path)

        assert result.artifact.spatial.crs is None

    def test_cpg_encoding_detection(self, shp_file_with_cpg, tmp_path):
        """.cpg file is read for encoding."""
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file_with_cpg), tmp_path)

        assert result.artifact.metadata["encoding"] == "UTF-8"
        assert result.artifact.metadata["sidecars"]["cpg"] is True

    def test_cpg_encoding_in_lineage(self, shp_file_with_cpg, tmp_path):
        """Encoding is recorded in lineage params."""
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file_with_cpg), tmp_path)

        assert result.artifact.lineage.params["encoding"] == "UTF-8"

    def test_missing_shx_raises(self, tmp_path):
        """Missing .shx file raises MaterializeError."""
        # Create a shapefile
        path = tmp_path / "broken.shp"
        geometries = [{"type": "Point", "coordinates": [1.0, 2.0]}]
        properties = {"name": ["test"]}
        _create_shp_file(path, geometries, properties)

        # Remove .shx
        shx_path = path.with_suffix(".shx")
        shx_path.unlink()

        connector = ShapefileConnector()
        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize(str(path), tmp_path)

        assert ".shx" in str(exc_info.value)

    def test_missing_dbf_raises(self, tmp_path):
        """Missing .dbf file raises MaterializeError."""
        # Create a shapefile
        path = tmp_path / "broken.shp"
        geometries = [{"type": "Point", "coordinates": [1.0, 2.0]}]
        properties = {"name": ["test"]}
        _create_shp_file(path, geometries, properties)

        # Remove .dbf
        dbf_path = path.with_suffix(".dbf")
        dbf_path.unlink()

        connector = ShapefileConnector()
        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize(str(path), tmp_path)

        assert ".dbf" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestShapefileDiscover:
    """Validate file discovery with sidecar info."""

    def test_discover_lists_shp_files(self, directory_with_shp):
        connector = ShapefileConnector()
        entries = connector.discover(str(directory_with_shp))

        names = {e.name for e in entries}
        assert "points" in names
        assert "zones" in names

    def test_discover_source_refs(self, directory_with_shp):
        connector = ShapefileConnector()
        entries = connector.discover(str(directory_with_shp))

        for entry in entries:
            assert entry.source_ref.endswith(".shp") or entry.source_ref.endswith(".SHP")

    def test_discover_sidecar_info_in_metadata(self, directory_with_shp):
        connector = ShapefileConnector()
        entries = connector.discover(str(directory_with_shp))

        for entry in entries:
            assert "sidecars" in entry.metadata
            assert "complete" in entry.metadata
            assert "has_prj" in entry.metadata
            # All fixtures have complete sidecars
            assert entry.metadata["complete"] is True
            assert entry.metadata["has_prj"] is True

    def test_discover_with_dict_query(self, directory_with_shp):
        connector = ShapefileConnector()
        entries = connector.discover({"path": str(directory_with_shp)})

        assert len(entries) >= 2

    def test_discover_empty_directory(self, tmp_path):
        connector = ShapefileConnector()
        entries = connector.discover(str(tmp_path))

        assert entries == []

    def test_discover_nonexistent_directory_raises(self, tmp_path):
        connector = ShapefileConnector()
        nonexistent = tmp_path / "no_such_dir"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))

    def test_discover_file_raises(self, shp_file):
        connector = ShapefileConnector()

        with pytest.raises(MaterializeError):
            connector.discover(str(shp_file))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestShapefileMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_schema(self, shp_file):
        connector = ShapefileConnector()
        meta = connector.metadata(str(shp_file))

        assert "schema" in meta
        assert meta["schema"]["geometry"] == "Point"

    def test_metadata_feature_count(self, shp_file):
        connector = ShapefileConnector()
        meta = connector.metadata(str(shp_file))

        assert meta["feature_count"] == 3

    def test_metadata_crs(self, shp_file):
        connector = ShapefileConnector()
        meta = connector.metadata(str(shp_file))

        assert meta["crs"] is not None
        assert "4326" in meta["crs"]

    def test_metadata_extent(self, shp_file):
        connector = ShapefileConnector()
        meta = connector.metadata(str(shp_file))

        extent = meta["extent"]
        assert extent is not None
        assert len(extent) == 4

    def test_metadata_driver(self, shp_file):
        connector = ShapefileConnector()
        meta = connector.metadata(str(shp_file))

        assert meta["driver"] == "ESRI Shapefile"

    def test_metadata_sidecars(self, shp_file):
        connector = ShapefileConnector()
        meta = connector.metadata(str(shp_file))

        assert "sidecars" in meta
        assert meta["sidecars"]["shx"] is True
        assert meta["sidecars"]["dbf"] is True
        assert meta["sidecars"]["prj"] is True

    def test_metadata_encoding(self, shp_file_with_cpg):
        connector = ShapefileConnector()
        meta = connector.metadata(str(shp_file_with_cpg))

        assert meta["encoding"] == "UTF-8"

    def test_metadata_missing_prj(self, shp_file_no_prj):
        connector = ShapefileConnector()
        meta = connector.metadata(str(shp_file_no_prj))

        assert meta["missing_prj"] is True
        assert meta["sidecars"]["prj"] is False


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestShapefileErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path):
        connector = ShapefileConnector()
        nonexistent = tmp_path / "does_not_exist.shp"

        with pytest.raises(MaterializeError):
            connector.materialize(str(nonexistent), tmp_path)

    def test_non_shp_file_raises(self, tmp_path):
        connector = ShapefileConnector()
        bad_file = tmp_path / "not_shp.txt"
        bad_file.write_text("this is not a shapefile")

        with pytest.raises(MaterializeError):
            connector.materialize(str(bad_file), tmp_path)

    def test_metadata_nonexistent_file_raises(self, tmp_path):
        connector = ShapefileConnector()
        nonexistent = tmp_path / "does_not_exist.shp"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))

    def test_discover_no_path_raises(self):
        connector = ShapefileConnector()

        with pytest.raises(MaterializeError):
            connector.discover()


# ---------------------------------------------------------------------------
# SourceRef Support
# ---------------------------------------------------------------------------


class TestShapefileSourceRef:
    """Validate SourceRef handling."""

    def test_sourceref_local(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        ref = SourceRef.local(str(shp_file))
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_raw_string_path(self, shp_file, tmp_path):
        connector = ShapefileConnector()
        result = connector.materialize(str(shp_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestShapefileCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        connector = ShapefileConnector()
        assert connector.name == "shapefile"

    def test_capabilities(self):
        from quarry_core.connector import ConnectorCapability

        connector = ShapefileConnector()
        caps = connector.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps
