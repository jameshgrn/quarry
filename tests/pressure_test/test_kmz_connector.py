"""
Pressure test: KMZConnector.

Lane: connector

Validates KMZ file materialization:
- source_ref parsing (local path)
- local eager: extract KML, LOCAL_FILE backing pointing to extracted .kml
- local lazy: metadata only, LAZY_HANDLE backing
- discover: list .kmz files in directory
- metadata: read without materializing
- extraction: doc.kml found, fallback to other .kml, no .kml raises error
- error handling: nonexistent files, not a zip, no kml inside, corrupt zip
- CRS is always WGS84 (EPSG:4326) per KML spec
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from quarry_connectors.kmz import KMZConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# KML driver availability check
# ---------------------------------------------------------------------------


def _kml_driver_available() -> bool:
    """Check if fiona has a KML driver available."""
    try:
        import fiona

        return "LIBKML" in fiona.supported_drivers or "KML" in fiona.supported_drivers
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Minimal KML with point and polygon placemarks
MINIMAL_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Test</name>
    <Placemark>
      <name>Point1</name>
      <Point><coordinates>-122.3,47.5,0</coordinates></Point>
    </Placemark>
    <Placemark>
      <name>Area1</name>
      <Polygon>
        <outerBoundaryIs><LinearRing>
          <coordinates>
            -122.3,47.5,0 -122.2,47.5,0 -122.2,47.6,0 -122.3,47.6,0 -122.3,47.5,0
          </coordinates>
        </LinearRing></outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>"""


# KML with only point placemarks
POINTS_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Points</name>
    <Placemark>
      <name>Point1</name>
      <Point><coordinates>-122.3,47.5,0</coordinates></Point>
    </Placemark>
    <Placemark>
      <name>Point2</name>
      <Point><coordinates>-122.2,47.6,0</coordinates></Point>
    </Placemark>
    <Placemark>
      <name>Point3</name>
      <Point><coordinates>-122.1,47.7,0</coordinates></Point>
    </Placemark>
  </Document>
</kml>"""


# KML with only polygon placemarks
POLYGONS_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Polygons</name>
    <Placemark>
      <name>Poly1</name>
      <Polygon>
        <outerBoundaryIs><LinearRing>
          <coordinates>0,0,0 1,0,0 1,1,0 0,1,0 0,0,0</coordinates>
        </LinearRing></outerBoundaryIs>
      </Polygon>
    </Placemark>
    <Placemark>
      <name>Poly2</name>
      <Polygon>
        <outerBoundaryIs><LinearRing>
          <coordinates>2,2,0 4,2,0 4,4,0 2,4,0 2,2,0</coordinates>
        </LinearRing></outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>"""


def _create_kmz_file(path: Path, kml_content: str, kml_filename: str = "doc.kml") -> Path:
    """Create a KMZ file by zipping KML content.

    Args:
        path: Output KMZ file path
        kml_content: KML XML content as string
        kml_filename: Name of the KML file inside the zip (default: doc.kml)

    Returns:
        Path to the created KMZ file
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(kml_filename, kml_content)
    return path


@pytest.fixture()
def kmz_file_points(tmp_path: Path) -> Path:
    """Create a KMZ file with point placemarks."""
    path = tmp_path / "points.kmz"
    _create_kmz_file(path, POINTS_KML, "doc.kml")
    return path


@pytest.fixture()
def kmz_file_polygons(tmp_path: Path) -> Path:
    """Create a KMZ file with polygon placemarks."""
    path = tmp_path / "polygons.kmz"
    _create_kmz_file(path, POLYGONS_KML, "doc.kml")
    return path


@pytest.fixture()
def kmz_file_mixed(tmp_path: Path) -> Path:
    """Create a KMZ file with mixed geometry types."""
    path = tmp_path / "mixed.kmz"
    _create_kmz_file(path, MINIMAL_KML, "doc.kml")
    return path


@pytest.fixture()
def kmz_file_alt_name(tmp_path: Path) -> Path:
    """Create a KMZ file with KML named differently than doc.kml."""
    path = tmp_path / "alt_name.kmz"
    _create_kmz_file(path, POINTS_KML, "layer.kml")
    return path


@pytest.fixture()
def directory_with_kmz(tmp_path: Path, kmz_file_points: Path, kmz_file_polygons: Path) -> Path:
    """Create a directory with multiple KMZ files."""
    # Files are already in tmp_path from fixtures
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Local Materialization
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _kml_driver_available(), reason="KML/LIBKML driver not available")
class TestKMZEagerLocal:
    """Validate eager materialization of local KMZ files."""

    def test_eager_produces_vector(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.name == "points"

    def test_eager_produces_local_file_backing(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".kml")
        # Backing should point to extracted KML, not original KMZ
        assert "_extracted" in result.artifact.backing.uri

    def test_eager_normalized_strategy(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        assert result.strategy == "normalized"

    def test_eager_feature_count(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_eager_extent(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(-122.3, abs=0.1)
        assert ymin == pytest.approx(47.5, abs=0.1)
        assert xmax == pytest.approx(-122.1, abs=0.1)
        assert ymax == pytest.approx(47.7, abs=0.1)

    def test_eager_crs_is_wgs84(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        assert result.artifact.spatial.crs == "EPSG:4326"

    def test_eager_content_hash_present(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        assert result.artifact.lineage.params["source"] == "kmz"
        assert result.artifact.lineage.params["original_kmz_path"] == str(kmz_file_points.resolve())
        assert "extracted_kml_path" in result.artifact.lineage.params
        assert result.artifact.lineage.params["lazy"] is False

    def test_eager_metadata_schema(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        assert "schema" in result.artifact.metadata
        assert "original_kmz" in result.artifact.metadata

    def test_eager_polygons(self, kmz_file_polygons: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_polygons), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 2

    def test_eager_extracted_file_exists(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        extracted_path = Path(result.artifact.backing.uri)
        assert extracted_path.exists()
        assert extracted_path.is_file()


# ---------------------------------------------------------------------------
# Lazy Local Materialization
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _kml_driver_available(), reason="KML/LIBKML driver not available")
class TestKMZLazyLocal:
    """Validate lazy (metadata-only) materialization of local KMZ files."""

    def test_lazy_backing_kind(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_feature_count(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path, lazy=True)

        assert result.artifact.spatial.feature_count == 3

    def test_lazy_extent(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path, lazy=True)

        extent = result.artifact.spatial.extent
        assert extent is not None
        assert len(extent) == 4

    def test_lazy_backing_uri(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path, lazy=True)

        # URI should point to extracted KML path
        assert result.artifact.backing.uri.endswith(".kml")
        assert "_extracted" in result.artifact.backing.uri

    def test_lazy_lineage(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True

    def test_lazy_crs_is_wgs84(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path, lazy=True)

        assert result.artifact.spatial.crs == "EPSG:4326"


# ---------------------------------------------------------------------------
# Extraction Behavior
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _kml_driver_available(), reason="KML/LIBKML driver not available")
class TestKMZExtraction:
    """Validate KMZ extraction behavior."""

    def test_extracts_doc_kml(self, kmz_file_points: Path, tmp_path: Path):
        """KMZ with doc.kml should extract it."""
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        extracted_path = Path(result.artifact.backing.uri)
        assert extracted_path.name == "doc.kml"

    def test_fallback_to_other_kml_name(self, kmz_file_alt_name: Path, tmp_path: Path):
        """KMZ without doc.kml should fallback to any .kml file."""
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_alt_name), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 3

    def test_extraction_directory_naming(self, kmz_file_points: Path, tmp_path: Path):
        """Extracted KML should be in {stem}_extracted/ directory."""
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        extracted_path = Path(result.artifact.backing.uri)
        parent_dir = extracted_path.parent
        assert parent_dir.name == "points_extracted"


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestKMZDiscover:
    """Validate file discovery."""

    def test_discover_lists_kmz_files(self, directory_with_kmz: Path):
        connector = KMZConnector()
        entries = connector.discover(str(directory_with_kmz))

        names = {e.name for e in entries}
        assert "points" in names
        assert "polygons" in names

    def test_discover_source_refs(self, directory_with_kmz: Path):
        connector = KMZConnector()
        entries = connector.discover(str(directory_with_kmz))

        for entry in entries:
            assert entry.source_ref.endswith(".kmz") or entry.source_ref.endswith(".KMZ")

    def test_discover_with_dict_query(self, directory_with_kmz: Path):
        connector = KMZConnector()
        entries = connector.discover({"path": str(directory_with_kmz)})

        assert len(entries) >= 2

    def test_discover_empty_directory(self, tmp_path: Path):
        connector = KMZConnector()
        entries = connector.discover(str(tmp_path))

        assert entries == []

    def test_discover_nonexistent_directory_raises(self, tmp_path: Path):
        connector = KMZConnector()
        nonexistent = tmp_path / "no_such_dir"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _kml_driver_available(), reason="KML/LIBKML driver not available")
class TestKMZMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_schema(self, kmz_file_points: Path):
        connector = KMZConnector()
        meta = connector.metadata(str(kmz_file_points))

        assert "schema" in meta

    def test_metadata_feature_count(self, kmz_file_points: Path):
        connector = KMZConnector()
        meta = connector.metadata(str(kmz_file_points))

        assert meta["feature_count"] == 3

    def test_metadata_crs_is_wgs84(self, kmz_file_points: Path):
        connector = KMZConnector()
        meta = connector.metadata(str(kmz_file_points))

        assert meta["crs"] == "EPSG:4326"

    def test_metadata_extent(self, kmz_file_points: Path):
        connector = KMZConnector()
        meta = connector.metadata(str(kmz_file_points))

        extent = meta["extent"]
        assert extent is not None
        assert len(extent) == 4

    def test_metadata_driver(self, kmz_file_points: Path):
        connector = KMZConnector()
        meta = connector.metadata(str(kmz_file_points))

        # Driver should be KML or LIBKML
        assert meta["driver"] in ("KML", "LIBKML")

    def test_metadata_kml_filename(self, kmz_file_points: Path):
        connector = KMZConnector()
        meta = connector.metadata(str(kmz_file_points))

        assert meta["kml_file_in_archive"] == "doc.kml"

    def test_metadata_original_kmz_path(self, kmz_file_points: Path):
        connector = KMZConnector()
        meta = connector.metadata(str(kmz_file_points))

        assert meta["original_kmz"] == str(kmz_file_points)


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestKMZErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path: Path):
        connector = KMZConnector()
        nonexistent = tmp_path / "does_not_exist.kmz"

        with pytest.raises(MaterializeError):
            connector.materialize(str(nonexistent), tmp_path)

    def test_not_a_zip_raises(self, tmp_path: Path):
        connector = KMZConnector()
        bad_file = tmp_path / "not_kmz.kmz"
        bad_file.write_text("this is not a zip file")

        with pytest.raises(MaterializeError):
            connector.materialize(str(bad_file), tmp_path)

    def test_no_kml_inside_raises(self, tmp_path: Path):
        connector = KMZConnector()
        kmz_path = tmp_path / "no_kml.kmz"

        # Create zip with no KML file
        with zipfile.ZipFile(kmz_path, "w") as zf:
            zf.writestr("readme.txt", "This is not a KML file")

        with pytest.raises(MaterializeError):
            connector.materialize(str(kmz_path), tmp_path)

    def test_corrupt_zip_raises(self, tmp_path: Path):
        connector = KMZConnector()
        kmz_path = tmp_path / "corrupt.kmz"
        kmz_path.write_bytes(b"PK\x03\x04corrupted data")

        with pytest.raises(MaterializeError):
            connector.materialize(str(kmz_path), tmp_path)

    def test_metadata_nonexistent_file_raises(self, tmp_path: Path):
        connector = KMZConnector()
        nonexistent = tmp_path / "does_not_exist.kmz"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))

    def test_metadata_not_a_zip_raises(self, tmp_path: Path):
        connector = KMZConnector()
        bad_file = tmp_path / "not_kmz.kmz"
        bad_file.write_text("this is not a zip file")

        with pytest.raises(MaterializeError):
            connector.metadata(str(bad_file))

    def test_metadata_no_kml_inside_raises(self, tmp_path: Path):
        connector = KMZConnector()
        kmz_path = tmp_path / "no_kml.kmz"

        with zipfile.ZipFile(kmz_path, "w") as zf:
            zf.writestr("readme.txt", "This is not a KML file")

        with pytest.raises(MaterializeError):
            connector.metadata(str(kmz_path))

    def test_discover_file_raises(self, kmz_file_points: Path):
        connector = KMZConnector()

        with pytest.raises(MaterializeError):
            connector.discover(str(kmz_file_points))

    def test_discover_no_path_raises(self):
        connector = KMZConnector()

        with pytest.raises(MaterializeError):
            connector.discover()


# ---------------------------------------------------------------------------
# SourceRef Support
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _kml_driver_available(), reason="KML/LIBKML driver not available")
class TestKMZSourceRef:
    """Validate SourceRef handling."""

    def test_sourceref_local(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        ref = SourceRef.local(str(kmz_file_points))
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_raw_string_path(self, kmz_file_points: Path, tmp_path: Path):
        connector = KMZConnector()
        result = connector.materialize(str(kmz_file_points), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestKMZCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        connector = KMZConnector()
        assert connector.name == "kmz"

    def test_capabilities(self):
        from quarry_core.connector import ConnectorCapability

        connector = KMZConnector()
        caps = connector.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps
