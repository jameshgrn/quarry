"""
Pressure test: LASPointCloudConnector.

Lane: connector

Validates LAS/LAZ lidar point cloud file materialization:
- source_ref parsing (local path)
- local eager: wrap in place, LOCAL_FILE backing
- local lazy: metadata only, LAZY_HANDLE backing
- discover: list .las/.laz files in directory
- metadata: read without materializing
- error handling: nonexistent files, invalid formats, missing laspy
"""

from __future__ import annotations

import pytest

# Skip all tests if laspy is not installed
pytest.importorskip("laspy")

import numpy as np
from quarry_connectors.las import LASPointCloudConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_las_file(
    path,
    points_xyz,
    point_format_id=0,
    crs_wkt=None,
    version="1.4",
):
    """Create a LAS file using laspy.

    Args:
        path: Output file path
        points_xyz: Nx3 array of XYZ coordinates
        point_format_id: LAS point format ID (0-20)
        crs_wkt: Optional WKT CRS string
        version: LAS version string (e.g., "1.4")
    """
    import laspy
    from laspy.vlrs.known import WktCoordinateSystemVlr

    # Create header - version can be string or tuple
    header = laspy.LasHeader(point_format=point_format_id, version=version)

    # Add CRS if provided - use WktCoordinateSystemVlr directly
    if crs_wkt:
        wkt_vlr = WktCoordinateSystemVlr()
        wkt_vlr.string = crs_wkt
        header.vlrs.append(wkt_vlr)

    # Create LasData and set points
    las = laspy.LasData(header)

    if len(points_xyz) > 0:
        las.x = points_xyz[:, 0]
        las.y = points_xyz[:, 1]
        las.z = points_xyz[:, 2]
    else:
        las.x = np.array([], dtype=np.float64)
        las.y = np.array([], dtype=np.float64)
        las.z = np.array([], dtype=np.float64)

    las.write(str(path))


@pytest.fixture()
def las_file(tmp_path):
    """Create a LAS file with 100 random 3D points."""
    path = tmp_path / "points.las"

    # Generate 100 random points in a known bounding box
    np.random.seed(42)
    n_points = 100
    x = np.random.uniform(1000.0, 2000.0, n_points)
    y = np.random.uniform(3000.0, 4000.0, n_points)
    z = np.random.uniform(50.0, 150.0, n_points)
    points = np.column_stack([x, y, z])

    _create_las_file(path, points, point_format_id=1, version="1.4")
    return path


@pytest.fixture()
def las_file_with_crs(tmp_path):
    """Create a LAS file with CRS."""
    path = tmp_path / "points_crs.las"

    np.random.seed(123)
    n_points = 50
    x = np.random.uniform(500000.0, 501000.0, n_points)
    y = np.random.uniform(5000000.0, 5001000.0, n_points)
    z = np.random.uniform(100.0, 200.0, n_points)
    points = np.column_stack([x, y, z])

    # Use a simple WKT for testing
    wkt = (
        'PROJCS["WGS 84 / UTM zone 33N",'
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
        'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
        'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
        'AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],'
        'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],'
        'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
        'PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
        'AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32633"]]'
    )

    _create_las_file(path, points, point_format_id=3, crs_wkt=wkt, version="1.4")
    return path


@pytest.fixture()
def las_file_known_extent(tmp_path):
    """Create a LAS file with known coordinates for extent testing."""
    path = tmp_path / "known_extent.las"

    # Create points with known min/max
    points = np.array(
        [
            [10.0, 20.0, 5.0],
            [15.0, 25.0, 10.0],
            [20.0, 30.0, 15.0],
            [100.0, 200.0, 50.0],  # Max point
        ]
    )

    _create_las_file(path, points, point_format_id=0, version="1.2")
    return path


@pytest.fixture()
def laz_file(tmp_path):
    """Create a LAZ (compressed) file."""
    path = tmp_path / "points.laz"

    np.random.seed(456)
    n_points = 75
    x = np.random.uniform(0.0, 1000.0, n_points)
    y = np.random.uniform(0.0, 1000.0, n_points)
    z = np.random.uniform(0.0, 100.0, n_points)
    points = np.column_stack([x, y, z])

    # LAZ requires laszip support
    try:
        _create_las_file(path, points, point_format_id=1, version="1.4")
    except Exception as e:
        pytest.skip(f"LAZ compression not available: {e}")

    return path


@pytest.fixture()
def directory_with_las(tmp_path, las_file, las_file_known_extent):
    """Create a directory with multiple LAS files."""
    # Files are already in tmp_path from fixtures
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Local Materialization
# ---------------------------------------------------------------------------


class TestLASEagerLocal:
    """Validate eager materialization of local LAS files."""

    def test_eager_produces_vector(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.name == "points"

    def test_eager_produces_local_file_backing(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".las")

    def test_eager_wrapped_local_strategy(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_feature_count(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path)

        assert result.artifact.spatial.feature_count == 100

    def test_eager_extent(self, las_file_known_extent, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file_known_extent), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(10.0)
        assert ymin == pytest.approx(20.0)
        assert xmax == pytest.approx(100.0)
        assert ymax == pytest.approx(200.0)

    def test_eager_content_hash_present(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path)

        assert result.artifact.lineage.params["source"] == "las"
        assert result.artifact.lineage.params["path"] == str(las_file)
        assert result.artifact.lineage.params["lazy"] is False
        assert result.artifact.lineage.params["point_count"] == 100

    def test_eager_metadata_point_format(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path)

        assert result.artifact.metadata["point_format_id"] == 1
        assert result.artifact.metadata["las_version"] == "1.4"

    def test_eager_metadata_scales_offsets(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path)

        assert "scales" in result.artifact.metadata
        assert "offsets" in result.artifact.metadata
        assert len(result.artifact.metadata["scales"]) == 3
        assert len(result.artifact.metadata["offsets"]) == 3


# ---------------------------------------------------------------------------
# Lazy Local Materialization
# ---------------------------------------------------------------------------


class TestLASLazyLocal:
    """Validate lazy (metadata-only) materialization of local LAS files."""

    def test_lazy_backing_kind(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_feature_count(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path, lazy=True)

        assert result.artifact.spatial.feature_count == 100

    def test_lazy_extent(self, las_file_known_extent, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file_known_extent), tmp_path, lazy=True)

        extent = result.artifact.spatial.extent
        assert extent is not None
        assert extent[0] == pytest.approx(10.0)
        assert extent[2] == pytest.approx(100.0)

    def test_lazy_backing_uri(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path, lazy=True)

        assert result.artifact.backing.uri == str(las_file)

    def test_lazy_lineage(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True
        assert result.artifact.lineage.params["point_count"] == 100

    def test_lazy_no_content_hash(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path, lazy=True)

        # Lazy handles don't compute content hash
        assert result.artifact.backing.content_hash is None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestLASMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_point_count(self, las_file):
        connector = LASPointCloudConnector()
        meta = connector.metadata(str(las_file))

        assert meta["point_count"] == 100

    def test_metadata_returns_point_format(self, las_file):
        connector = LASPointCloudConnector()
        meta = connector.metadata(str(las_file))

        assert meta["point_format_id"] == 1

    def test_metadata_returns_version(self, las_file):
        connector = LASPointCloudConnector()
        meta = connector.metadata(str(las_file))

        assert meta["las_version"] == "1.4"

    def test_metadata_returns_extent(self, las_file_known_extent):
        connector = LASPointCloudConnector()
        meta = connector.metadata(str(las_file_known_extent))

        extent = meta["extent"]
        assert extent is not None
        assert len(extent) == 4
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(10.0)
        assert ymin == pytest.approx(20.0)
        assert xmax == pytest.approx(100.0)
        assert ymax == pytest.approx(200.0)

    def test_metadata_returns_extent_3d(self, las_file_known_extent):
        connector = LASPointCloudConnector()
        meta = connector.metadata(str(las_file_known_extent))

        extent_3d = meta["extent_3d"]
        assert extent_3d is not None
        assert len(extent_3d) == 6
        zmin, zmax = extent_3d[2], extent_3d[5]
        assert zmin == pytest.approx(5.0)
        assert zmax == pytest.approx(50.0)

    def test_metadata_returns_scales_offsets(self, las_file):
        connector = LASPointCloudConnector()
        meta = connector.metadata(str(las_file))

        assert "scales" in meta
        assert "offsets" in meta
        assert len(meta["scales"]) == 3
        assert len(meta["offsets"]) == 3

    def test_metadata_has_color_gps_time_flags(self, las_file):
        connector = LASPointCloudConnector()
        meta = connector.metadata(str(las_file))

        assert "has_color" in meta
        assert "has_gps_time" in meta
        assert isinstance(meta["has_color"], bool)
        assert isinstance(meta["has_gps_time"], bool)

    def test_metadata_extra_dims(self, las_file):
        connector = LASPointCloudConnector()
        meta = connector.metadata(str(las_file))

        assert "extra_dims" in meta
        assert isinstance(meta["extra_dims"], list)


# ---------------------------------------------------------------------------
# CRS Extraction
# ---------------------------------------------------------------------------


class TestLASCRS:
    """Validate CRS extraction from LAS files."""

    def test_crs_extracted_when_present(self, las_file_with_crs, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file_with_crs), tmp_path)

        # CRS should be present (either as WKT or parsed)
        assert result.artifact.spatial.crs is not None
        assert result.artifact.metadata["crs"] is not None

    def test_crs_none_when_not_present(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path)

        # No CRS in this file
        assert result.artifact.spatial.crs is None
        assert result.artifact.metadata["crs"] is None

    def test_metadata_crs(self, las_file_with_crs):
        connector = LASPointCloudConnector()
        meta = connector.metadata(str(las_file_with_crs))

        assert meta["crs"] is not None


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestLASDiscover:
    """Validate file discovery."""

    def test_discover_lists_las_files(self, directory_with_las):
        connector = LASPointCloudConnector()
        entries = connector.discover(str(directory_with_las))

        names = {e.name for e in entries}
        assert "points" in names
        assert "known_extent" in names

    def test_discover_source_refs(self, directory_with_las):
        connector = LASPointCloudConnector()
        entries = connector.discover(str(directory_with_las))

        for entry in entries:
            assert entry.source_ref.endswith(".las") or entry.source_ref.endswith(".LAS")

    def test_discover_with_dict_query(self, directory_with_las):
        connector = LASPointCloudConnector()
        entries = connector.discover({"path": str(directory_with_las)})

        assert len(entries) >= 2

    def test_discover_empty_directory(self, tmp_path):
        connector = LASPointCloudConnector()
        entries = connector.discover(str(tmp_path))

        assert entries == []

    def test_discover_nonexistent_directory_raises(self, tmp_path):
        connector = LASPointCloudConnector()
        nonexistent = tmp_path / "no_such_dir"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))

    def test_discover_includes_laz(self, tmp_path):
        """Test that .laz files are also discovered."""
        connector = LASPointCloudConnector()

        # Create a LAZ file if possible
        try:
            laz_path = tmp_path / "compressed.laz"
            points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            _create_las_file(laz_path, points, point_format_id=1)
        except Exception:
            pytest.skip("LAZ compression not available")

        entries = connector.discover(str(tmp_path))
        names = {e.name for e in entries}
        assert "compressed" in names


# ---------------------------------------------------------------------------
# LAZ Support
# ---------------------------------------------------------------------------


class TestLASLAZSupport:
    """Validate LAZ (compressed) file support."""

    def test_laz_eager_materialization(self, laz_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(laz_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.spatial.feature_count == 75

    def test_laz_lazy_materialization(self, laz_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(laz_file), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE
        assert result.artifact.spatial.feature_count == 75

    def test_laz_metadata(self, laz_file):
        connector = LASPointCloudConnector()
        meta = connector.metadata(str(laz_file))

        assert meta["point_count"] == 75


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestLASErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path):
        connector = LASPointCloudConnector()
        nonexistent = tmp_path / "does_not_exist.las"

        with pytest.raises(MaterializeError):
            connector.materialize(str(nonexistent), tmp_path)

    def test_non_las_file_raises(self, tmp_path):
        connector = LASPointCloudConnector()
        bad_file = tmp_path / "not_las.txt"
        bad_file.write_text("this is not a las file")

        with pytest.raises(MaterializeError):
            connector.materialize(str(bad_file), tmp_path)

    def test_metadata_nonexistent_file_raises(self, tmp_path):
        connector = LASPointCloudConnector()
        nonexistent = tmp_path / "does_not_exist.las"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))

    def test_discover_no_path_raises(self):
        connector = LASPointCloudConnector()

        with pytest.raises(MaterializeError):
            connector.discover()

    def test_discover_file_raises(self, las_file):
        connector = LASPointCloudConnector()

        with pytest.raises(MaterializeError):
            connector.discover(str(las_file))


# ---------------------------------------------------------------------------
# SourceRef Support
# ---------------------------------------------------------------------------


class TestLASSourceRef:
    """Validate SourceRef handling."""

    def test_sourceref_local(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        ref = SourceRef.local(str(las_file))
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_raw_string_path(self, las_file, tmp_path):
        connector = LASPointCloudConnector()
        result = connector.materialize(str(las_file), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestLASCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        connector = LASPointCloudConnector()
        assert connector.name == "las"

    def test_capabilities(self):
        from quarry_core.connector import ConnectorCapability

        connector = LASPointCloudConnector()
        caps = connector.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps
