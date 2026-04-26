"""
Pressure test: CSVXYConnector.

Lane: connector

Validates CSV/TSV file materialization with coordinate columns:
- source_ref parsing (local path, explicit columns via ::)
- local eager: convert to GeoPackage, LOCAL_FILE backing
- local lazy: metadata only, LAZY_HANDLE backing
- column detection: auto-detect lat/lon, x/y, explicit via ::
- table fallback: CSV without coords → ArtifactType.TABLE
- delimiter detection: comma, tab, semicolon
- missing values: skip rows with empty/non-numeric coords
- discover: list .csv/.tsv files
- metadata: read without materializing
- error handling: nonexistent files, empty files, malformed CSV
"""

from __future__ import annotations

import pytest
from quarry_connectors.csv_xy import CSVXYConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind
from quarry_core.connector import MaterializeError
from quarry_core.source_ref import SourceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def csv_latlon(tmp_path):
    """Create a CSV file with lat/lon columns (3 points)."""
    path = tmp_path / "points.csv"
    content = """name,lat,lon,value
alpha,2.0,1.0,10.5
beta,4.0,3.0,20.5
gamma,6.0,5.0,30.5
"""
    path.write_text(content)
    return path


@pytest.fixture()
def csv_xy(tmp_path):
    """Create a CSV file with x/y columns."""
    path = tmp_path / "projected.csv"
    content = """id,x,y,description
A,100.0,200.0,point_a
B,300.0,400.0,point_b
C,500.0,600.0,point_c
"""
    path.write_text(content)
    return path


@pytest.fixture()
def tsv_latitude_longitude(tmp_path):
    """Create a TSV file with latitude/longitude columns."""
    path = tmp_path / "points.tsv"
    content = """name\tlatitude\tlongitude\tcategory
alpha\t2.0\t1.0\tA
beta\t4.0\t3.0\tB
gamma\t6.0\t5.0\tC
"""
    path.write_text(content)
    return path


@pytest.fixture()
def csv_no_coords(tmp_path):
    """Create a CSV file with no coordinate columns (table only)."""
    path = tmp_path / "data.csv"
    content = """name,age,city
Alice,30,NYC
Bob,25,LA
Charlie,35,Chicago
"""
    path.write_text(content)
    return path


@pytest.fixture()
def csv_missing_values(tmp_path):
    """Create a CSV file with missing values in coordinates."""
    path = tmp_path / "incomplete.csv"
    content = """name,lat,lon
alpha,2.0,1.0
beta,,3.0
gamma,4.0,
delta,invalid,5.0
epsilon,6.0,7.0
"""
    path.write_text(content)
    return path


@pytest.fixture()
def csv_semicolon(tmp_path):
    """Create a semicolon-delimited CSV file."""
    path = tmp_path / "european.csv"
    content = """name;lat;lon;value
alpha;2,0;1,0;10,5
beta;4,0;3,0;20,5
"""
    path.write_text(content)
    return path


@pytest.fixture()
def directory_with_csvs(tmp_path, csv_latlon, csv_xy, csv_no_coords):
    """Create a directory with multiple CSV files."""
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Local Materialization
# ---------------------------------------------------------------------------


class TestCSVXYEagerLocal:
    """Validate eager materialization of local CSV files."""

    def test_eager_produces_vector(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.name == "points"

    def test_eager_produces_local_file_backing(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".gpkg")

    def test_eager_wrapped_local_strategy(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_feature_count(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_eager_extent(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(1.0)
        assert ymin == pytest.approx(2.0)
        assert xmax == pytest.approx(5.0)
        assert ymax == pytest.approx(6.0)

    def test_eager_crs(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path)

        assert result.artifact.spatial.crs is not None
        assert "4326" in result.artifact.spatial.crs

    def test_eager_content_hash_present(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path)

        assert result.artifact.lineage.params["source"] == "csv_xy"
        assert result.artifact.lineage.params["path"] == str(csv_latlon)
        assert result.artifact.lineage.params["lazy"] is False
        assert result.artifact.lineage.params["has_coordinates"] is True
        assert result.artifact.lineage.params["detected_lon_col"] == "lon"
        assert result.artifact.lineage.params["detected_lat_col"] == "lat"

    def test_eager_metadata_columns(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path)

        assert "columns" in result.artifact.metadata
        assert "name" in result.artifact.metadata["columns"]
        assert "lat" in result.artifact.metadata["columns"]
        assert "lon" in result.artifact.metadata["columns"]
        assert "value" in result.artifact.metadata["columns"]


# ---------------------------------------------------------------------------
# Lazy Local Materialization
# ---------------------------------------------------------------------------


class TestCSVXYLazyLocal:
    """Validate lazy (metadata-only) materialization of local CSV files."""

    def test_lazy_backing_kind(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_feature_count(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path, lazy=True)

        assert result.artifact.spatial.feature_count == 3

    def test_lazy_extent(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path, lazy=True)

        extent = result.artifact.spatial.extent
        assert extent is not None
        assert extent[0] == pytest.approx(1.0)

    def test_lazy_backing_uri(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path, lazy=True)

        assert result.artifact.backing.uri == str(csv_latlon)

    def test_lazy_lineage(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True
        assert result.artifact.lineage.params["has_coordinates"] is True

    def test_lazy_detected_columns_in_metadata(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path, lazy=True)

        assert result.artifact.metadata["detected_lon_col"] == "lon"
        assert result.artifact.metadata["detected_lat_col"] == "lat"


# ---------------------------------------------------------------------------
# Column Detection
# ---------------------------------------------------------------------------


class TestCSVXYColumnDetection:
    """Validate coordinate column detection."""

    def test_auto_detect_lat_lon(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path, lazy=True)

        assert result.artifact.metadata["detected_lon_col"] == "lon"
        assert result.artifact.metadata["detected_lat_col"] == "lat"
        assert result.artifact.spatial.crs == "EPSG:4326"

    def test_auto_detect_latitude_longitude(self, tsv_latitude_longitude, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(tsv_latitude_longitude), tmp_path, lazy=True)

        assert result.artifact.metadata["detected_lon_col"] == "longitude"
        assert result.artifact.metadata["detected_lat_col"] == "latitude"
        assert result.artifact.spatial.crs == "EPSG:4326"

    def test_auto_detect_x_y(self, csv_xy, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_xy), tmp_path, lazy=True)

        assert result.artifact.metadata["detected_lon_col"] == "x"
        assert result.artifact.metadata["detected_lat_col"] == "y"
        # x/y uses default_crs (EPSG:4326 by default)
        assert result.artifact.spatial.crs == "EPSG:4326"

    def test_explicit_columns_via_syntax(self, csv_xy, tmp_path):
        """Test "path/to/file.csv::x,y" syntax for explicit columns."""
        connector = CSVXYConnector()
        source_ref = f"{csv_xy}::x,y"
        result = connector.materialize(source_ref, tmp_path, lazy=True)

        assert result.artifact.metadata["detected_lon_col"] == "x"
        assert result.artifact.metadata["detected_lat_col"] == "y"

    def test_explicit_columns_different_names(self, tmp_path):
        """Test explicit columns with non-standard names."""
        path = tmp_path / "custom.csv"
        content = """name,easting,northing,value
alpha,100,200,10
beta,300,400,20
"""
        path.write_text(content)

        connector = CSVXYConnector()
        source_ref = f"{path}::easting,northing"
        result = connector.materialize(source_ref, tmp_path, lazy=True)

        assert result.artifact.metadata["detected_lon_col"] == "easting"
        assert result.artifact.metadata["detected_lat_col"] == "northing"


# ---------------------------------------------------------------------------
# Table Fallback
# ---------------------------------------------------------------------------


class TestCSVXYTableFallback:
    """Validate CSV without coordinates falls back to TABLE artifact."""

    def test_no_coords_produces_table(self, csv_no_coords, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_no_coords), tmp_path)

        assert result.artifact.type == ArtifactType.TABLE

    def test_no_coords_local_file_backing(self, csv_no_coords, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_no_coords), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri == str(csv_no_coords)

    def test_no_coords_row_count(self, csv_no_coords, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_no_coords), tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_no_coords_no_crs(self, csv_no_coords, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_no_coords), tmp_path)

        assert result.artifact.spatial.crs is None
        assert result.artifact.spatial.extent is None


# ---------------------------------------------------------------------------
# Delimiter Detection
# ---------------------------------------------------------------------------


class TestCSVXYDelimiter:
    """Validate delimiter detection for various formats."""

    def test_comma_delimited(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path, lazy=True)

        assert result.artifact.metadata["delimiter"] == ","

    def test_tab_delimited(self, tsv_latitude_longitude, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(tsv_latitude_longitude), tmp_path, lazy=True)

        assert result.artifact.metadata["delimiter"] == "\t"

    def test_semicolon_delimited(self, csv_semicolon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_semicolon), tmp_path, lazy=True)

        assert result.artifact.metadata["delimiter"] == ";"


# ---------------------------------------------------------------------------
# Missing Values
# ---------------------------------------------------------------------------


class TestCSVXYMissingValues:
    """Validate handling of missing/invalid coordinate values."""

    def test_skip_empty_coords(self, csv_missing_values, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_missing_values), tmp_path)

        # Only alpha and epsilon have valid coordinates
        assert result.artifact.spatial.feature_count == 2

    def test_skip_invalid_coords(self, csv_missing_values, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_missing_values), tmp_path)

        # delta has invalid lat value, should be skipped
        assert result.artifact.spatial.feature_count == 2

    def test_valid_extent_with_missing(self, csv_missing_values, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_missing_values), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        # Only alpha (1,2) and epsilon (7,6) contribute to extent
        assert extent[0] == pytest.approx(1.0)  # xmin
        assert extent[1] == pytest.approx(2.0)  # ymin
        assert extent[2] == pytest.approx(7.0)  # xmax
        assert extent[3] == pytest.approx(6.0)  # ymax


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


class TestCSVXYDiscover:
    """Validate file discovery."""

    def test_discover_lists_csv_files(self, directory_with_csvs):
        connector = CSVXYConnector()
        entries = connector.discover(str(directory_with_csvs))

        names = {e.name for e in entries}
        assert "points" in names
        assert "projected" in names
        assert "data" in names

    def test_discover_source_refs(self, directory_with_csvs):
        connector = CSVXYConnector()
        entries = connector.discover(str(directory_with_csvs))

        for entry in entries:
            assert entry.source_ref.endswith(".csv")

    def test_discover_with_dict_query(self, directory_with_csvs):
        connector = CSVXYConnector()
        entries = connector.discover({"path": str(directory_with_csvs)})

        assert len(entries) >= 3

    def test_discover_empty_directory(self, tmp_path):
        connector = CSVXYConnector()
        entries = connector.discover(str(tmp_path))

        assert entries == []

    def test_discover_nonexistent_directory_raises(self, tmp_path):
        connector = CSVXYConnector()
        nonexistent = tmp_path / "no_such_dir"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))

    def test_discover_includes_tsv(self, tmp_path):
        """Discover should include .tsv files."""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("a\tb\tc\n1\t2\t3\n")

        connector = CSVXYConnector()
        entries = connector.discover(str(tmp_path))

        names = {e.name for e in entries}
        assert "data" in names


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestCSVXYMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_columns(self, csv_latlon):
        connector = CSVXYConnector()
        meta = connector.metadata(str(csv_latlon))

        assert "columns" in meta
        assert "name" in meta["columns"]
        assert "lat" in meta["columns"]
        assert "lon" in meta["columns"]

    def test_metadata_row_count(self, csv_latlon):
        connector = CSVXYConnector()
        meta = connector.metadata(str(csv_latlon))

        assert meta["row_count"] == 3

    def test_metadata_detected_coords(self, csv_latlon):
        connector = CSVXYConnector()
        meta = connector.metadata(str(csv_latlon))

        assert meta["detected_lon_col"] == "lon"
        assert meta["detected_lat_col"] == "lat"

    def test_metadata_crs(self, csv_latlon):
        connector = CSVXYConnector()
        meta = connector.metadata(str(csv_latlon))

        assert meta["crs"] == "EPSG:4326"

    def test_metadata_extent(self, csv_latlon):
        connector = CSVXYConnector()
        meta = connector.metadata(str(csv_latlon))

        extent = meta["extent"]
        assert extent is not None
        assert len(extent) == 4

    def test_metadata_delimiter(self, csv_latlon):
        connector = CSVXYConnector()
        meta = connector.metadata(str(csv_latlon))

        assert meta["delimiter"] == ","


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestCSVXYErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path):
        connector = CSVXYConnector()
        nonexistent = tmp_path / "does_not_exist.csv"

        with pytest.raises(MaterializeError):
            connector.materialize(str(nonexistent), tmp_path)

    def test_empty_file_raises(self, tmp_path):
        connector = CSVXYConnector()
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        with pytest.raises(MaterializeError):
            connector.materialize(str(empty_file), tmp_path)

    def test_header_only_csv(self, tmp_path):
        """CSV with only header row should be handled gracefully."""
        connector = CSVXYConnector()
        header_only = tmp_path / "header_only.csv"
        header_only.write_text("name,lat,lon\n")

        result = connector.materialize(str(header_only), tmp_path)
        # Should produce vector with 0 features
        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 0

    def test_metadata_nonexistent_file_raises(self, tmp_path):
        connector = CSVXYConnector()
        nonexistent = tmp_path / "does_not_exist.csv"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))

    def test_explicit_invalid_column_raises(self, tmp_path):
        """Explicit column that doesn't exist should raise error."""
        path = tmp_path / "test.csv"
        path.write_text("a,b,c\n1,2,3\n")

        connector = CSVXYConnector()
        source_ref = f"{path}::nonexistent,also_missing"

        with pytest.raises(MaterializeError):
            connector.materialize(source_ref, tmp_path)


# ---------------------------------------------------------------------------
# SourceRef Support
# ---------------------------------------------------------------------------


class TestCSVXYSourceRef:
    """Validate SourceRef handling."""

    def test_sourceref_local(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        ref = SourceRef.local(str(csv_latlon))
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_raw_string_path(self, csv_latlon, tmp_path):
        connector = CSVXYConnector()
        result = connector.materialize(str(csv_latlon), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCSVXYCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        connector = CSVXYConnector()
        assert connector.name == "csv_xy"

    def test_capabilities(self):
        from quarry_core.connector import ConnectorCapability

        connector = CSVXYConnector()
        caps = connector.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps

    def test_default_crs_constructor(self):
        """Test that default_crs can be customized."""
        connector = CSVXYConnector(default_crs="EPSG:32633")
        assert connector.default_crs == "EPSG:32633"
