"""
Pressure test: ExcelXYConnector.

Lane: connector

Validates Excel file (.xlsx/.xls) materialization with coordinate columns:
- source_ref parsing (local path, sheet selection, explicit columns via ::)
- local eager: convert to GeoPackage, LOCAL_FILE backing
- local lazy: metadata only, LAZY_HANDLE backing
- column detection: auto-detect lat/lon, x/y, explicit via ::
- sheet selection: default (first) sheet, explicit sheet via ::
- table fallback: Excel without coords → ArtifactType.TABLE
- missing values: skip rows with empty/non-numeric coords
- discover: list .xlsx/.xls files
- metadata: read without materializing
- error handling: nonexistent files, empty files, bad sheet name
"""

from __future__ import annotations

import pytest

# Skip all tests if openpyxl is not installed
openpyxl = pytest.importorskip("openpyxl")

from quarry_connectors.excel_xy import ExcelXYConnector  # noqa: E402
from quarry_core.artifact import ArtifactType, BackingStoreKind  # noqa: E402
from quarry_core.connector import MaterializeError  # noqa: E402
from quarry_core.source_ref import SourceRef  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_xlsx(path, headers, rows, sheet_name="Sheet1"):
    """Create an Excel file with the given headers and rows."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(headers)
    for row in rows:
        ws.append(row)
    wb.save(str(path))


@pytest.fixture()
def xlsx_latlon(tmp_path):
    """Create an Excel file with lat/lon columns (3 points)."""
    path = tmp_path / "points.xlsx"
    _create_xlsx(
        path,
        headers=["name", "lat", "lon", "value"],
        rows=[
            ["alpha", 2.0, 1.0, 10.5],
            ["beta", 4.0, 3.0, 20.5],
            ["gamma", 6.0, 5.0, 30.5],
        ],
    )
    return path


@pytest.fixture()
def xlsx_xy(tmp_path):
    """Create an Excel file with x/y columns."""
    path = tmp_path / "projected.xlsx"
    _create_xlsx(
        path,
        headers=["id", "x", "y", "description"],
        rows=[
            ["A", 100.0, 200.0, "point_a"],
            ["B", 300.0, 400.0, "point_b"],
            ["C", 500.0, 600.0, "point_c"],
        ],
    )
    return path


@pytest.fixture()
def xlsx_latitude_longitude(tmp_path):
    """Create an Excel file with latitude/longitude columns."""
    path = tmp_path / "points.xlsx"
    _create_xlsx(
        path,
        headers=["name", "latitude", "longitude", "category"],
        rows=[
            ["alpha", 2.0, 1.0, "A"],
            ["beta", 4.0, 3.0, "B"],
            ["gamma", 6.0, 5.0, "C"],
        ],
    )
    return path


@pytest.fixture()
def xlsx_no_coords(tmp_path):
    """Create an Excel file with no coordinate columns (table only)."""
    path = tmp_path / "data.xlsx"
    _create_xlsx(
        path,
        headers=["name", "age", "city"],
        rows=[
            ["Alice", 30, "NYC"],
            ["Bob", 25, "LA"],
            ["Charlie", 35, "Chicago"],
        ],
    )
    return path


@pytest.fixture()
def xlsx_missing_values(tmp_path):
    """Create an Excel file with missing values in coordinates."""
    path = tmp_path / "incomplete.xlsx"
    _create_xlsx(
        path,
        headers=["name", "lat", "lon"],
        rows=[
            ["alpha", 2.0, 1.0],
            ["beta", None, 3.0],
            ["gamma", 4.0, None],
            ["delta", "invalid", 5.0],
            ["epsilon", 6.0, 7.0],
        ],
    )
    return path


@pytest.fixture()
def xlsx_multiple_sheets(tmp_path):
    """Create an Excel file with multiple sheets."""
    path = tmp_path / "multi_sheet.xlsx"
    wb = openpyxl.Workbook()

    # First sheet
    ws1 = wb.active
    ws1.title = "Locations"
    ws1.append(["name", "lat", "lon"])
    ws1.append(["point1", 1.0, 2.0])
    ws1.append(["point2", 3.0, 4.0])

    # Second sheet
    ws2 = wb.create_sheet("OtherData")
    ws2.append(["id", "value"])
    ws2.append([1, "a"])
    ws2.append([2, "b"])

    wb.save(str(path))
    return path


@pytest.fixture()
def directory_with_xlsx(tmp_path, xlsx_latlon, xlsx_xy, xlsx_no_coords):
    """Create a directory with multiple Excel files."""
    return tmp_path


# ---------------------------------------------------------------------------
# Eager Local Materialization
# ---------------------------------------------------------------------------


class TestExcelXYEagerLocal:
    """Validate eager materialization of local Excel files."""

    def test_eager_produces_vector(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.name == "points"

    def test_eager_produces_local_file_backing(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri.endswith(".gpkg")

    def test_eager_wrapped_local_strategy(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path)

        assert result.strategy == "wrapped_local"

    def test_eager_feature_count(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_eager_extent(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path)

        extent = result.artifact.spatial.extent
        assert extent is not None
        xmin, ymin, xmax, ymax = extent
        assert xmin == pytest.approx(1.0)
        assert ymin == pytest.approx(2.0)
        assert xmax == pytest.approx(5.0)
        assert ymax == pytest.approx(6.0)

    def test_eager_crs(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path)

        assert result.artifact.spatial.crs is not None
        assert "4326" in result.artifact.spatial.crs

    def test_eager_content_hash_present(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path)

        assert result.artifact.backing.content_hash is not None
        assert len(result.artifact.backing.content_hash) == 64  # SHA-256 hex

    def test_eager_lineage(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path)

        assert result.artifact.lineage.params["source"] == "excel_xy"
        assert result.artifact.lineage.params["path"] == str(xlsx_latlon)
        assert result.artifact.lineage.params["lazy"] is False
        assert result.artifact.lineage.params["has_coordinates"] is True
        assert result.artifact.lineage.params["detected_lon_col"] == "lon"
        assert result.artifact.lineage.params["detected_lat_col"] == "lat"

    def test_eager_metadata_columns(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path)

        assert "columns" in result.artifact.metadata
        assert "name" in result.artifact.metadata["columns"]
        assert "lat" in result.artifact.metadata["columns"]
        assert "lon" in result.artifact.metadata["columns"]
        assert "value" in result.artifact.metadata["columns"]

    def test_eager_metadata_sheet_name(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path)

        assert result.artifact.metadata["sheet_name"] == "Sheet1"
        assert "Sheet1" in result.artifact.metadata["sheet_names"]


# ---------------------------------------------------------------------------
# Lazy Local Materialization
# ---------------------------------------------------------------------------


class TestExcelXYLazyLocal:
    """Validate lazy (metadata-only) materialization of local Excel files."""

    def test_lazy_backing_kind(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path, lazy=True)

        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE

    def test_lazy_strategy(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path, lazy=True)

        assert result.strategy == "lazy_handle"

    def test_lazy_feature_count(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path, lazy=True)

        assert result.artifact.spatial.feature_count == 3

    def test_lazy_extent(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path, lazy=True)

        extent = result.artifact.spatial.extent
        assert extent is not None
        assert extent[0] == pytest.approx(1.0)

    def test_lazy_backing_uri(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path, lazy=True)

        assert result.artifact.backing.uri == str(xlsx_latlon)

    def test_lazy_lineage(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path, lazy=True)

        assert result.artifact.lineage.params["lazy"] is True
        assert result.artifact.lineage.params["has_coordinates"] is True

    def test_lazy_detected_columns_in_metadata(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path, lazy=True)

        assert result.artifact.metadata["detected_lon_col"] == "lon"
        assert result.artifact.metadata["detected_lat_col"] == "lat"


# ---------------------------------------------------------------------------
# Column Detection
# ---------------------------------------------------------------------------


class TestExcelXYColumnDetection:
    """Validate coordinate column detection."""

    def test_auto_detect_lat_lon(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path, lazy=True)

        assert result.artifact.metadata["detected_lon_col"] == "lon"
        assert result.artifact.metadata["detected_lat_col"] == "lat"
        assert result.artifact.spatial.crs == "EPSG:4326"

    def test_auto_detect_latitude_longitude(self, xlsx_latitude_longitude, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latitude_longitude), tmp_path, lazy=True)

        assert result.artifact.metadata["detected_lon_col"] == "longitude"
        assert result.artifact.metadata["detected_lat_col"] == "latitude"
        assert result.artifact.spatial.crs == "EPSG:4326"

    def test_auto_detect_x_y(self, xlsx_xy, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_xy), tmp_path, lazy=True)

        assert result.artifact.metadata["detected_lon_col"] == "x"
        assert result.artifact.metadata["detected_lat_col"] == "y"
        # x/y uses default_crs (EPSG:4326 by default)
        assert result.artifact.spatial.crs == "EPSG:4326"

    def test_explicit_columns_via_syntax(self, xlsx_xy, tmp_path):
        """Test "path/to/file.xlsx::Sheet1::x,y" syntax for explicit columns."""
        connector = ExcelXYConnector()
        source_ref = f"{xlsx_xy}::Sheet1::x,y"
        result = connector.materialize(source_ref, tmp_path, lazy=True)

        assert result.artifact.metadata["detected_lon_col"] == "x"
        assert result.artifact.metadata["detected_lat_col"] == "y"

    def test_explicit_columns_different_names(self, tmp_path):
        """Test explicit columns with non-standard names."""
        path = tmp_path / "custom.xlsx"
        _create_xlsx(
            path,
            headers=["name", "easting", "northing", "value"],
            rows=[
                ["alpha", 100, 200, 10],
                ["beta", 300, 400, 20],
            ],
        )

        connector = ExcelXYConnector()
        source_ref = f"{path}::Sheet1::easting,northing"
        result = connector.materialize(source_ref, tmp_path, lazy=True)

        assert result.artifact.metadata["detected_lon_col"] == "easting"
        assert result.artifact.metadata["detected_lat_col"] == "northing"


# ---------------------------------------------------------------------------
# Sheet Selection
# ---------------------------------------------------------------------------


class TestExcelXYSheetSelection:
    """Validate sheet selection behavior."""

    def test_default_first_sheet(self, xlsx_multiple_sheets, tmp_path):
        """Default should use first sheet (Locations)."""
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_multiple_sheets), tmp_path, lazy=True)

        assert result.artifact.metadata["sheet_name"] == "Locations"
        assert result.artifact.spatial.feature_count == 2

    def test_explicit_sheet_via_syntax(self, xlsx_multiple_sheets, tmp_path):
        """Test "path/to/file.xlsx::sheet_name" syntax."""
        connector = ExcelXYConnector()
        source_ref = f"{xlsx_multiple_sheets}::OtherData"
        result = connector.materialize(source_ref, tmp_path, lazy=True)

        assert result.artifact.metadata["sheet_name"] == "OtherData"
        # OtherData has no coordinates, so it's a table
        assert result.artifact.type == ArtifactType.TABLE

    def test_explicit_sheet_with_columns(self, xlsx_multiple_sheets, tmp_path):
        """Test "path/to/file.xlsx::sheet_name::lon_col,lat_col" syntax."""
        connector = ExcelXYConnector()
        source_ref = f"{xlsx_multiple_sheets}::Locations::lon,lat"
        result = connector.materialize(source_ref, tmp_path, lazy=True)

        assert result.artifact.metadata["sheet_name"] == "Locations"
        assert result.artifact.metadata["detected_lon_col"] == "lon"
        assert result.artifact.metadata["detected_lat_col"] == "lat"


# ---------------------------------------------------------------------------
# Table Fallback
# ---------------------------------------------------------------------------


class TestExcelXYTableFallback:
    """Validate Excel without coordinates falls back to TABLE artifact."""

    def test_no_coords_produces_table(self, xlsx_no_coords, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_no_coords), tmp_path)

        assert result.artifact.type == ArtifactType.TABLE

    def test_no_coords_local_file_backing(self, xlsx_no_coords, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_no_coords), tmp_path)

        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert result.artifact.backing.uri == str(xlsx_no_coords)

    def test_no_coords_row_count(self, xlsx_no_coords, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_no_coords), tmp_path)

        assert result.artifact.spatial.feature_count == 3

    def test_no_coords_no_crs(self, xlsx_no_coords, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_no_coords), tmp_path)

        assert result.artifact.spatial.crs is None
        assert result.artifact.spatial.extent is None


# ---------------------------------------------------------------------------
# Missing Values
# ---------------------------------------------------------------------------


class TestExcelXYMissingValues:
    """Validate handling of missing/invalid coordinate values."""

    def test_skip_empty_coords(self, xlsx_missing_values, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_missing_values), tmp_path)

        # Only alpha and epsilon have valid coordinates
        assert result.artifact.spatial.feature_count == 2

    def test_skip_invalid_coords(self, xlsx_missing_values, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_missing_values), tmp_path)

        # delta has invalid lat value, should be skipped
        assert result.artifact.spatial.feature_count == 2

    def test_valid_extent_with_missing(self, xlsx_missing_values, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_missing_values), tmp_path)

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


class TestExcelXYDiscover:
    """Validate file discovery."""

    def test_discover_lists_xlsx_files(self, directory_with_xlsx):
        connector = ExcelXYConnector()
        entries = connector.discover(str(directory_with_xlsx))

        names = {e.name for e in entries}
        assert "points" in names
        assert "projected" in names
        assert "data" in names

    def test_discover_source_refs(self, directory_with_xlsx):
        connector = ExcelXYConnector()
        entries = connector.discover(str(directory_with_xlsx))

        for entry in entries:
            assert entry.source_ref.endswith((".xlsx", ".xls", ".XLSX", ".XLS"))

    def test_discover_with_dict_query(self, directory_with_xlsx):
        connector = ExcelXYConnector()
        entries = connector.discover({"path": str(directory_with_xlsx)})

        assert len(entries) >= 3

    def test_discover_empty_directory(self, tmp_path):
        connector = ExcelXYConnector()
        entries = connector.discover(str(tmp_path))

        assert entries == []

    def test_discover_nonexistent_directory_raises(self, tmp_path):
        connector = ExcelXYConnector()
        nonexistent = tmp_path / "no_such_dir"

        with pytest.raises(MaterializeError):
            connector.discover(str(nonexistent))

    def test_discover_includes_xls(self, tmp_path):
        """Discover should include .xls files."""
        # Note: openpyxl only supports .xlsx, but we test the pattern matching
        # Create a .xlsx file but name it with .xls extension for pattern test
        xls_file = tmp_path / "legacy.xls"
        _create_xlsx(
            xls_file,
            headers=["a", "b", "c"],
            rows=[[1, 2, 3]],
        )

        connector = ExcelXYConnector()
        entries = connector.discover(str(tmp_path))

        names = {e.name for e in entries}
        assert "legacy" in names

    def test_discover_metadata_has_sheet_names(self, xlsx_multiple_sheets):
        connector = ExcelXYConnector()
        entries = connector.discover(str(xlsx_multiple_sheets.parent))

        multi_entry = next(e for e in entries if e.name == "multi_sheet")
        assert "sheet_names" in multi_entry.metadata
        assert "Locations" in multi_entry.metadata["sheet_names"]
        assert "OtherData" in multi_entry.metadata["sheet_names"]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestExcelXYMetadata:
    """Validate metadata() without materialization."""

    def test_metadata_returns_columns(self, xlsx_latlon):
        connector = ExcelXYConnector()
        meta = connector.metadata(str(xlsx_latlon))

        assert "columns" in meta
        assert "name" in meta["columns"]
        assert "lat" in meta["columns"]
        assert "lon" in meta["columns"]

    def test_metadata_row_count(self, xlsx_latlon):
        connector = ExcelXYConnector()
        meta = connector.metadata(str(xlsx_latlon))

        assert meta["row_count"] == 3

    def test_metadata_detected_coords(self, xlsx_latlon):
        connector = ExcelXYConnector()
        meta = connector.metadata(str(xlsx_latlon))

        assert meta["detected_lon_col"] == "lon"
        assert meta["detected_lat_col"] == "lat"

    def test_metadata_crs(self, xlsx_latlon):
        connector = ExcelXYConnector()
        meta = connector.metadata(str(xlsx_latlon))

        assert meta["crs"] == "EPSG:4326"

    def test_metadata_extent(self, xlsx_latlon):
        connector = ExcelXYConnector()
        meta = connector.metadata(str(xlsx_latlon))

        extent = meta["extent"]
        assert extent is not None
        assert len(extent) == 4

    def test_metadata_sheet_names(self, xlsx_multiple_sheets):
        connector = ExcelXYConnector()
        meta = connector.metadata(str(xlsx_multiple_sheets))

        assert "sheet_names" in meta
        assert "Locations" in meta["sheet_names"]
        assert "OtherData" in meta["sheet_names"]

    def test_metadata_sheet_name(self, xlsx_multiple_sheets):
        connector = ExcelXYConnector()
        meta = connector.metadata(str(xlsx_multiple_sheets))

        assert meta["sheet_name"] == "Locations"  # First sheet by default


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestExcelXYErrors:
    """Validate error cases."""

    def test_nonexistent_file_raises(self, tmp_path):
        connector = ExcelXYConnector()
        nonexistent = tmp_path / "does_not_exist.xlsx"

        with pytest.raises(MaterializeError):
            connector.materialize(str(nonexistent), tmp_path)

    def test_empty_file_raises(self, tmp_path):
        connector = ExcelXYConnector()
        empty_file = tmp_path / "empty.xlsx"
        empty_file.write_bytes(b"")

        with pytest.raises(MaterializeError):
            connector.materialize(str(empty_file), tmp_path)

    def test_bad_sheet_name_raises(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        source_ref = f"{xlsx_latlon}::NonExistentSheet"

        with pytest.raises(MaterializeError) as exc_info:
            connector.materialize(source_ref, tmp_path)

        assert "Sheet not found" in str(exc_info.value)

    def test_header_only_xlsx(self, tmp_path):
        """Excel with only header row should be handled gracefully."""
        connector = ExcelXYConnector()
        header_only = tmp_path / "header_only.xlsx"
        _create_xlsx(
            header_only,
            headers=["name", "lat", "lon"],
            rows=[],
        )

        result = connector.materialize(str(header_only), tmp_path)
        # Should produce vector with 0 features
        assert result.artifact.type == ArtifactType.VECTOR
        assert result.artifact.spatial.feature_count == 0

    def test_metadata_nonexistent_file_raises(self, tmp_path):
        connector = ExcelXYConnector()
        nonexistent = tmp_path / "does_not_exist.xlsx"

        with pytest.raises(MaterializeError):
            connector.metadata(str(nonexistent))

    def test_explicit_invalid_column_raises(self, tmp_path):
        """Explicit column that doesn't exist should raise error."""
        path = tmp_path / "test.xlsx"
        _create_xlsx(
            path,
            headers=["a", "b", "c"],
            rows=[[1, 2, 3]],
        )

        connector = ExcelXYConnector()
        source_ref = f"{path}::Sheet1::nonexistent,also_missing"

        with pytest.raises(MaterializeError):
            connector.materialize(source_ref, tmp_path)


# ---------------------------------------------------------------------------
# SourceRef Support
# ---------------------------------------------------------------------------


class TestExcelXYSourceRef:
    """Validate SourceRef handling."""

    def test_sourceref_local(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        ref = SourceRef.local(str(xlsx_latlon))
        result = connector.materialize(ref, tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR

    def test_raw_string_path(self, xlsx_latlon, tmp_path):
        connector = ExcelXYConnector()
        result = connector.materialize(str(xlsx_latlon), tmp_path)

        assert result.artifact.type == ArtifactType.VECTOR


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestExcelXYCapabilities:
    """Validate connector capabilities."""

    def test_connector_name(self):
        connector = ExcelXYConnector()
        assert connector.name == "excel_xy"

    def test_capabilities(self):
        from quarry_core.connector import ConnectorCapability

        connector = ExcelXYConnector()
        caps = connector.capabilities

        assert ConnectorCapability.MATERIALIZE in caps
        assert ConnectorCapability.DISCOVER in caps
        assert ConnectorCapability.MATERIALIZE_LAZY in caps
        assert ConnectorCapability.METADATA_ONLY in caps

    def test_default_crs_constructor(self):
        """Test that default_crs can be customized."""
        connector = ExcelXYConnector(default_crs="EPSG:32633")
        assert connector.default_crs == "EPSG:32633"
