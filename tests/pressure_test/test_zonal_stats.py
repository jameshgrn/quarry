"""ZonalStatsOperator pressure test.

Lane: operator

Stress points:
1. Happy path — correct stats for known raster + polygon zones
2. Mismatched CRS rejected at validation
3. Empty geometries produce NaN rows, row count preserved
4. All-nodata zone produces NaN row
5. Partial overlap — zone partially outside raster gets stats for covered pixels
6. Zone fully outside raster gets NaN row
7. Schema preservation — all stat columns always present
8. Row count == input feature count (stable)
9. Single-pixel zone gets exact values
10. Multi-band raster — band param selects correct band
11. Nodata pixels excluded from statistics
12. Operator protocol compliance (spec, validate_inputs, declared_checks)
13. OperatorResult contains valid TABLE artifact with fresh metadata
14. Lineage records operation params
"""

from __future__ import annotations

import csv
from pathlib import Path

import fiona
import numpy as np
import pytest
import rasterio
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    SpatialDescriptor,
    ValidationState,
    content_hash,
)
from quarry_core.operator import Operator, ResourceScale
from quarry_operators.zonal_stats import (
    STAT_COLUMNS,
    ZonalStatsOperator,
    ZonalStatsParams,
)
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, mapping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_raster(path, data, crs_epsg=32610, nodata=None, extent=None):
    """Write a GeoTIFF. data shape: (bands, rows, cols) or (rows, cols) for single band."""
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    bands, nrows, ncols = data.shape
    if extent is None:
        extent = (0, 0, ncols, nrows)
    xmin, ymin, xmax, ymax = extent
    transform = from_bounds(xmin, ymin, xmax, ymax, ncols, nrows)
    meta = {
        "driver": "GTiff",
        "height": nrows,
        "width": ncols,
        "count": bands,
        "dtype": str(data.dtype),
        "crs": CRS.from_epsg(crs_epsg),
        "transform": transform,
    }
    if nodata is not None:
        meta["nodata"] = nodata
    with rasterio.open(path, "w", **meta) as dst:
        for b in range(bands):
            dst.write(data[b], b + 1)


def _write_vector(path, polygons, crs_epsg=32610, properties=None):
    """Write polygons to a GeoJSON file. polygons: list of shapely Polygons."""
    schema = {"geometry": "Polygon", "properties": {}}
    if properties and len(properties) > 0:
        for k, v in properties[0].items():
            schema["properties"][k] = "str"
    crs = CRS.from_epsg(crs_epsg).to_dict()
    with fiona.open(path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        for i, poly in enumerate(polygons):
            props = properties[i] if properties else {}
            dst.write({"geometry": mapping(poly), "properties": props})


def _make_raster_artifact(path, crs_epsg=32610):
    """Create Artifact for a raster file."""
    with rasterio.open(path) as src:
        bounds = src.bounds
        return Artifact(
            type=ArtifactType.RASTER,
            name=path.stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(path),
                size_bytes=path.stat().st_size,
                content_hash=content_hash(path),
            ),
            spatial=SpatialDescriptor(
                crs=str(src.crs),
                extent=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                resolution=(src.res[0], src.res[1]),
                band_count=src.count,
            ),
        )


def _make_vector_artifact(path, crs_epsg=32610):
    """Create Artifact for a vector file."""
    with fiona.open(path) as src:
        fc = len(src)
        bounds = src.bounds
        return Artifact(
            type=ArtifactType.VECTOR,
            name=Path(path).stem,
            backing=BackingStore(
                kind=BackingStoreKind.LOCAL_FILE,
                uri=str(path),
                size_bytes=Path(path).stat().st_size,
                content_hash=content_hash(Path(path)),
            ),
            spatial=SpatialDescriptor(
                crs=f"EPSG:{crs_epsg}",
                extent=(bounds[0], bounds[1], bounds[2], bounds[3]),
                feature_count=fc,
            ),
        )


def _read_csv(path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def op():
    return ZonalStatsOperator()


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Protocol compliance
# ---------------------------------------------------------------------------


def test_protocol_compliance(op):
    """Operator satisfies the Operator protocol."""
    assert isinstance(op, Operator)


def test_spec(op):
    spec = op.spec
    assert spec.input_types == (ArtifactType.RASTER, ArtifactType.VECTOR)
    assert spec.output_type == ArtifactType.TABLE
    assert spec.min_inputs == 2
    assert spec.max_inputs == 2
    assert spec.resource_scale == ResourceScale.MEDIUM


def test_declared_checks(op):
    checks = op.declared_checks()
    assert "row_count_matches" in checks
    assert "schema_complete" in checks
    assert "crs_valid" in checks


# ---------------------------------------------------------------------------
# 2. Happy path — known values
# ---------------------------------------------------------------------------


def test_happy_path_known_values(op, workspace):
    """4x4 raster with two non-overlapping rectangular zones. Stats hand-verifiable."""
    # Raster: 4x4 grid, values 1..16
    # Extent: (0,0)-(4,4), pixel size 1x1
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    # Zone A: left half (x: 0-2, y: 0-4) → columns 0,1
    # Zone B: right half (x: 2-4, y: 0-4) → columns 2,3
    zone_a = Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])
    zone_b = Polygon([(2, 0), (4, 0), (4, 4), (2, 4)])
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, [zone_a, zone_b])

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    result = op.execute([raster_art, vector_art], params)

    assert result.artifact.type == ArtifactType.TABLE
    rows = _read_csv(workspace / "out.csv")
    assert len(rows) == 2

    # Figure out which pixels fall in each zone by reading back
    # Zone A covers columns 0,1 → pixels: col0 + col1 for each row
    # Row 0 (top of raster, y=4→3): 1,2  Row 1: 5,6  Row 2: 9,10  Row 3: 13,14
    a_pixels = [1, 2, 5, 6, 9, 10, 13, 14]
    b_pixels = [3, 4, 7, 8, 11, 12, 15, 16]

    row_a = rows[0]
    assert int(row_a["count"]) == 8
    assert float(row_a["mean"]) == pytest.approx(np.mean(a_pixels))
    assert float(row_a["min"]) == pytest.approx(min(a_pixels))
    assert float(row_a["max"]) == pytest.approx(max(a_pixels))
    assert float(row_a["sum"]) == pytest.approx(sum(a_pixels))
    assert float(row_a["std"]) == pytest.approx(float(np.std(a_pixels)))

    row_b = rows[1]
    assert int(row_b["count"]) == 8
    assert float(row_b["mean"]) == pytest.approx(np.mean(b_pixels))


# ---------------------------------------------------------------------------
# 3. Mismatched CRS rejected
# ---------------------------------------------------------------------------


def test_crs_mismatch_rejected(op, workspace):
    """Raster in EPSG:32610, vector in EPSG:4326 → validation error."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, crs_epsg=32610)
    raster_art = _make_raster_artifact(raster_path, crs_epsg=32610)

    zone = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, [zone], crs_epsg=4326)
    vector_art = _make_vector_artifact(vector_path, crs_epsg=4326)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    errors = op.validate_inputs([raster_art, vector_art], params)
    assert any("CRS mismatch" in e for e in errors)


# ---------------------------------------------------------------------------
# 4. Empty geometry → NaN row, row count preserved
# ---------------------------------------------------------------------------


def test_empty_geometry_nan_row(op, workspace):
    """An empty polygon produces NaN stats but still appears as a row."""
    data = np.ones((4, 4), dtype=np.float32) * 5.0
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    normal = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    empty = Polygon()  # empty geometry
    vector_path = workspace / "zones.geojson"
    # Write manually since fiona may reject empty geom
    schema = {"geometry": "Polygon", "properties": {}}
    crs = CRS.from_epsg(32610).to_dict()
    with fiona.open(vector_path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        dst.write({"geometry": mapping(normal), "properties": {}})
        dst.write({"geometry": mapping(empty), "properties": {}})

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")
    assert len(rows) == 2

    # Second row should be NaN
    row_empty = rows[1]
    assert row_empty["count"] == "nan"
    assert row_empty["mean"] == "nan"


# ---------------------------------------------------------------------------
# 5. All-nodata zone → NaN row
# ---------------------------------------------------------------------------


def test_all_nodata_zone_nan(op, workspace):
    """A zone covering only nodata pixels produces NaN stats."""
    data = np.array([[1, 2], [-9999, -9999], [5, 6], [7, 8]], dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, nodata=-9999, extent=(0, 0, 2, 4))

    # Zone covering only row 1 (the nodata row)
    # Raster row 1 is at y: 3→2 (top-down pixel order, extent 0-4 tall)
    nodata_zone = Polygon([(0, 2), (2, 2), (2, 3), (0, 3)])
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, [nodata_zone])

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")
    assert len(rows) == 1
    assert rows[0]["count"] == "nan"


# ---------------------------------------------------------------------------
# 6. Partial overlap — zone extends beyond raster
# ---------------------------------------------------------------------------


def test_partial_overlap(op, workspace):
    """Zone partially outside raster → stats computed only for covered pixels."""
    data = np.ones((4, 4), dtype=np.float32) * 10.0
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    # Zone extends from x:-2 to x:2, only x:0-2 overlaps
    partial = Polygon([(-2, 0), (2, 0), (2, 4), (-2, 4)])
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, [partial])

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")
    assert len(rows) == 1

    # Should have ~8 pixels (columns 0,1 × 4 rows), all value 10
    row = rows[0]
    count = int(row["count"])
    assert count > 0
    assert float(row["mean"]) == pytest.approx(10.0)
    assert float(row["std"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. Fully outside raster → NaN row
# ---------------------------------------------------------------------------


def test_fully_outside_nan(op, workspace):
    """Zone entirely outside raster extent → NaN row."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    outside = Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, [outside])

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")
    assert len(rows) == 1
    assert rows[0]["count"] == "nan"


# ---------------------------------------------------------------------------
# 8. Schema always complete
# ---------------------------------------------------------------------------


def test_schema_always_complete(op, workspace):
    """Every row has all stat columns regardless of data."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    zone = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    outside = Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, [zone, outside])

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    result = op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    expected_keys = {"zone_id", *STAT_COLUMNS}
    for row in rows:
        assert set(row.keys()) == expected_keys

    # Check that schema_complete check passed
    schema_check = [c for c in result.checks if c.check_name == "schema_complete"]
    assert len(schema_check) == 1
    assert schema_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 9. Row count stability — always equals feature count
# ---------------------------------------------------------------------------


def test_row_count_equals_feature_count(op, workspace):
    """Output row count matches input feature count, even with mixed coverage."""
    data = np.ones((4, 4), dtype=np.float32) * 3.0
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    zones = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),  # fully inside
        Polygon([(100, 100), (200, 100), (200, 200), (100, 200)]),  # fully outside
        Polygon([(-1, 0), (1, 0), (1, 2), (-1, 2)]),  # partial overlap
    ]
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, zones)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    result = op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")
    assert len(rows) == 3

    row_count_check = [c for c in result.checks if c.check_name == "row_count_matches"]
    assert len(row_count_check) == 1
    assert row_count_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 10. Single-pixel zone → exact values
# ---------------------------------------------------------------------------


def test_single_pixel_zone(op, workspace):
    """A zone covering exactly one pixel returns that pixel's value for all stats."""
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    # Pixel at row=0, col=0 has value 1, covers (0,3)-(1,4)
    single = Polygon([(0.1, 3.1), (0.9, 3.1), (0.9, 3.9), (0.1, 3.9)])
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, [single])

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")
    assert len(rows) == 1
    row = rows[0]
    assert int(row["count"]) == 1
    assert float(row["mean"]) == pytest.approx(1.0)
    assert float(row["min"]) == pytest.approx(1.0)
    assert float(row["max"]) == pytest.approx(1.0)
    assert float(row["std"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11. Multi-band raster — band selection
# ---------------------------------------------------------------------------


def test_multi_band_selection(op, workspace):
    """band param selects correct band from multi-band raster."""
    band1 = np.ones((4, 4), dtype=np.float32) * 10.0
    band2 = np.ones((4, 4), dtype=np.float32) * 20.0
    data = np.stack([band1, band2])  # shape (2, 4, 4)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    zone = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, [zone])

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    # Band 1 → all 10s
    params1 = ZonalStatsParams(output_path=str(workspace / "out1.csv"), band=1)
    op.execute([raster_art, vector_art], params1)
    rows1 = _read_csv(workspace / "out1.csv")
    assert float(rows1[0]["mean"]) == pytest.approx(10.0)

    # Band 2 → all 20s
    params2 = ZonalStatsParams(output_path=str(workspace / "out2.csv"), band=2)
    op.execute([raster_art, vector_art], params2)
    rows2 = _read_csv(workspace / "out2.csv")
    assert float(rows2[0]["mean"]) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 12. Nodata exclusion
# ---------------------------------------------------------------------------


def test_nodata_excluded_from_stats(op, workspace):
    """Nodata pixels are excluded from statistics."""
    data = np.array(
        [[1, 2, -9999, 4], [5, -9999, 7, 8], [9, 10, 11, -9999], [13, 14, 15, 16]],
        dtype=np.float32,
    )
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, nodata=-9999, extent=(0, 0, 4, 4))

    zone = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, [zone])

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    valid_pixels = [1, 2, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16]
    row = rows[0]
    assert int(row["count"]) == len(valid_pixels)
    assert float(row["sum"]) == pytest.approx(sum(valid_pixels))
    assert float(row["mean"]) == pytest.approx(np.mean(valid_pixels))


# ---------------------------------------------------------------------------
# 13. NaN nodata handling
# ---------------------------------------------------------------------------


def test_nan_nodata_excluded(op, workspace):
    """NaN nodata pixels are excluded from statistics."""
    data = np.array([[1, 2], [np.nan, 4]], dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, nodata=float("nan"), extent=(0, 0, 2, 2))

    zone = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, [zone])

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    row = rows[0]
    assert int(row["count"]) == 3
    assert float(row["sum"]) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# 14. Lineage records params
# ---------------------------------------------------------------------------


def test_lineage_records_params(op, workspace):
    """Output artifact lineage includes operation name and params."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    zone = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, [zone])

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"), band=1)
    result = op.execute([raster_art, vector_art], params)

    assert result.artifact.lineage is not None
    assert result.artifact.lineage.operation == "zonal_stats"
    assert result.artifact.lineage.params["band"] == 1
    assert set(result.artifact.lineage.inputs) == {raster_art.id, vector_art.id}


# ---------------------------------------------------------------------------
# 15. Output artifact metadata
# ---------------------------------------------------------------------------


def test_output_artifact_metadata(op, workspace):
    """Output artifact has correct type, backing, spatial descriptor."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    zones = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
    ]
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, zones)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    result = op.execute([raster_art, vector_art], params)

    art = result.artifact
    assert art.type == ArtifactType.TABLE
    assert art.backing.kind == BackingStoreKind.LOCAL_FILE
    assert Path(art.backing.uri).exists()
    assert art.spatial.feature_count == 2
    assert art.spatial.crs is not None
    assert art.metadata["format"] == "csv"
    assert result.timing_seconds is not None
    assert result.timing_seconds > 0


# ---------------------------------------------------------------------------
# 16. Validation: wrong input count
# ---------------------------------------------------------------------------


def test_validate_wrong_input_count(op, workspace):
    """Validation rejects 0 or 1 inputs."""
    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    errors = op.validate_inputs([], params)
    assert any("Exactly 2" in e for e in errors)


# ---------------------------------------------------------------------------
# 17. Validation: wrong input types
# ---------------------------------------------------------------------------


def test_validate_wrong_types(op, workspace):
    """Validation rejects non-raster first input or non-vector second input."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    raster_art = _make_raster_artifact(raster_path)
    # Pass two rasters instead of raster + vector
    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    errors = op.validate_inputs([raster_art, raster_art], params)
    assert any("vector" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 18. zone_id_field param
# ---------------------------------------------------------------------------


def test_zone_id_field(op, workspace):
    """zone_id_field extracts zone ID from feature properties."""
    data = np.ones((4, 4), dtype=np.float32) * 7.0
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    zones = [
        Polygon([(0, 0), (2, 0), (2, 4), (0, 4)]),
        Polygon([(2, 0), (4, 0), (4, 4), (2, 4)]),
    ]
    props = [{"name": "north"}, {"name": "south"}]
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, zones, properties=props)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"), zone_id_field="name")
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")
    assert rows[0]["zone_id"] == "north"
    assert rows[1]["zone_id"] == "south"


# ---------------------------------------------------------------------------
# 19. Checks all pass on happy path
# ---------------------------------------------------------------------------


def test_all_checks_pass_happy_path(op, workspace):
    """All declared checks pass on a well-formed run."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    zone = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    vector_path = workspace / "zones.geojson"
    _write_vector(vector_path, [zone])

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = ZonalStatsParams(output_path=str(workspace / "out.csv"))
    result = op.execute([raster_art, vector_art], params)

    for check in result.checks:
        assert check.state == ValidationState.VALID, (
            f"Check {check.check_name} not VALID: {check.message}"
        )
