"""SampleRasterOperator pressure test.

Lane: operator

Stress points:
1. Happy path — correct sampled values for known raster + point locations
2. Mismatched CRS rejected at validation
3. Points outside raster extent → NaN, row count preserved
4. Nodata cells → NaN for affected bands
5. Multiband raster — explicit band selection
6. All bands sampled when bands param is empty
7. Row count == input point count (stable)
8. Empty input layer → zero rows, checks WARN
9. Single point at pixel center → exact value
10. Point on raster edge boundary → valid sample
11. NaN nodata handling
12. Nodata override via params
13. Operator protocol compliance (spec, validate_inputs, declared_checks)
14. OperatorResult contains valid TABLE artifact with fresh metadata
15. Lineage records operation params
16. Schema always complete (point_id + band columns)
17. All checks pass on happy path
18. Validation rejects wrong input count
19. Validation rejects wrong input types
20. Mixed coverage — some points inside, some outside
"""

from __future__ import annotations

import csv
import math
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
from quarry_operators.sample_raster import (
    SampleRasterOperator,
    SampleRasterParams,
)
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Point, mapping

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


def _write_points(path, points, crs_epsg=32610):
    """Write point geometries to GeoJSON. points: list of shapely Points."""
    schema = {"geometry": "Point", "properties": {}}
    crs = CRS.from_epsg(crs_epsg).to_dict()
    with fiona.open(path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        for pt in points:
            dst.write({"geometry": mapping(pt), "properties": {}})


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
    return SampleRasterOperator()


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
    assert spec.resource_scale == ResourceScale.LIGHT


def test_declared_checks(op):
    checks = op.declared_checks()
    assert "row_count_matches" in checks
    assert "schema_complete" in checks
    assert "crs_valid" in checks


# ---------------------------------------------------------------------------
# 2. Happy path — known values
# ---------------------------------------------------------------------------


def test_happy_path_known_values(op, workspace):
    """4x4 raster, sample at known pixel centers, verify exact values."""
    # Raster: 4x4, values 1..16, extent (0,0)-(4,4), pixel size 1x1
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    # Pixel (row=0, col=0) value=1 → center at (0.5, 3.5)
    # Pixel (row=3, col=3) value=16 → center at (3.5, 0.5)
    points = [Point(0.5, 3.5), Point(3.5, 0.5)]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    result = op.execute([raster_art, vector_art], params)

    assert result.artifact.type == ArtifactType.TABLE
    rows = _read_csv(workspace / "out.csv")
    assert len(rows) == 2

    assert float(rows[0]["band_1"]) == pytest.approx(1.0)
    assert float(rows[1]["band_1"]) == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# 3. CRS mismatch rejected
# ---------------------------------------------------------------------------


def test_crs_mismatch_rejected(op, workspace):
    """Raster in EPSG:32610, vector in EPSG:4326 → validation error."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, crs_epsg=32610)
    raster_art = _make_raster_artifact(raster_path, crs_epsg=32610)

    vector_path = workspace / "points.geojson"
    _write_points(vector_path, [Point(1, 1)], crs_epsg=4326)
    vector_art = _make_vector_artifact(vector_path, crs_epsg=4326)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    errors = op.validate_inputs([raster_art, vector_art], params)
    assert any("CRS mismatch" in e for e in errors)


# ---------------------------------------------------------------------------
# 4. Points outside raster → NaN, row count preserved
# ---------------------------------------------------------------------------


def test_points_outside_raster_nan(op, workspace):
    """Points outside raster extent produce NaN values, row count preserved."""
    data = np.ones((4, 4), dtype=np.float32) * 42.0
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    points = [
        Point(2, 2),  # inside
        Point(100, 100),  # outside
        Point(-5, -5),  # outside
    ]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    result = op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    assert len(rows) == 3
    assert float(rows[0]["band_1"]) == pytest.approx(42.0)
    assert rows[1]["band_1"] == "nan"
    assert rows[2]["band_1"] == "nan"

    # Row count check passes
    rc = [c for c in result.checks if c.check_name == "row_count_matches"]
    assert rc[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 5. Nodata cells → NaN
# ---------------------------------------------------------------------------


def test_nodata_cells_nan(op, workspace):
    """Points landing on nodata pixels produce NaN."""
    data = np.array([[10, -9999], [-9999, 20]], dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, nodata=-9999, extent=(0, 0, 2, 2))

    # Pixel (0,0) val=10 center (0.5, 1.5); pixel (0,1) val=-9999 center (1.5, 1.5)
    points = [Point(0.5, 1.5), Point(1.5, 1.5)]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    assert float(rows[0]["band_1"]) == pytest.approx(10.0)
    assert rows[1]["band_1"] == "nan"


# ---------------------------------------------------------------------------
# 6. Multiband — explicit band selection
# ---------------------------------------------------------------------------


def test_multiband_explicit_selection(op, workspace):
    """Select specific bands from multiband raster."""
    band1 = np.ones((4, 4), dtype=np.float32) * 10.0
    band2 = np.ones((4, 4), dtype=np.float32) * 20.0
    band3 = np.ones((4, 4), dtype=np.float32) * 30.0
    data = np.stack([band1, band2, band3])
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    points = [Point(2, 2)]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    # Select only bands 1 and 3
    params = SampleRasterParams(
        output_path=str(workspace / "out.csv"),
        bands=[1, 3],
    )
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    assert len(rows) == 1
    assert float(rows[0]["band_1"]) == pytest.approx(10.0)
    assert float(rows[0]["band_3"]) == pytest.approx(30.0)
    assert "band_2" not in rows[0]


# ---------------------------------------------------------------------------
# 7. All bands when bands param empty
# ---------------------------------------------------------------------------


def test_all_bands_default(op, workspace):
    """Empty bands param samples all bands."""
    band1 = np.ones((2, 2), dtype=np.float32) * 5.0
    band2 = np.ones((2, 2), dtype=np.float32) * 15.0
    data = np.stack([band1, band2])
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 2, 2))

    points = [Point(1, 1)]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    assert float(rows[0]["band_1"]) == pytest.approx(5.0)
    assert float(rows[0]["band_2"]) == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# 8. Row count always equals point count
# ---------------------------------------------------------------------------


def test_row_count_equals_point_count(op, workspace):
    """Row count matches input point count with mixed inside/outside."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    points = [
        Point(1, 1),  # inside
        Point(3, 3),  # inside
        Point(100, 100),  # outside
        Point(-1, -1),  # outside
        Point(2, 2),  # inside
    ]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    result = op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")
    assert len(rows) == 5

    rc = [c for c in result.checks if c.check_name == "row_count_matches"]
    assert rc[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 9. Empty input layer → zero rows
# ---------------------------------------------------------------------------


def test_empty_input_layer(op, workspace):
    """Zero points → zero rows, schema check WARN."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    vector_path = workspace / "points.geojson"
    _write_points(vector_path, [])

    raster_art = _make_raster_artifact(raster_path)
    # Empty vector — can't compute bounds, so build artifact manually
    vector_art = Artifact(
        type=ArtifactType.VECTOR,
        name="points",
        backing=BackingStore(
            kind=BackingStoreKind.LOCAL_FILE,
            uri=str(vector_path),
            size_bytes=vector_path.stat().st_size,
            content_hash=content_hash(vector_path),
        ),
        spatial=SpatialDescriptor(
            crs="EPSG:32610",
            feature_count=0,
        ),
    )

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    result = op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")
    assert len(rows) == 0

    schema_check = [c for c in result.checks if c.check_name == "schema_complete"]
    assert schema_check[0].state == ValidationState.WARN


# ---------------------------------------------------------------------------
# 10. Single point at pixel center → exact value
# ---------------------------------------------------------------------------


def test_single_point_pixel_center(op, workspace):
    """One point at exact pixel center returns that pixel's value."""
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    # Pixel (row=2, col=1) value=10 → center at (1.5, 1.5)
    points = [Point(1.5, 1.5)]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    assert len(rows) == 1
    assert float(rows[0]["band_1"]) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 11. Point on raster boundary edge
# ---------------------------------------------------------------------------


def test_point_on_boundary_edge(op, workspace):
    """Point exactly at raster origin (0,0) is within bounds and samples correctly."""
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    # Point at (0.01, 0.01) — just inside bottom-left corner
    # Pixel (row=3, col=0) value=13
    points = [Point(0.01, 0.01)]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    assert len(rows) == 1
    val = float(rows[0]["band_1"])
    assert not math.isnan(val)
    assert val == pytest.approx(13.0)


# ---------------------------------------------------------------------------
# 12. NaN nodata handling
# ---------------------------------------------------------------------------


def test_nan_nodata_handling(op, workspace):
    """NaN nodata values produce NaN in output."""
    data = np.array([[1, 2], [np.nan, 4]], dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, nodata=float("nan"), extent=(0, 0, 2, 2))

    # Pixel (1,0) is NaN → center (0.5, 0.5)
    points = [Point(0.5, 1.5), Point(0.5, 0.5)]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    assert float(rows[0]["band_1"]) == pytest.approx(1.0)
    assert rows[1]["band_1"] == "nan"


# ---------------------------------------------------------------------------
# 13. Nodata override via params
# ---------------------------------------------------------------------------


def test_nodata_override(op, workspace):
    """nodata_value param overrides raster's native nodata."""
    # Raster has nodata=-9999 but we override to treat 42 as nodata
    data = np.array([[10, 42], [42, 20]], dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, nodata=-9999, extent=(0, 0, 2, 2))

    points = [Point(0.5, 1.5), Point(1.5, 1.5)]  # pixel (0,0)=10, pixel (0,1)=42
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    # Override: treat 42 as nodata
    params = SampleRasterParams(
        output_path=str(workspace / "out.csv"),
        nodata_value=42.0,
    )
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    assert float(rows[0]["band_1"]) == pytest.approx(10.0)
    assert rows[1]["band_1"] == "nan"


# ---------------------------------------------------------------------------
# 14. Lineage records params
# ---------------------------------------------------------------------------


def test_lineage_records_params(op, workspace):
    """Output artifact lineage includes operation name and params."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    points = [Point(2, 2)]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(
        output_path=str(workspace / "out.csv"),
        bands=[1],
    )
    result = op.execute([raster_art, vector_art], params)

    assert result.artifact.lineage is not None
    assert result.artifact.lineage.operation == "sample_raster"
    assert result.artifact.lineage.params["bands"] == [1]
    assert set(result.artifact.lineage.inputs) == {raster_art.id, vector_art.id}


# ---------------------------------------------------------------------------
# 15. Output artifact metadata
# ---------------------------------------------------------------------------


def test_output_artifact_metadata(op, workspace):
    """Output artifact has correct type, backing, spatial descriptor."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    points = [Point(1, 1), Point(3, 3)]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
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
# 16. Schema always complete
# ---------------------------------------------------------------------------


def test_schema_always_complete(op, workspace):
    """Every row has point_id + all band columns."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    points = [Point(2, 2), Point(100, 100)]  # one inside, one outside
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    result = op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    expected_keys = {"point_id", "band_1"}
    for row in rows:
        assert set(row.keys()) == expected_keys

    schema_check = [c for c in result.checks if c.check_name == "schema_complete"]
    assert schema_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 17. All checks pass on happy path
# ---------------------------------------------------------------------------


def test_all_checks_pass_happy_path(op, workspace):
    """All declared checks pass on a well-formed run."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    points = [Point(2, 2)]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    result = op.execute([raster_art, vector_art], params)

    for check in result.checks:
        assert check.state == ValidationState.VALID, (
            f"Check {check.check_name} not VALID: {check.message}"
        )


# ---------------------------------------------------------------------------
# 18. Validation rejects wrong input count
# ---------------------------------------------------------------------------


def test_validate_wrong_input_count(op, workspace):
    """Validation rejects 0 or 1 inputs."""
    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    errors = op.validate_inputs([], params)
    assert any("Exactly 2" in e for e in errors)


# ---------------------------------------------------------------------------
# 19. Validation rejects wrong input types
# ---------------------------------------------------------------------------


def test_validate_wrong_types(op, workspace):
    """Validation rejects non-raster first input or non-vector second input."""
    data = np.ones((4, 4), dtype=np.float32)
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    raster_art = _make_raster_artifact(raster_path)
    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    errors = op.validate_inputs([raster_art, raster_art], params)
    assert any("vector" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 20. Mixed coverage — point_id stable across inside/outside
# ---------------------------------------------------------------------------


def test_point_id_sequential(op, workspace):
    """point_id is sequential 0..N-1 regardless of sample success."""
    data = np.ones((4, 4), dtype=np.float32) * 7.0
    raster_path = workspace / "raster.tif"
    _write_raster(raster_path, data, extent=(0, 0, 4, 4))

    points = [Point(2, 2), Point(100, 100), Point(1, 1)]
    vector_path = workspace / "points.geojson"
    _write_points(vector_path, points)

    raster_art = _make_raster_artifact(raster_path)
    vector_art = _make_vector_artifact(vector_path)

    params = SampleRasterParams(output_path=str(workspace / "out.csv"))
    op.execute([raster_art, vector_art], params)
    rows = _read_csv(workspace / "out.csv")

    assert [r["point_id"] for r in rows] == ["0", "1", "2"]
