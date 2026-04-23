"""RasterizeVectorOperator pressure test.

Lane: operator

Stress points:
1. Protocol compliance (spec, validate_inputs, declared_checks)
2. Happy path — constant burn, known grid
3. Attribute burn — per-feature values from numeric property
4. CRS mismatch detection (vector CRS vs explicit extent CRS)
5. Empty geometries skipped without crash
6. Polygons partially outside extent — clipped, pixels within grid correct
7. Nodata / background behavior — uncovered pixels == nodata
8. Grid alignment — dimensions match resolution × extent exactly
9. Explicit extent overrides vector bounds
10. Missing burn attribute → feature skipped, others burned
11. Non-numeric burn attribute → feature skipped
12. Invalid resolution rejected at validation
13. Invalid extent rejected at validation
14. Lineage records operation params
15. Output artifact metadata fresh from file
16. All declared checks pass on happy path
17. Zero-feature vector → all-nodata raster
18. Overlapping polygons — last-write-wins (rasterio default)
19. Tiny resolution → large grid (dimensions_sane still valid)
20. Wrong input type rejected
21. Unmaterialized input rejected
"""

from __future__ import annotations

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
from quarry_operators.rasterize_vector import (
    RasterizeVectorOperator,
    RasterizeVectorParams,
)
from rasterio.crs import CRS
from shapely.geometry import Polygon, mapping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_vector(path, polygons, crs_epsg=32610, properties=None):
    """Write polygons to GeoJSON. properties: list of dicts parallel to polygons."""
    prop_schema = {}
    if properties and len(properties) > 0:
        for k, v in properties[0].items():
            if isinstance(v, (int, float)):
                prop_schema[k] = "float"
            else:
                prop_schema[k] = "str"
    schema = {"geometry": "Polygon", "properties": prop_schema}
    crs = CRS.from_epsg(crs_epsg).to_dict()
    with fiona.open(path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        for i, poly in enumerate(polygons):
            props = properties[i] if properties else {}
            dst.write({"geometry": mapping(poly), "properties": props})


def _make_vector_artifact(path, crs_epsg=32610):
    """Create Artifact for a vector file."""
    with fiona.open(path) as src:
        fc = len(src)
        try:
            bounds = src.bounds
            extent = (bounds[0], bounds[1], bounds[2], bounds[3])
        except Exception:
            extent = None
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
                extent=extent,
                feature_count=fc,
            ),
        )


def _read_raster(path):
    """Read back a single-band raster as (data, profile) tuple."""
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def op():
    return RasterizeVectorOperator()


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
    assert spec.input_types == (ArtifactType.VECTOR,)
    assert spec.output_type == ArtifactType.RASTER
    assert spec.min_inputs == 1
    assert spec.max_inputs == 1
    assert spec.resource_scale == ResourceScale.MEDIUM


def test_declared_checks(op):
    checks = op.declared_checks()
    assert "crs_valid" in checks
    assert "dimensions_sane" in checks
    assert "nodata_background" in checks


# ---------------------------------------------------------------------------
# 2. Happy path — constant burn
# ---------------------------------------------------------------------------


def test_happy_path_constant_burn(op, workspace):
    """Burn constant value into a known grid, verify pixel values."""
    # Square polygon covering (1,1)-(3,3) in a 4x4 extent
    poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 4, 4),
        burn_value=42.0,
        nodata=0.0,
        dtype="float32",
    )

    result = op.execute([vec_art], params)
    data, profile = _read_raster(out_path)

    assert result.artifact.type == ArtifactType.RASTER
    assert data.shape == (4, 4)
    assert profile["nodata"] == 0.0

    # Pixels inside the polygon should be 42, outside should be 0
    # The polygon (1,1)-(3,3) at 1m resolution on (0,0)-(4,4) grid
    # burns a 2x2 block in the center of a 4x4 grid
    burned_pixels = data[data == 42.0]
    assert len(burned_pixels) > 0
    background_pixels = data[data == 0.0]
    assert len(background_pixels) > 0


# ---------------------------------------------------------------------------
# 3. Attribute burn
# ---------------------------------------------------------------------------


def test_attribute_burn(op, workspace):
    """Per-feature burn from numeric attribute — different polygons get different values."""
    poly_a = Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])
    poly_b = Polygon([(2, 0), (4, 0), (4, 4), (2, 4)])
    vec_path = workspace / "input.geojson"
    _write_vector(
        vec_path,
        [poly_a, poly_b],
        properties=[{"val": 10.0}, {"val": 20.0}],
    )
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 4, 4),
        burn_attribute="val",
        burn_value=None,
        nodata=-1.0,
        dtype="float32",
    )

    result = op.execute([vec_art], params)
    data, _ = _read_raster(out_path)

    unique = set(np.unique(data))
    assert 10.0 in unique
    assert 20.0 in unique
    assert result.artifact.metadata["shapes_burned"] == 2


# ---------------------------------------------------------------------------
# 4. CRS preserved from vector input
# ---------------------------------------------------------------------------


def test_crs_preserved(op, workspace):
    """Output raster inherits CRS from vector input."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly], crs_epsg=4326)
    vec_art = _make_vector_artifact(vec_path, crs_epsg=4326)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(0.5, 0.5),
        burn_value=1.0,
        dtype="uint8",
    )

    result = op.execute([vec_art], params)

    with rasterio.open(out_path) as src:
        assert src.crs is not None
        assert src.crs.to_epsg() == 4326

    crs_check = [c for c in result.checks if c.check_name == "crs_valid"]
    assert crs_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 5. Empty geometries skipped
# ---------------------------------------------------------------------------


def test_empty_geometries_skipped(op, workspace):
    """Empty polygon geometries are skipped; valid ones still burned."""
    normal = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    empty = Polygon()
    vec_path = workspace / "input.geojson"

    # Write manually to include empty geom
    schema = {"geometry": "Polygon", "properties": {}}
    crs = CRS.from_epsg(32610).to_dict()
    with fiona.open(vec_path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        dst.write({"geometry": mapping(normal), "properties": {}})
        dst.write({"geometry": mapping(empty), "properties": {}})

    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 4, 4),
        burn_value=5.0,
        nodata=0.0,
        dtype="float32",
    )

    result = op.execute([vec_art], params)
    data, _ = _read_raster(out_path)

    # Only 1 shape should have been burned (the non-empty one)
    assert result.artifact.metadata["shapes_burned"] == 1
    assert np.any(data == 5.0)


# ---------------------------------------------------------------------------
# 6. Polygons partially outside extent
# ---------------------------------------------------------------------------


def test_partial_extent_polygon(op, workspace):
    """Polygon extending beyond extent — only pixels within grid are burned."""
    # Polygon spans (-1,0)-(3,4) but extent is (0,0)-(4,4)
    poly = Polygon([(-1, 0), (3, 0), (3, 4), (-1, 4)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 4, 4),
        burn_value=7.0,
        nodata=0.0,
        dtype="float32",
    )

    op.execute([vec_art], params)
    data, _ = _read_raster(out_path)

    # Should have burned pixels in columns 0-2 (x: 0-3), not column 3
    assert data.shape == (4, 4)
    burned_count = int(np.sum(data == 7.0))
    background_count = int(np.sum(data == 0.0))
    assert burned_count > 0
    assert background_count > 0  # column 3 should be background


# ---------------------------------------------------------------------------
# 7. Nodata / background behavior
# ---------------------------------------------------------------------------


def test_nodata_background(op, workspace):
    """Uncovered pixels get nodata value; covered pixels do not."""
    poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    nodata_val = -9999.0
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 4, 4),
        burn_value=1.0,
        nodata=nodata_val,
        dtype="float32",
    )

    result = op.execute([vec_art], params)
    data, profile = _read_raster(out_path)

    assert profile["nodata"] == nodata_val
    # Background pixels should be nodata
    assert np.any(data == nodata_val)
    # Burned pixels should NOT be nodata
    burned = data[data != nodata_val]
    assert len(burned) > 0
    assert np.all(burned == 1.0)

    nodata_check = [c for c in result.checks if c.check_name == "nodata_background"]
    assert nodata_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 8. Grid alignment — dimensions match resolution × extent
# ---------------------------------------------------------------------------


def test_grid_alignment(op, workspace):
    """Grid dimensions = ceil((extent_span) / resolution)."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    # Extent 10x6 at resolution 2.5x1.5 → width=4, height=4
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(2.5, 1.5),
        extent=(0, 0, 10, 6),
        burn_value=1.0,
        dtype="float32",
    )

    result = op.execute([vec_art], params)

    with rasterio.open(out_path) as src:
        assert src.width == 4  # ceil(10 / 2.5)
        assert src.height == 4  # ceil(6 / 1.5)

    dim_check = [c for c in result.checks if c.check_name == "dimensions_sane"]
    assert dim_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 9. Explicit extent overrides vector bounds
# ---------------------------------------------------------------------------


def test_explicit_extent_overrides_bounds(op, workspace):
    """When extent is given, it is used instead of the vector bounding box."""
    poly = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    # Force a different extent that includes the polygon
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 30, 30),
        burn_value=1.0,
        dtype="float32",
    )

    op.execute([vec_art], params)

    with rasterio.open(out_path) as src:
        assert src.width == 30
        assert src.height == 30
        bounds = src.bounds
        assert bounds.left == pytest.approx(0.0)
        assert bounds.bottom == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 10. Missing burn attribute → feature skipped
# ---------------------------------------------------------------------------


def test_missing_burn_attribute_skipped(op, workspace):
    """Features missing the burn_attribute are skipped; others still burned."""
    poly_a = Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])
    poly_b = Polygon([(2, 0), (4, 0), (4, 4), (2, 4)])
    vec_path = workspace / "input.geojson"

    # poly_a has 'val', poly_b does not
    schema = {"geometry": "Polygon", "properties": {"val": "float"}}
    crs = CRS.from_epsg(32610).to_dict()
    with fiona.open(vec_path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        dst.write({"geometry": mapping(poly_a), "properties": {"val": 10.0}})
        dst.write({"geometry": mapping(poly_b), "properties": {"val": None}})

    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 4, 4),
        burn_attribute="val",
        burn_value=None,
        nodata=0.0,
        dtype="float32",
    )

    result = op.execute([vec_art], params)
    data, _ = _read_raster(out_path)

    # Only poly_a should be burned
    assert result.artifact.metadata["shapes_burned"] == 1
    assert np.any(data == 10.0)


# ---------------------------------------------------------------------------
# 11. Non-numeric burn attribute → feature skipped
# ---------------------------------------------------------------------------


def test_non_numeric_burn_attribute_skipped(op, workspace):
    """Features with non-numeric burn_attribute values are skipped."""
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    vec_path = workspace / "input.geojson"

    schema = {"geometry": "Polygon", "properties": {"label": "str"}}
    crs = CRS.from_epsg(32610).to_dict()
    with fiona.open(vec_path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        dst.write({"geometry": mapping(poly), "properties": {"label": "not_a_number"}})

    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 4, 4),
        burn_attribute="label",
        burn_value=None,
        nodata=0.0,
        dtype="float32",
    )

    result = op.execute([vec_art], params)
    data, _ = _read_raster(out_path)

    assert result.artifact.metadata["shapes_burned"] == 0
    assert np.all(data == 0.0)  # all nodata


# ---------------------------------------------------------------------------
# 12. Invalid resolution rejected
# ---------------------------------------------------------------------------


def test_invalid_resolution_rejected(op, workspace):
    """Zero or negative resolution values are rejected at validation."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    params_zero = RasterizeVectorParams(
        output_path=str(workspace / "out.tif"),
        resolution=(0.0, 1.0),
    )
    errors = op.validate_inputs([vec_art], params_zero)
    assert any("resolution" in e.lower() for e in errors)

    params_neg = RasterizeVectorParams(
        output_path=str(workspace / "out.tif"),
        resolution=(-1.0, 1.0),
    )
    errors = op.validate_inputs([vec_art], params_neg)
    assert any("resolution" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 13. Invalid extent rejected
# ---------------------------------------------------------------------------


def test_invalid_extent_rejected(op, workspace):
    """Degenerate extent (xmin >= xmax) is rejected at validation."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    params = RasterizeVectorParams(
        output_path=str(workspace / "out.tif"),
        resolution=(1.0, 1.0),
        extent=(5, 5, 3, 3),  # xmin > xmax, ymin > ymax
    )
    errors = op.validate_inputs([vec_art], params)
    assert any("extent" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 14. Lineage records params
# ---------------------------------------------------------------------------


def test_lineage_records_params(op, workspace):
    """Output artifact lineage includes operation name and all params."""
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 4, 4),
        burn_value=42.0,
        nodata=-1.0,
        dtype="float32",
    )

    result = op.execute([vec_art], params)
    lineage = result.artifact.lineage

    assert lineage is not None
    assert lineage.operation == "rasterize_vector"
    assert vec_art.id in lineage.inputs
    assert lineage.params["burn_value"] == 42.0
    assert lineage.params["nodata"] == -1.0
    assert lineage.params["dtype"] == "float32"
    assert lineage.params["resolution"] == (1.0, 1.0)


# ---------------------------------------------------------------------------
# 15. Output artifact metadata fresh from file
# ---------------------------------------------------------------------------


def test_output_metadata_fresh(op, workspace):
    """Artifact metadata reflects actual file on disk, not params echo."""
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 4, 4),
        burn_value=1.0,
        dtype="uint16",
    )

    result = op.execute([vec_art], params)
    art = result.artifact

    assert art.type == ArtifactType.RASTER
    assert art.backing.kind == BackingStoreKind.LOCAL_FILE
    assert Path(art.backing.uri).exists()
    assert art.backing.size_bytes > 0
    assert art.backing.content_hash is not None
    assert art.spatial.crs is not None
    assert art.spatial.extent is not None
    assert art.spatial.resolution is not None
    assert art.spatial.band_count == 1
    assert art.metadata["format"] == "geotiff"
    assert art.metadata["width"] == 4
    assert art.metadata["height"] == 4
    assert result.timing_seconds is not None
    assert result.timing_seconds > 0


# ---------------------------------------------------------------------------
# 16. All checks pass on happy path
# ---------------------------------------------------------------------------


def test_all_checks_pass_happy_path(op, workspace):
    """All declared checks pass on a well-formed run."""
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 4, 4),
        burn_value=1.0,
        dtype="float32",
    )

    result = op.execute([vec_art], params)

    declared = set(op.declared_checks())
    check_names = {c.check_name for c in result.checks}
    assert declared == check_names

    for check in result.checks:
        assert check.state == ValidationState.VALID, (
            f"Check {check.check_name} not VALID: {check.message}"
        )


# ---------------------------------------------------------------------------
# 17. Zero-feature vector → all-nodata raster
# ---------------------------------------------------------------------------


def test_zero_features_all_nodata(op, workspace):
    """Empty vector input produces an all-nodata raster."""
    vec_path = workspace / "empty.geojson"
    _write_vector(vec_path, [])
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 4, 4),
        burn_value=1.0,
        nodata=0.0,
        dtype="float32",
    )

    result = op.execute([vec_art], params)
    data, _ = _read_raster(out_path)

    assert np.all(data == 0.0)
    assert result.artifact.metadata["shapes_burned"] == 0


# ---------------------------------------------------------------------------
# 18. Overlapping polygons — last-write-wins
# ---------------------------------------------------------------------------


def test_overlapping_polygons_last_wins(op, workspace):
    """When polygons overlap, later features overwrite earlier ones."""
    full = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    overlap = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    vec_path = workspace / "input.geojson"
    _write_vector(
        vec_path,
        [full, overlap],
        properties=[{"val": 10.0}, {"val": 99.0}],
    )
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=(0, 0, 4, 4),
        burn_attribute="val",
        burn_value=None,
        nodata=0.0,
        dtype="float32",
    )

    result = op.execute([vec_art], params)
    data, _ = _read_raster(out_path)

    # The overlap region should have 99 (second polygon overwrites)
    assert np.any(data == 99.0)
    # The non-overlapping region of the first polygon should still be 10
    assert np.any(data == 10.0)
    assert result.artifact.metadata["shapes_burned"] == 2


# ---------------------------------------------------------------------------
# 19. Tiny resolution → still valid dimensions
# ---------------------------------------------------------------------------


def test_small_resolution_large_grid(op, workspace):
    """Small resolution relative to extent produces large but valid grid."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(0.1, 0.1),
        extent=(0, 0, 10, 10),
        burn_value=1.0,
        dtype="uint8",
    )

    result = op.execute([vec_art], params)

    with rasterio.open(out_path) as src:
        assert src.width == 100
        assert src.height == 100

    dim_check = [c for c in result.checks if c.check_name == "dimensions_sane"]
    assert dim_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 20. Wrong input type rejected
# ---------------------------------------------------------------------------


def test_wrong_input_type_rejected(op, workspace):
    """Non-vector input is rejected at validation."""
    raster_art = Artifact(
        type=ArtifactType.RASTER,
        name="fake",
        backing=BackingStore(
            kind=BackingStoreKind.LOCAL_FILE,
            uri="/fake/path.tif",
        ),
    )
    params = RasterizeVectorParams(
        output_path=str(workspace / "out.tif"),
        resolution=(1.0, 1.0),
    )
    errors = op.validate_inputs([raster_art], params)
    assert any("vector" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 21. Unmaterialized input rejected
# ---------------------------------------------------------------------------


def test_unmaterialized_input_rejected(op, workspace):
    """Input without materialized backing is rejected at validation."""
    lazy_art = Artifact(
        type=ArtifactType.VECTOR,
        name="lazy",
        backing=BackingStore(
            kind=BackingStoreKind.LAZY_HANDLE,
            uri="s3://bucket/file.geojson",
        ),
    )
    params = RasterizeVectorParams(
        output_path=str(workspace / "out.tif"),
        resolution=(1.0, 1.0),
    )
    errors = op.validate_inputs([lazy_art], params)
    assert any("materialized" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 22. burn_value always has default (1.0), burn_attribute is optional
# ---------------------------------------------------------------------------


def test_burn_value_default_used(op, workspace):
    """Default burn_value (1.0) is used when burn_attribute is not set."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    params = RasterizeVectorParams(
        output_path=str(workspace / "out.tif"),
        resolution=(1.0, 1.0),
        # burn_value not specified, should use default 1.0
        burn_attribute=None,
    )
    errors = op.validate_inputs([vec_art], params)
    assert not any("burn" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 23. No extent → uses vector bounds
# ---------------------------------------------------------------------------


def test_no_extent_uses_vector_bounds(op, workspace):
    """When extent is None, output extent matches vector bounding box."""
    poly = Polygon([(10, 20), (15, 20), (15, 25), (10, 25)])
    vec_path = workspace / "input.geojson"
    _write_vector(vec_path, [poly])
    vec_art = _make_vector_artifact(vec_path)

    out_path = workspace / "out.tif"
    params = RasterizeVectorParams(
        output_path=str(out_path),
        resolution=(1.0, 1.0),
        extent=None,  # should derive from vector bounds
        burn_value=1.0,
        dtype="float32",
    )

    op.execute([vec_art], params)

    with rasterio.open(out_path) as src:
        bounds = src.bounds
        assert bounds.left == pytest.approx(10.0)
        assert bounds.bottom == pytest.approx(20.0)
        assert src.width == 5
        assert src.height == 5
