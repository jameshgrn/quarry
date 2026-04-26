"""BufferOperator pressure test.

Lane: operator

Stress points:
1. Protocol compliance (spec, validate_inputs, declared_checks)
2. Happy path — positive buffer expands polygon
3. Negative buffer shrinks polygon
4. Negative buffer collapses tiny polygon to empty → geometry_valid WARN
5. Point buffer → circular polygon
6. Line buffer → polygon
7. Empty geometry preserved (null in output)
8. Feature count preserved (5 in → 5 out)
9. Properties preserved unchanged through buffer
10. CRS preserved in output
11. Lineage records all params (distance, resolution, cap_style, join_style)
12. Output artifact metadata fresh from file
13. Validation: raster input rejected
14. Validation: zero distance rejected
15. Validation: invalid cap_style rejected
16. Validation: invalid join_style rejected
17. Validation: unmaterialized input rejected
18. Cap style square — rectangular corners, larger area than round
19. Resolution parameter — low resolution → fewer vertices
20. All checks pass on happy path
21. Timing recorded > 0
"""

from __future__ import annotations

import math
from pathlib import Path

import fiona
import pytest
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
from quarry_operators.buffer import BufferOperator, BufferParams
from rasterio.crs import CRS
from shapely.geometry import LineString, Point, Polygon, mapping, shape

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_vector(path, geometries, crs_epsg=32610, properties=None, schema_props=None):
    """Write geometries to GeoJSON. Infers geometry type from first geom."""
    if not geometries:
        geom_type = "Polygon"
    else:
        first_non_empty = next((g for g in geometries if not g.is_empty), None)
        if first_non_empty is None:
            geom_type = "Polygon"
        else:
            gt = first_non_empty.geom_type
            if gt in ("Point", "LineString"):
                geom_type = gt
            else:
                geom_type = "Polygon"

    if schema_props is None:
        schema_props = {}
        if properties and len(properties) > 0:
            for k in properties[0]:
                schema_props[k] = "str"

    schema = {"geometry": geom_type, "properties": schema_props}
    crs = CRS.from_epsg(crs_epsg).to_dict()
    with fiona.open(path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        for i, geom in enumerate(geometries):
            props = properties[i] if properties else {}
            dst.write({"geometry": mapping(geom), "properties": props})


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


def _read_output(path):
    """Read GeoJSON output features as list of dicts with geometry + properties."""
    with fiona.open(path) as src:
        return [
            {
                "geometry": dict(f["geometry"]) if f["geometry"] else None,
                "properties": dict(f.get("properties", {})),
            }
            for f in src
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def op():
    return BufferOperator()


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
    assert spec.output_type == ArtifactType.VECTOR
    assert spec.min_inputs == 1
    assert spec.max_inputs == 1
    assert spec.resource_scale == ResourceScale.LIGHT


def test_declared_checks(op):
    checks = op.declared_checks()
    assert "crs_valid" in checks
    assert "feature_count_preserved" in checks
    assert "geometry_valid" in checks


# ---------------------------------------------------------------------------
# 2. Happy path — positive buffer expands polygon
# ---------------------------------------------------------------------------


def test_happy_path_positive_buffer(op, workspace):
    """Buffering a polygon outward produces a larger area."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    input_area = poly.area

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly])

    art = _make_vector_artifact(input_path)
    params = BufferParams(output_path=str(workspace / "out.geojson"), distance=5.0)
    result = op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    output_area = shape(features[0]["geometry"]).area
    assert output_area > input_area
    assert result.artifact.type == ArtifactType.VECTOR


# ---------------------------------------------------------------------------
# 3. Negative buffer shrinks polygon
# ---------------------------------------------------------------------------


def test_negative_buffer_shrinks_polygon(op, workspace):
    """Negative buffer on a large polygon produces smaller area."""
    poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    input_area = poly.area

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly])

    art = _make_vector_artifact(input_path)
    params = BufferParams(output_path=str(workspace / "out.geojson"), distance=-5.0)
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    output_area = shape(features[0]["geometry"]).area
    assert output_area < input_area


# ---------------------------------------------------------------------------
# 4. Negative buffer collapses tiny polygon → geometry_valid WARN
# ---------------------------------------------------------------------------


def test_negative_buffer_collapses_tiny_polygon(op, workspace):
    """Small polygon with large negative distance collapses to empty; geometry_valid is WARN."""
    tiny = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [tiny])

    art = _make_vector_artifact(input_path)
    params = BufferParams(output_path=str(workspace / "out.geojson"), distance=-10.0)
    result = op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    assert features[0]["geometry"] is None

    geom_check = [c for c in result.checks if c.check_name == "geometry_valid"]
    assert len(geom_check) == 1
    assert geom_check[0].state == ValidationState.WARN


# ---------------------------------------------------------------------------
# 5. Point buffer → polygon
# ---------------------------------------------------------------------------


def test_point_buffer_creates_polygon(op, workspace):
    """Buffering a point produces a circular polygon with area ≈ pi*r²."""
    pt = Point(500000, 4500000)
    distance = 100.0

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [pt])

    art = _make_vector_artifact(input_path)
    params = BufferParams(output_path=str(workspace / "out.geojson"), distance=distance)
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    assert features[0]["geometry"] is not None

    output_geom = shape(features[0]["geometry"])
    assert output_geom.geom_type in ("Polygon", "MultiPolygon")

    expected_area = math.pi * distance**2
    assert output_geom.area == pytest.approx(expected_area, rel=0.01)


# ---------------------------------------------------------------------------
# 6. Line buffer → polygon
# ---------------------------------------------------------------------------


def test_line_buffer_creates_polygon(op, workspace):
    """Buffering a linestring produces a polygon."""
    line = LineString([(0, 0), (100, 0)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [line])

    art = _make_vector_artifact(input_path)
    params = BufferParams(output_path=str(workspace / "out.geojson"), distance=10.0)
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    assert features[0]["geometry"] is not None

    output_geom = shape(features[0]["geometry"])
    assert output_geom.geom_type in ("Polygon", "MultiPolygon")
    assert output_geom.area > 0


# ---------------------------------------------------------------------------
# 7. Empty geometry preserved (null in output)
# ---------------------------------------------------------------------------


def test_empty_geometry_preserved(op, workspace):
    """Empty geometry in input stays null in output."""
    normal = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    empty = Polygon()

    input_path = workspace / "input.geojson"
    schema = {"geometry": "Polygon", "properties": {"fid": "str"}}
    crs = CRS.from_epsg(32610).to_dict()
    with fiona.open(input_path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        dst.write({"geometry": mapping(normal), "properties": {"fid": "A"}})
        dst.write({"geometry": mapping(empty), "properties": {"fid": "B"}})

    art = _make_vector_artifact(input_path)
    params = BufferParams(output_path=str(workspace / "out.geojson"), distance=5.0)
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 2

    feat_b = [f for f in features if f["properties"]["fid"] == "B"][0]
    assert feat_b["geometry"] is None


# ---------------------------------------------------------------------------
# 8. Feature count preserved (5 in → 5 out)
# ---------------------------------------------------------------------------


def test_feature_count_preserved(op, workspace):
    """5 features in → 5 features out, feature_count_preserved check VALID."""
    polys = [
        Polygon([(i * 20, 0), (i * 20 + 10, 0), (i * 20 + 10, 10), (i * 20, 10)]) for i in range(5)
    ]
    props = [{"fid": str(i)} for i in range(5)]

    input_path = workspace / "input.geojson"
    _write_vector(input_path, polys, properties=props)

    art = _make_vector_artifact(input_path)
    params = BufferParams(output_path=str(workspace / "out.geojson"), distance=2.0)
    result = op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 5

    fc_check = [c for c in result.checks if c.check_name == "feature_count_preserved"]
    assert len(fc_check) == 1
    assert fc_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 9. Properties preserved unchanged
# ---------------------------------------------------------------------------


def test_properties_preserved(op, workspace):
    """All feature properties are carried through the buffer unchanged."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    input_path = workspace / "input.geojson"
    _write_vector(
        input_path,
        [poly],
        properties=[{"name": "site_alpha", "code": "S01", "value": "42.5"}],
        schema_props={"name": "str", "code": "str", "value": "str"},
    )

    art = _make_vector_artifact(input_path)
    params = BufferParams(output_path=str(workspace / "out.geojson"), distance=3.0)
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    props = features[0]["properties"]
    assert props["name"] == "site_alpha"
    assert props["code"] == "S01"
    assert props["value"] == "42.5"


# ---------------------------------------------------------------------------
# 10. CRS preserved
# ---------------------------------------------------------------------------


def test_crs_preserved(op, workspace):
    """Output CRS matches input CRS."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly], crs_epsg=32610)

    art = _make_vector_artifact(input_path, crs_epsg=32610)
    params = BufferParams(output_path=str(workspace / "out.geojson"), distance=5.0)
    result = op.execute([art], params)

    assert result.artifact.spatial.crs is not None

    crs_check = [c for c in result.checks if c.check_name == "crs_valid"]
    assert len(crs_check) == 1
    assert crs_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 11. Lineage records all params
# ---------------------------------------------------------------------------


def test_lineage_records_params(op, workspace):
    """Lineage includes distance, resolution, cap_style, join_style."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly])

    art = _make_vector_artifact(input_path)
    params = BufferParams(
        output_path=str(workspace / "out.geojson"),
        distance=7.5,
        resolution=8,
        cap_style="flat",
        join_style="mitre",
    )
    result = op.execute([art], params)

    lineage = result.artifact.lineage
    assert lineage is not None
    assert lineage.operation == "buffer"
    assert lineage.params["distance"] == 7.5
    assert lineage.params["resolution"] == 8
    assert lineage.params["cap_style"] == "flat"
    assert lineage.params["join_style"] == "mitre"
    assert art.id in lineage.inputs


# ---------------------------------------------------------------------------
# 12. Output artifact metadata fresh from file
# ---------------------------------------------------------------------------


def test_output_artifact_metadata(op, workspace):
    """Output artifact has correct type, backing, spatial from actual file."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly], properties=[{"a": "1"}])

    art = _make_vector_artifact(input_path)
    params = BufferParams(output_path=str(workspace / "out.geojson"), distance=5.0)
    result = op.execute([art], params)

    out = result.artifact
    assert out.type == ArtifactType.VECTOR
    assert out.backing.kind == BackingStoreKind.LOCAL_FILE
    assert Path(out.backing.uri).exists()
    assert out.spatial.feature_count == 1
    assert out.spatial.crs is not None
    assert out.spatial.extent is not None
    assert out.metadata["format"] == "geojson"


# ---------------------------------------------------------------------------
# 13. Validation: raster input rejected
# ---------------------------------------------------------------------------


def test_validate_raster_input_rejected(op):
    """Validation rejects a raster artifact."""
    raster_art = Artifact(
        type=ArtifactType.RASTER,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = BufferParams(output_path="/fake/out.geojson", distance=5.0)
    errors = op.validate_inputs([raster_art], params)
    assert any("vector" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 14. Validation: zero distance rejected
# ---------------------------------------------------------------------------


def test_validate_zero_distance_rejected(op, workspace):
    """Validation rejects distance=0."""
    vector_art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = BufferParams(output_path="/fake/out.geojson", distance=0.0)
    errors = op.validate_inputs([vector_art], params)
    assert any("zero" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 15. Validation: invalid cap_style rejected
# ---------------------------------------------------------------------------


def test_validate_invalid_cap_style_rejected(op):
    """Validation rejects an unknown cap_style."""
    vector_art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = BufferParams(output_path="/fake/out.geojson", distance=5.0, cap_style="pointy")
    errors = op.validate_inputs([vector_art], params)
    assert any("cap_style" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 16. Validation: invalid join_style rejected
# ---------------------------------------------------------------------------


def test_validate_invalid_join_style_rejected(op):
    """Validation rejects an unknown join_style."""
    vector_art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = BufferParams(output_path="/fake/out.geojson", distance=5.0, join_style="zigzag")
    errors = op.validate_inputs([vector_art], params)
    assert any("join_style" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 17. Validation: unmaterialized input rejected
# ---------------------------------------------------------------------------


def test_validate_unmaterialized_input_rejected(op):
    """Validation rejects a lazy (unmaterialized) input."""
    lazy_art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LAZY_HANDLE, uri=""),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = BufferParams(output_path="/fake/out.geojson", distance=5.0)
    errors = op.validate_inputs([lazy_art], params)
    assert any("materialized" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 18. Cap style square — larger area than round
# ---------------------------------------------------------------------------


def test_cap_style_square(op, workspace):
    """Square cap on a line buffer produces rectangular corners → larger area than round."""
    line = LineString([(0, 0), (100, 0)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [line])

    art = _make_vector_artifact(input_path)

    # Round cap
    params_round = BufferParams(
        output_path=str(workspace / "out_round.geojson"),
        distance=10.0,
        cap_style="round",
    )
    op.execute([art], params_round)
    feat_round = _read_output(workspace / "out_round.geojson")
    area_round = shape(feat_round[0]["geometry"]).area

    # Square cap
    params_square = BufferParams(
        output_path=str(workspace / "out_square.geojson"),
        distance=10.0,
        cap_style="square",
    )
    op.execute([art], params_square)
    feat_square = _read_output(workspace / "out_square.geojson")
    area_square = shape(feat_square[0]["geometry"]).area

    assert area_square > area_round


# ---------------------------------------------------------------------------
# 19. Resolution parameter — low resolution → fewer vertices
# ---------------------------------------------------------------------------


def test_resolution_parameter(op, workspace):
    """Low resolution (4 segments) produces fewer vertices on a point buffer."""
    pt = Point(500000, 4500000)

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [pt])

    art = _make_vector_artifact(input_path)

    # High resolution (default 16)
    params_hi = BufferParams(
        output_path=str(workspace / "out_hi.geojson"),
        distance=100.0,
        resolution=16,
    )
    op.execute([art], params_hi)
    feat_hi = _read_output(workspace / "out_hi.geojson")
    geom_hi = shape(feat_hi[0]["geometry"])

    # Low resolution (4)
    params_lo = BufferParams(
        output_path=str(workspace / "out_lo.geojson"),
        distance=100.0,
        resolution=4,
    )
    op.execute([art], params_lo)
    feat_lo = _read_output(workspace / "out_lo.geojson")
    geom_lo = shape(feat_lo[0]["geometry"])

    # Extract vertex count from exterior ring
    def vertex_count(geom):
        if geom.geom_type == "MultiPolygon":
            return sum(len(p.exterior.coords) for p in geom.geoms)
        return len(geom.exterior.coords)

    assert vertex_count(geom_lo) < vertex_count(geom_hi)


# ---------------------------------------------------------------------------
# 20. All checks pass on happy path
# ---------------------------------------------------------------------------


def test_all_checks_pass_happy_path(op, workspace):
    """All declared checks pass on a well-formed run."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly])

    art = _make_vector_artifact(input_path)
    params = BufferParams(output_path=str(workspace / "out.geojson"), distance=5.0)
    result = op.execute([art], params)

    for check in result.checks:
        assert check.state == ValidationState.VALID, (
            f"Check {check.check_name} not VALID: {check.message}"
        )


# ---------------------------------------------------------------------------
# 21. Timing recorded > 0
# ---------------------------------------------------------------------------


def test_timing_recorded(op, workspace):
    """Result timing_seconds is present and positive."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly])

    art = _make_vector_artifact(input_path)
    params = BufferParams(output_path=str(workspace / "out.geojson"), distance=5.0)
    result = op.execute([art], params)

    assert result.timing_seconds is not None
    assert result.timing_seconds > 0
