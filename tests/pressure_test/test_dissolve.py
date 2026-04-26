"""DissolveOperator pressure test.

Lane: operator

Stress points:
1. Protocol compliance (spec, validate_inputs, declared_checks)
2. Dissolve all — two polygons merge into single geometry
3. Dissolve by field — two groups
4. Dissolve all — overlapping polygons merge into single Polygon
5. Dissolve all — disjoint polygons produce MultiPolygon
6. Feature count reduced check — output <= input, check VALID
7. Properties: _dissolved_count present per group
8. Properties: by field value preserved in output
9. Missing by field → __null__ group
10. Single feature dissolve — trivial case
11. CRS preserved — output CRS matches input
12. Lineage records by field
13. Output metadata fresh from file
14. Validation: wrong input type (raster rejected)
15. Validation: unmaterialized input rejected
16. Validation: wrong input count (0 or 2 inputs rejected)
17. All checks pass happy path
18. Empty geometry in group — omitted, geometry_valid WARN
19. Mixed by values — 3 groups verify correct grouping
20. Timing recorded
"""

from __future__ import annotations

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
from quarry_operators.dissolve import DissolveOperator, DissolveParams
from rasterio.crs import CRS
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_vector(path, geometries, crs_epsg=32610, properties=None, schema_props=None):
    """Write geometries to GeoJSON. Infers geometry type from first geom."""
    if not geometries:
        geom_type = "Polygon"
    else:
        first_non_empty = next((g for g in geometries if not g.is_empty), None)
        geom_type = "Polygon" if first_non_empty is None else first_non_empty.geom_type

    if schema_props is None:
        schema_props = {}
        if properties and len(properties) > 0:
            for k, v in properties[0].items():
                schema_props[k] = "int" if isinstance(v, int) else "str"

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
            {"geometry": dict(f["geometry"]), "properties": dict(f.get("properties", {}))}
            for f in src
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def op():
    return DissolveOperator()


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
    assert spec.resource_scale == ResourceScale.MEDIUM


def test_declared_checks(op):
    checks = op.declared_checks()
    assert "crs_valid" in checks
    assert "feature_count_reduced" in checks
    assert "geometry_valid" in checks


# ---------------------------------------------------------------------------
# 2. Dissolve all — two polygons merge
# ---------------------------------------------------------------------------


def test_dissolve_all_two_polygons_merge(op, workspace):
    """Two adjacent polygons, by=None → 1 feature, geometry is union."""
    poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly_a, poly_b])

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    assert result.artifact.type == ArtifactType.VECTOR

    # Verify geometry is the union of both
    expected = unary_union([poly_a, poly_b])
    actual = shape(features[0]["geometry"])
    assert actual.equals_exact(expected, tolerance=1e-6)


# ---------------------------------------------------------------------------
# 3. Dissolve by field — two groups
# ---------------------------------------------------------------------------


def test_dissolve_by_field_two_groups(op, workspace):
    """4 features with field 'type' having 2 values → 2 output features."""
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
        Polygon([(1, 2), (2, 2), (2, 3), (1, 3)]),
    ]
    props = [
        {"type": "forest"},
        {"type": "forest"},
        {"type": "water"},
        {"type": "water"},
    ]

    input_path = workspace / "input.geojson"
    _write_vector(input_path, polys, properties=props)

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"), by="type")
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 2

    type_values = {f["properties"]["type"] for f in features}
    assert type_values == {"forest", "water"}


# ---------------------------------------------------------------------------
# 4. Dissolve all — overlapping polygons → single Polygon
# ---------------------------------------------------------------------------


def test_dissolve_all_overlapping_polygons(op, workspace):
    """Overlapping polygons merge into a single Polygon (not Multi)."""
    poly_a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    poly_b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly_a, poly_b])

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"))
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1

    actual = shape(features[0]["geometry"])
    expected = unary_union([poly_a, poly_b])
    assert actual.geom_type == "Polygon"
    assert actual.equals_exact(expected, tolerance=1e-6)


# ---------------------------------------------------------------------------
# 5. Dissolve all — disjoint polygons → MultiPolygon
# ---------------------------------------------------------------------------


def test_dissolve_all_disjoint_polygons(op, workspace):
    """Disjoint polygons dissolved together → MultiPolygon."""
    poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly_a, poly_b])

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"))
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1

    actual = shape(features[0]["geometry"])
    assert isinstance(actual, MultiPolygon)
    assert len(actual.geoms) == 2


# ---------------------------------------------------------------------------
# 6. Feature count reduced check — VALID
# ---------------------------------------------------------------------------


def test_feature_count_reduced_check(op, workspace):
    """Dissolve reduces feature count; feature_count_reduced check is VALID."""
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
    ]

    input_path = workspace / "input.geojson"
    _write_vector(input_path, polys)

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([art], params)

    check = [c for c in result.checks if c.check_name == "feature_count_reduced"]
    assert len(check) == 1
    assert check[0].state == ValidationState.VALID
    assert result.artifact.spatial.feature_count <= 3


# ---------------------------------------------------------------------------
# 7. Properties: _dissolved_count present
# ---------------------------------------------------------------------------


def test_dissolved_count_present(op, workspace):
    """Each output feature has _dissolved_count property."""
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
    ]
    props = [{"type": "a"}, {"type": "b"}]

    input_path = workspace / "input.geojson"
    _write_vector(input_path, polys, properties=props)

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"), by="type")
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    for f in features:
        assert "_dissolved_count" in f["properties"]
        assert f["properties"]["_dissolved_count"] == 1


# ---------------------------------------------------------------------------
# 8. Properties: by field value preserved
# ---------------------------------------------------------------------------


def test_by_field_value_preserved(op, workspace):
    """Grouped field value appears in output properties."""
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
    ]
    props = [{"zone": "north"}, {"zone": "north"}]

    input_path = workspace / "input.geojson"
    _write_vector(input_path, polys, properties=props)

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"), by="zone")
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    assert features[0]["properties"]["zone"] == "north"
    assert features[0]["properties"]["_dissolved_count"] == 2


# ---------------------------------------------------------------------------
# 9. Missing by field → __null__ group
# ---------------------------------------------------------------------------


def test_missing_by_field_null_group(op, workspace):
    """Features without the grouping field are collected into __null__ group."""
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        Polygon([(3, 0), (4, 0), (4, 1), (3, 1)]),  # adjacent to previous → single Polygon
    ]
    # First has the field, second and third do not
    props = [{"zone": "north"}, {"zone": None}, {"zone": None}]

    input_path = workspace / "input.geojson"
    _write_vector(input_path, polys, properties=props)

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"), by="zone")
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 2

    null_group = [f for f in features if f["properties"]["zone"] is None]
    assert len(null_group) == 1
    assert null_group[0]["properties"]["_dissolved_count"] == 2

    north_group = [f for f in features if f["properties"]["zone"] == "north"]
    assert len(north_group) == 1
    assert north_group[0]["properties"]["_dissolved_count"] == 1


# ---------------------------------------------------------------------------
# 10. Single feature dissolve
# ---------------------------------------------------------------------------


def test_single_feature_dissolve(op, workspace):
    """1 feature in → 1 feature out (trivial dissolve)."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly])

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"))
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    assert features[0]["properties"]["_dissolved_count"] == 1

    actual = shape(features[0]["geometry"])
    assert actual.equals_exact(poly, tolerance=1e-6)


# ---------------------------------------------------------------------------
# 11. CRS preserved
# ---------------------------------------------------------------------------


def test_crs_preserved(op, workspace):
    """Output CRS matches input CRS."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly], crs_epsg=4326)

    art = _make_vector_artifact(input_path, crs_epsg=4326)
    params = DissolveParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([art], params)

    # Check artifact CRS
    assert result.artifact.spatial.crs is not None

    # Check actual file CRS
    with fiona.open(workspace / "out.geojson") as src:
        assert src.crs is not None

    # crs_valid check passes
    crs_check = [c for c in result.checks if c.check_name == "crs_valid"]
    assert crs_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 12. Lineage records by field
# ---------------------------------------------------------------------------


def test_lineage_records_by_field(op, workspace):
    """Lineage params include the 'by' field value."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly], properties=[{"zone": "A"}])

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"), by="zone")
    result = op.execute([art], params)

    assert result.artifact.lineage is not None
    assert result.artifact.lineage.operation == "dissolve"
    assert result.artifact.lineage.params["by"] == "zone"
    assert art.id in result.artifact.lineage.inputs


def test_lineage_by_none(op, workspace):
    """Lineage records by=None when dissolving all."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly])

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([art], params)

    assert result.artifact.lineage.params["by"] is None


# ---------------------------------------------------------------------------
# 13. Output metadata fresh from file
# ---------------------------------------------------------------------------


def test_output_metadata_fresh(op, workspace):
    """Output artifact metadata is read fresh from the output file."""
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
    ]

    input_path = workspace / "input.geojson"
    _write_vector(input_path, polys)

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([art], params)

    out_art = result.artifact
    assert out_art.type == ArtifactType.VECTOR
    assert out_art.backing.kind == BackingStoreKind.LOCAL_FILE
    assert Path(out_art.backing.uri).exists()
    assert out_art.spatial.feature_count == 1
    assert out_art.spatial.crs is not None
    assert out_art.spatial.extent is not None
    assert out_art.metadata["format"] == "geojson"
    assert out_art.metadata["input_feature_count"] == 2
    assert out_art.metadata["output_feature_count"] == 1


# ---------------------------------------------------------------------------
# 14. Validation: wrong input type
# ---------------------------------------------------------------------------


def test_validate_wrong_input_type(op):
    """Raster input rejected at validation."""
    raster_art = Artifact(
        type=ArtifactType.RASTER,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = DissolveParams(output_path="/fake/out.geojson")
    errors = op.validate_inputs([raster_art], params)
    assert any("vector" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 15. Validation: unmaterialized input
# ---------------------------------------------------------------------------


def test_validate_unmaterialized_input(op):
    """Lazy (unmaterialized) input rejected at validation."""
    lazy_art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LAZY_HANDLE, uri="s3://bucket/key"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = DissolveParams(output_path="/fake/out.geojson")
    errors = op.validate_inputs([lazy_art], params)
    assert any("materialized" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 16. Validation: wrong input count
# ---------------------------------------------------------------------------


def test_validate_wrong_input_count_zero(op):
    """Validation rejects 0 inputs."""
    params = DissolveParams(output_path="/fake/out.geojson")
    errors = op.validate_inputs([], params)
    assert any("1" in e for e in errors)


def test_validate_wrong_input_count_two(op, workspace):
    """Validation rejects 2 inputs."""
    art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = DissolveParams(output_path=str(workspace / "out.geojson"))
    errors = op.validate_inputs([art, art], params)
    assert any("1" in e for e in errors)


# ---------------------------------------------------------------------------
# 17. All checks pass happy path
# ---------------------------------------------------------------------------


def test_all_checks_pass_happy_path(op, workspace):
    """All declared checks pass on a well-formed dissolve run."""
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
    ]

    input_path = workspace / "input.geojson"
    _write_vector(input_path, polys)

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([art], params)

    for check in result.checks:
        assert check.state == ValidationState.VALID, (
            f"Check {check.check_name} not VALID: {check.message}"
        )


# ---------------------------------------------------------------------------
# 18. Empty geometry in group — omitted, geometry_valid WARN
# ---------------------------------------------------------------------------


def test_empty_geometry_group_omitted(op, workspace):
    """Group with only empty geometries is omitted; geometry_valid check is WARN."""
    normal = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    empty = Polygon()

    input_path = workspace / "input.geojson"
    schema = {"geometry": "Polygon", "properties": {"zone": "str"}}
    crs = CRS.from_epsg(32610).to_dict()
    with fiona.open(input_path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        dst.write({"geometry": mapping(normal), "properties": {"zone": "a"}})
        dst.write({"geometry": mapping(empty), "properties": {"zone": "b"}})

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"), by="zone")
    result = op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    # "a" group kept, "b" group omitted (empty geometry)
    assert len(features) == 1
    assert features[0]["properties"]["zone"] == "a"

    geom_check = [c for c in result.checks if c.check_name == "geometry_valid"]
    assert geom_check[0].state == ValidationState.WARN


# ---------------------------------------------------------------------------
# 19. Mixed by values — 3 groups
# ---------------------------------------------------------------------------


def test_mixed_by_values_three_groups(op, workspace):
    """Three distinct group values produce 3 output features with correct counts."""
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
        Polygon([(1, 2), (2, 2), (2, 3), (1, 3)]),  # adjacent to previous
        Polygon([(0, 4), (1, 4), (1, 5), (0, 5)]),
        Polygon([(1, 4), (2, 4), (2, 5), (1, 5)]),
        Polygon([(2, 4), (3, 4), (3, 5), (2, 5)]),
    ]
    props = [
        {"category": "urban"},
        {"category": "urban"},
        {"category": "forest"},
        {"category": "forest"},
        {"category": "wetland"},
        {"category": "wetland"},
        {"category": "wetland"},
    ]

    input_path = workspace / "input.geojson"
    _write_vector(input_path, polys, properties=props)

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"), by="category")
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 3

    by_category = {
        f["properties"]["category"]: f["properties"]["_dissolved_count"] for f in features
    }
    assert by_category["urban"] == 2
    assert by_category["forest"] == 2
    assert by_category["wetland"] == 3


# ---------------------------------------------------------------------------
# 20. Timing recorded
# ---------------------------------------------------------------------------


def test_timing_recorded(op, workspace):
    """Timing is recorded in result."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    input_path = workspace / "input.geojson"
    _write_vector(input_path, [poly])

    art = _make_vector_artifact(input_path)
    params = DissolveParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([art], params)

    assert result.timing_seconds is not None
    assert result.timing_seconds > 0
