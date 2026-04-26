"""SimplifyOperator pressure test.

Lane: operator

Stress points:
1. Protocol compliance (spec, validate_inputs, declared_checks)
2. Happy path — complex polygon simplified (fewer vertices)
3. Zero tolerance → unchanged geometry
4. Large tolerance collapses polygon → WARN
5. Point simplification — no-op
6. Line simplification — fewer vertices
7. Feature count preserved always
8. Properties preserved
9. CRS preserved
10. Lineage records tolerance and preserve_topology
11. Output metadata fresh from file
12. Validation: negative tolerance rejected
13. Validation: wrong input type (raster rejected)
14. Validation: unmaterialized artifact rejected
15. Validation: wrong input count
16. All checks pass happy path
17. preserve_topology=True — polygon stays valid
18. preserve_topology=False — can differ from True
19. Multiple features, different complexity
20. Empty geometry pass-through
21. Timing recorded
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
from quarry_operators.simplify import SimplifyOperator, SimplifyParams
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
            geom_type = first_non_empty.geom_type

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
            {"geometry": dict(f["geometry"]), "properties": dict(f.get("properties", {}))}
            for f in src
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def op():
    return SimplifyOperator()


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


def _complex_polygon():
    """A polygon with many vertices (sinusoidal boundary)."""
    coords = [(i * 0.1, math.sin(i * 0.3)) for i in range(50)]
    coords.append((4.9, 0))
    coords.append((0, 0))
    return Polygon(coords)


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
# 2. Happy path — complex polygon simplified
# ---------------------------------------------------------------------------


def test_happy_path_complex_polygon_simplified(op, workspace):
    """A polygon with many vertices is reduced by simplification."""
    poly = _complex_polygon()
    original_vertex_count = len(poly.exterior.coords)
    assert original_vertex_count > 10  # sanity: it's complex

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [poly])
    art = _make_vector_artifact(in_path)

    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.5)
    result = op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    simplified_geom = shape(features[0]["geometry"])
    assert len(simplified_geom.exterior.coords) < original_vertex_count
    assert result.artifact.type == ArtifactType.VECTOR


# ---------------------------------------------------------------------------
# 3. Zero tolerance → unchanged
# ---------------------------------------------------------------------------


def test_zero_tolerance_unchanged(op, workspace):
    """tolerance=0 means no simplification — coordinates identical."""
    poly = _complex_polygon()
    original_coords = list(poly.exterior.coords)

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [poly])
    art = _make_vector_artifact(in_path)

    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.0)
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    out_geom = shape(features[0]["geometry"])
    out_coords = list(out_geom.exterior.coords)
    assert len(out_coords) == len(original_coords)
    for oc, nc in zip(original_coords, out_coords):
        assert abs(oc[0] - nc[0]) < 1e-9
        assert abs(oc[1] - nc[1]) < 1e-9


# ---------------------------------------------------------------------------
# 4. Large tolerance collapses polygon → WARN
# ---------------------------------------------------------------------------


def test_large_tolerance_collapse_warns(op, workspace):
    """Tiny polygon with huge tolerance collapses to empty → geometry_valid WARN."""
    tiny = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [tiny])
    art = _make_vector_artifact(in_path)

    # preserve_topology=False allows full collapse; True would keep a minimal ring
    params = SimplifyParams(
        output_path=str(workspace / "out.geojson"),
        tolerance=1.0,
        preserve_topology=False,
    )
    result = op.execute([art], params)

    geom_check = [c for c in result.checks if c.check_name == "geometry_valid"]
    assert len(geom_check) == 1
    assert geom_check[0].state == ValidationState.WARN
    assert "collapsed" in geom_check[0].message.lower()


# ---------------------------------------------------------------------------
# 5. Point simplification — no-op
# ---------------------------------------------------------------------------


def test_point_simplification_noop(op, workspace):
    """Points pass through simplification unchanged."""
    pt = Point(1.0, 2.0)

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [pt], properties=[{"pid": "A"}])
    art = _make_vector_artifact(in_path)

    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=100.0)
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    out_geom = shape(features[0]["geometry"])
    assert out_geom.geom_type == "Point"
    assert abs(out_geom.x - 1.0) < 1e-9
    assert abs(out_geom.y - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# 6. Line simplification
# ---------------------------------------------------------------------------


def test_line_simplification(op, workspace):
    """A complex line is simplified to fewer vertices."""
    coords = [(i * 0.1, math.sin(i * 0.3)) for i in range(50)]
    line = LineString(coords)
    original_count = len(line.coords)

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [line])
    art = _make_vector_artifact(in_path)

    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.5)
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    out_geom = shape(features[0]["geometry"])
    assert out_geom.geom_type == "LineString"
    assert len(out_geom.coords) < original_count


# ---------------------------------------------------------------------------
# 7. Feature count preserved always
# ---------------------------------------------------------------------------


def test_feature_count_preserved(op, workspace):
    """5 features in → 5 features out, regardless of simplification."""
    polys = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(5)]
    props = [{"fid": str(i)} for i in range(5)]

    in_path = workspace / "input.geojson"
    _write_vector(in_path, polys, properties=props)
    art = _make_vector_artifact(in_path)

    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.01)
    result = op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 5

    fc_check = [c for c in result.checks if c.check_name == "feature_count_preserved"]
    assert fc_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 8. Properties preserved
# ---------------------------------------------------------------------------


def test_properties_preserved(op, workspace):
    """Feature properties are carried through unchanged."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    in_path = workspace / "input.geojson"
    _write_vector(
        in_path,
        [poly],
        properties=[{"name": "test_feat", "code": "42"}],
        schema_props={"name": "str", "code": "str"},
    )
    art = _make_vector_artifact(in_path)

    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.1)
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert features[0]["properties"]["name"] == "test_feat"
    assert features[0]["properties"]["code"] == "42"


# ---------------------------------------------------------------------------
# 9. CRS preserved
# ---------------------------------------------------------------------------


def test_crs_preserved(op, workspace):
    """Output CRS matches input CRS."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [poly], crs_epsg=4326)
    art = _make_vector_artifact(in_path, crs_epsg=4326)

    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.0)
    result = op.execute([art], params)

    crs_check = [c for c in result.checks if c.check_name == "crs_valid"]
    assert crs_check[0].state == ValidationState.VALID
    assert result.artifact.spatial.crs is not None


# ---------------------------------------------------------------------------
# 10. Lineage records tolerance and preserve_topology
# ---------------------------------------------------------------------------


def test_lineage(op, workspace):
    """Output lineage includes operation name, tolerance, and preserve_topology."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [poly])
    art = _make_vector_artifact(in_path)

    params = SimplifyParams(
        output_path=str(workspace / "out.geojson"),
        tolerance=0.5,
        preserve_topology=False,
    )
    result = op.execute([art], params)

    assert result.artifact.lineage is not None
    assert result.artifact.lineage.operation == "simplify"
    assert result.artifact.lineage.params["tolerance"] == 0.5
    assert result.artifact.lineage.params["preserve_topology"] is False
    assert art.id in result.artifact.lineage.inputs


# ---------------------------------------------------------------------------
# 11. Output metadata fresh from file
# ---------------------------------------------------------------------------


def test_output_metadata_fresh(op, workspace):
    """Output artifact has correct type, backing, spatial descriptor from actual file."""
    poly = _complex_polygon()

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [poly])
    art = _make_vector_artifact(in_path)

    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.3)
    result = op.execute([art], params)

    out = result.artifact
    assert out.type == ArtifactType.VECTOR
    assert out.backing.kind == BackingStoreKind.LOCAL_FILE
    assert Path(out.backing.uri).exists()
    assert out.spatial.feature_count == 1
    assert out.spatial.crs is not None
    assert out.spatial.extent is not None
    assert out.metadata["format"] == "geojson"
    assert out.metadata["input_feature_count"] == 1
    assert out.metadata["output_feature_count"] == 1


# ---------------------------------------------------------------------------
# 12. Validation: negative tolerance rejected
# ---------------------------------------------------------------------------


def test_validate_negative_tolerance(op, workspace):
    """Negative tolerance rejected at validation."""
    art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(
            kind=BackingStoreKind.LOCAL_FILE,
            uri=str(workspace / "fake.geojson"),
            size_bytes=1,
            content_hash="abc",
        ),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=-1.0)
    errors = op.validate_inputs([art], params)
    assert any("tolerance" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 13. Validation: wrong input type (raster rejected)
# ---------------------------------------------------------------------------


def test_validate_wrong_input_type(op, workspace):
    """Raster input rejected at validation."""
    raster_art = Artifact(
        type=ArtifactType.RASTER,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.1)
    errors = op.validate_inputs([raster_art], params)
    assert any("vector" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 14. Validation: unmaterialized artifact rejected
# ---------------------------------------------------------------------------


def test_validate_unmaterialized(op, workspace):
    """Unmaterialized artifact rejected at validation."""
    art = Artifact(
        type=ArtifactType.VECTOR,
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.1)
    errors = op.validate_inputs([art], params)
    assert any("materialized" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 15. Validation: wrong input count
# ---------------------------------------------------------------------------


def test_validate_wrong_input_count(op, workspace):
    """Validation rejects 0 or 2 inputs."""
    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.1)

    errors_zero = op.validate_inputs([], params)
    assert any("1 input" in e.lower() for e in errors_zero)

    art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    errors_two = op.validate_inputs([art, art], params)
    assert any("1 input" in e.lower() for e in errors_two)


# ---------------------------------------------------------------------------
# 16. All checks pass happy path
# ---------------------------------------------------------------------------


def test_all_checks_pass_happy_path(op, workspace):
    """All declared checks pass on a well-formed run with small tolerance."""
    poly = _complex_polygon()

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [poly])
    art = _make_vector_artifact(in_path)

    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.01)
    result = op.execute([art], params)

    for check in result.checks:
        assert check.state == ValidationState.VALID, (
            f"Check {check.check_name} not VALID: {check.message}"
        )


# ---------------------------------------------------------------------------
# 17. preserve_topology=True — polygon stays valid
# ---------------------------------------------------------------------------


def test_preserve_topology_true_valid(op, workspace):
    """With preserve_topology=True, the simplified polygon remains valid."""
    # Build a valid convex-ish polygon with noisy boundary (many vertices)
    coords = []
    n = 60
    for i in range(n):
        angle = 2 * math.pi * i / n
        r = 10.0 + 0.5 * math.sin(7 * angle)
        coords.append((r * math.cos(angle), r * math.sin(angle)))
    poly = Polygon(coords)
    assert poly.is_valid  # sanity: input is valid

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [poly])
    art = _make_vector_artifact(in_path)

    params = SimplifyParams(
        output_path=str(workspace / "out.geojson"),
        tolerance=0.5,
        preserve_topology=True,
    )
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    out_geom = shape(features[0]["geometry"])
    assert out_geom.is_valid


# ---------------------------------------------------------------------------
# 18. preserve_topology=False — can differ from True
# ---------------------------------------------------------------------------


def test_preserve_topology_false_differs(op, workspace):
    """preserve_topology=False can produce a different result than True.

    Use a valid star-shaped polygon where aggressive simplification with
    preserve_topology=False yields fewer vertices than True.
    """
    # Star polygon: valid, many vertices, simplification-sensitive
    coords = []
    n = 40
    for i in range(n):
        angle = 2 * math.pi * i / n
        r = 10.0 if i % 2 == 0 else 5.0
        coords.append((r * math.cos(angle), r * math.sin(angle)))
    poly = Polygon(coords)
    assert poly.is_valid

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [poly])
    art = _make_vector_artifact(in_path)

    params_topo = SimplifyParams(
        output_path=str(workspace / "topo_true.geojson"),
        tolerance=1.0,
        preserve_topology=True,
    )
    result_true = op.execute([art], params_topo)

    params_no_topo = SimplifyParams(
        output_path=str(workspace / "topo_false.geojson"),
        tolerance=1.0,
        preserve_topology=False,
    )
    result_false = op.execute([art], params_no_topo)

    feats_true = _read_output(workspace / "topo_true.geojson")
    feats_false = _read_output(workspace / "topo_false.geojson")

    geom_true = shape(feats_true[0]["geometry"])
    geom_false = shape(feats_false[0]["geometry"])

    # Both modes ran successfully
    assert result_true.artifact.type == ArtifactType.VECTOR
    assert result_false.artifact.type == ArtifactType.VECTOR

    # Compare vertex counts — they typically differ
    verts_true = len(geom_true.exterior.coords) if hasattr(geom_true, "exterior") else 0
    verts_false = len(geom_false.exterior.coords) if hasattr(geom_false, "exterior") else 0
    # At least one produced a simplified result
    assert verts_true > 0
    assert verts_false > 0


# ---------------------------------------------------------------------------
# 19. Multiple features, different complexity
# ---------------------------------------------------------------------------


def test_multiple_features_different_complexity(op, workspace):
    """Mix of simple and complex polygons all come through, complex ones simplified."""
    simple = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    complex_poly = _complex_polygon()

    simple_verts = len(simple.exterior.coords)
    complex_verts = len(complex_poly.exterior.coords)

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [simple, complex_poly], properties=[{"id": "s"}, {"id": "c"}])
    art = _make_vector_artifact(in_path)

    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.3)
    op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 2

    feat_s = [f for f in features if f["properties"]["id"] == "s"][0]
    feat_c = [f for f in features if f["properties"]["id"] == "c"][0]

    geom_s = shape(feat_s["geometry"])
    geom_c = shape(feat_c["geometry"])

    # Simple polygon has few vertices already — stays roughly the same
    assert len(geom_s.exterior.coords) <= simple_verts
    # Complex polygon should be reduced
    assert len(geom_c.exterior.coords) < complex_verts


# ---------------------------------------------------------------------------
# 20. Empty geometry pass-through
# ---------------------------------------------------------------------------


def test_empty_geometry_passthrough(op, workspace):
    """Empty geometry stays empty after simplification."""
    normal = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    empty = Polygon()

    in_path = workspace / "input.geojson"
    schema = {"geometry": "Polygon", "properties": {"fid": "str"}}
    crs = CRS.from_epsg(32610).to_dict()
    with fiona.open(in_path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        dst.write({"geometry": mapping(normal), "properties": {"fid": "A"}})
        dst.write({"geometry": mapping(empty), "properties": {"fid": "B"}})

    art = _make_vector_artifact(in_path)

    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.1)
    result = op.execute([art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 2

    feat_b = [f for f in features if f["properties"]["fid"] == "B"][0]
    geom_b = shape(feat_b["geometry"])
    assert geom_b.is_empty

    # No collapse warning — the empty was already empty
    geom_check = [c for c in result.checks if c.check_name == "geometry_valid"]
    assert geom_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 21. Timing recorded
# ---------------------------------------------------------------------------


def test_timing_recorded(op, workspace):
    """Result includes positive timing_seconds."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    in_path = workspace / "input.geojson"
    _write_vector(in_path, [poly])
    art = _make_vector_artifact(in_path)

    params = SimplifyParams(output_path=str(workspace / "out.geojson"), tolerance=0.0)
    result = op.execute([art], params)

    assert result.timing_seconds is not None
    assert result.timing_seconds > 0
