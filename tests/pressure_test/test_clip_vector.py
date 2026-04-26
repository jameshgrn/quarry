"""ClipVectorOperator pressure test.

Lane: operator

Stress points:
1. Protocol compliance (spec, validate_inputs, declared_checks)
2. Happy path — polygon clipped to mask
3. Feature entirely inside mask — unchanged in output
4. Feature entirely outside mask — dropped from output
5. Multiple features, mixed overlap
6. Point clipping — points inside mask kept, outside dropped
7. Line clipping — lines clipped to mask boundary
8. CRS mismatch rejected
9. CRS preserved in output
10. Properties preserved on surviving features
11. Feature count check — output <= input
12. Output within clip extent check — VALID
13. Lineage records both input IDs
14. Output metadata fresh from file
15. Validation: wrong input count
16. Validation: raster input rejected
17. Validation: unmaterialized input
18. All checks pass on happy path
19. Empty geometries in input skipped
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
from quarry_operators.clip_vector import (
    ClipVectorOperator,
    ClipVectorParams,
)
from rasterio.crs import CRS
from shapely.geometry import LineString, Point, Polygon, mapping, shape

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_vector(path, geometries, crs_epsg=32610, properties=None, schema_props=None):
    """Write geometries to GeoJSON. Infers geometry type from first non-empty geom."""
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
    return ClipVectorOperator()


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# Shared geometry: mask is a box from (0,0) to (5,5)
# ---------------------------------------------------------------------------

MASK_POLY = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])

# ---------------------------------------------------------------------------
# 1. Protocol compliance
# ---------------------------------------------------------------------------


def test_protocol_compliance(op):
    """Operator satisfies the Operator protocol."""
    assert isinstance(op, Operator)


def test_spec(op):
    spec = op.spec
    assert spec.input_types == (ArtifactType.VECTOR, ArtifactType.VECTOR)
    assert spec.output_type == ArtifactType.VECTOR
    assert spec.min_inputs == 2
    assert spec.max_inputs == 2
    assert spec.resource_scale == ResourceScale.MEDIUM


def test_declared_checks(op):
    checks = op.declared_checks()
    assert "crs_valid" in checks
    assert "output_within_clip" in checks
    assert "feature_count" in checks


# ---------------------------------------------------------------------------
# 2. Happy path — polygon clipped to mask
# ---------------------------------------------------------------------------


def test_happy_path_polygon_clipped(op, workspace):
    """Polygon partially overlapping mask is clipped; output area < original."""
    # Feature spans (3,3) to (7,7) — overlaps mask at (3,3)-(5,5)
    feature_poly = Polygon([(3, 3), (7, 3), (7, 7), (3, 7)])

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(feat_path, [feature_poly], properties=[{"fid": "A"}])
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([feat_art, mask_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1

    clipped_geom = shape(features[0]["geometry"])
    assert clipped_geom.area < feature_poly.area
    assert clipped_geom.area == pytest.approx(4.0, abs=0.01)  # (5-3)*(5-3)
    assert result.artifact.type == ArtifactType.VECTOR


# ---------------------------------------------------------------------------
# 3. Feature entirely inside mask — unchanged in output
# ---------------------------------------------------------------------------


def test_feature_entirely_inside_mask(op, workspace):
    """Feature fully within mask passes through unchanged."""
    inner_poly = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(feat_path, [inner_poly], properties=[{"fid": "inside"}])
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    op.execute([feat_art, mask_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    out_geom = shape(features[0]["geometry"])
    assert out_geom.area == pytest.approx(inner_poly.area, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Feature entirely outside mask — dropped from output
# ---------------------------------------------------------------------------


def test_feature_entirely_outside_mask(op, workspace):
    """Feature fully outside mask is dropped."""
    outside_poly = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(feat_path, [outside_poly], properties=[{"fid": "outside"}])
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    op.execute([feat_art, mask_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 0


# ---------------------------------------------------------------------------
# 5. Multiple features, mixed overlap
# ---------------------------------------------------------------------------


def test_multiple_features_mixed_overlap(op, workspace):
    """Some inside, some partial, some outside — correct counts."""
    inside = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    partial = Polygon([(3, 3), (7, 3), (7, 7), (3, 7)])
    outside = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(
        feat_path,
        [inside, partial, outside],
        properties=[{"fid": "in"}, {"fid": "partial"}, {"fid": "out"}],
    )
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    op.execute([feat_art, mask_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 2  # inside + partial survive, outside dropped
    fids = {f["properties"]["fid"] for f in features}
    assert fids == {"in", "partial"}

    # Partial feature is clipped — smaller area than original
    partial_feat = [f for f in features if f["properties"]["fid"] == "partial"][0]
    assert shape(partial_feat["geometry"]).area < partial.area


# ---------------------------------------------------------------------------
# 6. Point clipping
# ---------------------------------------------------------------------------


def test_point_clipping(op, workspace):
    """Points inside mask kept, points outside dropped."""
    pt_in = Point(2, 2)
    pt_out = Point(10, 10)
    pt_edge = Point(0, 0)  # on mask boundary — intersects

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(
        feat_path,
        [pt_in, pt_out, pt_edge],
        properties=[{"pid": "in"}, {"pid": "out"}, {"pid": "edge"}],
    )
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    op.execute([feat_art, mask_art], params)

    features = _read_output(workspace / "out.geojson")
    pids = {f["properties"]["pid"] for f in features}
    assert "in" in pids
    assert "out" not in pids
    # Edge point may or may not survive depending on intersection semantics
    assert len(features) >= 1


# ---------------------------------------------------------------------------
# 7. Line clipping
# ---------------------------------------------------------------------------


def test_line_clipping(op, workspace):
    """Line crossing mask boundary is clipped to shorter segment."""
    line = LineString([(2, 2), (10, 2)])  # crosses mask at x=5

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(feat_path, [line], properties=[{"lid": "cross"}])
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    op.execute([feat_art, mask_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    clipped_geom = shape(features[0]["geometry"])
    assert clipped_geom.length < line.length
    assert clipped_geom.length == pytest.approx(3.0, abs=0.01)  # from x=2 to x=5


# ---------------------------------------------------------------------------
# 8. CRS mismatch rejected
# ---------------------------------------------------------------------------


def test_crs_mismatch_rejected(op, workspace):
    """Features in EPSG:32610, mask in EPSG:4326 — validation error."""
    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(feat_path, [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs_epsg=32610)
    _write_vector(mask_path, [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs_epsg=4326)

    feat_art = _make_vector_artifact(feat_path, crs_epsg=32610)
    mask_art = _make_vector_artifact(mask_path, crs_epsg=4326)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    errors = op.validate_inputs([feat_art, mask_art], params)
    assert any("CRS mismatch" in e for e in errors)


# ---------------------------------------------------------------------------
# 9. CRS preserved in output
# ---------------------------------------------------------------------------


def test_crs_preserved_in_output(op, workspace):
    """Output CRS matches input features CRS."""
    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(feat_path, [Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])], crs_epsg=32610)
    _write_vector(mask_path, [MASK_POLY], crs_epsg=32610)

    feat_art = _make_vector_artifact(feat_path, crs_epsg=32610)
    mask_art = _make_vector_artifact(mask_path, crs_epsg=32610)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([feat_art, mask_art], params)

    assert result.artifact.spatial.crs is not None
    # CRS check is VALID
    crs_check = [c for c in result.checks if c.check_name == "crs_valid"]
    assert len(crs_check) == 1
    assert crs_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 10. Properties preserved on surviving features
# ---------------------------------------------------------------------------


def test_properties_preserved(op, workspace):
    """Clipped features retain all original properties."""
    poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(
        feat_path,
        [poly],
        properties=[{"name": "river", "code": "42", "class": "A"}],
        schema_props={"name": "str", "code": "str", "class": "str"},
    )
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    op.execute([feat_art, mask_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    props = features[0]["properties"]
    assert props["name"] == "river"
    assert props["code"] == "42"
    assert props["class"] == "A"


# ---------------------------------------------------------------------------
# 11. Feature count check — output <= input
# ---------------------------------------------------------------------------


def test_feature_count_check(op, workspace):
    """feature_count check VALID: output features <= input features."""
    inside = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    outside = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(feat_path, [inside, outside], properties=[{"fid": "1"}, {"fid": "2"}])
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([feat_art, mask_art], params)

    fc_check = [c for c in result.checks if c.check_name == "feature_count"]
    assert len(fc_check) == 1
    assert fc_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 12. Output within clip extent check
# ---------------------------------------------------------------------------


def test_output_within_clip_extent_check(op, workspace):
    """output_within_clip check VALID on normal clip."""
    poly = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(feat_path, [poly])
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([feat_art, mask_art], params)

    clip_check = [c for c in result.checks if c.check_name == "output_within_clip"]
    assert len(clip_check) == 1
    assert clip_check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 13. Lineage records both input IDs
# ---------------------------------------------------------------------------


def test_lineage_records_both_inputs(op, workspace):
    """Output lineage includes both feature and mask artifact IDs."""
    poly = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(feat_path, [poly])
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([feat_art, mask_art], params)

    assert result.artifact.lineage is not None
    assert result.artifact.lineage.operation == "clip_vector"
    assert set(result.artifact.lineage.inputs) == {feat_art.id, mask_art.id}


# ---------------------------------------------------------------------------
# 14. Output metadata fresh from file
# ---------------------------------------------------------------------------


def test_output_metadata_fresh(op, workspace):
    """Output artifact metadata is read from the actual output file."""
    poly = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(feat_path, [poly], properties=[{"a": "1"}])
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([feat_art, mask_art], params)

    art = result.artifact
    assert art.type == ArtifactType.VECTOR
    assert art.backing.kind == BackingStoreKind.LOCAL_FILE
    assert Path(art.backing.uri).exists()
    assert art.spatial.feature_count == 1
    assert art.spatial.crs is not None
    assert art.spatial.extent is not None
    assert art.metadata["format"] == "geojson"
    assert art.backing.size_bytes > 0
    assert art.backing.content_hash is not None


# ---------------------------------------------------------------------------
# 15. Validation: wrong input count
# ---------------------------------------------------------------------------


def test_validate_wrong_input_count_zero(op, workspace):
    """Validation rejects 0 inputs."""
    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    errors = op.validate_inputs([], params)
    assert any("Exactly 2" in e for e in errors)


def test_validate_wrong_input_count_one(op, workspace):
    """Validation rejects 1 input."""
    art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    errors = op.validate_inputs([art], params)
    assert any("Exactly 2" in e for e in errors)


def test_validate_wrong_input_count_three(op, workspace):
    """Validation rejects 3 inputs."""
    art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    errors = op.validate_inputs([art, art, art], params)
    assert any("Exactly 2" in e for e in errors)


# ---------------------------------------------------------------------------
# 16. Validation: raster input rejected
# ---------------------------------------------------------------------------


def test_validate_raster_input_rejected(op, workspace):
    """Validation rejects non-vector inputs."""
    raster_art = Artifact(
        type=ArtifactType.RASTER,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    vector_art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    errors = op.validate_inputs([raster_art, vector_art], params)
    assert any("vector" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 17. Validation: unmaterialized input
# ---------------------------------------------------------------------------


def test_validate_unmaterialized_input(op, workspace):
    """Validation rejects unmaterialized artifacts (no backing)."""
    no_backing = Artifact(
        type=ArtifactType.VECTOR,
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    with_backing = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    errors = op.validate_inputs([no_backing, with_backing], params)
    assert any("materialized" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 18. All checks pass on happy path
# ---------------------------------------------------------------------------


def test_all_checks_pass_happy_path(op, workspace):
    """All declared checks pass on a well-formed clip."""
    poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(feat_path, [poly], properties=[{"a": "1"}])
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([feat_art, mask_art], params)

    for check in result.checks:
        assert check.state == ValidationState.VALID, (
            f"Check {check.check_name} not VALID: {check.message}"
        )


# ---------------------------------------------------------------------------
# 19. Empty geometries in input skipped
# ---------------------------------------------------------------------------


def test_empty_geometries_skipped(op, workspace):
    """Empty geometries in input are skipped, not written to output."""
    normal = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    empty = Polygon()

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"

    # Write features with one empty geometry
    schema = {"geometry": "Polygon", "properties": {"fid": "str"}}
    crs = CRS.from_epsg(32610).to_dict()
    with fiona.open(feat_path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        dst.write({"geometry": mapping(normal), "properties": {"fid": "A"}})
        dst.write({"geometry": mapping(empty), "properties": {"fid": "B"}})

    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    op.execute([feat_art, mask_art], params)

    features = _read_output(workspace / "out.geojson")
    # Only the non-empty feature survives
    assert len(features) == 1
    assert features[0]["properties"]["fid"] == "A"


# ---------------------------------------------------------------------------
# 20. Timing recorded
# ---------------------------------------------------------------------------


def test_timing_recorded(op, workspace):
    """OperatorResult.timing_seconds is recorded and positive."""
    poly = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])

    feat_path = workspace / "features.geojson"
    mask_path = workspace / "mask.geojson"
    _write_vector(feat_path, [poly])
    _write_vector(mask_path, [MASK_POLY])

    feat_art = _make_vector_artifact(feat_path)
    mask_art = _make_vector_artifact(mask_path)

    params = ClipVectorParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([feat_art, mask_art], params)

    assert result.timing_seconds is not None
    assert result.timing_seconds > 0
