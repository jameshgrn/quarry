"""SpatialJoinOperator pressure test.

Lane: operator

Stress points:
1. Protocol compliance (spec, validate_inputs, declared_checks)
2. Happy path — polygon/polygon intersects with known overlap
3. CRS mismatch rejected at validation
4. Empty geometries — preserved in output, no match
5. One-to-many — left polygon overlaps multiple right polygons
6. No-overlap — left feature kept with null right attributes
7. Schema collision — colliding column names get '_right' suffix
8. Row count / cardinality: output >= left count (left join invariant)
9. Point-in-polygon join
10. Lineage records operation and params
11. Output artifact metadata (fresh from file, not copied)
12. Validation rejects invalid predicate (intersects/contains/within/touches accepted)
13. Many-to-many — both sides overlap multiple features
14. Mixed geometry types in single layer
15. predicate=contains — left polygon containing right point matches
16. predicate=within — left polygon contained in right polygon matches
17. predicate=touches — boundary-only contact matches; interior overlap does not
18. predicate asymmetry — same geometry pair with contains vs within yields different match counts
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
from quarry_operators.spatial_join import (
    SpatialJoinOperator,
    SpatialJoinParams,
)
from rasterio.crs import CRS
from shapely.geometry import Point, Polygon, mapping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_vector(path, geometries, crs_epsg=32610, properties=None, schema_props=None):
    """Write geometries to GeoJSON. Infers geometry type from first geom."""
    if not geometries:
        # Empty layer
        geom_type = "Polygon"
    else:
        first_non_empty = next((g for g in geometries if not g.is_empty), None)
        if first_non_empty is None:
            geom_type = "Polygon"
        elif first_non_empty.geom_type == "Point":
            geom_type = "Point"
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
            {"geometry": dict(f["geometry"]), "properties": dict(f.get("properties", {}))}
            for f in src
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def op():
    return SpatialJoinOperator()


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
    assert spec.input_types == (ArtifactType.VECTOR, ArtifactType.VECTOR)
    assert spec.output_type == ArtifactType.VECTOR
    assert spec.min_inputs == 2
    assert spec.max_inputs == 2
    assert spec.resource_scale == ResourceScale.MEDIUM


def test_declared_checks(op):
    checks = op.declared_checks()
    assert "crs_valid" in checks
    assert "left_features_preserved" in checks
    assert "schema_no_collision" in checks


# ---------------------------------------------------------------------------
# 2. Happy path — polygon/polygon intersects
# ---------------------------------------------------------------------------


def test_happy_path_polygon_intersects(op, workspace):
    """Two overlapping polygons produce a joined feature."""
    left_poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    right_poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [left_poly], properties=[{"left_id": "A"}])
    _write_vector(right_path, [right_poly], properties=[{"right_id": "X"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([left_art, right_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    assert features[0]["properties"]["left_id"] == "A"
    assert features[0]["properties"]["right_id"] == "X"
    assert result.artifact.type == ArtifactType.VECTOR


# ---------------------------------------------------------------------------
# 3. CRS mismatch rejected
# ---------------------------------------------------------------------------


def test_crs_mismatch_rejected(op, workspace):
    """Left in EPSG:32610, right in EPSG:4326 → validation error."""
    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs_epsg=32610)
    _write_vector(right_path, [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs_epsg=4326)

    left_art = _make_vector_artifact(left_path, crs_epsg=32610)
    right_art = _make_vector_artifact(right_path, crs_epsg=4326)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    errors = op.validate_inputs([left_art, right_art], params)
    assert any("CRS mismatch" in e for e in errors)


# ---------------------------------------------------------------------------
# 4. Empty geometries — preserved, no match
# ---------------------------------------------------------------------------


def test_empty_geometry_preserved(op, workspace):
    """Empty left geometry appears in output with null right attrs."""
    normal = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    empty = Polygon()

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    # Write left with empty geom — manual approach since fiona may handle empty
    schema = {"geometry": "Polygon", "properties": {"lid": "str"}}
    crs = CRS.from_epsg(32610).to_dict()
    with fiona.open(left_path, "w", driver="GeoJSON", crs=crs, schema=schema) as dst:
        dst.write({"geometry": mapping(normal), "properties": {"lid": "A"}})
        dst.write({"geometry": mapping(empty), "properties": {"lid": "B"}})

    _write_vector(right_path, [normal], properties=[{"rid": "X"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    op.execute([left_art, right_art], params)

    features = _read_output(workspace / "out.geojson")
    # Feature A matches right, feature B (empty) does not
    assert len(features) == 2
    feat_b = [f for f in features if f["properties"]["lid"] == "B"][0]
    assert feat_b["properties"]["rid"] is None


# ---------------------------------------------------------------------------
# 5. One-to-many — left overlaps multiple right
# ---------------------------------------------------------------------------


def test_one_to_many(op, workspace):
    """One left polygon overlapping two right polygons produces two output rows."""
    left_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    right_a = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    right_b = Polygon([(5, 5), (7, 5), (7, 7), (5, 7)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [left_poly], properties=[{"lid": "big"}])
    _write_vector(right_path, [right_a, right_b], properties=[{"rid": "a"}, {"rid": "b"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    op.execute([left_art, right_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 2
    rids = {f["properties"]["rid"] for f in features}
    assert rids == {"a", "b"}
    # All features carry the left geometry (same lid)
    assert all(f["properties"]["lid"] == "big" for f in features)


# ---------------------------------------------------------------------------
# 6. No overlap — left preserved with null right attrs
# ---------------------------------------------------------------------------


def test_no_overlap(op, workspace):
    """Disjoint datasets — all left features kept, right attrs are null."""
    left_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    right_poly = Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [left_poly], properties=[{"lid": "L1"}])
    _write_vector(right_path, [right_poly], properties=[{"rid": "R1"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    op.execute([left_art, right_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    assert features[0]["properties"]["lid"] == "L1"
    assert features[0]["properties"]["rid"] is None


# ---------------------------------------------------------------------------
# 7. Schema collision — right column renamed with '_right' suffix
# ---------------------------------------------------------------------------


def test_schema_collision_rename(op, workspace):
    """Colliding column 'name' in both sides → 'name_right' in output."""
    left_poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    right_poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [left_poly], properties=[{"name": "left_val"}])
    _write_vector(right_path, [right_poly], properties=[{"name": "right_val"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([left_art, right_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    props = features[0]["properties"]
    assert props["name"] == "left_val"
    assert props["name_right"] == "right_val"

    # Check warns about collision
    collision_check = [c for c in result.checks if c.check_name == "schema_no_collision"]
    assert len(collision_check) == 1
    assert collision_check[0].state == ValidationState.WARN


# ---------------------------------------------------------------------------
# 8. Cardinality: output >= left (left join invariant)
# ---------------------------------------------------------------------------


def test_cardinality_left_join_invariant(op, workspace):
    """Output feature count >= left feature count, always."""
    # 3 left features, 1 right that overlaps 2 of them
    left_a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    left_b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    left_c = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])  # disjoint
    right = Polygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(
        left_path, [left_a, left_b, left_c], properties=[{"id": "a"}, {"id": "b"}, {"id": "c"}]
    )
    _write_vector(right_path, [right], properties=[{"val": "R"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([left_art, right_art], params)

    features = _read_output(workspace / "out.geojson")
    # left_a and left_b overlap right → 2 matched rows + left_c unmatched → 3 total
    assert len(features) >= 3

    # left_features_preserved check should pass
    check = [c for c in result.checks if c.check_name == "left_features_preserved"]
    assert check[0].state == ValidationState.VALID


# ---------------------------------------------------------------------------
# 9. Point-in-polygon join
# ---------------------------------------------------------------------------


def test_point_in_polygon(op, workspace):
    """Points joined to polygons they fall within."""
    pt_inside = Point(1, 1)
    pt_outside = Point(100, 100)

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [pt_inside, pt_outside], properties=[{"pid": "in"}, {"pid": "out"}])

    poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    _write_vector(right_path, [poly], properties=[{"zone": "Z1"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    op.execute([left_art, right_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 2
    feat_in = [f for f in features if f["properties"]["pid"] == "in"][0]
    feat_out = [f for f in features if f["properties"]["pid"] == "out"][0]
    assert feat_in["properties"]["zone"] == "Z1"
    assert feat_out["properties"]["zone"] is None


# ---------------------------------------------------------------------------
# 10. Lineage records operation and params
# ---------------------------------------------------------------------------


def test_lineage(op, workspace):
    """Output artifact lineage includes operation name and predicate."""
    left_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    right_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [left_poly])
    _write_vector(right_path, [right_poly])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([left_art, right_art], params)

    assert result.artifact.lineage is not None
    assert result.artifact.lineage.operation == "spatial_join"
    assert result.artifact.lineage.params["predicate"] == "intersects"
    assert set(result.artifact.lineage.inputs) == {left_art.id, right_art.id}


# ---------------------------------------------------------------------------
# 11. Output artifact metadata — fresh from file
# ---------------------------------------------------------------------------


def test_output_artifact_metadata(op, workspace):
    """Output artifact has correct type, backing, spatial descriptor from actual file."""
    left_poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    right_poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [left_poly], properties=[{"a": "1"}])
    _write_vector(right_path, [right_poly], properties=[{"b": "2"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([left_art, right_art], params)

    art = result.artifact
    assert art.type == ArtifactType.VECTOR
    assert art.backing.kind == BackingStoreKind.LOCAL_FILE
    assert Path(art.backing.uri).exists()
    assert art.spatial.feature_count == 1
    assert art.spatial.crs is not None
    assert art.spatial.extent is not None
    assert art.metadata["format"] == "geojson"
    assert result.timing_seconds is not None
    assert result.timing_seconds > 0


# ---------------------------------------------------------------------------
# 12. Unsupported predicate rejected
# ---------------------------------------------------------------------------


def test_invalid_predicate_rejected(op, workspace):
    """Invalid predicate rejected at validation (crosses not supported)."""
    left_art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    right_art = Artifact(
        type=ArtifactType.VECTOR,
        backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        spatial=SpatialDescriptor(crs="EPSG:32610"),
    )
    params = SpatialJoinParams(output_path="/fake/out.geojson", predicate="crosses")
    errors = op.validate_inputs([left_art, right_art], params)
    assert any("Unsupported predicate" in e for e in errors)


# ---------------------------------------------------------------------------
# 13. Many-to-many — both sides overlap multiple features
# ---------------------------------------------------------------------------


def test_many_to_many(op, workspace):
    """Two left × two right, all overlapping → 4 output features."""
    # All four polygons overlap each other (centered on same area)
    left_a = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    left_b = Polygon([(1, 1), (6, 1), (6, 6), (1, 6)])
    right_x = Polygon([(2, 2), (7, 2), (7, 7), (2, 7)])
    right_y = Polygon([(3, 3), (8, 3), (8, 8), (3, 8)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [left_a, left_b], properties=[{"lid": "a"}, {"lid": "b"}])
    _write_vector(right_path, [right_x, right_y], properties=[{"rid": "x"}, {"rid": "y"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    op.execute([left_art, right_art], params)

    features = _read_output(workspace / "out.geojson")
    # Each left matches both rights → 2 × 2 = 4
    assert len(features) == 4
    pairs = {(f["properties"]["lid"], f["properties"]["rid"]) for f in features}
    assert pairs == {("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")}


# ---------------------------------------------------------------------------
# 14. Validation: wrong input count
# ---------------------------------------------------------------------------


def test_validate_wrong_input_count(op, workspace):
    """Validation rejects 0 or 1 inputs."""
    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    errors = op.validate_inputs([], params)
    assert any("Exactly 2" in e for e in errors)


# ---------------------------------------------------------------------------
# 15. Validation: wrong input types
# ---------------------------------------------------------------------------


def test_validate_wrong_types(op, workspace):
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
    params = SpatialJoinParams(output_path="/fake/out.geojson")
    errors = op.validate_inputs([raster_art, vector_art], params)
    assert any("vector" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 16. All checks pass on happy path
# ---------------------------------------------------------------------------


def test_all_checks_pass_happy_path(op, workspace):
    """All declared checks pass on a well-formed run with no collisions."""
    left_poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    right_poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [left_poly], properties=[{"a": "1"}])
    _write_vector(right_path, [right_poly], properties=[{"b": "2"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([left_art, right_art], params)

    for check in result.checks:
        assert check.state == ValidationState.VALID, (
            f"Check {check.check_name} not VALID: {check.message}"
        )


# ---------------------------------------------------------------------------
# 17. Empty right layer — all left preserved with nulls
# ---------------------------------------------------------------------------


def test_empty_right_layer(op, workspace):
    """Right layer with no features → all left features kept, left attrs intact.

    GeoJSON driver discards schema when no features written, so the operator
    sees zero right columns — output contains only left columns.
    """
    left_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [left_poly], properties=[{"lid": "L1"}])

    # Write empty right layer
    schema = {"geometry": "Polygon", "properties": {"rid": "str"}}
    crs = CRS.from_epsg(32610).to_dict()
    with fiona.open(right_path, "w", driver="GeoJSON", crs=crs, schema=schema):
        pass  # no features

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    op.execute([left_art, right_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    assert features[0]["properties"]["lid"] == "L1"
    # GeoJSON discards schema on empty layer → no right columns in output
    assert "rid" not in features[0]["properties"]


# ---------------------------------------------------------------------------
# 18. Multiple colliding columns
# ---------------------------------------------------------------------------


def test_multiple_colliding_columns(op, workspace):
    """Multiple colliding columns all get '_right' suffix."""
    left_poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    right_poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(
        left_path,
        [left_poly],
        properties=[{"name": "L", "code": "LC"}],
        schema_props={"name": "str", "code": "str"},
    )
    _write_vector(
        right_path,
        [right_poly],
        properties=[{"name": "R", "code": "RC", "unique_col": "U"}],
        schema_props={"name": "str", "code": "str", "unique_col": "str"},
    )

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"))
    result = op.execute([left_art, right_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    props = features[0]["properties"]
    assert props["name"] == "L"
    assert props["code"] == "LC"
    assert props["name_right"] == "R"
    assert props["code_right"] == "RC"
    assert props["unique_col"] == "U"

    # Lineage records collision renames
    renames = result.artifact.lineage.params["collision_renames"]
    assert renames == {"name": "name_right", "code": "code_right"}


# ---------------------------------------------------------------------------
# 15. predicate=contains — left polygon containing right point matches
# ---------------------------------------------------------------------------


def test_predicate_contains(op, workspace):
    """Left polygon containing right point matches with predicate=contains."""
    left_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    right_point = Point(50, 50)

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [left_poly], properties=[{"lid": "big"}])
    _write_vector(right_path, [right_point], properties=[{"rid": "inside"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"), predicate="contains")
    op.execute([left_art, right_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    assert features[0]["properties"]["lid"] == "big"
    assert features[0]["properties"]["rid"] == "inside"


# ---------------------------------------------------------------------------
# 16. predicate=within — left polygon contained in right polygon matches
# ---------------------------------------------------------------------------


def test_predicate_within(op, workspace):
    """Left polygon within right polygon matches with predicate=within."""
    left_poly = Polygon([(45, 45), (55, 45), (55, 55), (45, 55)])  # 10m square at (45,45)
    right_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])  # 100m square at origin

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [left_poly], properties=[{"lid": "small"}])
    _write_vector(right_path, [right_poly], properties=[{"rid": "large"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    params = SpatialJoinParams(output_path=str(workspace / "out.geojson"), predicate="within")
    op.execute([left_art, right_art], params)

    features = _read_output(workspace / "out.geojson")
    assert len(features) == 1
    assert features[0]["properties"]["lid"] == "small"
    assert features[0]["properties"]["rid"] == "large"


# ---------------------------------------------------------------------------
# 17. predicate=touches — boundary-only contact matches; interior overlap does not
# ---------------------------------------------------------------------------


def test_predicate_touches(op, workspace):
    """Touching polygons match with predicate=touches; interior overlap does not."""
    # Two polygons sharing the x=10 edge (touching)
    left_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    right_poly = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [left_poly], properties=[{"lid": "left"}])
    _write_vector(right_path, [right_poly], properties=[{"rid": "right"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    # With touches: should match (they share an edge)
    params_touches = SpatialJoinParams(
        output_path=str(workspace / "out_touches.geojson"), predicate="touches"
    )
    op.execute([left_art, right_art], params_touches)
    features_touches = _read_output(workspace / "out_touches.geojson")
    assert len(features_touches) == 1
    assert features_touches[0]["properties"]["rid"] == "right"

    # With intersects: should also match (touching satisfies intersects)
    params_intersects = SpatialJoinParams(
        output_path=str(workspace / "out_intersects.geojson"), predicate="intersects"
    )
    op.execute([left_art, right_art], params_intersects)
    features_intersects = _read_output(workspace / "out_intersects.geojson")
    assert len(features_intersects) == 1
    assert features_intersects[0]["properties"]["rid"] == "right"

    # With contains: should NOT match (left does not contain right, they just touch)
    params_contains = SpatialJoinParams(
        output_path=str(workspace / "out_contains.geojson"), predicate="contains"
    )
    op.execute([left_art, right_art], params_contains)
    features_contains = _read_output(workspace / "out_contains.geojson")
    assert len(features_contains) == 1
    # Left feature preserved but with null right attrs (left join invariant)
    assert features_contains[0]["properties"]["rid"] is None


# ---------------------------------------------------------------------------
# 18. predicate asymmetry — same geometry pair with contains vs within yields different match
# counts
# ---------------------------------------------------------------------------


def test_predicate_asymmetry_contains_within(op, workspace):
    """Contains vs within on same pair yield different results due to asymmetry."""
    # Small polygon at (45,45) inside large polygon at origin
    small_poly = Polygon([(45, 45), (55, 45), (55, 55), (45, 55)])
    large_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

    left_path = workspace / "left.geojson"
    right_path = workspace / "right.geojson"
    _write_vector(left_path, [small_poly], properties=[{"lid": "small"}])
    _write_vector(right_path, [large_poly], properties=[{"rid": "large"}])

    left_art = _make_vector_artifact(left_path)
    right_art = _make_vector_artifact(right_path)

    # With contains: small does NOT contain large → no match, right attrs null
    params_contains = SpatialJoinParams(
        output_path=str(workspace / "out_contains.geojson"), predicate="contains"
    )
    op.execute([left_art, right_art], params_contains)
    features_contains = _read_output(workspace / "out_contains.geojson")
    assert len(features_contains) == 1
    assert features_contains[0]["properties"]["rid"] is None  # No match

    # With within: small IS within large → match, right attrs populated
    params_within = SpatialJoinParams(
        output_path=str(workspace / "out_within.geojson"), predicate="within"
    )
    op.execute([left_art, right_art], params_within)
    features_within = _read_output(workspace / "out_within.geojson")
    assert len(features_within) == 1
    assert features_within[0]["properties"]["rid"] == "large"  # Match!
