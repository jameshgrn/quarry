"""Reproject operator pressure test.

Stress points:
1. Spatial descriptor updates (CRS changes, extent transforms, resolution recalc)
2. Output metadata regeneration (fresh from actual output, not copied from input)
3. Check composition (input CRS valid, output CRS matches target, extent sane)
4. Lineage captures transform intent (source CRS, target CRS, resampling)
5. Lazy artifact rejection (can't reproject without data)
6. OperatorResult still naturally single-output
7. Vector reprojection (second artifact type through same operator)
"""


import fiona
import numpy as np
import pytest
import rasterio
from fiona.crs import CRS as FionaCRS
from quarry_connectors.local_file import LocalFileConnector
from quarry_core.artifact import (
    Artifact,
    ArtifactType,
    BackingStore,
    BackingStoreKind,
    ValidationState,
)
from quarry_core.executor import RunStatus
from quarry_core.executors.local import LocalExecutor
from quarry_core.operator import Operator
from quarry_operators.reproject import ReprojectOperator, ReprojectParams
from quarry_registry.registry import Registry
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Point, mapping


@pytest.fixture
def sample_raster_4326(tmp_path):
    """Small raster in EPSG:4326."""
    path = tmp_path / "sample_4326.tif"
    data = np.random.rand(3, 100, 100).astype(np.float32)
    transform = from_bounds(-10.0, -10.0, 10.0, 10.0, 100, 100)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=3,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def sample_vector_4326(tmp_path):
    """Small vector in EPSG:4326."""
    path = tmp_path / "points_4326.geojson"
    schema = {"geometry": "Point", "properties": {"name": "str", "value": "float"}}
    with fiona.open(
        path,
        "w",
        driver="GeoJSON",
        crs=FionaCRS.from_epsg(4326),
        schema=schema,
    ) as dst:
        for i in range(10):
            dst.write(
                {
                    "geometry": mapping(Point(-5.0 + i, -5.0 + i)),
                    "properties": {"name": f"pt_{i}", "value": float(i)},
                }
            )
    return path


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_satisfies_operator_protocol(self):
        op = ReprojectOperator()
        assert isinstance(op, Operator)

    def test_declared_checks(self):
        op = ReprojectOperator()
        checks = op.declared_checks()
        assert "crs_valid" in checks
        assert "crs_matches_target" in checks
        assert "extent_sane" in checks
        assert "backing_accessible" in checks


# ---------------------------------------------------------------------------
# Stress point 1: spatial descriptor updates
# ---------------------------------------------------------------------------


class TestSpatialDescriptorUpdates:
    def test_crs_changes_on_output(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)
        assert mat.artifact.spatial.crs == "EPSG:4326"

        output_path = workspace / "reprojected.tif"
        params = ReprojectParams(target_crs="EPSG:32610", output_path=str(output_path))

        op = ReprojectOperator()
        result = op.execute([mat.artifact], params)

        assert result.artifact.spatial.crs is not None
        assert "32610" in str(result.artifact.spatial.crs)
        assert result.artifact.spatial.crs != mat.artifact.spatial.crs

    def test_extent_transforms_to_projected_coords(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)
        in_ext = mat.artifact.spatial.extent
        # Input extent is in degrees (-10, -10, 10, 10)

        output_path = workspace / "projected.tif"
        params = ReprojectParams(target_crs="EPSG:32610", output_path=str(output_path))

        op = ReprojectOperator()
        result = op.execute([mat.artifact], params)
        out_ext = result.artifact.spatial.extent

        # Output extent should be in meters (much larger numbers)
        assert out_ext is not None
        assert abs(out_ext[2] - out_ext[0]) > abs(in_ext[2] - in_ext[0])

    def test_resolution_recalculated(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)
        in_res = mat.artifact.spatial.resolution
        # Input resolution is in degrees (~0.2 deg/pixel)

        output_path = workspace / "res_change.tif"
        params = ReprojectParams(target_crs="EPSG:32610", output_path=str(output_path))

        op = ReprojectOperator()
        result = op.execute([mat.artifact], params)
        out_res = result.artifact.spatial.resolution

        # Output resolution should be in meters (much larger than degrees)
        assert out_res is not None
        assert out_res[0] > in_res[0]  # meters > degrees

    def test_band_count_preserved(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)

        output_path = workspace / "bands.tif"
        params = ReprojectParams(target_crs="EPSG:32610", output_path=str(output_path))

        op = ReprojectOperator()
        result = op.execute([mat.artifact], params)
        assert result.artifact.spatial.band_count == mat.artifact.spatial.band_count == 3


# ---------------------------------------------------------------------------
# Stress point 2: output metadata is fresh, not copied
# ---------------------------------------------------------------------------


class TestMetadataRegeneration:
    def test_output_metadata_from_actual_file(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)

        output_path = workspace / "fresh_meta.tif"
        params = ReprojectParams(target_crs="EPSG:32610", output_path=str(output_path))

        op = ReprojectOperator()
        result = op.execute([mat.artifact], params)

        # Verify metadata comes from actual output file, not copied
        assert result.artifact.metadata.get("driver") == "GTiff"
        assert result.artifact.backing.size_bytes > 0
        assert result.artifact.backing.content_hash is not None

        # The hash should be different from input (different data due to resampling)
        assert result.artifact.backing.content_hash != mat.artifact.backing.content_hash

    def test_output_has_distinct_identity(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)

        output_path = workspace / "new_id.tif"
        params = ReprojectParams(target_crs="EPSG:32610", output_path=str(output_path))

        op = ReprojectOperator()
        result = op.execute([mat.artifact], params)

        assert result.artifact.id != mat.artifact.id
        assert result.artifact.name != mat.artifact.name


# ---------------------------------------------------------------------------
# Stress point 3: check composition
# ---------------------------------------------------------------------------


class TestCheckComposition:
    def test_all_declared_checks_present(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)

        output_path = workspace / "checked.tif"
        params = ReprojectParams(target_crs="EPSG:32610", output_path=str(output_path))

        op = ReprojectOperator()
        result = op.execute([mat.artifact], params)

        check_names = [c.check_name for c in result.checks]
        assert "crs_valid" in check_names
        assert "crs_matches_target" in check_names
        assert "extent_sane" in check_names
        assert "backing_accessible" in check_names

    def test_all_checks_pass_for_valid_reproject(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)

        output_path = workspace / "valid.tif"
        params = ReprojectParams(target_crs="EPSG:32610", output_path=str(output_path))

        op = ReprojectOperator()
        result = op.execute([mat.artifact], params)

        for check in result.checks:
            assert check.state in (ValidationState.VALID, ValidationState.WARN), (
                f"Check '{check.check_name}' failed: {check.message}"
            )


# ---------------------------------------------------------------------------
# Stress point 4: lineage captures transform intent
# ---------------------------------------------------------------------------


class TestLineage:
    def test_lineage_captures_crs_transform(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)

        output_path = workspace / "lineage.tif"
        params = ReprojectParams(
            target_crs="EPSG:32610",
            output_path=str(output_path),
            resampling="bilinear",
        )

        op = ReprojectOperator()
        result = op.execute([mat.artifact], params)

        lineage = result.artifact.lineage
        assert lineage is not None
        assert lineage.operation == "reproject"
        assert mat.artifact.id in lineage.inputs
        assert lineage.params["target_crs"] == "EPSG:32610"
        assert lineage.params["source_crs"] == "EPSG:4326"
        assert lineage.params["resampling"] == "bilinear"


# ---------------------------------------------------------------------------
# Stress point 5: lazy artifact rejection
# ---------------------------------------------------------------------------


class TestLazyRejection:
    def test_lazy_artifact_rejected(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace, lazy=True)

        params = ReprojectParams(target_crs="EPSG:32610", output_path="/out.tif")
        op = ReprojectOperator()
        errors = op.validate_inputs([mat.artifact], params)
        assert any("materialized" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Stress point 6: single-output still natural
# ---------------------------------------------------------------------------


class TestSingleOutput:
    def test_result_has_exactly_one_artifact(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)

        output_path = workspace / "single.tif"
        params = ReprojectParams(target_crs="EPSG:32610", output_path=str(output_path))

        op = ReprojectOperator()
        result = op.execute([mat.artifact], params)

        # OperatorResult.artifact is singular — no awkwardness here
        assert result.artifact is not None
        assert result.artifact.type == ArtifactType.RASTER


# ---------------------------------------------------------------------------
# Stress point 7: vector reprojection
# ---------------------------------------------------------------------------


class TestVectorReproject:
    def test_vector_reproject(self, sample_vector_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_vector_4326), workspace)
        assert mat.artifact.type == ArtifactType.VECTOR
        assert mat.artifact.spatial.crs is not None

        output_path = workspace / "projected_pts.geojson"
        params = ReprojectParams(
            target_crs="EPSG:32610",
            output_path=str(output_path),
        )

        op = ReprojectOperator()
        result = op.execute([mat.artifact], params)

        assert result.artifact.type == ArtifactType.VECTOR
        assert "32610" in str(result.artifact.spatial.crs)
        assert result.artifact.spatial.feature_count == 10

    def test_vector_lineage(self, sample_vector_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_vector_4326), workspace)

        output_path = workspace / "lineage_pts.geojson"
        params = ReprojectParams(target_crs="EPSG:32610", output_path=str(output_path))

        op = ReprojectOperator()
        result = op.execute([mat.artifact], params)

        assert result.artifact.lineage.params["source_crs"] is not None
        assert result.artifact.lineage.params["target_crs"] == "EPSG:32610"


# ---------------------------------------------------------------------------
# Validation edge cases
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_no_crs_input(self, workspace):
        artifact = Artifact(
            type=ArtifactType.RASTER,
            name="no_crs",
            backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake.tif"),
            # spatial.crs is None by default
        )
        params = ReprojectParams(target_crs="EPSG:32610", output_path="/out.tif")
        op = ReprojectOperator()
        errors = op.validate_inputs([artifact], params)
        assert any("no CRS" in e for e in errors)

    def test_rejects_same_crs(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)

        params = ReprojectParams(target_crs="EPSG:4326", output_path="/out.tif")
        op = ReprojectOperator()
        errors = op.validate_inputs([mat.artifact], params)
        assert any("already" in e.lower() for e in errors)

    def test_rejects_missing_target_crs(self, sample_raster_4326, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)

        params = ReprojectParams(target_crs="", output_path="/out.tif")
        op = ReprojectOperator()
        errors = op.validate_inputs([mat.artifact], params)
        assert any("target_crs" in e for e in errors)


# ---------------------------------------------------------------------------
# Full loop: connector → operator → executor → registry
# ---------------------------------------------------------------------------


class TestFullLoop:
    def test_reproject_through_full_substrate(self, sample_raster_4326, workspace):
        """End-to-end: materialize → reproject → execute → persist → recover."""
        registry = Registry(workspace)

        # Materialize
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster_4326), workspace)
        registry.save_artifact(mat.artifact)

        # Reproject via executor
        output_path = workspace / "full_loop.tif"
        params = ReprojectParams(target_crs="EPSG:32610", output_path=str(output_path))
        executor = LocalExecutor()
        record = executor.submit(ReprojectOperator(), [mat.artifact], params)

        # Persist
        registry.save_run(record)

        # Recover
        recovered_run = registry.get_run(record.id)
        assert recovered_run.status == RunStatus.COMPLETED
        assert recovered_run.operator_name == "reproject"

        output_id = record.output.artifact.id
        recovered_art = registry.get_artifact(output_id)
        assert recovered_art is not None
        assert "32610" in str(recovered_art.spatial.crs)

        # Lineage
        parents = registry.get_parents(output_id)
        assert len(parents) == 1
        assert parents[0]["artifact_id"] == mat.artifact.id
        assert parents[0]["operation"] == "reproject"

        # Checks persisted
        checks = registry.get_checks(artifact_id=output_id)
        assert len(checks) > 0
