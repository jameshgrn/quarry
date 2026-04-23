"""Pressure test: end-to-end flow through the substrate.

Flow:
  local file → LocalFileConnector.materialize → Artifact
  → ClipRasterOperator.execute → clipped Artifact
  → LocalExecutor.submit (wraps the above)
  → RunRecord with checks
  → validate everything is coherent

This test exercises:
  - Connector → Artifact pathway
  - Artifact identity (not path-based)
  - Operator input validation
  - Operator execution producing new artifact with lineage
  - Executor producing RunRecord with lifecycle
  - Checks attaching to results
  - Materialization provenance

Failure signals we're watching for:
  - artifact vs run validation confusion
  - single-output operator awkwardness
  - mutability causing identity drift
  - source refs feeling too loose
  - normalization/materialization boundary confusion
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from quarry_connectors.local_file import LocalFileConnector
from quarry_core.artifact import ArtifactType, BackingStoreKind, ValidationState
from quarry_core.check import BackingStoreAccessible, CRSValid, ExtentSane
from quarry_core.connector import Connector, MaterializeResult
from quarry_core.executor import RunStatus
from quarry_core.executors.local import LocalExecutor
from quarry_core.operator import Operator
from quarry_operators.clip_raster import ClipRasterOperator, ClipRasterParams
from rasterio.crs import CRS
from rasterio.transform import from_bounds


@pytest.fixture
def sample_raster(tmp_path):
    """Create a small synthetic raster for testing."""
    raster_path = tmp_path / "sample.tif"
    data = np.random.rand(3, 100, 100).astype(np.float32)
    transform = from_bounds(-10.0, -10.0, 10.0, 10.0, 100, 100)

    with rasterio.open(
        raster_path,
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

    return raster_path


@pytest.fixture
def workspace(tmp_path):
    """Provide a workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


class TestConnectorPathway:
    """Test: source → connector → artifact."""

    def test_materialize_produces_artifact(self, sample_raster, workspace):
        conn = LocalFileConnector()
        result = conn.materialize(str(sample_raster), workspace)

        assert isinstance(result, MaterializeResult)
        assert result.strategy == "wrapped_local"
        assert result.source_ref == str(sample_raster)

        art = result.artifact
        assert art.type == ArtifactType.RASTER
        assert art.name == "sample"
        assert art.backing is not None
        assert art.backing.kind == BackingStoreKind.LOCAL_FILE
        assert art.spatial.crs == "EPSG:4326"
        assert art.spatial.band_count == 3
        assert art.spatial.extent is not None

    def test_artifact_identity_not_path(self, sample_raster, workspace):
        """Two materializations of the same file produce distinct artifact IDs."""
        conn = LocalFileConnector()
        r1 = conn.materialize(str(sample_raster), workspace)
        r2 = conn.materialize(str(sample_raster), workspace)

        assert r1.artifact.id != r2.artifact.id
        assert r1.artifact.backing.uri == r2.artifact.backing.uri

    def test_lazy_materialization(self, sample_raster, workspace):
        conn = LocalFileConnector()
        result = conn.materialize(str(sample_raster), workspace, lazy=True)

        assert result.strategy == "lazy_handle"
        assert result.artifact.backing.kind == BackingStoreKind.LAZY_HANDLE
        assert not result.artifact.is_materialized
        assert result.artifact.spatial.crs is None  # lazy = no inspection

    def test_connector_satisfies_protocol(self):
        """LocalFileConnector satisfies the Connector protocol."""
        conn = LocalFileConnector()
        assert isinstance(conn, Connector)

    def test_discover(self, sample_raster):
        conn = LocalFileConnector()
        entries = conn.discover(str(sample_raster.parent))
        assert len(entries) >= 1
        assert any(e.name == "sample" for e in entries)


class TestOperatorExecution:
    """Test: artifact → operator → new artifact."""

    def test_clip_by_bounds(self, sample_raster, workspace):
        # Materialize input
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        input_artifact = mat.artifact

        # Set up clip
        output_path = workspace / "clipped.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )

        # Execute
        op = ClipRasterOperator()
        result = op.execute([input_artifact], params)

        # Verify output artifact
        assert result.artifact.type == ArtifactType.RASTER
        assert result.artifact.id != input_artifact.id
        assert result.artifact.backing.kind == BackingStoreKind.LOCAL_FILE
        assert Path(result.artifact.backing.uri).exists()

        # Verify lineage
        assert result.artifact.lineage is not None
        assert result.artifact.lineage.operation == "clip_raster"
        assert input_artifact.id in result.artifact.lineage.inputs

        # Verify spatial properties changed
        assert result.artifact.spatial.extent is not None
        out_ext = result.artifact.spatial.extent
        # Output should be smaller than input
        in_ext = input_artifact.spatial.extent
        assert out_ext[2] - out_ext[0] <= in_ext[2] - in_ext[0]

    def test_validate_inputs_rejects_bad_type(self, workspace):
        from quarry_core.artifact import Artifact, BackingStore

        # Create a vector artifact (wrong type for clip_raster)
        bad_input = Artifact(
            type=ArtifactType.VECTOR,
            name="not_a_raster",
            backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        )
        params = ClipRasterParams(bounds=(0, 0, 1, 1), output_path="/out.tif")

        op = ClipRasterOperator()
        errors = op.validate_inputs([bad_input], params)
        assert len(errors) > 0
        assert "raster" in errors[0].lower()

    def test_validate_inputs_rejects_lazy(self, sample_raster, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace, lazy=True)

        params = ClipRasterParams(bounds=(0, 0, 1, 1), output_path="/out.tif")
        op = ClipRasterOperator()
        errors = op.validate_inputs([mat.artifact], params)
        assert any("materialized" in e.lower() for e in errors)

    def test_operator_satisfies_protocol(self):
        op = ClipRasterOperator()
        assert isinstance(op, Operator)

    def test_checks_attached_to_result(self, sample_raster, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)

        output_path = workspace / "checked.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )

        op = ClipRasterOperator()
        result = op.execute([mat.artifact], params)

        assert len(result.checks) > 0
        check_names = [c.check_name for c in result.checks]
        assert "crs_valid" in check_names
        assert "backing_accessible" in check_names


class TestExecutorLifecycle:
    """Test: operator + executor → RunRecord."""

    def test_full_lifecycle(self, sample_raster, workspace):
        # Materialize
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)

        # Set up operation
        output_path = workspace / "executed.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        op = ClipRasterOperator()

        # Submit to executor
        executor = LocalExecutor()
        record = executor.submit(op, [mat.artifact], params)

        # Verify RunRecord
        assert record.status == RunStatus.COMPLETED
        assert record.operator_name == "clip_raster"
        assert record.input_ids == [mat.artifact.id]
        assert record.output is not None
        assert record.output.artifact.id != mat.artifact.id
        assert record.started_at is not None
        assert record.completed_at is not None
        assert record.duration_seconds is not None
        assert record.duration_seconds >= 0
        assert record.executor_name == "local"
        assert record.error is None

        # Checks propagate to RunRecord
        assert len(record.checks) > 0

    def test_executor_captures_validation_failure(self, workspace):
        from quarry_core.artifact import Artifact, BackingStore

        bad_input = Artifact(
            type=ArtifactType.VECTOR,
            name="wrong_type",
            backing=BackingStore(kind=BackingStoreKind.LOCAL_FILE, uri="/fake"),
        )
        params = ClipRasterParams(bounds=(0, 0, 1, 1), output_path="/out.tif")
        op = ClipRasterOperator()
        executor = LocalExecutor()

        record = executor.submit(op, [bad_input], params)
        assert record.status == RunStatus.FAILED
        assert record.error is not None
        assert "Validation failed" in record.error

    def test_run_record_retrievable(self, sample_raster, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)

        output_path = workspace / "retrievable.tif"
        params = ClipRasterParams(
            bounds=(-5.0, -5.0, 5.0, 5.0),
            output_path=str(output_path),
        )
        op = ClipRasterOperator()
        executor = LocalExecutor()

        record = executor.submit(op, [mat.artifact], params)
        retrieved = executor.status(record.id)

        assert retrieved.id == record.id
        assert retrieved.status == RunStatus.COMPLETED


class TestChecksIndependent:
    """Test: checks work independently of operator/executor."""

    def test_checks_on_artifact_directly(self, sample_raster, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        artifact = mat.artifact

        # Run checks independently
        crs_check = CRSValid()
        extent_check = ExtentSane()
        backing_check = BackingStoreAccessible()

        r1 = crs_check.run(artifact)
        r2 = extent_check.run(artifact)
        r3 = backing_check.run(artifact)

        assert r1.state == ValidationState.VALID
        assert r2.state == ValidationState.VALID
        assert r3.state == ValidationState.VALID

    def test_check_accumulation(self, sample_raster, workspace):
        conn = LocalFileConnector()
        mat = conn.materialize(str(sample_raster), workspace)
        artifact = mat.artifact

        assert artifact.validation_state == ValidationState.UNCHECKED

        # Add checks
        r1 = CRSValid().run(artifact)
        updated = artifact.with_check(r1)

        assert updated.validation_state == ValidationState.VALID
        assert len(updated.checks) == 1
        # Original unchanged
        assert len(artifact.checks) == 0
